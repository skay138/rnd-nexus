from __future__ import annotations
import ast
import json
import json as _json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from api.schemas import QueryRequest
from common.config.query_config import QueryConfig, RequestConfig

logger = logging.getLogger(__name__)
router = APIRouter()


def _item_to_ref(d: dict) -> "dict | None":
    if "paper_id" in d:
        return {"type": "논문", "id": d["paper_id"], "title": d.get("title", "")}
    if "patent_id" in d:
        return {"type": "특허", "id": d["patent_id"], "title": d.get("title", "")}
    if "researcher_id" in d:
        return {"type": "연구자", "id": d["researcher_id"], "title": d.get("name", "")}
    if "technology_id" in d:
        return {"type": "기술", "id": d["technology_id"], "title": d.get("name", "")}
    if "project_id" in d:
        return {"type": "과제", "id": d["project_id"], "title": d.get("title", d.get("name", ""))}
    if "node_type" in d:
        return {
            "type": d["node_type"],
            "id": str(d.get("id", "") or d.get("entity_id", "")),
            "title": d.get("name", d.get("title", "")),
        }
    return None


def _summarize_result(result_str: str) -> str:
    """tool result 문자열을 한 줄 요약으로 변환."""
    s = str(result_str)
    if s.startswith("[ERROR]"):
        return "오류"
    try:
        items = ast.literal_eval(s)
        if not isinstance(items, list):
            return "1건"
        texts = [i["text"] for i in items if isinstance(i, dict) and i.get("type") == "text"]
        if not texts:
            return f"{len(items)}건"
        previews: list = []
        for text in texts[:5]:
            try:
                d = _json.loads(text)
                entries = d if isinstance(d, list) else [d]
                for entry in entries[:3]:
                    if not isinstance(entry, dict):
                        continue
                    label = str(entry.get("name") or entry.get("title") or entry.get("id") or "")[:25]
                    score = entry.get("score")
                    if score is not None:
                        label += f"({score:.2f})"
                    if label:
                        previews.append(label)
            except Exception:
                pass
        count = len(texts)
        return f"{count}건" + (f": {', '.join(previews[:3])}" if previews else "")
    except Exception:
        return "결과 있음"


def _build_references(tool_results: dict) -> list:
    refs: list = []
    seen: set = set()
    for results in tool_results.values():
        for result_str in results:
            if str(result_str).startswith("[ERROR]"):
                continue
            try:
                items = ast.literal_eval(str(result_str))
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    text = item.get("text", "")
                    if not text:
                        continue
                    try:
                        data = _json.loads(text)
                        if not isinstance(data, list):
                            data = [data]
                        for d in data[:5]:
                            if not isinstance(d, dict):
                                continue
                            ref = _item_to_ref(d)
                            if ref and ref["id"] and ref["id"] not in seen:
                                seen.add(ref["id"])
                                ref["num"] = len(refs) + 1
                                refs.append(ref)
                    except Exception:
                        continue
            except Exception:
                continue
    return refs


@router.post("/agent/query")
async def agent_query(body: QueryRequest, request: Request) -> Any:
    graph           = request.app.state.graph
    llm_with_tools  = request.app.state.llm_with_tools
    tools_by_name   = request.app.state.tools_by_name
    config_repo     = request.app.state.config_repo

    # 설정 우선순위: API 파라미터 > DB > 기본값
    override = QueryConfig(
        generate_model = body.config.generate_model  if body.config else None,
        max_replan     = body.config.max_replan      if body.config else None,
        temperature    = body.config.temperature     if body.config else None,
        semantic_top_k = body.config.semantic_top_k  if body.config else None,
        dense_weight   = body.config.dense_weight    if body.config else None,
        sparse_weight  = body.config.sparse_weight   if body.config else None,
    )
    resolved = RequestConfig._resolve(config_repo, override)
    RequestConfig.set_current(resolved, original_query=body.query)

    thread_id = body.session_id or str(uuid.uuid4())

    lg_config: dict[str, Any] = {
        "configurable": {
            "thread_id":      thread_id,
            "llm_with_tools": llm_with_tools,
            "tools_by_name":  tools_by_name,
            "generate_model": resolved.generate_model,
            "max_replan":     resolved.max_replan,
        },
        "recursion_limit": 50,
    }
    initial_state = {
        "messages":        [HumanMessage(content=body.query)],
        "iteration_count": 0,
        "tool_results":    {},
        "executed_tasks":  [],
    }

    return EventSourceResponse(_stream_events(graph, initial_state, lg_config))


async def _stream_events(
    graph: Any,
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> AsyncGenerator[str, None]:
    last_tool_results: dict   = {}
    prev_tool_results: dict   = {}
    last_iteration_count: int = 0

    try:
        async for item in graph.astream(initial_state, config, stream_mode=["values", "messages"]):
            mode, data = item

            # ── 실시간 토큰 (generate 노드에서만) ─────────────────────────────
            if mode == "messages":
                msg_chunk, metadata = data
                if metadata.get("langgraph_node") == "generate":
                    content = getattr(msg_chunk, "content", "")
                    if content:
                        yield json.dumps({"type": "token", "content": content})
                continue

            # ── 노드 완료 후 상태 스냅샷 (values) ─────────────────────────────
            messages        = data.get("messages", [])
            tool_results    = data.get("tool_results", {})
            iteration_count = data.get("iteration_count", 0)
            pending_tasks   = data.get("pending_tasks", [])

            # 오케스트레이터 라운드 이벤트
            if iteration_count > last_iteration_count:
                last_iteration_count = iteration_count
                orch_msg = next(
                    (m for m in reversed(messages) if getattr(m, "name", None) == "orchestrator"),
                    None,
                )
                if orch_msg:
                    yield json.dumps({
                        "type":      "orchestrator",
                        "round":     iteration_count,
                        "reasoning": str(orch_msg.content)[:400],
                        "tasks":     pending_tasks,
                    })

            # 병렬 실행 결과 (parallel_executor)
            if tool_results:
                for tool_name, results in tool_results.items():
                    prev_count = len(prev_tool_results.get(tool_name, []))
                    if len(results) > prev_count:
                        summary = _summarize_result(results[-1])
                        yield json.dumps({"type": "tool_result", "tool": tool_name, "summary": summary})
                last_tool_results = tool_results
                prev_tool_results = {k: list(v) for k, v in tool_results.items()}

    except Exception as e:
        logger.exception("agent_query 스트리밍 오류")
        yield json.dumps({"type": "error", "message": str(e)})

    references = _build_references(last_tool_results)
    yield json.dumps({"type": "done", "references": references})
