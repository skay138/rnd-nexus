from __future__ import annotations
import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessageChunk

from api.schemas import QueryRequest
from common.config.query_config import QueryConfig, RequestConfig
from common.parsers import iter_entities

logger = logging.getLogger(__name__)
router = APIRouter()


_ENTITY_ID_KEYS: list[tuple[str, str, str | None]] = [
    # (id_key, type_label, title_key)  — title_key=None → try name then title
    ("paper_id",      "Paper",        "title"),
    ("patent_id",     "Patent",       "title"),
    ("researcher_id", "Researcher",   "name"),
    ("tech_id",       "Technology",   "name"),
    ("technology_id", "Technology",   "name"),
    ("project_id",    "Project",      "title"),
    ("org_id",        "Organization", "name"),
]


def _entity_to_ref(d: dict) -> dict | None:
    for id_key, label, title_key in _ENTITY_ID_KEYS:
        if id_key in d:
            title = d.get(title_key or "name") or d.get("title") or d.get("name") or ""
            return {"type": label, "id": d[id_key], "title": title}
    if "node_type" in d:
        return {
            "type":  d["node_type"],
            "id":    str(d.get("id") or d.get("entity_id") or ""),
            "title": d.get("name") or d.get("title") or "",
        }
    return None


def _build_references(task_execution_results: list) -> list:
    refs: list = []
    seen: set = set()
    for result in task_execution_results:
        for tc in result.get("tool_calls", []):
            if tc.get("is_error"):
                continue
            for d in iter_entities(tc.get("result_text", "")):
                ref = _entity_to_ref(d)
                if ref and ref["id"] and ref["id"] not in seen:
                    seen.add(ref["id"])
                    ref["num"] = len(refs) + 1
                    refs.append(ref)
    return refs


@router.post("/agent/query")
async def agent_query(body: QueryRequest, request: Request) -> Any:
    graph         = request.app.state.graph
    tools_by_name = request.app.state.tools_by_name
    config_repo   = request.app.state.config_repo

    # 설정 우선순위: API 파라미터 > DB > 기본값
    override = QueryConfig(
        generate_model = body.config.generate_model  if body.config else None,
        max_iterations = body.config.max_iterations  if body.config else None,
        temperature    = body.config.temperature     if body.config else None,
        semantic_top_k = body.config.semantic_top_k  if body.config else None,
        keyword_weight = body.config.keyword_weight  if body.config else None,
    )
    resolved = RequestConfig._resolve(config_repo, override)
    RequestConfig.set_current(resolved, original_query=body.query)

    thread_id = body.session_id or str(uuid.uuid4())

    lg_config: dict[str, Any] = {
        "configurable": {
            "thread_id":     thread_id,
            "tools_by_name": tools_by_name,
            "max_iterations": resolved.max_iterations,
        },
        "recursion_limit": 50,
    }
    # 멀티턴 시 이전 체크포인트 위에 덮어쓰는 필드만 포함
    # task_execution_results / executed_task_ids 초기화 → 이번 턴이 깨끗하게 시작
    initial_state = {
        "messages":               [HumanMessage(content=body.query)],
        "iteration_count":        0,
        "task_execution_results": [],
        "pending_tasks":          [],
        "out_of_scope":           False,
    }

    return EventSourceResponse(_stream_events(graph, initial_state, lg_config))


async def _stream_events(
    graph: Any,
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> AsyncGenerator[str, None]:
    sse_queue: asyncio.Queue = asyncio.Queue()
    config["configurable"]["sse_queue"] = sse_queue

    last_task_execution_results: list = []
    last_iteration_count: int = 0
    tokens_sent: bool         = False
    last_final_answer: str    = ""
    think_active: bool        = False

    async def _run_graph() -> None:
        try:
            async for item in graph.astream(initial_state, config, stream_mode=["values", "messages"]):
                await sse_queue.put(("graph", item))
        except Exception as e:
            await sse_queue.put(("error", e))
        finally:
            await sse_queue.put(None)

    asyncio.create_task(_run_graph())

    try:
        while True:
            item = await sse_queue.get()
            if item is None:
                break

            kind = item[0]

            if kind == "error":
                yield json.dumps({"type": "error", "message": str(item[1])})
                break

            if kind == "worker_result":
                yield json.dumps(item[1])
                continue

            # kind == "graph"
            mode, data = item[1]

            # ── 실시간 토큰 (generate 노드에서만) ─────────────────────────────
            if mode == "messages":
                msg_chunk, metadata = data
                if (metadata.get("langgraph_node") == "generate"
                        and isinstance(msg_chunk, AIMessageChunk)):
                    raw = getattr(msg_chunk, "content", "")
                    if raw:
                        if "<think>" in raw:
                            think_active = True
                        elif "</think>" in raw:
                            think_active = False
                            after = raw.split("</think>", 1)[1]
                            if after:
                                tokens_sent = True
                                yield json.dumps({"type": "token", "content": after})
                        elif not think_active:
                            tokens_sent = True
                            yield json.dumps({"type": "token", "content": raw})
                continue

            # ── 노드 완료 후 상태 스냅샷 (values) ─────────────────────────────
            messages               = data.get("messages", [])
            task_execution_results = data.get("task_execution_results", [])
            iteration_count        = data.get("iteration_count", 0)
            pending_tasks          = data.get("pending_tasks", [])

            for msg in reversed(messages):
                if getattr(msg, "name", None) == "final_answer":
                    last_final_answer = str(msg.content)
                    break

            if iteration_count > last_iteration_count:
                last_iteration_count = iteration_count
                orch_msg = next(
                    (m for m in reversed(messages) if getattr(m, "name", None) == "orchestrator"),
                    None,
                )
                if orch_msg:
                    raw_content = str(orch_msg.content)
                    # reasoning 뒤의 상태 마커·태스크 목록 제거 (tasks는 별도 필드로 전송)
                    reasoning_only = raw_content
                    for marker in ("[계획한 태스크]", "[수집 완료", "[범위 외"):
                        reasoning_only = reasoning_only.split(marker)[0]
                    reasoning_only = reasoning_only.strip()
                    yield json.dumps({
                        "type":      "orchestrator",
                        "round":     iteration_count,
                        "reasoning": reasoning_only,
                        "tasks":     [t["description"] for t in pending_tasks],
                    })

            if task_execution_results:
                last_task_execution_results = task_execution_results

    except Exception as e:
        logger.exception("agent_query 스트리밍 오류")
        yield json.dumps({"type": "error", "message": str(e)})

    if not tokens_sent and last_final_answer:
        logger.warning("[query] 토큰 미수신 — final_answer fallback 전송 (len=%d)", len(last_final_answer))
        yield json.dumps({"type": "token", "content": last_final_answer})

    references = _build_references(last_task_execution_results)
    yield json.dumps({"type": "done", "references": references})
