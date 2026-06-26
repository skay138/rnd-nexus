from __future__ import annotations
import ast
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessageChunk

from api.schemas import QueryRequest
from common.config.query_config import QueryConfig, RequestConfig

logger = logging.getLogger(__name__)
router = APIRouter()


from common.parsers import item_to_ref, iter_entities


def _build_references(tool_results: dict) -> list:
    refs: list = []
    seen: set = set()
    for results in tool_results.values():
        for result_str in results:
            for d in iter_entities(result_str):
                ref = item_to_ref(d)
                if ref and ref["id"] and ref["id"] not in seen:
                    seen.add(ref["id"])
                    ref["num"] = len(refs) + 1
                    refs.append(ref)
    return refs


@router.post("/agent/query")
async def agent_query(body: QueryRequest, request: Request) -> Any:
    graph         = request.app.state.graph
    tools_by_name = request.app.state.tools_by_name
    config_repo     = request.app.state.config_repo

    # 설정 우선순위: API 파라미터 > DB > 기본값
    override = QueryConfig(
        generate_model = body.config.generate_model  if body.config else None,
        max_iterations = body.config.max_iterations  if body.config else None,
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
            "thread_id":     thread_id,
            "tools_by_name": tools_by_name,
            "max_iterations": resolved.max_iterations,
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
    last_tool_results: dict    = {}
    prev_task_result_count: int = 0
    last_iteration_count: int  = 0
    tokens_sent: bool          = False
    last_final_answer: str     = ""
    _in_think: bool            = False  # Qwen3 <think> 블록 내부 여부

    try:
        async for item in graph.astream(initial_state, config, stream_mode=["values", "messages"]):
            mode, data = item

            # ── 실시간 토큰 (generate 노드에서만) ─────────────────────────────
            if mode == "messages":
                msg_chunk, metadata = data
                if (metadata.get("langgraph_node") == "generate"
                        and isinstance(msg_chunk, AIMessageChunk)):
                    content = getattr(msg_chunk, "content", "")
                    if not content:
                        continue
                    # Qwen3 thinking 토큰 필터: <think>...</think> 구간은 SSE로 내보내지 않음
                    if "<think>" in content:
                        _in_think = True
                    if "</think>" in content:
                        _in_think = False
                        after = content.split("</think>", 1)[1]
                        if after:
                            tokens_sent = True
                            yield json.dumps({"type": "token", "content": after})
                        continue
                    if _in_think:
                        continue
                    tokens_sent = True
                    yield json.dumps({"type": "token", "content": content})
                continue

            # ── 노드 완료 후 상태 스냅샷 (values) ─────────────────────────────
            messages        = data.get("messages", [])
            tool_results    = data.get("tool_results", {})
            iteration_count = data.get("iteration_count", 0)
            pending_tasks   = data.get("pending_tasks", [])

            # final_answer 추적 (토큰 미수신 시 fallback용)
            for msg in reversed(messages):
                if getattr(msg, "name", None) == "final_answer":
                    last_final_answer = str(msg.content)
                    break

            # 오케스트레이터 라운드 이벤트
            if iteration_count > last_iteration_count:
                last_iteration_count = iteration_count
                orch_msg = next(
                    (m for m in reversed(messages) if getattr(m, "name", None) == "orchestrator"),
                    None,
                )
                if orch_msg:
                    # msg_content = reasoning + "\n\n[계획한 태스크]..." 형식 — 순수 reasoning만 추출
                    raw = str(orch_msg.content)
                    reasoning_only = raw.split("\n\n[계획한 태스크]")[0].split("\n\n[수집 완료")[0]
                    yield json.dumps({
                        "type":      "orchestrator",
                        "round":     iteration_count,
                        "reasoning": reasoning_only,
                        "tasks":     pending_tasks,
                    })

            # task별 실행 결과 (parallel_executor)
            task_results_list = data.get("task_results", [])
            if len(task_results_list) > prev_task_result_count:
                for tr in task_results_list[prev_task_result_count:]:
                    yield json.dumps({"type": "task_result", "round": tr["round"],
                                      "task": tr["task"], "tools": tr["tools"]})
                prev_task_result_count = len(task_results_list)

            # _build_references용 tool_results 유지
            if tool_results:
                last_tool_results = tool_results

    except Exception as e:
        logger.exception("agent_query 스트리밍 오류")
        yield json.dumps({"type": "error", "message": str(e)})

    # 토큰이 전혀 스트림되지 않았지만 final_answer가 있으면 한번에 전송
    if not tokens_sent and last_final_answer:
        logger.warning("[query] 토큰 미수신 — final_answer fallback 전송 (len=%d)", len(last_final_answer))
        yield json.dumps({"type": "token", "content": last_final_answer})

    references = _build_references(last_tool_results)
    yield json.dumps({"type": "done", "references": references})
