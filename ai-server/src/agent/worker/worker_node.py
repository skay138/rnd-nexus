"""parallel_executor 노드 — 라운드의 태스크들을 Worker Agent로 병렬 실행.

담당 (코드 레벨 제어):
- task_id 기준 중복 태스크 차단
- 워커 병렬 실행 (asyncio.gather) 및 에러 격리
- 사후 상세 보강 (agent.enrichment)
- 라운드 요약 메시지 생성 — state.messages에 올라가는 유일한 산출물
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent.worker.enrichment import enrich_selected_entities
from agent.state import RDAgentState, TaskExecutionResult, TaskSpec
from agent.worker.worker import run_worker
from common.config.query_config import RequestConfig
from config import get_settings

logger = logging.getLogger(__name__)


def _build_history_summary(task_execution_results: list[TaskExecutionResult]) -> str:
    """이전 라운드 완료 태스크·결과 요약 — 워커 중복 수집 방지용."""
    completed: list[str] = []
    empty_tools: list[str] = []

    for result in task_execution_results:
        task_desc = result.get("task_description", "")[:50]
        summaries = [
            tc["summary"]
            for tc in result.get("tool_calls", [])
            if not tc.get("is_error") and tc.get("summary")
        ]
        if summaries:
            completed.append(f"- {task_desc}: {', '.join(summaries[:3])}")
        else:
            for tc in result.get("tool_calls", []):
                if not tc.get("is_error"):
                    entry = f"{tc['tool_name']}: 빈 결과"
                    if entry not in empty_tools:
                        empty_tools.append(entry)

    lines: list[str] = []
    if completed:
        lines.append("[Completed tasks from previous rounds]")
        lines.extend(completed[:20])
    if empty_tools:
        lines.append("[Searches with no results in previous rounds — do not repeat]")
        lines.extend(f"- {e}" for e in empty_tools[:10])

    return "\n".join(lines) if lines else ""


def _format_round_results(results: list[TaskExecutionResult]) -> str:
    """라운드 결과를 읽기 쉬운 요약으로 변환 — state.messages에 올라가는 유일한 산출물.

    전체 result_text는 task_execution_results에만 저장되고, 공유 컨텍스트에는
    태스크별 도구 요약·선별 ID·워커 보고만 노출한다.
    """
    lines = ["[수집 결과]"]
    for r in results:
        lines.append(f"태스크: {r['task_description']}")
        for tc in r["tool_calls"]:
            mark = "✗" if tc["is_error"] else "-"
            lines.append(f"  {mark} {tc['tool_name']}: {tc['summary']}")
        if not r["tool_calls"]:
            lines.append("  (도구 호출 없음)")
        if r.get("selected_ids"):
            # orchestrator가 다음 라운드 태스크 설명에 ID를 직접 포함할 수 있도록 노출
            lines.append(f"  선별 ID: {', '.join(r['selected_ids'][:10])}")
        if r.get("worker_note"):
            lines.append(f"  보고: {r['worker_note']}")
    return "\n".join(lines)


async def worker_node(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    pending_tasks: list[TaskSpec] = state.get("pending_tasks", [])

    if not pending_tasks:
        return {"pending_tasks": []}

    # 코드 레벨 중복 태스크 차단 — 이번 턴에 이미 실행된 task_id는 재실행하지 않음
    executed_ids = {r["task_id"] for r in state.get("task_execution_results", [])}
    fresh_tasks = [t for t in pending_tasks if t["id"] not in executed_ids]
    if len(fresh_tasks) < len(pending_tasks):
        logger.warning(
            "[EXEC] 중복 태스크 %d건 건너뜀: %s",
            len(pending_tasks) - len(fresh_tasks),
            [t["description"][:40] for t in pending_tasks if t["id"] in executed_ids],
        )
    if not fresh_tasks:
        note = AIMessage(
            content="[수집 결과]\n(계획된 태스크가 모두 이미 실행된 태스크와 중복되어 건너뜀 — 새 데이터 없음. "
                    "다른 각도의 태스크를 계획하거나 수집을 종료하세요.)",
            name="tool_results",
        )
        return {"messages": [note], "pending_tasks": []}

    current_round   = state.get("iteration_count", 0)
    original_query  = RequestConfig.current().original_query
    history_summary = _build_history_summary(state.get("task_execution_results", []))
    sse_queue: asyncio.Queue | None = config["configurable"].get("sse_queue")

    logger.debug(
        "[EXEC] round=%d  workers=%d\n%s",
        current_round, len(fresh_tasks),
        "\n".join(f"  dispatch | {t['description'][:60]}" for t in fresh_tasks),
    )

    t0_all = time.perf_counter()
    new_results: list[TaskExecutionResult] = list(await asyncio.gather(*[
        run_worker(
            task, tools_by_name, settings,
            current_round=current_round,
            original_query=original_query,
            history_summary=history_summary,
            sse_queue=sse_queue,
        )
        for task in fresh_tasks
    ]))
    total_elapsed = time.perf_counter() - t0_all

    logger.debug(
        "[EXEC] done  %.2fs\n%s",
        total_elapsed,
        "\n".join(
            f"  {'✓' if r['status'] == 'completed' else '✗'} \"{r['task_description'][:50]}\"  "
            f"{len(r['tool_calls'])} calls  {r['status']}"
            for r in new_results
        ),
    )

    # 사후 상세 보강 — 자동 조인을 빠져나간 선별 ID(그래프 유래 등)를 get_entities로 채움
    enrich = await enrich_selected_entities(
        new_results,
        list(state.get("task_execution_results", [])) + new_results,
        tools_by_name, current_round, sse_queue,
    )
    if enrich is not None:
        new_results.append(enrich)

    # 정제된 라운드 요약만 공유 컨텍스트에 올림 — raw 도구 트래픽은 워커에 격리,
    # 전체 result_text는 task_execution_results에 단일 보관 (generate가 직접 사용)
    summary_message = AIMessage(
        content=_format_round_results(new_results),
        name="tool_results",
    )

    return {
        "messages":               [summary_message],
        "task_execution_results": list(state.get("task_execution_results", [])) + new_results,
        "pending_tasks":          [],
    }