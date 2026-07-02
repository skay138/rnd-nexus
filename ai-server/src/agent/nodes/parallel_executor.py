import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.state import RDAgentState, TaskSpec, TaskExecutionResult, ToolCallRecord
from common.config.query_config import RequestConfig
from common.llm import get_llm
from common.parsers import (
    clean_tool_result,
    strip_code_fence,
    strip_think,
    summarize_tool_result,
    try_parse,
)
from config import get_settings

logger = logging.getLogger(__name__)

_WORKER_MAX_STEPS = 5


def _build_history_summary(task_execution_results: list[dict]) -> str:
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
    태스크별 도구 요약과 워커 보고만 노출한다.
    """
    lines = ["[수집 결과]"]
    for r in results:
        lines.append(f"태스크: {r['task_description']}")
        for tc in r["tool_calls"]:
            mark = "✗" if tc["is_error"] else "-"
            lines.append(f"  {mark} {tc['tool_name']}: {tc['summary']}")
        if not r["tool_calls"]:
            lines.append("  (도구 호출 없음)")
        if r.get("worker_note"):
            lines.append(f"  보고: {r['worker_note']}")
    return "\n".join(lines)


def _parse_worker_final(text: str) -> tuple[str, list[str]]:
    """워커 최종 응답({"summary", "relevant_ids"} JSON) 파싱.

    JSON이 아니면(모델이 평문으로 답한 경우) 원문을 보고로, 선별 없음으로 처리.
    """
    data = try_parse(strip_code_fence(text))
    if isinstance(data, dict):
        summary = str(data.get("summary", "")).strip()
        ids = [str(i) for i in data.get("relevant_ids") or [] if i]
        return (summary or text), ids
    return text, []


async def _run_worker(
    task: TaskSpec,
    tools_by_name: dict[str, Any],
    settings: Any,
    current_round: int = 0,
    original_query: str = "",
    history_summary: str = "",
    sse_queue: asyncio.Queue | None = None,
) -> TaskExecutionResult:
    """Mini ReAct agent — TaskSpec을 받아 도구를 선택·실행하고 TaskExecutionResult를 반환한다.

    도구 호출 트래픽(AIMessage(tool_calls)+ToolMessage)은 워커 내부에 격리되며
    state.messages에는 올라가지 않는다. 전체 result_text는 task_execution_results로만 전달.
    """
    task_id = task["id"]
    task_description = task["description"]
    tool_calls: list[ToolCallRecord] = []
    worker_note = ""                  # 워커 최종 보고 한 줄 — orchestrator 수집 완료 판단용
    selected_ids: list[str] = []      # 워커가 선별한 태스크 관련 엔티티 ID

    llm = get_llm(model=RequestConfig.current().worker_model or settings.rnd_model)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    system = SystemMessage(content="""<role>
You are an R&D data collection worker. Collect the requested data by calling tools — do not write analysis or interpretation.
</role>

<instructions>
- [태스크] is the top priority. Use [원본 질문] only as supplementary context when keywords are missing or pronouns are used.
- Select tools autonomously based on the task. If sufficient IDs are obtained from search, consider calling a detail-retrieval tool to collect full fields (affiliation, abstract, h_index, etc.).
- Stop when sufficient data is collected or all reasonable retrieval paths are exhausted.
- If [Completed tasks from previous rounds] is provided, do not repeat the same searches.
- When you finish, reply with ONLY this JSON object (no other text):
  {"summary": "한 줄 보고 — 무엇을 수집했는지 또는 왜 찾지 못했는지 (Korean)", "relevant_ids": ["ID1", "ID2"]}
- relevant_ids: IDs of entities DIRECTLY relevant to [태스크], chosen only from IDs that appeared in THIS round's tool results — never invent IDs. If no tools were called this round, relevant_ids must be []. Exclude unrelated or only tangentially related entities.
</instructions>""")

    task_content = (
        f"[원본 질문]\n{original_query}\n\n[태스크]\n{task_description}"
        if original_query
        else task_description
    )
    if history_summary:
        task_content += f"\n\n{history_summary}"

    messages: list = [system, HumanMessage(content=task_content)]
    seen_calls: set[str] = set()

    try:
        for step in range(_WORKER_MAX_STEPS):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                worker_note, selected_ids = _parse_worker_final(
                    strip_think(str(response.content))
                )
                logger.debug(
                    "[WORKER] %-38s  ✓ done  %d steps  %d tools  selected=%d  note=%s",
                    f'"{task_description[:35]}"', step + 1, len(tool_calls),
                    len(selected_ids), worker_note[:60],
                )
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"

                if call_key in seen_calls:
                    logger.debug("[worker:%s] 중복 호출 건너뜀: %s", task_description[:40], call_key[:80])
                    messages.append(ToolMessage(
                        content="[SKIP] 이미 동일한 호출 결과가 있습니다.",
                        tool_call_id=tc["id"],
                    ))
                    continue
                seen_calls.add(call_key)

                t0 = time.perf_counter()
                try:
                    result = await tools_by_name[tool_name].ainvoke(tool_args)
                    result_str = str(result)
                except KeyError:
                    result_str = f"[ERROR] 알 수 없는 도구: {tool_name}"
                except Exception as e:
                    result_str = f"[ERROR] {tool_name} 실패: {type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0

                result_text = clean_tool_result(result_str)
                summary = summarize_tool_result(result_str)
                is_error = result_str.startswith("[ERROR]")

                record: ToolCallRecord = {
                    "tool_name":   tool_name,
                    "args":        tool_args,
                    "result_text": result_text,
                    "summary":     summary,
                    "is_error":    is_error,
                }
                tool_calls.append(record)

                messages.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))

                logger.debug(
                    "[WORKER] %-38s  step=%d  %s  %.2fs\n  in  | %s\n  out | %s",
                    f'"{task_description[:35]}"', step + 1, tool_name, elapsed,
                    json.dumps(tool_args, ensure_ascii=False),
                    summary,
                )

                if sse_queue is not None:
                    await sse_queue.put(("worker_result", {
                        "type":    "task_result",
                        "task_id": task_id,
                        "round":   current_round,
                        "task":    task_description,
                        "tools":   [{"name": tool_name, "summary": summary}],
                    }))

        has_data  = any(not r["is_error"] for r in tool_calls)
        has_error = any(r["is_error"] for r in tool_calls)
        status = "error" if (has_error and not has_data) else ("empty" if not tool_calls else "completed")

        result: TaskExecutionResult = {
            "task_id":          task_id,
            "task_description": task_description,
            "round":            current_round,
            "status":           status,
            "tool_calls":       tool_calls,
            "worker_note":      worker_note,
            "selected_ids":     selected_ids,
        }
        return result

    except Exception as e:
        logger.error("[worker:%s] 에러: %s", task_description[:40], e)
        err_result: TaskExecutionResult = {
            "task_id":          task_id,
            "task_description": task_description,
            "round":            current_round,
            "status":           "error",
            "tool_calls":       tool_calls + [{
                "tool_name":   "worker_error",
                "args":        {},
                "result_text": f"[ERROR] 워커 실행 중 에러 ({type(e).__name__}): {e}",
                "summary":     "워커 오류",
                "is_error":    True,
            }],
            "worker_note":      "",
            "selected_ids":     [],
        }
        return err_result


async def parallel_executor(state: RDAgentState, config: RunnableConfig) -> dict:
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

    async def _run_and_emit(task: TaskSpec) -> TaskExecutionResult:
        return await _run_worker(
            task, tools_by_name, settings,
            current_round=current_round,
            original_query=original_query,
            history_summary=history_summary,
            sse_queue=sse_queue,
        )

    t0_all = time.perf_counter()
    new_results: list[TaskExecutionResult] = list(
        await asyncio.gather(*[_run_and_emit(t) for t in fresh_tasks])
    )
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

    # 정제된 라운드 요약만 공유 컨텍스트에 올림 — raw 도구 트래픽은 워커에 격리,
    # 전체 result_text는 task_execution_results에 단일 보관 (generate가 직접 사용)
    summary_message = AIMessage(
        content=_format_round_results(new_results),
        name="tool_results",
    )

    accumulated = list(state.get("task_execution_results", [])) + new_results

    return {
        "messages":               [summary_message],
        "task_execution_results": accumulated,
        "pending_tasks":          [],
    }
