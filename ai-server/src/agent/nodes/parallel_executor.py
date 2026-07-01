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
from common.parsers import clean_tool_result, summarize_tool_result
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


async def _run_worker(
    task: TaskSpec,
    tools_by_name: dict[str, Any],
    settings: Any,
    current_round: int = 0,
    original_query: str = "",
    history_summary: str = "",
    sse_queue: asyncio.Queue | None = None,
) -> tuple[TaskExecutionResult, list]:
    """Mini ReAct agent — TaskSpec을 받아 도구를 선택·실행하고
    (TaskExecutionResult, worker_messages) 튜플을 반환한다.

    worker_messages: state.messages에 추가할 AIMessage(tool_calls) + ToolMessage 쌍.
    """
    task_id = task["id"]
    task_description = task["description"]
    tool_calls: list[ToolCallRecord] = []
    messages_for_state: list = []   # state.messages에 추가할 tool 상호작용 메시지

    llm = get_llm(model=RequestConfig.current().worker_model or settings.rnd_model)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    system = SystemMessage(content="""<role>
You are an R&D data collection worker. Execute tool calls only — do not analyze, summarize, or explain.
</role>

<instructions>
- [태스크] is the top priority. Use [원본 질문] only as supplementary context when keywords are missing or pronouns are used.
- After obtaining IDs from a search, call detail-retrieval tools to collect affiliation, specialty, abstract, and other detailed fields.
- Stop when sufficient data is collected or there is nothing more to retrieve.
- If [Completed tasks from previous rounds] is provided, do not repeat the same searches.
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
                logger.debug(
                    "[WORKER] %-38s  ✓ done  %d steps  %d tools",
                    f'"{task_description[:35]}"', step + 1, len(tool_calls),
                )
                break

            # tool_calls가 있는 AIMessage → state에 포함
            messages_for_state.append(response)

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"

                if call_key in seen_calls:
                    logger.debug("[worker:%s] 중복 호출 건너뜀: %s", task_description[:40], call_key[:80])
                    skip_msg = ToolMessage(
                        content="[SKIP] 이미 동일한 호출 결과가 있습니다.",
                        tool_call_id=tc["id"],
                    )
                    messages.append(skip_msg)
                    messages_for_state.append(skip_msg)
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

                tool_msg = ToolMessage(content=result_text, tool_call_id=tc["id"])
                messages.append(tool_msg)
                messages_for_state.append(tool_msg)   # state에 포함

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
        }
        return result, messages_for_state

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
        }
        return err_result, messages_for_state


async def parallel_executor(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    pending_tasks: list[TaskSpec] = state.get("pending_tasks", [])

    if not pending_tasks:
        return {"pending_tasks": []}

    fresh_tasks = pending_tasks

    current_round   = state.get("iteration_count", 0)
    original_query  = RequestConfig.current().original_query
    history_summary = _build_history_summary(state.get("task_execution_results", []))
    sse_queue: asyncio.Queue | None = config["configurable"].get("sse_queue")

    logger.debug(
        "[EXEC] round=%d  workers=%d\n%s",
        current_round, len(fresh_tasks),
        "\n".join(f"  dispatch | {t['description'][:60]}" for t in fresh_tasks),
    )

    async def _run_and_emit(task: TaskSpec) -> tuple[TaskExecutionResult, list]:
        return await _run_worker(
            task, tools_by_name, settings,
            current_round=current_round,
            original_query=original_query,
            history_summary=history_summary,
            sse_queue=sse_queue,
        )

    t0_all = time.perf_counter()
    worker_outputs: list[tuple[TaskExecutionResult, list]] = list(
        await asyncio.gather(*[_run_and_emit(t) for t in fresh_tasks])
    )
    total_elapsed = time.perf_counter() - t0_all

    new_results    = [r for r, _ in worker_outputs]
    # 각 워커의 AIMessage(tool_calls) + ToolMessage 쌍 — state.messages에 직접 포함
    worker_messages = [msg for _, msgs in worker_outputs for msg in msgs]

    logger.debug(
        "[EXEC] done  %.2fs\n%s",
        total_elapsed,
        "\n".join(
            f"  {'✓' if r['status'] == 'completed' else '✗'} \"{r['task_description'][:50]}\"  "
            f"{len(r['tool_calls'])} calls  {r['status']}"
            for r in new_results
        ),
    )

    # 구조화된 요약 AIMessage — multi-turn에서 extract_results_from_messages가 파싱하는 마커
    msg_data = [
        {
            "task_id":          r["task_id"],
            "task_description": r["task_description"],
            "round":            r["round"],
            "tool_calls": [
                {
                    "tool_name":   tc["tool_name"],
                    "result_text": tc["result_text"],
                    "summary":     tc["summary"],
                    "is_error":    tc["is_error"],
                }
                for tc in r["tool_calls"]
            ],
        }
        for r in new_results
    ]
    summary_message = AIMessage(
        content=json.dumps(msg_data, ensure_ascii=False),
        name="tool_results",
    )

    accumulated = list(state.get("task_execution_results", [])) + new_results

    return {
        "messages":               worker_messages + [summary_message],
        "task_execution_results": accumulated,
        "pending_tasks":          [],
    }
