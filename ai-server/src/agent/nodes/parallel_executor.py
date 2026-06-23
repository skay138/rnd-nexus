import asyncio
import json
import logging
import time
from typing import Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings

logger = logging.getLogger(__name__)

_WORKER_MAX_STEPS = 5


def _task_key(description: str) -> str:
    return description.strip().lower()


async def _run_worker(
    task: dict,
    tools_by_name: dict[str, Any],
    settings,
) -> list[tuple[str, str]]:
    """
    Mini ReAct agent — 태스크 설명을 받아 필요한 도구를 스스로 선택·실행하고
    [(tool_name, result_str), ...] 목록을 반환합니다.
    """
    label = task.get("label") or task["description"][:40]
    description = task["description"]

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    system = SystemMessage(content="""당신은 R&D 데이터 수집 워커입니다. 주어진 태스크를 완료하기 위해 도구를 사용하세요.
도구 결과를 분석하고 추가 정보가 필요하면 계속 호출하세요. 충분한 데이터를 수집했으면 멈추세요.
도구 없이 데이터를 추측하거나 생성하지 마세요.""")
    messages: list = [system, HumanMessage(content=description)]
    collected: list[tuple[str, str]] = []

    for step in range(_WORKER_MAX_STEPS):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            logger.debug("[worker:%s] 완료 step=%d", label, step + 1)
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            t0 = time.perf_counter()
            try:
                result = await tools_by_name[tool_name].ainvoke(tool_args)
                result_str = str(result)
            except KeyError:
                result_str = f"[ERROR] 알 수 없는 도구: {tool_name}"
            except Exception as e:
                result_str = f"[ERROR] {tool_name} 실패: {type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            logger.debug(
                "[worker:%s] %s elapsed=%.2fs\nargs: %s\nresult: %s",
                label, tool_name, elapsed,
                json.dumps(tool_args, ensure_ascii=False),
                result_str,
            )
            collected.append((tool_name, result_str))
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    return collected


async def parallel_executor(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    pending_tasks: list[dict] = state.get("pending_tasks", [])
    executed: list[dict] = list(state.get("executed_tasks", []))

    if not pending_tasks:
        return {"pending_tasks": [], "executed_tasks": executed}

    already_run = {_task_key(t["description"]) for t in executed}
    fresh_tasks = [t for t in pending_tasks
                   if _task_key(t["description"]) not in already_run]

    skipped = len(pending_tasks) - len(fresh_tasks)
    if skipped:
        logger.debug("[parallel_executor] %d개 태스크 중복으로 건너뜀", skipped)

    if not fresh_tasks:
        logger.debug("[parallel_executor] 실행할 새 태스크 없음 — generate로 단락")
        return {"pending_tasks": [], "executed_tasks": executed, "no_new_data": True}

    logger.debug("[parallel_executor] 워커 %d개 병렬 실행", len(fresh_tasks))

    t0_all = time.perf_counter()
    worker_results = await asyncio.gather(*[
        _run_worker(t, tools_by_name, settings)
        for t in fresh_tasks
    ])
    total_elapsed = time.perf_counter() - t0_all
    logger.debug("[parallel_executor] 전체 완료 elapsed=%.2fs", total_elapsed)

    updated: dict[str, list[str]] = dict(state.get("tool_results", {}))
    msg_lines: list[str] = []

    for task, tool_pairs in zip(fresh_tasks, worker_results):
        task_label = task.get("label") or task["description"][:40]
        tool_lines = [f"[{tool_name}]\n{result_str}" for tool_name, result_str in tool_pairs]
        msg_lines.append(f"# {task_label}\n" + "\n\n".join(tool_lines))
        for tool_name, result_str in tool_pairs:
            updated[tool_name] = updated.get(tool_name, []) + [result_str]

    executed.extend(
        {"description": t["description"], "label": t.get("label", "")}
        for t in fresh_tasks
    )

    result_message = AIMessage(
        content="\n\n---\n\n".join(msg_lines),
        name="tool_results",
    )

    return {
        "messages":       [result_message],
        "tool_results":   updated,
        "pending_tasks":  [],
        "executed_tasks": executed,
        "no_new_data":    False,
    }
