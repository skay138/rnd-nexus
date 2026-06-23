import asyncio
import json
import logging
from typing import Any
from langchain_core.runnables import RunnableConfig
from agent.state import RDAgentState

logger = logging.getLogger(__name__)


def _task_key(tool: str, args: dict) -> str:
    return f"{tool}::{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


async def _run_one(tool_name: str, tool_args: dict, tools_by_name: dict[str, Any]) -> tuple[str, str]:
    try:
        tool_fn = tools_by_name[tool_name]
        result = await tool_fn.ainvoke(tool_args)
        result_str = str(result)
        logger.debug("[parallel_executor] %s(args=%s) → %s", tool_name, tool_args, result_str[:300])
        return tool_name, result_str
    except KeyError:
        msg = f"[ERROR] 알 수 없는 도구: {tool_name}"
        logger.error(msg)
        return tool_name, msg
    except Exception as e:
        msg = f"[ERROR] {tool_name} 호출 실패: {type(e).__name__}: {e}"
        logger.error("[parallel_executor] %s(args=%s) failed: %s", tool_name, tool_args, e, exc_info=True)
        return tool_name, msg


async def parallel_executor(state: RDAgentState, config: RunnableConfig) -> dict:
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    pending_tasks: list[dict] = state.get("pending_tasks", [])
    executed: list[dict] = list(state.get("executed_tasks", []))

    if not pending_tasks:
        return {"pending_tasks": [], "executed_tasks": executed}

    # 동일 {tool+args} 중복 차단
    already_run = {_task_key(t["tool"], t.get("args", {})) for t in executed}
    fresh_tasks = [t for t in pending_tasks if _task_key(t["tool"], t.get("args", {})) not in already_run]

    skipped = len(pending_tasks) - len(fresh_tasks)
    if skipped:
        logger.debug("[parallel_executor] %d개 태스크 중복으로 건너뜀", skipped)

    if not fresh_tasks:
        logger.debug("[parallel_executor] 실행할 새 태스크 없음 — 수집 완료로 처리")
        return {"pending_tasks": [], "executed_tasks": executed}

    logger.debug("[parallel_executor] 태스크 %d개 병렬 실행", len(fresh_tasks))

    pairs = await asyncio.gather(*[
        _run_one(t["tool"], t.get("args", {}), tools_by_name)
        for t in fresh_tasks
    ])

    updated: dict[str, list[str]] = dict(state.get("tool_results", {}))
    for tool_name, result_str in pairs:
        updated[tool_name] = updated.get(tool_name, []) + [result_str]

    executed.extend({"tool": t["tool"], "args": t.get("args", {})} for t in fresh_tasks)

    return {
        "tool_results":  updated,
        "pending_tasks": [],
        "executed_tasks": executed,
    }
