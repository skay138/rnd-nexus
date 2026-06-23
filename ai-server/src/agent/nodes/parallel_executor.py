import asyncio
import logging
from typing import Any
from langchain_core.runnables import RunnableConfig
from agent.state import RDAgentState

logger = logging.getLogger(__name__)


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

    if not pending_tasks:
        return {"pending_tasks": []}

    logger.debug("[parallel_executor] 태스크 %d개 병렬 실행", len(pending_tasks))

    pairs = await asyncio.gather(*[
        _run_one(t["tool"], t.get("args", {}), tools_by_name)
        for t in pending_tasks
    ])

    updated: dict[str, list[str]] = dict(state.get("tool_results", {}))
    for tool_name, result_str in pairs:
        updated[tool_name] = updated.get(tool_name, []) + [result_str]

    return {
        "tool_results":  updated,
        "pending_tasks": [],
    }
