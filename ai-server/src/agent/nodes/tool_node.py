import logging
from typing import Any, cast
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from agent.state import RDAgentState

logger = logging.getLogger(__name__)


class _ToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


async def tool_node(state: RDAgentState, config: RunnableConfig) -> dict:
    """LLM이 요청한 도구(ToolCall)들을 실행하고 결과를 ToolMessage로 반환합니다."""
    logger.debug("[Node] tool_node 시작")
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    result = []

    for raw_call in state["messages"][-1].tool_calls:
        tc = cast(_ToolCall, raw_call)
        tool_name = tc["name"]
        tool_args = tc["args"]

        try:
            tool_fn = tools_by_name[tool_name]
            observation_raw: Any = await tool_fn.ainvoke(tool_args)
            observation_str = str(observation_raw)
            logger.debug("[Tool Call] %s(args=%s) -> %s", tool_name, tool_args, observation_str[:1000])
            
            if len(observation_str) > 3000:
                from memory.tool_cache import save_tool_result
                redis_key = await save_tool_result(observation_str)
                if redis_key:
                    observation_str = f"[데이터가 길어 캐시되었습니다]\n* Redis Key: {redis_key} (전체 내용 조회 시 read_redis_data 도구 사용)\n* 미리보기:\n{observation_str[:2500]}..."
        except Exception as e:
            observation_str = f"[ERROR] {tool_name} 호출 실패: {type(e).__name__}: {e}. 재시도하거나 다른 방법을 시도하세요."
            logger.error("[Tool Call] %s(args=%s) failed: %s: %s", tool_name, tool_args, type(e).__name__, e, exc_info=True)

        result.append(
            ToolMessage(
                content=observation_str,
                tool_call_id=tc["id"],
            )
        )

    updated_cache: dict[str, list[str]] = dict(state.get("tool_results", {}))
    for raw_call, msg in zip(state["messages"][-1].tool_calls, result):
        tc = cast(_ToolCall, raw_call)
        updated_cache[tc["name"]] = updated_cache.get(tc["name"], []) + [msg.content]

    return {"messages": result, "tool_results": updated_cache}
