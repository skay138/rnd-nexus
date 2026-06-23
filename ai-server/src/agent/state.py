from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: dict[str, list[str]]
    iteration_count: int
    pending_tasks: list[dict]   # orchestrator → parallel_executor
    executed_tasks: list[dict]  # 실행된 {tool, args} 기록 (중복 차단용)
    no_new_data: bool           # parallel_executor에서 새 태스크 없을 때 → generate로 단락
