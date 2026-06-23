from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: dict[str, list[str]]
    iteration_count: int
    pending_tasks: list[dict]   # orchestrator → parallel_executor
