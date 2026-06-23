from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    reflection_result: str
    reflection_feedback: str
    replan_count: int
    tool_results: dict[str, list[str]]
