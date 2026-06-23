from typing import Literal
from agent.state import RDAgentState


def should_continue(state: RDAgentState) -> Literal["tool_node", "reflection"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return "reflection"
