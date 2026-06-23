from typing import Literal
from agent.state import RDAgentState


def should_replan(state: RDAgentState) -> Literal["planner", "generate"]:
    if state.get("reflection_result") == "insufficient":
        return "planner"
    return "generate"
