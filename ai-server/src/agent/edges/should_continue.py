from typing import Literal
from agent.state import RDAgentState


def should_continue(state: RDAgentState) -> Literal["parallel_executor", "generate"]:
    if state.get("pending_tasks"):
        return "parallel_executor"
    return "generate"
