from typing import Literal
from langchain_core.runnables import RunnableConfig
from agent.state import RDAgentState


def should_continue(state: RDAgentState, config: RunnableConfig) -> Literal["parallel_executor", "generate"]:
    max_iterations: int = config.get("configurable", {}).get("max_iterations", 3)
    if state.get("iteration_count", 0) >= max_iterations:
        return "generate"
    if state.get("pending_tasks"):
        return "parallel_executor"
    return "generate"
