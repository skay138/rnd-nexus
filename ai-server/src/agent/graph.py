from typing import Literal
from langgraph.graph import StateGraph, START, END

from agent.state import RDAgentState
from agent.orchestrator.orchestrator_node import orchestrator_node
from agent.worker.worker_node import worker_node
from agent.generator.generator_node import generator_node


def _should_continue(state: RDAgentState) -> Literal["worker", "generator"]:
    if state.get("pending_tasks"):
        return "worker"
    return "generator"


def _after_worker(state: RDAgentState, config) -> Literal["orchestrator", "generator"]:
    max_iterations = config.get("configurable", {}).get("max_iterations", 3)
    if state.get("iteration_count", 0) >= max_iterations:
        return "generator"
    return "orchestrator"


def build_graph(memory):
    builder = StateGraph(RDAgentState)  # pyrefly: ignore[bad-specialization]

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("worker",       worker_node)
    builder.add_node("generator",    generator_node)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        _should_continue,
        ["worker", "generator"],
    )

    builder.add_conditional_edges(
        "worker",
        _after_worker,
        ["orchestrator", "generator"],
    )

    builder.add_edge("generator", END)

    return builder.compile(checkpointer=memory)
