from langgraph.graph import StateGraph, START, END

from agent.state import RDAgentState
from agent.nodes.orchestrator import orchestrator
from agent.nodes.parallel_executor import parallel_executor
from agent.nodes.generate import generate
from agent.edges.should_continue import should_continue


def build_graph(memory):
    builder = StateGraph(RDAgentState)  # pyrefly: ignore[bad-specialization]

    builder.add_node("orchestrator",      orchestrator)
    builder.add_node("parallel_executor", parallel_executor)
    builder.add_node("generate",          generate)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        should_continue,
        ["parallel_executor", "generate"],
    )

    builder.add_edge("parallel_executor", "orchestrator")
    builder.add_edge("generate",          END)

    return builder.compile(checkpointer=memory)
