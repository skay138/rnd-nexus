from typing import Literal
from langgraph.graph import StateGraph, START, END

from agent.state import RDAgentState
from agent.nodes.planner import planner
from agent.nodes.llm_call import llm_call
from agent.nodes.tool_node import tool_node
from agent.nodes.reflection import reflection
from agent.nodes.generate import generate


def should_continue(state: RDAgentState) -> Literal["tool_node", "reflection"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return "reflection"


def should_replan(state: RDAgentState) -> Literal["planner", "generate"]:
    if state.get("reflection_result") == "insufficient":
        return "planner"
    return "generate"


def build_graph(memory):
    builder = StateGraph(RDAgentState)  # pyrefly: ignore[bad-specialization]

    builder.add_node("planner",    planner)
    builder.add_node("llm_call",   llm_call)
    builder.add_node("tool_node",  tool_node)
    builder.add_node("reflection", reflection)
    builder.add_node("generate",   generate)

    builder.add_edge(START,      "planner")
    builder.add_edge("planner",  "llm_call")

    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", "reflection"],
    )
    builder.add_edge("tool_node", "llm_call")

    builder.add_conditional_edges(
        "reflection",
        should_replan,
        ["planner", "generate"],
    )

    builder.add_edge("generate", END)

    return builder.compile(checkpointer=memory)
