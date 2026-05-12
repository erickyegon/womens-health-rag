"""
LangGraph Agent — Episodes 18–23
==================================
Stateful, self-correcting RAG agent with adaptive routing.

Graph topology:
    START → router → [direct_answer | retrieve → grade → (answer | rewrite↺)]
                                                    ↓
                                          hallucination_check → [END | rewrite↺]

Episode 18: basic graph structure
Episode 19: router + adaptive retrieval
Episode 20: grade + rewrite self-correction loop
Episode 21: LangSmith tracing
Episode 22: human-in-the-loop checkpoints
Episode 23: multi-agent via supervisor (separate file)
"""
from __future__ import annotations
import logging
from typing import Any
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from rag.agent.state import AgentState, initial_state
from rag.agent.nodes import (
    answer_node, direct_answer_node, grade_node,
    hallucination_check_node, retrieve_node, rewrite_node, router_node,
    route_after_grade, route_after_hallucination, route_after_router,
)

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None, human_in_loop: bool = False):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointer:   Checkpointer for state persistence (Episode 22).
                        Pass MemorySaver() for in-memory, SqliteSaver for disk.
        human_in_loop:  If True, add an interrupt before answer_node (Ep 22).

    Returns:
        Compiled LangGraph graph.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("router",              router_node)
    graph.add_node("retrieve",            retrieve_node)
    graph.add_node("grade",               grade_node)
    graph.add_node("rewrite",             rewrite_node)
    graph.add_node("answer",              answer_node)
    graph.add_node("direct_answer",       direct_answer_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # ── Conditional routing after router ─────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"direct": "direct_answer", "retrieve": "retrieve"},
    )

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "answer", "rewrite": "rewrite"},
    )

    # ── Rewrite loop ──────────────────────────────────────────────────────────
    graph.add_edge("rewrite", "retrieve")

    # ── Post-generation ───────────────────────────────────────────────────────
    graph.add_edge("answer",        "hallucination_check")
    graph.add_edge("direct_answer", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"end": END, "rewrite": "rewrite"},
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if human_in_loop:
        compile_kwargs["interrupt_before"] = ["answer"]

    return graph.compile(**compile_kwargs)


def run_agent(question: str, thread_id: str = "default",
              checkpointer=None, **kwargs) -> dict:
    """
    Run the agent for a single question.

    Args:
        question:    User's question.
        thread_id:   Conversation thread ID for checkpointing.
        checkpointer: Optional checkpointer for persistence.

    Returns:
        Final AgentState dict.
    """
    app    = build_graph(checkpointer=checkpointer)
    state  = initial_state(question)
    config = {"configurable": {"thread_id": thread_id}}

    final = app.invoke(state, config=config)
    logger.info("Agent complete — answer: %s chars, grounded: %s",
                len(final.get("answer","") or ""), final.get("grounded"))
    return final


def stream_agent(question: str, thread_id: str = "default",
                 checkpointer=None):
    """Stream agent execution — yields (node_name, state_update) tuples."""
    app    = build_graph(checkpointer=checkpointer)
    state  = initial_state(question)
    config = {"configurable": {"thread_id": thread_id}}
    yield from app.stream(state, config=config, stream_mode="updates")
