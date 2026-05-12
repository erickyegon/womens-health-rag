"""
Agent State — Episode 18
=========================
TypedDict defining the complete LangGraph agent state.
Single source of truth — all nodes read/write this schema.
"""
from __future__ import annotations
from typing import Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(dict):
    """
    LangGraph state schema for the Women's Health RAG agent.

    Fields:
        question        Original user question
        chat_history    Conversation history (accumulated via add_messages)
        strategy        Routing decision: direct|vector|hybrid|multihop
        query           Active search query (may be rewritten)
        rewrites        Number of query rewrites so far (loop budget)
        documents       Retrieved document chunks
        graded_docs     Documents after relevance grading
        answer          Generated answer string
        structured      Structured RAGResponse (if using structured chain)
        grounded        Hallucination check result
        sources         Source citations for display
        filters         Active metadata filters {country, year, ...}
        metadata        Arbitrary telemetry for LangSmith
    """
    question:     str
    chat_history: Annotated[list[BaseMessage], add_messages]
    strategy:     Literal["direct", "vector", "hybrid", "multihop"] | None
    query:        str | None
    rewrites:     int
    documents:    list[Document]
    graded_docs:  list[Document]
    answer:       str | None
    grounded:     bool | None
    sources:      list[dict]
    filters:      dict
    metadata:     dict


def initial_state(question: str, chat_history: list | None = None) -> AgentState:
    """Create a fresh state for a new query."""
    return AgentState(
        question=question,
        chat_history=chat_history or [],
        strategy=None,
        query=question,
        rewrites=0,
        documents=[],
        graded_docs=[],
        answer=None,
        grounded=None,
        sources=[],
        filters={},
        metadata={},
    )
