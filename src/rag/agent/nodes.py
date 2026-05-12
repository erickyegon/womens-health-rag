"""
Graph Nodes — Episodes 18–23
==============================
Every node in the LangGraph agent. Each is a pure function:
    AgentState → AgentState (partial update dict)

Nodes: router, retrieve, grade, rewrite, answer, hallucination_check
"""
from __future__ import annotations
import json, logging, re
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag.agent.state import AgentState
from rag.config.prompts import (
    GRADE_DOC_PROMPT, HALLUCINATION_PROMPT,
    REWRITE_PROMPT, ROUTER_PROMPT,
)
from rag.config.settings import get_settings
from rag.chains.rag_chain import format_docs
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

MAX_REWRITES = 2   # Episode 20: loop budget


def get_llm():
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_model, temperature=0,
        openai_api_key=s.openai_api_key.get_secret_value())  # type: ignore


# ── Episode 19: Router ────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify query complexity → choose retrieval strategy."""
    llm  = get_llm()
    resp = llm.invoke(ROUTER_PROMPT.format_messages(question=state["question"]))
    text = resp.content.strip()
    text = re.sub(r"```(?:json)?\n?", "", text).strip()
    try:
        data     = json.loads(text)
        strategy = data.get("strategy", "vector")
        reason   = data.get("reason", "")
    except Exception:
        strategy, reason = "vector", "parse error"

    logger.info("Router: %s → %s (%s)", state["question"][:50], strategy, reason)
    return {
        "strategy": strategy,
        "metadata": {**state.get("metadata", {}),
                     "router_strategy": strategy,
                     "router_reason": reason},
    }


# ── Episode 19: Retrieve ──────────────────────────────────────────────────────

def retrieve_node(state: AgentState,
                  retriever: VectorRetriever | None = None) -> dict:
    """Retrieve documents based on the chosen strategy."""
    ret     = retriever or VectorRetriever()
    query   = state.get("query") or state["question"]
    filters = state.get("filters", {})
    docs    = ret.retrieve(query, filters=filters)
    logger.info("Retrieved %d docs for: %s", len(docs), query[:50])
    return {"documents": docs,
            "metadata": {**state.get("metadata", {}), "retrieved": len(docs)}}


# ── Episode 20: Grade ─────────────────────────────────────────────────────────

def grade_node(state: AgentState) -> dict:
    """
    Grade each retrieved document for relevance to the question.
    Relevant docs move to graded_docs. Irrelevant ones are dropped.
    If too few relevant docs remain, we rewrite the query.
    """
    llm      = get_llm()
    question = state["question"]
    docs     = state.get("documents", [])
    relevant = []

    for doc in docs:
        try:
            resp = llm.invoke(
                GRADE_DOC_PROMPT.format_messages(
                    question=question, document=doc.page_content[:500]))
            text = re.sub(r"```(?:json)?\n?", "", resp.content.strip()).strip()
            data = json.loads(text)
            if data.get("relevant", True):
                relevant.append(doc)
        except Exception:
            relevant.append(doc)  # keep on error

    logger.info("Grader: %d/%d docs relevant", len(relevant), len(docs))
    return {
        "graded_docs": relevant,
        "metadata": {**state.get("metadata", {}),
                     "graded_relevant": len(relevant),
                     "graded_total":    len(docs)},
    }


# ── Episode 20: Rewrite ───────────────────────────────────────────────────────

def rewrite_node(state: AgentState) -> dict:
    """Rewrite the query when grading finds insufficient relevant docs."""
    llm      = get_llm()
    question = state["question"]
    rewrites = state.get("rewrites", 0) + 1
    new_q    = llm.invoke(
        REWRITE_PROMPT.format_messages(question=question)).content.strip()
    logger.info("Rewrite #%d: %s → %s", rewrites, question[:40], new_q[:40])
    return {
        "query":   new_q,
        "rewrites": rewrites,
        "metadata": {**state.get("metadata", {}),
                     f"rewrite_{rewrites}": new_q},
    }


# ── Episode 20: Answer ────────────────────────────────────────────────────────

def answer_node(state: AgentState) -> dict:
    """Generate a grounded answer from the relevant documents."""
    from rag.config.prompts import RAG_PROMPT
    llm      = get_llm()
    question = state["question"]
    docs     = state.get("graded_docs") or state.get("documents", [])
    context  = format_docs(docs)

    messages = RAG_PROMPT.format_messages(
        question=question, context=context,
        chat_history=state.get("chat_history", []))
    answer   = llm.invoke(messages).content

    sources  = [
        {"n": i+1,
         "title":   d.metadata.get("report_title", ""),
         "page":    d.metadata.get("page_number"),
         "country": d.metadata.get("country", ""),
         "year":    d.metadata.get("year", "")}
        for i, d in enumerate(docs[:5])
    ]
    return {"answer": answer, "sources": sources}


# ── Episode 20: Hallucination check ──────────────────────────────────────────

def hallucination_check_node(state: AgentState) -> dict:
    """Verify the answer is grounded in retrieved documents."""
    llm     = get_llm()
    answer  = state.get("answer", "")
    docs    = state.get("graded_docs") or state.get("documents", [])
    docs_text = "\n\n".join(f"[Source {i+1}] {d.page_content[:300]}"
                            for i, d in enumerate(docs[:5]))
    try:
        resp = llm.invoke(
            HALLUCINATION_PROMPT.format_messages(
                answer=answer, documents=docs_text))
        text     = re.sub(r"```(?:json)?\n?", "", resp.content.strip()).strip()
        data     = json.loads(text)
        grounded = data.get("grounded", True)
    except Exception:
        grounded = True   # fail open

    logger.info("Hallucination check: grounded=%s", grounded)
    return {"grounded": grounded,
            "metadata": {**state.get("metadata", {}), "grounded": grounded}}


# ── Episode 22: Direct answer (no retrieval) ──────────────────────────────────

def direct_answer_node(state: AgentState) -> dict:
    """Answer simple questions directly without retrieval."""
    llm = get_llm()
    ans = llm.invoke(state["question"]).content
    return {"answer": ans, "sources": [], "graded_docs": []}


# ── Routing edges ──────────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Edge: router_node → (direct | retrieve)"""
    return "direct" if state.get("strategy") == "direct" else "retrieve"


def route_after_grade(state: AgentState) -> str:
    """Edge: grade_node → (answer | rewrite)"""
    relevant = state.get("graded_docs", [])
    rewrites = state.get("rewrites", 0)
    if relevant or rewrites >= MAX_REWRITES:
        return "answer"
    return "rewrite"


def route_after_hallucination(state: AgentState) -> str:
    """Edge: hallucination_check → (end | rewrite)"""
    grounded = state.get("grounded", True)
    rewrites = state.get("rewrites", 0)
    if grounded or rewrites >= MAX_REWRITES:
        return "end"
    return "rewrite"
