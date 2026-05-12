"""
RAG Chain — Episode 5

The first working end-to-end RAG chain using LangChain LCEL.
Takes a question, retrieves relevant chunks, and generates a grounded answer.

LCEL (LangChain Expression Language) uses the pipe operator (|) to compose
runnables. The chain reads left to right, which maps to how you'd describe
the pipeline in plain English:
    retrieve context → format prompt → call LLM → parse output

Episode 5 walkthrough:
    - Build the chain step by step
    - Inspect intermediate outputs with .invoke() vs .stream()
    - Observe how system prompt prevents hallucination
    - See the difference with and without retrieved context
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.config.prompts import RAG_PROMPT
from rag.config.settings import get_settings
from rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def build_rag_chain(retriever: VectorRetriever | None = None):
    """
    Build and return the base RAG chain.

    Chain architecture:
        {"context": retriever, "question": passthrough}
            → RAG_PROMPT
            → ChatOpenAI
            → StrOutputParser

    Args:
        retriever: Optional pre-built retriever. Creates one with defaults if None.

    Returns:
        A compiled LCEL chain that accepts {"question": str} and returns str.

    Episode 5 note:
        We start with StrOutputParser for simplicity.
        Episode 12 upgrades this to a Pydantic structured output parser
        that returns typed objects with citations and confidence scores.
    """
    settings  = get_settings()
    retriever = retriever or VectorRetriever()

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,         # deterministic — important for evaluation
        streaming=True,        # enables .stream() in Episode 5
        openai_api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]
    )

    chain = (
        {
            "context":  retriever.as_langchain_retriever() | _format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def invoke(question: str, retriever: VectorRetriever | None = None) -> str:
    """
    Convenience wrapper — ask a question and get a complete string answer.

    Usage:
        from rag.chains.rag_chain import invoke
        answer = invoke("What is maternal mortality in Nigeria?")
        print(answer)
    """
    chain  = build_rag_chain(retriever)
    result = chain.invoke({"question": question})
    return result


def stream(question: str, retriever: VectorRetriever | None = None) -> Iterator[str]:
    """
    Stream the answer token by token.

    Usage:
        for token in stream("What is maternal mortality in Nigeria?"):
            print(token, end="", flush=True)
    """
    chain = build_rag_chain(retriever)
    yield from chain.stream({"question": question})


async def astream(
    question: str,
    retriever: VectorRetriever | None = None,
) -> AsyncIterator[str]:
    """
    Async streaming — used by the FastAPI SSE endpoint in Phase 4.
    """
    chain = build_rag_chain(retriever)
    async for token in chain.astream({"question": question}):
        yield token


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into the context block injected into the prompt.

    Each document gets a [Source N] label and its metadata printed,
    so the LLM can use these labels in citations.

    Episode 5 insight: The quality of this formatting directly affects
    how well the LLM can construct citations. More structured = better citations.
    """
    if not docs:
        return "No relevant documents were retrieved."

    formatted = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source_label = (
            f"[Source {i}] "
            f"{meta.get('report_title', meta.get('file_name', 'Unknown'))} "
            f"| {meta.get('country', '')} "
            f"| {meta.get('year', '')} "
            f"| Page {meta.get('page_number', '?')}"
        ).strip()
        formatted.append(f"{source_label}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)
