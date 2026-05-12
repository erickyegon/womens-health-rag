"""
RAG Chain — Episode 5
======================
First working end-to-end RAG chain using LangChain LCEL.
Teaches: pipe operator composition, streaming, context formatting.
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
    Build the base RAG chain:
        {"context": retriever | format_docs, "question": passthrough}
        → RAG_PROMPT → ChatOpenAI → StrOutputParser
    """
    settings  = get_settings()
    retriever = retriever or VectorRetriever()
    llm = ChatOpenAI(
        model=settings.openai_model, temperature=0, streaming=True,
        openai_api_key=settings.openai_api_key.get_secret_value())  # type: ignore
    chain = (
        {"context": retriever.as_langchain_retriever() | format_docs,
         "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents as a numbered context block."""
    if not docs:
        return "No relevant documents retrieved."
    parts = []
    for i, doc in enumerate(docs, 1):
        m   = doc.metadata
        hdr = (f"[Source {i}] {m.get('report_title', m.get('file_name', 'Unknown'))} "
               f"| {m.get('country','')} | {m.get('year','')} | Page {m.get('page_number','?')}")
        parts.append(f"{hdr}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def invoke(question: str, retriever: VectorRetriever | None = None) -> str:
    return build_rag_chain(retriever).invoke({"question": question})


def stream(question: str, retriever: VectorRetriever | None = None) -> Iterator[str]:
    yield from build_rag_chain(retriever).stream({"question": question})


async def astream(question: str, retriever: VectorRetriever | None = None) -> AsyncIterator[str]:
    async for token in build_rag_chain(retriever).astream({"question": question}):
        yield token
