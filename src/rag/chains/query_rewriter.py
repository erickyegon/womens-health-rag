"""
Query Rewriter — Episode 11
=============================
Three strategies: direct rewrite, HyDE, multi-query.
Teaches: why user queries are bad retrieval queries and how to fix them.
"""
from __future__ import annotations
import logging
from enum import Enum
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag.config.prompts import HYDE_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT
from rag.config.settings import get_settings
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.hybrid_retriever import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RewriteStrategy(str, Enum):
    DIRECT      = "direct"       # rewrite to cleaner query
    HYDE        = "hyde"         # hypothetical document embedding
    MULTI_QUERY = "multi_query"  # N query variants, merge results


class QueryRewriter:
    """
    Rewrites queries before retrieval to improve semantic matching.

    Episode 11:
        DIRECT:  "MMR in NG 2021" → "maternal mortality ratio in Nigeria 2021"
        HYDE:    generate a hypothetical answer, embed it, retrieve similar
        MULTI:   generate 3 variants, retrieve each, merge with RRF
    """
    def __init__(self, retriever: VectorRetriever | None = None, n_queries: int = 3):
        self.settings  = get_settings()
        self.retriever = retriever or VectorRetriever()
        self.n_queries = n_queries
        self._llm = ChatOpenAI(
            model=self.settings.openai_model, temperature=0,
            openai_api_key=self.settings.openai_api_key.get_secret_value())  # type: ignore

    def retrieve(self, question: str,
                 strategy: RewriteStrategy = RewriteStrategy.DIRECT,
                 top_k: int | None = None) -> tuple[list[Document], dict]:
        """
        Returns (documents, rewrite_info) where rewrite_info shows what was done.
        """
        if strategy == RewriteStrategy.DIRECT:
            return self._direct(question, top_k)
        if strategy == RewriteStrategy.HYDE:
            return self._hyde(question, top_k)
        return self._multi_query(question, top_k)

    def _direct(self, question: str, top_k) -> tuple[list[Document], dict]:
        rewritten = self._llm.invoke(
            REWRITE_PROMPT.format_messages(question=question)).content.strip()
        docs = self.retriever.retrieve(rewritten, top_k=top_k)
        return docs, {"strategy": "direct", "rewritten": rewritten}

    def _hyde(self, question: str, top_k) -> tuple[list[Document], dict]:
        hypo_doc = self._llm.invoke(
            HYDE_PROMPT.format_messages(question=question)).content.strip()
        # Embed the hypothetical document and search
        docs = self.retriever.retrieve(hypo_doc, top_k=top_k)
        return docs, {"strategy": "hyde", "hypothetical_doc": hypo_doc}

    def _multi_query(self, question: str, top_k) -> tuple[list[Document], dict]:
        resp = self._llm.invoke(
            MULTI_QUERY_PROMPT.format_messages(
                question=question, n=self.n_queries)).content.strip()
        queries = [q.strip() for q in resp.split("\n") if q.strip()][:self.n_queries]
        all_results = [self.retriever.retrieve(q, top_k=top_k or 10) for q in queries]
        fused = reciprocal_rank_fusion(all_results, top_n=top_k or 20)
        return fused, {"strategy": "multi_query", "queries": queries}
