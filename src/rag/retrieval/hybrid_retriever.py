"""
Hybrid Retriever — Episode 6
==============================
Combines vector and BM25 results using Reciprocal Rank Fusion (RRF).
Teaching: why neither vector nor keyword alone is sufficient.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from langchain_core.documents import Document
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 20,
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ  1 / (k + rank_i(d))
    where rank_i is the position in list i (1-indexed).

    k=60 is the standard constant — from the original RRF paper.

    Episode 6 talking point:
        "RRF doesn't care about the absolute scores from either system.
         A document ranked #1 by BM25 and #15 by vector gets a better
         combined score than a document ranked #5 by both. It rewards
         consistency across retrieval systems, not score magnitude."
    """
    scores: dict[str, float]    = defaultdict(float)
    docs:   dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content[:100]   # stable identifier
            scores[key] += 1.0 / (k + rank)
            if key not in docs:
                docs[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for key, score in ranked:
        doc = docs[key]
        d   = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": round(score, 6)},
        )
        results.append(d)
    return results


class HybridRetriever:
    """
    Two-stage hybrid retriever: vector + BM25 fused with RRF.

    Usage:
        hybrid = HybridRetriever(vector_retriever, bm25_retriever)
        docs   = hybrid.retrieve("maternal mortality Kenya 2022")
    """
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever:   BM25Retriever,
        vector_top_k: int = 20,
        bm25_top_k:   int = 20,
        final_top_n:  int = 20,
        rrf_k:        int = 60,
    ):
        self.vector   = vector_retriever
        self.bm25     = bm25_retriever
        self.v_top_k  = vector_top_k
        self.b_top_k  = bm25_top_k
        self.final_n  = final_top_n
        self.rrf_k    = rrf_k

    def retrieve(self, query: str, filters: dict | None = None) -> list[Document]:
        vector_docs = self.vector.retrieve(query, top_k=self.v_top_k, filters=filters)
        bm25_docs   = self.bm25.retrieve(query, top_k=self.b_top_k)
        fused       = reciprocal_rank_fusion(
            [vector_docs, bm25_docs], k=self.rrf_k, top_n=self.final_n)
        logger.info(
            "Hybrid: vector=%d bm25=%d → fused=%d",
            len(vector_docs), len(bm25_docs), len(fused),
        )
        return fused

    def as_langchain_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        from typing import Any

        outer = self

        class HybridLC(BaseRetriever):
            def _get_relevant_documents(self, query: str, **_: Any) -> list[Document]:
                return outer.retrieve(query)
            async def _aget_relevant_documents(self, query: str, **_: Any) -> list[Document]:
                return self._get_relevant_documents(query)

        return HybridLC()
