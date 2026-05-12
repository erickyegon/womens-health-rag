"""
Multi-hop Chain — Episode 13
==============================
Decomposes compound questions into sub-questions and chains retrievals.
Teaches: "How does female education affect child mortality?" → two retrievals.
"""
from __future__ import annotations
import json, logging, re
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from rag.config.settings import get_settings
from rag.chains.rag_chain import format_docs, invoke
from rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """Break this compound question into 2-3 simpler sub-questions
that can each be answered by searching a women's health database.
Return ONLY JSON: {"sub_questions": ["q1", "q2", ...]}"""


class MultiHopChain:
    """
    Two-stage chain: decompose → retrieve each sub-question → synthesise.

    Episode 13 demo:
        "How does female education affect child mortality in Kenya?"
        → sub1: "female secondary education rates Kenya"
        → sub2: "under-5 child mortality rates Kenya"
        → synthesis: LLM combines both contexts into one answer
    """
    def __init__(self, retriever: VectorRetriever | None = None):
        self.settings  = get_settings()
        self.retriever = retriever or VectorRetriever()
        self._llm = ChatOpenAI(
            model=self.settings.openai_model, temperature=0,
            openai_api_key=self.settings.openai_api_key.get_secret_value())  # type: ignore

    def invoke(self, question: str) -> dict:
        """
        Returns:
            {"answer": str, "sub_questions": list, "sub_contexts": list}
        """
        sub_qs = self._decompose(question)
        logger.info("Multi-hop sub-questions: %s", sub_qs)

        all_docs: list[Document] = []
        sub_contexts = []
        for sq in sub_qs:
            docs = self.retriever.retrieve(sq)
            all_docs.extend(docs)
            sub_contexts.append({"question": sq, "docs": len(docs)})

        # Deduplicate by content
        seen   = set()
        unique = []
        for doc in all_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        context = format_docs(unique[:10])  # cap at 10 chunks
        synth_prompt = (
            f"Using the context below, answer this compound question comprehensively:\n\n"
            f"Question: {question}\n\n"
            f"Sub-questions addressed: {', '.join(sub_qs)}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer (cite [Source N] for every claim):"
        )
        answer = self._llm.invoke(synth_prompt).content

        return {
            "answer":        answer,
            "sub_questions": sub_qs,
            "sub_contexts":  sub_contexts,
            "docs_used":     len(unique),
        }

    def _decompose(self, question: str) -> list[str]:
        msg = [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user",   "content": question},
        ]
        resp = self._llm.invoke(msg).content.strip()
        resp = re.sub(r"```(?:json)?\n?", "", resp).strip()
        try:
            data = json.loads(resp)
            return data.get("sub_questions", [question])
        except Exception:
            return [question]
