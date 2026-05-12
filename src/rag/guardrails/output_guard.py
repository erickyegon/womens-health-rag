"""
Output Guardrails — Episode 16
================================
Detects hallucinations and low-confidence answers before returning to user.
"""
from __future__ import annotations
import json, logging, re
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag.config.prompts import HALLUCINATION_PROMPT
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OutputGuardResult:
    grounded:           bool
    unsupported_claims: list[str] = field(default_factory=list)
    confidence_score:   float     = 1.0
    flagged:            bool      = False
    flag_reason:        str       = ""


class OutputGuard:
    """
    Episode 16 demo:
        guard = OutputGuard()
        result = guard.check(answer, retrieved_docs)
        if not result.grounded:
            print("Unsupported:", result.unsupported_claims)
    """
    def __init__(self, min_confidence: float = 0.3):
        self.settings       = get_settings()
        self.min_confidence = min_confidence
        self._llm = ChatOpenAI(
            model=self.settings.openai_model, temperature=0,
            openai_api_key=self.settings.openai_api_key.get_secret_value())  # type: ignore

    def check(self, answer: str, documents: list[Document]) -> OutputGuardResult:
        if not answer.strip():
            return OutputGuardResult(grounded=False, flagged=True,
                                     flag_reason="Empty answer")
        docs_text = "\n\n".join(
            f"[Source {i+1}] {d.page_content[:300]}"
            for i, d in enumerate(documents[:5]))
        try:
            resp = self._llm.invoke(
                HALLUCINATION_PROMPT.format_messages(
                    answer=answer, documents=docs_text))
            text = resp.content.strip()
            text = re.sub(r"```(?:json)?\n?", "", text).strip()
            data = json.loads(text)
            unsupported = data.get("unsupported_claims", [])
            grounded    = data.get("grounded", True)
            confidence  = 1.0 - (len(unsupported) * 0.2)
            flagged     = not grounded or confidence < self.min_confidence
            return OutputGuardResult(
                grounded=grounded,
                unsupported_claims=unsupported,
                confidence_score=max(0.0, confidence),
                flagged=flagged,
                flag_reason=f"{len(unsupported)} unsupported claim(s)" if not grounded else "",
            )
        except Exception as e:
            logger.warning("Output guard error: %s", e)
            return OutputGuardResult(grounded=True)  # fail open
