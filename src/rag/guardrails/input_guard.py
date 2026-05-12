"""
Input Guardrails — Episode 16
================================
Filters incoming queries: medical advice, off-topic, PII.
Teaches: why guardrails are essential for health domain RAG.
"""
from __future__ import annotations
import json, logging, re
from dataclasses import dataclass
from typing import Literal
from langchain_openai import ChatOpenAI
from rag.config.prompts import INPUT_GUARD_PROMPT
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    classification: Literal["safe", "medical", "off_topic", "sensitive"]
    reason:         str
    allowed:        bool

    @property
    def rejection_message(self) -> str:
        if self.classification == "medical":
            return ("I'm a data analysis assistant — I can share statistics and "
                    "research findings, but I can't provide medical advice. "
                    "Please consult a qualified healthcare professional.")
        if self.classification == "off_topic":
            return ("This question falls outside my scope. I specialise in "
                    "women's health data from DHS and Status of Women reports.")
        return ""


class InputGuard:
    """
    Episode 16 demo:
        guard = InputGuard()
        result = guard.check("What drug should a pregnant woman take?")
        # → GuardResult(classification='medical', allowed=False)
    """
    def __init__(self, use_llm: bool = True):
        self.use_llm  = use_llm
        self.settings = get_settings()
        self._llm = ChatOpenAI(
            model=self.settings.openai_model, temperature=0,
            openai_api_key=self.settings.openai_api_key.get_secret_value()  # type: ignore
        ) if use_llm else None

    def check(self, question: str) -> GuardResult:
        # Fast rule-based check first (no LLM cost)
        result = self._rule_based(question)
        if result.classification != "safe":
            return result
        # LLM-based classification for ambiguous cases
        if self.use_llm:
            return self._llm_check(question)
        return result

    def _rule_based(self, question: str) -> GuardResult:
        q = question.lower()
        medical_keywords = ["dosage", "take this", "prescribe", "medication",
                            "drug", "treatment", "diagnosis", "symptoms of"]
        for kw in medical_keywords:
            if kw in q:
                return GuardResult("medical", f"Contains medical keyword: '{kw}'", False)
        return GuardResult("safe", "Passed rule-based check", True)

    def _llm_check(self, question: str) -> GuardResult:
        try:
            resp = self._llm.invoke(
                INPUT_GUARD_PROMPT.format_messages(question=question))
            text = resp.content.strip()
            text = re.sub(r"```(?:json)?\n?", "", text).strip()
            data = json.loads(text)
            cls  = data.get("classification", "safe")
            return GuardResult(
                classification=cls,
                reason=data.get("reason", ""),
                allowed=cls in ("safe", "sensitive"),
            )
        except Exception as e:
            logger.warning("Input guard LLM error: %s — defaulting to safe", e)
            return GuardResult("safe", "Guard error — defaulting safe", True)
