"""
Self-Query Retriever — Episode 7
==================================
LLM extracts structured metadata filters from natural language queries.
"Data from Nigeria between 2018 and 2022" → {country: Nigeria, year: [2018,2022]}
"""
from __future__ import annotations
import json, logging, re
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag.config.settings import get_settings
from rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

EXTRACT_FILTERS_PROMPT = """Extract structured metadata filters from the query below.
Return ONLY valid JSON. Use null for fields not mentioned.

Allowed fields:
  country      string  e.g. "Nigeria", "Kenya", "Ghana", "Ethiopia"
  year         string  e.g. "2021", "2022"
  report_type  string  "dhs" or "status_of_women"

Query: {query}

JSON:"""


class SelfQueryRetriever:
    """
    Automatically extracts metadata filters from a natural language query,
    then applies them to the vector retriever.

    Episode 7 demo:
        "What does the 2022 Kenya report say about contraception?"
        → filters: {country: "Kenya", year: "2022"}
        → pgvector: WHERE country='Kenya' AND year='2022'
    """
    def __init__(self, vector_retriever: VectorRetriever | None = None):
        self.settings  = get_settings()
        self.retriever = vector_retriever or VectorRetriever()
        self._llm      = ChatOpenAI(
            model=self.settings.openai_model, temperature=0,
            openai_api_key=self.settings.openai_api_key.get_secret_value())  # type: ignore

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[Document], dict]:
        """
        Returns (documents, extracted_filters).
        The filters dict is returned for transparency / debugging in LangSmith.
        """
        filters = self._extract_filters(query)
        # Remove None values
        active  = {k: v for k, v in filters.items() if v}
        docs    = self.retriever.retrieve(query, top_k=top_k, filters=active)
        logger.info("Self-query filters: %s → %d docs", active, len(docs))
        return docs, active

    def _extract_filters(self, query: str) -> dict:
        prompt   = EXTRACT_FILTERS_PROMPT.format(query=query)
        response = self._llm.invoke(prompt)
        text     = response.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse filter JSON: %s", text)
            return {}
