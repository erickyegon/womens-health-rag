"""
Structured Output Chain — Episode 12
======================================
Returns typed Pydantic objects with citations and confidence.
Teaches: OpenAI structured outputs, citation grounding, Pydantic v2.
"""
from __future__ import annotations
import json, logging, re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from rag.config.prompts import STRUCTURED_RAG_PROMPT
from rag.config.settings import get_settings
from rag.chains.rag_chain import format_docs
from rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    n:       int    = Field(description="Source number matching [Source N] in answer")
    title:   str    = Field(description="Report title")
    page:    int | None = Field(default=None)
    country: str | None = Field(default=None)
    year:    str | None = Field(default=None)


class RAGResponse(BaseModel):
    answer:     str                         = Field(description="Answer with [Source N] citations")
    sources:    list[SourceCitation]        = Field(default_factory=list)
    confidence: Literal["high","medium","low"] = Field(default="medium")
    caveat:     str | None                  = Field(default=None)


def build_structured_chain(retriever: VectorRetriever | None = None):
    """
    Build a chain that returns RAGResponse Pydantic objects.
    Input:  {"question": str}
    Output: RAGResponse
    """
    settings  = get_settings()
    retriever = retriever or VectorRetriever()
    llm = ChatOpenAI(model=settings.openai_model, temperature=0,
                     openai_api_key=settings.openai_api_key.get_secret_value())  # type: ignore

    def run(inputs: dict) -> RAGResponse:
        question = inputs["question"]
        docs     = retriever.retrieve(question)
        context  = format_docs(docs)
        messages = STRUCTURED_RAG_PROMPT.format_messages(
            question=question, context=context)
        response = llm.invoke(messages)
        text     = response.content.strip()
        text     = re.sub(r"```(?:json)?\n?", "", text).strip()
        try:
            data = json.loads(text)
            return RAGResponse(**data)
        except Exception as e:
            logger.warning("Could not parse structured response: %s", e)
            return RAGResponse(answer=text, confidence="low")

    return run


def invoke_structured(question: str,
                       retriever: VectorRetriever | None = None) -> RAGResponse:
    return build_structured_chain(retriever)({"question": question})
