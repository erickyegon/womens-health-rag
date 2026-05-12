"""
Conversational RAG Chain — Episode 14
=======================================
Multi-turn chain with chat history compression.
Teaches: history management, token budgets, contextual query rewriting.
"""
from __future__ import annotations
import logging
from typing import Iterator
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from rag.config.prompts import RAG_PROMPT
from rag.config.settings import get_settings
from rag.chains.rag_chain import format_docs
from rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

CONDENSE_SYSTEM = """Given a conversation history and a follow-up question,
rewrite the follow-up to be a standalone search query that captures all
necessary context from the history.
Return ONLY the rewritten query."""


def build_conversational_chain(retriever: VectorRetriever | None = None,
                                max_history_turns: int = 5):
    """
    Conversational RAG chain.
    Input: {"question": str, "chat_history": list[BaseMessage]}
    Output: str answer token stream
    """
    settings  = get_settings()
    retriever = retriever or VectorRetriever()
    llm = ChatOpenAI(model=settings.openai_model, temperature=0, streaming=True,
                     openai_api_key=settings.openai_api_key.get_secret_value())  # type: ignore

    def condense_question(inputs: dict) -> str:
        history  = inputs.get("chat_history", [])[-max_history_turns * 2:]
        question = inputs["question"]
        if not history:
            return question
        history_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in history)
        prompt = f"History:\n{history_text}\n\nFollow-up: {question}"
        return llm.invoke(
            [{"role": "system", "content": CONDENSE_SYSTEM},
             {"role": "user",   "content": prompt}]
        ).content

    chain = (
        RunnablePassthrough.assign(
            condensed_q=RunnableLambda(condense_question),
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.retrieve(x["condensed_q"])),
            question=lambda x: x["condensed_q"],
        )
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain


class ConversationSession:
    """
    Manages a multi-turn conversation with automatic history.

    Usage:
        session = ConversationSession()
        answer  = session.ask("What is the MMR in Nigeria?")
        answer2 = session.ask("How does that compare to Kenya?")  # context preserved
    """
    def __init__(self, retriever: VectorRetriever | None = None,
                 max_turns: int = 5):
        self._chain   = build_conversational_chain(retriever, max_turns)
        self._history: list[BaseMessage] = []
        self.max_turns = max_turns

    def ask(self, question: str) -> str:
        result = self._chain.invoke(
            {"question": question, "chat_history": self._history})
        self._history.extend([
            HumanMessage(content=question),
            AIMessage(content=result),
        ])
        # Trim to max_turns
        self._history = self._history[-(self.max_turns * 2):]
        return result

    def stream(self, question: str) -> Iterator[str]:
        tokens = []
        for token in self._chain.stream(
                {"question": question, "chat_history": self._history}):
            tokens.append(token)
            yield token
        full = "".join(tokens)
        self._history.extend([
            HumanMessage(content=question),
            AIMessage(content=full),
        ])
        self._history = self._history[-(self.max_turns * 2):]

    def clear(self) -> None:
        self._history = []

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)
