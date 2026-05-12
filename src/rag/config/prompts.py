"""
Prompt Templates — used across Episodes 5–16
All prompts in one file: one place to audit, A/B test, and version.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Episode 5: Base RAG ───────────────────────────────────────────────────────
RAG_SYSTEM = """You are a specialist assistant for global women's health data.
Answer ONLY using the retrieved document excerpts provided below.

Rules:
1. Never invent statistics. If the context lacks the answer, say so clearly.
2. Cite every claim with [Source N] notation matching the numbered excerpts.
3. Express uncertainty when data is ambiguous or from different time periods.
4. Never provide medical advice — you are a data analysis assistant.
5. If the question is outside the scope of women's health data, decline politely.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])

# ── Episode 12: Structured output ────────────────────────────────────────────
STRUCTURED_RAG_SYSTEM = RAG_SYSTEM + """
Respond ONLY with a JSON object matching this schema exactly:
{
  "answer": "<your answer text with inline [Source N] citations>",
  "sources": [{"n": 1, "title": "...", "page": ..., "country": "...", "year": "..."}],
  "confidence": "high|medium|low",
  "caveat": "<any important limitation or uncertainty, or null>"
}"""

STRUCTURED_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRUCTURED_RAG_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])

# ── Episode 11: Query rewriting ──────────────────────────────────────────────
REWRITE_SYSTEM = """You are an expert at reformulating search queries for a
vector database of women's health DHS reports.

Rewrite the query to be more specific and retrieval-friendly:
- Expand abbreviations (MMR → maternal mortality ratio)
- Add domain context if implied
- Keep it concise — one clear retrieval query
- Output ONLY the rewritten query, nothing else."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM),
    ("human", "Original: {question}"),
])

HYDE_SYSTEM = """Generate a hypothetical document excerpt that would perfectly
answer the following question about women's health data.
Write it as if it were extracted from a DHS report — include plausible statistics.
Output ONLY the hypothetical excerpt, 2-3 sentences."""

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HYDE_SYSTEM),
    ("human", "{question}"),
])

MULTI_QUERY_SYSTEM = """Generate {n} different search queries that would help
retrieve documents to answer this question from a women's health database.
Each query should explore a different aspect.
Output ONLY the queries, one per line, no numbering."""

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MULTI_QUERY_SYSTEM),
    ("human", "Question: {question}"),
])

# ── Episode 18-20: LangGraph agent nodes ─────────────────────────────────────
ROUTER_SYSTEM = """Classify this query for a women's health RAG system.
Choose ONE strategy and return ONLY JSON:
{{"strategy": "direct|vector|hybrid|multihop", "reason": "<one sentence>"}}

direct   → answerable without retrieval (greetings, definitions)
vector   → single semantic search covers it
hybrid   → contains specific terms: country codes, years, acronyms (DHS, MMR)
multihop → needs multiple retrievals (causal: "how does X affect Y")"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("human", "{question}"),
])

GRADE_DOC_SYSTEM = """Grade whether this document excerpt is relevant to the question.
Return ONLY JSON: {{"relevant": true|false, "reason": "<one sentence>"}}
Be strict: tangentially related = not relevant."""

GRADE_DOC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GRADE_DOC_SYSTEM),
    ("human", "Question: {question}\n\nExcerpt:\n{document}"),
])

HALLUCINATION_SYSTEM = """Check if this answer is grounded in the source documents.
Return ONLY JSON:
{{"grounded": true|false, "unsupported_claims": ["claim1", ...]}}"""

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HALLUCINATION_SYSTEM),
    ("human", "Answer:\n{answer}\n\nSources:\n{documents}"),
])

# ── Episode 16: Guardrails ────────────────────────────────────────────────────
INPUT_GUARD_SYSTEM = """You are a content safety filter for a women's health
data assistant. Classify the input as:
  "safe"       → health data question, appropriate to answer
  "medical"    → requests specific medical advice (reject — we're data only)
  "off_topic"  → unrelated to women's health or demographics
  "sensitive"  → requires extra care (flag but answer)

Return ONLY JSON: {{"classification": "safe|medical|off_topic|sensitive",
                    "reason": "<one sentence>"}}"""

INPUT_GUARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INPUT_GUARD_SYSTEM),
    ("human", "{question}"),
])
