# RAGAS Score History

This file tracks evaluation scores across all four course phases.
Updated after each phase recap episode.

## Metrics explained

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **Faithfulness** | Are all claims in the answer supported by retrieved context? | 0–1 (higher = better) |
| **Answer Relevance** | Does the answer actually address the question asked? | 0–1 |
| **Context Precision** | Are the retrieved chunks relevant to the question? | 0–1 |
| **Context Recall** | Does the retrieved context contain all necessary information? | 0–1 |

## Score table

| Phase | Episode | Faithfulness | Answer Relevance | Context Precision | Context Recall | Notes |
|-------|---------|-------------|------------------|-------------------|----------------|-------|
| Baseline | E08 | – | – | – | – | Established in Episode 9 |
| +Reranking | E10 | – | – | – | – | Cohere Rerank added |
| +Query rewrite | E11 | – | – | – | – | HyDE added |
| +Structured output | E12 | – | – | – | – | Pydantic citations |
| Phase 2 final | E17 | – | – | – | – | Full Phase 2 pipeline |
| Phase 3 final | E24 | – | – | – | – | LangGraph agent |
| Phase 4 final | E28 | – | – | – | – | Deployed system |

*Scores added as episodes are recorded.*

## Test set

30 questions covering:
- Single-hop factual queries (e.g., "What is the MMR in Nigeria 2021?")
- Multi-country comparisons (e.g., "Compare contraceptive prevalence in East Africa")
- Trend analysis (e.g., "How has under-5 mortality changed in Ghana 2014–2022?")
- Causal/relational (e.g., "How does female education affect child mortality?")
- Temporal filtering (e.g., "DHS data from West Africa between 2018 and 2022")

Full test set: [`src/rag/evaluation/test_set/questions.json`](../src/rag/evaluation/test_set/questions.json)
