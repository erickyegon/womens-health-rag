# 🌍 Women's Health RAG

> A production-grade Retrieval-Augmented Generation system for querying global women's health data — developed as a series of 28 companion notebooks.

[![CI](https://github.com/erickyegon/womens-health-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/erickyegon/womens-health-rag/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-teal.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## What this is

This system answers questions from decades of Demographic Health Survey (DHS) reports and Status of Women publications — the gold standard datasets for public health data across 90+ countries.

Ask it: *"What is the trend in maternal mortality in West Africa between 2010 and 2022?"*  
It retrieves the exact relevant passages, cites the source reports, and generates a grounded answer — never fabricating statistics.

```
Query → Adaptive router → Hybrid retrieval → Reranking → Self-correction → Grounded answer + citations
```

---

## Evaluation

The RAGAS evaluation pipeline is in `src/rag/evaluation/` (`make eval`). Scores have not yet been published for this repo; they will be added once a full run on the ingested DHS corpus is complete.

---

## Technology stack

| Layer          | Tool                          | Purpose                                      |
|----------------|-------------------------------|----------------------------------------------|
| LLM            | OpenAI GPT-4o-mini            | Answer generation                            |
| Orchestration  | LangChain + LangGraph         | Chains (Phase 1–2), Stateful agents (Phase 3)|
| Vector store   | pgvector on PostgreSQL 16     | Embedding storage and similarity search      |
| Embeddings     | OpenAI text-embedding-3-small | Cloud embeddings (1536 dims)                 |
| Local embedder | ONNX (all-MiniLM-L12-v2)      | Private/offline embedding alternative        |
| Reranking      | Cohere Rerank + cross-encoder | Two-stage retrieval quality (Phase 2)        |
| Evaluation     | RAGAS                         | Faithfulness, relevance, precision, recall   |
| Observability  | LangSmith                     | Full agent trace logging                     |
| Backend API    | FastAPI + uvicorn             | Streaming SSE endpoint (Phase 4)             |
| Frontend       | Streamlit                     | Chat UI with source citations (Phase 4)      |
| Packaging      | Docker + Docker Compose        | One-command local stack                      |
| Deployment     | Railway / Fly.io               | Live public URL                              |
| Package mgr    | uv                            | Fast, reproducible Python deps               |

---

## Quick start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker + Docker Compose (for Phase 4)
- OpenAI API key

### 1. Clone and set up

```bash
git clone https://github.com/erickyegon/womens-health-rag.git
cd womens-health-rag


# Install dependencies
make setup

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Verify the setup

```bash
python -c "
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')
print(llm.invoke('Say hello in one sentence.').content)
"
```

### 3. Download data and ingest

Follow the instructions in [`data/README.md`](data/README.md) to download DHS reports.

```bash
# Place PDFs in data/raw/
# Create data/metadata.json with country/year info

make ingest
# Or: python scripts/ingest.py --metadata data/metadata.json
```

### 4. Run the full stack (Phase 4+)

```bash
make up
# API:      http://localhost:8000
# UI:       http://localhost:8501
# Docs:     http://localhost:8000/docs
```

---

## Repository structure

```
womens-health-rag/
├── src/rag/
│   ├── config/         Settings, prompts
│   ├── ingestion/      PDF loading, cleaning, chunking, embedding, indexing
│   ├── retrieval/      Vector, BM25, hybrid, reranking, self-query
│   ├── chains/         LangChain LCEL pipelines (Phase 1–2)
│   ├── agent/          LangGraph stateful agent (Phase 3)
│   ├── evaluation/     RAGAS pipeline + golden test set
│   ├── guardrails/     Input/output safety filters
│   ├── api/            FastAPI backend
│   └── ui/             Streamlit frontend
├── tests/              pytest — unit, integration, e2e
├── notebooks/          Episode companion notebooks
├── scripts/            CLI tools: ingest, eval, seed
├── docker/             Dockerfiles + init.sql
├── docs/               Architecture, RAGAS scores, deployment guide
└── data/               PDFs (not committed — see data/README.md)
```

---

## Episode guide

| Episode | Title | Notebook |
|---------|-------|----------|
| E01 | Why RAG? Environment setup | [episode_01_intro.ipynb](notebooks/episode_01_intro.ipynb) |
| E02 | Document ingestion & chunking | [episode_02_chunking.ipynb](notebooks/episode_02_chunking.ipynb) |
| E02b | Multimodal ingestion (supplementary) | [episode_02b_multimodal_ingestion.ipynb](notebooks/episode_02b_multimodal_ingestion.ipynb) |
| E03 | Embeddings: OpenAI vs ONNX | [episode_03_embeddings__turning_text_into.ipynb](notebooks/episode_03_embeddings__turning_text_into.ipynb) |
| E04 | pgvector: production vector storage | [episode_04_pgvector__production_vector_st.ipynb](notebooks/episode_04_pgvector__production_vector_st.ipynb) |
| E05 | First RAG chain with LangChain | [episode_05_your_first_rag_chain_with_lang.ipynb](notebooks/episode_05_your_first_rag_chain_with_lang.ipynb) |
| E06 | Hybrid search: BM25 + vector | [episode_06_hybrid_search__bm25_plus_vecto.ipynb](notebooks/episode_06_hybrid_search__bm25_plus_vecto.ipynb) |
| E07 | Metadata filtering + self-query | [episode_07_metadata_filtering__selfqueryi.ipynb](notebooks/episode_07_metadata_filtering__selfqueryi.ipynb) |
| E08 | Phase 1 recap & diagnostics | [episode_08_phase_1_recap__what_we_built_a.ipynb](notebooks/episode_08_phase_1_recap__what_we_built_a.ipynb) |
| E09 | RAGAS evaluation pipeline | [episode_09_ragas_evaluation_pipeline.ipynb](notebooks/episode_09_ragas_evaluation_pipeline.ipynb) |
| E10 | Reranking with Cohere | [episode_10_reranking.ipynb](notebooks/episode_10_reranking.ipynb) |
| E11 | Query rewriting: HyDE & multi-query | [episode_11_query_rewriting.ipynb](notebooks/episode_11_query_rewriting.ipynb) |
| E12 | Structured outputs with Pydantic | [episode_12_structured_outputs.ipynb](notebooks/episode_12_structured_outputs.ipynb) |
| E13 | Multi-hop retrieval | [episode_13_multihop.ipynb](notebooks/episode_13_multihop.ipynb) |
| E14 | Conversational RAG with memory | [episode_14_conversational_rag.ipynb](notebooks/episode_14_conversational_rag.ipynb) |
| E15 | Semantic & late chunking | [episode_15_advanced_chunking.ipynb](notebooks/episode_15_advanced_chunking.ipynb) |
| E16 | Guardrails & hallucination detection | [episode_16_guardrails.ipynb](notebooks/episode_16_guardrails.ipynb) |
| E17 | Phase 2 RAGAS report card | [episode_17_phase2_recap.ipynb](notebooks/episode_17_phase2_recap.ipynb) |
| E18 | LangGraph fundamentals | [episode_18_langgraph_fundamentals.ipynb](notebooks/episode_18_langgraph_fundamentals.ipynb) |
| E19 | Adaptive retrieval routing | [episode_19_adaptive_routing.ipynb](notebooks/episode_19_adaptive_routing.ipynb) |
| E20 | Self-correcting RAG | [episode_20_self_correcting_rag.ipynb](notebooks/episode_20_self_correcting_rag.ipynb) |
| E21 | LangSmith observability | [episode_21_langsmith.ipynb](notebooks/episode_21_langsmith.ipynb) |
| E22 | Human-in-the-loop checkpoints | [episode_22_human_in_the_loop.ipynb](notebooks/episode_22_human_in_the_loop.ipynb) |
| E23 | Multi-agent systems | [episode_23_multi_agent.ipynb](notebooks/episode_23_multi_agent.ipynb) |
| E24 | Phase 3 recap | [episode_24_phase3_recap.ipynb](notebooks/episode_24_phase3_recap.ipynb) |
| E25 | FastAPI backend | [episode_25_fastapi_backend.ipynb](notebooks/episode_25_fastapi_backend.ipynb) |
| E26 | Streamlit frontend | [episode_26_streamlit_frontend.ipynb](notebooks/episode_26_streamlit_frontend.ipynb) |
| E27 | Docker & cloud deployment | [episode_27_deployment.ipynb](notebooks/episode_27_deployment.ipynb) |
| E28 | Capstone & interview guide | [episode_28_capstone.ipynb](notebooks/episode_28_capstone.ipynb) |

---

## Key commands

```bash
make setup       # Install deps + pre-commit hooks
make ingest      # Run full PDF ingestion pipeline
make eval        # Run RAGAS evaluation suite
make test        # Run full test suite with coverage
make lint        # Ruff linter + formatter check
make typecheck   # mypy type checking
make up          # Start full Docker stack
make deploy      # Deploy to Fly.io
make help        # Show all commands
```

---

## Interview guide

See [`docs/interview_guide.md`](docs/interview_guide.md) for a structured walkthrough of how to present this project in AI engineering interviews — covering the architecture decisions, RAGAS score story, and the specific talking points that land.

---

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for step-by-step instructions for deploying to:
- **Railway** — easiest, $5/month, great for demos
- **Fly.io** — more control, free tier available, production-grade

---

## Contributing

This repo is primarily a teaching resource. If you find a bug or want to improve the code:

1. Open an issue using the appropriate template
2. Fork the repo and create a feature branch
3. All PRs require CI to pass and RAGAS scores to remain stable

---

## License

MIT — you are free to fork, modify, and use this commercially.  
Attribution appreciated but not required.

---

*Built with [LangChain](https://langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [pgvector](https://github.com/pgvector/pgvector) · [RAGAS](https://docs.ragas.io)*

---

Author: Erick Kiprotich Yegon, epidemiologist and data scientist (real-world evidence, HEOR, causal inference) · Portfolio: https://erickyegon.github.io · LinkedIn: https://linkedin.com/in/erickyegon
