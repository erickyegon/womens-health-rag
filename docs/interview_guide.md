# Interview Guide — Presenting This Project

This guide helps you tell the story of this project in AI engineering interviews.
The best answers are specific, measurable, and show production thinking.

---

## The 2-minute summary

> "I built a production RAG system for querying global women's health data —
> specifically Demographic Health Survey reports from 90+ countries.
> The system uses a self-correcting LangGraph agent with adaptive query routing,
> two-stage retrieval with Cohere reranking, and full LangSmith observability.
> I evaluated it with RAGAS throughout and improved context precision from X to Y.
> It's deployed on Fly.io — here's the live URL."

---

## Common interview questions and strong answers

### "Walk me through your RAG architecture."

Start from the user's query and trace through every component:

1. **Query router** — LangGraph node classifies the query as simple/complex/multi-hop
2. **Hybrid retrieval** — vector search (pgvector cosine) + BM25 keyword search fused with RRF
3. **Reranking** — Cohere Rerank reduces 20 chunks to 5 highest-quality
4. **Document grading** — LLM scores each chunk for relevance; irrelevant chunks trigger query rewrite
5. **Answer generation** — grounded only in trusted chunks; Pydantic model returns typed answer + citations
6. **Hallucination check** — LLM-as-judge verifies claims against source chunks

Then show the LangSmith trace for a real query.

### "How did you handle chunking?"

"I compared three strategies on the same DHS report:
- Fixed 800-char splitting — fast but cuts through sentences
- Recursive splitting — tries paragraph → sentence → word boundaries
- Parent-child retrieval — retrieves small chunks, passes parent paragraph to the LLM

The parent-child approach improved RAGAS context precision by the largest margin,
because the LLM gets more context while retrieval still uses focused chunks."

### "How do you evaluate RAG quality?"

"I use RAGAS from Episode 9 onward with a 30-question golden test set.
The four metrics I track are faithfulness, answer relevance, context precision,
and context recall. I set up a GitHub Actions workflow that runs the eval suite
on every PR to main and fails the merge if scores drop — the same pattern
production ML teams use for regression testing."

### "What was your biggest RAG quality improvement?"

"Reranking. Adding Cohere Rerank as a second stage after vector search
gave the largest single RAGAS score improvement of any Phase 2 technique.
The intuition is: vector retrieval casts a wide net, but cosine similarity
is a weak signal for relevance. A cross-encoder sees the full query + chunk pair
and scores relevance much more accurately."

### "How do you handle hallucinations?"

"Three layers:
1. System prompt — explicitly instructs the model to only use provided context
2. Pydantic citations — the model must reference specific source chunks
3. LangGraph hallucination checker — an LLM-as-judge node verifies each claim
   against the retrieved documents before the answer is returned

If the hallucination check fails, the agent rewrites the query and retries.
LangSmith shows every retry in the trace."

### "Why LangGraph over plain LangChain chains?"

"Chains are fixed sequences — A→B→C. LangGraph is a directed cyclic graph
that can loop, branch, and recover from failures. In production, you need:
- Conditional routing (simple query vs multi-hop vs web search)
- Retry loops when retrieval quality is poor
- Human-in-the-loop checkpoints for sensitive queries
- Persistent state across long conversations

None of that is possible with a linear chain. LangGraph also integrates
directly with LangSmith for full observability — you can see exactly which
node made a wrong decision and why."

### "How is it deployed?"

"Docker Compose packages the FastAPI backend, Streamlit frontend, and pgvector
into a single stack. The API uses Server-Sent Events for streaming responses.
It's deployed on Fly.io — the whole stack runs for under $20/month.
The GitHub Actions workflow builds and pushes new Docker images to GHCR
on every merge to main."

---

## Questions to ask the interviewer

These show production thinking:

- "How do you currently evaluate your RAG pipelines — do you have a golden test set?"
- "Are you using LangGraph or a custom orchestration layer for your agents?"
- "What's your current observability setup for agent traces?"
- "How do you handle retrieval quality regression when you update your embedding model?"
