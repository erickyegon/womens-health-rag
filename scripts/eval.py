"""
Evaluation Script — run with: make eval  OR  python scripts/eval.py [tag]

Runs the 30-question RAGAS evaluation suite and logs scores.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

from rag.retrieval.vector_retriever import VectorRetriever
from rag.chains.rag_chain import build_rag_chain
from rag.evaluation.runner import RAGASRunner

tag = sys.argv[1] if len(sys.argv) > 1 else "manual"

print(f"Running RAGAS evaluation [{tag}]...")
retriever = VectorRetriever()
chain     = build_rag_chain(retriever)
runner    = RAGASRunner(rag_chain=chain, retriever=retriever)
result    = runner.run(tag=tag)
runner.save(result)

print("\nScores:")
for k, v in result["scores"].items():
    print(f"  {k:<25} {v:.4f}")
