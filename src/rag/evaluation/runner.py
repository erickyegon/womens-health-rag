"""
RAGAS Evaluation Runner — Episode 9
======================================
Runs the 30-question test set through the RAG pipeline and scores with RAGAS.
Logs scores to CSV for tracking across episodes.
Teaches: faithfulness, answer relevance, context precision, context recall.
"""
from __future__ import annotations
import csv, json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)

SCORES_FILE = Path("docs/RAGAS_scores.md")
CSV_FILE    = Path("docs/ragas_history.csv")
TEST_SET    = Path("src/rag/evaluation/test_set/questions.json")


class RAGASRunner:
    """
    Evaluates the RAG pipeline on the 30-question test set using RAGAS.

    Usage:
        runner  = RAGASRunner(rag_chain=build_rag_chain(), retriever=retriever)
        results = runner.run(tag="phase2-reranking")
        runner.save(results)
    """

    def __init__(self, rag_chain=None, retriever=None):
        self.chain     = rag_chain
        self.retriever = retriever
        self.settings  = get_settings()

    def load_test_set(self) -> list[dict]:
        if not TEST_SET.exists():
            raise FileNotFoundError(f"Test set not found: {TEST_SET}")
        return json.loads(TEST_SET.read_text())

    def run(self, tag: str = "baseline",
            question_types: list[str] | None = None) -> dict[str, Any]:
        """
        Run evaluation on the test set.

        Args:
            tag:            Label for this run (e.g. "phase2-reranking")
            question_types: Filter to specific types. None = all.

        Returns dict with keys:
            tag, timestamp, scores (faithfulness etc), per_question, n_questions
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy, context_precision,
                context_recall, faithfulness,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "RAGAS not installed. Run: uv add ragas datasets")

        questions_raw = self.load_test_set()
        if question_types:
            questions_raw = [q for q in questions_raw
                             if q.get("type") in question_types]

        # Build dataset
        data: dict[str, list] = {
            "question": [], "answer": [], "contexts": [], "ground_truth": []
        }

        logger.info("Running RAGAS on %d questions [%s]...", len(questions_raw), tag)

        for q in questions_raw:
            question = q["question"]
            logger.info("  Q: %s", question[:60])

            # Retrieve
            if self.retriever:
                docs     = self.retriever.retrieve(question)
                contexts = [d.page_content for d in docs]
            else:
                contexts = ["(no retriever configured)"]

            # Generate
            if self.chain:
                answer = self.chain.invoke({"question": question})
            else:
                answer = "(no chain configured)"

            data["question"].append(question)
            data["answer"].append(answer if isinstance(answer, str) else str(answer))
            data["contexts"].append(contexts)
            data["ground_truth"].append(q.get("ground_truth", ""))

        dataset = Dataset.from_dict(data)
        result  = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_precision, context_recall],
        )

        scores = {
            "faithfulness":       round(float(result["faithfulness"]), 4),
            "answer_relevancy":   round(float(result["answer_relevancy"]), 4),
            "context_precision":  round(float(result["context_precision"]), 4),
            "context_recall":     round(float(result["context_recall"]), 4),
        }

        return {
            "tag":          tag,
            "timestamp":    datetime.now().isoformat(),
            "scores":       scores,
            "n_questions":  len(questions_raw),
            "raw":          result,
        }

    def save(self, result: dict[str, Any]) -> None:
        """Save scores to CSV and update the markdown table."""
        self._save_csv(result)
        self._update_markdown(result)
        logger.info("Scores saved — %s", result["scores"])

    def _save_csv(self, result: dict) -> None:
        CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_header = not CSV_FILE.exists()
        with open(CSV_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["tag","timestamp","faithfulness","answer_relevancy",
                             "context_precision","context_recall","n_questions"])
            s = result["scores"]
            w.writerow([
                result["tag"], result["timestamp"],
                s["faithfulness"], s["answer_relevancy"],
                s["context_precision"], s["context_recall"],
                result["n_questions"],
            ])

    def _update_markdown(self, result: dict) -> None:
        s = result["scores"]
        row = (f"| {result['tag']} | {result['timestamp'][:10]} | "
               f"{s['faithfulness']:.3f} | {s['answer_relevancy']:.3f} | "
               f"{s['context_precision']:.3f} | {s['context_recall']:.3f} |")
        logger.info("RAGAS row: %s", row)


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO)
    tag = sys.argv[1] if len(sys.argv) > 1 else "manual-run"

    from rag.retrieval.vector_retriever import VectorRetriever
    from rag.chains.rag_chain import build_rag_chain

    retriever = VectorRetriever()
    chain     = build_rag_chain(retriever)
    runner    = RAGASRunner(rag_chain=chain, retriever=retriever)
    result    = runner.run(tag=tag)
    runner.save(result)
    print("\nScores:")
    for k, v in result["scores"].items():
        print(f"  {k:<25} {v:.4f}")


if __name__ == "__main__":
    main()
