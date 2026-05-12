"""
Checkpointer — Episode 22
===========================
Persistent state for human-in-the-loop workflows.
Teaches: LangGraph interrupt_before, SQLite checkpointing, resume from state.
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_memory_checkpointer():
    """In-memory checkpointer — state lost on restart. Good for demos."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def get_sqlite_checkpointer(db_path: str = "checkpoints/agent.db"):
    """
    SQLite checkpointer — state persists across restarts.
    Used in Episode 22 for human-in-the-loop.

    The agent saves state before the 'answer' node.
    A human can review retrieved documents and approve/modify.
    Then resume() continues from the checkpoint.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        raise ImportError("Run: uv add langgraph[sqlite]")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(db_path)


def run_with_hitl(question: str, thread_id: str = "hitl-demo",
                  auto_approve: bool = False) -> dict:
    """
    Episode 22 demo: run agent with human-in-the-loop.

    Flow:
        1. Agent runs until interrupt_before=['answer']
        2. Human sees retrieved docs, can approve or modify
        3. Agent resumes and generates the answer

    Args:
        auto_approve: Skip human review (for automated testing).
    """
    from rag.agent.graph import build_graph
    from rag.agent.state import initial_state

    checkpointer = get_sqlite_checkpointer()
    app          = build_graph(checkpointer=checkpointer, human_in_loop=True)
    state        = initial_state(question)
    config       = {"configurable": {"thread_id": thread_id}}

    # Run until interrupt
    partial = app.invoke(state, config=config)

    if not auto_approve:
        # Show docs for human review
        docs = partial.get("graded_docs", [])
        print(f"\n{'='*60}")
        print(f"HUMAN REVIEW — {len(docs)} documents retrieved")
        print(f"Question: {question}")
        for i, doc in enumerate(docs[:3], 1):
            print(f"\n[Source {i}] {doc.metadata.get('report_title','')}")
            print(doc.page_content[:300])
        approval = input("\nApprove retrieval? [y/n]: ").strip().lower()
        if approval != "y":
            logger.info("Human rejected retrieval — aborting")
            return {**partial, "answer": "Query rejected during human review."}

    # Resume
    final = app.invoke(None, config=config)
    return final
