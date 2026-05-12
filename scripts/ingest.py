"""
Ingestion Script — run with: make ingest  OR  python scripts/ingest.py

Runs the complete pipeline:
    data/raw/*.pdf
        → loader.py    (PDF extraction)
        → cleaner.py   (text normalisation)
        → chunker.py   (splitting)
        → embedder.py  (vectorisation)
        → indexer.py   (pgvector upsert)

Accepts optional CLI flags for strategy selection and dry-run mode.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add src/ to path so we can import rag.*
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.ingestion.chunker import ChunkStrategy, chunk_pages, chunk_stats
from rag.ingestion.cleaner import clean_pages
from rag.ingestion.embedder import Embedder, EmbedderBackend
from rag.ingestion.indexer import VectorIndex
from rag.ingestion.loader import load_directory

app     = typer.Typer(help="Ingest PDFs into the womens-health-rag vector index.")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@app.command()
def main(
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir", "-d",
        help="Directory containing PDF files.",
    ),
    metadata_file: Path | None = typer.Option(
        None,
        "--metadata", "-m",
        help="JSON file mapping filename stems to metadata (country, year, etc.).",
    ),
    strategy: ChunkStrategy = typer.Option(
        ChunkStrategy.RECURSIVE,
        "--strategy", "-s",
        help="Chunking strategy: fixed | recursive | semantic.",
    ),
    backend: EmbedderBackend = typer.Option(
        EmbedderBackend.OPENAI,
        "--backend", "-b",
        help="Embedding backend: openai | onnx.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run pipeline without writing to the database.",
    ),
    drop_existing: bool = typer.Option(
        False,
        "--drop",
        help="Drop and recreate the index table before ingestion. USE WITH CAUTION.",
    ),
    multimodal: bool = typer.Option(
        False,
        "--multimodal",
        help="[Episode 2B] Use Docling + GPT-4o vision + table-to-prose. Requires: uv add docling",
    ),
    vision_flag: bool = typer.Option(
        True,
        "--vision/--no-vision",
        help="Enable GPT-4o figure descriptions (only active with --multimodal).",
    ),
) -> None:
    """
    Run the full PDF ingestion pipeline.

    Basic (Episodes 1-2):      python scripts/ingest.py --metadata data/metadata.json
    Multimodal (Episode 2B):   python scripts/ingest.py --multimodal --metadata data/metadata.json
    Docling only (no vision):  python scripts/ingest.py --multimodal --no-vision
    Dry run:                   python scripts/ingest.py --dry-run
    """
    console.rule("[bold teal]Women's Health RAG — Ingestion Pipeline[/]")

    if multimodal:
        console.print("[bold magenta]🔬 Multimodal mode:[/] Docling + GPT-4o Vision + Table Prose")

    # ── Load metadata map ─────────────────────────────────────────────────────
    metadata_map: dict = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file) as f:
            metadata_map = json.load(f)
        console.print(f"[green]✓[/] Loaded metadata for {len(metadata_map)} documents")

    start = time.perf_counter()

    # ── Step 1: Load PDFs ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/5:[/] Loading PDFs from", str(data_dir))

    if multimodal:
        # Multimodal path: Docling + Vision + Prose
        from rag.ingestion.multimodal_loader import MultimodalLoader, element_stats
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            t = prog.add_task("Multimodal loading (Docling + Vision)...", total=None)
            mm_loader = MultimodalLoader(vision_enabled=vision_flag, table_prose=True)
            elements  = mm_loader.load_directory(data_dir, metadata_map=metadata_map)
            docs      = mm_loader.to_documents(elements)
            prog.update(t, completed=True)

        stats = element_stats(elements)
        console.print(f"  [green]✓[/] Loaded [bold]{stats['total']}[/] elements → [bold]{len(docs)}[/] documents")
        console.print(f"     Text: {stats['text']} | Tables: {stats['tables']} (×2 w/prose) | Figures: {stats['figures']}")

        # Skip basic clean/chunk steps — multimodal docs are already structured
        if dry_run:
            console.print("\n[yellow]Dry run — no data written to database.[/]")
            return

        embedder = Embedder(backend=backend)
        index    = VectorIndex(embedder=embedder)
        if drop_existing:
            console.print("[yellow]⚠ Dropping existing index...[/]")
            index.drop_and_recreate()
        else:
            index.init_schema()

        console.print(f"\n[bold]Upserting {len(docs)} documents into pgvector...")
        batch_size = 50
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), TaskProgressColumn(), console=console) as prog:
            t = prog.add_task("Upserting...", total=len(docs))
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                index.upsert_documents(batch)
                prog.advance(t, len(batch))

        elapsed = time.perf_counter() - start
        console.rule()
        console.print(
            f"[bold green]✓ Multimodal ingestion complete[/] in {elapsed:.1f}s\n"
            f"  {len(docs)} documents indexed | {index.count()} total in database"
        )
        return
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        t = prog.add_task("Loading...", total=None)
        pages = load_directory(data_dir, metadata_map=metadata_map)
        prog.update(t, completed=True)

    console.print(f"  [green]✓[/] Loaded [bold]{len(pages)}[/] pages")

    if not pages:
        console.print("[red]No pages loaded. Exiting.[/]")
        raise typer.Exit(1)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/5:[/] Cleaning extracted text")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        t = prog.add_task("Cleaning...", total=None)
        cleaned = clean_pages(pages)
        prog.update(t, completed=True)

    skipped = len(pages) - len(cleaned)
    console.print(f"  [green]✓[/] {len(cleaned)} pages kept, {skipped} skipped (empty/noise)")

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 3/5:[/] Chunking with strategy=[bold]{strategy.value}[/]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        t = prog.add_task("Chunking...", total=None)
        docs = chunk_pages(cleaned, strategy=strategy)
        prog.update(t, completed=True)

    stats = chunk_stats(docs)
    _print_stats_table(stats)

    # ── Dry run exit ──────────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[yellow]Dry run mode — no data written to database.[/]")
        return

    # ── Step 4: Embed + Upsert ────────────────────────────────────────────────
    console.print(f"\n[bold]Step 4/5:[/] Embedding with backend=[bold]{backend.value}[/]")
    embedder = Embedder(backend=backend)
    index    = VectorIndex(embedder=embedder)

    if drop_existing:
        console.print("[yellow]⚠ Dropping existing index...[/]")
        index.drop_and_recreate()
    else:
        index.init_schema()

    console.print(f"\n[bold]Step 5/5:[/] Upserting {len(docs)} chunks into pgvector")
    batch_size = 50
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as prog:
        t = prog.add_task("Upserting...", total=len(docs))
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            index.upsert_documents(batch)
            prog.advance(t, len(batch))

    elapsed = time.perf_counter() - start
    total_in_db = index.count()

    console.rule()
    console.print(
        f"[bold green]✓ Ingestion complete[/] in {elapsed:.1f}s\n"
        f"  {len(docs)} chunks indexed | {total_in_db} total in database"
    )


def _print_stats_table(stats: dict) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total chunks",  str(stats["total_chunks"]))
    table.add_row("Avg chars",     str(stats["avg_chars"]))
    table.add_row("Min chars",     str(stats["min_chars"]))
    table.add_row("Max chars",     str(stats["max_chars"]))
    table.add_row("Total chars",   f"{stats['total_chars']:,}")
    console.print(table)


if __name__ == "__main__":
    app()
