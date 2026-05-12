"""
PDF Loader — Episode 2

Loads PDFs from disk and returns raw text with page-level metadata.
Uses PyMuPDF as the primary extractor (fast, handles most PDFs well)
with pdfplumber as a fallback for PDFs where PyMuPDF struggles with
table layout or multi-column text.

Design decision: We return a list of dicts rather than LangChain Document
objects at this stage. The Document wrapper happens in chunker.py after
cleaning. This keeps the loader pure and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RawPage:
    """A single extracted page before chunking or cleaning."""

    text: str
    page_number: int           # 1-indexed
    source_file: str           # absolute path
    file_name: str             # basename
    total_pages: int

    # Enriched by the pipeline — set to empty string if unknown
    country: str = ""
    year: str = ""
    report_type: str = ""      # "dhs" | "status_of_women" | "other"
    report_title: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 50  # fewer than 50 chars = skip


def load_pdf(
    path: Path | str,
    country: str = "",
    year: str = "",
    report_type: str = "dhs",
    report_title: str = "",
    use_fallback: bool = True,
) -> list[RawPage]:
    """
    Load a PDF and return one RawPage per page.

    Args:
        path:         Path to the PDF file.
        country:      Country name or code (e.g. "Nigeria", "NGA").
        year:         Publication year as string (e.g. "2021").
        report_type:  "dhs" | "status_of_women" | "other".
        report_title: Full report title if known.
        use_fallback: If True, try pdfplumber when PyMuPDF yields < 100 chars/page.

    Returns:
        List of RawPage objects, one per page, with metadata attached.

    Episode 2 walkthrough:
        - We'll inspect the raw text quality on a real DHS report
        - Observe the messy artefacts: broken tables, repeated headers, footnote numbers
        - This motivates the cleaner.py module
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF: %s", path.name)

    pages = _load_with_pymupdf(path)

    # Fallback: if average text per page is suspiciously short, try pdfplumber
    if use_fallback and pages:
        avg_chars = sum(len(p.text) for p in pages) / len(pages)
        if avg_chars < 100:
            logger.warning(
                "PyMuPDF yielded only %.0f avg chars/page for %s — trying pdfplumber",
                avg_chars,
                path.name,
            )
            pages = _load_with_pdfplumber(path)

    # Attach metadata to all pages
    for page in pages:
        page.country      = country
        page.year         = year
        page.report_type  = report_type
        page.report_title = report_title or path.stem

    non_empty = [p for p in pages if not p.is_empty]
    logger.info(
        "Loaded %d pages (%d non-empty) from %s",
        len(pages),
        len(non_empty),
        path.name,
    )
    return non_empty


def load_directory(
    directory: Path | str,
    metadata_map: dict[str, dict] | None = None,
    glob: str = "**/*.pdf",
) -> list[RawPage]:
    """
    Load all PDFs in a directory.

    Args:
        directory:    Directory to search recursively.
        metadata_map: Map from filename stem to metadata dict.
                      Keys: country, year, report_type, report_title.
                      Example: {"nigeria_dhs_2021": {"country": "Nigeria", "year": "2021", ...}}
        glob:         Glob pattern for finding PDFs.

    Returns:
        All RawPage objects from all PDFs, sorted by file then page number.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pdf_files = sorted(directory.glob(glob))
    if not pdf_files:
        logger.warning("No PDFs found in %s matching %s", directory, glob)
        return []

    logger.info("Found %d PDFs in %s", len(pdf_files), directory)

    all_pages: list[RawPage] = []
    for pdf in pdf_files:
        meta = (metadata_map or {}).get(pdf.stem, {})
        pages = load_pdf(
            pdf,
            country      = meta.get("country", ""),
            year         = meta.get("year", ""),
            report_type  = meta.get("report_type", "dhs"),
            report_title = meta.get("report_title", ""),
        )
        all_pages.extend(pages)

    logger.info("Total pages loaded: %d", len(all_pages))
    return all_pages


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_with_pymupdf(path: Path) -> list[RawPage]:
    """Primary PDF extractor using PyMuPDF (fitz)."""
    try:
        import pymupdf  # type: ignore[import]
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: uv add pymupdf")

    pages: list[RawPage] = []
    doc = pymupdf.open(str(path))

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text extraction
        pages.append(
            RawPage(
                text        = text,
                page_number = i,
                source_file = str(path.resolve()),
                file_name   = path.name,
                total_pages = len(doc),
            )
        )

    doc.close()
    return pages


def _load_with_pdfplumber(path: Path) -> list[RawPage]:
    """Fallback PDF extractor using pdfplumber — better for tables."""
    try:
        import pdfplumber  # type: ignore[import]
    except ImportError:
        raise ImportError("pdfplumber not installed. Run: uv add pdfplumber")

    pages: list[RawPage] = []

    with pdfplumber.open(str(path)) as doc:
        total = len(doc.pages)
        for i, page in enumerate(doc.pages, start=1):
            text = page.extract_text() or ""

            # Also extract tables and append as pipe-delimited text
            tables = page.extract_tables()
            if tables:
                table_texts = []
                for table in tables:
                    rows = [
                        " | ".join(str(cell) if cell else "" for cell in row)
                        for row in table
                        if any(cell for cell in row)
                    ]
                    table_texts.append("\n".join(rows))
                text = text + "\n\n" + "\n\n".join(table_texts)

            pages.append(
                RawPage(
                    text        = text,
                    page_number = i,
                    source_file = str(path.resolve()),
                    file_name   = path.name,
                    total_pages = total,
                )
            )

    return pages
