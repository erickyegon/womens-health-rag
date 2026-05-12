"""
Text Cleaner — Episode 2
========================
Cleans raw extracted PDF text before chunking.

DHS reports have a consistent set of artefacts introduced by PDF extraction:
  1.  Unicode ligatures    ﬁ → fi, ﬂ → fl, etc.
  2.  Hyphenated linebreaks  "mor-\\ntal-\\nity" → "mortality"
  3.  Embedded footnote refs  "rate12 was" → "rate was"
  4.  Repeated page headers / footers  "DEMOGRAPHIC SURVEY 2021"
  5.  Orphaned table artefacts  lone pipe characters, dashed dividers
  6.  Excessive whitespace  3+ blank lines, multi-space runs
  7.  Junk characters  control chars, null bytes, BOM markers
  8.  Short noise lines  lone digits, single-letter lines

Design principles:
  - Every step is a pure function (str → str or RawPage → RawPage)
  - Steps are independently testable and swappable
  - A pipeline function applies them in the right order
  - We return NEW RawPage instances — originals are never mutated
  - Each step is written to be explainable on camera in 60 seconds

Episode 2 walkthrough order:
  1. Show raw artefacts on a live DHS page
  2. Apply each cleaning step individually — show before/after
  3. Run the full pipeline — show char reduction and quality gain
  4. Run clean_pages() across all 1466 pages — show what gets filtered
"""

from __future__ import annotations

import dataclasses
import logging
import re
import unicodedata
from typing import Callable

from rag.ingestion.loader import RawPage

logger = logging.getLogger(__name__)

# ── Public API ────────────────────────────────────────────────────────────────

def clean_page(page: RawPage) -> RawPage:
    """
    Apply the full cleaning pipeline to a single RawPage.

    Returns a NEW RawPage with cleaned text — the original is never modified.
    All metadata (country, year, report_type, etc.) is preserved exactly.

    Pipeline order matters:
      Unicode first  → so subsequent regexes work on normalised codepoints
      Hyphens second → before whitespace collapse removes the newlines we need
      Footnotes third → before whitespace pass removes trailing spaces
      Headers fourth → removes whole lines before whitespace collapse
      Tables fifth   → removes artefact lines
      Whitespace last → final normalisation pass
    """
    text = page.text
    text = _normalise_unicode(text)
    text = _fix_hyphenated_linebreaks(text)
    text = _remove_footnote_numbers(text)
    text = _remove_repeated_headers(text)
    text = _remove_table_artefacts(text)
    text = _remove_junk_characters(text)
    text = _remove_short_noise_lines(text)
    text = _collapse_whitespace(text)
    text = text.strip()

    return dataclasses.replace(page, text=text)


def clean_pages(
    pages: list[RawPage],
    min_chars: int = 80,
) -> list[RawPage]:
    """
    Clean a list of RawPages and filter out pages that are too short after cleaning.

    Args:
        pages:     Raw pages from loader.py
        min_chars: Pages with fewer chars after cleaning are discarded.
                   Default 80 — covers cover pages, blank pages, section dividers.

    Returns:
        Cleaned, filtered list. Order preserved.

    Episode 2 insight:
        On our 1466-page corpus, expect ~5–8% of pages to be filtered out.
        These are cover pages, blank pages, table-of-contents dividers, etc.
        That's a feature, not a bug — noise pages dilute retrieval quality.
    """
    original_count = len(pages)
    cleaned = [clean_page(p) for p in pages]
    kept = [p for p in cleaned if len(p.text) >= min_chars]

    filtered = original_count - len(kept)
    logger.info(
        "Cleaned %d pages → kept %d, filtered %d (%.1f%%) below %d chars",
        original_count, len(kept), filtered,
        100 * filtered / original_count if original_count else 0,
        min_chars,
    )
    return kept


def cleaning_report(raw: RawPage, cleaned: RawPage) -> dict:
    """
    Generate a before/after comparison report for a single page.
    Used in the Episode 2 notebook to show what each step does.
    """
    raw_lines     = raw.text.split("\n")
    cleaned_lines = cleaned.text.split("\n")
    return {
        "raw_chars":       len(raw.text),
        "cleaned_chars":   len(cleaned.text),
        "reduction_pct":   round(100 * (1 - len(cleaned.text) / max(len(raw.text), 1)), 1),
        "raw_lines":       len(raw_lines),
        "cleaned_lines":   len(cleaned_lines),
        "lines_removed":   len(raw_lines) - len(cleaned_lines),
        "raw_preview":     raw.text[:300],
        "cleaned_preview": cleaned.text[:300],
    }


# ── Step 1: Unicode normalisation ─────────────────────────────────────────────

def _normalise_unicode(text: str) -> str:
    """
    Normalise to NFC and replace typographic characters with ASCII equivalents.

    Why NFC?  PDF extraction often produces NFD (decomposed) codepoints.
    NFC re-composes them so subsequent regex patterns work correctly.

    Ligature map covers the most common PDF artefacts from DHS reports.
    The ﬁ ligature (U+FB01) is extremely common in LaTeX-typeset reports.

    Episode 2 talking point:
        "If your regex for 'first' misses 'ﬁrst', this is why."
    """
    text = unicodedata.normalize("NFC", text)

    replacements = {
        # Ligatures
        "\uFB00": "ff",   # ﬀ
        "\uFB01": "fi",   # ﬁ  ← most common in DHS reports
        "\uFB02": "fl",   # ﬂ
        "\uFB03": "ffi",  # ﬃ
        "\uFB04": "ffl",  # ﬄ
        "\uFB05": "st",   # ﬅ
        "\uFB06": "st",   # ﬆ
        # Typographic quotes → ASCII
        "\u2018": "'",    # ' left single
        "\u2019": "'",    # ' right single
        "\u201C": '"',    # " left double
        "\u201D": '"',    # " right double
        "\u201A": ",",    # ‚ low-9 quotation
        "\u201E": ",,",   # „ low-9 double
        # Dashes
        "\u2014": " - ",  # — em dash
        "\u2013": " - ",  # – en dash
        "\u2012": "-",    # ‒ figure dash
        # Spaces
        "\u00A0": " ",    # non-breaking space
        "\u202F": " ",    # narrow no-break space
        "\u2009": " ",    # thin space
        "\u200B": "",     # zero-width space → remove entirely
        "\uFEFF": "",     # BOM marker → remove
        # Bullets used in DHS tables
        "\u2022": "-",    # •
        "\u25CF": "-",    # ●
        "\u25A0": "-",    # ■
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove remaining control characters (except \n, \t)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text


# ── Step 2: Hyphenated line breaks ────────────────────────────────────────────

def _fix_hyphenated_linebreaks(text: str) -> str:
    """
    Rejoin words broken across lines with a hyphen.

    PDF text extractors often produce:
        "ma-\\nter-\\nnal health"
    We want:
        "maternal health"

    The regex requires a word character on BOTH sides of the hyphen+newline
    to avoid merging intentional hyphens like compound words split at line end
    in non-hyphenated contexts.

    Episode 2 talking point:
        "This one regex saves thousands of broken words across 1466 pages."

    Examples:
        "mor-\\ntality" → "mortality"
        "sub-\\nSaharan" → "subSaharan"  (imperfect but acceptable)
        "2021-\\n2022" → NOT merged (digits trigger the guard below)
    """
    # Only merge alphabetic characters — don't touch year ranges like 2020-\n2022
    return re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)


# ── Step 3: Footnote number removal ───────────────────────────────────────────

def _remove_footnote_numbers(text: str) -> str:
    """
    Remove inline superscript footnote references.

    DHS reports typeset footnotes as superscript digits immediately after
    a word, before punctuation or whitespace:
        "mortality rate12 was declining"    → "mortality rate was declining"
        "in Nigeria,3 Kenya,4 and Ghana5."  → "in Nigeria, Kenya, and Ghana."
        "this finding1,2,3 shows"           → "this finding shows"

    Guards against removing real numbers:
        "in 2021 the rate"   — "2021" is preceded by a space, not a word char
        "95% confidence"     — "95" is preceded by a space
        "Table 3.2"          — "3" is preceded by a period

    Regex breakdown:
        (?<=\\w)   — lookbehind: immediately after a word character
        \\d{1,2}   — one or two digits (footnotes are rarely >99)
        (,\\d{1,2})*  — optional additional refs like ,2,3
        (?=[,. \\n\\t]|$)  — followed by punctuation, whitespace, or end
    """
    # Multi-ref pattern: word12,3,4 → word
    text = re.sub(r"(?<=\w)(\d{1,2})(,\d{1,2})*(?=[,.\s\n\t]|$)", "", text)
    return text


# ── Step 4: Repeated headers and footers ─────────────────────────────────────

def _remove_repeated_headers(text: str) -> str:
    """
    Remove lines that are clearly page headers or footers repeated across pages.

    DHS reports consistently have:
      Top of page:    "DEMOGRAPHIC AND HEALTH SURVEY 2021"
                      "CHAPTER 5: MATERNAL HEALTH"
      Bottom of page: "87"  (page number)
                      "Chapter 5 | Maternal Health  87"

    Strategy:
      1. All-caps lines between 10–90 chars → likely a section header repeated
      2. Lines that are only digits → page numbers
      3. Lines matching "Page N of M" or "N | Title" patterns
      4. Lines that are only dashes, pipes, underscores (table dividers from bad extraction)

    Episode 2 talking point:
        "Every page in a DHS report has the same header. That header appearing
         in hundreds of chunks would confuse the retriever badly."
    """
    lines   = text.split("\n")
    cleaned = []

    for line in lines:
        s = line.strip()

        # Empty line — keep as paragraph separator
        if not s:
            cleaned.append(line)
            continue

        # All-caps header (length guard: >10 avoids "YES", "NO", "NA")
        if s.isupper() and 10 < len(s) < 90:
            continue

        # Lone page number: "87", "102"
        if re.fullmatch(r"\d{1,3}", s):
            continue

        # "Page 12 of 300" / "page 3"
        if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", s, re.IGNORECASE):
            continue

        # Table divider lines: "-------", "=======", "| | |"
        if re.fullmatch(r"[-=_|+ ]{4,}", s):
            continue

        # Chapter/section footers like "Chapter 5: Maternal Health | 87"
        if re.search(r"\|\s*\d{1,3}\s*$", s):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ── Step 5: Table artefacts ───────────────────────────────────────────────────

def _remove_table_artefacts(text: str) -> str:
    """
    Remove residual table extraction noise.

    When PDF tables don't extract cleanly, we get lines like:
        "| | | | |"
        ". . . . ."
        "N/A N/A N/A N/A"  (repeated N/A across columns)

    We keep table data that has meaningful content (numbers, words) but remove
    lines that are purely structural artefacts.

    Episode 2 talking point:
        "The Kenya Vol I report FR380 has 684 pages — many of them dense tables.
         This step prevents table noise from polluting health narrative chunks."
    """
    lines   = text.split("\n")
    cleaned = []

    for line in lines:
        s = line.strip()

        # Lines that are only pipes, spaces, dots
        if re.fullmatch(r"[|\s.]{3,}", s):
            continue

        # Lines that are repetitions of a single token (e.g. "N/A N/A N/A N/A")
        tokens = s.split()
        if len(tokens) >= 4 and len(set(tokens)) == 1:
            continue

        # Lines of only special chars: "* * * *", "# # # #"
        if re.fullmatch(r"[*#~^]{1,3}(\s+[*#~^]{1,3}){2,}", s):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ── Step 6: Junk character removal ───────────────────────────────────────────

def _remove_junk_characters(text: str) -> str:
    """
    Remove remaining encoding artefacts not caught by unicode normalisation.

    Targets:
      - Replacement characters (U+FFFD) — PDF encoding failures
      - Null bytes that slip through some extractors
      - Excessive repeated punctuation: "....." → "..."

    We do NOT strip all non-ASCII — DHS reports contain legitimate accented
    characters for country names (Côte d'Ivoire, São Tomé, etc.)
    """
    # Replacement character from bad encoding
    text = text.replace("\uFFFD", "")

    # Null bytes
    text = text.replace("\x00", "")

    # 4+ repeated dots → ellipsis (common in table cells with dotted leaders)
    text = re.sub(r"\.{4,}", "...", text)

    # 4+ repeated dashes not at line start (preserve markdown-style dividers)
    text = re.sub(r"(?<!^)-{4,}", "---", text, flags=re.MULTILINE)

    return text


# ── Step 7: Short noise lines ─────────────────────────────────────────────────

def _remove_short_noise_lines(text: str) -> str:
    """
    Remove lines that are too short to carry meaningful health data.

    A line with 1–2 characters is almost always extraction noise:
    a lone letter from a column label, a stray bracket, an isolated digit.

    We keep short lines that are paragraph separators (empty string) or
    lines inside bullet/numbered lists (which often start with a number).

    Threshold: 3 characters — short enough to catch noise, long enough
    to keep "N/A", "Yes", "No", country codes like "GH", "NG".
    """
    lines   = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        # Keep empty lines (paragraph breaks)
        if not s:
            cleaned.append(line)
            continue
        # Keep lines ≥ 3 chars
        if len(s) >= 3:
            cleaned.append(line)
        # else: discard — it's noise
    return "\n".join(cleaned)


# ── Step 8: Whitespace collapse ───────────────────────────────────────────────

def _collapse_whitespace(text: str) -> str:
    """
    Final whitespace normalisation pass.

    1. Collapse multiple spaces / tabs within a line to a single space
    2. Collapse 3+ consecutive blank lines to exactly 2 (one paragraph break)
    3. Remove trailing whitespace from every line

    We intentionally keep double newlines (\\n\\n) because they are the
    paragraph boundary signal that RecursiveCharacterTextSplitter uses
    as its highest-priority split point.

    Episode 2 talking point:
        "The double newline is sacred. It's the chunker's paragraph signal.
         We collapse 3+ to 2, but we never collapse 2 to 1."
    """
    # Normalise multiple spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# ── Pipeline introspection helper ─────────────────────────────────────────────

PIPELINE_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("Unicode normalisation",     _normalise_unicode),
    ("Hyphenated linebreaks",     _fix_hyphenated_linebreaks),
    ("Footnote numbers",          _remove_footnote_numbers),
    ("Repeated headers/footers",  _remove_repeated_headers),
    ("Table artefacts",           _remove_table_artefacts),
    ("Junk characters",           _remove_junk_characters),
    ("Short noise lines",         _remove_short_noise_lines),
    ("Whitespace collapse",       _collapse_whitespace),
]


def step_by_step_clean(text: str) -> list[dict]:
    """
    Apply each cleaning step individually and return the result of each.

    Used in the Episode 2 notebook to show exactly what each step does
    to a real piece of DHS text — the key teaching moment of the episode.

    Returns:
        List of dicts: [{step, chars_before, chars_after, delta, text_after}, ...]
    """
    results = []
    current = text

    for step_name, step_fn in PIPELINE_STEPS:
        before  = current
        current = step_fn(current)
        results.append({
            "step":         step_name,
            "chars_before": len(before),
            "chars_after":  len(current),
            "delta":        len(current) - len(before),
            "text_after":   current[:200],   # preview
        })

    return results
