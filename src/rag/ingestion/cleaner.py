"""
Text Cleaner — Episode 2

Cleans raw page text extracted from PDFs before chunking.
DHS reports have specific artefacts we need to handle:
- Repeated page headers / footers
- Footnote numbers embedded mid-sentence (e.g. "mortality rate12 was")
- Hyphenated line breaks ("mor-\ntality")
- Table noise when extraction fails
- Unicode ligatures (ﬁ → fi)

Design: pure functions — no state, easy to test and swap out.
"""

from __future__ import annotations

import re
import unicodedata

from rag.ingestion.loader import RawPage


def clean_page(page: RawPage) -> RawPage:
    """
    Apply the full cleaning pipeline to one page.

    Returns a NEW RawPage with cleaned text — originals are immutable.
    """
    text = page.text
    text = _normalise_unicode(text)
    text = _fix_hyphenated_linebreaks(text)
    text = _remove_footnote_numbers(text)
    text = _remove_repeated_headers(text)
    text = _collapse_whitespace(text)
    text = text.strip()

    # Return a new dataclass instance with cleaned text
    import dataclasses
    return dataclasses.replace(page, text=text)


def clean_pages(pages: list[RawPage]) -> list[RawPage]:
    """Clean a list of pages, filtering out anything that becomes empty."""
    cleaned = [clean_page(p) for p in pages]
    return [p for p in cleaned if not p.is_empty]


# ── Individual cleaning steps ─────────────────────────────────────────────────

def _normalise_unicode(text: str) -> str:
    """
    Normalise Unicode to NFC and replace common ligatures.
    DHS PDFs often contain ﬁ (U+FB01) instead of 'fi', etc.
    """
    text = unicodedata.normalize("NFC", text)
    ligature_map = {
        "\uFB00": "ff",  # ﬀ
        "\uFB01": "fi",  # ﬁ
        "\uFB02": "fl",  # ﬂ
        "\uFB03": "ffi", # ﬃ
        "\uFB04": "ffl", # ﬄ
        "\u2019": "'",   # right single quotation mark
        "\u2018": "'",   # left single quotation mark
        "\u201C": '"',   # left double quotation mark
        "\u201D": '"',   # right double quotation mark
        "\u2014": " — ", # em dash
        "\u2013": " - ", # en dash
        "\u00A0": " ",   # non-breaking space
    }
    for char, replacement in ligature_map.items():
        text = text.replace(char, replacement)
    return text


def _fix_hyphenated_linebreaks(text: str) -> str:
    """
    Rejoin words broken across lines with a hyphen.
    'mor-\\ntal-\\nity rates' → 'mortality rates'
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _remove_footnote_numbers(text: str) -> str:
    """
    Remove superscript footnote references embedded in running text.
    'rate12 was' → 'rate was'
    'mortality1,2,3' → 'mortality'

    We match digits that appear immediately after a word character
    with no space, and are followed by either a space or punctuation.
    This avoids removing real numbers like '2021' or '95%'.
    """
    return re.sub(r"(?<=\w)\d{1,2}(?=[,.\s])", "", text)


def _remove_repeated_headers(text: str) -> str:
    """
    Remove lines that appear to be repeated page headers or footers.

    DHS reports often have:
    "DEMOGRAPHIC AND HEALTH SURVEY 2021" at the top of every page.
    "Chapter 5: Maternal Health | 87" at the bottom.

    Strategy: remove lines that are ALL CAPS and fewer than 80 chars,
    or lines that match a page number pattern at the end.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip all-caps header-like lines (but not short data lines like "YES")
        if stripped.isupper() and 10 < len(stripped) < 80:
            continue
        # Skip lone page numbers
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        # Skip "Page 12 of 300" patterns
        if re.match(r"^(page\s+)?\d+\s*(of\s+\d+)?$", stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _collapse_whitespace(text: str) -> str:
    """
    Collapse multiple blank lines into at most two.
    Normalise multiple spaces to single space within lines.
    """
    # Collapse multiple spaces within a line
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
