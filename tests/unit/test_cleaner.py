"""
Unit tests for rag.ingestion.cleaner

These tests are fast (no network, no DB) and cover each cleaning step
individually. Run with: make test-unit
"""

import pytest

from rag.ingestion.cleaner import (
    _collapse_whitespace,
    _fix_hyphenated_linebreaks,
    _normalise_unicode,
    _remove_footnote_numbers,
    _remove_repeated_headers,
    clean_page,
    clean_pages,
)
from rag.ingestion.loader import RawPage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_page(text: str, page_number: int = 1) -> RawPage:
    return RawPage(
        text=text,
        page_number=page_number,
        source_file="/tmp/test.pdf",
        file_name="test.pdf",
        total_pages=10,
        country="Nigeria",
        year="2021",
        report_type="dhs",
    )


# ── Unicode normalisation ─────────────────────────────────────────────────────

class TestNormaliseUnicode:
    def test_ligatures_replaced(self):
        assert _normalise_unicode("ﬁrst ﬂoor") == "first floor"

    def test_smart_quotes_replaced(self):
        result = _normalise_unicode("\u201CHello\u201D it\u2019s")
        assert result == '"Hello" it\'s'

    def test_non_breaking_space_replaced(self):
        assert _normalise_unicode("hello\u00A0world") == "hello world"

    def test_em_dash_replaced(self):
        assert _normalise_unicode("one\u2014two") == "one — two"

    def test_plain_text_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert _normalise_unicode(text) == text


# ── Hyphenated linebreaks ─────────────────────────────────────────────────────

class TestFixHyphenatedLinebreaks:
    def test_simple_hyphenated_word(self):
        assert _fix_hyphenated_linebreaks("mor-\ntality") == "mortality"

    def test_multiple_breaks(self):
        result = _fix_hyphenated_linebreaks("ma-\nter-\nnal")
        assert result == "maternal"

    def test_intentional_hyphen_preserved(self):
        # Hyphen at end of line followed by blank line — not a word break
        text = "see table 3-\n\n4 below"
        result = _fix_hyphenated_linebreaks(text)
        # Should not merge across blank lines
        assert "3-" in result

    def test_no_false_positives(self):
        text = "2021-2022 rates"
        assert _fix_hyphenated_linebreaks(text) == text


# ── Footnote number removal ───────────────────────────────────────────────────

class TestRemoveFootnoteNumbers:
    def test_removes_superscript_after_word(self):
        result = _remove_footnote_numbers("mortality rate12 was")
        assert result == "mortality rate was"

    def test_removes_before_comma(self):
        result = _remove_footnote_numbers("Nigeria,3 Kenya,4 Ghana")
        assert result == "Nigeria, Kenya, Ghana"

    def test_preserves_real_numbers(self):
        # "2021" should NOT be stripped
        assert "2021" in _remove_footnote_numbers("In 2021 the rate")

    def test_preserves_percentages(self):
        assert "95" in _remove_footnote_numbers("95% confidence interval")

    def test_removes_end_of_sentence(self):
        result = _remove_footnote_numbers("this is true.3")
        # Footnote after period
        assert "true." in result


# ── Repeated header removal ───────────────────────────────────────────────────

class TestRemoveRepeatedHeaders:
    def test_removes_allcaps_header(self):
        text = "DEMOGRAPHIC AND HEALTH SURVEY 2021\nSome real content here."
        result = _remove_repeated_headers(text)
        assert "DEMOGRAPHIC AND HEALTH SURVEY" not in result
        assert "Some real content here." in result

    def test_removes_lone_page_number(self):
        text = "Some text.\n87\nMore text."
        result = _remove_repeated_headers(text)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        assert "87" not in lines

    def test_preserves_short_allcaps_labels(self):
        # "YES", "NO" are data values, not headers — under 10 chars
        text = "Response: YES\nResponse: NO"
        result = _remove_repeated_headers(text)
        assert "YES" in result

    def test_preserves_normal_content(self):
        text = "The maternal mortality ratio in Nigeria was 512 per 100,000 live births."
        result = _remove_repeated_headers(text)
        assert result == text


# ── Whitespace collapse ───────────────────────────────────────────────────────

class TestCollapseWhitespace:
    def test_multiple_newlines_collapsed(self):
        text = "para 1\n\n\n\n\npara 2"
        result = _collapse_whitespace(text)
        assert "\n\n\n" not in result
        assert "para 1" in result
        assert "para 2" in result

    def test_multiple_spaces_collapsed(self):
        text = "word1   word2    word3"
        result = _collapse_whitespace(text)
        assert "  " not in result

    def test_single_newlines_preserved(self):
        text = "line 1\nline 2"
        result = _collapse_whitespace(text)
        assert "\n" in result


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestCleanPage:
    def test_returns_new_page_instance(self):
        page = make_page("some text ﬁrst")
        cleaned = clean_page(page)
        assert cleaned is not page  # immutable — new object

    def test_preserves_metadata(self):
        page = make_page("Nigeria DHS data")
        cleaned = clean_page(page)
        assert cleaned.country == "Nigeria"
        assert cleaned.year == "2021"
        assert cleaned.report_type == "dhs"

    def test_cleans_text(self):
        page = make_page("ﬁrst sec-\ntion header12. DEMO HEALTH SURVEY 2021\n\n\n\nReal content.")
        cleaned = clean_page(page)
        assert "ﬁ" not in cleaned.text
        assert "sec-\ntion" not in cleaned.text


class TestCleanPages:
    def test_filters_empty_pages(self):
        pages = [
            make_page("This is a real page with substantial content that should be kept.", 1),
            make_page("   \n\n   ", 2),   # effectively empty after cleaning
            make_page("Another good page with relevant health statistics from the survey.", 3),
        ]
        cleaned = clean_pages(pages)
        assert len(cleaned) == 2  # middle page filtered out

    def test_empty_list_input(self):
        assert clean_pages([]) == []
