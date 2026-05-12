"""
Unit tests for rag.ingestion.cleaner — Episode 2

Tests every cleaning step individually, then the full pipeline.
All tests are fast — no network, no DB, no LLM calls.

Run with: make test-unit  or  pytest tests/unit/test_cleaner.py -v
"""

from __future__ import annotations

import pytest

from rag.ingestion.cleaner import (
    PIPELINE_STEPS,
    _collapse_whitespace,
    _fix_hyphenated_linebreaks,
    _normalise_unicode,
    _remove_footnote_numbers,
    _remove_junk_characters,
    _remove_repeated_headers,
    _remove_short_noise_lines,
    _remove_table_artefacts,
    clean_page,
    clean_pages,
    cleaning_report,
    step_by_step_clean,
)
from rag.ingestion.loader import RawPage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_page(text: str, country: str = "Nigeria", year: str = "2021") -> RawPage:
    return RawPage(
        text=text,
        page_number=42,
        source_file="/tmp/PR157.pdf",
        file_name="PR157.pdf",
        total_pages=350,
        country=country,
        year=year,
        report_type="dhs",
        report_title="Nigeria Demographic and Health Survey 2021",
    )


REALISTIC_DHS_TEXT = """DEMOGRAPHIC AND HEALTH SURVEY 2021

Chapter 5: Maternal Health

The maternal mortality ratio (MMR) in Nigeria declined signiﬁcantly between 2013 and 2021.
According to ﬁeld surveys,1 the MMR stood at 512 deaths per 100,000 live births.2
This represents an 11% reduction from the 2013 ﬁgure of 576.3

Skilled birth attend-
ance increased from 38% to 43% during this period.4,5 Antenatal care cov-
erage remained stable at 67%.

CHAPTER 5 | 87

Regional variation: The North West zone showed the highest MMR at 1,549,
while the South West showed 165 per 100,000 live births.

| | | | |
N/A N/A N/A N/A N/A

. . . . . . . .

87
"""


# ── Step 1: Unicode normalisation ─────────────────────────────────────────────

class TestNormaliseUnicode:

    def test_fi_ligature_replaced(self):
        assert _normalise_unicode("ﬁrst result") == "first result"

    def test_fl_ligature_replaced(self):
        assert _normalise_unicode("ﬂoor plan") == "floor plan"

    def test_ff_ligature_replaced(self):
        assert _normalise_unicode("ﬀect") == "ffect"

    def test_ffi_ligature_replaced(self):
        assert _normalise_unicode("eﬃciency") == "efficiency"

    def test_smart_quotes_replaced(self):
        result = _normalise_unicode("\u201CHello\u201D it\u2019s fine")
        assert '"Hello"' in result
        assert "it's" in result

    def test_em_dash_replaced(self):
        result = _normalise_unicode("one\u2014two")
        assert "-" in result
        assert "\u2014" not in result

    def test_non_breaking_space_replaced(self):
        result = _normalise_unicode("hello\u00A0world")
        assert result == "hello world"

    def test_zero_width_space_removed(self):
        result = _normalise_unicode("hello\u200Bworld")
        assert "\u200B" not in result
        assert "helloworld" in result

    def test_bom_removed(self):
        result = _normalise_unicode("\uFEFFsome text")
        assert "\uFEFF" not in result
        assert "some text" in result

    def test_control_characters_removed(self):
        result = _normalise_unicode("hello\x07world")
        assert "\x07" not in result
        assert "hello" in result
        assert "world" in result

    def test_plain_ascii_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog. 1234567890."
        assert _normalise_unicode(text) == text

    def test_legitimate_accented_chars_preserved(self):
        # Country names with accents must be preserved
        result = _normalise_unicode("Côte d'Ivoire and São Tomé")
        assert "Côte" in result
        assert "São" in result


# ── Step 2: Hyphenated line breaks ────────────────────────────────────────────

class TestFixHyphenatedLinebreaks:

    def test_simple_word_rejoined(self):
        assert _fix_hyphenated_linebreaks("mor-\ntality") == "mortality"

    def test_multi_syllable_word_rejoined(self):
        result = _fix_hyphenated_linebreaks("ma-\nter-\nnal")
        assert result == "maternal"

    def test_mid_sentence_hyphen_fixed(self):
        text = "skilled birth attend-\nance increased"
        result = _fix_hyphenated_linebreaks(text)
        assert "attendance" in result
        assert "-\n" not in result

    def test_year_range_not_merged(self):
        # "2020-\n2022" — digits should NOT be merged
        text = "from 2020-\n2022"
        result = _fix_hyphenated_linebreaks(text)
        assert "2020-" in result   # hyphen preserved

    def test_compound_adjective_not_merged(self):
        # Intentional hyphen at end of line with digit follows
        text = "under-\n5 mortality"
        result = _fix_hyphenated_linebreaks(text)
        # "under" ends with letter, "5" is digit — should not merge
        assert "under-" in result

    def test_no_hyphen_unchanged(self):
        text = "The maternal mortality rate was declining steadily."
        assert _fix_hyphenated_linebreaks(text) == text

    def test_lowercase_both_sides(self):
        assert _fix_hyphenated_linebreaks("sig-\nnificant") == "significant"

    def test_uppercase_preserved(self):
        result = _fix_hyphenated_linebreaks("Sub-\nSaharan")
        assert "SubSaharan" in result  # merged but case preserved


# ── Step 3: Footnote numbers ──────────────────────────────────────────────────

class TestRemoveFootnoteNumbers:

    def test_inline_footnote_removed(self):
        result = _remove_footnote_numbers("mortality rate12 was declining")
        assert "rate was" in result
        assert "12" not in result

    def test_footnote_before_comma(self):
        result = _remove_footnote_numbers("Nigeria,3 Kenya,4 Ghana5.")
        assert "Nigeria," in result
        assert "Kenya," in result
        assert ",3" not in result
        assert ",4" not in result

    def test_multi_ref_removed(self):
        result = _remove_footnote_numbers("this finding1,2,3 shows")
        assert ",2,3" not in result
        assert "this finding" in result

    def test_year_not_removed(self):
        result = _remove_footnote_numbers("In 2021 the rate")
        assert "2021" in result

    def test_percentage_not_removed(self):
        result = _remove_footnote_numbers("95% confidence interval")
        assert "95" in result

    def test_decimal_not_removed(self):
        result = _remove_footnote_numbers("rate of 3.5 per 1000")
        assert "3.5" in result

    def test_table_numbers_preserved(self):
        # "Table 3.2" — the "3" is part of a table reference
        result = _remove_footnote_numbers("see Table 3.2 for details")
        assert "3.2" in result or "Table 3" in result


# ── Step 4: Repeated headers ──────────────────────────────────────────────────

class TestRemoveRepeatedHeaders:

    def test_allcaps_header_removed(self):
        text = "DEMOGRAPHIC AND HEALTH SURVEY 2021\nSome real content here."
        result = _remove_repeated_headers(text)
        assert "DEMOGRAPHIC AND HEALTH SURVEY" not in result
        assert "Some real content here." in result

    def test_lone_page_number_removed(self):
        text = "Some content.\n87\nMore content."
        result = _remove_repeated_headers(text)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        assert "87" not in lines

    def test_three_digit_page_number_removed(self):
        text = "Content.\n234\nMore content."
        result = _remove_repeated_headers(text)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        assert "234" not in lines

    def test_chapter_pipe_footer_removed(self):
        text = "Some content.\nChapter 5 | Maternal Health | 87\nMore content."
        result = _remove_repeated_headers(text)
        assert "Chapter 5 | Maternal Health | 87" not in result

    def test_dash_divider_removed(self):
        text = "Content.\n---------\nMore content."
        result = _remove_repeated_headers(text)
        assert "-------" not in result

    def test_pipe_divider_removed(self):
        text = "Content.\n| | | | |\nMore content."
        result = _remove_repeated_headers(text)
        assert "| | | | |" not in result

    def test_yes_no_preserved(self):
        # Short uppercase data values must NOT be removed
        text = "Response: YES\nStatus: NO\nResult: N/A"
        result = _remove_repeated_headers(text)
        assert "YES" in result
        assert "NO" in result

    def test_country_code_preserved(self):
        # "NG" (Nigeria code) is 2 chars — under threshold
        text = "Country code: NG\nCountry code: KE"
        result = _remove_repeated_headers(text)
        assert "NG" in result

    def test_normal_content_untouched(self):
        text = "The maternal mortality ratio in Nigeria was 512 per 100,000 live births."
        assert _remove_repeated_headers(text) == text

    def test_empty_lines_preserved(self):
        text = "Para 1.\n\nPara 2."
        result = _remove_repeated_headers(text)
        assert "\n\n" in result


# ── Step 5: Table artefacts ───────────────────────────────────────────────────

class TestRemoveTableArtefacts:

    def test_pipe_only_line_removed(self):
        text = "Content.\n| | | | |\nMore content."
        result = _remove_table_artefacts(text)
        assert "| | | | |" not in result

    def test_dot_leader_line_removed(self):
        text = "Content.\n. . . . . .\nMore content."
        result = _remove_table_artefacts(text)
        assert ". . . . ." not in result

    def test_repeated_na_line_removed(self):
        text = "Content.\nN/A N/A N/A N/A N/A\nMore content."
        result = _remove_table_artefacts(text)
        assert "N/A N/A N/A N/A" not in result

    def test_meaningful_table_data_preserved(self):
        # A line with actual data should survive
        text = "Nigeria | 512 | 2021 | DHS\nGhana | 319 | 2022 | DHS"
        result = _remove_table_artefacts(text)
        assert "Nigeria" in result

    def test_three_repeated_tokens_preserved(self):
        # 3 repeated tokens is borderline — our threshold is 4
        text = "Yes Yes Yes"
        result = _remove_table_artefacts(text)
        assert "Yes" in result


# ── Step 6: Junk characters ───────────────────────────────────────────────────

class TestRemoveJunkCharacters:

    def test_replacement_char_removed(self):
        result = _remove_junk_characters("hello\uFFFDworld")
        assert "\uFFFD" not in result

    def test_null_bytes_removed(self):
        result = _remove_junk_characters("hello\x00world")
        assert "\x00" not in result

    def test_dot_leaders_collapsed(self):
        result = _remove_junk_characters("Chapter 5.........")
        assert "........." not in result
        assert "..." in result

    def test_normal_text_unchanged(self):
        text = "The rate was 3.5% in 2021."
        assert _remove_junk_characters(text) == text


# ── Step 7: Short noise lines ─────────────────────────────────────────────────

class TestRemoveShortNoiseLines:

    def test_single_char_line_removed(self):
        text = "Content.\na\nMore content."
        result = _remove_short_noise_lines(text)
        lines = [l for l in result.split("\n") if l.strip() == "a"]
        assert len(lines) == 0

    def test_two_char_line_removed(self):
        text = "Content.\nab\nMore content."
        result = _remove_short_noise_lines(text)
        lines = [l for l in result.split("\n") if l.strip() == "ab"]
        assert len(lines) == 0

    def test_three_char_line_preserved(self):
        # "N/A", "Yes", "GHA" are valid short values
        text = "Content.\nN/A\nMore content."
        result = _remove_short_noise_lines(text)
        assert "N/A" in result

    def test_empty_lines_preserved(self):
        text = "Para 1.\n\nPara 2."
        result = _remove_short_noise_lines(text)
        assert "\n\n" in result

    def test_normal_lines_unchanged(self):
        text = "The maternal mortality ratio was 512 per 100,000."
        assert _remove_short_noise_lines(text) == text


# ── Step 8: Whitespace collapse ───────────────────────────────────────────────

class TestCollapseWhitespace:

    def test_multiple_spaces_collapsed(self):
        result = _collapse_whitespace("word1   word2    word3")
        assert "  " not in result
        assert "word1 word2 word3" in result

    def test_three_newlines_to_two(self):
        text = "Para 1.\n\n\n\nPara 2."
        result = _collapse_whitespace(text)
        assert "\n\n\n" not in result
        assert "\n\n" in result

    def test_two_newlines_preserved(self):
        # Sacred paragraph separator — must not be collapsed to one
        text = "Para 1.\n\nPara 2."
        result = _collapse_whitespace(text)
        assert "\n\n" in result

    def test_trailing_spaces_stripped(self):
        text = "hello   \nworld   "
        result = _collapse_whitespace(text)
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_tabs_collapsed(self):
        result = _collapse_whitespace("col1\t\tcol2\t\t\tcol3")
        assert "\t\t" not in result


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestCleanPage:

    def test_returns_new_instance(self):
        page = make_page("Some text with ﬁgures and data.")
        cleaned = clean_page(page)
        assert cleaned is not page

    def test_text_is_modified(self):
        page = make_page(REALISTIC_DHS_TEXT)
        cleaned = clean_page(page)
        # Ligature fixed
        assert "ﬁ" not in cleaned.text
        # Footnote numbers removed
        assert "rate12" not in cleaned.text
        # Header removed
        assert "DEMOGRAPHIC AND HEALTH SURVEY 2021" not in cleaned.text

    def test_metadata_preserved(self):
        page = make_page("Some content", country="Nigeria", year="2021")
        cleaned = clean_page(page)
        assert cleaned.country      == "Nigeria"
        assert cleaned.year         == "2021"
        assert cleaned.report_type  == "dhs"
        assert cleaned.page_number  == 42
        assert cleaned.file_name    == "PR157.pdf"

    def test_hyphen_break_fixed(self):
        page = make_page("attend-\nance increased")
        cleaned = clean_page(page)
        assert "attendance" in cleaned.text

    def test_full_realistic_page(self):
        page = make_page(REALISTIC_DHS_TEXT)
        cleaned = clean_page(page)
        # Should still contain the real content
        assert "512" in cleaned.text
        assert "maternal mortality" in cleaned.text.lower() or "MMR" in cleaned.text

    def test_is_empty_property_works(self):
        empty_page = make_page("   \n\n   ")
        cleaned = clean_page(empty_page)
        assert cleaned.is_empty

    def test_not_empty_after_cleaning_real_content(self):
        page = make_page(REALISTIC_DHS_TEXT)
        cleaned = clean_page(page)
        assert not cleaned.is_empty


class TestCleanPages:

    def test_filters_empty_pages(self):
        pages = [
            make_page(REALISTIC_DHS_TEXT),          # real content — keep
            make_page("   \n\n   "),                 # empty — filter
            make_page("x"),                          # too short — filter
            make_page("Skilled birth attendance rates in Kenya were 62%."),  # keep
        ]
        cleaned = clean_pages(pages, min_chars=80)
        assert len(cleaned) == 2

    def test_order_preserved(self):
        pages = [make_page(f"This is page {i} with enough content to pass the filter threshold.") for i in range(5)]
        cleaned = clean_pages(pages)
        for i, page in enumerate(cleaned):
            assert str(i) in page.text

    def test_empty_input(self):
        assert clean_pages([]) == []

    def test_custom_min_chars(self):
        pages = [
            make_page("Short."),    # 6 chars — filtered at min_chars=10
            make_page(REALISTIC_DHS_TEXT),
        ]
        cleaned = clean_pages(pages, min_chars=10)
        assert len(cleaned) == 1

    def test_all_filtered_returns_empty(self):
        pages = [make_page("   "), make_page("\n\n")]
        cleaned = clean_pages(pages)
        assert cleaned == []


# ── Pipeline introspection ────────────────────────────────────────────────────

class TestStepByStepClean:

    def test_returns_correct_number_of_steps(self):
        results = step_by_step_clean("Some text")
        assert len(results) == len(PIPELINE_STEPS)

    def test_result_keys_present(self):
        results = step_by_step_clean("Some text")
        for r in results:
            assert "step" in r
            assert "chars_before" in r
            assert "chars_after" in r
            assert "delta" in r
            assert "text_after" in r

    def test_chars_before_matches_previous_after(self):
        results = step_by_step_clean(REALISTIC_DHS_TEXT)
        for i in range(1, len(results)):
            assert results[i]["chars_before"] == results[i-1]["chars_after"]

    def test_delta_is_consistent(self):
        results = step_by_step_clean(REALISTIC_DHS_TEXT)
        for r in results:
            assert r["delta"] == r["chars_after"] - r["chars_before"]


class TestCleaningReport:

    def test_all_keys_present(self):
        page    = make_page(REALISTIC_DHS_TEXT)
        cleaned = clean_page(page)
        report  = cleaning_report(page, cleaned)
        expected_keys = [
            "raw_chars", "cleaned_chars", "reduction_pct",
            "raw_lines", "cleaned_lines", "lines_removed",
            "raw_preview", "cleaned_preview",
        ]
        for key in expected_keys:
            assert key in report

    def test_reduction_is_positive_for_noisy_text(self):
        page    = make_page(REALISTIC_DHS_TEXT)
        cleaned = clean_page(page)
        report  = cleaning_report(page, cleaned)
        assert report["reduction_pct"] > 0

    def test_reduction_near_zero_for_clean_text(self):
        clean_text = "The maternal mortality ratio in Nigeria was 512 per 100,000 live births in 2021."
        page    = make_page(clean_text)
        cleaned = clean_page(page)
        report  = cleaning_report(page, cleaned)
        # Should be very small — maybe 0 or tiny trailing space removal
        assert report["reduction_pct"] < 5
