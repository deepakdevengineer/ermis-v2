"""Truncations and stutters.

CLAUDE.md section 6, with section 5 for the contrast against fully voiced
repetitions.

Mandatory test 19 from section 54, and cases M and N from section 55.

Note on scope: several section 6 rules cannot be checked from text alone --
whether a hyphen marks a genuine cut-off, or whether a complete-looking word
was actually truncated, is only audible. The validator checks the syntax it can
see and stays silent on the rest rather than guessing (section 59).
"""

from __future__ import annotations

import pytest

from app.ermis.findings import Code
from app.ermis.validator import validate


class TestValidTruncations:
    @pytest.mark.parametrize("text", [
        "elev-",
        "sh- song",
        "wh- wh- what",
        "play that sh- song",
        "ama- [hn] -zing",
        "-ey stop",
    ])
    def test_documented_examples_pass(self, assert_valid, text):
        assert_valid(text)

    def test_case_n_word_cut_off(self, assert_valid):
        # Case N from section 55: the speaker cuts off a word.
        assert_valid("fragment-")

    def test_continuation_may_begin_with_a_hyphen(self, assert_valid):
        # Section 6: "-ey stop".
        assert_valid("-ey stop")

    def test_self_correction_with_a_real_truncation(self, assert_valid):
        assert_valid("play that sh- song")

    def test_interrupted_word_around_a_tag(self, assert_valid):
        # Section 40's documented placement.
        assert_valid("ama- [hn] -zing")


class TestInvalidHyphens:
    def test_double_hyphen_flagged(self, check):
        # Section 6 explicitly lists "play that sh-- song" as wrong.
        report = check("play that sh-- song", Code.MULTI_HYPHEN)
        assert report.by_code(Code.MULTI_HYPHEN)[0].suggestion == "-"

    def test_triple_hyphen_flagged(self, check):
        check("sh--- song", Code.MULTI_HYPHEN)

    def test_stray_hyphen_flagged(self, check):
        report = check("play - music", Code.STRAY_HYPHEN)
        assert "sh-" in report.by_code(Code.STRAY_HYPHEN)[0].message

    def test_hyphen_offset_points_into_the_token(self):
        report = validate("play that sh-- song")
        finding = report.by_code(Code.MULTI_HYPHEN)[0]
        assert report.text[finding.start : finding.end] == "--"


class TestCompoundHyphens:
    """A hyphen is not always a truncation; compounds and brand names use one."""

    @pytest.mark.parametrize("text", [
        "twenty-three",
        "ninety-nine",
        "t-pain",
        "oral-b",
        "mm-hm",
        "uh-huh",
        "mm-mm",
        "nuh-huh",
    ])
    def test_compounds_pass(self, assert_valid, text):
        assert_valid(text)

    def test_compound_is_not_reported_as_a_truncation(self, check):
        check("twenty-three", absent=Code.MULTI_HYPHEN)
        check("twenty-three", absent=Code.STRAY_HYPHEN)


class TestRepetitionVersusTruncation:
    """Section 5 versus section 6: fully voiced repeats take no hyphen."""

    def test_case_m_whole_word_stutter_has_no_hyphen(self, assert_valid):
        assert_valid("i i i i i want to go")

    def test_fully_voiced_repetition_passes_without_hyphens(self, assert_valid):
        assert_valid("very very very much")

    def test_truncated_stutter_keeps_its_hyphens(self, assert_valid):
        assert_valid("wh- wh- what")

    def test_validator_does_not_demand_hyphens_on_repeats(self, check):
        # Section 5: do not add hyphens to fully voiced repeated words. The
        # validator must not push the annotator toward hyphenating these.
        report = validate("very very very much")
        assert not report.findings
        check("i i i i i want to go", absent=Code.STRAY_HYPHEN)


class TestAudioDependentRulesAreNotGuessed:
    """Section 6 rules that need the audio produce no text-only finding.

    The validator must not invent a truncation finding it cannot support, and
    must not clear one either. These stay for the human-confirmation workflow
    and the event cross-check (sections 47, 49).
    """

    def test_complete_looking_word_is_not_forced_to_truncate(self, assert_valid):
        # Only the audio can say whether "sh" was cut off; the text alone
        # cannot, so a plain word passes.
        assert_valid("play that song")

    def test_fast_speech_is_not_a_truncation_signal(self, assert_valid):
        # Section 6: do not add a hyphen merely because speech is fast.
        assert_valid("play that song right now really quickly")

    def test_hyphenated_fragment_is_accepted_without_second_guessing(self, assert_valid):
        # The validator trusts the annotator's ear here.
        assert_valid("sh- song")
        assert_valid("elev-")
