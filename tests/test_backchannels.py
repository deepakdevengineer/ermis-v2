"""Backchannels and filled pauses.

CLAUDE.md section 7. Mandatory test 4 from section 54.

The exact spellings matter: these are the forms the workflow accepts, and a
near-miss spelling is a finding rather than something to normalize silently.
"""

from __future__ import annotations

import pytest

from app.ermis import rules as R
from app.ermis import tags as T
from app.ermis.findings import Code
from app.ermis.validator import validate


class TestCanonicalSpellings:
    @pytest.mark.parametrize("form,intent", [
        ("mm-hm", "closed-mouth agreement"),
        ("uh-huh", "open-mouth agreement"),
        ("hmm", "thinking"),
        ("uh", "short open-vowel hesitation"),
        ("um", "closed-lip/nasal hesitation"),
        ("mm-mm", "closed-mouth disagreement"),
        ("nuh-huh", "open-mouth disagreement"),
        ("oh", "surprise/realization"),
        ("ooh", "excitement/impressed"),
    ])
    def test_documented_form_passes_and_has_its_intent(self, assert_valid, form, intent):
        assert_valid(form)
        assert R.BACKCHANNEL_INTENT[form] == intent

    def test_all_canonical_forms_pass_in_a_sentence(self, assert_valid):
        for form in R.CANONICAL_BACKCHANNELS:
            assert_valid(f"{form} i see")

    def test_table_matches_the_documented_set(self):
        assert R.CANONICAL_BACKCHANNELS == {
            "mm-hm", "uh-huh", "hmm", "uh", "um", "mm-mm", "nuh-huh", "oh", "ooh",
        }


class TestMisspellings:
    """Every "NOT" line in section 7."""

    @pytest.mark.parametrize("wrong,right", [
        ("mhm", "mm-hm"),
        ("uh-hu", "uh-huh"),
        ("nuh-uh", "nuh-huh"),
        ("hm", "hmm"),
        ("hmmm", "hmm"),
        ("uhh", "uh"),
        ("umm", "um"),
        ("ohh", "ooh"),
    ])
    def test_single_token_misspellings(self, wrong, right, check):
        report = check(wrong, Code.BACKCHANNEL_SPELLING)
        finding = report.by_code(Code.BACKCHANNEL_SPELLING)[0]
        assert finding.suggestion == right
        assert right in finding.message

    @pytest.mark.parametrize("wrong,right", [
        ("uh huh", "uh-huh"),
        ("mm hm", "mm-hm"),
        ("mm mm", "mm-mm"),
        ("nuh uh", "nuh-huh"),
    ])
    def test_two_word_forms_must_be_hyphenated(self, wrong, right, check):
        report = check(wrong, Code.BACKCHANNEL_SPELLING)
        assert report.by_code(Code.BACKCHANNEL_SPELLING)[0].suggestion == right

    def test_misspelling_in_context(self, check):
        check("mhm i agree", Code.BACKCHANNEL_SPELLING)
        check("i said uh huh to him", Code.BACKCHANNEL_SPELLING)

    def test_two_word_match_requires_adjacency(self, check):
        # A tag between the two words means they are not one backchannel.
        check("uh [hn] huh", absent=Code.BACKCHANNEL_SPELLING)

    def test_message_mentions_the_fp_fallback(self, check):
        report = check("mhm", Code.BACKCHANNEL_SPELLING)
        assert "[fp]" in report.by_code(Code.BACKCHANNEL_SPELLING)[0].message

    def test_one_finding_per_misspelling(self):
        # 'hmmm' is both a backchannel misspelling and a stretched spelling.
        # It should be reported once, as the more specific of the two.
        report = validate("hmmm")
        assert len(report.errors) == 1
        assert report.has(Code.BACKCHANNEL_SPELLING)
        assert not report.has(Code.STRETCHED_LETTERS)


class TestFilledPause:
    def test_fp_tag_valid(self, assert_valid):
        assert_valid("[fp]")
        assert_valid("i want [fp] that song")

    def test_fp_is_the_fallback_for_an_undeterminable_vowel(self):
        # Section 7: use [fp] when the exact filled pause cannot be determined.
        assert T.FILLED_PAUSE == "[fp]"
        assert validate("[fp] play music").is_valid

    def test_two_back_to_back_fp_collapse_to_one(self, check):
        # Section 7: if two filled pauses occur back-to-back use one [fp].
        report = check("[fp] [fp] play music", Code.DUPLICATE_TAG)
        finding = report.by_code(Code.DUPLICATE_TAG)[0]
        assert finding.suggestion == "[fp]"
        assert "filled pause" in finding.message

    def test_single_fp_passes(self, assert_valid):
        assert_valid("[fp] play music")


class TestBackchannelsAreNeverRemoved:
    """Section 7 and 56.4-56.6: these must not be treated as noise."""

    @pytest.mark.parametrize("text", [
        "mm-hm i know",
        "uh-huh that's right",
        "um i think so",
        "uh let me see",
        "hmm maybe",
        "oh really",
        "ooh nice",
        "mm-mm no thanks",
        "nuh-huh not me",
    ])
    def test_they_pass_validation(self, assert_valid, text):
        assert_valid(text)

    def test_repeated_backchannels_are_kept(self, assert_valid):
        # Repetition rules apply to words, so repeated backchannels are fine.
        assert_valid("uh uh uh i mean")
        assert_valid("mm-hm mm-hm")


class TestNonBackchannelWordsUntouched:
    @pytest.mark.parametrize("text", ["huh", "ah ha", "oh no", "so um yeah"])
    def test_ordinary_words_are_not_forced_into_the_table(self, text):
        # 'huh' on its own is not in the section 7 table, so the validator has
        # no basis to rewrite it.
        report = validate(text)
        assert not report.has(Code.UNKNOWN_TAG)
