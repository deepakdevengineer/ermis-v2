"""Overlapping speech and cross talk.

CLAUDE.md sections 18 (<ol>), 19 (<ct>), 20 (simultaneous speaker sounds),
22 (crowd).

Mandatory tests 17 and 18 from section 54.
"""

from __future__ import annotations

import pytest

from app.ermis.findings import Code
from app.ermis.validator import validate


# -------------------------------------------------------------- section 18
class TestOverlapBlocks:
    """Mandatory test 17 -- <ol>."""

    def test_documented_example_passes(self, assert_valid):
        assert_valid("<ol> <s1> stop <s2> stop </ol>")

    def test_overlap_inside_a_turn_passes(self, assert_valid):
        assert_valid("<s1> set a timer <ol> <s2> how long <s1> for five minutes </ol>")

    def test_simultaneous_non_lexical_sounds_pass(self, assert_valid):
        # Section 20.
        assert_valid("<ol> <s1> [laughter] <s2> [laughter] </ol>")

    def test_overlap_followed_by_cross_talk_passes(self, assert_valid):
        # Section 19: close the block first, then use <ct>.
        assert_valid("<ol> <s1> wait <s2> no </ol> <ct>")

    def test_three_speakers_in_one_overlap(self, assert_valid):
        assert_valid("<ol> <s1> a <s2> b <s3> c </ol>")

    def test_two_separate_overlap_blocks(self, assert_valid):
        assert_valid("<ol> <s1> a <s2> b </ol> ok <ol> <s1> c <s2> d </ol>")


class TestOverlapSyntaxErrors:
    def test_unclosed_block_flagged(self, check):
        check("<ol> <s1> stop <s2> stop", Code.OVERLAP_UNCLOSED)

    def test_unmatched_close_flagged(self, check):
        check("stop </ol>", Code.OVERLAP_UNMATCHED_CLOSE)

    def test_nested_blocks_flagged(self, check):
        check("<ol> <s1> a <ol> <s2> b </ol> </ol>", Code.OVERLAP_NESTED)

    def test_empty_block_flagged(self, check):
        check("<ol> </ol>", Code.OVERLAP_EMPTY)

    def test_close_before_open_flagged(self, check):
        check("</ol> <s1> a <s2> b <ol>", Code.OVERLAP_UNMATCHED_CLOSE)


class TestOverlapSpeakerTagging:
    """Section 18: every speaker must be re-tagged inside the overlap block."""

    def test_single_speaker_inside_flagged(self, check):
        report = check("<ol> <s1> stop </ol>", Code.OVERLAP_MISSING_SPEAKERS)
        assert "at least two" in report.by_code(Code.OVERLAP_MISSING_SPEAKERS)[0].message

    def test_no_speaker_tags_inside_flagged(self, check):
        check("<ol> stop stop </ol>", Code.OVERLAP_MISSING_SPEAKERS)

    def test_content_must_begin_with_a_speaker_tag(self, check):
        check("<ol> stop <s1> a <s2> b </ol>", Code.OVERLAP_NO_LEADING_SPEAKER)

    def test_leading_speaker_tag_passes(self, check):
        check("<ol> <s1> a <s2> b </ol>", absent=Code.OVERLAP_NO_LEADING_SPEAKER)

    def test_unclosed_block_content_is_still_checked(self, check):
        # An unclosed block should not hide the fact that it only names one
        # speaker.
        report = validate("<ol> <s1> stop")
        assert report.has(Code.OVERLAP_UNCLOSED)
        assert report.has(Code.OVERLAP_MISSING_SPEAKERS)


# -------------------------------------------------------------- section 19
class TestCrossTalk:
    """Mandatory test 18 -- <ct>."""

    def test_cross_talk_alone_passes(self, assert_valid):
        # Section 19: the entire clip is unintelligible overlapping speech.
        assert_valid("<ct>")

    def test_cross_talk_after_a_closed_block_passes(self, assert_valid):
        assert_valid("<ol> <s1> wait <s2> no </ol> <ct>")

    def test_cross_talk_nested_in_overlap_flagged(self, check):
        report = check("<ol> <s1> a <s2> b <ct> </ol>", Code.CROSS_TALK_IN_OVERLAP)
        message = report.by_code(Code.CROSS_TALK_IN_OVERLAP)[0].message
        assert "</ol>" in message

    def test_cross_talk_at_the_start_of_an_overlap_flagged(self, check):
        check("<ol> <ct> <s1> a <s2> b </ol>", Code.CROSS_TALK_IN_OVERLAP)

    def test_cross_talk_in_mixed_content_passes(self, assert_valid):
        assert_valid("<s1> hello <ct>")

    def test_cross_talk_is_not_for_uncertain_speaker_order(self, assert_valid):
        # Section 19 and 56.16: uncertainty about order is not cross talk, so
        # ordinary speaker tags remain valid and unflagged.
        assert_valid("<s1> a <s2> b")


class TestOverlapAndCrossTalkInteraction:
    def test_case_j_intelligible_overlap(self, assert_valid):
        # Case J from section 55.
        assert_valid("<ol> <s1> wait <s2> no </ol>")

    def test_case_k_unintelligible_overlap(self, assert_valid):
        # Case K from section 55.
        assert_valid("<ct>")

    def test_intelligible_becoming_unintelligible(self, assert_valid):
        # Section 19's documented structure for speech that degrades.
        assert_valid("<ol> <s1> wait <s2> no </ol> <ct>")

    @pytest.mark.parametrize("text", [
        "<ol> <s1> a <s2> b </ol>",
        "<s1> x <ol> <s2> a <s1> b </ol>",
        "<ol> <s1> a <s2> b </ol> <ct>",
    ])
    def test_valid_structures_stay_valid(self, assert_valid, text):
        assert_valid(text)
