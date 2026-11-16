"""Tag inventory, placement, ordering, and retired V1 tags.

CLAUDE.md sections 8 ([hn], laughter), 9 ([artifact]), 13 (no pause tag),
14 ((())), 15 ({{}}), 20 (simultaneous sounds), 21 ([bg]), 38 (inventory),
39 (retired tags), 40 (placement), 41 (multiple tags).

Mandatory tests 5, 6, 7, 8, 14, 15, 26 from section 54.
"""

from __future__ import annotations

import pytest

from app.ermis import tags as T
from app.ermis.findings import Code
from app.ermis.validator import validate


# ------------------------------------------------------------- section 38
class TestTagInventory:
    def test_every_documented_v2_tag_is_recognized(self, assert_valid):
        # Section 38's complete fixed inventory, each in a valid context.
        assert_valid("play [fp] music")
        assert_valid("play [hn] music")
        assert_valid("play [laughter] music")
        assert_valid("play [bg] music")
        assert_valid("play [artifact] music")
        assert_valid("play (()) music")
        assert_valid("play {{music}} now")
        assert_valid("<s1> hi <s2> hello")
        assert_valid("<ol> <s1> stop <s2> stop </ol>")
        assert_valid("<ol> <s1> a <s2> b </ol> <ct>")
        for whole_file in ("<ga>", "<na>", "<ns>", "<nt>", "<sc>"):
            assert_valid(whole_file)

    @pytest.mark.parametrize("text", [
        "play [nonsense] music",
        "play <xyz> music",
        "play [speech] music",
        "play <overlap> music",
    ])
    def test_unknown_tags_flagged(self, text, check):
        check(text, Code.UNKNOWN_TAG)

    def test_tag_sets_are_disjoint(self):
        assert not T.BRACKET_TAGS & T.WHOLE_FILE_TAGS
        assert not T.STRUCTURAL_ANGLE_TAGS & T.WHOLE_FILE_TAGS

    def test_retired_tags_are_not_in_the_valid_inventory(self):
        for retired in T.RETIRED_TAGS:
            assert retired not in T.ALL_FIXED_TAGS


# --------------------------------------------------------------- section 8
class TestNonLexicalSound:
    """Mandatory test 5 -- [hn]."""

    @pytest.mark.parametrize("text", [
        "play [hn] music",
        "i'm [hn] so tired",
        "[hn] play music",
        "play music [hn]",
    ])
    def test_hn_valid_anywhere(self, assert_valid, text):
        assert_valid(text)

    def test_repeated_same_sound_uses_one_tag(self, check):
        # Case B from section 55: two coughs -> a single [hn].
        report = check("play [hn] [hn] music", Code.DUPLICATE_TAG)
        assert report.by_code(Code.DUPLICATE_TAG)[0].suggestion == "[hn]"

    def test_single_tag_for_two_coughs_passes(self, assert_valid):
        assert_valid("play [hn] music")

    def test_hn_alone_is_a_complete_transcription(self, assert_valid):
        # Section 42: a clip containing only a cough or sigh.
        assert_valid("[hn]")


# --------------------------------------------------------------- section 8
class TestLaughter:
    """Mandatory test 6 -- [laughter]."""

    def test_laughter_valid(self, assert_valid):
        assert_valid("[laughter] you're funny")
        assert_valid("that's funny [laughter]")

    def test_laughter_alone_is_a_complete_transcription(self, assert_valid):
        assert_valid("[laughter]")

    def test_laughter_is_its_own_tag_not_hn(self):
        # Section 8 and 56.7: laughter is never converted to [hn].
        assert T.LAUGHTER in T.BRACKET_TAGS
        assert T.LAUGHTER != T.NON_LEXICAL
        assert T.RETIRED_TAGS.get("[laughter]") is None

    def test_repeated_laughter_collapses(self, check):
        check("[laughter] [laughter] funny", Code.DUPLICATE_TAG)

    def test_simultaneous_laughter_uses_overlap(self, assert_valid):
        # Section 20.
        assert_valid("<ol> <s1> [laughter] <s2> [laughter] </ol>")


# --------------------------------------------------------------- section 9
class TestArtifact:
    """Mandatory test 7 -- [artifact]."""

    def test_artifact_valid(self, assert_valid):
        assert_valid("play [artifact] music")
        assert_valid("[artifact] stop")

    def test_artifact_and_garbled_are_distinct_decisions(self):
        from app.ermis.whole_file import classify_noise

        # Section 9's decision rule: can speech still be transcribed?
        assert classify_noise(speech_remains_usable=True) == T.ARTIFACT
        assert classify_noise(speech_remains_usable=False) == T.GARBLED

    def test_artifact_cannot_be_combined_with_a_whole_file_tag(self, check):
        check("<ga> [artifact]", Code.WHOLE_FILE_NOT_ALONE)


# -------------------------------------------------------------- section 21
class TestBackground:
    """Mandatory test 8 -- [bg]."""

    def test_bg_valid(self, assert_valid):
        assert_valid("play music [bg]")
        assert_valid("[bg] play music")

    def test_bg_with_laughter_as_documented(self, assert_valid):
        # Section 20 documents exactly this output for an unidentified
        # person's laughter alongside an identified speaker's.
        assert_valid("[laughter] [bg]")

    def test_bg_after_interrupted_content(self, assert_valid):
        # Section 21: place [bg] after the interrupted content.
        assert_valid("<s1> i want that [bg] <s2> ok")

    def test_duplicate_bg_flagged(self, check):
        check("play [bg] [bg] music", Code.DUPLICATE_TAG)


# -------------------------------------------------------------- section 14
class TestUnintelligible:
    """Mandatory test 14 -- (())."""

    def test_bare_marker_valid(self, assert_valid):
        assert_valid("play (())")
        assert_valid("(()) but i don't know what to get")

    def test_guess_inside_valid(self, assert_valid):
        assert_valid("play ((songs))")

    def test_partial_word_guess_valid(self, assert_valid):
        assert_valid("play ((musi-))")

    def test_single_parentheses_flagged(self, check):
        report = check("play ()", Code.SINGLE_PARENTHESES)
        assert report.by_code(Code.SINGLE_PARENTHESES)[0].suggestion == "(())"

    def test_spaces_inside_flagged(self, check):
        report = check("play (( songs ))", Code.UNINTELLIGIBLE_SPACING)
        assert report.by_code(Code.UNINTELLIGIBLE_SPACING)[0].suggestion == "((songs))"

    def test_unmatched_marker_flagged(self, check):
        check("play ((songs", Code.UNINTELLIGIBLE_MALFORMED)

    def test_guess_obeys_character_rules(self, check):
        check("play ((Songs))", Code.UPPERCASE)
        check("play ((song5))", Code.DIGIT_USED)

    def test_whole_file_unintelligible(self, assert_valid):
        # Case H from section 55.
        assert_valid("(())")


# -------------------------------------------------------------- section 15
class TestMisSpoken:
    """Mandatory test 15 -- {{}}."""

    def test_valid(self, assert_valid):
        assert_valid("{{authentication}}")
        assert_valid("i need {{authentication}} now")

    def test_empty_flagged(self, check):
        check("{{}}", Code.MISSPOKEN_EMPTY)

    def test_spaces_inside_flagged(self, check):
        report = check("{{ authentication }}", Code.MISSPOKEN_MALFORMED)
        assert report.by_code(Code.MISSPOKEN_MALFORMED)[0].suggestion == "{{authentication}}"

    def test_unmatched_flagged(self, check):
        check("{{authentication", Code.MISSPOKEN_MALFORMED)

    def test_body_obeys_character_rules(self, check):
        check("{{Authentication}}", Code.UPPERCASE)


# -------------------------------------------------------------- section 39
class TestRetiredTags:
    """Mandatory test 26 -- retired V1 tags."""

    @pytest.mark.parametrize("retired", [
        "[cough]", "[sigh]", "[throat-clear]", "[sneeze]", "[lip-smack]",
    ])
    def test_retired_sound_tags_map_to_hn(self, retired, check):
        report = check(f"play {retired} music", Code.RETIRED_TAG)
        finding = report.by_code(Code.RETIRED_TAG)[0]
        assert finding.suggestion == "[hn]"
        assert "[hn]" in finding.message

    def test_hum_maps_to_hn(self, check):
        report = check("play <hum> music", Code.RETIRED_TAG)
        assert report.by_code(Code.RETIRED_TAG)[0].suggestion == "[hn]"

    @pytest.mark.parametrize("retired", ["<i>", "<sg>", "<ws>", "<pws>", "<lv>"])
    def test_removed_v2_tags_have_no_replacement(self, retired, check):
        report = check(f"play {retired} music", Code.RETIRED_TAG)
        finding = report.by_code(Code.RETIRED_TAG)[0]
        assert finding.suggestion is None
        assert "removed in V2" in finding.message

    def test_cough_is_not_silently_accepted(self):
        assert not validate("play [cough] music").is_valid


# -------------------------------------------------------------- section 13
class TestNoPauseTag:
    """Mandatory: a pause never becomes a transcription tag."""

    @pytest.mark.parametrize("tag", ["[pause]", "<pause>", "[silence]", "<silence>"])
    def test_pause_tags_rejected(self, tag, check):
        report = check(f"play {tag} music", Code.PAUSE_TAG)
        assert "no transcription tag for pauses" in report.by_code(Code.PAUSE_TAG)[0].message

    def test_pause_tag_is_not_reported_as_merely_unknown(self, check):
        # It gets its own code so the message can explain the V2 rule.
        check("play [pause] music", absent=Code.UNKNOWN_TAG)


# -------------------------------------------------------------- section 41
class TestMultipleTags:
    def test_alphabetical_order_passes(self, assert_valid):
        assert_valid("play [artifact] [fp] [hn] music")
        assert_valid("play [fp] [laughter] music")

    def test_wrong_order_flagged(self, check):
        report = check("play [laughter] [artifact] music", Code.TAG_ORDER)
        assert report.by_code(Code.TAG_ORDER)[0].suggestion == "[artifact] [laughter]"

    def test_sort_key_ignores_brackets(self):
        tags = ["[laughter]", "[artifact]", "[hn]", "[bg]", "[fp]"]
        assert sorted(tags, key=T.sort_key) == [
            "[artifact]", "[bg]", "[fp]", "[hn]", "[laughter]"
        ]

    def test_bg_is_positional_not_alphabetical(self, check):
        # Sections 20 and 21 place [bg] by position, so it is exempt from the
        # ordering rule. "[laughter] [bg]" is documented as correct.
        check("[laughter] [bg]", absent=Code.TAG_ORDER)

    def test_speaker_tags_are_not_sorted(self, check):
        # Speaker order follows the audio, never the alphabet.
        check("<s2> hi", absent=Code.TAG_ORDER)


# -------------------------------------------------------------- section 40
class TestTagPlacement:
    @pytest.mark.parametrize("text", [
        "play [hn] music",
        "[laughter] you're funny",
        "play [artifact] music",
        "ama- [hn] -zing",
        "[artifact] stop",
    ])
    def test_documented_placements_pass(self, assert_valid, text):
        assert_valid(text)

    def test_stray_bracket_flagged(self, check):
        check("play [hn music", Code.MALFORMED_TAG)
        check("play hn] music", Code.MALFORMED_TAG)

    def test_stray_angle_flagged(self, check):
        check("play < music", Code.MALFORMED_TAG)


# ------------------------------------------------------------------ helpers
class TestTagHelpers:
    def test_speaker_number(self):
        assert T.speaker_number("<s1>") == 1
        assert T.speaker_number("<s12>") == 12
        assert T.speaker_number("<ol>") is None
        assert T.speaker_number("[hn]") is None

    def test_is_speaker_tag(self):
        assert T.is_speaker_tag("<s3>")
        assert not T.is_speaker_tag("<sc>")  # sensitive content, not speaker 'c'

    def test_retired_lookup_is_case_insensitive(self):
        assert T.is_retired_tag("[COUGH]")
        assert T.replacement_for_retired("[Cough]") == "[hn]"

    def test_standalone_valid_set(self):
        for tag in ("<ga>", "<na>", "<ns>", "<nt>", "<sc>", "(())", "<ct>",
                    "[laughter]", "[hn]"):
            assert tag in T.STANDALONE_VALID
        # A tag needing surrounding context is not standalone-valid.
        assert "[artifact]" not in T.STANDALONE_VALID
        assert "[fp]" not in T.STANDALONE_VALID
