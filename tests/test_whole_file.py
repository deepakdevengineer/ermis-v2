"""Whole-file conditions.

CLAUDE.md sections 10 (<ga>), 11 (<na>), 12 (<ns>), 35 (<nt>), 36 (<sc>),
42 (whole-file tag logic).

Mandatory tests 9 to 13 and 25 from section 54.
"""

from __future__ import annotations

import pytest

from app.ermis import tags as T
from app.ermis import whole_file
from app.ermis.findings import Code
from app.ermis.validator import validate

WHOLE_FILE_TAGS = ["<ga>", "<na>", "<ns>", "<nt>", "<sc>"]


class TestWholeFileTagsAlone:
    """Mandatory tests 9-13 and 25."""

    @pytest.mark.parametrize("tag", WHOLE_FILE_TAGS)
    def test_tag_alone_is_valid(self, assert_valid, tag):
        assert_valid(tag)

    @pytest.mark.parametrize("tag", WHOLE_FILE_TAGS)
    def test_tag_with_text_is_invalid(self, check, tag):
        report = check(f"{tag} hello", Code.WHOLE_FILE_NOT_ALONE)
        assert report.by_code(Code.WHOLE_FILE_NOT_ALONE)[0].suggestion == tag

    @pytest.mark.parametrize("tag", WHOLE_FILE_TAGS)
    def test_text_before_tag_is_invalid(self, check, tag):
        check(f"hello {tag}", Code.WHOLE_FILE_NOT_ALONE)

    @pytest.mark.parametrize("tag", WHOLE_FILE_TAGS)
    def test_tag_with_another_tag_is_invalid(self, check, tag):
        check(f"{tag} [hn]", Code.WHOLE_FILE_NOT_ALONE)

    def test_two_whole_file_tags_flagged(self, check):
        check("<na> <ga>", Code.MULTIPLE_WHOLE_FILE_TAGS)

    def test_documented_wrong_examples(self, assert_invalid):
        # Section 42 lists these explicitly as WRONG.
        assert_invalid("<na> hello")
        assert_invalid("<ga> hello")

    def test_documented_correct_examples(self, assert_valid):
        assert_valid("<na>")
        assert_valid("<ga>")

    def test_surrounding_whitespace_does_not_break_a_standalone_tag(self):
        report = validate(" <na> ")
        assert not report.errors  # only the whitespace warning
        assert report.has(Code.LEADING_TRAILING_SPACE)


class TestNoAudio:
    """Mandatory test 10 -- <na>. Section 11."""

    def test_valid_alone(self, assert_valid):
        assert_valid("<na>")

    def test_meaning(self):
        condition = whole_file.detect("<na>")
        assert condition is not None
        assert condition.tag == "<na>"
        assert "no audio signal" in condition.meaning

    def test_measured_from_the_audio_not_decided(self, wav_silence):
        # Case E from section 55: a completely blank file. The loader measures
        # silence; the annotator decides <na>.
        from app.audio.loader import load_audio

        buffer = load_audio(wav_silence)
        assert buffer.is_silent


class TestGarbled:
    """Mandatory test 9 -- <ga>. Section 10."""

    def test_valid_alone(self, assert_valid):
        assert_valid("<ga>")

    def test_meaning(self):
        condition = whole_file.detect("<ga>")
        assert condition is not None
        assert "impossible to evaluate" in condition.meaning

    def test_distinct_from_artifact(self):
        # Section 9's decision rule.
        assert whole_file.classify_noise(True) == T.ARTIFACT
        assert whole_file.classify_noise(False) == T.GARBLED

    def test_brief_artifact_does_not_use_ga(self, assert_valid):
        # Section 56.17: <ga> is not for a brief artifact.
        assert_valid("play [artifact] music")

    def test_unintelligible_is_not_ga(self, assert_valid):
        # Sections 14 and 56.18: (()) is for unclear speech, <ga> for
        # recording problems. They are different findings.
        assert_valid("play (())")


class TestNoSpeech:
    """Mandatory test 11 -- <ns>. Section 12."""

    def test_valid_alone(self, assert_valid):
        assert_valid("<ns>")

    def test_meaning(self):
        condition = whole_file.detect("<ns>")
        assert condition is not None
        assert "no human speech" in condition.meaning

    def test_not_combined_with_text(self, check):
        check("<ns> room noise", Code.WHOLE_FILE_NOT_ALONE)

    def test_another_applicable_tag_takes_precedence(self, assert_valid):
        # Section 12 and 56.19: do not use <ns> when another tag applies. A
        # clip of only laughter is [laughter], not <ns>.
        assert_valid("[laughter]")
        assert_valid("[hn]")


class TestNonTarget:
    """Mandatory test 12 -- <nt>. Section 35."""

    def test_valid_alone(self, assert_valid):
        assert_valid("<nt>")

    def test_meaning(self):
        condition = whole_file.detect("<nt>")
        assert condition is not None
        assert "non-target language" in condition.meaning

    def test_understandable_foreign_speech_is_transcribed(self, assert_valid):
        # Section 35: if understandable, transcribe it.
        assert_valid("jose gonzalez")

    def test_partly_unintelligible_uses_double_parentheses(self, assert_valid):
        assert_valid("hola (()) amigo")

    def test_background_foreign_speech_uses_bg_not_nt(self, assert_valid):
        # Section 35: background non-target language from an unidentified
        # speaker is [bg], NOT <nt>.
        assert_valid("play music [bg]")


class TestSensitiveContent:
    """Mandatory test 13 -- <sc>. Section 36."""

    def test_valid_alone(self, assert_valid):
        assert_valid("<sc>")

    def test_not_combined(self, check):
        check("<sc> something", Code.WHOLE_FILE_NOT_ALONE)

    def test_not_parsed_as_a_speaker_tag(self):
        assert not T.is_speaker_tag("<sc>")


class TestStandaloneNonAngleConditions:
    """Section 42 -- conditions written without an angle tag."""

    def test_unintelligible_whole_file(self, assert_valid):
        # Case H from section 55.
        assert_valid("(())")

    def test_cross_talk_whole_file(self, assert_valid):
        assert_valid("<ct>")

    def test_laughter_only(self, assert_valid):
        assert_valid("[laughter]")

    def test_non_lexical_only(self, assert_valid):
        assert_valid("[hn]")

    @pytest.mark.parametrize("tag", ["(())", "<ct>", "[laughter]", "[hn]"])
    def test_recognized_as_standalone(self, tag):
        assert whole_file.is_standalone_valid(tag)

    @pytest.mark.parametrize("tag", ["[artifact]", "[fp]", "[bg]", "play music"])
    def test_not_standalone(self, tag):
        assert not whole_file.is_standalone_valid(tag)


class TestDetection:
    def test_detect_requires_an_exact_standalone_match(self):
        assert whole_file.detect("<na>") is not None
        assert whole_file.detect("<na> hello") is None
        assert whole_file.detect("play music") is None

    def test_detect_tolerates_surrounding_whitespace(self):
        assert whole_file.detect("  <ga>  ") is not None

    def test_angle_tags_are_marked_as_such(self):
        assert whole_file.detect("<ga>").is_angle_whole_file
        assert not whole_file.detect("(())").is_angle_whole_file

    def test_every_whole_file_tag_has_a_meaning(self):
        for tag in T.WHOLE_FILE_TAGS:
            assert tag in whole_file.WHOLE_FILE_MEANING
            assert whole_file.WHOLE_FILE_MEANING[tag]

    def test_empty_transcription_is_not_a_whole_file_condition(self, check):
        # A blank clip is <na>, and an empty text box is an error, not a
        # silently accepted whole-file condition.
        check("", Code.EMPTY_TRANSCRIPTION)
        assert whole_file.detect("") is None
