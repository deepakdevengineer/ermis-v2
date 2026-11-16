"""Speaker tags and speaker metadata.

CLAUDE.md sections 17 (multiple speakers), 18A (speaker profile and metadata),
22 (large group / crowd), 37 (user descriptors).

Mandatory test 16 from section 54, plus the twelve validation requirements of
section 18A.10.
"""

from __future__ import annotations

import pytest

from app.ermis import tags as T
from app.ermis.findings import Code, Severity
from app.ermis.validator import validate
from app.storage.database import ALLOWED_GENDERS, ALLOWED_NATIVITY, SpeakerProfile


# -------------------------------------------------------------- section 17
class TestSpeakerTags:
    """Mandatory test 16 -- speaker tags."""

    def test_two_speakers_pass(self, assert_valid):
        assert_valid("<s1> play <s2> songs")

    def test_three_speakers_pass(self, assert_valid):
        assert_valid("<s1> hi <s2> hello <s3> hey")

    def test_single_speaker_audio_takes_no_tag(self, assert_valid):
        # Section 17: "play music", NOT "<s1> play music".
        assert_valid("play music")

    def test_s1_alone_is_an_error(self, check):
        report = check("<s1> play music", Code.SPEAKER_SINGLE_TAGGED)
        assert report.by_code(Code.SPEAKER_SINGLE_TAGGED)[0].suggestion == ""

    def test_speaker_count_one_makes_any_tag_an_error(self, check):
        check("<s1> hi <s2> hello", Code.SPEAKER_SINGLE_TAGGED, speaker_count=1)

    def test_speaker_count_two_allows_tags(self, assert_valid):
        assert_valid("<s1> hi <s2> hello", speaker_count=2)

    def test_tag_at_the_start_of_each_turn(self, assert_valid):
        assert_valid("<s1> ready <s2> yes")


# ------------------------------------------------------------ section 18A.2
class TestSpeakerNumbering:
    """Numbering follows first appearance with no gaps."""

    def test_sequential_numbering_passes(self, assert_valid):
        assert_valid("<s1> a <s2> b <s3> c")

    def test_starting_at_s2_flagged(self, check):
        report = check("<s2> a <s3> b", Code.SPEAKER_NUMBERING)
        finding = report.by_code(Code.SPEAKER_NUMBERING)[0]
        assert finding.suggestion == "<s1>"

    def test_skipped_number_flagged(self, check):
        report = check("<s1> a <s3> b", Code.SPEAKER_NUMBERING)
        assert report.by_code(Code.SPEAKER_NUMBERING)[0].suggestion == "<s2>"

    def test_returning_to_an_earlier_speaker_is_fine(self, assert_valid):
        # s1, s2, then back to s1: normal turn-taking.
        assert_valid("<s1> a <s2> b <s1> c")

    def test_first_appearance_order_not_numeric_order(self, check):
        # Hearing s2 first is wrong regardless of what follows.
        check("<s2> first <s1> second", Code.SPEAKER_NUMBERING)

    def test_numbering_message_cites_first_appearance(self, check):
        report = check("<s2> a <s3> b", Code.SPEAKER_NUMBERING)
        assert "first appearance" in report.by_code(Code.SPEAKER_NUMBERING)[0].message


# ----------------------------------------------------------- section 18A.10
class TestSpeakerReferenceValidation:
    def test_unknown_speaker_reference_flagged(self, check):
        # Item 7: a speaker tag referencing a nonexistent speaker.
        check("<s1> a <s2> b <s3> c", Code.SPEAKER_UNKNOWN_REFERENCE,
              known_speakers={1, 2})

    def test_known_speakers_pass(self, assert_valid):
        assert_valid("<s1> a <s2> b", known_speakers={1, 2})

    def test_no_check_when_speaker_panel_not_supplied(self, check):
        check("<s1> a <s2> b <s9> c", absent=Code.SPEAKER_UNKNOWN_REFERENCE)

    def test_repeated_turn_warns(self):
        # Item 12: inconsistent speaker assignment. A warning, since a long
        # pause between two turns by the same speaker can justify it.
        report = validate("<s1> hello <s1> there <s2> hi")
        assert report.has(Code.SPEAKER_REPEATED_TURN)
        assert report.by_code(Code.SPEAKER_REPEATED_TURN)[0].severity is Severity.WARNING

    def test_repeated_turn_allowed_inside_overlap(self, check):
        # Section 18: each speaker is re-tagged inside an overlap, so the same
        # tag legitimately appears again there.
        check("<s1> a <ol> <s2> b <s1> c </ol>", absent=Code.SPEAKER_REPEATED_TURN)

    def test_too_many_speakers_warns(self):
        text = " ".join(f"<s{i}> word" for i in range(1, 23))
        report = validate(text)
        assert report.has(Code.SPEAKER_TOO_MANY)
        assert report.by_code(Code.SPEAKER_TOO_MANY)[0].severity is Severity.WARNING


# -------------------------------------------------------------- section 22
class TestCrowd:
    def test_crowd_treated_as_one_speaker(self, assert_valid):
        # Section 22: <s2> represents the crowd.
        assert_valid("<s1> ready <s2> yes")

    def test_unintelligible_crowd_uses_cross_talk(self, assert_valid):
        assert_valid("<ct>")


# ------------------------------------------------------ sections 18A.3, 37
class TestSpeakerGender:
    @pytest.mark.parametrize("gender", ["msv", "fsv", "unknown"])
    def test_allowed_values(self, gender):
        assert SpeakerProfile("s1", "Speaker 1", gender=gender).gender == gender

    @pytest.mark.parametrize("gender", ["male", "female", "m", "f", "MSV", "other", ""])
    def test_disallowed_values_rejected(self, gender):
        with pytest.raises(ValueError, match="gender"):
            SpeakerProfile("s1", "Speaker 1", gender=gender)

    def test_default_is_unknown(self):
        # The system must not force a classification (section 18A.3).
        assert SpeakerProfile("s1", "Speaker 1").gender == "unknown"

    def test_allowed_set_matches_the_specification(self):
        assert set(ALLOWED_GENDERS) == {"msv", "fsv", "unknown"}


# ------------------------------------------------------ sections 18A.4, 37
class TestSpeakerNativity:
    @pytest.mark.parametrize("nativity", ["native", "nonnative", "unknown", "tts"])
    def test_allowed_values(self, nativity):
        assert SpeakerProfile("s1", "Speaker 1", nativity=nativity).nativity == nativity

    @pytest.mark.parametrize("nativity", ["fluent", "non-native", "NATIVE", "robot", ""])
    def test_disallowed_values_rejected(self, nativity):
        with pytest.raises(ValueError, match="nativity"):
            SpeakerProfile("s1", "Speaker 1", nativity=nativity)

    def test_default_is_unknown(self):
        assert SpeakerProfile("s1", "Speaker 1").nativity == "unknown"

    def test_allowed_set_matches_the_specification(self):
        assert set(ALLOWED_NATIVITY) == {"native", "nonnative", "unknown", "tts"}


# ----------------------------------------------------------- section 18A.5
class TestSpeakerProfile:
    def test_profile_fields(self):
        profile = SpeakerProfile(
            speaker_id="s2",
            display_name="Speaker 2",
            gender="fsv",
            nativity="native",
            first_seen=8.41,
            last_seen=51.22,
            confidence=0.91,
        )
        payload = profile.to_dict()
        assert payload == {
            "speaker_id": "s2",
            "display_name": "Speaker 2",
            "gender": "fsv",
            "nativity": "native",
            "first_seen": 8.41,
            "last_seen": 51.22,
            "confidence": 0.91,
        }

    def test_confidence_is_optional(self):
        assert SpeakerProfile("s1", "Speaker 1").confidence is None


# ---------------------------------------------------------- section 18A.11
class TestHumanDecisionsPreserved:
    """Automatic detection is assistive only; a human value is never silently
    overwritten (sections 18A.11 and 56.15)."""

    def test_automatic_pass_cannot_overwrite_a_human_value(self, db, audio_row):
        db.upsert_speaker(
            audio_row,
            SpeakerProfile("s1", "Speaker 1", gender="msv", nativity="native"),
            human_edited=True,
        )
        # A later detector run proposes different values.
        db.upsert_speaker(
            audio_row,
            SpeakerProfile("s1", "Speaker 1", gender="fsv", nativity="nonnative"),
            human_edited=False,
        )
        stored = db.list_speakers(audio_row)[0]
        assert stored.gender == "msv"
        assert stored.nativity == "native"
        assert stored.human_edited is True

    def test_human_can_change_their_own_value(self, db, audio_row):
        db.upsert_speaker(
            audio_row, SpeakerProfile("s1", "Speaker 1", gender="msv"), human_edited=True
        )
        db.upsert_speaker(
            audio_row, SpeakerProfile("s1", "Speaker 1", gender="fsv"), human_edited=True
        )
        assert db.list_speakers(audio_row)[0].gender == "fsv"

    def test_automatic_pass_may_set_an_untouched_value(self, db, audio_row):
        db.upsert_speaker(
            audio_row, SpeakerProfile("s1", "Speaker 1"), human_edited=False
        )
        db.upsert_speaker(
            audio_row, SpeakerProfile("s1", "Speaker 1", gender="msv"), human_edited=False
        )
        assert db.list_speakers(audio_row)[0].gender == "msv"

    def test_timeline_bounds_still_refresh_on_a_human_edited_speaker(self, db, audio_row):
        db.upsert_speaker(
            audio_row,
            SpeakerProfile("s1", "Speaker 1", gender="msv", first_seen=1.0, last_seen=2.0),
            human_edited=True,
        )
        db.upsert_speaker(
            audio_row,
            SpeakerProfile("s1", "Speaker 1", gender="fsv", first_seen=0.5, last_seen=9.0),
            human_edited=False,
        )
        stored = db.list_speakers(audio_row)[0]
        assert stored.gender == "msv"       # metadata protected
        assert stored.first_seen == 0.5     # timeline refreshed
        assert stored.last_seen == 9.0


class TestSpeakerHelpers:
    def test_speaker_numbers_from_db(self, db, audio_row):
        db.upsert_speaker(audio_row, SpeakerProfile("s1", "Speaker 1"))
        db.upsert_speaker(audio_row, SpeakerProfile("s2", "Speaker 2"))
        assert db.speaker_numbers(audio_row) == {1, 2}

    def test_sc_is_not_parsed_as_a_speaker(self):
        # <sc> is the sensitive-content tag, not "speaker c".
        assert not T.is_speaker_tag("<sc>")
        assert validate("<sc>").is_valid

    def test_multi_digit_speaker_numbers(self):
        assert T.speaker_number("<s10>") == 10
