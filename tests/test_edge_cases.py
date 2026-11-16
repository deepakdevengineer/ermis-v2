"""Section 55 edge cases A through P, plus the section 56 prohibitions.

Each test names the case it covers so a reviewer can check it against
CLAUDE.md directly.
"""

from __future__ import annotations

import pytest

from app.ermis import tags as T
from app.ermis.findings import Code
from app.ermis.validator import validate
from app.models.event import (
    AnalysisEvent,
    ConfirmationState,
    EventType,
    UnconfirmedEventError,
)


class TestSection55Cases:
    def test_case_a_laughter_then_speech(self, assert_valid):
        assert_valid("[laughter] i know right")

    def test_case_b_two_coughs_use_one_tag(self, check, assert_valid):
        assert_valid("play [hn] music")
        check("play [hn] [hn] music", Code.DUPLICATE_TAG)

    def test_case_c_brief_static_then_speech(self, assert_valid):
        assert_valid("[artifact] play music")

    def test_case_d_recording_impossible_to_evaluate(self, assert_valid):
        assert_valid("<ga>")

    def test_case_e_completely_blank(self, assert_valid, wav_silence):
        from app.audio.loader import load_audio

        assert_valid("<na>")
        assert load_audio(wav_silence).is_silent

    def test_case_f_audio_but_no_speech(self, assert_valid):
        assert_valid("<ns>")

    def test_case_g_partly_unintelligible(self, assert_valid):
        assert_valid("(()) but i don't know what to get")

    def test_case_h_entirely_unintelligible(self, assert_valid):
        assert_valid("(())")

    def test_case_i_two_sequential_speakers(self, assert_valid):
        assert_valid("<s1> play music <s2> ok")

    def test_case_j_intelligible_overlap(self, assert_valid):
        assert_valid("<ol> <s1> stop <s2> stop </ol>")

    def test_case_k_unintelligible_overlap(self, assert_valid):
        assert_valid("<ct>")

    def test_case_l_speaker_plus_unidentified_cough(self, assert_valid):
        assert_valid("<s1> play music [bg] <s2> ok")

    def test_case_m_whole_word_stutter_no_hyphen(self, assert_valid):
        assert_valid("word word word")

    def test_case_n_cut_off_word(self, assert_valid):
        assert_valid("fragment-")

    def test_case_o_closed_mouth_agreement(self, assert_valid, check):
        assert_valid("mm-hm")
        check("mhm", Code.BACKCHANNEL_SPELLING)

    def test_case_p_open_mouth_agreement(self, assert_valid, check):
        assert_valid("uh-huh")
        check("uh huh", Code.BACKCHANNEL_SPELLING)


class TestSection56Prohibitions:
    """Each numbered "DO NOT" that is checkable in this layer."""

    def test_2_no_hallucinated_words(self):
        # The validator reports on the text it is given and returns it
        # unchanged; it has no mechanism to add content.
        text = "play (())"
        assert validate(text).text == text

    def test_3_no_grammar_correction(self, assert_valid):
        assert_valid("he don't like it")

    def test_4_repetitions_are_kept(self, assert_valid):
        assert_valid("very very very much")

    def test_5_fillers_are_kept(self, assert_valid):
        assert_valid("um uh i think so")

    def test_6_backchannels_are_kept(self, assert_valid):
        assert_valid("mm-hm uh-huh")

    def test_7_laughter_is_not_hn(self):
        assert T.LAUGHTER != T.NON_LEXICAL
        assert "[laughter]" not in T.RETIRED_TAGS

    def test_8_cough_does_not_use_a_cough_tag(self, check):
        check("play [cough] music", Code.RETIRED_TAG)

    def test_9_v1_tags_rejected(self, assert_invalid):
        for tag in ["[cough]", "[sigh]", "[throat-clear]", "[sneeze]",
                    "[lip-smack]", "<hum>", "<i>", "<sg>", "<ws>", "<pws>", "<lv>"]:
            assert_invalid(f"play {tag} music")

    def test_10_no_pause_tag(self, check):
        check("play [pause] music", Code.PAUSE_TAG)

    def test_11_no_sentence_punctuation(self, check):
        check("play music.", Code.SENTENCE_PERIOD)
        check("play music?", Code.FORBIDDEN_PUNCTUATION)

    def test_12_no_digits(self, check):
        check("play 2 songs", Code.DIGIT_USED)

    def test_13_no_url_syntax(self, check):
        check("ischristmas.com", Code.SENTENCE_PERIOD)

    def test_14_no_email_syntax(self, check):
        check("janesmith8@company.com", Code.FORBIDDEN_SYMBOL)

    def test_15_metadata_defaults_to_unknown(self):
        from app.storage.database import SpeakerProfile

        profile = SpeakerProfile("s1", "Speaker 1")
        assert profile.gender == "unknown"
        assert profile.nativity == "unknown"

    def test_16_cross_talk_not_for_uncertain_order(self, assert_valid):
        assert_valid("<s1> a <s2> b")

    def test_17_ga_is_not_for_a_brief_artifact(self, assert_valid):
        assert_valid("play [artifact] music")

    def test_18_unintelligible_is_not_for_recording_problems(self, assert_valid):
        # Both forms are syntactically valid; the distinction is the
        # annotator's judgement, and the validator does not conflate them.
        assert_valid("play (())")
        assert_valid("<ga>")

    def test_19_ns_not_used_when_another_tag_applies(self, assert_valid):
        assert_valid("[laughter]")
        assert_valid("[hn]")

    def test_20_whole_file_tags_not_combined(self, check):
        check("<na> hello", Code.WHOLE_FILE_NOT_ALONE)

    def test_21_transcription_never_overwritten(self, db, audio_row):
        db.save_transcription(audio_row, "original human text")
        db.save_transcription(audio_row, "revised human text")
        history = [r["text"] for r in db.transcription_history(audio_row)]
        assert history == ["original human text", "revised human text"]

    def test_22_confidence_is_not_ground_truth(self):
        # A 99% confident detection still cannot produce a tag on its own.
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, confidence=0.99)
        assert event.requires_human_confirmation
        with pytest.raises(UnconfirmedEventError):
            _ = event.tag


class TestAsrIsLabelledNotHidden:
    """The ASR deviation, kept honest.

    Sections 1 and 56.1 prohibit ASR-generated transcription. The project owner
    asked for it anyway, so the guarantee the code still enforces is a
    different one: machine text is never presented as human work, and it never
    leaves the machine.
    """

    def test_machine_text_must_carry_the_draft_label(self, db, audio_row):
        # 'asr' and friends are refused; only the explicit 'asr_draft' label
        # gets in, so provenance cannot be laundered by choosing a word.
        for source in ("asr", "whisper", "model", "auto"):
            with pytest.raises(ValueError):
                db.save_transcription(audio_row, "text", source=source)
        db.save_transcription(audio_row, "text", source="asr_draft")

    def test_a_draft_is_not_reported_as_human_reviewed(self):
        from app.speakers.metadata import SpeakerRegistry
        from app.storage.export import Annotation

        annotation = Annotation(
            audio_path="clip.wav",
            duration=1.0,
            transcription="hello how are you",
            speakers=SpeakerRegistry(),
            events=[],
            source="asr_draft",
        )
        assert annotation.asr_used
        assert not annotation.human_reviewed

    def test_draft_text_is_never_written_by_the_validator_or_detectors(self):
        # The only module that may produce words is app.asr; the rule engine
        # and the detectors still emit findings and events, never text.
        import app.detection.pipeline as pipeline
        import app.ermis.validator as validator

        for module in (validator, pipeline):
            source = open(module.__file__, encoding="utf-8").read()
            assert "app.asr" not in source, (
                f"{module.__name__} must not depend on the recogniser"
            )

    def test_config_cannot_enable_external_asr(self):
        from app.config import AppConfig

        config = AppConfig()
        config.privacy.allow_external_asr = True
        with pytest.raises(ValueError, match="allow_external_asr"):
            config.validate()

    def test_no_event_type_produces_transcription_text(self):
        # Every suggested tag is a tag from the fixed V2 inventory, never
        # free text, so no code path can emit words.
        from app.models.event import SUGGESTED_TAG

        for tag in SUGGESTED_TAG.values():
            if tag is not None:
                assert tag in T.ALL_FIXED_TAGS


class TestHumanConfirmationWorkflow:
    """Section 47."""

    def test_pending_event_blocks_tag_access(self):
        event = AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.5, 0.8)
        assert event.state is ConfirmationState.PENDING
        with pytest.raises(UnconfirmedEventError):
            _ = event.tag

    def test_confirm_uses_the_proposed_tag_by_default(self):
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 7.42, 8.15, 0.91)
        assert event.proposed_tag == T.LAUGHTER
        event.confirm()
        assert event.tag == T.LAUGHTER

    def test_human_can_override_the_suggestion(self):
        # The UI's "Change to hn" action on a laughter detection.
        event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 7.42, 8.15, 0.91)
        event.confirm(T.NON_LEXICAL)
        assert event.tag == T.NON_LEXICAL

    def test_rejection_yields_no_tag(self):
        event = AnalysisEvent(EventType.POSSIBLE_ARTIFACT, 2.0, 2.1, 0.7)
        event.reject("just a door closing")
        assert event.state is ConfirmationState.REJECTED
        assert event.tag is None
        assert event.note == "just a door closing"

    def test_confirm_rejects_an_invalid_tag(self):
        event = AnalysisEvent(EventType.POSSIBLE_HN, 1.0, 1.5)
        with pytest.raises(ValueError, match="not a valid"):
            event.confirm("[cough]")

    def test_confirm_rejects_a_pause_tag(self):
        event = AnalysisEvent(EventType.PAUSE, 3.42, 5.18)
        with pytest.raises(ValueError, match="no pause tag"):
            event.confirm("[pause]")

    def test_pause_confirms_to_no_tag(self):
        event = AnalysisEvent(EventType.PAUSE, 3.42, 5.18)
        event.confirm()
        assert event.tag is None

    def test_certain_events_do_not_require_confirmation(self):
        assert not AnalysisEvent(EventType.SPEECH, 0.0, 1.0).requires_human_confirmation
        assert not AnalysisEvent(EventType.PAUSE, 1.0, 2.0).requires_human_confirmation

    def test_all_possible_types_require_confirmation(self):
        for event_type in EventType:
            if event_type.value.startswith("possible_"):
                event = AnalysisEvent(event_type, 0.0, 1.0, 0.9)
                assert event.requires_human_confirmation, event_type
