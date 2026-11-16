"""Local ASR engine helpers and the draft builder.

No model is loaded here. The engine's plumbing -- windowing, resampling, token
grouping, timing repair -- is tested directly, and the draft builder is fed
recogniser words supplied by the test, which is what `draft_from_words` exists
for. That keeps this file fast and makes the merge logic, not the model, the
thing under test.

The properties that matter:

  - a draft always validates, or says why it does not
  - a draft never claims more than the audio supports: a low-confidence word
    becomes unintelligible, a weak detection contributes nothing, and a
    single-speaker clip gets no speaker tag
  - a whole-file condition stands alone
  - pause never becomes a tag (CLAUDE.md sections 13, 56.10)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.asr.draft import DraftBuilder, draft_from_words
from app.asr.engine import (
    AsrError,
    AsrResult,
    AsrWord,
    _finalize_word_times,
    _resample_for_model,
    _windows,
    build_engine,
)
from app.audio.loader import load_audio
from app.config import AsrConfig
from app.detection.pipeline import AnalysisPipeline
from app.ermis import tags as T
from app.models.event import AnalysisEvent, EventType

#: Backend names. The memory figures differ by roughly a factor of three,
#: so a test that asserts one of them must not let "auto" choose.
TORCH = "transformers-whisper"
FASTER = "faster-whisper"


def analyse(path):
    buffer = load_audio(path)
    return buffer, AnalysisPipeline().run(buffer)


def words(*specs: tuple[str, float, float, float]) -> list[AsrWord]:
    return [AsrWord(text, start, end, conf) for text, start, end, conf in specs]


# --------------------------------------------------------------------- engine
class TestWindowing:
    def test_short_audio_is_one_window(self):
        samples = np.zeros(16_000, dtype=np.float32)
        assert len(_windows(samples, 16_000)) == 1

    def test_long_audio_is_split_and_covers_everything(self):
        samples = np.zeros(16_000 * 70, dtype=np.float32)
        chunks = _windows(samples, 16_000)
        assert len(chunks) >= 3
        assert sum(chunk.size for _offset, chunk in chunks) == samples.size

    def test_offsets_are_cumulative_seconds(self):
        samples = np.zeros(16_000 * 70, dtype=np.float32)
        chunks = _windows(samples, 16_000)
        assert chunks[0][0] == 0.0
        for (offset, chunk), (next_offset, _next) in zip(chunks, chunks[1:]):
            assert next_offset == pytest.approx(offset + chunk.size / 16_000)

    def test_cut_lands_on_a_quiet_frame(self):
        # A window boundary should fall in the silence, not mid-word, so a word
        # is never split across two windows.
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 0.3, 16_000 * 60).astype(np.float32)
        quiet_at = int(27.0 * 16_000)
        samples[quiet_at : quiet_at + 16_000] = 0.0
        chunks = _windows(samples, 16_000)
        first_cut = chunks[0][1].size / 16_000
        assert 27.0 <= first_cut <= 28.0


class TestResampling:
    def test_model_rate_passes_through(self, wav_tone):
        buffer = load_audio(wav_tone)  # loader delivers 16 kHz
        assert _resample_for_model(buffer) is not None
        assert _resample_for_model(buffer).size == buffer.samples.size

    def test_empty_buffer_is_handled(self, wav_tone):
        from dataclasses import replace

        buffer = replace(load_audio(wav_tone), samples=np.zeros(0, dtype=np.float32))
        assert _resample_for_model(buffer).size == 0


class TestWordTiming:
    def test_zero_length_word_is_extended_to_the_next_word(self):
        fixed = _finalize_word_times(
            words(("how", 1.0, 1.0, 0.9), ("are", 1.4, 1.4, 0.9)), duration=2.0
        )
        assert fixed[0].end == pytest.approx(1.4)
        assert fixed[1].end == pytest.approx(2.0)

    def test_last_word_is_clamped_to_the_audio(self):
        fixed = _finalize_word_times(words(("you", 4.0, 9.0, 0.9)), duration=5.0)
        assert fixed[0].end <= 5.0

    def test_ordinary_spans_survive(self):
        fixed = _finalize_word_times(words(("hello", 0.0, 0.8, 0.9)), duration=2.0)
        assert (fixed[0].start, fixed[0].end) == (0.0, 0.8)


class TestEngineSelection:
    def test_disabled_config_refuses_to_build(self):
        with pytest.raises(AsrError, match="disabled"):
            build_engine(AsrConfig(enabled=False))

    def test_unknown_backend_is_named_in_the_error(self):
        with pytest.raises(AsrError, match="unknown ASR backend"):
            build_engine(AsrConfig(backend="magic"))

    def test_missing_backend_explains_how_to_install(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_backends", lambda config=None: [])
        with pytest.raises(AsrError, match="pip install"):
            engine.build_engine(AsrConfig())


class TestEngineCache:
    """One loaded model per process.

    Regression test from a real run: drafting a folder of ten clips loaded the
    model once per clip, and on Windows the second concurrent load failed with
    "the paging file is too small" -- so every draft silently lost its words and
    fell back to `(())`. The failure was honest but the words were gone.
    """

    def setup_method(self):
        from app.asr.engine import reset_engine_cache

        reset_engine_cache()

    teardown_method = setup_method

    def _counting_backend(self, monkeypatch):
        import app.asr.engine as engine

        # The memory gate must not make these tests depend on how much RAM the
        # machine running them happens to have free.
        monkeypatch.setattr(engine, "available_memory_gb", lambda: 64.0)
        loads = []

        class StubEngine(engine.AsrEngine):
            name = "stub"

            def load(self):
                loads.append(1)
                self._loaded = True

            def _transcribe_window(self, window):
                return []

        monkeypatch.setattr(engine, "build_engine", lambda config=None: StubEngine(config))
        return loads

    def test_repeated_get_engine_loads_once(self, monkeypatch):
        from app.asr.engine import get_engine

        loads = self._counting_backend(monkeypatch)
        first = get_engine(AsrConfig())
        second = get_engine(AsrConfig())
        assert first is second
        assert len(loads) == 1

    def test_changing_the_model_loads_again(self, monkeypatch):
        from app.asr.engine import get_engine

        loads = self._counting_backend(monkeypatch)
        get_engine(AsrConfig())
        get_engine(AsrConfig(model="openai/whisper-base.en"))
        assert len(loads) == 2

    def test_drafting_many_clips_loads_one_model(self, monkeypatch, wav_speech):
        loads = self._counting_backend(monkeypatch)
        buffer, analysis = analyse(wav_speech)
        for _ in range(3):
            DraftBuilder().build(buffer, analysis)
        assert len(loads) == 1

    def test_a_supplied_engine_is_used_as_is(self, monkeypatch, wav_speech):
        loads = self._counting_backend(monkeypatch)
        from app.asr.engine import get_engine

        engine = get_engine(AsrConfig())
        buffer, analysis = analyse(wav_speech)
        DraftBuilder(engine=engine).build(buffer, analysis)
        assert len(loads) == 1

    def test_reset_forces_a_reload(self, monkeypatch):
        from app.asr.engine import get_engine, reset_engine_cache

        loads = self._counting_backend(monkeypatch)
        get_engine(AsrConfig())
        reset_engine_cache()
        get_engine(AsrConfig())
        assert len(loads) == 2


class TestMemoryGate:
    """A model that will not fit must not be loaded.

    Regression from a real crash: the window preloaded whisper-small.en on a
    machine with 2 GB free and segfaulted during quantisation, taking the whole
    application with it. A load that does not fit cannot be caught afterwards,
    so it has to be refused beforehand.
    """

    def setup_method(self):
        from app.asr.engine import reset_engine_cache

        reset_engine_cache()

    teardown_method = setup_method

    def test_estimates_are_per_model_size(self):
        from app.asr.engine import model_memory_estimate_gb

        tiny = model_memory_estimate_gb("openai/whisper-tiny.en")
        small = model_memory_estimate_gb("openai/whisper-small.en")
        medium = model_memory_estimate_gb("openai/whisper-medium")
        assert tiny < small < medium

    def test_unknown_model_name_assumes_the_common_default(self):
        from app.asr.engine import model_memory_estimate_gb

        assert model_memory_estimate_gb("some/other-model") == pytest.approx(2.6)

    def test_plenty_of_memory_keeps_the_requested_model(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 32.0)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=TORCH))
        assert fitted.model == AsrConfig().model
        assert note is None

    def test_tight_memory_downgrades_and_says_so(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 2.0)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=TORCH))
        assert fitted.model == "openai/whisper-base.en"
        assert note is not None
        assert "not enough" in note

    def test_english_only_family_is_preserved_on_downgrade(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 2.0)
        fitted, _ = engine.fit_model_to_memory(
            AsrConfig(model="openai/whisper-medium", backend=TORCH)
        )
        # The largest size that fits in 2 GB is base; the multilingual family is
        # kept, so a non-English clip does not silently become English-only.
        assert fitted.model == "openai/whisper-base"
        assert not fitted.model.endswith(".en")

    def test_downgrade_can_be_refused(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 2.0)
        config = AsrConfig(downgrade_when_memory_is_tight=False, backend=TORCH)
        with pytest.raises(AsrError, match="needs about"):
            engine.fit_model_to_memory(config)

    def test_no_memory_at_all_is_a_clear_refusal(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.2)
        with pytest.raises(AsrError, match="smallest model"):
            engine.fit_model_to_memory(AsrConfig(backend=TORCH))

    def test_explicit_requirement_overrides_the_estimate(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 4.0)
        config = AsrConfig(min_free_memory_gb=32.0, backend=TORCH)
        fitted, note = engine.fit_model_to_memory(config)
        assert fitted.model != config.model
        assert note is not None

    def test_without_psutil_the_gate_stays_out_of_the_way(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: None)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=TORCH))
        assert fitted.model == AsrConfig().model
        assert note is None

    # ------------------------------------------------- backend-aware costs
    def test_faster_whisper_is_charged_less_than_transformers(self):
        from app.asr.engine import model_memory_estimate_gb

        for size in ("tiny", "base", "small", "medium", "large"):
            model = f"openai/whisper-{size}"
            assert model_memory_estimate_gb(model, FASTER) < model_memory_estimate_gb(
                model, TORCH
            )

    def test_unknown_backend_is_charged_the_dearest_price(self):
        from app.asr.engine import model_memory_estimate_gb

        # Guessing low means a load that does not fit, which kills the process.
        assert model_memory_estimate_gb(
            "openai/whisper-small.en", None
        ) == model_memory_estimate_gb("openai/whisper-small.en", TORCH)

    def test_faster_whisper_keeps_the_model_where_transformers_downgrades(
        self, monkeypatch
    ):
        import app.asr.engine as engine

        # The bug this covers: 1.5 GB free refused a draft that CTranslate2
        # would have run comfortably, because the gate charged every backend
        # the transformers conversion peak.
        monkeypatch.setattr(engine, "available_memory_gb", lambda: 1.5)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=FASTER))
        assert fitted.model == "openai/whisper-small.en"
        assert note is None

        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=TORCH))
        assert fitted.model != "openai/whisper-small.en"
        assert note is not None

    def test_faster_whisper_drafts_where_transformers_refuses_outright(
        self, monkeypatch
    ):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.7)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=FASTER))
        # base.en fits in 0.7 GB on this backend (0.35 + 0.15 margin), so the
        # annotator gets the better of the two small models, not the worst.
        assert fitted.model == "openai/whisper-base.en"
        assert note is not None

        with pytest.raises(AsrError, match="smallest model"):
            engine.fit_model_to_memory(AsrConfig(backend=TORCH))

    def test_faster_whisper_reserves_less_headroom(self):
        from app.asr.engine import memory_headroom_gb

        # The margin covers an allocation spike. CTranslate2 reads weights that
        # were quantised ahead of time and does not spike; the torch path does.
        assert memory_headroom_gb(FASTER) < memory_headroom_gb(TORCH)
        assert memory_headroom_gb(None) == memory_headroom_gb(TORCH)

    def test_a_tiny_model_fits_in_the_memory_a_laptop_actually_has(
        self, monkeypatch
    ):
        import app.asr.engine as engine

        # Regression from this project's own 8 GB machine: 0.5 GB free refused
        # every model, though CTranslate2 tiny.en runs in well under that.
        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.45)
        fitted, note = engine.fit_model_to_memory(AsrConfig(backend=FASTER))
        assert fitted.model == "openai/whisper-tiny.en"
        assert note is not None

    def test_the_refusal_names_the_cheaper_backend_when_it_would_help(
        self, monkeypatch
    ):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.7)
        with pytest.raises(AsrError, match="faster-whisper"):
            engine.fit_model_to_memory(AsrConfig(backend=TORCH))

    def test_no_cheaper_backend_hint_when_it_would_not_help(self, monkeypatch):
        import app.asr.engine as engine

        # Below what even CTranslate2 needs, the suggestion would be a wrong
        # answer rather than a helpful one.
        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.2)
        with pytest.raises(AsrError) as excinfo:
            engine.fit_model_to_memory(AsrConfig(backend=TORCH))
        assert "faster-whisper" not in str(excinfo.value)

    def test_no_hint_when_already_on_the_cheaper_backend(self, monkeypatch):
        import app.asr.engine as engine

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 0.2)
        with pytest.raises(AsrError) as excinfo:
            engine.fit_model_to_memory(AsrConfig(backend=FASTER))
        assert "faster-whisper" not in str(excinfo.value)

    def test_resolve_backend_honours_an_explicit_name(self):
        from app.asr.engine import resolve_backend

        assert resolve_backend(AsrConfig(backend=FASTER)) == FASTER
        assert resolve_backend(AsrConfig(backend=TORCH)) == TORCH
        assert resolve_backend(AsrConfig(backend="nonsense")) is None

    def test_resolve_backend_matches_what_build_engine_picks(self):
        from app.asr.engine import available_backends, build_engine, resolve_backend

        if not available_backends():
            pytest.skip("no local ASR backend installed")
        assert resolve_backend(AsrConfig()) == build_engine(AsrConfig()).name

    def test_the_note_is_readable_after_loading(self, monkeypatch):
        import app.asr.engine as engine

        loaded = []

        class StubEngine(engine.AsrEngine):
            name = "stub"

            def load(self):
                loaded.append(self.config.model)
                self._loaded = True

            def _transcribe_window(self, window):
                return []

        monkeypatch.setattr(engine, "available_memory_gb", lambda: 2.0)
        monkeypatch.setattr(engine, "build_engine", lambda config=None: StubEngine(config))
        engine.get_engine(AsrConfig(backend=TORCH))
        assert loaded == ["openai/whisper-base.en"]
        assert "not enough" in (engine.active_memory_note() or "")


class TestAsrResult:
    def test_text_joins_words(self):
        result = AsrResult(words(("hello", 0.0, 0.5, 0.9), ("there", 0.5, 1.0, 0.8)),
                           backend="t", model="m")
        assert result.text == "hello there"
        assert result.mean_confidence == pytest.approx(0.85)

    def test_words_between_uses_the_midpoint(self):
        result = AsrResult(
            words(("a", 0.0, 1.0, 1.0), ("b", 1.0, 2.0, 1.0), ("c", 2.0, 3.0, 1.0)),
            backend="t",
            model="m",
        )
        assert [w.text for w in result.words_between(0.9, 2.1)] == ["b"]

    def test_dict_is_json_shaped(self):
        import json

        result = AsrResult(words(("hi", 0.0, 0.4, 0.7)), backend="t", model="m")
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["word_count"] == 1
        assert payload["raw_text"] == "hi"


# ---------------------------------------------------------------------- draft
class TestSingleSpeakerDraft:
    def test_words_are_normalised_into_spoken_form(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(
            words(("Hello,", 0.1, 0.6, 0.95), ("okay", 0.7, 1.1, 0.93)),
            analysis,
            buffer,
        )
        assert draft.text == "hello ok"
        assert draft.report.is_valid

    def test_single_speaker_gets_no_speaker_tag(self, wav_speech):
        # Section 17: single-speaker audio does not take <s1>.
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(words(("play", 0.2, 0.6, 0.9)), analysis, buffer)
        assert "<s1>" not in draft.text

    def test_low_confidence_word_becomes_unintelligible(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(
            words(("play", 0.2, 0.5, 0.95), ("mumble", 0.6, 0.9, 0.10)),
            analysis,
            buffer,
        )
        assert draft.text == "play ((mumble))"
        assert draft.unclear_count == 1
        assert draft.report.is_valid

    def test_consecutive_unclear_words_share_one_marker(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(
            words(
                ("play", 0.1, 0.3, 0.95),
                ("some", 0.4, 0.6, 0.05),
                ("thing", 0.6, 0.8, 0.05),
            ),
            analysis,
            buffer,
        )
        assert draft.text == "play ((some thing))"

    def test_confidence_threshold_is_configurable(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        config = AsrConfig(min_word_confidence=0.05)
        draft = draft_from_words(
            words(("mumble", 0.2, 0.5, 0.10)), analysis, buffer, config=config
        )
        assert draft.text == "mumble"

    def test_backend_without_confidence_marks_nothing_unclear(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        result = AsrResult(
            words(("mumble", 0.2, 0.5, 0.0)),
            backend="stub",
            model="m",
            has_confidence=False,
        )
        draft = DraftBuilder().build(buffer, analysis, asr=result)
        assert draft.text == "mumble"
        assert any("no per-word confidence" in note for note in draft.notes)


class TestSpeakerTags:
    def test_two_speakers_are_tagged_at_their_turns(self, wav_two_speakers):
        buffer, analysis = analyse(wav_two_speakers)
        if analysis.speaker_count < 2:
            pytest.skip("diarizer did not separate the fixture's voices")
        segments = sorted(analysis.diarization.segments, key=lambda s: s.start)
        first, second = segments[0], next(
            s for s in segments if s.speaker_id != segments[0].speaker_id
        )
        draft = draft_from_words(
            words(
                ("hello", first.start + 0.1, first.start + 0.4, 0.95),
                ("goodbye", second.start + 0.1, second.start + 0.4, 0.95),
            ),
            analysis,
            buffer,
        )
        assert draft.text.startswith("<s1>")
        assert "<s2>" in draft.text
        assert draft.report.is_valid

    def test_same_speaker_is_not_retagged(self, wav_two_speakers):
        buffer, analysis = analyse(wav_two_speakers)
        segments = sorted(analysis.diarization.segments, key=lambda s: s.start)
        span = segments[0]
        draft = draft_from_words(
            words(
                ("one", span.start + 0.1, span.start + 0.3, 0.95),
                ("two", span.start + 0.35, span.start + 0.5, 0.95),
            ),
            analysis,
            buffer,
        )
        # Section 17 tags turns, not words; a repeated tag is a validator error.
        assert draft.text.count("<s1>") <= 1


class TestSpeakerNumbering:
    """Regression tests from a real clip whose draft opened on <s2>.

    The diarizer reported three speakers for one voice, the first of them
    carried no recognised words, and the draft emitted the diarizer's own ids --
    producing "<s2> ... <s3>" with no <s1>, a numbering error in text that was
    otherwise correct (section 18A.2).
    """

    @staticmethod
    def _diarization(spans):
        from app.speakers.diarization import DiarizationResult, SpeakerSegment

        segments = [
            SpeakerSegment(start, end, speaker_id, 0.6)
            for start, end, speaker_id in spans
        ]
        return DiarizationResult(
            segments, speaker_count=len({s.speaker_id for s in segments}), confidence=0.52
        )

    def test_speakers_are_renumbered_by_first_appearance_in_the_draft(
        self, wav_two_speakers
    ):
        buffer, analysis = analyse(wav_two_speakers)
        # s1 owns audio the recogniser produced no words for.
        analysis.diarization = self._diarization(
            [(0.0, 0.2, "s1"), (1.0, 1.5, "s2"), (2.0, 2.5, "s3")]
        )
        draft = draft_from_words(
            words(("hello", 1.1, 1.3, 0.95), ("there", 2.1, 2.4, 0.95)),
            analysis,
            buffer,
        )
        assert draft.text == "<s1> hello <s2> there"
        assert draft.report.is_valid

    def test_word_outside_every_segment_takes_the_nearest_speaker(
        self, wav_two_speakers
    ):
        buffer, analysis = analyse(wav_two_speakers)
        analysis.diarization = self._diarization(
            [(1.0, 1.5, "s1"), (2.0, 2.5, "s2")]
        )
        # 0.6s falls before the first segment; diarization boundaries and
        # recogniser timings come from different algorithms.
        draft = draft_from_words(
            words(("hello", 0.6, 0.8, 0.95), ("there", 2.1, 2.4, 0.95)),
            analysis,
            buffer,
        )
        assert draft.text.startswith("<s1> hello")
        assert draft.report.is_valid

    def test_uncertain_speaker_structure_is_flagged(self, wav_two_speakers):
        buffer, analysis = analyse(wav_two_speakers)
        analysis.diarization = self._diarization(
            [(1.0, 1.5, "s1"), (2.0, 2.5, "s2")]
        )
        draft = draft_from_words(
            words(("hello", 1.1, 1.3, 0.95), ("there", 2.1, 2.4, 0.95)),
            analysis,
            buffer,
        )
        assert any("speaker structure is a suggestion" in n for n in draft.notes)
        assert any("merge them in the Speakers panel" in n for n in draft.notes)

    def test_no_tags_when_every_word_landed_on_one_speaker(self, wav_two_speakers):
        """Regression: a two-speaker diarization whose words are all one voice.

        The draft tagged "<s1> ..." alone, which section 17 forbids -- the
        validator flagged speaker-single-tagged on otherwise correct text.
        """
        buffer, analysis = analyse(wav_two_speakers)
        analysis.diarization = self._diarization(
            [(1.0, 1.5, "s1"), (3.0, 3.5, "s2")]
        )
        draft = draft_from_words(
            words(("hello", 1.1, 1.2, 0.95), ("there", 1.3, 1.4, 0.95)),
            analysis,
            buffer,
        )
        assert "<s1>" not in draft.text
        assert draft.report.is_valid
        assert any("no speaker tag was written" in note for note in draft.notes)

    def test_no_speaker_tag_when_the_diarizer_found_one_voice(self, wav_two_speakers):
        buffer, analysis = analyse(wav_two_speakers)
        analysis.diarization = self._diarization([(1.0, 2.5, "s1")])
        draft = draft_from_words(
            words(("hello", 1.1, 1.3, 0.95), ("there", 2.1, 2.4, 0.95)),
            analysis,
            buffer,
        )
        assert "<s1>" not in draft.text
        assert draft.report.is_valid


class TestEventTags:
    def _confident(self, analysis, event_type, tag_time=0.5):
        event = AnalysisEvent(event_type, tag_time, tag_time + 0.3, confidence=0.9)
        analysis.events.append(event)
        return event

    def test_confident_laughter_reaches_the_draft(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        self._confident(analysis, EventType.POSSIBLE_LAUGHTER, 0.45)
        draft = draft_from_words(
            words(("hello", 0.1, 0.4, 0.95), ("there", 0.9, 1.2, 0.95)),
            analysis,
            buffer,
        )
        assert T.LAUGHTER in draft.text
        assert draft.report.is_valid

    def test_weak_detection_contributes_nothing(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        analysis.events.append(
            AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 0.45, 0.7, confidence=0.2)
        )
        draft = draft_from_words(
            words(("hello", 0.1, 0.4, 0.95), ("there", 0.9, 1.2, 0.95)),
            analysis,
            buffer,
        )
        assert T.LAUGHTER not in draft.text
        assert draft.used_events == []

    def test_tags_at_one_place_are_alphabetical(self, wav_speech):
        # Section 41: several tags in one position go in alphabetical order.
        buffer, analysis = analyse(wav_speech)
        self._confident(analysis, EventType.POSSIBLE_LAUGHTER, 0.45)
        self._confident(analysis, EventType.POSSIBLE_ARTIFACT, 0.46)
        draft = draft_from_words(
            words(("hello", 0.1, 0.4, 0.95), ("there", 0.9, 1.2, 0.95)),
            analysis,
            buffer,
        )
        assert f"{T.ARTIFACT} {T.LAUGHTER}" in draft.text
        assert draft.report.is_valid

    def test_repeated_sound_collapses_to_one_tag(self, wav_speech):
        # Section 8: the same sound repeated in immediate succession is one tag.
        buffer, analysis = analyse(wav_speech)
        self._confident(analysis, EventType.POSSIBLE_HN, 0.45)
        self._confident(analysis, EventType.POSSIBLE_HN, 0.47)
        draft = draft_from_words(
            words(("hello", 0.1, 0.4, 0.95), ("there", 0.9, 1.2, 0.95)),
            analysis,
            buffer,
        )
        assert draft.text.count(T.NON_LEXICAL) == 1

    def test_pause_never_becomes_a_tag(self, wav_two_speakers):
        # Sections 13 and 56.10.
        buffer, analysis = analyse(wav_two_speakers)
        assert analysis.pause_events, "fixture should contain a pause"
        draft = draft_from_words(words(("hello", 0.4, 0.8, 0.95)), analysis, buffer)
        assert "[pause]" not in draft.text
        assert all(
            event.type is not EventType.PAUSE for event in draft.used_events
        )

    def test_used_events_are_reported_for_the_caller_to_record(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        event = self._confident(analysis, EventType.POSSIBLE_LAUGHTER, 0.45)
        draft = draft_from_words(
            words(("hello", 0.1, 0.4, 0.95), ("there", 0.9, 1.2, 0.95)),
            analysis,
            buffer,
        )
        assert event in draft.used_events
        # The builder does not confirm anything itself; that is the caller's
        # decision to record (section 47).
        assert not event.is_confirmed


class TestOverlap:
    def test_overlap_needs_two_speakers_inside_the_block(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        analysis.events.append(
            AnalysisEvent(EventType.POSSIBLE_OVERLAP, 0.2, 1.0, confidence=0.9)
        )
        draft = draft_from_words(
            words(("stop", 0.3, 0.5, 0.95), ("stop", 0.6, 0.8, 0.95)),
            analysis,
            buffer,
        )
        # One speaker means an <ol> block would be malformed under section 18,
        # so the detection is left for a human instead.
        assert T.OVERLAP_OPEN not in draft.text
        assert any("not drafted as <ol>" in note for note in draft.notes)
        assert draft.report.is_valid


class TestWholeFileConditions:
    def test_silence_drafts_na_alone(self, wav_silence):
        buffer, analysis = analyse(wav_silence)
        draft = DraftBuilder().build(buffer, analysis, asr=AsrResult([], "stub", "m"))
        assert draft.text == T.NO_AUDIO
        assert draft.whole_file == T.NO_AUDIO
        assert draft.report.is_valid

    def test_no_audio_skips_recognition_entirely(self, wav_silence):
        # Running ASR over a blank file could only invent words.
        buffer, analysis = analyse(wav_silence)
        draft = DraftBuilder().build(buffer, analysis)
        assert draft.text == T.NO_AUDIO
        assert draft.asr is None

    def test_hiss_with_no_words_drafts_ns(self, wav_hiss):
        buffer, analysis = analyse(wav_hiss)
        draft = DraftBuilder().build(buffer, analysis, asr=AsrResult([], "stub", "m"))
        assert draft.text in (T.NO_SPEECH, T.GARBLED)
        assert draft.report.is_valid

    def test_speech_with_no_words_drafts_unintelligible(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = DraftBuilder().build(buffer, analysis, asr=AsrResult([], "stub", "m"))
        assert draft.text == T.UNINTELLIGIBLE
        assert any("unintelligible" in note for note in draft.notes)

    def test_laughter_only_clip_drafts_the_laughter_tag(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        analysis.events.append(
            AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 0.2, 1.0, confidence=0.9)
        )
        draft = DraftBuilder().build(buffer, analysis, asr=AsrResult([], "stub", "m"))
        assert draft.text == T.LAUGHTER
        assert draft.report.is_valid

    def test_whole_file_text_stands_alone(self, wav_silence):
        # Section 42: nothing accompanies a whole-file condition.
        buffer, analysis = analyse(wav_silence)
        draft = DraftBuilder().build(buffer, analysis)
        assert draft.text.split() == [draft.text.strip()]


class TestDraftMetadata:
    def test_source_label_is_the_draft_label(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(words(("hello", 0.1, 0.4, 0.9)), analysis, buffer)
        assert draft.source_label == "asr_draft"

    def test_changes_are_listed(self, wav_speech):
        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(words(("Hello,", 0.1, 0.4, 0.9)), analysis, buffer)
        assert draft.changes
        assert any("lowercas" in change for change in draft.changes)

    def test_dict_round_trips_through_json(self, wav_speech):
        import json

        buffer, analysis = analyse(wav_speech)
        draft = draft_from_words(words(("hello", 0.1, 0.4, 0.9)), analysis, buffer)
        payload = json.loads(json.dumps(draft.to_dict()))
        assert payload["source"] == "asr_draft"
        assert payload["text"] == "hello"

    def test_unavailable_recogniser_is_reported_not_raised(self, wav_speech, monkeypatch):
        import app.asr.draft as draft_module

        def boom(config):
            raise AsrError("no backend installed")

        monkeypatch.setattr(draft_module, "get_engine", boom)
        buffer, analysis = analyse(wav_speech)
        draft = DraftBuilder().build(buffer, analysis)
        # No words, but the clip still gets an honest draft plus the reason.
        assert any("no backend installed" in note for note in draft.notes)
        assert draft.text

    def test_long_audio_is_refused_by_the_engine_limit(self, wav_speech):
        from dataclasses import replace

        buffer, analysis = analyse(wav_speech)
        long_buffer = replace(
            buffer, samples=np.zeros(buffer.sample_rate * 60, dtype=np.float32)
        )
        config = AsrConfig(max_audio_seconds=10)

        # The ceiling lives on AsrEngine.transcribe, before any backend work.
        from app.asr.engine import AsrEngine

        engine = AsrEngine(config)
        engine._loaded = True
        with pytest.raises(AsrError, match="above the ASR limit"):
            engine.transcribe(long_buffer)
