"""Build an Ermis V2 draft transcription from audio.

Inputs: the recogniser's words (Layer A words), the analysis events, and the
diarization result. Output: one line of V2-shaped text plus the record of how
it was produced.

    ASR words ---> spoken form ---> speaker turns ---> event tags ---> validate

What the builder writes:

  - the words, in spoken form (`app.ermis.spoken_form`)
  - `<s1>` / `<s2>` ... at speaker turns, only when more than one speaker was
    detected, numbered by first appearance (sections 17, 18A.2)
  - `<ol> ... </ol>` around intelligible overlap, but only when the words
    inside really do come from two speakers (section 18)
  - `[laughter]` / `[hn]` / `[artifact]` / `[bg]` where a confident detector
    put them, alphabetically ordered and collapsed when repeated
    (sections 8, 40, 41)
  - `((word))` where the recogniser itself was unsure, and `(())` when it
    produced nothing for speech that is audibly there (section 14)
  - a whole-file condition alone -- `<na>`, `<ga>`, `<ns>` -- when the audio
    says so, with nothing else beside it (section 42)

What it will not write: a pause tag (section 13), a retired V1 tag (section 39),
or a guess dressed up as certainty. A low-confidence word becomes unintelligible
rather than an assertion, and a detector below threshold contributes nothing and
stays in the confirmation queue.

The draft is a starting point for a human, not a submission. Every draft is
validated before it is handed over, and the audio stays the source of truth
(section 59).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.asr.engine import AsrEngine, AsrError, AsrResult, AsrWord, get_engine
from app.audio.loader import AudioBuffer
from app.config import AsrConfig, ValidationConfig
from app.detection.pipeline import AnalysisResult
from app.ermis import tags as T
from app.ermis.findings import ValidationReport
from app.ermis.spoken_form import normalize_tokens
from app.ermis.validator import validate
from app.models.event import AnalysisEvent, EventType

log = logging.getLogger("ermis.draft")

#: Event types that contribute an inline tag to a draft, and the tag each one
#: contributes. Pause and silence are absent on purpose (section 13).
_TAGGED_EVENTS: dict[EventType, str] = {
    EventType.POSSIBLE_LAUGHTER: T.LAUGHTER,
    EventType.POSSIBLE_HN: T.NON_LEXICAL,
    EventType.POSSIBLE_ARTIFACT: T.ARTIFACT,
    EventType.POSSIBLE_BACKGROUND: T.BACKGROUND,
}


@dataclass(frozen=True)
class DraftWord:
    """One word in the draft, with where it came from in the audio."""

    text: str
    start: float
    end: float
    confidence: float
    speaker_id: str | None = None
    #: True when the recogniser's confidence was below the threshold, so the
    #: word is written as unintelligible rather than asserted.
    unclear: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "speaker": self.speaker_id,
            "unclear": self.unclear,
        }


@dataclass
class DraftResult:
    """A machine-produced draft, ready for human correction."""

    text: str
    words: list[DraftWord] = field(default_factory=list)
    speaker_count: int = 0
    #: Set when the whole clip is one condition and the text is that tag alone.
    whole_file: str | None = None
    #: (time, tag) for every tag the builder inserted.
    inserted_tags: list[tuple[float, str]] = field(default_factory=list)
    #: Events whose tags reached the draft. The caller records the decision.
    used_events: list[AnalysisEvent] = field(default_factory=list)
    asr: AsrResult | None = None
    report: ValidationReport | None = None
    #: Form changes applied to the recogniser's words, deduplicated.
    changes: list[str] = field(default_factory=list)
    #: Things the annotator should know before trusting the draft.
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def unclear_count(self) -> int:
        return sum(1 for word in self.words if word.unclear)

    @property
    def source_label(self) -> str:
        """How this text must be labelled wherever it is stored."""
        return "asr_draft"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source_label,
            "speaker_count": self.speaker_count,
            "whole_file": self.whole_file,
            "word_count": len(self.words),
            "unclear_words": self.unclear_count,
            "inserted_tags": [
                {"at": round(at, 3), "tag": tag} for at, tag in self.inserted_tags
            ],
            "asr": self.asr.to_dict() if self.asr else None,
            "validation": self.report.to_dict() if self.report else None,
            "changes": list(self.changes),
            "notes": list(self.notes),
            "words": [w.to_dict() for w in self.words],
        }


class DraftBuilder:
    """Turns audio plus analysis into a validated V2 draft."""

    def __init__(
        self,
        config: AsrConfig | None = None,
        validation: ValidationConfig | None = None,
        engine: AsrEngine | None = None,
    ) -> None:
        self.config = config or AsrConfig()
        self.validation = validation or ValidationConfig()
        # A caller that already holds a loaded engine passes it in; everyone
        # else shares the process-wide one. A builder must never load a model
        # of its own: drafting a folder of clips would then load the model once
        # per clip, which on Windows fails with "the paging file is too small"
        # and quietly costs every draft its words.
        self._engine = engine

    # ------------------------------------------------------------------ entry
    def build(
        self,
        buffer: AudioBuffer,
        analysis: AnalysisResult,
        asr: AsrResult | None = None,
    ) -> DraftResult:
        """Draft one clip. Recognises the audio unless `asr` is supplied."""
        notes: list[str] = []

        whole_file = self._whole_file_condition(analysis)
        if whole_file in (T.NO_AUDIO, T.GARBLED):
            # Nothing may accompany these, and running ASR over unusable audio
            # would only invent words (sections 10, 11, 42).
            return self._whole_file_result(whole_file, analysis, notes)

        if asr is None:
            asr = self._recognise(buffer, notes)

        if asr is None or not asr.words:
            return self._nothing_recognised(analysis, whole_file, asr, notes)

        words = self._to_draft_words(asr, analysis, notes)
        if not words:
            return self._nothing_recognised(analysis, whole_file, asr, notes)

        tokens = self._assemble(words, analysis, notes)
        text = " ".join(tokens).strip()

        # Validate against the speakers the draft actually tagged. The diarizer
        # may report more than the words support, and checking the text against
        # a count it does not use would produce findings about the detector
        # rather than about the transcription.
        tagged = len(self._numbering)
        report = validate(
            text,
            speaker_count=tagged or analysis.speaker_count or None,
            known_speakers=set(range(1, tagged + 1)) or None,
        )
        if tagged > 1 and analysis.diarization.confidence < 0.7:
            notes.append(
                f"speaker structure is a suggestion: {tagged} speaker(s) tagged at "
                f"{analysis.diarization.confidence * 100:.0f}% confidence. If one "
                "person was split in two, merge them in the Speakers panel and fix "
                "the tags (sections 18A.8, 18A.11)."
            )
        if not report.is_valid:
            notes.append(
                "the draft has validation errors: "
                + ", ".join(sorted({f.code.value for f in report.errors}))
                + " -- fix them against the audio before submitting"
            )

        return DraftResult(
            text=text,
            words=words,
            speaker_count=analysis.speaker_count,
            whole_file=None,
            inserted_tags=self._inserted,
            used_events=self._used_events,
            asr=asr,
            report=report,
            changes=self._changes,
            notes=notes,
        )

    # ------------------------------------------------------------------- ASR
    def _recognise(self, buffer: AudioBuffer, notes: list[str]) -> AsrResult | None:
        try:
            if self._engine is None:
                self._engine = get_engine(self.config)
            return self._engine.transcribe(buffer)
        except AsrError as exc:
            notes.append(f"no words were drafted: {exc}")
            log.warning("ASR unavailable: %s", exc)
            return None

    # -------------------------------------------------------- whole-file path
    def _whole_file_condition(self, analysis: AnalysisResult) -> str | None:
        """The whole-file tag the audio suggests, if any (section 42)."""
        for candidate in (T.NO_AUDIO, T.GARBLED, T.NO_SPEECH):
            if candidate in analysis.whole_file_candidates:
                return candidate
        return None

    def _whole_file_result(
        self, tag: str, analysis: AnalysisResult, notes: list[str]
    ) -> DraftResult:
        notes.append(
            f"whole-file condition {tag} drafted from the audio measurements; "
            "it must stand alone (section 42). Confirm by listening."
        )
        notes.extend(analysis.quality.notes)
        return DraftResult(
            text=tag,
            speaker_count=0,
            whole_file=tag,
            report=validate(tag),
            notes=notes,
        )

    def _nothing_recognised(
        self,
        analysis: AnalysisResult,
        whole_file: str | None,
        asr: AsrResult | None,
        notes: list[str],
    ) -> DraftResult:
        """No words came back. Decide what the clip is instead of guessing."""
        laughter = self._confident(analysis, EventType.POSSIBLE_LAUGHTER)
        non_lexical = self._confident(analysis, EventType.POSSIBLE_HN)

        if laughter:
            tag = T.LAUGHTER
            notes.append("no speech was recognised; the clip is laughter only (section 42)")
            used = laughter
        elif non_lexical:
            tag = T.NON_LEXICAL
            notes.append(
                "no speech was recognised; the clip is a non-lexical sound only "
                "(section 42)"
            )
            used = non_lexical
        elif whole_file:
            tag = whole_file
            notes.append(f"no speech was recognised; {tag} suggested by the audio")
            used = []
        elif analysis.speech_events:
            tag = T.UNINTELLIGIBLE
            notes.append(
                "speech was detected but nothing could be recognised, so the "
                "draft is unintelligible (section 14). Replay and slow the audio "
                "down before accepting it."
            )
            used = []
        else:
            tag = T.NO_SPEECH
            notes.append("no speech detected and no other condition applies (section 12)")
            used = []

        return DraftResult(
            text=tag,
            speaker_count=analysis.speaker_count,
            whole_file=tag if tag in T.WHOLE_FILE_TAGS else None,
            used_events=used,
            asr=asr,
            report=validate(tag),
            notes=notes,
        )

    # ------------------------------------------------------------------ words
    def _to_draft_words(
        self, asr: AsrResult, analysis: AnalysisResult, notes: list[str]
    ) -> list[DraftWord]:
        """Normalise recognised words and attach speakers and confidence."""
        raw = list(asr.words)
        normalized = normalize_tokens([word.text for word in raw])

        changes: list[str] = []
        words: list[DraftWord] = []
        for item in normalized:
            sources = [raw[i] for i in item.source_indices if i < len(raw)]
            if not sources:
                continue
            start = min(source.start for source in sources)
            end = max(source.end for source in sources)
            confidence = min(source.confidence for source in sources)
            unclear = (
                asr.has_confidence and confidence < self.config.min_word_confidence
            )
            words.append(
                DraftWord(
                    text=item.text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    speaker_id=_speaker_at(analysis, (start + end) / 2),
                    unclear=unclear,
                )
            )
            changes.extend(item.changes)

        self._changes = list(dict.fromkeys(changes))
        if not asr.has_confidence:
            notes.append(
                "this backend reports no per-word confidence, so no word was "
                "marked unintelligible automatically -- check the audio yourself"
            )
        elif any(word.unclear for word in words):
            count = sum(1 for word in words if word.unclear)
            notes.append(
                f"{count} word(s) were recognised with low confidence and are "
                "written as unintelligible; replay them and either correct the "
                "word or keep the (()) form (section 14)"
            )
        return words

    # --------------------------------------------------------------- assembly
    def _assemble(
        self, words: list[DraftWord], analysis: AnalysisResult, notes: list[str]
    ) -> list[str]:
        """Interleave words, speaker tags, event tags and overlap blocks."""
        self._inserted: list[tuple[float, str]] = []
        self._used_events: list[AnalysisEvent] = []

        # Tagging needs two speakers in the WORDS, not just in the diarization.
        # A clip where the diarizer heard two voices but every recognised word
        # landed on one of them would otherwise be tagged "<s1> ..." alone,
        # which section 17 forbids: single-speaker audio takes no speaker tag.
        voices = {word.speaker_id for word in words if word.speaker_id}
        multi_speaker = analysis.speaker_count > 1 and len(voices) > 1
        if analysis.speaker_count > 1 and len(voices) < 2:
            notes.append(
                f"the diarizer suggested {analysis.speaker_count} speakers but every "
                "recognised word fell on one of them, so no speaker tag was written. "
                "If a second speaker really is audible, tag the turns yourself "
                "(sections 17, 18A.11)."
            )
        overlap_spans = self._overlap_spans(words, analysis, notes)
        event_slots = self._event_slots(words, analysis)

        tokens: list[str] = []
        current_speaker: str | None = None
        open_overlap = False
        unclear_run: list[str] = []

        # Speaker tags are numbered by first appearance IN THE DRAFT, not by the
        # diarizer's own ids (section 18A.2). The diarizer can report a speaker
        # whose segments carry no recognised words, and emitting its id would
        # leave a gap -- "<s2> ... <s3>" with no <s1> -- which is a numbering
        # error the annotator would have to clean up by hand.
        numbering: dict[str, int] = {}
        self._numbering = numbering

        def tag_for(speaker_id: str) -> str:
            if speaker_id not in numbering:
                numbering[speaker_id] = len(numbering) + 1
            return f"<s{numbering[speaker_id]}>"

        def flush_unclear() -> None:
            nonlocal unclear_run
            if unclear_run:
                tokens.append(f"(({' '.join(unclear_run)}))")
                unclear_run = []

        for index, word in enumerate(words):
            for tag in event_slots.get(index, ()):
                flush_unclear()
                _append_tag(tokens, tag)

            span = _span_starting_at(overlap_spans, index)
            if span is not None:
                flush_unclear()
                tokens.append(T.OVERLAP_OPEN)
                open_overlap = True
                # Every speaker is re-tagged inside an overlap block, so the
                # speaker that opened it does not carry over (section 18).
                current_speaker = None

            if multi_speaker and word.speaker_id and word.speaker_id != current_speaker:
                flush_unclear()
                tokens.append(tag_for(word.speaker_id))
                current_speaker = word.speaker_id

            if word.unclear:
                unclear_run.append(word.text)
            else:
                flush_unclear()
                tokens.append(word.text)

            if open_overlap and _span_ending_at(overlap_spans, index):
                flush_unclear()
                tokens.append(T.OVERLAP_CLOSE)
                open_overlap = False
                current_speaker = None

        flush_unclear()
        if open_overlap:  # pragma: no cover - spans are clamped to the words
            tokens.append(T.OVERLAP_CLOSE)

        for tag in event_slots.get(len(words), ()):
            _append_tag(tokens, tag)
        return tokens

    def _event_slots(
        self, words: list[DraftWord], analysis: AnalysisResult
    ) -> dict[int, list[str]]:
        """Map word index -> tags that belong immediately before that word."""
        slots: dict[int, list[str]] = {}
        for event in analysis.events:
            tag = _TAGGED_EVENTS.get(event.type)
            if tag is None:
                continue
            if event.confidence < self.config.min_event_confidence:
                continue
            index = _slot_for(words, event.start)
            slots.setdefault(index, [])
            if tag not in slots[index]:
                slots[index].append(tag)
            self._used_events.append(event)
            self._inserted.append((event.start, tag))
        # Section 41: several tags at one place go in alphabetical order.
        for index in slots:
            slots[index].sort()
        return slots

    def _overlap_spans(
        self, words: list[DraftWord], analysis: AnalysisResult, notes: list[str]
    ) -> list[tuple[int, int]]:
        """Word ranges to wrap in <ol> ... </ol>.

        A block is only produced when the words inside come from at least two
        speakers. Otherwise the block would be malformed -- section 18 requires
        every speaker to be tagged inside it -- so the detection is left in the
        confirmation queue for a human instead.
        """
        spans: list[tuple[int, int]] = []
        for event in analysis.of_type(EventType.POSSIBLE_OVERLAP):
            if event.confidence < self.config.min_event_confidence:
                continue
            inside = [
                index
                for index, word in enumerate(words)
                if event.start <= (word.start + word.end) / 2 <= event.end
            ]
            speakers = {words[i].speaker_id for i in inside if words[i].speaker_id}
            if len(inside) < 2 or len(speakers) < 2:
                notes.append(
                    f"possible overlap at {event.start:.2f}s was not drafted as "
                    "<ol>: the recogniser attributed the words to one speaker. "
                    "Listen and decide between <ol> and <ct> (sections 18, 19)."
                )
                continue
            spans.append((min(inside), max(inside)))
            self._used_events.append(event)
            self._inserted.append((event.start, T.OVERLAP_OPEN))
        return spans

    def _confident(
        self, analysis: AnalysisResult, event_type: EventType
    ) -> list[AnalysisEvent]:
        return [
            event
            for event in analysis.of_type(event_type)
            if event.confidence >= self.config.min_event_confidence
        ]

    # Defaults so a whole-file path can return without running _assemble.
    _inserted: list[tuple[float, str]] = []
    _used_events: list[AnalysisEvent] = []
    _changes: list[str] = []
    _numbering: dict[str, int] = {}


# ---------------------------------------------------------------- helpers
def _append_tag(tokens: list[str], tag: str) -> None:
    """Add a tag, collapsing an immediate repeat of the same sound (section 8)."""
    if tokens and tokens[-1] == tag and tag in T.NON_LEXICAL_SOUND_TAGS:
        return
    if tokens and tokens[-1] == tag:
        return
    tokens.append(tag)


def _slot_for(words: list[DraftWord], when: float) -> int:
    """Index of the first word starting at or after `when`."""
    for index, word in enumerate(words):
        if word.start >= when:
            return index
    return len(words)


def _span_starting_at(spans: list[tuple[int, int]], index: int) -> tuple[int, int] | None:
    for span in spans:
        if span[0] == index:
            return span
    return None


def _span_ending_at(spans: list[tuple[int, int]], index: int) -> bool:
    return any(span[1] == index for span in spans)


def _speaker_at(analysis: AnalysisResult, when: float) -> str | None:
    """Which speaker the diarizer put at this moment.

    A word can fall just outside every segment -- diarization boundaries and
    recogniser timings come from different algorithms -- so the nearest segment
    is used rather than leaving the word unattributed. An untagged first word
    would produce a draft that opens on `<s2>`, which is a numbering error in
    text that is otherwise correct.
    """
    segments = analysis.diarization.segments
    if not segments:
        return None
    for segment in segments:
        if segment.start <= when <= segment.end:
            return segment.speaker_id
    nearest = min(
        segments,
        key=lambda s: 0.0 if s.start <= when <= s.end else min(
            abs(when - s.start), abs(when - s.end)
        ),
    )
    return nearest.speaker_id


def build_draft(
    buffer: AudioBuffer,
    analysis: AnalysisResult,
    config: AsrConfig | None = None,
    asr: AsrResult | None = None,
    engine: AsrEngine | None = None,
) -> DraftResult:
    """Draft one clip. Repeated calls share one loaded model."""
    return DraftBuilder(config, engine=engine).build(buffer, analysis, asr=asr)


def draft_from_words(
    words: list[AsrWord],
    analysis: AnalysisResult,
    buffer: AudioBuffer,
    config: AsrConfig | None = None,
    backend: str = "supplied",
) -> DraftResult:
    """Draft from words that were recognised elsewhere.

    Used by the tests and by any caller that already has recogniser output, so
    the merge logic can be exercised without loading a model.
    """
    result = AsrResult(
        words=list(words),
        backend=backend,
        model=(config or AsrConfig()).model,
        duration=buffer.duration,
    )
    return DraftBuilder(config).build(buffer, analysis, asr=result)
