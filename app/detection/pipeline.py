"""Analysis pipeline (CLAUDE.md sections 3, 45, 46, 47).

Runs every detector in dependency order on one shared frame grid and returns a
single event timeline.

Order matters, and encodes rules rather than convenience:

  1. VAD          -- speech regions; everything downstream is scoped by them
  2. silence      -- gaps, and which gaps are pauses
  3. laughter     -- first among the vocal detectors, because section 8 gives
                     laughter its own tag and it must win ties
  4. non-lexical  -- [hn], suppressed where laughter already claimed the audio
  5. artifact     -- technical noise only, suppressed where a vocal sound was
                     found (section 9)
  6. background   -- unattributable sound, outside the identified speech
  7. overlap      -- simultaneous voices
  8. quality      -- whole-file conditions, informed by whether speech and
                     non-lexical sound were found (section 12 needs both)
  9. diarization  -- speaker attribution over the speech regions

The result is Layer A only (CLAUDE.md section 3). No step here writes
transcription text, and every uncertain event arrives as `possible_*` awaiting
human confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.audio.loader import AudioBuffer
from app.audio.preprocessing import FrameGrid, compute_grid
from app.audio.quality import QualityAnalyzer, QualityReport
from app.config import DetectionConfig
from app.detection.artifact import ArtifactDetector
from app.detection.background import BackgroundDetector
from app.detection.laughter import LaughterDetector
from app.detection.nonlexical import NonLexicalDetector
from app.detection.overlap import OverlapDetector
from app.detection.silence import SilenceDetector, SilenceResult
from app.detection.vad import EnergyVad, VadResult
from app.models.event import AnalysisEvent, EventType, format_timeline
from app.speakers.diarization import Diarizer, DiarizationResult


@dataclass
class AnalysisResult:
    """Everything the analysis engine found for one file."""

    buffer: AudioBuffer
    grid: FrameGrid
    vad: VadResult
    silence: SilenceResult
    quality: QualityReport
    diarization: DiarizationResult
    events: list[AnalysisEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    @property
    def pending(self) -> list[AnalysisEvent]:
        """Events awaiting human confirmation (section 47)."""
        return [e for e in self.events if e.requires_human_confirmation]

    @property
    def speech_events(self) -> list[AnalysisEvent]:
        return [e for e in self.events if e.type is EventType.SPEECH]

    @property
    def pause_events(self) -> list[AnalysisEvent]:
        return [e for e in self.events if e.type is EventType.PAUSE]

    def of_type(self, event_type: EventType) -> list[AnalysisEvent]:
        return [e for e in self.events if e.type is event_type]

    @property
    def speaker_count(self) -> int:
        """Suggested speaker count. Editable by the annotator (section 18A.1)."""
        return self.diarization.speaker_count

    @property
    def whole_file_candidates(self) -> list[str]:
        return list(self.quality.candidates)

    def timeline(self) -> str:
        """The analysis timeline in the format of CLAUDE.md section 3."""
        return format_timeline(self.events)

    def summary(self) -> dict:
        return {
            "duration": round(self.buffer.duration, 3),
            "speech_regions": len(self.speech_events),
            "speech_ratio": round(self.vad.speech_ratio, 3),
            "pauses": len(self.pause_events),
            "longest_pause": round(self.silence.longest_pause, 3),
            "speaker_count": self.speaker_count,
            "speaker_count_confidence": round(self.diarization.confidence, 2),
            "pending_confirmations": len(self.pending),
            "whole_file_candidates": self.whole_file_candidates,
            "quality": self.quality.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }


class AnalysisPipeline:
    """Runs the full assistive analysis for one audio file."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.vad = EnergyVad(self.config)
        self.silence = SilenceDetector(self.config)
        self.laughter = LaughterDetector(self.config)
        self.non_lexical = NonLexicalDetector(self.config)
        self.artifact = ArtifactDetector(self.config)
        self.background = BackgroundDetector(self.config)
        self.overlap = OverlapDetector(self.config)
        self.quality = QualityAnalyzer(self.config)
        self.diarizer = Diarizer(self.config)

    def run(
        self,
        buffer: AudioBuffer,
        include_diarization: bool = True,
        max_speakers: int | None = None,
    ) -> AnalysisResult:
        grid = compute_grid(buffer.samples, buffer.sample_rate)

        vad_result = self.vad.analyse(buffer, grid)
        speech_spans = vad_result.speech
        silence_result = self.silence.analyse(buffer, grid, vad_result)

        events: list[AnalysisEvent] = []
        events.extend(self.silence.detect(buffer, grid, vad_result))

        # Laughter runs before [hn] so it can claim its audio first: section 8
        # forbids writing laughter as [hn].
        laughter_events = self.laughter.detect(buffer, grid, speech_spans=speech_spans)
        events.extend(laughter_events)

        hn_events = self.non_lexical.detect(
            buffer, grid, speech_spans=speech_spans, laughter_events=laughter_events
        )
        events.extend(hn_events)

        # [artifact] is technical noise; a vocal sound is not an artifact.
        events.extend(
            self.artifact.detect(buffer, grid, vocal_events=laughter_events + hn_events)
        )
        events.extend(self.background.detect(buffer, grid, speech_spans=speech_spans))
        events.extend(self.overlap.detect(buffer, grid, speech_spans=speech_spans))

        # Section 12 needs both facts: <ns> requires no speech AND no other
        # applicable tag, so the quality analyser is told what was found.
        quality_report = self.quality.analyse(
            buffer,
            grid,
            has_speech=vad_result.has_speech,
            has_non_lexical=bool(laughter_events or hn_events),
        )
        events.extend(
            self.quality.detect(
                buffer,
                grid,
                has_speech=vad_result.has_speech,
                has_non_lexical=bool(laughter_events or hn_events),
            )
        )

        if include_diarization and speech_spans:
            diarization = self.diarizer.diarize(
                buffer, speech_spans, grid, max_speakers=max_speakers
            )
            # Diarized speech events carry speaker attribution, so they replace
            # the unattributed SPEECH events from the VAD.
            events = [e for e in events if e.type is not EventType.SPEECH]
            events.extend(diarization.to_events())
        else:
            diarization = DiarizationResult([], 0, 0.0)
            events.extend(self.vad.detect(buffer, grid))

        events.sort(key=lambda e: (e.start, e.end, e.type.value))
        return AnalysisResult(
            buffer=buffer,
            grid=grid,
            vad=vad_result,
            silence=silence_result,
            quality=quality_report,
            diarization=diarization,
            events=events,
        )


def analyse(
    buffer: AudioBuffer, config: DetectionConfig | None = None
) -> AnalysisResult:
    """Run the full analysis pipeline on a decoded buffer."""
    return AnalysisPipeline(config).run(buffer)
