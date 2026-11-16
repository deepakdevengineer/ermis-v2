"""Application configuration.

Privacy contract (CLAUDE.md section 53):
  - all processing is local
  - no automatic upload, no external transcription API, no telemetry
  - no cloud storage

SECURITY: this module stores NO credentials. There are deliberately no
password, token, api_key or secret fields, and none may be added. The
application never authenticates against a remote service, so it never needs
one. If an opt-in external service is ever introduced it must read its
credential from the process environment at call time and must not persist it.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "ermis-local-assistant"
APP_VERSION = "0.1.0"

# Ermis workflow version this build implements. The validator rejects V1 tags.
ERMIS_VERSION = "V2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_FORBIDDEN_CONFIG_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
)


@dataclass
class AudioConfig:
    """Audio decode / analysis parameters."""

    # Analysis sample rate. 16 kHz mono is the standard rate for VAD and
    # diarization models, and is enough for every QA decision this tool makes.
    target_sample_rate: int = 16_000
    target_channels: int = 1

    # Waveform peak envelope resolution, in buckets per second of audio.
    waveform_buckets_per_second: int = 200

    # Playback speeds offered in the editor (CLAUDE.md section 48).
    playback_speeds: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

    # Containers the loader accepts. WAV is handled by stdlib; the rest need
    # the ffmpeg executable on PATH.
    supported_extensions: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".wma",
        ".webm",
    )

    # Hard ceiling so a stray multi-hour file cannot exhaust memory.
    max_duration_seconds: float = 3 * 60 * 60

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


@dataclass
class RecordingConfig:
    """Microphone capture parameters (CLAUDE.md sections 45, 53).

    Capture is local: the input device is written to a WAV file under
    `PathsConfig.recordings_dir` and then loaded through the ordinary
    AudioLoader, so a recorded clip and an attached file take the same analysis
    path. Nothing is uploaded, and recording produces audio only -- never
    transcription text.
    """

    # Preferred capture rate, requested from the device first. When the device
    # refuses it, capture falls back to the device's own default rate and the
    # file is written at that rate; the loader resamples the analysis copy
    # either way, so the file on disk stays the source of truth (section 59).
    sample_rate: int = 48_000

    # Mono by default: the analysis copy is mono anyway, and a single channel
    # halves the file for no loss of QA information.
    channels: int = 1

    # PCM width written to disk: 16, 24 or 32 bit.
    sample_width_bits: int = 16

    # Hard ceiling. Capture auto-stops here, so a session left running cannot
    # fill the disk.
    max_duration_seconds: float = 30 * 60

    # Input device index, or None for the system default.
    device: int | None = None

    filename_prefix: str = "recording"

    # Level-meter refresh interval used by the recording UI, in milliseconds.
    meter_interval_ms: int = 50

    # Samples at or above this amplitude count as clipped. The count is
    # reported to the annotator, who decides whether it means [artifact],
    # <ga> or nothing at all (sections 9, 10, 47).
    clip_threshold: float = 0.999


@dataclass
class AsrConfig:
    """Local speech recognition used to DRAFT a transcription.

    Deviation notice: CLAUDE.md sections 1 and 56.1 prohibit ASR-generated
    transcription. This section exists because the project owner explicitly
    asked for the words to be produced from the audio. The compliance decision
    is theirs; the code's job is to keep it honest:

      - the model runs locally, and no audio is uploaded (`allow_external_asr`
        stays False and still refuses to be enabled)
      - draft text is labelled `asr_draft` everywhere it is stored or exported,
        never as human work
      - the draft stays editable, and the audio remains the source of truth
        (section 59)
    """

    enabled: bool = True

    # Run the recogniser automatically when a clip is loaded or recorded.
    # Turn it off to keep the load path analysis-only.
    auto_draft_on_load: bool = True

    # Hugging Face model id, or a size name for the faster-whisper backend.
    # ".en" models are English-only and noticeably better on English clips.
    model: str = "openai/whisper-small.en"

    # "auto" prefers faster-whisper, then transformers.
    backend: str = "auto"

    # Local directory holding the model, when it is not in the standard cache.
    model_dir: str = ""

    device: str = "cpu"

    # faster-whisper compute type. int8 is the fast CPU default.
    compute_type: str = "int8"

    # Dynamically quantise the transformers backend's linear layers to int8 on
    # CPU. Measured on this project: about four times faster with no change to
    # the recognised words on clean speech, which is what makes drafting usable
    # at roughly one times realtime instead of five.
    quantize_dynamic: bool = True

    # Run one throwaway inference at load time so the first real clip does not
    # pay for lazy kernel initialisation (which costs several seconds).
    warm_up: bool = True

    # Load the model in the background when the window opens, rather than on
    # the first clip. Loading takes tens of seconds; paying it at start-up means
    # a recording is usually drafted the moment it stops.
    preload_on_start: bool = True

    language: str = "en"

    # 1 = greedy. Higher is slower and only slightly better on clean speech.
    beam_size: int = 1

    # Refuse to reach the network for weights. Set True once the model is
    # cached to guarantee a fully offline run.
    local_files_only: bool = False

    # Below this model probability a word is written as unintelligible --
    # `((word))` -- rather than asserted as heard (section 14). Draft output is
    # allowed to say "unclear"; it is not allowed to invent certainty.
    min_word_confidence: float = 0.45

    # Detector confidence needed before a candidate event contributes its tag
    # to the draft. Everything below it stays in the confirmation queue.
    min_event_confidence: float = 0.50

    # Ceiling on clip length sent to the recogniser.
    max_audio_seconds: float = 20 * 60

    # Free memory required before a model is loaded, in GB. 0 means "work it
    # out from the model size". A load that does not fit does not fail cleanly:
    # on Windows it dies with "the paging file is too small", or segfaults
    # part-way through quantisation and takes the process with it.
    min_free_memory_gb: float = 0.0

    # When the configured model does not fit, fall back to a smaller one and
    # say so, rather than refusing to draft at all.
    downgrade_when_memory_is_tight: bool = True


@dataclass
class SpeakerProfilingConfig:
    """Automatic speaker-descriptor suggestions (CLAUDE.md sections 18A.3, 18A.4).

    Gender is suggested from pitch. Nativity is suggested from fluency features
    and a synthetic-speech check -- a documented deviation from section 18A.4,
    which forbids guessing nativity at all (deviation log, section 60.4). Both
    are assistive: below the thresholds here the value stays `unknown`, and a
    profile the annotator edited is never overwritten (section 18A.11).
    """

    suggest_gender: bool = True
    suggest_nativity: bool = True

    # A suggestion below this confidence is reported but not written into the
    # registry, so a thin guess never appears as a filled-in field.
    min_gender_confidence: float = 0.55
    min_nativity_confidence: float = 0.50

    # Fluency needs material to measure. Under either threshold nativity stays
    # `unknown` regardless of what the features say.
    min_words_for_nativity: int = 8
    min_speaking_seconds: float = 2.5

    # How decisively the fluency score must lean before native/nonnative is
    # claimed at all.
    nativity_decision_margin: float = 0.45


@dataclass
class DetectionConfig:
    """Thresholds for the assistive analysis engine (Phase 3 / Phase 7).

    Every value here feeds *analysis events only*. No threshold in this class
    may ever cause a tag to be written into the final transcription; that
    requires explicit human confirmation (CLAUDE.md sections 46 and 47).
    """

    silence_dbfs_threshold: float = -40.0
    min_silence_duration: float = 0.30

    # A silence at or above this length is reported as a PAUSE analysis event.
    # Pause is internal metadata and is NEVER inserted as a transcription tag
    # (CLAUDE.md section 13).
    pause_min_duration: float = 0.50

    min_speech_duration: float = 0.20

    # Below this confidence a suggestion is surfaced as "unknown" rather than
    # as a concrete proposal.
    low_confidence_threshold: float = 0.50

    # Every "possible_*" event requires human confirmation. Not configurable
    # to False on purpose -- it is a workflow rule, not a preference.
    require_human_confirmation: bool = True


@dataclass
class ValidationConfig:
    """Validator behaviour."""

    # Validation is advisory. The engine reports findings and never edits the
    # annotator's text (CLAUDE.md sections 49 and 56.21).
    autofix_enabled: bool = False

    # Findings that depend on audio the validator cannot hear (for example
    # "this hyphen may not be a real truncation") are emitted as warnings.
    report_audio_dependent_warnings: bool = True

    # Emit INFO findings for detected pauses, reminding the annotator that
    # pause is metadata only.
    report_pause_info: bool = True

    max_speakers: int = 20


@dataclass
class PrivacyConfig:
    """Hard privacy switches. All default to the locked-down value."""

    allow_network: bool = False
    telemetry_enabled: bool = False
    allow_external_asr: bool = False  # must stay False; see AppConfig.validate()
    allow_cloud_storage: bool = False


@dataclass
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    projects_dir: Path = PROJECT_ROOT / "data" / "projects"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    log_dir: Path = PROJECT_ROOT / "data" / "logs"
    exports_dir: Path = PROJECT_ROOT / "data" / "exports"
    #: Where microphone captures are written. Local only, and gitignored with
    #: the rest of data/ because it holds the audio being worked on.
    recordings_dir: Path = PROJECT_ROOT / "data" / "recordings"
    database_path: Path = PROJECT_ROOT / "data" / "ermis.sqlite3"

    def ensure(self) -> None:
        for f in fields(self):
            value = Path(getattr(self, f.name))
            if f.name == "database_path":
                value.parent.mkdir(parents=True, exist_ok=True)
            else:
                value.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    speakers: SpeakerProfilingConfig = field(default_factory=SpeakerProfilingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> AppConfig:
        """Load config from JSON, falling back to defaults.

        Unknown keys are ignored rather than fatal, so a config written by a
        newer build still opens. Credential-shaped keys are rejected outright.
        """
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.is_file():
            return cfg

        raw = json.loads(p.read_text(encoding="utf-8"))
        _reject_credentials(raw)

        for section in (
            "audio",
            "recording",
            "asr",
            "speakers",
            "detection",
            "validation",
            "privacy",
        ):
            if section not in raw or not isinstance(raw[section], dict):
                continue
            target = getattr(cfg, section)
            known = {f.name for f in fields(target)}
            for key, value in raw[section].items():
                if key not in known:
                    continue
                # Preserve tuple-typed defaults.
                if isinstance(getattr(target, key), tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(target, key, value)

        if isinstance(raw.get("paths"), dict):
            known = {f.name for f in fields(cfg.paths)}
            for key, value in raw["paths"].items():
                if key in known:
                    setattr(cfg.paths, key, Path(value))

        if isinstance(raw.get("log_level"), str):
            cfg.log_level = raw["log_level"].upper()

        cfg.validate()
        return cfg

    def save(self, path: str | os.PathLike[str]) -> None:
        payload = {
            "audio": asdict(self.audio),
            "recording": asdict(self.recording),
            "asr": asdict(self.asr),
            "speakers": asdict(self.speakers),
            "detection": asdict(self.detection),
            "validation": asdict(self.validation),
            "privacy": asdict(self.privacy),
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
            "log_level": self.log_level,
        }
        _reject_credentials(payload)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def validate(self) -> None:
        """Enforce invariants that are workflow rules rather than preferences."""
        if self.privacy.allow_external_asr:
            raise ValueError(
                "allow_external_asr must be False: ASR-generated final "
                "transcription is prohibited by the Ermis V2 workflow "
                "(CLAUDE.md sections 1 and 56.1)."
            )
        if not self.detection.require_human_confirmation:
            raise ValueError(
                "require_human_confirmation must be True: every uncertain "
                "audio event needs human confirmation (CLAUDE.md section 47)."
            )
        if self.audio.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.audio.target_channels != 1:
            raise ValueError("analysis audio is mono; target_channels must be 1")
        if self.recording.sample_rate <= 0:
            raise ValueError("recording sample_rate must be positive")
        if self.recording.channels not in (1, 2):
            raise ValueError("recording channels must be 1 or 2")
        if self.recording.sample_width_bits not in (16, 24, 32):
            raise ValueError(
                "recording sample_width_bits must be 16, 24 or 32; capture is "
                "written as PCM WAV"
            )
        if self.recording.max_duration_seconds <= 0:
            raise ValueError("recording max_duration_seconds must be positive")
        if not 0.0 <= self.asr.min_word_confidence <= 1.0:
            raise ValueError("asr min_word_confidence must be within [0, 1]")
        if not 0.0 <= self.asr.min_event_confidence <= 1.0:
            raise ValueError("asr min_event_confidence must be within [0, 1]")
        if self.asr.max_audio_seconds <= 0:
            raise ValueError("asr max_audio_seconds must be positive")
        if self.asr.beam_size < 1:
            raise ValueError("asr beam_size must be at least 1")
        if self.asr.min_free_memory_gb < 0:
            raise ValueError("asr min_free_memory_gb cannot be negative")
        for name in ("min_gender_confidence", "min_nativity_confidence"):
            value = getattr(self.speakers, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"speakers {name} must be within [0, 1]")
        if self.speakers.min_words_for_nativity < 1:
            raise ValueError("speakers min_words_for_nativity must be at least 1")
        if self.speakers.min_speaking_seconds <= 0:
            raise ValueError("speakers min_speaking_seconds must be positive")


def _reject_credentials(payload: dict) -> None:
    """Refuse to read or write credential-shaped keys anywhere in the config."""

    def walk(node: object, trail: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(bad in lowered for bad in _FORBIDDEN_CONFIG_KEYS):
                    raise ValueError(
                        f"refusing to handle credential-like config key: {trail}{key}. "
                        "This application stores no secrets."
                    )
                walk(value, f"{trail}{key}.")
        elif isinstance(node, list):
            for item in node:
                walk(item, trail)

    walk(payload)


def setup_logging(cfg: AppConfig) -> logging.Logger:
    """Configure rotating local file logging plus console output."""
    cfg.paths.ensure()
    logger = logging.getLogger("ermis")
    logger.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        cfg.paths.log_dir / "ermis.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.propagate = False
    return logger


CONFIG = AppConfig()
