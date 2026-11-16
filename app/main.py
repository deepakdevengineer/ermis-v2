"""Entry point.

Phase 1 scope. The GUI shell arrives in Phase 5; today this exposes the parts
that are finished, and it works without PySide6 installed:

    python -m app.main validate "play [hn] music"
    python -m app.main validate --file transcription.txt
    python -m app.main record --seconds 15
    python -m app.main record --list-devices
    python -m app.main info data/samples/tone.wav
    python -m app.main waveform data/samples/tone.wav
    python -m app.main analyse data/samples/tone.wav
    python -m app.main draft data/samples/clip.wav
    python -m app.main speakers data/samples/clip.wav
    python -m app.main initdb
    python -m app.main gui            # needs PySide6

`validate` checks transcription text; `info` and `waveform` describe the audio;
`record` captures from a local input device; `analyse` runs the assistive QA
engine; `draft` runs the local recogniser and produces a DRAFT transcription.

DEVIATION NOTICE: CLAUDE.md sections 1 and 56.1 prohibit ASR-generated
transcription. `draft` exists because the project owner asked for the words to
be produced from the audio. Draft text is labelled `asr_draft` everywhere it is
stored or exported, never as human work, and it runs locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import (
    APP_NAME,
    APP_VERSION,
    ERMIS_VERSION,
    PROJECT_ROOT,
    AppConfig,
    setup_logging,
)

#: Optional local settings file, picked up by every command when present.
#: Git-ignored, so machine-specific choices (a smaller ASR model, a different
#: recordings folder) live here instead of in the code.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.local.json"


def load_config(args: argparse.Namespace) -> AppConfig:
    """Config from --config, else config.local.json, else the defaults."""
    path = getattr(args, "config", None) or (
        DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None
    )
    config = AppConfig.load(path)
    if path is not None:
        print(f"settings: {path}", file=sys.stderr)
    return config


def cmd_validate(args: argparse.Namespace) -> int:
    from app.ermis.validator import validate

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip("\n")
    elif args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read().strip("\n")

    report = validate(
        text,
        speaker_count=args.speaker_count,
    )

    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"text: {text!r}")
        print(report)
        print(
            f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s), "
            f"{len(report.infos)} info"
        )
    return 0 if report.is_valid else 1


def cmd_info(args: argparse.Namespace) -> int:
    import json

    from app.audio.loader import AudioLoadError, load_audio

    try:
        buffer = load_audio(args.path)
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(buffer.to_dict(), indent=2))
    if buffer.is_silent:
        print(
            "\nnote: no audio signal detected. The whole-file condition <na> may "
            "apply -- confirm by listening (CLAUDE.md section 11)."
        )
    return 0


def cmd_waveform(args: argparse.Namespace) -> int:
    from app.audio.loader import AudioLoadError, load_audio
    from app.audio.waveform import compute_waveform, render_ascii

    try:
        buffer = load_audio(args.path)
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    waveform = compute_waveform(
        buffer, buckets=args.width, start=args.start, end=args.end
    )
    print(f"{buffer.path.name}  {buffer.duration:.2f}s  {buffer.sample_rate} Hz mono")
    print(render_ascii(waveform, width=args.width, height=args.height))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Capture audio from a local input device to a WAV file.

    Capture only. The recorded file is analysed by `analyse` and transcribed by
    a human in the editor; nothing here produces transcription text
    (CLAUDE.md sections 1 and 56.1).
    """
    import json

    from app.audio.recorder import Recorder, RecordingError, bluetooth_mic_note

    cfg = load_config(args)
    cfg.paths.ensure()
    recorder = Recorder(cfg.audio, cfg.recording, output_dir=cfg.paths.recordings_dir)

    if args.list_devices:
        # Always re-enumerate: PortAudio caches its device table, so a headset
        # switched on after this process started is otherwise invisible.
        devices = recorder.list_devices(rescan=True)
        if not devices:
            print(
                "no input devices found. Microphone capture needs the "
                "sounddevice package and a working PortAudio library.",
                file=sys.stderr,
            )
            return 2
        default = recorder.default_device()
        for device in devices:
            marks = []
            if default and device.index == default.index:
                marks.append("default")
            if device.is_bluetooth:
                marks.append("bluetooth")
            if device.unnamed:
                marks.append("unnamed endpoint -- device not connected")
            suffix = f" ({'; '.join(marks)})" if marks else ""
            print(f"{device}{suffix}")
        note = bluetooth_mic_note(devices)
        if note:
            print()
            print("No Bluetooth microphone is listed above.")
            print(note)
        return 0

    try:
        path = recorder.start(args.output, device=args.device)
    except RecordingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    limit = args.seconds if args.seconds and args.seconds > 0 else None
    print(
        f"recording to {path} at {recorder.sample_rate} Hz, "
        f"{recorder.channels} channel(s)"
    )
    print(
        "press Ctrl+C to stop"
        if limit is None
        else f"stopping automatically after {limit:.1f}s (Ctrl+C stops sooner)"
    )

    import time

    deadline = time.monotonic() + limit if limit is not None else None
    try:
        while not recorder.limit_reached:
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            if not args.quiet:
                peak = recorder.read_level()
                bar = "#" * int(round(min(1.0, peak) * 30))
                print(
                    f"\r{recorder.elapsed:6.1f}s |{bar:<30}|",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        if not args.quiet:
            print()

    try:
        result = recorder.stop()
    except RecordingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if result is None:
        print("nothing was captured, so no file was written.", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    for note in result.notes():
        print(f"note: {note}")
    print(
        "\nCapture is audio only. Run `analyse` on it for the QA timeline, and "
        "type the transcription yourself (sections 1 and 3)."
    )

    if args.analyse:
        print()
        args.path = str(result.path)
        args.json = False
        return cmd_analyse(args)
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    """Draft a transcription from the audio with the local recogniser.

    The output is a DRAFT. It is labelled `asr_draft` wherever it is stored,
    and it is meant to be corrected against the audio, which remains the source
    of truth (CLAUDE.md section 59). Drafting words from audio is a deliberate
    deviation from sections 1 and 56.1, made at the project owner's request.
    """
    import json

    from app.asr import build_draft
    from app.audio.loader import AudioLoadError, load_audio
    from app.detection.pipeline import AnalysisPipeline
    from app.speakers.metadata import SpeakerRegistry
    from app.speakers.nativity import apply_nativity, suggest_nativity
    from app.speakers.voice_profile import apply_suggestions, suggest_voice_profiles

    cfg = load_config(args)
    if args.model:
        cfg.asr.model = args.model
    if args.backend:
        cfg.asr.backend = args.backend
    if args.offline:
        cfg.asr.local_files_only = True

    try:
        buffer = load_audio(args.path)
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    analysis = AnalysisPipeline(cfg.detection).run(buffer)
    draft = build_draft(buffer, analysis, cfg.asr)

    registry = SpeakerRegistry.from_diarization(analysis.diarization)
    suggestions = suggest_voice_profiles(buffer, analysis.diarization)
    nativity = suggest_nativity(buffer, analysis.diarization, draft.words, cfg.speakers)
    applied = apply_suggestions(
        registry, suggestions, min_confidence=cfg.speakers.min_gender_confidence
    ) + apply_nativity(registry, nativity, cfg.speakers)

    if args.json:
        payload = draft.to_dict()
        payload["speakers"] = registry.to_dict()
        payload["voice_suggestions"] = [s.to_dict() for s in suggestions.values()]
        payload["nativity_suggestions"] = [s.to_dict() for s in nativity.values()]
        print(json.dumps(payload, indent=2))
        return 0 if draft.report is None or draft.report.is_valid else 1

    print(f"{buffer.path.name}  {buffer.duration:.2f}s")
    print()
    print("DRAFT TRANSCRIPTION (machine-produced, unreviewed)")
    print(draft.text or "(nothing drafted)")
    print()
    from app.asr.engine import active_memory_note

    note = active_memory_note()
    if note:
        print(f"note: {note}")
        print()
    if draft.asr is not None:
        print(
            f"recogniser: {draft.asr.backend} / {draft.asr.model}  ·  "
            f"{len(draft.asr.words)} word(s)  ·  mean confidence "
            f"{draft.asr.mean_confidence * 100:.0f}%"
        )
        print(f"raw output: {draft.asr.text!r}")
    if draft.changes:
        print()
        print("spoken-form changes applied:")
        for change in draft.changes:
            print(f"  - {change}")
    if draft.inserted_tags:
        print()
        print("tags inserted from the analysis engine:")
        for at, tag in draft.inserted_tags:
            print(f"  {at:7.2f}s  {tag}")

    print()
    print(f"Total Speakers: {registry.count} (editable)")
    for speaker_id in registry.speaker_ids:
        profile = registry.profiles[speaker_id]
        gender = suggestions.get(speaker_id)
        native = nativity.get(speaker_id)
        print(
            f"  {speaker_id}: gender {profile.gender}, nativity {profile.nativity}"
        )
        if gender:
            print(f"      gender   -- {gender.reason}")
        if native:
            print(f"      nativity -- {native.reason}")
    for line in applied:
        print(f"  applied: {line}")

    if draft.report is not None:
        print()
        print("VALIDATION OF THE DRAFT")
        print(draft.report or "(clean)")

    if draft.notes:
        print()
        print("NOTES")
        for note in draft.notes:
            print(f"  - {note}")

    print()
    print(
        "This is a draft, not a submission. Replay the audio, correct the words, "
        "and confirm every tag before use."
    )
    return 0 if draft.report is None or draft.report.is_valid else 1


def cmd_speakers(args: argparse.Namespace) -> int:
    """Report the speaker panel for one clip: count, gender, nativity.

    Gender comes from pitch (section 18A.3). Nativity comes from fluency
    features and a synthetic-speech check -- a deviation from section 18A.4,
    which forbids guessing it; see CLAUDE.md section 60.4. Both are suggestions
    with confidences, and both stay editable (section 18A.11).
    """
    import json

    from app.audio.loader import AudioLoadError, load_audio
    from app.detection.pipeline import AnalysisPipeline
    from app.speakers.metadata import SpeakerRegistry
    from app.speakers.nativity import apply_nativity, suggest_nativity
    from app.speakers.voice_profile import apply_suggestions, suggest_voice_profiles

    cfg = load_config(args)
    if args.model:
        cfg.asr.model = args.model

    try:
        buffer = load_audio(args.path)
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    analysis = AnalysisPipeline(cfg.detection).run(buffer)
    registry = SpeakerRegistry.from_diarization(analysis.diarization)

    words: list = []
    draft_text = None
    if not args.no_asr:
        from app.asr import build_draft

        draft = build_draft(buffer, analysis, cfg.asr)
        words = draft.words
        draft_text = draft.text

    gender = suggest_voice_profiles(buffer, analysis.diarization)
    nativity = suggest_nativity(buffer, analysis.diarization, words, cfg.speakers)
    applied = apply_suggestions(
        registry, gender, min_confidence=cfg.speakers.min_gender_confidence
    ) + apply_nativity(registry, nativity, cfg.speakers)

    if args.json:
        print(
            json.dumps(
                {
                    "audio": buffer.to_dict(),
                    "total_speakers": registry.count,
                    "speaker_count_confidence": round(analysis.diarization.confidence, 2),
                    "speakers": registry.to_dict(),
                    "gender_suggestions": [s.to_dict() for s in gender.values()],
                    "nativity_suggestions": [s.to_dict() for s in nativity.values()],
                    "applied": applied,
                    "draft": draft_text,
                },
                indent=2,
            )
        )
        return 0

    print(f"{buffer.path.name}  {buffer.duration:.2f}s")
    print()
    print("-" * 62)
    print("SPEAKERS")
    print("-" * 62)
    print()
    print(
        f"Total Speakers: {registry.count}   "
        f"(detected at {analysis.diarization.confidence * 100:.0f}% confidence, editable)"
    )
    print()
    for speaker_id in registry.speaker_ids:
        profile = registry.profiles[speaker_id]
        g = gender.get(speaker_id)
        n = nativity.get(speaker_id)
        print(f"{profile.display_name}")
        print(f"  ID       : {speaker_id}")
        print(
            f"  Gender   : {profile.gender}"
            + (
                f"   (suggested {g.gender} at {g.confidence * 100:.0f}%)"
                if g and g.gender != profile.gender
                else ""
            )
        )
        print(
            f"  Nativity : {profile.nativity}"
            + (
                f"   (suggested {n.nativity} at {n.confidence * 100:.0f}%)"
                if n and n.nativity != profile.nativity
                else ""
            )
        )
        if profile.first_seen is not None:
            print(
                f"  Heard    : {profile.first_seen:.2f}s - {profile.last_seen:.2f}s"
            )
        if g:
            print(f"  Pitch    : {g.median_f0:.0f} Hz" if g.median_f0 else "  Pitch    : n/a")
            print(f"  why      : {g.reason}")
        if n:
            print(
                f"  Fluency  : {n.words} word(s), {n.words_per_second:.1f} words/s, "
                f"long pauses {n.long_pause_ratio:.2f}/word, "
                f"recogniser confidence {n.mean_word_confidence * 100:.0f}%"
            )
            print(f"  why      : {n.reason}")
        print()

    if applied:
        print("applied to the profile:")
        for line in applied:
            print(f"  - {line}")
        print()
    if draft_text:
        print(f"draft used for the fluency features: {draft_text!r}")
        print()
    print(
        "Allowed values: gender msv | fsv | unknown, nativity native | nonnative "
        "| unknown | tts. All suggestions are assistive and editable "
        "(section 18A.11)."
    )
    return 0


def cmd_initdb(args: argparse.Namespace) -> int:
    from app.storage.database import Database

    cfg = load_config(args)
    cfg.paths.ensure()
    path = Path(args.path) if args.path else cfg.paths.database_path
    db = Database(path)
    tables = [
        r["name"]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    print(f"database: {path}")
    print(f"schema version: {db.schema_version}")
    print(f"tables: {', '.join(tables)}")
    db.close()
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Run the assistive analysis engine and print the event timeline.

    Prints Layer A analysis only (CLAUDE.md section 3). No transcription text is
    produced or implied.
    """
    import json

    from app.audio.loader import AudioLoadError, load_audio
    from app.detection.pipeline import AnalysisPipeline

    try:
        buffer = load_audio(args.path)
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = AnalysisPipeline().run(buffer)

    if args.json:
        print(json.dumps(result.summary(), indent=2))
        return 0

    print(f"{buffer.path.name}  {buffer.duration:.2f}s")
    print()
    print("ANALYSIS TIMELINE (internal QA metadata, not transcription)")
    print(result.timeline() or "(no events)")
    print()
    print(
        f"Total Speakers: {result.speaker_count} "
        f"(confidence {result.diarization.confidence * 100:.0f}%, editable)"
    )
    if result.diarization.segments:
        print()
        print(result.diarization.render_timeline(width=64, duration=buffer.duration))
    if result.whole_file_candidates:
        print()
        print(f"Whole-file candidates: {' '.join(result.whole_file_candidates)}")
        for note in result.quality.notes:
            print(f"  - {note}")

    pending = result.pending
    if pending:
        print()
        print(
            f"{len(pending)} candidate(s) require human confirmation before any "
            "tag may be used:"
        )
        for event in pending:
            print(f"  {event}")
    print()
    print(
        "Pause is internal metadata and is never inserted as a transcription tag "
        "(section 13)."
    )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "error: the GUI needs PySide6. Install it with `pip install PySide6`.\n"
            "The validate, info, waveform and analyse commands work without it.",
            file=sys.stderr,
        )
        return 2

    from app.ui.main_window import MainWindow

    cfg = load_config(args)
    setup_logging(cfg)

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow(cfg)
    if args.path:
        window.load_audio(args.path)
    window.show()
    return int(app.exec())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ermis-assistant",
        description=(
            f"{APP_NAME} {APP_VERSION} -- local Ermis {ERMIS_VERSION} annotation "
            "and QA assistant. Assistive only: the final transcription is always "
            "human-entered."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "settings file to load; defaults to config.local.json in the project "
            "root when it exists"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate transcription text against V2 rules")
    p.add_argument("text", nargs="*", help="transcription text")
    p.add_argument("--file", help="read the transcription from a file")
    p.add_argument("--json", action="store_true", help="emit findings as JSON")
    p.add_argument(
        "--speaker-count",
        type=int,
        default=None,
        help="human-confirmed number of speakers; 1 makes any speaker tag an error",
    )
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("info", help="show decoded audio properties")
    p.add_argument("path")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("waveform", help="print an ASCII waveform")
    p.add_argument("path")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--height", type=int, default=11)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.set_defaults(func=cmd_waveform)

    p = sub.add_parser("record", help="capture audio from a local input device")
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="stop after this many seconds; omit to record until Ctrl+C",
    )
    p.add_argument(
        "--output",
        default=None,
        help="WAV file to write; default is a timestamped name under data/recordings",
    )
    p.add_argument("--device", type=int, default=None, help="input device index")
    p.add_argument(
        "--list-devices", action="store_true", help="list input devices and exit"
    )
    p.add_argument(
        "--analyse",
        action="store_true",
        help="run the assistive analysis engine on the recording once it is done",
    )
    p.add_argument("--quiet", action="store_true", help="do not draw the level meter")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser(
        "draft", help="draft a transcription from audio with the local recogniser"
    )
    p.add_argument("path")
    p.add_argument("--json", action="store_true", help="emit the full draft as JSON")
    p.add_argument("--model", default=None, help="override the ASR model id")
    p.add_argument(
        "--backend",
        default=None,
        choices=["auto", "faster-whisper", "transformers-whisper"],
        help="override the ASR backend",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="refuse to fetch model weights; use only what is already cached",
    )
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser(
        "speakers", help="report speaker count, gender and nativity for one clip"
    )
    p.add_argument("path")
    p.add_argument("--json", action="store_true", help="emit the panel as JSON")
    p.add_argument(
        "--no-asr",
        action="store_true",
        help="skip recognition; nativity then has no fluency evidence to use",
    )
    p.add_argument("--model", default=None, help="override the ASR model id")
    p.set_defaults(func=cmd_speakers)

    p = sub.add_parser("initdb", help="create the local SQLite database")
    p.add_argument("--path", default=None)
    p.set_defaults(func=cmd_initdb)

    p = sub.add_parser("analyse", help="run the assistive analysis engine")
    p.add_argument("path")
    p.add_argument("--json", action="store_true", help="emit the full result as JSON")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("gui", help="launch the annotation window (needs PySide6)")
    p.add_argument("path", nargs="?", default=None)
    p.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
