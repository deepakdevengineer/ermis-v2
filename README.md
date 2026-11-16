# Ermis V2 Local Audio Annotation & QA Assistant

A local desktop tool that helps a human annotator produce a correct Ermis
Transcription V2 annotation and catches rule violations before submission.

**The tool drafts, checks and audits; a human decides.** It records audio,
analyses it, drafts a transcription with a local recogniser, and validates that
text against every V2 rule — but the draft is machine output and is labelled as
such until a person corrects it.

> **Deviation notice.** CLAUDE.md §1 and §56.1 prohibit ASR-generated
> transcription, and the Ermis V2 workflow the spec describes forbids it. The
> `draft` feature exists because the project owner explicitly asked for the
> words to be produced from the audio. That compliance decision is the owner's.
> What the code still guarantees: recognition runs **locally**, draft text is
> stored, exported and audited as `asr_draft` — never as human work — and the
> audio stays the source of truth (§59). See [Drafting from audio](#drafting-from-audio-asr).

Everything runs locally. No network calls, no telemetry, no cloud storage, and
no credentials are stored anywhere.

**1552 tests, all passing.**

---

## Status: all 9 phases of §57 complete

| Phase | Scope | State |
|---|---|---|
| 1 | Project setup, config, logging, SQLite | Done |
| 2 | Audio engine: load, playback, waveform, speed, segment replay | Done |
| 3 | Analysis: VAD, silence/pause, quality, event timeline | Done |
| 4 | Ermis rule engine (all validators) | Done |
| 5 | Transcription editor GUI | Done |
| 6 | Speaker metadata, merge/split | Done |
| 7 | Audio-event assistance, diarization | Done |
| 8 | Audit, storage, export | Done |
| 9 | Testing and packaging | Done |

Read **[Detector reliability](#detector-reliability--read-this-before-trusting-output)**
before relying on the Phase 3/7 detectors. They are hand-built heuristics, and
some are weak.

---

## Install

```bash
pip install -r requirements.txt
```

Core (validator, storage, loader, waveform, **and the full analysis engine**)
needs only `numpy` and `scipy`. `PySide6` is needed for the GUI; `sounddevice`
for audio output **and for microphone capture**; `ffmpeg` on `PATH` for any
format other than PCM WAV.

Drafting needs one local speech backend — neither is required for anything else:

```bash
pip install faster-whisper        # recommended: several times faster on CPU
pip install torch transformers    # alternative, useful if torch is already there
```

The model downloads once (`openai/whisper-small.en` by default, ~1 GB for the
torch path) and then runs offline. Set `asr.local_files_only = true` in the
config to forbid any network use, or `asr.enabled = false` to switch drafting
off entirely.

Everything except the GUI runs headless.

---

## Use

Settings live in an optional, git-ignored `config.local.json` in the project
root; every command reads it (or `--config path.json`):

```json
{
  "asr": { "model": "openai/whisper-base.en" },
  "speakers": { "suggest_nativity": true }
}
```

```bash
# Launch the annotation window
python -m app.main gui clip.wav

# Validate transcription text a human wrote
python -m app.main validate "play [hn] music"
python -m app.main validate --json "Play 23 songs?"
python -m app.main validate --speaker-count 1 "<s1> play music"

# Run the assistive analysis engine (Layer A only — no transcription)
python -m app.main analyse clip.wav
python -m app.main analyse --json clip.wav

# Draft the words from the audio with the local recogniser (machine output)
python -m app.main draft clip.wav
python -m app.main draft --json clip.wav
python -m app.main draft --model openai/whisper-base.en --offline clip.wav

# Record a clip from the microphone (audio only, never transcription)
python -m app.main record --list-devices
python -m app.main record --seconds 30
python -m app.main record --output take1.wav --device 1 --analyse

# Inspect audio
python -m app.main info clip.mp3
python -m app.main waveform clip.wav --width 100 --start 1.0 --end 3.5

python -m app.main initdb
```

`validate` exits `0` when there are no errors and `1` otherwise, so it drops
into a pre-submission check.

```
$ python -m app.main analyse two_speakers.wav

ANALYSIS TIMELINE (internal QA metadata, not transcription)
00:00.28 - 00:01.31  SPEECH  s1
00:01.31 - 00:01.78  PAUSE
00:01.78 - 00:02.71  SPEECH  s2
00:02.20 - 00:02.23  POSSIBLE_ARTIFACT  conf=0.54
...

Total Speakers: 2 (confidence 69%, editable)

S1     ############                   ##########
S2                     ###########                  ############

1 candidate(s) require human confirmation before any tag may be used

Pause is internal metadata and is never inserted as a transcription tag.
```

---

## Recording

The window has a capture bar next to the playback transport, and the CLI has a
`record` command. Both do the same thing:

```
microphone -> WAV under data/recordings/ -> AudioLoader -> analysis -> editor
```

Capture is written to disk first and then loaded back through the ordinary
loader, so a recorded clip and an attached file follow exactly the same path.
The file is the source of truth (CLAUDE.md §59): whatever the detectors saw,
the annotator can replay later and hear.

- Device picker, level meter (dBFS) and elapsed clock, so a muted input or a
  clipping level is visible while recording rather than afterwards.
- Pause keeps the file open; Discard deletes the capture; closing the window
  keeps an in-progress recording, because audio is the one thing that cannot be
  recreated.
- Capture stops by itself at `RecordingConfig.max_duration_seconds`
  (30 minutes by default) and keeps what it captured.
- Mono 16-bit PCM WAV at 48 kHz by default. If the device refuses that rate,
  capture falls back to the device default and writes the file at that rate;
  the analysis copy is resampled either way.
- The capture is measured, not judged: peak level, clipped-sample count and
  input overflows are reported as notes, and whether they mean `[artifact]`,
  `<ga>`, `<na>` or nothing stays a human decision (§9, §10, §11, §47).
- Provenance is audited. A recorded clip gets a `recording_captured` row with
  its rate, duration and measurements, so a later reviewer can see the audio
  was captured here rather than supplied as an attachment.

Recording adds an **input** path only. It produces audio and stops there — the
transcription box stays empty until a person types in it (§1, §3, §56.1).

---

## Drafting from audio (ASR)

```
audio -> local recogniser -> spoken form -> speaker turns -> event tags -> validate
```

Press **Ctrl+D** (or Analysis > Draft transcription from audio) and the editor
fills with a draft. With `asr.auto_draft_on_load` on — the default — every clip
you open or record is drafted automatically, on a worker thread, so the window
stays usable.

What the draft contains:

- **The words**, converted to V2 spoken form by `app/ermis/spoken_form.py`:
  lowercase, no sentence punctuation, numbers and symbols as words, `okay` ->
  `ok`, `Mr.` -> `mister`, canonical backchannel spellings (`mhm` -> `mm-hm`),
  accents folded, stretched letters collapsed. It reuses the validator's own
  tables, so a draft passes the checks this tool applies.
- **`((word))`** where the recogniser itself was unsure (below
  `asr.min_word_confidence`), and `(())` when speech is audible but nothing
  could be recognised. A draft is allowed to say "unclear"; it is not allowed to
  invent certainty.
- **Speaker tags** `<s1>`/`<s2>`… at diarized turns, numbered by first
  appearance **in the draft** and omitted entirely for single-speaker audio
  (§17, §18A.2). The diarizer can report a speaker whose segments carry no
  recognised words; renumbering keeps the tags contiguous instead of opening on
  `<s2>`. When the speaker count is uncertain the draft says so and points at
  the Speakers panel's merge/split — an over-split voice is the most common
  diarizer error on short clips.
- **`[laughter]` / `[hn]` / `[artifact]` / `[bg]`** where a detector above
  `asr.min_event_confidence` found one — alphabetically ordered, repeats
  collapsed (§8, §41).
- **`<ol> … </ol>`** only when the words inside really come from two speakers;
  otherwise the overlap is left in the confirmation queue with a note, because a
  one-speaker overlap block is malformed under §18.
- **A whole-file condition alone** — `<na>`, `<ga>`, `<ns>`, `(())`,
  `[laughter]`, `[hn]` — when that is what the clip is (§42). A blank or garbled
  file is never sent to the recogniser at all.
- **Never a pause tag** (§13), and never a retired V1 tag (§39).

What it never does:

- Overwrite text you typed. If the editor has your words, the draft waits and
  asks (§56.21).
- Claim to be human work. The draft is saved as source `asr_draft`, the window
  title says `[UNREVIEWED MACHINE DRAFT]`, the export carries
  `"source": "asr_draft"`, `"asr_used": true`, `"human_reviewed": false`, and
  exporting it is refused until you say so explicitly.
- Hide that a tag came from a machine. Tags a draft inserts are audited as
  `auto_confirmed_event`, distinct from a human's `confirmed_event`, with the
  note "verify against the audio".

### Speaker descriptors

```bash
python -m app.main speakers clip.wav          # the §18A.9 panel, with evidence
python -m app.main speakers --json clip.wav
python -m app.main speakers --no-asr clip.wav # skip recognition (no fluency data)
```

- **Count** comes from diarization, with its confidence, and stays editable
  (merge/split in the Speakers panel, §18A.8).
- **`gender`** (`msv`/`fsv`/`unknown`) from median pitch per speaker. A pitch in
  the 155–190 Hz overlap band, or too little voiced audio, stays `unknown`; a
  suggestion below `speakers.min_gender_confidence` is reported but not written
  (§18A.3).
- **`nativity`** (`native`/`nonnative`/`unknown`/`tts`) — **a second documented
  deviation** (§18A.4 forbids guessing it; see CLAUDE.md §60.4). `tts` comes
  from a real signature: flat pitch, near-zero jitter, digital silence between
  phrases. `native`/`nonnative` comes from fluency — speaking rate, long-pause
  ratio, filler density, recogniser confidence. Recogniser confidence is partly
  accent, which is exactly the evidence §18A.4 excludes, so it carries the least
  weight, confidence is capped at 0.75, and anything thin or borderline returns
  `unknown`. Treat it as a first guess to correct, not an answer.
  `speakers.suggest_nativity = false` turns it off.

Nothing here overwrites a value you edited (§18A.11), and every suggestion
carries its evidence — pitch, word count, rate, pause ratio — so you can see why
it said what it said.

Speed, measured on this project (4-core CPU, `whisper-small.en`):

| Path | Warm speed |
|---|---|
| torch fp32 | ~4× realtime |
| torch int8 (`asr.quantize_dynamic`, default) | ~1.6× realtime |
| faster-whisper int8 | faster still; install it if drafting feels slow |

**Memory.** A load that does not fit does not fail cleanly: on Windows it dies
with "the paging file is too small", or segfaults part-way through quantisation
and takes the process with it. So it is checked *before* loading:

| Model | Peak to load |
|---|---|
| `tiny(.en)` | ~0.6 GB |
| `base(.en)` | ~1.0 GB |
| `small(.en)` (default) | ~2.6 GB |
| `medium(.en)` | ~6.0 GB |

When the configured model does not fit, drafting falls back to the largest one
that does and says so in the status bar — a labelled draft from a smaller model
beats a crash. Set `asr.downgrade_when_memory_is_tight = false` to make it refuse
instead, or `asr.min_free_memory_gb` to demand more headroom. Two copies of the
model (the window plus a CLI run) will not fit on an 8 GB machine, so close one
before running `draft` or `speakers` from the terminal.

The model loads once per session (~40 s for the torch path, including
quantisation and warm-up). The window starts that load in the background as it
opens (`asr.preload_on_start`), so recording and analysis stay usable and a clip
recorded a minute later is drafted immediately instead of waiting.
`whisper-base.en` is roughly three times faster than `small.en` and noticeably
less accurate — set it with `--model` or `asr.model`.

---

## Architecture

```
app/
├── main.py                 CLI: validate / info / waveform / record / draft / speakers / analyse / initdb / gui
├── config.py               configuration, privacy switches, logging
├── audio/
│   ├── loader.py           decode to mono float32 (stdlib wave / FFmpeg)
│   ├── playback.py         transport, speed control, segment replay
│   ├── recorder.py         microphone capture to a local WAV file
│   ├── preprocessing.py    shared frame grid + spectral features
│   ├── waveform.py         peak + RMS envelope, ASCII renderer
│   └── quality.py          <na> / <ga> / <ns> measurement
├── asr/                    local drafting (the §1 / §56.1 deviation)
│   ├── engine.py           faster-whisper / transformers backends, word timings
│   └── draft.py            merges words + events + speakers into V2 text
├── detection/
│   ├── vad.py              adaptive-threshold voice activity
│   ├── silence.py          silence, and which gaps are pauses
│   ├── laughter.py         [laughter] candidates
│   ├── nonlexical.py       [hn] candidates
│   ├── artifact.py         [artifact] candidates
│   ├── background.py       [bg] candidates
│   ├── overlap.py          <ol> candidates
│   └── pipeline.py         runs all of the above in dependency order
├── speakers/
│   ├── diarization.py      MFCC + deterministic agglomerative clustering
│   ├── voice_profile.py    pitch-based gender suggestion
│   ├── nativity.py         fluency + TTS-signature nativity suggestion (§60.4)
│   └── metadata.py         registry, merge/split, renumbering
├── ermis/                  the rule engine
│   ├── tags.py             V2 tag inventory — single source of truth
│   ├── rules.py            lexical tables
│   ├── tokenizer.py        offset-preserving tokenizer
│   ├── spoken_form.py      recogniser prose -> V2 spoken form
│   ├── whole_file.py       whole-file condition logic
│   ├── findings.py         Finding / Severity / ValidationReport
│   └── validator.py        the deterministic validator
├── models/event.py         analysis events + human-confirmation gate
├── storage/
│   ├── schema.sql          SQLite schema
│   ├── database.py         data access
│   ├── audit.py            append-only audit trail
│   └── export.py           JSON / TXT / CSV export
└── ui/
    ├── main_window.py      shell wiring everything together
    ├── record_bar.py       capture transport: device, level meter, elapsed
    ├── draft_worker.py     drafting off the GUI thread, model cached once
    ├── waveform_view.py    waveform, events, speaker lanes
    ├── transcription_editor.py  editor + tag palette
    ├── validation_panel.py findings, grouped by severity
    ├── speaker_panel.py    §18A.9 speaker panel
    └── event_panel.py      §47 confirmation workflow
```

### The two layers, and where drafting sits

**Layer A (analysis)** produces internal events. **Layer B (transcription)** is
text a human typed. The only bridge is explicit human confirmation:

```python
event = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 7.42, 8.15, confidence=0.91)
event.tag          # raises UnconfirmedEventError — a guess cannot become a tag
event.confirm()    # a person listened and agreed
event.tag          # '[laughter]'
```

A 99%-confident detection still cannot write a tag (§56.22) through this API.
The event panel is one route from a detection to the transcription, and it needs
a click.

Drafting is the second route, and it is the deviation: when a draft inserts a
tag it calls `confirm()` on your behalf. That is visible rather than hidden —
the event is noted as "auto-confirmed by ASR draft; verify against the audio",
the audit row is `auto_confirmed_event` rather than `confirmed_event`, and the
transcription is labelled `asr_draft` until you edit it. Detections below
`asr.min_event_confidence` are never auto-confirmed; they stay in the queue for
a human.

---

## Detector reliability — read this before trusting output

The Phase 3/7 detectors are **deterministic signal-processing heuristics, not
trained models**. No model weights are downloaded, which keeps the privacy
guarantee intact but costs real accuracy. Ranked by how much you can rely on
them:

| Detector | Reliability | Notes |
|---|---|---|
| Silence / pause | High | A direct measurement. Boundaries land within ~20 ms. |
| VAD | Good | Adaptive noise floor; rejects broadband noise. Harmonic non-speech (music, singing, engine hum) can still read as speech. |
| Audio quality | Good | Clipping and dropouts are objective. `<ga>` vs `[artifact]` needs a listener (§9). |
| `[artifact]` | Moderate | Distinguishes a spike from a speech onset. Cannot tell a mic pop from a door slam. |
| Diarization | Moderate | Correct on clean 1- and 2-speaker audio. Threshold calibrated on **synthetic** fixtures; real within-speaker variance is much larger. |
| `[hn]` | Moderate | Finds non-speech vocal-ish sounds; cannot tell a cough from a sneeze from a desk thump. |
| `[bg]` | Low | Attribution is inherently a human judgement. |
| `[laughter]` | **Low** | Speech syllables occupy the same 3–8 Hz range as laughter. Tuned for precision over recall; still produces occasional false positives on very regular speech. |
| `<ol>` overlap | **Lowest** | Confidence capped at 0.55. Reverberation and music mimic a second voice. |

Confidence values are **display figures, not probabilities**, and are capped
per detector. This is exactly why §47's confirmation workflow exists and why
every uncertain event is `possible_*`.

**Validation caveat:** detectors were tuned against synthetic fixtures
generated by this repo (glottal pulses through formant resonators), not a real
speech corpus. Real-world hit rates are unmeasured. Thresholds are named
constants at the top of each module, documented for retuning.

---

## What the validator checks

All 26 mandatory rules from §54, reported as `ERROR`, `WARNING` or `INFO`, each
citing its CLAUDE.md section. Findings are advisory — the validator returns them
and never edits the annotator's text.

Full reference: **[docs/validation-rules.md](docs/validation-rules.md)**.

### What it deliberately does not check

Rules only the audio can settle are `WARNING` or absent: whether a hyphen marks
a real truncation, `[artifact]` vs `<ga>`, `(())` vs `<ga>`, whether a
repetition was spoken, and speaker gender/nativity.

Guessing would contradict §59: **audio is the source of truth.** A false
positive is worse than a missing check, because it trains the annotator to
ignore the panel. The corpus of documented-correct examples from CLAUDE.md is
asserted to produce zero errors *and* zero warnings.

### Two places CLAUDE.md conflicts with itself

1. **§20 documents `[laughter] [bg]` as correct**, while §41 requires
   alphabetical order (`[bg] [laughter]`). Reading them with §21 ("place `[bg]`
   … after the interrupted content"), `[bg]` is positional, so it is excluded
   from the ordering check and kept in the duplicate check.
2. **`(())` and `<ct>` are dual-purpose** — valid alone *and* inline (§14's
   `(()) but i don't know what to get`, §19's `<ol>…</ol> <ct>`). Only the five
   angle tags `<ga> <na> <ns> <nt> <sc>` must stand alone.

---

## Privacy and security

- Local processing only; the loader, analysis pipeline and exporter are each
  asserted in tests to open no socket.
- **Audio never leaves the machine**, drafting included: recognition runs on a
  local model. The one network touch in the whole application is downloading
  model weights the first time, and `asr.local_files_only = true` forbids even
  that.
- No telemetry, no external transcription API, no cloud storage. All default to
  off in `PrivacyConfig`, and `allow_external_asr = True` still raises: the
  local recogniser is the only ASR path, and there is no cloud one.
- **No credentials are stored.** No password/token/secret/api-key fields exist,
  `AppConfig.load` refuses a credential-shaped key at any nesting depth, and
  tests assert no database column and no export file looks like one.
- The audit log is append-only, enforced by SQLite triggers.
- Transcription saves are append-only revisions, so no automated process can
  destroy a human's text.
- Microphone capture writes only to `data/recordings/` on this machine. It
  opens no socket, and it produces audio only -- never transcription text.
- `data/` holds recordings, the database, logs and exports. It is git-ignored.

---

## Tests

```bash
python -m pytest                      # 1552 tests
python -m pytest tests/test_rules.py -v
```

| File | Tests | Covers |
|---|---|---|
| `test_rules.py` | 107 | spoken form, repetitions, contractions, letter sequences, entities |
| `test_tags.py` | 72 | tag inventory, `[hn]`, `[laughter]`, `[artifact]`, `[bg]`, retired tags, ordering |
| `test_numbers.py` | 45 | numbers as words, acronyms, alphanumeric sequences |
| `test_backchannels.py` | 45 | every canonical spelling and documented misspelling |
| `test_speakers.py` | 53 | speaker tags, numbering, metadata vocabularies, human-edit protection |
| `test_overlap.py` | 28 | `<ol>` syntax, `<ct>`, nesting |
| `test_whole_file.py` | 61 | `<ga>` `<na>` `<ns>` `<nt>` `<sc>` and standalone conditions |
| `test_truncations.py` | 30 | truncation hyphens vs compounds and repetitions |
| `test_tokenizer.py` | 59 | offset round-tripping, malformed-shape classification |
| `test_edge_cases.py` | 50 | §55 cases A–P and each §56 prohibition |
| `test_coverage.py` | 215 | every finding code reachable; §54 checklist; false-positive corpus |
| `test_detection.py` | 84 | framing, VAD, silence/pause, quality, all candidate detectors, pipeline |
| `test_diarization.py` | 66 | clustering, registry, merge/split/renumbering |
| `test_playback.py` | 34 | transport, speed, segment replay, no-device fallback |
| `test_recorder.py` | 46 | capture lifecycle, WAV round-trip, levels, limits, no-backend fallback |
| `test_asr.py` | 41 | windowing, word timing, draft merge, whole-file drafts, provenance |
| `test_spoken_form.py` | 93 | prose -> spoken form; every output validates |
| `test_voice_profile.py` | 24 | pitch tracking, unknown-when-unsure, human-edit protection |
| `test_nativity.py` | 20 | fluency features, TTS signature, capped confidence, thresholds |
| `test_audio.py` | 43 | decoding, resampling, downmixing, levels, envelope |
| `test_storage.py` | 48 | schema constraints, revisions, audit immutability |
| `test_export.py` | 30 | export formats, confirmation boundary, guards |
| `test_ui.py` | 110 | every widget including the record bar and drafting, headless via Qt offscreen |
| `test_config.py` | 30 | privacy defaults, workflow invariants, credential refusal |
| `test_cli.py` | 25 | every command, and that no transcribe command exists |

Audio fixtures are synthesised at test time, so there are no binary assets.
UI tests run under `QT_QPA_PLATFORM=offscreen` and skip if PySide6 is absent.

---

## Packaging

```bash
pip install pyinstaller
pyinstaller packaging/ermis-assistant.spec
```

See **[packaging/README.md](packaging/README.md)**. FFmpeg is not bundled
(licensing); no ASR runtime is bundled, and the spec blocks them as transitive
dependencies.
