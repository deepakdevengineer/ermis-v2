# Packaging

Builds a self-contained Windows folder so the tool runs on a machine without a
Python install.

## Build

From the project root:

```bash
pip install pyinstaller
pyinstaller packaging/ermis-assistant.spec
```

Output: `dist/ermis-assistant/ermis-assistant.exe`

Expect roughly 150–250 MB, almost all of it PySide6's Qt libraries.

## Verify a build

The spec excludes several Qt modules and the speech runtimes, so confirm the
build still works and stays the size it should:

```bash
# launches, loads audio, records, runs analysis
dist/ermis-assistant/ermis-assistant.exe

# no multi-gigabyte speech runtime was pulled in transitively
ls dist/ermis-assistant/_internal | grep -iE "whisper|vosk|torch|tensorflow"   # expect no matches
```

A packaged build therefore has **no drafting**: Analysis > Draft transcription
reports that no local backend is installed, and the annotator types the
transcription as CLAUDE.md §1 originally required.

## What is not bundled, and why

**FFmpeg.** GPL/LGPL-licensed and large. Without it the loader still handles
PCM WAV through the Python standard library, and reports a clear message naming
FFmpeg when another container is opened. Bundling it would raise licensing
questions for whoever redistributes the build.

To give users the other formats, put `ffmpeg.exe` on `PATH`, or set
`AudioConfig.ffmpeg_path` to an absolute path in a local config file.

**The speech runtimes.** Excluded for size, not policy: torch plus a Whisper
model runs to several gigabytes, against 150-250 MB for the whole rest of the
build. The `excludes` list blocks whisper, vosk, speechbrain, deepspeech, torch,
tensorflow and transformers so none arrives as a transitive dependency.

Drafting (CLAUDE.md §60) is consequently a run-from-source feature. To ship a
build that can draft, remove the relevant names from `excludes`, install one
backend in the build environment, and expect the size to grow accordingly; the
model weights still download on first use unless you ship a local `model_dir`.

**`sounddevice` / PortAudio** is bundled when installed, but the application
degrades gracefully without a working output device: the transport still tracks
position and the status bar explains that no device was found.

## Distribution notes

- The build writes its database, logs and exports under `data/` next to the
  executable. On a locked-down machine where that directory is not writable,
  point `PathsConfig` at a user-writable location via a config file.
- Nothing in the build phones home. There is no telemetry, no update check and
  no network code, so the folder can be copied onto an air-gapped machine.
- No credentials are bundled or stored. The application authenticates against
  nothing.
