# PyInstaller spec for the Ermis V2 Local Audio Annotation & QA Assistant.
#
# Build (from the project root):
#     pip install pyinstaller
#     pyinstaller packaging/ermis-assistant.spec
#
# Output: dist/ermis-assistant/ermis-assistant.exe
#
# Notes:
#   - onedir rather than onefile. A onefile build unpacks to a temp directory
#     on every launch, which is slow with PySide6 and makes the bundled Qt
#     plugins harder to diagnose when something is missing.
#   - FFmpeg is NOT bundled. It is GPL/LGPL-licensed and large; the loader
#     handles PCM WAV with the standard library alone and reports a clear
#     message when ffmpeg is needed but absent.
#   - No ASR runtime is bundled. torch/transformers/faster-whisper are large
#     (gigabytes with weights) and `excludes` below keeps them out, so a
#     packaged build has NO drafting: `asr.enabled` effectively falls back to
#     unavailable and the editor says what to install. Drafting is for running
#     from source, or for a build made after removing the relevant excludes.
#     See CLAUDE.md section 60 for the deviation this refers to.

import sys
from pathlib import Path

# SPECPATH is injected by PyInstaller; fall back for editors and linters.
PROJECT_ROOT = Path(globals().get("SPECPATH", ".")).resolve().parent

block_cipher = None

a = Analysis(
    ["../app/main.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        # The schema is read at runtime by app.storage.database.
        ("../app/storage/schema.sql", "app/storage"),
        ("../CLAUDE.md", "."),
        ("../docs/validation-rules.md", "docs"),
    ],
    hiddenimports=[
        "app.ui.main_window",
        "app.ui.waveform_view",
        "app.ui.transcription_editor",
        "app.ui.validation_panel",
        "app.ui.speaker_panel",
        "app.ui.event_panel",
        "app.ui.record_bar",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Kept out of the bundle for size, not for policy (section 60):
        # drafting works from source, where the user installs a backend.
        "whisper",
        "faster_whisper",
        "vosk",
        "speechbrain",
        "deepspeech",
        "torch",
        "torchaudio",
        "tensorflow",
        "transformers",
        # Unused Qt modules; dropping them saves well over 100 MB.
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtMultimedia",
        "PySide6.QtNetwork",
        # Test-only.
        "pytest",
        "_pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ermis-assistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressed Qt binaries are a common source of crashes.
    console=False,  # GUI application; the CLI is available via python -m app.main
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ermis-assistant",
)
