"""Background drafting (CLAUDE.md sections 45, 47, 59).

Recognition takes seconds per clip on CPU, so it runs on a worker thread and
the window stays usable while it works. The worker touches no widget: it emits
the finished `DraftResult` and the main thread decides what to do with it.

The model is loaded once per process and reused. A cold load costs tens of
seconds -- weights, quantisation and kernel warm-up -- and paying that for
every clip would make the feature unusable in a real annotation session.

Requires PySide6.
"""

from __future__ import annotations

import logging

try:
    from PySide6.QtCore import QThread, Signal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "app.ui.draft_worker requires PySide6. Install it with `pip install PySide6`."
    ) from exc

from app.asr.draft import DraftBuilder, DraftResult
from app.asr.engine import AsrError, get_engine, reset_engine_cache
from app.audio.loader import AudioBuffer
from app.config import AsrConfig, ValidationConfig
from app.detection.pipeline import AnalysisResult

log = logging.getLogger("ermis.ui.draft")

#: The loaded-model cache lives in app.asr.engine so every caller -- the UI
#: worker, the CLI and the draft builder -- shares one model per process.
_engine_for = get_engine

__all__ = ["DraftWorker", "ModelPreloadWorker", "draft_blocking", "reset_engine_cache"]


class ModelPreloadWorker(QThread):
    """Loads the speech model in the background at start-up.

    Without this the cost lands on the first clip: the annotator records, waits
    through a silent minute of weight loading and quantisation, and only then
    sees a draft. Doing it while the window opens means the model is usually
    ready before there is anything to draft.
    """

    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, config: AsrConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            engine = get_engine(self.config)
        except AsrError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - backend specific
            log.exception("model preload failed")
            self.failed.emit(f"model preload failed: {exc}")
            return
        message = f"{engine.name} / {engine.config.model}"
        if engine.memory_note:
            message += f" — {engine.memory_note}"
        self.ready.emit(message)


class DraftWorker(QThread):
    """Runs the recogniser and the draft builder for one clip."""

    #: Emitted with the finished DraftResult.
    drafted = Signal(object)
    #: Emitted with a human-readable reason when no draft could be produced.
    failed = Signal(str)
    #: Progress text for the status bar.
    progress = Signal(str)

    def __init__(
        self,
        buffer: AudioBuffer,
        analysis: AnalysisResult,
        config: AsrConfig,
        validation: ValidationConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.buffer = buffer
        self.analysis = analysis
        self.config = config
        self.validation = validation

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.progress.emit("Drafting from the audio...")
            engine = get_engine(self.config)
            if engine.memory_note:
                self.progress.emit(engine.memory_note)
            self.progress.emit(
                f"Recognising {self.buffer.duration:.1f}s of audio with "
                f"{engine.name} ({engine.config.model})..."
            )
            asr = engine.transcribe(self.buffer)
            builder = DraftBuilder(self.config, self.validation)
            result = builder.build(self.buffer, self.analysis, asr=asr)
        except AsrError as exc:
            log.warning("drafting unavailable: %s", exc)
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - backend specific
            log.exception("drafting failed")
            self.failed.emit(f"drafting failed: {exc}")
            return
        self.drafted.emit(result)


def draft_blocking(
    buffer: AudioBuffer,
    analysis: AnalysisResult,
    config: AsrConfig,
    validation: ValidationConfig | None = None,
) -> DraftResult:
    """Draft on the calling thread, reusing the cached engine.

    Used by tests, and by any caller that already runs off the GUI thread.
    """
    engine = get_engine(config)
    asr = engine.transcribe(buffer)
    return DraftBuilder(config, validation, engine=engine).build(
        buffer, analysis, asr=asr
    )
