"""Local ASR backends producing word-level timings and confidences.

Backends, in preference order:

  1. `faster-whisper` (CTranslate2). Fastest on CPU and gives a real per-word
     probability, which is what the draft builder uses to decide whether a word
     is confident enough to write plainly or should be wrapped as
     unintelligible.
  2. `transformers` + `torch` Whisper. Present on more machines; word
     timestamps come from the model's alignment heads and per-word confidence
     from the generation scores.

Both run entirely offline once the model is in the local cache. Neither uploads
audio; there is no cloud ASR path in this application and
`PrivacyConfig.allow_external_asr` still refuses to be enabled.

Long audio is processed in windows with overlap-free stitching, so a
twenty-minute clip does not need a twenty-minute context.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.audio.loader import AudioBuffer
from app.config import AsrConfig

log = logging.getLogger("ermis.asr")

#: Whisper's own input rate. Buffers are resampled to this before inference.
_MODEL_RATE = 16_000

#: Window length used to chunk long audio, in seconds. Whisper's receptive
#: field is 30 s; windows are cut on silence where possible.
_WINDOW_SECONDS = 28.0

#: Tokens Whisper emits that are not words.
_SPECIAL_RE = re.compile(r"^<\|.*\|>$")


class AsrError(RuntimeError):
    """Raised when no backend is usable, or inference fails."""


@dataclass(frozen=True)
class AsrWord:
    """One recognised word with its timing and the model's confidence."""

    text: str
    start: float
    end: float
    #: Model probability in [0, 1]. 1.0 when a backend cannot report one.
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def shifted(self, offset: float) -> AsrWord:
        return AsrWord(self.text, self.start + offset, self.end + offset, self.confidence)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class AsrResult:
    """Raw recogniser output, before any Ermis normalisation.

    This is *not* a transcription. It is the machine's reading of the audio,
    which the draft builder converts to spoken form and the annotator then
    corrects against the audio.
    """

    words: list[AsrWord]
    backend: str
    model: str
    language: str = "en"
    #: Seconds of audio processed.
    duration: float = 0.0
    #: True when the backend reports genuine per-word probabilities.
    has_confidence: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(np.mean([w.confidence for w in self.words]))

    def words_between(self, start: float, end: float) -> list[AsrWord]:
        """Words whose midpoint falls inside [start, end)."""
        return [w for w in self.words if start <= (w.start + w.end) / 2 < end]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "duration": round(self.duration, 3),
            "word_count": len(self.words),
            "mean_confidence": round(self.mean_confidence, 3),
            "has_confidence": self.has_confidence,
            "raw_text": self.text,
            "words": [w.to_dict() for w in self.words],
            "notes": list(self.notes),
        }


class AsrEngine:
    """Base class. Subclasses implement `_transcribe_window`."""

    name = "base"

    #: Set when the requested model was swapped for a smaller one to fit in
    #: memory, so the UI can say so instead of quietly getting worse.
    memory_note: str | None = None

    def __init__(self, config: AsrConfig | None = None) -> None:
        self.config = config or AsrConfig()
        self._loaded = False

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        """Load the model. Called lazily by `transcribe`."""
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------ inference
    def transcribe(self, buffer: AudioBuffer) -> AsrResult:
        """Recognise a whole buffer, chunking it if it is long."""
        if buffer.duration > self.config.max_audio_seconds:
            raise AsrError(
                f"{buffer.duration / 60:.1f} minutes is above the ASR limit of "
                f"{self.config.max_audio_seconds / 60:.0f} minutes; split the clip"
            )
        if not self.is_loaded:
            self.load()

        samples = _resample_for_model(buffer)
        if samples.size == 0:
            return AsrResult([], self.name, self.config.model, self.config.language, 0.0)

        words: list[AsrWord] = []
        for offset, window in _windows(samples, _MODEL_RATE):
            for word in self._transcribe_window(window):
                words.append(word.shifted(offset))

        words.sort(key=lambda w: (w.start, w.end))
        return AsrResult(
            words=words,
            backend=self.name,
            model=self.config.model,
            language=self.config.language,
            duration=buffer.duration,
            has_confidence=self.reports_confidence,
        )

    reports_confidence = True

    def _transcribe_window(self, window: np.ndarray) -> list[AsrWord]:
        raise NotImplementedError


class FasterWhisperEngine(AsrEngine):
    """faster-whisper backend: int8 CTranslate2 inference on CPU."""

    name = "faster-whisper"

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional backend
            raise AsrError(
                "faster-whisper is not installed; `pip install faster-whisper`"
            ) from exc
        model_id = self.config.model_dir or _short_model_name(self.config.model)
        log.info("loading faster-whisper model %s", model_id)
        self._model = WhisperModel(
            model_id,
            device=self.config.device,
            compute_type=self.config.compute_type,
            local_files_only=self.config.local_files_only,
        )
        self._loaded = True

    def _transcribe_window(self, window: np.ndarray) -> list[AsrWord]:  # pragma: no cover
        segments, _info = self._model.transcribe(
            window,
            language=self.config.language or None,
            beam_size=self.config.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        words: list[AsrWord] = []
        for segment in segments:
            for word in segment.words or ():
                text = word.word.strip()
                if not text:
                    continue
                words.append(
                    AsrWord(text, float(word.start), float(word.end), float(word.probability))
                )
        return words


class TransformersWhisperEngine(AsrEngine):
    """transformers + torch Whisper backend with token timestamps and scores."""

    name = "transformers-whisper"

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
        except ImportError as exc:
            raise AsrError(
                "the transformers Whisper backend needs `torch` and "
                "`transformers`; install them or install faster-whisper instead"
            ) from exc

        source = self.config.model_dir or self.config.model
        log.info("loading transformers Whisper model %s", source)
        try:
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(
                source, local_files_only=self.config.local_files_only
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                source, local_files_only=self.config.local_files_only
            )
        except Exception as exc:
            raise AsrError(f"could not load the ASR model {source!r}: {exc}") from exc
        self._model.eval()
        if self.config.device not in ("cpu", "", None):
            try:  # pragma: no cover - depends on hardware
                self._model.to(self.config.device)
            except Exception as exc:
                log.warning("could not move the model to %s: %s", self.config.device, exc)
        elif self.config.quantize_dynamic:
            self._quantize()
        self._loaded = True
        if self.config.warm_up:
            self._warm_up()

    def _quantize(self) -> None:
        """Dynamically quantise linear layers to int8 for CPU inference."""
        torch = self._torch
        try:
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
            log.info("quantised the ASR model to int8 for CPU inference")
        except Exception as exc:  # pragma: no cover - platform specific
            log.warning("dynamic quantisation unavailable (%s); using float32", exc)

    def _warm_up(self) -> None:
        """One short inference so the first real clip is not the slow one."""
        try:
            self._transcribe_window(np.zeros(_MODEL_RATE, dtype=np.float32))
            log.info("ASR warm-up complete")
        except Exception as exc:  # pragma: no cover - depends on backend
            log.warning("ASR warm-up failed (%s); continuing", exc)

    def _transcribe_window(self, window: np.ndarray) -> list[AsrWord]:
        torch = self._torch
        inputs = self._processor(
            window,
            sampling_rate=_MODEL_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs["input_features"].to(self._model.device)
        mask = inputs.get("attention_mask")
        kwargs = dict(
            return_token_timestamps=True,
            num_beams=max(1, int(self.config.beam_size)),
            do_sample=False,
        )
        if mask is not None:
            # Without the mask the alignment heads report coarser times.
            kwargs["attention_mask"] = mask.to(self._model.device)
        if self.config.language and not str(self.config.model).endswith(".en"):
            kwargs["language"] = self.config.language
            kwargs["task"] = "transcribe"

        with torch.no_grad():
            output = self._model.generate(features, **kwargs)

        sequence, stamps = _unpack_generation(output)
        probabilities = self._token_probabilities(features, sequence)
        words = _tokens_to_words(
            self._processor.tokenizer, sequence.tolist(), stamps, probabilities
        )
        return _finalize_word_times(words, len(window) / _MODEL_RATE)

    def _token_probabilities(self, features, sequence) -> list[float] | None:
        """Per-token probability, from one teacher-forced forward pass.

        `generate` in this version returns a plain dict with no scores when
        token timestamps are requested, so the probabilities are recovered by
        scoring the produced sequence once. That is a single extra
        encoder-decoder pass, cheap next to generation, and it gives the exact
        probability the model assigned to each token it chose.

        The generated sequence arrives without its forced prefix, so the prefix
        is put back before scoring; feeding the bare sequence would misalign
        every position and produce meaningless numbers.
        """
        torch = self._torch
        tokenizer = self._processor.tokenizer
        try:
            prefix = [
                token_id
                for token_id in tokenizer.convert_tokens_to_ids(
                    ["<|startoftranscript|>", "<|notimestamps|>"]
                )
                if token_id is not None
            ]
            if not prefix:
                start = self._model.generation_config.decoder_start_token_id
                prefix = [int(start)] if start is not None else []
            full = torch.cat(
                [torch.tensor(prefix, dtype=sequence.dtype, device=sequence.device), sequence]
            )
            with torch.no_grad():
                logits = self._model(
                    features, decoder_input_ids=full[:-1].unsqueeze(0)
                ).logits[0]
            probs = torch.softmax(logits.float(), dim=-1)
            offset = len(prefix)
            return [
                float(probs[offset + index - 1, int(token_id)])
                for index, token_id in enumerate(sequence.tolist())
            ]
        except Exception as exc:
            # Confidence is a nice-to-have; losing it must not lose the words.
            # The result then reports has_confidence=False and the draft
            # builder stops wrapping low-confidence words as unintelligible.
            log.warning("could not score ASR tokens (%s); confidence unavailable", exc)
            self.reports_confidence = False
            return None


def _tokens_to_words(
    tokenizer,
    token_ids: list[int],
    timestamps: list[float] | None,
    probabilities: list[float] | None,
) -> list[AsrWord]:
    """Group Whisper tokens into words with timings and mean probability.

    Whisper BPE marks a word start with a leading space, so a token that starts
    with one opens a new word and everything else continues the current one.
    """
    words: list[AsrWord] = []
    current = ""
    start = 0.0
    end = 0.0
    probs: list[float] = []

    def flush() -> None:
        nonlocal current, probs
        text = current.strip()
        if text:
            confidence = float(np.mean(probs)) if probs else 1.0
            words.append(AsrWord(text, start, max(end, start), confidence))
        current = ""
        probs = []

    for index, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if _SPECIAL_RE.match(piece.strip()) or not piece:
            continue
        stamp = timestamps[index] if timestamps and index < len(timestamps) else None
        probability = (
            probabilities[index]
            if probabilities and index < len(probabilities)
            else 1.0
        )
        if piece.startswith(" ") or not current:
            flush()
            if stamp is not None:
                start = float(stamp)
        current += piece
        probs.append(probability)
        if stamp is not None:
            end = float(stamp)
    flush()

    # A model that reported no timestamps still produced usable words; spread
    # them evenly so the draft can at least place tags between them.
    if timestamps is None and words:
        words = _spread_evenly(words)
    return words


def _unpack_generation(output) -> tuple:
    """Return (sequence, token stamps) from either a dict or a ModelOutput."""
    if isinstance(output, dict):
        sequence = output["sequences"][0]
        stamps = output.get("token_timestamps")
    else:  # pragma: no cover - older transformers return a ModelOutput
        sequence = output.sequences[0]
        stamps = getattr(output, "token_timestamps", None)
    return sequence, (stamps[0].tolist() if stamps is not None else None)


def _finalize_word_times(words: list[AsrWord], duration: float) -> list[AsrWord]:
    """Give every word a non-empty span.

    Whisper reports one alignment time per token, so a single-token word comes
    out with end == start. Tag placement needs real spans, so a zero-length
    word is extended to the next word's start.
    """
    fixed: list[AsrWord] = []
    for index, word in enumerate(words):
        end = word.end
        next_start = words[index + 1].start if index + 1 < len(words) else duration
        if end <= word.start:
            end = next_start if next_start > word.start else word.start + 0.2
        end = min(end, max(next_start, word.start + 0.05), max(duration, word.start + 0.05))
        fixed.append(AsrWord(word.text, word.start, end, word.confidence))
    return fixed


def _spread_evenly(words: list[AsrWord]) -> list[AsrWord]:
    step = 1.0 / max(1, len(words))
    return [
        AsrWord(word.text, index * step, (index + 1) * step, word.confidence)
        for index, word in enumerate(words)
    ]


# ------------------------------------------------------------------ helpers
def _resample_for_model(buffer: AudioBuffer) -> np.ndarray:
    """Mono float32 at the model rate."""
    samples = np.ascontiguousarray(buffer.samples, dtype=np.float32)
    if buffer.sample_rate == _MODEL_RATE or samples.size == 0:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(buffer.sample_rate), _MODEL_RATE)
        out = resample_poly(
            samples.astype(np.float64),
            _MODEL_RATE // divisor,
            buffer.sample_rate // divisor,
        )
        return np.ascontiguousarray(out, dtype=np.float32)
    except ImportError:  # pragma: no cover - scipy is a core dependency
        count = int(round(samples.size * _MODEL_RATE / buffer.sample_rate))
        source = np.linspace(0.0, samples.size - 1, samples.size)
        target = np.linspace(0.0, samples.size - 1, count)
        return np.interp(target, source, samples).astype(np.float32)


def _windows(samples: np.ndarray, rate: int) -> list[tuple[float, np.ndarray]]:
    """Split into model-sized windows, cutting at the quietest nearby frame.

    Cutting on a quiet frame rather than a fixed boundary keeps a word from
    being split across two windows, which would produce two half-words.
    """
    length = int(_WINDOW_SECONDS * rate)
    if samples.size <= length:
        return [(0.0, samples)]

    out: list[tuple[float, np.ndarray]] = []
    position = 0
    search = int(1.5 * rate)  # look this far back for a quiet cut point
    while position < samples.size:
        stop = min(samples.size, position + length)
        if stop < samples.size:
            window_start = max(position + length - search, position + 1)
            tail = samples[window_start:stop]
            if tail.size:
                # Quietest 20 ms block in the search region.
                block = max(1, int(0.02 * rate))
                usable = (tail.size // block) * block
                if usable:
                    energy = np.abs(tail[:usable]).reshape(-1, block).mean(axis=1)
                    stop = window_start + int(np.argmin(energy)) * block
        out.append((position / rate, samples[position:stop]))
        position = stop
    return out


def _short_model_name(model: str) -> str:
    """faster-whisper takes bare size names such as "small.en"."""
    return model.split("/")[-1].replace("whisper-", "")


def available_backends(config: AsrConfig | None = None) -> list[str]:
    """Backend names that can actually be imported on this machine."""
    found: list[str] = []
    try:
        import faster_whisper  # noqa: F401

        found.append(FasterWhisperEngine.name)
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        found.append(TransformersWhisperEngine.name)
    except ImportError:
        pass
    return found


def build_engine(config: AsrConfig | None = None) -> AsrEngine:
    """Pick a backend according to config, or raise with an install hint."""
    config = config or AsrConfig()
    if not config.enabled:
        raise AsrError(
            "ASR is disabled in configuration (asr.enabled is False). The tag "
            "and QA assistance works without it."
        )

    wanted = (config.backend or "auto").lower()
    engines = {
        FasterWhisperEngine.name: FasterWhisperEngine,
        TransformersWhisperEngine.name: TransformersWhisperEngine,
    }
    if wanted in engines:
        return engines[wanted](config)
    if wanted not in ("auto", ""):
        raise AsrError(
            f"unknown ASR backend {config.backend!r}; use one of: auto, "
            f"{', '.join(engines)}"
        )

    for name in available_backends(config):
        return engines[name](config)
    raise AsrError(
        "no local ASR backend is installed. Install one:\n"
        "  pip install faster-whisper      (recommended, fastest on CPU)\n"
        "  pip install torch transformers  (heavier)\n"
        "Both run offline once the model is cached."
    )


#: Process-wide loaded-engine cache. Loading a model costs tens of seconds and
#: over a gigabyte of memory, so a second copy in the same process is never
#: what the caller wants -- on Windows a concurrent second load fails outright
#: with "the paging file is too small" and every draft silently loses its words.
_CACHE_LOCK = threading.Lock()
_CACHED_ENGINE: AsrEngine | None = None
_CACHED_KEY: tuple | None = None


def _cache_key(config: AsrConfig) -> tuple:
    return (
        config.backend,
        config.model,
        config.model_dir,
        config.device,
        config.compute_type,
        config.quantize_dynamic,
        config.local_files_only,
    )


def get_engine(config: AsrConfig | None = None) -> AsrEngine:
    """A loaded engine for `config`, reusing the process-wide cached one.

    This is what callers should use. `build_engine` constructs a fresh,
    unloaded engine and is for callers that deliberately want their own.
    """
    global _CACHED_ENGINE, _CACHED_KEY
    config = config or AsrConfig()
    key = _cache_key(config)
    with _CACHE_LOCK:
        if _CACHED_ENGINE is not None and _CACHED_KEY == key:
            return _CACHED_ENGINE
        # Checked before the load, because a load that does not fit crashes the
        # process rather than raising.
        fitted, note = fit_model_to_memory(config)
        engine = build_engine(fitted)
        engine.load()
        engine.memory_note = note
        _CACHED_ENGINE, _CACHED_KEY = engine, key
        return engine


#: Peak memory each Whisper size needs while loading, in GB, per backend.
#:
#: The cost is a property of the backend, not of the weights alone, so one
#: table cannot serve both. Charging the transformers price to a backend that
#: does not pay it refuses a draft the machine could actually have run.
#:
#: transformers: the peak is not the model size. `quantize_dynamic` builds the
#: int8 copy while the float32 weights are still resident, so both exist for a
#: moment and the peak is roughly twice the resident figure.
#:
#: faster-whisper: CTranslate2 reads weights that were quantised ahead of time,
#: so there is no second copy and no conversion peak -- roughly the resident
#: size plus the runtime's working set.
_TRANSFORMERS_MODEL_MEMORY_GB: dict[str, float] = {
    "tiny": 0.6,
    "base": 1.0,
    "small": 2.6,
    "medium": 6.0,
    "large": 12.0,
}

_FASTER_WHISPER_MODEL_MEMORY_GB: dict[str, float] = {
    "tiny": 0.25,
    "base": 0.35,
    "small": 0.8,
    "medium": 2.0,
    "large": 3.6,
}

#: Used when the backend is not known. The expensive one, because guessing low
#: here means a load that does not fit, and that kills the process.
_MODEL_MEMORY_GB = _TRANSFORMERS_MODEL_MEMORY_GB

_BACKEND_MODEL_MEMORY_GB: dict[str, dict[str, float]] = {
    FasterWhisperEngine.name: _FASTER_WHISPER_MODEL_MEMORY_GB,
    TransformersWhisperEngine.name: _TRANSFORMERS_MODEL_MEMORY_GB,
}

#: Headroom left for the rest of the application, per backend.
#:
#: transformers needs the larger margin because its peak is a spike it does not
#: announce: `quantize_dynamic` allocates the int8 copy while the float32
#: weights are still held, and if that spike does not fit the process dies
#: rather than raising.
#:
#: faster-whisper loads weights that were quantised ahead of time, so its
#: requirement is close to the resident size and does not spike. Charging it
#: the same margin refused drafts on this project's own 8 GB machine that ran
#: without trouble once the margin matched the backend.
_TRANSFORMERS_HEADROOM_GB = 0.4
_FASTER_WHISPER_HEADROOM_GB = 0.15

_BACKEND_HEADROOM_GB: dict[str, float] = {
    FasterWhisperEngine.name: _FASTER_WHISPER_HEADROOM_GB,
    TransformersWhisperEngine.name: _TRANSFORMERS_HEADROOM_GB,
}

#: Used when the backend is not known: the larger margin, for the same reason
#: the dearer table is the fallback.
_MEMORY_HEADROOM_GB = _TRANSFORMERS_HEADROOM_GB


def memory_headroom_gb(backend: str | None) -> float:
    """Margin to leave free beyond the model itself, for `backend`."""
    return _BACKEND_HEADROOM_GB.get(backend or "", _MEMORY_HEADROOM_GB)

#: Sizes to fall back through, largest first.
_FALLBACK_ORDER = ("small", "base", "tiny")


def resolve_backend(config: AsrConfig | None = None) -> str | None:
    """The backend `build_engine` would choose, without building anything.

    Returns None when the choice cannot be made -- nothing installed, or a
    name that is not a backend -- leaving the caller to assume the worst.
    """
    config = config or AsrConfig()
    wanted = (config.backend or "auto").lower()
    if wanted in _BACKEND_MODEL_MEMORY_GB:
        return wanted
    if wanted not in ("auto", ""):
        return None
    for name in available_backends(config):
        return name
    return None


def memory_table(backend: str | None) -> dict[str, float]:
    """Per-size memory costs for `backend`, falling back to the dearest."""
    return _BACKEND_MODEL_MEMORY_GB.get(backend or "", _MODEL_MEMORY_GB)


def model_memory_estimate_gb(model: str, backend: str | None = None) -> float:
    """Peak memory the named model needs to load, in GB.

    Without a backend the transformers figures are used, which are the higher
    of the two: an over-estimate costs a smaller model, an under-estimate costs
    the process.
    """
    table = memory_table(backend)
    name = model.lower()
    for size, cost in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if size in name:
            return cost
    return table["small"]  # unknown name: assume small, the common default


def available_memory_gb() -> float | None:
    """Free physical memory in GB, or None when it cannot be measured."""
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        # psutil is optional. Without it the memory gate does not engage, which
        # is the pre-existing behaviour rather than a new failure mode.
        return None


def _cheaper_backend_hint(backend: str | None, available: float) -> str:
    """Suggest faster-whisper when it is the difference between draft and none.

    Only when it is installable and would actually change the answer, so the
    hint never appears on a machine where it would not help.
    """
    if backend == FasterWhisperEngine.name:
        return ""
    smallest = _FASTER_WHISPER_MODEL_MEMORY_GB["tiny"] + _FASTER_WHISPER_HEADROOM_GB
    if available < smallest:
        return ""
    return (
        " Installing faster-whisper (pip install faster-whisper) would fit in "
        f"the {available:.1f} GB you have: it reads weights that are already "
        "quantised, so it needs a fraction of the memory this backend does."
    )


def fit_model_to_memory(config: AsrConfig) -> tuple[AsrConfig, str | None]:
    """Return a config that can actually load, plus a note when it changed.

    A model that does not fit does not fail cleanly: on Windows the load dies
    with "the paging file is too small", or segfaults part-way through
    quantisation and takes the whole process with it. Neither can be caught
    afterwards, so the check has to happen before the load.

    The cost is read from the table for the backend that will actually be
    used, because the two differ by roughly a factor of three and the dearer
    figure would refuse drafts the cheaper backend could run.

    With `asr.downgrade_when_memory_is_tight` (the default) a smaller model in
    the same family is used and the caller is told, because a labelled draft
    from a smaller model beats no drafting at all. Otherwise this raises.
    """
    from dataclasses import replace

    available = available_memory_gb()
    if available is None:
        return config, None

    backend = resolve_backend(config)
    table = memory_table(backend)

    headroom = memory_headroom_gb(backend)
    required = model_memory_estimate_gb(config.model, backend) + headroom
    if config.min_free_memory_gb > 0:
        required = config.min_free_memory_gb
    if available >= required:
        return config, None

    if not config.downgrade_when_memory_is_tight:
        raise AsrError(
            f"{config.model} needs about {required:.1f} GB free to load and only "
            f"{available:.1f} GB is available. Close other applications, set a "
            "smaller asr.model (openai/whisper-base.en), or turn drafting off "
            "with asr.enabled = false." + _cheaper_backend_hint(backend, available)
        )

    suffix = ".en" if config.model.endswith(".en") else ""
    for size in _FALLBACK_ORDER:
        candidate = f"openai/whisper-{size}{suffix}"
        if candidate == config.model:
            continue
        cost = table[size] + headroom
        if cost <= available:
            note = (
                f"only {available:.1f} GB of memory is free, which is not enough "
                f"for {config.model} (~{required:.1f} GB), so drafting is using "
                f"{candidate} instead. It is faster and less accurate -- close "
                "other applications and restart for the larger model."
            )
            log.warning("%s", note)
            return replace(config, model=candidate), note

    raise AsrError(
        f"only {available:.1f} GB of memory is free and even the smallest model "
        f"needs about {table['tiny'] + headroom:.1f} GB. "
        "Close some applications, or turn drafting off with asr.enabled = false; "
        "recording, analysis and validation do not need a model."
        + _cheaper_backend_hint(backend, available)
    )


def active_memory_note() -> str | None:
    """The note from the loaded engine, when a smaller model was substituted."""
    with _CACHE_LOCK:
        return getattr(_CACHED_ENGINE, "memory_note", None)


def reset_engine_cache() -> None:
    """Drop the cached engine. Used by tests and after a settings change."""
    global _CACHED_ENGINE, _CACHED_KEY
    with _CACHE_LOCK:
        _CACHED_ENGINE, _CACHED_KEY = None, None


def model_is_cached(config: AsrConfig | None = None) -> bool:
    """True when the configured model appears to be in the local cache.

    Used by the UI to warn before a first run downloads a model, which is the
    only moment this feature touches the network.
    """
    config = config or AsrConfig()
    if config.model_dir and Path(config.model_dir).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for filename in ("config.json", "model.safetensors"):
        try:
            hit = try_to_load_from_cache(config.model, filename)
        except Exception:
            return False
        if isinstance(hit, str) and Path(hit).exists():
            continue
        if filename == "config.json":
            return False
    return True


def transcribe(buffer: AudioBuffer, config: AsrConfig | None = None) -> AsrResult:
    """Recognise one buffer, reusing the process-wide loaded engine."""
    return get_engine(config).transcribe(buffer)
