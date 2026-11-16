"""Shared test fixtures.

Audio fixtures are synthesised into tmp_path rather than committed, so the test
suite has no binary assets and runs on a clean checkout.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.ermis.findings import Code, ValidationReport
from app.ermis.validator import validate

SAMPLE_RATE = 44100


# ---------------------------------------------------------------- validation
@pytest.fixture
def check():
    """Assert that `text` produces (or does not produce) a finding code.

        check("play music?", Code.FORBIDDEN_PUNCTUATION)
        check("play music", absent=Code.FORBIDDEN_PUNCTUATION)
    """

    def _check(
        text: str,
        code: Code | None = None,
        *,
        absent: Code | None = None,
        **kwargs,
    ) -> ValidationReport:
        report = validate(text, **kwargs)
        if code is not None:
            assert report.has(code), (
                f"{text!r} should raise {code.value}; got "
                f"{sorted(c.value for c in report.codes()) or 'nothing'}"
            )
        if absent is not None:
            assert not report.has(absent), (
                f"{text!r} should not raise {absent.value}:\n{report}"
            )
        return report

    return _check


@pytest.fixture
def assert_valid():
    """Assert a transcription produces no ERROR findings."""

    def _assert_valid(text: str, **kwargs) -> ValidationReport:
        report = validate(text, **kwargs)
        assert report.is_valid, f"{text!r} should be valid but reported:\n{report}"
        return report

    return _assert_valid


@pytest.fixture
def assert_invalid():
    """Assert a transcription produces at least one ERROR finding."""

    def _assert_invalid(text: str, **kwargs) -> ValidationReport:
        report = validate(text, **kwargs)
        assert not report.is_valid, f"{text!r} should be invalid but passed"
        return report

    return _assert_invalid


# --------------------------------------------------------------------- audio
def write_wav(
    path: Path,
    data: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    """Write float samples in [-1, 1] as a PCM WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if sample_width == 2:
            raw = np.clip(data * 32767.0, -32768, 32767).astype("<i2").tobytes()
        elif sample_width == 1:
            raw = np.clip(data * 127.0 + 128.0, 0, 255).astype(np.uint8).tobytes()
        elif sample_width == 4:
            raw = np.clip(data * 2147483647.0, -2147483648, 2147483647).astype("<i4").tobytes()
        else:
            raise ValueError(f"unsupported width {sample_width}")
        wf.writeframes(raw)
    return path


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def wav_tone(tmp_path: Path) -> Path:
    """One second of a 220 Hz tone, mono 16-bit 44.1 kHz."""
    return write_wav(tmp_path / "tone.wav", tone(1.0))


@pytest.fixture
def wav_silence(tmp_path: Path) -> Path:
    """One second of digital silence -- the <na> measurement case."""
    return write_wav(tmp_path / "silence.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))


@pytest.fixture
def wav_stereo(tmp_path: Path) -> Path:
    """Two channels with different content, to exercise downmixing."""
    left = tone(1.0, 200.0, 0.2)
    right = tone(1.0, 400.0, 0.4)
    interleaved = np.stack([left, right], axis=1).ravel()
    return write_wav(tmp_path / "stereo.wav", interleaved, channels=2)


#: Duration of the synthetic click, in seconds. Real pops and clicks last on
#: the order of a millisecond or more; a single-sample impulse would be
#: attenuated by any correct anti-aliasing filter during resampling, so it
#: would not be a fair model of the artifact the QA workflow looks for.
CLICK_DURATION = 0.002


@pytest.fixture
def wav_speech_gap(tmp_path: Path) -> Path:
    """Tone, silent gap, tone, plus a brief loud click at 2.0 s.

    Mirrors the shape the analysis engine will look for: speech regions, a
    pause, and a brief artifact.
    """
    signal = tone(3.0)
    signal[int(1.0 * SAMPLE_RATE) : int(1.6 * SAMPLE_RATE)] = 0.0
    click_start = int(2.0 * SAMPLE_RATE)
    click_end = click_start + int(CLICK_DURATION * SAMPLE_RATE)
    signal[click_start:click_end] = 0.95
    return write_wav(tmp_path / "speech_gap.wav", signal)


# ------------------------------------------------------------------ database
@pytest.fixture
def db():
    """An in-memory database with schema applied."""
    from app.storage.database import Database

    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def audio_row(db):
    """A project plus audio file, returning the audio_file id."""
    project_id = db.create_project("test-project")
    return db.add_audio_file(project_id, "clip.wav", 12.5, 16000, 1)


# ---------------------------------------------------------------------------
# Speech-like synthesis for the detection tests.
#
# A plain sine tone is not a usable proxy for speech: it has no formants, so
# it reads as humming to the [hn] detector and its flat envelope defeats the
# rhythm detectors. These helpers drive a glottal pulse train through formant
# resonators, which produces the harmonic-plus-formant structure the detectors
# actually key on.
# ---------------------------------------------------------------------------
def _resonator(signal: np.ndarray, frequency: float, bandwidth: float = 90.0,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Single two-pole formant resonator."""
    from scipy.signal import lfilter

    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2 * np.pi * frequency / sample_rate
    return lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], signal)


def syllable_envelope(
    duration: float,
    rate: float = 4.5,
    jitter: float = 0.40,
    seed: int = 0,
    floor: float = 0.35,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Syllabic amplitude envelope.

    `floor` > 0 models connected speech, where voicing continues between
    syllables. `floor` near 0 models a pulsed sound such as laughter, where the
    signal drops away between bursts.

    `jitter` is the relative variation in syllable spacing: real speech timing
    varies with stress and word length, laughter is close to metronomic.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    envelope = np.full(n, floor)
    position = 0.0
    while position < duration:
        start = int(position * sample_rate)
        length = int(0.09 * sample_rate)
        if start + length > n:
            break
        burst = floor + (1 - floor) * np.exp(-np.linspace(0, 4.5, length))
        envelope[start : start + length] = np.maximum(
            envelope[start : start + length], burst
        )
        position += (1.0 / rate) * (1.0 + rng.normal(0, jitter))
    return envelope


def speech(
    duration: float,
    f0: float = 110.0,
    formants: tuple[float, ...] = (600.0, 1100.0, 2400.0),
    amplitude: float = 0.28,
    rate: float = 4.5,
    jitter: float = 0.40,
    seed: int = 0,
    floor: float = 0.35,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Speech-like signal: glottal pulses through formant resonators."""
    n = int(duration * sample_rate)
    pulses = np.zeros(n)
    pulses[:: max(1, int(sample_rate / f0))] = 1.0
    voiced = sum(
        gain * _resonator(pulses, formant, sample_rate=sample_rate)
        for formant, gain in zip(formants, (1.0, 0.6, 0.35))
    )
    peak = float(np.max(np.abs(voiced))) or 1.0
    envelope = syllable_envelope(
        duration, rate, jitter, seed, floor, sample_rate
    )
    return (amplitude * voiced / peak * envelope).astype(np.float32)


def laughter(duration: float = 1.6, seed: int = 9) -> np.ndarray:
    """Metronomic pulsed bursts that drop to near-silence between them."""
    return speech(
        duration,
        f0=190.0,
        formants=(850.0, 1300.0, 2700.0),
        amplitude=0.30,
        rate=5.0,
        jitter=0.0,
        seed=seed,
        floor=0.04,
    )


def burst(duration: float = 0.12, seed: int = 3) -> np.ndarray:
    """Broadband decaying noise burst: a cough, or a static pop."""
    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    return (rng.normal(0, 0.7, n) * np.exp(-np.linspace(0, 6, n))).astype(np.float32)


def silence(duration: float) -> np.ndarray:
    return np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)


def room_noise(duration: float, level: float = 0.0012, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, level, int(duration * SAMPLE_RATE)).astype(np.float32)


#: Two distinct voices: low/dark and high/bright.
VOICE_A = dict(f0=110.0, formants=(600.0, 1100.0, 2400.0))
VOICE_B = dict(f0=225.0, formants=(850.0, 1500.0, 2900.0))


@pytest.fixture
def wav_speech(tmp_path: Path) -> Path:
    """Two seconds of connected single-speaker speech."""
    return write_wav(tmp_path / "speech.wav", speech(2.0, seed=5, **VOICE_A))


@pytest.fixture
def wav_two_speakers(tmp_path: Path) -> Path:
    """Four sequential turns, A B A B, with a room-noise floor."""
    # 0.7s gaps, comfortably above DetectionConfig.pause_min_duration (0.5s),
    # so the pause classification is unambiguous rather than borderline.
    parts = [
        silence(0.3),
        speech(1.0, seed=1, **VOICE_A),
        silence(0.7),
        speech(0.9, seed=2, **VOICE_B),
        silence(0.7),
        speech(0.8, seed=3, **VOICE_A),
        silence(0.7),
        speech(1.0, seed=4, **VOICE_B),
        silence(0.3),
    ]
    signal = np.concatenate(parts)
    signal = signal + room_noise(len(signal) / SAMPLE_RATE, seed=7)[: len(signal)]
    return write_wav(tmp_path / "two_speakers.wav", signal.astype(np.float32))


@pytest.fixture
def wav_one_speaker_turns(tmp_path: Path) -> Path:
    """Four turns from ONE speaker -- the single-speaker diarization case."""
    parts = [silence(0.3)]
    for index, seed in enumerate((21, 22, 23, 24)):
        parts.append(speech(0.9, seed=seed, **VOICE_A))
        parts.append(silence(0.5))
    return write_wav(tmp_path / "one_speaker.wav", np.concatenate(parts))


@pytest.fixture
def wav_laughter(tmp_path: Path) -> Path:
    """Speech, then a metronomic laughter bout at 1.2-2.8s."""
    signal = np.concatenate(
        [speech(0.8, seed=6, **VOICE_A), silence(0.4), laughter(), silence(0.3)]
    )
    return write_wav(tmp_path / "laughter.wav", signal)


@pytest.fixture
def wav_two_coughs(tmp_path: Path) -> Path:
    """Speech, two coughs in immediate succession, speech (case B, section 55)."""
    signal = np.concatenate(
        [
            speech(0.7, seed=7, **VOICE_A),
            silence(0.35),
            burst(),
            silence(0.35),
            burst(seed=4),
            silence(0.4),
            speech(0.6, seed=8, **VOICE_A),
        ]
    )
    return write_wav(tmp_path / "two_coughs.wav", signal)


@pytest.fixture
def wav_click(tmp_path: Path) -> Path:
    """Speech with a 4 ms broadband click at exactly 1.0s."""
    signal = speech(2.0, seed=9, **VOICE_A).copy()
    rng = np.random.default_rng(13)
    length = int(0.004 * SAMPLE_RATE)
    start = int(1.0 * SAMPLE_RATE)
    signal[start : start + length] = (
        rng.normal(0, 1, length) * np.exp(-np.linspace(0, 3, length)) * 0.95
    ).astype(np.float32)
    return write_wav(tmp_path / "click.wav", signal)


@pytest.fixture
def wav_overlap(tmp_path: Path) -> Path:
    """Solo A, then A and B simultaneously, then solo B."""
    both = (speech(0.9, seed=11, **VOICE_A) + speech(0.9, seed=12, **VOICE_B)) * 0.5
    signal = np.concatenate(
        [
            speech(0.7, seed=10, **VOICE_A),
            silence(0.3),
            both.astype(np.float32),
            silence(0.3),
            speech(0.6, seed=13, **VOICE_B),
        ]
    )
    return write_wav(tmp_path / "overlap.wav", signal)


@pytest.fixture
def wav_hiss(tmp_path: Path) -> Path:
    """Steady broadband noise, no speech -- the <ns> case (section 12)."""
    return write_wav(tmp_path / "hiss.wav", room_noise(2.0, level=0.05, seed=1))


@pytest.fixture
def wav_clipped(tmp_path: Path) -> Path:
    """Heavily clipped audio -- a <ga> candidate (section 10).

    Amplified far past full scale so a large share of samples sit at the rail,
    rather than hovering at the threshold where the test would be flaky.
    """
    signal = speech(2.0, seed=14, **VOICE_A) * 40.0
    return write_wav(tmp_path / "clipped.wav", np.clip(signal, -1.0, 1.0))
