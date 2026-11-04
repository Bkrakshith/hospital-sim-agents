"""Placing utterances on the run's timeline and mixing them (pure numpy, no TTS and no video)."""
from __future__ import annotations

import dataclasses
import math
import wave
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SAMPLE_RATE = 22050
REF_DB = 60.0  # the loudness of ordinary speech in the sim (world.audio.SPEECH_DB)
MAX_BOOST_DB, MAX_CUT_DB = 6.0, 18.0
DISTANCE_SLOPE = 10.0  # dB per decade of distance (half the free-field rate: it is a film, not a measurement)
PAN_RANGE_M = 8.0  # a source this far left or right of the listener is fully to that side
PEAK = 0.89  # about -1 dBFS


@dataclasses.dataclass
class Placed:
    """One synthesised line: mono float samples at SAMPLE_RATE, its run time, loudness and place."""
    t: float
    audio: np.ndarray
    loudness_db: float = REF_DB
    x: float = 0.0
    y: float = 0.0


def start_sample(t: float, t0: float, sr: int = SAMPLE_RATE) -> int:
    """Sample at which a line said at run time ``t`` starts, with video frame 0 at run time ``t0``."""
    return int(round((t - t0) * sr))


def resample(a: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or a.size == 0:
        return a.astype(np.float32)
    n = int(round(a.size * sr_to / sr_from))
    return np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a).astype(np.float32)


def gain_db(loudness_db: float, distance_m: float) -> float:
    """Relative level of a line: its loudness against ordinary speech, less a gentle distance loss."""
    g = (float(loudness_db) - REF_DB) - DISTANCE_SLOPE * math.log10(max(float(distance_m), 1.0))
    return max(-MAX_CUT_DB, min(MAX_BOOST_DB, g))


def pan_gains(dx: float) -> tuple[float, float]:
    """(left, right) constant-power gains for a source ``dx`` metres to the right of the listener."""
    p = max(-1.0, min(1.0, dx / PAN_RANGE_M))
    th = (p + 1.0) * math.pi / 4.0
    return math.cos(th) * math.sqrt(2.0), math.sin(th) * math.sqrt(2.0)


def mix(items: Sequence[Placed], t0: float, duration_s: float, listener: tuple = (0.0, 0.0), sr: int = SAMPLE_RATE,
        pan: bool = True) -> np.ndarray:
    """A stereo float32 array (n, 2) of exactly ``duration_s``. Overlapping lines are summed; lines before ``t0`` are
    trimmed at the start and lines past the end at the end. The sum is scaled so its peak is PEAK: it never clips."""
    n = int(round(duration_s * sr))
    out = np.zeros((n, 2), np.float32)
    for it in items:
        a = np.asarray(it.audio, np.float32)
        s = start_sample(it.t, t0, sr)
        if s + a.size <= 0 or s >= n or a.size == 0:
            continue
        if s < 0:
            a, s = a[-s:], 0
        a = a[:n - s]
        d = math.hypot(it.x - listener[0], it.y - listener[1])
        g = 10.0 ** (gain_db(it.loudness_db, d) / 20.0)
        l, r = pan_gains(it.x - listener[0]) if pan else (1.0, 1.0)
        out[s:s + a.size, 0] += a * g * l
        out[s:s + a.size, 1] += a * g * r
    peak = float(np.abs(out).max()) if n else 0.0
    if peak > 0:
        out *= PEAK / peak
    return out


def write_wav(path: Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """16-bit PCM WAV of an (n, channels) float array in [-1, 1]."""
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        ch, sr = w.getnchannels(), w.getframerate()
    a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    return (a.reshape(-1, ch) if ch > 1 else a), sr
