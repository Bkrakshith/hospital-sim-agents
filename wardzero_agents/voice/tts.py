"""Piper text-to-speech, offline and on the CPU. A model is loaded once and kept for the whole run."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .catalogue import MODELS, Voice, voices_dir

log = logging.getLogger("wardzero.voice")

ROBOT_BUZZ_HZ, ROBOT_BUZZ_DEPTH = 55.0, 0.3


class PiperUnavailable(RuntimeError):
    pass


def robotise(a: np.ndarray, sr: int) -> np.ndarray:
    """A light, pleasant machine tone: a 55 Hz amplitude buzz over the speech (the words stay clear)."""
    t = np.arange(a.size) / float(sr)
    return (a * (1.0 - ROBOT_BUZZ_DEPTH + ROBOT_BUZZ_DEPTH * np.sin(2 * np.pi * ROBOT_BUZZ_HZ * t))).astype(np.float32)


class Synth:
    """``say(voice, text)`` returns (mono float32 in [-1, 1], sample rate). Same text and voice are synthesised once."""

    def __init__(self, directory: Optional[Path] = None):
        try:
            from piper import PiperVoice, SynthesisConfig  # noqa: F401
        except ImportError as exc:
            raise PiperUnavailable('piper-tts is not installed: pip install "wardzero-agents[voice]"') from exc
        self.dir = voices_dir() if directory is None else Path(directory)
        self._models: dict[str, object] = {}
        self._cache: dict[tuple, tuple] = {}
        self.lines = 0
        self.seconds = 0.0

    def _model(self, stem: str):
        if stem not in self._models:
            from piper import PiperVoice

            p = self.dir / f"{MODELS[stem].stem}.onnx"
            self._models[stem] = PiperVoice.load(str(p))
        return self._models[stem]

    def say(self, voice: Voice, text: str, robot: bool = False) -> tuple:
        key = (voice.key, text, robot)
        if key in self._cache:
            return self._cache[key]
        import time

        from piper import SynthesisConfig

        m = self._model(voice.model)
        speaker_id = None
        if voice.speaker is not None:
            speaker_id = m.config.speaker_id_map[voice.speaker]
        cfg = SynthesisConfig(speaker_id=speaker_id, length_scale=voice.length_scale)
        t0 = time.monotonic()
        chunks = [c.audio_int16_array.astype(np.float32) / 32768.0 for c in m.synthesize(text, cfg)]
        a = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
        sr = int(m.config.sample_rate)
        if robot:
            a = robotise(a, sr)
        self.seconds += time.monotonic() - t0
        self.lines += 1
        self._cache[key] = (a, sr)
        return a, sr
