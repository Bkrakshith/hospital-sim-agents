"""Piper text-to-speech, offline and on the CPU, in its own process.

Piper is GPL-3 and this package is Apache-2.0, so nothing here imports it: ``voice/_piper_runner.py`` (the only code
that does) is run with the venv's python, given every line of a run in one JSON job file, and writes one WAV per line.
Models load once per call.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .catalogue import MODELS, Voice, voices_dir
from .mix import read_wav

RUNNER = Path(__file__).resolve().parent / "_piper_runner.py"  # package data, run by path, never imported
TIMEOUT_S = 900
ROBOT_BUZZ_HZ, ROBOT_BUZZ_DEPTH = 55.0, 0.3


class PiperUnavailable(RuntimeError):
    pass


def robotise(a: np.ndarray, sr: int) -> np.ndarray:
    """A light, pleasant machine tone: a 55 Hz amplitude buzz over the speech (the words stay clear)."""
    t = np.arange(a.size) / float(sr)
    return (a * (1.0 - ROBOT_BUZZ_DEPTH + ROBOT_BUZZ_DEPTH * np.sin(2 * np.pi * ROBOT_BUZZ_HZ * t))).astype(np.float32)


def piper_installed() -> bool:
    return importlib.util.find_spec("piper") is not None  # find_spec does not import it


class Synth:
    """``run([(voice, text, robot)])`` returns [(mono float32 in [-1, 1], sample rate)] in the same order. Equal
    (voice, text) pairs are synthesised once. ``seconds`` and ``lines`` are the runner's own timing."""

    def __init__(self, directory: Optional[Path] = None):
        if not piper_installed():
            raise PiperUnavailable('piper-tts is not installed: pip install "wardzero-agents[voice]"')
        if not RUNNER.is_file():
            raise PiperUnavailable(f"the Piper runner {RUNNER} is missing")
        self.dir = voices_dir() if directory is None else Path(directory)
        self.lines = 0
        self.seconds = 0.0

    def run(self, items: Sequence[tuple]) -> list:
        unique: dict = {}
        for v, text, _robot in items:
            unique.setdefault((v.key, text), (v, text))
        if not unique:
            return []
        with tempfile.TemporaryDirectory(prefix="wz_voice_") as tmp:
            tmp = Path(tmp)
            jobs, outs = [], {}
            for i, (key, (v, text)) in enumerate(unique.items()):
                out = tmp / f"line_{i:05d}.wav"
                outs[key] = out
                jobs.append({"model": str(self.dir / f"{MODELS[v.model].stem}.onnx"), "speaker": v.speaker,
                             "length_scale": v.length_scale, "text": text, "out": str(out)})
            (tmp / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, str(RUNNER), str(tmp / "jobs.json")], capture_output=True, text=True,
                                   timeout=TIMEOUT_S)
            except (OSError, subprocess.SubprocessError) as exc:
                raise PiperUnavailable(f"the Piper runner did not run ({type(exc).__name__})") from exc
            if r.returncode != 0:
                raise PiperUnavailable(f"the Piper runner failed (exit {r.returncode}): "
                                       f"{(r.stderr or r.stdout).strip().splitlines()[-1:] or ['no output']}")
            try:
                summary = json.loads(r.stdout.strip().splitlines()[-1])
                self.seconds += float(summary["seconds"])
                self.lines += int(summary["lines"])
            except (ValueError, KeyError, IndexError):
                pass
            audio = {}
            for key, out in outs.items():
                a, sr = read_wav(out)
                audio[key] = (a if a.ndim == 1 else a.mean(axis=1), sr)
        res = []
        for v, text, robot in items:
            a, sr = audio[(v.key, text)]
            res.append((robotise(a, sr) if robot else a, sr))
        return res
