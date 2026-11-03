"""Run folder recorder: every run writes JSONL streams, configs and meta so it can be replayed."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .keys import redact

STREAMS = (
    "decisions", "intents", "says", "nudges", "reflex", "world",
    "cards", "audio", "status", "tokens", "fairness", "summaries",
)


def _redact_all(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _redact_all(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_all(v) for v in value]
    return value


class RunLog:
    def __init__(self, dir: Path):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, stream: str, record: dict) -> None:
        if stream not in STREAMS:
            raise ValueError(f"unknown run log stream '{stream}'")
        line = redact(json.dumps(_redact_all(record), sort_keys=True, default=str))
        with self._lock:
            with open(self.dir / f"{stream}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def write_meta(self, meta: dict) -> None:
        text = redact(json.dumps(_redact_all(meta), sort_keys=True, indent=2, default=str))
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w", dir=self.dir, prefix="meta.", suffix=".tmp", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                tmp = f.name
            os.replace(tmp, self.dir / "meta.json")

    def copy_configs(self, loaded) -> None:
        """Copy the scenario, personas, area, signs, storyboard and deck YAML into config/."""
        dest = self.dir / "config"
        dest.mkdir(exist_ok=True)
        sc = loaded.scenario
        spath = Path(loaded.scenario_path)
        sources = [
            spath,
            loaded.root / "personas.yaml",
            loaded.root / f"{sc.area}.yaml",
            loaded.root / "signs.yaml",
            (spath.parent / sc.storyboard),
            (spath.parent / sc.deck),
        ]
        for src in sources:
            if src.is_file():
                shutil.copyfile(src, dest / src.name)


class RunTail:
    """Follow some streams of a run folder another process is writing.

    poll() returns the complete lines appended since the last poll, as (stream, record) pairs.
    A half-written last line waits for the next poll; a line that is not JSON is skipped.
    """

    def __init__(self, dir: Path, streams: "tuple[str, ...] | list[str]"):
        self.dir = Path(dir)
        self.streams = tuple(streams)
        self._pos = {s: 0 for s in self.streams}

    def poll(self) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        for stream in self.streams:
            path = self.dir / f"{stream}.jsonl"
            try:
                with open(path, "rb") as f:
                    f.seek(self._pos[stream])
                    data = f.read()
            except FileNotFoundError:
                continue
            end = data.rfind(b"\n")
            if end < 0:
                continue
            self._pos[stream] += end + 1
            for line in data[: end + 1].splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append((stream, rec))
        return out


def new_run_dir(root: Path, scenario_id: str, now: datetime) -> Path:
    base = f"{now.strftime('%Y%m%d-%H%M%S')}_{scenario_id}"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = root / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            n += 1
