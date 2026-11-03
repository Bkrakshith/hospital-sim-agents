"""Replay readers for a recorded run folder."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from .contracts import Intent, Say, WorldEvent, from_json

log = logging.getLogger(__name__)


def iter_stream(run_dir: Path, stream: str) -> Iterator[dict]:
    path = Path(run_dir) / f"{stream}.jsonl"
    if not path.exists():
        return iter(())
    lines = path.read_text(encoding="utf-8").split("\n")
    records: list[dict] = []
    last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            records.append(json.loads(ln))
        except ValueError:
            if i == last:
                log.warning("%s: skipping truncated last line", path)
            else:
                raise
    records.sort(key=lambda r: r.get("t", 0.0))  # stable
    return iter(records)


def load_meta(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))


class _Script:
    def __init__(self, items: list[tuple[float, object]]):
        self._items = sorted(items, key=lambda x: x[0])  # stable
        self._i = 0

    def due(self, t: float) -> list:
        out = []
        while self._i < len(self._items) and self._items[self._i][0] <= t:
            out.append(self._items[self._i][1])
            self._i += 1
        return out


class IntentScript(_Script):
    def __init__(self, run_dir: Path):
        items: list[tuple[float, object]] = []
        for stream, cls in (("intents", Intent), ("says", Say), ("world", WorldEvent)):
            for r in iter_stream(run_dir, stream):
                items.append((r.get("t", 0.0), from_json(cls, json.dumps(r))))
        super().__init__(items)


class ExactScript(_Script):
    def __init__(self, run_dir: Path):
        items: list[tuple[float, object]] = []
        for stream in ("status", "audio", "cards"):
            for r in iter_stream(run_dir, stream):
                items.append((r.get("t", 0.0), (stream, r)))
        super().__init__(items)
