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

import yaml

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

    def copy_configs(self, loaded, overrides: "dict | None" = None) -> Path:
        """Copy the scenario, personas, area, signs, storyboard and deck YAML into config/.

        The copy is self-contained: the copied scenario's storyboard and deck fields name the
        copied files, so load_scenario(config/<scenario>.yaml) works from the run folder alone
        (rerun relies on it). overrides are top-level scenario fields to set in the copy.
        Copying a run's own copy onto itself changes nothing. Returns the copied scenario path.
        """
        dest = self.dir / "config"
        dest.mkdir(exist_ok=True)
        sc = loaded.scenario
        spath = Path(loaded.scenario_path)
        storyboard, deck = spath.parent / sc.storyboard, spath.parent / sc.deck
        plain = [loaded.root / "personas.yaml", loaded.root / f"{sc.area}.yaml", loaded.root / "signs.yaml"]
        taken = {spath.name, *(p.name for p in plain)}
        names = {}
        for kind, src in (("storyboard", storyboard), ("deck", deck)):
            name = src.name if src.name not in taken else f"{kind}_{src.name}"
            taken.add(name)
            names[kind] = name
        for src, name in [*((p, p.name) for p in plain), (storyboard, names["storyboard"]), (deck, names["deck"])]:
            dst = dest / name
            if src.is_file() and not (dst.exists() and os.path.samefile(src, dst)):
                shutil.copyfile(src, dst)
        raw = yaml.safe_load(spath.read_text(encoding="utf-8"))
        want = dict(raw)
        want.update(storyboard=names["storyboard"], deck=names["deck"])
        want.update(overrides or {})
        out = dest / spath.name
        if out.exists() and os.path.samefile(spath, out) and want == raw:
            return out  # the run's own copy, unchanged: never rewrite a file other nodes may be reading
        text = yaml.safe_dump(want, sort_keys=False)
        with tempfile.NamedTemporaryFile("w", dir=dest, prefix=".scenario.", suffix=".tmp", delete=False,
                                         encoding="utf-8") as f:
            f.write(text)
            tmp = f.name
        os.replace(tmp, out)
        return out


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
