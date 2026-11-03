"""Run folder recorder: every run writes JSONL streams, configs and meta so it can be replayed."""
from __future__ import annotations

import json
import os
import shutil
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
        return {(redact(k) if isinstance(k, str) else k): _redact_all(v) for k, v in value.items()}
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
        line = json.dumps(_redact_all(record), sort_keys=True, default=str)
        with self._lock:
            with open(self.dir / f"{stream}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def write_meta(self, meta: dict) -> None:
        text = json.dumps(_redact_all(meta), sort_keys=True, indent=2, default=str)
        with self._lock:
            tmp = self.dir / "meta.json.tmp"
            tmp.write_text(text, encoding="utf-8")
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
