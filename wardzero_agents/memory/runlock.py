"""Run lock: a pid file that stops memory resets and restores while a run is active."""
from __future__ import annotations

import json
import os
from pathlib import Path


class RunActive(Exception):
    pass


def _lock_file(data_dir: Path) -> Path:
    return Path(data_dir) / "active_run.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_run(data_dir: Path) -> str | None:
    f = _lock_file(data_dir)
    try:
        info = json.loads(f.read_text())
        run_id, pid = info["run_id"], int(info["pid"])
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError):
        f.unlink(missing_ok=True)
        return None
    if _pid_alive(pid):
        return run_id
    f.unlink(missing_ok=True)
    return None


class RunLock:
    def __init__(self, data_dir: Path, run_id: str):
        self.data_dir = Path(data_dir)
        self.run_id = run_id

    def __enter__(self) -> "RunLock":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        other = active_run(self.data_dir)
        if other is not None:
            raise RunActive(f"run {other} is active; stop it first")
        _lock_file(self.data_dir).write_text(
            json.dumps({"run_id": self.run_id, "pid": os.getpid()}))
        return self

    def __exit__(self, *exc) -> None:
        _lock_file(self.data_dir).unlink(missing_ok=True)
