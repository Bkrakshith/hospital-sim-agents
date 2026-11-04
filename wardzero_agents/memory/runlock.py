"""Run lock: a pid file that stops memory resets and restores while a run is active."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class RunActive(Exception):
    pass


def _lock_file(data_dir: Path) -> Path:
    return Path(data_dir) / "active_run.json"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, OSError):
        return False
    return True


_YOUNG_S = 5.0


def _read(f: Path):
    """Return (run_id, pid), None when unparsable, raising FileNotFoundError when absent."""
    try:
        info = json.loads(f.read_text())
        return info["run_id"], int(info["pid"])
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, TypeError, OSError):
        return None


def active_run(data_dir: Path, ignore_pid: int | None = None) -> str | None:
    """The id of the run holding the lock, or None. A lock held by ignore_pid (the caller's own) does not count."""
    f = _lock_file(data_dir)
    try:
        info = _read(f)
        if info is None:
            # half-written by a live acquirer, or corrupt: only remove once it is old
            if time.time() - f.stat().st_mtime < _YOUNG_S:
                return "<starting>"
            f.unlink(missing_ok=True)
            return None
    except FileNotFoundError:
        return None
    run_id, pid = info
    if ignore_pid is not None and pid == ignore_pid and _pid_alive(pid):
        return None
    if _pid_alive(pid):
        return run_id
    f.unlink(missing_ok=True)
    return None


def held_by(data_dir: Path, pid: int) -> bool:
    """Whether the live process pid holds the lock (a parent that took it for the whole run)."""
    try:
        info = _read(_lock_file(data_dir))
    except FileNotFoundError:
        return False
    return info is not None and info[1] == pid and _pid_alive(pid)


class RunLock:
    def __init__(self, data_dir: Path, run_id: str):
        self.data_dir = Path(data_dir)
        self.run_id = run_id

    def _try_create(self) -> bool:
        try:
            fd = os.open(_lock_file(self.data_dir), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"run_id": self.run_id, "pid": os.getpid()}))
        return True

    def __enter__(self) -> "RunLock":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self._try_create():
            return self
        other = active_run(self.data_dir)  # removes a stale lock
        if other is not None or not self._try_create():
            other = active_run(self.data_dir) or other or "unknown"
            raise RunActive(f"run {other} is active; stop it first")
        return self

    def __exit__(self, *exc) -> None:
        f = _lock_file(self.data_dir)
        try:
            if _read(f) == (self.run_id, os.getpid()):
                f.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
