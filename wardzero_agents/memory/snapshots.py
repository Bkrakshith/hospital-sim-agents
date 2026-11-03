"""Named snapshots of the long-term memory db, via the sqlite backup API."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from wardzero_agents.kb.embed import Embedder, HashEmbedder
from wardzero_agents.memory import memory_path
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.memory.runlock import RunActive, active_run

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


RESERVED_PREFIXES = ("run-", "pre-restore-")  # saved automatically by runs and restores


def check_user_name(name: str) -> None:
    """Refuse names the tools save automatically, so a hand-made snapshot never replaces one."""
    _check(name)
    for p in RESERVED_PREFIXES:
        if name.startswith(p):
            raise ValueError(f"Snapshot names starting with '{p}' are reserved for the snapshots runs "
                             "save automatically. Choose another name.")


def _check(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid snapshot name {name!r}: use 1-64 of A-Z a-z 0-9 _ . -")
    if name in (".", ".."):
        raise ValueError(f"invalid snapshot name {name!r}")


def _snap_dir(mem_path: Path) -> Path:
    return Path(mem_path).parent.parent / "snapshots"


def _backup(src: Path, dst: Path) -> None:
    with closing(sqlite3.connect(src)) as s, closing(sqlite3.connect(dst)) as d:
        s.backup(d)


def save_snapshot(mem_path: Path, name: str) -> Path:
    _check(name)
    mem_path = Path(mem_path)
    if not mem_path.exists():
        raise FileNotFoundError(f"memory db {mem_path} does not exist yet")
    out_dir = _snap_dir(mem_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{name}.sqlite"
    dst.unlink(missing_ok=True)
    _backup(mem_path, dst)
    with closing(sqlite3.connect(dst)) as c:
        rows = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        keys = [r[0] for r in c.execute(
            "SELECT DISTINCT person_key FROM memories ORDER BY person_key")]
    (out_dir / f"{name}.json").write_text(json.dumps(
        {"created": datetime.now(timezone.utc).isoformat(), "rows": rows, "keys": keys}))
    return dst


def restore_snapshot(mem_path: Path, name: str) -> None:
    _check(name)
    mem_path = Path(mem_path)
    src = _snap_dir(mem_path) / f"{name}.sqlite"
    if not src.exists():
        raise FileNotFoundError(f"snapshot {name!r} not found ({src})")
    try:
        with closing(sqlite3.connect(f"file:{src}?mode=ro", uri=True)) as c:
            c.execute("SELECT 1 FROM memories LIMIT 1")
    except sqlite3.Error as e:
        raise ValueError(f"snapshot {name!r} is not a valid memory db: {e}") from e
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    _backup(src, mem_path)


def save_pre_restore(mem_path: Path, name: str, restoring: str) -> str | None:
    """Before a restore overwrites the memory db, save it as `name` and say so.

    Returns the name, or None when there is no memory db yet (nothing to lose).
    """
    if not Path(mem_path).exists():
        return None
    save_snapshot(mem_path, name)
    print(f"Saved the current long-term memory as snapshot {name} before restoring {restoring}; "
          f"`hhumans memory snapshot restore {name}` brings it back.", flush=True)
    return name


def list_snapshots(data_dir: Path) -> list[dict]:
    out = []
    for f in sorted((Path(data_dir) / "snapshots").glob("*.sqlite")):
        meta = {}
        try:
            meta = json.loads(f.with_suffix(".json").read_text())
        except (OSError, ValueError):
            pass
        out.append({"name": f.stem, **meta})
    return out


def _refuse_if_active(data_dir: Path) -> None:
    run_id = active_run(data_dir)
    if run_id is not None:
        raise RunActive(f"run {run_id} is active; stop it first")


def guarded_reset(data_dir: Path, keys: list[str], seeds: dict[str, list[str]],
                  embedder: Embedder | None = None) -> None:
    _refuse_if_active(data_dir)
    LongTermMemory(memory_path(data_dir), embedder or HashEmbedder()).reset(keys, seeds)


def guarded_restore(data_dir: Path, name: str, keep_as: str | None = None) -> str | None:
    """Restore snapshot `name` unless a run is active. With keep_as, the memory db is first saved
    under that name (see save_pre_restore). Returns the name it was kept as, if any."""
    _refuse_if_active(data_dir)
    kept = None
    if keep_as is not None:
        src = Path(data_dir) / "snapshots" / f"{name}.sqlite"
        _check(name)
        if not src.exists():
            raise FileNotFoundError(f"snapshot {name!r} not found ({src})")
        kept = save_pre_restore(memory_path(data_dir), keep_as, name)
    restore_snapshot(memory_path(data_dir), name)
    return kept
