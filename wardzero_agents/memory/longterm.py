"""Per-person long-term memory in SQLite with recency and importance weighted search."""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from wardzero_agents.campaign.trust import TRUST_DDL
from wardzero_agents.kb.embed import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY, person_key TEXT, run_id TEXT, run_index INTEGER, t REAL,
  kind TEXT CHECK(kind IN ('seed','episode','fact')), text TEXT, importance REAL, vector BLOB,
  tag TEXT);
CREATE INDEX IF NOT EXISTS memories_person ON memories(person_key);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, run_index INTEGER, started_at TEXT);
""" + TRUST_DDL


@dataclass
class MemoryHit:
    text: str
    kind: str
    importance: float
    run_index: int
    score: float
    tag: Optional[str] = None  # "robot:<robot_id>" for a memory about a robot


def robot_tag(robot_id: str) -> str:
    return f"robot:{robot_id}"


def render_memory(hit: MemoryHit, robot_id: str) -> str:
    """A memory as the person recalls it. One about another robot is marked as such."""
    if hit.tag and hit.tag.startswith("robot:") and hit.tag != robot_tag(robot_id):
        return f"a different robot ({hit.text})"
    return hit.text


class LongTermMemory:
    def __init__(self, path: Path, embedder: Embedder):
        self.path = Path(path)
        self.embedder = embedder
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)
            with c:
                if "tag" not in {r[1] for r in c.execute("PRAGMA table_info(memories)")}:
                    c.execute("ALTER TABLE memories ADD COLUMN tag TEXT")
                row = c.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
                want = f"{embedder.name}:{embedder.dim}"
                if row is None:
                    c.execute("INSERT INTO meta VALUES('embedder',?)", (want,))
                elif row[0] != want:
                    raise ValueError(
                        f"memory db was built with embedder {row[0]}, opened with {want}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def begin_run(self, run_id: str) -> int:
        with closing(self._conn()) as c, c:
            row = c.execute("SELECT run_index FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row:
                return row[0]
            idx = (c.execute("SELECT COALESCE(MAX(run_index),0) FROM runs").fetchone()[0]) + 1
            c.execute("INSERT INTO runs VALUES(?,?,?)",
                      (run_id, idx, datetime.now(timezone.utc).isoformat()))
            return idx

    def _run_index(self, c: sqlite3.Connection, run_id: str) -> int:
        row = c.execute("SELECT run_index FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown run_id {run_id!r}; call begin_run first")
        return row[0]

    def _insert(self, c, person_key, run_id, run_index, t, kind, texts, importance, tag=None):
        vecs = self.embedder.embed(list(texts))
        for text, v in zip(texts, vecs):
            c.execute(
                "INSERT INTO memories(person_key,run_id,run_index,t,kind,text,importance,vector,tag)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (person_key, run_id, run_index, t, kind, text, importance,
                 np.asarray(v, dtype=np.float32).tobytes(), tag))

    def add(self, person_key: str, run_id: str, t: float, kind: str, text: str,
            importance: float, tag: Optional[str] = None) -> None:
        with closing(self._conn()) as c, c:
            self._insert(c, person_key, run_id, self._run_index(c, run_id), t, kind, [text],
                         importance, tag)

    def ensure_seed(self, person_key: str, seed_texts: list[str]) -> None:
        with closing(self._conn()) as c, c:
            if c.execute("SELECT 1 FROM memories WHERE person_key=? LIMIT 1",
                         (person_key,)).fetchone():
                return
            self._insert(c, person_key, "seed", 0, 0.0, "seed", seed_texts, 0.5)

    def search(self, person_key: str, query: str, k: int = 5, *, run_index: int) -> list[MemoryHit]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT text,kind,importance,run_index,t,vector,tag FROM memories WHERE person_key=?",
                (person_key,)).fetchall()
        if not rows:
            return []
        q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        scored = []
        for text, kind, imp, ri, t, blob, tag in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape[0] != self.embedder.dim:
                raise ValueError(
                    f"stored vector dim {v.shape[0]} != embedder dim {self.embedder.dim}")
            score = (0.6 * float(v @ q) + 0.25 * math.exp(-max(0, run_index - ri) / 5)
                     + 0.15 * imp)
            scored.append((score, t, MemoryHit(text, kind, imp, ri, score, tag)))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [s[2] for s in scored[:k]]

    def reset(self, person_keys: list[str], seeds: dict[str, list[str]]) -> None:
        with closing(self._conn()) as c, c:
            for key in person_keys:
                c.execute("DELETE FROM memories WHERE person_key=?", (key,))
                c.execute("DELETE FROM trust WHERE person_key=?", (key,))
                if seeds.get(key):
                    self._insert(c, key, "seed", 0, 0.0, "seed", seeds[key], 0.5)

    def has_robot_memories(self, person_key: str, robot_id: str) -> bool:
        with closing(self._conn()) as c:
            return c.execute("SELECT 1 FROM memories WHERE person_key=? AND tag=? LIMIT 1",
                             (person_key, robot_tag(robot_id))).fetchone() is not None

    def all_keys(self) -> list[str]:
        with closing(self._conn()) as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT person_key FROM memories ORDER BY person_key")]
