"""Trust in the robot that rises and falls with experience (research_notes trust_dynamics.md, section 2).

All numbers are first guesses to tune, not measured constants."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ..config.presets import RolePreset

FAIL_STEP = 0.15
EASY_X = 1.5
BAD_OUTCOME_X = 1.5
SUCCESS_STEP = 0.05
REPAIR_AT_NEXT_REQUEST = 0.60  # share of the last drop restored by a repair at the next request
REPAIR_IMMEDIATE = 0.10  # share restored by one right after the error
STRIKES = 3
DRIFT_CLEAN_RUN = 0.03
HEARSAY_W = 0.3
BANDS = [0.6, 0.45, 0.3, 0.15]  # stage 0 at or above the first, stage 4 below the last

Kind = Literal["success", "failure", "repair_immediate", "repair_at_request", "hearsay"]


@dataclass(frozen=True)
class Experience:
    t: float
    person: str
    robot_id: str
    kind: Kind
    easy: bool = False
    bad_outcome: bool = False
    source_person: Optional[str] = None  # hearsay: who told them. A hearsay with bad_outcome is bad news.


@dataclass(frozen=True)
class TrustState:
    person_key: str
    robot_id: str
    trust: float
    self_confidence: float
    violations: int = 0
    repairs: int = 0
    ceiling: float = 1.0
    stage: int = 0
    runs: int = 0
    last_run: int = 0
    last_drop: float = 0.0  # the last failure's drop that a repair may still win back


def stage_for(trust: float) -> int:
    for stage, floor in enumerate(BANDS):
        if trust >= floor:
            return stage
    return len(BANDS)


def _scale(roll: int) -> float:
    if roll == 1:
        return 2.0  # a critical: the change doubles
    if roll == 20:
        return 0.0  # a natural 20: nothing moves
    return 0.5 + roll / 20


def _set(s: TrustState, trust: float, **kw) -> TrustState:
    trust = min(max(0.0, trust), s.ceiling if "ceiling" not in kw else kw["ceiling"])
    return replace(s, trust=trust, stage=stage_for(trust), **kw)


def apply_event(s: TrustState, ev: Experience, roll: int) -> TrustState:
    k = _scale(roll)
    if ev.kind == "failure":
        step = FAIL_STEP * (EASY_X if ev.easy else 1) * (BAD_OUTCOME_X if ev.bad_outcome else 1) * k
        drop = min(step, s.trust)
        n = s.violations + 1
        # the third strike caps what can ever be won back at about the level before it
        ceiling = min(s.ceiling, s.trust) if n >= STRIKES else s.ceiling
        return _set(s, s.trust - step, violations=n, ceiling=ceiling, last_drop=drop)
    if ev.kind == "success":
        return _set(s, s.trust + SUCCESS_STEP * k)
    if ev.kind in ("repair_immediate", "repair_at_request"):
        share = REPAIR_AT_NEXT_REQUEST if ev.kind == "repair_at_request" else REPAIR_IMMEDIATE
        gain = share * s.last_drop * k
        return _set(s, s.trust + gain, repairs=s.repairs + 1,
                    last_drop=max(0.0, s.last_drop - gain))
    if ev.kind == "hearsay":
        base = -FAIL_STEP if ev.bad_outcome else SUCCESS_STEP
        return _set(s, s.trust + HEARSAY_W * base * k)
    raise ValueError(f"unknown experience kind {ev.kind!r}")


def end_of_run(s: TrustState, clean: bool) -> TrustState:
    """Close a run: a clean run lets trust drift up a little, never past the ceiling."""
    t = s.trust + (DRIFT_CLEAN_RUN if clean else 0.0)
    return _set(s, t, runs=s.runs + 1, last_run=s.runs + 1)


TRUST_DDL = """
CREATE TABLE IF NOT EXISTS trust(
  person_key TEXT NOT NULL, robot_id TEXT NOT NULL,
  trust REAL NOT NULL CHECK(trust >= 0 AND trust <= 1),
  self_confidence REAL NOT NULL, violations INTEGER NOT NULL, repairs INTEGER NOT NULL,
  ceiling REAL NOT NULL, stage INTEGER NOT NULL CHECK(stage BETWEEN 0 AND 4),
  runs INTEGER NOT NULL, last_run INTEGER NOT NULL, last_drop REAL NOT NULL,
  PRIMARY KEY(person_key, robot_id));
"""
_COLS = [f.name for f in fields(TrustState)]


class TrustStore:
    """Trust per (person key, robot). It lives in the same SQLite file as long-term memory."""

    def __init__(self, db_path: "str | Path"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as c, c:
            c.executescript(TRUST_DDL)

    @staticmethod
    def initial(person_key: str, robot_id: str, defaults: "RolePreset") -> TrustState:
        return TrustState(person_key, robot_id, defaults.trust_start, defaults.self_confidence,
                          stage=stage_for(defaults.trust_start))

    def get(self, person_key: str, robot_id: str, defaults: "RolePreset") -> TrustState:
        with closing(sqlite3.connect(self.path)) as c:
            row = c.execute(f"SELECT {','.join(_COLS)} FROM trust WHERE person_key=? AND robot_id=?",
                            (person_key, robot_id)).fetchone()
        return TrustState(*row) if row else self.initial(person_key, robot_id, defaults)

    def put_many(self, states: list[TrustState]) -> None:
        """All rows or none."""
        rows = [tuple(asdict(s)[c] for c in _COLS) for s in states]
        with closing(sqlite3.connect(self.path)) as c, c:
            c.executemany(f"INSERT OR REPLACE INTO trust({','.join(_COLS)}) "
                          f"VALUES({','.join('?' * len(_COLS))})", rows)

    def reset(self, person_keys: list[str]) -> None:
        with closing(sqlite3.connect(self.path)) as c, c:
            c.executemany("DELETE FROM trust WHERE person_key=?", [(k,) for k in person_keys])

    def snapshot_rows(self) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as c:
            rows = c.execute(f"SELECT {','.join(_COLS)} FROM trust ORDER BY person_key, robot_id").fetchall()
        return [dict(zip(_COLS, r)) for r in rows]
