"""Storyboard engine: a deterministic, timed and conditional sequence of card firings."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from wardzero_agents.config.models import Beat, CardDef, Storyboard
from wardzero_agents.contracts import Status


@dataclass
class WorldView:
    t: float
    statuses: dict[str, Status]
    robot_xy: tuple[float, float]
    seated_since: dict[str, float] = field(default_factory=dict)
    present: set[str] = field(default_factory=set)


@dataclass
class CardFiring:
    id: str
    t: float
    card: str
    target: Optional[str]
    params: dict
    source: Literal["storyboard", "panel", "chaos"]


def parse_at(at: str) -> float:
    mm, ss = at[2:].split(":")
    return int(mm) * 60 + int(ss)


class StoryboardEngine:
    def __init__(self, storyboard: Storyboard, deck: dict[str, CardDef]):
        self.storyboard = storyboard
        self.deck = deck
        self._beats: dict[str, Beat] = {b.id: b for b in storyboard.beats}
        self._state = {b.id: "pending" for b in storyboard.beats}
        self._fired_t: dict[str, Optional[float]] = {b.id: None for b in storyboard.beats}
        self._fired_tick: dict[str, int] = {}
        self._reason: dict[str, Optional[str]] = {b.id: None for b in storyboard.beats}
        self._n = 0
        self._tick_no = 0

    def _condition_holds(self, beat: Beat, view: WorldView) -> bool:
        c = beat.when
        st = view.statuses.get(c.person)
        if c.kind == "posture_is":
            return st is not None and st.posture == c.value
        if c.kind == "robot_within":
            if st is None:
                return False
            return math.hypot(st.x - view.robot_xy[0], st.y - view.robot_xy[1]) <= float(c.value)
        since = view.seated_since.get(c.person)
        return since is not None and view.t - since >= float(c.value)

    def _ready(self, beat: Beat, view: WorldView) -> bool:
        if beat.at is not None:
            return view.t >= parse_at(beat.at)
        if beat.after is not None:
            return (self._state.get(beat.after) == "fired"
                    and self._fired_tick.get(beat.after, self._tick_no) < self._tick_no)
        return self._condition_holds(beat, view)

    def tick(self, view: WorldView) -> list[CardFiring]:
        self._tick_no += 1
        out: list[CardFiring] = []
        for beat in self.storyboard.beats:
            if self._state[beat.id] != "pending" or not self._ready(beat, view):
                continue
            if beat.target is not None and beat.target not in view.present:
                self._mark_skipped(beat.id, f"target {beat.target} is not in the room")
                continue
            self._n += 1
            self._state[beat.id] = "fired"
            self._fired_t[beat.id] = view.t
            self._fired_tick[beat.id] = self._tick_no
            self._reason[beat.id] = None
            out.append(CardFiring(f"storyboard-{self._n}", view.t, beat.card, beat.target,
                                  dict(beat.params), "storyboard"))
        return out

    def skip(self, beat_id: str) -> None:
        if beat_id not in self._state:
            raise KeyError(beat_id)
        self._mark_skipped(beat_id, "skipped by operator")

    def _dependents(self, beat_id: str) -> list[str]:
        return [b.id for b in self.storyboard.beats if b.after == beat_id]

    def _mark_skipped(self, beat_id: str, reason: str) -> None:
        self._state[beat_id] = "skipped"
        self._reason[beat_id] = reason
        for dep in self._dependents(beat_id):
            if self._state[dep] == "pending":
                self._mark_skipped(dep, f"predecessor {beat_id} skipped")

    def rerun(self, beat_id: str) -> None:
        if beat_id not in self._state:
            raise KeyError(beat_id)
        self._state[beat_id] = "pending"
        self._fired_t[beat_id] = None
        self._fired_tick.pop(beat_id, None)
        self._reason[beat_id] = None
        for dep in self._dependents(beat_id):
            if self._state[dep] == "skipped" and (self._reason[dep] or "").startswith("predecessor "):
                self.rerun(dep)

    def timeline(self) -> list[dict]:
        rows = []
        for b in self.storyboard.beats:
            row = {"id": b.id, "card": b.card, "target": b.target,
                   "state": self._state[b.id], "fired_t": self._fired_t[b.id]}
            if self._reason[b.id]:
                row["reason"] = self._reason[b.id]
            rows.append(row)
        return rows
