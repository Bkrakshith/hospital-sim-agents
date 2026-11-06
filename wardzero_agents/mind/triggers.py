"""When does a person stop and think? A cheap gate in front of the LLM decision."""
from __future__ import annotations

from typing import Optional

from ..contracts import Nudge, Observation, Status

URGENT = {"addressed_speech", "nudge"}
REASONS = ("addressed_speech", "speech_nearby", "robot_near", "new_sign", "new_person",
           "intent_done", "intent_failed", "cough_nearby", "patience_timer", "nudge", "start")


class TriggerGate:
    def __init__(self, min_interval_s: float = 4.0, robot_near_m: float = 2.0,
                 patience_check_s: float = 60.0):
        self.min_interval_s = min_interval_s
        self.robot_near_m = robot_near_m
        self.patience_check_s = patience_check_s
        self._last_decision: Optional[float] = None
        self._first_check: Optional[float] = None
        self._reported_ends: set[tuple[str, str]] = set()
        self._candidate_end: Optional[tuple[str, str]] = None
        # A person who has seen nothing new would otherwise stand idle until the patience timer: everyone takes
        # stock once at the start. It stays a reason until a decision has been started on it.
        self._started = False

    def _robot_near(self, obs: Optional[Observation]) -> bool:
        return obs is not None and any(
            s.kind == "robot" and s.distance <= self.robot_near_m for s in obs.visible)

    def check(self, t: float, obs: Observation, status: Status, nudges: list[Nudge],
              last_obs: Optional[Observation]) -> list[str]:
        """Reasons to decide now, or [] if there are none or it is too soon since the last decision.

        last_obs is the observation the last decision was made on, so something new stays new
        until a decision has seen it.
        """
        if self._first_check is None:
            self._first_check = t
        found: set[str] = set()
        for h in obs.heard:
            if h.source_kind == "cough":
                found.add("cough_nearby")
            elif h.source_kind in ("robot", "person", "pa"):
                found.add("addressed_speech" if h.addressed_to_me else "speech_nearby")
        prev = last_obs.visible if last_obs is not None else []
        prev_ids = {(s.kind, s.id) for s in prev}
        for s in obs.visible:
            if s.kind == "person" and (s.kind, s.id) not in prev_ids:
                found.add("new_person")
            elif s.kind == "sign" and (s.kind, s.id) not in prev_ids:
                found.add("new_sign")
        if self._robot_near(obs) and not self._robot_near(last_obs):
            found.add("robot_near")
        self._candidate_end = None
        if status.state in ("done", "failed"):
            key = (status.intent_id or str(status.intent_kind), status.state)
            if key not in self._reported_ends:
                found.add("intent_done" if status.state == "done" else "intent_failed")
                self._candidate_end = key
        if nudges:
            found.add("nudge")
        ref = self._last_decision if self._last_decision is not None else self._first_check
        if t - ref >= self.patience_check_s:
            found.add("patience_timer")
        if not self._started:
            found.add("start")

        if (self._last_decision is not None and t - self._last_decision < self.min_interval_s
                and not (found & URGENT)):
            self._candidate_end = None
            return []
        return [r for r in REASONS if r in found]

    def mark(self, t: float) -> None:
        """Record that a decision was started at t, on the reasons the last check returned."""
        self._last_decision = t
        self._started = True
        if self._candidate_end is not None:
            self._reported_ends.add(self._candidate_end)
            self._candidate_end = None
