"""Reflexes: coughing, fidgeting and turning towards a voice. No LLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from wardzero_agents.config.models import PersonInstance
from wardzero_agents.contracts import Heard, Status

GAZE_MIN_GAP_S = 3.0
COUGH_DURATION_S = 1.0
FIDGET_DURATION_S = 2.0


@dataclass(frozen=True)
class ReflexAction:
    kind: Literal["cough", "fidget", "look_at"]
    target: Optional[str] = None
    duration_s: float = 0.0
    severity: Optional[str] = None


class ReflexEngine:
    def __init__(self, instance: PersonInstance, seed: int):
        self.inst = instance
        self.rng = np.random.default_rng(seed)
        self._next_cough: Optional[float] = None
        self._next_fidget: Optional[float] = None
        self._last_gaze = -1e18
        self._bout_end: Optional[float] = None
        self._bout_severity: Optional[str] = None
        self._next_bout_cough = 0.0

    def _cough_gap(self) -> float:
        return float(self.rng.exponential(3600.0 / self.inst.cough_per_hour))

    def start_bout(self, t: float, duration_s: float, severity: str) -> None:
        self._bout_end = t + duration_s
        self._bout_severity = severity
        self._next_bout_cough = t

    def tick(self, t: float, status: Status, heard: list[Heard], idle: bool) -> list[ReflexAction]:
        acts: list[ReflexAction] = []
        rate = self.inst.cough_per_hour

        if self._bout_end is not None and t >= self._bout_end:
            self._bout_end = None
            self._next_cough = None  # resume background process afresh after the bout

        if self._bout_end is not None:
            if t >= self._next_bout_cough:
                acts.append(ReflexAction("cough", None, COUGH_DURATION_S, self._bout_severity))
                self._next_bout_cough = t + float(self.rng.uniform(2.0, 4.0))
        elif rate > 0:
            if self._next_cough is None:
                self._next_cough = t + self._cough_gap()
            if t >= self._next_cough:
                acts.append(ReflexAction("cough", None, COUGH_DURATION_S, self.inst.cough_severity))
                self._next_cough += self._cough_gap()
                if self._next_cough < t:
                    self._next_cough = t + self._cough_gap()

        if status.posture == "seated" and idle:
            if self._next_fidget is None:
                self._next_fidget = t + float(self.rng.uniform(45.0, 120.0))
            elif t >= self._next_fidget:
                acts.append(ReflexAction("fidget", None, FIDGET_DURATION_S, None))
                self._next_fidget = t + float(self.rng.uniform(45.0, 120.0))
        else:
            self._next_fidget = None

        if heard and t - self._last_gaze >= GAZE_MIN_GAP_S:
            acts.append(ReflexAction("look_at", heard[0].source_id, 1.5, None))
            self._last_gaze = t
        return acts
