"""Audio routing: what a listener (a person or the robot) hears, given distance and hearing loss."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from wardzero_agents.contracts import AudioEvent, Heard
from wardzero_agents.world.oracle import Pose

SPEECH_DB = 60
ROBOT_SPEECH_DB = 72
COUGH_DB = {"mild": 62, "moderate": 68, "severe": 74}
PA_RECEIVED_DB = 62
ALARM_RECEIVED_DB = 85
BASE_THRESHOLD_DB = 30
PARTIAL_MARGIN_DB = 6
PARTIAL_KEEP = 0.6

_FIXED = {"pa": PA_RECEIVED_DB, "alarm": ALARM_RECEIVED_DB}


def received_db(loudness: float, distance: float) -> float:
    return loudness - 20.0 * math.log10(max(distance, 1.0))


def _degrade(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    if not words:
        return text
    keep = [bool(rng.random() < PARTIAL_KEEP) for _ in words]
    if not any(keep):
        keep[int(rng.integers(len(words)))] = True
    return " ".join(w if k else "…" for w, k in zip(words, keep))


def route(
    ev: AudioEvent,
    listener: Pose,
    listener_id: str,
    hearing_loss_db: float,
    rng: np.random.Generator,
) -> Optional[Heard]:
    if ev.source_id == listener_id:
        return None
    if ev.source_kind in _FIXED:
        level = float(_FIXED[ev.source_kind])
    else:
        level = received_db(ev.loudness_db, math.hypot(ev.x - listener.x, ev.y - listener.y))
    threshold = BASE_THRESHOLD_DB + hearing_loss_db
    if level < threshold:
        return None
    partial = level < threshold + PARTIAL_MARGIN_DB
    text = None if ev.source_kind == "cough" else ev.text
    if partial and text:
        text = _degrade(text, rng)
    return Heard(
        source_kind=ev.source_kind,
        source_id=ev.source_id,
        text=text,
        received_db=level,
        partial=partial,
        addressed_to_me=ev.addressee == listener_id,
    )
