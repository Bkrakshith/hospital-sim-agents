"""Audio routing: what a listener (a person or the robot) hears, given distance, walls and hearing loss.

Each wall between the source and the listener costs WALL_LOSS_DB (up to MAX_WALLS of them); a doorway is a gap in
the wall, so it costs nothing. The PA and the fire alarm are heard everywhere at a fixed level."""
from __future__ import annotations

import math
from typing import Callable, Optional

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
WALL_LOSS_DB = 15.0  # one wall between source and listener: talk a room away is lost, a loud voice or cough is faint
MAX_WALLS = 3
WALL_END_SKIP_M = 0.2  # a wall this close to either end (someone standing against it) is not between them
WALL_STEP_M = 0.05

_FIXED = {"pa": PA_RECEIVED_DB, "alarm": ALARM_RECEIVED_DB}


def received_db(loudness: float, distance: float) -> float:
    return loudness - 20.0 * math.log10(max(distance, 1.0))


def count_walls(sight, grid, a, b) -> int:
    """How many walls the segment a-b crosses: runs of wall cells in ``sight`` (a bool array over ``grid``'s cells,
    doorways already gaps). The first and last WALL_END_SKIP_M are not looked at."""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    if d <= 2 * WALL_END_SKIP_M:
        return 0
    n = int(math.ceil((d - 2 * WALL_END_SKIP_M) / WALL_STEP_M))
    walls, inside, last = 0, False, None
    for k in range(n + 1):
        f = (WALL_END_SKIP_M + k * (d - 2 * WALL_END_SKIP_M) / n) / d
        cell = grid.to_cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        if cell == last:
            continue
        last = cell
        wall = grid.in_grid(*cell) and bool(sight[cell[0], cell[1]])
        if wall and not inside:
            walls += 1
        inside = wall
    return walls


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
    walls_between: Optional[Callable[[tuple, tuple], int]] = None,
) -> Optional[Heard]:
    """What the listener hears of ev, or None. walls_between((x, y), (x, y)) -> the number of walls between two
    points (``count_walls`` over the world's wall cells); without it, sound carries as if there were no walls."""
    if ev.source_id == listener_id:
        return None
    threshold = BASE_THRESHOLD_DB + hearing_loss_db
    if ev.source_kind in _FIXED:
        level = float(_FIXED[ev.source_kind])
    else:
        level = received_db(ev.loudness_db, math.hypot(ev.x - listener.x, ev.y - listener.y))
        if level >= threshold and walls_between is not None:
            level -= WALL_LOSS_DB * min(MAX_WALLS, int(walls_between((ev.x, ev.y), (listener.x, listener.y))))
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
