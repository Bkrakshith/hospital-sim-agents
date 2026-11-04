"""Which voice each speaker gets: deterministic per person per run, matched to sex and age band, and no two people in a
scene share a voice while the installed pool has one left."""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from ..isaac_host.characters import age_band
from .catalogue import ANNOUNCER_VOICE, DEFAULT_ROBOT_VOICE, VOICES, Voice


def _rank(seed: object, pid: str, key: str) -> str:
    return hashlib.sha1(f"{seed}|{pid}|{key}".encode()).hexdigest()


def usable(models: set) -> list[Voice]:
    return [v for v in VOICES.values() if v.model in models]


def robot_voice(name: Optional[str], models: set) -> Optional[Voice]:
    """The robot's voice by key; its default when it names none or one that is not installed; any installed voice as a
    last resort. None only when no model is installed."""
    for key in (name, DEFAULT_ROBOT_VOICE):
        v = VOICES.get(key or "")
        if v is not None and v.model in models:
            return v
    pool = usable(models)
    return pool[0] if pool else None


def announcer_voice(models: set, robot: Optional[Voice] = None) -> Optional[Voice]:
    """The PA / non-person voice: its own speaker when installed, else any installed voice but the robot's."""
    v = VOICES.get(ANNOUNCER_VOICE)
    if v is not None and v.model in models:
        return v
    pool = [x for x in usable(models) if robot is None or x.key != robot.key]
    return pool[0] if pool else robot


def assign_voices(people: Iterable[tuple], models: set, seed: object = 0, reserved: Iterable[str] = ()) -> dict:
    """people: (person_id, sex, age_years) with sex "female" / "male" / anything else for unknown and age None for
    unknown. Returns {person_id: Voice}; empty when no model is installed. ``reserved`` voice keys (the robot's) are left
    out of the pool unless nothing else is left."""
    pool = usable(models)
    skip = set(reserved)
    if any(v.key not in skip for v in pool):
        pool = [v for v in pool if v.key not in skip]
    if not pool:
        return {}
    used: dict[str, int] = {}
    out: dict[str, Voice] = {}
    for pid, sex, age in sorted(people, key=lambda p: str(p[0])):
        sex = (sex or "").lower()
        if sex not in ("female", "male"):  # unknown: a stable pick from the id
            sex = "female" if int(_rank(seed, str(pid), "sex")[:8], 16) % 2 == 0 else "male"
        band = age_band(int(age)) if age is not None else "adult"
        cands = [v for v in pool if v.sex == sex] or pool  # no voice of that sex installed: any voice

        def order(v: Voice):
            ages = v.ages
            return (used.get(v.key, 0), ages.index(band) if band in ages else len(ages), _rank(seed, str(pid), v.key))

        pick = min(cands, key=order)
        used[pick.key] = used.get(pick.key, 0) + 1
        out[str(pid)] = pick
    return out
