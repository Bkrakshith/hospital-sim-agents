"""Placing characters on the hospital map, and their memory and trust."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from ..config.presets import CharacterSpec, MoodPreset, RolePreset
from .project import Project, ProjectError

SEAT_SNAP_M = 0.3
_CID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def memory_key(project_id: str, character_id: str) -> str:
    return f"studio:{project_id}:{character_id}"


def new_character_id(existing: list[CharacterSpec]) -> str:
    taken = {c.id for c in existing}
    n = 1
    while f"C{n}" in taken:
        n += 1
    return f"C{n}"


def check_character(c: CharacterSpec, site, roles: dict[str, RolePreset], moods: dict[str, MoodPreset],
                    others: list[CharacterSpec]) -> CharacterSpec:
    """Validate one character against the site and the other characters. Returns it (a bed patient is
    snapped to the bed); raises ProjectError with a message a person can act on."""
    if not _CID.match(c.id):
        raise ProjectError("A character id is a letter followed by up to 15 letters, digits or underscores.")
    if any(o.id == c.id for o in others):
        raise ProjectError(f"There is already a character '{c.id}'.")
    if not c.name.strip() or len(c.name) > 60:
        raise ProjectError("A character needs a name of 1 to 60 characters.")
    if c.role not in roles:
        raise ProjectError(f"Unknown role '{c.role}'.")
    if c.mood not in moods:
        raise ProjectError(f"Unknown mood '{c.mood}'.")
    for label, text in (("role_prompt", c.role_prompt), ("mood_prompt", c.mood_prompt)):
        if text is not None and len(text) > 4000:
            raise ProjectError(f"'{label}' is too long (4000 characters at most).")
    if not all(math.isfinite(v) for v in (c.x, c.y, c.yaw)):
        raise ProjectError("The position must be numbers.")
    occupied = [o.bed for o in others if o.bed]
    if c.bed:
        bed = site.beds.get(c.bed)
        if bed is None:
            raise ProjectError(f"There is no bed '{c.bed}' on this map.")
        if c.bed in occupied:
            raise ProjectError(f"Bed {c.bed} already has a patient.")
        # the bed is solid in the occupancy grid, so a bed patient takes the bed's own pose
        return c.model_copy(update={"x": bed.x, "y": bed.y, "yaw": bed.yaw, "room": bed.room})
    # a seat is solid in the occupancy grid too; a click near a free seat sits the person in it
    seat = min(site.seats.values(), key=lambda s: math.hypot(s.x - c.x, s.y - c.y), default=None)
    if seat is not None and math.hypot(seat.x - c.x, seat.y - c.y) <= SEAT_SNAP_M:
        if any(math.hypot(seat.x - o.x, seat.y - o.y) <= SEAT_SNAP_M and not o.bed for o in others):
            raise ProjectError(f"Seat {seat.id} is already taken.")
        return c.model_copy(update={"x": seat.x, "y": seat.y, "yaw": seat.yaw,
                                    "room": site.room_at(seat.x, seat.y) or c.room or "outdoors"})
    problem = site.check_place(c.x, c.y, occupied_beds=occupied)
    if problem:
        raise ProjectError(f"{c.name or c.id} cannot be placed there: {problem}.")
    room = site.room_at(c.x, c.y)
    return c.model_copy(update={"room": room or c.room or "outdoors"})


def check_all(project: Project, site, roles, moods) -> Project:
    """Validate every character of a project (and the robot start); returns the project with beds snapped."""
    done: list[CharacterSpec] = []
    for c in project.characters:
        done.append(check_character(c, site, roles, moods, done))
    if project.robot_start is not None:
        problem = site.check_place(project.robot_start.x, project.robot_start.y)
        if problem:
            raise ProjectError(f"The robot cannot start there: {problem}.")
    return project.model_copy(update={"characters": done})


def memory_rows(project: Project, store, ltm_path: Path, roles: dict[str, RolePreset]) -> list[dict]:
    """Per character: trust, stage and what trust has been through, from the campaign store."""
    from ..campaign import robot_key, role_defaults
    robot_id = robot_key(project.robot_profile)  # trust is kept per robot: this is the selected robot's
    out = []
    for c in project.characters:
        key = memory_key(project.id, c.id)
        s = store.get(key, robot_id, role_defaults(roles, c.role))
        out.append({"id": c.id, "name": c.name, "memory_key": key, "trust": round(s.trust, 4), "stage": s.stage,
                    "runs": s.runs, "violations": s.violations, "repairs": s.repairs})
    return out


def reset_memory(project: Project, which, data: Path, embedder_name: str) -> list[str]:
    """Reset long-term memory and trust of some characters (or "all") to the start. Prompts are part of
    the project, not of memory, so they stay. Raises ProjectError for an unknown id, RunActive when a
    run is on. Returns the character ids reset."""
    from ..kb.embed import get_embedder
    from ..campaign.trust import TrustStore
    from ..memory import memory_path
    from ..memory.snapshots import guarded_reset
    ids = [c.id for c in project.characters]
    chosen = ids if which == "all" else which
    if not isinstance(chosen, list) or not all(isinstance(i, str) for i in chosen):
        raise ProjectError("'characters' must be a list of character ids, or \"all\".")
    unknown = [i for i in chosen if i not in ids]
    if unknown:
        raise ProjectError(f"No such character: {', '.join(unknown)}.")
    keys = [memory_key(project.id, i) for i in chosen]
    TrustStore(memory_path(data))  # makes sure the db and its trust table exist
    try:
        guarded_reset(data, keys, {k: [] for k in keys}, get_embedder(embedder_name))
    except ValueError as e:  # embedder mismatch
        raise ProjectError(f"Cannot reset: {e}") from None
    return list(chosen)
