"""Placing characters on the hospital map, and their memory and trust."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from ..config.presets import CharacterSpec, MoodPreset, RolePreset
from .project import Project, ProjectError

SEAT_SNAP_M = 0.3
WAITING_ROOM = "AE_Waiting"  # "Start seated" picks one of this room's seats (A1-A3, B1-B3, C1, D3)
STAND_OFF_M = 1.0  # unticking "Start seated" stands the person this far in front of the seat
_CID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def memory_key(project_id: str, character_id: str) -> str:
    return f"studio:{project_id}:{character_id}"


def new_character_id(existing: list[CharacterSpec]) -> str:
    taken = {c.id for c in existing}
    n = 1
    while f"C{n}" in taken:
        n += 1
    return f"C{n}"


def waiting_seats(site) -> list:
    """The A&E waiting-room seats a character can start seated on (staff-only seats left out)."""
    return [s for s in site.seats.values()
            if site.room_at(s.x, s.y) == WAITING_ROOM and not getattr(s, "staff_only", False)]


def _seat_taken(seat, others: list[CharacterSpec]) -> bool:
    return any(math.hypot(seat.x - o.x, seat.y - o.y) <= SEAT_SNAP_M and not o.bed for o in others)


def _waiting_seat_at(site, x: float, y: float):
    return next((s for s in waiting_seats(site) if math.hypot(s.x - x, s.y - y) <= SEAT_SNAP_M), None)


def _stand_in_front(site, seat, others: list[CharacterSpec]) -> tuple[float, float, float]:
    """A free standing spot STAND_OFF_M in front of a seat (the way it faces), else a little nearer or further,
    beside or behind it. The waiting area is open plan, so the spot may be just outside the room's outline."""
    for turn in (0.0, math.pi / 2, -math.pi / 2, math.pi):
        for dist in (STAND_OFF_M, 0.8, 1.2, 0.6):
            a = seat.yaw + turn
            x, y = round(seat.x + dist * math.cos(a), 2), round(seat.y + dist * math.sin(a), 2)
            near_seat = any(math.hypot(q.x - x, q.y - y) <= SEAT_SNAP_M for q in site.seats.values())
            crowded = any(math.hypot(o.x - x, o.y - y) < 0.5 for o in others)
            if not near_seat and not crowded and site.check_place(x, y) is None:
                return x, y, seat.yaw
    raise ProjectError(f"There is no room to stand in front of seat {seat.id}.")


def apply_start_seated(new: CharacterSpec, old: Optional[CharacterSpec], site,
                       others: list[CharacterSpec], ticked: bool = False) -> CharacterSpec:
    """Act on the "Start seated" box before the usual checks.

    Ticked (newly, or on a new character): sit them on the free A&E waiting seat nearest to where they are;
    every seat taken is an error. Unticked while still on their seat: stand them 1 m in front of it.
    Moved (dragged) while ticked: they stay seated only when dropped on a free waiting seat, unless the box
    was sent ticked with the move (ticked: the dialog), which seats them nearest to the new spot."""
    was = bool(old and old.start_seated)
    moved = old is not None and (abs(new.x - old.x) > 1e-6 or abs(new.y - old.y) > 1e-6)
    if new.start_seated and new.bed:
        raise ProjectError("A patient in a bed cannot also start seated.")
    if new.start_seated and was and moved and not ticked:
        seat = _waiting_seat_at(site, new.x, new.y)
        if seat is None or _seat_taken(seat, others):
            return new.model_copy(update={"start_seated": False})
        return new.model_copy(update={"x": seat.x, "y": seat.y, "yaw": seat.yaw, "room": WAITING_ROOM})
    if new.start_seated and not (was and not moved):
        here = _waiting_seat_at(site, new.x, new.y)
        if here is not None and not _seat_taken(here, others):
            seat = here
        else:
            free = [s for s in waiting_seats(site) if not _seat_taken(s, others)]
            if not free:
                raise ProjectError(f"All {len(waiting_seats(site))} A&E waiting seats are taken, "
                                   f"so {new.name or new.id} cannot start seated.")
            seat = min(free, key=lambda s: math.hypot(s.x - new.x, s.y - new.y))
        return new.model_copy(update={"x": seat.x, "y": seat.y, "yaw": seat.yaw, "room": WAITING_ROOM,
                                      "bed": None})
    if not new.start_seated and was and not moved:
        seat = _waiting_seat_at(site, new.x, new.y)
        if seat is not None:
            x, y, yaw = _stand_in_front(site, seat, others)
            return new.model_copy(update={"x": x, "y": y, "yaw": yaw, "room": site.room_at(x, y) or new.room})
    return new


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
    if c.start_seated:
        seat = _waiting_seat_at(site, c.x, c.y)
        if c.bed or seat is None:
            raise ProjectError(f"{c.name or c.id} starts seated, so they must be on an A&E waiting seat.")
        if _seat_taken(seat, others):
            raise ProjectError(f"Seat {seat.id} is already taken.")
    if c.bed:
        bed = site.beds.get(c.bed)
        if bed is None:
            raise ProjectError(f"There is no bed '{c.bed}' on this map.")
        if c.bed in occupied:
            raise ProjectError(f"Bed {c.bed} already has a patient.")
        # the bed is solid in the occupancy grid, so a bed patient takes the bed's own pose
        return c.model_copy(update={"x": bed.x, "y": bed.y, "yaw": bed.yaw, "room": bed.room})
    # a seat is solid in the occupancy grid too; a click near a free seat sits the person in it
    # staff-only chairs (the receptionist's, R1) are never snapped to
    seat = min((s for s in site.seats.values() if not getattr(s, "staff_only", False)),
               key=lambda s: math.hypot(s.x - c.x, s.y - c.y), default=None)
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
