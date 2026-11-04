"""Reach-zone geometry for item hand-overs (spec 9). Pure Python: no ROS, no world state."""
from __future__ import annotations

import math
from typing import Optional

GIVER_HORIZONTAL_M = 0.8
LATERAL_M = 0.4  # how far to either side of a person's facing a hand-over point may be

# (forward min, forward max, z min, z max)
STANDING_ZONE = (0.3, 0.7, 0.8, 1.4)
SEATED_ZONE = (0.3, 0.6, 0.6, 1.1)
BED_SITTING_RADIUS_M = 0.6
BED_SITTING_Z = (0.75, 1.0)
LYING_RADIUS_M = 0.4
LYING_Z = (0.0, 1.0)


def giver_reach_ok(giver_xy: tuple[float, float], reach: tuple[float, float],
                   place: tuple[float, float, float]) -> bool:
    """The giver can put something at place: within 0.8 m horizontally and z inside reach=(low, high)."""
    return (math.hypot(place[0] - giver_xy[0], place[1] - giver_xy[1]) <= GIVER_HORIZONTAL_M
            and reach[0] <= place[2] <= reach[1])


def _in_front(person_xy, yaw, place, zone) -> bool:
    fmin, fmax, zmin, zmax = zone
    dx, dy = place[0] - person_xy[0], place[1] - person_xy[1]
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return fmin <= fwd <= fmax and abs(lat) <= LATERAL_M and zmin <= place[2] <= zmax


def receiver_zone_ok(posture: str, in_bed: bool, sitting_up: bool, person_xy: tuple[float, float],
                     person_yaw: float, bed: Optional[object], place: tuple[float, float, float],
                     *, item_kind: Optional[str] = None) -> bool:
    """True when place lies in the receiver's reach zone.

    standing: 0.3-0.7 m in front, z 0.8-1.4. seated: 0.3-0.6 m, z 0.6-1.1.
    Sitting up in bed: within 0.6 m of the torso, z 0.75-1.0 (the over-bed table zone).
    Lying flat: within 0.4 m of the shoulder, and a tray there never counts (cannot eat lying flat).
    bed is accepted for callers that have one; the person's xy is the torso position in bed.
    """
    if in_bed and posture == "lying":
        d = math.hypot(place[0] - person_xy[0], place[1] - person_xy[1])
        if sitting_up:
            return d <= BED_SITTING_RADIUS_M and BED_SITTING_Z[0] <= place[2] <= BED_SITTING_Z[1]
        if item_kind == "tray":
            return False
        return d <= LYING_RADIUS_M and LYING_Z[0] <= place[2] <= LYING_Z[1]
    if posture in ("standing", "walking"):
        return _in_front(person_xy, person_yaw, place, STANDING_ZONE)
    if posture == "seated":
        return _in_front(person_xy, person_yaw, place, SEATED_ZONE)
    return False
