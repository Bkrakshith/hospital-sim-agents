"""World oracle: what a viewer can see. Pure geometry, no simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from wardzero_agents.config.models import Sign
from wardzero_agents.contracts import Seen


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float  # radians, counter-clockwise from +x


@dataclass(frozen=True)
class Entity:
    kind: str  # "person" | "robot" | "seat"
    id: str
    x: float
    y: float
    detail: Optional[str] = None


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def bearing_deg(viewer: Pose, x: float, y: float) -> float:
    """Signed bearing of (x, y) from the viewer's heading; left (counter-clockwise) is positive."""
    if math.hypot(x - viewer.x, y - viewer.y) < 1e-6:
        return 0.0
    return wrap_deg(math.degrees(math.atan2(y - viewer.y, x - viewer.x) - viewer.yaw))


def _always(a, b) -> bool:
    return True


def visible(
    viewer: Pose,
    entities: list[Entity],
    signs: list[Sign],
    fov_deg: float = 120.0,
    range_m: float = 12.0,
    los: Callable[[tuple, tuple], bool] = _always,
) -> list[Seen]:
    out: list[Seen] = []
    half = fov_deg / 2.0

    def in_view(x: float, y: float, limit: float) -> Optional[tuple[float, float]]:
        d = math.hypot(x - viewer.x, y - viewer.y)
        if d > limit:
            return None
        b = bearing_deg(viewer, x, y)
        if abs(b) > half:
            return None
        if not los((viewer.x, viewer.y), (x, y)):
            return None
        return d, b

    for e in entities:
        r = in_view(e.x, e.y, range_m)
        if r is not None:
            out.append(Seen(kind=e.kind, id=e.id, distance=r[0], bearing_deg=r[1], detail=e.detail))

    for s in signs:
        # A sign is readable only from its front: the viewer must lie within 90 degrees of facing_yaw.
        to_viewer = math.degrees(math.atan2(viewer.y - s.y, viewer.x - s.x))
        if abs(wrap_deg(to_viewer - math.degrees(s.facing_yaw))) > 90.0:
            continue
        r = in_view(s.x, s.y, min(s.read_distance_m, range_m))
        if r is not None:
            out.append(Seen(kind="sign", id=s.id, distance=r[0], bearing_deg=r[1], detail=s.text))
    return out
