"""What the robot's camera sees: people by anonymous track, objects, and the bbox_3d message.

Pure functions over a StubWorld and a RobotBody. Output never carries a person id, name, mood or
persona: only what a camera shows (where, posture, moving, what is in hand, which bed). Track ids
(t1, t2, ...) are handed out in the order people are first seen in a run and never change.
"""
from __future__ import annotations

import math
from typing import Optional

from .oracle import bearing_deg, Pose
from .robot_body import RobotBody

# (size x, size y, size z) of the person's bounding box, in the person's own frame
BOX_STANDING = (0.5, 0.5, 1.7)
BOX_SEATED = (0.6, 0.6, 1.2)
BOX_LYING = (1.8, 0.6, 0.5)


class Tracks:
    """person id <-> anonymous track id for one run. Sim-side only: the robot sees the track."""

    def __init__(self):
        self._by_person: dict[str, str] = {}
        self._by_track: dict[str, str] = {}

    def track(self, person: str) -> str:
        t = self._by_person.get(person)
        if t is None:
            t = self._by_person[person] = f"t{len(self._by_person) + 1}"
            self._by_track[t] = person
        return t

    def person(self, track: str) -> Optional[str]:
        return self._by_track.get(track)


def _in_view(world, body: RobotBody, x: float, y: float) -> bool:
    cfg = world.loaded.scenario.robot
    d = math.hypot(x - body.x, y - body.y)
    if d > cfg.range_m:
        return False
    if d > 1e-6 and abs(bearing_deg(Pose(*body.pose), x, y)) > cfg.fov_deg / 2:
        return False
    return world._los((body.x, body.y), (x, y))


def _visible(world, body: RobotBody):
    out = []
    for pid, p in world.people.items():
        if p.posture != "gone" and _in_view(world, body, p.x, p.y):
            out.append((pid, p))
    return out


def _holding(world, pid: str) -> Optional[str]:
    for it in world.objects.items.values():
        if it.holder == pid:
            return it.kind
    return None


def _moving(p) -> bool:
    return bool(p.plan) and p.plan[0].kind == "walk"


def perceive_people(world, body: RobotBody) -> list[dict]:
    out = []
    for pid, p in _visible(world, body):
        out.append({"track": world.tracks.track(pid), "x": round(p.x, 3), "y": round(p.y, 3),
                    "posture": p.posture, "moving": _moving(p), "holding": _holding(world, pid),
                    "bed": p.bed})
    return out


def perceive_objects(world, body: RobotBody) -> dict:
    items = [{"id": it.id, "kind": it.kind, "label": it.label, "x": round(it.x, 3), "y": round(it.y, 3),
              "z": round(it.z, 3)}
             for it in world.objects.items.values()
             if it.holder is None and _in_view(world, body, it.x, it.y)]
    held = [{"id": it.id, "kind": it.kind, "label": it.label}
            for it in world.objects.items.values() if it.holder == world_robot_id(world)]
    occupied = {p.bed for p in world.people.values() if p.bed and p.posture == "lying"}
    beds = [{"id": b.id, "x": round(b.x, 3), "y": round(b.y, 3), "occupied": b.id in occupied}
            for b in world.site.beds.values() if _in_view(world, body, b.x, b.y)]
    fires = [{"id": f.id, "x": round(f.x, 3), "y": round(f.y, 3), "size": round(f.size, 3)}
             for f in world.objects.fires.values() if not f.out and _in_view(world, body, f.x, f.y)]
    lights = []
    for b in world.objects.bells.values():
        bed = world.site.beds.get(b.bed_id)
        if b.active and bed is not None and _in_view(world, body, bed.x, bed.y):
            lights.append({"bed": b.bed_id, "x": round(bed.x, 3), "y": round(bed.y, 3)})
    taken = {q.seat for q in world.people.values() if q.posture != "gone" and q.seat}
    taken |= {q.claim for q in world.people.values() if q.posture != "gone" and getattr(q, "claim", None)}
    seats = [{"id": c.id, "x": round(c.x, 3), "y": round(c.y, 3), "free": c.id not in taken}
             for c in world.site.seats.values() if not c.staff_only and _in_view(world, body, c.x, c.y)]
    return {"items": items, "held": held, "beds": beds, "fires": fires, "call_lights": lights, "seats": seats}


def world_robot_id(world) -> str:
    from .stub_world import ROBOT_ID
    return ROBOT_ID


def bbox_3d(world, body: RobotBody, t: float) -> dict:
    """vision_msgs/Detection3DArray-shaped JSON (published as std_msgs/String by the stub)."""
    dets = []
    for pid, p in _visible(world, body):
        size = BOX_LYING if p.posture == "lying" else BOX_SEATED if p.posture == "seated" else BOX_STANDING
        dets.append({
            "id": world.tracks.track(pid),
            "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": round(p.x, 3), "y": round(p.y, 3), "z": round(size[2] / 2, 3),
                                "yaw": round(p.yaw, 3)},
                     "size": {"x": size[0], "y": size[1], "z": size[2]}},
        })
    return {"header": {"stamp": t, "frame_id": "map"}, "detections": dets}
