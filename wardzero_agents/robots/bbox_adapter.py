"""Isaac's camera bbox_3d to WardZero's bbox_3d. Pure Python (no rclpy, no Isaac).

Isaac's camera helper publishes a vision_msgs/Detection3DArray in the **world frame** (whatever its frame_id says)
whose ``center`` is the prim origin, the feet, not the middle of the box, and whose class is the numeric semantic id
(``semantic_labels`` maps it to a class). The robot gets the stub's shape instead (docs/robot-topics.md): the box centre
in the ``map`` frame, the class as a word, and an anonymous track ``t<n>``. The track comes from the world: each
detection is matched to the nearest person it can be (one box per person) and takes that person's track
(``world.tracks``), so ``hand_over {to: "t1"}`` means the person the camera showed. The person's id never leaves the
host: the resolver is given positions and returns a track.
"""
from __future__ import annotations

import json
import math
from typing import Callable, Iterable, Optional

PERSON = "person"
GATE_M = 0.8  # a box is a person when it is this close to where the world has them (feet to feet)

Resolver = Callable[[list], "list[Optional[str]]"]  # [(x, y)] -> [track or None]


def _yaw(q) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def nearest_person_resolver(people: Callable[[], dict], track: Callable[[str], str], gate_m: float = GATE_M) -> Resolver:
    """A resolver over ``people()`` ({person id: (x, y)}) and ``track(person id) -> "t3"``.

    Pairs are taken closest first, each person and each box once. Tracks are asked for in box order, so the first
    box seen in a run gets t1 when nobody has a track yet."""
    def resolve(points: list) -> list:
        pairs = sorted((math.hypot(px - x, py - y), i, pid)
                       for i, (x, y) in enumerate(points) for pid, (px, py) in people().items()
                       if math.hypot(px - x, py - y) <= gate_m)
        taken_box, taken_person, who = set(), set(), {}
        for _d, i, pid in pairs:
            if i in taken_box or pid in taken_person:
                continue
            taken_box.add(i)
            taken_person.add(pid)
            who[i] = pid
        return [track(who[i]) if i in who else None for i in range(len(points))]

    return resolve


class BboxAdapter:
    """Turns raw detections (``detections_from_msg``) into the robot's bbox_3d dict."""

    def __init__(self, resolver: Resolver, frame_id: str = "map"):
        self.resolver = resolver
        self.frame_id = frame_id
        self.labels: dict[str, str] = {}  # numeric class id -> class word
        self.unknown_class = 0  # boxes whose class is not a person (or has no label yet)
        self.unmatched = 0  # person boxes that matched nobody in the world (a stranger, or a ghost)

    def set_labels(self, text: str) -> None:
        """The camera helper's semantic_labels message: {"0": {"class": "person"}}. A bad message changes nothing."""
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]
        if not isinstance(data, dict):
            return
        labels = {}
        for key, val in data.items():
            cls = val.get("class") if isinstance(val, dict) else val
            if isinstance(cls, str):
                labels[str(key)] = cls
        self.labels = labels

    def _class_of(self, class_id: str) -> Optional[str]:
        cls = self.labels.get(str(class_id))
        if cls is not None and PERSON in [c.strip() for c in cls.split(",")]:
            return PERSON
        return None

    def convert(self, stamp: float, dets: Iterable[dict]) -> dict:
        dets = list(dets)
        people = [d for d in dets if self._class_of(d["class_id"]) == PERSON]
        self.unknown_class += len(dets) - len(people)
        tracks = self.resolver([(d["center"][0], d["center"][1]) for d in people])
        out = []
        for d, tr in zip(people, tracks):
            if tr is None:
                self.unmatched += 1
                continue
            (x, y, z0), (sx, sy, sz) = d["center"], d["size"]
            out.append({"id": tr, "results": [{"class_id": PERSON, "score": 1.0}],
                        "bbox": {"center": {"x": round(x, 3), "y": round(y, 3), "z": round(z0 + sz / 2.0, 3),
                                            "yaw": round(_yaw(d["orientation"]), 3)},
                                 "size": {"x": round(sx, 3), "y": round(sy, 3), "z": round(sz, 3)}}})
        return {"header": {"stamp": stamp, "frame_id": self.frame_id}, "detections": out}


def detections_from_msg(msg) -> tuple[float, list]:
    """(stamp seconds, [raw detection dicts]) from a vision_msgs/Detection3DArray (or anything shaped like one)."""
    st = msg.header.stamp
    t = float(st.sec) + float(st.nanosec) * 1e-9
    out = []
    for d in msg.detections:
        pos, ori, size = d.bbox.center.position, d.bbox.center.orientation, d.bbox.size
        cls = str(d.results[0].hypothesis.class_id) if d.results else ""
        out.append({"class_id": cls, "center": (float(pos.x), float(pos.y), float(pos.z)),
                    "orientation": (float(ori.x), float(ori.y), float(ori.z), float(ori.w)),
                    "size": (float(size.x), float(size.y), float(size.z))})
    return t, out
