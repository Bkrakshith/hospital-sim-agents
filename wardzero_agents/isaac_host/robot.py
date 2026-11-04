"""The Isaac robot's logic that needs no Kit: the obstacle cylinder's throttle, the arm poses and how targets ease
towards them, and the base pose read from a transform. ``robot_isaac`` does the stage work with these."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Optional

from ..robots.bbox_adapter import BboxAdapter, detections_from_msg, nearest_person_resolver
from ..robots.sensors import SensorSpec, catalogue_entry
from ..ros import topics

PROXY_MOVE_M = 0.2  # S1b: moving the people's obstacle costs ~40 ms on that frame, so move it per 0.2 m ...
PROXY_EVERY_S = 0.5  # ... or per half second while the base is creeping
PROXY_MIN_STEP_M = 0.02  # a base that has moved less than this is standing still

# G1 arm poses as joint targets (rad). Zero is arms hanging. "offer": the right arm held out in front at chest height
# with the forearm up, the hand-over gesture.
G1_ARM_POSES = {
    "rest": {},
    "offer": {"right_shoulder_pitch_joint": -1.0, "right_shoulder_roll_joint": -0.15, "right_elbow_joint": 1.0,
              "left_shoulder_pitch_joint": 0.0},
}
ARM_RATE_RAD_S = 1.5
EMPTY_AFTER_S = 0.5  # sim seconds without a message to the robot before it is told its view is empty (then every 0.5 s)


class ProxyMover:
    """When to move the obstacle cylinder that stands in for the robot (it is registered with the people's NavMesh
    avoidance; the robot itself is not, S1b). ``update`` returns the new (x, y), or None to leave it."""

    def __init__(self, x: float, y: float, move_m: float = PROXY_MOVE_M, every_s: float = PROXY_EVERY_S,
                 t0: float = 0.0):
        self.x, self.y = float(x), float(y)
        self.move_m, self.every_s = move_m, every_s
        self._t = float(t0)  # when the cylinder last moved

    def update(self, x: float, y: float, t: float) -> Optional[tuple]:
        d = math.hypot(x - self.x, y - self.y)
        if d >= self.move_m or (d >= PROXY_MIN_STEP_M and t - self._t >= self.every_s):
            self.x, self.y, self._t = float(x), float(y), t
            return (self.x, self.y)
        return None


def arm_pose_name(world) -> str:
    """The pose the world calls for: the hand held out while the robot has an open hand-over offer."""
    offers = getattr(world, "_offers", {}) or {}
    return "offer" if any(o.get("giver") == "g1" for o in offers.values()) else "rest"


class ArmTargets:
    """Joint targets that ease towards a named pose at ``rate`` rad/s per joint. ``step`` returns the targets to write
    ({joint: rad}), or None when nothing changed (a PhysX write every frame is not free)."""

    def __init__(self, poses: dict, rate: float = ARM_RATE_RAD_S):
        self.poses, self.rate = poses, rate
        self.joints = sorted({j for pose in poses.values() for j in pose})
        self.q = {j: 0.0 for j in self.joints}

    def step(self, name: str, dt: float) -> Optional[dict]:
        goal = self.poses[name]
        changed = False
        for j in self.joints:
            want = goal.get(j, 0.0)
            diff = want - self.q[j]
            if abs(diff) < 1e-6:
                continue
            self.q[j] += max(-self.rate * dt, min(self.rate * dt, diff))
            changed = True
        return dict(self.q) if changed else None


def pose_from_matrix(m) -> tuple[float, float, float]:
    """(x, y, yaw) of a row-vector 4x4 world transform (USD's convention; x forward)."""
    return (float(m[3][0]), float(m[3][1]), math.atan2(float(m[0][1]), float(m[0][0])))


def camera_resolution(sensor: SensorSpec) -> tuple[int, int]:
    """(width, height) for the sensor's render product: its params, else its catalogue row, else 640x480."""
    res = sensor.params.get("resolution")
    if not res and sensor.model:
        try:
            res = catalogue_entry(sensor.model).get("resolution")
        except KeyError:
            res = None
    return (int(res[0]), int(res[1])) if res else (640, 480)


class DetectionLink:
    """Isaac's raw camera boxes (private domain) to the robot's bbox_3d (the robot's bridge).

    The boxes are matched to the world's people for their anonymous tracks (``robots.bbox_adapter``); the robot gets
    only classes, boxes and tracks. The camera helper may publish nothing when no box is in view, so the robot is told
    ``detections: []`` when the camera has been quiet for ``EMPTY_AFTER_S``."""

    def __init__(self, world, ns: str, private, robot, msgs=None, parse=detections_from_msg):
        if msgs is None:
            from std_msgs.msg import String
            from vision_msgs.msg import Detection3DArray

            from ..ros.rosutil import QOS
            msgs = SimpleNamespace(Detection3DArray=Detection3DArray, String=String, QOS=QOS)
        self.world, self.robot = world, robot
        self.topic = topics.robot_topics(ns)["bbox_3d"]
        self.adapter = BboxAdapter(nearest_person_resolver(self._positions, world.tracks.track))
        self._last = 0.0  # sim time of the last message published
        self.raw_messages = 0  # raw messages and boxes seen (diagnostics)
        self.raw_boxes = 0
        self._t = 0.0
        raw = topics.raw_detection_topics(ns)
        private.subscribe(msgs.Detection3DArray, raw["bbox_3d"], self._on_boxes, msgs.QOS, convert=parse)
        private.subscribe(msgs.String, raw["labels"], self.adapter.set_labels, msgs.QOS, convert=lambda m: m.data)

    def _positions(self) -> dict:
        return {pid: (p.x, p.y) for pid, p in self.world.people.items() if p.posture != "gone"}

    def _publish(self, msg: dict) -> None:
        self.robot.publish_text(self.topic, json.dumps(msg))
        self._last = self._t

    def _on_boxes(self, parsed) -> None:
        stamp, dets = parsed
        self.raw_messages += 1
        self.raw_boxes += len(dets)
        self._publish(self.adapter.convert(stamp, dets))

    def update(self, t: float) -> None:
        self._t = t
        if t - self._last >= EMPTY_AFTER_S:
            self._publish({"header": {"stamp": t, "frame_id": self.adapter.frame_id}, "detections": []})
