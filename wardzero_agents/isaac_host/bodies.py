"""IsaacBodies: Isaac Sim owns the bodies; every frame their poses are written into the StubWorld.

A pose source is a callable returning ``(x, y, yaw)`` in the site frame (or None while it has no
pose yet). I2 registers one per IRA character (``agent.get_world_translation`` and rotation), I4 one
for the robot base. In I1 both sets are empty, so the StubWorld still moves everyone kinematically.
Until I2 hands an intent to the Behavior System, a registered person keeps the StubWorld's plan and
state; only the pose comes from Isaac.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

from ..world.bodies import Bodies

if TYPE_CHECKING:
    from ..world.stub_world import StubWorld

Pose3 = tuple[float, float, float]
PoseSource = Callable[[], Optional[Pose3]]


def _finite(pose) -> bool:
    return pose is not None and len(pose) == 3 and all(math.isfinite(float(v)) for v in pose)


class IsaacBodies(Bodies):
    name = "isaac"

    def __init__(self, people: Optional[dict[str, PoseSource]] = None, robot: Optional[PoseSource] = None):
        self.people: dict[str, PoseSource] = dict(people or {})
        self.robot: Optional[PoseSource] = robot

    def add_person(self, pid: str, source: PoseSource) -> None:
        self.people[pid] = source

    def set_robot(self, source: Optional[PoseSource]) -> None:
        self.robot = source

    def step(self, world: "StubWorld", t: float) -> None:
        world.step(t)  # the logic (objects, tasks, contacts); bodies Isaac does not own yet stay kinematic
        self.read_back(world)

    def read_back(self, world: "StubWorld") -> None:
        for pid, source in self.people.items():
            p = world.people.get(pid)
            pose = source()
            if p is None or not _finite(pose):
                continue
            p.x, p.y, p.yaw = (float(v) for v in pose)
        if self.robot is not None:
            pose = self.robot()
            if _finite(pose):
                r = world.robot
                r.x, r.y, r.yaw = (float(v) for v in pose)
