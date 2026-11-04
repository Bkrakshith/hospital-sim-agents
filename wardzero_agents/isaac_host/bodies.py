"""IsaacBodies: Isaac Sim owns the bodies; every frame their poses are written into the StubWorld.

A pose source is a callable returning ``(x, y, yaw)`` in the site frame (or None while it has no
pose yet). I2 registers one per IRA character (``agent.get_world_translation`` and rotation), I4 one
for the robot base. In I1 both sets are empty, so the StubWorld still moves everyone kinematically.
A person with a controller (``add_controller``, I2) is body-driven: the StubWorld validates their intents and
builds the plan, the controller carries each phase out with the Behavior System and ends it in the world
(``wardzero_agents.isaac_host.people``). Anyone else keeps the StubWorld's plan and state, and if they have a pose
source only the pose comes from Isaac.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

from ..world.bodies import Bodies
from .people import PENDING

if TYPE_CHECKING:
    from ..world.stub_world import StubWorld
    from .people import PersonController

Pose3 = tuple[float, float, float]
PoseSource = Callable[[], Optional[Pose3]]


def _finite(pose) -> bool:
    return pose is not None and len(pose) == 3 and all(math.isfinite(float(v)) for v in pose)


class IsaacBodies(Bodies):
    name = "isaac"

    def __init__(self, people: Optional[dict[str, PoseSource]] = None, robot: Optional[PoseSource] = None,
                 spawner: Optional[Callable[["StubWorld", str], Optional["PersonController"]]] = None):
        self.people: dict[str, PoseSource] = dict(people or {})
        self.robot: Optional[PoseSource] = robot
        self.controllers: dict[str, "PersonController"] = {}
        self.spawner = spawner  # makes a controller for a person who joined after the start (None: stays kinematic)
        self._unserved: set[str] = set()
        self._last_t: Optional[float] = None

    def add_person(self, pid: str, source: PoseSource) -> None:
        self.people[pid] = source

    def set_robot(self, source: Optional[PoseSource]) -> None:
        self.robot = source

    def add_controller(self, world: "StubWorld", ctrl: "PersonController") -> None:
        """Make a person's body an Isaac character: the world stops walking them, the controller does."""
        self.controllers[ctrl.pid] = ctrl
        self.people[ctrl.pid] = ctrl.api.pose
        world.body_driven.add(ctrl.pid)

    def on_intent(self, world: "StubWorld", intent) -> None:
        ctrl = self.controllers.get(intent.person)
        p = world.people.get(intent.person)
        if ctrl is not None and p is not None and p.state != "failed":
            ctrl.on_intent(world, intent)

    def step(self, world: "StubWorld", t: float) -> None:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self._join(world)
        world.step(t)  # the logic (objects, tasks, contacts); body-driven people are not walked here
        for pid, ctrl in list(self.controllers.items()):
            ctrl.update(world, dt)
        self.read_back(world)

    def _join(self, world: "StubWorld") -> None:
        """People who joined after the start (a spawn) get a character if there is a spawner."""
        for pid in world.people:
            if pid in self.controllers or pid in self._unserved or pid in self.people:
                continue
            ctrl = self.spawner(world, pid) if self.spawner is not None else None
            if ctrl is PENDING:
                continue  # being made: ask again next frame
            if ctrl is None:
                self._unserved.add(pid)  # no body for them: the StubWorld keeps moving them
            else:
                self.add_controller(world, ctrl)

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
