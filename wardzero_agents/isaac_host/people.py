"""Intents to Behavior System calls: the minds decide, Isaac's characters carry it out.

Every decision still comes from a persona mind as an Intent. The StubWorld validates it and builds the plan (stand
up, walk, sit), exactly as in the stub. For a person whose body is an Isaac character (``world.body_driven``), the
world no longer walks the plan: ``PersonController`` hands each phase to the character through five calls
(``move_to``, ``sit``, ``idle``, ``look_at``, ``cancel``), watches the character, and tells the world when a phase has
ended (``end_phase``) or cannot be done (``fail_intent``). The character's pose is read back into the world every
frame by ``IsaacBodies``, so everything else in the world (hearing, sight, the robot's contacts) uses the real pose.

This module never imports pxr, carb or Omniverse: it talks to a small ``BehaviorApi``, which
``people_isaac.IraAgent`` implements on top of the IRA Behavior System (``IBehaviorAgent``). The unit tests
drive it with a fake.

What the Behavior System does badly, and what is done about it (from spike S1, Q2):
- crowd avoidance (a ``Dodge`` task) can pre-empt a MoveTo and leave it failed short of the goal, silently:
  a walk that ended short is re-issued, up to ``WALK_RETRIES`` times, then the intent fails with "blocked";
- a seated character who is given any new task stands up: look_at and idle are never sent to a seated person;
- the LookAround clip shows a T-pose: "looking around" is a short sweep of look_at targets;
- a sit effector is a point on the chair: the walk before a sit stops on the floor in front of the seat.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Protocol

log = logging.getLogger("wardzero.isaac.people")

WALK_RETRIES = 3  # a walk that ended short is re-issued this many times, then it fails "blocked"
ARRIVE_BODY_M = 0.4  # a walk is done when the character is this close to its goal
ARRIVE_ENDED_M = 0.8  # ... or when the task reports success this close
SEAT_APPROACH_M = 0.85  # the floor point in front of a seat, where the walk before a sit stops
SIT_MIN_S = 2.5  # a sit phase lasts at least as long as the sit animation
STAND_MIN_S = 1.5
STUCK_S = 6.0  # a running walk that has moved less than STUCK_MOVE_M in this long counts as ended short
STUCK_MOVE_M = 0.1
TIMEOUT_FACTOR = 3.0  # a walk gets this many times its nominal time, plus TIMEOUT_EXTRA_S
TIMEOUT_EXTRA_S = 20.0
REROUTE_M = 0.5  # a walk to a person who moves is re-issued when they have moved this far
SPEED_RANGE = (0.4, 1.8)
EYE_Z = 1.55
LOOK_S = 2.5
SWEEP_DEG = (60.0, -60.0, 0.0)  # "look around": left, right, back to the front
SWEEP_STEP_S = 1.3
SIT_RETRIES = 2
WALKING = ("MoveTo", "GoTo", "Dodge", "MoveAlong", "Follow")


class BehaviorApi(Protocol):
    """One character, as the controller needs it. ``people_isaac.IraAgent`` is the Isaac implementation."""

    def move_to(self, x: float, y: float) -> int: ...
    def sit(self, prop: str) -> int: ...
    def idle(self) -> int: ...
    def look_at(self, x: float, y: float, z: float, duration: float) -> int: ...
    def cancel(self, task: int) -> None: ...
    def status(self, task: int) -> str: ...  # "running" | "succeeded" | "failed" | "none"
    def pose(self) -> Optional[tuple[float, float, float]]: ...  # x, y, yaw in the site frame
    def current_action(self) -> str: ...  # the running task's name: MoveTo, Dodge, Sit, Idle, LookAt, ...
    def set_speed(self, speed: float) -> None: ...
    def hide(self) -> None: ...


PENDING = object()  # a spawner's answer: the character is being made, ask again next frame


def seat_prim_path(seat_id: str) -> str:
    return f"/World/Seats/{seat_id}"


def clamp_speed(v: float) -> float:
    return min(max(float(v), SPEED_RANGE[0]), SPEED_RANGE[1])


def seat_approach(seat) -> tuple[float, float]:
    """The point on the floor in front of a seat (seat.yaw is the way the sitter faces)."""
    return seat.x + SEAT_APPROACH_M * math.cos(seat.yaw), seat.y + SEAT_APPROACH_M * math.sin(seat.yaw)


class PersonController:
    """Drives one Isaac character from its person's plan in the StubWorld. ``update`` runs once per frame."""

    def __init__(self, pid: str, api: BehaviorApi, seat_prim: Callable[[str], str] = seat_prim_path):
        self.pid, self.api, self.seat_prim = pid, api, seat_prim
        self._phase = None  # the plan phase being carried out (compared by identity)
        self._task: Optional[int] = None
        self._goal: Optional[tuple[float, float]] = None
        self._retries = 0
        self._elapsed = 0.0
        self._timeout = 0.0
        self._anchor: Optional[tuple[float, float]] = None
        self._still = 0.0
        self._left = 0.0
        self._sweep: list[tuple[float, tuple[float, float, float]]] = []
        self.hidden = False
        self.events: list[tuple[str, str]] = []  # (kind, detail) of what the controller did, for the run log

    # ---- start ---------------------------------------------------------------------------------
    def start(self, world) -> None:
        """A person who begins seated sits down on their seat (before the run's clock starts)."""
        p = world.people[self.pid]
        if p.posture == "seated" and p.seat:
            self._task = self.api.sit(self.seat_prim(p.seat))

    # ---- intents that are not part of the plan ---------------------------------------------------
    def on_intent(self, world, intent) -> None:
        p = world.people.get(self.pid)
        if p is None or p.posture != "standing" or p.plan:
            return  # a seated person's head turn is not drawn (any new task would stand them up)
        kind = intent.kind
        target = None
        if kind == "look_at" and intent.target:
            target = world.look_target(intent.target)
        elif kind == "read_sign" and intent.target:
            target = world.look_target(intent.target)
        elif kind == "talk" and intent.addressee:
            target = world.look_target(intent.addressee)
        if target is not None:
            self._task = self.api.look_at(target[0], target[1], target[2], LOOK_S)
            self.events.append(("look_at", intent.target or intent.addressee or ""))
        elif kind == "fidget":
            self.look_around(p.x, p.y, p.yaw)

    def look_around(self, x: float, y: float, yaw: float) -> None:
        """The LookAround clip is a T-pose in this Isaac version: sweep the head with look_at targets instead."""
        self._sweep = []
        for i, deg in enumerate(SWEEP_DEG):
            a = yaw + math.radians(deg)
            self._sweep.append((i * SWEEP_STEP_S, (x + 3.0 * math.cos(a), y + 3.0 * math.sin(a), EYE_Z)))
        self._sweep_t = 0.0
        self.events.append(("look_around", ""))

    # ---- per frame -------------------------------------------------------------------------------
    def update(self, world, dt: float) -> None:
        p = world.people.get(self.pid)
        if p is None:
            return
        if p.posture == "gone":
            if not self.hidden:
                self.api.hide()
                self.hidden = True
            return
        self._tick_sweep(p, dt)
        for _ in range(4):  # a phase that ends hands over to the next one in the same frame
            if not p.plan:
                self._phase = None
                if p.posture == "walking":
                    p.posture = "standing"
                return
            ph = p.plan[0]
            if ph is not self._phase:
                self._begin(world, p, ph)
            if not self._run(world, p, ph, dt):
                return
            dt = 0.0
        self._set_posture(p)

    def _tick_sweep(self, p, dt: float) -> None:
        if not self._sweep:
            return
        if p.plan or p.posture != "standing":
            self._sweep = []
            return
        self._sweep_t += dt
        while self._sweep and self._sweep[0][0] <= self._sweep_t:
            _, (x, y, z) = self._sweep.pop(0)
            self._task = self.api.look_at(x, y, z, SWEEP_STEP_S + 0.4)

    # ---- phases --------------------------------------------------------------------------------
    def _begin(self, world, p, ph) -> None:
        self._phase = ph
        self._sweep = []
        self._retries = 0
        self._elapsed = 0.0
        self._still = 0.0
        pose = self.api.pose()
        self._anchor = (pose[0], pose[1]) if pose else None
        if ph.kind == "stand":
            self._left = max(ph.left, STAND_MIN_S)
            self._task = self.api.idle()  # an Idle task is how a sitter stands up
        elif ph.kind == "sit":
            self._left = max(ph.left, SIT_MIN_S)
            self._task = self.api.sit(self.seat_prim(p.claim))
        else:
            self._goal = self._walk_goal(world, p, ph)
            self.api.set_speed(clamp_speed(ph.speed))
            self._issue_walk(self._goal)
            d = math.hypot(self._goal[0] - p.x, self._goal[1] - p.y)
            self._timeout = TIMEOUT_FACTOR * d / max(ph.speed, 0.1) + TIMEOUT_EXTRA_S

    def _walk_goal(self, world, p, ph) -> tuple[float, float]:
        goal = world.walk_goal(self.pid) or (p.x, p.y)
        if len(p.plan) > 1 and p.plan[1].kind == "sit" and p.claim:
            return seat_approach(world.seat(p.claim))
        return goal

    def _issue_walk(self, goal: tuple[float, float]) -> None:
        self._task = self.api.move_to(goal[0], goal[1])
        self.events.append(("move_to", f"{goal[0]:.2f},{goal[1]:.2f}"))

    def _run(self, world, p, ph, dt: float) -> bool:
        """Advance the head phase. True when it ended and the next phase may start now."""
        self._elapsed += dt
        if ph.kind in ("stand", "sit"):
            self._left -= dt
            if ph.kind == "sit" and self.api.status(self._task) == "failed" and self._retries < SIT_RETRIES:
                self._retries += 1
                self._task = self.api.sit(self.seat_prim(p.claim))
            if self._left <= 0.0:
                world.end_phase(self.pid)
                return True
            self._set_posture(p)
            return False
        return self._run_walk(world, p, ph, dt)

    def _run_walk(self, world, p, ph, dt: float) -> bool:
        pose = self.api.pose()
        if pose is None:
            return False
        x, y = pose[0], pose[1]
        if ph.follow is not None:  # the person being approached moves: chase them
            nd = world.walk_goal(self.pid)
            if nd is not None and math.hypot(nd[0] - self._goal[0], nd[1] - self._goal[1]) > REROUTE_M:
                self._goal = nd
                self._issue_walk(nd)
                self._still = 0.0
        d = math.hypot(self._goal[0] - x, self._goal[1] - y)
        st = self.api.status(self._task)
        if d <= ARRIVE_BODY_M or (st == "succeeded" and d <= ARRIVE_ENDED_M):
            world.end_phase(self.pid)
            return True
        if self._anchor is not None and math.hypot(x - self._anchor[0], y - self._anchor[1]) > STUCK_MOVE_M:
            self._anchor, self._still = (x, y), 0.0
        else:
            self._still += dt
        stuck = st == "running" and self._still >= STUCK_S
        if st in ("failed", "none", "succeeded") or stuck:
            if self._retries >= WALK_RETRIES:
                return self._block(world, p)
            if stuck:
                self.api.cancel(self._task)
            self._retries += 1
            self._still = 0.0
            self.events.append(("retry", f"{self._retries}"))
            self._issue_walk(self._goal)
        elif self._elapsed > self._timeout:
            return self._block(world, p)
        self._set_posture(p)
        return False

    def _block(self, world, p) -> bool:
        log.warning("%s: walk to %s blocked after %d re-issues", self.pid, self._goal, self._retries)
        self.api.cancel(self._task)
        self._task = self.api.idle()
        world.fail_intent(self.pid, "blocked")
        self._phase = None
        self.events.append(("blocked", ""))
        return False

    def _set_posture(self, p) -> None:
        """Posture comes from what the character's behavior is doing."""
        if p.posture in ("seated", "lying", "gone"):
            return
        p.posture = "walking" if self.api.current_action() in WALKING else "standing"
