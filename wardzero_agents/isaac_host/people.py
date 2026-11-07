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
- a sit effector is a point on the chair: the walk before a sit stops on the floor in front of the seat;
- a character who stands up from a seat stands inside the seat row, off the NavMesh, and every MoveTo from there fails
  at once: the first failed walk steps them out to the floor in front of the seat (``teleport``), and a walk that still
  fails for good while they stand in furniture puts them back in the seat they left, or on the nearest free floor.
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
WAYPOINT_REACH_M = 0.6  # a route waypoint counts as passed this close (the last one is the goal itself)
REROUTE_M = 0.5  # a walk to a person who moves is re-issued when they have moved this far
SPEED_RANGE = (0.4, 1.8)
EYE_Z = 1.55
LOOK_S = 2.5
SWEEP_DEG = (60.0, -60.0, 0.0)  # "look around": left, right, back to the front
SWEEP_STEP_S = 1.3
SIT_RETRIES = 2
TRAIL_NEAR_M = 0.3  # a led person within this of their trail point stands
TRAIL_REISSUE_S = 0.4  # at most one new move_to this often
TRAIL_SPEED = 1.5
RESEAT_M = 1.5  # a person stuck in furniture within this of the seat they left sits back down in it
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
    def show(self) -> None: ...
    # optional: def teleport(self, x: float, y: float, yaw: float) -> None  (puts the character there at once)


class BedProp(Protocol):
    """The posed character lying in a bed, for a person who has a mind (the walking character is hidden meanwhile)."""

    def show(self) -> None: ...
    def hide(self) -> None: ...
    def place(self, x: float, y: float, yaw: float) -> None: ...  # a bed being wheeled


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

    def __init__(self, pid: str, api: BehaviorApi, seat_prim: Callable[[str], str] = seat_prim_path,
                 bed_prop: Optional[BedProp] = None):
        self.pid, self.api, self.seat_prim = pid, api, seat_prim
        self.bed_prop = bed_prop  # a minded patient who starts in a bed: lying prop shown, walking character hidden
        self.in_bed = bed_prop is not None
        self._trail_goal: Optional[tuple[float, float]] = None
        self._trail_age = 0.0
        self._phase = None  # the plan phase being carried out (compared by identity)
        self._task: Optional[int] = None
        self._goal: Optional[tuple[float, float]] = None
        self._wps: list = []  # route waypoints still to pass before the goal (the world's route, round the robot)
        self._retries = 0
        self._stepped = False  # this walk already stepped the character out of furniture
        self._left_seat: Optional[str] = None  # the seat this person last stood up from
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
                if self.bed_prop is not None:
                    self.bed_prop.hide()
                self.hidden = True
            return
        if self.bed_prop is not None and self._swap_bed(p):
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

    def _swap_bed(self, p) -> bool:
        """A minded patient swaps between the lying prop and the walking character. True while they are in bed."""
        lying = p.posture == "lying" and not (p.plan and p.plan[0].kind == "stand")
        if lying and not self.in_bed:  # lay back down: the character goes, the prop comes back
            self.api.hide()
            self.bed_prop.show()
            self.in_bed = True
            self.events.append(("lie_down", ""))
        elif not lying and self.in_bed:  # getting up: the character appears at the bedside
            self.bed_prop.hide()
            self.api.show()
            self.in_bed = False
            self.events.append(("get_up", ""))
        return self.in_bed

    def trail(self, world, point: tuple[float, float], dt: float) -> None:
        """This person is being led (escorted, evacuated): their character follows the leader's trail point."""
        pose = self.api.pose()
        if pose is None:
            return
        self._trail_age += dt
        d = math.hypot(point[0] - pose[0], point[1] - pose[1])
        ended = self.api.status(self._task) != "running" if self._task is not None else True
        if d > TRAIL_NEAR_M and (self._trail_goal is None or ended
                                 or math.hypot(point[0] - self._trail_goal[0], point[1] - self._trail_goal[1]) > REROUTE_M) \
                and self._trail_age >= TRAIL_REISSUE_S:
            self._trail_age = 0.0
            self._trail_goal = point
            self._phase = None
            self.api.set_speed(clamp_speed(TRAIL_SPEED))
            self._issue_walk(point)
        p = world.people[self.pid]
        if p.posture not in ("lying", "gone"):
            p.posture = "walking" if self.api.current_action() in WALKING else "standing"

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
        self._stepped = False
        self._elapsed = 0.0
        self._still = 0.0
        pose = self.api.pose()
        self._anchor = (pose[0], pose[1]) if pose else None
        if ph.kind == "stand":
            self._left_seat = p.seat or self._left_seat
            self._left = max(ph.left, STAND_MIN_S)
            self._task = self.api.idle()  # an Idle task is how a sitter stands up
        elif ph.kind == "sit":
            self._left_seat = None
            self._left = max(ph.left, SIT_MIN_S)
            self._task = self.api.sit(self.seat_prim(p.claim))
        else:
            self._goal = self._walk_goal(world, p, ph)
            self.api.set_speed(clamp_speed(ph.speed))
            self._plan_route(world, self._goal)
            self._issue_walk(self._target())
            d = math.hypot(self._goal[0] - p.x, self._goal[1] - p.y)
            self._timeout = TIMEOUT_FACTOR * d / max(ph.speed, 0.1) + TIMEOUT_EXTRA_S

    def _walk_goal(self, world, p, ph) -> tuple[float, float]:
        goal = world.walk_goal(self.pid) or (p.x, p.y)
        if len(p.plan) > 1 and p.plan[1].kind == "sit" and p.claim:
            return seat_approach(world.seat(p.claim))
        return goal

    def _plan_route(self, world, goal) -> None:
        """The world's route to the goal (it keeps clear of the robot): the NavMesh's straight path to the goal walked
        the receptionist right through the G1 standing at the counter. Without one the character walks straight."""
        route = getattr(world, "route_for", None)
        pts = route(self.pid, goal) if route is not None else None
        self._wps = [tuple(q) for q in pts[1:-1]] if pts and len(pts) > 2 else []
        if pts and len(pts) >= 2:  # the goal may have been moved out of the robot's contact distance
            self._goal = tuple(pts[-1])

    def _target(self) -> tuple[float, float]:
        return self._wps[0] if self._wps else self._goal

    def _next_waypoint(self, x: float, y: float) -> None:
        if self._wps and math.hypot(self._wps[0][0] - x, self._wps[0][1] - y) < WAYPOINT_REACH_M:
            self._wps.pop(0)
            self._issue_walk(self._target())

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
        if getattr(ph, "trail", False):  # following the robot: the world says when it is over, never "arrived"
            if world.trail_done(self.pid):
                world.end_phase(self.pid)
                return True
            nd = world.walk_goal(self.pid)
            st = self.api.status(self._task)
            if nd is not None and (math.hypot(nd[0] - self._goal[0], nd[1] - self._goal[1]) > REROUTE_M
                                   or (st != "running" and math.hypot(nd[0] - x, nd[1] - y) > ARRIVE_ENDED_M)):
                self._goal = nd
                self._issue_walk(nd)
            self._set_posture(p)
            return False
        if ph.follow is not None:  # the person being approached moves: chase them
            nd = world.walk_goal(self.pid)
            if nd is not None and math.hypot(nd[0] - self._goal[0], nd[1] - self._goal[1]) > REROUTE_M:
                self._goal = nd
                self._plan_route(world, nd)
                self._issue_walk(self._target())
                self._still = 0.0
        self._next_waypoint(x, y)
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
            if not self._stepped and self._in_furniture(world, x, y) and self._step_out(world, p, x, y):
                self._still = 0.0
                self._anchor = None
                self._issue_walk(self._target())  # from the free floor now: not a re-issue of a walk that ended short
                self._set_posture(p)
                return False
            if self._retries >= WALK_RETRIES:
                return self._block(world, p)
            if stuck:
                self.api.cancel(self._task)
            self._retries += 1
            self._still = 0.0
            self.events.append(("retry", f"{self._retries}"))
            self._issue_walk(self._target())
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
        pose = self.api.pose()
        if pose is not None and self._in_furniture(world, pose[0], pose[1]):
            self._recover(world, p, pose[0], pose[1])
        return False

    # ---- never left standing in furniture --------------------------------------------------------
    @staticmethod
    def _in_furniture(world, x: float, y: float) -> bool:
        grid = getattr(getattr(world, "site", None), "grid", None)
        return grid is not None and bool(grid.blocked(x, y))

    def _free_floor(self, world, x: float, y: float) -> Optional[tuple[float, float]]:
        """Where a character stuck in furniture can stand: the floor in front of the seat they left (the seat rows'
        own way out), else the nearest free spot the world finds; None when there is none."""
        seat = self._seat_left(world, x, y)
        tries = [seat_approach(world.seat(seat))] if seat is not None else []
        tries.append((x, y))
        free = getattr(world, "free_spot", None)
        for tx, ty in tries:
            fx, fy = free(tx, ty, self.pid) if free is not None else (tx, ty)
            if not self._in_furniture(world, fx, fy):
                return fx, fy
        return None

    def _seat_left(self, world, x: float, y: float) -> Optional[str]:
        sid = self._left_seat
        if sid is None or sid not in getattr(world.site, "seats", {}):
            return None
        seat = world.seat(sid)
        return sid if math.hypot(seat.x - x, seat.y - y) <= RESEAT_M else None

    def _step_out(self, world, p, x: float, y: float) -> bool:
        """Put the character on the free floor at once (a short step out of a seat row). False when it cannot."""
        tp = getattr(self.api, "teleport", None)
        spot = self._free_floor(world, x, y) if tp is not None else None
        self._stepped = True
        if spot is None:
            return False
        yaw = math.atan2(spot[1] - y, spot[0] - x) if math.hypot(spot[0] - x, spot[1] - y) > 1e-6 else (p.yaw or 0.0)
        tp(spot[0], spot[1], yaw)
        self.events.append(("step_out", f"{spot[0]:.2f},{spot[1]:.2f}"))
        log.info("%s: stepped out of furniture at (%.2f, %.2f) to (%.2f, %.2f)", self.pid, x, y, spot[0], spot[1])
        return True

    def _recover(self, world, p, x: float, y: float) -> None:
        """A walk failed for good with the character inside furniture: back in the seat they left when it is still
        free, else onto the nearest free floor."""
        sid = self._seat_left(world, x, y)
        taken = getattr(world, "_seat_taken", None)
        if sid is not None and not (taken is not None and taken(sid, self.pid)):
            self._task = self.api.sit(self.seat_prim(sid))
            p.posture, p.seat = "seated", sid
            self._left_seat = None
            self.events.append(("reseat", sid))
            log.info("%s: back in seat %s after the failed walk", self.pid, sid)
            return
        self._step_out(world, p, x, y)

    def _set_posture(self, p) -> None:
        """Posture comes from what the character's behavior is doing."""
        if p.posture in ("seated", "lying", "gone"):
            return
        p.posture = "walking" if self.api.current_action() in WALKING else "standing"
