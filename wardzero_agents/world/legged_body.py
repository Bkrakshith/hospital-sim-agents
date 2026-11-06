"""The world's stand-in for a robot that walks on its own legs in Isaac Sim (the built-in G1 with GEAR-SONIC).

``RobotBody`` moves itself (a unicycle on the grid). A ``LeggedBody`` never moves itself: Isaac owns the body and the
host writes the measured pelvis pose into ``x, y, yaw`` every frame (``IsaacBodies.read_back``). What it keeps is the
goal side the task layer talks to (``set_goal``, ``turn_to``, ``cancel_goal``, ``has_goal``, ``take_goal_event``), run
by a ``RouteFollower`` on a router with the G1's clearance. Each ``step`` (the world's) asks the follower for a walking
command at ``FOLLOW_HZ`` and keeps it in ``command``; the Isaac host's SONIC driver turns that into planner commands at
its own 50 Hz tick. Results come back as goal events: ``arrived``, ``stuck`` (5 s of sim time without progress) and
``fell``. A fall ends the robot's part of the run: every later goal fails with ``fell``.

Pure Python, no simulator.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from .robot_body import CONTACT_MARGIN_M, RobotBody
from .route_follower import (ARRIVED, FELL, IDLE, NO_GOAL, RESULTS, STUCK, FollowerCommand, RouteFollower,
                             SLOW_WALK, wrap)

FOLLOW_HZ = 10.0  # follower updates per sim second (the planner decides at 10 Hz too)
MIN_WALK_SPEED = 0.2  # m/s: SONIC's SLOW_WALK takes 0.2-0.8


def door_points(site) -> list[tuple[float, float]]:
    """The middle of every door in the site map (the follower slows near them)."""
    out = []
    for d in getattr(site, "doors", None) or []:
        try:
            a, b = d["a"], d["b"]
            out.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
        except (KeyError, TypeError, IndexError):
            continue
    return out


def block_segments(router, segments, clearance: float, step_m: float = 0.05) -> int:
    """Make every map cell a segment passes through impassable in this router (only this one: the walking G1's),
    then rebuild its clearance band with ``clearance``. For door leaves: the site map and the NavMesh leave them out,
    and an open leaf standing into a corridor stopped the gate-test-style walk back from the ward doors. Returns the
    cells newly blocked."""
    import numpy as np

    g = router.grid
    raw = np.array(router._raw, dtype=bool)
    n0 = int(raw.sum())
    for (x0, y0), (x1, y1) in segments:
        k = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step_m)))
        for i in range(k + 1):
            iy, ix = g.to_cell(x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k)
            if g.in_grid(iy, ix):
                raw[iy, ix] = True
    router._raw = raw.tolist()
    router._b = router._raw
    router.clearance = float(clearance)
    router._near = router._inflate(raw, clearance).tolist()
    router._cache.clear()
    router.version = getattr(router, "version", 0) + 1  # an AreaRouter over it rebuilds its copy
    return int(raw.sum()) - n0


class LeggedBody(RobotBody):
    """``RobotBody``'s interface over a ``RouteFollower``; the pose is written from outside (Isaac)."""

    def __init__(self, x: float, y: float, yaw: float, radius: float = 0.3, max_speed: float = 0.6, router=None,
                 doors: Iterable = (), follower: Optional[RouteFollower] = None, follow_hz: float = FOLLOW_HZ):
        super().__init__(x, y, yaw, radius, max_speed, router)
        self.follower = follower or RouteFollower(router)
        self.doors = [tuple(d) for d in doors]
        self.follow_dt = 1.0 / follow_hz
        self.command: FollowerCommand = FollowerCommand(IDLE, 0.0, None, None, NO_GOAL)
        self.ready = False  # the walking controller is up (start-up done); goals wait until then
        self.fallen: Optional[str] = None  # why the robot fell (set by the host's driver), then it never walks again
        self.exempt: set[str] = set()  # people who do not slow the robot (the person it guides)
        self.speed_scale = 1.0  # set by a task layer: 0 holds the robot where it is (the stuck clock does not run)
        self.speed_cap: Optional[float] = None
        self._t = 0.0
        self._next_follow = 0.0
        self._last_xy: Optional[tuple[float, float]] = None
        self._goal_silent = False  # a turn_to: no goal event when it ends

    # ---- the task layer's side -----------------------------------------------------------
    @property
    def has_goal(self) -> bool:
        return self.follower.active  # a turn_to counts, as in RobotBody

    def set_goal(self, x: float, y: float, yaw: Optional[float] = None, avoid=()) -> bool:
        """Follow a route to (x, y), ending facing yaw if given. False when there is none. A fallen robot takes the goal
        and ends it at once with ``fell``, so the brain is told why."""
        if self.fallen:
            self._had_goal = False
            self.goal_event = FELL
            return True
        if self.router is not None and self.follower.router is not self.router:
            self.follower.router = self.router  # a task layer may swap the router (forbidden areas)
        ok = self.follower.set_goal((self.x, self.y), (x, y), yaw, avoid)
        self._goal_silent = False
        self._had_goal = ok
        self.goal_event = None
        self._next_follow = self._t  # the next step asks the follower at once
        return ok

    def turn_to(self, yaw: float) -> None:
        """Turn on the spot to yaw. Not a goal: no goal event when it is done."""
        if self.fallen:
            return
        self.follower.set_route([], final_yaw=float(yaw))
        self._goal_silent = True
        self._had_goal = False
        self._next_follow = self._t

    def cancel_goal(self) -> None:
        self.follower.cancel()
        self._goal_silent = False
        self._had_goal = False
        self.command = FollowerCommand(IDLE, 0.0, None, None, NO_GOAL)

    def set_cmd_vel(self, v: float, w: float) -> None:
        """A legged robot takes no raw velocities from the brain here: it stops (and drops its goal)."""
        self.cancel_goal()
        self._v = self._w = 0.0

    def teleport(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.cancel_goal()

    # ---- the host's side -----------------------------------------------------------------
    def report_fall(self, reason: str) -> None:
        """The walking controller saw the robot fall. A goal in flight ends with ``fell``."""
        if self.fallen:
            return
        self.fallen = reason or "fell"
        if self.follower.active and not self._goal_silent:
            self.goal_event = FELL
        self.follower.cancel()
        self._had_goal = False
        self.command = FollowerCommand(IDLE, 0.0, None, None, FELL)

    # ---- the world's step ----------------------------------------------------------------
    def step(self, dt: float, grid, people: Optional[Iterable[tuple[str, float, float]]] = None) -> list[str]:
        """No motion here (Isaac moves the body). Measured speed, the follower's command and the contacts."""
        people = list(people or [])
        if dt > 0:
            self._t += dt
            if self._last_xy is not None:
                self.vx = math.hypot(self.x - self._last_xy[0], self.y - self._last_xy[1]) / dt
            self._last_xy = (self.x, self.y)
            self._follow(people)
        new: list[str] = []
        near = {pid for pid, px, py in people if math.hypot(px - self.x, py - self.y) <= self.radius + CONTACT_MARGIN_M}
        new = sorted(near - self._touching)
        self._touching = near
        return new

    def _follow(self, people) -> None:
        f = self.follower
        if self.fallen or not f.active:
            return
        if not self.ready or self.speed_scale <= 0:
            f.reset_progress()  # held on purpose: start-up, or waiting for the person it guides
            self.command = FollowerCommand(IDLE, 0.0, None, None, "holding")
            return
        if self._t + 1e-9 < self._next_follow:
            return
        self._next_follow = self._t + self.follow_dt
        others = [(px, py) for pid, px, py in people if pid not in self.exempt]
        cmd = f.update(self._t, (self.x, self.y, self.yaw), self.doors, others)
        if cmd.mode == SLOW_WALK:
            top = self.max_speed * min(1.0, self.speed_scale)
            if self.speed_cap is not None and not f._in_door_zone(self.x, self.y):
                top = min(top, self.speed_cap)  # not while it pushes a door's closer: 0.3 m/s wedges it (door spike)
            cmd = FollowerCommand(cmd.mode, max(MIN_WALK_SPEED, min(cmd.speed, top)), cmd.move_yaw, cmd.face_yaw,
                                  cmd.status, cmd.slow)
        self.command = cmd
        ev = f.take_event()
        if ev in RESULTS:
            if not self._goal_silent:
                self.goal_event = ev
            self._goal_silent = False
            self._had_goal = False

    def heading_error(self) -> Optional[float]:
        """How far the commanded facing is from the measured yaw (rad), for logs."""
        c = self.command
        return None if c.face_yaw is None else wrap(c.face_yaw - self.yaw)


__all__ = ["LeggedBody", "door_points", "FOLLOW_HZ", "ARRIVED", "STUCK", "FELL"]
