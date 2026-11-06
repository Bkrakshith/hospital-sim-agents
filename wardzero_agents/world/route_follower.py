"""Route follower: a goal and a Router route in, walking commands out. Simulator-agnostic and pure Python.

The same follower drives the walking G1 (its commands become GEAR-SONIC planner commands,
``isaac_host.sonic.SonicRunner.follow``) and the 2D unicycle ``RobotBody`` (``UnicycleDrive``), so routes in
Quick 2D and in Isaac match.

Each ``update`` (any rate; 10 Hz is plenty) takes the sim time, the measured pose, door points and people
positions, and returns a ``FollowerCommand``: mode IDLE or SLOW_WALK, a speed, and world-frame movement and facing
yaw angles. It walks facing the next waypoint, slows to 0.3 m/s within 1.5 m of a door or a person (0.6 m/s
otherwise), turns in place (IDLE with a facing) when the waypoint is far off to the side, and ends with
``arrived``, ``stuck`` (no progress for 5 s of sim time) or ``fell``.

The heading loop is closed on the measured yaw. SONIC's planner turns by about 90 % of the step between its
current heading and a new facing (an in-place 90 deg turn stops at 81 deg), so once the robot has stopped turning
with more than ``yaw_tol`` left, the facing is re-issued from the measured yaw: ``yaw + correct_gain * error``.
The facing is held steady otherwise (it moves only when the bearing moves more than ``retarget``), because every
facing change makes the planner re-plan.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Same numbers as GEAR-SONIC's planner modes.
IDLE, SLOW_WALK = 0, 1
OPEN_SPEED = 0.6  # m/s commanded in open space
SLOW_SPEED = 0.3  # m/s within SLOW_RADIUS of a door or a person
SLOW_RADIUS = 1.5
ARRIVE_M = 0.3
WAYPOINT_M = 0.4
STUCK_S = 5.0
STUCK_M = 0.1
STUCK_YAW = math.radians(10)  # turning this much also counts as progress
YAW_TOL = math.radians(8)
TURN_IN_PLACE = math.radians(50)  # heading error that stops the walk for an in-place turn
WALK_AGAIN = math.radians(20)  # ... and the error at which walking resumes
SETTLE_S = 0.6  # the robot counts as no longer turning when its yaw moved < SETTLE_YAW over this long
SETTLE_YAW = math.radians(2)
CORRECT_GAIN = 1.2  # over-asks a little, since the planner delivers ~90 % of a facing step
RETARGET = math.radians(4)

FOLLOWING, TURNING, ARRIVED, STUCK, FELL, NO_GOAL = "following", "turning", "arrived", "stuck", "fell", "no_goal"
RESULTS = (ARRIVED, STUCK, FELL)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class FollowerCommand:
    mode: int  # IDLE or SLOW_WALK
    speed: float  # m/s; 0 when IDLE
    move_yaw: Optional[float]  # world-frame direction of travel (rad), None when IDLE
    face_yaw: Optional[float]  # world-frame facing (rad), None = keep the current facing
    status: str  # following | turning | arrived | stuck | fell | no_goal
    slow: bool = False  # inside a slow zone


def near_any(x: float, y: float, points: Iterable[Sequence[float]], radius: float) -> bool:
    return any(math.hypot(p[0] - x, p[1] - y) <= radius for p in points)


class RouteFollower:
    def __init__(self, router=None, *, open_speed: float = OPEN_SPEED, slow_speed: float = SLOW_SPEED,
                 slow_radius: float = SLOW_RADIUS, arrive_m: float = ARRIVE_M, waypoint_m: float = WAYPOINT_M,
                 stuck_s: float = STUCK_S, stuck_m: float = STUCK_M, yaw_tol: float = YAW_TOL,
                 turn_in_place: float = TURN_IN_PLACE, walk_again: float = WALK_AGAIN, settle_s: float = SETTLE_S,
                 correct_gain: float = CORRECT_GAIN, retarget: float = RETARGET) -> None:
        self.router = router
        self.open_speed, self.slow_speed, self.slow_radius = open_speed, slow_speed, slow_radius
        self.arrive_m, self.waypoint_m = arrive_m, waypoint_m
        self.stuck_s, self.stuck_m = stuck_s, stuck_m
        self.yaw_tol, self.turn_in_place, self.walk_again = yaw_tol, turn_in_place, walk_again
        self.settle_s, self.correct_gain, self.retarget = settle_s, correct_gain, retarget
        self.path: list[tuple[float, float]] = []
        self.final_yaw: Optional[float] = None
        self.result: Optional[str] = None
        self._event: Optional[str] = None
        self._active = False
        self._in_place = False
        self._final_turn = False
        self._target: Optional[float] = None
        self.face_cmd: Optional[float] = None  # the facing being commanded (world yaw)
        self.corrections = 0
        self._yaws: deque[tuple[float, float]] = deque()
        self._anchor: Optional[tuple[float, float, float, float]] = None  # x, y, yaw, t

    # ---- goals -------------------------------------------------------------------------------
    def set_goal(self, start: Sequence[float], goal: Sequence[float], final_yaw: Optional[float] = None) -> bool:
        """Route from start (x, y) to goal (x, y) with the Router. False (and no goal) when there is none."""
        route = self.router.route((start[0], start[1]), (goal[0], goal[1])) if self.router is not None else None
        if route is None:
            self.cancel()
            return False
        self.set_route(route[1:] or [tuple(goal[:2])], final_yaw)
        return True

    def set_route(self, waypoints: Sequence[Sequence[float]], final_yaw: Optional[float] = None) -> None:
        """Follow these world (x, y) waypoints (the start point left out), then face final_yaw if given."""
        self.path = [(float(p[0]), float(p[1])) for p in waypoints]
        self.final_yaw = final_yaw
        self.result = self._event = None
        self._active = True
        self._in_place = self._final_turn = False
        self._target = None
        self.face_cmd = None
        self.corrections = 0
        self._yaws.clear()
        self._anchor = None

    def cancel(self) -> None:
        """Dropped on purpose: no result event."""
        self.path = []
        self.final_yaw = None
        self._active = False

    def take_event(self) -> Optional[str]:
        """``arrived``, ``stuck`` or ``fell``, once."""
        ev, self._event = self._event, None
        return ev

    @property
    def active(self) -> bool:
        return self._active

    # ---- the loop ----------------------------------------------------------------------------
    def _finish(self, result: str, hold_yaw: Optional[float] = None) -> FollowerCommand:
        """End the goal. hold_yaw: face the measured yaw, so the planner does not finish a turn it was asked for."""
        self._active = False
        self.result = self._event = result
        self.path = []
        return FollowerCommand(IDLE, 0.0, None, hold_yaw, result)

    def _settled(self, t: float, yaw: float) -> bool:
        self._yaws.append((t, yaw))
        while len(self._yaws) > 1 and self._yaws[1][0] <= t - self.settle_s:
            self._yaws.popleft()
        t0, y0 = self._yaws[0]
        return t - t0 >= self.settle_s - 1e-9 and abs(wrap(yaw - y0)) < SETTLE_YAW

    def _face(self, target: float, yaw: float, settled: bool) -> float:
        """The facing to command for a target yaw: the target itself, re-issued from the measured yaw when the
        robot has stopped turning short of it."""
        if self.face_cmd is None or self._target is None or abs(wrap(target - self._target)) > self.retarget:
            self._target = target
            self.face_cmd = target
        err = wrap(self._target - yaw)
        if settled and abs(err) > self.yaw_tol:
            self.face_cmd = wrap(yaw + self.correct_gain * err)
            self.corrections += 1
            self._yaws.clear()  # let the robot react before judging again
        return self.face_cmd

    def _progress(self, t: float, x: float, y: float, yaw: float) -> bool:
        """False once there was no progress for stuck_s of sim time."""
        a = self._anchor
        if a is None or math.hypot(x - a[0], y - a[1]) >= self.stuck_m or abs(wrap(yaw - a[2])) >= STUCK_YAW:
            self._anchor = (x, y, yaw, t)
            return True
        return t - a[3] < self.stuck_s

    def update(self, t: float, pose: Sequence[float], doors: Iterable[Sequence[float]] = (),
               people: Iterable[Sequence[float]] = (), fallen: bool = False) -> FollowerCommand:
        """One step at sim time t (s). pose = measured (x, y, yaw); doors and people are (x, y) points (leave out a
        person who is being guided). fallen: the robot's own fall flag."""
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        if not self._active:
            return FollowerCommand(IDLE, 0.0, None, None, self.result or NO_GOAL)
        if fallen:
            return self._finish(FELL)
        if not self._progress(t, x, y, yaw):
            return self._finish(STUCK, yaw)
        settled = self._settled(t, yaw)
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < (
                self.arrive_m if len(self.path) == 1 else self.waypoint_m):
            self.path.pop(0)
        if not self.path:
            if self.final_yaw is None:
                return self._finish(ARRIVED, yaw)
            if not self._final_turn:
                self._final_turn = True
                self._yaws.clear()
                self._yaws.append((t, yaw))
                settled = False
            face = self._face(self.final_yaw, yaw, settled)
            if abs(wrap(self.final_yaw - yaw)) <= self.yaw_tol and settled:
                return self._finish(ARRIVED, yaw)
            return FollowerCommand(IDLE, 0.0, None, face, TURNING)
        tx, ty = self.path[0]
        bearing = math.atan2(ty - y, tx - x)
        err = abs(wrap(bearing - yaw))
        if self._in_place and err <= self.walk_again:
            self._in_place = False
        elif not self._in_place and err > self.turn_in_place:
            self._in_place = True
        face = self._face(bearing, yaw, settled)
        if self._in_place:
            return FollowerCommand(IDLE, 0.0, None, face, TURNING)
        slow = near_any(x, y, doors, self.slow_radius) or near_any(x, y, people, self.slow_radius)
        speed = self.slow_speed if slow else self.open_speed
        return FollowerCommand(SLOW_WALK, speed, face, face, FOLLOWING, slow)


class UnicycleDrive:
    """Drives the 2D ``RobotBody`` with a ``RouteFollower`` (instead of the body's own goal following).

    Each ``step`` asks the follower, turns its command into the unicycle's (v, w) and steps the body."""

    def __init__(self, body, follower: Optional[RouteFollower] = None, turn_rate: float = 3.0) -> None:
        self.body = body
        self.follower = follower or RouteFollower(body.router)
        self.turn_rate = turn_rate
        self.last: Optional[FollowerCommand] = None

    def set_goal(self, x: float, y: float, yaw: Optional[float] = None) -> bool:
        return self.follower.set_goal((self.body.x, self.body.y), (x, y), yaw)

    def step(self, t: float, dt: float, grid, people: Optional[Iterable[tuple[str, float, float]]] = None,
             doors: Iterable[Sequence[float]] = (), exempt: Iterable[str] = ()) -> list[str]:
        """people = (id, x, y) as for RobotBody.step; ``exempt`` ids do not slow the robot (a guided person)."""
        people = list(people or [])
        skip = set(exempt)
        cmd = self.follower.update(t, self.body.pose, doors, [(px, py) for pid, px, py in people if pid not in skip])
        self.last = cmd
        v, w = 0.0, 0.0
        if cmd.face_yaw is not None:
            err = wrap(cmd.face_yaw - self.body.yaw)
            w = max(-self.turn_rate, min(self.turn_rate, err / max(dt, 1e-3)))
            if cmd.mode == SLOW_WALK and abs(err) < 1.2:
                v = cmd.speed * max(0.0, math.cos(err))
        self.body.set_cmd_vel(v, w)
        return self.body.step(dt, grid, people)
