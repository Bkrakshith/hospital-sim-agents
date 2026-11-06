"""Route follower: a goal and a Router route in, walking commands out. Simulator-agnostic and pure Python.

The same follower drives the walking G1 (its commands become GEAR-SONIC planner commands,
``isaac_host.sonic.SonicRunner.follow``) and the 2D unicycle ``RobotBody`` (``UnicycleDrive``), so routes in
Quick 2D and in Isaac match.

Each ``update`` (any rate; 10 Hz is plenty) takes the sim time, the measured pose, door points and people
positions, and returns a ``FollowerCommand``: mode IDLE or SLOW_WALK, a speed, and world-frame movement and facing
yaw angles. It walks facing the next waypoint, slows to 0.3 m/s within 1.5 m of a person (0.6 m/s otherwise, doors
included: the door spike found the G1 passes a door leaf cleanly at 0.6 m/s and gets stuck against it at 0.3), turns in place (IDLE with a facing) when the waypoint is far off to the side, and ends with
``arrived``, ``stuck`` (no progress for 5 s of sim time) or ``fell``.

Cross-track correction: the robot steers at a point ``lookahead`` metres ahead of its foot on the current route segment
(pure pursuit), not at the waypoint itself, so a robot pushed off the line (a desk corner, a sideways drift of the
walking gait) walks back onto it instead of cutting the corner. Give the follower a Router with a clearance band
wider than the robot's half-width with arms (``G1_CLEARANCE_M`` for the G1): the gate test's G1 rubbed along the end
of the reception desk on a 0.3 m clearance route.

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
SLOW_SPEED = 0.3  # m/s within SLOW_RADIUS of a person (not of a door: at 0.3 m/s the G1 stalled against a leaf)
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
DOOR_ZONE_M = 1.5  # within this of a swinging door's line the G1 keeps walking speed even with people near
PRECISE_M = 0.2  # a door pass point counts as reached this close (the robot lines up on the leaf before it walks on)
LOOKAHEAD_M = 0.8  # pure-pursuit point ahead on the route segment (cross-track correction)
# route clearance band for the walking G1 (half-width with arms ~0.35 m). The band is soft and the map's cells are
# 0.2 m, so a 0.45 m band still let the route pass 0.31 m from the end of the reception desk; 0.6 keeps it >= 0.44 m
G1_CLEARANCE_M = 0.6

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
                 correct_gain: float = CORRECT_GAIN, retarget: float = RETARGET,
                 lookahead: float = LOOKAHEAD_M, slow_near_doors: bool = False) -> None:
        self.router = router
        self.open_speed, self.slow_speed, self.slow_radius = open_speed, slow_speed, slow_radius
        self.arrive_m, self.waypoint_m = arrive_m, waypoint_m
        self.stuck_s, self.stuck_m = stuck_s, stuck_m
        self.yaw_tol, self.turn_in_place, self.walk_again = yaw_tol, turn_in_place, walk_again
        self.settle_s, self.correct_gain, self.retarget = settle_s, correct_gain, retarget
        self.lookahead = lookahead
        self.slow_near_doors = slow_near_doors
        self.door_lines: list = []  # swinging doors ((ax, ay), (bx, by)): routes pass straight through a leaf
        self.precise: set = set()  # door pass points: reached within PRECISE_M, so the pass starts on its line
        self.path: list[tuple[float, float]] = []
        self._prev: Optional[tuple[float, float]] = None  # start of the current route segment
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
    def set_goal(self, start: Sequence[float], goal: Sequence[float], final_yaw: Optional[float] = None,
                 avoid=()) -> bool:
        """Route from start (x, y) to goal (x, y) with the Router. False (and no goal) when there is none.
        avoid: (x, y, r) discs to keep out of, passed on to a Router that takes them."""
        if self.router is None:
            route = None
        elif avoid:
            route = self.router.route((start[0], start[1]), (goal[0], goal[1]), avoid)
        else:
            route = self.router.route((start[0], start[1]), (goal[0], goal[1]))
        if route is None:
            self.cancel()
            return False
        precise = set()
        if self.door_lines:
            free = getattr(self.router, "is_free", None)
            before = {tuple(q) for q in route}
            route = through_leaves(route, self.door_lines, free, avoid=avoid)
            precise = {tuple(q) for q in route} - before
        self.set_route(route[1:] or [tuple(goal[:2])], final_yaw)
        self.precise = precise
        self._prev = (float(start[0]), float(start[1]))
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
        self._prev = None

    def reset_progress(self) -> None:
        """The robot is being held on purpose (waiting for someone): the stuck clock starts again."""
        self._anchor = None
        self._yaws.clear()

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
        if self._prev is None:
            self._prev = (x, y)
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < (
                self.arrive_m if len(self.path) == 1 else
                (PRECISE_M if tuple(self.path[0]) in self.precise else self.waypoint_m)):
            self._prev = self.path.pop(0)
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
        tx, ty = self.carrot(x, y)
        bearing = math.atan2(ty - y, tx - x)
        err = abs(wrap(bearing - yaw))
        if self._in_place and err <= self.walk_again:
            self._in_place = False
        elif not self._in_place and err > self.turn_in_place:
            self._in_place = True
        face = self._face(bearing, yaw, settled)
        if self._in_place:
            return FollowerCommand(IDLE, 0.0, None, face, TURNING)
        slow = near_any(x, y, people, self.slow_radius) or (
            self.slow_near_doors and near_any(x, y, doors, self.slow_radius))
        if slow and self._in_door_zone(x, y):
            slow = False  # pushing a closer at 0.3 m/s wedges the G1 against the leaf: walking speed through a door
        speed = self.slow_speed if slow else self.open_speed
        return FollowerCommand(SLOW_WALK, speed, face, face, FOLLOWING, slow)


    def _in_door_zone(self, x: float, y: float, zone: float = DOOR_ZONE_M) -> bool:
        """Within ``zone`` of a swinging door's line (its leaves sweep about a metre each way)."""
        for (ax, ay), (bx, by) in self.door_lines:
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            if math.hypot(x - (ax + u * dx), y - (ay + u * dy)) < zone:
                return True
        return False

    def carrot(self, x: float, y: float) -> tuple[float, float]:
        """The point to steer at: ``lookahead`` ahead of the robot's foot on the current segment, at most its end."""
        bx, by = self.path[0]
        if self._prev is None or self.lookahead <= 0:
            return bx, by
        ax, ay = self._prev
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            return bx, by
        s = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
        seg = math.sqrt(seg2)
        s2 = s + self.lookahead / seg
        if s2 >= 1.0:
            return bx, by
        return ax + s2 * dx, ay + s2 * dy


DOOR_BEFORE_M = 2.0  # door spike: a straight segment through the centre of one leaf, from about 2 m before the door
DOOR_AFTER_M = 1.5
SINGLE_LEAF_MAX_M = 1.3  # a door narrower than this has one leaf


def _seg_cross(p, q, a, b):
    """Where segment p-q crosses segment a-b: (t along p-q, u along a-b), or None."""
    rx, ry = q[0] - p[0], q[1] - p[1]
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    t = ((a[0] - p[0]) * sy - (a[1] - p[1]) * sx) / den
    u = ((a[0] - p[0]) * ry - (a[1] - p[1]) * rx) / den
    return (t, u) if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def _disc_gap(p, q, avoid) -> float:
    """The smallest gap between segment p-q and the (x, y, r) discs (negative: it cuts one)."""
    best = float("inf")
    for x, y, r in avoid:
        dx, dy = q[0] - p[0], q[1] - p[1]
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - p[0]) * dx + (y - p[1]) * dy) / L2))
        best = min(best, math.hypot(x - (p[0] + u * dx), y - (p[1] + u * dy)) - r)
    return best


def through_leaves(route, doors, free=None, before: float = DOOR_BEFORE_M, after: float = DOOR_AFTER_M,
                   avoid=()):
    """A route (start first) with every crossing of a swinging door replaced by a straight pass through the centre of
    the leaf nearer the crossing: a point ``before`` metres ahead of the door line, then ``after`` beyond it (shorter
    when ``free(x, y)`` says the point is blocked). doors: ((ax, ay), (bx, by)) door lines."""
    if not route or not doors:
        return list(route or [])
    out = [tuple(route[0])]
    for q in route[1:]:
        p = out[-1]
        hits = []
        for a, b in doors:
            c = _seg_cross(p, q, a, b)
            if c is not None:
                hits.append((c[0], c[1], a, b))
        for t, u, a, b in sorted(hits):
            w = math.hypot(b[0] - a[0], b[1] - a[1])
            if w < 1e-6:
                continue
            nx, ny = -(b[1] - a[1]) / w, (b[0] - a[0]) / w
            if (q[0] - p[0]) * nx + (q[1] - p[1]) * ny < 0:
                nx, ny = -nx, -ny
            if w <= SINGLE_LEAF_MAX_M:
                uc = 0.5
            else:
                uc = 0.25 if u < 0.5 else 0.75
                if avoid:  # someone standing on that leaf's line (the person just guided): the other leaf
                    def gap(c):
                        cx_, cy_ = a[0] + c * (b[0] - a[0]), a[1] + c * (b[1] - a[1])
                        return _disc_gap((cx_ - before * nx, cy_ - before * ny), (cx_ + after * nx, cy_ + after * ny),
                                         avoid)
                    other = 1.0 - uc
                    if gap(uc) < 0.0 <= gap(other) or (gap(uc) < 0.0 and gap(other) > gap(uc)):
                        uc = other
            cx, cy = a[0] + uc * (b[0] - a[0]), a[1] + uc * (b[1] - a[1])
            pts = []
            for dist, sign in ((before, -1.0), (after, 1.0)):
                d = dist
                while d > 0.4:
                    x, y = cx + sign * d * nx, cy + sign * d * ny
                    if free is None or free(x, y):
                        break
                    d -= 0.25
                pts.append((cx + sign * max(d, 0.4) * nx, cy + sign * max(d, 0.4) * ny))
            out += pts
        out.append(tuple(q))
    return out


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
