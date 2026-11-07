"""Mobile robot body for the stub world: a unicycle on the occupancy grid. Pure Python.

It is the fast test double for the Isaac robot: /cmd_vel drives it, /goal_pose sends it along a
Router route, walls and furniture stop it, and a person inside radius + CONTACT_MARGIN_M is a
contact when the robot's own motion brought it there. Isaac replaces it with the real controller; the topics stay
the same.

A step of any length behaves like many short ones (at most MAX_SUBSTEP_S each): a long sim step (a stalled wall step
at x10 is 5 s of sim) turns and then walks, as short steps would, instead of spending the whole step turning.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

CONTACT_MARGIN_M = 0.25
SUBSTEP_M = 0.04  # never move further than this between blocked-cell checks
TURN_RATE = 3.0  # rad/s while following a route
STUCK_S = 5.0  # a goal with less than STUCK_M progress for this long is given up
STUCK_M = 0.1
ARRIVE_M = 0.12
WAYPOINT_M = 0.15
MAX_SUBSTEP_S = 0.1  # a step is cut into pieces no longer than this
TURN_PROGRESS_RAD = 1e-3  # turning on the spot towards the route by more than this is progress, not being stuck
REPLAN_S = 0.5  # on a route, blocked this long (no slide either): plan again from where it stands
CLOSE_EPS_M = 1e-3  # a contact is the robot's when its own motion closed the gap by more than this


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class RobotBody:
    def __init__(self, x: float, y: float, yaw: float, radius: float = 0.3, max_speed: float = 1.0,
                 router=None):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.radius = radius
        self.max_speed = max_speed
        self.router = router
        self._v = 0.0  # commanded forward speed
        self._w = 0.0
        self.vx = 0.0  # measured, body frame
        self.wz = 0.0
        self._path: list[tuple[float, float]] = []
        self._final_yaw: Optional[float] = None
        self._touching: set[str] = set()
        self._had_goal = False
        self.goal_event: Optional[str] = None  # "arrived" | "stuck", read once by take_goal_event
        self._anchor = (self.x, self.y)
        self._anchor_t = 0.0
        self._goal_xy: Optional[tuple[float, float]] = None  # where the route goes, to plan again when blocked
        self._avoid: tuple = ()  # the avoid discs the goal was set with: a new plan keeps out of them too
        self._blocked_t = 0.0
        # set by the task layer each step while a route is followed: 0 holds the robot (waiting for the person it
        # guides; the stuck timer does not run), a cap slows it (near a person)
        self.speed_scale = 1.0
        self.speed_cap: Optional[float] = None

    def take_goal_event(self) -> Optional[str]:
        ev, self.goal_event = self.goal_event, None
        return ev

    # ---- state ---------------------------------------------------------------------------
    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    @property
    def speed(self) -> float:
        return abs(self.vx)

    @property
    def has_goal(self) -> bool:
        return bool(self._path) or self._final_yaw is not None

    def teleport(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.cancel_goal()
        self._v = self._w = self.vx = self.wz = 0.0

    # ---- commands ------------------------------------------------------------------------
    def set_cmd_vel(self, v: float, w: float) -> None:
        """Forward speed (m/s, clamped to max_speed) and turn rate (rad/s). Cancels a goal."""
        self.cancel_goal()
        self._v = max(-self.max_speed, min(self.max_speed, float(v)))
        self._w = float(w)

    def set_goal(self, x: float, y: float, yaw: Optional[float] = None, avoid=()) -> bool:
        """Follow a Router route to (x, y) at max_speed. False when there is none. avoid: (x, y, r) discs the route
        keeps out of if it can (people standing in the way)."""
        self._v = self._w = 0.0
        if self.router is None:
            return False
        route = self.router.route((self.x, self.y), (x, y), avoid) if avoid else self.router.route((self.x, self.y), (x, y))
        if route is None:
            return False
        self._path = [tuple(p) for p in route[1:]] or [(float(x), float(y))]
        self._final_yaw = yaw
        self._had_goal = True
        self.goal_event = None
        self._anchor, self._anchor_t = (self.x, self.y), 0.0
        self._goal_xy, self._blocked_t = (float(x), float(y)), 0.0
        self._avoid = tuple(avoid or ())
        return True

    def turn_to(self, yaw: float) -> None:
        """Turn on the spot to yaw. Not a goal: no goal event when it is done."""
        self._v = self._w = 0.0
        self._path = []
        self._final_yaw = float(yaw)
        self._had_goal = False

    def cancel_goal(self) -> None:
        """Dropped on purpose (new command): no goal event."""
        self._path = []
        self._final_yaw = None
        self._had_goal = False
        self._goal_xy = None
        self._avoid = ()

    # ---- motion --------------------------------------------------------------------------
    def _move(self, dist: float, grid, slide: bool = False) -> float:
        """Move dist metres along the heading in small steps; stop before a blocked cell. Returns the distance made
        good (signed). slide (on a route): where the heading is blocked, move along whichever axis is still free (along
        the wall), and a body that is inside a blocked cell already may walk on out of it."""
        done = 0.0
        sign = 1.0 if dist >= 0 else -1.0
        left = abs(dist)
        c, s_ = math.cos(self.yaw) * sign, math.sin(self.yaw) * sign
        while left > 1e-9:
            d = min(SUBSTEP_M, left)
            nx, ny = self.x + c * d, self.y + s_ * d
            if grid.blocked(nx, ny) and not (slide and grid.blocked(self.x, self.y)):
                if not slide:
                    break
                # along the wall: the free axis that keeps the most of the step
                for ax, ay in sorted(((c * d, 0.0), (0.0, s_ * d)), key=lambda v: -abs(v[0] + v[1])):
                    if abs(ax + ay) > 0.25 * d and not grid.blocked(self.x + ax, self.y + ay):
                        nx, ny = self.x + ax, self.y + ay
                        break
                else:
                    break
            step = math.hypot(nx - self.x, ny - self.y)
            self.x, self.y = nx, ny
            done += step
            left -= d
        return done * sign

    def _heading_error(self) -> Optional[float]:
        """How far the heading is from the next waypoint (rad), or None with no route."""
        if not self._path:
            return None
        tx, ty = self._path[0]
        return abs(_wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw))

    def _follow(self, dt: float) -> tuple[float, float]:
        """Commanded (v, w) that steers towards the next waypoint."""
        while self._path and math.hypot(self._path[0][0] - self.x, self._path[0][1] - self.y) < (
                ARRIVE_M if len(self._path) == 1 else WAYPOINT_M):
            self._path.pop(0)
        if self._path:
            tx, ty = self._path[0]
            err = _wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
            w = max(-TURN_RATE, min(TURN_RATE, err / max(dt, 1e-3)))
            top = self.max_speed * max(0.0, min(1.0, self.speed_scale))
            if self.speed_cap is not None:
                top = min(top, self.speed_cap)
            v = top * max(0.0, math.cos(err)) if abs(err) < 1.2 else 0.0
            dist = math.hypot(tx - self.x, ty - self.y)
            return min(v, dist / max(dt, 1e-3)), w
        if self._final_yaw is not None:
            err = _wrap(self._final_yaw - self.yaw)
            if abs(err) < 0.05:
                self._final_yaw = None
                return 0.0, 0.0
            return 0.0, max(-TURN_RATE, min(TURN_RATE, err / max(dt, 1e-3)))
        return 0.0, 0.0

    def _replan(self) -> None:
        """Blocked on the route (no slide either): a new route from where the body stands, if there is one, round the
        same avoid discs the goal was set with."""
        if self.router is None or self._goal_xy is None:
            return
        a = (self.x, self.y)
        route = self.router.route(a, self._goal_xy, self._avoid) if self._avoid else self.router.route(a, self._goal_xy)
        if route is not None:
            self._path = [tuple(p) for p in route[1:]] or [self._goal_xy]

    def _substep(self, dt: float, grid) -> tuple[float, float]:
        """One piece of a step (dt <= MAX_SUBSTEP_S): turn, move, and keep the goal's progress clock. Returns (distance
        moved, turn rate)."""
        on_route = self.has_goal
        if on_route:
            v, w = self._follow(dt)
            err0 = self._heading_error()
        else:
            v, w = self._v, self._w
            err0 = None
        self.yaw = _wrap(self.yaw + w * dt)
        moved = self._move(v * dt, grid, slide=on_route and bool(self._path))
        if on_route and v > 0 and abs(moved) < 1e-9 and self._path:
            self._blocked_t += dt
            if self._blocked_t >= REPLAN_S:
                self._blocked_t = 0.0
                self._replan()
        else:
            self._blocked_t = 0.0
        if self._had_goal:
            if not self.has_goal:
                self._had_goal = False
                self._goal_xy = None
                self.goal_event = "arrived"
            elif self.speed_scale <= 0.0:  # held on purpose: waiting is not being stuck
                self._anchor, self._anchor_t = (self.x, self.y), 0.0
            else:
                err1 = self._heading_error()
                turning = (abs(moved) < 1e-9 and err0 is not None and err1 is not None
                           and err1 < err0 - TURN_PROGRESS_RAD)
                self._anchor_t += dt
                if turning or math.hypot(self.x - self._anchor[0], self.y - self._anchor[1]) >= STUCK_M:
                    self._anchor, self._anchor_t = (self.x, self.y), 0.0
                elif self._anchor_t >= STUCK_S:
                    self.cancel_goal()
                    self.goal_event = "stuck"
        return moved, w

    def step(self, dt: float, grid, people: Optional[Iterable[tuple[str, float, float]]] = None) -> list[str]:
        """Advance dt seconds, in pieces of at most MAX_SUBSTEP_S. people = (id, x, y) of everyone to check contact
        against, where they stand now. Returns the ids that just came into contact through the robot's own motion
        (once per touch, not every step): a person who walks into a standing or retreating robot is not one."""
        x0, y0 = self.x, self.y
        if dt > 0:
            n = max(1, int(math.ceil(dt / MAX_SUBSTEP_S - 1e-9)))
            h = dt / n
            moved = turned = 0.0
            for _ in range(n):
                m, w = self._substep(h, grid)
                moved += m
                turned += w * h
            self.vx = moved / dt
            self.wz = turned / dt
        if people is None:
            return []
        return self._contacts(people, (x0, y0))

    def _contacts(self, people: Iterable[tuple[str, float, float]], before: tuple[float, float]) -> list[str]:
        """People inside the contact distance whom the robot's own motion this step brought closer (each once, until
        they are out of reach again)."""
        reach = self.radius + CONTACT_MARGIN_M
        near: set[str] = set()
        new: list[str] = []
        for pid, px, py in people:
            d = math.hypot(px - self.x, py - self.y)
            if d > reach:
                continue
            near.add(pid)
            if pid not in self._touching and d < math.hypot(px - before[0], py - before[1]) - CLOSE_EPS_M:
                new.append(pid)
        self._touching = (self._touching & near) | set(new)
        return sorted(new)
