"""Mobile robot body for the stub world: a unicycle on the occupancy grid. Pure Python.

It is the fast test double for the Isaac robot: /cmd_vel drives it, /goal_pose sends it along a
Router route, walls and furniture stop it, and a person inside radius + CONTACT_MARGIN_M is a
contact. Isaac replaces it with the real controller; the topics stay the same.
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

    def set_goal(self, x: float, y: float, yaw: Optional[float] = None) -> bool:
        """Follow a Router route to (x, y) at max_speed. False when there is none."""
        self._v = self._w = 0.0
        if self.router is None:
            return False
        route = self.router.route((self.x, self.y), (x, y))
        if route is None:
            return False
        self._path = [tuple(p) for p in route[1:]] or [(float(x), float(y))]
        self._final_yaw = yaw
        self._had_goal = True
        self.goal_event = None
        self._anchor, self._anchor_t = (self.x, self.y), 0.0
        return True

    def cancel_goal(self) -> None:
        """Dropped on purpose (new command): no goal event."""
        self._path = []
        self._final_yaw = None
        self._had_goal = False

    # ---- motion --------------------------------------------------------------------------
    def _move(self, dist: float, grid) -> float:
        """Move dist metres along the heading in small steps; stop before a blocked cell."""
        done = 0.0
        sign = 1.0 if dist >= 0 else -1.0
        left = abs(dist)
        while left > 1e-9:
            d = min(SUBSTEP_M, left)
            nx, ny = self.x + math.cos(self.yaw) * d * sign, self.y + math.sin(self.yaw) * d * sign
            if grid.blocked(nx, ny):
                break
            self.x, self.y = nx, ny
            done += d
            left -= d
        return done * sign

    def _follow(self, dt: float) -> tuple[float, float]:
        """Commanded (v, w) that steers towards the next waypoint."""
        while self._path and math.hypot(self._path[0][0] - self.x, self._path[0][1] - self.y) < (
                ARRIVE_M if len(self._path) == 1 else WAYPOINT_M):
            self._path.pop(0)
        if self._path:
            tx, ty = self._path[0]
            err = _wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
            w = max(-TURN_RATE, min(TURN_RATE, err / max(dt, 1e-3)))
            v = self.max_speed * max(0.0, math.cos(err)) if abs(err) < 1.2 else 0.0
            dist = math.hypot(tx - self.x, ty - self.y)
            return min(v, dist / max(dt, 1e-3)), w
        if self._final_yaw is not None:
            err = _wrap(self._final_yaw - self.yaw)
            if abs(err) < 0.05:
                self._final_yaw = None
                return 0.0, 0.0
            return 0.0, max(-TURN_RATE, min(TURN_RATE, err / max(dt, 1e-3)))
        return 0.0, 0.0

    def step(self, dt: float, grid, people: Optional[Iterable[tuple[str, float, float]]] = None) -> list[str]:
        """Advance dt seconds. people = (id, x, y) of everyone to check contact against.
        Returns the ids that just came into contact (once per touch, not every step)."""
        if dt > 0:
            if self.has_goal:
                v, w = self._follow(dt)
            else:
                v, w = self._v, self._w
            self.yaw = _wrap(self.yaw + w * dt)
            moved = self._move(v * dt, grid)
            self.vx = moved / dt
            self.wz = w
            if self._had_goal:
                if not self.has_goal:
                    self._had_goal = False
                    self.goal_event = "arrived"
                else:
                    self._anchor_t += dt
                    if math.hypot(self.x - self._anchor[0], self.y - self._anchor[1]) >= STUCK_M:
                        self._anchor, self._anchor_t = (self.x, self.y), 0.0
                    elif self._anchor_t >= STUCK_S:
                        self.cancel_goal()
                        self.goal_event = "stuck"
        new: list[str] = []
        if people is not None:
            near = {pid for pid, px, py in people
                    if math.hypot(px - self.x, py - self.y) <= self.radius + CONTACT_MARGIN_M}
            new = sorted(near - self._touching)
            self._touching = near
        return new
