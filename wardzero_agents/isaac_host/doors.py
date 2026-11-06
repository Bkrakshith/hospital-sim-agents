"""Door physics for the Isaac host, the parts that need no Kit: which doors swing, their closer, the leaf geometry, and
the people's capsules. ``doors_isaac`` authors and drives them.

Design §8 and the door spike (``research_notes/isaac_host/door_spike_report.md``): a door on the run's routes becomes a
double-action hinged pair (or single leaf) with a closer, pushed open by real PhysX contact. People get kinematic
capsules that follow their characters every physics step; the walking G1 pushes with its body. The final values of
the spike:

* leaves start closed (joint state 0), 40 kg (inertia scaled), hinge limits +-100 deg;
* the closer: an angular force drive to 0 with stiffness 18.2 N m/rad (30 N at the leading edge holds 90 deg) and
  direction-dependent damping, 5 N m s/rad while the leaf swings open and 30 while it closes (switched every step);
* joint speed cap 3 rad/s, leaf max depenetration velocity 1 m/s, articulation solver iterations 16/4;
* the door root is out of the NavMesh, and the leaves' copies are out of the baked collision USD (a static copy of a
  leaf would block the moving one);
* people: capsule radius 0.25 m from z 0.45 to 1.50, kinematic, in a collision group that filters itself, the robot
  proxy and the robot; every agent ignores the door roots, the leaves and the capsules in both avoidance lists; a
  person within 1 m of a leaf walks at 0.8 m/s.

Doors that are not on the routes stay as drawn (held open) and static: they only need to be passable. The old reason
for the static Compartment leaves (a rotate op against the joint frame) was wrong: the baked Junction side panels
overlap the hinge end of ``Compartment/Leaf1``; their baked meshes are left out (``JUNCTION_PANELS``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

DOOR_ROOTS = ("/World/Fixtures/Doors", "/World/Fixtures/Doors1")
LEAF_NAMES = ("Leaf0", "Leaf1")
MASS_KG = 40.0
LIMIT_DEG = 100.0
STIFFNESS = 18.2  # N m / rad
DAMPING_CLOSE = 30.0  # N m s / rad while the leaf swings back to closed
DAMPING_OPEN = 5.0  # ... and while it swings open (a person or the G1 pushing it)
MAX_SPEED = 3.0  # rad/s
MAX_FORCE = 1.0e4
LEAF_DEPENETRATION = 1.0  # m/s
SOLVER_ITERATIONS = (16, 4)

CAPSULES_ROOT = "/World/PeopleCapsules"
GROUPS_ROOT = "/World/CollisionGroups"
CAPSULE_R, CAPSULE_Z0, CAPSULE_Z1 = 0.25, 0.45, 1.50
CAPSULE_MASS = 70.0
CAPSULE_CONTACT_OFFSET = 0.04
SPARE_CAPSULES = 4  # for people who join after the start
PARK = (-300.0, -300.0)  # where an unused capsule waits, far from everything
VEL_EMA = 0.5  # weight of the newest frame in a capsule's velocity estimate
MAX_PERSON_SPEED = 3.0  # m/s: a frame-to-frame jump faster than this (a teleport) counts as standing still
SLOW_SPEED = 0.8  # m/s within SLOW_RADIUS of a leaf
SLOW_RADIUS = 1.0
DOOR_MARGIN_M = 0.5  # a door counts as on the routes when its middle is this close to a kept room

# baked collision meshes the moving leaves must not meet: every leaf's own copy, and the Junction side panels whose start
# sits on Compartment/Leaf1's hinge axis (the cause of that pair's non-finite poses)
BAKED_LEAF = re.compile(r"_Fixtures_Doors\d*_.*_Leaf\d")
JUNCTION_PANELS = re.compile(r"Junction_CompartmentSidePanel_b[0-2]$")


def per_degree(v: float) -> float:
    """USD angular drive gains are per degree; the spike's values are per radian."""
    return v * math.pi / 180.0


def closed_yaw(authored_yaw_deg: float, joint_state_deg: float) -> float:
    """The leaf's yaw (deg) with its joint at 0, from the yaw it is drawn with and its authored joint state: the stage
    draws the leaves held open (state +-90)."""
    y = authored_yaw_deg - joint_state_deg
    return (y + 180.0) % 360.0 - 180.0


def closer_damping(q: float, w: float) -> float:
    """Swinging away from closed (q and w of one sign): the light opening damping; else the closing one."""
    return DAMPING_OPEN if q * w > 0 else DAMPING_CLOSE


def baked_off(names: Iterable[str]) -> list[str]:
    """The baked collision prims (names under the collision USD's root) to leave out."""
    return sorted(n for n in names if BAKED_LEAF.search(n) or JUNCTION_PANELS.search(n))


def route_doors(site, rects: Optional[dict], margin: float = DOOR_MARGIN_M) -> list[str]:
    """Site doors whose middle is within ``margin`` of a kept room (all of them for a full-site run)."""
    out = []
    for d in getattr(site, "doors", None) or []:
        try:
            mx, my = (d["a"][0] + d["b"][0]) / 2.0, (d["a"][1] + d["b"][1]) / 2.0
        except (KeyError, TypeError, IndexError):
            continue
        if not rects or any(r[0] - margin <= mx <= r[2] + margin and r[1] - margin <= my <= r[3] + margin
                            for r in rects.values()):
            out.append(str(d["id"]))
    return out


@dataclass
class Leaf:
    """A leaf in the floor plane: hinge (x, y), yaw (deg) when closed, collider length along x from x0 to x1 and
    half thickness. Its yaw at joint angle q (rad) is closed_yaw + q."""
    door: str
    name: str
    hinge: tuple[float, float]
    closed_yaw_deg: float
    x0: float = 0.005
    x1: float = 0.96
    half_t: float = 0.027

    def dist(self, q: float, px: float, py: float) -> float:
        """Distance from (px, py) to the leaf's rectangle, negative inside."""
        hx, hy = self.hinge
        yaw = math.radians(self.closed_yaw_deg) + q
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (px - hx) + s * (py - hy)
        ly = -s * (px - hx) + c * (py - hy)
        cx, hxl = (self.x0 + self.x1) / 2.0, (self.x1 - self.x0) / 2.0
        dx, dy = abs(lx - cx) - hxl, abs(ly) - self.half_t
        if dx <= 0 and dy <= 0:
            return max(dx, dy)
        return math.hypot(max(dx, 0.0), max(dy, 0.0))


class CapsuleTracker:
    """Each person's capsule: which capsule (a stable slot), where it was at the last frame and how fast it went.
    ``targets(j, dt)`` is each slot's position for sub-step j of the next frame (it leads the drawn body by up to a
    frame, as the spike did); unused slots stay parked."""

    def __init__(self, n_slots: int, ema: float = VEL_EMA, max_speed: float = MAX_PERSON_SPEED):
        self.n = n_slots
        self.ema, self.max_speed = ema, max_speed
        self.slot: dict[str, int] = {}
        self.pos = [PARK] * n_slots
        self.vel = [(0.0, 0.0)] * n_slots
        self.overflow: set[str] = set()

    def assign(self, pid: str) -> Optional[int]:
        if pid in self.slot:
            return self.slot[pid]
        free = [i for i in range(self.n) if i not in self.slot.values()]
        if not free:
            self.overflow.add(pid)
            return None
        self.slot[pid] = free[0]
        return free[0]

    def frame(self, poses: dict, dt: float) -> None:
        """poses: person id -> (x, y), or None for someone who has no body now (gone, in bed: their capsule is parked)."""
        for pid, xy in poses.items():
            i = self.assign(pid)
            if i is None:
                continue
            if xy is None:
                self.pos[i], self.vel[i] = PARK, (0.0, 0.0)
                continue
            px, py = self.pos[i]
            if self.pos[i] == PARK or dt <= 0:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = (xy[0] - px) / dt, (xy[1] - py) / dt
                if math.hypot(vx, vy) > self.max_speed:
                    vx, vy = 0.0, 0.0
                ox, oy = self.vel[i]
                vx, vy = self.ema * vx + (1 - self.ema) * ox, self.ema * vy + (1 - self.ema) * oy
            self.pos[i], self.vel[i] = (float(xy[0]), float(xy[1])), (vx, vy)

    def targets(self, j: int, dt: float) -> list[tuple[float, float]]:
        return [(x + vx * j * dt, y + vy * j * dt) if (x, y) != PARK else PARK
                for (x, y), (vx, vy) in zip(self.pos, self.vel)]


def near_leaf(leaves: Sequence[Leaf], qs: Sequence[float], px: float, py: float, radius: float = SLOW_RADIUS) -> bool:
    return any(lf.dist(q, px, py) < radius for lf, q in zip(leaves, qs))
