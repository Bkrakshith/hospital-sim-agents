"""Typed view of config/site/hospital.json and its occupancy grid."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..config.loader import _find_root

SIDE_OFFSET = 0.6


def default_root(start: "str | Path | None" = None) -> Path:
    """Config root: the nearest parent holding personas.yaml (same walk as the scenario loader)."""
    start = Path(start) if start else Path(__file__).resolve()
    return _find_root(start / "x" if start.is_dir() else start)


@dataclass
class OccGrid:
    occ: np.ndarray  # uint8, 1 = blocked, indexed [iy, ix]
    res: float
    origin: tuple[float, float]

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((y - self.origin[1]) / self.res)), int(math.floor((x - self.origin[0]) / self.res))

    def to_world(self, iy: int, ix: int) -> tuple[float, float]:
        return self.origin[0] + (ix + 0.5) * self.res, self.origin[1] + (iy + 0.5) * self.res

    def in_grid(self, iy: int, ix: int) -> bool:
        return 0 <= iy < self.occ.shape[0] and 0 <= ix < self.occ.shape[1]

    def blocked(self, x: float, y: float) -> bool:
        iy, ix = self.to_cell(x, y)
        return (not self.in_grid(iy, ix)) or bool(self.occ[iy, ix])


@dataclass
class Room:
    id: str
    label: str
    use: str
    compartment: Optional[str]
    poly: list[tuple[float, float]]

    def contains(self, x: float, y: float) -> bool:
        ok = False
        p = self.poly
        for i in range(len(p)):
            x0, y0 = p[i]
            x1, y1 = p[(i + 1) % len(p)]
            if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
                ok = not ok
        return ok

    @property
    def area(self) -> float:
        p = self.poly
        return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2

    @property
    def centre(self) -> tuple[float, float]:
        xs = [q[0] for q in self.poly]
        ys = [q[1] for q in self.poly]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


@dataclass
class Bed:
    id: str
    room: str
    x: float
    y: float
    yaw: float

    def side_point(self, side: Literal["left", "right"]) -> tuple[float, float]:
        """0.6 m beside the bed's mid-length; left is the bed's left when lying along +yaw."""
        sign = 1.0 if side == "left" else -1.0
        return (self.x - sign * SIDE_OFFSET * math.sin(self.yaw), self.y + sign * SIDE_OFFSET * math.cos(self.yaw))


@dataclass
class Landmark:
    id: str
    x: float
    y: float
    label: str = ""


@dataclass
class Seat:
    id: str
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class Furniture:
    id: str
    kind: str
    x: float
    y: float
    yaw: float = 0.0
    w: float = 0.0
    d: float = 0.0
    h: float = 0.0


@dataclass
class Site:
    rooms: dict[str, Room]
    beds: dict[str, Bed]
    landmarks: dict[str, Landmark]
    seats: dict[str, Seat]
    furniture: dict[str, Furniture]
    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    grid: OccGrid
    walls: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    compartments: list = field(default_factory=list)
    lifts: list = field(default_factory=list)
    stairs: list = field(default_factory=list)

    def room_at(self, x: float, y: float) -> Optional[str]:
        hits = [r for r in self.rooms.values() if r.contains(x, y)]
        return min(hits, key=lambda r: r.area).id if hits else None

    def resolve(self, place: str) -> Optional[tuple[float, float]]:
        """Kinds: landmark, seat, room, bed, furniture. A bare id tries them in that order."""
        kinds = ("landmark", "seat", "room", "bed", "furniture")
        if ":" in place:
            kind, _, name = place.partition(":")
            if kind not in kinds:
                return None
            order = (kind,)
        else:
            kind, name, order = "", place, kinds
        for k in order:
            if k == "landmark" and name in self.landmarks:
                m = self.landmarks[name]
                return (m.x, m.y)
            if k == "seat" and name in self.seats:
                return (self.seats[name].x, self.seats[name].y)
            if k == "room" and name in self.rooms:
                c = self.landmarks.get(f"{name}_centre")
                return (c.x, c.y) if c else self.rooms[name].centre
            if k == "bed" and name in self.beds:
                return self.beds[name].side_point("right")
            if k == "furniture" and name in self.furniture:
                return (self.furniture[name].x, self.furniture[name].y)
        return None

    def check_place(self, x: float, y: float, *, occupied_beds=()) -> Optional[str]:
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return "outside the building"
        iy, ix = self.grid.to_cell(x, y)
        if not self.grid.in_grid(iy, ix):
            return "outside the building"
        for bid in occupied_beds:  # the bed itself is blocked in the grid, so test it before walls
            b = self.beds.get(bid)
            if b and math.hypot(x - b.x, y - b.y) < 0.8:
                return f"bed {bid} already has a patient"
        if self.grid.occ[iy, ix]:
            return "inside a wall"
        return None

    def summary_for_prompt(self, known_rooms: list[str]) -> str:
        lines = []
        for rid in known_rooms:
            r = self.rooms.get(rid)
            if r:
                lines.append(f"- {rid}: {r.label} ({r.use})")
        return "Places you know:\n" + "\n".join(lines) if lines else "You do not know any named places."


def load_site(root: "str | Path") -> Site:
    """Read root/site/hospital.json and hospital_occ.npz (root is the config directory)."""
    d = Path(root) / "site"
    j = json.loads((d / "hospital.json").read_text())
    z = np.load(d / "hospital_occ.npz")
    grid = OccGrid(np.asarray(z["occ"], dtype=np.uint8), float(z["res"]), (float(z["origin"][0]), float(z["origin"][1])))
    return Site(
        rooms={r["id"]: Room(r["id"], r.get("label", r["id"]), r.get("use", ""), r.get("compartment"),
                             [tuple(p) for p in r["poly"]]) for r in j["rooms"]},
        beds={b["id"]: Bed(b["id"], b["room"], b["x"], b["y"], b["yaw"]) for b in j["beds"]},
        landmarks={k: Landmark(k, v["x"], v["y"], v.get("label", "")) for k, v in j["landmarks"].items()},
        seats={s["id"]: Seat(s["id"], s["x"], s["y"], s.get("yaw", 0.0)) for s in j["seats"]},
        furniture={f["id"]: Furniture(f["id"], f["kind"], f["x"], f["y"], f.get("yaw", 0.0), f.get("w", 0.0),
                                      f.get("d", 0.0), f.get("h", 0.0)) for f in j["furniture"]},
        bounds=tuple(j["bounds"]), grid=grid, walls=j.get("walls", []), doors=j.get("doors", []),
        compartments=j.get("compartments", []), lifts=j.get("lifts", []), stairs=j.get("stairs", []),
    )
