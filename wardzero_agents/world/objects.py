"""World objects: items, fires and call bells. Pure Python (no ROS, no stub-world state), so an
Isaac Sim host can reuse it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from wardzero_agents.contracts import ObjectState

ACCEPT_WINDOW_S = 10.0
SURGEON_WINDOW_S = 5.0
EXTINGUISH_RANGE_M = 2.0
ROBOT_EXTINGUISH_PER_S = 0.1
PERSON_EXTINGUISH_PER_S = 0.08
FIRE_START_SIZE = 0.1


@dataclass
class Item:
    id: str
    kind: Literal["tray", "specimen_box", "instrument", "extinguisher"]
    label: str = ""
    owner: Optional[str] = None
    holder: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Fire:
    id: str
    x: float
    y: float
    size: float = FIRE_START_SIZE
    grow_per_s: float = 0.01
    out: bool = False
    alarmed: bool = False

    def grow(self, dt: float) -> bool:
        """Advance dt seconds. Returns True the first time the fire reaches full size."""
        if self.out:
            return False
        self.size = min(1.0, self.size + self.grow_per_s * dt)
        if self.size >= 1.0 and not self.alarmed:
            self.alarmed = True
            return True
        return False

    def reduce(self, amount: float) -> None:
        if self.out:
            return
        self.size = max(0.0, self.size - amount)
        if self.size <= 0.0:
            self.out = True


@dataclass
class CallBell:
    bed_id: str
    active: bool = False
    since: Optional[float] = None


@dataclass
class WorldObjects:
    items: dict[str, Item] = field(default_factory=dict)
    fires: dict[str, Fire] = field(default_factory=dict)
    bells: dict[str, CallBell] = field(default_factory=dict)
    _fire_n: int = 0

    def add_item(self, item: Item) -> Item:
        self.items[item.id] = item
        return item

    def start_fire(self, x: float, y: float, size: float = FIRE_START_SIZE,
                   grow_per_s: float = 0.01) -> Fire:
        self._fire_n += 1
        f = Fire(f"fire_{self._fire_n}", x, y, size, grow_per_s)
        self.fires[f.id] = f
        return f

    def ring(self, bed_id: str, t: float) -> CallBell:
        b = self.bells.setdefault(bed_id, CallBell(bed_id))
        b.active, b.since = True, t
        return b

    def nearest_fire(self, x: float, y: float, within: float = EXTINGUISH_RANGE_M) -> Optional[Fire]:
        best, bd = None, within
        for f in self.fires.values():
            d = math.hypot(f.x - x, f.y - y)
            if not f.out and d <= bd:
                best, bd = f, d
        return best

    def step(self, dt: float) -> list[Fire]:
        """Grow every fire. Returns the fires that just reached full size (they raise the alarm)."""
        return [f for f in self.fires.values() if f.grow(dt)]

    def snapshot(self) -> list[ObjectState]:
        out = [ObjectState(id=i.id, kind=i.kind, label=i.label, owner=i.owner, holder=i.holder,
                           x=i.x, y=i.y, z=i.z) for i in self.items.values()]
        out += [ObjectState(id=f.id, kind="fire", x=f.x, y=f.y, size=f.size, active=not f.out)
                for f in self.fires.values()]
        out += [ObjectState(id=f"bell_{b.bed_id}", kind="call_bell", label=b.bed_id, active=b.active)
                for b in self.bells.values()]
        return out
