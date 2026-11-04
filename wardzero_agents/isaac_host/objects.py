"""World objects to stage prims: where each item is drawn, how big a fire's emitter is, which call lights are lit.

The StubWorld owns the logic (``world/objects.py``: items, fires, call bells; ``world/reach.py``); Isaac only mirrors it.
``ObjectMirror`` reads ``world.objects`` once a frame and tells a stage what to change, writing only what changed. The
stage is anything with the methods used below: ``objects_isaac.UsdObjectStage`` in the host, a recording fake in the
tests. Pure Python (no pxr, no Kit).

Where an item is drawn:
- on the floor or a surface (no holder): at its (x, y, z). An extinguisher stands on the floor: its z is the height
  of the handle the robot grabs, not where the body sits;
- held by the robot or by a person who is up (standing, walking or seated): in front of the holder, ``CARRY_AHEAD_M``
  ahead of them at hand height (at least ``HANDS_MIN_Z_M``). The world's x, y of a held item is only where it was handed over;
- held by a patient lying in a bed: where the hand-over put it (``world.hand_over`` writes the place into the item), on
  the bed within the patient's reach;
- held by someone who has left: hidden.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..world.stub_world import StubWorld

log = logging.getLogger("wardzero.isaac.objects")

ROBOT_HOLDER = "g1"
CARRY_AHEAD_M = 0.35
HANDS_MIN_Z_M = 0.95
BED_HALF_LENGTH_M = 1.277  # bed centre to the head end (the site map's "head" point)
BELL_Z_M = 1.8  # the call light is on the wall above the head of the bed
BELL_WALL_GAP_M = 0.02  # ... this far beyond the head end (the head end touches the wall's room face; the lamp stands off it)
FIRE_STEP = 0.02  # a fire's size is rewritten when it moved by at least this much
INSTRUMENT_ASSETS = ("scalpel_handle", "scissors", "adson_forceps", "angled_tweezers", "needle_driver", "veress_needle", "trocar")
_KEYWORDS = (("scalpel", "scalpel_handle"), ("scissor", "scissors"), ("adson", "adson_forceps"), ("forcep", "adson_forceps"),
             ("tweezer", "angled_tweezers"), ("needle driver", "needle_driver"), ("driver", "needle_driver"),
             ("veress", "veress_needle"), ("trocar", "trocar"))


def instrument_asset(label: str) -> str:
    """The instrument model a label names (a stable pick from the list when it names none)."""
    low = (label or "").lower()
    for word, name in _KEYWORDS:
        if word in low:
            return name
    return INSTRUMENT_ASSETS[sum(map(ord, low)) % len(INSTRUMENT_ASSETS)]


def bed_head(bed) -> tuple[float, float]:
    return bed.x + math.cos(bed.yaw) * BED_HALF_LENGTH_M, bed.y + math.sin(bed.yaw) * BED_HALF_LENGTH_M


@dataclass(frozen=True)
class FireParams:
    """What a fire of a given size asks of its emitter (Flow: a sphere emitter; a fallback proxy scales with radius)."""
    enabled: bool
    radius: float  # m
    fuel: float
    temperature: float


def fire_params(size: float) -> FireParams:
    """Size 0..1 -> an emitter. Out (size <= 0) is disabled. A 0.1 fire is a small flame, a full one fills a corridor."""
    if size <= 0.0:
        return FireParams(False, 0.0, 0.0, 0.0)
    s = min(1.0, size)
    return FireParams(True, round(0.10 + 0.30 * s, 3), round(0.5 + 0.4 * s, 3), round(1.5 + 1.5 * s, 3))


class ObjectMirror:
    def __init__(self, world: "StubWorld", stage):
        self.world, self.stage = world, stage
        self._items: dict[str, tuple] = {}  # key -> (pose, visible) last written
        self._fires: dict[str, tuple] = {}  # key -> (x, y, size step) while burning
        self._gone_fires: set[str] = set()
        self._bells: dict[str, bool] = {}
        self._dead: set[str] = set()  # items that could not be added
        self._reported: set[str] = set()  # what has failed (and was logged once)

    # ---- the pure mapping --------------------------------------------------------------------
    def item_pose(self, item) -> tuple[bool, float, float, float, float]:
        """(visible, x, y, z, yaw) of an item from the world's state."""
        w = self.world
        if item.holder is None:
            return True, item.x, item.y, 0.0 if item.kind == "extinguisher" else item.z, 0.0
        if item.holder == ROBOT_HOLDER:
            return self._in_hands(item, w.robot.x, w.robot.y, w.robot.yaw)
        p = w.people.get(item.holder)
        if p is None or p.posture == "gone":
            return False, item.x, item.y, item.z, 0.0
        if p.posture == "lying" and p.bed is not None:
            return True, item.x, item.y, item.z, p.yaw
        return self._in_hands(item, p.x, p.y, p.yaw)

    @staticmethod
    def _in_hands(item, x: float, y: float, yaw: float):
        return (True, x + math.cos(yaw) * CARRY_AHEAD_M, y + math.sin(yaw) * CARRY_AHEAD_M,
                max(item.z, HANDS_MIN_Z_M), yaw)

    # ---- once a frame ------------------------------------------------------------------------
    def _guard(self, what: str, fn, *args) -> bool:
        """Run one stage call; a failure is logged once per (what) and the rest of the frame goes on. True when it worked."""
        try:
            fn(*args)
            return True
        except Exception:
            if what not in self._reported:
                self._reported.add(what)
                log.exception("drawing %s failed; it is skipped from now on", what)
            return False

    def sync(self) -> None:
        """Mirror every item, bell and fire. One failing object never stops the others."""
        objs = self.world.objects
        for item in list(objs.items.values()):
            if item.id in self._dead:
                continue
            if item.id not in self._items:
                if not self._guard(f"item {item.id}", self.stage.add_item, item.id, item.kind, item.label):
                    self._dead.add(item.id)
                    continue
                self._items[item.id] = (None, None)
            self._sync_item(item)
        for bell in list(objs.bells.values()):
            if f"bell {bell.bed_id}" not in self._reported:
                self._sync_bell(bell)
        for fire in list(objs.fires.values()):
            if f"fire {fire.id}" not in self._reported:
                self._sync_fire(fire)

    def _sync_item(self, item) -> None:
        visible, x, y, z, yaw = self.item_pose(item)
        pose, shown = self._items[item.id]
        new_pose = (round(x, 3), round(y, 3), round(z, 3), round(yaw, 3))
        if visible and new_pose != pose:
            if self._guard(f"item {item.id}", self.stage.place_item, item.id, x, y, z, yaw):
                pose = new_pose
        if visible != shown and self._guard(f"item {item.id}", self.stage.show_item, item.id, visible):
            shown = visible
        self._items[item.id] = (pose, shown)

    def _sync_bell(self, bell) -> None:
        if bell.bed_id not in self._bells:
            bed = self.world.site.beds.get(bell.bed_id)
            if bed is None:
                return
            hx, hy = bed_head(bed)
            if not self._guard(f"bell {bell.bed_id}", self.stage.add_bell, bell.bed_id,
                               hx + math.cos(bed.yaw) * BELL_WALL_GAP_M, hy + math.sin(bed.yaw) * BELL_WALL_GAP_M,
                               BELL_Z_M, bed.yaw):
                return
            self._bells[bell.bed_id] = None
        if self._bells.get(bell.bed_id) is not bell.active and self._guard(
                f"bell {bell.bed_id}", self.stage.light_bell, bell.bed_id, bell.active):
            self._bells[bell.bed_id] = bell.active

    def _sync_fire(self, fire) -> None:
        if fire.out:
            if fire.id not in self._gone_fires:
                self._gone_fires.add(fire.id)
                if self._fires.pop(fire.id, None) is not None:
                    self._guard(f"fire {fire.id}", self.stage.remove_fire, fire.id)
            return
        key = (fire.x, fire.y, math.floor(fire.size / FIRE_STEP + 1e-9))
        if self._fires.get(fire.id) != key and self._guard(f"fire {fire.id}", self.stage.set_fire, fire.id, fire.x,
                                                           fire.y, fire_params(fire.size)):
            self._fires[fire.id] = key
