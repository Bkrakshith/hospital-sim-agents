"""Fixes to the hospital stage that live in the host's wrapper layer (``<run>/stage.usda``), never in the source stage.

1. NavMesh exclude volumes over the bedrooms. ``hospital_g1.usd`` carries ``NavMeshVolume_01..09`` under
   ``/World/Hospital/Navmesh``: Exclude boxes laid over the ward bedrooms (and the nurse-base bays) when the stage only
   had a lobby film in mind. They cut those rooms out of the NavMesh, so a person could not walk to B1, B3, B4, B5 or
   B6 (nor to a nurse base): the walk ended "blocked" although the StubWorld router, which follows the site map,
   reaches them. The door openings themselves are clear (1.22 m wide at every ward door in the baked collision USD, and
   the stage has no door leaf there). Fix: every Exclude volume that overlaps a bedroom, nurse-base or A&E treatment room that the run
   keeps is deactivated in the wrapper layer *before* the stage opens (deactivating it after the open has no effect on a
   bake; the NavMesh reads the volumes as they appear).
2. A door whose leaves are posed with a rotate op. ``/World/Fixtures/Doors/Compartment`` (the 4 m link between the lobby
   and the ward) is the only door built differently from the rest: its leaf rigid bodies carry ``xformOp:rotateZ`` (-90 and
   270 degrees) while the revolute joints that hold them assume leaf frames without it. PhysX then gives both leaves a
   non-finite pose on the first step ("Invalid PhysX transform detected for .../Compartment/Leaf0" and "Illegal
   BroadPhaseUpdateData - non-finite bounds", hundreds of times in a full-site run). Fix: such leaves are made static
   (``physics:rigidBodyEnabled = False``) in the wrapper layer. They stay where they are drawn, swung open against the
   wall, so the link stays passable (the site map gives the link no fire rating and no door).

The selection logic is pure Python (unit-tested); the stage reads need pxr, so they import it inside the function.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .stage import Box, CullPlan, intersects, room_rect

if TYPE_CHECKING:
    from ..site.model import Site

log = logging.getLogger("wardzero.isaac.fixes")

NAV_PARENT = "/World/Hospital/Navmesh"
FREE_USES = ("bedroom", "nurse_base", "ae_treatment")  # rooms that people must be able to walk into (ward and A&E bays)
MIN_OVERLAP_M = 0.3  # an exclude that overlaps a room by less than this (a wall's thickness) is left alone
DOOR_ROOT = "/World/Fixtures/Doors"  # door leaves only: lifts and call buttons are not touched
KNOWN_ROTATED = ("/World/Fixtures/Doors/Compartment/Leaf0", "/World/Fixtures/Doors/Compartment/Leaf1")


def _overlap(a: Box, r) -> float:
    """The smaller side of the overlap between a box and a rectangle (x0, y0, x1, y1); 0 when they do not meet."""
    (ax0, ay0), (ax1, ay1) = a
    return max(0.0, min(min(ax1, r[2]) - max(ax0, r[0]), min(ay1, r[3]) - max(ay0, r[1])))


def nav_volumes_off(volumes: Mapping[str, tuple[str, Box]], site: "Site", plan: CullPlan) -> list[str]:
    """The Exclude volumes (path -> (type, xy box)) to deactivate: those over a bedroom, nurse-base or A&E treatment room the run keeps."""
    rooms = [room_rect(r) for rid, r in site.rooms.items()
             if r.use in FREE_USES and (plan.full or rid in plan.rects)]
    return sorted(path for path, (kind, box) in volumes.items()
                  if kind == "Exclude" and any(_overlap(box, r) >= MIN_OVERLAP_M for r in rooms))


def rotated_leaves(bodies: Mapping[str, bool]) -> list[str]:
    """Door-leaf rigid bodies (path -> has a rotate xform op) that PhysX cannot hold to their joints: the ones with a
    rotate op. Only prims under ``DOOR_ROOT`` count."""
    return sorted(p for p, rotated in bodies.items() if rotated and p.startswith(DOOR_ROOT + "/"))


# ---- reading the source stage (pxr) -----------------------------------------------------------------
def read_nav_volumes(st) -> dict[str, tuple[str, Box]]:
    """The NavMeshVolume prims of an opened pxr stage: path -> (Include or Exclude, xy box)."""
    from pxr import UsdGeom

    root = st.GetPrimAtPath(NAV_PARENT)
    out: dict[str, tuple[str, Box]] = {}
    if not root:
        return out
    xc = UsdGeom.XformCache()
    for p in root.GetChildren():
        if p.GetTypeName() != "NavMeshVolume":
            continue
        m = xc.GetLocalToWorldTransform(p)
        pts = [m.Transform((sx * 0.5, sy * 0.5, 0.0)) for sx in (-1, 1) for sy in (-1, 1)]  # the unit cube's corners
        kind = p.GetAttribute("nav:volume:type").Get() or "Include"
        out[str(p.GetPath())] = (str(kind), ((min(q[0] for q in pts), min(q[1] for q in pts)),
                                             (max(q[0] for q in pts), max(q[1] for q in pts))))
    return out


def read_door_bodies(st) -> dict[str, bool]:
    """Every rigid body under /World/Fixtures/Doors of an opened pxr stage: does its xform stack have a rotate op?"""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = st.GetPrimAtPath(DOOR_ROOT)
    out: dict[str, bool] = {}
    if not root:
        return out
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            ops = UsdGeom.Xformable(p).GetOrderedXformOps()
            out[str(p.GetPath())] = any("rotate" in op.GetOpName() or "orient" in op.GetOpName() for op in ops)
    return out


def stage_fixes(site: "Site", plan: CullPlan, stage_path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    """(prims to deactivate, {prim: {attribute: value}} to author) for the wrapper layer."""
    from pxr import Usd

    # One more open of the source stage (about a second): ``stage.compute_cull`` opens it too, but it returns only the
    # list of prims to cull and the two passes want different things, so they are not merged.
    st = Usd.Stage.Open(str(stage_path))
    off = nav_volumes_off(read_nav_volumes(st), site, plan)
    authored = {p: {"physics:rigidBodyEnabled": False} for p in rotated_leaves(read_door_bodies(st))}
    unexpected = [p for p in authored if p not in KNOWN_ROTATED]
    if unexpected:
        log.warning("the door-leaf fix changed more than the known Compartment pair: %s", ", ".join(unexpected))
    log.info("stage fixes: %d NavMesh exclude volume(s) off (%s), %d door leaf body(ies) made static (%s)",
             len(off), ", ".join(Path(p).name for p in off) or "none", len(authored),
             ", ".join(Path(p).parent.name + "/" + Path(p).name for p in authored) or "none")
    return off, authored
