"""The hospital stage for a run: which areas to keep, the culling wrapper layer, physics and colliders.

On the 8 GB RTX 3070 the full hospital costs about 5 GB of VRAM before any person or robot, so the
host keeps only the rooms a scenario uses (spike S1 Q1). ``hospital_g1.usd`` is not grouped by
room: the cull is spatial. Every child of the ``CULL_PARENTS`` groups whose world bounding box
misses all the kept rectangles (plus a margin) is deactivated by an ``over ... (active = false)`` in
a small wrapper root layer that sublayers the real stage. Opening the wrapper means RTX never loads
the culled prims, and the source stage is never edited. The baked collision USD is culled the same
way.

Which rooms are kept (``cull_plan``, pure Python, unit-tested):
- the scenario preset's ``focus_rooms`` and ``route_rooms`` from ``site/navmesh_hints.json``;
- plus the room of every placement: cast seats, start points and beds, the entrance of people who
  arrive, the robot's start and the scenario's objects (a small box when a point is in no room);
- ``--rooms A,B`` replaces all of that; ``--full-site`` keeps everything, as does a scenario that
  gives nothing to go on.

Everything that needs pxr or Kit imports it inside the function (pxr exists only once Kit is up).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

if TYPE_CHECKING:
    from ..config.loader import LoadedScenario

log = logging.getLogger("wardzero.isaac.stage")

HG1 = Path(__file__).resolve().parents[3]
STAGE_USD = HG1 / "hospital-sim-env" / "isaac" / "usd" / "hospital_g1.usd"
COLLISION_USD = HG1 / "hospital-sim-humans" / "sim" / "assets" / "hospital_collision.usd"

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1 in the site frame (the stage's world frame)
Box = tuple[tuple[float, float], tuple[float, float]]  # ((xmin, ymin), (xmax, ymax))

MARGIN_M = 0.3  # a prim this close to a kept rectangle is kept (walls straddle room edges)
POINT_BOX_M = 2.5  # half-size of the area kept around a placement that is in no room

# Groups whose children are culled one by one; anything else under /World is kept as it is.
CULL_PARENTS = (
    "/World/Hospital",
    "/World/Building/Dressing",
    "/World/BuildingMods/Partitions",
    "/World/BuildingMods/LightPanels",
    "/World/HospitalLighting",
    "/World/Fixtures/Doors", "/World/Fixtures/Doors1", "/World/Fixtures/Lifts", "/World/Fixtures/Calls",
    "/World/Fixtures/Static",
    "/World/Wing/Shell", "/World/Wing/ORSuite", "/World/Wing/ORSuiteMods", "/World/Wing/Dressing",
    "/World/Wing/Lighting",
    "/World/Exterior/Ground", "/World/Exterior/Ambulance", "/World/Exterior/Canopy", "/World/Exterior/Signs",
)
COLLIDERS_PATH = "/World/BakedColliders"  # where the baked collision USD is referenced
PHYSICS_SCENE_PATH = "/World/PhysicsScene"
ROBOT_PROXY_PATH = "/World/RobotProxy"


# ---- which rooms (pure) -----------------------------------------------------------------------
@dataclass
class CullPlan:
    rooms: list[str] = field(default_factory=list)
    rects: dict[str, Rect] = field(default_factory=dict)
    source: str = "full"  # "hints" | "placements" | "rooms" | "full"

    @property
    def full(self) -> bool:
        return self.source == "full"

    def describe(self) -> str:
        if self.full:
            return "the full site"
        extra = [k for k in self.rects if k not in self.rooms]
        return ", ".join(self.rooms + extra) + f" (from {self.source})"


def room_rect(room) -> Rect:
    xs = [p[0] for p in room.poly]
    ys = [p[1] for p in room.poly]
    return (min(xs), min(ys), max(xs), max(ys))


def read_hints(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def placements(loaded: "LoadedScenario") -> list[tuple[float, float]]:
    """Every point the scenario puts someone or something at."""
    site, scen = loaded.site, loaded.scenario
    pts: list[tuple[float, float]] = []

    def at(obj) -> None:
        if obj is not None:
            pts.append((float(obj.x), float(obj.y)))

    for c in scen.cast:
        if c.seat is not None:
            at(site.seats.get(c.seat))
        elif c.start is not None and c.start.bed is not None:
            at(site.beds.get(c.start.bed))
        elif c.start is not None:
            pts.append((float(c.start.x), float(c.start.y)))
        else:
            at(site.landmarks.get(c.entrance or "entrance"))
    if scen.robot.enabled:
        pts.append((float(scen.robot.x), float(scen.robot.y)))
    for o in scen.objects:
        if isinstance(o, dict):
            if "x" in o and "y" in o:
                pts.append((float(o["x"]), float(o["y"])))
            elif o.get("bed"):
                at(site.beds.get(o["bed"]))
    return [p for p in pts if all(math.isfinite(v) for v in p)]


def cull_plan(loaded: "LoadedScenario", hints: Optional[Mapping] = None, rooms: Optional[Iterable[str]] = None,
              full_site: bool = False) -> CullPlan:
    """The rooms (and rectangles) to keep for this scenario."""
    if full_site:
        return CullPlan()
    site = loaded.site
    if rooms is not None:
        rooms = list(rooms)
        unknown = [r for r in rooms if r not in site.rooms]
        if unknown:
            raise ValueError(f"unknown room(s): {', '.join(unknown)}")
        return CullPlan(rooms, {r: room_rect(site.rooms[r]) for r in rooms}, "rooms")
    if hints is None:
        hints = read_hints(Path(loaded.root) / "site" / "navmesh_hints.json")
    scen = loaded.scenario
    hint = hints.get(scen.preset_id or "") or hints.get(scen.id) or {}
    names: set[str] = set()
    for r in list(hint.get("focus_rooms", [])) + list(hint.get("route_rooms", [])):
        if r in site.rooms:
            names.add(r)
        else:
            log.warning("navmesh_hints: room %s is not on the site map; ignored", r)
    extra: dict[str, Rect] = {}
    for i, (x, y) in enumerate(placements(loaded)):
        room = next((rid for rid, rm in site.rooms.items() if rm.contains(x, y)), None)
        if room is not None:
            names.add(room)
        else:
            extra[f"near_{i}"] = (x - POINT_BOX_M, y - POINT_BOX_M, x + POINT_BOX_M, y + POINT_BOX_M)
    if not names and not extra:
        return CullPlan()
    ordered = sorted(names)
    rects = {r: room_rect(site.rooms[r]) for r in ordered}
    rects.update(extra)
    return CullPlan(ordered, rects, "hints" if hint else "placements")


def intersects(box: Box, rects: Mapping[str, Rect], margin: float = MARGIN_M) -> bool:
    (bx0, by0), (bx1, by1) = box
    for x0, y0, x1, y1 in rects.values():
        if not (bx1 < x0 - margin or bx0 > x1 + margin or by1 < y0 - margin or by0 > y1 + margin):
            return True
    return False


def select_off(boxes: Mapping[str, Optional[Box]], rects: Mapping[str, Rect], margin: float = MARGIN_M) -> list[str]:
    """The prim paths to deactivate: those with bounds that touch no kept rectangle."""
    return sorted(p for p, b in boxes.items() if b is not None and not intersects(b, rects, margin))


# ---- the wrapper layer (pxr) ------------------------------------------------------------------
def _boxes(stage, parents: Iterable[str], remap: Optional[tuple[str, str]] = None) -> dict[str, Optional[Box]]:
    from pxr import Usd, UsdGeom

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                           useExtentsHint=True)
    xc = UsdGeom.XformCache()
    out: dict[str, Optional[Box]] = {}
    for parent in parents:
        pp = stage.GetPrimAtPath(parent)
        if not pp:
            continue
        for c in pp.GetChildren():
            r = bc.ComputeWorldBound(c).ComputeAlignedRange()
            if r.IsEmpty():
                if not c.IsA(UsdGeom.Xformable):
                    out[str(c.GetPath())] = None  # materials, scopes: kept
                    continue
                t = xc.GetLocalToWorldTransform(c).ExtractTranslation()  # lights and bare xforms: the origin
                b: Box = ((t[0], t[1]), (t[0], t[1]))
            else:
                b = ((r.GetMin()[0], r.GetMin()[1]), (r.GetMax()[0], r.GetMax()[1]))
            path = str(c.GetPath())
            if remap is not None and path.startswith(remap[0]):
                path = remap[1] + path[len(remap[0]):]
            out[path] = b
    return out


def compute_cull(plan: CullPlan, stage_path: Path = STAGE_USD, collision_path: Optional[Path] = COLLISION_USD) -> list[str]:
    """Pure-USD pass over the stage (and the collision USD as mounted at COLLIDERS_PATH)."""
    if plan.full:
        return []
    from pxr import Usd

    st = Usd.Stage.Open(str(stage_path))
    off = select_off(_boxes(st, CULL_PARENTS), plan.rects)
    if collision_path is not None and Path(collision_path).exists():
        col = Usd.Stage.Open(str(collision_path))
        dp = col.GetDefaultPrim()
        if dp:
            root = str(dp.GetPath())
            off += select_off(_boxes(col, [root], remap=(root, COLLIDERS_PATH)), plan.rects)
    return off


def write_wrapper(out_path: Path, off: Iterable[str], stage_path: Path = STAGE_USD,
                  authored: Optional[Mapping[str, Mapping[str, object]]] = None) -> Path:
    """A root layer that sublayers the real stage, deactivates ``off`` and authors ``authored`` ({prim: {attribute: bool or float}})."""
    from pxr import Sdf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lay = Sdf.Layer.FindOrOpen(str(out_path)) if out_path.exists() else Sdf.Layer.CreateNew(str(out_path))
    lay.Clear()
    lay.subLayerPaths.append(str(stage_path))
    src = Sdf.Layer.FindOrOpen(str(stage_path))
    lay.defaultPrim = src.defaultPrim or "World"
    for k in ("upAxis", "metersPerUnit"):
        if src.pseudoRoot.GetInfo(k) is not None:
            lay.pseudoRoot.SetInfo(k, src.pseudoRoot.GetInfo(k))
    for p in off:
        spec = Sdf.CreatePrimInLayer(lay, p)
        spec.specifier = Sdf.SpecifierOver
        spec.active = False
    for p, attrs in (authored or {}).items():
        spec = Sdf.CreatePrimInLayer(lay, p)
        spec.specifier = Sdf.SpecifierOver
        for name, value in attrs.items():
            kind = Sdf.ValueTypeNames.Bool if isinstance(value, bool) else Sdf.ValueTypeNames.Float
            Sdf.AttributeSpec(spec, name, kind).default = value if isinstance(value, bool) else float(value)
    lay.Save()
    return out_path


def open_stage(app, path: Path):
    """Open a stage in the Kit context and wait until it has loaded."""
    import omni.usd

    ctx = omni.usd.get_context()
    ctx.open_stage(str(path))
    while ctx.get_stage_loading_status()[2] > 0 or not ctx.get_stage():
        app.update()
    return ctx.get_stage()


def add_physics_scene(stage) -> str:
    """CPU PhysX with the MBP broadphase (S1b: GPU dynamics waits ~50 ms/frame on one articulation)."""
    from pxr import PhysxSchema, UsdPhysics

    scene = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
    if scene is None:
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH).GetPrim()
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene)
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    return str(scene.GetPath())


def add_collision(stage, collision_path: Path = COLLISION_USD) -> Optional[str]:
    """Reference the baked, invisible collision USD (the render stage's own colliders are incomplete)."""
    from pxr import UsdGeom

    if not Path(collision_path).exists():
        log.warning("collision USD %s is missing; the people and robots get no walls", collision_path)
        return None
    col = UsdGeom.Xform.Define(stage, COLLIDERS_PATH)
    col.GetPrim().GetReferences().AddReference(str(collision_path))
    UsdGeom.Imageable(col).MakeInvisible()
    return COLLIDERS_PATH


def add_robot_proxy(stage, x: float, y: float, radius: float, height: float = 1.4) -> str:
    """One coarse obstacle for the people's avoidance, created before play (the robot itself is added
    after play so the Behavior System never tracks its links, S1b). Invisible and non-colliding: a
    colliding proxy pushed the robot's links."""
    from pxr import Gf, UsdGeom, UsdPhysics

    pc = UsdGeom.Cylinder.Define(stage, ROBOT_PROXY_PATH)
    pc.CreateRadiusAttr(float(radius))
    pc.CreateHeightAttr(float(height))
    pc.CreateAxisAttr("Z")
    pc.CreatePurposeAttr("guide")
    UsdGeom.Xformable(pc).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), height / 2))
    UsdPhysics.RigidBodyAPI.Apply(pc.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(pc.GetPrim()).CreateCollisionEnabledAttr(False)
    return ROBOT_PROXY_PATH


def view_of(plan: CullPlan, site) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """An eye and a target that frame the kept area from above at an angle (for the headed window)."""
    if plan.full or not plan.rects:
        x0, x1, y0, y1 = site.bounds
    else:
        x0 = min(r[0] for r in plan.rects.values())
        y0 = min(r[1] for r in plan.rects.values())
        x1 = max(r[2] for r in plan.rects.values())
        y1 = max(r[3] for r in plan.rects.values())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    span = max(x1 - x0, y1 - y0, 8.0)
    return (cx, cy - 0.6 * span, 0.7 * span), (cx, cy, 0.0)
