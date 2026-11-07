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
from ..isaac_assets import collision_usd as _collision_usd, stage_usd as _stage_usd

# WARDZERO_STAGE_USD and WARDZERO_COLLISION_USD override where the two files are (see isaac_assets)
STAGE_USD = _stage_usd()
COLLISION_USD = _collision_usd()

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
    exterior_rects: dict[str, Rect] = field(default_factory=dict)  # extra keep area for /World/Exterior only (a robot's view outdoors)

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


ROBOT_VIEW_FOV_DEG = 90.0  # the head camera's colour FOV (D455 asset, catalogue color_fov_h)
ROBOT_VIEW_RANGE_M = 15.0


def view_wedge(x: float, y: float, yaw: float, fov_deg: float, range_m: float, arc: int = 8) -> list:
    """The camera's view on the floor plan as a convex polygon: the apex, then points along the arc at range_m."""
    half = math.radians(fov_deg) / 2.0
    pts = [(x, y)]
    for i in range(arc + 1):
        a = yaw - half + 2.0 * half * i / arc
        pts.append((x + range_m * math.cos(a), y + range_m * math.sin(a)))
    return pts


def wedge_hits_rect(poly: list, rect: Rect) -> bool:
    """Separating-axis test of a convex polygon (fov < 180) against an axis-aligned rectangle."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if max(p[0] for p in poly) < x0 or min(p[0] for p in poly) > x1:
        return False
    if max(p[1] for p in poly) < y0 or min(p[1] for p in poly) > y1:
        return False
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        nx, ny = ay - by, bx - ax  # edge normal
        proj = [nx * (px - ax) + ny * (py - ay) for px, py in poly]
        side = max(proj, key=abs)  # the polygon lies on this side of its own edge
        rp = [nx * (cx - ax) + ny * (cy - ay) for cx, cy in corners]
        if (side >= 0 and max(rp) < -1e-9) or (side < 0 and min(rp) > 1e-9):
            return False
    return True


EXTERIOR_VIEW_RANGE_M = 10.0  # the outdoors a robot looks onto through an entrance, kept for /World/Exterior only
SIGHT_STEP_M = 0.1
ROOM_SAMPLE_M = 1.0


def sight_clear(grid, a: tuple, b: tuple) -> bool:
    """No occupied cell of the site grid between a and b (cells outside the grid are open: outdoors)."""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    for k in range(1, int(d / SIGHT_STEP_M)):
        f = k * SIGHT_STEP_M / d
        iy, ix = grid.to_cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        if grid.in_grid(iy, ix) and grid.occ[iy, ix]:
            return False
    return True


def rooms_in_view(site, pose: tuple, fov_deg: float = ROBOT_VIEW_FOV_DEG, range_m: float = ROBOT_VIEW_RANGE_M) -> list:
    """Room ids a camera at pose (x, y, yaw) can see: the room it stands in, and every room with a point inside the
    view wedge that has a clear line of sight from the camera on the site grid (walls hide what is behind them;
    doorways are open cells). Points are sampled every ROOM_SAMPLE_M."""
    x, y, yaw = pose
    wedge = view_wedge(x, y, yaw, fov_deg, range_m)
    out = []
    for rid, rm in site.rooms.items():
        if rm.contains(x, y):
            out.append(rid)
            continue
        x0, y0, x1, y1 = room_rect(rm)
        if not wedge_hits_rect(wedge, (x0, y0, x1, y1)):
            continue
        xs = [x0 + ROOM_SAMPLE_M / 2 + i * ROOM_SAMPLE_M for i in range(max(1, int((x1 - x0) / ROOM_SAMPLE_M)))]
        ys = [y0 + ROOM_SAMPLE_M / 2 + j * ROOM_SAMPLE_M for j in range(max(1, int((y1 - y0) / ROOM_SAMPLE_M)))]
        if any(wedge_hits_rect(wedge, (px, py, px, py)) and rm.contains(px, py) and sight_clear(site.grid, (x, y), (px, py))
               for px in xs for py in ys):
            out.append(rid)
    return sorted(out)


def cull_plan(loaded: "LoadedScenario", hints: Optional[Mapping] = None, rooms: Optional[Iterable[str]] = None,
              full_site: bool = False, robot_view: bool = True) -> CullPlan:
    """The rooms (and rectangles) to keep for this scenario, plus the rooms in the robot camera's view (a camera that
    looks into culled space sees black); ``robot_view=False`` leaves them out."""
    plan = _base_plan(loaded, hints, rooms, full_site)
    r = loaded.scenario.robot
    if plan.full or not robot_view or not r.enabled:
        return plan
    seen = [rid for rid in rooms_in_view(loaded.site, (r.x, r.y, r.yaw)) if rid not in plan.rooms]
    rects = dict(plan.rects)
    rects.update({rid: room_rect(loaded.site.rooms[rid]) for rid in seen})
    wedge = view_wedge(r.x, r.y, r.yaw, ROBOT_VIEW_FOV_DEG, EXTERIOR_VIEW_RANGE_M)
    outdoors = {"robot_view": (min(p[0] for p in wedge), min(p[1] for p in wedge),
                               max(p[0] for p in wedge), max(p[1] for p in wedge))}
    return CullPlan(plan.rooms + seen, rects, plan.source, outdoors)


def _base_plan(loaded: "LoadedScenario", hints: Optional[Mapping] = None, rooms: Optional[Iterable[str]] = None,
               full_site: bool = False) -> CullPlan:
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


FRAME_BIG_ROOM_M = 15.0  # a placement in a room longer than this (a corridor) frames only its stretch of the room
FRAME_POINT_M = 5.0  # ... this far round the placement


def frame_rects(loaded: "LoadedScenario", plan: CullPlan, hints: Optional[Mapping] = None) -> list[Rect]:
    """What the overhead and preview cameras fit: the scenario's places, not every room kept for it. The preset's
    focus rooms, plus where the cast, the robot and the objects start (a long room, such as the main corridor, only
    round the placement). Rooms kept for the route in and the robot's view are left out: the 47 m main corridor kept
    for ward_fire's route squeezed the ward into a third of the frame. Rooms given by hand (``--rooms``), the full
    site, or nothing to go on: the kept rooms as they are."""
    kept = list(plan.rects.values())
    if plan.full or plan.source == "rooms":
        return kept
    site = loaded.site
    if hints is None:
        hints = read_hints(Path(loaded.root) / "site" / "navmesh_hints.json")
    scen = loaded.scenario
    hint = hints.get(scen.preset_id or "") or hints.get(scen.id) or {}
    out: dict[str, Rect] = {r: room_rect(site.rooms[r]) for r in hint.get("focus_rooms", []) if r in site.rooms}
    for i, (x, y) in enumerate(placements(loaded)):
        rid = next((k for k, rm in site.rooms.items() if rm.contains(x, y)), None)
        if rid is None:
            out[f"near_{i}"] = (x - POINT_BOX_M, y - POINT_BOX_M, x + POINT_BOX_M, y + POINT_BOX_M)
            continue
        r = room_rect(site.rooms[rid])
        if rid in out or max(r[2] - r[0], r[3] - r[1]) <= FRAME_BIG_ROOM_M:
            out.setdefault(rid, r)
        else:
            out[f"near_{i}"] = (max(r[0], x - FRAME_POINT_M), max(r[1], y - FRAME_POINT_M),
                                min(r[2], x + FRAME_POINT_M), min(r[3], y + FRAME_POINT_M))
    if not out:
        return kept
    if kept:  # never wider than what is kept (a camera that sees culled space sees black)
        kx0, ky0 = min(r[0] for r in kept), min(r[1] for r in kept)
        kx1, ky1 = max(r[2] for r in kept), max(r[3] for r in kept)
        out = {k: (max(r[0], kx0), max(r[1], ky0), min(r[2], kx1), min(r[3], ky1)) for k, r in out.items()}
    return list(out.values())


def intersects(box: Box, rects: Mapping[str, Rect], margin: float = MARGIN_M) -> bool:
    (bx0, by0), (bx1, by1) = box
    for x0, y0, x1, y1 in rects.values():
        if not (bx1 < x0 - margin or bx0 > x1 + margin or by1 < y0 - margin or by0 > y1 + margin):
            return True
    return False


# Prims never culled, wherever their bounds say they are. The hospital's floor is one infinite physics plane
# (/World/Hospital/GroundPlane/CollisionPlane, purpose "guide"): its bounds come out empty, so it was placed at its
# origin by the lobby and culled from every ward scenario, and the walking G1 fell through the floor.
NEVER_CULL = ("GroundPlane",)


def never_culled(path: str) -> bool:
    return path.rstrip("/").rsplit("/", 1)[-1] in NEVER_CULL


def select_off(boxes: Mapping[str, Optional[Box]], rects: Mapping[str, Rect], margin: float = MARGIN_M) -> list[str]:
    """The prim paths to deactivate: those with bounds that touch no kept rectangle (never the floor)."""
    return sorted(p for p, b in boxes.items()
                  if b is not None and not never_culled(p) and not intersects(b, rects, margin))


# ---- the wrapper layer (pxr) ------------------------------------------------------------------
def _boxes(stage, parents: Iterable[str], remap: Optional[tuple[str, str]] = None) -> dict[str, Optional[Box]]:
    from pxr import Usd, UsdGeom, UsdLux

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                           useExtentsHint=True)
    xc = UsdGeom.XformCache()
    out: dict[str, Optional[Box]] = {}
    for parent in parents:
        pp = stage.GetPrimAtPath(parent)
        if not pp:
            continue
        for c in pp.GetChildren():
            if c.IsA(UsdLux.DomeLight) or c.IsA(UsdLux.DistantLight):
                out[str(c.GetPath())] = None  # the sky and the sun light everything outdoors: they have no place to cull by
                continue
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
    boxes = _boxes(st, CULL_PARENTS)
    outdoors = {p: b for p, b in boxes.items() if p.startswith("/World/Exterior/")}
    off = select_off({p: b for p, b in boxes.items() if p not in outdoors}, plan.rects)
    off += select_off(outdoors, {**plan.rects, **plan.exterior_rects})
    if collision_path is not None and Path(collision_path).exists():
        col = Usd.Stage.Open(str(collision_path))
        dp = col.GetDefaultPrim()
        if dp:
            root = str(dp.GetPath())
            off += select_off(_boxes(col, [root], remap=(root, COLLIDERS_PATH)), plan.rects)
    return off


NAV_Z_CENTER_M, NAV_Z_SIZE_M = -0.033, 0.5  # the stage's own include slab: 0.5 m thick about the floor


def nav_include_attrs(box: Box) -> dict[str, tuple[str, object]]:
    """The attributes of an Include NavMeshVolume over an xy box, as the stage's own include slab has them
    ({name: (Sdf value type name, value)}; a unit cube scaled and moved over the box)."""
    (x0, y0), (x1, y1) = box
    return {
        "extent": ("Float3Array", [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]),
        "nav:area": ("String", "Walkable"),
        "nav:volume:type": ("Token", "Include"),
        "xformOp:translate": ("Double3", ((x0 + x1) / 2, (y0 + y1) / 2, NAV_Z_CENTER_M)),
        "xformOp:scale": ("Float3", (x1 - x0, y1 - y0, NAV_Z_SIZE_M)),
        "xformOpOrder": ("TokenArray", ["xformOp:translate", "xformOp:scale"]),
    }


def write_wrapper(out_path: Path, off: Iterable[str], stage_path: Path = STAGE_USD,
                  authored: Optional[Mapping[str, Mapping[str, object]]] = None,
                  nav_include: Optional[Box] = None) -> Path:
    """A root layer that sublayers the real stage, deactivates ``off``, authors ``authored`` ({prim: {attribute: bool or
    float}}) and, given ``nav_include`` (an xy box), defines one more NavMesh Include volume at ``fixes.NAV_INCLUDE_PATH``."""
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
    if nav_include is not None:
        from .fixes import NAV_INCLUDE_PATH

        parent = Sdf.CreatePrimInLayer(lay, NAV_INCLUDE_PATH.rsplit("/", 1)[0])
        parent.specifier, parent.typeName = Sdf.SpecifierDef, "Scope"  # defined, or the volume under it is not traversed
        spec = Sdf.CreatePrimInLayer(lay, NAV_INCLUDE_PATH)
        spec.specifier, spec.typeName = Sdf.SpecifierDef, "NavMeshVolume"
        spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["NavMeshAreaAPI"]))
        for name, (kind, value) in nav_include_attrs(nav_include).items():
            attr = Sdf.AttributeSpec(spec, name, getattr(Sdf.ValueTypeNames, kind),
                                     variability=Sdf.VariabilityUniform if name == "xformOpOrder" else Sdf.VariabilityVarying)
            attr.default = value
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


def add_physics_scene(stage, steps_per_s: Optional[int] = None) -> str:
    """CPU PhysX with the MBP broadphase (S1b: GPU dynamics waits ~50 ms/frame on one articulation). steps_per_s: the
    physics rate (200 for the walking G1, whose controller runs PD every 5 ms); None leaves PhysX's default."""
    from pxr import PhysxSchema, UsdPhysics

    scene = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
    if scene is None:
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH).GetPrim()
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene)
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    if steps_per_s:
        ps.CreateTimeStepsPerSecondAttr(int(steps_per_s))
    return str(scene.GetPath())


def add_collision(stage, collision_path: Path = COLLISION_USD) -> Optional[str]:
    """Reference the baked, invisible collision USD (the render stage's own colliders are incomplete)."""
    from pxr import UsdGeom

    if not Path(collision_path).exists():
        # `hhumans run` refuses to start without it (exit 2); this is for a host started by hand
        log.error("collision USD %s is missing; the people and robots get no walls. Build it with "
                  "scripts/build_collision_usd.py in hospital-sim-agents", collision_path)
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
