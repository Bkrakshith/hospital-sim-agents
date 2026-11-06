"""The Isaac side of the people: spawn IRA characters, seats and lying patients, and wrap the Behavior System.

Everything here needs Kit (pxr, carb, omni.*); each import is inside the function that uses it. The mapping from
intents to calls is in ``people`` and tested without Isaac. What this module does, in the order the host calls it:

1. ``IsaacPeople.spawn`` (before the NavMesh bake and before ``timeline.play()``): the shared HumanMotionLibrary,
   one character per person with a mind (payload, BehaviorAgentAPI, NavMeshExcludeAPI, IRACharacterAPI and a
   ``person`` semantic label, the calls IRA's CharacterLoader makes), a sit effector on every seat of the site, and
   a character lying on the bed top for every static patient (no behavior).
2. ``IsaacPeople.bind`` (after play): fetch each character's IBehaviorAgent, give it to a ``PersonController``,
   start the people who begin seated and let them settle into their seats.
3. ``IsaacPeople.spawn_runtime``: a person who joins during the run is spawned after play and bound when the Behavior
   System has registered the agent (a few frames).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import characters as ch
from . import people as pe

if TYPE_CHECKING:
    from ..world.stub_world import StubWorld

log = logging.getLogger("wardzero.isaac.people")

ASSETS_ROOT = Path(os.environ.get("ISAAC_ASSETS_ROOT", Path.home() / "isaacsim_assets" / "Assets" / "Isaac" / "6.0"))
MOTION_LIBRARY = "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
CHARACTERS_ROOT = "/World/Characters"
SEATS_ROOT = "/World/Seats"
STAFF_CHAIRS_ROOT = "/World/StaffChairs"
# the hospital's office chair (SM_Chair_04a, the one inside the reception counter); its front is +x
STAFF_CHAIR_REL = "assets/isaac/Isaac/Environments/Hospital/Props/SM_Chair_04a.usd"
STAFF_CHAIR_YAW_OFFSET_DEG = 0.0


def staff_chair_usd() -> "Optional[Path]":
    """The office chair asset next to the hospital stage (hospital-sim-env/isaac/<STAFF_CHAIR_REL>), or None."""
    from .stage import STAGE_USD
    p = Path(STAGE_USD).resolve().parents[1] / STAFF_CHAIR_REL
    return p if p.exists() else None
STATIC_ROOT = "/World/StaticPatients"
SEAT_HIPS_Z = 0.62  # sit effector 6 cm above the 0.56 m cushion top of the lobby chairs (spike S1 Q2)
BED_FLAT_Z = 1.0  # mattress top of SM_HospitalBed_01b (measured from its mesh: 0.80 to 1.00, flat part)
BED_TOP_Z = BED_FLAT_Z + 0.09  # where a lying character's origin rides: its back plane sits 0.09 m under the origin
HEAD_RISE_START_T, HEAD_RISE_END_T, HEAD_RISE_M = 0.3, 1.0, 0.2  # the head half of that mesh slopes up 0.2 m (t along the bed to its head)
BEDSIDE_M = 0.95
BODY_LENGTH_M = 1.75
OVERLAP_M = 0.5  # people who would start closer than this are spread apart at spawn
SPREAD_M = 0.65
SEATED_WITHIN_M = 0.12  # a character is in its seat when its root is this close to the seat point
SETTLE_MAX_S = 10.0  # at most this long (sim seconds) for the people who start seated to sit down
AGENT_WAIT_FRAMES = 300
FRONT_LOCAL = (0.0, -1.0, 0.0)  # the way a character faces in its own frame: -Y (seen on the contact sheet)


def asset(rel: str) -> str:
    return str(ASSETS_ROOT / rel)


def lying_matrix(bed, top_z: float = BED_TOP_Z, front_local: tuple = FRONT_LOCAL):
    """Row-vector 4x4 (as 16 numbers) that lays a standing character on its back on a bed (top_z: where its origin rides): head towards the bed's head
    end (bed.yaw points at the head), face up, body centred on the bed."""
    c, s = math.cos(bed.yaw), math.sin(bed.yaw)
    h = (c, s, 0.0)  # local up (the head end) -> along the bed to its head
    up = (0.0, 0.0, 1.0)  # the facing axis -> up (face up)
    w = (s, -c, 0.0)  # the remaining axis, so that the three images keep the rotation proper
    fx, fy = front_local[0], front_local[1]
    x_axis = tuple(fx * u - fy * v for u, v in zip(up, w))
    y_axis = tuple(fy * u + fx * v for u, v in zip(up, w))
    z_axis = h
    ox = bed.x - c * BODY_LENGTH_M / 2
    oy = bed.y - s * BODY_LENGTH_M / 2
    oz = top_z
    return (x_axis[0], x_axis[1], x_axis[2], 0.0, y_axis[0], y_axis[1], y_axis[2], 0.0,
            z_axis[0], z_axis[1], z_axis[2], 0.0, ox, oy, oz, 1.0)


def flatten_dz(points_xy, centre, head, start: float = HEAD_RISE_START_T, end: float = HEAD_RISE_END_T,
               rise: float = HEAD_RISE_M):
    """Vertical shift (<= 0) that lays the raised head half of a bed mesh flat. ``points_xy``: Nx2 world x, y;
    ``head``: unit vector from the bed's centre to its head end. A shear along the bed, so thickness is kept."""
    import numpy as np

    t = (np.asarray(points_xy, dtype=float) - np.asarray(centre, dtype=float)) @ np.asarray(head, dtype=float)
    return -rise * np.clip((t - start) / (end - start), 0.0, 1.0)


def bedside(bed, blocked) -> tuple[float, float]:
    """A floor point beside a bed (across it from the centre), on the side where ``blocked(x, y)`` is false."""
    c, s = math.cos(bed.yaw), math.sin(bed.yaw)
    for sign in (1.0, -1.0):
        x, y = bed.x - s * BEDSIDE_M * sign, bed.y + c * BEDSIDE_M * sign
        if not blocked(x, y):
            return x, y
    return bed.x - c * 1.7, bed.y - s * 1.7  # the foot end


def pick_bedside(bed, blocked, reachable) -> tuple[float, float]:
    """The side of a bed to get up on: a free floor point the NavMesh can reach (``reachable(x, y)``), else the first free one."""
    c, s = math.cos(bed.yaw), math.sin(bed.yaw)
    first = None
    for dist in (BEDSIDE_M, BEDSIDE_M + 0.25):
        for sign in (1.0, -1.0):
            x, y = bed.x - s * dist * sign, bed.y + c * dist * sign
            if blocked(x, y):
                continue
            first = first or (x, y)
            if reachable(x, y):
                return x, y
    return first or bedside(bed, blocked)


def spread(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """(x, y, yaw) starts: people who would stand on top of each other are moved sideways."""
    out: list[tuple[float, float, float]] = []
    for x, y, yaw in points:
        k = 0
        while any(math.hypot(x - ox, y - oy) < OVERLAP_M for ox, oy, _ in out):
            k += 1
            sign = 1 if k % 2 else -1
            off = SPREAD_M * ((k + 1) // 2) * sign
            x2, y2 = x - math.sin(yaw) * off, y + math.cos(yaw) * off
            if k > 12:
                break
            if not any(math.hypot(x2 - ox, y2 - oy) < OVERLAP_M for ox, oy, _ in out):
                x, y = x2, y2
                break
        out.append((x, y, yaw))
    return out


class IraAgent:
    """``people.BehaviorApi`` on one IBehaviorAgent."""

    def __init__(self, agent, prim_path: str, carb_module=None, bedside=None):
        if carb_module is None:
            import carb as carb_module
        self._carb = carb_module
        self.a = agent
        self.path = prim_path
        self.bedside = bedside  # (x, y, face_x, face_y) for a patient in a bed: hide and show move them under the floor

    def _f3(self, x, y, z=0.0):
        return self._carb.Float3(float(x), float(y), float(z))

    def move_to(self, x, y):
        return self.a.move_to(target=self._f3(x, y))

    def sit(self, prop):
        return self.a.sit(prop)

    def idle(self):
        return self.a.idle()

    def look_at(self, x, y, z, duration):
        return self.a.look_at(target=self._f3(x, y, z), duration=float(duration))

    def cancel(self, task):
        try:
            self.a.cancel_task(task)
        except Exception as exc:
            log.debug("cancel_task(%s): %s", task, exc)

    def status(self, task):
        if task is None:
            return "none"
        try:
            if self.a.is_task_running(task):
                return "running"
            name = str(self.a.get_task_status(task)).split(".")[-1].lower()
        except Exception:
            return "none"
        if "succe" in name or "complete" in name or "done" in name:
            return "succeeded"
        if "run" in name:
            return "running"
        return "failed"

    def pose(self):
        t = self.a.get_world_translation()
        f = self.a.get_facing_direction()
        return float(t.x), float(t.y), math.atan2(float(f.y), float(f.x))

    def current_action(self):
        try:
            return str(self.a.get_task_name(self.a.get_action_task_id()))
        except Exception:
            return ""

    def set_speed(self, v):
        """The walking speed the controller asks for (m/s); near a door leaf it is capped (set_door_slow)."""
        self._want = float(v)
        self._apply_speed()

    def set_door_slow(self, on: bool) -> None:
        """Within 1 m of a door leaf a person walks at 0.8 m/s at most (door spike), then back to what was asked."""
        self._door_slow = bool(on)
        self._apply_speed()

    def _apply_speed(self) -> None:
        from .doors import SLOW_SPEED

        want = getattr(self, "_want", None)
        if want is None:
            try:
                want = self._want = float(self.a.get_speed())
            except Exception:
                want = 1.5
        v = min(want, SLOW_SPEED) if getattr(self, "_door_slow", False) else want
        try:
            self.a.set_speed(float(v))
        except Exception as exc:
            log.debug("set_speed: %s", exc)

    def hide(self):
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        pid = self.path[len(CHARACTERS_ROOT) + 1:].split("/")[0]  # the character's own prim under /World/Characters
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{CHARACTERS_ROOT}/{pid}")).MakeInvisible()
        if self.bedside is None:  # a patient in bed stays registered (an invisible agent made before play has none)
            try:
                self.a.set_enabled(False)
            except Exception:
                pass

    def show(self):
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        pid = self.path[len(CHARACTERS_ROOT) + 1:].split("/")[0]
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{CHARACTERS_ROOT}/{pid}")).MakeVisible()
        if self.bedside is not None:  # up beside the bed, facing it
            x, y, fx, fy = self.bedside
            self.a.teleport(target=self._f3(x, y, 0.0), facing=self._f3(fx, fy, 0.0))
            return
        try:
            self.a.set_enabled(True)
        except Exception:
            pass


class RestingPose:
    """A one-frame SkelAnimation on a lying character's skeleton: arms down beside the body, torso bent forward when propped up.

    The character is drawn without the Behavior System, so nothing animates it and it would stand in a T-pose. The joint
    maths is ``rest_pose.resting``; this writes the result as an animation bound to the skeleton.
    """

    SIT_UP_DEG = 50.0

    def __init__(self, stage, skeleton_path: str):
        import numpy as np
        from pxr import Sdf, UsdSkel

        from . import rest_pose

        self._rp, self._np = rest_pose, np
        skel = UsdSkel.Skeleton(stage.GetPrimAtPath(skeleton_path))
        self.names = [str(j) for j in skel.GetJointsAttr().Get()]
        self.rest = np.array([[list(row) for row in m] for m in skel.GetRestTransformsAttr().Get()])
        anim = UsdSkel.Animation.Define(stage, Sdf.Path(skeleton_path).AppendChild("RestingPose"))
        anim.CreateJointsAttr(self.names)
        self.anim = anim
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        self.sit_up = None
        self.set(False)

    def set(self, sit_up: bool) -> None:
        from pxr import Gf, Vt, UsdSkel

        if sit_up == self.sit_up:
            return
        self.sit_up = sit_up
        new = self._rp.resting(self.names, self.rest, self.SIT_UP_DEG if sit_up else 0.0)
        parts = [UsdSkel.DecomposeTransform(Gf.Matrix4d(*[float(v) for v in m.reshape(16)])) for m in new]
        self.anim.CreateTranslationsAttr(Vt.Vec3fArray([p[0] for p in parts]))
        self.anim.CreateRotationsAttr(Vt.QuatfArray([p[1] for p in parts]))
        self.anim.CreateScalesAttr(Vt.Vec3hArray([p[2] for p in parts]))


class LyingProp:
    """The posed character on a bed (``people.BedProp``): shown, hidden, or carried along with a bed being wheeled."""

    def __init__(self, stage, path: str, op, bed=None, pose: Optional[RestingPose] = None):
        self.stage, self.path, self.op = stage, path, op
        self.bed = bed  # BedMover: the bed itself goes along when the patient is wheeled
        self.pose = pose  # RestingPose: arms at the sides, torso raised when propped up

    def set_sitting_up(self, sitting_up: bool) -> None:
        if self.pose is not None:
            self.pose.set(sitting_up)

    def _img(self):
        from pxr import UsdGeom

        return UsdGeom.Imageable(self.stage.GetPrimAtPath(self.path))

    def show(self):
        self._img().MakeVisible()

    def hide(self):
        self._img().MakeInvisible()

    def place(self, x, y, yaw):
        from types import SimpleNamespace

        from pxr import Gf

        self.op.Set(Gf.Matrix4d(*lying_matrix(SimpleNamespace(x=x, y=y, yaw=yaw))))
        if self.bed is not None:
            self.bed.move_to(x, y, yaw)


class BedMover:
    """Moves the stage's bed (its de-instanced Xform) to follow a wheeled patient: a rigid shift and turn about the bed's centre."""

    def __init__(self, stage, path: str, x: float, y: float, yaw: float):
        from pxr import Gf, UsdGeom

        self.x0, self.y0, self.yaw0 = x, y, yaw
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        self.m0 = xf.GetLocalTransformation()
        self.parent = UsdGeom.Xformable(prim).ComputeParentToWorldTransform(0.0) if hasattr(xf, "ComputeParentToWorldTransform") \
            else Gf.Matrix4d(1.0)
        xf.ClearXformOpOrder()
        self.op = xf.AddTransformOp()
        self.op.Set(self.m0)

    def move_to(self, x, y, yaw):
        from pxr import Gf

        d = yaw - self.yaw0
        c0 = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-self.x0, -self.y0, 0.0))
        r = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(d)))
        t = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, 0.0))
        delta = c0 * r * t  # row vectors: about the old centre, then to the new one
        self.op.Set(self.m0 * (self.parent * delta * self.parent.GetInverse()))


class IsaacPeople:
    def __init__(self, stage, loaded, allow: Optional[ch.AllowList] = None):
        self.stage = stage
        self.loaded = loaded
        self.allow = allow or ch.load_allowlist()
        self.skelroots: dict[str, str] = {}  # person id -> SkelRoot prim path
        self.choice: dict[str, ch.Character] = {}
        self.static: list[str] = []
        self.props: dict[str, LyingProp] = {}  # person id -> the lying prop (static patients and minded ones in bed)
        self._flat_beds: set[str] = set()
        self._bed_paths: dict[str, str] = {}
        self.bedsides: dict[str, tuple[float, float, float, float]] = {}
        self.pending: dict[str, int] = {}  # runtime joiners waiting for their agent -> frames waited
        self._motion_lib: Optional[str] = None

    # ---- spawning ---------------------------------------------------------------------------
    def spawn(self, world: "StubWorld", app=None) -> None:
        self._seats(world)
        brains = [(pid, p) for pid, p in world.people.items() if self._has_body(p)]
        self.choice = self.allow.assign([(pid, p.inst.role, p.inst.sex, p.inst.age) for pid, p in brains])
        starts = spread([self._start_pose(world, p) for _, p in brains])
        for (pid, p), (x, y, yaw) in zip(brains, starts):
            lying = p.posture == "lying"
            # a character who is invisible at spawn gets no Behavior agent: one in bed is hidden after play (bind)
            self.skelroots[pid] = self._spawn_character(pid, self.choice[pid], x, y, yaw)
            if lying:
                self.bedsides[pid] = (x, y, p.x, p.y)  # where they get up, and what they face (the bed)
        for pid, p in world.people.items():
            if p.kind == "static" or (p.bed and p.posture == "lying"):
                self._lying(pid, p, world)
        log.info("spawned %d characters (%s) and %d lying patients", len(self.skelroots),
                 ", ".join(f"{pid}={self.choice[pid].id}" for pid in self.skelroots), len(self.static))

    @staticmethod
    def _has_body(p) -> bool:
        return p.kind != "static"  # a minded patient in a bed has a walking character too (hidden until they get up)

    def _start_pose(self, world, p) -> tuple[float, float, float]:
        if p.posture == "seated" and p.seat:
            seat = world.seat(p.seat)
            ax, ay = pe.seat_approach(seat)
            return ax, ay, seat.yaw + math.pi  # in front of the seat, facing it: the first task sits them down
        if p.posture == "lying" and p.bed in world.site.beds:
            bed = world.site.beds[p.bed]
            x, y = bedside(bed, world.site.grid.blocked)
            return x, y, math.atan2(bed.y - y, bed.x - x)  # beside the bed, facing it
        return p.x, p.y, p.yaw

    def _root_prim(self, pid: str):
        return self.stage.GetPrimAtPath(f"{CHARACTERS_ROOT}/{pid}")

    def _motion_library(self) -> str:
        import omni.kit.commands
        import omni.usd
        from pxr import Sdf

        if self._motion_lib is None:
            if not self.stage.GetPrimAtPath(CHARACTERS_ROOT):
                self.stage.DefinePrim(CHARACTERS_ROOT, "Scope")
            path = Sdf.Path(CHARACTERS_ROOT).AppendChild("HumanMotionLibrary")
            omni.kit.commands.execute("CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to=path,
                                      asset_path=asset(MOTION_LIBRARY), prim_path=None, instanceable=False,
                                      select_prim=False)
            self._motion_lib = str(path)
        return self._motion_lib

    def _spawn_character(self, pid: str, char: ch.Character, x: float, y: float, yaw: float, z: float = 0.0) -> str:
        """Payload, then the Behavior System's APIs, the way IRA's CharacterLoader does it (spike S1 Q2)."""
        import NavSchema
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

        ml = Sdf.Path(self._motion_library())
        cp = Sdf.Path(CHARACTERS_ROOT).AppendChild(pid)
        omni.kit.commands.execute("CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to=cp,
                                  asset_path=asset(char.usd), prim_path=None, instanceable=False, select_prim=False)
        root = self.stage.GetPrimAtPath(cp)
        sr = next(p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Root))
        for name in char.hide:  # safety helmets, ear defenders, gloves and glasses of the DH characters
            child = self.stage.GetPrimAtPath(sr.GetPath().AppendChild(name))
            if child:
                child.SetActive(False)
        omni.kit.commands.execute("ApplyBehaviorAgentAPICommand", skelroot_prim_paths=[sr.GetPath()],
                                  motion_library_prim_path=ml, motion_library_skeleton_rig="Human")
        omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=sr.GetPath(), api=NavSchema.NavMeshExcludeAPI)
        omni.kit.commands.execute("ApplyIRACharacterAPICommand", skelroot_prim_paths=[sr.GetPath()])
        rep.functional.modify.semantics([sr], {"class": "person"})
        xf = UsdGeom.Xformable(root)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        xf.AddRotateZOp().Set(math.degrees(float(yaw)))
        return str(sr.GetPath())

    def _seats(self, world: "StubWorld") -> None:
        """A Sit effector on every seat of the site. Y-up effector frame: +Y is world up, +Z the way the sitter faces."""
        from pxr import Gf, Sdf, UsdGeom

        if not self.stage.GetPrimAtPath(SEATS_ROOT):
            self.stage.DefinePrim(SEATS_ROOT, "Scope")
        for sid, seat in world.site.seats.items():
            base = f"{SEATS_ROOT}/{sid}"
            xf = UsdGeom.Xform.Define(self.stage, base)
            xf.AddTranslateOp().Set(Gf.Vec3d(seat.x, seat.y, 0.0))
            xf.AddRotateZOp().Set(math.degrees(seat.yaw))
            prim = xf.GetPrim()
            prim.AddAppliedSchema("BehaviorPropAPI")
            eff = self.stage.DefinePrim(f"{base}/Behavior/Sit", "BehaviorTaskEffectors")
            eff.AddAppliedSchema("BehaviorTaskEffectorAPI")
            eff.CreateAttribute("behavior:task", Sdf.ValueTypeNames.Token).Set("Sit")
            hips = UsdGeom.Xform.Define(self.stage, f"{base}/Behavior/Sit_Hips")
            hips.AddTransformOp().Set(Gf.Matrix4d(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0.0, 0.0, SEAT_HIPS_Z, 1))
            eff.CreateRelationship("behavior:task:effectorHips").SetTargets([hips.GetPath()])
            prim.CreateRelationship("behavior:task:effectors").SetTargets([eff.GetPath()])
            if getattr(seat, "staff_only", False):
                self._staff_chair(sid, seat)

    def _staff_chair(self, sid: str, seat) -> Optional[str]:
        """A staff seat (the receptionist's R1) has no chair in the stage: the hospital's own office chair goes there,
        facing the way the sitter faces, out of the NavMesh (she walks into it to sit)."""
        import NavSchema
        import omni.kit.commands
        from pxr import Gf, Sdf, UsdGeom

        usd = staff_chair_usd()
        if usd is None:
            log.warning("seat %s: no chair asset (%s); she sits on nothing", sid, STAFF_CHAIR_REL)
            return None
        path = f"{STAFF_CHAIRS_ROOT}/{sid}"
        xf = UsdGeom.Xform.Define(self.stage, path)
        xf.GetPrim().GetReferences().AddReference(str(usd))
        xf.AddTranslateOp().Set(Gf.Vec3d(seat.x, seat.y, 0.0))
        xf.AddRotateZOp().Set(math.degrees(seat.yaw) + STAFF_CHAIR_YAW_OFFSET_DEG)
        omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=Sdf.Path(path), api=NavSchema.NavMeshExcludeAPI)
        log.info("staff chair at seat %s (%.2f, %.2f)", sid, seat.x, seat.y)
        return path

    def _lying(self, pid: str, p, world: "StubWorld") -> None:
        """A patient in a bed: a posed character on the mattress, no Behavior System (so no animation)."""
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

        bed = world.site.beds.get(p.bed)
        if bed is None:
            return
        char = self.choice.get(pid) or self.allow.choose(pid, "patient", p.inst.sex, p.inst.age)
        self._flatten_bed(bed, world)
        if not self.stage.GetPrimAtPath(STATIC_ROOT):
            self.stage.DefinePrim(STATIC_ROOT, "Scope")
        path = Sdf.Path(STATIC_ROOT).AppendChild(pid)
        omni.kit.commands.execute("CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to=path,
                                  asset_path=asset(char.usd), prim_path=None, instanceable=False, select_prim=False)
        root = self.stage.GetPrimAtPath(path)
        sr = next((q for q in Usd.PrimRange(root) if q.IsA(UsdSkel.Root)), None)
        if sr is not None:
            for name in char.hide:
                child = self.stage.GetPrimAtPath(sr.GetPath().AppendChild(name))
                if child:
                    child.SetActive(False)
            import NavSchema
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=sr.GetPath(), api=NavSchema.NavMeshExcludeAPI)
        xf = UsdGeom.Xformable(root)
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        op.Set(Gf.Matrix4d(*lying_matrix(bed)))
        mover = None
        if bed.id in self._bed_paths:
            mover = BedMover(self.stage, self._bed_paths[bed.id], bed.x, bed.y, bed.yaw)
        pose = None
        skel = next((q for q in Usd.PrimRange(root) if q.IsA(UsdSkel.Skeleton)), None)
        if skel is not None:
            try:
                pose = RestingPose(self.stage, str(skel.GetPath()))
            except Exception:
                log.exception("could not set the resting pose of %s: it stays in a T-pose", pid)
        self.props[pid] = LyingProp(self.stage, str(path), op, mover, pose)
        self.static.append(pid)

    def _flatten_bed(self, bed, world) -> None:
        """Lay the raised head half of the bed mattress flat (a rigid lying body cannot follow its slope)."""
        import numpy as np
        from pxr import Gf, Usd, UsdGeom

        if bed.id in self._flat_beds:
            return
        self._flat_beds.add(bed.id)
        base = self.stage.GetPrimAtPath(f"/World/Building/Dressing/Ward_{bed.room}/Bed")
        if not base:
            log.warning("bed %s: no mattress prim to flatten", bed.id)
            return
        self._bed_paths[bed.id] = str(base.GetPath())
        instances = [q.GetPath() for q in Usd.PrimRange(base, Usd.TraverseInstanceProxies()) if q.IsInstance()]
        for path in instances:
            self.stage.GetPrimAtPath(path).SetInstanceable(False)  # a private copy of the asset, so the mesh can be edited
        head = np.array([math.cos(bed.yaw), math.sin(bed.yaw)])
        for q in Usd.PrimRange(base):
            if q.IsA(UsdGeom.Mesh) and q.GetName() == "Section1":
                mesh = UsdGeom.Mesh(q)
                pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
                m = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                world_pts = np.c_[pts, np.ones(len(pts))] @ m
                world_pts[:, 2] += flatten_dz(world_pts[:, :2], (bed.x, bed.y), head)
                local = np.c_[world_pts[:, :3], np.ones(len(pts))] @ np.linalg.inv(m)
                mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, r[:3])) for r in local])
                log.info("bed %s: head section laid flat (%d points)", bed.id, len(pts))

    # ---- binding after play -------------------------------------------------------------------
    def _agent(self, pid: str):
        import omni.anim.behavior.core as bh

        a = bh.acquire_interface().get_agent(self.skelroots[pid])
        return None if a is None else IraAgent(a, self.skelroots[pid], bedside=self.bedsides.get(pid))

    def bind(self, app, bodies, world: "StubWorld", settle_s: float = SETTLE_MAX_S, fps: int = 30) -> None:
        """Give every character's agent to a controller, start the people who begin seated and let them sit."""
        agents: dict[str, IraAgent] = {}
        for k in range(AGENT_WAIT_FRAMES):
            app.update()
            for pid in self.skelroots:
                if pid not in agents:
                    a = self._agent(pid)
                    if a is not None:
                        agents[pid] = a
            if len(agents) == len(self.skelroots):
                break
        missing = sorted(set(self.skelroots) - set(agents))
        if missing:
            log.error("no Behavior agent for %s after %d frames; they stay kinematic", ", ".join(missing), k + 1)
        seated = []
        bodies.props.update(self.props)
        self._choose_bedsides(world)
        for pid, api in agents.items():
            ctrl = pe.PersonController(pid, api, bed_prop=self.props.get(pid))
            if ctrl.in_bed:
                api.hide()  # the posed prop is what is seen; the walking character appears when they get up
            bodies.add_controller(world, ctrl)
            ctrl.start(world)
            if world.people[pid].posture == "seated":
                seated.append(pid)
        for k in range(int(settle_s * fps)):
            app.update()
            if all(self._seated(world, bodies, pid) for pid in seated):
                break
        log.info("people bound: %d agents, %d seated after %d frames", len(agents), len(seated), k + 1)

    def _choose_bedsides(self, world: "StubWorld") -> None:
        """Now that the NavMesh is baked: people in bed get up on a side from which they can walk to the corridor."""
        if not self.bedsides:
            return
        import omni.anim.navigation.core as nav

        nm = nav.acquire_interface().get_navmesh()
        anchor = (world.site.resolve("ward_doors") or (11.2, 7.25))
        for pid, (x, y, fx, fy) in list(self.bedsides.items()):
            bed = world.site.beds.get(world.people[pid].bed or "")
            if bed is None or nm is None:
                continue

            def reachable(px, py):
                path = nm.query_shortest_path((anchor[0], anchor[1], 0.0), (px, py, 0.0))
                pts = path.get_points() if path is not None and hasattr(path, "get_points") else path
                return bool(pts) and math.hypot(pts[-1][0] - px, pts[-1][1] - py) < 0.4

            nx, ny = pick_bedside(bed, world.site.grid.blocked, reachable)
            self.bedsides[pid] = (nx, ny, fx, fy)
            log.info("%s gets up at (%.2f, %.2f) beside bed %s", pid, nx, ny, bed.id)

    def _seated(self, world, bodies, pid: str) -> bool:
        api = bodies.controllers[pid].api
        seat = world.seat(world.people[pid].seat)
        x, y, _ = api.pose()
        return api.current_action() == "Sit" and math.hypot(x - seat.x, y - seat.y) < SEATED_WITHIN_M

    # ---- people who join during the run ------------------------------------------------------------
    def spawn_runtime(self, world: "StubWorld", pid: str):
        """The IsaacBodies spawner: a joiner gets a character after play. PENDING until the agent is registered."""
        p = world.people[pid]
        if not self._has_body(p):
            return None
        if pid not in self.skelroots:
            taken = {c.id for q, c in self.choice.items()}
            char = self.allow.choose(pid, p.inst.role, p.inst.sex, p.inst.age, taken)
            self.choice[pid] = char
            x, y, yaw = self._start_pose(world, p)
            self.skelroots[pid] = self._spawn_character(pid, char, x, y, yaw)
            self.pending[pid] = 0
            return pe.PENDING
        api = self._agent(pid)
        if api is None:
            self.pending[pid] = self.pending.get(pid, 0) + 1
            if self.pending[pid] > AGENT_WAIT_FRAMES:
                log.error("no Behavior agent for the joiner %s; it stays kinematic", pid)
                return None
            return pe.PENDING
        self.pending.pop(pid, None)
        ctrl = pe.PersonController(pid, api)
        ctrl.start(world)
        return ctrl
