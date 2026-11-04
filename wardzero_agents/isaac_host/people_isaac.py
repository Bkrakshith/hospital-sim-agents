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
STATIC_ROOT = "/World/StaticPatients"
SEAT_HIPS_Z = 0.62  # sit effector 6 cm above the 0.56 m cushion top of the lobby chairs (spike S1 Q2)
BED_TOP_Z = 0.86  # top of the mattress of SM_HospitalBed_01b (the stage's Section1 spans 0.77 to 1.28 with the head raised)
LYING_THICKNESS_M = 0.13  # a supine body's back is this far below its centre line
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
    """Row-vector 4x4 (as 16 numbers) that lays a standing character on its back on a bed: head towards the bed's head
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
    oz = top_z + LYING_THICKNESS_M
    return (x_axis[0], x_axis[1], x_axis[2], 0.0, y_axis[0], y_axis[1], y_axis[2], 0.0,
            z_axis[0], z_axis[1], z_axis[2], 0.0, ox, oy, oz, 1.0)


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

    def __init__(self, agent, prim_path: str, carb_module=None):
        import carb
        self._carb = carb_module or carb
        self.a = agent
        self.path = prim_path

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
        try:
            self.a.set_enabled(False)
        except Exception:
            pass


class IsaacPeople:
    def __init__(self, stage, loaded, allow: Optional[ch.AllowList] = None):
        self.stage = stage
        self.loaded = loaded
        self.allow = allow or ch.load_allowlist()
        self.skelroots: dict[str, str] = {}  # person id -> SkelRoot prim path
        self.choice: dict[str, ch.Character] = {}
        self.static: list[str] = []
        self.pending: dict[str, int] = {}  # runtime joiners waiting for their agent -> frames waited
        self._motion_lib: Optional[str] = None

    # ---- spawning ---------------------------------------------------------------------------
    def spawn(self, world: "StubWorld", app=None) -> None:
        self._seats(world)
        brains = [(pid, p) for pid, p in world.people.items() if self._has_body(p)]
        self.choice = self.allow.assign([(pid, p.inst.role, p.inst.sex, p.inst.age) for pid, p in brains])
        starts = spread([self._start_pose(world, p) for _, p in brains])
        for (pid, p), (x, y, yaw) in zip(brains, starts):
            self.skelroots[pid] = self._spawn_character(pid, self.choice[pid], x, y, yaw)
        for pid, p in world.people.items():
            if p.kind == "static" or (p.bed and p.posture == "lying"):
                self._lying(pid, p, world)
        log.info("spawned %d characters (%s) and %d lying patients", len(self.skelroots),
                 ", ".join(f"{pid}={self.choice[pid].id}" for pid in self.skelroots), len(self.static))

    @staticmethod
    def _has_body(p) -> bool:
        return p.kind != "static" and not (p.bed and p.posture == "lying")

    def _start_pose(self, world, p) -> tuple[float, float, float]:
        if p.posture == "seated" and p.seat:
            seat = world.seat(p.seat)
            ax, ay = pe.seat_approach(seat)
            return ax, ay, seat.yaw + math.pi  # in front of the seat, facing it: the first task sits them down
        return p.x, p.y, p.yaw

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

    def _spawn_character(self, pid: str, char: ch.Character, x: float, y: float, yaw: float) -> str:
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
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.0))
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

    def _lying(self, pid: str, p, world: "StubWorld") -> None:
        """A patient in a bed: a posed character on the mattress, no Behavior System (so no animation)."""
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

        bed = world.site.beds.get(p.bed)
        if bed is None:
            return
        char = self.allow.choose(pid, "patient", p.inst.sex, p.inst.age)
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
        xf.AddTransformOp().Set(Gf.Matrix4d(*lying_matrix(bed)))
        self.static.append(pid)

    # ---- binding after play -------------------------------------------------------------------
    def _agent(self, pid: str):
        import omni.anim.behavior.core as bh

        a = bh.acquire_interface().get_agent(self.skelroots[pid])
        return None if a is None else IraAgent(a, self.skelroots[pid])

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
        for pid, api in agents.items():
            ctrl = pe.PersonController(pid, api)
            bodies.add_controller(world, ctrl)
            ctrl.start(world)
            if world.people[pid].posture == "seated":
                seated.append(pid)
        for k in range(int(settle_s * fps)):
            app.update()
            if all(self._seated(world, bodies, pid) for pid in seated):
                break
        log.info("people bound: %d agents, %d seated after %d frames", len(agents), len(seated), k + 1)

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
