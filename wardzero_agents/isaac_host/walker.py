"""The walking G1 in Isaac Sim: GEAR-SONIC drives the floating-base robot from a PhysX pre-step callback (Kit side).

What it does is the deploy contract the Isaac gate test proved (``research_notes/isaac_host/sonic_isaac_report.md``;
the numbers are in ``walking``):

* ``author`` (in the robot's authoring pass, before PhysX parses it): the 29 body joints in force mode with zero drive
  gains, our USD's effort and armature; the fingers held open by stiff position drives; hand colliders filtered
  against the legs (articulation self-collision is off anyway, so this only guards a USD that turns it on);
* ``start`` (after one update): a ``omni.physics.tensors`` view of the articulation, the joint map by name, SONIC from
  the weights, and one pre-step physics callback that, every physics step (200 Hz), does the PD
  ``tau = kp (q* - q) - kd dq`` clipped to effort and sends it as joint actuation force; every 4th step (50 Hz) one
  SONIC tick on the pelvis IMU (wxyz quaternion, body-frame gyro); the planner decides every 5th tick and runs in a
  worker thread. During start-up the pelvis is held (pose and zero velocity, every step) while the joints ramp to
  SONIC's default pose, lowered, and released on the tick the planner starts;
* the walking commands come from the world's ``LeggedBody`` (its ``RouteFollower``), the measured pose goes back to it
  through ``pose`` (``IsaacBodies.read_back``), and a fall is reported to it once.

Every Kit import is inside the function that uses it.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import numpy as np

from . import walking as W
from .sonic.quat import qmat, tilt_deg, yaw_of

log = logging.getLogger("wardzero.isaac.walker")

LOG_EVERY_S = 0.5  # sim seconds between robot_walk records


DOOR_ROOT = "/World/Fixtures/Doors"


def door_leaf_segments(stage, root: str = DOOR_ROOT, skip=()) -> list:
    """Each door leaf (a rigid body under the doors) as the floor segment along its long side, from its world-space
    oriented box, as it stands now. Kit side; ``legged_body.block_segments`` puts them in the G1's router. Doors named
    in ``skip`` swing (``doors_isaac``): the G1 walks through them, pushing the leaf."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out = []
    skip = set(skip)
    prims = []
    for r in (root, root + "1"):
        top = stage.GetPrimAtPath(r)
        if top and top.IsValid():
            prims += [p for p in Usd.PrimRange(top) if p.GetParent().GetName() not in skip]
    for p in prims:
        if not p.HasAPI(UsdPhysics.RigidBodyAPI) or not p.GetName().startswith("Leaf"):
            continue
        box = cache.ComputeWorldBound(p)
        r = box.GetRange()
        if r.IsEmpty():
            continue
        m = box.GetMatrix()
        lo, hi = r.GetMin(), r.GetMax()
        corners = [m.Transform(Gf.Vec3d(x, y, lo[2])) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
        # the long side in the floor plane: the pair of opposite edge midpoints farthest apart
        a, b, c, d = ((v[0], v[1]) for v in corners)  # (x0,y0) (x0,y1) (x1,y0) (x1,y1)
        mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)  # noqa: E731
        s1 = (mid(a, b), mid(c, d))
        s2 = (mid(a, c), mid(b, d))
        seg = max((s1, s2), key=lambda s: math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
        out.append(seg)
    return out


class SonicWalker:
    def __init__(self, stage, robot_path: str, spawn: tuple[float, float, float], body, *,
                 runlog=None, weights_dir=None, threads: int = 4, on_fall: Optional[Callable[[str], None]] = None):
        self.stage, self.robot_path = stage, robot_path
        self.spawn = tuple(float(v) for v in spawn)  # x, y, yaw
        self.body = body  # world.legged_body.LeggedBody
        self.runlog, self.weights_dir, self.threads = runlog, weights_dir, threads
        self.on_fall = on_fall
        self.art = self.view = self.jm = self.runner = self.ctrl = None
        self._sub = None
        self.sched = W.StepSchedule()
        self.root: Optional[np.ndarray] = None  # x y z qx qy qz qw of the pelvis, last step
        self.targets = None
        self.yaw0: Optional[float] = None
        self._facing = (1.0, 0.0, 0.0)
        self._forces = None
        self._idx = None
        self._quat_spawn_xyzw = None
        self._frame_steps = 0
        self.frame_steps: list[int] = []
        self.cb_ms: list[float] = []
        self.max_tilt = 0.0
        self.sat_steps = 0
        self.policy_steps = 0
        self._next_log = 0.0
        self.fell = None
        self.error: Optional[str] = None
        self.stats: dict = {}

    # ---- authoring (before PhysX parses the robot) -------------------------------------------------
    def author(self) -> dict:
        """Joint drives, effort and armature, finger drives, colliders on and hand-leg filtering. Returns counts."""
        from pxr import PhysxSchema, Usd, UsdPhysics

        prim = self.stage.GetPrimAtPath(self.robot_path)
        joints = [p for p in Usd.PrimRange(prim) if p.IsA(UsdPhysics.RevoluteJoint)]
        by_name = {p.GetName(): p for p in joints}
        n_body = 0
        for s in W.joint_settings(list(by_name)):
            p = by_name[s.name]
            d = UsdPhysics.DriveAPI.Apply(p, "angular")
            d.CreateStiffnessAttr(float(s.stiffness))
            d.CreateDampingAttr(float(s.damping))
            d.CreateTargetPositionAttr(0.0)
            if s.max_force is not None:
                d.CreateMaxForceAttr(float(s.max_force))
                n_body += 1
            if s.armature is not None:
                PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(float(s.armature))
        hands, legs = [], []
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                if W.is_hand_link(p.GetName()):
                    hands.append(p)
                elif W.is_leg_link(p.GetName()):
                    legs.append(p.GetPath())
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(True)
        for h in hands:
            UsdPhysics.FilteredPairsAPI.Apply(h).CreateFilteredPairsRel().SetTargets(legs)
        out = {"body_joints": n_body, "finger_joints": len(by_name) - n_body, "hand_links": len(hands),
               "leg_links": len(legs)}
        if n_body != 29:
            raise RuntimeError(f"the walking G1 needs SONIC's 29 body joints; the robot USD has {n_body}")
        log.info("walker authored: %s", out)
        return out

    # ---- start (after one update: PhysX has the articulation) --------------------------------------
    def start(self, art_root: str) -> None:
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        from .sonic import JointMap, SonicController, SonicRunner

        self.view = omni.physics.tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        self.view.set_subspace_roots("/")
        self.art = self.view.create_articulation_view(art_root)
        names = list(self.art.shared_metatype.dof_names)
        self.jm = JointMap(names)
        self._idx = np.arange(1, dtype=np.int32)
        self._forces = np.zeros((1, self.jm.n_sim), np.float32)
        x, y, yaw = self.spawn
        self._quat_spawn_xyzw = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        t0 = time.monotonic()
        self.ctrl = SonicController.from_weights(self.weights_dir, threads=self.threads, planner_threaded=True)
        self.runner = SonicRunner(self.ctrl, ramp_s=W.RAMP_S, settle_s=0.0, idle_s=W.IDLE_S, lower_s=W.LOWER_S,
                                  harness_z=W.SPAWN_Z, low_z=W.LOW_Z)
        self.stats["load_s"] = round(time.monotonic() - t0, 2)
        self.stats["dofs"] = self.jm.n_sim
        self._hold()  # place it now; the callback holds it from the next step
        self._sub = omni.physx.get_physx_interface().subscribe_physics_on_step_events(self._on_step, True, 0)
        log.info("walker up: %d DOFs (%d SONIC joints by name), SONIC loaded in %.1f s", self.jm.n_sim, 29,
                 self.stats["load_s"])

    def _hold(self, z: float = W.SPAWN_Z) -> None:
        x, y, _ = self.spawn
        self.art.set_root_transforms(np.array([[x, y, z, *self._quat_spawn_xyzw]], np.float32), self._idx)
        self.art.set_root_velocities(np.zeros((1, 6), np.float32), self._idx)

    # ---- the physics callback ------------------------------------------------------------------------
    def _on_step(self, dt: float) -> None:
        try:
            self._step()
        except Exception as exc:  # one bad step must not take Kit down; stop driving and say why
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("walker step failed; the G1 is no longer driven")
                self._report_fall(f"walking controller error: {self.error}")
            self._sub = None

    def _step(self) -> None:
        t0 = time.perf_counter()
        self._frame_steps += 1
        art, jm = self.art, self.jm
        q_all = np.asarray(art.get_dof_positions()).reshape(-1)
        dq_all = np.asarray(art.get_dof_velocities()).reshape(-1)
        root = np.asarray(art.get_root_transforms()).reshape(7).copy()  # pelvis: x y z qx qy qz qw
        self.root = root
        q, dq = jm.to_sonic(q_all), jm.to_sonic(dq_all)
        sch = self.sched
        if sch.tick_now() or self.targets is None:
            vel = np.asarray(art.get_root_velocities()).reshape(6)
            wxyz = np.array([root[6], root[3], root[4], root[5]], float)
            gyro_b = qmat(wxyz).T @ vel[3:6]
            self._command()
            out = self.runner.tick(sch.t, q, dq, wxyz, gyro_b, float(root[2]))
            self.targets = out
            if out.phase.value == "fallen" and self.fell is None:
                self._report_fall(out.fall or "fell")
            if self.runner.ready and not self.body.ready:
                self.body.ready = True
                log.info("walker ready at sim %.2f s (pelvis %.3f m)", sch.t, root[2])
            if self.yaw0 is None and self.ctrl.ready:
                self.yaw0 = yaw_of(wxyz)  # the planner frame: the heading when the planner started
            self.max_tilt = max(self.max_tilt, tilt_deg(wxyz)) if self.ctrl.ready else self.max_tilt
            self._log(sch.t, root, wxyz)
        out = self.targets
        if out.harness:
            self._hold(out.harness_z if out.harness_z is not None else W.SPAWN_Z)
        tg = out.targets
        effort = W.EFFORT_OURS
        raw = tg.kp * (tg.q - q) - tg.kd * dq
        tau = np.clip(raw, -effort, effort)
        if self.ctrl.ready:
            self.policy_steps += 1
            self.sat_steps += bool((np.abs(raw) >= effort - 1e-6).any())
        self._forces[0, jm.sim_index] = tau
        art.set_dof_actuation_forces(self._forces, self._idx)
        sch.advance()
        if len(self.cb_ms) < 200000:
            self.cb_ms.append((time.perf_counter() - t0) * 1e3)

    def _command(self) -> None:
        """The body's latest follower command, in the planner frame, for the runner (applied once READY)."""
        if self.yaw0 is None or self.fell:
            return
        cmd = W.planner_command(self.body.command, self.yaw0, self._facing)
        self._facing = cmd.facing
        self.runner.command(cmd)

    def _report_fall(self, why: str) -> None:
        self.fell = why
        x, y = (float(self.root[0]), float(self.root[1])) if self.root is not None else self.spawn[:2]
        log.error("the G1 fell at sim %.2f s at (%.2f, %.2f): %s", self.sched.t, x, y, why)
        try:
            self.body.report_fall(why)
        finally:
            if self.on_fall is not None:
                self.on_fall(why)

    def _log(self, t: float, root, wxyz) -> None:
        if self.runlog is None or t + 1e-9 < self._next_log:
            return
        self._next_log = t + LOG_EVERY_S
        c = self.body.command
        try:
            self.runlog.write("robot_walk", {
                "t_phys": round(t, 3), "phase": self.runner.phase.value, "x": round(float(root[0]), 3),
                "y": round(float(root[1]), 3), "z": round(float(root[2]), 3), "yaw": round(yaw_of(wxyz), 3),
                "tilt_deg": round(tilt_deg(wxyz), 1), "mode": int(c.mode), "speed": round(float(c.speed), 2),
                "status": c.status, "face": None if c.face_yaw is None else round(float(c.face_yaw), 3)})
        except Exception as exc:  # the log is a nicety: never let it stop the legs
            log.warning("robot_walk record failed: %s", exc)
            self.runlog = None

    # ---- per frame (host) ----------------------------------------------------------------------------
    def begin_frame(self) -> None:
        self._frame_steps = 0

    def end_frame(self) -> None:
        if len(self.frame_steps) < 200000:
            self.frame_steps.append(self._frame_steps)

    def pose(self) -> Optional[tuple[float, float, float]]:
        """(x, y, yaw) of the pelvis, as of the last physics step."""
        r = self.root
        if r is None:
            return None
        return (float(r[0]), float(r[1]), yaw_of(np.array([r[6], r[3], r[4], r[5]], float)))

    def summary(self) -> dict:
        def pct(v, q):
            return round(float(np.percentile(v, q)), 2) if len(v) else None
        tm = self.ctrl.timing if self.ctrl is not None else {}
        fs = [n for n in self.frame_steps if n > 0]
        s = dict(self.stats)
        s.update(phase=self.runner.phase.value if self.runner else None, fell=self.fell, error=self.error,
                 max_tilt_deg=round(self.max_tilt, 1), physics_steps=self.sched.step,
                 sim_s=round(self.sched.t, 2),
                 ankle_or_any_saturation_pct=round(100.0 * self.sat_steps / self.policy_steps, 2)
                 if self.policy_steps else None,
                 steps_per_frame={"min": min(fs), "max": max(fs), "median": float(np.median(fs))} if fs else None,
                 callback_ms={"median": pct(self.cb_ms, 50), "p99": pct(self.cb_ms, 99)},
                 tick_ms={"median": pct(tm.get("tick", []), 50), "p99": pct(tm.get("tick", []), 99)},
                 planner_ms_last=round(self.ctrl.planner.last_ms, 1) if self.ctrl is not None else None,
                 replans=self.ctrl.replans if self.ctrl is not None else 0)
        return s

    def close(self) -> None:
        self._sub = None
        if self.ctrl is not None:
            self.ctrl.close()
