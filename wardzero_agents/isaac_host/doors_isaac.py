"""Door physics in the Isaac stage (Kit side): hinged leaves with a closer on the run's routes, the people's capsules,
both avoidance ignore lists, the slow zone at the leaves, and the per-step closer damping and capsule targets.
The numbers and the reasons are in ``doors``; every Kit import is inside the function that uses it.

Order (``app.IsaacHost``): ``author`` in the physics step (before play, before the NavMesh bake), ``add_capsules``
after the people are spawned (before the bake and play), ``start`` after play (a tensor view per door and one for the
capsules, and one pre-step physics callback), ``bind_agents`` once the people have their agents, ``end_frame`` after
every frame.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import numpy as np

from . import doors as D

log = logging.getLogger("wardzero.isaac.doors")


def find_door(stage, name: str) -> Optional[str]:
    for root in D.DOOR_ROOTS:
        p = stage.GetPrimAtPath(f"{root}/{name}")
        if p and p.IsValid():
            return str(p.GetPath())
    return None


def _yaw_deg(xf) -> float:
    from pxr import Gf

    m = xf.GetLocalTransformation()
    r = Gf.Matrix4d(m).ExtractRotation()
    v = r.TransformDir(Gf.Vec3d(1, 0, 0))
    return math.degrees(math.atan2(v[1], v[0]))


class IsaacDoors:
    def __init__(self, stage, runlog=None):
        self.stage, self.runlog = stage, runlog
        self.doors: dict[str, str] = {}  # name -> prim path
        self.leaves: list[D.Leaf] = []
        self.leaf_dof: list[tuple[int, int]] = []  # per leaf: (door view index, dof index)
        self.views: list = []
        self.dof_names: list[list[str]] = []
        self.capsule_paths: list[str] = []
        self.tracker: Optional[D.CapsuleTracker] = None
        self.caps = None
        self.view = None
        self._sub = None
        self._j = 0
        self._dt = 0.005
        self.q = np.zeros(0)
        self._ignored: set[str] = set()
        self._slow: dict[str, bool] = {}
        self._pen_run: dict[str, float] = {}
        self._last_t: Optional[float] = None
        self.stats = {"nonfinite_steps": 0, "past_limit_steps": 0, "max_abs_deg": 0.0, "max_rad_s": 0.0,
                      "steps": 0, "max_penetration_m": 0.0, "through_leaf": [], "slowed": 0}

    # ---- before play ---------------------------------------------------------------------------------
    def author(self, names) -> list[str]:
        """Make each named door a closed, double-action hinged door with a closer; returns the doors found."""
        import NavSchema
        import omni.kit.commands
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        st = self.stage
        for name in names:
            path = find_door(st, name)
            if path is None:
                continue
            root = st.GetPrimAtPath(path)
            joints = {p.GetName(): p for p in root.GetChildren() if p.IsA(UsdPhysics.RevoluteJoint)}
            ok = False
            for leaf_name in D.LEAF_NAMES:
                leaf = st.GetPrimAtPath(f"{path}/{leaf_name}")
                j = joints.get(f"fx_{leaf_name.lower()}")
                if not leaf or j is None:
                    continue
                xf = UsdGeom.Xformable(leaf)
                tr = Gf.Vec3d(0, 0, 0)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        tr = Gf.Vec3d(op.Get())
                state = float(j.GetAttribute("state:angular:physics:position").Get() or 0.0)
                yaw = D.closed_yaw(_yaw_deg(xf), state)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(tr)
                h = math.radians(yaw) / 2
                xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), 0, 0, math.sin(h)))
                ma = UsdPhysics.MassAPI(leaf)
                m0 = ma.GetMassAttr().Get() or D.MASS_KG
                i0 = ma.GetDiagonalInertiaAttr().Get()
                ma.CreateMassAttr(float(D.MASS_KG))
                if i0 is not None:
                    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) * D.MASS_KG / m0 for v in i0]))
                rj = UsdPhysics.RevoluteJoint(j)
                rj.CreateLowerLimitAttr(-D.LIMIT_DEG)
                rj.CreateUpperLimitAttr(D.LIMIT_DEG)
                j.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(0.0)
                j.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
                d = UsdPhysics.DriveAPI.Apply(j, "angular")
                d.CreateTypeAttr("force")
                d.CreateTargetPositionAttr(0.0)
                d.CreateTargetVelocityAttr(0.0)
                d.CreateStiffnessAttr(D.per_degree(D.STIFFNESS))
                d.CreateDampingAttr(D.per_degree(D.DAMPING_CLOSE))
                d.CreateMaxForceAttr(D.MAX_FORCE)
                PhysxSchema.PhysxJointAPI.Apply(j).CreateMaxJointVelocityAttr(math.degrees(D.MAX_SPEED))
                PhysxSchema.PhysxRigidBodyAPI.Apply(leaf).CreateMaxDepenetrationVelocityAttr(D.LEAF_DEPENETRATION)
                UsdPhysics.RigidBodyAPI(leaf).CreateRigidBodyEnabledAttr(True)
                hinge = UsdGeom.Xformable(leaf.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) \
                    .Transform(tr)
                x0, x1, ht = self._collider(leaf)
                self.leaves.append(D.Leaf(name, leaf_name, (float(hinge[0]), float(hinge[1])), yaw, x0, x1, ht))
                ok = True
            if not ok:
                continue
            pa = PhysxSchema.PhysxArticulationAPI.Apply(root)
            pa.CreateSolverPositionIterationCountAttr(D.SOLVER_ITERATIONS[0])
            pa.CreateSolverVelocityIterationCountAttr(D.SOLVER_ITERATIONS[1])
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=Sdf.Path(path), api=NavSchema.NavMeshExcludeAPI)
            self.doors[name] = path
        log.info("hinged doors with a closer: %s (%d leaves)", ", ".join(self.doors) or "none", len(self.leaves))
        return list(self.doors)

    @staticmethod
    def _collider(leaf) -> tuple[float, float, float]:
        from pxr import UsdGeom

        c = leaf.GetChild("Collider")
        if not c or not c.IsA(UsdGeom.Cube):
            return 0.005, 0.96, 0.027
        size = float(UsdGeom.Cube(c).GetSizeAttr().Get() or 2.0)
        tx = sx = sy = None
        for op in UsdGeom.Xformable(c).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                tx = op.Get()[0]
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                sx, sy = op.Get()[0], op.Get()[1]
        if tx is None or sx is None:
            return 0.005, 0.96, 0.027
        hx, hy = size / 2 * sx, size / 2 * sy
        return float(tx - hx), float(tx + hx), float(hy)

    def add_capsules(self, n: int) -> None:
        """n kinematic capsules (one per person plus spares), parked, and the collision groups."""
        import NavSchema
        import omni.kit.commands
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        st = self.stage
        UsdGeom.Scope.Define(st, D.CAPSULES_ROOT)
        caps = UsdPhysics.CollisionGroup.Define(st, f"{D.GROUPS_ROOT}/PeopleCapsules")
        proxy = UsdPhysics.CollisionGroup.Define(st, f"{D.GROUPS_ROOT}/RobotProxy")
        robot = UsdPhysics.CollisionGroup.Define(st, f"{D.GROUPS_ROOT}/Robot")
        caps.CreateFilteredGroupsRel().SetTargets([caps.GetPath(), proxy.GetPath(), robot.GetPath()])
        proxy.CreateFilteredGroupsRel().SetTargets([caps.GetPath()])
        robot.CreateFilteredGroupsRel().SetTargets([caps.GetPath()])
        proxy.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/RobotProxy")
        robot.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Robot")
        zc = (D.CAPSULE_Z0 + D.CAPSULE_Z1) / 2
        for i in range(n):
            path = f"{D.CAPSULES_ROOT}/Cap{i:02d}"
            cap = UsdGeom.Capsule.Define(st, path)
            cap.CreateRadiusAttr(D.CAPSULE_R)
            cap.CreateHeightAttr(max(0.0, D.CAPSULE_Z1 - D.CAPSULE_Z0 - 2 * D.CAPSULE_R))
            cap.CreateAxisAttr("Z")
            cap.CreatePurposeAttr("guide")
            xf = UsdGeom.Xformable(cap)
            xf.AddTranslateOp().Set(Gf.Vec3d(D.PARK[0] - 2.0 * i, D.PARK[1], zc))
            xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
            pc = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
            pc.CreateContactOffsetAttr(D.CAPSULE_CONTACT_OFFSET)
            pc.CreateRestOffsetAttr(0.0)
            UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim()).CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(cap.GetPrim()).CreateMassAttr(D.CAPSULE_MASS)
            caps.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(cap.GetPath())
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=cap.GetPath(), api=NavSchema.NavMeshExcludeAPI)
            self.capsule_paths.append(path)
        self.tracker = D.CapsuleTracker(n)
        log.info("%d people capsules (kinematic, radius %.2f m, z %.2f-%.2f m)", n, D.CAPSULE_R, D.CAPSULE_Z0,
                 D.CAPSULE_Z1)

    # ---- after play -----------------------------------------------------------------------------------
    def start(self, physics_dt: float) -> None:
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        self._dt = physics_dt
        if not self.doors and not self.capsule_paths:
            return
        self.view = omni.physics.tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        self.view.set_subspace_roots("/")
        index = {}
        for name, path in self.doors.items():
            v = self.view.create_articulation_view(path)
            index[name] = len(self.views)
            self.views.append(v)
            self.dof_names.append(list(v.shared_metatype.dof_names))
        for lf in self.leaves:
            vi = index[lf.door]
            self.leaf_dof.append((vi, self.dof_names[vi].index(f"fx_{lf.name.lower()}")))
        self.q = np.zeros(len(self.leaves))
        if self.capsule_paths:
            self.caps = self.view.create_rigid_body_view(f"{D.CAPSULES_ROOT}/*")
        self._sub = omni.physx.get_physx_interface().subscribe_physics_on_step_events(self._on_step, True, 1)
        log.info("door physics up: %d doors, %d capsules", len(self.views), len(self.capsule_paths))

    def _on_step(self, dt: float) -> None:
        try:
            self._step()
        except Exception:
            log.exception("door step failed; doors and capsules are no longer driven")
            self._sub = None

    def _step(self) -> None:
        s = self.stats
        s["steps"] += 1
        idx = np.arange(1, dtype=np.int32)
        for vi, v in enumerate(self.views):
            q = np.asarray(v.get_dof_positions()).reshape(-1)
            w = np.asarray(v.get_dof_velocities()).reshape(-1)
            if not (np.isfinite(q).all() and np.isfinite(w).all()):
                s["nonfinite_steps"] += 1
                continue
            a = float(np.degrees(np.abs(q)).max()) if q.size else 0.0
            s["max_abs_deg"] = max(s["max_abs_deg"], a)
            s["max_rad_s"] = max(s["max_rad_s"], float(np.abs(w).max()) if w.size else 0.0)
            if a > D.LIMIT_DEG + 0.5:
                s["past_limit_steps"] += 1
            d = np.asarray(v.get_dof_dampings()).reshape(1, -1).copy()
            for i in range(q.size):
                d[0, i] = D.closer_damping(float(q[i]), float(w[i]))
            v.set_dof_dampings(d.astype(np.float32), idx)
        for k, (vi, di) in enumerate(self.leaf_dof):
            q = np.asarray(self.views[vi].get_dof_positions()).reshape(-1)
            if np.isfinite(q[di]):
                self.q[k] = q[di]
        if self.caps is not None and self.tracker is not None:
            self._j += 1
            zc = (D.CAPSULE_Z0 + D.CAPSULE_Z1) / 2
            tgt = np.array([(x, y, zc, 0, 0, 0, 1) for x, y in self.tracker.targets(self._j, self._dt)], np.float32)
            self.caps.set_kinematic_targets(tgt, np.arange(len(tgt), dtype=np.int32))

    # ---- people ------------------------------------------------------------------------------------------
    def bind_agents(self, controllers: dict) -> None:
        """Every agent ignores the door roots, the leaves and every capsule, in both avoidance lists (else a person
        dodges the leaf they should push, or their own capsule)."""
        paths = list(self.doors.values()) + [f"{p}/{lf}" for p in self.doors.values() for lf in D.LEAF_NAMES] + \
            self.capsule_paths
        for pid, ctrl in controllers.items():
            if pid in self._ignored:
                continue
            a = getattr(getattr(ctrl, "api", None), "a", None)
            if a is None:
                continue
            for p in paths:
                for fn in ("add_obstacle_avoidance_ignored_object", "add_auto_avoidance_ignored_object"):
                    try:
                        getattr(a, fn)(p)
                    except Exception as exc:
                        log.debug("%s(%s): %s", fn, p, exc)
            self._ignored.add(pid)

    def end_frame(self, poses: dict, controllers: dict, t: float) -> None:
        """poses: person id -> (x, y) or None (no body now). Capsules follow, new agents get the ignore lists, and
        people near a leaf slow down. Penetration of a capsule into a leaf is logged (> 5 cm for > 0.2 s)."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        frame_dt = dt if dt > 0 else self._dt * 8
        self._last_t = t
        if self.tracker is not None:
            self.tracker.frame(poses, frame_dt)
            self._j = 0
        if len(self._ignored) < len(controllers):
            self.bind_agents(controllers)
        if not self.leaves:
            return
        qs = list(self.q)
        for pid, xy in poses.items():
            if xy is None:
                continue
            near = D.near_leaf(self.leaves, qs, xy[0], xy[1])
            if near != self._slow.get(pid, False):
                self._slow[pid] = near
                api = getattr(controllers.get(pid), "api", None)
                if api is not None and hasattr(api, "set_door_slow"):
                    api.set_door_slow(near)
                    self.stats["slowed"] += int(near)
            depth = max(D.CAPSULE_R - lf.dist(q, xy[0], xy[1]) for lf, q in zip(self.leaves, qs))
            self.stats["max_penetration_m"] = max(self.stats["max_penetration_m"], round(depth, 3))
            run = self._pen_run.get(pid, 0.0) + frame_dt if depth > 0.05 else 0.0
            self._pen_run[pid] = run
            if run > 0.2 and (not self.stats["through_leaf"] or self.stats["through_leaf"][-1]["who"] != pid):
                self.stats["through_leaf"].append({"who": pid, "t": round(t, 2), "x": round(xy[0], 2),
                                                   "y": round(xy[1], 2), "depth": round(depth, 3)})
                log.warning("%s is inside a door leaf at (%.2f, %.2f)", pid, xy[0], xy[1])

    def leaf_angles_deg(self) -> dict:
        return {f"{lf.door}/{lf.name}": round(math.degrees(q), 1) for lf, q in zip(self.leaves, self.q)}

    def summary(self) -> dict:
        s = dict(self.stats)
        s["doors"] = list(self.doors)
        s["max_abs_deg"] = round(s["max_abs_deg"], 1)
        s["max_rad_s"] = round(s["max_rad_s"], 2)
        s["capsule_overflow"] = sorted(self.tracker.overflow) if self.tracker else []
        return s

    def close(self) -> None:
        self._sub = None
