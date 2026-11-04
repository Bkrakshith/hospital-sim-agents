"""The robot in the Isaac stage (I4): reference its ready USD after ``timeline.play()``, build its OmniGraph on the robot
domain, and each frame move the arm, throttle the obstacle cylinder and feed the detection link.

Everything here needs Kit (pxr, omni.*); each import is inside the function that uses it. The decisions are in
``robot`` (poses, throttle, detection link) and ``robot_graph`` (the graph as data), both tested without Isaac.

The order is the one S1b found: the robot is added after ``play()`` so the Behavior System never registers its links as
dynamic NavMesh obstacles (an awake G1 would cost 220 ms a frame), in one authoring pass (the reference and its pose,
no ``app.update()`` between), as a ready USD (``robots.sanitise``) so PhysX never parses it half-edited. The people
avoid the robot through one cylinder created before play (``stage.add_robot_proxy``), which only moves with a mobile
robot. A built-in controller (``builtin_claude``) gets arm gestures; a developer's own controller owns its joints.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Optional

from ..robots import sanitise
from ..robots.profile import RobotProfile
from . import robot as rb
from . import stage as stage_mod
from .robot_graph import CAMERA_HZ, PrimPath, build_robot_graph

log = logging.getLogger("wardzero.isaac.robot")

ROBOT_PATH = "/World/Robot"
SETTLE_UPDATES = 5  # PhysX parses the articulation while playing (S1b)


class IsaacRobot:
    def __init__(self, stage, app, loaded, profile: RobotProfile, usd: Path, fps: int, robot_domain: int,
                 private_domain: int):
        self.cfg = loaded.scenario.robot
        # the scenario's namespace is the one SimRos and the robot use: the graph follows it, not robot.yaml's
        self.profile, self.downgraded = rb.effective_profile(dataclasses.replace(profile, ns=self.cfg.ns))
        self.stage, self.app, self.loaded, self.usd = stage, app, loaded, Path(usd)
        self.fps, self.robot_domain, self.private_domain = fps, robot_domain, private_domain
        self.art_root: Optional[str] = None
        self.base_path: Optional[str] = None
        self.camera_path: Optional[str] = None
        self.render_product: Optional[str] = None
        self.graph = None
        self.link: Optional[rb.DetectionLink] = None
        self.art = None
        self.arm = None
        self._arm_idx = None
        self._proxy = None
        self.gestures = profile.controller == "builtin_claude" and profile.name == "g1_claude"
        self.stats = {"proxy_moves": 0, "arm_writes": 0}

    # ---- step 9: one authoring pass --------------------------------------------------------
    def add(self, world=None) -> None:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        layer = Sdf.Layer.FindOrOpen(str(self.usd))
        if layer is None:
            raise FileNotFoundError(f"robot USD {self.usd} is missing")
        sanitise.require_ready(dict(layer.customLayerData or {}), str(self.usd))
        c = self.cfg
        # --- no app.update() until the pose is authored: the robot must not be parsed while half placed ---
        xf = UsdGeom.Xform.Define(self.stage, ROBOT_PATH)
        prim = xf.GetPrim()
        prim.GetReferences().AddReference(str(self.usd))
        xf.AddTranslateOp().Set(Gf.Vec3d(c.x, c.y, self.profile.base_z))
        xf.AddRotateZOp().Set(math.degrees(c.yaw))
        roots = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError(f"{self.usd.name}: expected one articulation root, found {len(roots)}")
        self.art_root = str(roots[0].GetPath())
        self.base_path = self._base_link(prim, UsdPhysics, Usd) or self.art_root
        self.camera_path = self._camera(prim, Usd, UsdGeom)
        if self.camera_path is not None:
            prim_cam = self.stage.GetPrimAtPath(self.camera_path)  # the tick rate throttles the render product itself
            prim_cam.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(float(CAMERA_HZ))
            self.render_product = self._render_product()
        self._build_graph()
        for _ in range(SETTLE_UPDATES):
            self.app.update()
        if self.profile.can_move:
            self._proxy = rb.ProxyMover(c.x, c.y)
        if self.downgraded and world is not None:
            rb.immobilise(world)
        self._articulation()
        log.info("robot %s at (%.2f, %.2f) facing %.0f deg: articulation %s, camera %s, graph %d nodes", self.profile.name,
                 c.x, c.y, math.degrees(c.yaw), self.art_root, self.camera_path, len(self.graph.nodes))

    @staticmethod
    def _base_link(prim, UsdPhysics, Usd) -> Optional[str]:
        """The body the world anchor pins (the pelvis for a fixed-base G1): its pose is the robot's."""
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdPhysics.FixedJoint) and p.GetName() == "root_joint":
                body1 = UsdPhysics.Joint(p).GetBody1Rel().GetTargets()
                if body1:
                    return str(body1[0])
        return None

    def _camera(self, prim, Usd, UsdGeom) -> Optional[str]:
        """The colour camera of the first camera sensor (a catalogue asset has several; a generic sensor has one)."""
        for s in self.profile.sensors:
            if s.kind not in ("rgb", "rgbd"):
                continue
            holder = next((p for p in Usd.PrimRange(prim) if p.GetName() == f"wz_{_ident(s.id)}"), None)
            cams = [p for p in Usd.PrimRange(holder) if p.IsA(UsdGeom.Camera)] if holder is not None else []
            if cams:
                best = next((p for p in cams if "color" in p.GetName().lower()), cams[0])
                return str(best.GetPath())
            log.warning("sensor %s has no camera prim in %s", s.id, self.usd.name)
        return None

    def _render_product(self) -> str:
        import omni.replicator.core as rep

        sensor = next(s for s in self.profile.sensors if s.kind in ("rgb", "rgbd"))
        rp = rep.create.render_product(self.camera_path, rb.camera_resolution(sensor))
        return rp.path if hasattr(rp, "path") else str(rp)

    def _build_graph(self) -> None:
        import omni.graph.core as og
        from pxr import Sdf

        self.graph = g = build_robot_graph(self.profile, self.art_root, self.robot_domain, self.private_domain,
                                           self.render_product, self.fps)
        values = [(k, [Sdf.Path(v) if isinstance(v, PrimPath) else v for v in val] if isinstance(val, list)
                   else val) for k, val in g.values]
        keys = og.Controller.Keys
        og.Controller.edit({"graph_path": g.path, "evaluator_name": "execution"},
                           {keys.CREATE_NODES: list(g.nodes), keys.SET_VALUES: values, keys.CONNECT: list(g.connects)})

    def _articulation(self) -> None:
        if not self.gestures:
            return
        import numpy as np  # noqa: F401
        from isaacsim.core.experimental.prims import Articulation

        self.art = Articulation(self.art_root)
        self.arm = rb.ArmTargets(rb.G1_ARM_POSES)
        names = self.art.dof_names
        self.arm.joints = [j for j in self.arm.joints if j in names]
        self._arm_idx = self.art.get_dof_indices(self.arm.joints).numpy().flatten()

    # ---- ROS side (called once the bridges exist) --------------------------------------------
    def attach_ros(self, world, private, robot) -> None:
        """Raw boxes (private bridge) become the robot's bbox_3d (the robot's bridge, or the private one when shared)."""
        if self.graph is not None and self.graph.topics_private:
            self.link = rb.DetectionLink(world, self.profile.ns, private, robot or private,
                                         close_range_m=rb.CLOSE_RANGE_M)

    @property
    def graph_owned(self) -> tuple:
        """robot_topics keys the graph publishes, so SimRos does not."""
        return tuple(k for k in ("bbox_3d", "odom") if self.graph is not None and
                     (k in self.graph.topics or (k == "bbox_3d" and self.graph.topics_private)))

    @property
    def owns_clock(self) -> bool:
        return self.graph is not None and "clock" in self.graph.topics

    # ---- per frame ---------------------------------------------------------------------------
    def pose(self):
        """(x, y, yaw) of the base link in the site frame, for IsaacBodies."""
        import omni.usd

        prim = self.stage.GetPrimAtPath(self.base_path or self.art_root)
        if not prim.IsValid():
            return None
        return rb.pose_from_matrix(omni.usd.get_world_transform_matrix(prim))

    def update(self, world, t: float, dt: float) -> None:
        if self.link is not None:
            self.link.update(t)
        if self.arm is not None and self.art is not None and self.arm.joints:
            targets = self.arm.step(rb.arm_pose_name(world), dt)
            if targets is not None:
                import numpy as np

                self.art.set_dof_position_targets(np.array([[targets[j] for j in self.arm.joints]]),
                                                  dof_indices=self._arm_idx)
                self.stats["arm_writes"] += 1
        if self._proxy is not None:
            r = world.robot
            move = self._proxy.update(r.x, r.y, t)
            if move is not None:
                from pxr import Gf, UsdGeom

                prim = self.stage.GetPrimAtPath(stage_mod.ROBOT_PROXY_PATH)
                if prim.IsValid():
                    UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(move[0], move[1], 0.7))
                    self.stats["proxy_moves"] += 1

    def camera_pitch(self) -> Optional[float]:
        """The mounted camera's pitch against its sensor link (rad, positive down), read from the stage."""
        if self.camera_path is None:
            return None
        import omni.usd
        from pxr import Gf

        cam = omni.usd.get_world_transform_matrix(self.stage.GetPrimAtPath(self.camera_path))
        link_prim = self.stage.GetPrimAtPath(self.camera_path)
        sensor = next(s for s in self.profile.sensors if s.kind in ("rgb", "rgbd"))
        while link_prim.IsValid() and link_prim.GetName() != sensor.link:
            link_prim = link_prim.GetParent()
        link = omni.usd.get_world_transform_matrix(link_prim)
        fwd = cam.TransformDir(Gf.Vec3d(0, 0, -1))  # a USD camera looks down its -Z
        local = link.GetInverse().TransformDir(fwd)
        return rb.pitch_down((local[0], local[1], local[2]))

    def describe(self) -> dict:
        return {"camera_pitch": self.camera_pitch(), "art_root": self.art_root, "base": self.base_path, "camera": self.camera_path,
                "render_product": self.render_product, "nodes": len(self.graph.nodes) if self.graph else 0,
                "topics": dict(self.graph.topics) if self.graph else {}, **self.stats}


def _ident(s: str) -> str:
    import re
    return re.sub(r"\W", "_", s)
