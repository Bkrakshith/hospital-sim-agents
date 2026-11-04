"""A headless demo of the Isaac robot (I4): the G1 at reception, three people, the robot's own ROS domain.

    WARDZERO_ISAAC_MODULE=wardzero_agents.isaac_host.robot_demo scripts/isaac_host.sh \\
        --run-dir /tmp/run --scenario <scenario> --out artifacts/screens/isaac_robot --robot-domain 41 \\
        --private-domain 42 --fire none --rooms Lobby,AE_Waiting,Main_LiftLobby [--seconds 14]

The host starts exactly as ``app`` does (stage, physics, people, props, NavMesh, play, bind, robot) and its real frame
loop (``HostLoop``, the ROS bridges, SimRos, the detection link) runs for ``--seconds``. A second bridge on the robot
domain plays the robot under test: it listens to every topic of the robot graph, speaks, sends a task action, and
looks for person ids in anything it can read. Frame cost and VRAM are measured over the run, stills are taken after it.
``demo_stats.json`` is what tests/isaac/test_isaac_robot_smoke.py asserts on.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from ..contracts import Intent
from . import app as app_mod
from . import stage as stage_mod

log = logging.getLogger("wardzero.isaac.robot_demo")

PEDESTAL = (1.05, 1.62)
CAMS = {  # name: (eye, target, focal mm)
    "threequarter": ((3.4, -0.9, 1.9), (1.05, 1.3, 1.0), 12.0),
    "top": ((1.05, 0.9, 9.0), (1.05, 0.91, 0.0), 11.0),
}
TOP_NEAR = 6.8  # the near plane cuts the ceiling away without deactivating it (S1b)


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=14.0)
    p.add_argument("--mode", choices=("smoke", "budget"), default="smoke")
    a, rest = p.parse_known_args(argv)
    return a, app_mod.parse_args(rest)


def boot(args):
    """The host's start-up up to and including the robot, and the loop pieces built the way ``IsaacHost.loop`` builds them."""
    from ..config.loader import load_scenario

    loaded = load_scenario(args.scenario)
    plan = stage_mod.cull_plan(loaded, rooms=args.rooms, full_site=args.full_site)
    host = app_mod.IsaacHost(args, loaded, plan)
    host.prepare_robot()
    host.start_app()
    host.enable_extensions()
    host.open_stage()
    host.physics()
    host.make_service()
    host.spawn_people()
    host.spawn_objects()
    host.bake_navmesh()
    host.robot_proxy()
    host.play()
    host.bind_people()
    host.add_robots()
    return host


def make_loop(host, extra_bridges=()):
    from ..ros.rosutil import RosBridge
    from ..ros.sim_service import SimRos
    from .service import HostLoop, VramSampler

    a, svc, robot = host.args, host.svc, host.robot
    private = RosBridge("wardzero_isaac_host", None, use_sim_time=False, domain_id=a.private_domain)
    rb = RosBridge("wardzero_isaac_robot_side", None, use_sim_time=False, domain_id=host.robot_domain) \
        if host.robot_domain != a.private_domain else None
    host.bridges = [b for b in (private, rb) if b is not None]
    clocks = [rb] if rb is not None and not robot.owns_clock else []
    sim_ros = SimRos(svc, private, clock_bridges=clocks, robot_ros=rb, graph_owned=robot.graph_owned)
    robot.attach_ros(svc.world, private, rb)
    for b in host.bridges + list(extra_bridges):
        b.start()
    vram = VramSampler().start()
    vram.sample()
    loop = HostLoop(svc, sim_ros, host.app, host.timeline, bridges=host.bridges + list(extra_bridges), time_scale=0.0,
                    vram=lambda: vram.last, vram_peak=lambda: vram.peak)
    return loop, vram


class RobotProbe:
    """The robot under test: a node on the robot domain whose name does not start with wardzero_."""

    def __init__(self, host, out: Path):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage
        from rosgraph_msgs.msg import Clock

        from ..ros import topics
        from ..ros.rosutil import RosBridge

        self.host, self.out = host, out
        self.rt = rt = topics.robot_topics(host.loaded.scenario.robot.ns)
        self.bridge = RosBridge("robot_under_test", None, use_sim_time=False, domain_id=host.robot_domain)
        self.counts: dict[str, int] = {}
        self.texts: dict[str, list] = {}
        self.last: dict[str, object] = {}
        self.image = None
        self.String = String
        q = qos_profile_sensor_data
        for key, mt in (("clock", Clock), ("tf", TFMessage), ("joint_states", JointState), ("camera_rgb", Image),
                        ("camera_depth", Image), ("camera_info", CameraInfo), ("bbox_3d", String), ("objects", String),
                        ("heard", String), ("task_result", String), ("task_brief", String), ("odom", None)):
            if mt is None or key not in rt:
                continue
            self.bridge.subscribe(mt, rt[key], lambda m, k=key: self._on(k, m), q)
        self.pub_say = self.bridge.publisher(rt["say"], String)
        self.pub_action = self.bridge.publisher(rt["task_action"], String)
        self.pub_joint = None

    def _on(self, key, msg):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.last[key] = msg
        if hasattr(msg, "data") and isinstance(msg.data, str):
            self.texts.setdefault(key, []).append(msg.data)
        if key == "camera_rgb":
            self.image = msg

    def say(self, text):
        m = self.String()
        m.data = text
        self.pub_say.publish(m)

    def action(self, d):
        m = self.String()
        m.data = json.dumps(d)
        self.pub_action.publish(m)

    def topics_seen(self) -> dict:
        return {name: types for name, types in self.bridge.node.get_topic_names_and_types()}

    def save_image(self, path: Path) -> bool:
        import numpy as np
        from PIL import Image

        m = self.image
        if m is None:
            return False
        ch = {"rgb8": 3, "rgba8": 4, "bgr8": 3}.get(m.encoding, 3)
        a = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(m.height, m.width, ch)[..., :3]
        Image.fromarray(a[..., ::-1] if m.encoding == "bgr8" else a).save(path)
        return True


def camera(stage, path, eye, target, focal, near=0.05, w=1280, h=720):
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(near, 300.0))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    return path


def person_ids(world) -> set:
    ids = set(world.people)
    for p in world.people.values():
        inst = getattr(p, "inst", None)
        for attr in ("person_id", "name", "id"):
            v = getattr(inst, attr, None)
            if isinstance(v, str) and len(v) > 2:
                ids.add(v)
    return ids


def run_smoke(host, demo, out: Path) -> dict:
    import omni.replicator.core as rep
    from PIL import Image
    import numpy as np

    svc, world, robot = host.svc, host.svc.world, host.robot
    probe = RobotProbe(host, out)
    loop, vram = make_loop(host, [probe.bridge])
    fps = host.args.fps
    n = int(demo.seconds * fps)
    joints = {}
    events = {int(2.0 * fps): "walk", int(3.0 * fps): "talk", int(5.0 * fps): "action", int(6.0 * fps): "say",
              int(8.0 * fps): "offer", int(11.0 * fps): "withdraw"}
    arm_track = []
    for k in range(n):
        ev = events.get(k)
        if ev == "walk":
            svc.on_intent(Intent(id="demo-walk", person="P1", t=svc.t, kind="walk_to", point=(1.05, -0.3)))
        elif ev == "talk":
            svc.on_intent(Intent(id="demo-talk", person="P2", t=svc.t, kind="talk", text="Excuse me, where is the pharmacy?"))
        elif ev == "action":
            probe.action({"id": "demo-call", "action": "press_call"})
        elif ev == "say":
            probe.say("The pharmacy is down the clinic corridor.")
        elif ev == "offer":
            world._offers["P1"] = {"giver": "g1", "item": "none", "place": (1.0, 0.9, 1.1), "t": svc.t, "intent": "demo"}
        elif ev == "withdraw":
            world._offers.pop("P1", None)
        loop.frame()
        if robot.arm is not None and k % 15 == 0:
            arm_track.append({"t": round(svc.t, 2), "q": {j: round(float(v), 3) for j, v in
                                                          zip(robot.arm.joints, robot.art.get_dof_positions(
                                                              dof_indices=robot._arm_idx).numpy().flatten())}})
    stats = loop.stats()
    stats["vram_peak_mib"] = vram.peak
    stats["frames_measured"] = n
    # stills after the measured run
    saved = {}
    if probe.save_image(out / "robot_camera.png"):
        saved["robot_camera"] = "robot_camera.png"
    annots = {}
    for name, (eye, target, focal) in CAMS.items():
        cp = camera(host.stage, f"/World/DemoCams/{name}", eye, target, focal, near=TOP_NEAR if name == "top" else 0.05)
        an = rep.AnnotatorRegistry.get_annotator("rgb")
        an.attach([rep.create.render_product(cp, (1280, 720))])
        annots[name] = an
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=3, pause_timeline=False)
        loop.frame()
    rep.orchestrator.step(rt_subframes=3, pause_timeline=False)
    for name, an in annots.items():
        img = np.asarray(an.get_data())
        if img.size:
            Image.fromarray(img[..., :3]).save(out / f"still_{name}.png")
            saved[name] = f"still_{name}.png"
    diag = {}
    try:
        ab = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
        ab.attach([robot.render_product])
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=3, pause_timeline=False)
        d = ab.get_data()
        diag = {"n": len(d["data"]), "labels": d["info"].get("idToLabels"), "rows": [str(r) for r in d["data"][:3]]}
    except Exception as exc:
        diag = {"error": repr(exc)}
    ids = person_ids(world)
    leaked = sorted({pid for texts in probe.texts.values() for t in texts for pid in ids if f'"{pid}"' in t or pid in t.split()})
    seen = probe.topics_seen()
    from ..ros import topics
    forbidden = sorted(t for t in seen if any(t.startswith(p) for p in topics.FORBIDDEN_FOR_ROBOT) or t.startswith("/wz/"))
    result = {"stats": stats, "counts": probe.counts, "topics": {k: v for k, v in seen.items()},
              "forbidden_topics": forbidden, "leaked_ids": leaked, "stills": saved, "robot": robot.describe(),
              "arm_track": arm_track, "people": {pid: [round(p.x, 2), round(p.y, 2), p.posture] for pid, p in world.people.items()},
              "robot_pose": [round(v, 3) for v in world.robot.pose],
              "bbox_last": probe.texts.get("bbox_3d", [None])[-1],
              "bbox_with_people": sum(1 for t in probe.texts.get("bbox_3d", []) if json.loads(t)["detections"]),
              "task_results": [json.loads(t) for t in probe.texts.get("task_result", [])],
              "heard": [json.loads(t) for t in probe.texts.get("heard", [])],
              "annotator_diag": diag, "dropped_boxes": robot.link.adapter.dropped if robot.link else None,
              "raw": {"messages": robot.link.raw_messages, "boxes": robot.link.raw_boxes,
                      "labels": robot.link.adapter.labels} if robot.link else None,
              "vram_mib": vram.last}
    probe.bridge.close()
    return result


def run_budget(host, demo, out: Path) -> dict:
    """The S1b regression probe (qR1_probe.py --edits late) on the host's own start-up: the G1 added after play, its arm
    driven every frame, people walking. A regression of the Behavior System tracking the robot's links shows as >150 ms."""
    import numpy as np

    svc, world, robot = host.svc, host.svc.world, host.robot
    loop, vram = make_loop(host)
    fps = host.args.fps
    for i, pid in enumerate(sorted(world.people)):  # everyone walks to a spot in front of the robot
        svc.on_intent(Intent(id=f"b-{pid}", person=pid, t=0.0, kind="walk_to", point=(-0.5 + 0.8 * i, 0.0)))
    idx = robot._arm_idx
    names = robot.arm.joints
    pitch = names.index("right_shoulder_pitch_joint")
    n = int(demo.seconds * fps)
    for k in range(n):
        tgt = np.zeros((1, len(names)))
        tgt[0, pitch] = -1.2
        tgt[0, names.index("right_elbow_joint")] = 0.3 + 0.4 * math.sin(k / 10)
        robot.art.set_dof_position_targets(tgt, dof_indices=idx)
        loop.frame()
    stats = loop.stats()
    stats["vram_peak_mib"] = vram.peak
    q = robot.art.get_dof_positions(dof_indices=idx).numpy().flatten()
    stats["arm_q"] = {j: round(float(v), 3) for j, v in zip(names, q)}
    stats["people"] = len(world.people)
    return {"stats": stats}


def main(argv=None) -> int:
    demo, args = parse(argv)
    for k in app_mod.KEY_ENV:
        os.environ.pop(k, None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = Path(demo.out)
    out.mkdir(parents=True, exist_ok=True)
    code, host = 1, None
    try:
        host = boot(args)
        result = (run_smoke if demo.mode == "smoke" else run_budget)(host, demo, out)
        (out / "demo_stats.json").write_text(json.dumps(result, indent=1, default=str))
        print("DEMO_STATS", json.dumps(result["stats"]), flush=True)
        code = 0
    except Exception:
        log.exception("the demo failed")
    finally:
        if host is not None:
            host.close(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
