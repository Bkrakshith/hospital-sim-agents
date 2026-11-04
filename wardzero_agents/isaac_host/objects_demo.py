"""A short headless demo of the world's objects in Isaac: a tray handed to a patient, and a ward fire that grows and goes out.

    WARDZERO_ISAAC_MODULE=wardzero_agents.isaac_host.objects_demo scripts/isaac_host.sh \\
        --run-dir /tmp/run --scenario <scenario with static patients> --out artifacts/screens/isaac_objects \\
        --view tray|fire [--seconds 10] [--frames] [--no-cams]

The host starts exactly as ``app`` does (stage, physics, people, props, NavMesh, play, bind); the SimService is then
stepped by hand and the world's objects are driven the way the sim does it: a nurse carries a tray to a bed and hands it
over with ``world.hand_over``; a fire is started with ``world.objects.start_fire`` and put out with ``Fire.reduce``.
Stills and frames go to ``--out``; ``demo_stats.json`` has the frame cost and the objects' last states.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from ..contracts import Intent
from . import app as app_mod
from . import stage as stage_mod

log = logging.getLogger("wardzero.isaac.objects_demo")

# camera: (eye, target, focal length mm)
VIEWS = {
    "tray": {"cams": {"bed": ((26.5, 14.35, 2.3), (24.5, 16.4, 1.0), 12.0),
                      "top": ((24.0, 16.0, 7.5), (24.0, 16.01, 0.0), 14.0)}},
    "fire": {"cams": {"corridor": ((19.0, 6.5, 2.2), (19.0, 14.5, 0.9), 12.0),
                      "top": ((19.0, 14.0, 14.0), (19.0, 14.01, 0.0), 14.0)}},
    "bells": {"cams": {"bed": ((22.6, 19.0, 2.2), (25.1, 19.1, 1.4), 11.0),
                       "top": ((24.0, 19.0, 7.5), (24.0, 19.01, 0.0), 14.0)}},
}
FIRE_AT = (19.0, 14.5)
BED_FOR_TRAY = "B3"


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--view", choices=sorted(VIEWS), default="tray")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--no-cams", action="store_true", help="no render products: measures the host's own frame cost")
    p.add_argument("--frames", action="store_true", help="save a frame every third sim frame (10 fps) for a video")
    p.add_argument("--video-cam", default=None)
    a, rest = p.parse_known_args(argv)
    return a, app_mod.parse_args(rest)


def main(argv=None) -> int:
    demo, args = parse(argv)
    for k in app_mod.KEY_ENV:
        os.environ.pop(k, None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from ..config.loader import load_scenario

    loaded = load_scenario(args.scenario)
    plan = stage_mod.cull_plan(loaded, rooms=args.rooms, full_site=args.full_site)
    host = app_mod.IsaacHost(args, loaded, plan)
    out = Path(demo.out)
    out.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
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
        code = run(host, demo, out)
    except Exception:
        log.exception("the demo failed")
    finally:
        host.close(code)
    return code


def run(host, demo, out: Path) -> int:
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    svc, stage, app = host.svc, host.stage, host.app
    world = svc.world
    view = VIEWS[demo.view]

    def camera(path, eye, target, focal, w=1280, h=720):
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 300.0))
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)
        return path

    cams = {} if demo.no_cams else {n: camera(f"/World/DemoCams/{n}", *v) for n, v in view["cams"].items()}
    annots = {}
    for name, cp in cams.items():
        an = rep.AnnotatorRegistry.get_annotator("rgb")
        an.attach([rep.create.render_product(cp, (1280, 720))])
        annots[name] = an
    from PIL import Image
    import numpy as np

    def shot(name, path):
        rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
        img = np.asarray(annots[name].get_data())
        if img.size:
            Image.fromarray(img[..., :3]).save(path)

    fps = host.args.fps
    dt = 1.0 / fps
    n = int(demo.seconds * fps)
    video_cam = demo.video_cam or next(iter(view["cams"]), None)
    frames_dir = out / "frames"
    if demo.frames:
        frames_dir.mkdir(exist_ok=True)
    events = {"tray": tray_events, "fire": fire_events, "bells": bell_events}[demo.view](world, svc)
    log_rows: list[dict] = []
    stills = {int(t * fps) for t in (0.2, 3.0, 5.0, 7.0, demo.seconds - 0.2) if t < demo.seconds} if cams else set()
    ms = []
    fi = 0
    for k in range(n):
        t0 = time.perf_counter()
        t = (k + 1) * dt
        events(t, dt)
        app.update()
        svc.step(dt)
        ms.append((time.perf_counter() - t0) * 1000.0)
        if k % 15 == 0:
            log_rows.append({"t": round(t, 2), "objects": [o.model_dump(exclude={"schema_"}) for o in world.objects.snapshot()]})
        if k in stills:
            for name in cams:
                shot(name, out / f"still_{name}_t{t:05.1f}.png")
        if demo.frames and cams and k % 3 == 0:
            shot(video_cam, frames_dir / f"{video_cam}_{fi:04d}.png")
            if "top" in cams:
                shot("top", frames_dir / f"top_{fi:04d}.png")
            fi += 1
    ms.sort()
    from .service import nvidia_smi_mib

    stats = {"frames": n, "ms_median": round(ms[len(ms) // 2], 2), "ms_p90": round(ms[int(0.9 * len(ms))], 2),
             "ms_mean": round(sum(ms) / len(ms), 2), "vram_mib": nvidia_smi_mib(), "fire_mode": host.args.fire,
             "flow_ready": getattr(getattr(host.objects, "ustage", None), "_flow_ready", None)}
    (out / "demo_stats.json").write_text(json.dumps({"stats": stats, "rows": log_rows}, indent=1))
    log.info("demo done: %s", stats)
    print("DEMO_STATS", json.dumps(stats), flush=True)
    return 0


def tray_events(world, svc):
    """W1 carries the tray from the door of B3 to the bedside and puts it down within the patient's reach."""
    bed = world.site.beds[BED_FOR_TRAY]
    patient = next((q for q in world.people.values() if q.bed == BED_FOR_TRAY and q.posture == "lying"), None)
    if patient is not None:
        patient.sitting_up = True  # propped up: the over-bed zone applies (the body drawn in the bed stays posed lying)
    state = {"walked": False, "given": False, "bell": False}
    place = (bed.x - 0.45, bed.y, 1.0)  # on the bed beside the patient, within 0.6 m of their torso

    def step(t, dt):
        if not state["bell"] and t >= 0.5:
            world.objects.ring(BED_FOR_TRAY, t)
            state["bell"] = True
        if not state["walked"] and t >= 2.0:
            svc.on_intent(Intent(id="demo-walk", person="W1", t=t, kind="walk_to", point=(bed.x - 0.95, bed.y)))
            state["walked"] = True
        w = world.people.get("W1")
        if state["walked"] and not state["given"] and w is not None and w.state == "done":
            ok, why = world.hand_over("W1", patient.inst.person_id, "tray_1", place, t)
            log.info("hand_over: %s %s at t=%.1f (W1 at %.2f, %.2f; place %s)", ok, why, t, w.x, w.y, place)
            if ok:
                world.objects.bells[BED_FOR_TRAY].active = False  # answered
                state["given"] = True

    return step


def fire_events(world, svc):
    """A fire starts in the corridor, grows, and is put out."""
    state = {"fire": None}

    def step(t, dt):
        if state["fire"] is None and t >= 0.3:
            state["fire"] = world.objects.start_fire(*FIRE_AT, size=0.1, grow_per_s=0.14)
        f = state["fire"]
        if f is not None and t >= 5.5 and not f.out:
            f.reduce(0.30 * dt)  # sprayed: the size falls to 0 over about 4 s

    return step


def bell_events(world, svc):
    """Call lights come on in B2 and B3, one goes out."""
    def step(t, dt):
        if abs(t - 1.0) < dt / 2:
            world.objects.ring("B2", t)
        if abs(t - 2.5) < dt / 2:
            world.objects.ring("B3", t)
        if abs(t - 6.0) < dt / 2:
            world.objects.bells["B2"].active = False

    return step


if __name__ == "__main__":
    sys.exit(main())
