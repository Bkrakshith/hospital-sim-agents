"""A short headless demo of the Isaac people, with stills and frames for a video (no ROS, no minds).

    WARDZERO_ISAAC_MODULE=wardzero_agents.isaac_host.people_demo scripts/isaac_host.sh \\
        --run-dir /tmp/run --scenario config/scenarios/waiting_area_v1.yaml --out artifacts/screens/isaac_people \\
        [--seconds 12] [--frames]

It starts the host exactly as ``app`` does (stage, physics, the people, NavMesh, play, bind), then steps the
SimService by hand and feeds it a scripted list of Intents, the same objects a persona mind would send. Used for the I2
smoke stills and the demo clip; the real runs go through ``app``.
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

log = logging.getLogger("wardzero.isaac.demo")

# (sim seconds, person, intent kind, fields): walk, sit, stand and look, as a mind would choose them
SCRIPT = [
    (1.0, "P5", "walk_to", {"target": "reception_front"}),
    (2.0, "P4", "sit_down", {"target": "A1"}),
    (3.0, "P2", "leave", {}),
    (6.0, "P3", "stand_up", {}),
    (8.0, "P3", "walk_to", {"target": "reception_front"}),
    (14.0, "P5", "look_at", {"target": "P3"}),
    (16.0, "P5", "fidget", {}),
]


SCRIPT_WARD = [
    (1.0, "P1", "walk_to", {"target": "nurse_call_point"}),
    (4.0, "P2", "approach", {"target": "P1"}),
    (4.0, "__arrival__", "arrival", {"persona": "ward_visitor"}),
    (10.0, "W1", "walk_to", {"target": "ward_doors"}),
]
SCRIPT_SMOKE = [  # three people: one walks, one moves seat and sits, one stays put
    (1.0, "P1", "walk_to", {"target": "reception_front"}),
    (1.0, "P2", "sit_down", {"target": "A1"}),
]
SCRIPT_WARD2 = [  # an escort, and a minded bed patient who gets up and leaves
    (1.0, "W1", "escort", {"target": "V1", "point": (19.0, 22.0)}),
    (2.0, "B2P", "walk_to", {"target": "ward_doors"}),
]
SCRIPT_WARD3 = [(1.0, "W2", "escort", {"target": "B2P", "point": (19.0, 13.0)})]
VIEWS = {
    "lobby": {"script": SCRIPT, "cams": {
        "top": ((-3.5, 0.3, 22.0), (-3.5, 0.31, 0.0), 15.0, 19.4),
        "wide": ((1.2, -2.9, 2.5), (-4.0, 0.5, 0.8), 11.0, 0.05),
        "seats": ((-3.2, -2.2, 2.0), (-6.6, 0.5, 0.8), 14.0, 0.05)}},
    "smoke": {"script": SCRIPT_SMOKE, "cams": {
        "wide": ((1.2, -2.9, 2.5), (-4.0, 0.5, 0.8), 11.0, 0.05)}},
    "ward2": {"script": SCRIPT_WARD2, "cams": {
        "top": ((22.0, 13.0, 22.0), (22.0, 13.01, 0.0), 9.0, 19.4),
        "wide": ((19.0, 6.5, 2.4), (19.0, 16.0, 1.0), 11.0, 0.05),
        "bed": ((22.9, 20.2, 2.4), (25.1, 19.1, 0.9), 10.0, 0.05)}},
    "ward3": {"script": SCRIPT_WARD3, "cams": {
        "top": ((22.0, 17.0, 22.0), (22.0, 17.01, 0.0), 9.0, 19.4),
        "wide": ((18.2, 22.0, 2.4), (22.0, 17.0, 0.9), 11.0, 0.05),
        "bed": ((22.9, 20.2, 2.4), (25.1, 19.1, 0.9), 10.0, 0.05)}},
    "ward": {"script": SCRIPT_WARD, "cams": {
        "top": ((19.0, 14.0, 22.0), (19.0, 14.01, 0.0), 8.0, 19.4),
        "wide": ((22.9, 6.9, 2.4), (25.1, 8.0, 0.9), 10.0, 0.05),
        "seats": ((17.0, 10.0, 2.2), (12.5, 7.3, 0.9), 12.0, 0.05)}},
}


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--view", choices=sorted(VIEWS), default="lobby")
    p.add_argument("--no-cams", action="store_true", help="no render products: measures the host's own frame cost")
    p.add_argument("--probe-nav", action="store_true", help="print NavMesh path lengths from the bedsides, then exit")
    p.add_argument("--video-cam", default="wide", help="camera of the frames for the video (the top camera is saved too)")
    p.add_argument("--frames", action="store_true", help="save a frame every third sim frame (10 fps) for a video")
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
    if demo.probe_nav:
        import math as _m

        import omni.anim.navigation.core as nav

        nm = nav.acquire_interface().get_navmesh()
        src = (19.0, 9.0, 0.0)
        for bid, bed in svc.world.site.beds.items():
            for dx, dy in ((1.0, 0), (-1.0, 0), (0, 1.0), (0, -1.0), (0.0, 0.0)):
                pt = (bed.x + dx * 1.1, bed.y + dy * 1.1, 0.0)
                path = nm.query_shortest_path(src, pt)
                pts = path.get_points() if path is not None and hasattr(path, "get_points") else path
                L = sum(_m.dist(pts[i][:2], pts[i + 1][:2]) for i in range(len(pts) - 1)) if pts else None
                end = pts[-1][:2] if pts else None
                print("NAVPROBE", bid, (dx, dy), "len", L and round(L, 1), "end", end and (round(end[0], 2), round(end[1], 2)), flush=True)
        return 0

    def camera(path, eye, target, focal=14.0, near=0.05, w=1280, h=720):
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

    view = VIEWS[demo.view]
    cams = {} if demo.no_cams else {name: camera(f"/World/DemoCams/{name}", eye, tgt, focal=f, near=near)
            for name, (eye, tgt, f, near) in view["cams"].items()}  # inside the kept area: outside it the lights are culled
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
    script = list(view["script"])
    track = []
    frames_dir = out / "frames"
    if demo.frames:
        frames_dir.mkdir(exist_ok=True)
    ms = []
    n = int(demo.seconds * fps)
    stills = {int(t * fps) for t in (0.1, 4.0, 8.0, 12.0, 16.0, demo.seconds - 0.1) if t < demo.seconds} if cams else set()
    fi = 0
    for k in range(n):
        t0 = time.perf_counter()
        t = (k + 1) * dt
        while script and script[0][0] <= t:
            _, pid, kind, kw = script.pop(0)
            if kind == "arrival":  # a person joins during the run: they get a character after play
                from ..contracts import WorldEvent

                svc.on_world_event(WorldEvent(id=f"demo-arrival-{k}", t=t, kind="arrival", **kw))
                continue
            svc.on_intent(Intent(id=f"demo-{kind}-{pid}-{k}", person=pid, t=t, kind=kind, **kw))
        app.update()
        svc.step(dt)
        ms.append((time.perf_counter() - t0) * 1000.0)
        if k % 15 == 0:
            track.append({"t": round(t, 2), "people": {s.person: [round(s.x, 2), round(s.y, 2), s.posture, s.state,
                                                                    s.intent_kind] for s in svc.world.statuses()
                                                       if s.person in svc.bodies.controllers}})
        if k in stills:
            for name in cams:
                shot(name, out / f"still_{name}_t{t:05.1f}.png")
        if demo.frames and cams and k % 3 == 0:
            shot(demo.video_cam, frames_dir / f"{demo.video_cam}_{fi:04d}.png")
            shot("top", frames_dir / f"top_{fi:04d}.png")
            fi += 1
    ms.sort()
    stats = {"frames": n, "ms_median": round(ms[len(ms) // 2], 2), "ms_p90": round(ms[int(0.9 * len(ms))], 2),
             "ms_mean": round(sum(ms) / len(ms), 2)}
    from .service import nvidia_smi_mib

    stats["vram_mib"] = nvidia_smi_mib()
    (out / "demo_track.json").write_text(json.dumps({"stats": stats, "track": track,
                                                     "choice": {p: c.id for p, c in host.people.choice.items()},
                                                     "events": {p: c.events for p, c in svc.bodies.controllers.items()}},
                                                    indent=1))
    log.info("demo done: %s", stats)
    print("DEMO_STATS", json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
