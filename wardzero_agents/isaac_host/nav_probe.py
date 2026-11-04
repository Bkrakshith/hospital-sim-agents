"""Bake the NavMesh the way the host does and report which ward bedsides and nurse bases a person can walk to.

    WARDZERO_ISAAC_MODULE=wardzero_agents.isaac_host.nav_probe scripts/isaac_host.sh --run-dir /tmp/run \\
        --scenario config/scenarios/waiting_area_v1.yaml --headless --wait-for none --rooms <ward rooms>

Prints ``NAVPROBE <point> OK|NO`` lines and a ``NAVSUM`` line: paths are asked from the ward corridor (19, 9) to a point
0.95 m beside each side of every ward bed (B1..B6) and to the centre of each ward nurse base. A dev tool for the stage fixes in
``fixes.py``; the Isaac smoke test ``tests/isaac/test_isaac_bedsides_smoke.py`` walks real characters instead.
"""
from __future__ import annotations

import math
import sys

from . import app as app_mod
from . import stage as stage_mod

START = (19.0, 9.0, 0.0)  # the ward corridor, by the entrance
REACH_M = 0.4  # a path that ends this close counts as arriving


def main(argv=None) -> int:
    args = app_mod.parse_args(sys.argv[1:] if argv is None else argv)
    from ..config.loader import load_scenario

    loaded = load_scenario(args.scenario)
    host = app_mod.IsaacHost(args, loaded, stage_mod.cull_plan(loaded, rooms=args.rooms, full_site=args.full_site))
    host.start_app()
    host.enable_extensions()
    host.open_stage()
    host.physics()
    host.bake_navmesh()
    import omni.anim.navigation.core as nav

    navmesh = nav.acquire_interface().get_navmesh()
    site = loaded.site
    results: dict[str, bool] = {}

    def probe(label: str, pt: tuple[float, float]) -> None:
        path = navmesh.query_shortest_path(START, (pt[0], pt[1], 0.0))
        pts = path.get_points() if path is not None and hasattr(path, "get_points") else path
        ok = bool(pts) and math.hypot(pts[-1][0] - pt[0], pts[-1][1] - pt[1]) < REACH_M
        end = (round(pts[-1][0], 2), round(pts[-1][1], 2)) if pts else None
        print("NAVPROBE", label, "OK" if ok else "NO", "end", end, flush=True)
        results[label] = ok

    for bid, bed in site.beds.items():
        if bid.startswith("B"):
            c, s = math.cos(bed.yaw), math.sin(bed.yaw)
            for side in (1, -1):
                probe(f"{bid}{'+' if side > 0 else '-'}", (bed.x - s * 0.95 * side, bed.y + c * 0.95 * side))
    for rid, room in site.rooms.items():
        if room.use == "nurse_base" and rid.startswith("Ward"):
            probe(rid, room.centre)
    print("NAVSUM ok", sum(results.values()), "/", len(results), "fail:", ",".join(k for k, v in results.items() if not v),
          flush=True)
    host.close(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
