"""Bake the NavMesh the way the host does and check which bedsides and nurse bases a person can walk to.

    WARDZERO_ISAAC_MODULE=wardzero_agents.isaac_host.nav_probe scripts/isaac_host.sh --run-dir /tmp/run \\
        --scenario config/scenarios/waiting_area_v1.yaml --headless --wait-for none --rooms <ward and A&E rooms>

For every kept bed (a point 0.95 m beside each side) and every kept nurse base, a path is asked from the centre of the
nearest corridor room. Points the site's own grid blocks (furniture) are not targets. Prints ``NAVPROBE <point> OK|NO``,
a ``NAVFREED`` line per NavMesh exclude volume that the stage fixes switched off with the rooms it covers, and a
``NAVSUM`` line; the exit code is 1 when a target could not be reached. The Isaac test
``tests/isaac/test_isaac_navmesh_smoke.py`` runs it on the ward and the A&E bays; ``test_isaac_bedsides_smoke.py`` walks real characters.
"""
from __future__ import annotations

import math
import sys

from . import app as app_mod
from . import stage as stage_mod

REACH_M = 0.4  # a path that ends this close counts as arriving
FREE_USES = ("bedroom", "nurse_base", "ae_treatment")


def corridor_start(site, x: float, y: float) -> tuple[float, float, float]:
    cs = [r.centre for r in site.rooms.values() if r.use == "corridor"]
    cx, cy = min(cs, key=lambda c: math.hypot(c[0] - x, c[1] - y))
    return (cx, cy, 0.0)


def targets(site, kept) -> list[tuple[str, tuple[float, float]]]:
    """(label, point) of every bedside and nurse base in the kept rooms that the site grid leaves free."""
    out = []
    for bid, bed in site.beds.items():
        if bed.room in kept:
            c, s = math.cos(bed.yaw), math.sin(bed.yaw)
            for side in (1, -1):
                pt = (bed.x - s * 0.95 * side, bed.y + c * 0.95 * side)
                if not site.grid.blocked(*pt):
                    out.append((f"{bid}{'+' if side > 0 else '-'}", pt))
    for rid, room in site.rooms.items():
        if room.use == "nurse_base" and rid in kept and not site.grid.blocked(*room.centre):
            out.append((rid, room.centre))
    return out


def main(argv=None) -> int:
    args = app_mod.parse_args(sys.argv[1:] if argv is None else argv)
    from ..config.loader import load_scenario
    from . import fixes

    loaded = load_scenario(args.scenario)
    plan = stage_mod.cull_plan(loaded, rooms=args.rooms, full_site=args.full_site)
    host = app_mod.IsaacHost(args, loaded, plan)
    code = 1
    try:
        host.start_app()
        host.enable_extensions()
        site = loaded.site
        from pxr import Usd

        volumes = fixes.read_nav_volumes(Usd.Stage.Open(str(args.stage)))
        for path in fixes.nav_volumes_off(volumes, site, plan):
            (x0, y0), (x1, y1) = volumes[path][1]
            covered = [f"{rid}({r.use})" for rid, r in site.rooms.items()
                       if min(x1, max(p[0] for p in r.poly)) - max(x0, min(p[0] for p in r.poly)) > 0.3
                       and min(y1, max(p[1] for p in r.poly)) - max(y0, min(p[1] for p in r.poly)) > 0.3]
            print("NAVFREED", path.rsplit("/", 1)[-1], ", ".join(covered), flush=True)
        host.open_stage()
        host.physics()
        host.bake_navmesh()
        import omni.anim.navigation.core as nav

        navmesh = nav.acquire_interface().get_navmesh()
        kept = set(plan.rooms) if not plan.full else set(site.rooms)
        results: dict[str, bool] = {}
        for label, pt in targets(site, kept):
            start = corridor_start(site, *pt)
            path = navmesh.query_shortest_path(start, (pt[0], pt[1], 0.0))
            pts = path.get_points() if path is not None and hasattr(path, "get_points") else path
            ok = bool(pts) and math.hypot(pts[-1][0] - pt[0], pts[-1][1] - pt[1]) < REACH_M
            print("NAVPROBE", label, "OK" if ok else "NO", "from", (round(start[0], 1), round(start[1], 1)), flush=True)
            results[label] = ok
        failed = [k for k, v in results.items() if not v]
        print("NAVSUM ok", len(results) - len(failed), "/", len(results), "fail:", ",".join(failed), flush=True)
        code = 1 if failed or not results else 0
    finally:
        host.close(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
