"""Site model and grid routing over the committed whole-hospital map."""
import math
import random
import time
from pathlib import Path

import numpy as np
import pytest

from wardzero_agents.site import Router, load_site

ROOT = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def site():
    return load_site(ROOT)


@pytest.fixture(scope="module")
def router(site):
    return Router(site.grid)


def free_points(site, router, n, seed):
    rng = random.Random(seed)
    xmin, xmax, ymin, ymax = site.bounds
    out = []
    while len(out) < n:
        p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if router.is_free(*p):
            out.append(p)
    return out


def test_route_never_crosses_blocked_cells(site, router):
    ent = site.resolve("entrance")
    pts = [p for p in free_points(site, router, 60, 7) if router.route(ent, p)][:12]
    assert len(pts) >= 8
    rng = random.Random(3)
    for _ in range(20):
        a, b = rng.sample(pts, 2)
        r = router.route(a, b)
        assert r is not None and r[0] == a and r[-1] == b
        for (x0, y0), (x1, y1) in zip(r, r[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
            for i in range(n + 1):
                t = i / n
                assert not site.grid.blocked(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


def test_every_room_reachable_from_entrance(site, router):
    ent = site.resolve("entrance")
    for rid, room in site.rooms.items():
        if room.use in ("plant", "stair"):
            continue
        assert router.route(ent, site.resolve(f"room:{rid}")) is not None, rid


def test_route_none_when_unreachable(site, router):
    """A free cell inside a sealed pocket (found by flood fill from the entrance) has no route."""
    g = site.grid.occ
    start = site.grid.to_cell(*site.resolve("entrance"))
    seen, todo = {start}, [start]
    while todo:
        y, x = todo.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (y + dy, x + dx)
            if 0 <= n[0] < g.shape[0] and 0 <= n[1] < g.shape[1] and n not in seen and g[n] == 0:
                seen.add(n)
                todo.append(n)
    sealed = [(iy, ix) for iy, ix in np.argwhere(g == 0) if (int(iy), int(ix)) not in seen]
    assert sealed, "map has no sealed pocket"
    pit = site.grid.to_world(*[int(v) for v in sealed[0]])
    assert router.route(site.resolve("entrance"), pit) is None


def test_long_route_is_fast(site, router):
    a = site.resolve("entrance")
    b = site.resolve("room:Theatre")
    far = site.resolve("room:B6")
    t = time.perf_counter()
    r = Router(site.grid).route(a, far)  # cold, uncached
    dt = time.perf_counter() - t
    assert r is not None and dt < 0.5, dt
    assert router.route(a, b) is not None


def test_route_cache_returns_same(router, site):
    a, b = site.resolve("entrance"), site.resolve("room:B1")
    assert router.route(a, b) == router.route(a, b)
