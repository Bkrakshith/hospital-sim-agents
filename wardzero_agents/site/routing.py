"""8-connected A* over the occupancy grid with clearance inflation and string-pulling."""
from __future__ import annotations

import copy
import heapq
import math
from collections import OrderedDict
from typing import Optional

import numpy as np

from .model import OccGrid

CLEARANCE = 0.3
NEAR_PENALTY = 3.0  # extra cost per step inside the clearance band
TIGHT_M = 0.35  # a robot's narrower berth: its radius (0.3 m) and a little; the robot_near_furniture_s distance
DETOUR_RATIO = 1.25  # the hard band's route is kept unless it is this much longer than the tight band's ...
DETOUR_M = 2.0  # ... plus this
_SQ2 = math.sqrt(2.0)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


class Router:
    """Raw-blocked cells are impassable. Cells within `clearance` of a blocked cell are passable but cost
    more, so routes keep clear of walls where the map allows and still squeeze through tight doorways.

    hard (metres, 0 = off): a stricter band for a robot's body. A route first keeps out of every cell within ``hard`` of
    a blocked cell (except near its two ends, so it can start or stop beside furniture). Where that route does not
    exist, or is a long detour (more than DETOUR_RATIO x the other + DETOUR_M: a 1 m link the wide berth cannot pass,
    such as the corridor-to-ambulance-bay link), the route keeps out of the ``tight`` band instead (TIGHT_M: the body's
    radius and a little, so the body still clears every wall and piece of furniture). Only where neither exists (a
    doorway narrower than the tight band) does it fall back to the soft band."""

    def __init__(self, grid: OccGrid, clearance: float = CLEARANCE, cache_size: int = 4096, hard: float = 0.0,
                 tight: float = TIGHT_M):
        self.grid = grid
        self.clearance = float(clearance)  # kept: block() and with_rects_blocked() rebuild the band with it
        self.hard = float(hard)
        self.tight = float(tight) if 0.0 < float(tight) < self.hard else 0.0  # a middle tier only below the hard one
        occ = grid.occ.astype(bool)
        near = self._inflate(occ, clearance)
        self._raw = occ.tolist()
        self._near = near.tolist()
        self._hard = self._inflate(occ, self.hard).tolist() if self.hard > 0 else None
        self._tight = self._inflate(occ, self.tight).tolist() if self._hard is not None and self.tight > 0 else None
        self._b = self._raw
        self._cache: "OrderedDict[tuple, Optional[list]]" = OrderedDict()
        self._size = cache_size
        self.version = 0  # goes up with every block(): a router built from this one knows to rebuild
        self.hazards: tuple = ()  # (x, y, r) discs no route may enter (a fire), unlike the avoid discs, which give way

    def _inflate(self, occ: np.ndarray, clearance: float) -> np.ndarray:
        r = clearance / self.grid.res
        k = int(math.ceil(r))
        inf = occ.copy()
        h, w = occ.shape
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if (dy or dx) and dy * dy + dx * dx <= r * r + 1e-9:
                    ys, xs = slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx))
                    yd, xd = slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx))
                    inf[yd, xd] |= occ[ys, xs]
        return inf

    def block(self, x: float, y: float, radius: float) -> None:
        """Make every cell within radius metres of (x, y) impassable (a blocked doorway or corridor)."""
        g = self.grid
        iy, ix = g.to_cell(x, y)
        n = int(math.ceil(radius / g.res))
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                if dy * dy + dx * dx <= (radius / g.res) ** 2 and g.in_grid(iy + dy, ix + dx):
                    self._raw[iy + dy][ix + dx] = True
        self.rebuild_bands()
        self.version += 1

    def rebuild_bands(self) -> None:
        """Rebuild the soft band (and the hard one) from the raw cells, after cells were blocked."""
        raw = np.array(self._raw, dtype=bool)
        self._b = self._raw
        self._near = self._inflate(raw, self.clearance).tolist()
        self._hard = self._inflate(raw, self.hard).tolist() if getattr(self, "hard", 0.0) > 0 else None
        tight = getattr(self, "tight", 0.0)
        self._tight = self._inflate(raw, tight).tolist() if self._hard is not None and tight > 0 else None
        self._hard_cells = None
        self._tight_cells = None
        self._cache.clear()

    def with_rects_blocked(self, rects) -> "Router":
        """A new router with this one's blocked cells (anything block_segments or block put in, door leaves
        included) plus every cell whose centre is inside one of the rectangles (x0, y0, x1, y1): forbidden
        areas only the robot keeps out of. It keeps this router's clearance band (0.6 m for the walking G1)."""
        g = self.grid
        raw = np.array(self._raw, dtype=bool)
        for x0, y0, x1, y1 in rects:
            iy0, ix0 = g.to_cell(x0, y0)
            iy1, ix1 = g.to_cell(x1, y1)
            for iy in range(max(0, iy0), min(raw.shape[0], iy1 + 1)):
                for ix in range(max(0, ix0), min(raw.shape[1], ix1 + 1)):
                    cx, cy = g.to_world(iy, ix)
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        raw[iy, ix] = True
        out = copy.copy(self)  # keeps any other settings; the cell tables and the cache are its own
        out.version = 0
        out._raw = raw.tolist()
        out._cache = OrderedDict()
        out.rebuild_bands()
        return out

    def set_hazards(self, discs) -> None:
        """(x, y, r) discs no route enters while they last (a fire): a route goes round them or there is none."""
        self.hazards = tuple(discs)

    def _hazard_cells(self, a) -> frozenset:
        """The cells of the hazard discs, less any disc a (the route's start) is already in: one can walk out."""
        hz = [d for d in getattr(self, "hazards", ()) if math.hypot(a[0] - d[0], a[1] - d[1]) > d[2]]
        return self._discs(hz, ()) if hz else frozenset()

    def is_free(self, x: float, y: float) -> bool:
        """Free of walls and of the clearance band."""
        iy, ix = self.grid.to_cell(x, y)
        return self.grid.in_grid(iy, ix) and not self._near[iy][ix]

    def is_clear(self, x: float, y: float) -> bool:
        """Free of walls and of the hard band (the soft band when there is no hard one): room for a robot's body."""
        iy, ix = self.grid.to_cell(x, y)
        if not self.grid.in_grid(iy, ix) or self._raw[iy][ix]:
            return False
        band = self._hard if getattr(self, "_hard", None) is not None else self._near
        return not band[iy][ix]

    def _snap(self, c):
        """The cell itself, else the nearest raw-free cell within 1 m, else None."""
        if self.grid.in_grid(*c) and not self._raw[c[0]][c[1]]:
            return c
        n = int(1.0 / self.grid.res)
        best, bd = None, 1e9
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                y, x = c[0] + dy, c[1] + dx
                if self.grid.in_grid(y, x) and not self._raw[y][x] and dy * dy + dx * dx < bd:
                    best, bd = (y, x), dy * dy + dx * dx
        return best

    def _los(self, p, q, allow, extra=frozenset()) -> bool:
        """Walk every cell the segment touches (supercover) and refuse raw-blocked or out-of-band cells."""
        g = self.grid
        raw, near = self._raw, self._near
        fx, fy = (p[0] - g.origin[0]) / g.res, (p[1] - g.origin[1]) / g.res
        tx, ty = (q[0] - g.origin[0]) / g.res, (q[1] - g.origin[1]) / g.res
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        ex, ey = int(math.floor(tx)), int(math.floor(ty))
        dx, dy = tx - fx, ty - fy
        sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        inf = float("inf")
        tdx = abs(1.0 / dx) if dx else inf
        tdy = abs(1.0 / dy) if dy else inf
        mx = ((ix + (1 if dx > 0 else 0)) - fx) / dx if dx else inf
        my = ((iy + (1 if dy > 0 else 0)) - fy) / dy if dy else inf
        for _ in range(abs(ex - ix) + abs(ey - iy) + 2):
            if not g.in_grid(iy, ix) or raw[iy][ix] or (iy, ix) in extra:
                return False
            if near[iy][ix] and (iy, ix) not in allow:
                return False
            if ix == ex and iy == ey:
                return True
            if abs(mx - my) < 1e-12:  # exact corner: both neighbours must be passable too
                for cy, cx in ((iy, ix + sx), (iy + sy, ix)):
                    if not g.in_grid(cy, cx) or raw[cy][cx]:
                        return False
                ix += sx
                iy += sy
                mx += tdx
                my += tdy
            elif mx < my:
                ix += sx
                mx += tdx
            else:
                iy += sy
                my += tdy
        return True

    def _astar(self, s, g, extra=frozenset()):
        b, near = self._raw, self._near
        h, w = len(b), len(b[0])
        gy, gx = g

        def hf(y, x):
            dy, dx = abs(y - gy), abs(x - gx)
            return (dy + dx) + (_SQ2 - 2.0) * min(dy, dx)

        cost = {s: 0.0}
        prev = {}
        heap = [(hf(*s), s)]
        done = set()
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == g:
                path = [cur]
                while cur in prev:
                    cur = prev[cur]
                    path.append(cur)
                return path[::-1]
            if cur in done:
                continue
            done.add(cur)
            cy, cx = cur
            cc = cost[cur]
            for dy, dx, step in _NB:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < h and 0 <= nx < w) or b[ny][nx] or (ny, nx) in extra:
                    continue
                if dy and dx and (b[cy + dy][cx] or b[cy][cx + dx]):
                    continue  # no corner cutting
                n = (ny, nx)
                nc = cc + step + (NEAR_PENALTY * step if near[ny][nx] else 0.0)
                if nc < cost.get(n, 1e18):
                    cost[n] = nc
                    prev[n] = cur
                    heapq.heappush(heap, (nc + hf(ny, nx), n))
        return None

    def _discs(self, avoid, keep) -> frozenset:
        """Cells within r of each (x, y, r) in avoid, less the cells in keep and their neighbours (the route's ends)."""
        g = self.grid
        keep = {(y + dy, x + dx) for y, x in keep for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        out: set = set()
        for x, y, r in avoid:
            iy, ix = g.to_cell(x, y)
            n = int(math.ceil(r / g.res))
            disc = set()
            for dy in range(-n, n + 1):
                for dx in range(-n, n + 1):
                    cx, cy = g.to_world(iy + dy, ix + dx)
                    if math.hypot(cx - x, cy - y) <= r:
                        disc.add((iy + dy, ix + dx))
            out |= disc - keep
        return frozenset(out)

    def route(self, a: tuple, b: tuple, avoid=()) -> Optional[list[tuple[float, float]]]:
        """Waypoints from a to b. An endpoint in a wall is moved to the nearest free cell centre (within 1 m);
        None if there is none or no route. Cached on the (cell_a, cell_b) pair.

        avoid: (x, y, radius) discs to keep out of, for this route only (people standing in the way). The cells at
        either end stay open; with no route around the discs, the plain route is returned. Not cached."""
        ca, cb = self._snap(self.grid.to_cell(*a)), self._snap(self.grid.to_cell(*b))
        if ca is None or cb is None:
            return None

        def endpoint(p, c):
            return tuple(p) if self.grid.to_cell(*p) == c else self.grid.to_world(*c)

        ea, eb = endpoint(a, ca), endpoint(b, cb)
        return self._ends_clear(self._route(ca, cb, ea, eb, a, avoid), ca, cb)

    def _seg_clear(self, p, q) -> bool:
        """No raw-blocked cell on the straight segment p-q (sampled at a quarter cell)."""
        g = self.grid
        n = max(1, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / (g.res / 4))))
        for k in range(n + 1):
            iy, ix = g.to_cell(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n)
            if not g.in_grid(iy, ix) or self._raw[iy][ix]:
                return False
        return True

    def _ends_clear(self, pts, ca, cb):
        """The inner waypoints are planned from cell centres; the route's own ends are the exact points. Where the leg
        from the exact start (or to the exact goal) would cut the corner of a blocked cell, it goes by its cell's centre
        first (or last), so a body beside furniture is never sent into it."""
        if pts is None or len(pts) < 2:
            return pts
        pts = list(pts)
        c0, c1 = self.grid.to_world(*ca), self.grid.to_world(*cb)
        if not self._seg_clear(pts[0], pts[1]) and tuple(pts[0]) != c0 and tuple(pts[1]) != c0:
            pts.insert(1, c0)
        if not self._seg_clear(pts[-2], pts[-1]) and tuple(pts[-1]) != c1 and tuple(pts[-2]) != c1:
            pts.insert(len(pts) - 1, c1)
        return pts

    def _route(self, ca, cb, ea, eb, a, avoid):
        hz = self._hazard_cells(a)
        if cb in hz:
            return None  # the goal is in a fire
        if getattr(self, "_hard", None) is not None:
            key = ("band", ca, cb)
            if not avoid and not hz and key in self._cache:  # the band routes are cached like the plain ones
                self._cache.move_to_end(key)
                mid = self._cache[key]
            else:
                mid = self._banded(ca, cb, avoid, hz)
                if not avoid and not hz:
                    self._cache[key] = mid
                    if len(self._cache) > self._size:
                        self._cache.popitem(last=False)
            if mid is not None:
                return [ea] + list(mid) + [eb]
        extra = self._discs(avoid, (ca, cb)) if avoid else frozenset()
        if extra:
            mid = self._plan(ca, cb, extra | hz)
            if mid is not None:
                return [ea] + list(mid) + [eb]
        if hz:  # not cached: hazards come and go
            mid = self._plan(ca, cb, hz)
            return None if mid is None else [ea] + list(mid) + [eb]
        key = (ca, cb)
        if key in self._cache:
            self._cache.move_to_end(key)
            mid = self._cache[key]
        else:
            mid = self._plan(ca, cb)
            self._cache[key] = mid
            if len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return None if mid is None else [ea] + list(mid) + [eb]

    def _cells_len(self, ca, cb, mid) -> float:
        pts = [self.grid.to_world(*ca)] + list(mid) + [self.grid.to_world(*cb)]
        return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))

    def _banded(self, ca, cb, avoid, hz) -> Optional[list]:
        """The inner waypoints of the hard band's route, or the tight band's where the hard one has none or is a long
        detour; None when neither band has a route."""
        strict = self._strict(ca, cb, avoid, hz)
        if getattr(self, "_tight", None) is None:
            return strict
        straight = self._cells_len(ca, cb, [])
        if strict is not None and self._cells_len(ca, cb, strict) <= DETOUR_RATIO * straight + DETOUR_M:
            return strict  # no route can be much shorter: the wide berth it is
        tight = self._strict(ca, cb, avoid, hz, which="tight")
        if strict is None:
            return tight
        if tight is not None and self._cells_len(ca, cb, strict) > DETOUR_RATIO * self._cells_len(ca, cb, tight) + DETOUR_M:
            return tight
        return strict

    def _strict(self, ca, cb, avoid, hz=frozenset(), which: str = "hard") -> Optional[list]:
        """The inner waypoints of a route that keeps out of the hard band (or the tight one) and of the avoid discs if
        it can, or None. Band cells within the band's width + 0.5 m of either end stay open, so a route can leave or
        reach a spot by furniture."""
        g = self.grid
        hard = self._hard if which == "hard" else self._tight
        width = self.hard if which == "hard" else self.tight
        attr = "_hard_cells" if which == "hard" else "_tight_cells"
        h, w = len(hard), len(hard[0])
        open_r = int(math.ceil((width + 0.5) / g.res))
        ends = {(c[0] + dy, c[1] + dx) for c in (ca, cb) for dy in range(-open_r, open_r + 1)
                for dx in range(-open_r, open_r + 1) if dy * dy + dx * dx <= open_r * open_r}
        base = getattr(self, attr, None)
        if base is None or base[0] != self.version:  # the band's cells, once per change of the map
            base = (self.version, frozenset((y, x) for y in range(h) for x in range(w) if hard[y][x] and not self._raw[y][x]))
            setattr(self, attr, base)
        band = (base[1] - ends) | hz
        discs = self._discs(avoid, (ca, cb)) if avoid else frozenset()
        if discs:
            mid = self._plan(ca, cb, band | discs)
            if mid is not None:
                return mid
        return self._plan(ca, cb, band)

    def _plan(self, ca, cb, extra=frozenset()) -> Optional[list]:
        """A* over cells, then string-pulled: the inner waypoints, or None."""
        cells = self._astar(ca, cb, extra)
        if cells is None:
            return None
        allow = {(y + dy, x + dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        pts = [self.grid.to_world(*c) for c in cells]
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._los(pts[i], pts[j], allow, extra):
                j -= 1
            out.append(pts[j])
            i = j
        return out[1:-1]
