"""8-connected A* over the occupancy grid with clearance inflation and string-pulling."""
from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from typing import Optional

import numpy as np

from .model import OccGrid

CLEARANCE = 0.3
NEAR_PENALTY = 3.0  # extra cost per step inside the clearance band
_SQ2 = math.sqrt(2.0)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


class Router:
    """Raw-blocked cells are impassable. Cells within `clearance` of a blocked cell are passable but cost
    more, so routes keep clear of walls where the map allows and still squeeze through tight doorways."""

    def __init__(self, grid: OccGrid, clearance: float = CLEARANCE, cache_size: int = 4096):
        self.grid = grid
        occ = grid.occ.astype(bool)
        near = self._inflate(occ, clearance)
        self._raw = occ.tolist()
        self._near = near.tolist()
        self._b = self._raw
        self._cache: "OrderedDict[tuple, Optional[list]]" = OrderedDict()
        self._size = cache_size
        self.version = 0  # goes up with every block(): a router built from this one knows to rebuild

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
        self._near = self._inflate(np.array(self._raw, dtype=bool), CLEARANCE).tolist()
        self._cache.clear()
        self.version += 1

    def with_rects_blocked(self, rects) -> "Router":
        """A new router with this one's blocked cells plus every cell whose centre is inside one of the
        rectangles (x0, y0, x1, y1): forbidden areas only the robot keeps out of."""
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
        out = Router.__new__(Router)
        out.grid, out._size, out.version = g, self._size, 0
        out._raw = raw.tolist()
        out._near = out._inflate(raw, CLEARANCE).tolist()
        out._b = out._raw
        out._cache = OrderedDict()
        return out

    def is_free(self, x: float, y: float) -> bool:
        """Free of walls and of the clearance band."""
        iy, ix = self.grid.to_cell(x, y)
        return self.grid.in_grid(iy, ix) and not self._near[iy][ix]

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

    def _los(self, p, q, allow) -> bool:
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
            if not g.in_grid(iy, ix) or raw[iy][ix]:
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

    def _astar(self, s, g):
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
                if not (0 <= ny < h and 0 <= nx < w) or b[ny][nx]:
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

    def route(self, a: tuple, b: tuple) -> Optional[list[tuple[float, float]]]:
        """Waypoints from a to b. An endpoint in a wall is moved to the nearest free cell centre (within 1 m);
        None if there is none or no route. Cached on the (cell_a, cell_b) pair."""
        ca, cb = self._snap(self.grid.to_cell(*a)), self._snap(self.grid.to_cell(*b))
        if ca is None or cb is None:
            return None

        def endpoint(p, c):
            return tuple(p) if self.grid.to_cell(*p) == c else self.grid.to_world(*c)

        ea, eb = endpoint(a, ca), endpoint(b, cb)
        key = (ca, cb)
        if key in self._cache:
            self._cache.move_to_end(key)
            mid = self._cache[key]
        else:
            cells = self._astar(ca, cb)
            if cells is None:
                mid = None
            else:
                allow = {(y + dy, x + dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                pts = [self.grid.to_world(*c) for c in cells]
                out = [pts[0]]
                i = 0
                while i < len(pts) - 1:
                    j = len(pts) - 1
                    while j > i + 1 and not self._los(pts[i], pts[j], allow):
                        j -= 1
                    out.append(pts[j])
                    i = j
                mid = out[1:-1]
            self._cache[key] = mid
            if len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return None if mid is None else [ea] + list(mid) + [eb]
