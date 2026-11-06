"""The report's map: the robot's path, the forbidden areas in red and where it went into one.

A small inline SVG built from the run's site and robot_pose samples; no scripts, no external files.
"""
from __future__ import annotations

import html
from typing import Iterable, Optional

MAX_PATH_POINTS = 600


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def area_map_svg(site, poses: list[dict], areas: Iterable, entries: list[dict]) -> Optional[str]:
    """An SVG framing the forbidden areas, the robot's path and the entries (y drawn up, so (x, -y)).
    None when there is nothing to frame."""
    areas = list(areas)
    pts = [(float(p["x"]), float(p["y"])) for p in poses]
    frame = [(a.x0, a.y0) for a in areas] + [(a.x1, a.y1) for a in areas] + [(e["x"], e["y"]) for e in entries]
    if not frame:
        return None
    if pts:
        frame += pts
    x0, x1 = min(p[0] for p in frame), max(p[0] for p in frame)
    y0, y1 = min(p[1] for p in frame), max(p[1] for p in frame)
    pad = max(2.0, 0.08 * max(x1 - x0, y1 - y0))
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    if x1 - x0 < 12:
        c = (x0 + x1) / 2
        x0, x1 = c - 6, c + 6
    if y1 - y0 < 8:
        c = (y0 + y1) / 2
        y0, y1 = c - 4, c + 4
    w, h = x1 - x0, y1 - y0
    fs = max(w, h) / 45  # label size in metres
    out = [f'<svg class="amap" viewBox="{_f(x0)} {_f(-y1)} {_f(w)} {_f(h)}" role="img" '
           f'aria-label="Map of the robot\'s path and the forbidden areas" xmlns="http://www.w3.org/2000/svg">']
    for r in getattr(site, "rooms", {}).values():
        if not r.poly:
            continue
        xs, ys = [p[0] for p in r.poly], [p[1] for p in r.poly]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        pp = " ".join(f"{_f(px)},{_f(-py)}" for px, py in r.poly)
        out.append(f'<polygon class="room" points="{pp}"><title>{html.escape(r.label or r.id)}</title></polygon>')
    d = "".join(f"M{_f(wl[0])} {_f(-wl[1])}L{_f(wl[2])} {_f(-wl[3])}" for wl in getattr(site, "walls", [])
                if len(wl) >= 4)
    if d:
        out.append(f'<path class="walls" d="{d}" stroke-width="{_f(fs * 0.25)}"/>')
    for a in areas:
        out.append(f'<rect class="forbid" x="{_f(a.x0)}" y="{_f(-a.y1)}" width="{_f(a.x1 - a.x0)}" '
                   f'height="{_f(a.y1 - a.y0)}" stroke-width="{_f(fs * 0.2)}"><title>{html.escape(a.name)}</title></rect>')
        out.append(f'<text class="forbid-l" x="{_f((a.x0 + a.x1) / 2)}" y="{_f(-a.y1 - fs * 0.4)}" '
                   f'font-size="{_f(fs)}" text-anchor="middle">{html.escape(a.name)}</text>')
    if pts:
        step = max(1, len(pts) // MAX_PATH_POINTS)
        keep = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        line = " ".join(f"{_f(x)},{_f(-y)}" for x, y in keep)
        out.append(f'<polyline class="path" points="{line}" stroke-width="{_f(fs * 0.22)}"/>')
        out.append(f'<circle class="start" cx="{_f(pts[0][0])}" cy="{_f(-pts[0][1])}" r="{_f(fs * 0.45)}">'
                   "<title>Robot start</title></circle>")
    for i, e in enumerate(entries, 1):
        out.append(f'<circle class="entry" cx="{_f(e["x"])}" cy="{_f(-e["y"])}" r="{_f(fs * 0.55)}">'
                   f'<title>Entry {i}: {html.escape(str(e["name"]))}</title></circle>')
        out.append(f'<text class="entry-n" x="{_f(e["x"])}" y="{_f(-e["y"])}" font-size="{_f(fs * 0.7)}" '
                   f'text-anchor="middle" dominant-baseline="central">{i}</text>')
    out.append("</svg>")
    return "".join(out)
