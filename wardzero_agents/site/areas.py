"""Forbidden areas the user drew in the Studio: rectangles the robot must keep out of.

The scenario carries them (Scenario.forbidden_areas). This module turns them into what each part needs:
polygons for the task brief (developer robots), a router for the built-in robot that treats them as
walls, and the list a finished run's report reads back from the run's config copy. Isaac shows nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from ..config.models import ForbiddenArea


def polygon(a: ForbiddenArea) -> list[list[float]]:
    """The rectangle's corners, counter-clockwise from (x0, y0), metres in the map frame."""
    return [[a.x0, a.y0], [a.x1, a.y0], [a.x1, a.y1], [a.x0, a.y1]]


def brief_areas(areas: Iterable[ForbiddenArea]) -> list[dict]:
    """The task brief's forbidden_areas: [{id, name, polygon}]."""
    return [{"id": a.id, "name": a.name, "polygon": polygon(a)} for a in areas]


def task_brief(brief: Optional[dict], areas: Iterable[ForbiddenArea] = ()) -> Optional[dict]:
    """What the sim publishes latched on <ns>/task/brief: the scenario's robot_brief, plus forbidden_areas
    when there are any. None when there is neither."""
    areas = list(areas or [])
    if brief is None and not areas:
        return None
    out = dict(brief or {})
    if areas:
        out["forbidden_areas"] = brief_areas(areas)
    return out


def areas_from_brief(brief: Optional[dict]) -> list[ForbiddenArea]:
    """Forbidden areas back from a task brief (each polygon's bounding box); bad entries are skipped."""
    out = []
    for i, d in enumerate((brief or {}).get("forbidden_areas") or []):
        try:
            xs = [float(p[0]) for p in d["polygon"]]
            ys = [float(p[1]) for p in d["polygon"]]
            out.append(ForbiddenArea(id=str(d.get("id") or f"area{i + 1}"), name=str(d.get("name") or "area"),
                                     x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys)))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


class AreaRouter:
    """The robot's router: the world's router with the forbidden areas blocked. It follows the world's
    router (a blocked corridor there is blocked here too). A route that starts inside an area uses the
    world's router so the robot can walk out; a route that ends inside one is refused (None)."""

    def __init__(self, base, areas: Iterable[ForbiddenArea]):
        self.base = base
        self.areas = list(areas)
        self.grid = base.grid
        self.areas_key = tuple((a.x0, a.y0, a.x1, a.y1) for a in self.areas)
        self._blocked = None
        self._version = None

    def _router(self):
        v = getattr(self.base, "version", 0)
        if self._blocked is None or v != self._version:
            self._blocked = self.base.with_rects_blocked(self.areas_key)
            self._version = v
        return self._blocked

    def inside(self, x: float, y: float) -> Optional[ForbiddenArea]:
        return next((a for a in self.areas if a.contains(x, y)), None)

    def route(self, a: tuple, b: tuple):
        if self.inside(*b[:2]) is not None:
            return None
        if self.inside(*a[:2]) is not None:
            return self.base.route(a, b)
        return self._router().route(a, b)

    def is_free(self, x: float, y: float) -> bool:
        return self.inside(x, y) is None and self.base.is_free(x, y)

    def block(self, x: float, y: float, radius: float) -> None:
        self.base.block(x, y, radius)


def robot_router(base, areas: Iterable[ForbiddenArea]):
    """The base router itself when there are no areas, else an AreaRouter over it."""
    areas = list(areas or [])
    return AreaRouter(base, areas) if areas else base


def run_areas(run_dir: "str | Path", meta: Optional[dict] = None) -> list[ForbiddenArea]:
    """The forbidden areas of a finished run, from the scenario in its config copy (config/<scenario file>).
    [] when the copy is missing or has none."""
    meta = meta or {}
    cfg = Path(run_dir) / "config"
    names = []
    if isinstance(meta.get("scenario_path"), str):
        names.append(Path(meta["scenario_path"]).name)
    if isinstance(meta.get("scenario_id"), str):
        names.append(f"{meta['scenario_id']}.yaml")
    for name in names:
        f = cfg / name
        if not f.is_file():
            continue
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            return [ForbiddenArea.model_validate(a) for a in data.get("forbidden_areas") or []]
        except (OSError, yaml.YAMLError, ValueError, AttributeError, TypeError):
            return []
    return []
