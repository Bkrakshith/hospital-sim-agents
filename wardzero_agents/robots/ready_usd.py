"""A robot's ready USD: where it lives, whether it is out of date, and how it is made.

The URDF importer (``scripts/isaac_import_urdf.py``, run with Isaac's Python) writes ``robot.usd`` next to the robot's
``robot.yaml`` with the sanitising rules of ``robots.sanitise`` applied, so the host can reference it in one authoring
pass after ``timeline.play()``. The Studio converts a developer's robot with the same script; the host makes the
built-in G1's the first time it is needed.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .profile import G1_NAME, RobotProfile, profile_path
from .sensors import write_sensors_yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "isaac_import_urdf.py"
# $HG1_ROOT (set by env.sh) is the Hospital_G1 checkout; a git worktree of this repo is not next to the URDF
G1_URDF = (Path(os.environ.get("HG1_ROOT") or REPO.parent) / "g1-brainco-description" / "g1_29dof_brainco" / "urdf"
           / "g1_29dof_brainco.urdf")
CONVERT_TIMEOUT_S = 600


def robot_urdf(profile: RobotProfile) -> Path:
    if profile.urdf is not None:
        return Path(profile.urdf)
    if profile.name == G1_NAME:
        return G1_URDF
    raise ValueError(f"robot {profile.name!r} has no URDF to convert")


def ready_usd_path(profile: RobotProfile, data: Optional[Path] = None, floating: bool = False) -> Path:
    """robot.usd (fixed base, anchored to the world) or robot_floating.usd (a free base: the walking G1)."""
    return profile_path(profile.name, data).parent / ("robot_floating.usd" if floating else "robot.usd")


def floating_base(profile: RobotProfile) -> bool:
    """A robot that walks on its own legs gets a free base; every other robot keeps the fixed one."""
    return profile.base == "legged" and profile.can_move


def _sensors_text(profile: RobotProfile) -> str:
    import yaml
    return yaml.safe_dump({"sensors": [s.to_dict() for s in profile.sensors]}, sort_keys=False)


def is_stale(usd: Path, inputs) -> bool:
    return not usd.exists() or any(Path(i).exists() and Path(i).stat().st_mtime > usd.stat().st_mtime for i in inputs)


def convert_command(urdf: Path, sensors: Path, out: Path, python: Optional[str] = None,
                    floating: bool = False) -> list[str]:
    py = python or os.environ.get("WARDZERO_ISAAC_PYTHON") or str(Path.home() / "isaacsim" / "python.sh")
    return [py, str(SCRIPT), "--urdf", str(urdf), "--sensors", str(sensors), "--out", str(out)] + (
        ["--floating-base"] if floating else [])


def ensure_ready_usd(profile: RobotProfile, data: Optional[Path] = None,
                     runner: Callable = subprocess.run) -> Path:
    """The ready USD for the robot, converting first when it is missing or older than its URDF, sensors or importer."""
    urdf = robot_urdf(profile)
    floating = floating_base(profile)
    out = ready_usd_path(profile, data, floating)
    out.parent.mkdir(parents=True, exist_ok=True)
    sensors = out.parent / "wardzero_sensors.yaml"
    text = _sensors_text(profile)
    if not sensors.exists() or sensors.read_text() != text:
        write_sensors_yaml(out.parent, profile.sensors)
    if not is_stale(out, (urdf, sensors, SCRIPT)):
        return out
    res = runner(convert_command(urdf, sensors, out, floating=floating), capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S)
    last = (res.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        info = json.loads(last[0])
    except ValueError:
        info = {"ok": False, "errors": [(res.stderr or res.stdout or "no output")[-300:]]}
    if res.returncode != 0 or not info.get("ok") or not out.exists():
        out.unlink(missing_ok=True)
        raise RuntimeError(f"converting {urdf.name} failed: {'; '.join(info.get('errors') or ['unknown error'])}")
    return out
