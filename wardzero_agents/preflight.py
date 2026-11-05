"""Checks made before an Isaac run starts, so a run that cannot work is refused with the fix instead of failing
a minute later inside Isaac Sim. No Isaac imports: the CLI and the Studio call these first."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .robots.profile import RobotProfile

FIX_ROBOT = ("On the Robot page, pick the built-in G1, or give your robot's URDF folder and press Check the URDF. "
             "Or choose Quick 2D on the Run page.")


def robot_problem(profile: RobotProfile) -> Optional[str]:
    """Why Isaac cannot build this robot, or None. The built-in G1 uses the URDF that ships next to this repo."""
    from .robots.ready_usd import robot_urdf
    try:
        urdf = robot_urdf(profile)
    except ValueError:
        return f"The robot '{profile.name}' has no URDF, so Isaac Sim cannot build it. {FIX_ROBOT}"
    if not Path(urdf).is_file():
        return f"The robot '{profile.name}' URDF is missing: {urdf}. {FIX_ROBOT}"
    return None


def isaac_pids() -> list[int]:
    """Isaac Sim (Kit) processes running now. Empty when there are none or pgrep is not there."""
    try:
        out = subprocess.run(["pgrep", "-x", "kit"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(t) for t in out.split() if t.isdigit()]


def isaac_busy_problem(pids: Optional[list[int]] = None) -> Optional[str]:
    pids = isaac_pids() if pids is None else pids
    if not pids:
        return None
    listed = " ".join(str(p) for p in pids)
    return (f"another Isaac Sim is running (process {listed}), and the graphics card cannot hold two. Close it, or "
            f"stop it with `kill {listed}`, then start again.")


def isaac_start_problems(profile: Optional[RobotProfile]) -> list[str]:
    """Everything that stops an Isaac run with this robot (None: no robot), one sentence and fix each."""
    out = []
    if profile is not None:
        p = robot_problem(profile)
        if p:
            out.append(p)
    busy = isaac_busy_problem()
    if busy:
        out.append(busy[0].upper() + busy[1:])
    return out
