"""Where the Isaac mode's files live, and what is missing. No Isaac imports: the CLI and the Studio use it
before anything is started.

Nothing here is published with the repo. The stage is part of the hospital-sim-env checkout and the collision
USD is built once from it with ``hospital-sim-humans/sim/build_collision_usd.py``."""
from __future__ import annotations

import os
from pathlib import Path

HG1 = Path(__file__).resolve().parents[2]
STAGE_ENV, COLLISION_ENV, ROS_ENV_SH_ENV = "WARDZERO_STAGE_USD", "WARDZERO_COLLISION_USD", "WARDZERO_ROS_ENV_SH"
BUILD_SCRIPT = "hospital-sim-humans/sim/build_collision_usd.py"


def _path(env: str, default: Path) -> Path:
    v = os.environ.get(env, "").strip()
    return Path(v).expanduser() if v else default


def stage_usd() -> Path:
    return _path(STAGE_ENV, HG1 / "hospital-sim-env" / "isaac" / "usd" / "hospital_g1.usd")


def collision_usd() -> Path:
    return _path(COLLISION_ENV, HG1 / "hospital-sim-humans" / "sim" / "assets" / "hospital_collision.usd")


def ros_env_sh() -> Path:
    return _path(ROS_ENV_SH_ENV, HG1 / "env.sh")


def isaac_python(isaac_root: Path) -> Path:
    return Path(isaac_root) / "python.sh"


def missing_for_isaac(isaac_root: Path) -> list[str]:
    """What an Isaac run needs and does not find, one clear sentence each (empty: all there)."""
    out = []
    if not isaac_python(isaac_root).is_file():
        out.append(f"Isaac Sim's python.sh is not in {isaac_root} (set ISAAC_SIM_ROOT to an Isaac Sim 6.0 install).")
    if not stage_usd().is_file():
        out.append(f"the hospital stage is missing: {stage_usd()}. It comes with the hospital-sim-env checkout; "
                   f"set {STAGE_ENV} if it lives elsewhere.")
    if not collision_usd().is_file():
        out.append(f"the collision USD is missing: {collision_usd()}. Without it the people walk through walls. "
                   f"Make it with {BUILD_SCRIPT} (run with Isaac Sim's python.sh), or set {COLLISION_ENV}.")
    return out
