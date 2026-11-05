"""``hhumans doctor``: check this computer can run WardZero, before a run fails. Each check says what it found and,
when something is wrong, the steps that fix it. Nothing here calls the AI provider or starts Isaac Sim."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

OK, NOTE, FIX = "ok", "note", "FIX"
MIN_DISK_GB = 5.0
MIN_FREE_VRAM_MIB = 5000  # the hospital, three people and the G1 peak at about 6.6 GiB on an 8 GB card with a window


@dataclass(frozen=True)
class Check:
    status: str  # OK, NOTE (works, but worth knowing) or FIX (a run will fail or lose something)
    what: str
    fix: tuple[str, ...] = ()


def _gpu() -> Optional[tuple[int, int]]:
    """(free, total) MiB of the first GPU, or None without nvidia-smi."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=5)
        free, total = (int(v) for v in out.splitlines()[0].split(","))
        return free, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def check_ros(ros_available: Callable[[], bool]) -> list[Check]:
    out = []
    if ros_available():
        out.append(Check(OK, "ROS 2 is set up in this terminal."))
    else:
        out.append(Check(FIX, "ROS 2 is not set up in this terminal, so no run can start.",
                         ("Run `source ~/ntu_dev/Hospital_G1/env.sh`, then start the Studio from this terminal.",)))
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "conda" in ld:
        out.append(Check(FIX, "conda libraries are on LD_LIBRARY_PATH; they break ROS and Isaac Sim.",
                         ("Run `conda deactivate`, then `source ~/ntu_dev/Hospital_G1/env.sh`.",)))
    return out


def check_key(key_source: Callable[[str], str], not_set: str) -> list[Check]:
    if os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1":
        return [Check(NOTE, "Test mode (WARDZERO_TEST_FAKE_LLM=1): the people and robot do not use the AI provider "
                            "and nothing is billed.")]
    src = key_source("claude")
    if src == not_set:
        return [Check(FIX, "No Claude key is stored, so the people and the robot cannot think.",
                      ("Run `hhumans keys set claude` (or `ant auth login`).",
                       "To try the Studio without a key: `WARDZERO_TEST_FAKE_LLM=1 hhumans studio`."))]
    return [Check(OK, f"Claude key: {src} (it is checked against the models when a run starts).")]


def check_isaac(isaac_root: Path, missing: Callable[[Path], list[str]], busy: Callable[[], Optional[str]],
                gpu: Callable[[], Optional[tuple[int, int]]] = _gpu) -> list[Check]:
    if not (Path(isaac_root) / "python.sh").is_file():
        return [Check(NOTE, f"Isaac Sim is not installed at {isaac_root}: only Quick 2D runs work.",
                      ("Install Isaac Sim 6.0 there, or set ISAAC_SIM_ROOT to where it is.",))]
    out = []
    gaps = missing(isaac_root)
    if gaps:
        out += [Check(FIX, g[0].upper() + g[1:], ()) for g in gaps]
    else:
        out.append(Check(OK, f"Isaac Sim, the hospital stage and the collision USD are there ({isaac_root})."))
    b = busy()
    if b:
        out.append(Check(FIX, b[0].upper() + b[1:]))
    g = gpu()
    if g is None:
        out.append(Check(FIX, "nvidia-smi does not answer, so Isaac Sim will not find the graphics card.",
                         ("Check the NVIDIA driver: run `nvidia-smi`; reinstall the driver and reboot if it fails.",)))
    else:
        free, total = g
        if free < MIN_FREE_VRAM_MIB:
            out.append(Check(FIX, f"Only {free} MiB of {total} MiB graphics memory is free; an Isaac run needs about "
                                  f"{MIN_FREE_VRAM_MIB} MiB.",
                             ("Close other programs using the GPU (`nvidia-smi` lists them).",
                              "Run headless, which saves the window's share.")))
        else:
            out.append(Check(OK, f"Graphics memory: {free} MiB free of {total} MiB."))
            if total < 10000:
                out.append(Check(NOTE, "On this card a fire scenario with a window may not have room for the full "
                                       "fire effect; it is then drawn as a simple stand-in. Headless leaves more room."))
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        out.append(Check(NOTE, "No desktop display here: choose headless on the Run page (the preview still shows)."))
    return out


def check_disk(runs_dir: Path) -> list[Check]:
    p = Path(runs_dir)
    while not p.exists() and p != p.parent:
        p = p.parent
    free_gb = shutil.disk_usage(p).free / 1e9
    if free_gb < MIN_DISK_GB:
        return [Check(FIX, f"Only {free_gb:.1f} GB of disk is free where runs are saved ({runs_dir}); a recorded run "
                           "needs a few GB.", ("Delete old run folders you no longer need.",))]
    return [Check(OK, f"Disk: {free_gb:.0f} GB free for runs.")]


def check_voices(voices: Path) -> list[Check]:
    models = sorted(Path(voices).glob("*.onnx")) if Path(voices).is_dir() else []
    piper = importlib.util.find_spec("piper") is not None
    if models and piper:
        return [Check(OK, f"Voices: {len(models)} voice models installed.")]
    fix = []
    if not piper:
        fix.append("Run `pip install '.[voice]'` in hospital-sim-agents.")
    if not models:
        fix.append("Run `python scripts/fetch_voices.py` to download the voice models.")
    return [Check(NOTE, "Voices are not installed: recorded runs will be silent.", tuple(fix))]


def check_robots(data: Path, problem: Callable) -> list[Check]:
    from .robots.profile import load_profile
    out = []
    root = Path(data) / "robots"
    for f in sorted(root.glob("*/robot.yaml")) if root.is_dir() else []:
        try:
            prof = load_profile(f.parent.name, data)
        except (KeyError, ValueError, OSError) as exc:
            out.append(Check(FIX, f"Robot '{f.parent.name}' cannot be read: {exc}.",
                             ("Open it on the Studio's Robot page and save it again.",)))
            continue
        p = problem(prof)
        if p:
            out.append(Check(NOTE, f"Robot '{prof.name}' has no usable URDF yet: it runs in Quick 2D only.",
                             ("On the Robot page give its URDF folder and press Check the URDF.",)))
    return out


def run_checks() -> list[Check]:
    from . import cli
    from .isaac_assets import missing_for_isaac
    from .memory import data_dir
    from .preflight import isaac_busy_problem, robot_problem
    from .voice.catalogue import voices_dir

    return [*check_ros(cli.ros_available),
            *check_key(cli.key_source, cli.KEY_LABELS[3]),
            *check_isaac(cli.ISAAC_ROOT, missing_for_isaac, isaac_busy_problem),
            *check_disk(cli.runs_dir()),
            *check_voices(voices_dir()),
            *check_robots(data_dir(), robot_problem)]


def format_checks(checks: list[Check]) -> str:
    lines = ["WardZero setup check", ""]
    for c in checks:
        lines.append(f"  {c.status:<5} {c.what}")
        lines += [f"        {i}. {step}" for i, step in enumerate(c.fix, 1)]
    fixes = sum(c.status == FIX for c in checks)
    lines += ["", "All good." if not fixes else f"{fixes} thing{'s' if fixes > 1 else ''} to fix before a run."]
    return "\n".join(lines)
