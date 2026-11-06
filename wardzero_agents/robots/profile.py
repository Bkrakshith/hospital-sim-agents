"""Robot profiles: what a robot body can do, which the robot's mind is told and must not exceed.

The built-in g1_claude profile lives here; bring-your-own profiles are saved by the Studio to
<data>/robots/<name>/robot.yaml.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from .sensors import SensorSpec


G1_NAME = "g1_claude"


@dataclass(frozen=True)
class RobotProfile:
    name: str
    can_move: bool
    base: Literal["static", "legged", "wheeled", "mobile"]
    radius: float
    reach: tuple[float, float]  # [zmin, zmax] the hands can put things at, metres
    max_speed: float
    ns: str
    sensors: tuple[SensorSpec, ...]
    urdf: Optional[Path] = None
    controller: Literal["builtin_claude", "external", "none"] = "builtin_claude"
    base_z: float = 0.0  # where the URDF's root link sits above the floor when the robot is placed (the G1's pelvis)
    voice: Optional[str] = None  # the spoken voice's key in wardzero_agents.voice.catalogue (None: the default robot voice)

    def abilities(self, desk_staffed: bool = True) -> str:
        """The robot's body in words, for the prompt. desk_staffed False (nobody reads the reception desk computer)
        leaves out the desk note."""
        lo, hi = self.reach
        if self.name == G1_NAME:
            body = "Your body: a Unitree G1 humanoid with a head camera and a microphone and speaker."
        else:
            kind = {"legged": "legged", "wheeled": "wheeled", "static": "fixed-base"}.get(self.base, "mobile")
            body = (f"Your body: a {kind} robot called {self.name}, body radius {self.radius:g} m, with "
                    f"{', '.join(x.label() for x in self.sensors) or 'no sensors'}.")
        lines = [
            body,
            f"Sensors: {', '.join(x.label() for x in self.sensors)}.",
            f"Your hands reach between {lo:g} m and {hi:g} m above the floor, within arm's length of you.",
            "You can hand an item you are holding to a person, take an item lying within arm's length, "
            "spray a fire extinguisher at a small fire, press the staff call button, speak, and wait.",
            "You can tell the staff that someone needs help (tell_staff): you go to the reception desk (or the "
            "nurse base on a ward) and speak to the member of staff there.",
        ]
        if desk_staffed:
            lines.append("You can leave a note about someone on the reception desk computer (note_desk): the "
                         "receptionist reads it on her screen when she is at the desk, nobody is interrupted and you "
                         "stay where you are.")
        if self.can_move:
            lines.append(f"You can walk (up to {self.max_speed:g} m/s, body radius {self.radius:g} m): "
                         "to a named place, up to a person (approach), or while leading a person to a place or to a "
                         "free seat there.")
            lines.append("While you walk, your body slows to 0.3 m/s near people and stops for, or steps round, "
                         "anyone in your way by itself; you do not need to use wait for that, and wait cancels your "
                         "walk, guide or tell_staff. If someone stays in your way for about 8 s the walk ends as "
                         "blocked_by_person; then choose again. extinguish walks you up to a fire you can see.")
        else:
            lines.append("You are standing at a fixed post and cannot walk in this build; to help someone "
                         "get somewhere, give clear directions or press the call button for staff. extinguish "
                         "reaches only a fire within 2 m.")
        return "\n".join(lines)


# The URDF mounts the head D435 pitched 47.6 degrees down, which sees a visitor's legs and the floor at reception. The
# built-in G1's camera is tilted back up by 0.70 rad so it looks about 7 degrees down (a developer's robot keeps its own mount).
G1_HEAD_CAMERA_PITCH = -0.70
G1_CLAUDE = RobotProfile(
    name="g1_claude", can_move=False, base="static", radius=0.3, reach=(0.6, 1.3), max_speed=1.0,
    ns="robot", base_z=0.80, voice="en_GB-cori-medium", sensors=(
        SensorSpec("head_camera", "rgbd", "realsense_d455", "d435_link", (0.0, 0.0, 0.0), (0.0, G1_HEAD_CAMERA_PITCH, 0.0),
                   {"label": "head RealSense D455 camera"}),
        SensorSpec("mic", "mic", None, "head_link", params={"label": "microphone"}),
        SensorSpec("speaker", "speaker", None, "head_link", params={"label": "speaker"})))

PROFILES = {G1_CLAUDE.name: G1_CLAUDE}

# THE SWITCH for the built-in G1's legs. In Quick 2D (the stub) the G1 walks as a simple moving body on the same routes.
# In Isaac Sim it walks (NVIDIA GEAR-SONIC in the host) when the SONIC weights are on this machine
# (`python scripts/fetch_sonic.py`, checked by `hhumans doctor`); without them it stands at its post.
# WARDZERO_G1_WALKS_IN_ISAAC=0 or 1 forces the Isaac side either way.
G1_WALK_SPEED = 0.6  # m/s: SONIC's SLOW_WALK as the gate test drove it (the ankles saturate more in WALK mode)


def g1_walks_in_isaac() -> bool:
    env = os.environ.get("WARDZERO_G1_WALKS_IN_ISAAC", "").strip()
    if env in ("0", "1"):
        return env == "1"
    from ..isaac_host.sonic.weights import missing_files  # pure Python: no onnxruntime, no Kit
    return not missing_files()


def g1_static_reason() -> str:
    """Why the built-in G1 stands still in Isaac Sim, and the fix (for logs and the run's output)."""
    from ..isaac_host.sonic.weights import resolve_dir
    if os.environ.get("WARDZERO_G1_WALKS_IN_ISAAC", "").strip() == "0":
        return "WARDZERO_G1_WALKS_IN_ISAAC=0 keeps the G1 at its post in Isaac Sim."
    return (f"The G1 stands at its post in Isaac Sim: the GEAR-SONIC walking controller weights are not in "
            f"{resolve_dir()}. Run `python scripts/fetch_sonic.py` in hospital-sim-agents (or set WARDZERO_SONIC_DIR).")


def profile_for_sim(profile: RobotProfile, sim: str) -> RobotProfile:
    """The profile the robot's mind and body work with in this simulator. Only the built-in G1 changes: it walks in
    Quick 2D, and in Isaac when g1_walks_in_isaac(). Other robots keep what their robot.yaml says."""
    if profile.name != G1_NAME or profile.can_move:
        return profile
    if sim == "isaac" and not g1_walks_in_isaac():
        return profile
    return dataclasses.replace(profile, can_move=True, base="legged",
                              max_speed=G1_WALK_SPEED if sim == "isaac" else profile.max_speed)


def get_profile(name: str) -> RobotProfile:
    if name in PROFILES:
        return PROFILES[name]
    try:  # a bring-your-own robot saved by the Studio
        return load_profile(name)
    except (KeyError, ValueError, OSError):
        raise KeyError(f"unknown robot profile {name!r}") from None


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def profile_path(name: str, data: Optional[Path] = None) -> Path:
    if not _NAME.match(name or ""):
        raise ValueError(f"bad robot name {name!r}: use letters, digits, - and _")
    if data is None:
        from ..memory import data_dir
        data = data_dir()
    return Path(data) / "robots" / name / "robot.yaml"


def save_profile(profile: RobotProfile, data: Optional[Path] = None) -> Path:
    path = profile_path(profile.name, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"name": profile.name, "urdf": str(profile.urdf) if profile.urdf else None, "base": profile.base,
           "can_move": profile.can_move, "radius": profile.radius, "reach": list(profile.reach),
           "max_speed": profile.max_speed, "ns": profile.ns, "controller": profile.controller,
           "base_z": profile.base_z, "voice": profile.voice, "sensors": [x.to_dict() for x in profile.sensors]}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def load_profile(name: str, data: Optional[Path] = None) -> RobotProfile:
    """Rule: the dataclass default controller ("builtin_claude") is for the built-in G1 only. A saved profile
    always records its controller; one saved without it was written by hand, and is "external" (the developer's own)."""
    path = profile_path(name, data)
    if not path.is_file():
        raise KeyError(f"unknown robot profile {name!r}")
    d = yaml.safe_load(path.read_text()) or {}
    base = d.get("base", "static")
    return RobotProfile(
        name=d["name"], can_move=bool(d.get("can_move", base != "static")), base=base,
        radius=float(d.get("radius", 0.3)), reach=tuple(float(v) for v in d.get("reach", (0.6, 1.3))),
        max_speed=float(d.get("max_speed", 1.0)), ns=d.get("ns") or d["name"],
        sensors=tuple(SensorSpec.from_dict(x) for x in d.get("sensors", [])),
        urdf=Path(d["urdf"]) if d.get("urdf") else None, controller=d.get("controller", "external"),
        base_z=float(d.get("base_z", 0.0)), voice=d.get("voice") or None)
