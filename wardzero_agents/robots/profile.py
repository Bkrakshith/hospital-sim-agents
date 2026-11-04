"""Robot profiles: what a robot body can do, which the robot's mind is told and must not exceed.

Only the built-in g1_claude profile lives here for now; the Studio adds bring-your-own profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RobotProfile:
    name: str
    can_move: bool
    base: Literal["static", "mobile"]
    radius: float
    reach: tuple[float, float]  # [zmin, zmax] the hands can put things at, metres
    max_speed: float
    ns: str
    sensors: tuple[str, ...]

    def abilities(self) -> str:
        """The robot's body in words, for the prompt."""
        lo, hi = self.reach
        lines = [
            "Your body: a Unitree G1 humanoid with a head camera and a microphone and speaker.",
            f"Sensors: {', '.join(self.sensors)}.",
            f"Your hands reach between {lo:g} m and {hi:g} m above the floor, within arm's length of you.",
            "You can hand an item you are holding to a person, take an item lying within arm's length, "
            "spray a fire extinguisher at a small fire in range, press the staff call button, speak, and wait.",
        ]
        if self.can_move:
            lines.append(f"You can walk (up to {self.max_speed:g} m/s, body radius {self.radius:g} m): "
                         "to a named place, or while leading a person there.")
        else:
            lines.append("You are standing at a fixed post and cannot walk in this build; to help someone "
                         "get somewhere, give clear directions or press the call button for staff.")
        return "\n".join(lines)


G1_CLAUDE = RobotProfile(
    name="g1_claude", can_move=False, base="static", radius=0.3, reach=(0.6, 1.3), max_speed=1.0,
    ns="robot", sensors=("head RealSense D455 camera", "microphone", "speaker"))

PROFILES = {G1_CLAUDE.name: G1_CLAUDE}


def get_profile(name: str) -> RobotProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown robot profile {name!r}") from None
