"""The walking G1 in the Isaac host, the parts that need no Kit: the frame and physics rates, the start-up numbers, the
per-step schedule (PD every physics step, a SONIC tick every 4th, a planner decision every 5th tick) and the
session-layer joint settings. ``walker`` does the Kit side with these.

The numbers are the ones the Isaac gate test proved (``research_notes/isaac_host/sonic_isaac_report.md``): physics
200 Hz on a fixed-step timeline whose frame rate divides 200 (25 fps = 8 physics steps a frame), Kit's minimum frame
rate lowered so it never drops a substep, explicit PD with our USD's efforts (35 N m ankles and waist), a gantry
start-up (pelvis held at 0.80 m while the joints ramp to SONIC's default pose for 1 s, lowered to 0.762 m over 0.4 s,
released on the tick the planner starts), SLOW_WALK at most 0.6 m/s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .sonic.params import ARMATURE_OURS, CONTROL_DT, EFFORT_OURS, MJ_JOINTS, PHYSICS_DT

PHYSICS_HZ = int(round(1.0 / PHYSICS_DT))  # 200
DECIMATION = int(round(CONTROL_DT / PHYSICS_DT))  # 4: one SONIC tick every 4th physics step
WALK_FPS = 25  # 8 physics steps a frame
DEFAULT_FPS = 30  # without a walking robot (the physics rate is left to PhysX)
MIN_FRAME_RATE = 1  # /persistent/simulation/minFrameRate: Kit never drops physics substeps on a slow frame

SPAWN_Z = 0.80  # pelvis height the robot is placed and held at
LOW_Z = 0.762  # lowered to this before the release (the feet just touch)
RAMP_S, LOWER_S, IDLE_S = 1.0, 0.4, 1.0

# the obstacle cylinder the people avoid follows a walking robot, but moving it costs ~40 ms on that frame (S1b)
WALK_PROXY_MOVE_M = 0.4
WALK_PROXY_EVERY_S = 1.0

# finger joints (BrainCo Revo2): held open (target 0) by stiff PhysX position drives; SONIC drives only the 29 body joints
FINGER_STIFFNESS = 150.0  # what the URDF importer gives every joint (N m / rad)
FINGER_DAMPING = 8.0

HAND_LINK_KEYS = ("_thumb_", "_index_", "_middle_", "_ring_", "_pinky_", "_base_link", "_base2_link",
                  "wrist_yaw_link")
LEG_LINK_KEYS = ("hip_", "knee_", "ankle_")


def frame_plan(fps: Optional[int], walking: bool) -> tuple[int, Optional[int]]:
    """(frames per sim second, physics steps per sim second or None for PhysX's default).

    A walking robot needs physics at 200 Hz in whole steps per frame, so its fps must divide 200 (default 25)."""
    if not walking:
        return (fps or DEFAULT_FPS), None
    fps = fps or WALK_FPS
    if fps <= 0 or PHYSICS_HZ % fps:
        raise ValueError(f"--fps {fps}: with the walking G1 the frame rate must divide {PHYSICS_HZ} "
                         f"(physics {PHYSICS_HZ} Hz in whole steps a frame), for example 20, 25, 40 or 50")
    return fps, PHYSICS_HZ


def steps_per_frame(fps: int, physics_hz: Optional[int]) -> Optional[int]:
    return None if not physics_hz else physics_hz // fps


@dataclass(frozen=True)
class JointSetting:
    """What the walker authors on one joint before PhysX parses the robot."""
    name: str
    stiffness: float
    damping: float
    max_force: Optional[float] = None  # None: keep the USD's
    armature: Optional[float] = None


def joint_settings(joint_names) -> list[JointSetting]:
    """SONIC's 29 joints: force mode with zero drive gains (the PD is the walker's), our USD's effort and armature
    (effort B of the gate test). Every other joint (the fingers): a stiff position drive holding it open."""
    body = {n: i for i, n in enumerate(MJ_JOINTS)}
    out = []
    for n in joint_names:
        i = body.get(n)
        if i is None:
            out.append(JointSetting(n, FINGER_STIFFNESS, FINGER_DAMPING))
        else:
            out.append(JointSetting(n, 0.0, 0.0, float(EFFORT_OURS[i]), float(ARMATURE_OURS[i])))
    return out


def is_hand_link(name: str) -> bool:
    return any(k in name for k in HAND_LINK_KEYS)


def is_leg_link(name: str) -> bool:
    return any(k in name for k in LEG_LINK_KEYS)


class StepSchedule:
    """Which physics steps run a SONIC tick (every DECIMATION-th, from the first) and the sim time of each step."""

    def __init__(self, decimation: int = DECIMATION, dt: float = PHYSICS_DT):
        self.decimation, self.dt = decimation, dt
        self.step = 0

    @property
    def t(self) -> float:
        return self.step * self.dt

    def tick_now(self) -> bool:
        return self.step % self.decimation == 0

    def advance(self) -> None:
        self.step += 1


def planner_command(fc, yaw0: float, hold_facing=(1.0, 0.0, 0.0)):
    """A route follower's command (world-frame yaw angles) -> SONIC's ``PlannerCommand`` in the planner frame.

    The planner frame is the robot's heading when the planner started (``yaw0``), and stays so: world directions are
    sent as ``angle - yaw0`` (gate test, item 8). IDLE with no facing keeps ``hold_facing`` (the last one sent), so an
    arrived robot does not turn back to where it started."""
    import math

    from .sonic.params import IDLE
    from .sonic.planner import PlannerCommand

    def unit(a: float) -> tuple[float, float, float]:
        a = a - yaw0
        return (float(math.cos(a)), float(math.sin(a)), 0.0)

    facing = unit(fc.face_yaw) if fc.face_yaw is not None else tuple(hold_facing)
    if fc.mode == IDLE or fc.move_yaw is None:
        return PlannerCommand(IDLE, (0.0, 0.0, 0.0), facing, -1.0, -1.0)
    return PlannerCommand(int(fc.mode), unit(fc.move_yaw), facing, float(fc.speed), -1.0)
