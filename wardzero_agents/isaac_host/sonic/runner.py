"""Start-up sequence, fall detection and command hand-off around ``SonicController``, driven by sim time.

Call ``SonicRunner.tick`` every 50 Hz policy tick (every 4th physics step at 200 Hz) with the robot state in
SONIC's MuJoCo joint order; apply ``pd_torque`` with the returned targets on every physics step; pin the pelvis
while ``harness`` is True. All timings use the sim time the caller passes, so a slow sim stays correct.

Start-up (``sonic_prep.md`` 8.4): HARNESS (pelvis pinned, joints ramp to SONIC's default pose over ``ramp_s``) ->
SETTLE (harness released, default pose held ``settle_s``) -> planner initialised -> IDLE (``idle_s``) -> READY
(commands are passed on). Any time after release, a fall (pelvis below 0.45 m or tilt over 60 deg) -> FALLEN:
damping only (kp = 0) and no more commands; the run reports it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .controller import SonicController, Targets, default_pose_targets
from .params import DEFAULT_ANGLES, IDLE, N_JOINTS
from .planner import PlannerCommand
from .quat import tilt_deg, yaw_of

FALL_Z_M = 0.45
FALL_TILT_DEG = 60.0
HARNESS_Z_M = 0.79  # pelvis height to pin at during the ramp


class Phase(str, Enum):
    HARNESS = "harness"
    SETTLE = "settle"
    IDLE = "idle"
    READY = "ready"
    FALLEN = "fallen"


def fall_reason(pelvis_z: float, base_quat_wxyz, floor_z: float = 0.0,
                min_z: float = FALL_Z_M, max_tilt_deg: float = FALL_TILT_DEG) -> Optional[str]:
    """Why the robot counts as fallen, or None: pelvis below min_z above the floor, or tilted over max_tilt_deg."""
    if pelvis_z - floor_z < min_z:
        return f"pelvis {pelvis_z - floor_z:.2f} m above the floor (< {min_z} m)"
    tilt = tilt_deg(base_quat_wxyz)
    if tilt > max_tilt_deg:
        return f"tilted {tilt:.0f} deg (> {max_tilt_deg:.0f} deg)"
    return None


@dataclass
class TickOutput:
    targets: Targets  # MuJoCo order; JointMap.to_sim puts them in the simulator's order
    harness: bool  # True: keep the pelvis pinned at HARNESS_Z_M
    phase: Phase
    fall: Optional[str] = None  # the reason, once fallen


class SonicRunner:
    def __init__(self, controller: SonicController, *, ramp_s: float = 1.0, settle_s: float = 0.5,
                 idle_s: float = 1.0, floor_z: float = 0.0) -> None:
        self.ctrl = controller
        self.ramp_s, self.settle_s, self.idle_s = ramp_s, settle_s, idle_s
        self.floor_z = floor_z
        self.phase = Phase.HARNESS
        self.fall: Optional[str] = None
        self._t_phase: Optional[float] = None
        self._q0: Optional[np.ndarray] = None
        self._hold = default_pose_targets(controller_gain_scale(controller))
        self._wanted: PlannerCommand = PlannerCommand()
        self.fell_at: Optional[float] = None

    # ---- commands ---------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.phase == Phase.READY

    def command(self, cmd: PlannerCommand) -> None:
        """Planner-frame command; applied once READY (IDLE before that)."""
        self._wanted = cmd

    def command_world(self, mode: int, speed: float = -1.0, move_yaw: Optional[float] = None,
                      face_yaw: Optional[float] = None, quantum_rad: float = 0.0) -> PlannerCommand:
        """World-frame yaw angles (rad) -> planner command (see SonicController.command_from_world)."""
        if not self.ctrl.ready:
            return self._wanted
        cmd = self.ctrl.command_from_world(mode, speed, move_yaw, face_yaw, quantum_rad=quantum_rad)
        if face_yaw is None:  # keep the facing last asked for
            cmd = PlannerCommand(cmd.mode, cmd.movement, self._wanted.facing, cmd.speed, cmd.height)
        self._wanted = cmd
        return cmd

    def follow(self, fc, quantum_rad: float = 0.0) -> PlannerCommand:
        """A ``world.route_follower.FollowerCommand`` (mode, speed, world move/face yaw) -> the planner command.
        The follower already holds its facing steady; quantum_rad > 0 also snaps angles to a grid."""
        if fc.mode == IDLE:
            return self.command_world(IDLE, -1.0, None, fc.face_yaw, quantum_rad)
        return self.command_world(int(fc.mode), float(fc.speed), fc.move_yaw, fc.face_yaw, quantum_rad)

    # ---- the 50 Hz tick ----------------------------------------------------------------------
    def tick(self, t: float, q_mj, dq_mj, base_quat_wxyz, gyro_body, pelvis_z: float) -> TickOutput:
        q_mj = np.asarray(q_mj, float)
        if self._t_phase is None:
            self._t_phase, self._q0 = t, q_mj.copy()
        if self.phase not in (Phase.HARNESS, Phase.FALLEN):
            why = fall_reason(pelvis_z, base_quat_wxyz, self.floor_z)
            if why:
                self.phase, self.fall, self.fell_at = Phase.FALLEN, why, t
        if self.phase == Phase.FALLEN:
            h = self._hold
            return TickOutput(Targets(q_mj.copy(), np.zeros(N_JOINTS), h.kd.copy(), np.zeros(N_JOINTS)),
                              False, self.phase, self.fall)
        if self.phase == Phase.HARNESS:
            a = 1.0 if self.ramp_s <= 0 else min(1.0, (t - self._t_phase) / self.ramp_s)
            q = self._q0 + a * (DEFAULT_ANGLES - self._q0)
            if a >= 1.0:
                self.phase, self._t_phase = Phase.SETTLE, t
                return TickOutput(self._held(q), False, Phase.SETTLE)
            return TickOutput(self._held(q), True, Phase.HARNESS)
        if self.phase == Phase.SETTLE:
            if t - self._t_phase < self.settle_s:
                return TickOutput(self._held(DEFAULT_ANGLES), False, self.phase)
            self.ctrl.planner_initialize(q_mj)
            self.phase, self._t_phase = Phase.IDLE, t
        if self.phase == Phase.IDLE and t - self._t_phase >= self.idle_s:
            self.phase = Phase.READY
        self.ctrl.set_command(self._wanted if self.phase == Phase.READY else PlannerCommand(IDLE))
        out = self.ctrl.step(q_mj, dq_mj, base_quat_wxyz, gyro_body)
        return TickOutput(out, False, self.phase)

    def _held(self, q) -> Targets:
        h = self._hold
        return Targets(np.asarray(q, float).copy(), h.kp, h.kd, np.zeros(N_JOINTS))

    # ---- what a route follower needs ---------------------------------------------------------
    @staticmethod
    def yaw(base_quat_wxyz) -> float:
        return yaw_of(base_quat_wxyz)


def controller_gain_scale(ctrl: SonicController) -> dict[int, float]:
    """The gain scale the controller was built with, recovered from its kp (so the hold uses the same gains)."""
    from .params import KP
    return {i: float(ctrl.kp[i] / KP[i]) for i in range(N_JOINTS) if abs(ctrl.kp[i] / KP[i] - 1.0) > 1e-12}
