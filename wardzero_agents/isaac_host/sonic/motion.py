"""The planner's reference motion at 50 Hz: resampling the planner output, sampling its context, and blending.

Ported from ``localmotion_kplanner*.hpp`` and ``CurrentFrameAdvancement`` in ``g1_deploy_onnx_ref.cpp``.

Modified port of NVIDIA code (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), Apache License 2.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import BLEND_FRAMES, MUJOCO_TO_ISAACLAB, N_JOINTS
from .quat import slerp

CONTEXT_FRAMES = 4
QPOS_DIM = 7 + N_JOINTS  # [x, y, z, qw, qx, qy, qz, 29 joints in MuJoCo order]
MAX_PLANNER_FRAMES = 64


class PlannerOutputError(RuntimeError):
    """The planner returned NaN or more frames than its output holds."""


@dataclass
class Motion:
    """Root position and wxyz quaternion in the planner frame, joints in IsaacLab order (absolute angles)."""
    pos: np.ndarray  # [T, 3]
    quat: np.ndarray  # [T, 4]
    jp: np.ndarray  # [T, 29]
    jv: np.ndarray  # [T, 29]

    @property
    def T(self) -> int:
        return len(self.jp)

    def copy(self) -> "Motion":
        return Motion(self.pos.copy(), self.quat.copy(), self.jp.copy(), self.jv.copy())


def resample_50hz(q30: np.ndarray) -> Motion:
    """Planner output [n, 36] at 30 Hz to 50 Hz (lerp, slerp), joint velocities by finite difference x 50."""
    T30 = len(q30)
    if T30 > MAX_PLANNER_FRAMES or T30 < 2 or np.isnan(q30).any():
        raise PlannerOutputError(f"planner output invalid ({T30} frames, NaN={bool(np.isnan(q30).any())})")
    T50 = int(math.floor(T30 / 30.0 * 50))
    pos = np.zeros((T50, 3))
    quat = np.zeros((T50, 4))
    jp = np.zeros((T50, N_JOINTS))
    for f in range(T50):
        f30 = f / 50.0 * 30.0
        f0 = int(math.floor(f30))
        f1 = min(f0 + 1, T30 - 1)
        w1 = f30 - f0
        w0 = 1 - w1
        pos[f] = w0 * q30[f0, :3] + w1 * q30[f1, :3]
        quat[f] = slerp(q30[f0, 3:7], q30[f1, 3:7], f30 - f0)
        jp[f] = w0 * q30[f0, 7 + MUJOCO_TO_ISAACLAB] + w1 * q30[f1, 7 + MUJOCO_TO_ISAACLAB]
    jv = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) * 50.0
    jv[-1] = jv[-2]
    return Motion(pos, quat, jp, jv)


def initial_context(joint_pos_mj: np.ndarray, height: float) -> np.ndarray:
    """First planner call: 4 frames of the measured joints at the start height, identity orientation, origin."""
    ctx = np.zeros((CONTEXT_FRAMES, QPOS_DIM))
    ctx[:, 2] = height
    ctx[:, 3] = 1.0
    ctx[:, 7:] = joint_pos_mj
    return ctx


def context_from_motion(m: Motion, gen_frame: int) -> np.ndarray:
    """Later planner calls: 4 samples of the current motion (not the robot) from gen_frame, 1/30 s apart."""
    ctx = np.zeros((CONTEXT_FRAMES, QPOS_DIM))
    for n in range(CONTEXT_FRAMES):
        t = gen_frame / 50.0 + n / 30.0
        f50 = t * 50.0
        f0 = min(int(math.floor(f50)), m.T - 1)
        f1 = min(f0 + 1, m.T - 1)
        w1 = f50 - f0
        w0 = 1 - w1
        ctx[n, 3:7] = slerp(m.quat[f0], m.quat[f1], f50 - f0)
        ctx[n, :3] = w0 * m.pos[f0] + w1 * m.pos[f1]
        ctx[n, 7 + MUJOCO_TO_ISAACLAB] = w0 * m.jp[f0] + w1 * m.jp[f1]
    return ctx


def blend(old: Motion, frame: int, new: Motion, gen_frame: int, nb: int = BLEND_FRAMES) -> Motion | None:
    """Cross-fade ``new`` (generated from ``gen_frame`` of ``old``) into ``old`` seen from ``frame``.

    The result starts at the current frame (index 0 = old[frame]); the weight of the new motion ramps 0 -> 1 over
    ``nb`` frames from gen_frame - frame. None when the new motion would already be over (a very late result)."""
    L = gen_frame - frame + new.T
    if L <= 0:
        return None
    bs = max(0, gen_frame - frame)
    pos = np.zeros((L, 3))
    quat = np.zeros((L, 4))
    jp = np.zeros((L, N_JOINTS))
    jv = np.zeros((L, N_JOINTS))
    for f in range(L):
        fo = int(np.clip(f + frame, 0, old.T - 1))
        fn = int(np.clip(f + frame - gen_frame, 0, new.T - 1))
        wn = float(np.clip((f - bs) / nb, 0.0, 1.0))
        wo = 1 - wn
        jp[f] = wo * old.jp[fo] + wn * new.jp[fn]
        jv[f] = wo * old.jv[fo] + wn * new.jv[fn]
        pos[f] = wo * old.pos[fo] + wn * new.pos[fn]
        quat[f] = slerp(old.quat[fo], new.quat[fn], wn)
    return Motion(pos, quat, jp, jv)
