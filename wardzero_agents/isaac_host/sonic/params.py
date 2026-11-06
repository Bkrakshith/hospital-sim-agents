"""GEAR-SONIC v1.1 robot parameters: joint names and orders, PD gains, action scale, default pose, rates.

Ported from ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`` (NVlabs
GR00T-WholeBodyControl, Apache-2.0, commit 1e1ec4f). Arrays are in SONIC's MuJoCo/hardware order (``MJ_JOINTS``)
unless the name says IsaacLab (``IL_``). Both permutations are derived from joint names, never typed by hand.

Modified port of NVIDIA code (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), Apache License 2.0.
"""
from __future__ import annotations

import numpy as np

# MuJoCo / hardware order: equals our contract order (robot_spec.json joint_names), index for index.
MJ_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# IsaacLab order: the policy's internal order (breadth-first over the training URDF).
IL_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
N_JOINTS = 29

# The C++ names are misleading; kept for cross-reference:
#   ISAACLAB_TO_MUJOCO[mj] = IsaacLab index of MuJoCo joint mj   (action[ISAACLAB_TO_MUJOCO] -> MuJoCo order)
#   MUJOCO_TO_ISAACLAB[il] = MuJoCo index of IsaacLab joint il   (q_mj[MUJOCO_TO_ISAACLAB] -> IsaacLab order)
ISAACLAB_TO_MUJOCO = np.array([IL_JOINTS.index(n) for n in MJ_JOINTS])
MUJOCO_TO_ISAACLAB = np.array([MJ_JOINTS.index(n) for n in IL_JOINTS])

# Gains: kp = armature * w^2, kd = 2 * zeta * armature * w, w = 2 pi 10 Hz (BeyondMimic); ankles and waist roll/pitch 2x.
ARM_5020, ARM_7520_14, ARM_7520_22, ARM_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_W = 10 * 2.0 * 3.1415926535
_ZETA = 2.0
_MOTOR = (["22", "22", "14", "22", "5020", "5020"] * 2 + ["14", "5020", "5020"]
          + ["5020"] * 5 + ["4010"] * 2 + ["5020"] * 5 + ["4010"] * 2)
_ARM = {"22": ARM_7520_22, "14": ARM_7520_14, "5020": ARM_5020, "4010": ARM_4010}
_EFF = {"22": 139.0, "14": 88.0, "5020": 25.0, "4010": 5.0}
_DOUBLE = {4, 5, 10, 11, 13, 14}


def _kp(a: float) -> float:
    return a * _W * _W


def _kd(a: float) -> float:
    return 2.0 * _ZETA * a * _W


KP = np.array([_kp(_ARM[m]) * (2.0 if i in _DOUBLE else 1.0) for i, m in enumerate(_MOTOR)])
KD = np.array([_kd(_ARM[m]) * (2.0 if i in _DOUBLE else 1.0) for i, m in enumerate(_MOTOR)])
ACTION_SCALE = np.array([0.25 * _EFF[m] / _kp(_ARM[m]) for m in _MOTOR])
# The v1.1 README's tested tuning: kp and kd x1.5 on the ankle-pitch motors (MuJoCo indices 4 and 10).
V11_GAIN_SCALE = {4: 1.5, 10: 1.5}
# Armature and effort SONIC was trained with (G1_CYLINDER_MODEL_12_DEX_CFG): the reference for the sim's joints.
ARMATURE = np.array([_ARM[m] * (2.0 if i in _DOUBLE else 1.0) for i, m in enumerate(_MOTOR)])
EFFORT_SONIC = np.array([139, 139, 88, 139, 50, 50, 139, 139, 88, 139, 50, 50, 88, 50, 50,
                         25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], float)
# Our G1 rev 1.0 URDF: hip pitch 7520-14 (88 N m), ankles and waist roll/pitch 35 N m.
EFFORT_OURS = np.array([88, 139, 88, 139, 35, 35, 88, 139, 88, 139, 35, 35, 88, 35, 35,
                        25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], float)

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# Planner modes (localmotion_kplanner.hpp). 4-8 squat/kneel/crawl, 9-16 boxing, 17-26 styled walks.
IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 9})
SLOW_WALK_SPEED = (0.2, 0.8)  # m/s accepted in SLOW_WALK

CONTROL_DT = 0.02  # 50 Hz policy
PLANNER_DT = 0.1  # planner decision every 5th tick
PLANNER_EVERY = 5
PHYSICS_DT = 0.005  # 200 Hz, as trained (decimation 4)
LOOK_AHEAD = 2  # planner context starts this many 50 Hz frames ahead of the current one
BLEND_FRAMES = 8
DEFAULT_HEIGHT = 0.788740  # planner-initialisation pelvis height
RANDOM_SEED = 1234

ENCODER_DIM, DECODER_DIM, TOKEN_DIM = 1751, 994, 64
HISTORY = 10  # decoder history frames, 0.02 s apart, oldest first
REF_FRAMES, REF_STEP = 10, 5  # encoder reference frames: current + 5k (k = 0..9), clamped to the last frame
