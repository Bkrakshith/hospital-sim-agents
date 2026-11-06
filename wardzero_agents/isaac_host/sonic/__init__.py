"""NVIDIA GEAR-SONIC whole-body controller for the G1, simulator-agnostic (no Isaac imports).

Whole-body controller: NVIDIA GEAR-SONIC (NVlabs GR00T-WholeBodyControl, Apache-2.0 code; weights under the NVIDIA
Open Model License, fetched by ``scripts/fetch_sonic.py``, never committed). Luo et al., arXiv:2511.07820.

The Isaac glue stays thin:

    jm = JointMap(dof_names)                        # by name, never by position
    runner = SonicRunner(SonicController.from_weights())
    every 4th physics step (50 Hz):
        out = runner.tick(t, jm.to_sonic(q), jm.to_sonic(dq), pelvis_quat_wxyz, pelvis_gyro_body, pelvis_z)
        pin the pelvis while out.harness
    every physics step (200 Hz):
        tau = pd_torque(out.targets.q, jm.to_sonic(q), jm.to_sonic(dq), out.targets.kp, out.targets.kd, effort)
        apply jm.to_sim(tau, fill=0) as joint efforts
"""
from .controller import LogEntry, SonicController, Targets, default_pose_targets
from .joint_map import JointMap, pd_torque
from .motion import Motion, PlannerOutputError
from .params import (ARMATURE_OURS, CONTROL_DT, DEFAULT_ANGLES, EFFORT_OURS, EFFORT_SONIC, IDLE, IL_JOINTS, MJ_JOINTS, PHYSICS_DT,
                     RUN, SLOW_WALK, WALK)
from .planner import PlannerCommand, PlannerWorker, needs_replan
from .runner import FALL_TILT_DEG, FALL_Z_M, HARNESS_Z_M, LOW_Z_M, Phase, SonicRunner, TickOutput, fall_reason
from .weights import LICENCE_NOTE, missing_files, resolve_dir

__all__ = [
    "SonicController", "SonicRunner", "Targets", "TickOutput", "Phase", "LogEntry", "Motion", "PlannerCommand",
    "PlannerWorker", "PlannerOutputError", "JointMap", "pd_torque", "fall_reason", "needs_replan",
    "default_pose_targets", "missing_files", "resolve_dir", "LICENCE_NOTE", "MJ_JOINTS", "IL_JOINTS",
    "DEFAULT_ANGLES", "ARMATURE_OURS", "EFFORT_OURS", "EFFORT_SONIC", "IDLE", "SLOW_WALK", "WALK", "RUN", "CONTROL_DT", "PHYSICS_DT",
    "FALL_Z_M", "FALL_TILT_DEG", "HARNESS_Z_M", "LOW_Z_M",
]
