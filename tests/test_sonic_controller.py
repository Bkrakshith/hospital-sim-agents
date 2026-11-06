"""GEAR-SONIC controller pieces: joint maps, history order, the 6-D anchor, replan rules, the planner thread,
the start-up sequence and fall detection. No weights or onnxruntime needed."""
from __future__ import annotations

import math
import random
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from tests.sonic_fakes import fake_sessions, standing_qpos
from wardzero_agents.isaac_host.sonic import (DEFAULT_ANGLES, IL_JOINTS, MJ_JOINTS, JointMap, Phase, PlannerCommand,
                                              SonicController, SonicRunner, fall_reason, needs_replan, pd_torque)
from wardzero_agents.isaac_host.sonic import params as P
from wardzero_agents.isaac_host.sonic.motion import Motion, blend, resample_50hz, PlannerOutputError
from wardzero_agents.isaac_host.sonic.planner import ReplanTimer
from wardzero_agents.isaac_host.sonic.quat import yaw_quat


def _ctrl(**kw) -> SonicController:
    return SonicController(*fake_sessions(), **kw)


def _stand(c: SonicController, **kw):
    return c.step(DEFAULT_ANGLES, np.zeros(29), np.array([1.0, 0, 0, 0]), np.zeros(3), **kw)


# ---- joint maps --------------------------------------------------------------------------------------------------
def test_isaaclab_permutation_from_names_matches_sonic():
    # policy_parameters.hpp, as copied into sonic_prep.md 5
    assert P.ISAACLAB_TO_MUJOCO.tolist() == [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                                             11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    assert P.MUJOCO_TO_ISAACLAB.tolist() == [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                                             16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
    assert [MJ_JOINTS[i] for i in P.MUJOCO_TO_ISAACLAB] == list(IL_JOINTS)
    assert P.KP[4] * 1.5 == pytest.approx(42.75, abs=0.01) and P.KD[4] * 1.5 == pytest.approx(2.722, abs=0.001)
    assert P.ACTION_SCALE[0] == pytest.approx(0.3507, abs=1e-4) and P.ACTION_SCALE[20] == pytest.approx(0.0745, abs=1e-4)


def test_joint_map_is_built_by_name_from_a_physx_style_order():
    fingers = [f"left_{f}_proximal_joint" for f in ("index", "middle", "thumb")]
    names = list(MJ_JOINTS[:22]) + fingers + list(MJ_JOINTS[22:]) + ["right_index_proximal_joint"]
    random.Random(3).shuffle(names)
    names = [f"/World/G1/joints/{n}" if i % 3 == 0 else n for i, n in enumerate(names)]
    jm = JointMap(names)
    assert jm.n_sim == 33 and len(jm.other) == 4
    q_sim = np.arange(33, dtype=float)
    q = jm.to_sonic(q_sim)
    for i, n in enumerate(MJ_JOINTS):
        assert names[int(q[i])].rsplit("/", 1)[-1] == n
    back = jm.to_sim(q, fill=-1.0)
    assert (back[jm.sim_index] == q).all() and (back[jm.other] == -1.0).all()
    kept = jm.to_sim(np.zeros(29), out=q_sim)  # untouched joints keep the given values
    assert (kept[jm.other] == q_sim[jm.other]).all() and not kept[jm.sim_index].any()


def test_joint_map_accepts_names_without_the_joint_suffix_and_reports_missing_ones():
    jm = JointMap([n[:-6] for n in reversed(MJ_JOINTS)])
    assert jm.sim_index.tolist() == list(range(28, -1, -1))
    with pytest.raises(ValueError, match="left_knee_joint"):
        JointMap([n for n in MJ_JOINTS if n != "left_knee_joint"])
    with pytest.raises(ValueError, match="twice"):
        JointMap(list(MJ_JOINTS) + ["left_knee"])
    with pytest.raises(ValueError):
        jm.to_sonic(np.zeros(30))


def test_pd_torque_is_clipped_to_the_effort():
    tau = pd_torque([1.0, 0.0], [0.0, 0.0], [0.0, 2.0], [100.0, 10.0], [1.0, 1.0], [35.0, 50.0])
    assert tau.tolist() == [35.0, -2.0]


# ---- history and observations ----------------------------------------------------------------------------------
def test_decoder_history_is_oldest_first_and_zero_padded():
    c = _ctrl()
    c.planner_initialize(DEFAULT_ANGLES)
    dec = c.decoder
    for k in range(3):
        q = DEFAULT_ANGLES.copy()
        q[3] += 0.1 * (k + 1)  # left knee
        c.step(q, np.full(29, k + 1.0), np.array([1.0, 0, 0, 0]), np.array([k + 1.0, 0, 0]))
    o = dec.feeds[-1]["obs_dict"].reshape(-1)
    assert o.shape == (994,)
    gyro = o[64:94].reshape(10, 3)
    assert gyro[:7].tolist() == [[0, 0, 0]] * 7 and gyro[7:, 0].tolist() == [1.0, 2.0, 3.0]
    body_q = o[94:384].reshape(10, 29)
    knee_il = IL_JOINTS.index("left_knee_joint")
    np.testing.assert_allclose(body_q[7:, knee_il], [0.1, 0.2, 0.3], atol=1e-6)  # minus the default pose
    assert not body_q[:7].any()
    assert o[384:674].reshape(10, 29)[7:].tolist() == [[1.0] * 29, [2.0] * 29, [3.0] * 29]
    grav = o[964:994].reshape(10, 3)
    assert not grav[:7].any() and grav[7:].tolist() == [[0, 0, -1.0]] * 3  # zero for padded frames


def test_last_action_history_is_the_previous_ticks_output():
    acts = iter([np.full((1, 29), v, np.float32) for v in (0.5, -0.25, 0.125)])
    enc, _, pl = fake_sessions()
    from tests.sonic_fakes import Session
    dec = Session(fn=lambda f: [next(acts)])
    c = SonicController(enc, dec, pl)
    c.planner_initialize(DEFAULT_ANGLES)
    outs = [_stand(c) for _ in range(3)]
    la = dec.feeds[-1]["obs_dict"].reshape(-1)[674:964].reshape(10, 29)
    assert la[-3:, 0].tolist() == [0.0, 0.5, -0.25]
    np.testing.assert_allclose(outs[0].q, DEFAULT_ANGLES + 0.5 * P.ACTION_SCALE)  # q* = default + a[perm] * scale


def test_six_d_anchor_is_the_reference_relative_to_the_robot_heading():
    c = _ctrl()
    c.planner_initialize(DEFAULT_ANGLES)
    _stand(c)  # heading re-init: init base = robot (yaw 0), init ref = motion (yaw 0)
    theta, phi = math.radians(30), math.radians(-20)
    c.motion.quat[:] = yaw_quat(theta)
    c.history[-1].base_quat = yaw_quat(phi)
    six = c.encoder_obs()[584:644].reshape(10, 6)
    a = theta - phi
    np.testing.assert_allclose(six, np.tile([math.cos(a), -math.sin(a), math.sin(a), math.cos(a), 0, 0], (10, 1)),
                               atol=1e-6)


def test_heading_reinit_maps_the_planner_frame_onto_the_robot_heading():
    c = _ctrl()
    c.planner_initialize(DEFAULT_ANGLES)
    psi = math.radians(70)
    c.step(DEFAULT_ANGLES, np.zeros(29), yaw_quat(psi), np.zeros(3))
    assert c.planner_yaw_offset() == pytest.approx(psi)
    assert c.reference_yaw() == pytest.approx(psi, abs=1e-6)
    # a world-frame facing of 160 deg is 90 deg in the planner frame
    cmd = c.command_from_world(P.IDLE, face_yaw=math.radians(160))
    assert cmd.facing[0] == pytest.approx(0.0, abs=1e-9) and cmd.facing[1] == pytest.approx(1.0)
    q = c.command_from_world(P.SLOW_WALK, 0.6, move_yaw=psi + math.radians(7), face_yaw=psi + math.radians(7),
                             quantum_rad=math.radians(5))
    assert math.degrees(math.atan2(q.movement[1], q.movement[0])) == pytest.approx(5.0)
    assert q.speed == 0.6 and q.mode == P.SLOW_WALK


def test_encoder_reference_frames_step_five_and_clamp_to_the_end():
    c = _ctrl()
    c.planner_initialize(DEFAULT_ANGLES)
    _stand(c)
    T = c.motion.T  # 30 planner frames at 30 Hz -> 50 frames at 50 Hz
    assert T == 50
    c.motion.jp[:] = np.arange(T)[:, None]
    c.frame = 30
    o = c.encoder_obs()
    assert o[4:294].reshape(10, 29)[:, 0].tolist() == [30, 35, 40, 45] + [49] * 6
    assert not o[:4].any() and not o[644:].any()


# ---- planner: resampling, blending, replan rules ----------------------------------------------------------------
def test_resample_and_invalid_planner_output():
    m = resample_50hz(standing_qpos(n=30, vx=0.3)[:30].astype(float))
    assert m.T == 50 and m.pos[-1, 0] == pytest.approx(0.3 * 29 / 30)  # frame 49 -> 30 Hz frame 29.4, clamped
    assert m.jv[0, 0] == 0.0 and m.jp[0].tolist() == DEFAULT_ANGLES[P.MUJOCO_TO_ISAACLAB].astype(np.float32).tolist()
    bad = standing_qpos()[:30].astype(float)
    bad[3, 9] = np.nan
    with pytest.raises(PlannerOutputError):
        resample_50hz(bad)


def test_blend_cross_fades_over_eight_frames_from_the_generation_frame():
    old = Motion(np.zeros((40, 3)), np.tile([1.0, 0, 0, 0], (40, 1)), np.zeros((40, 29)), np.zeros((40, 29)))
    new = Motion(np.ones((30, 3)), np.tile([1.0, 0, 0, 0], (30, 1)), np.ones((30, 29)), np.zeros((30, 29)))
    b = blend(old, 5, new, 7)
    assert b.T == 7 - 5 + 30
    w = b.jp[:, 0]
    assert w[:3].tolist() == [0, 0, 0] and w[3] == pytest.approx(1 / 8) and w[10:].tolist() == [1.0] * (b.T - 10)
    assert blend(old, 40, new, 2) is None  # too late to matter


def test_replan_rules():
    walk = PlannerCommand(P.SLOW_WALK, (1, 0, 0), (1, 0, 0), 0.6)
    assert not needs_replan(walk, walk, timer_expired=False)
    assert needs_replan(walk, walk, timer_expired=True)  # periodic, moving
    assert needs_replan(PlannerCommand(P.SLOW_WALK, (1, 0, 0), (1, 0, 0), 0.3), walk, False)  # speed
    assert needs_replan(PlannerCommand(P.SLOW_WALK, (0, 1, 0), (1, 0, 0), 0.6), walk, False)  # movement
    assert needs_replan(PlannerCommand(P.SLOW_WALK, (1, 0, 0), (0, 1, 0), 0.6), walk, False)  # facing
    idle = PlannerCommand(P.IDLE)
    assert needs_replan(idle, walk, False)  # mode
    assert not needs_replan(PlannerCommand(P.IDLE, speed=0.5), idle, True)  # static: speed and timer ignored
    assert needs_replan(PlannerCommand(P.IDLE, facing=(0, 1, 0)), idle, False)  # static: facing still counts
    assert not needs_replan(PlannerCommand(P.SLOW_WALK, (1, 0, 0), (1, 0, 0), 0.0),
                            PlannerCommand(P.SLOW_WALK, (1, 0, 0), (1, 0, 0), 0.0), True)  # zero speed: no timer
    t = ReplanTimer()
    # 10 x 0.1 sums to 0.9999999999999999 in floating point, so the 1 s timer fires on the 11th planner tick,
    # as in the C++ planner thread and the reference port
    assert [t.tick(P.SLOW_WALK) for _ in range(11)] == [False] * 10 + [True]
    assert all(t.tick(P.RUN) for _ in range(3))


def test_planner_runs_every_fifth_tick_and_on_command_change_only():
    c = _ctrl()
    c.planner_initialize(DEFAULT_ANGLES)
    pl = c.planner.session
    for _ in range(12):
        _stand(c)
    assert len(pl.feeds) == 1  # IDLE, unchanged: only the initial call
    c.set_command(PlannerCommand(P.IDLE, facing=(0.0, 1.0, 0.0)))
    _stand(c)  # tick 12: not a planner tick
    assert len(pl.feeds) == 1
    for _ in range(3):
        _stand(c)  # tick 15 is
    assert len(pl.feeds) == 2 and c.replans == 1
    f = pl.feeds[-1]
    assert f["facing_direction"].reshape(-1).tolist() == [0.0, 1.0, 0.0]
    # the context comes from the current motion, LOOK_AHEAD frames ahead, not from the robot
    ctx = f["context_mujoco_qpos"].reshape(4, 36)
    np.testing.assert_allclose(ctx[:, 7:], np.tile(DEFAULT_ANGLES, (4, 1)), atol=1e-6)
    assert ctx[0, 2] == pytest.approx(0.79, abs=1e-6)


def test_threaded_planner_never_blocks_a_tick_and_blends_the_late_result():
    enc, dec, pl = fake_sessions(delay_s=0.3)
    c = SonicController(enc, dec, pl, planner_threaded=True)
    pl.delay_s = 0.0
    c.planner_initialize(DEFAULT_ANGLES)  # synchronous
    pl.delay_s = 0.3
    for _ in range(5):
        _stand(c)
    c.set_command(PlannerCommand(P.IDLE, facing=(0.0, 1.0, 0.0)))
    t0 = time.perf_counter()
    _stand(c)  # tick 5: submits
    assert c.worker.busy and time.perf_counter() - t0 < 0.2
    before = c.motion
    for _ in range(4):
        _stand(c)
    assert c.motion is not before or c.worker.busy  # still busy (0.3 s) -> old motion kept
    c.worker.join()
    _stand(c)
    assert c.motion.quat[-1] @ yaw_quat(math.pi / 2) == pytest.approx(1.0, abs=1e-6)  # the new motion is in
    c.close()


# ---- start-up sequence and falls --------------------------------------------------------------------------------
def _run(runner, t, q=None, z=0.78, quat=(1.0, 0, 0, 0)):
    return runner.tick(t, DEFAULT_ANGLES if q is None else q, np.zeros(29), np.array(quat), np.zeros(3), z)


def test_startup_harness_ramp_release_settle_planner_idle_ready():
    c = _ctrl()
    r = SonicRunner(c, ramp_s=1.0, settle_s=0.5, idle_s=1.0)
    q0 = np.zeros(29)
    out = _run(r, 0.0, q=q0, z=0.2)  # harnessed: no fall check while pinned
    assert out.harness and out.phase == Phase.HARNESS and not out.targets.q.any()
    out = _run(r, 0.5, q=q0)
    np.testing.assert_allclose(out.targets.q, 0.5 * DEFAULT_ANGLES)
    assert out.targets.kp[4] == pytest.approx(P.KP[4] * 1.5)  # SONIC gains, v1.1 ankle tuning
    out = _run(r, 1.0)
    assert not out.harness and out.phase == Phase.SETTLE
    np.testing.assert_allclose(out.targets.q, DEFAULT_ANGLES)
    r.command(PlannerCommand(P.SLOW_WALK, (1, 0, 0), (1, 0, 0), 0.6))
    assert _run(r, 1.48).phase == Phase.SETTLE and not c.ready
    out = _run(r, 1.5)
    assert out.phase == Phase.IDLE and c.ready and c.cmd.mode == P.IDLE  # planner initialised, IDLE first
    assert len(c.planner.session.feeds) == 1
    assert _run(r, 2.48).phase == Phase.IDLE and c.cmd.mode == P.IDLE
    out = _run(r, 2.5)
    assert out.phase == Phase.READY and r.ready and c.cmd.mode == P.SLOW_WALK


def test_a_fall_ends_control_with_damping_only():
    c = _ctrl()
    r = SonicRunner(c, ramp_s=0.0, settle_s=0.0, idle_s=0.0)
    _run(r, 0.0)
    assert _run(r, 0.02).phase == Phase.READY
    out = _run(r, 0.04, z=0.40)
    assert out.phase == Phase.FALLEN and "pelvis" in out.fall and r.fell_at == 0.04
    assert not out.targets.kp.any() and out.targets.kd.any()
    assert _run(r, 0.06, z=0.8).phase == Phase.FALLEN  # stays fallen


def test_fall_thresholds():
    assert fall_reason(0.46, [1, 0, 0, 0]) is None
    assert "pelvis" in fall_reason(0.44, [1, 0, 0, 0])
    assert fall_reason(1.44, [1, 0, 0, 0], floor_z=1.0) is not None  # height above the floor
    tilt = lambda deg: [math.cos(math.radians(deg) / 2), math.sin(math.radians(deg) / 2), 0, 0]
    assert fall_reason(0.78, tilt(59)) is None
    assert "tilted 61" in fall_reason(0.78, tilt(61))


def test_no_simulator_or_onnxruntime_imported_by_the_package():
    code = ("import sys, wardzero_agents.isaac_host.sonic; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & "
            "{'onnxruntime', 'omni', 'isaacsim', 'pxr', 'mujoco', 'rclpy', 'torch'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]"
