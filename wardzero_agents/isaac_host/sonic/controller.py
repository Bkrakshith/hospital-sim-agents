"""The GEAR-SONIC whole-body controller (v1.1 encoder + decoder with the V2 planner), simulator-agnostic.

A port of ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref`` (NVlabs GR00T-WholeBodyControl, Apache-2.0) by way of
the CPU reference ``sonic_cpu.py``; it matches that reference bit for bit on recorded MuJoCo ticks
(tests/test_sonic_replay.py). The caller feeds the robot state every 50 Hz tick in SONIC's MuJoCo joint order
(``JointMap`` converts from a simulator's own order by name) and gets joint position targets and PD gains back.
It applies ``tau = kp (q* - q) - kd dq``, clipped to the effort limit, on every physics step (200 Hz).

Per tick, in the C++ order: (planner decision every 5th tick) -> log state -> heading re-init -> encoder ->
decoder -> PD targets -> frame advance (installs or blends a finished planner motion).

Modified port of NVIDIA code (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), Apache License 2.0.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .motion import Motion, blend, context_from_motion, initial_context
from .params import (ACTION_SCALE, DECODER_DIM, DEFAULT_ANGLES, DEFAULT_HEIGHT, ENCODER_DIM, HISTORY,
                     ISAACLAB_TO_MUJOCO, KD, KP, LOOK_AHEAD, MUJOCO_TO_ISAACLAB, N_JOINTS, PLANNER_EVERY,
                     REF_FRAMES, REF_STEP, V11_GAIN_SCALE)
from .planner import PlannerCommand, PlannerModel, PlannerWorker, ReplanTimer, needs_replan
from .quat import heading_quat, heading_quat_inv, qconj, qmat, qmul, qrot, yaw_of


@dataclass
class LogEntry:
    """One tick of robot state as the deploy code logs it (``GatherRobotStateToLogger``)."""
    base_quat: np.ndarray  # pelvis IMU, wxyz
    base_ang_vel: np.ndarray  # pelvis gyro, body frame
    body_q: np.ndarray  # IsaacLab order, minus SONIC's default pose
    body_dq: np.ndarray  # IsaacLab order
    last_action: np.ndarray  # IsaacLab order, the raw decoder output of the previous tick


_ZERO_ENTRY = LogEntry(np.zeros(4), np.zeros(3), np.zeros(N_JOINTS), np.zeros(N_JOINTS), np.zeros(N_JOINTS))


@dataclass
class Targets:
    """One tick's output, all in SONIC's MuJoCo joint order. dq* = 0 and no feed-forward."""
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action: np.ndarray = field(repr=False)  # raw decoder output, IsaacLab order


class SonicController:
    """Encoder/decoder observation assembly, history buffers, planner context/blending/replan and PD targets.

    ``encoder``, ``decoder`` and ``planner`` are onnxruntime-like sessions (``run(None, feeds)``); use
    ``SonicController.from_weights`` for the real ones. ``planner_threaded=True`` runs planner calls in a worker
    thread as the deploy code does (a 50 Hz tick never waits; a late result is blended in when it arrives);
    False runs them inline, which is deterministic (tests, and sims running slower than real time)."""

    def __init__(self, encoder, decoder, planner, *, gain_scale: Optional[dict[int, float]] = None,
                 planner_threaded: bool = False) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.planner = PlannerModel(planner)
        self.worker = PlannerWorker(self.planner) if planner_threaded else None
        self.kp = KP.copy()
        self.kd = KD.copy()
        for idx, s in (V11_GAIN_SCALE if gain_scale is None else gain_scale).items():
            self.kp[idx] *= s
            self.kd[idx] *= s
        self.reset()

    @classmethod
    def from_weights(cls, weights_dir=None, *, threads: int = 4, **kw) -> "SonicController":
        """Load the three ONNX models (onnxruntime, CPU) from the weights cache (see ``weights.resolve_dir``)."""
        from .weights import load_sessions
        enc, dec, plan = load_sessions(weights_dir, threads=threads)
        return cls(enc, dec, plan, **kw)

    def reset(self) -> None:
        self.history: deque[LogEntry] = deque(maxlen=HISTORY)
        self.last_action = np.zeros(N_JOINTS)
        self.motion: Optional[Motion] = None
        self.frame = 0
        self.pending: Optional[tuple[Motion, int]] = None  # (new 50 Hz motion, gen_frame)
        self.first_motion = True
        self.reinit_heading = False
        self.init_base_quat = np.array([1.0, 0, 0, 0])
        self.init_ref_quat = np.array([1.0, 0, 0, 0])
        self.cmd = PlannerCommand()
        self.last_cmd = PlannerCommand()
        self.timer = ReplanTimer()
        self.tick_count = 0
        self.replans = 0
        self.timing: dict[str, list[float]] = {"encoder": [], "decoder": [], "tick": []}

    # ------------------------------------------------------------------ planner
    @property
    def ready(self) -> bool:
        """A planner motion is installed, so ``step`` can run."""
        return self.motion is not None

    def planner_initialize(self, joint_pos_mj, height: float = DEFAULT_HEIGHT) -> None:
        """First planner call (synchronous): IDLE from the measured joints, then install the motion."""
        m = self.planner(initial_context(np.asarray(joint_pos_mj, float), height), PlannerCommand())
        self.pending = (m, 0)
        self.first_motion = True
        self.timer = ReplanTimer()
        self.tick_count = 0
        self._frame_advance(play=False)  # the control thread picks up the first motion without playing it

    def set_command(self, cmd: PlannerCommand) -> None:
        """Takes effect at the next planner tick (every 5th control tick), as the deploy code's input thread."""
        self.cmd = cmd

    def planner_tick(self) -> bool:
        """The 10 Hz planner decision. True when a replan was started."""
        timer = self.timer.tick(self.cmd.mode)
        if self.motion is None or not needs_replan(self.cmd, self.last_cmd, timer):
            return False
        if self.worker is not None and self.worker.busy:
            return False  # decide again on the next planner tick
        c = self.cmd
        gen_frame = self.frame + LOOK_AHEAD
        ctx = context_from_motion(self.motion, gen_frame)
        if self.worker is not None:
            self.worker.submit(ctx, c, gen_frame)
        else:
            self.pending = (self.planner(ctx, c), gen_frame)
        self.last_cmd = c
        self.replans += 1
        return True

    def _frame_advance(self, play: bool = True) -> None:
        if self.worker is not None:
            done = self.worker.poll()
            if done is not None:
                self.pending = done
        if self.pending is not None:
            gen, fgen = self.pending
            self.pending = None
            if self.motion is None or self.first_motion:
                self.motion = gen.copy()
                self.reinit_heading = True
                self.first_motion = False
                self.frame = 0
            else:
                blended = blend(self.motion, self.frame, gen, fgen)
                if blended is not None:
                    self.motion = blended
                    self.frame = 0
        if play and self.motion is not None and self.motion.T > 0:
            self.frame = min(self.frame + 1, self.motion.T - 1)

    # ------------------------------------------------------------------ frames
    def apply_delta_heading(self) -> np.ndarray:
        """Planner frame -> world: heading(init base) * heading^-1(init reference root)."""
        return qmul(heading_quat(self.init_base_quat), heading_quat_inv(self.init_ref_quat))

    def planner_yaw_offset(self) -> float:
        """World yaw minus planner-frame yaw (rad)."""
        return yaw_of(self.apply_delta_heading())

    def reference_yaw(self) -> float:
        """World yaw (rad) of the reference root at the current frame: where the policy is steering the pelvis."""
        if self.motion is None:
            return 0.0
        return yaw_of(qmul(self.apply_delta_heading(), self.motion.quat[self.frame]))

    def command_from_world(self, mode: int, speed: float = -1.0, move_yaw: Optional[float] = None,
                           face_yaw: Optional[float] = None, height: float = -1.0,
                           quantum_rad: float = 0.0) -> PlannerCommand:
        """A planner command from world-frame yaw angles (rad). move_yaw None = no movement; face_yaw None =
        keep the current reference heading. quantum_rad > 0 snaps both angles to a grid, so a slowly drifting
        target does not re-plan on every planner tick (any facing or movement change re-plans)."""
        off = self.planner_yaw_offset()

        def snap(a: float) -> float:
            a = math.atan2(math.sin(a), math.cos(a))
            return round(a / quantum_rad) * quantum_rad if quantum_rad > 0 else a

        def unit(a: float) -> tuple[float, float, float]:
            return (float(math.cos(a)), float(math.sin(a)), 0.0)

        if face_yaw is None:
            face = tuple(self.cmd.facing)
        else:
            face = unit(snap(face_yaw - off))
        move = (0.0, 0.0, 0.0) if move_yaw is None else unit(snap(move_yaw - off))
        return PlannerCommand(int(mode), move, face, float(speed), float(height))

    # ------------------------------------------------------------------ observations
    def _hist(self, n: int) -> list[LogEntry]:
        h = list(self.history)[-n:]
        if len(h) < n:
            h = [_ZERO_ENTRY] * (n - len(h)) + h
        return h  # oldest first, zero-padded at the old end

    def _motion_frames(self) -> list[int]:
        return [min(self.frame + k * REF_STEP, self.motion.T - 1) for k in range(REF_FRAMES)]

    def encoder_obs(self) -> np.ndarray:
        """1751 floats: mode (4, zero = "g1"), reference joints (290) and velocities (290), 10 x 6-D anchor
        orientation (60), then teleop/SMPL inputs that are zero in mode 0."""
        o = np.zeros(ENCODER_DIM, np.float32)
        frames = self._motion_frames()
        o[4:294] = np.concatenate([self.motion.jp[f] for f in frames])
        o[294:584] = np.concatenate([self.motion.jv[f] for f in frames])
        left = heading_quat(self.history[-1].base_quat)
        adh = self.apply_delta_heading()
        six = []
        for f in frames:
            r = qmat(qmul(qconj(left), qmul(adh, self.motion.quat[f])))
            six.append(r[:, :2].reshape(-1))  # first two columns, row-wise: R00 R01 R10 R11 R20 R21
        o[584:644] = np.concatenate(six)
        return o

    def decoder_obs(self, token: np.ndarray) -> np.ndarray:
        """994 floats: token (64), then 10 frames (oldest first) of gyro, body_q, body_dq, last action, gravity."""
        h = self._hist(HISTORY)
        grav = [qrot(qconj(e.base_quat), [0, 0, -1.0]) if np.any(e.base_quat) else np.zeros(3) for e in h]
        out = np.concatenate([token,
                              np.concatenate([e.base_ang_vel for e in h]),
                              np.concatenate([e.body_q for e in h]),
                              np.concatenate([e.body_dq for e in h]),
                              np.concatenate([e.last_action for e in h]),
                              np.concatenate(grav)]).astype(np.float32)
        assert out.shape == (DECODER_DIM,)
        return out

    # ------------------------------------------------------------------ one 50 Hz tick
    def step(self, q_mj, dq_mj, base_quat_wxyz, gyro_body, play: bool = True) -> Targets:
        """One 50 Hz tick. q_mj/dq_mj: the 29 body joints in MuJoCo order; base_quat_wxyz: the pelvis
        orientation; gyro_body: the pelvis angular velocity in the pelvis frame."""
        if self.motion is None:
            raise RuntimeError("call planner_initialize() before step()")
        t_tick = time.perf_counter()
        if self.tick_count % PLANNER_EVERY == 0 and self.tick_count > 0:
            self.planner_tick()
        q_mj = np.asarray(q_mj, float)
        dq_mj = np.asarray(dq_mj, float)
        self.history.append(LogEntry(np.asarray(base_quat_wxyz, float), np.asarray(gyro_body, float),
                                     q_mj[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB],
                                     dq_mj[MUJOCO_TO_ISAACLAB], self.last_action.copy()))
        if self.reinit_heading:
            self.init_base_quat = self.history[-1].base_quat.copy()
            self.init_ref_quat = self.motion.quat[self.frame].copy()
            self.reinit_heading = False
        if self.frame == 0:
            self.init_ref_quat = self.motion.quat[0].copy()
        t0 = time.perf_counter()
        token = np.asarray(self.encoder.run(None, {"obs_dict": self.encoder_obs()[None]})[0]).reshape(-1)
        t1 = time.perf_counter()
        action = np.asarray(self.decoder.run(None, {"obs_dict": self.decoder_obs(token)[None]})[0]
                            ).reshape(-1).astype(float)
        t2 = time.perf_counter()
        q_target = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        self.last_action = action
        self._frame_advance(play)
        self.tick_count += 1
        self.timing["encoder"].append((t1 - t0) * 1e3)
        self.timing["decoder"].append((t2 - t1) * 1e3)
        self.timing["tick"].append((time.perf_counter() - t_tick) * 1e3)
        for v in self.timing.values():
            if len(v) > 4096:
                del v[:2048]
        return Targets(q_target, self.kp, self.kd, action)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.join(timeout=2.0)


def default_pose_targets(kp_scale: Optional[dict[int, float]] = None) -> Targets:
    """Hold SONIC's default pose with SONIC's gains (start-up ramp end point and settle)."""
    kp, kd = KP.copy(), KD.copy()
    for idx, s in (V11_GAIN_SCALE if kp_scale is None else kp_scale).items():
        kp[idx] *= s
        kd[idx] *= s
    return Targets(DEFAULT_ANGLES.copy(), kp, kd, np.zeros(N_JOINTS))

