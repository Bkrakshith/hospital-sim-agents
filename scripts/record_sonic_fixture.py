#!/usr/bin/env python3
"""Record a GEAR-SONIC tick log in MuJoCo (CPU) for tests/test_sonic_replay.py.

Runs the CPU reference controller (``sonic_cpu.py``, the port the S2 preparation validated in MuJoCo) on our G1 +
BrainCo, exactly as its ``sim2sim_cpu.py`` does (200 Hz physics, 50 Hz policy, PD every physics step), and saves
every tick's robot state, command, encoder token, decoder action and PD targets, the full encoder and decoder
inputs on a subset of ticks, and every planner call (feeds and output). The test replays the robot states through
``wardzero_agents.isaac_host.sonic`` and checks the observations and actions against this log.

Needs mujoco, onnxruntime and the weights. CPU only. Example:

  taskset -c 0-7,10-31 python scripts/record_sonic_fixture.py \\
      --reference ~/src/gear_sonic/scripts/sonic_cpu.py --weights ~/src/gear_sonic/weights
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = Path("~/ntu_dev/Hospital_G1/g1-brainco-description/g1_29dof_brainco/mjcf/scene_flat.xml")
DEFAULT_OUT = ROOT / "tests" / "data" / "sonic" / "replay_ours_walk_turn.npz"


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("sonic_cpu_reference", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Recorder:
    """Wraps an onnxruntime session; keeps each call's feeds and outputs."""

    def __init__(self, sess):
        self.sess = sess
        self.calls = []

    def get_inputs(self):
        return self.sess.get_inputs()

    def run(self, names, feeds):
        out = self.sess.run(names, feeds)
        self.calls.append(({k: np.array(v) for k, v in feeds.items()}, [np.array(o) for o in out]))
        return out


def schedule(t: float, S):
    """IDLE 0.5 s, SLOW_WALK 0.6 m/s forward to 2.0 s, then IDLE facing +y (an in-place left turn)."""
    if t < 0.5:
        return S.Command(S.IDLE)
    if t < 2.0:
        return S.Command(S.SLOW_WALK, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.6, -1.0)
    return S.Command(S.IDLE, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.0, -1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=Path("~/src/gear_sonic/scripts/sonic_cpu.py"))
    ap.add_argument("--weights", type=Path, default=Path("~/src/gear_sonic/weights"))
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    import mujoco
    S = load_reference(a.reference.expanduser())
    m = mujoco.MjModel.from_xml_path(str(a.mjcf.expanduser()))
    dt = 0.005
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in S.MJ_JOINTS]
    assert min(jid) >= 0, "joint missing"
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    act_of = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT}
    body_act = np.array([act_of[j] for j in jid])
    other_act = [i for i in range(m.nu) if i not in set(body_act.tolist())]
    lim = np.abs(m.actuator_ctrlrange[body_act, 1]).copy()

    mujoco.mj_resetData(m, d)
    d.qpos[qadr] = S.DEFAULT_ANGLES
    d.qpos[3:7] = [1, 0, 0, 0]
    lo, hi = 0.5, 1.1
    for _ in range(40):  # lowest pelvis height with the feet just touching the floor
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        pen = min([c.dist for c in d.contact[:d.ncon]
                   if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0] or [1.0])
        if pen < -1e-4:
            lo = z
        else:
            hi = z
    d.qpos[2] = hi + 0.002
    mujoco.mj_forward(m, d)

    ctrl = S.SonicController(str(a.weights.expanduser()), threads=a.threads, kp_kd_scale={4: 1.5, 10: 1.5})
    ctrl.enc, ctrl.dec, ctrl.planner = Recorder(ctrl.enc), Recorder(ctrl.dec), Recorder(ctrl.planner)

    def apply_pd(qt, kp, kd, steps):
        for _ in range(steps):
            tau = np.clip(kp * (qt - d.qpos[qadr]) - kd * d.qvel[vadr], -lim, lim)
            d.ctrl[body_act] = tau
            for i in other_act:
                d.ctrl[i] = 0.0  # BrainCo position actuators: fingers open
            mujoco.mj_step(m, d)

    apply_pd(S.DEFAULT_ANGLES, ctrl.kp, ctrl.kd, int(0.5 / dt))
    init_q = d.qpos[qadr].copy()
    ctrl.planner_initialize(init_q)
    ctrl._frame_advance(play=False)

    n_ticks = int(round(a.seconds / S.CONTROL_DT))
    rec = {k: [] for k in ("q", "dq", "quat", "gyro", "mode", "movement", "facing", "speed", "height",
                           "token", "action", "q_target", "pelvis", "planner_tick")}
    for tick in range(n_ticks):
        t = tick * S.CONTROL_DT
        ctrl.cmd = schedule(t, S)
        n_before = len(ctrl.planner.calls)
        if tick % 5 == 0 and tick > 0:
            ctrl.planner_tick()
        q, dq, quat, gyro = d.qpos[qadr].copy(), d.qvel[vadr].copy(), d.qpos[3:7].copy(), d.qvel[3:6].copy()
        qt, kp, kd = ctrl.step(q, dq, quat, gyro)
        c = ctrl.cmd
        for k, v in (("q", q), ("dq", dq), ("quat", quat), ("gyro", gyro), ("mode", c.mode),
                     ("movement", c.movement), ("facing", c.facing), ("speed", c.speed), ("height", c.height),
                     ("token", ctrl.enc.calls[-1][1][0].reshape(-1)), ("action", ctrl.dec.calls[-1][1][0].reshape(-1)),
                     ("q_target", qt), ("pelvis", d.qpos[:3].copy()),
                     ("planner_tick", len(ctrl.planner.calls) > n_before)):
            rec[k].append(np.array(v))
        apply_pd(qt, kp, kd, int(round(S.CONTROL_DT / dt)))
        up = S.qrot(d.qpos[3:7], [0, 0, 1.0])
        if d.qpos[2] < 0.45 or math.degrees(math.acos(np.clip(up[2], -1, 1))) > 60:
            raise SystemExit(f"fell at tick {tick}")

    obs_ticks = np.array(sorted(set(range(12)) | set(range(12, n_ticks, 3)) | {n_ticks - 1}))
    enc_in = np.stack([ctrl.enc.calls[i][0]["obs_dict"].reshape(-1) for i in obs_ticks])
    dec_in = np.stack([ctrl.dec.calls[i][0]["obs_dict"].reshape(-1) for i in obs_ticks])
    assert not enc_in[:, 644:].any(), "encoder inputs past 644 are zero in mode 0"
    pc = ctrl.planner.calls
    out = {k: np.stack(v) for k, v in rec.items()}
    out.update(
        init_q=init_q, kp=ctrl.kp, kd=ctrl.kd, obs_ticks=obs_ticks,
        enc_in=enc_in[:, :644], dec_in=dec_in,
        plan_context=np.stack([f["context_mujoco_qpos"].reshape(4, 36) for f, _ in pc]),
        plan_cmd=np.stack([np.concatenate([f["mode"].reshape(1).astype(float), f["movement_direction"].reshape(3),
                                           f["facing_direction"].reshape(3), f["target_vel"].reshape(1),
                                           f["height"].reshape(1)]) for f, _ in pc]),
        plan_qpos=np.stack([o[0].reshape(-1, 36) for _, o in pc]),
        plan_n=np.array([int(np.asarray(o[1]).reshape(-1)[0]) for _, o in pc]),
        plan_input_names=np.array([i.name for i in ctrl.planner.get_inputs()]),
        plan_input_shapes=np.array([str(list(i.shape)) for i in ctrl.planner.get_inputs()]),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"{n_ticks} ticks, {len(pc)} planner calls, final pelvis {d.qpos[:3].round(3).tolist()}, "
          f"yaw {math.degrees(S.yaw_of(d.qpos[3:7])):.1f} deg -> {a.out} ({a.out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
