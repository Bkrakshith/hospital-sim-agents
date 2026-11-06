"""Replay a MuJoCo tick log of the CPU reference controller through the ported GEAR-SONIC controller.

The log (scripts/record_sonic_fixture.py) is our G1 + BrainCo in MuJoCo: 0.5 s IDLE, slow walk at 0.6 m/s to 2.0 s,
then an in-place turn to +y. The robot states are fed back open loop, tick by tick.

- Without the weights: the recorded tokens, actions and planner outputs stand in for the networks, and the
  observations, planner contexts and PD targets must match the reference bit for bit.
- With the weights and onnxruntime: the real networks run, and tokens, actions, planner outputs and PD targets
  must match within 1e-5.
"""
from __future__ import annotations

import ast

import numpy as np
import pytest

from tests.sonic_fakes import FIXTURE, Session, real_weights_dir
from wardzero_agents.isaac_host.sonic import PlannerCommand, SonicController
from wardzero_agents.isaac_host.sonic.weights import onnxruntime_available


@pytest.fixture(scope="module")
def log():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _cmd(log, i) -> PlannerCommand:
    return PlannerCommand(int(log["mode"][i]), tuple(float(v) for v in log["movement"][i]),
                          tuple(float(v) for v in log["facing"][i]), float(log["speed"][i]), float(log["height"][i]))


def _replay(ctrl: SonicController, log):
    ctrl.planner_initialize(log["init_q"])
    out = []
    for i in range(len(log["q"])):
        ctrl.set_command(_cmd(log, i))
        n = ctrl.replans
        t = ctrl.step(log["q"][i], log["dq"][i], log["quat"][i], log["gyro"][i])
        out.append((t, ctrl.replans > n))
    return out


def _planner_inputs(log):
    return list(zip(log["plan_input_names"].tolist(), [ast.literal_eval(s) for s in log["plan_input_shapes"].tolist()]))


def test_observations_and_targets_match_the_reference_bit_for_bit(log):
    n = len(log["q"])
    enc = Session([[t[None]] for t in log["token"]])
    dec = Session([[a[None]] for a in log["action"]])
    plan = Session([[q[None], np.array([k], np.int32)] for q, k in zip(log["plan_qpos"], log["plan_n"])],
                   inputs=_planner_inputs(log))
    ctrl = SonicController(enc, dec, plan, gain_scale={4: 1.5, 10: 1.5})
    out = _replay(ctrl, log)

    assert len(enc.feeds) == len(dec.feeds) == n
    ticks = log["obs_ticks"]
    got_enc = np.stack([enc.feeds[i]["obs_dict"].reshape(-1) for i in ticks])
    got_dec = np.stack([dec.feeds[i]["obs_dict"].reshape(-1) for i in ticks])
    assert got_enc.dtype == np.float32 and got_enc.shape[1] == 1751 and got_dec.shape[1] == 994
    np.testing.assert_array_equal(got_enc[:, :644], log["enc_in"])
    assert not got_enc[:, 644:].any()  # teleop/SMPL inputs are zero in encoder mode 0
    np.testing.assert_array_equal(got_dec, log["dec_in"])
    np.testing.assert_array_equal(np.stack([t.q for t, _ in out]), log["q_target"])
    np.testing.assert_array_equal(ctrl.kp, log["kp"])
    np.testing.assert_array_equal(ctrl.kd, log["kd"])

    # planner: same calls on the same ticks, same contexts and commands
    assert [r for _, r in out] == log["planner_tick"].tolist()
    assert len(plan.feeds) == len(log["plan_qpos"]) == 4
    for f, ctx, c in zip(plan.feeds, log["plan_context"], log["plan_cmd"]):
        np.testing.assert_array_equal(f["context_mujoco_qpos"].reshape(4, 36), ctx)
        got = np.concatenate([f["mode"].reshape(1).astype(float), f["movement_direction"].reshape(3),
                              f["facing_direction"].reshape(3), f["target_vel"].reshape(1), f["height"].reshape(1)])
        np.testing.assert_array_equal(got, c)
        assert f["allowed_pred_num_tokens"].reshape(-1).tolist() == [1] * 6 + [0] * 5
        assert int(f["random_seed"].reshape(-1)[0]) == 1234


@pytest.mark.skipif(not onnxruntime_available(), reason="onnxruntime is not installed")
@pytest.mark.skipif(real_weights_dir() is None, reason="GEAR-SONIC weights are not fetched (scripts/fetch_sonic.py)")
def test_the_real_networks_reproduce_the_recorded_actions(log):
    ctrl = SonicController.from_weights(real_weights_dir(), threads=4, gain_scale={4: 1.5, 10: 1.5})
    rec_plan = []
    run = ctrl.planner.session.run

    def spy(names, feeds):
        out = run(names, feeds)
        rec_plan.append(out)
        return out

    ctrl.planner.session = type("S", (), {"run": staticmethod(spy), "get_inputs": ctrl.planner.session.get_inputs})()
    tokens = []
    enc_run = ctrl.encoder.run

    class Enc:
        def run(self, names, feeds):
            out = enc_run(names, feeds)
            tokens.append(np.asarray(out[0]).reshape(-1))
            return out

    ctrl.encoder = Enc()
    out = _replay(ctrl, log)
    np.testing.assert_allclose(np.stack(tokens), log["token"], atol=1e-5)
    np.testing.assert_allclose(np.stack([t.action for t, _ in out]), log["action"], atol=1e-5)
    np.testing.assert_allclose(np.stack([t.q for t, _ in out]), log["q_target"], atol=1e-5)
    assert len(rec_plan) == len(log["plan_qpos"])
    for (q, k), want_q, want_n in zip(rec_plan, log["plan_qpos"], log["plan_n"]):
        assert int(np.asarray(k).reshape(-1)[0]) == int(want_n)
        np.testing.assert_allclose(np.asarray(q).reshape(-1, 36)[:want_n], want_q[:want_n], atol=1e-5)
