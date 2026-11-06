"""Stand-ins for the GEAR-SONIC onnxruntime sessions, so the controller is tested without the weights."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wardzero_agents.isaac_host.sonic import weights as W
from wardzero_agents.isaac_host.sonic.params import DEFAULT_ANGLES

FIXTURE = Path(__file__).parent / "data" / "sonic" / "replay_ours_walk_turn.npz"
V2_INPUTS = [("context_mujoco_qpos", [1, 4, 36]), ("target_vel", [1]), ("mode", [1]), ("movement_direction", [1, 3]),
             ("facing_direction", [1, 3]), ("random_seed", [1]), ("has_specific_target", [1, 1]),
             ("specific_target_positions", [1, 4, 3]), ("specific_target_headings", [1, 4]),
             ("allowed_pred_num_tokens", [1, 11]), ("height", [1])]


@dataclass
class In:
    name: str
    shape: list


class Session:
    """Returns the queued outputs in order (or ``fn(feeds)``) and keeps every call's feeds."""

    def __init__(self, outputs=None, fn=None, inputs=None, delay_s: float = 0.0):
        self._outs = iter(outputs) if outputs is not None else None
        self._fn = fn
        self._inputs = [In(n, s) for n, s in (inputs or [("obs_dict", [1, 0])])]
        self.feeds: list[dict] = []
        self.delay_s = delay_s

    def get_inputs(self):
        return self._inputs

    def run(self, names, feeds):
        self.feeds.append({k: np.array(v) for k, v in feeds.items()})
        if self.delay_s:
            time.sleep(self.delay_s)
        return self._fn(feeds) if self._fn is not None else next(self._outs)


def standing_qpos(n: int = 30, yaw: float = 0.0, vx: float = 0.0) -> np.ndarray:
    """A planner output: n frames at 30 Hz standing in SONIC's default pose, facing yaw, moving at vx along x."""
    q = np.zeros((64, 36), np.float32)
    q[:, 0] = vx * np.arange(64) / 30.0
    q[:, 2] = 0.79
    q[:, 3] = np.cos(yaw / 2)
    q[:, 6] = np.sin(yaw / 2)
    q[:, 7:] = DEFAULT_ANGLES
    return q


def fake_sessions(planner_fn=None, delay_s: float = 0.0):
    """(encoder, decoder, planner): zero token, zero action, a standing planner motion facing the command."""
    enc = Session(fn=lambda f: [np.zeros((1, 64), np.float32)])
    dec = Session(fn=lambda f: [np.zeros((1, 29), np.float32)])

    def plan(feeds):
        face = np.asarray(feeds["facing_direction"]).reshape(-1)
        return [standing_qpos(yaw=float(np.arctan2(face[1], face[0])))[None], np.array([30], np.int32)]

    pl = Session(fn=planner_fn or plan, inputs=V2_INPUTS, delay_s=delay_s)
    return enc, dec, pl


def real_weights_dir():
    """The first folder with the real weights: $WARDZERO_SONIC_DIR, the cache, or the S2 preparation's folder."""
    return W.first_present([W.resolve_dir(), Path("~/src/gear_sonic/weights").expanduser()])
