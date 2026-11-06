"""Joint maps by name between a simulator's joint list and SONIC's MuJoCo order. Never slice joints by position.

A PhysX articulation lists DOFs in tree-traversal order, and with the BrainCo hands the left fingers sit between
the two arms, so the 29 body joints are found by name. Extra joints (fingers) are not SONIC's: the simulator holds
them with its own drives.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .params import MJ_JOINTS, N_JOINTS


def _variants(name: str) -> tuple[str, ...]:
    base = name[:-6] if name.endswith("_joint") else name
    return (name, base, base + "_joint")


class JointMap:
    """``sim_index[i]`` is the simulator index of SONIC (MuJoCo-order) joint i.

    Names may carry a path prefix (``/World/G1/joints/left_knee_joint``) and may omit the ``_joint`` suffix."""

    def __init__(self, sim_names: Sequence[str]) -> None:
        self.sim_names = list(sim_names)
        index: dict[str, int] = {}
        for i, n in enumerate(self.sim_names):
            short = str(n).rsplit("/", 1)[-1]
            for v in _variants(short):
                if v in index and index[v] != i:
                    raise ValueError(f"joint name {v!r} appears twice in the simulator's joint list")
                index[v] = i
        missing = [n for n in MJ_JOINTS if n not in index and n[:-6] not in index]
        if missing:
            raise ValueError(f"the simulator has no joint named {', '.join(missing)}")
        self.sim_index = np.array([index.get(n, index.get(n[:-6])) for n in MJ_JOINTS], dtype=int)
        self.n_sim = len(self.sim_names)
        self.other = np.array(sorted(set(range(self.n_sim)) - set(self.sim_index.tolist())), dtype=int)

    def to_sonic(self, sim_values: Iterable[float]) -> np.ndarray:
        """Simulator-order array (length n_sim) -> the 29 SONIC joints, MuJoCo order."""
        a = np.asarray(sim_values, float)
        if a.shape[-1] != self.n_sim:
            raise ValueError(f"expected {self.n_sim} simulator joints, got {a.shape[-1]}")
        return a[..., self.sim_index]

    def to_sim(self, sonic_values: Iterable[float], fill: float = np.nan, out=None) -> np.ndarray:
        """29 MuJoCo-order values -> a simulator-order array; joints SONIC does not drive get ``fill``
        (or keep what ``out`` holds)."""
        v = np.asarray(sonic_values, float)
        if v.shape[-1] != N_JOINTS:
            raise ValueError(f"expected {N_JOINTS} SONIC joints, got {v.shape[-1]}")
        res = np.full(v.shape[:-1] + (self.n_sim,), fill, float) if out is None else np.array(out, float)
        res[..., self.sim_index] = v
        return res


def pd_torque(q_target, q, dq, kp, kd, effort) -> np.ndarray:
    """``tau = kp (q* - q) - kd dq`` clipped to +-effort: apply on every physics step (200 Hz), not 50 Hz."""
    effort = np.asarray(effort, float)
    tau = np.asarray(kp) * (np.asarray(q_target) - np.asarray(q)) - np.asarray(kd) * np.asarray(dq)
    return np.clip(tau, -effort, effort)
