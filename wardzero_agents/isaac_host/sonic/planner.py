"""The SONIC kinematic planner: commands, the ONNX call, the 10 Hz replan rules and a worker thread.

Ported from ``Planner()`` in ``g1_deploy_onnx_ref.cpp`` and ``localmotion_kplanner*.hpp``. Directions are unit
vectors in the *planner frame* (x = the robot's heading when the planner started); ``SonicController`` converts
world-frame yaw angles for you (``command_from_world``).

Modified port of NVIDIA code (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), Apache License 2.0.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .motion import Motion, resample_50hz
from .params import IDLE, PLANNER_DT, RANDOM_SEED, RUN, STATIC_MODES


@dataclass(frozen=True)
class PlannerCommand:
    """mode: IDLE 0, SLOW_WALK 1 (0.2-0.8 m/s), WALK 2, RUN 3, ... ; speed -1 = the mode's default;
    height -1 for standing modes."""
    mode: int = IDLE
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (1.0, 0.0, 0.0)
    speed: float = -1.0
    height: float = -1.0


def replan_interval(mode: int) -> float:
    """Periodic replan timer: 0.1 s running, 0.2 s crawling (mode 8), 1 s otherwise."""
    return 0.1 if mode == RUN else (0.2 if mode == 8 else 1.0)


def needs_replan(cmd: PlannerCommand, last: PlannerCommand, timer_expired: bool) -> bool:
    """Always on a change of mode, facing or height; in non-static modes also on a change of speed or movement,
    and when the periodic timer expired with a non-zero speed."""
    if cmd.mode != last.mode or tuple(cmd.facing) != tuple(last.facing) or cmd.height != last.height:
        return True
    if cmd.mode in STATIC_MODES:
        return False
    return (cmd.speed != last.speed or tuple(cmd.movement) != tuple(last.movement)
            or (timer_expired and cmd.speed != 0))


class ReplanTimer:
    """The C++ planner thread's counter: += PLANNER_DT per planner tick; expires (and resets) at the interval."""

    def __init__(self) -> None:
        self.counter = 0.0

    def tick(self, mode: int) -> bool:
        self.counter += PLANNER_DT
        if self.counter >= replan_interval(mode):
            self.counter = 0.0
            return True
        return False


class PlannerModel:
    """The V2 planner ONNX session behind a plain call: context [4, 36] + command -> 50 Hz Motion.

    ``session`` is anything with onnxruntime's ``get_inputs()`` / ``run(None, feeds)``; tests pass a fake."""

    def __init__(self, session) -> None:
        self.session = session
        self.inputs = {i.name: i for i in session.get_inputs()}
        self.last_ms = 0.0

    def _shape(self, name: str, default):
        shp = getattr(self.inputs[name], "shape", None)
        return tuple(int(s) if isinstance(s, (int, np.integer)) else 1 for s in shp) if shp else default

    def feeds(self, context: np.ndarray, cmd: PlannerCommand) -> dict[str, np.ndarray]:
        f = {
            "context_mujoco_qpos": context.reshape(self._shape("context_mujoco_qpos", (1, 4, 36))).astype(np.float32),
            "target_vel": np.array([cmd.speed], np.float32).reshape(self._shape("target_vel", (1,))),
            "mode": np.array([cmd.mode], np.int64).reshape(self._shape("mode", (1,))),
            "movement_direction": np.array(cmd.movement, np.float32).reshape(self._shape("movement_direction", (1, 3))),
            "facing_direction": np.array(cmd.facing, np.float32).reshape(self._shape("facing_direction", (1, 3))),
            "random_seed": np.array([RANDOM_SEED], np.int64).reshape(self._shape("random_seed", (1,))),
        }
        if "height" in self.inputs:  # V2
            f["height"] = np.array([cmd.height], np.float32).reshape(self._shape("height", (1,)))
            f["has_specific_target"] = np.zeros(self._shape("has_specific_target", (1, 1)), np.int64)
            f["specific_target_positions"] = np.zeros(self._shape("specific_target_positions", (1, 4, 3)), np.float32)
            f["specific_target_headings"] = np.zeros(self._shape("specific_target_headings", (1, 4)), np.float32)
            shape = self._shape("allowed_pred_num_tokens", (1, 11))
            a = np.zeros(shape, np.int64).reshape(-1)
            a[:6] = 1  # C++ default: 6..11 tokens (24..44 frames at 30 Hz)
            f["allowed_pred_num_tokens"] = a.reshape(shape)
        return f

    def __call__(self, context: np.ndarray, cmd: PlannerCommand) -> Motion:
        t0 = time.perf_counter()
        qpos, n = self.session.run(None, self.feeds(context, cmd))
        self.last_ms = (time.perf_counter() - t0) * 1e3
        n = int(np.asarray(n).reshape(-1)[0])
        return resample_50hz(np.asarray(qpos).reshape(-1, 36)[:n])


class PlannerWorker:
    """Runs planner calls off the control thread, one at a time (the C++ planner thread).

    ``submit`` returns False while a call is still running, so the caller re-decides on its next planner tick.
    ``poll`` hands back the finished (motion, gen_frame) once, or None. A failed call is kept in ``error``."""

    def __init__(self, fn: Callable[[np.ndarray, PlannerCommand], Motion]) -> None:
        self.fn = fn
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[tuple[Motion, int]] = None
        self.error: Optional[BaseException] = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, context: np.ndarray, cmd: PlannerCommand, gen_frame: int) -> bool:
        if self.busy:
            return False

        def run() -> None:
            try:
                m = self.fn(context, cmd)
            except Exception as exc:  # reported through .error, the control loop keeps the old motion
                with self._lock:
                    self.error = exc
                return
            with self._lock:
                self._result = (m, gen_frame)

        self._thread = threading.Thread(target=run, name="sonic-planner", daemon=True)
        self._thread.start()
        return True

    def poll(self) -> Optional[tuple[Motion, int]]:
        with self._lock:
            r, self._result = self._result, None
        return r

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
