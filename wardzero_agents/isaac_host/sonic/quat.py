"""Quaternion helpers, wxyz, with the semantics of the SONIC deploy code's ``math_utils.hpp``.

Kept as plain float64 numpy so the observations match the reference port bit for bit.

Modified port of NVIDIA code (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), Apache License 2.0.
"""
from __future__ import annotations

import math

import numpy as np


def qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v) -> np.ndarray:
    """Rotate vector v by unit quaternion q."""
    return qmul(qmul(q, np.array([0.0, *v])), qconj(q))[1:]


def qunit(q) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def heading_quat(q) -> np.ndarray:
    """``calc_heading_quat_d``: normalise (w, 0, 0, z). Not an exact yaw extraction, by design."""
    return qunit(np.array([q[0], 0.0, 0.0, q[3]]))


def heading_quat_inv(q) -> np.ndarray:
    h = heading_quat(q)
    return np.array([h[0], 0.0, 0.0, -h[3]])


def qmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, t: float) -> np.ndarray:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return qunit(q0 + t * (q1 - q0))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def yaw_of(q) -> float:
    """Exact yaw (rad) of a wxyz quaternion."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def tilt_deg(q) -> float:
    """Angle (degrees) between the body z axis and world up."""
    up = qrot(q, [0.0, 0.0, 1.0])
    return math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))


def xyzw_to_wxyz(q) -> np.ndarray:
    """PhysX / Isaac tensor API order (x, y, z, w) to the wxyz order SONIC uses."""
    return np.array([q[3], q[0], q[1], q[2]], float)
