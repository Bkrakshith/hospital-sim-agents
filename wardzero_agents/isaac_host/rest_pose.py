"""A patient's resting pose in bed, from a character skeleton's rest transforms (numpy only, unit-tested).

The Isaac People characters are in a T-pose at rest. A patient lying in a bed (``people_isaac.LyingProp``) is drawn
without the Behavior System, so the pose is the skeleton's own: the arms stick straight out. ``resting`` returns new
local rest transforms in which the upper arms hang beside the body and, for a patient who is propped up, the spine is
bent forward about the hips. The caller writes them as a one-frame SkelAnimation bound to the skeleton.

Matrices follow USD: 4x4, row vectors (a point is ``p @ M``), the translation in the last row, and a joint's world
matrix is ``local @ parent_world``. Two skeleton families exist (the generic ``L_Upperarm`` / ``Spine01`` one and the
DH ``LeftArm`` / ``Spine1`` one); joints are found by the last name of their path.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

LEFT_ARM = ("L_Upperarm", "LeftArm")
RIGHT_ARM = ("R_Upperarm", "RightArm")
LEFT_FORE = ("L_Forearm", "LeftForeArm")
RIGHT_FORE = ("R_Forearm", "RightForeArm")
HIPS = ("Hip", "Hips")
SPINE = ("Spine01", "Spine1")
HEAD = ("Head",)
LEFT_FOOT = ("L_Foot", "LeftFoot")
LEFT_TOE = ("L_ToeBase", "LeftToeBase")
ARM_OUT = 0.25  # how far an arm hangs away from the body (a fraction of its length, as a direction mix)


def find(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for n in names:
        if n.rsplit("/", 1)[-1] in candidates:
            return n
    return None


def _parent(n: str) -> Optional[str]:
    return n.rsplit("/", 1)[0] if "/" in n else None


def _worlds(names, local) -> dict[str, np.ndarray]:
    w: dict[str, np.ndarray] = {}
    for n, m in zip(names, local):
        p = _parent(n)
        w[n] = m @ (w[p] if p in w else np.eye(4))
    return w


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _rotation_between(a, b) -> np.ndarray:
    """A row-vector 3x3 rotation R with ``a @ R == b`` (the smallest turn from direction a to direction b)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    col = np.eye(3) + k + k @ k * (1.0 / (1.0 + c))
    return col.T


def _about(axis, deg) -> np.ndarray:
    """A row-vector 3x3 rotation about a unit axis."""
    x, y, z = _unit(axis)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    col = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    return col.T


def _turn(names, local, joint: str, rot3: np.ndarray) -> np.ndarray:
    """Rotate a joint (and so everything under it) by a world-space rotation about the joint's own position."""
    local = local.copy()
    i = names.index(joint)
    w = _worlds(names, local)
    new_w = w[joint].copy()
    new_w[:3, :3] = w[joint][:3, :3] @ rot3
    p = _parent(joint)
    parent_w = w[p] if p in w else np.eye(4)
    local[i] = new_w @ np.linalg.inv(parent_w)
    return local


def resting(names: Sequence[str], rest: np.ndarray, sit_up_deg: float = 0.0) -> np.ndarray:
    """New local rest transforms: arms down beside the body; the spine bent forward ``sit_up_deg`` when propped up."""
    names = list(names)
    local = np.array(rest, dtype=float)
    hips, head = find(names, HIPS), find(names, HEAD)
    la, lf = find(names, LEFT_ARM), find(names, LEFT_FORE)
    ra, rf = find(names, RIGHT_ARM), find(names, RIGHT_FORE)
    spine = find(names, SPINE)
    foot, toe = find(names, LEFT_FOOT), find(names, LEFT_TOE)
    if None in (hips, head, la, lf, ra, rf):
        return local  # not a skeleton this knows: left as it is
    w = _worlds(names, local)
    pos = {n: m[3, :3] for n, m in w.items()}
    up = _unit(pos[head] - pos[hips])
    lateral = _unit(pos[la] - pos[ra])
    front = _unit(pos[toe] - pos[foot]) if foot and toe else _unit(np.cross(lateral, up))
    if sit_up_deg and spine:
        for sign in (1.0, -1.0):  # the turn that takes the torso towards the front
            r = _about(lateral, sign * sit_up_deg)
            if float(np.dot(up @ r, front)) > 0:
                break
        local = _turn(names, local, spine, r)
    w = _worlds(names, local)
    pos = {n: m[3, :3] for n, m in w.items()}
    down = -_unit(pos[head] - pos[hips])
    for arm, fore, side in ((la, lf, 1.0), (ra, rf, -1.0)):
        target = _unit(down + ARM_OUT * side * lateral)
        local = _turn(names, local, arm, _rotation_between(pos[fore] - pos[arm], target))
        w = _worlds(names, local)
        pos = {n: m[3, :3] for n, m in w.items()}
    return local
