"""The resting pose of a patient in bed, computed from a skeleton's rest transforms (numpy only, no Isaac)."""
import numpy as np
import pytest

from wardzero_agents.isaac_host import rest_pose as rp

# A T-pose skeleton, Y up, facing +Z (toes ahead of the feet): joint path -> world position
JOINTS = {
    "Root": (0, 0, 0), "Root/Hips": (0, 1.0, 0), "Root/Hips/Spine1": (0, 1.15, 0), "Root/Hips/Spine1/Chest": (0, 1.35, 0),
    "Root/Hips/Spine1/Chest/Head": (0, 1.6, 0),
    "Root/Hips/Spine1/Chest/LeftArm": (0.2, 1.45, 0), "Root/Hips/Spine1/Chest/LeftArm/LeftForeArm": (0.5, 1.45, 0),
    "Root/Hips/Spine1/Chest/LeftArm/LeftForeArm/LeftHand": (0.8, 1.45, 0),
    "Root/Hips/Spine1/Chest/RightArm": (-0.2, 1.45, 0), "Root/Hips/Spine1/Chest/RightArm/RightForeArm": (-0.5, 1.45, 0),
    "Root/Hips/Spine1/Chest/RightArm/RightForeArm/RightHand": (-0.8, 1.45, 0),
    "Root/Hips/LeftUpLeg": (0.1, 0.95, 0), "Root/Hips/LeftUpLeg/LeftFoot": (0.1, 0.1, 0),
    "Root/Hips/LeftUpLeg/LeftFoot/LeftToeBase": (0.1, 0.0, 0.15),
    "Root/Hips/RightUpLeg": (-0.1, 0.95, 0), "Root/Hips/RightUpLeg/RightFoot": (-0.1, 0.1, 0),
    "Root/Hips/RightUpLeg/RightFoot/RightToeBase": (-0.1, 0.0, 0.15),
}


def rest():
    names = list(JOINTS)
    pos = {n: np.array(p, float) for n, p in JOINTS.items()}
    local = np.zeros((len(names), 4, 4))
    for i, n in enumerate(names):
        par = n.rsplit("/", 1)[0] if "/" in n else None
        m = np.eye(4)
        m[3, :3] = pos[n] - (pos[par] if par else 0)  # row-vector convention: translation in the last row
        local[i] = m
    return names, local


def world_pos(names, local):
    w = {}
    for n, m in zip(names, local):
        par = n.rsplit("/", 1)[0] if "/" in n else None
        w[n] = m @ (w[par] if par else np.eye(4))
    return {n: m[3, :3] for n, m in w.items()}


def unit(v):
    return v / np.linalg.norm(v)


def test_arms_come_down_beside_the_body():
    names, local = rest()
    new = rp.resting(names, local)
    p = world_pos(names, new)
    arm = unit(p["Root/Hips/Spine1/Chest/LeftArm/LeftForeArm"] - p["Root/Hips/Spine1/Chest/LeftArm"])
    assert arm[1] < -0.8  # pointing down, no longer out to the side (a T-pose has arm[1] = 0)
    assert arm[0] > 0.0  # a little away from the body, on its own side
    right = unit(p["Root/Hips/Spine1/Chest/RightArm/RightForeArm"] - p["Root/Hips/Spine1/Chest/RightArm"])
    assert right[1] < -0.8 and right[0] < 0.0


def test_the_hands_hang_lower_than_the_shoulders_and_the_legs_do_not_move():
    names, local = rest()
    p0, p1 = world_pos(names, local), world_pos(names, rp.resting(names, local))
    assert p1["Root/Hips/Spine1/Chest/LeftArm/LeftForeArm/LeftHand"][1] < p0["Root/Hips/Spine1/Chest/LeftArm"][1] - 0.4
    for n in names:
        if "UpLeg" in n or n == "Root/Hips":
            assert p1[n] == pytest.approx(p0[n], abs=1e-9)


def test_sitting_up_raises_the_torso_towards_the_front_and_keeps_the_hips():
    names, local = rest()
    new = rp.resting(names, local, sit_up_deg=50.0)
    p = world_pos(names, new)
    torso = unit(p["Root/Hips/Spine1/Chest/Head"] - p["Root/Hips/Spine1"])
    assert torso[2] == pytest.approx(np.sin(np.radians(50)), abs=1e-6)  # leaning to the front (the toes' side), 50 degrees from the old vertical
    assert p["Root/Hips"] == pytest.approx(np.array(JOINTS["Root/Hips"]), abs=1e-9)
    assert p["Root/Hips/LeftUpLeg/LeftFoot"] == pytest.approx(np.array(JOINTS["Root/Hips/LeftUpLeg/LeftFoot"]), abs=1e-9)
    assert abs(np.linalg.norm(p["Root/Hips/Spine1/Chest/Head"] - p["Root/Hips/Spine1"]) - 0.45) < 1e-6  # bones keep their length


def test_the_arms_still_hang_down_when_sitting_up():
    names, local = rest()
    p = world_pos(names, rp.resting(names, local, sit_up_deg=50.0))
    arm = unit(p["Root/Hips/Spine1/Chest/LeftArm/LeftForeArm"] - p["Root/Hips/Spine1/Chest/LeftArm"])
    torso = unit(p["Root/Hips/Spine1/Chest/Head"] - p["Root/Hips"])
    assert np.dot(arm, torso) < -0.8  # along the torso, towards the hips


def test_the_two_skeleton_families_name_their_joints_differently():
    assert rp.find(["a/b/L_Upperarm", "a/R_Upperarm"], rp.LEFT_ARM) == "a/b/L_Upperarm"
    assert rp.find(["a/LeftArm", "a/RightArm"], rp.LEFT_ARM) == "a/LeftArm"
    assert rp.find(["a/Spine01", "a/Waist"], rp.SPINE) == "a/Spine01"
    assert rp.find(["a/x"], rp.LEFT_ARM) is None
