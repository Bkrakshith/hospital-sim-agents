"""Where a robot's ready USD lives, when it is stale, and how the host gets one made. The conversion itself needs Kit
and is run (and checked) by tests/isaac/test_isaac_robot_smoke.py; here the runner is a fake."""
import json
import os
import time

import pytest

from wardzero_agents.robots import ready_usd as ru
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile, load_profile, save_profile


def test_the_g1_has_a_standing_height_and_other_robots_start_on_the_floor():
    assert G1_CLAUDE.base_z == pytest.approx(0.80)  # pelvis of the standing G1 (spike S1 Q4)
    other = RobotProfile(name="cart", can_move=True, base="wheeled", radius=0.3, reach=(0.3, 1.0), max_speed=1.0, ns="cart",
                         sensors=())
    assert other.base_z == 0.0


def test_base_z_survives_the_robot_yaml(tmp_path):
    p = RobotProfile(**{**G1_CLAUDE.__dict__, "name": "tall", "base_z": 0.95, "controller": "external"})
    save_profile(p, tmp_path)
    assert load_profile("tall", tmp_path).base_z == 0.95


def test_paths(tmp_path):
    assert ru.ready_usd_path(G1_CLAUDE, tmp_path) == tmp_path / "robots" / "g1_claude" / "robot.usd"
    assert ru.robot_urdf(G1_CLAUDE).name == "g1_29dof_brainco.urdf"
    mine = RobotProfile(**{**G1_CLAUDE.__dict__, "name": "mine", "urdf": tmp_path / "mine.urdf"})
    assert ru.robot_urdf(mine) == tmp_path / "mine.urdf"
    with pytest.raises(ValueError, match="no URDF"):
        ru.robot_urdf(RobotProfile(**{**mine.__dict__, "urdf": None}))


class Runner:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = cmd[cmd.index("--out") + 1]
        if self.ok:
            open(out, "w").write("#usda 1.0\n")
        line = json.dumps({"ok": True, "usd": out} if self.ok else {"ok": False, "errors": ["sensor asset missing"]})

        class R:
            returncode = 0 if self.ok else 1
            stdout = "noise\n" + line + "\n"
            stderr = ""
        return R()


@pytest.fixture
def robot(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text("<robot name='r'/>")
    return RobotProfile(**{**G1_CLAUDE.__dict__, "name": "r", "urdf": urdf}), tmp_path


def test_a_missing_usd_is_converted_with_the_robots_sensors(robot):
    p, data = robot
    run = Runner()
    out = ru.ensure_ready_usd(p, data, runner=run)
    assert out == data / "robots" / "r" / "robot.usd" and out.exists()
    (cmd,) = run.calls
    assert cmd[cmd.index("--urdf") + 1] == str(p.urdf) and cmd[cmd.index("--out") + 1] == str(out)
    sensors = (data / "robots" / "r" / "wardzero_sensors.yaml").read_text()
    assert "realsense_d455" in sensors and "d435_link" in sensors


def test_a_fresh_usd_is_left_alone_and_a_changed_input_converts_again(robot):
    p, data = robot
    run = Runner()
    ru.ensure_ready_usd(p, data, runner=run)
    ru.ensure_ready_usd(p, data, runner=run)
    assert len(run.calls) == 1
    later = time.time() + 5
    os.utime(p.urdf, (later, later))  # the URDF changed after the USD was written
    ru.ensure_ready_usd(p, data, runner=run)
    assert len(run.calls) == 2
    q = RobotProfile(**{**p.__dict__, "sensors": (G1_CLAUDE.sensors[1],)})  # the sensors changed
    ru.ensure_ready_usd(q, data, runner=run)
    assert len(run.calls) == 3


def test_a_failed_conversion_says_why(robot):
    p, data = robot
    with pytest.raises(RuntimeError, match="sensor asset missing"):
        ru.ensure_ready_usd(p, data, runner=Runner(ok=False))
    assert not ru.ready_usd_path(p, data).exists()
