"""The stub sim on the robot's own ROS domain: a robot node there gets the brief, the clock and a person's box, what
it says reaches the people, and it sees none of the people's topics. Also the built-in G1 (fake LLM) on that domain.

Needs a sourced ROS 2 Humble. Run with: pytest -m ros tests/integration/test_robot_domain_e2e.py -q
"""
import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

AGENTS_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ID = "waiting_area_robot_domain"
BRIEF = {"role": "greeter", "job": "Welcome people and answer where things are."}
SAY = "Hello from the test robot, can I help?"

pytestmark = [
    pytest.mark.ros,
    pytest.mark.skipif(os.environ.get("ROS_DISTRO") != "humble", reason="needs a sourced ROS 2 Humble"),
]

# a robot on the robot domain only: records what it receives, says one line when it sees a person
PROBE = textwrap.dedent('''
    import json, sys, time
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    out, say = sys.argv[1], sys.argv[2]
    rclpy.init()
    n = Node("test_robot")
    got = {"brief": None, "person_boxes": 0, "clock": 0, "said": False}
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = n.create_publisher(String, "/robot/speech/say", qos)
    def brief(m): got["brief"] = json.loads(m.data)
    def boxes(m):
        d = json.loads(m.data)
        d = d.get("detections", []) if isinstance(d, dict) else d
        if any(r.get("class_id") == "person" for b in d for r in b.get("results", [])):
            got["person_boxes"] += 1
            if not got["said"]:
                got["said"] = True
                pub.publish(String(data=say))
    n.create_subscription(String, "/robot/task/brief", brief, latched)
    n.create_subscription(String, "/robot/camera/bbox_3d", boxes, qos)
    n.create_subscription(Clock, "/clock", lambda m: got.__setitem__("clock", got["clock"] + 1), qos)
    end = time.time() + float(sys.argv[3])
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    got["topics"] = sorted(t for t, _ in n.get_topic_names_and_types())
    open(out, "w").write(json.dumps(got))
''')


def write_variant(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(AGENTS_ROOT / "config", cfg)
    src = cfg / "scenarios" / "waiting_area_v1.yaml"
    scenario = yaml.safe_load(src.read_text())
    scenario.update(id=SCENARIO_ID, duration_min=3, robot_brief=BRIEF)
    scenario["robot"] = {**scenario.get("robot", {}), "x": -2.0, "y": 0.6, "yaw": 3.1416}  # facing the waiting area
    out = cfg / "scenarios" / f"{SCENARIO_ID}.yaml"
    out.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return out


def read(run: Path, stream: str) -> list[dict]:
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def run_cli(tmp_path, *extra, probe=False):
    scenario = write_variant(tmp_path)
    private = random.randint(1, 100)
    robot = private + 1
    runs = tmp_path / "runs"
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_DATA_DIR=str(tmp_path / "data"),
               WARDZERO_RUNS_DIR=str(runs), ROS_DOMAIN_ID=str(private), PYTHONUNBUFFERED="1")
    cli = "import sys; from wardzero_agents.cli import main; sys.exit(main(sys.argv[1:]))"
    result = tmp_path / "probe.json"
    side = None
    if probe:
        script = tmp_path / "probe.py"
        script.write_text(PROBE)
        side = subprocess.Popen([sys.executable, str(script), str(result), SAY, "40"],
                                env=dict(os.environ, ROS_DOMAIN_ID=str(robot)))
    try:
        proc = subprocess.run([sys.executable, "-c", cli, "run", str(scenario), "--no-panel", "--time-scale", "30",
                               "--robot-domain", str(robot), *extra],
                              env=env, cwd=AGENTS_ROOT, capture_output=True, text=True, timeout=300)
        if side is not None:
            side.wait(timeout=60)
    finally:
        if side is not None and side.poll() is None:
            side.kill()
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    (run,) = [p for p in runs.glob(f"*_{SCENARIO_ID}*") if p.is_dir()]
    return run, result


def test_a_robot_on_the_robot_domain_gets_the_world_and_is_heard(tmp_path):
    run, result = run_cli(tmp_path, "--bot", "none", probe=True)
    got = json.loads(result.read_text())
    assert got["brief"] == {**BRIEF, "desk_staffed": False}  # no receptionist in this scenario
    assert got["clock"] > 5
    assert got["person_boxes"] > 0, "no person ever appeared on the robot's camera topic"
    assert not [t for t in got["topics"] if t.startswith(("/humans", "/scenario", "/sim", "/agents"))]
    assert any(a["source_kind"] == "robot" and a.get("text") == SAY for a in read(run, "audio"))


def test_the_built_in_g1_on_the_robot_domain_greets_a_person_in_view(tmp_path):
    run, _ = run_cli(tmp_path, "--bot", "claude")
    from wardzero_agents.llm.fake import ROBOT_GREETING
    assert any(a["source_kind"] == "robot" and ROBOT_GREETING in (a.get("text") or "") for a in read(run, "audio"))
    assert json.loads((run / "robot_config.json").read_text())["domain"] is not None
