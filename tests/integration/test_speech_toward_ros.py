"""The robot's speech wire over ROS: a robot node on its own domain names who each line is for on speech/toward, just
before speech/say. Two pairs sent back to back keep their addressees, and the toward ``room`` addresses nobody and
marks the audio row ``to_room``; every robot row carries ``heard_by``.

Needs a sourced ROS 2 Humble. Run with: pytest -m ros tests/integration/test_speech_toward_ros.py -q
The people's domain is ROS_DOMAIN_ID (default 140), the robot's the next one.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

AGENTS_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ID = "waiting_area_speech_toward"
LINES = ("Hello, you on the left, can I help?", "And hello to you as well.", "The clinic is running late, everyone.")

pytestmark = [
    pytest.mark.ros,
    pytest.mark.skipif(os.environ.get("ROS_DISTRO") != "humble", reason="needs a sourced ROS 2 Humble"),
]

# On the first camera frame with two people within 3.8 m: toward A + say, toward B + say, toward room + say, back
# to back. Records the chosen boxes' centres so the test can tell who they were.
PROBE = textwrap.dedent('''
    import json, math, sys, time
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    out, lines = sys.argv[1], json.loads(sys.argv[2])
    rclpy.init()
    n = Node("test_robot_toward")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    say = n.create_publisher(String, "/robot/speech/say", qos)
    toward = n.create_publisher(String, "/robot/speech/toward", qos)
    got = {"sent": None, "pose": None}
    def odom(m):
        got["pose"] = (m.pose.pose.position.x, m.pose.pose.position.y)
    def boxes(m):
        if got["sent"] or got["pose"] is None:
            return
        d = json.loads(m.data)
        rx, ry = got["pose"]
        near = []
        for b in d.get("detections", []):
            c = b["bbox"]["center"]
            dist = math.hypot(c["x"] - rx, c["y"] - ry)
            if dist <= 3.8:
                near.append((dist, b["id"], c["x"], c["y"]))
        if len(near) < 2:
            return
        near.sort()
        a, b = near[0], near[1]
        got["sent"] = {"a": [a[2], a[3]], "b": [b[2], b[3]]}
        for track, text in ((a[1], lines[0]), (b[1], lines[1]), ("room", lines[2])):
            toward.publish(String(data=track))
            say.publish(String(data=text))
    n.create_subscription(Odometry, "/robot/odom", odom, qos)
    n.create_subscription(String, "/robot/camera/bbox_3d", boxes, qos)
    end = time.time() + float(sys.argv[3])
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    open(out, "w").write(json.dumps(got))
''')


def write_variant(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(AGENTS_ROOT / "config", cfg)
    src = cfg / "scenarios" / "waiting_area_v1.yaml"
    scenario = yaml.safe_load(src.read_text())
    scenario.update(id=SCENARIO_ID, duration_min=2)
    scenario["robot"] = {**scenario.get("robot", {}), "x": -2.0, "y": 0.6, "yaw": 3.1416}  # facing the waiting area
    out = cfg / "scenarios" / f"{SCENARIO_ID}.yaml"
    out.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return out


def read(run: Path, stream: str) -> list[dict]:
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def who_was_at(status, t, xy):
    rows = [r for r in status if abs(r["t"] - t) <= 1.0]
    return min(rows, key=lambda r: math.hypot(r["x"] - xy[0], r["y"] - xy[1]))["person"]


def test_toward_pairs_and_room_over_ros(tmp_path):
    scenario = write_variant(tmp_path)
    private = int(os.environ.get("ROS_DOMAIN_ID") or 140)
    robot = private + 1
    runs = tmp_path / "runs"
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_DATA_DIR=str(tmp_path / "data"),
               WARDZERO_RUNS_DIR=str(runs), ROS_DOMAIN_ID=str(private), PYTHONUNBUFFERED="1")
    env.pop("WARDZERO_TEST_CLAUDE_BOT", None)
    cli = "import sys; from wardzero_agents.cli import main; sys.exit(main(sys.argv[1:]))"
    result = tmp_path / "probe.json"
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    side = subprocess.Popen([sys.executable, str(script), str(result), json.dumps(LINES), "40"],
                            env=dict(os.environ, ROS_DOMAIN_ID=str(robot)))
    try:
        proc = subprocess.run([sys.executable, "-c", cli, "run", str(scenario), "--no-panel", "--no-report",
                               "--time-scale", "5", "--robot-domain", str(robot), "--bot", "none"],
                              env=env, cwd=AGENTS_ROOT, capture_output=True, text=True, timeout=300)
        side.wait(timeout=60)
    finally:
        if side.poll() is None:
            side.kill()
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    (run,) = [p for p in runs.glob(f"*_{SCENARIO_ID}*") if p.is_dir()]
    got = json.loads(result.read_text())
    assert got["sent"], "the probe never saw two people within 3.8 m"
    rows = {r["text"]: r for r in read(run, "audio") if r["source_kind"] == "robot"}
    assert set(LINES) <= set(rows), sorted(rows)
    status = read(run, "status")
    a, b, room = (rows[x] for x in LINES)
    assert a["addressee"] == who_was_at(status, a["t"], got["sent"]["a"])
    assert b["addressee"] == who_was_at(status, b["t"], got["sent"]["b"])
    assert a["addressee"] != b["addressee"]
    assert room["addressee"] is None and room["to_room"] is True
    assert all(isinstance(r.get("heard_by"), list) and r["heard_by"] for r in (a, b, room))
