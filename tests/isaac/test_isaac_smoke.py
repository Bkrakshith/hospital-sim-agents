"""Isaac host smoke test: boot headless on the culled lobby + ward, no people, no robot.

Checks /clock on the private and the robot domain, the latched roster, the run folder (wrapper stage,
host stream) and a clean exit on Control stop after 10 s of sim time. Prints ms/frame and peak VRAM.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac -q -s

Needs Isaac Sim 6.0 at ~/isaacsim and the system ROS 2 Humble at /opt/ros/humble (the listeners run
there, outside Isaac's Python). It starts one Kit process and only ever signals that one.
"""
import json
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
HOST_SH = ROOT / "scripts" / "isaac_host.sh"
ROS_SETUP = "/opt/ros/humble/setup.bash"
SIM_SECONDS = 10.0
BOOT_TIMEOUT_S = 420.0


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data.update(id="isaac_smoke", cast=[], static_patients=False, duration_min=1.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml")
    q = cfg / "scenarios" / "isaac_smoke.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


def lobby_and_ward_rooms() -> list[str]:
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    return ["Lobby", "AE_Waiting", "Main_LiftLobby"] + [r["id"] for r in site["rooms"] if r["compartment"] == "C3"]


def system_ros(domain: int, cmd: str, timeout: float) -> subprocess.Popen:
    """A system-ROS CLI command with a clean environment (Isaac's and conda's paths break it)."""
    env = {"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"}
    script = (f"source {ROS_SETUP} && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID={domain} "
              f"&& timeout {int(timeout)} {cmd}")
    return subprocess.Popen(["bash", "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def host_records(run: Path) -> list[dict]:
    p = run / "host.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    if not Path(ROS_SETUP).exists():
        pytest.skip("no system ROS 2 Humble")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_boots_headless_and_stops_on_control(tmp_path, isaac_free):
    scen = write_scenario(tmp_path)
    run = tmp_path / "run"
    private = random.randint(110, 160)
    robot = private + 1
    log = open(tmp_path / "host.log", "w")
    host = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(run), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", ",".join(lobby_and_ward_rooms()), "--private-domain", str(private), "--robot-domain", str(robot)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    listeners = []
    try:
        start = time.monotonic()
        while not any(r.get("kind") == "startup" for r in host_records(run)):
            assert host.poll() is None, (tmp_path / "host.log").read_text()[-4000:]
            assert time.monotonic() - start < BOOT_TIMEOUT_S, "the host did not come up"
            time.sleep(1.0)
        listeners = [
            system_ros(private, "ros2 topic echo --once /clock rosgraph_msgs/msg/Clock", 60),
            system_ros(robot, "ros2 topic echo --once /clock rosgraph_msgs/msg/Clock", 60),
            system_ros(private, "ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable "
                                "/humans/roster std_msgs/msg/String", 60),
        ]
        while not any(r.get("kind") == "progress" and r["t"] >= SIM_SECONDS for r in host_records(run)):
            assert host.poll() is None, (tmp_path / "host.log").read_text()[-4000:]
            assert time.monotonic() - start < BOOT_TIMEOUT_S + 120, "sim time did not reach 10 s"
            time.sleep(0.5)
        stop = system_ros(private, "ros2 topic pub --times 6 -r 2 /scenario/control std_msgs/msg/String "
                                   "\"{data: '{\\\"schema\\\": \\\"wz.v1\\\", \\\"action\\\": \\\"stop\\\"}'}\"", 30)
        listeners.append(stop)
        assert host.wait(timeout=180) == 0, (tmp_path / "host.log").read_text()[-4000:]
        outs = [p.communicate(timeout=90)[0] for p in listeners]
    finally:
        for p in [host, *listeners]:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        log.close()
    clock_private, clock_robot, roster = outs[:3]
    assert "sec:" in clock_private, clock_private
    assert "sec:" in clock_robot, clock_robot
    assert "wz.v1" in roster and "people" in roster, roster
    recs = host_records(run)
    final = recs[-1]
    assert final["kind"] == "final" and SIM_SECONDS <= final["t"] < 60.0  # stopped by Control, not the end
    assert (run / "stage.usda").exists()
    startup = next(r for r in recs if r["kind"] == "startup")
    print(f"\n[isaac smoke] start-up {json.dumps(startup)}")
    print(f"[isaac smoke] final {json.dumps(final)}")
