"""Isaac smoke test for the robot (I4): the G1 stands at reception with three people, on a ROS domain of its own.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_robot_smoke.py -q -s

Runs ``wardzero_agents.isaac_host.robot_demo`` (the host's start-up and real frame loop) on the lobby and saves the
robot's camera image and two stills to artifacts/screens/isaac_robot/ (git-ignored). A probe node on the robot domain
plays the robot under test. Needs Isaac Sim 6.0 at ~/isaacsim; starts one Kit process. The first run converts the G1's
URDF into its ready USD (about 15 s).
"""
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
HOST_SH = ROOT / "scripts" / "isaac_host.sh"
OUT = ROOT / "artifacts" / "screens" / "isaac_robot"
BOOT_TIMEOUT_S = 900.0
ROBOT_DOMAIN, PRIVATE_DOMAIN = 41, 42
SECONDS = 14.0


def write_scenario(tmp: Path, cast=("P1", "P2", "P4")) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data["cast"] = [c for c in data["cast"] if c["id"] in cast]
    data.update(id="isaac_robot_smoke", static_patients=False, duration_min=3.0, objects=[],
                robot={"enabled": True}, storyboard="../storyboards/empty.yaml")
    q = cfg / "scenarios" / "isaac_robot_smoke.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def run_demo(tmp_path: Path, mode: str, seconds: float, out: Path, cast=("P1", "P2", "P4")) -> dict:
    scen = write_scenario(tmp_path, cast)
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.robot_demo")
    log = open(tmp_path / "demo.log", "w")
    proc = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", "Lobby,AE_Waiting,Main_LiftLobby", "--fire", "none", "--private-domain", str(PRIVATE_DOMAIN),
         "--robot-domain", str(ROBOT_DOMAIN), "--out", str(out), "--mode", mode, "--seconds", str(seconds)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        assert proc.wait(timeout=BOOT_TIMEOUT_S) == 0, (tmp_path / "demo.log").read_text()[-4000:]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    return json.loads((out / "demo_stats.json").read_text())


def test_the_g1_stands_at_reception_and_talks_on_its_own_domain(tmp_path, isaac_free):
    r = run_demo(tmp_path, "smoke", SECONDS, OUT)
    st = r["stats"]
    assert st["ms_median"] < 60.0, st  # the brief's budget: one 1280x720 camera, 3 people, no fire
    c = r["counts"]
    # the graph: clock, tf, joint states, the camera and the host's bbox_3d, all heard on the robot domain
    for key in ("clock", "tf", "joint_states", "camera_rgb", "camera_depth", "camera_info", "bbox_3d"):
        assert c.get(key, 0) > 0, (key, c)
    # the StubWorld's robot logic on the same domain: a task action answered, speech heard, the brief
    assert any(t["id"] == "demo-call" and t["ok"] for t in r["task_results"])
    assert any(h["text"] and "pharmacy" in h["text"] and h["source_id"] == "unknown" for h in r["heard"])
    # a person walked up to the desk and the camera saw them: anonymous track, class only
    assert r["bbox_with_people"] > 0
    boxes = [d for t in [json.loads(x) for x in [r["bbox_last"]] if x] for d in t["detections"]]
    for d in boxes:
        assert d["id"].startswith("t") and d["results"][0]["class_id"] == "person"
    # fairness: no person id, no private topic on the robot's domain
    assert r["leaked_ids"] == [] and r["forbidden_topics"] == []
    # the hand-over gesture reached the arm and came back
    shoulder = [row["q"]["right_shoulder_pitch_joint"] for row in r["arm_track"]]
    assert min(shoulder) < -0.8 and abs(shoulder[-1]) < 0.3
    # the G1 stayed where it was put (static, fixed base) and the obstacle cylinder never moved
    x, y, yaw = r["robot_pose"]
    assert (x, y) == pytest.approx((1.05, 1.62), abs=0.05) and r["robot"]["proxy_moves"] == 0
    for name in ("robot_camera", "threequarter", "top"):
        assert (OUT / r["stills"][name]).stat().st_size > 20_000
