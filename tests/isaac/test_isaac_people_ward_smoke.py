"""Isaac ward smoke: a nurse escorts a visitor down the corridor, and a minded patient gets up out of bed and leaves.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_people_ward_smoke.py -q -s

Stills and frames go to artifacts/screens/isaac_people/ward_smoke/ (git-ignored). One Kit process.
"""
import json
import math
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
OUT = ROOT / "artifacts" / "screens" / "isaac_people" / "ward_smoke"
BOOT_TIMEOUT_S = 900.0


def ward_rooms() -> str:
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    return ",".join(["Lobby", "AE_Waiting", "Main_LiftLobby"] + [r["id"] for r in site["rooms"] if r["compartment"] == "C3"])


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data.update(id="isaac_ward_smoke", static_patients=True, duration_min=2.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml",
                cast=[{"id": "W1", "persona": "ward_visitor", "start": {"x": 19.0, "y": 11.0, "yaw": 1.57}},
                      {"id": "V1", "persona": "ward_visitor", "start": {"x": 19.0, "y": 9.0, "yaw": 1.57}},
                      {"id": "B2P", "persona": "older_outpatient_stick", "start": {"bed": "B2"}}])
    q = cfg / "scenarios" / "isaac_ward_smoke.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_escort_and_a_patient_who_gets_up(tmp_path, isaac_free):
    scen = write_scenario(tmp_path)
    if OUT.exists():
        shutil.rmtree(OUT)
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.people_demo")
    log = open(tmp_path / "demo.log", "w")
    proc = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", ward_rooms(), "--out", str(OUT), "--view", "ward2", "--seconds", "20"],
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
    track = json.loads((OUT / "demo_track.json").read_text())
    last = track["track"][-1]["people"]
    w1, v1 = last["W1"], last["V1"]
    assert (w1[3], w1[4]) == ("done", "escort")  # the escort arrived
    assert math.hypot(w1[0] - v1[0], w1[1] - v1[1]) < 3.0  # the visitor came along, not left at the start
    assert v1[1] > 15.0  # ... 6+ m up the corridor from y = 9
    ev = track["events"]["B2P"]
    assert ("get_up", "") in [tuple(e) for e in ev]  # the lying prop was swapped for a walking character
    b = last["B2P"]
    assert b[2] == "walking" or b[2] == "standing"
    assert math.hypot(b[0] - 25.1, b[1] - 19.1) > 5.0  # and they walked away from the bed
    assert (OUT / "still_top_t000.1.png").exists()
