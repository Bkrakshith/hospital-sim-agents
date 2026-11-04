"""Isaac people smoke test: three people walk, sit and idle as Isaac characters, driven by Intents.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_people_smoke.py -q -s

Runs ``wardzero_agents.isaac_host.people_demo`` (the host's start-up, then a scripted list of Intents, no ROS and no
minds) on the lobby with three people from the waiting-area scenario, and saves stills to
artifacts/screens/isaac_people/smoke/ (git-ignored). Needs Isaac Sim 6.0 at ~/isaacsim; starts one Kit process.
"""
import json
import math
import os
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
STILLS = ROOT / "artifacts" / "screens" / "isaac_people" / "smoke"
SECONDS = 24.0
BOOT_TIMEOUT_S = 600.0


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data["cast"] = [c for c in data["cast"] if c["id"] in ("P1", "P2", "P4")]  # P2 is seated at C1, P4 at A3, P1 at B2
    data.update(id="isaac_people_smoke", static_patients=False, duration_min=2.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml")
    q = cfg / "scenarios" / "isaac_people_smoke.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_three_people_walk_sit_and_idle(tmp_path, isaac_free):
    scen = write_scenario(tmp_path)
    if STILLS.exists():
        shutil.rmtree(STILLS)
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.people_demo")
    log = open(tmp_path / "demo.log", "w")
    proc = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", "Lobby,AE_Waiting,Main_LiftLobby", "--out", str(STILLS), "--view", "smoke", "--seconds", str(SECONDS)],
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
    track = json.loads((STILLS / "demo_track.json").read_text())
    last = track["track"][-1]["people"]
    # P1 walked to the reception desk
    x, y, posture, state, kind = last["P1"]
    assert (state, kind) == ("done", "walk_to") and posture == "standing"
    assert math.hypot(x - 1.0, y - 0.57) < 0.6
    # P2 moved seats: it stood, walked and sat on A1
    assert last["P2"][2] == "seated" and last["P2"][3:] == ["done", "sit_down"]
    # P4 never got an intent: it sits (idle) where it started
    assert last["P4"][2] == "seated" and last["P4"][3] == "idle"
    walked = {r["people"]["P1"][2] for r in track["track"]}
    assert "walking" in walked
    assert track["stats"]["ms_median"] < 200  # a sanity bound with a render product on; the perf gate is in the report
    stills = sorted(STILLS.glob("still_wide_*.png"))
    assert len(stills) >= 3 and all(s.stat().st_size > 20000 for s in stills)
    print(f"\n[isaac people smoke] {json.dumps(track['stats'])}")
    print(f"[isaac people smoke] stills in {STILLS}")
