"""Isaac smoke: six people walk from the ward corridor to a bedside of each of B1..B6, and two to the nurse bases WS1 and WS2.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_bedsides_smoke.py -q -s

Before the stage fixes in ``isaac_host/fixes.py`` the NavMesh could not reach B1, B3, B4, B5 or B6 (the stage's exclude
volumes cut the bedrooms out) and the walks ended "blocked". Runs the people demo's ``bedsides`` view (the host's start-up, then
six scripted walk_to intents); stills go to artifacts/screens/isaac_objects/bedsides/ (git-ignored). One Kit process.
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
OUT = ROOT / "artifacts" / "screens" / "isaac_objects" / "bedsides"
BOOT_TIMEOUT_S = 900.0
SEATS = [(18.3, 8.0), (19.0, 8.0), (19.8, 8.0), (18.3, 9.3), (19.0, 9.3), (19.8, 9.3), (18.3, 10.6), (19.8, 10.6)]


def ward_rooms() -> str:
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    return ",".join(["Lobby", "AE_Waiting", "Main_LiftLobby"] + [r["id"] for r in site["rooms"] if r["compartment"] == "C3"])


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data.update(id="isaac_bedsides", static_patients=False, duration_min=2.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml",
                cast=[{"id": f"W{i + 1}", "persona": "ward_visitor", "start": {"x": x, "y": y, "yaw": 1.57}}
                      for i, (x, y) in enumerate(SEATS)])
    q = cfg / "scenarios" / "isaac_bedsides.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_a_person_walks_to_every_bedside(tmp_path, isaac_free):
    scen = write_scenario(tmp_path)
    if OUT.exists():
        shutil.rmtree(OUT)
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.people_demo")
    log = open(tmp_path / "demo.log", "w")
    proc = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", ward_rooms(), "--out", str(OUT), "--view", "bedsides", "--seconds", "50"],
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
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    beds = {b["id"]: b for b in site["beds"]}
    for i, bid in enumerate(("B1", "B2", "B3", "B4", "B5", "B6")):
        x, y, posture, state, kind = last[f"W{i + 1}"]
        assert (state, kind) == ("done", "walk_to"), f"{bid}: walk ended {state!r} ({x:.1f}, {y:.1f})"  # never "blocked"
        assert math.hypot(x - beds[bid]["x"], y - beds[bid]["y"]) < 1.6, f"{bid}: ended at ({x:.1f}, {y:.1f}), not at the bed"
    rooms = {r["id"]: r["poly"] for r in site["rooms"]}
    for i, rid in enumerate(("Ward_WS1", "Ward_WS2")):  # the nurse bases
        x, y, posture, state, kind = last[f"W{7 + i}"]
        xs, ys = [p[0] for p in rooms[rid]], [p[1] for p in rooms[rid]]
        assert (state, kind) == ("done", "walk_to"), f"{rid}: walk ended {state!r} ({x:.1f}, {y:.1f})"
        assert min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys), f"{rid}: ended at ({x:.1f}, {y:.1f}), outside the room"
