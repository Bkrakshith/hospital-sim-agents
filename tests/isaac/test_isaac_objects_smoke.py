"""Isaac smoke: a Flow fire grows and goes out, and a tray is carried to a bed and handed over, all drawn from the world's state.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_objects_smoke.py -q -s

Runs ``wardzero_agents.isaac_host.objects_demo`` (the host's start-up, then the world's objects driven by hand) on the ward
and saves stills to artifacts/screens/isaac_objects/smoke_<view>/ (git-ignored). One Kit process per view.
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
OUT = ROOT / "artifacts" / "screens" / "isaac_objects"
BOOT_TIMEOUT_S = 900.0


def ward_rooms() -> str:
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    return ",".join(["Lobby", "AE_Waiting", "Main_LiftLobby"] + [r["id"] for r in site["rooms"] if r["compartment"] == "C3"])


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data.update(id="isaac_objects", static_patients=True, duration_min=2.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml",
                cast=[{"id": "W1", "persona": "ward_visitor", "start": {"x": 19.4, "y": 15.8, "yaw": 0.0}}],
                objects=[{"kind": "tray", "id": "tray_1", "label": "soft diet", "owner": "SB3", "holder": "W1", "x": 19.4, "y": 15.8, "z": 1.0},
                         {"kind": "extinguisher", "id": "ext_1", "label": "wall extinguisher", "x": 18.2, "y": 12.0, "z": 0.9},
                         {"kind": "specimen_box", "id": "spec_1", "x": 18.4, "y": 9.0, "z": 0.0},
                         {"kind": "instrument", "id": "ins_1", "label": "scissors", "x": 22.4, "y": 16.5, "z": 0.0},
                         {"kind": "call_bell", "bed": "B3"}])
    q = cfg / "scenarios" / "isaac_objects.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def run_demo(tmp_path: Path, view: str, seconds: float) -> Path:
    scen = write_scenario(tmp_path)
    out = OUT / f"smoke_{view}"
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.objects_demo")
    log = open(tmp_path / "demo.log", "w")
    proc = subprocess.Popen(
        [str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(scen), "--headless", "--wait-for", "none",
         "--rooms", ward_rooms(), "--out", str(out), "--view", view, "--seconds", str(seconds)],
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
    return out


def test_a_fire_grows_and_goes_out(tmp_path, isaac_free):
    out = run_demo(tmp_path, "fire", 12.0)
    data = json.loads((out / "demo_stats.json").read_text())
    sizes = [o["size"] for row in data["rows"] for o in row["objects"] if o["kind"] == "fire"]
    assert max(sizes) > 0.5  # it grew
    final = [o for o in data["rows"][-1]["objects"] if o["kind"] == "fire"][0]
    assert final["active"] is False and final["size"] == 0.0  # and went out
    assert data["stats"]["flow_ready"] is True  # drawn with Flow, not the proxy
    drawn = [(row["t"], row["drawn"]["fires"].get("fire_1")) for row in data["rows"]]  # the emitter prim, read back
    radii = [f["radius"] for _, f in drawn if f is not None]
    assert drawn[0][1] is None and radii, "no Flow emitter prim appeared"  # no emitter before the fire starts
    assert all(f["kind"] == "flow" for _, f in drawn if f is not None)
    assert max(radii) > radii[0] + 0.1  # the emitter grew with the fire's size
    assert drawn[-1][1] is None  # and the emitter prim is gone once the fire is out
    assert list(out.glob("still_corridor_*.png")) and list(out.glob("still_top_*.png"))


def test_a_tray_is_carried_to_a_bed_and_handed_over(tmp_path, isaac_free):
    out = run_demo(tmp_path, "tray", 10.0)
    data = json.loads((out / "demo_stats.json").read_text())
    tray = [o for o in data["rows"][-1]["objects"] if o["id"] == "tray_1"][0]
    assert tray["holder"] == "SB3"  # handed to the patient in B3 (sitting up)
    assert math.hypot(tray["x"] - 25.1, tray["y"] - 16.2) < 0.6 and tray["z"] == pytest.approx(1.0)
    bell = [o for o in data["rows"][-1]["objects"] if o["kind"] == "call_bell"][0]
    assert bell["active"] is False  # the call was answered
    drawn = [row["drawn"] for row in data["rows"]]
    first, last = drawn[0]["items"]["tray_1"]["pos"], drawn[-1]["items"]["tray_1"]["pos"]
    assert math.hypot(first[0] - 19.4, first[1] - 15.8) < 1.0  # the tray prim started with the nurse in the corridor ...
    assert math.hypot(last[0] - 24.66, last[1] - 16.2) < 0.1 and last[2] == pytest.approx(1.0)  # ... and lies on the bed where it was handed over
    mid = [d["items"]["tray_1"]["pos"] for d in drawn[3:8]]
    assert any(m[0] > first[0] + 1.0 for m in mid)  # it travelled with the holder
    lit = [d["bells"]["B3"][0] for d in drawn]
    assert max(lit) > 1.0 and lit[-1] < 0.1  # the lamp's emissive colour: on while ringing, dark once answered
    assert list(out.glob("still_bed_*.png"))
