"""Isaac smoke: every bedside and nurse base of the kept ward and A&E bays is reachable on the baked NavMesh.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_navmesh_smoke.py -q -s

Runs ``isaac_host.nav_probe`` (the host's start-up up to the bake, then a path query from the nearest corridor to each
bedside and nurse base the site grid leaves free). Also checks what the freed NavMesh exclude volumes cover: only the
rooms of the ward and A&E bays, never the lobby or the corridors. One Kit process.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
HOST_SH = ROOT / "scripts" / "isaac_host.sh"
SCENARIO = ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_every_bedside_and_nurse_base_is_reachable(tmp_path, isaac_free):
    site = json.loads((ROOT / "config" / "site" / "hospital.json").read_text())
    rooms = [r["id"] for r in site["rooms"] if r["compartment"] in ("C3", "C1")] + ["Lobby", "AE_Waiting", "Main_LiftLobby"]
    env = dict(os.environ, WARDZERO_ISAAC_MODULE="wardzero_agents.isaac_host.nav_probe")
    r = subprocess.run([str(HOST_SH), "--run-dir", str(tmp_path / "run"), "--scenario", str(SCENARIO), "--headless",
                        "--wait-for", "none", "--rooms", ",".join(rooms)], env=env, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    sums = re.findall(r"NAVSUM ok (\d+) / (\d+) fail: (.*)", out)
    assert sums, out[-3000:]
    ok, total, fail = int(sums[-1][0]), int(sums[-1][1]), sums[-1][2].strip()
    probes = re.findall(r"NAVPROBE (\S+) (OK|NO)", out)
    labels = {k for k, _ in probes}
    assert {"B1+", "B2+", "B3+", "B4-", "B5-", "B6+", "Ward_WS1", "Ward_WS2"} <= labels  # the ward, both nurse bases
    assert any(k.startswith("AE_") for k in labels), labels  # and the A&E bays
    assert r.returncode == 0 and fail == "", f"unreachable: {fail}\n" + out[-2000:]
    freed = re.findall(r"NAVFREED (\S+) (.*)", out)
    assert freed, "the ward's exclude volumes were not freed"
    for name, covered in freed:
        for item in covered.split(", "):
            use = re.search(r"\((.*)\)", item).group(1)
            assert use not in ("lobby", "corridor", "entrance", "ward_entrance", "lift_lobby"), f"{name} frees {item}"
