"""Isaac smoke: a full-site run gives PhysX no non-finite door pose, and the wrapper layer carries the NavMesh fixes.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_stage_fixes_smoke.py -q -s

Before the fix, ``/World/Fixtures/Doors/Compartment/Leaf0`` and ``Leaf1`` (rotate ops on bodies that their joints hold
unrotated) logged "Invalid PhysX transform detected" and "Illegal BroadPhaseUpdateData - non-finite bounds" on every step.
One Kit process; stopped with SIGINT at a frame boundary.
"""
import json
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
BOOT_TIMEOUT_S = 600.0
SIM_SECONDS = 6.0


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    for probe in (["pgrep", "-x", "kit"], ["pgrep", "-f", "isaacsim/kit/python"]):  # detection only
        if subprocess.run(probe, capture_output=True).returncode == 0:
            pytest.skip("another Isaac (kit) process is running")


def test_full_site_has_no_invalid_physx_transform(tmp_path, isaac_free):
    cfg = tmp_path / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data.update(id="isaac_full", cast=[], static_patients=False, duration_min=1.0, robot={"enabled": False},
                storyboard="../storyboards/empty.yaml")
    scen = cfg / "scenarios" / "isaac_full.yaml"
    scen.write_text(yaml.safe_dump(data, sort_keys=False))
    run = tmp_path / "run"
    logp = tmp_path / "host.log"
    log = open(logp, "w")
    host = subprocess.Popen([str(HOST_SH), "--run-dir", str(run), "--scenario", str(scen), "--headless", "--wait-for", "none",
                             "--full-site"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        start = time.monotonic()
        host_jsonl = run / "host.jsonl"
        while True:
            assert host.poll() is None, logp.read_text()[-4000:]
            assert time.monotonic() - start < BOOT_TIMEOUT_S, "sim time did not advance"
            recs = [json.loads(x) for x in host_jsonl.read_text().splitlines()] if host_jsonl.exists() else []
            if any(r.get("kind") == "progress" and r["t"] >= SIM_SECONDS for r in recs):
                break
            time.sleep(1.0)
        os.killpg(host.pid, signal.SIGINT)
        assert host.wait(timeout=120) == 0, logp.read_text()[-4000:]
    finally:
        if host.poll() is None:
            os.killpg(host.pid, signal.SIGKILL)
        log.close()
    text = logp.read_text()
    assert "Invalid PhysX transform" not in text and "non-finite bounds" not in text
    wrapper = (run / "stage.usda").read_text()
    assert 'over "NavMeshVolume_02"' in wrapper and "physics:rigidBodyEnabled" in wrapper
