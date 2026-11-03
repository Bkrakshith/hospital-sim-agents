"""End to end: a 3-minute scenario through `hhumans run` with the reception bot, in test mode.

Needs a sourced ROS 2 Humble (source $HG1_ROOT/env.sh) and the hhumans CLI (Task 14).
Run with: pytest -m ros tests/integration -q
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

AGENTS_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ID = "waiting_area_e2e"
KEEP = {"b1": "t+00:20", "b3": "t+00:40", "b6": "t+01:00"}  # clinic_delay, lost_visitor P5, coughing_fit P2

pytestmark = [
    pytest.mark.ros,
    pytest.mark.skipif(os.environ.get("ROS_DISTRO") != "humble", reason="needs a sourced ROS 2 Humble"),
]


def write_variant(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    shutil.copytree(AGENTS_ROOT / "config", cfg)
    src = cfg / "scenarios" / "waiting_area_v1.yaml"
    scenario = yaml.safe_load(src.read_text())
    scenario.update(id=SCENARIO_ID, duration_min=3)
    out = cfg / "scenarios" / f"{SCENARIO_ID}.yaml"
    out.write_text(yaml.safe_dump(scenario, sort_keys=False))
    sb_path = (src.parent / scenario["storyboard"]).resolve()
    sb = yaml.safe_load(sb_path.read_text())
    sb["beats"] = [dict(b, at=KEEP[b["id"]]) for b in sb["beats"] if b["id"] in KEEP]
    assert [b["card"] for b in sb["beats"]] == ["clinic_delay", "lost_visitor", "coughing_fit"]
    sb_path.write_text(yaml.safe_dump(sb, sort_keys=False))
    return out


def read(run: Path, stream: str) -> list[dict]:
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def find_run(runs_root: Path, started: float) -> Path:
    found = sorted(p for p in runs_root.glob(f"*_{SCENARIO_ID}*") if p.is_dir())
    if not found:  # a CLI that ignores WARDZERO_RUNS_DIR records under the component's runs/
        found = sorted(p for p in (AGENTS_ROOT / "runs").glob(f"*_{SCENARIO_ID}*")
                       if p.is_dir() and p.stat().st_mtime >= started)
    assert found, "no run folder was written"
    return found[-1]


def test_end_to_end(tmp_path):
    scenario = write_variant(tmp_path)
    runs_root = tmp_path / "runs"
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_DATA_DIR=str(tmp_path / "data"),
               WARDZERO_RUNS_DIR=str(runs_root), ROS_DOMAIN_ID=str(random.randint(100, 200)),
               PYTHONUNBUFFERED="1")
    started = time.time()
    cli = "import sys; from wardzero_agents.cli import main; sys.exit(main(sys.argv[1:]))"
    proc = subprocess.run(
        [sys.executable, "-c", cli, "run", str(scenario), "--no-panel", "--time-scale", "30"],
        env=env, cwd=AGENTS_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "TEST MODE: no AI provider" in proc.stdout + proc.stderr

    run = find_run(runs_root, started)
    meta = json.loads((run / "meta.json").read_text())
    cards, intents, says = read(run, "cards"), read(run, "intents"), read(run, "says")
    audio, fairness = read(run, "audio"), read(run, "fairness")

    assert meta["test_mode"] is True and meta["snapshot_in"].startswith("run-") and meta["snapshot_out"].endswith("-end")
    assert any(c["card"] == "clinic_delay" for c in cards) and any(c["card"] == "coughing_fit" for c in cards)
    assert any(r["kind"] == "cough" for r in intents if r["person"] == "P2")                        # reflex + bout
    assert any(s["person"] == "P3" for s in says)                                                  # companion asked for help (fake policy approaches robot on "help")
    assert any(a["source_kind"] == "robot" and "nurse" in (a["text"] or "") for a in audio)          # robot heard the coughs
    assert fairness == [] or all(v["node"] != "reception_bot" for v in fairness)
