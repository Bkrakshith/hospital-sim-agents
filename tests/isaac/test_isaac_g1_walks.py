"""GPU: the built-in G1 walks in a recorded Isaac run of Reception and guide (GEAR-SONIC in the host).

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_g1_walks.py -q -s
    WARDZERO_WALK_OUT=/big/disk/folder ...   # where the run goes (default: pytest's tmp folder)

A whole `hhumans run` (fake LLM people, the placeholder robot mind, no key, no spend) of a 3-minute Reception and guide
with WARDZERO_TEST_SCRIPTED_SAY sending the G1 `go_to ward_doors` at 6 s and back to its post at 75 s. It asserts the
walk: both goals end `arrived` on the goal-result path, the G1 never falls, physics ran 8 steps a 25 fps frame, and
both videos exist. A few frames of follow.mp4 go to artifacts/screens/isaac_walk/ (git-ignored). Needs Isaac Sim 6.0,
the GEAR-SONIC weights (scripts/fetch_sonic.py or WARDZERO_SONIC_DIR) and ROS 2 (env.sh). One Kit process.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
HG1 = Path(os.environ.get("HG1_ROOT") or ROOT.parent)  # the Hospital_G1 checkout (env.sh); a worktree is elsewhere
sys.path.insert(0, str(Path(__file__).parent))
from test_isaac_record_smoke import ffmpeg_exe, probe  # noqa: E402

OUT = ROOT / "artifacts" / "screens" / "isaac_walk"
SCRIPT = [
    {"t": 4.0, "who": "robot", "text": "I will walk to the ward doors and come back."},
    {"t": 6.0, "who": "robot", "do": "go_to", "place": "ward_doors"},
    {"t": 75.0, "who": "robot", "do": "go_to", "place": "post"},
]
MINUTES = 3.0
TIMEOUT_S = 3600


def rows(run: Path, name: str) -> list:
    f = run / f"{name}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []


def test_the_g1_walks_to_the_ward_doors_and_back_in_a_recorded_run(tmp_path):
    from wardzero_agents.isaac_host.sonic.weights import missing_files
    if missing_files():
        pytest.skip("no GEAR-SONIC weights (scripts/fetch_sonic.py or WARDZERO_SONIC_DIR)")
    if subprocess.run(["pgrep", "-x", "kit"], capture_output=True).returncode == 0:
        pytest.skip("another Isaac (kit) process is running")
    from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
    pre = load_scenario_presets(ROOT / "config")["reception_guide"]
    scen = preset_to_scenario(pre, (ROOT / "config").resolve(), tmp_path / "scen",
                              scenario_extra={"duration_min": MINUTES}).scenario_path
    script = tmp_path / "script.json"
    script.write_text(json.dumps(SCRIPT))
    runs = Path(os.environ.get("WARDZERO_WALK_OUT") or tmp_path / "runs")
    runs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_TEST_SCRIPTED_SAY=str(script),
               WARDZERO_RUNS_DIR=str(runs), ROS_DOMAIN_ID=os.environ.get("ROS_DOMAIN_ID", "57"),
               PYTHONPATH=str(ROOT), WARDZERO_G1_WALKS_IN_ISAAC="1")
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE",
              "WARDZERO_TEST_CLAUDE_BOT"):
        env.pop(k, None)
    log_path = runs / f"walk_{int(time.time())}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        code = subprocess.run(
            ["bash", "-c", f'source "{HG1 / "env.sh"}" >/dev/null 2>&1; '
                           f'exec taskset -c 0-7,10-31 python -m wardzero_agents.cli run "{scen}" '
                           f"--sim isaac --record --headless --no-panel --no-report --no-voice --bot reception "
                           f"--record-res 960x540"],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=TIMEOUT_S).returncode
    text = log_path.read_text()
    print(f"\nrun wall time {time.time() - t0:.0f} s, log {log_path}")
    m = re.search(r"Run folder: (.+)", text)
    assert m, text[-3000:]
    run = Path(m.group(1).strip())
    assert code == 0, text[-3000:]

    check_walk(run)
    OUT.mkdir(parents=True, exist_ok=True)
    for t in (10, 30, 60, 90, 120):
        subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-ss", str(t), "-i", str(run / "video" / "follow.mp4"),
                        "-frames:v", "1", str(OUT / f"follow_{t:03d}s.png")], check=False)
    shutil.copy(log_path, OUT / "run.log")


def check_walk(run: Path) -> dict:
    """The walk's evidence from a run folder (also usable on a kept run: check_walk(Path(...)))."""
    goals = [r for r in rows(run, "robot_tasks") if r.get("action") == "goal" and "reason" in r]
    walker = [r for r in rows(run, "host") if r.get("kind") == "walker"]
    final = [r for r in rows(run, "host") if r.get("kind") == "final"]
    fell = [r for r in rows(run, "world") if r.get("kind") == "robot_fell"]
    evidence = {"run": str(run), "goals": goals, "walker": walker[-1] if walker else None,
                "final": final[-1] if final else None, "fell": fell}
    print(json.dumps(evidence, indent=2, default=str))
    assert not fell and walker and walker[-1]["fell"] is None, "the G1 fell"
    assert walker[-1]["steps_per_frame"] == {"min": 8, "max": 8, "median": 8.0}
    assert [g["reason"] for g in goals] == ["arrived", "arrived"] and all(g["ok"] for g in goals), goals
    for name in ("overhead.mp4", "follow.mp4"):
        mp4 = run / "video" / name
        assert mp4.is_file() and probe(mp4)["duration"] > 60, name
    return evidence
