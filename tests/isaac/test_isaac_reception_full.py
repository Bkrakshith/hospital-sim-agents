"""GPU: the whole Reception and guide scenario in Isaac (design §15), recorded, free (fake LLM, scripted lines).

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_reception_full.py -q -s
    WARDZERO_FULL_OUT=/big/disk/folder ...   # where the run and the assembled film go (default: pytest's tmp folder)

A `hhumans run` of Reception and guide with the receptionist (RCP, in her chair at R1), the walking G1 (GEAR-SONIC),
hinged doors with closers on the routes, one user-drawn forbidden area beside the route to the ward, the people's
fake minds and the built-in robot's fake mind (WARDZERO_TEST_CLAUDE_BOT=1), with WARDZERO_TEST_SCRIPTED_SAY:
  1. Priya walks up and asks for a nurse: the G1 tells the receptionist (tell_staff), who gets up, goes to Priya,
     speaks to her and goes back to her chair;
  2. Daniel asks for the ward: the G1 guides him through the C2C3_East doors to ward_entrance, he follows and arrives,
     and the G1 walks back to its post;
  3. Arthur asks the way twice (hard of hearing) and is answered both times.
Pass: no G1 fall, no non-finite door leaf, nobody inside a leaf, the report's reached / unanswered / contacts pass, the
forbidden area never entered. ``check_full(run)`` checks a kept run folder too; ``film(run, out)`` assembles the video.
"""
import json
import math
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
HG1 = Path(os.environ.get("HG1_ROOT") or ROOT.parent)
sys.path.insert(0, str(Path(__file__).parent))
from test_isaac_record_smoke import ffmpeg_exe, probe  # noqa: E402

MINUTES = 5.0
TIMEOUT_S = 5400
AREA = {"id": "a1", "name": "Wet floor", "x0": 8.0, "y0": 4.6, "x1": 9.5, "y1": 5.5}
SCRIPT = [
    {"t": 2.0, "who": "V1", "do": "approach", "target": "g1"},
    {"t": 9.0, "who": "V1", "text": "Excuse me, I'm Priya. I need a nurse, please.", "to": "g1"},
    # Daniel asks from where he stands: a person the script moves ignores their own mind from then on, and his mind
    # is what accepts the robot's offer and follows it
    {"t": 66.0, "who": "V2", "text": "Hello, could you take me to the ward, please?", "to": "g1"},
    {"t": 205.0, "who": "H1", "do": "approach", "target": "g1"},
    {"t": 215.0, "who": "H1", "text": "Excuse me, where are the toilets?", "to": "g1"},
    {"t": 219.0, "who": "robot", "text": "The toilets are just past the lifts, on your left.", "to": "H1"},
    {"t": 226.0, "who": "H1", "text": "Sorry, I didn't catch that. Where did you say?", "to": "g1"},
    {"t": 230.0, "who": "robot", "text": "The toilets. Past the lifts, then on your left. I can show you if you like.",
     "to": "H1"},
]


def rows(run: Path, name: str) -> list:
    f = run / f"{name}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []


def stage(tmp: Path):
    from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
    from wardzero_agents.studio import auto_staff
    pre = load_scenario_presets(ROOT / "config")["reception_guide"]
    chars = auto_staff.with_auto(list(pre.characters), pre)
    loaded = preset_to_scenario(pre, (ROOT / "config").resolve(), tmp / "scen", characters=chars,
                                scenario_extra={"duration_min": MINUTES, "forbidden_areas": [AREA]})
    return pre, loaded


def check_full(run: Path) -> dict:
    """The §15 table for a run folder: {check: (passed, detail)}."""
    from wardzero_agents.bench.report import build_report
    from wardzero_agents.config.presets import load_scenario_presets
    pre = load_scenario_presets(ROOT / "config")["reception_guide"]
    host = rows(run, "host")
    walker = ([r for r in host if r.get("kind") == "walker"] or [{}])[-1]
    doors = ([r for r in host if r.get("kind") == "doors"] or [{}])[-1]
    tasks = rows(run, "robot_tasks")
    status = rows(run, "status")
    audio = rows(run, "audio")
    out = {}
    out["g1_no_fall"] = (walker.get("fell") is None and bool(walker), f"max tilt {walker.get('max_tilt_deg')} deg")
    out["leaves_finite"] = (doors.get("nonfinite_steps") == 0 and doors.get("past_limit_steps") == 0,
                            f"doors {doors.get('doors')}, max {doors.get('max_abs_deg')} deg, "
                            f"{doors.get('max_rad_s')} rad/s")
    out["nobody_through_a_leaf"] = (not doors.get("through_leaf") and "through_leaf" in doors,
                                    f"deepest {doors.get('max_penetration_m')} m")
    told = [r for r in tasks if r.get("action") == "tell_staff" and "reason" in r]
    out["tell_staff"] = (bool(told) and told[0]["ok"], str([(r["reason"], round(r["t"], 1)) for r in told]))
    pos = {}
    for s in status:
        pos.setdefault(s["person"], []).append((s["t"], s["x"], s["y"], s["posture"]))
    t_told = told[0]["t"] if told else 1e9
    v1 = {round(t, 2): (x, y) for t, x, y, _ in pos.get("V1", [])}
    near = [t for t, x, y, _ in pos.get("RCP", []) if t > t_told and round(t, 2) in v1
            and math.hypot(x - v1[round(t, 2)][0], y - v1[round(t, 2)][1]) <= 1.8]
    spoke = [a for a in audio if a.get("source_id") == "RCP" and a.get("addressee") == "V1"]
    back = [t for t, x, y, p in pos.get("RCP", []) if near and t > near[0] and p == "seated"
            and math.hypot(x - 2.0, y - 4.75) < 0.6]
    out["receptionist_errand"] = (bool(near and spoke and back),
                                  f"reached Priya {near[:1]}, spoke {len(spoke)}, back seated {back[:1]}")
    gs = [r for r in tasks if r.get("kind") == "guide"]
    start = [r for r in gs if r.get("event") == "start"]
    end = [r for r in gs if r.get("event") == "end"]
    through = any(x > 11.3 for t, x, y, p in pos.get("V2", []))
    out["guide_daniel"] = (bool(start and end and end[0].get("reason") == "arrived" and through),
                           f"start {[(r.get('person'), r.get('place')) for r in start]}, end "
                           f"{[r.get('reason') for r in end]}, through the doors {through}")
    ret = [r for r in tasks if str(r.get("id", "")).endswith("-return") and "reason" in r]
    out["g1_back_at_post"] = (bool(ret) and ret[-1].get("ok"), str([(r.get("reason"), round(r["t"], 1)) for r in ret]))
    arthur = [a for a in audio if a.get("source_kind") == "robot" and a.get("addressee") == "H1"]
    out["arthur_answered_twice"] = (len(arthur) >= 2, f"{len(arthur)} robot lines to Arthur")
    build_report(run, pre, None)
    rep = json.loads((run / "report.json").read_text())
    crit = {c["id"]: c for c in rep["criteria"]}
    for cid in ("reached", "unanswered", "contacts"):
        out[f"report_{cid}"] = (crit[cid]["passed"] is True, f"value {crit[cid].get('value')}")
    entries = [e for e in rep.get("forbidden_entries", []) if e.get("kind") == "user"]
    out["forbidden_area_never_entered"] = (not entries, f"{len(entries)} entries")
    return out


def film(run: Path, out: Path) -> Path:
    """follow.mp4 full frame with overhead.mp4 as a picture-in-picture at the top right, 1280x720, H.264."""
    v = run / "video"
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(v / "follow.mp4"), "-i", str(v / "overhead.mp4"),
           "-filter_complex", "[0:v]scale=1280:720[a];[1:v]scale=448:252[b];[a][b]overlay=W-w-12:12[o]",
           "-map", "[o]"]
    voice = v / "voice.wav"
    if voice.is_file():
        cmd += ["-i", str(voice), "-map", "2:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return out


def test_reception_and_guide_in_isaac(tmp_path):
    from wardzero_agents.isaac_host.sonic.weights import missing_files
    if missing_files():
        pytest.skip("no GEAR-SONIC weights (scripts/fetch_sonic.py or WARDZERO_SONIC_DIR)")
    if subprocess.run(["pgrep", "-x", "kit"], capture_output=True).returncode == 0:
        pytest.skip("another Isaac (kit) process is running")
    pre, loaded = stage(tmp_path)
    script = tmp_path / "script.json"
    script.write_text(json.dumps(SCRIPT))
    out = Path(os.environ.get("WARDZERO_FULL_OUT") or tmp_path / "runs")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_TEST_CLAUDE_BOT="1",
               WARDZERO_TEST_SCRIPTED_SAY=str(script), WARDZERO_RUNS_DIR=str(out),
               ROS_DOMAIN_ID=os.environ.get("ROS_DOMAIN_ID", "57"), PYTHONPATH=str(ROOT), WARDZERO_G1_WALKS_IN_ISAAC="1")
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE"):
        env.pop(k, None)
    log_path = out / f"full_{int(time.time())}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        code = subprocess.run(
            ["bash", "-c", f'source "{HG1 / "env.sh"}" >/dev/null 2>&1; '
                           f'exec taskset -c 0-7,10-31 python -m wardzero_agents.cli run "{loaded.scenario_path}" '
                           f"--sim isaac --record --headless --no-panel --no-report --bot claude "
                           f"--record-res 1280x720"],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=TIMEOUT_S).returncode
    text = log_path.read_text()
    print(f"\nrun wall time {time.time() - t0:.0f} s, log {log_path}")
    m = re.search(r"Run folder: (.+)", text)
    assert m, text[-3000:]
    run = Path(m.group(1).strip())
    assert code == 0, text[-3000:]
    table = check_full(run)
    print(json.dumps({k: [bool(v[0]), v[1]] for k, v in table.items()}, indent=2))
    for name in ("overhead.mp4", "follow.mp4"):
        assert probe(run / "video" / name)["duration"] > 0.9 * MINUTES * 60, name
    film(run, run / "video" / "reception_guide_film.mp4")
    assert all(ok for ok, _ in table.values()), {k: v for k, v in table.items() if not v[0]}
