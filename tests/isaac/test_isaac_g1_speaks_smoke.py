"""C1 smoke: in an Isaac run started from the Studio, the built-in G1 (fake model, no key) lives on the robot domain: a test node
there gets ``task/brief`` and a ``bbox_3d`` with a person in view, and the robot's ``say`` reaches the people (``audio.jsonl``,
``source_kind: robot``). Before the fix the robot process joined the people's domain, so it was deaf and mute.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_g1_speaks_smoke.py -q -s

Free path: WARDZERO_TEST_FAKE_LLM=1 (its robot greets once when a person is in view) and WARDZERO_TEST_CLAUDE_BOT=1 (the
Studio starts the built-in robot instead of the placeholder). One Kit process. Writes artifacts/screens/final/run_g1_speaks.png
and c1_smoke.json (git-ignored).
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

sys.path.insert(0, str(Path(__file__).parent))
from test_isaac_studio_e2e import H, _up, call, free_port, wait  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "screens" / "final"
PRIVATE_DOMAIN = 44
ROBOT_DOMAIN = PRIVATE_DOMAIN + 1  # the Studio's default: private + 1

# a robot-side test node: records what the robot domain carries; it says nothing (the built-in G1 does the talking)
LISTENER = textwrap.dedent('''
    import json, sys, time
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    out = sys.argv[1]
    rclpy.init()
    n = Node("test_listener")
    got = {"brief": None, "person_boxes": 0, "clock": 0, "robot_say_seen": []}
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def brief(m): got["brief"] = json.loads(m.data)
    def boxes(m):
        d = json.loads(m.data)
        d = d.get("detections", []) if isinstance(d, dict) else d
        if any(r.get("class_id") == "person" for b in d for r in b.get("results", [])):
            got["person_boxes"] += 1
    n.create_subscription(String, "/robot/task/brief", brief, latched)
    n.create_subscription(String, "/robot/camera/bbox_3d", boxes, qos)
    n.create_subscription(String, "/robot/speech/say", lambda m: got["robot_say_seen"].append(m.data), qos)
    n.create_subscription(Clock, "/clock", lambda m: got.__setitem__("clock", got["clock"] + 1), qos)
    def dump():
        got["topics"] = sorted(t for t, _ in n.get_topic_names_and_types())
        open(out, "w").write(json.dumps(got))
    end = time.time() + float(sys.argv[2])
    last = 0
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time() - last > 2:
            dump(); last = time.time()
    dump()
''')


def screenshot_when_robot_speaks(base, path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base + "/#/run")
        page.wait_for_function("() => { const i = document.getElementById('preview'); return i && !i.hidden && i.naturalWidth > 0; }",
                               timeout=120000)
        page.wait_for_function("() => [...document.querySelectorAll('.tail li b')].some((e) => e.textContent === 'Robot')",
                               timeout=120000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(path))
        b.close()


def test_the_built_in_g1_hears_sees_and_speaks_on_the_robot_domain_in_isaac(tmp_path):
    if subprocess.run(["pgrep", "-x", "kit"], capture_output=True).returncode == 0:
        pytest.skip("another Isaac (kit) process is running")
    data, runs, port = tmp_path / "data", tmp_path / "runs", free_port()
    runs.mkdir()
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_TEST_CLAUDE_BOT="1", WARDZERO_DATA_DIR=str(data),
               WARDZERO_RUNS_DIR=str(runs), ROS_DOMAIN_ID=str(PRIVATE_DOMAIN), PYTHONPATH=str(ROOT))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE", "WARDZERO_SIM",
              "WARDZERO_STUDIO_SIM", "WARDZERO_TEST_SCRIPTED_SAY"):
        env.pop(k, None)
    listener = tmp_path / "listener.py"
    listener.write_text(LISTENER)
    got_file = tmp_path / "listener.json"
    log = open(tmp_path / "studio.log", "w")
    studio = subprocess.Popen(
        ["bash", "-c", f'source "{ROOT.parent / "env.sh"}" >/dev/null 2>&1 && exec taskset -c 0-7,10-31 python -m '
                       f"wardzero_agents.cli studio --no-browser --port {port}"],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    side = None
    base = f"http://127.0.0.1:{port}"
    try:
        wait(lambda: _up(base), 60, "the Studio")
        assert call(base, "GET", "/api/connection")["robot_domain"] == ROBOT_DOMAIN
        side = subprocess.Popen(["bash", "-c", f'source "{ROOT.parent / "env.sh"}" >/dev/null 2>&1 && exec python "{listener}" "{got_file}" 1500'],
                                env=dict(os.environ, ROS_DOMAIN_ID=str(ROBOT_DOMAIN)), start_new_session=True)
        p = call(base, "POST", "/api/projects", {"name": "C1", "scenario_id": "reception_guide"})
        p = call(base, "PUT", f"/api/projects/{p['id']}", {"sim": "isaac", "headed": False, "record": False,
                                                           "duration_min": float(os.environ.get("WARDZERO_C1_MIN", "0.75"))})
        call(base, "POST", "/api/run/start", {"project_id": p["id"]})
        wait(lambda: (lambda s: s if s.get("preview_url") else None)(call(base, "GET", "/api/run/state")), 900, "the preview")
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            screenshot_when_robot_speaks(base, OUT / "run_g1_speaks.png")
        except Exception as exc:  # the evidence below is what counts; say why the picture is missing
            print("screenshot failed:", type(exc).__name__, str(exc)[:200])
        final = wait(lambda: (lambda s: s if s.get("phase") == "done" else None)(call(base, "GET", "/api/run/state")), 1500,
                     "the run to finish")
        assert final["exit_code"] == 0 and not final.get("error"), final
        run = runs / final["run_id"]
        time.sleep(3)
        got = json.loads(got_file.read_text())
        audio = [json.loads(x) for x in (run / "audio.jsonl").read_text().splitlines() if x.strip()]
        robot_lines = [a for a in audio if a.get("source_kind") == "robot" and a.get("text")]
        from wardzero_agents.llm.fake import ROBOT_GREETING
        evidence = {"brief_received": got["brief"], "person_boxes_seen": got["person_boxes"], "clock_msgs": got["clock"],
                    "robot_say_on_robot_domain": got["robot_say_seen"][:3],
                    "robot_lines_in_audio_jsonl": [{"t": a["t"], "text": a["text"], "addressee": a.get("addressee")}
                                                   for a in robot_lines][:5],
                    "private_topics_visible_to_robot": [t for t in got["topics"] if t.startswith(("/humans", "/scenario", "/sim", "/agents"))]}
        (OUT / "c1_smoke.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2))
        assert got["brief"], "the robot never got task/brief"
        assert got["person_boxes"] > 0, "no bbox_3d with a person reached the robot domain"
        assert any(ROBOT_GREETING in a["text"] for a in robot_lines), "the robot's say never reached the people"
        assert evidence["private_topics_visible_to_robot"] == []
    finally:
        try:
            call(base, "POST", "/api/run/stop")
        except Exception:
            pass
        if side is not None:
            try:
                os.killpg(side.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        os.killpg(studio.pid, signal.SIGINT)
        try:
            studio.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(studio.pid, signal.SIGKILL)
        log.close()
