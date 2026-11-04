"""End to end (I5 fix 1): the Studio starts a headless, recording Isaac run from its API, the Run page shows the live
preview, the run finishes and leaves videos, a rosbag and a report. Free path only: WARDZERO_TEST_FAKE_LLM=1 (the minds,
the robot and the debrief use the fake gateway, never a paid call).

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_studio_e2e.py -q -s

Starts ``hhumans studio --no-browser`` as a real process (system ROS sourced from env.sh), one Kit process, and headless
Chrome for the screenshot (artifacts/screens/isaac_record/studio_run.png, git-ignored). Takes several minutes.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "screens" / "isaac_record"
H = {"X-WardZero-Panel": "1", "Content-Type": "application/json"}
PRIVATE_DOMAIN = 44
PREVIEW_TIMEOUT_S, FINISH_TIMEOUT_S = 900, 1500
RECORD = os.environ.get("WARDZERO_E2E_RECORD", "1") == "1"  # 0: to tell a recording problem from a run problem


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(base, method, path, body=None):
    req = urllib.request.Request(base + path, method=method, headers=H,
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


def wait(fn, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(1.0)
    raise AssertionError(f"timed out waiting for {what}")


def read_frames(url, n=2):
    """The first n distinct JPEGs of an MJPEG stream."""
    out, buf = [], b""
    with urllib.request.urlopen(url, timeout=30) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        while len(out) < n:
            chunk = r.read1(65536)
            assert chunk, "the stream ended"
            buf += chunk
            while b"\xff\xd8" in buf and b"\xff\xd9" in buf[buf.index(b"\xff\xd8"):]:
                i = buf.index(b"\xff\xd8")
                j = buf.index(b"\xff\xd9", i) + 2
                jpg, buf = buf[i:j], buf[j:]
                if not out or jpg != out[-1]:
                    out.append(jpg)
    return out


def screenshot(base, path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1280, "height": 800})
        page.goto(base + "/#/run")
        page.wait_for_function("() => { const i = document.getElementById('preview'); return i && !i.hidden && i.naturalWidth > 0; }",
                               timeout=60000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(path))
        b.close()


def test_studio_runs_isaac_headless_with_recording_and_a_live_preview(tmp_path):
    if subprocess.run(["pgrep", "-x", "kit"], capture_output=True).returncode == 0:
        pytest.skip("another Isaac (kit) process is running")
    data, runs, port = tmp_path / "data", tmp_path / "runs", free_port()
    runs.mkdir()  # default Studio settings: no connection.yaml, so the robot gets its own default domain
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_DATA_DIR=str(data), WARDZERO_RUNS_DIR=str(runs),
               ROS_DOMAIN_ID=str(PRIVATE_DOMAIN), PYTHONPATH=str(ROOT))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE", "WARDZERO_SIM",
              "WARDZERO_STUDIO_SIM"):
        env.pop(k, None)
    log = open(tmp_path / "studio.log", "w")
    studio = subprocess.Popen(
        ["bash", "-c", f'source "{ROOT.parent / "env.sh"}" >/dev/null 2>&1 && exec taskset -c 0-7,10-31 python -m '
                       f"wardzero_agents.cli studio --no-browser --port {port}"],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    base = f"http://127.0.0.1:{port}"
    try:
        wait(lambda: _up(base), 60, "the Studio")
        assert call(base, "GET", "/api/presets")["isaac_available"] is True
        conn = call(base, "GET", "/api/connection")
        assert conn["explicit"] is False and conn["robot_domain"] == PRIVATE_DOMAIN + 1
        p = call(base, "POST", "/api/projects", {"name": "E2E", "scenario_id": "reception_guide"})
        p = call(base, "PUT", f"/api/projects/{p['id']}", {"sim": "isaac", "headed": False, "record": RECORD,
                                                           "record_res": "1280x720", "duration_min": 0.25})
        call(base, "POST", "/api/run/start", {"project_id": p["id"]})
        state = wait(lambda: (lambda s: s if s.get("preview_url") else None)(call(base, "GET", "/api/run/state")),
                     PREVIEW_TIMEOUT_S, "preview_url in the run state")
        frames = read_frames(base + state["preview_url"])
        assert len(frames) == 2 and all(f[:2] == b"\xff\xd8" for f in frames)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "studio_preview_frame.jpg").write_bytes(frames[-1])
        screenshot(base, OUT / "studio_run.png")
        final = wait(lambda: (lambda s: s if s.get("phase") == "done" else None)(call(base, "GET", "/api/run/state")),
                     FINISH_TIMEOUT_S, "the run to finish")
        assert final["exit_code"] == 0 and final["has_report"], final
        run = runs / final["run_id"]
        for f in (("video/overhead.mp4", "video/follow.mp4", "robot_bag/metadata.yaml") if RECORD else ()) + ("report.html",):
            assert (run / f).is_file() and (run / f).stat().st_size > 0, f
        counts = bag_counts(run / "robot_bag")
        print("bag messages:", json.dumps(counts))
        assert counts.get("/clock", 0) > 0 and counts.get("/tf", 0) > 0, counts  # recorded with no user setting
        assert "preview_url" not in call(base, "GET", "/api/run/state")  # the stream ended with the run
        assert (OUT / "studio_run.png").stat().st_size > 20_000
    finally:
        try:
            call(base, "POST", "/api/run/stop")
        except Exception:
            pass
        os.killpg(studio.pid, signal.SIGINT)
        try:
            studio.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(studio.pid, signal.SIGKILL)
        log.close()


def bag_counts(bag: Path) -> dict:
    import sqlite3
    db = next(iter(sorted(bag.glob("*.db3"))))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return dict(con.execute("select t.name, count(*) from messages m join topics t on t.id = m.topic_id group by t.id"))
    finally:
        con.close()


def _up(base):
    try:
        urllib.request.urlopen(base + "/api/presets", timeout=2).read()
        return True
    except Exception:
        return False
