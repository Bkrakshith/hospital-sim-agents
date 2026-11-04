"""Final-review fixes in the Studio: the report video, failures shown as errors, nothing editable during a run's debrief,
the scenario card that stays out of public capture, and the Isaac prerequisites."""
import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents import cli, isaac_assets
from wardzero_agents.cli import main
from wardzero_agents.studio.runner import Runner
from wardzero_agents.studio.server import Ctx, make_app

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}

# a fake `hhumans run`: prints the run folder, "Run finished", and then lingers in its debrief (no trust.jsonl yet)
DEBRIEFING = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    runs = Path(os.environ["WARDZERO_RUNS_DIR"]); pid = sys.argv[sys.argv.index("--project-id") + 1]
    d = runs / ("run-" + str(os.getpid())); d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "project_id": pid, "scenario_id": "ae_waiting"}))
    print("Run folder: " + str(d), flush=True)
    print("Run finished (exit code 0). Run folder: " + str(d), flush=True)
    time.sleep(30)
''')
CRASH = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    runs = Path(os.environ["WARDZERO_RUNS_DIR"]); pid = sys.argv[sys.argv.index("--project-id") + 1]
    d = runs / ("run-" + str(os.getpid())); d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "project_id": pid}))
    print("Run folder: " + str(d), flush=True)
    print("isaac_host exited with code 2", file=sys.stderr, flush=True)
    sys.exit(2)
''')


def make_env(tmp_path, monkeypatch, script_text):
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(data))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(runs))
    script = tmp_path / "fake_cli.py"
    script.write_text(script_text)
    runner = Runner(data, ROOT, runs, command=lambda p, s, b, d: [sys.executable, str(script), "--project-id", p.id],
                    stop_publisher=lambda d: False, key_present=lambda: False)
    return data, runs, runner


@pytest.fixture
async def studio(tmp_path, monkeypatch, request):
    data, runs, runner = make_env(tmp_path, monkeypatch, getattr(request, "param", DEBRIEFING))
    c = TestClient(TestServer(make_app(Ctx(data, ROOT, runs, runner))))
    await c.start_server()
    c.runs, c.runner, c.data = runs, runner, data
    yield c
    await asyncio.to_thread(runner.shutdown)
    await c.close()


async def mkproject(c):
    r = await c.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting"}, headers=H)
    return await r.json()


async def until(fn, timeout=15.0):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        v = await fn()
        if v:
            return v
        await asyncio.sleep(0.1)
    raise AssertionError("timed out")


async def state(c):
    return await (await c.get("/api/run/state")).json()


# ---- I2: the report video plays in the Studio -----------------------------------------------------------
async def test_the_reports_video_url_serves_mp4_with_ranges(studio):
    rd = studio.runs / "run-a"
    (rd / "video").mkdir(parents=True)
    (rd / "video" / "overhead.mp4").write_bytes(bytes(range(256)) * 40)
    (rd / "video" / "secret.mp4").write_bytes(b"x")
    (rd / "report.html").write_text('<html><body><video controls src="video/overhead.mp4"></video></body></html>')
    page = await studio.get("/runs/run-a/report.html")
    assert "media-src 'self'" in page.headers["Content-Security-Policy"]
    assert "video/overhead.mp4" in await page.text()
    url = "/runs/run-a/video/overhead.mp4"  # what the page's relative src resolves to
    r = await studio.get(url)
    assert r.status == 200 and r.headers["Content-Type"] == "video/mp4" and r.headers["X-Content-Type-Options"] == "nosniff"
    assert len(await r.read()) == 256 * 40
    part = await studio.get(url, headers={"Range": "bytes=10-19"})
    assert part.status == 206 and await part.read() == bytes(range(10, 20))
    assert part.headers["Content-Range"] == "bytes 10-19/10240"
    assert (await studio.get("/runs/run-a/video/secret.mp4")).status == 404  # allow-list
    assert (await studio.get("/runs/run-a/video/follow.mp4")).status == 404  # allowed name, no file
    assert (await studio.get("/runs/..%2Fx/video/overhead.mp4")).status == 404


# ---- I3: a crash after the run folder exists is an error -----------------------------------------------
@pytest.mark.parametrize("studio", [CRASH], indirect=True)
async def test_a_failure_after_the_folder_exists_is_shown_with_its_exit_code(studio):
    p = await mkproject(studio)
    assert (await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202
    s = await until(lambda: _done(studio))
    assert s["exit_code"] == 2 and "exit code 2" in s["error"] and "isaac_host exited with code 2" in s["error"]


async def _done(c):
    s = await state(c)
    return s if s["phase"] == "done" else None


# ---- I4: nothing changes under a run's debrief ----------------------------------------------------------
async def test_reset_edit_and_delete_are_refused_while_the_run_debriefs(studio):
    p = await mkproject(studio)
    cid = p["characters"][0]["id"]
    assert (await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202
    s = await until(lambda: _phase(studio, "debrief"))
    assert s["running"]
    refused = [
        await studio.post(f"/api/projects/{p['id']}/memory/reset", json={"characters": "all"}, headers=H),
        await studio.delete(f"/api/projects/{p['id']}", headers=H),
        await studio.put(f"/api/projects/{p['id']}/characters/{cid}", json={"name": "Zed"}, headers=H),
        await studio.delete(f"/api/projects/{p['id']}/characters/{cid}", headers=H),
        await studio.post(f"/api/projects/{p['id']}/characters", json={"name": "New", "role": "visitor", "mood": "calm", "x": 0, "y": 0}, headers=H),
        await studio.put(f"/api/projects/{p['id']}", json={"characters": []}, headers=H),
    ]
    assert [r.status for r in refused] == [409] * 6
    assert (await studio.get(f"/api/projects/{p['id']}")).status == 200  # still there, unchanged
    # a second Start during the debrief is refused too
    again = await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert again.status == 409
    # other settings (not characters) can still be saved
    assert (await studio.put(f"/api/projects/{p['id']}", json={"budget_usd": 3}, headers=H)).status == 200


async def _phase(c, name):
    s = await state(c)
    return s if s["phase"] == name else None


async def test_edits_work_when_no_run_is_going(studio):
    p = await mkproject(studio)
    r = await studio.post(f"/api/projects/{p['id']}/memory/reset", json={"characters": "all"}, headers=H)
    assert r.status == 200


# ---- I7: the cost bar is for the whole Start ------------------------------------------------------------
def test_the_cost_adds_up_across_the_run_folders_of_one_start(tmp_path, monkeypatch):
    data, runs, runner = make_env(tmp_path, monkeypatch, "")
    from wardzero_agents.studio.project import Project
    runner._project = Project(id="p1", name="n", scenario_id="ae_waiting", budget_usd=5)
    for n, cost in (("r1", 0.4), ("r2", 0.3)):
        d = runs / n
        d.mkdir()
        (d / "tokens.jsonl").write_text(json.dumps({"person": "P1", "cost_usd": cost}) + "\n")
    runner._folder = str(runs / "r1")
    runner._follow()
    assert runner.state()["cost_usd"] == pytest.approx(0.4)
    (runs / "r1" / "tokens.jsonl").write_text(
        (runs / "r1" / "tokens.jsonl").read_text() + json.dumps({"person": "report", "cost_usd": 0.1}) + "\n")  # its debrief
    runner._folder = str(runs / "r2")
    s = runner.state()
    assert s["cost_usd"] == pytest.approx(0.4 + 0.1 + 0.3) and s["budget_usd"] == 5


# ---- I8: the scenario card that stays out of public capture -------------------------------------------
async def test_the_preset_is_listed_last_and_hidden_in_public_capture(studio, monkeypatch):
    names = [s["id"] for s in (await (await studio.get("/api/presets")).json())["scenarios"]]
    assert names[-1] == "scrub_nurse" and names.count("scrub_nurse") == 1 and len(names) > 3
    assert "scrub_nurse" not in names[:-1]
    monkeypatch.setenv("WARDZERO_STUDIO_PUBLIC_CAPTURE", "1")
    cap = [s["id"] for s in (await (await studio.get("/api/presets")).json())["scenarios"]]
    assert "scrub_nurse" not in cap and cap == names[:-1]
    # compile and tests still reach it by id
    from wardzero_agents.config.presets import load_scenario_presets
    assert load_scenario_presets(ROOT)["scrub_nurse"].listed_in_public_capture is False


def test_no_public_text_mentions_the_preset():
    agents = Path(__file__).resolve().parents[1]
    for f in [agents / "README.md", *(agents / "docs").glob("*.md")]:
        assert "scrub" not in f.read_text(encoding="utf-8").lower(), f


# ---- I9: Isaac prerequisites ------------------------------------------------------------------------
@pytest.fixture
def isaac_root(tmp_path, monkeypatch):
    root = tmp_path / "isaacsim"
    root.mkdir()
    monkeypatch.setattr(cli, "ISAAC_ROOT", root)
    monkeypatch.setenv("WARDZERO_STAGE_USD", str(tmp_path / "stage.usd"))
    monkeypatch.setenv("WARDZERO_COLLISION_USD", str(tmp_path / "collision.usd"))
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    return root


def test_isaac_installed_needs_python_sh(isaac_root):
    assert not cli.isaac_installed()  # a folder is not an install
    (isaac_root / "python.sh").write_text("#!/bin/sh\n")
    assert cli.isaac_installed()


def test_the_isaac_start_fails_with_exit_2_naming_the_fix(isaac_root, capsys):
    (isaac_root / "python.sh").write_text("#!/bin/sh\n")
    V1 = str(Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml")
    assert main(["run", V1, "--sim", "isaac"]) == 2
    err = capsys.readouterr().err
    assert "stage" in err and "build_collision_usd.py" in err and "WARDZERO_COLLISION_USD" in err


def test_the_paths_can_be_overridden_and_complete_assets_pass(isaac_root, tmp_path):
    (isaac_root / "python.sh").write_text("")
    assert len(isaac_assets.missing_for_isaac(isaac_root)) == 2
    (tmp_path / "stage.usd").write_text("x")
    (tmp_path / "collision.usd").write_text("x")
    assert isaac_assets.missing_for_isaac(isaac_root) == []
    assert isaac_assets.collision_usd() == tmp_path / "collision.usd"


def test_a_missing_env_sh_is_a_warning_in_the_run_log(isaac_root, tmp_path, monkeypatch, capsys):
    (isaac_root / "python.sh").write_text("")
    (tmp_path / "stage.usd").write_text("x")
    (tmp_path / "collision.usd").write_text("x")
    monkeypatch.setenv("WARDZERO_ROS_ENV_SH", str(tmp_path / "nope.sh"))
    args = cli.build_parser().parse_args(["run", "x.yaml", "--sim", "isaac", "--record"])
    cli._resolve_sim(args)
    assert "no rosbag will be recorded" in capsys.readouterr().out


async def test_the_studio_says_what_isaac_is_missing(tmp_path, monkeypatch, isaac_root):
    (isaac_root / "python.sh").write_text("")
    data, runs, runner = make_env(tmp_path, monkeypatch, "")
    c = TestClient(TestServer(make_app(Ctx(data, ROOT, runs, runner))))
    await c.start_server()
    try:
        pr = await (await c.get("/api/presets")).json()
        assert pr["isaac_installed"] and not pr["isaac_available"]
        assert any("collision USD is missing" in m and "build_collision_usd.py" in m for m in pr["isaac_missing"])
        r = await c.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting", "sim": "isaac"}, headers=H)
        assert r.status == 400 and "collision USD" in (await r.json())["error"]
    finally:
        await c.close()


# ---- minors --------------------------------------------------------------------------------------------
async def test_a_deleted_robot_profile_is_a_400_at_start(studio):
    p = await mkproject(studio)
    f = studio.data / "studio" / "projects" / p["id"] / "project.yaml"
    f.write_text(f.read_text().replace("robot_profile: g1_claude", "robot_profile: gone_robot"))
    r = await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 400 and "gone_robot" in (await r.json())["error"]


def test_the_convert_and_probe_children_get_no_api_key(monkeypatch):
    from wardzero_agents.studio.server import keyless_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-not-a-key")
    env = keyless_env(ROS_DOMAIN_ID="5")
    assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env and env["ROS_DOMAIN_ID"] == "5"


def test_a_port_in_use_is_a_clean_message(capsys):
    import socket
    from wardzero_agents.studio.server import serve
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert serve(port, open_browser=False) == 2
    finally:
        s.close()
    assert f"Port {port} is already in use" in capsys.readouterr().err


def test_the_scripted_say_hook_warns_loudly(tmp_path, monkeypatch, caplog):
    from wardzero_agents.ros import sim_service
    f = tmp_path / "say.json"
    f.write_text("[]")
    monkeypatch.setenv("WARDZERO_TEST_SCRIPTED_SAY", str(f))
    with caplog.at_level("WARNING"):
        sim_service._scripted_lines()
    assert "TEST HOOK ACTIVE" in caplog.text
