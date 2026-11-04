"""Simulator controls: saved per project, validated by the API, turned into the right `hhumans run` flags."""
import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents.studio import runner as runner_mod
from wardzero_agents.studio.project import Project
from wardzero_agents.studio.runner import Runner
from wardzero_agents.studio.server import Ctx, make_app

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}


def argv_for(tmp_path, monkeypatch, **fields):
    monkeypatch.delenv("WARDZERO_STUDIO_SIM", raising=False)
    p = Project(id="p1", name="n", scenario_id="reception_guide", **fields)
    return runner_mod.default_command(p, tmp_path, "none", 43)


def opt(argv, flag):
    return argv[argv.index(flag) + 1]


def test_defaults_are_the_quick_2d_sim(tmp_path, monkeypatch):
    a = argv_for(tmp_path, monkeypatch)
    assert opt(a, "--sim") == "stub"
    assert not {"--headed", "--headless", "--record", "--preview", "--record-res"} & set(a)


@pytest.mark.parametrize("fields,flags,absent", [
    (dict(sim="isaac"), ["--headless", "--preview"], ["--headed", "--record"]),
    (dict(sim="isaac", headed=True), ["--headed"], ["--headless", "--preview", "--record"]),  # the window is the preview
    (dict(sim="isaac", record=True), ["--headless", "--preview", "--record"], ["--headed"]),
    (dict(sim="isaac", headed=True, record=True), ["--headed", "--record"], ["--headless", "--preview"]),
])
def test_isaac_argv(tmp_path, monkeypatch, fields, flags, absent):
    a = argv_for(tmp_path, monkeypatch, **fields)
    assert opt(a, "--sim") == "isaac"
    assert set(flags) <= set(a) and not set(absent) & set(a)


def test_record_resolution_is_passed_only_when_recording(tmp_path, monkeypatch):
    a = argv_for(tmp_path, monkeypatch, sim="isaac", record=True, record_res="1920x1080")
    assert opt(a, "--record-res") == "1920x1080"
    assert "--record-res" not in argv_for(tmp_path, monkeypatch, sim="isaac", record_res="1920x1080")


def test_the_stub_ignores_isaac_only_settings(tmp_path, monkeypatch):
    a = argv_for(tmp_path, monkeypatch, sim="stub", headed=True, record=True)
    assert opt(a, "--sim") == "stub" and not {"--headed", "--record", "--preview"} & set(a)  # the CLI would refuse them


def test_the_environment_still_overrides_the_project(tmp_path, monkeypatch):
    p = Project(id="p1", name="n", scenario_id="reception_guide", sim="stub")
    monkeypatch.setenv("WARDZERO_STUDIO_SIM", "isaac")
    assert opt(runner_mod.default_command(p, tmp_path, "none", None), "--sim") == "isaac"


@pytest.fixture
async def client(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    state = {"isaac": True}
    runner = Runner(data, ROOT, runs, command=lambda *a: ["true"], stop_publisher=lambda d: False, key_present=lambda: False)
    c = TestClient(TestServer(make_app(Ctx(data, ROOT, runs, runner, isaac_installed=lambda: state["isaac"]))))
    await c.start_server()
    c.state = state
    yield c
    await asyncio.to_thread(runner.shutdown)
    await c.close()


async def new_project(c):
    r = await c.post("/api/projects", json={"name": "T", "scenario_id": "reception_guide"}, headers=H)
    assert r.status == 201
    return await r.json()


async def test_a_new_project_has_the_defaults_and_the_page_learns_whether_isaac_is_there(client):
    p = await new_project(client)
    assert (p["sim"], p["headed"], p["record"], p["record_res"]) == ("stub", False, False, "1280x720")
    assert (await (await client.get("/api/presets")).json())["isaac_available"] is True
    client.state["isaac"] = False
    assert (await (await client.get("/api/presets")).json())["isaac_available"] is False


async def test_the_settings_are_saved_per_project(client):
    p = await new_project(client)
    r = await client.put(f"/api/projects/{p['id']}", json={"sim": "isaac", "headed": True, "record": True,
                                                           "record_res": "1920x1080"}, headers=H)
    assert r.status == 200
    got = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert (got["sim"], got["headed"], got["record"], got["record_res"]) == ("isaac", True, True, "1920x1080")


@pytest.mark.parametrize("patch", [{"sim": "unreal"}, {"record_res": "640x480"}, {"headed": "yes"}, {"record": 3}])
async def test_bad_values_are_refused(client, patch):
    p = await new_project(client)
    r = await client.put(f"/api/projects/{p['id']}", json=patch, headers=H)
    assert r.status == 400


async def test_isaac_is_refused_when_it_is_not_installed(client):
    p = await new_project(client)
    client.state["isaac"] = False
    r = await client.put(f"/api/projects/{p['id']}", json={"sim": "isaac"}, headers=H)
    assert r.status == 400 and "Isaac Sim is not installed" in (await r.json())["error"]
    assert (await client.put(f"/api/projects/{p['id']}", json={"sim": "stub"}, headers=H)).status == 200
    # a project saved while Isaac was there, started after it went: the run refuses too
    client.state["isaac"] = True
    await client.put(f"/api/projects/{p['id']}", json={"sim": "isaac"}, headers=H)
    client.state["isaac"] = False
    r = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 400 and "Isaac Sim is not installed" in (await r.json())["error"]


def test_voices_default_on_and_no_voice_is_passed_only_when_recording_with_them_off(tmp_path, monkeypatch):
    assert Project(id="p", name="n", scenario_id="reception_guide").voice is True
    assert "--no-voice" not in argv_for(tmp_path, monkeypatch, sim="isaac", record=True)
    assert "--no-voice" in argv_for(tmp_path, monkeypatch, sim="isaac", record=True, voice=False)
    assert "--no-voice" not in argv_for(tmp_path, monkeypatch, sim="isaac", voice=False)  # not recording: nothing to voice


def test_the_cli_takes_no_voice_and_a_voice_command():
    from wardzero_agents import cli
    import argparse
    live = cli._live_flags()
    assert live.parse_args([]).no_voice is False and live.parse_args(["--no-voice"]).no_voice is True
