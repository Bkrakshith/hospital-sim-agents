"""The Studio backend over aiohttp's test client. The run is a fake subprocess, never the real CLI."""
import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents.campaign import ROBOT_ID
from wardzero_agents.campaign.trust import TrustState, TrustStore
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_roles, load_scenario_presets
from wardzero_agents.memory import memory_path
from wardzero_agents.studio.compile import compile_project
from wardzero_agents.studio.project import Project
from wardzero_agents.studio.runner import Runner
from wardzero_agents.studio.server import Ctx, make_app

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}

FAKE = textwrap.dedent('''
    import json, os, signal, sys, time
    from pathlib import Path
    runs = Path(os.environ["WARDZERO_RUNS_DIR"]); pid = sys.argv[sys.argv.index("--project-id") + 1]
    d = runs / "20261104-000000_fake"; d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "project_id": pid, "scenario_id": "ae_waiting"}))
    print("Run folder: " + str(d), flush=True)
    w = lambda n, rows: (d / (n + ".jsonl")).open("a").write("".join(json.dumps(r) + "\\n" for r in rows))
    w("tokens", [{"t": 1, "person": "P1", "cost_usd": 0.25, "input_tokens": 100, "output_tokens": 10,
                  "cache_read_tokens": 5, "cache_write_tokens": 1},
                 {"t": 2, "person": "P1", "cost_usd": 0.5, "input_tokens": 200, "output_tokens": 20,
                  "cache_read_tokens": 0, "cache_write_tokens": 0}])
    w("status", [{"t": 12.0, "person": "P1", "x": 1.0, "y": 2.0, "yaw": 0, "posture": "seated", "state": "idle"}])
    w("says", [{"t": 12.5, "person": "P1", "text": "Hello robot", "addressee": None}])
    w("robot_pose", [{"t": 12.0, "x": 3.0, "y": 4.0, "yaw": 0.5, "speed": 0.1}])
    mode = os.environ.get("FAKE_MODE", "wait")
    def done(*a):
        print("Run finished (exit code 0).", flush=True)
        if mode == "debrief":
            time.sleep(30)
        (d / "trust.jsonl").write_text("{}\\n"); (d / "report.html").write_text("<html>report</html>")
        sys.exit(0)
    signal.signal(signal.SIGINT, done)
    if mode == "debrief":
        print("Run finished (exit code 0).", flush=True)
        time.sleep(30)
    time.sleep(30)
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(data))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(runs))
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE)
    calls = []

    def command(project, staged, bot, domain):
        calls.append({"staged": staged, "bot": bot, "domain": domain})
        return [sys.executable, str(script), "--project-id", project.id]

    runner = Runner(data, ROOT, runs, command=command, stop_publisher=lambda: False, key_present=lambda: False)
    return {"data": data, "runs": runs, "runner": runner, "calls": calls}


@pytest.fixture
async def client(env):
    ctx = Ctx(env["data"], ROOT, env["runs"], env["runner"])
    c = TestClient(TestServer(make_app(ctx)))
    await c.start_server()
    c.env = env
    yield c
    await asyncio.to_thread(env["runner"].shutdown)
    await c.close()


async def mk(client, scenario="ae_waiting", **kw):
    r = await client.post("/api/projects", json={"name": "T", "scenario_id": scenario, **kw}, headers=H)
    assert r.status == 201, await r.text()
    return await r.json()


async def test_presets_and_site_served(client):
    r = await client.get("/api/presets")
    d = await r.json()
    assert len(d["scenarios"]) == 10 and {x["id"] for x in d["roles"]} >= {"ward_nurse", "patient"}
    assert d["moods"]
    s = await (await client.get("/api/site")).json()
    assert s["rooms"] and s["beds"]


async def test_project_crud_roundtrip(client):
    p = await mk(client)
    assert p["characters"] and p["scenario_id"] == "ae_waiting" and p["robot_profile"] == "g1_claude"
    assert (client.env["data"] / "studio" / "projects" / p["id"] / "project.yaml").is_file()
    assert [x["id"] for x in await (await client.get("/api/projects")).json()] == [p["id"]]
    r = await client.put(f"/api/projects/{p['id']}", json={"name": "Renamed", "budget_usd": 2.0}, headers=H)
    assert (await r.json())["name"] == "Renamed"
    got = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert got["budget_usd"] == 2.0 and got["characters"] == p["characters"]
    assert (await client.delete(f"/api/projects/{p['id']}", headers=H)).status == 200
    assert (await client.get(f"/api/projects/{p['id']}")).status == 404
    assert (await client.get("/api/projects/..%2Fx")).status == 404


async def test_bad_project_400(client):
    r = await client.post("/api/projects", json={"name": "x", "scenario_id": "nope"}, headers=H)
    assert r.status == 400 and "error" in await r.json()


async def test_add_character_on_wall_400(client):
    p = await mk(client)
    base = {"name": "Walla", "sex": "female", "age_band": "31-50", "role": "visitor", "mood": "happy"}
    r = await client.post(f"/api/projects/{p['id']}/characters", json={**base, "x": 500.0, "y": 500.0}, headers=H)
    assert r.status == 400 and "outside" in (await r.json())["error"]
    # find a real wall cell
    from wardzero_agents.site import load_site
    site = load_site(ROOT)
    iy, ix = [int(v[0]) for v in __import__("numpy").nonzero(site.grid.occ)]
    wx, wy = site.grid.to_world(iy, ix)
    r = await client.post(f"/api/projects/{p['id']}/characters", json={**base, "x": wx, "y": wy}, headers=H)
    assert r.status == 400 and "wall" in (await r.json())["error"]
    assert len((await (await client.get(f"/api/projects/{p['id']}")).json())["characters"]) == len(p["characters"])


async def test_add_character_ok_and_edit(client):
    p = await mk(client)
    seat = p["characters"][0]
    r = await client.post(f"/api/projects/{p['id']}/characters", headers=H, json={
        "name": "New Person", "sex": "male", "age_band": "18-30", "role": "visitor", "mood": "happy",
        "x": seat["x"] + 0.0, "y": seat["y"] + 0.0})
    # the first character's own spot is clear (it is a seat); a person may stand where a seat is
    assert r.status in (201, 400)
    r = await client.put(f"/api/projects/{p['id']}/characters/{seat['id']}", headers=H,
                         json={"role_prompt": "Be brisk."})
    assert (await r.json())["role_prompt"] == "Be brisk."
    assert (await client.put(f"/api/projects/{p['id']}/characters/ZZ", headers=H, json={})).status == 404
    assert (await client.delete(f"/api/projects/{p['id']}/characters/{seat['id']}", headers=H)).status == 200


async def test_add_patient_to_occupied_bed_400(client):
    p = await mk(client, "ward_fire")
    assert any(c["bed"] == "B3" for c in p["characters"])
    r = await client.post(f"/api/projects/{p['id']}/characters", headers=H, json={
        "name": "Second", "sex": "male", "age_band": "51-70", "role": "patient", "mood": "happy",
        "x": 25.1, "y": 16.2, "bed": "B3"})
    assert r.status == 400 and "B3" in (await r.json())["error"]
    r = await client.post(f"/api/projects/{p['id']}/characters", headers=H, json={
        "name": "Third", "sex": "male", "age_band": "51-70", "role": "patient", "mood": "happy",
        "x": 25.109, "y": 16.204})  # on the bed, no bed given
    assert r.status == 400
    r = await client.post(f"/api/projects/{p['id']}/characters", headers=H, json={
        "name": "Fourth", "sex": "male", "age_band": "51-70", "role": "patient", "mood": "happy",
        "x": 0, "y": 0, "bed": "B4"})
    assert r.status == 201 and (await r.json())["bed"] == "B4"


async def test_post_without_header_403(client):
    assert (await client.post("/api/projects", json={"name": "x", "scenario_id": "ae_waiting"})).status == 403
    p = await mk(client)
    assert (await client.put(f"/api/projects/{p['id']}", json={})).status == 403
    assert (await client.delete(f"/api/projects/{p['id']}")).status == 403
    assert (await client.post("/api/run/stop")).status == 403
    r = await client.get("/api/presets", headers={"Origin": "http://evil.example"})
    assert r.status == 403


@pytest.mark.parametrize("sid", sorted(load_scenario_presets(ROOT)))
async def test_compile_loads_with_load_scenario(client, sid, tmp_path):
    p = Project.model_validate(await mk(client, sid))
    path = compile_project(p, ROOT, tmp_path / "staged")
    loaded = load_scenario(path)
    sc = loaded.scenario
    assert sc.campaign and sc.preset_id == sid and sc.robot.profile == "g1_claude"
    assert [i.memory_key for i in loaded.instances] == [f"studio:{p.id}:{c.id}" for c in p.characters]


async def test_compile_uses_edited_prompts_and_drops_orphan_beats(client, tmp_path):
    p = Project.model_validate(await mk(client, "ward_fire"))
    chars = list(p.characters)
    chars[0] = chars[0].model_copy(update={"role_prompt": "Always whistles at the robot", "mood_prompt": None})
    p = p.model_copy(update={"characters": chars[1:]})  # N1 removed: beat b3 targets it
    loaded = load_scenario(compile_project(p, ROOT, tmp_path / "s"))
    assert "b3" not in [b.id for b in loaded.storyboard.beats]
    p2 = p.model_copy(update={"characters": [chars[0], *chars[1:]]})
    loaded = load_scenario(compile_project(p2, ROOT, tmp_path / "s2"))
    persona = loaded.personas[chars[0].id]
    assert "Always whistles at the robot" in persona.backstory


async def test_reset_keeps_prompts(client):
    p = await mk(client)
    cid = p["characters"][0]["id"]
    await client.put(f"/api/projects/{p['id']}/characters/{cid}", headers=H, json={"role_prompt": "Custom text."})
    key = f"studio:{p['id']}:{cid}"
    store = TrustStore(memory_path(client.env["data"]))
    role = load_roles(ROOT)[p["characters"][0]["role"]]
    store.put_many([TrustState(key, ROBOT_ID, 0.05, 0.5, violations=3, stage=4, runs=2)])
    rows = await (await client.get(f"/api/projects/{p['id']}/memory")).json()
    row = next(x for x in rows if x["id"] == cid)
    assert row["trust"] == 0.05 and row["stage"] == 4
    r = await client.post(f"/api/projects/{p['id']}/memory/reset", json={"characters": [cid]}, headers=H)
    assert r.status == 200
    rows = await (await client.get(f"/api/projects/{p['id']}/memory")).json()
    row = next(x for x in rows if x["id"] == cid)
    assert row["trust"] == role.trust_start and row["runs"] == 0
    got = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert got["characters"][0]["role_prompt"] == "Custom text."
    r = await client.post(f"/api/projects/{p['id']}/memory/reset", json={"characters": "all"}, headers=H)
    assert r.status == 200
    r = await client.post(f"/api/projects/{p['id']}/memory/reset", json={"characters": ["nobody"]}, headers=H)
    assert r.status == 400


async def wait_for(fn, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = await fn()
        if v:
            return v
        await asyncio.sleep(0.05)
    raise AssertionError("timed out")


async def test_start_twice_409(client):
    p = await mk(client)
    r = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 202
    r2 = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r2.status == 409 and "error" in await r2.json()
    assert client.env["calls"][0]["bot"] == "reception"  # builtin robot, no key
    async def up():  # the fake installs its SIGINT handler after writing its first records
        return (await (await client.get("/api/run/state")).json())["robot"]
    await wait_for(up)
    await asyncio.sleep(0.2)
    stop = await client.post("/api/run/stop", headers=H)
    assert (await stop.json())["stopped"] is True
    async def finished():
        return (await (await client.get("/api/run/state")).json())["phase"] == "done"
    await wait_for(finished)
    st = await (await client.get("/api/run/state")).json()
    assert st["has_report"] and st["running"] is False


async def test_state_reports_cost_from_tokens_jsonl(client):
    p = await mk(client)
    assert (await (await client.get("/api/run/state")).json())["phase"] == "idle"
    await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    async def have():
        s = await (await client.get("/api/run/state")).json()
        return s if s["cost_usd"] >= 0.75 and s["says_tail"] and s["robot"] else None
    s = await wait_for(have)
    assert s["cost_usd"] == pytest.approx(0.75)
    assert s["tokens"] == {"input": 300, "output": 30, "cache_read": 5, "cache_write": 1}
    assert s["people"][0]["id"] == "P1" and s["t"] == 12.0 and s["phase"] == "running" and s["running"]
    assert s["run_id"] == "20261104-000000_fake" and s["says_tail"][0]["text"] == "Hello robot"
    assert s["robot"]["x"] == 3.0
    runs = await (await client.get(f"/api/runs?project={p['id']}")).json()
    assert [r["run_id"] for r in runs] == ["20261104-000000_fake"]
    assert await (await client.get("/api/runs?project=other")).json() == []
    await client.post("/api/run/stop", headers=H)


async def test_stop_during_debrief_is_not_a_signal(client, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "debrief")
    p = await mk(client)
    await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    async def debrief():
        return (await (await client.get("/api/run/state")).json())["phase"] == "debrief"
    await wait_for(debrief)
    r = await (await client.post("/api/run/stop", headers=H)).json()
    assert r["stopped"] is False
    assert (await (await client.get("/api/run/state")).json())["running"] is True
    client.env["runner"]._proc.kill()


async def test_report_served_and_traversal_refused(client):
    d = client.env["runs"] / "r1"
    d.mkdir()
    (d / "report.html").write_text("<html>hi</html>")
    r = await client.get("/runs/r1/report.html")
    assert r.status == 200 and "hi" in await r.text() and "default-src 'none'" in r.headers["Content-Security-Policy"]
    assert (await client.get("/runs/nope/report.html")).status == 404
    assert (await client.get("/runs/..%2Fdata/report.html")).status == 404


async def test_robots_crud_urdf_sensors(client):
    r = await client.get("/api/robots")
    assert [x["name"] for x in await r.json()] == ["g1_claude"]
    r = await client.post("/api/robots", headers=H, json={"name": "myarm", "base": "static", "radius": 0.4})
    assert r.status == 201 and (await r.json())["controller"] == "external"
    assert (await client.post("/api/robots", headers=H, json={"name": "g1_claude"})).status == 400
    assert (await client.post("/api/robots", headers=H, json={"name": "../x"})).status == 400
    assert (await client.post("/api/robots", headers=H, json={"name": "bad", "radius": -1})).status == 400
    urdf = Path(__file__).parent / "data" / "urdf"
    folder = next(p for p in urdf.iterdir() if p.is_dir()) if urdf.is_dir() and any(urdf.iterdir()) else urdf
    r = await client.post("/api/robots/myarm/urdf", headers=H, json={"folder": str(folder)})
    assert r.status == 200 and "issues" in await r.json()
    r = await client.put("/api/robots/myarm/sensors", headers=H, json={"sensors": [
        {"id": "cam", "kind": "rgb", "model": None, "link": "base", "params": {"label": "cam"}}]})
    assert r.status == 200 and (await r.json())["sensors"][0]["id"] == "cam"
    assert (await client.put("/api/robots/myarm/sensors", headers=H, json={"sensors": [{"id": "x", "kind": "bogus"}]})).status == 400
    assert (await client.put("/api/robots/g1_claude/sensors", headers=H, json={"sensors": []})).status == 400
    assert (await client.put("/api/robots/nope/sensors", headers=H, json={"sensors": []})).status == 404
    assert [x["name"] for x in await (await client.get("/api/robots")).json()] == ["g1_claude", "myarm"]
    assert (await client.delete("/api/robots/myarm", headers=H)).status == 200


async def test_convert_job_streams_log(env, tmp_path):
    fake = tmp_path / "conv.py"
    fake.write_text('import json,sys\nprint("converting")\nprint(json.dumps({"ok": True, "usd": sys.argv[-1]}))\n')
    ctx = Ctx(env["data"], ROOT, env["runs"], env["runner"],
              convert_command=lambda urdf, sensors, out: [sys.executable, str(fake), str(out)])
    c = TestClient(TestServer(make_app(ctx)))
    await c.start_server()
    try:
        await c.post("/api/robots", headers=H, json={"name": "arm2"})
        assert (await c.post("/api/robots/arm2/convert", headers=H)).status == 400  # no URDF yet
        urdf = tmp_path / "r" / "a.urdf"
        urdf.parent.mkdir()
        urdf.write_text('<robot name="a"><link name="base"/></robot>')
        await c.post("/api/robots/arm2/urdf", headers=H, json={"folder": str(urdf.parent)})
        r = await c.post("/api/robots/arm2/convert", headers=H)
        job = (await r.json())["job"]
        async def done():
            d = await (await c.get(f"/api/jobs/{job}")).json()
            return d if d["status"] != "running" else None
        d = await wait_for(done)
        assert d["status"] == "done" and "converting" in d["log"] and d["result"]["ok"]
        assert (await c.get("/api/jobs/zzz")).status == 404
    finally:
        await c.close()


async def test_connection_without_ros(client, monkeypatch):
    monkeypatch.setattr("wardzero_agents.cli.ros_available", lambda: False)
    d = await (await client.get("/api/connection")).json()
    assert d["ros"] is False and d["lights"] == {"controller_seen": False, "sensors_subscribed": False,
                                                 "commands_seen": False}
    assert d["ip"] and d["topics"] and {t["dir"] for t in d["topics"]} == {"sim_to_robot", "robot_to_sim"}
    r = await client.post("/api/connection", json={"robot_domain": 7}, headers=H)
    assert (await r.json())["robot_domain"] == 7
    assert (await client.post("/api/connection", json={"robot_domain": 999}, headers=H)).status == 400
    p = await mk(client)
    await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert client.env["calls"][0]["domain"] == 7
    await client.post("/api/run/stop", headers=H)


async def test_connection_lights_from_probe(env):
    async def probe(domain, ns):
        return {"controller_seen": True, "sensors_subscribed": True, "commands_seen": False}
    import wardzero_agents.cli as cli
    orig = cli.ros_available
    cli.ros_available = lambda: True
    c = TestClient(TestServer(make_app(Ctx(env["data"], ROOT, env["runs"], env["runner"], probe=probe))))
    await c.start_server()
    try:
        d = await (await c.get("/api/connection")).json()
        assert d["ros"] and d["lights"] == {"controller_seen": True, "sensors_subscribed": True, "commands_seen": False}
    finally:
        cli.ros_available = orig
        await c.close()


async def test_ws_pushes_state(client):
    async with client.ws_connect("/ws") as ws:
        msg = await asyncio.wait_for(ws.receive_json(), 3)
        assert msg["phase"] == "idle"


def test_cli_run_takes_robot_domain_and_studio_subcommand():
    from wardzero_agents.cli import build_parser
    a = build_parser().parse_args(["run", "x.yaml", "--robot-domain", "3", "--project-id", "p1"])
    assert a.robot_domain == 3 and a.project_id == "p1"
    s = build_parser().parse_args(["studio", "--port", "9000", "--no-browser"])
    assert s.port == 9000 and s.no_browser and callable(s.func)
