"""Studio placement: dragging characters, "Start seated" on the A&E waiting seats, and forbidden areas
(project API, compile, task brief, the built-in robot's routes and brief, the report)."""
import json
import math
from pathlib import Path

import pytest
import yaml

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.models import ForbiddenArea
from wardzero_agents.site import Router, load_site
from wardzero_agents.site.areas import AreaRouter, areas_from_brief, run_areas, task_brief
from wardzero_agents.studio.compile import compile_project
from wardzero_agents.studio.project import Project

from test_studio_api import H, ROOT, client, env, mk  # noqa: F401  (fixtures)

WAITING = {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "D3"}
SITE = load_site(ROOT)
BASE = {"sex": "female", "age_band": "31-50", "role": "visitor", "mood": "happy"}


def seat_of(c):
    return next((s.id for s in SITE.seats.values() if math.hypot(s.x - c["x"], s.y - c["y"]) < 0.05), None)


async def empty_project(client):
    p = await mk(client)
    r = await client.put(f"/api/projects/{p['id']}", json={"characters": []}, headers=H)
    assert r.status == 200, await r.text()
    return await r.json()


async def add(client, pid, **kw):
    return await client.post(f"/api/projects/{pid}/characters", json={**BASE, **kw}, headers=H)


# ---- drag ----------------------------------------------------------------------------------------
async def test_drag_saves_position_room_and_snaps_to_a_seat(client):
    p = await empty_project(client)
    c = await (await add(client, p["id"], name="Ann", x=-7.0, y=-1.5)).json()
    r = await client.put(f"/api/projects/{p['id']}/characters/{c['id']}", headers=H,
                         json={"x": -7.5, "y": 0.3, "yaw": 1.0})  # dropped near seat A2
    moved = await r.json()
    assert r.status == 200 and seat_of(moved) == "A2" and moved["room"] == "AE_Waiting"
    r = await client.put(f"/api/projects/{p['id']}/characters/{c['id']}", headers=H,
                         json={"x": -7.0, "y": -1.5, "yaw": 0.5})
    back = await r.json()
    assert (back["x"], back["y"], back["yaw"]) == (-7.0, -1.5, 0.5) and seat_of(back) is None


# ---- start seated --------------------------------------------------------------------------------
async def test_start_seated_picks_the_nearest_free_waiting_seat_and_unticking_stands_them_up(client):
    p = await empty_project(client)
    c = await (await add(client, p["id"], name="Bea", x=-5.9, y=-1.5)).json()
    assert c["start_seated"] is False
    r = await client.put(f"/api/projects/{p['id']}/characters/{c['id']}", headers=H, json={"start_seated": True})
    sat = await r.json()
    assert r.status == 200 and sat["start_seated"] and seat_of(sat) == "C1" and sat["room"] == "AE_Waiting"
    r = await client.put(f"/api/projects/{p['id']}/characters/{c['id']}", headers=H, json={"start_seated": False})
    up = await r.json()
    seat = SITE.seats["C1"]
    assert not up["start_seated"] and seat_of(up) is None
    assert math.hypot(up["x"] - seat.x, up["y"] - seat.y) == pytest.approx(1.0, abs=0.02)
    assert up["y"] < seat.y  # in front: C1 faces south


async def test_start_seated_when_all_eight_are_taken_is_a_clear_error(client):
    p = await empty_project(client)
    for i in range(8):
        r = await add(client, p["id"], name=f"P{i}", x=-7.0, y=-1.5, start_seated=True)
        assert r.status == 201, await r.text()
    got = (await (await client.get(f"/api/projects/{p['id']}")).json())["characters"]
    assert {seat_of(c) for c in got} == WAITING
    last = await (await add(client, p["id"], name="Late", x=-7.0, y=-1.5)).json()
    r = await client.put(f"/api/projects/{p['id']}/characters/{last['id']}", headers=H, json={"start_seated": True})
    assert r.status == 400 and "All 8 A&E waiting seats are taken" in (await r.json())["error"]
    after = (await (await client.get(f"/api/projects/{p['id']}")).json())["characters"]
    assert not next(c for c in after if c["id"] == last["id"])["start_seated"]


async def test_dragging_a_seated_character_off_the_seat_unticks_it(client):
    p = await empty_project(client)
    c = await (await add(client, p["id"], name="Cat", x=-7.0, y=-1.5, start_seated=True)).json()
    assert c["start_seated"]
    r = await client.put(f"/api/projects/{p['id']}/characters/{c['id']}", headers=H, json={"x": -7.0, "y": -1.6})
    moved = await r.json()
    assert r.status == 200 and not moved["start_seated"] and (moved["x"], moved["y"]) == (-7.0, -1.6)


async def test_old_projects_without_the_flag_stay_valid_and_compile_to_seats(client, tmp_path):
    p = await mk(client)  # ae_waiting: people on seats, no start_seated in the preset
    raw = Project.model_validate(p).model_dump(mode="json")
    for c in raw["characters"]:
        c.pop("start_seated", None)
    proj = Project.model_validate(raw)
    assert all(not c.start_seated for c in proj.characters)
    p2 = await empty_project(client)
    await add(client, p2["id"], name="Dee", x=-7.0, y=-1.5, start_seated=True)
    proj2 = Project.model_validate(await (await client.get(f"/api/projects/{p2['id']}")).json())
    sc = yaml.safe_load(compile_project(proj2, ROOT, tmp_path / "s").read_text())
    assert sc["cast"][0]["seat"] in WAITING and "start" not in sc["cast"][0]


# ---- forbidden areas -----------------------------------------------------------------------------
async def test_forbidden_areas_add_remove_and_compile(client, tmp_path):
    p = await mk(client)
    url = f"/api/projects/{p['id']}/areas"
    r = await client.post(url, json={"name": "Resus", "x0": 2, "y0": 3, "x1": -1, "y1": 1}, headers=H)
    a = await r.json()
    assert r.status == 201 and a == {"id": "fa1", "name": "Resus", "x0": -1, "y0": 1, "x1": 2, "y1": 3}
    assert (await client.post(url, json={"name": "", "x0": 0, "y0": 0, "x1": 1, "y1": 1}, headers=H)).status == 400
    assert (await client.post(url, json={"name": "Thin", "x0": 0, "y0": 0, "x1": 0.1, "y1": 1}, headers=H)).status == 400
    assert (await client.post(url, json={"name": "resus", "x0": 0, "y0": 0, "x1": 1, "y1": 1}, headers=H)).status == 400
    r = await client.post(url, json={"name": "Staff only", "x0": 10, "y0": 10, "x1": 12, "y1": 12}, headers=H)
    assert (await r.json())["id"] == "fa2"
    proj = Project.model_validate(await (await client.get(f"/api/projects/{p['id']}")).json())
    loaded = load_scenario(compile_project(proj, ROOT, tmp_path / "s"))
    assert [a.name for a in loaded.scenario.forbidden_areas] == ["Resus", "Staff only"]
    brief = task_brief(loaded.scenario.robot_brief, loaded.scenario.forbidden_areas)
    assert brief["job"] and brief["forbidden_areas"][0] == {
        "id": "fa1", "name": "Resus", "polygon": [[-1, 1], [2, 1], [2, 3], [-1, 3]]}
    assert [a.name for a in areas_from_brief(brief)] == ["Resus", "Staff only"]
    assert (await client.delete(f"{url}/fa1", headers=H)).status == 200
    assert (await client.delete(f"{url}/fa1", headers=H)).status == 404
    got = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert [a["name"] for a in got["forbidden_areas"]] == ["Staff only"]


def test_a_scenario_without_areas_publishes_the_brief_unchanged():
    assert task_brief({"role": "r", "job": "j"}, []) == {"role": "r", "job": "j"}
    assert task_brief(None, []) is None


def test_area_router_routes_round_an_area_and_follows_world_blocks():
    base = Router(SITE.grid)
    a, b = (-7.0, -1.5), (-2.0, -1.5)  # along the front of the waiting room
    straight = base.route(a, b)
    assert straight is not None
    area = ForbiddenArea(id="fa1", name="Middle", x0=-5.0, y0=-2.6, x1=-4.0, y1=-0.6)
    r = AreaRouter(base, [area])
    route = r.route(a, b)

    def crosses(pts):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            for k in range(21):
                f = k / 20
                if area.contains(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f):
                    return True
        return False
    assert crosses(straight)
    assert route is None or not crosses(route)
    assert r.route(a, (-4.5, -1.5)) is None  # a goal inside the area is refused
    assert r.route((-4.5, -1.5), a) is not None  # from inside, the robot can walk out
    v = base.version
    base.block(-6.0, -1.5, 0.4)
    assert base.version == v + 1 and r._router() is not None and r._version == base.version


def test_robot_brief_lists_the_areas_by_name():
    from wardzero_agents.robots.claude_robot import RobotMind
    brief = {"role": "Receptionist", "job": "Greet people.",
             "forbidden_areas": [{"id": "fa1", "name": "Resus", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                                 {"id": "fa2", "name": "Staff only", "polygon": [[3, 3], [4, 3], [4, 4], [3, 4]]}]}
    mind = RobotMind(SITE, brief, gateway=None, clock=lambda: 0.0, known_places=["AE_Triage"], pose=(-7.0, -1.5, 0))
    assert "Do not enter: Resus, Staff only." in mind.card()


def test_entries_into_a_user_area_are_counted_and_listed_in_the_report(tmp_path):
    from wardzero_agents.bench.metrics import RunData, forbidden_area_entries, forbidden_area_entry_list
    from wardzero_agents.bench.report import build_facts, render_html
    run = tmp_path / "run"
    (run / "config").mkdir(parents=True)
    (run / "config" / "ae_waiting.yaml").write_text(yaml.safe_dump({"forbidden_areas": [
        {"id": "fa1", "name": "Resus", "x0": -5.0, "y0": -2.6, "x1": -4.0, "y1": -0.6}]}))
    (run / "meta.json").write_text(json.dumps({"run_id": "r1", "scenario_id": "ae_waiting"}))
    poses = [{"t": float(t), "x": -7.0 + 0.5 * t, "y": -1.5, "yaw": 0, "speed": 0.5} for t in range(12)]
    poses += [{"t": 12.0 + t, "x": -1.5 - 0.5 * t, "y": -1.5, "yaw": 3.1, "speed": 0.5} for t in range(12)]
    (run / "robot_pose.jsonl").write_text("".join(json.dumps(p) + "\n" for p in poses))
    assert [a.name for a in run_areas(run, {"scenario_id": "ae_waiting"})] == ["Resus"]
    d = RunData.load(run)
    entries = [e for e in forbidden_area_entry_list(d) if e["kind"] == "user"]
    assert [(e["name"], e["t"]) for e in entries] == [("Resus", 4.0), ("Resus", 17.0)]
    assert forbidden_area_entries(d) >= 2.0
    facts = build_facts(run, None)
    assert [e["name"] for e in facts["forbidden_entries"] if e["kind"] == "user"] == ["Resus", "Resus"]
    page = render_html(facts)
    assert 'id="areas"' in page and "Resus" in page and 'class="forbid"' in page and "00:04" in page


def test_a_run_without_areas_has_no_areas_section(tmp_path):
    from wardzero_agents.bench.report import build_facts, render_html
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"run_id": "r1", "scenario_id": "ae_waiting"}))
    (run / "robot_pose.jsonl").write_text(json.dumps({"t": 0.0, "x": -7.0, "y": -1.5, "yaw": 0}) + "\n")
    facts = build_facts(run, None)
    assert facts["forbidden_areas"] == [] and 'id="areas"' not in render_html(facts)
