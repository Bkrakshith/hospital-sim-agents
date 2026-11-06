"""Reception and guide: the staff seat, the receptionist, tell_staff, follow, guide pacing and the arrival metric."""
from pathlib import Path

import pytest

from wardzero_agents.config.presets import CharacterSpec, load_moods, load_roles
from wardzero_agents.site import load_site
from wardzero_agents.studio import characters as chars

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG)


def _char(**kw):
    d = dict(id="C1", name="Test Person", sex="female", age_band="31-50", role="visitor", mood="happy", room="",
             x=0.0, y=0.0)
    d.update(kw)
    return CharacterSpec(**d)


def test_placing_near_the_staff_chair_never_snaps_to_it(site):
    roles, moods = load_roles(CONFIG), load_moods(CONFIG)
    r1 = site.seats["R1"]
    c = chars.check_character(_char(x=r1.x + 0.2, y=r1.y), site, roles, moods, [])
    assert (c.x, c.y) == (r1.x + 0.2, r1.y)  # stands where it was put: R1 is not a seat to snap to
    a1 = site.seats["A1"]
    c = chars.check_character(_char(x=a1.x + 0.2, y=a1.y + 0.1), site, roles, moods, [])
    assert (c.x, c.y) == (a1.x, a1.y)  # a waiting seat still snaps


# ---- the world: a reception_guide scene with the auto receptionist ------------------------------------------------
import json  # noqa: E402
import math  # noqa: E402

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario  # noqa: E402
from wardzero_agents.contracts import Intent  # noqa: E402
from wardzero_agents.ros.stub_sim_node import StubSimService  # noqa: E402
from wardzero_agents.runlog import RunLog  # noqa: E402
from wardzero_agents.studio import auto_staff  # noqa: E402
from wardzero_agents.world import robot_tasks as rt  # noqa: E402
from wardzero_agents.world.oracle import Pose  # noqa: E402
from wardzero_agents.world.perception import perceive_people  # noqa: E402

PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def reception(tmp_path_factory):
    preset = PRESETS["reception_guide"]
    chars_ = auto_staff.with_auto(list(preset.characters), preset)
    return preset_to_scenario(preset, CONFIG, tmp_path_factory.mktemp("staged"), characters=chars_)


def svc(loaded, tmp_path):
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


def results(outs, kind=None):
    return [r for o in outs for r in o.task_results if r.get("kind") == kind]


def put(w, pid, x, y, yaw=0.0):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.plan = x, y, yaw, "standing", None, []


def track_of(s, pid):
    w = s.world
    s.step(0.1)
    return next(d["track"] for d in perceive_people(w, w.robot) if w.tracks.person(d["track"]) == pid)


def rows(run, stream):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def test_the_receptionist_starts_seated_in_the_staff_chair(reception, tmp_path):
    s = svc(reception, tmp_path)
    rcp = s.world.people["RCP"]
    assert rcp.posture == "seated" and rcp.seat == "R1" and rcp.inst.role == "receptionist"


def test_a_visitor_cannot_sit_in_the_staff_chair_and_does_not_see_it(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "RCP", 3.0, 0.0)
    w.people["RCP"].posture = "standing"
    put(w, "V1", 2.0, 1.0)
    w.apply_intent(Intent(id="s1", person="V1", t=0.0, kind="sit_down", target="R1"))
    assert w.people["V1"].state == "failed" and "staff" in w.people["V1"].reason
    assert not any(v.kind == "seat" and v.id == "R1" for v in w.observation("V1", 0.0, []).visible)
    w.apply_intent(Intent(id="s2", person="RCP", t=0.0, kind="sit_down", target="R1"))
    advance(s, 20)
    assert w.people["RCP"].posture == "seated" and w.people["RCP"].seat == "R1"


def test_tell_staff_walks_to_the_counter_and_the_receptionist_gets_the_person(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    s.on_say(__import__("wardzero_agents.contracts", fromlist=["Say"]).Say(
        person="V1", t=0.0, text="I'm Priya, I need a nurse", addressee="g1"))
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "tell_staff", "about": tr, "message": "A visitor needs a nurse."})
    outs = advance(s, 15)
    res = [r for r in results(outs) if r["id"] == "rb1"]
    assert res == [{"id": "rb1", "action": "tell_staff", "ok": True, "reason": "told", "t": res[0]["t"]}]
    c = w.site.landmarks["reception_counter"]
    assert math.hypot(w.robot.x - c.x, w.robot.y - c.y) < 0.2
    nudges = [n for o in outs for n in o.nudges]
    assert [n.person for n in nudges] == ["RCP"] and nudges[0].card == rt.TELL_CARD
    msg = nudges[0].message
    assert "person id V1" in msg and "Priya Patel" in msg and "approach V1" in msg and "seat R1" in msg
    assert '"A visitor needs a nurse."' in msg
    said = [a for o in outs for a in o.audio if a.source_kind == "robot"]
    assert said and said[0].addressee == "RCP" and said[0].text == "A visitor needs a nurse."
    tell = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "tell"]
    assert tell and tell[0]["about"] == "V1" and tell[0]["staff"] == "RCP" and tell[0]["heard"]
    # the result the robot gets names no person
    assert "V1" not in json.dumps(res) and "RCP" not in json.dumps(res)


def test_tell_staff_without_a_name_and_when_she_is_away_or_missing(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    put(w, "RCP", -9.0, -1.0)  # away in the A&E waiting room, out of talking range
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "tell_staff", "about": tr})
    outs = advance(s, 15)
    assert [(r["ok"], r["reason"]) for r in results(outs) if r["id"] == "rb1"] == [(False, "staff_not_at_desk")]
    assert not [n for o in outs for n in o.nudges]
    w.people["RCP"].posture = "gone"
    s.on_task_action({"id": "rb2", "action": "tell_staff", "about": tr})
    outs = advance(s, 0.3)
    assert [(r["ok"], r["reason"]) for r in results(outs) if r["id"] == "rb2"] == [(False, "no_staff_to_tell")]
    s.on_task_action({"id": "rb3", "action": "tell_staff", "about": "t999"})
    assert [r["reason"] for r in results(advance(s, 0.2)) if r["id"] == "rb3"] == ["unknown_track"]


def test_tell_message_has_no_name_until_it_is_said(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    msg = s.tasks.tell_message("hi", "V1", w.people["RCP"])
    assert "Priya" not in msg and "not said their name" in msg and "x 0.4, y 0.6" in msg


def test_a_robot_that_cannot_walk_tells_the_staff_from_where_it_stands(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    s.tasks.mobile = False
    put(w, "V1", 0.4, 0.6)
    tr = track_of(s, "V1")
    start = w.robot.pose
    s.on_task_action({"id": "rb1", "action": "tell_staff", "about": tr})
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs) if r["id"] == "rb1"] == [(True, "told")]
    assert w.robot.pose == start


def guided(reception, tmp_path, pid="V2", at=(1.6, 0.0)):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, pid, *at)
    tr = track_of(s, pid)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": tr, "place": "ward_entrance"})
    return s, w


def test_the_guided_person_follows_behind_and_arrives_and_the_robot_goes_back(reception, tmp_path):
    s, w = guided(reception, tmp_path)
    outs = advance(s, 0.3)
    n = [x for o in outs for x in o.nudges]
    assert n and n[0].person == "V2" and "follow it (follow g1)" in n[0].message
    s.on_intent(Intent(id="f1", person="V2", t=0.0, kind="follow", target="g1"))
    p, gaps, slow = w.people["V2"], [], []
    we = w.site.landmarks["ward_entrance"]
    outs = []
    for _ in range(900):
        o = s.step(0.1)
        outs.append(o)
        if w.robot.has_goal and p.intent_kind == "follow" and p.state == "running":
            gaps.append(math.hypot(p.x - w.robot.x, p.y - w.robot.y))
        if w.robot.speed_cap is not None and s.tasks._guide is not None:
            slow.append(w.robot.speed_cap)
        if any(r["id"] == "gd-return" for r in results([o], "goal")):
            break
    assert p.state == "done" and math.hypot(p.x - we.x, p.y - we.y) <= 2.0
    assert max(gaps) <= rt.GUIDE_HOLD_M + 0.3 and min(gaps[20:]) >= 0.95  # trails, never presses on the robot
    assert not slow  # the guided person never slowed the robot
    goals = results(outs, "goal")
    assert ("gd", True, "arrived") in [(r["id"], r["ok"], r["reason"]) for r in goals]
    assert ("gd-return", True, "arrived") in [(r["id"], r["ok"], r["reason"]) for r in goals]
    post = w.site.landmarks["reception_g1"]
    assert math.hypot(w.robot.x - post.x, w.robot.y - post.y) < 0.2
    bye = [a for o in outs for a in o.audio if a.source_kind == "robot" and a.text == rt.GOODBYE]
    assert bye and bye[0].addressee == "V2"
    log = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "guide"]
    assert [r["event"] for r in log] == ["start", "end"] and log[1]["reason"] == "arrived"
    assert log[0]["person"] == "V2" and log[0]["place"] == "ward_entrance" and log[0]["route_m"] > 10
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]


def test_the_robot_waits_for_a_person_who_does_not_follow_and_gives_up(reception, tmp_path):
    s, w = guided(reception, tmp_path, at=(-1.0, -1.5))  # 3.6 m away: too far behind from the start
    start = w.robot.pose
    outs = advance(s, 10)
    assert math.hypot(w.robot.x - start[0], w.robot.y - start[1]) < 0.05 and w.robot.has_goal
    assert not results(outs, "goal")  # held on purpose: not stuck
    outs = advance(s, rt.GUIDE_GIVE_UP_S - 9)
    assert ("gd", False, "cancelled") in [(r["id"], r["ok"], r["reason"]) for r in results(outs, "goal")]
    log = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "guide"]
    assert log[-1]["event"] == "end" and log[-1]["reason"] == "left_behind"


def test_another_person_near_the_robot_slows_it(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 1.6, 2.6 - 1.6)  # beside the robot's way to the counter
    s.on_goal(4.0, 1.4, None)
    s.step(0.1)
    s.step(0.1)
    assert w.robot.speed_cap == rt.NEAR_PERSON_SPEED and w.robot.speed <= rt.NEAR_PERSON_SPEED + 1e-6


def test_follow_needs_the_robot():
    from wardzero_agents.mind.graph import check_decision
    loaded = PRESETS  # noqa: F841
    area = __import__("wardzero_agents.config.loader", fromlist=["load_robot_view"]).load_robot_view(CONFIG)[1]
    base = {"say": None, "addressee": None, "mood": "calm", "patience_delta": 0.0, "reason": "r"}
    ok, err = check_decision({**base, "intent": {"kind": "follow", "target": None, "text": None}}, area, [], ["V1"],
                             role="ward_visitor")
    assert err is None and ok["intent"]["target"] == "g1"
    ok, err = check_decision({**base, "intent": {"kind": "follow", "target": "V1", "text": None}}, area, [], ["V1"],
                             role="ward_visitor")
    assert err is None and ok["intent"]["kind"] == "approach"  # following a person is walking with them
    _, err = check_decision({**base, "intent": {"kind": "follow", "target": "Q9", "text": None}}, area, [], ["V1"],
                            role="ward_visitor")
    assert err == "invalid_target"
    _, err = check_decision({**base, "intent": {"kind": "follow", "target": "g1", "text": None}}, area, [], ["V1"],
                            role="surgeon")
    assert err == "invalid_decision"


# ---- the built-in robot -------------------------------------------------------------------------------------------
from wardzero_agents.robots import profile as prof  # noqa: E402
from wardzero_agents.robots.claude_robot import RobotMind, decision_outputs, parse_decision  # noqa: E402


def test_the_g1_walks_in_quick_2d_and_in_isaac_only_with_sonic(monkeypatch):
    g1 = prof.G1_CLAUDE
    assert g1.can_move is False  # the profile itself is unchanged
    stub = prof.profile_for_sim(g1, "stub")
    assert stub.can_move and stub.base == "legged" and stub.max_speed == g1.max_speed
    monkeypatch.setenv("WARDZERO_G1_WALKS_IN_ISAAC", "0")
    assert prof.profile_for_sim(g1, "isaac") is g1
    assert prof.profile_for_sim(g1, "stub").can_move  # the switch is about Isaac only
    monkeypatch.setenv("WARDZERO_G1_WALKS_IN_ISAAC", "1")
    assert prof.profile_for_sim(g1, "isaac").can_move and prof.profile_for_sim(g1, "isaac").max_speed == 0.6
    other = prof.RobotProfile(**{**g1.__dict__, "name": "cart", "controller": "external"})
    assert prof.profile_for_sim(other, "stub") is other


def test_robot_config_carries_the_simulator(reception, tmp_path):
    from wardzero_agents.robots.view import load_robot_view, write_robot_config
    v = load_robot_view(write_robot_config(tmp_path, reception, domain=None, budget_usd=1.0, sim="isaac"))
    assert v.sim == "isaac"
    v = load_robot_view(write_robot_config(tmp_path, reception, domain=None, budget_usd=1.0))
    assert v.sim == "stub"


def test_tell_staff_decision_on_the_wire(reception):
    class GW:
        async def complete(self, req):
            raise AssertionError

    m = RobotMind(reception.site, None, GW(), lambda: 0.0, profile=prof.profile_for_sim(prof.G1_CLAUDE, "stub"),
                  known_places=reception.scenario.known_places, pose=(1.05, 1.62, -1.57))
    base = {"say": "I will tell the receptionist.", "toward_track": "t2", "item": "", "destination": "", "place": [],
            "reason": "r"}
    d = parse_decision({**base, "action": "tell_staff", "target": "t2", "message": "She needs a nurse."},
                       m.profile, m)
    out = decision_outputs(d, m, "rb7")
    assert out.action == {"id": "rb7", "action": "tell_staff", "about": "t2", "message": "She needs a nurse."}
    d = parse_decision({**base, "action": "tell_staff", "target": "t2"}, prof.G1_CLAUDE, m)  # a fixed robot too
    assert decision_outputs(d, m, "rb8").action == {"id": "rb8", "action": "tell_staff", "about": "t2"}
    with pytest.raises(ValueError):
        parse_decision({**base, "action": "tell_staff", "target": ""}, m.profile, m)
    assert "for track t2" in m._events[-1]
    assert "tell_staff" in m.profile.abilities()
    d = parse_decision({**base, "action": "guide", "target": "t2", "destination": "Ward entrance"}, m.profile, m)
    assert decision_outputs(d, m, "rb9").action["place"] == "ward_entrance"


def test_the_reception_preset_guides_to_the_ward_entrance():
    p = PRESETS["reception_guide"]
    assert p.robot_brief["destination"] == "ward_entrance" and "Ward_Entrance" in p.focus_rooms
    assert "walk to the reception desk and tell the receptionist" in p.robot_brief["job"]
    assert "Lobby" in PRESETS["ae_waiting"].focus_rooms


# ---- the arrival metric -------------------------------------------------------------------------------------------
def _write(run, stream, rows_):
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{stream}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_))


def _guide_run(run, person_path, route_m=14.0, end_t=200.0):
    we = load_site(CONFIG).landmarks["ward_entrance"]
    _write(run, "robot_tasks", [
        {"t": 10.0, "kind": "action", "id": "gd", "action": "guide_start", "track": "t1", "place": "ward_entrance"},
        {"t": 10.0, "kind": "result", "id": "gd", "action": "guide_start", "ok": True, "reason": "started"},
        {"t": 10.0, "kind": "guide", "event": "start", "id": "gd", "person": "V2", "place": "ward_entrance",
         "x": we.x, "y": we.y, "room": None, "route_m": route_m},
        {"t": 30.0, "kind": "goal", "id": "gd", "action": "goal", "ok": True, "reason": "arrived"}])
    _write(run, "status", [{"t": t, "person": "V2", "x": x, "y": y, "yaw": 0.0, "posture": "standing",
                            "progress": 0.0, "state": "running"} for t, x, y in person_path]
           + [{"t": end_t, "person": "V2", "x": person_path[-1][1], "y": person_path[-1][2], "yaw": 0.0,
               "posture": "standing", "progress": 0.0, "state": "idle"}])
    (run / "meta.json").write_text(json.dumps({"run_id": "r", "people": {"V2": {"role": "ward_visitor"}}}))


def test_a_guide_counts_only_when_the_person_arrives(tmp_path):
    from wardzero_agents.bench.metrics import RunData, people_reached_destination_rate
    from wardzero_agents.campaign.experiences import _Run, guide_outcomes
    we = load_site(CONFIG).landmarks["ward_entrance"]
    # the robot arrived at 30 s, the person stayed in the lobby: not a success
    _guide_run(tmp_path / "a", [(11.0, 1.6, 0.0), (60.0, 1.6, 0.0)])
    assert people_reached_destination_rate(RunData.load(tmp_path / "a")) == 0.0
    o = guide_outcomes(_Run(tmp_path / "a"))[0]
    assert o["arrived"] is False and o["limit_s"] == pytest.approx(60 + 14 / 0.4)
    # the person got there at 40 s (30 s after the start)
    _guide_run(tmp_path / "b", [(11.0, 1.6, 0.0), (40.0, we.x - 1.5, we.y)])
    assert people_reached_destination_rate(RunData.load(tmp_path / "b")) == 1.0
    assert guide_outcomes(_Run(tmp_path / "b"))[0]["time_s"] == pytest.approx(30.0)
    # they got there, but after 60 s + route/0.4 m/s: too late
    _guide_run(tmp_path / "c", [(11.0, 1.6, 0.0), (10.0 + 96.0, we.x, we.y)])
    assert guide_outcomes(_Run(tmp_path / "c"))[0]["arrived"] is False
    # the run ended before the time was up: not judged
    _guide_run(tmp_path / "d", [(11.0, 1.6, 0.0)], end_t=50.0)
    assert guide_outcomes(_Run(tmp_path / "d"))[0]["arrived"] is None
    assert people_reached_destination_rate(RunData.load(tmp_path / "d")) is None


def test_the_report_lists_each_guide(tmp_path):
    from wardzero_agents.bench.report import build_facts, render_html
    we = load_site(CONFIG).landmarks["ward_entrance"]
    _guide_run(tmp_path, [(11.0, 1.6, 0.0), (40.0, we.x, we.y)])
    facts = build_facts(tmp_path, PRESETS["reception_guide"])
    assert facts["guides"] == [{"person": "V2", "who": "Daniel Evans", "place": "ward_entrance", "where": "Ward entrance",
                                "arrived": True, "time_s": 30.0, "limit_s": 95.0, "t": 10.0}]
    page = render_html(facts)
    assert "People the robot guided" in page and "Daniel Evans" in page and "arrived" in page


def test_tell_staff_counts_as_alerting_staff(tmp_path):
    from wardzero_agents.bench.metrics import RunData, _alert_after
    _write(tmp_path, "robot_tasks", [{"t": 5.0, "kind": "action", "id": "rb1", "action": "tell_staff", "about": "t1"},
                                     {"t": 9.0, "kind": "result", "id": "rb1", "action": "tell_staff", "ok": True,
                                      "reason": "told"}])
    assert _alert_after(RunData.load(tmp_path), 2.0) == pytest.approx(7.0)


# ---- the auto receptionist in the Studio --------------------------------------------------------------------------
from test_studio_api import H, client, env, mk  # noqa: E402,F401  (fixtures)


def test_only_reception_scenes_get_a_receptionist():
    scenes = {pid for pid, p in PRESETS.items() if auto_staff.is_reception_scene(p)}
    assert scenes == {"reception_guide", "evacuation_guide", "ae_waiting"}
    r = auto_staff.receptionist()
    assert (r.id, r.role, r.auto) == ("RCP", "receptionist", True)
    assert (r.x, r.y) == (load_site(CONFIG).seats["R1"].x, load_site(CONFIG).seats["R1"].y)


async def test_a_new_project_gets_the_receptionist_and_her_removal_sticks(client):
    p = await mk(client, "reception_guide")
    rcp = [c for c in p["characters"] if c["id"] == "RCP"]
    assert rcp and rcp[0]["auto"] is True and rcp[0]["room"] == "Lobby"
    assert all(not c["auto"] for c in p["characters"] if c["id"] != "RCP")
    ward = await mk(client, "meal_round")
    assert not any(c["id"] == "RCP" for c in ward["characters"])
    r = await client.delete(f"/api/projects/{p['id']}/characters/RCP", headers=H)
    assert r.status == 200
    again = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert not any(c["id"] == "RCP" for c in again["characters"]) and again["removed_auto"] == ["RCP"]
    listed = await (await client.get("/api/projects")).json()
    assert not any(c["id"] == "RCP" for q in listed if q["id"] == p["id"] for c in q["characters"])


async def test_an_old_project_gets_her_when_loaded_and_a_scenario_switch_brings_her(client):
    from wardzero_agents.studio.project import ProjectStore
    p = await mk(client, "meal_round")
    store = ProjectStore(client.env["data"])
    old = store.get(p["id"])
    preset = PRESETS["ae_waiting"]
    store.save(old.model_copy(update={"scenario_id": "ae_waiting", "characters": list(preset.characters)}))
    got = await (await client.get(f"/api/projects/{p['id']}")).json()
    assert [c["id"] for c in got["characters"] if c["auto"]] == ["RCP"]
    assert any(c.id == "RCP" for c in store.get(p["id"]).characters)  # saved, so it stays
    r = await client.put(f"/api/projects/{p['id']}", json={"scenario_id": "meal_round"}, headers=H)
    assert not any(c["id"] == "RCP" for c in (await r.json())["characters"])
    r = await client.put(f"/api/projects/{p['id']}", json={"scenario_id": "evacuation_guide"}, headers=H)
    assert any(c["id"] == "RCP" for c in (await r.json())["characters"])


# ---- Isaac people: following the robot (fake Behavior API, no Isaac) -----------------------------------------------
def test_an_isaac_character_follows_the_robot_until_the_world_says_stop(tmp_path):
    from test_isaac_people import DT, intent, make, run
    svc_, _, apis = make(tmp_path, ("P5",))
    w = svc_.world
    run(svc_, apis, 6.0)
    w.robot.teleport(-1.0, -1.0, 0.0)
    w.robot._path = [(10.0, -1.0)]  # on its way somewhere, held for its follower: not over yet
    w.robot.speed_scale = 0.0
    svc_.on_intent(intent("P5", "follow", target="g1"))
    run(svc_, apis, 8.0)
    p = w.people["P5"]
    moves = [c for c in apis["P5"].calls if c[0] == "move_to"]
    assert moves and p.intent_kind == "follow" and p.state == "running"
    assert math.hypot(p.x - w.robot.x, p.y - w.robot.y) <= 1.6
    w.trail_released.add("P5")  # the guide is over
    run(svc_, apis, DT * 3)
    assert p.state == "done"


def test_a_route_keeps_clear_of_a_disc_when_it_can(site):
    from wardzero_agents.site import Router
    r = Router(site.grid)
    a, b, robot = (5.0, 1.6), (0.8, 1.6), (2.9, 1.6)

    def closest(path):
        best = math.inf
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            for k in range(21):
                best = min(best, math.dist((x0 + (x1 - x0) * k / 20, y0 + (y1 - y0) * k / 20), robot))
        return best

    assert closest(r.route(a, b)) < 0.2
    assert closest(r.route(a, b, [(*robot, 0.75)])) >= 0.6
    assert r.route(a, (2.9, 1.9), [(*robot, 0.75)]) is not None  # an end inside the disc stays reachable


def test_a_room_listed_in_places_you_know_is_a_valid_walk_target():
    # The persona card lists rooms next to the start room ("AE_Waiting"); the world resolves them,
    # so the decision check must accept them too (it rejected them as invalid_target).
    from wardzero_agents.mind.graph import check_decision
    area = __import__("wardzero_agents.config.loader", fromlist=["load_robot_view"]).load_robot_view(CONFIG)[1]
    site = load_site(CONFIG)
    base = {"say": None, "addressee": None, "mood": "stressed", "patience_delta": 0.0, "reason": "r"}
    walk = {**base, "intent": {"kind": "walk_to", "target": "AE_Waiting", "text": None}}
    ok, err = check_decision(walk, area, [], ["P2"], role="patient", site=site)
    assert err is None and ok["intent"]["target"] == "AE_Waiting"
    bad = {**base, "intent": {"kind": "walk_to", "target": "narnia", "text": None}}
    assert check_decision(bad, area, [], ["P2"], role="patient", site=site) == (None, "invalid_target")
