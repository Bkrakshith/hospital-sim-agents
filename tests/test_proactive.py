"""A proactive robot: it walks up to people, walks them to a free seat, and thinks more often while someone near it
has not been greeted."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent
from wardzero_agents.robots.claude_robot import (GREET_GAP_S, RobotMind, actions_for, decision_outputs,
                                                 decision_schema, parse_decision, system_prompt)
from wardzero_agents.robots.profile import G1_CLAUDE
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt
from wardzero_agents.world.perception import perceive_objects, perceive_people

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
WALKING = dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")
LS = load_scenario(CONFIG / "scenarios" / "waiting_area_v1.yaml")
NONE = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
        "place": [], "message": "", "seat": False, "reason": "nothing to do"}


# ---- the robot's mind ---------------------------------------------------------------------------------------------
class Recorder:
    def __init__(self, answer=None):
        self.reqs, self.answer = [], answer if answer is not None else dict(NONE)

    async def complete(self, req):
        from wardzero_agents.llm.gateway import LLMResult
        self.reqs.append(req)
        return LLMResult(True, dict(self.answer), None, "fake", cost_usd=0.0)


def det(track, x, y, z=1.7):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": 0.5, "y": 0.5, "z": z}}}


def mind(gw, profile=WALKING):
    return RobotMind(LS.site, {"role": "guide", "job": "Help people."}, gw, lambda: 0.0, profile=profile,
                     known_places=LS.scenario.known_places, pose=(1.05, 1.62, -1.5708))


def test_the_prompt_asks_the_robot_to_go_to_people_and_walk_them_to_seats():
    s = system_prompt(WALKING)
    assert "Do not wait to be asked" in s and "approach" in s and "seat" in s
    assert "check on" in s  # people who have been waiting a while
    assert "approach" in WALKING.abilities() and "seat" in WALKING.abilities()


def test_approach_is_for_a_walking_robot_and_needs_a_track():
    assert "approach" in actions_for(WALKING) and "approach" not in actions_for(G1_CLAUDE)
    m = mind(Recorder())
    with pytest.raises(ValueError, match="target"):
        parse_decision({**NONE, "action": "approach"}, WALKING, m)
    d = parse_decision({**NONE, "action": "approach", "target": "t2"}, WALKING, m)
    assert decision_outputs(d, m, "rb1").action == {"id": "rb1", "action": "approach", "track": "t2"}


def test_guide_can_end_at_a_free_seat():
    assert decision_schema(WALKING)["properties"]["seat"] == {"type": "boolean"}
    assert "seat" in decision_schema(WALKING)["required"]
    m = mind(Recorder())
    place = LS.scenario.known_places[0]
    d = parse_decision({**NONE, "action": "guide", "target": "t1", "destination": place, "seat": True}, WALKING, m)
    out = decision_outputs(d, m, "rb1").action
    assert out["action"] == "guide_start" and out["seat"] is True
    d = parse_decision({**NONE, "action": "guide", "target": "t1", "destination": place}, WALKING, m)
    assert "seat" not in decision_outputs(d, m, "rb2").action
    with pytest.raises(ValueError, match="seat"):
        parse_decision({**NONE, "seat": "yes"}, WALKING, m)
    no_seat = dict(NONE)
    del no_seat["seat"]  # an older reply without the field is still read
    assert parse_decision(no_seat, WALKING, m).seat is False


async def test_while_someone_near_is_not_greeted_the_robot_thinks_more_often():
    gw = Recorder()
    m = mind(gw)
    m.on_people([det("t1", 1.05, 0.0)], 0.0)
    await m.tick(0.0)
    assert len(gw.reqs) == 1
    await m.tick(GREET_GAP_S - 0.1)
    assert len(gw.reqs) == 1
    await m.tick(GREET_GAP_S + 0.1)
    assert len(gw.reqs) == 2
    gw.answer = {**NONE, "say": "Hello!", "toward_track": "t1"}
    await m.tick(2 * GREET_GAP_S + 0.2)  # greets t1
    n = len(gw.reqs)
    await m.tick(3 * GREET_GAP_S + 0.4)
    assert len(gw.reqs) == n  # greeted: back to the normal 10 s heartbeat


async def test_free_seats_in_view_are_in_the_prompt():
    gw = Recorder()
    m = mind(gw)
    m.on_objects({"items": [], "seats": [{"id": "C3", "x": 1.05, "y": 0.0, "free": True},
                                         {"id": "C4", "x": 1.5, "y": 0.0, "free": False}]})
    m.on_people([det("t1", 1.05, 0.5)], 0.0)
    await m.tick(0.0)
    u = gw.reqs[-1].user
    assert "seat C3: free" in u and "C4" not in u


# ---- the world ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reception(tmp_path_factory):
    preset = PRESETS["reception_guide"]
    chars_ = auto_staff.with_auto(list(preset.characters), preset)
    return preset_to_scenario(preset, CONFIG, tmp_path_factory.mktemp("staged"), characters=chars_)


def svc(loaded, tmp_path):
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


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


def test_the_robot_walks_up_to_a_person_and_faces_them(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", -1.0, -1.0)
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "approach", "track": tr})
    outs = advance(s, 20)
    res = [(r["action"], r["ok"], r["reason"]) for o in outs for r in o.task_results if r["id"] == "rb1"]
    assert ("approach", True, "started") in res and ("goal", True, "arrived") in res
    p = w.people["V1"]
    d = math.hypot(w.robot.x - p.x, w.robot.y - p.y)
    assert rt.APPROACH_M - 0.3 <= d <= rt.APPROACH_M + 0.3
    facing = math.atan2(p.y - w.robot.y, p.x - w.robot.x)
    assert abs((w.robot.yaw - facing + math.pi) % (2 * math.pi) - math.pi) < 0.35
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]


def test_approach_fails_for_an_unknown_track_or_a_robot_that_cannot_walk(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", -1.0, -1.0)
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "approach", "track": "t999"})
    s.tasks.mobile = False
    s.on_task_action({"id": "rb2", "action": "approach", "track": tr})
    got = {r["id"]: (r["ok"], r["reason"]) for o in advance(s, 0.2) for r in o.task_results}
    assert got == {"rb1": (False, "unknown_track"), "rb2": (False, "cannot_walk")}


def test_the_robot_sees_free_and_taken_seats(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    seats = perceive_objects(w, w.robot)["seats"]
    assert all(set(x) == {"id", "x", "y", "free"} for x in seats)
    assert "R1" not in [x["id"] for x in seats]  # the staff chair is not offered


def guided_to_a_seat(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V2", 1.6, 0.0)
    tr = track_of(s, "V2")
    s.on_task_action({"id": "gd", "action": "guide_start", "track": tr, "place": "AE_Waiting", "seat": True})
    advance(s, 0.3)
    s.on_intent(Intent(id="f1", person="V2", t=0.0, kind="follow", target="g1"))
    return s, w


def test_a_guide_to_a_seat_ends_beside_a_free_seat_and_the_person_is_shown_it(reception, tmp_path):
    s, w = guided_to_a_seat(reception, tmp_path)
    outs = []
    for _ in range(1500):
        o = s.step(0.1)
        outs.append(o)
        if any(r["id"] == "gd-return" for r in o.task_results):
            break
    log = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "guide"]
    assert [r["event"] for r in log] == ["start", "seat", "end"], log
    seat_id = log[1]["seat"]
    seat = w.site.seats[seat_id]
    assert not seat.staff_only and w.site.room_at(seat.x, seat.y) == "AE_Waiting"
    assert log[2]["reason"] == "arrived"
    p = w.people["V2"]
    assert math.hypot(p.x - seat.x, p.y - seat.y) <= rt.NEAR_PLACE_M
    shown = [n for o in outs for n in o.nudges if n.card == rt.SEAT_CARD]
    assert shown and shown[0].person == "V2" and f"sit_down {seat_id}" in shown[0].message
    said = [a.text for o in outs for a in o.audio if a.source_kind == "robot"]
    assert rt.SEAT_SAY in said
    # the person can now sit in it
    s.on_intent(Intent(id="s1", person="V2", t=0.0, kind="sit_down", target=seat_id))
    advance(s, 15)
    assert p.posture == "seated" and p.seat == seat_id


def test_with_no_free_seat_in_sight_the_robot_says_so_and_the_guide_ends_in_the_room(reception, tmp_path):
    s, w = guided_to_a_seat(reception, tmp_path)
    s.tasks.free_seat = lambda *a, **k: None
    outs = []
    for _ in range(1500):
        o = s.step(0.1)
        outs.append(o)
        if any(r["id"] == "gd-return" for r in o.task_results):
            break
    said = [a.text for o in outs for a in o.audio if a.source_kind == "robot"]
    assert rt.NO_SEAT_SAY in said
    log = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "guide"]
    assert [r["event"] for r in log] == ["start", "end"] and log[-1]["reason"] == "arrived"
