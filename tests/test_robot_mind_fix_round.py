"""The robot's prompt after QA: routes in words from where it stands now (and round a blocked route it has seen), no
greeting while an alarm sounds, the scenario's destination key kept from the robot, the new actions and reasons (follow,
guide without the walk back, a line for the whole room, single-step hand_over, no_answer, asleep tracks), people standing
at a known spot, and the fallback brief knowing whether the desk is staffed."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.robots.claude_robot import (RobotMind, decision_outputs, decision_schema, parse_decision,
                                                 system_prompt)
from wardzero_agents.robots.profile import G1_CLAUDE
from wardzero_agents.studio import auto_staff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
WALKING = dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")
NONE = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
        "place": [], "message": "", "seat": False, "return_to_post": True, "reason": "nothing to do"}


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p)
            cache[preset_id] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[preset_id]
    return get


class Recorder:
    def __init__(self, answer=None):
        self.reqs, self.answer = [], answer if answer is not None else dict(NONE)

    async def complete(self, req):
        from wardzero_agents.llm.gateway import LLMResult
        self.reqs.append(req)
        return LLMResult(True, dict(self.answer), None, "fake", cost_usd=0.0)


def det(track, x, y, z=1.7, **extra):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": 0.5, "y": 0.5, "z": z}},
            **extra}


def mind_for(loaded, gw=None, pose=None, profile=WALKING):
    r = loaded.scenario.robot
    return RobotMind(loaded.site, loaded.scenario.robot_brief, gw or Recorder(), lambda: 0.0, profile=profile,
                     known_places=loaded.scenario.known_places, pose=pose or (r.x, r.y, r.yaw))


def routes_block(situation):
    return situation.split("Getting there from here", 1)[1].split("\n\n", 1)[0]


# ---- routes are worked out where the robot stands now -----------------------------------------------------------------
async def test_routes_are_in_the_situation_from_the_current_pose(staged):
    vr = staged("visiting_rush")
    gw = Recorder()
    m = mind_for(vr, gw)
    assert "Getting there" not in m.card()
    m.on_people([det("t1", 19.0, 9.0)], 0.0)
    await m.tick(0.0)
    first = routes_block(gw.reqs[0].user)
    m.on_pose(19.0, 20.0, math.pi / 2)  # most of the way up the corridor
    await m.tick(20.0)
    later = routes_block(gw.reqs[-1].user)
    line0 = next(ln for ln in first.splitlines() if "Ward nurse base" in ln)
    line1 = next(ln for ln in later.splitlines() if "Ward nurse base" in ln)
    assert int(line1.split("about ")[1].split(" m")[0]) < int(line0.split("about ")[1].split(" m")[0]) - 8


async def test_a_blocked_route_the_robot_saw_changes_its_routes(staged):
    ev = staged("evacuation_guide")
    gw = Recorder()
    m = mind_for(ev, gw)
    m.on_people([det("t1", 1.0, -0.9)], 0.0)
    await m.tick(0.0)
    before = next(ln for ln in routes_block(gw.reqs[0].user).splitlines() if ln.startswith("- Main entrance"))
    assert "about" in before
    m.on_objects({"blocked": [{"id": f"route_blocked_{i}", "x": x, "y": -2.6, "text": "Route blocked"}
                              for i, x in enumerate((-4.2, -1.2, 1.8), 1)]})
    await m.tick(20.0)
    after = next(ln for ln in routes_block(gw.reqs[-1].user).splitlines() if ln.startswith("- Main entrance"))
    assert after != before and "blocked" in after


# ---- the brief ----------------------------------------------------------------------------------------------------------
def test_the_destination_key_is_not_shown_to_the_robot(staged):
    card = mind_for(staged("reception_guide")).card()
    assert "Destination" not in card
    card = mind_for(staged("evacuation_guide")).card()
    assert "Destination" not in card and "Ambulance handover bay" in card.split("YOUR JOB", 1)[1]


def test_ward_fetch_rule_defers_to_the_job():
    s = system_prompt(WALKING, desk_staffed=False)
    assert "YOUR JOB says how to call" in s


# ---- no greeting during an alarm ----------------------------------------------------------------------------------------
async def test_greeting_rules_are_off_while_an_alarm_sounds(staged):
    ev = staged("evacuation_guide")
    gw = Recorder()
    m = mind_for(ev, gw)
    r = ev.scenario.robot
    m.on_people([det("t1", r.x - 2.0, r.y)], 0.0)
    await m.tick(0.0)
    assert "first point of contact" in gw.reqs[0].system_fixed
    await m.on_heard({"source_kind": "alarm", "source_id": "unknown", "text": "Fire alarm", "received_db": 90.0,
                      "partial": False, "addressed_to_me": False}, 5.0)
    m.on_people([det("t1", r.x - 2.0, r.y), det("t2", r.x - 3.0, r.y + 0.5)], 5.0)
    await m.tick(5.0)
    req = gw.reqs[-1]
    assert "first point of contact" not in req.system_fixed and "Do not wait to be asked" not in req.system_fixed
    assert "not spoken to yet" not in req.user
    assert "alarm" in req.user.lower()
    assert not m._someone_to_greet()


# ---- new actions ------------------------------------------------------------------------------------------------------
def _m(staged, preset="wandering_patient"):
    m = mind_for(staged(preset))
    r = staged(preset).scenario.robot
    m.on_people([det("t3", r.x + 1.0, r.y), det("t4", r.x, r.y + 1.5)], 0.0)
    return m


def test_follow_is_an_action_with_a_track(staged):
    m = _m(staged)
    assert "follow" in decision_schema(WALKING)["properties"]["action"]["enum"]
    assert "follow" not in decision_schema(G1_CLAUDE)["properties"]["action"]["enum"]  # a fixed post cannot walk
    d = parse_decision({**NONE, "action": "follow", "target": "t3"}, WALKING, m)
    assert decision_outputs(d, m, "rb1").action == {"id": "rb1", "action": "follow", "track": "t3"}
    with pytest.raises(ValueError):
        parse_decision({**NONE, "action": "follow"}, WALKING, m)
    assert "follow" in system_prompt(WALKING) and "follow" in WALKING.abilities()


def test_guide_can_stay_at_the_destination(staged):
    m = _m(staged)
    schema = decision_schema(WALKING)
    assert schema["properties"]["return_to_post"] == {"type": "boolean"}
    d = parse_decision({**NONE, "action": "guide", "target": "t3", "destination": "B4", "return_to_post": False},
                       WALKING, m)
    assert decision_outputs(d, m, "rb1").action == {"id": "rb1", "action": "guide_start", "track": "t3",
                                                    "place": "B4", "return_to_post": False}
    d = parse_decision({**NONE, "action": "guide", "target": "t3", "destination": "B4"}, WALKING, m)
    assert "return_to_post" not in decision_outputs(d, m, "rb2").action
    old = {k: v for k, v in NONE.items() if k != "return_to_post"}  # an answer without the field: back to the post
    d = parse_decision({**old, "action": "guide", "target": "t3", "destination": "B4"}, WALKING, m)
    assert "return_to_post" not in decision_outputs(d, m, "rb3").action
    with pytest.raises(ValueError):
        parse_decision({**NONE, "return_to_post": "no"}, WALKING, m)


async def test_a_line_for_the_whole_room(staged):
    m = _m(staged, "ae_waiting")
    d = parse_decision({**NONE, "say": "The wait is about three hours.", "toward_track": "room"}, WALKING, m)
    out = decision_outputs(d, m, "rb1")
    assert out.toward == "room" and out.say
    assert "room" in system_prompt(WALKING).split("toward_track", 1)[1].split("\n", 1)[0]


async def test_a_room_line_counts_as_speaking_to_everyone_in_the_room(staged):
    ae = staged("ae_waiting")
    gw = Recorder({**NONE, "say": "The wait is about three hours.", "toward_track": "room"})
    m = mind_for(ae, gw)
    r = ae.scenario.robot
    m.on_people([det("t1", r.x + 1.0, r.y + 1.0), det("t2", r.x - 0.5, r.y + 2.0)], 0.0)
    await m.tick(0.0)
    assert {"t1", "t2"} <= m._spoken


def test_abilities_cover_single_step_hand_over_and_its_reasons():
    a = WALKING.abilities()
    assert "in one action" in a and "no_answer" in a and "refused" in a
    assert "asleep" in system_prompt(WALKING)


def test_keep_clear_rule_allows_standing_beside_someone_to_pass_things():
    s = system_prompt(WALKING)
    line = next(ln for ln in s.splitlines() if ln.startswith("- Safety comes first"))
    assert "beside" in line and "hand" in line


# ---- what the camera shows ---------------------------------------------------------------------------------------------
async def test_an_asleep_track_is_shown_asleep(staged):
    meal = staged("meal_round")
    gw = Recorder()
    m = mind_for(meal, gw, pose=(23.0, 19.0, 0.0))
    m.on_people([det("t1", 25.1, 19.1, z=0.6, asleep=True), det("t2", 25.1, 16.2, z=0.6)], 0.0)
    await m.tick(0.0)
    tracks = gw.reqs[0].user.split("People your camera sees", 1)[1].split("Objects in view", 1)[0]
    line1 = next(ln for ln in tracks.splitlines() if ln.startswith("- t1"))
    line2 = next(ln for ln in tracks.splitlines() if ln.startswith("- t2"))
    assert "asleep" in line1 and "asleep" not in line2


async def test_a_person_at_a_known_spot_is_said_to_be_there(staged):
    sp = staged("specimen_run")
    gw = Recorder()
    m = mind_for(sp, gw, pose=(9.5, 3.0, -math.pi / 2))
    po1 = next(c for c in PRESETS["specimen_run"].characters if c.id == "PO1")
    m.on_people([det("t1", po1.x, po1.y), det("t2", 10.2, 0.0)], 0.0)
    await m.tick(0.0)
    tracks = gw.reqs[0].user.split("People your camera sees", 1)[1].split("Objects in view", 1)[0]
    assert "Porter waiting point" in next(ln for ln in tracks.splitlines() if ln.startswith("- t1"))
    assert "Porter waiting point" not in next(ln for ln in tracks.splitlines() if ln.startswith("- t2"))


# ---- the robot's own config -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("preset_id,staffed", [("meal_round", False), ("reception_guide", True)])
def test_fallback_brief_knows_whether_the_desk_is_staffed(staged, tmp_path, preset_id, staffed):
    from wardzero_agents.robots.view import load_robot_view, write_robot_config
    path = write_robot_config(tmp_path, staged(preset_id), domain=None, budget_usd=1.0)
    assert json.loads(path.read_text())["robot_brief"]["desk_staffed"] is staffed
    assert load_robot_view(path).brief["desk_staffed"] is staffed
