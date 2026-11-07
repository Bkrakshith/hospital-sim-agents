"""The scripted built-in robot after the re-QA, fed the prompt the real robot mind renders: it answers a call bell from
the bell's own words (source_kind "bell", "a call bell (bed B3)"), reads the places it knows by name, and in A&E spots
the patient who gets worse from the camera's "looks unwell" cue, with no cough needed."""
import asyncio
import dataclasses
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.llm.fake_robot import ScriptedRobot, parse_card, parse_situation
from wardzero_agents.robots.claude_robot import RobotMind, decision_schema
from wardzero_agents.robots.profile import G1_CLAUDE
from wardzero_agents.studio import auto_staff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
WALKING = dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}
    presets = load_scenario_presets(CONFIG)

    def get(preset_id):
        if preset_id not in cache:
            p = presets[preset_id]
            cache[preset_id] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id),
                                                  characters=auto_staff.with_auto(list(p.characters), p))
        return cache[preset_id]
    return get


def _mind(loaded):
    r = loaded.scenario.robot
    return RobotMind(loaded.site, loaded.scenario.robot_brief, None, lambda: 0.0, profile=WALKING,
                     known_places=loaded.scenario.known_places, pose=(r.x, r.y, r.yaw))


def _det(track, x, y, z=1.7, **extra):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": 0.5, "y": 0.5, "z": z}},
            **extra}


def _decide(robot, m, t):
    allowed = list(decision_schema(WALKING, m.desk_staffed)["properties"]["action"]["enum"])
    return robot.decide(m.card(), m._situation(t), allowed)


def test_the_card_lists_places_by_name_and_the_parser_still_reads_them(staged):
    c = parse_card(_mind(staged("meal_round")).card())
    assert "Ward bedroom B3" in c.rooms.values() and "Ward entrance" in c.rooms
    assert not any("_" in k for k in c.rooms)
    assert ScriptedRobot().room_of_bed(c, "B3") == "Ward bedroom B3"


def test_the_meal_assistant_goes_to_the_bed_a_bell_names(staged):
    """The re-QA meal_round in Isaac: the bell at 92.6 s was shown as someone nearby speaking, so the scripted robot
    went to it only 43 s later. Heard as a bell naming bed B3, it sets off for B3 at once."""
    m = _mind(staged("meal_round"))
    m.on_objects({"held": [{"id": "tray_M3", "label": "Joan Mason - vegetarian"}]})
    asyncio.run(m.on_heard({"source_kind": "bell", "source_id": "unknown", "text": "call bell, bed B3",
                            "received_db": 41.0, "partial": False, "addressed_to_me": False, "bearing_deg": 20.0},
                           92.6))
    assert m._speech is True
    s = parse_situation(m._situation(93.0))
    assert [h.who for h in s.heard] == ["bell"]
    d = _decide(ScriptedRobot(), m, 93.0)
    assert d["action"] == "go_to" and d["destination"] == "Ward bedroom B3", d


def test_its_own_staff_call_chime_is_not_a_bell_to_answer(staged):
    m = _mind(staged("night_call_bell"))
    asyncio.run(m.on_heard({"source_kind": "bell", "source_id": "unknown", "text": "staff call chime",
                            "received_db": 55.0, "partial": False, "addressed_to_me": False}, 10.0))
    r = ScriptedRobot()
    d = _decide(r, m, 10.5)
    assert d["action"] != "go_to" and not r.mem.get("bell")


def test_the_waiting_room_robot_alerts_staff_for_the_track_that_looks_unwell_without_a_cough(staged):
    m = _mind(staged("ae_waiting"))
    x, y, _ = m.pose
    m.on_people([_det("t1", x + 1.0, y + 2.0, z=1.2, unwell=True), _det("t2", x - 1.0, y + 1.5, z=1.2)], 70.0)
    r = ScriptedRobot()
    r.card_text = m.card()  # this run's robot: what it remembers is kept
    r.spoken |= {"t1", "t2"}  # both greeted and checked in earlier
    r.mem["noted"] = {"t1", "t2"}
    d = _decide(r, m, 70.5)
    assert d["action"] == "press_call" and d["target"] == "t1", d


def test_without_the_unwell_cue_a_cough_from_their_direction_still_counts(staged):
    m = _mind(staged("ae_waiting"))
    x, y, yaw = m.pose
    m.on_people([_det("t1", x, y + 2.0, z=1.2)], 70.0)  # straight ahead (the robot faces +y)
    asyncio.run(m.on_heard({"source_kind": "cough", "source_id": "unknown", "text": None, "received_db": 58.0,
                            "partial": False, "addressed_to_me": False, "bearing_deg": 0.0}, 70.2))
    r = ScriptedRobot()
    r.card_text = m.card()
    r.spoken.add("t1")
    r.mem["noted"] = {"t1"}
    d = _decide(r, m, 70.5)
    assert d["action"] == "press_call" and d["target"] == "t1", d
