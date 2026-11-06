"""The robot does its own job first: its prompt, its brief card, when it wakes, and what it can perceive of the
scenario's main task (fires, alarms, blocked routes, where a sound came from, which way a room faces)."""
import dataclasses
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.robots.claude_robot import (GREET_GAP_S, HEARTBEAT_S, RobotMind, decision_outputs,
                                                 decision_schema, parse_decision, route_words, system_prompt)
from wardzero_agents.robots.profile import G1_CLAUDE

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
WALKING = dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")
LS = load_scenario(CONFIG / "scenarios" / "waiting_area_v1.yaml")
POSE = (1.05, 1.62, -1.5708)
NONE = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
        "place": [], "message": "", "seat": False, "reason": "nothing to do"}


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            cache[preset_id] = preset_to_scenario(PRESETS[preset_id], CONFIG, tmp_path_factory.mktemp(preset_id))
        return cache[preset_id]
    return get


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


def at(pose, rel_deg, dist):
    """The point dist metres away at rel_deg from the robot's heading (positive = left)."""
    a = pose[2] + math.radians(rel_deg)
    return pose[0] + dist * math.cos(a), pose[1] + dist * math.sin(a)


def mind(brief=None, profile=WALKING, loaded=LS, pose=POSE, gw=None):
    brief = brief if brief is not None else {"role": "helper", "job": "Help people."}
    return RobotMind(loaded.site, brief, gw or Recorder(), lambda: 0.0, profile=profile,
                     known_places=loaded.scenario.known_places, pose=pose)


def tracks_block(situation):
    return situation.split("People your camera sees", 1)[1].split("Objects in view", 1)[0]


# ---- the prompt -----------------------------------------------------------------------------------------------------
def test_system_prompt_puts_your_job_first():
    for profile in (G1_CLAUDE, WALKING):
        s = system_prompt(profile)
        job = s.index("YOUR JOB is your main task and you do it yourself")
        first = s.index("When YOUR JOB is to greet or receive people")
        assert job < first
        assert "first point of contact" in s[first:first + 200]
        assert "start it at once" not in s.lower()  # jobs that wait for a call, a bell or an alarm


async def test_first_contact_false_drops_the_greet_marker_and_cadence():
    gw = Recorder()
    m = mind({"role": "runner", "job": "Walk to the nurse base.", "first_contact": False}, gw=gw)
    x, y = at(POSE, 0, 3.0)
    m.on_people([det("t1", x, y)], 0.0)
    await m.tick(0.0)
    assert len(gw.reqs) == 1
    assert "not spoken to yet" not in tracks_block(gw.reqs[0].user)
    m.on_people([det("t1", x, y)], 5.0)
    assert not m._due(5.0)
    assert m._due(HEARTBEAT_S)
    s = gw.reqs[0].system_fixed
    assert "When YOUR JOB is to greet" not in s and "Do not wait to be asked" not in s
    assert "keep your distance from people at work" in s
    assert "first_contact" not in m.card() and "False" not in m.card()


async def test_first_contact_default_keeps_reception_behaviour():
    gw = Recorder()
    m = mind(gw=gw)
    x, y = at(POSE, 0, 3.0)
    m.on_people([det("t1", x, y)], 0.0)
    await m.tick(0.0)
    assert "not spoken to yet" in tracks_block(gw.reqs[0].user)
    assert not m._due(GREET_GAP_S - 0.5)
    assert m._due(GREET_GAP_S)
    assert "first point of contact" in gw.reqs[0].system_fixed


async def test_unstaffed_desk_hides_note_desk():
    enum = decision_schema(WALKING, desk_staffed=False)["properties"]["action"]["enum"]
    assert "note_desk" not in enum and "tell_staff" in enum and "press_call" in enum
    assert "note_desk" in decision_schema(WALKING)["properties"]["action"]["enum"]
    s = system_prompt(WALKING, desk_staffed=False)
    assert "note_desk" not in s and "tell_staff, then press_call" in s
    assert "note_desk" in system_prompt(WALKING)
    assert "note_desk" not in WALKING.abilities(desk_staffed=False) and "note_desk" in WALKING.abilities()
    gw = Recorder({**NONE, "action": "note_desk", "target": "t1", "message": "x"})
    m = mind({"role": "ward helper", "job": "Help.", "desk_staffed": False}, gw=gw)
    await m.tick(0.0)
    d = await m.tick(HEARTBEAT_S)
    assert len(gw.reqs) == 1 and d is None  # not an action this robot has
    assert "note_desk" not in gw.reqs[0].schema["properties"]["action"]["enum"]
    assert "note_desk" not in gw.reqs[0].system_fixed and "note_desk" not in m.card()
    with pytest.raises(ValueError):
        parse_decision({**NONE, "action": "note_desk", "target": "t1", "message": "x"}, WALKING, m,
                       desk_staffed=False)


def test_confused_patient_leaving_skips_the_fetch_order():
    s = system_prompt(WALKING)
    line = next(ln for ln in s.splitlines() if ln.startswith("- Emergencies skip that order"))
    assert "disoriented" in line and "trying to leave" in line and "press_call" in line and "stay beside" in line


def test_extinguish_is_not_held_back_by_an_in_range_rule():
    s = system_prompt(WALKING)
    assert "a fire is in range" not in s
    assert "putting out a fire is part of your job" in s
    assert "you do not need to go_to it first" in s
    assert "within 2 m" in system_prompt(G1_CLAUDE)


def test_card_renders_meals_and_instruments_but_not_polygons(staged):
    meal = staged("meal_round")
    card = mind(meal.scenario.robot_brief, loaded=meal).card()
    for bed, tray in (("B1", "Margaret Hall - soft diet"), ("B2", "George Wright - diabetic"),
                      ("B3", "Joan Mason - vegetarian")):
        line = next(ln for ln in card.splitlines() if tray in ln)
        assert f"Ward bedroom {bed}" in line
    assert "Meals" in card
    scrub = staged("scrub_nurse")
    card = mind(scrub.scenario.robot_brief, loaded=scrub).card()
    assert "Instruments: scalpel, forceps, retractor, suture scissors, needle holder" in card
    brief = {"role": "helper", "job": "Help.", "first_contact": True, "speed": 0.5,
             "forbidden_areas": [{"id": "a1", "name": "the store room",
                                  "polygon": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]}]}
    card = mind(brief).card()
    assert card.count("Do not enter") == 1 and "the store room" in card
    assert "polygon" not in card.lower() and "[[" not in card and "Forbidden areas" not in card
    assert "First contact" not in card and "Speed" not in card


def test_go_to_a_bedroom_faces_the_bed(staged):
    meal = staged("meal_round")
    r = meal.scenario.robot
    m = mind(meal.scenario.robot_brief, loaded=meal, pose=(r.x, r.y, r.yaw))
    d = parse_decision({**NONE, "action": "go_to", "destination": "B2"}, WALKING, m)
    gx, gy, gyaw = decision_outputs(d, m, "rb1").goal
    bed = meal.site.beds["B2"]
    want = math.atan2(bed.y - gy, bed.x - gx)
    assert abs((gyaw - want + math.pi) % (2 * math.pi) - math.pi) < 0.1
    d = parse_decision({**NONE, "action": "go_to", "destination": "Ward entrance"}, WALKING, m)
    gx, gy, gyaw = decision_outputs(d, m, "rb2").goal
    assert gyaw == pytest.approx(math.atan2(gy - r.y, gx - r.x))


def test_route_words_use_the_first_real_waypoint(staged):
    vr = staged("visiting_rush")
    text = route_words(vr.site, (19.0, 6.5, math.pi / 2), ["nurse_base_ward"])
    line = next(ln for ln in text.splitlines() if ln.startswith("- "))
    assert "straight ahead" in line, line


# ---- waking up --------------------------------------------------------------------------------------------------------
async def test_alarm_wakes_the_robot():
    m = mind()
    await m.tick(0.0)
    assert not m._due(3.0)
    await m.on_heard({"source_kind": "alarm", "source_id": "unknown", "text": None, "received_db": 80.0,
                      "partial": False, "addressed_to_me": False}, 3.0)
    assert m._due(3.0)
    await m.tick(3.0)
    await m.on_heard({"source_kind": "pa", "source_id": "unknown", "text": "Fire: the route is blocked",
                      "received_db": 70.0, "partial": False, "addressed_to_me": False}, 6.0)
    assert m._due(6.0)


async def test_a_new_fire_wakes_once():
    m = mind()
    await m.tick(0.0)
    fx, fy = at(POSE, 0, 4.0)
    fire = {"items": [], "fires": [{"id": "fire_1", "x": fx, "y": fy, "size": 0.5}]}
    m.on_objects(fire)
    assert m._due(3.0)
    await m.tick(3.0)
    m.on_objects({"items": [], "fires": []})
    m.on_objects(fire)  # out of view and back again: not new
    assert not m._due(6.0)
    assert m._due(3.0 + HEARTBEAT_S)


async def test_blocked_sign_is_listed_and_remembered():
    gw = Recorder()
    m = mind(gw=gw)
    await m.tick(0.0)
    sx, sy = at(POSE, 0, 3.0)
    text = "Route blocked: use the other exit"
    m.on_objects({"items": [], "blocked": [{"id": "route_blocked_1", "x": sx, "y": sy, "text": text}]})
    assert m._due(2.5)
    await m.tick(2.5)
    line = next(ln for ln in gw.reqs[-1].user.splitlines() if ln.startswith("- route blocked sign"))
    assert f'"{text}"' in line and "3.0 m" in line and "straight ahead" in line
    m.on_objects({"items": []})
    for k in range(25):  # many later events: the sign stays in the robot's memory
        m.note(f"event {k}")
    await m.tick(20.0)
    user = gw.reqs[-1].user
    assert "- route blocked sign" not in user
    events = user.split("Your last events", 1)[1]
    assert f'saw sign "{text}"' in events


async def test_a_sign_without_a_position_does_not_stop_the_robot_thinking():
    gw = Recorder()
    m = mind(gw=gw)
    await m.tick(0.0)
    m.on_objects({"items": [], "blocked": [{"id": "route_blocked_1", "text": "Route blocked"}]})
    await m.tick(2.5)
    line = next(ln for ln in gw.reqs[-1].user.splitlines() if ln.startswith("- route blocked sign"))
    assert line == '- route blocked sign: "Route blocked"'


async def test_heard_bearing_names_the_matching_track():
    gw = Recorder()
    m = mind(gw=gw)
    a, b = at(POSE, -32, 2.0), at(POSE, 40, 2.5)
    m.on_people([det("t1", *a, z=1.2), det("t2", *b, z=1.2)], 0.0)
    await m.on_heard({"source_kind": "cough", "source_id": "unknown", "text": None, "received_db": 50.0,
                      "partial": False, "addressed_to_me": False, "bearing_deg": -32.0}, 0.0)
    await m.tick(0.0)
    line = next(ln for ln in gw.reqs[-1].user.splitlines() if "a cough" in ln)
    assert "from the direction of t1 (2.0 m)" in line
    c = at(POSE, -38, 3.0)  # a second track within 12 degrees: no guess
    m.on_people([det("t1", *a, z=1.2), det("t2", *b, z=1.2), det("t3", *c, z=1.2)], 1.0)
    await m.tick(11.0)
    line = next(ln for ln in gw.reqs[-1].user.splitlines() if "a cough" in ln)
    assert "from the direction of" not in line and "32 degrees to your right" in line
    m2 = mind(gw=Recorder())
    await m2.on_heard({"source_kind": "cough", "source_id": "unknown", "text": None, "received_db": 50.0,
                       "partial": False, "addressed_to_me": False}, 0.0)
    line = next(ln for ln in m2._situation(0.0).splitlines() if "a cough" in ln)
    assert "degrees" not in line and "direction" not in line  # no bearing: as before


async def test_an_old_sound_is_not_put_down_to_a_track():
    """Tracks are where they are now; a sound heard a while ago (the robot may have walked since) gets only its
    angle."""
    gw = Recorder()
    m = mind(gw=gw)
    m.on_people([det("t1", *at(POSE, -32, 2.0), z=1.2)], 0.0)
    await m.on_heard({"source_kind": "cough", "source_id": "unknown", "text": None, "received_db": 50.0,
                      "partial": False, "addressed_to_me": False, "bearing_deg": -32.0}, 0.0)
    m.on_people([det("t1", *at(POSE, -32, 2.0), z=1.2)], 6.0)
    line = next(ln for ln in m._situation(6.0).splitlines() if "a cough" in ln)
    assert "from the direction of" not in line and "32 degrees to your right" in line


def test_wait_cancels_is_explained():
    s = system_prompt(WALKING)
    assert "cancels any walk, guide or tell_staff" in s
    assert 'choose "none"' in s
    mob = WALKING.abilities()
    assert "0.3 m/s" in mob and "blocked_by_person" in mob and "extinguish walks you up to a fire" in mob
    assert "in range" not in mob
    fixed = G1_CLAUDE.abilities()
    assert "within 2 m" in fixed and "0.3 m/s" not in fixed


# ---- the wire ---------------------------------------------------------------------------------------------------------
def test_press_call_names_the_track_and_speech_names_its_listener():
    m = mind()
    m.on_people([det("t3", *at(POSE, 0, 2.0))], 0.0)
    d = parse_decision({**NONE, "action": "press_call", "target": "t3"}, WALKING, m)
    assert decision_outputs(d, m, "rb1").action == {"id": "rb1", "action": "press_call", "track": "t3"}
    for target in ("nurse", "N1", "t9"):  # not a track the robot sees: no name to pass on
        d = parse_decision({**NONE, "action": "press_call", "target": target}, WALKING, m)
        assert decision_outputs(d, m, "rb5").action == {"id": "rb5", "action": "press_call"}, target
    d = parse_decision({**NONE, "action": "press_call"}, WALKING, m)
    assert decision_outputs(d, m, "rb2").action == {"id": "rb2", "action": "press_call"}
    d = parse_decision({**NONE, "say": "Hello there.", "toward_track": "t2"}, WALKING, m)
    out = decision_outputs(d, m, "rb3")
    assert out.say == "Hello there." and out.toward == "t2"
    d = parse_decision({**NONE, "toward_track": "t2"}, WALKING, m)
    assert decision_outputs(d, m, "rb4").toward is None  # nothing said: nobody to address
