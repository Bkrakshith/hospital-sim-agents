"""The robot as the first point of contact: it greets people, helps them itself, writes a note on the desk computer
before it interrupts anyone, and the busy receptionist sends walk-ups to it first."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import (CharacterSpec, ROLE_INTENTS, load_moods, load_roles,
                                            load_scenario_presets, preset_to_scenario)
from wardzero_agents.robots.claude_robot import (RobotMind, actions_for, decision_outputs, parse_decision,
                                                 system_prompt)
from wardzero_agents.robots.profile import G1_CLAUDE
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt
from wardzero_agents.world.perception import perceive_people

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
WALKING = dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")
LS = load_scenario(CONFIG / "scenarios" / "waiting_area_v1.yaml")
BRIEF = {"role": "reception guide", "job": "Greet people who come to reception."}
NONE = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
        "place": [], "message": "", "reason": "nothing to do"}


# ---- the robot's mind ---------------------------------------------------------------------------------------------
class Recorder:
    def __init__(self, answer=None):
        self.reqs, self.answer = [], answer if answer is not None else dict(NONE)

    async def complete(self, req):
        from wardzero_agents.llm.gateway import LLMResult
        self.reqs.append(req)
        data = self.answer(req) if callable(self.answer) else self.answer
        return LLMResult(True, data, None, "fake", cost_usd=0.0)


def det(track, x, y, z=1.7):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": 0.5, "y": 0.5, "z": z}}}


def mind(gw, **kw):
    return RobotMind(LS.site, BRIEF, gw, lambda: 0.0, profile=G1_CLAUDE, known_places=LS.scenario.known_places,
                     pose=(1.05, 1.62, -1.5708), **kw)


def test_the_robot_is_told_it_is_the_first_point_of_contact_and_when_to_fetch_staff():
    for profile in (G1_CLAUDE, WALKING):
        s = system_prompt(profile)
        assert "first point of contact" in s
        assert "greet" in s.lower()
        assert "note_desk" in s and "tell_staff" in s and "press_call" in s
        # the order: help yourself, then a note, then telling staff, then the call button; emergencies skip it
        assert s.index("note_desk") < s.index("tell_staff", s.index("note_desk")) < s.rindex("press_call")
        assert "straight away" in s and "collapsed" in s


def test_note_desk_is_an_action_for_a_fixed_and_a_walking_robot():
    assert "note_desk" in actions_for(G1_CLAUDE) and "note_desk" in actions_for(WALKING)
    assert "note_desk" in G1_CLAUDE.abilities() and "computer" in G1_CLAUDE.abilities()


def test_note_desk_needs_a_track_and_a_message_and_goes_on_the_wire():
    m = mind(Recorder())
    with pytest.raises(ValueError, match="message"):
        parse_decision({**NONE, "action": "note_desk", "target": "t1"}, G1_CLAUDE, m)
    with pytest.raises(ValueError, match="target"):
        parse_decision({**NONE, "action": "note_desk", "message": "hi"}, G1_CLAUDE, m)
    d = parse_decision({**NONE, "action": "note_desk", "target": "t1", "message": "Waiting for the AE_Triage nurse."},
                       G1_CLAUDE, m)
    out = decision_outputs(d, m, "rb1")
    assert {k: out.action[k] for k in ("id", "action", "about")} == {"id": "rb1", "action": "note_desk", "about": "t1"}
    assert out.action["message"].startswith("Waiting for the ") and "_" not in out.action["message"]
    assert out.goal is None and not out.stop and out.say is None


async def test_people_the_robot_has_not_spoken_to_are_marked_until_it_speaks_to_them():
    gw = Recorder()
    m = mind(gw)
    m.on_people([det("t1", 1.05, 0.0), det("t2", 1.05, 12.0)], 0.0)
    await m.tick(0.0)
    user = gw.reqs[-1].user
    t1 = next(line for line in user.splitlines() if line.startswith("- t1:"))
    t2 = next(line for line in user.splitlines() if line.startswith("- t2:"))
    assert "not spoken to yet" in t1
    assert "not spoken to yet" not in t2  # too far away to greet
    gw.answer = {**NONE, "say": "Hello, can I help you?", "toward_track": "t1", "reason": "greet"}
    await m.tick(10.0)
    gw.answer = dict(NONE)
    await m.tick(20.0)
    t1 = next(line for line in gw.reqs[-1].user.splitlines() if line.startswith("- t1:"))
    assert "not spoken to yet" not in t1


async def test_every_robot_decision_is_reported_with_what_it_saw_and_why():
    rows = []
    gw = Recorder({**NONE, "say": "Hello there.", "toward_track": "t1", "reason": "greet the new arrival"})
    m = mind(gw, on_decision=rows.append)
    m.on_people([det("t1", 1.05, 0.0)], 0.0)
    await m.tick(0.0)
    gw.answer = {"action": "fly"}  # invalid: still reported, as dropped
    await m.tick(10.0)
    assert len(rows) == 2
    ok, bad = rows
    assert ok["t"] == 0.0 and ok["action"] == "none" and ok["say"] == "Hello there." and ok["toward_track"] == "t1"
    assert ok["reason"] == "greet the new arrival" and "t1" in ok["situation"] and ok["error"] is None
    assert bad["action"] is None and bad["error"] and "situation" in bad
    json.dumps(rows)  # plain data for the run log


async def test_the_robot_decisions_stream_is_a_run_log_stream(tmp_path):
    log = RunLog(tmp_path)
    log.write("robot_decisions", {"t": 0.0, "action": "none"})
    assert (tmp_path / "robot_decisions.jsonl").exists()


# ---- the desk computer ----------------------------------------------------------------------------------------------
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


def test_a_desk_note_reaches_the_receptionist_on_her_screen_and_the_robot_stays_put(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    tr = track_of(s, "V1")
    start = w.robot.pose
    s.on_task_action({"id": "rb1", "action": "note_desk", "about": tr,
                      "message": "Visitor waiting for the ward; I gave directions."})
    outs = advance(s, 1.0)
    res = [r for o in outs for r in o.task_results if r["id"] == "rb1"]
    assert [(r["action"], r["ok"], r["reason"]) for r in res] == [("note_desk", True, "noted")]
    assert w.robot.pose == start
    assert not [a for o in outs for a in o.audio if a.source_kind == "robot"]  # silent: nobody is interrupted
    nudges = [n for o in outs for n in o.nudges]
    assert [n.person for n in nudges] == ["RCP"] and nudges[0].card == rt.NOTE_CARD
    msg = nudges[0].message
    assert "screen" in msg and '"Visitor waiting for the ward; I gave directions."' in msg and "person id V1" in msg
    note = [r for r in rows(tmp_path / "run", "robot_tasks") if r.get("kind") == "note"]
    assert note and note[0]["about"] == "V1" and note[0]["staff"] == "RCP" and note[0]["read"]
    assert "V1" not in json.dumps(res) and "RCP" not in json.dumps(res)


def test_a_desk_note_waits_on_the_screen_until_she_is_back_at_the_desk(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    put(w, "RCP", -9.0, -1.0)  # away in the A&E waiting room
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "note_desk", "about": tr, "message": "A visitor needs a ward nurse."})
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for o in outs for r in o.task_results if r["id"] == "rb1"] == [(True, "noted")]
    assert not [n for o in outs for n in o.nudges]
    seat = w.site.seats["R1"]
    put(w, "RCP", seat.x, seat.y)
    outs = advance(s, 0.3)
    assert [n.person for o in outs for n in o.nudges] == ["RCP"]
    assert not [n for o in advance(s, 0.3) for n in o.nudges]  # read once


def test_a_desk_note_with_no_receptionist_an_unknown_track_or_no_message_fails(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.4, 0.6)
    tr = track_of(s, "V1")
    s.on_task_action({"id": "rb1", "action": "note_desk", "about": "t999", "message": "x"})
    s.on_task_action({"id": "rb2", "action": "note_desk", "about": tr})
    w.people["RCP"].posture = "gone"
    s.on_task_action({"id": "rb3", "action": "note_desk", "about": tr, "message": "x"})
    got = {r["id"]: r["reason"] for o in advance(s, 0.2) for r in o.task_results}
    assert got == {"rb1": "unknown_track", "rb2": "bad_request", "rb3": "no_desk_staff"}


# ---- the people ---------------------------------------------------------------------------------------------------
def test_the_receptionist_is_busy_at_the_computer_and_sends_walk_ups_to_the_robot_first():
    p = load_roles(CONFIG)["receptionist"].prompt
    assert "computer" in p and "robot first" in p
    assert "unwell" in p  # someone who looks very unwell is helped at once


def test_an_ae_patient_arrived_on_their_feet_and_can_follow_the_robot():
    roles = load_roles(CONFIG)
    r = roles["ae_patient"]
    assert r.persona_role == "ae_patient" and "follow" in ROLE_INTENTS["ae_patient"]
    assert "follow" in r.intents and "call_bell" not in r.intents
    assert "bed" not in r.prompt.split("\n")[0].lower()


def test_the_ae_waiting_scene_uses_ae_patients_and_a_first_contact_brief(tmp_path):
    p = PRESETS["ae_waiting"]
    assert {c.id: c.role for c in p.characters if c.id in ("P1", "P2")} == {"P1": "ae_patient", "P2": "ae_patient"}
    assert "first point of contact" in p.robot_brief["job"]
    loaded = preset_to_scenario(p, CONFIG, tmp_path)  # the personas compile with the new role
    assert {loaded.personas[pid].role for pid in ("P1", "P2")} == {"ae_patient"}


def test_a_character_can_be_an_ae_patient(tmp_path):
    c = CharacterSpec(id="P9", name="Sam Lee", sex="male", age_band="31-50", role="ae_patient", mood="anxious",
                      room="Lobby", x=0.0, y=0.0, yaw=0.0)
    assert c.role in load_roles(CONFIG) and c.mood in load_moods(CONFIG)


def test_someone_coming_in_the_main_entrance_reads_check_in_with_the_robot(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    e = w.site.landmarks["entrance"]
    put(w, "V1", e.x, e.y + 0.5, yaw=math.pi / 2)  # just inside, facing in
    assert "sign_checkin" in [v.id for v in w.observation("V1", 0.0, []).visible if v.kind == "sign"]
    signs = {sg.id: sg.text for sg in w.signs}
    assert "robot" in signs["sign_checkin"].lower() and "robot" in signs["sign_desk_robot"].lower()
    put(w, "V1", 1.0, 0.57, yaw=math.pi / 2)  # in front of the reception desk, facing it
    assert "sign_desk_robot" in [v.id for v in w.observation("V1", 0.0, []).visible if v.kind == "sign"]


def test_staff_are_told_the_robot_does_the_scenarios_main_task(tmp_path):
    p = PRESETS["meal_round"]
    loaded = preset_to_scenario(p, CONFIG, tmp_path)
    nurse = loaded.personas["N1"]
    text = " ".join(nurse.goals) + " " + nurse.backstory
    # staff read the line written to them, never the robot's own job text (QA: quoted, it reads as their job)
    assert "Leave that work to the robot" in text and p.staff_line.split(".")[0] in text
    assert p.robot_brief["job"].split(".")[0] not in text
    patient = loaded.personas["M1"]
    assert "Leave that work to the robot" not in " ".join(patient.goals) + patient.backstory


def test_a_new_project_shows_isaac_in_a_window():
    from wardzero_agents.studio.project import Project
    assert Project(id="p1", name="x", scenario_id="ae_waiting").headed is True
