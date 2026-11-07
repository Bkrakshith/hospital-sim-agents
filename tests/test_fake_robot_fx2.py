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


def test_a_guide_that_stays_at_the_place_is_left_to_end_before_the_robot_walks_on():
    """A guide with return_to_post false waits at the place for the person (up to 20 s) after its goal arrives, and
    only then ends arrived (what a flow credits). Walking on at once replaced it (ended superseded), so the robot holds
    its next walk until the guide has had time to end."""
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    job = ("When the alarm sounds, get everyone out by a safe open way. Once the people near you are on their way out, "
           "check the A&E waiting area and walk anyone still there out.")
    c = "\n".join(["THE HOSPITAL", "Places you know:", "- Main lobby and reception: Main lobby and reception (lobby)",
                   "- A&E walk-in reception: A&E walk-in reception + 15-seat waiting (waiting)", "", "YOUR JOB",
                   "Role: fire evacuation guide", f"Job: {job}",
                   "Safe exit: the safe way out is the Ambulance handover bay in A&E.", "", "YOUR BODY", "A humanoid."])

    def sit(t, results=()):
        return "\n".join([f"NOW t={t:.1f} s. You are at (x=0.00, y=0.00), facing 0 degrees, in Ambulance handover bay.",
                          "You are holding: nothing.", "Heard recently, newest last:", "- 30 s ago, an alarm, 85 dB",
                          "People your camera sees (anonymous tracks):",
                          "- t2: 1.5 m, behind you, standing, still, at (x=-1.50, y=0.00)", "Objects in view:",
                          "- nothing notable", *(["Results of your recent actions:", *results] if results else []),
                          "Your last events, oldest first:", "- t=40.0s You started guide as rb9 for track t2.",
                          "Decide what to do now."])
    actions = ["none", "wait", "go_to", "approach", "guide", "follow", "tell_staff", "press_call"]
    r.decide(c, sit(39.0), actions)
    r.mem.update({"alarm": True, "announced": True, "told": {"t2": 10.0}, "guided": {"t2"}})
    r.pending = _Pending("guide", 40.0, "t2", aid="rb9", extra={"back": False})
    done = ["- rb9 walking to a goal: ok (arrived)"]
    d = r.decide(c, sit(70.0, done), actions)
    assert r.busy() and d["action"] not in ("go_to", "guide", "approach", "follow"), d
    d = r.decide(c, sit(80.0, done), actions)
    assert r.busy() and d["action"] not in ("go_to", "guide", "approach", "follow"), d
    d = r.decide(c, sit(91.5, done), actions)
    assert d["action"] == "go_to", d


def test_a_guide_held_up_by_someone_in_the_way_is_tried_again_round_them():
    """A guide that ends blocked_by_person short of the place, with the person still beside the robot, is started
    again (a new guide plans its route round people standing in the way) before the robot falls back to telling them
    to walk on; a flow credits only a guide that ends arrived."""
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    job = "When the alarm sounds, get everyone out by a safe open way."
    c = "\n".join(["THE HOSPITAL", "Places you know:", "- Main lobby and reception: Main lobby and reception (lobby)",
                   "", "YOUR JOB", "Role: fire evacuation guide", f"Job: {job}",
                   "Safe exit: the safe way out is the Ambulance handover bay in A&E.", "", "YOUR BODY", "A humanoid."])

    def sit(t, results=()):
        return "\n".join([f"NOW t={t:.1f} s. You are at (x=0.00, y=0.00), facing 0 degrees, in an unnamed space.",
                          "You are holding: nothing.", "Heard recently, newest last:", "- 30 s ago, an alarm, 85 dB",
                          "People your camera sees (anonymous tracks):",
                          "- t3: 1.5 m, behind you, standing, still, at (x=-1.50, y=0.00)",
                          "- t2: 0.9 m, straight ahead, standing, still, at (x=0.90, y=0.00)", "Objects in view:",
                          "- nothing notable", *(["Results of your recent actions:", *results] if results else []),
                          "Your last events, oldest first:", "- t=140.0s You started guide as rb22 for track t3.",
                          "Decide what to do now."])
    actions = ["none", "wait", "go_to", "approach", "guide", "follow", "tell_staff", "press_call"]
    r.decide(c, sit(139.0), actions)
    r.mem.update({"alarm": True, "announced": True, "told": {"t2": 10.0, "t3": 100.0}, "guided": {"t2", "t3"},
                  ("guide_to", "t3"): "Ambulance handover bay"})
    r.pending = _Pending("guide", 140.0, "t3", aid="rb22", extra={"back": False})
    d = r.decide(c, sit(172.8, ["- rb22 walking to a goal: failed (blocked_by_person)"]), actions)
    assert d["action"] == "guide" and d["target"] == "t3" and d["destination"] == "Ambulance handover bay", d
    assert d["return_to_post"] is False


def test_the_guide_is_tried_again_with_someone_following_out_of_view_behind():
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    c = "\n".join(["THE HOSPITAL", "Places you know:", "- Main lobby and reception: Main lobby and reception (lobby)",
                   "", "YOUR JOB", "Role: fire evacuation guide", "Job: When the alarm sounds, get everyone out.",
                   "Safe exit: the safe way out is the Ambulance handover bay in A&E.", "", "YOUR BODY", "A humanoid."])

    def sit(t, tracks, results=()):
        return "\n".join([f"NOW t={t:.1f} s. You are at (x=0.00, y=0.00), facing 0 degrees, in an unnamed space.",
                          "You are holding: nothing.", "Heard recently, newest last:", "- 30 s ago, an alarm, 85 dB",
                          "People your camera sees (anonymous tracks):", *(tracks or ["- nobody in view"]),
                          "Objects in view:", "- nothing notable",
                          *(["Results of your recent actions:", *results] if results else []),
                          "Your last events, oldest first:", "- t=140.0s You started guide as rb22 for track t3.",
                          "Decide what to do now."])
    actions = ["none", "wait", "go_to", "approach", "guide", "follow", "tell_staff", "press_call"]
    r.decide(c, sit(160.0, ["- t3: 1.8 m, ahead and to your left, standing, moving, at (x=-1.00, y=1.50)"]), actions)
    r.mem.update({"alarm": True, "announced": True, "told": {"t3": 100.0}, "guided": {"t3"},
                  ("guide_to", "t3"): "Ambulance handover bay"})
    r.pending = _Pending("guide", 140.0, "t3", aid="rb22", extra={"back": False})
    d = r.decide(c, sit(172.8, [], ["- rb22 walking to a goal: failed (blocked_by_person)"]), actions)
    assert d["action"] == "guide" and d["target"] == "t3", d
