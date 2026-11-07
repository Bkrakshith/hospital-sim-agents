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


# ---- evacuation: every person walked out by the robot ------------------------------------------------------------------
_EVAC_JOB = ("When the alarm sounds, get everyone out by a safe open way. Once the people near you are on their way "
             "out, check the A&E waiting area and walk anyone still there out.")
_EVAC_CARD = "\n".join(["THE HOSPITAL", "Places you know:", "- Lobby: Main lobby and reception (lobby)",
                        "- AE_Waiting: A&E walk-in reception + 15-seat waiting (waiting)",
                        "- AE_Handover: Ambulance handover bay (handover)", "", "YOUR JOB",
                        "Role: fire evacuation guide", f"Job: {_EVAC_JOB}",
                        "Safe exit: the safe way out is the Ambulance handover bay in A&E.", "", "YOUR BODY",
                        "A humanoid."])
_EVAC_ACTIONS = ["none", "wait", "go_to", "approach", "guide", "follow", "tell_staff", "press_call"]


def _evac_sit(t, *, room="Main lobby and reception", x=0.0, y=0.0, heard=(), tracks=(), results=(), events=()):
    return "\n".join([f"NOW t={t:.1f} s. You are at (x={x:.2f}, y={y:.2f}), facing 0 degrees, in {room}.",
                      "You are holding: nothing.", "Heard recently, newest last:", "- 1 s ago, an alarm, 85 dB",
                      *heard, "People your camera sees (anonymous tracks):", *(tracks or ["- nobody in view"]),
                      "Objects in view:", "- nothing notable",
                      *(["Results of your recent actions:", *results] if results else []),
                      "Your last events, oldest first:", *events, "Decide what to do now."])


def test_someone_frozen_while_walked_out_is_calmed_and_offered_the_walk_again_at_once():
    """The person it walks out freezes and calls out from behind it (out of the camera's view): a calm word to them
    with the walk offered again, once for that line, while the walk goes on otherwise."""
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    r.decide(_EVAC_CARD, _evac_sit(11.0, tracks=["- t2: 1.5 m, straight ahead, standing, still, at (x=1.50, y=0.00)"]),
             _EVAC_ACTIONS)
    r.mem.update({"alarm": True, "alarm_t": 1.0, "announced": True, "guided": {"t2"},
                  ("guide_to", "t2"): "Ambulance handover bay"})
    r.pending = _Pending("guide", 12.2, "t2", aid="rb3", extra={"back": False})
    cry = ['- 0 s ago, someone nearby, 55 dB, from behind you: "I don\'t know where to go!"']
    d = r.decide(_EVAC_CARD, _evac_sit(18.9, x=-2.0, heard=cry), _EVAC_ACTIONS)
    assert d["action"] == "guide" and d["target"] == "t2" and d["toward_track"] == "t2", d
    assert d["destination"] == "Ambulance handover bay" and d["return_to_post"] is False
    assert "right here with you" in d["say"]
    d = r.decide(_EVAC_CARD, _evac_sit(19.5, x=-2.2, heard=cry), _EVAC_ACTIONS)
    assert d["action"] == "none" and d["reason"] == "guiding", d


def test_a_word_of_thanks_from_the_person_walked_out_does_not_restart_the_walk():
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    r.decide(_EVAC_CARD, _evac_sit(11.0), _EVAC_ACTIONS)
    r.mem.update({"alarm": True, "alarm_t": 1.0, "announced": True, "guided": {"t2"}})
    r.pending = _Pending("guide", 12.2, "t2", aid="rb3", extra={"back": False})
    d = r.decide(_EVAC_CARD, _evac_sit(14.0, heard=['- 0 s ago, someone nearby, 55 dB, from behind you: "Thank you"']),
                 _EVAC_ACTIONS)
    assert d["action"] == "none" and d["reason"] == "guiding", d


def test_after_walking_one_out_it_goes_back_for_the_person_it_left_in_the_lobby_first():
    """Back at the exit with nobody in view: it goes back for the person it saw standing still in the lobby when the
    alarm sounded before the one it passed on the way (the room the job names is checked after)."""
    from wardzero_agents.llm.fake_robot import _Pending
    r = ScriptedRobot()
    lobby = ["- t1: 4.7 m, straight ahead, standing, still, at (x=-2.00, y=1.00)",
             "- t2: 1.6 m, straight ahead, standing, still, at (x=1.00, y=-0.90)"]
    d = r.decide(_EVAC_CARD, _evac_sit(0.5, x=2.5, y=-0.5, tracks=lobby), _EVAC_ACTIONS)
    assert d["toward_track"] == "room", d  # the word to the room
    d = r.decide(_EVAC_CARD, _evac_sit(2.1, x=2.5, y=-0.5, tracks=lobby), _EVAC_ACTIONS)
    assert d["action"] == "guide" and d["target"] == "t2", d
    r.pending.aid = "rb3"
    # on the way it passes someone seated still in the A&E waiting area
    r.decide(_EVAC_CARD, _evac_sit(40.0, x=-8.0, y=2.0, room="A&E walk-in reception + 15-seat waiting",
                                   tracks=["- t3: 2.0 m, to your left, seated, still, at (x=-9.00, y=0.00)"]),
             _EVAC_ACTIONS)
    done = ["- rb3 walking to a goal: ok (arrived)"]
    r.decide(_EVAC_CARD, _evac_sit(60.2, x=-29.1, y=6.7, room="Ambulance handover bay", results=done), _EVAC_ACTIONS)
    d = r.decide(_EVAC_CARD, _evac_sit(82.0, x=-29.1, y=6.7, room="Ambulance handover bay", results=done),
                 _EVAC_ACTIONS)
    assert d["action"] == "approach" and d["target"] == "t1", d


def test_in_an_evacuation_someone_who_never_came_along_is_not_gone_back_for_or_sent_on():
    """The guide is given up when the person never comes within 2.5 m for 30 s (a staff member who stays at their
    post): on the wire its goal ends "failed (cancelled)". The robot neither turns back to them nor tells them they
    are nearly there; it walks the next person out."""
    from test_fake_robot import card, decide, situation, track
    r = ScriptedRobot()
    c = card("When the alarm sounds, get everyone out by a safe open way.", "Role: fire evacuation guide",
             "Safe exit: the safe way out is the Ambulance handover bay in A&E.")
    alarm = ["- 1 s ago, an alarm, 85 dB"]
    both = [track("t1", 1.5, 1.5, 0.0), track("t2", 3.0, 3.0, 0.0)]
    decide(r, c, situation(t=2.0, heard=alarm, tracks=both))
    d = decide(r, c, situation(t=3.0, heard=alarm, tracks=both))
    assert d["action"] == "guide" and d["target"] == "t1", d
    res = ["- rb1 guide_start: ok (started)", "- rb1 walking to a goal: failed (cancelled)"]
    ev = ["- t=3.0s You started guide as rb1 for track t1."]
    d = decide(r, c, situation(t=40.0, x=5, y=0, heard=alarm, results=res, events=ev,
                               tracks=[track("t1", 1.0, 6.0, 0.0), track("t2", 3.0, 3.0, 0.0)]))
    assert d["action"] == "guide" and d["target"] == "t2", d
    assert "nearly there" not in (d["say"] or "")


def test_a_go_to_is_not_ended_by_the_walk_it_replaced():
    """Third QA (Isaac meal_round): after the bell the scripted robot re-sent its walk every 2 s, each new goal_pose
    superseding the last, because the superseded result of the walk it had replaced counted as its own walk's end."""
    from wardzero_agents.llm.fake_robot import Result, _Pending
    bot = ScriptedRobot()
    bot.pending = _Pending("go_to", 90.0)
    assert bot._finished([Result("goal_pose-2", "walking to a goal", False, "superseded")], 90.6) is None
    end = Result("goal_pose-3", "walking to a goal", True, "arrived")
    assert bot._finished([end], 110.0) is end
