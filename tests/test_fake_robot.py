"""The scripted built-in robot (WARDZERO_TEST_CLAUDE_BOT with the scripted model): it reads only the robot's own prompt
(its card and situation) and does each preset's main task through the robot's real action schema and speech."""
from wardzero_agents.llm.fake import default_policy
from wardzero_agents.llm.fake_robot import ScriptedRobot, _Pending, parse_card, parse_situation
from wardzero_agents.llm.gateway import LLMRequest

ACTIONS = ["none", "wait", "go_to", "approach", "guide", "follow", "tell_staff", "note_desk", "press_call", "take",
           "hand_over", "extinguish"]


def card(job: str, *brief: str, places=("- Lobby: Main lobby and reception (lobby)",)) -> str:
    return "\n".join(["THE HOSPITAL", "Places you know:", *places, "", "YOUR JOB", *brief[:1], f"Job: {job}",
                      *brief[1:], "", "YOUR BODY", "Your body: a humanoid."])


def situation(t=5.0, x=0.0, y=0.0, yaw=0.0, room="Main lobby and reception", held="nothing.", heard=(), tracks=(),
              objects=(), routes=(), results=(), events=()) -> str:
    lines = [f"NOW t={t:.1f} s. You are at (x={x:.2f}, y={y:.2f}), facing {yaw:.0f} degrees, in {room}.",
             f"You are holding: {held}", "Heard recently, newest last:", *(heard or ["- nothing"]),
             "People your camera sees (anonymous tracks):", *(tracks or ["- nobody in view"]), "Objects in view:",
             *(objects or ["- nothing notable"])]
    if routes:
        lines += ["Getting there from here (Main lobby and reception):", *routes, ""]
    if results:
        lines += ["Results of your recent actions:", *results]
    lines += ["Your last events, oldest first:", *(events or []), "Decide what to do now."]
    return "\n".join(lines)


def track(tid, dist, x, y, posture="standing", moving="still", extra=""):
    return f"- {tid}: {dist:.1f} m, straight ahead, {posture}{extra and ', ' + extra if extra == 'asleep' else ''}, " \
           f"{moving}, at (x={x:.2f}, y={y:.2f}){'' if extra == 'asleep' else extra}"


def decide(robot, c, s):
    return robot.decide(c, s, ACTIONS)


# ---- reading the prompt ---------------------------------------------------------------------------------------------
def test_the_parser_reads_asleep_tracks_and_the_routes_from_here():
    s = parse_situation(situation(tracks=[track("t1", 1.1, 1.0, 0.0, "lying down", extra="asleep"),
                                          track("t2", 2.0, 2.0, 0.0, extra=", at Main entrance, not spoken to yet")],
                                  routes=["- Ward entrance: about 14 m, setting off to your left, through Main lobby.",
                                          "- Main entrance: the way there is blocked (you saw a route blocked sign); "
                                          "no other way found.", "- In front of the reception desk: right beside you."]))
    t1, t2 = s.tracks
    assert t1.posture == "lying down" and t1.asleep and not t2.asleep and t2.new
    assert s.places == ["Ward entrance", "Main entrance", "In front of the reception desk"]
    assert s.route_dirs == {"Ward entrance": "to your left"} and s.blocked == {"Main entrance"}


def test_the_destination_comes_from_an_exit_line_or_the_place_the_job_names_last():
    r = ScriptedRobot()
    evac = parse_card(card("When the alarm sounds, get everyone out.", "Role: fire evacuation guide",
                           "Safe exit: With a fire at the main entrance, the safe way out is the Ambulance handover "
                           "bay in A&E (the ambulance doors)."))
    assert r.destination(evac) == "Ambulance handover bay"
    run = parse_card(card("You start at the ward entrance holding the box. Carry it down the main corridor to the "
                          "porter, who waits in the main lift lobby, and hand it to the porter yourself.",
                          "Role: specimen runner",
                          places=["- Ward_Entrance: Ward entrance, reception + nurse base (ward_entrance)",
                                  "- Main_Corridor: Main corridor (corridor)",
                                  "- Main_LiftLobby: Protected bed-lift lobby (lift_lobby)"]))
    assert r.destination(run) == "Protected bed-lift lobby"
    rush = parse_card(card("Get through the crowd to the Ward nurse base (Nurse workstation 1), not the reception at "
                           "the ward entrance where you start.", "Role: ward corridor runner"))
    assert r.destination(rush) == "Ward nurse base"


# ---- who spoke --------------------------------------------------------------------------------------------------------
def test_a_voice_from_behind_is_not_put_down_to_the_person_in_front():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    d = decide(r, c, situation(t=0.0, tracks=[track("t1", 4.5, -4.5, 0.0)]))  # post, and someone settled ahead
    s = situation(t=40.0, heard=['- 0 s ago, someone speaking to you, 48 dB, from about 162 degrees to your left: '
                                 '"Hello, could you take me to the ward, please?"'],
                  tracks=[track("t2", 1.6, 1.6, 0.0)])
    d = decide(r, c, s)
    assert d["action"] == "none" and d["toward_track"] == "" and "moment" in d["say"]  # who is unseen: held over


def test_a_voice_much_quieter_than_the_matched_track_is_from_someone_unseen():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    s = situation(heard=['- 0 s ago, someone speaking to you, 40 dB, from the direction of t1 (1.0 m): '
                         '"Excuse me, I need to get to the ward entrance."'], tracks=[track("t1", 1.0, 1.0, 0.0)],
                  routes=["- Ward entrance: about 14 m, setting off to your left."])
    assert decide(r, c, s)["action"] == "none"
    loud = situation(t=9.0, heard=['- 0 s ago, someone speaking to you, 60 dB, from the direction of t1 (1.0 m): '
                                   '"Excuse me, I need to get to the ward entrance."'],
                     tracks=[track("t1", 1.0, 1.0, 0.0)], routes=["- Ward entrance: about 14 m, setting off to your "
                                                                  "left."])
    d = decide(r, c, loud)
    assert d["action"] == "guide" and d["target"] == "t1" and d["destination"] == "Ward entrance"


# ---- the jobs ---------------------------------------------------------------------------------------------------------
def test_the_fire_responder_puts_out_the_fire_and_warns_the_room():
    r = ScriptedRobot()
    c = card("Put out the small fire in the corridor. Keep clear of staff moving patients.", "Role: ward fire responder")
    d = decide(r, c, situation(objects=["- fire F1: 3.2 m, straight ahead, size 0.4"]))
    assert d["action"] == "extinguish" and d["target"] == "F1" and d["toward_track"] == "room"


def test_the_scrub_assistant_passes_the_named_instrument_in_one_hand_over():
    r = ScriptedRobot()
    c = card("Stand beside the surgeon. When she names an instrument, hand it to her in one action.",
             "Role: scrub nurse assistant", "Instruments: scalpel, forceps, suture_scissors (suture scissors)")
    s = situation(heard=['- 0 s ago, someone speaking to you, 60 dB, from straight ahead: "Suture scissors, please."'],
                  tracks=[track("t1", 0.9, 0.9, 0.0)])
    d = decide(r, c, s)
    assert d["action"] == "hand_over" and d["item"] == "suture_scissors" and d["target"] == "t1"
    assert d["say"] == "Suture scissors."


def test_the_meal_assistant_skips_a_sleeper_and_tells_a_nurse_without_waking_them():
    r = ScriptedRobot()
    c = card("Bring each patient the right tray. Skip anyone who is asleep and tell a nurse.",
             "Role: ward meal assistant", "Meals:", "- patient bed: Ward bedroom B2; tray label: George Wright - diabetic",
             places=["- B2: Ward bedroom B2 (bedroom)"])
    s = situation(room="Ward bedroom B2", held="George Wright - diabetic (item tray_M2)",
                  tracks=[track("t2", 1.1, 1.1, 0.0, "lying down", extra="asleep")])
    d = decide(r, c, s)
    assert d["action"] == "tell_staff" and d["target"] == "t2" and "asleep" in d["message"] and not d["say"]


def test_the_night_assistant_presses_the_call_with_the_patients_track():
    r = ScriptedRobot()
    c = card("Answer a call bell while the nurses are busy. Go to the patient and reassure them. To fetch a nurse, stay "
             "with the patient and press the staff call button.", "Role: night ward assistant",
             places=["- B2: Ward bedroom B2 (bedroom)"])
    s = situation(room="Ward bedroom B2", tracks=[track("t1", 1.2, 1.2, 0.0, "lying down")],
                  objects=["- call light on at bed B2, 1.5 m, straight ahead"])
    first = decide(r, c, s)
    assert first["action"] == "none" and first["toward_track"] == "t1" and "call bell" in first["say"]
    d = decide(r, c, situation(t=7.0, room="Ward bedroom B2", tracks=[track("t1", 1.2, 1.2, 0.0, "lying down")],
                               objects=["- call light on at bed B2, 1.5 m, straight ahead"]))
    assert d["action"] == "press_call" and d["target"] == "t1"


def test_the_companion_calls_a_nurse_then_walks_her_back_and_stays():
    r = ScriptedRobot()
    c = card("Notice a patient who gets up and starts to leave the ward. Stay beside them and press the staff call "
             "button. Gently suggest going back to bed in B4 and walk them there if they will come.",
             "Role: night ward companion", places=["- B4: Ward bedroom B4 (bedroom)"])
    up = [track("t1", 1.5, 1.5, 0.0, moving="moving")]
    assert decide(r, c, situation(t=1.0, tracks=up))["action"] == "approach"
    r.pending = None
    assert decide(r, c, situation(t=3.0, tracks=up))["action"] == "press_call"
    r.pending = None
    d = decide(r, c, situation(t=5.0, tracks=up))
    assert d["action"] == "guide" and d["destination"] == "Ward bedroom B4" and d["return_to_post"] is False
    r.pending = None
    assert decide(r, c, situation(t=7.0, tracks=up))["action"] == "follow"


def test_the_evacuation_guide_tells_the_room_the_safe_exit_first():
    r = ScriptedRobot()
    c = card("When the alarm sounds, get everyone out by a safe open way. Once the people near you are on their way "
             "out, check the A&E waiting area and walk anyone still there out.", "Role: fire evacuation guide",
             "Safe exit: the safe way out is the Ambulance handover bay in A&E.")
    d = decide(r, c, situation(heard=["- 0 s ago, an alarm, 85 dB"], tracks=[track("t1", 2.0, 2.0, 0.0)]))
    assert d["toward_track"] == "room" and "Ambulance handover bay" in d["say"]


def test_the_courier_walks_to_the_porter_and_hands_the_box_to_the_person_at_the_porter_point():
    r = ScriptedRobot()
    c = card("Carry the specimen box to the porter, who waits in the main lift lobby, and hand it to the porter.",
             "Role: specimen runner", "Items: specimen box",
             places=["- Main_LiftLobby: Protected bed-lift lobby (lift_lobby)"])
    d = decide(r, c, situation(held="specimen box (item specimen_1)"))
    assert d["action"] == "go_to" and d["destination"] == "Protected bed-lift lobby"
    r.pending, r.mem["there"] = None, True
    s = situation(t=40.0, held="specimen box (item specimen_1)",
                  tracks=[track("t1", 1.0, 1.0, 0.0), track("t2", 1.1, 0.0, 1.1, extra=", at Porter waiting point")])
    d = decide(r, c, s)
    assert d["action"] == "hand_over" and d["target"] == "t2" and d["item"] == "specimen_1"


def test_the_robot_role_in_the_scripted_gateway_uses_the_scripted_robot():
    req = LLMRequest(role="robot", person="g1", system_fixed="", persona_card=card("Put out the small fire.",
                                                                                     "Role: ward fire responder"),
                     user=situation(objects=["- fire F9: 2.0 m, straight ahead, size 0.3"]),
                     schema={"properties": {"action": {"enum": ACTIONS}}})
    assert default_policy(req)["action"] == "extinguish"


def test_the_evacuation_guide_tells_everyone_in_view_before_walking_anyone_out():
    r = ScriptedRobot()
    c = card("When the alarm sounds, get everyone out by a safe open way.", "Role: fire evacuation guide",
             "Safe exit: the safe way out is the Ambulance handover bay in A&E.")
    alarm = ["- 1 s ago, an alarm, 85 dB"]
    both = [track("t1", 1.5, 1.5, 0.0), track("t2", 3.0, 3.0, 0.0)]
    assert decide(r, c, situation(t=2.0, heard=alarm, tracks=both))["toward_track"] == "room"
    first = decide(r, c, situation(t=3.0, heard=alarm, tracks=both))
    assert first["toward_track"] == "t1" and "Ambulance handover bay" in first["say"]
    assert "watching" in decide(r, c, situation(t=5.0, heard=alarm, tracks=both))["reason"]
    second = decide(r, c, situation(t=8.0, heard=alarm, tracks=both))
    assert second["toward_track"] == "t2"
    # t1 has not moved 12 s after being told: walked out, and the robot stays at the exit afterwards
    out = decide(r, c, situation(t=21.0, heard=alarm, tracks=both))
    assert out["action"] == "guide" and out["target"] == "t1" and out["return_to_post"] is False


def test_of_two_people_that_way_the_one_walking_up_is_taken_for_the_speaker():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    s = situation(yaw=-90.0, heard=['- 1 s ago, someone speaking to you, 50 dB, from about 35 degrees to your right: '
                                    '"Excuse me, can you help us? I need to get to the ward entrance."'],
                  tracks=["- t1: 2.1 m, ahead and to your right, standing, moving, at (x=-1.35, y=-1.55)",
                          "- t2: 5.2 m, ahead and to your right, standing, still, at (x=-3.05, y=-4.22)"],
                  routes=["- Ward entrance: about 14 m, setting off to your left."])
    d = decide(r, c, s)
    assert d["action"] == "guide" and d["target"] == "t1"


def test_the_reception_guide_walks_back_to_its_post_after_a_walk_that_ended_away_from_it():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    post = ["- Main entrance: about 5 m, setting off ahead and to your right.",
            "- In front of the reception desk: right beside you.",
            "- Corridor to the outpatient clinics: about 8 m, setting off to your left."]
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=post))
    # a guide that ended blocked, far from the desk: nobody to help here, so back to the desk
    away = ["- Main entrance: about 12 m, setting off ahead and to your left.",
            "- In front of the reception desk: about 8 m, setting off ahead and to your left.",
            "- Corridor to the outpatient clinics: right beside you."]
    d = decide(r, c, situation(t=48.9, x=6.17, y=5.79, yaw=-161.0, room="Main corridor", routes=away,
                               tracks=["- t4: 4.3 m, straight ahead, seated, still, at (x=2.00, y=4.75), "
                                       "not spoken to yet"]))
    assert d["action"] == "go_to" and d["destination"] == "In front of the reception desk"
    # back at the desk, a newcomer at the entrance is greeted
    r.pending = None
    d = decide(r, c, situation(t=60.0, x=1.0, y=1.5, yaw=-140.0, routes=post,
                               tracks=["- t5: 4.8 m, ahead and to your right, standing, still, at (x=-1.20, y=-2.57), "
                                       "at Main entrance, not spoken to yet"]))
    assert d["action"] == "approach" and d["target"] == "t5"


def test_the_evacuation_guide_goes_back_for_someone_it_saw_in_the_room_it_checks():
    r = ScriptedRobot()
    c = card("When the alarm sounds, get everyone out by a safe open way. Once the people near you are on their way "
             "out, check the A&E waiting area and walk anyone still there out.", "Role: fire evacuation guide",
             "Safe exit: the safe way out is the Ambulance handover bay in A&E.")
    alarm = ["- 1 s ago, an alarm, 85 dB"]
    assert decide(r, c, situation(t=2.0, x=2.5, y=-0.5, heard=alarm))["toward_track"] == "room"
    # while walking someone else out, the camera catches someone standing still in the A&E waiting area
    r.pending = _Pending("guide", 41.0, "t2", aid="rb9")
    r.mem["guided"].add("t2")
    d = decide(r, c, situation(t=46.2, x=-3.94, y=1.84, yaw=176.0, heard=alarm,
                               tracks=["- t3: 5.4 m, ahead and to your left, standing, still, at (x=-9.00, y=0.00)"]))
    assert d["action"] == "none"
    # later, checking the A&E waiting area, they are just behind the robot, out of view: go back to them
    r.pending, r.mem["checked_other"] = None, True
    routes = ["- Main lobby and reception: about 9 m, setting off ahead and to your left, through A&E walk-in "
              "reception + 15-seat waiting, then Main lobby and reception.",
              "- A&E walk-in reception + 15-seat waiting: right beside you.",
              "- Ambulance handover bay: about 25 m, setting off behind you, through A&E walk-in reception + 15-seat "
              "waiting, then Ambulance handover bay."]
    d = decide(r, c, situation(t=136.8, x=-8.43, y=-0.54, yaw=-22.0, room="A&E walk-in reception + 15-seat waiting",
                               routes=routes))
    assert d["action"] == "approach" and d["target"] == "t3"
    # once beside them and they are in view, they are walked out
    r.pending = None
    d = decide(r, c, situation(t=140.0, x=-8.6, y=-0.4, yaw=140.0, room="A&E walk-in reception + 15-seat waiting",
                               routes=routes, tracks=[track("t3", 0.9, -9.0, 0.0)]))
    assert d["action"] == "guide" and d["target"] == "t3" and d["return_to_post"] is False


def test_an_action_the_body_cannot_do_leaves_nothing_pending():
    r = ScriptedRobot()
    c = card("Carry the specimen box to the porter, who waits in the main lift lobby, and hand it to the porter.",
             "Role: specimen runner", "Items: specimen box",
             places=["- Main_LiftLobby: Protected bed-lift lobby (lift_lobby)"])
    fixed = [a for a in ACTIONS if a not in ("go_to", "approach", "guide", "follow")]
    d = r.decide(c, situation(held="specimen box (item specimen_1)"), fixed)
    assert d["action"] == "none" and r.pending is None
