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


def test_the_evacuation_guide_walks_each_person_out_nearest_first():
    """A flow credits the robot only for a walk that arrived: after the word to the room it walks the nearest person
    out itself (rather than sending them alone), and stays at the exit afterwards."""
    r = ScriptedRobot()
    c = card("When the alarm sounds, get everyone out by a safe open way.", "Role: fire evacuation guide",
             "Safe exit: the safe way out is the Ambulance handover bay in A&E.")
    alarm = ["- 1 s ago, an alarm, 85 dB"]
    both = [track("t1", 1.5, 1.5, 0.0), track("t2", 3.0, 3.0, 0.0)]
    room = decide(r, c, situation(t=2.0, heard=alarm, tracks=both))
    assert room["toward_track"] == "room" and "walk each of you out" in room["say"]
    out = decide(r, c, situation(t=3.0, heard=alarm, tracks=both))
    assert out["action"] == "guide" and out["target"] == "t1" and out["return_to_post"] is False
    assert out["destination"] == "Ambulance handover bay"
    assert decide(r, c, situation(t=5.0, heard=alarm, tracks=both))["reason"] == "guiding"


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


def test_of_two_still_people_that_way_the_one_as_far_as_the_voice_sounds_is_the_speaker():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    # 49 dB is a voice from about 3.5 m: the person at 3.5 m, not the one at 5.2 m a degree nearer the bearing
    s = situation(t=0.8, x=1.05, y=1.62, yaw=-90.0,
                  heard=['- 0 s ago, someone speaking to you, 49 dB, from about 35 degrees to your right: '
                         '"Excuse me, can you help us? I need to get to the ward entrance."'],
                  tracks=["- t1: 3.5 m, ahead and to your right, standing, still, at (x=-1.20, y=-1.00), "
                          "not spoken to yet",
                          "- t2: 5.2 m, ahead and to your right, standing, still, at (x=-2.00, y=-2.60), at Main "
                          "entrance, not spoken to yet",
                          "- t3: 2.2 m, ahead and to your left, standing, still, at (x=2.00, y=-0.40)"],
                  routes=["- Ward entrance: about 14 m, setting off to your left."])
    d = decide(r, c, s)
    assert d["action"] == "guide" and d["target"] == "t1"


def test_a_faint_voice_from_far_off_is_not_put_down_to_someone_close_that_way():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    # t1 asked for the ward and was walked there; the walk ended short of it
    first = situation(t=1.4, heard=['- 0 s ago, someone speaking to you, 50 dB, from the direction of t1 (2.6 m): '
                                    '"Excuse me, I need to get to the ward entrance."'],
                      tracks=[track("t1", 2.6, -0.62, -0.32)], routes=["- Ward entrance: about 14 m, setting off to "
                                                                        "your left."])
    assert decide(r, c, first)["target"] == "t1"
    r.pending = None
    # 35 dB is a voice from well over 10 m: not the person 1.2 m away in that direction
    s = situation(t=28.5, x=11.91, y=7.20, yaw=152.0,
                  heard=['- 1 s ago, someone speaking to you, 35 dB (only part heard), from about 56 degrees to your '
                         'left: "… … … you help … I … … get to the corridor to the outpatient …"'],
                  tracks=["- t1: 1.2 m, ahead and to your left, standing, still, at (x=10.78, y=6.90)"],
                  routes=["- Corridor to the outpatient clinics: about 7 m, setting off ahead and to your left."])
    d = decide(r, c, s)
    assert d["action"] != "guide" and d["toward_track"] != "t1"
    # nor, held over, is the person it already walked somewhere, close by that way, taken for them afterwards
    later = situation(t=31.0, x=11.91, y=7.20, yaw=152.0,
                      tracks=["- t1: 1.2 m, ahead and to your left, standing, still, at (x=10.78, y=6.90)"],
                      routes=["- Corridor to the outpatient clinics: about 7 m, setting off ahead and to your left."])
    d = decide(r, c, later)
    assert d["target"] != "t1" and d["toward_track"] != "t1"


def test_the_reception_guide_looks_for_someone_who_asked_before_it_walks_back_to_its_post():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=["- In front of the reception desk: right beside "
                                                                     "you."]))
    r.mem["deferred"] = [("", None, (12.9, 7.2, 50.0), 17.0)]  # an unseen voice, heard at t=17
    r.mem["looked_t"] = 28.0  # it has just turned that way and is waiting to see who comes into view
    away = ["- In front of the reception desk: about 13 m, setting off behind you."]
    d = decide(r, c, situation(t=30.0, x=12.9, y=7.2, yaw=-138.0, room="Main corridor", routes=away))
    assert d["reason"] != "back to my post"
    d = decide(r, c, situation(t=200.0, x=12.9, y=7.2, yaw=-138.0, room="Main corridor", routes=away))
    assert d["action"] == "go_to" and d["reason"] == "back to my post"


def test_the_reception_guide_sends_someone_on_when_a_walk_with_them_is_held_up_short():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=["- In front of the reception desk: right beside "
                                                                     "you."]))
    ask = situation(t=34.8, x=9.21, y=6.94, yaw=-157.0,
                    heard=['- 0 s ago, someone speaking to you, 55 dB, from the direction of t2 (2.0 m): '
                           '"Excuse me, I need to get to the corridor to the outpatient clinics."'],
                    tracks=["- t2: 2.0 m, straight ahead, standing, moving, at (x=7.82, y=5.50)"],
                    routes=["- Corridor to the outpatient clinics: about 4 m, setting off straight ahead."])
    d = decide(r, c, ask)
    assert d["action"] == "guide" and d["target"] == "t2"
    held = situation(t=47.7, x=6.58, y=6.05, yaw=-147.0, room="Main corridor",
                     tracks=["- t2: 1.5 m, behind you, standing, still, at (x=7.60, y=6.90)"],
                     routes=["- Corridor to the outpatient clinics: about 2 m, setting off straight ahead.",
                             "- In front of the reception desk: about 8 m, setting off ahead and to your left."],
                     results=["- rb8 guide_start: ok (started)", "- rb8 walking to a goal: failed (blocked_by_person)"],
                     events=["- t=34.8s You started guide as rb8 for track t2."])
    d = decide(r, c, held)  # held up by someone in the way: tried once more (a new guide plans round them)
    assert d["action"] == "guide" and d["target"] == "t2" and d["destination"] == "Corridor to the outpatient clinics"
    again = situation(t=60.0, x=6.58, y=6.05, yaw=-147.0, room="Main corridor",
                      tracks=["- t2: 1.5 m, behind you, standing, still, at (x=7.60, y=6.90)"],
                      routes=["- Corridor to the outpatient clinics: about 2 m, setting off straight ahead.",
                              "- In front of the reception desk: about 8 m, setting off ahead and to your left."],
                      results=["- rb9 guide_start: ok (started)", "- rb9 walking to a goal: failed (blocked_by_person)"],
                      events=["- t=34.8s You started guide as rb8 for track t2.",
                              "- t=47.7s You started guide as rb9 for track t2."])
    d = decide(r, c, again)
    assert d["toward_track"] == "t2" and "walk on" in d["say"] and "Corridor to the outpatient clinics" in d["say"]


def test_the_reception_guide_answers_someone_it_walks_up_to_once_they_are_close():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0))
    req = '"Excuse me, can you help us? I need to get to the corridor to the outpatient clinics."'
    r.pending = _Pending("guide", 1.0, "t1", aid="rb2")  # busy walking someone else: held over
    d = decide(r, c, situation(t=17.5, x=13.1, y=7.2, yaw=-48.0,
                               heard=[f'- 0 s ago, someone speaking to you, 50 dB, from the direction of t2 (3.0 m): '
                                      f'{req}'], tracks=["- t2: 6.0 m, straight ahead, standing, still, at "
                                                         "(x=7.0, y=5.0)"]))
    assert "moment" in d["say"]
    r.pending = None
    d = decide(r, c, situation(t=32.3, x=10.9, y=7.9, yaw=-173.0,
                               tracks=["- t2: 5.7 m, straight ahead, standing, still, at (x=6.46, y=4.28)"]))
    assert d["action"] == "approach" and d["target"] == "t2"
    # they walk up too: once they are close, the robot answers without waiting for its walk to end
    d = decide(r, c, situation(t=38.9, x=8.71, y=6.25, yaw=-142.0,
                               tracks=["- t2: 1.0 m, straight ahead, standing, still, at (x=8.42, y=5.90)"],
                               routes=["- Corridor to the outpatient clinics: about 4 m, setting off straight ahead."]))
    assert d["action"] == "guide" and d["target"] == "t2" and d["destination"] == "Corridor to the outpatient clinics"


def test_the_reception_guide_leaves_the_walk_back_after_a_guide_to_the_robot_and_walks_back_only_if_it_fails():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    post = ["- In front of the reception desk: right beside you.", "- Ward entrance: about 15 m, setting off ahead."]
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=post))
    ask = situation(t=40.0, x=1.05, y=1.62, yaw=-90.0, routes=post,
                    heard=['- 0 s ago, someone speaking to you, 54 dB, from the direction of t3 (2.0 m): '
                           '"Hello, could you take me to the ward entrance, please?"'],
                    tracks=["- t3: 2.0 m, straight ahead, standing, still, at (x=1.05, y=-0.38)"])
    d = decide(r, c, ask)
    assert d["action"] == "guide" and d["target"] == "t3" and d.get("return_to_post", True) is not False
    away = ["- In front of the reception desk: about 15 m, setting off behind you.",
            "- Ward entrance: right beside you."]
    started = ["- rb15 guide_start: ok (started)"]
    ev = ["- t=40.0s You started guide as rb15 for track t3."]
    decide(r, c, situation(t=45.0, x=4.0, y=4.0, yaw=0.0, room="Main corridor", results=started, events=ev))
    # the guide's walk arrives: the robot walks itself back to its post (goal "rb15-return"), so no go_to of its own
    arrived = started + ["- rb15 walking to a goal: ok (arrived)"]
    for t in (56.8, 70.0, 85.0):
        d = decide(r, c, situation(t=t, x=13.09, y=7.24, yaw=0.0, room="Main corridor", routes=away,
                                   results=arrived, events=ev))
        assert d["action"] == "none" and d["reason"] != "back to my post", (t, d)
    # the walk back is stopped short (superseded): now the robot walks back itself
    stopped = arrived + ["- rb15-return walking to a goal: failed (superseded)"]
    d = decide(r, c, situation(t=90.0, x=8.0, y=5.0, yaw=0.0, room="Main corridor", routes=away, results=stopped,
                               events=ev))
    assert d["action"] == "go_to" and d["reason"] == "back to my post"


def test_the_reception_guide_walks_back_itself_when_no_walk_back_comes_after_a_guide():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    post = ["- In front of the reception desk: right beside you."]
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=post))
    r.pending = _Pending("guide", 40.0, "t3", aid="rb15")
    away = ["- In front of the reception desk: about 15 m, setting off behind you."]
    arrived = ["- rb15 guide_start: ok (started)", "- rb15 walking to a goal: ok (arrived)"]
    d = decide(r, c, situation(t=56.8, x=13.09, y=7.24, room="Main corridor", routes=away, results=arrived))
    assert d["reason"] != "back to my post"
    # long after the guide ended it is still away and no walk back reported: it goes back itself
    d = decide(r, c, situation(t=56.8 + ScriptedRobot.RETURN_WAIT_S + 1.0, x=13.09, y=7.24, room="Main corridor",
                               routes=away, results=arrived))
    assert d["action"] == "go_to" and d["reason"] == "back to my post"


def test_each_scripted_gateway_has_its_own_robot_so_one_run_leaves_nothing_to_the_next():
    import asyncio

    from wardzero_agents.llm.fake import FakeGateway
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    places = ["- In front of the reception desk: right beside you.", "- Ward entrance: about 15 m, setting off ahead."]

    def req(s):
        return LLMRequest(role="robot", person="g1", system_fixed="", persona_card=c, user=s,
                          schema={"properties": {"action": {"enum": ACTIONS}}})

    ask = situation(t=0.5, routes=places, heard=['- 0 s ago, someone speaking to you, 54 dB, from the direction of t3 '
                                                 '(2.0 m): "Could you take me to the ward entrance, please?"'],
                    tracks=["- t3: 2.0 m, straight ahead, standing, still, at (x=2.00, y=0.00)"])
    first = asyncio.run(FakeGateway().complete(req(ask)))
    assert first.data["action"] == "guide"
    # a second run in the same process, same card, starting at t=0: nothing pending from the first
    second = asyncio.run(FakeGateway().complete(req(situation(t=0.0, routes=places))))
    assert second.data["reason"] != "carrying on", second.data


def test_someone_who_walked_up_since_they_spoke_is_the_speaker_though_their_voice_was_from_farther():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    places = ["- In front of the reception desk: right beside you.", "- Ward entrance: about 15 m, setting off ahead."]
    decide(r, c, situation(t=0.8, x=1.05, y=1.62, yaw=-90.0, routes=places))
    # 49 dB is about 3.5 m away; they were walking up and are 1.3 m away now
    d = decide(r, c, situation(t=2.9, x=1.05, y=1.62, yaw=-90.0, routes=places,
                               heard=['- 1 s ago, someone speaking to you, 49 dB, from the direction of t1 (1.3 m): '
                                      '"Excuse me, can you help us? I need to get to the ward entrance."'],
                               tracks=["- t1: 1.3 m, ahead and to your right, standing, moving, at (x=0.18, y=0.63), "
                                       "at In front of the reception desk, not spoken to yet"]))
    assert d["action"] == "guide" and d["target"] == "t1", d


def test_a_far_voice_heard_after_a_guide_is_looked_for_once_the_walk_back_brings_the_robot_to_its_post():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    at_post = ["- In front of the reception desk: right beside you.",
               "- Corridor to the outpatient clinics: about 6 m, setting off to your left.",
               "- Ward entrance: about 15 m, setting off ahead and to your left."]
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=at_post))
    r.pending = _Pending("guide", 1.4, "t1", aid="rb2")
    away = ["- In front of the reception desk: about 15 m, setting off behind you.",
            "- Corridor to the outpatient clinics: about 9 m, setting off behind you.",
            "- Ward entrance: right beside you."]
    done = ["- rb2 guide_start: ok (started)", "- rb2 walking to a goal: ok (arrived)"]
    decide(r, c, situation(t=16.9, x=13.11, y=7.24, yaw=27.0, room="Ward entrance", routes=away, results=done))
    # a faint request from far behind, from someone out of view (by the main entrance)
    far = ('- 0 s ago, someone speaking to you, 35 dB (only part heard), from about 180 degrees to your left: '
           '"… … … you help … I … … get to the corridor to the outpatient …"')
    d = decide(r, c, situation(t=17.0, x=13.11, y=7.24, yaw=27.0, room="Ward entrance", routes=away, results=done,
                               heard=[far]))
    assert d["say"] and d["action"] == "none"
    # the walk back is under way: no walk of its own to look round, and a person seated at the desk on the way,
    # nearer than the voice sounded, is not taken for the speaker
    for t, x, y in ((27.4, 11.05, 7.71), (34.1, 7.64, 6.26)):
        d = decide(r, c, situation(t=t, x=x, y=y, yaw=-150.0, room="Main corridor", routes=away, results=done,
                                   tracks=["- t4: 5.8 m, straight ahead, seated, still, at (x=2.00, y=4.75), "
                                           "not spoken to yet"]))
        assert d["action"] == "none" and d["target"] != "t4", (t, d)
    # back at its post: the person by the main entrance, where the voice came from, is the one it goes to
    back = done + ["- rb2-return walking to a goal: ok (arrived)"]
    d = decide(r, c, situation(t=40.0, x=1.10, y=1.60, yaw=-150.0, routes=at_post, results=back,
                               tracks=["- t4: 3.3 m, ahead and to your right, seated, still, at (x=2.00, y=4.75)",
                                       "- t6: 5.0 m, ahead and to your right, standing, still, at (x=-1.92, y=-2.55), "
                                       "at Main entrance, not spoken to yet"]))
    assert d["target"] == "t6", d


def test_someone_who_comes_over_that_way_while_the_robot_walks_back_is_taken_for_the_far_speaker():
    r = ScriptedRobot()
    c = card("Greet people who come to reception and walk them where they are going.", "Role: reception guide")
    at_post = ["- In front of the reception desk: right beside you.",
               "- Corridor to the outpatient clinics: about 6 m, setting off to your left."]
    decide(r, c, situation(t=0.0, x=1.05, y=1.62, yaw=-90.0, routes=at_post))
    r.pending = _Pending("guide", 0.9, "t1", aid="rb2")
    away = ["- In front of the reception desk: about 15 m, setting off behind you.",
            "- Corridor to the outpatient clinics: about 9 m, setting off behind you."]
    done = ["- rb2 guide_start: ok (started)", "- rb2 walking to a goal: ok (arrived)"]
    decide(r, c, situation(t=16.8, x=12.89, y=7.21, yaw=7.0, room="Ward entrance", routes=away, results=done))
    far = ('- 0 s ago, someone speaking to you, 35 dB (only part heard), from about 160 degrees to your right: '
           '"… … … you help … I … … get to the corridor to the outpatient …"')
    decide(r, c, situation(t=16.9, x=12.89, y=7.21, yaw=7.0, room="Ward entrance", routes=away, results=done,
                           heard=[far]))
    d = decide(r, c, situation(t=36.9, x=8.61, y=6.95, yaw=-136.0, room="Main corridor", routes=away, results=done,
                               tracks=["- t2: 1.2 m, straight ahead, standing, still, at (x=7.84, y=6.03), "
                                       "not spoken to yet"]))
    assert d["action"] == "guide" and d["target"] == "t2" and "outpatient" in d["destination"].lower(), d
