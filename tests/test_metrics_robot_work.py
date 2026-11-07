"""Scoring credits the robot's own work (second fix round): people reached only by a walk the robot finished, the
frozen person helped in time, requests the robot heard answered to the asker, bells answered, and the speed near
people measured on the robot's own motion."""
import json
from pathlib import Path

import pytest

from wardzero_agents.bench.criteria import evaluate
from wardzero_agents.bench.metrics import METRICS, RunData, not_applicable
from wardzero_agents.config.presets import load_scenario_presets

ROOT = Path(__file__).resolve().parents[1]
PRESETS = load_scenario_presets(ROOT / "config")


def make_run(path, meta=None, **streams):
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta or {}))
    for name, rows in streams.items():
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return RunData.load(path)


def val(d, name):
    return METRICS[name](d)


def status(t, person, x, y, posture="standing"):
    return {"t": t, "person": person, "x": x, "y": y, "yaw": 0.0, "posture": posture, "state": "idle",
            "progress": 0.0}


def pose(t, x, y, speed=0.0, yaw=0.0):
    return {"t": t, "x": x, "y": y, "yaw": yaw, "speed": speed}


def row(by_id, crit):
    return next(r for r in by_id.rows if r["id"] == crit)


# ---- item 2: people reached only through a walk the robot finished ------------------------------------------------
REC_META = {"preset_id": "reception_guide", "scenario_id": "reception_guide",
            "people": {"V1": {"role": "ward_visitor"}, "H1": {"role": "outpatient"}, "RCP": {"role": "receptionist"}}}


def _walk(person, frm, to, t0, t1, step=1.0):
    out, t = [], t0
    while t <= t1 + 1e-9:
        f = (t - t0) / (t1 - t0)
        out.append(status(t, person, frm[0] + f * (to[0] - frm[0]), frm[1] + f * (to[1] - frm[1]), "walking"))
        t += step
    return out


def test_people_who_walk_to_their_places_alone_are_not_reached_by_the_robot(tmp_path):
    # no robot at all: V1 walks to the ward entrance and H1 to the clinics on their own
    d = make_run(tmp_path / "alone", REC_META,
                 status=_walk("V1", (-2.0, 1.0), (8.0, 6.0), 30.0, 60.0) + _walk("H1", (-2.0, -2.6), (30.0, 4.0),
                                                                                  30.0, 90.0)
                 + [status(300.0, "V1", 8.0, 6.0), status(300.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") is None
    r = evaluate(PRESETS["reception_guide"], d)
    assert row(r, "reached")["status"] == "fail" and r.passed is False


def test_a_guide_that_ended_blocked_does_not_count_when_the_person_then_walks_there(tmp_path):
    # the Isaac case: the guide with H1 starts, never moves and ends blocked; H1 walks to the clinics alone
    tasks = [{"t": 76.4, "kind": "action", "id": "rb17", "action": "guide_start", "track": "t2",
              "place": "clinic_corridor"},
             {"t": 76.4, "kind": "guide", "event": "start", "id": "rb17", "person": "H1", "place": "clinic_corridor",
              "x": 30.0, "y": 4.0, "room": None, "route_m": 20.0},
             {"t": 76.4, "kind": "result", "id": "rb17", "action": "guide_start", "ok": True, "reason": "started"},
             {"t": 84.5, "kind": "guide", "event": "end", "id": "rb17", "person": "H1",
              "reason": "blocked_by_person"}]
    d = make_run(tmp_path / "blocked", REC_META, robot_tasks=tasks,
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(300.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == 0.0
    assert evaluate(PRESETS["reception_guide"], d).passed is False
    # the same walk ending arrived is the robot's
    tasks[-1] = dict(tasks[-1], reason="arrived", t=118.0)
    d = make_run(tmp_path / "arrived", REC_META, robot_tasks=tasks,
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(300.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == 1.0


def test_a_guide_restarted_for_the_same_person_is_judged_once(tmp_path):
    tasks = [{"t": 10.0, "kind": "guide", "event": "start", "id": "a1", "person": "P1", "place": "AE_Handover"},
             {"t": 15.0, "kind": "guide", "event": "end", "id": "a1", "person": "P1", "reason": "superseded"},
             {"t": 15.0, "kind": "guide", "event": "start", "id": "a2", "person": "P1", "place": "AE_Handover"},
             {"t": 60.0, "kind": "guide", "event": "end", "id": "a2", "person": "P1", "reason": "arrived"},
             {"t": 70.0, "kind": "guide", "event": "start", "id": "a3", "person": "V1", "place": "AE_Handover"},
             {"t": 80.0, "kind": "guide", "event": "end", "id": "a3", "person": "V1", "reason": "left_behind"}]
    d = make_run(tmp_path, {}, robot_tasks=tasks, status=[status(200.0, "P1", 0.0, 0.0)])
    assert val(d, "people_reached_destination_rate") == pytest.approx(0.5)


_H1_HELD = [{"t": 76.4, "kind": "action", "id": "rb17", "action": "guide_start", "track": "t2",
             "place": "clinic_corridor"},
            {"t": 76.4, "kind": "guide", "event": "start", "id": "rb17", "person": "H1", "place": "clinic_corridor",
             "x": 30.0, "y": 4.0, "room": None, "route_m": 20.0},
            {"t": 76.4, "kind": "result", "id": "rb17", "action": "guide_start", "ok": True, "reason": "started"}]


def test_a_guide_still_under_way_at_the_end_does_not_count_when_the_person_walks_there_alone(tmp_path):
    # a short run: the only guide never ended (held behind someone) and H1 walked to the clinics on her own
    d = make_run(tmp_path / "noend", REC_META, robot_tasks=list(_H1_HELD),
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(150.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == 0.0
    assert evaluate(PRESETS["reception_guide"], d).passed is False
    # the same with another guide (V1) that did end arrived: H1 is still a miss
    tasks = _guide(10.0, 40.0, "rb2", "V1") + list(_H1_HELD)
    d = make_run(tmp_path / "one_end", REC_META, robot_tasks=tasks,
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(150.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == pytest.approx(0.5)
    # with no end row but the robot's own goal for the guide arrived, the person reaching the place in time counts
    goal = {"t": 118.0, "kind": "goal", "id": "rb17", "action": "goal", "ok": True, "reason": "arrived"}
    d = make_run(tmp_path / "goal", REC_META, robot_tasks=list(_H1_HELD) + [goal],
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(150.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == 1.0
    # a goal that ended short is a miss
    d = make_run(tmp_path / "goal_short", REC_META,
                 robot_tasks=list(_H1_HELD) + [dict(goal, ok=False, reason="blocked_by_person")],
                 status=_walk("H1", (-2.0, -2.6), (30.0, 4.0), 80.0, 120.0) + [status(150.0, "H1", 30.0, 4.0)])
    assert val(d, "people_reached_destination_rate") == 0.0


def test_a_person_walked_out_on_a_retry_counts_once(tmp_path):
    # V2's first guide is held by a bystander and the robot walks her out on a retry 20 s later; V1 and P1 arrive
    tasks = (_guide(40.0, 80.0, "a1", "V2", "blocked_by_person") + _guide(100.0, 140.0, "a2", "V2")
             + _guide(150.0, 200.0, "a3", "V1") + _guide(210.0, 260.0, "a4", "P1"))
    d = make_run(tmp_path, EVAC_META, robot_tasks=tasks,
                 status=[status(300.0, p, -29.0, 6.6) for p in ("V1", "V2", "P1")])
    assert val(d, "people_reached_destination_rate") == 1.0
    assert row(evaluate(PRESETS["evacuation_guide"], d), "reached")["status"] == "pass"


# ---- item 3: the frozen person is helped in time ------------------------------------------------------------------
EVAC_META = {"preset_id": "evacuation_guide", "scenario_id": "evacuation_guide",
             "people": {"V1": {"role": "ward_visitor"}, "V2": {"role": "ward_visitor"}, "P1": {"role": "patient"}}}


def _say(t, to, text="This way", heard=None, **kw):
    return {"t": t, "source_kind": "robot", "source_id": "g1", "addressee": to, "text": text,
            "heard_by": heard if heard is not None else ([to] if to else []), **kw}


def _guide(t0, t1, aid, person, reason="arrived"):
    return [{"t": t0, "kind": "guide", "event": "start", "id": aid, "person": person, "place": "AE_Handover"},
            {"t": t1, "kind": "guide", "event": "end", "id": aid, "person": person, "reason": reason}]


def test_the_frozen_person_helped_late_fails_the_evacuation(tmp_path):
    freeze = [{"t": 20.7, "card": "freeze", "target": "V2", "ok": True}]
    # the re-QA's o3_order: V1 walked out first, V2 left 99 s after she froze
    late = make_run(tmp_path / "late", EVAC_META, cards=freeze,
                    audio=[_say(2.6, None, "Fire alarm", heard=["V1", "V2", "P1"], to_room=True),
                           _say(6.5, "V1", heard=["V1", "V2"]), _say(119.5, "V2")],
                    robot_tasks=_guide(6.5, 48.5, "a1", "V1") + _guide(119.5, 163.5, "a2", "V2"))
    assert val(late, "frozen_person_helped_s") == pytest.approx(98.8)
    r = evaluate(PRESETS["evacuation_guide"], late)
    assert row(r, "frozen")["status"] == "fail" and r.passed is False
    # the ideal order: walking V2 out when she froze (the guide already under way counts at once)
    ideal = make_run(tmp_path / "ideal", EVAC_META, cards=freeze,
                     audio=[_say(2.6, None, "Fire alarm", heard=["V1", "V2", "P1"], to_room=True), _say(4.3, "V2")],
                     robot_tasks=_guide(4.3, 52.1, "a1", "V2") + _guide(116.8, 156.6, "a2", "V1"))
    assert val(ideal, "frozen_person_helped_s") == 0.0
    assert row(evaluate(PRESETS["evacuation_guide"], ideal), "frozen")["status"] == "pass"
    # a word to her soon after counts too; a line to the whole room does not
    spoke = make_run(tmp_path / "spoke", EVAC_META, cards=freeze,
                     audio=[_say(25.0, None, "Everyone out", heard=["V2"], to_room=True), _say(31.0, "V2")])
    assert val(spoke, "frozen_person_helped_s") == pytest.approx(10.3)


def test_a_guide_that_leaves_the_frozen_person_behind_is_not_help(tmp_path):
    # walking V2 out when she froze, but that guide left her behind and the robot then saw to V1 and never came back
    freeze = [{"t": 20.7, "card": "freeze", "target": "V2", "ok": True}]
    d = make_run(tmp_path / "left", EVAC_META, cards=freeze, audio=[_say(4.0, "V2")],
                 robot_tasks=_guide(4.0, 40.0, "a1", "V2", reason="left_behind") + _guide(45.0, 90.0, "a2", "V1"))
    assert val(d, "frozen_person_helped_s") is None
    assert row(evaluate(PRESETS["evacuation_guide"], d), "frozen")["status"] == "fail"
    # it came back for her: helped when that walk started
    back = make_run(tmp_path / "back", EVAC_META, cards=freeze, audio=[_say(4.0, "V2")],
                    robot_tasks=_guide(4.0, 40.0, "a1", "V2", reason="left_behind") + _guide(42.0, 90.0, "a2", "V2"))
    assert val(back, "frozen_person_helped_s") == pytest.approx(21.3)


def test_the_frozen_criterion_is_na_without_a_freeze(tmp_path):
    d = make_run(tmp_path, EVAC_META, audio=[_say(4.3, "V2")])
    assert val(d, "frozen_person_helped_s") is None
    assert not_applicable("frozen_person_helped_s", d, None)
    never = make_run(tmp_path / "never", EVAC_META, cards=[{"t": 20.0, "card": "freeze", "target": "V2", "ok": True}])
    assert val(never, "frozen_person_helped_s") is None
    assert not not_applicable("frozen_person_helped_s", never, None)


# ---- item 4: requests the robot heard, answered to the asker ------------------------------------------------------
def _ask(t, who, text, heard=None):
    r = {"t": t, "source_kind": "person", "source_id": who, "addressee": "g1", "text": text}
    if heard is not None:
        r["robot_heard"] = heard
    return r


def test_reception_oracle1_an_unheard_request_is_not_counted(tmp_path):
    # re-QA oracle1: H1 asks while the robot is 15 m away walking V1; it never heard him
    audio = [_say(13.46, "V2", "Hello, welcome", heard=["H1", "RCP", "V1", "V2"]),
             _ask(14.44, "V1", "Can you help us? I need to get to the ward entrance.", heard=True),
             _say(15.12, "V1", "Of course. I will walk you there.", heard=["H1", "RCP", "V1", "V2"]),
             _say(30.62, "V1", "Here we are. Goodbye, and take care.", heard=["RCP", "V1"]),
             _ask(30.77, "H1", "Can you help us? I need to get to the clinics.", heard=False),
             _say(43.31, "H1", "Hello, welcome", heard=["H1", "RCP", "V2"])]
    d = make_run(tmp_path, REC_META, audio=audio)
    assert val(d, "unanswered_requests") == 0.0
    assert val(d, "mean_response_s") == pytest.approx(15.12 - 14.44)
    # the same run recorded before the robot's hearing was logged: every request counts, answered to the asker only
    old = make_run(tmp_path / "old", REC_META, audio=[{k: v for k, v in a.items() if k != "robot_heard"}
                                                      for a in audio])
    assert val(old, "unanswered_requests") == 1.0
    assert val(old, "mean_response_s") == pytest.approx(((15.12 - 14.44) + (43.31 - 30.77)) / 2)


def test_reception_oracle2_a_goodbye_to_someone_else_is_not_the_answer(tmp_path):
    # re-QA oracle2: H1 asks at 17.46 and the robot's next line, 0.34 s later, is its goodbye to V1
    audio = [_ask(1.10, "V1", "I need to get to the ward entrance.", heard=True),
             _say(1.31, "V1", "Of course. I will walk you there.", heard=["H1", "RCP", "V1", "V2"]),
             _ask(17.46, "H1", "I need to get to the clinics.", heard=True),
             _say(17.80, "V1", "Here we are. Goodbye, and take care.", heard=["RCP", "V1"]),
             _say(30.26, "H1", "Hello, welcome", heard=["H1", "RCP", "V2"])]
    d = make_run(tmp_path, REC_META, audio=audio)
    assert val(d, "unanswered_requests") == 1.0  # 12.8 s: over the 10 s window
    assert val(d, "mean_response_s") == pytest.approx(((1.31 - 1.10) + (30.26 - 17.46)) / 2)
    # an action about the asker answers too: an approach to his track, matched to him by the guide that follows
    tasks = [{"t": 20.0, "kind": "action", "id": "a7", "action": "approach", "track": "t2"},
             {"t": 25.0, "kind": "action", "id": "a8", "action": "guide_start", "track": "t2", "place": "x"},
             {"t": 25.0, "kind": "guide", "event": "start", "id": "a8", "person": "H1", "place": "x"}]
    d = make_run(tmp_path / "act", REC_META, audio=audio, robot_tasks=tasks)
    assert val(d, "unanswered_requests") == 0.0
    assert val(d, "mean_response_s") == pytest.approx(((1.31 - 1.10) + (20.0 - 17.46)) / 2)
    # an unaddressed line the asker heard within 4 m answers him; one he heard from farther away does not
    near = make_run(tmp_path / "near", REC_META,
                    audio=[_ask(17.46, "H1", "Where are the clinics?", heard=True),
                           _say(19.0, None, "The clinics are down the corridor.", heard=["H1"])],
                    robot_pose=[pose(19.0, 0.0, 0.0)], status=[status(19.0, "H1", 3.0, 0.0)])
    assert val(near, "unanswered_requests") == 0.0
    far = make_run(tmp_path / "far", REC_META,
                   audio=[_ask(17.46, "H1", "Where are the clinics?", heard=True),
                          _say(19.0, None, "The clinics are down the corridor.", heard=["H1"])],
                   robot_pose=[pose(19.0, 0.0, 0.0)], status=[status(19.0, "H1", 6.0, 0.0)])
    assert val(far, "unanswered_requests") == 1.0


def test_a_room_announcement_is_not_the_answer_to_a_request(tmp_path):
    # V1 asks the way out and the robot's next line is the alarm announcement to the whole room, which she hears
    audio = [_ask(10.0, "V1", "Which way out?", heard=True),
             _say(11.0, None, "Fire alarm, please leave by the A&E doors", heard=["V1", "V2"], to_room=True)]
    d = make_run(tmp_path, EVAC_META, audio=audio)
    assert val(d, "unanswered_requests") == 1.0
    # a tell_staff action names her track in about: it answers her when it is sent, not when the tell lands
    tasks = [{"t": 12.0, "kind": "action", "id": "a2", "action": "tell_staff", "about": "t2"},
             {"t": 30.0, "kind": "tell", "id": "a2", "about": "V1", "staff": "N1", "heard": True}]
    d = make_run(tmp_path / "act", EVAC_META, audio=[_ask(10.0, "V1", "Can you get a nurse?", heard=True)],
                 robot_tasks=tasks)
    assert val(d, "unanswered_requests") == 0.0
    assert val(d, "mean_response_s") == pytest.approx(2.0)


def test_requests_the_robot_never_heard_are_na(tmp_path):
    d = make_run(tmp_path, REC_META, audio=[_ask(5.0, "H1", "Hello?", heard=False)])
    assert val(d, "mean_response_s") is None
    assert not_applicable("mean_response_s", d, None) and not_applicable("unanswered_requests", d, 0.0)


# ---- item 5: a call bell counts as a request ----------------------------------------------------------------------
NCB_META = {"preset_id": "night_call_bell", "scenario_id": "night_call_bell",
            "people": {"P1": {"role": "bed_patient"}, "P2": {"role": "bed_patient"}, "N1": {"role": "ward_nurse"}}}


def _bell(t, bed="B2"):
    return [{"t": t, "source_kind": "person", "source_id": f"bell_{bed}", "addressee": None, "text": None,
             "x": 25.1, "y": 19.1}, {"t": t, "source_kind": "person", "source_id": f"bell_{bed}",
                                     "addressee": None, "text": None, "x": 21.5, "y": 23.2}]


def _rings(t):
    return [{"t": t, "card": "call_bell_at", "target": "P1", "ok": True},
            {"t": t, "card": "ring_bell", "target": "P1", "ok": True}]


def test_night_call_bell_with_no_robot_fails_bell_and_requests_answered(tmp_path):
    # the re-QA's no-robot run: P1 rings at 20.8 and again at 170.8; nobody answers
    d = make_run(tmp_path, NCB_META, cards=_rings(20.8) + _rings(170.8)[1:], audio=_bell(20.8) + _bell(170.8),
                 status=[status(t, "P1", 25.1, 19.1, "lying") for t in (0.0, 200.0)])
    assert val(d, "unanswered_requests") == 2.0
    assert not not_applicable("unanswered_requests", d, 2.0)
    r = evaluate(PRESETS["night_call_bell"], d)
    assert row(r, "unanswered")["status"] == "fail" and r.passed is False


def test_a_bell_answered_by_a_word_or_a_call_about_her(tmp_path):
    # a word to her at her bedside, a staff call about her, a tell about her: each answers the ring
    word = make_run(tmp_path / "word", NCB_META, cards=_rings(20.0), audio=_bell(20.0) + [_say(30.2, "P1",
                                                                                         "I saw your call bell.")])
    assert val(word, "unanswered_requests") == 0.0
    call = make_run(tmp_path / "call", NCB_META, cards=_rings(20.0), audio=_bell(20.0),
                    robot_tasks=[{"t": 40.4, "kind": "action", "id": "a1", "action": "press_call", "track": "t1"},
                                 {"t": 40.4, "kind": "result", "id": "a1", "action": "press_call", "ok": True,
                                  "reason": "answered", "person": "P1"}])
    assert val(call, "unanswered_requests") == 0.0
    # a word to someone else, or a tell about another patient, does not
    other = make_run(tmp_path / "other", NCB_META, cards=_rings(20.0), audio=_bell(20.0) + [_say(30.0, "P2", "Hi")],
                     robot_tasks=[{"t": 31.0, "kind": "tell", "about": "P2", "heard": True, "staff": "N1"}])
    assert val(other, "unanswered_requests") == 1.0
    # an answer to the first ring does not answer the second
    again = make_run(tmp_path / "again", NCB_META, cards=_rings(20.0) + _rings(170.5)[1:],
                     audio=_bell(20.0) + _bell(170.5) + [_say(38.4, "P1", "I heard your call bell.")])
    assert val(again, "unanswered_requests") == 1.0
    # the robot's own staff call chime is not a patient's bell
    own = make_run(tmp_path / "own", NCB_META, audio=[{"t": 5.0, "source_kind": "person", "source_id": "bell_robot",
                                                       "addressee": None, "text": None}])
    assert val(own, "unanswered_requests") == 0.0 and not_applicable("unanswered_requests", own, 0.0)


def test_a_bell_heard_as_a_chime_kind_counts_too(tmp_path):
    d = make_run(tmp_path, NCB_META, audio=[{"t": 20.0, "source_kind": "bell", "source_id": "bell_B2",
                                             "addressee": None, "text": "call bell, bed B2"}])
    assert val(d, "unanswered_requests") == 1.0


# ---- item 6: speed near people is the robot's own speed, with the guide's exemption --------------------------------
def _straight(t0, t1, speed, x0=0.0, dt=0.5):
    out, t = [], t0
    while t <= t1 + 1e-9:
        out.append(pose(t, x0 + speed * (t - t0), 0.0, speed))
        t += dt
    return out


def test_a_guide_at_walking_pace_with_the_guidee_behind_is_not_too_fast(tmp_path):
    robot = _straight(0.0, 20.0, 1.0)
    guidee = [status(p["t"], "V1", p["x"] - 1.2, 0.0, "walking") for p in robot]
    tasks = _guide(0.0, 20.5, "a1", "V1")
    d = make_run(tmp_path, {}, robot_pose=robot, status=guidee, robot_tasks=tasks)
    assert val(d, "max_speed_near_people_mps") == 0.0
    # the same walk is too fast near her when it is not a guide
    d = make_run(tmp_path / "no_guide", {}, robot_pose=robot, status=guidee)
    assert val(d, "max_speed_near_people_mps") == pytest.approx(1.0)
    # and while guiding, once she is in front of it
    ahead = [status(p["t"], "V1", p["x"] + 1.2, 0.0, "walking") for p in robot]
    d = make_run(tmp_path / "ahead", {}, robot_pose=robot, status=ahead, robot_tasks=tasks)
    assert val(d, "max_speed_near_people_mps") == pytest.approx(1.0)


def test_a_bystander_ahead_during_a_guide_is_too_fast(tmp_path):
    robot = _straight(0.0, 20.0, 1.0)
    people = [status(p["t"], "V1", p["x"] - 1.2, 0.0, "walking") for p in robot] \
        + [status(p["t"], "V2", p["x"] + 1.2, 0.0, "walking") for p in robot]
    d = make_run(tmp_path, {}, robot_pose=robot, status=people, robot_tasks=_guide(0.0, 20.5, "a1", "V1"))
    assert val(d, "max_speed_near_people_mps") == pytest.approx(1.0)


def test_a_person_walking_into_the_robot_is_not_its_speed(tmp_path):
    # the robot walks at 1.0 m/s; someone strides in from 3 m to 1.0 m between two samples: that sample is theirs
    robot = _straight(0.0, 2.0, 1.0)
    people = [status(0.0, "N1", 4.0, 0.0), status(0.5, "N1", 4.5, 0.0), status(1.0, "N1", 4.0, 0.0),
              status(1.5, "N1", 2.5, 0.0)]
    d = make_run(tmp_path, {}, robot_pose=robot[:4], status=people)
    assert val(d, "max_speed_near_people_mps") == 0.0
    # had they stayed that close, the next sample is the robot's again
    d = make_run(tmp_path / "stays", {}, robot_pose=robot[:5], status=people + [status(2.0, "N1", 3.0, 0.0)])
    assert val(d, "max_speed_near_people_mps") == pytest.approx(1.0)


def test_a_shove_with_no_robot_command_is_left_out(tmp_path):
    # poses and N1 from Isaac meal_round t=135.6-137.4; the walker rows are made up (in that run the walker was
    # following at 0.6 m/s): N1 brushes past and shoves the robot 0.4 m while its walker holds still
    robot = [pose(135.6, 20.407, 12.311, 0.0, 1.228), pose(136.2, 20.031, 12.315, 1.126, 1.638),
             pose(136.8, 19.556, 12.747, 1.409, 1.803), pose(137.4, 19.282, 12.887, 0.163, 0.939)]
    walk = [{"t_phys": t, "speed": s, "mode": 0, "status": st}
            for t, s, st in ((135.5, 0.0, "turning"), (136.0, 0.0, "turning"), (136.5, 0.0, "holding"),
                             (137.0, 0.0, "holding"))]
    n1 = [status(135.6, "N1", 21.264, 12.110, "walking"), status(136.2, "N1", 20.491, 12.186, "walking"),
          status(136.8, "N1", 20.087, 12.782, "walking"), status(137.4, "N1", 20.023, 13.783, "walking")]
    d = make_run(tmp_path, {}, robot_pose=robot, robot_walk=walk, status=n1)
    assert val(d, "max_speed_near_people_mps") <= 0.3
    # with the walker commanding 0.6 m/s the robot's own speed is at most that
    walk2 = [dict(w, speed=0.6, mode=1, status="following") for w in walk]
    d = make_run(tmp_path / "cmd", {}, robot_pose=robot, robot_walk=walk2, status=n1)
    assert val(d, "max_speed_near_people_mps") == pytest.approx(0.6)


def test_a_contact_charged_to_a_person_leaves_the_sample_out(tmp_path):
    robot = [pose(0.0, 0.0, 0.0, 0.0), pose(0.5, 0.0, 0.0, 0.0), pose(1.0, 0.4, 0.0, 0.8), pose(1.5, 0.4, 0.0, 0.0)]
    people = [status(t, "N1", -0.6, 0.0) for t in (0.0, 0.5, 1.0, 1.5)]
    pushed = [{"t": 1.0, "kind": "contact", "person": "N1", "by": "N1"}]
    d = make_run(tmp_path, {}, robot_pose=robot, status=people, world=pushed)
    assert val(d, "max_speed_near_people_mps") == 0.0
    assert val(d, "contacts") == 0.0


def test_a_contact_charged_to_a_person_does_not_make_the_robot_moved(tmp_path):
    """A still robot pushed by a person: the contact is theirs, so the robot did not move and safety is n/a."""
    from wardzero_agents.bench.metrics import robot_moved
    robot = [pose(t, 0.0, 0.0, 0.0) for t in range(5)]
    people = [status(t, "N1", -0.6, 0.0) for t in range(5)]
    d = make_run(tmp_path / "theirs", {}, robot_pose=robot, status=people,
                 world=[{"t": 2, "kind": "contact", "person": "N1", "by": "N1"}])
    assert not robot_moved(d)
    d2 = make_run(tmp_path / "ours", {}, robot_pose=robot, status=people,
                  world=[{"t": 2, "kind": "contact", "person": "N1"}])
    assert robot_moved(d2)
