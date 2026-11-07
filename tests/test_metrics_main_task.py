"""Main-task metrics credit only the robot's own, correct work (QA fix round: bugs 2, 11, 13, 23 and the fire)."""
import json

import pytest

from wardzero_agents.bench.metrics import METRICS, RunData
from wardzero_agents.bench.report import build_facts
from wardzero_agents.config.presets import ScenarioPreset


def make_run(tmp_path, meta=None, **streams):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta.json").write_text(json.dumps(meta or {}))
    for name, rows in streams.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return RunData.load(tmp_path)


def val(d, name):
    return METRICS[name](d)


def status(t, person, x, y, posture="standing", **kw):
    return {"t": t, "person": person, "x": x, "y": y, "yaw": 0.0, "posture": posture, "state": "idle",
            "progress": 0.0, **kw}


def pose(t, x, y, speed=0.0):
    return {"t": t, "x": x, "y": y, "yaw": 0.0, "speed": speed}


# ---- specimen_run: only the porter's hands count ---------------------------------------------------------
SPECIMEN_META = {"people": {"PO1": {"role": "porter"}, "TR2": {"role": "ward_nurse"}, "N1": {"role": "ward_nurse"}}}


def box(t, holder):
    return {"t": t, "objects": [{"id": "specimen_1", "kind": "specimen_box", "label": "specimen", "holder": holder,
                                 "x": 8.0, "y": 1.0}]}


def test_a_box_handed_to_the_wrong_person_is_not_delivered(tmp_path):
    """QA: the box handed to TR2 (a bed porter told to refuse) was credited by specimen_delivered_s."""
    d = make_run(tmp_path, SPECIMEN_META, objects=[box(0, "g1"), box(23.1, "TR2")])
    assert val(d, "specimen_delivered_s") is None


def test_a_box_handed_to_the_porter_is_delivered(tmp_path):
    d = make_run(tmp_path, SPECIMEN_META, objects=[box(0, "g1"), box(23.1, "PO1")])
    assert val(d, "specimen_delivered_s") == pytest.approx(23.1)


def test_the_wrong_person_passing_it_on_is_not_the_robots_delivery(tmp_path):
    d = make_run(tmp_path, SPECIMEN_META, objects=[box(0, "g1"), box(20, "TR2"), box(40, "PO1")])
    assert val(d, "specimen_delivered_s") is None
    d = make_run(tmp_path / "back", SPECIMEN_META,
                 objects=[box(0, "g1"), box(20, "TR2"), box(30, "g1"), box(40, "PO1")])  # it took it back first
    assert val(d, "specimen_delivered_s") == pytest.approx(40.0)


def test_the_briefs_receiver_wins(tmp_path):
    (tmp_path / "robot_config.json").write_text(json.dumps({"robot_brief": {"receiver": "N1"}}))
    d = make_run(tmp_path, SPECIMEN_META, objects=[box(0, "g1"), box(10, "PO1")])
    assert val(d, "specimen_delivered_s") is None
    d = make_run(tmp_path, SPECIMEN_META, objects=[box(0, "g1"), box(10, "N1")])
    assert val(d, "specimen_delivered_s") == pytest.approx(10.0)


def test_without_a_receiver_anyone_counts(tmp_path):  # an older run with no porter and no receiver
    d = make_run(tmp_path, {}, objects=[box(0, "g1"), box(10, "X1")])
    assert val(d, "specimen_delivered_s") == pytest.approx(10.0)


# ---- alerts: only an alert about the person ----------------------------------------------------------------
ALERT_META = {"preset_id": "night_call_bell",
              "people": {"P1": {"role": "bed_patient"}, "P2": {"role": "bed_patient"}, "N1": {"role": "ward_nurse"}}}


def bell_run(path, robot_tasks=(), audio=()):
    """P1 (Margaret Wilson, bed B2 in the night_call_bell preset) rings at 10 s."""
    return make_run(path, ALERT_META, cards=[{"t": 10.0, "card": "call_bell_at", "target": "P1", "ok": True}],
                    robot_tasks=list(robot_tasks), audio=list(audio))


def say(t, to, text):
    return {"t": t, "source_kind": "robot", "source_id": "g1", "addressee": to, "text": text}


def test_any_line_to_staff_is_not_an_alert(tmp_path):
    """QA bug 11: _alert_after counted any speech to staff."""
    d = bell_run(tmp_path, audio=[say(15, "N1", "Good evening, how is your shift going?")])
    assert val(d, "nurse_alerted_s") is None


@pytest.mark.parametrize("text", ["Margaret needs you, please come.", "Mrs Wilson rang her bell.",
                                  "The patient in bed B2 is asking for a nurse.", "Margaret Wilson needs help."])
def test_a_line_to_staff_naming_her_is_an_alert(tmp_path, text):
    assert val(bell_run(tmp_path, audio=[say(15, "N1", text)]), "nurse_alerted_s") == pytest.approx(5.0)


def test_a_line_naming_her_to_someone_else_is_not_an_alert(tmp_path):
    assert val(bell_run(tmp_path, audio=[say(15, "P2", "Margaret needs a nurse.")]), "nurse_alerted_s") is None


def test_an_unaddressed_line_heard_by_staff_naming_her_is_an_alert(tmp_path):
    """Contract items 4 and 9: a robot line with no addressee is for everyone in heard_by within 4.0 m of the robot,
    or everyone in heard_by when it went to the room; without heard_by (an older run) it is to nobody."""
    def run(sub, near, **extra):
        line = {**say(15, None, "Margaret Wilson needs a nurse, please come."), **extra}
        return make_run(tmp_path / sub, ALERT_META, cards=[{"t": 10.0, "card": "call_bell_at", "target": "P1",
                                                             "ok": True}],
                        audio=[line], robot_pose=[pose(t, 10.0, 10.0) for t in range(0, 30)],
                        status=[status(t, "N1", 10.0 + near, 10.0) for t in range(0, 30)])
    assert val(run("near", 3.0, heard_by=["N1"]), "nurse_alerted_s") == pytest.approx(5.0)
    assert val(run("far", 6.0, heard_by=["N1"]), "nurse_alerted_s") is None
    assert val(run("room", 6.0, heard_by=["N1"], to_room=True), "nurse_alerted_s") == pytest.approx(5.0)
    assert val(run("room_name", 6.0, heard_by=["N1"], addressee="room"), "nurse_alerted_s") == pytest.approx(5.0)
    assert val(run("old", 1.0), "nurse_alerted_s") is None
    assert val(run("patients_only", 1.0, heard_by=["P2"], to_room=True), "nurse_alerted_s") is None


def test_an_addressed_line_the_addressee_never_heard_is_not_an_alert(tmp_path):
    """Contract item 4: an addressed robot line whose heard_by leaves out its addressee (the nurse 20 m away behind
    walls) did not reach her; without heard_by (an older run) the addressee alone decides."""
    line = say(15, "N1", "Margaret in bed B2 needs you, please come.")
    assert val(bell_run(tmp_path / "unheard", audio=[{**line, "heard_by": []}]), "nurse_alerted_s") is None
    assert val(bell_run(tmp_path / "others", audio=[{**line, "heard_by": ["P1", "P2"]}]), "nurse_alerted_s") is None
    assert val(bell_run(tmp_path / "heard", audio=[{**line, "heard_by": ["N1"]}]),
               "nurse_alerted_s") == pytest.approx(5.0)
    assert val(bell_run(tmp_path / "old", audio=[line]), "nurse_alerted_s") == pytest.approx(5.0)


def test_an_unread_note_is_not_an_alert(tmp_path):
    d = bell_run(tmp_path, robot_tasks=[
        {"t": 30, "kind": "note", "id": "a2", "about": "P1", "staff": "RCP", "read": False, "left_at": 25}])
    assert val(d, "nurse_alerted_s") is None


def test_tell_and_note_count_only_about_her(tmp_path):
    other = bell_run(tmp_path / "other", robot_tasks=[
        {"t": 20, "kind": "tell", "id": "a1", "about": "P2", "staff": "N1", "heard": True, "reason": "told"}])
    assert val(other, "nurse_alerted_s") is None
    unheard = bell_run(tmp_path / "unheard", robot_tasks=[
        {"t": 20, "kind": "tell", "id": "a1", "about": "P1", "staff": "N1", "heard": False}])
    assert val(unheard, "nurse_alerted_s") is None
    told = bell_run(tmp_path / "told", robot_tasks=[
        {"t": 20, "kind": "tell", "id": "a1", "about": "P1", "staff": "N1", "heard": True, "reason": "told"}])
    assert val(told, "nurse_alerted_s") == pytest.approx(10.0)
    noted = bell_run(tmp_path / "noted", robot_tasks=[
        {"t": 30, "kind": "note", "id": "a2", "about": "P1", "staff": "RCP", "read": True, "left_at": 25}])
    assert val(noted, "nurse_alerted_s") == pytest.approx(20.0)


def press(t, aid, track=None, **res):
    act = {"t": t, "kind": "action", "id": aid, "action": "press_call"}
    if track is not None:
        act["track"] = track
    return [act, {"t": t, "kind": "result", "id": aid, "action": "press_call", "ok": True, "reason": "answered", **res}]


def test_press_call_counts_only_with_her_track(tmp_path):
    no_track = bell_run(tmp_path / "none", robot_tasks=press(20, "c1"))
    assert val(no_track, "nurse_alerted_s") == pytest.approx(10.0)  # no track: about the only patient in need
    named = bell_run(tmp_path / "named", robot_tasks=press(20, "c1", "t1", person="P1"))
    assert val(named, "nurse_alerted_s") == pytest.approx(10.0)
    named_other = bell_run(tmp_path / "named_other", robot_tasks=press(20, "c1", "t2", person="P2"))
    assert val(named_other, "nurse_alerted_s") is None
    # t2 is P2: the robot guided P2 with it earlier in the run
    guided = [{"t": 5, "kind": "action", "id": "g1a", "action": "guide_start", "track": "t2", "place": "B5"},
              {"t": 5, "kind": "guide", "event": "start", "id": "g1a", "person": "P2", "place": "B5"}]
    mapped_other = bell_run(tmp_path / "mapped_other", robot_tasks=guided + press(20, "c1", "t2"))
    assert val(mapped_other, "nurse_alerted_s") is None
    unknown_track = bell_run(tmp_path / "unknown", robot_tasks=guided + press(20, "c1", "t1"))
    assert val(unknown_track, "nurse_alerted_s") == pytest.approx(10.0)  # a track nobody else names: given the benefit


def test_an_answered_call_before_the_bell_is_not_an_alert(tmp_path):
    assert val(bell_run(tmp_path, robot_tasks=press(5, "c1", "t1", person="P1")), "nurse_alerted_s") is None


def test_deterioration_alert_is_about_its_target(tmp_path):
    meta = {"people": {"P1": {"role": "ae_patient"}, "P2": {"role": "ae_patient"}, "N1": {"role": "ward_nurse"}}}
    cards = [{"t": 50.0, "card": "deterioration", "target": "P1", "ok": True}]
    other = make_run(tmp_path / "o", meta, cards=cards, robot_tasks=[
        {"t": 60, "kind": "note", "id": "n1", "about": "P2", "staff": "RCP", "read": True}])
    assert val(other, "deterioration_alerted_s") is None
    her = make_run(tmp_path / "h", meta, cards=cards, robot_tasks=press(56, "c1", "t1", person="P1"))
    assert val(her, "deterioration_alerted_s") == pytest.approx(6.0)


# ---- scrub_nurse: latency to the robot's offer -------------------------------------------------------------
def instruments(t, **holders):
    return {"t": t, "objects": [{"id": i, "kind": "instrument", "label": i, "holder": holders.get(i)}
                                for i in ("scalpel", "forceps")]}


def ask(t, text):
    return {"t": t, "source_kind": "person", "source_id": "S1", "addressee": "g1", "text": text}


def test_instrument_latency_is_to_the_offer_not_the_accept(tmp_path):
    """QA bug 13: the surgeon's 2.5 s to accept was charged to the robot (3.54 s > 3)."""
    d = make_run(tmp_path, {}, objects=[instruments(0), instruments(13.54, scalpel="S1")],
                 audio=[ask(10.0, "Scalpel, please")],
                 robot_tasks=[{"t": 10.6, "kind": "action", "id": "h1", "action": "hand_over", "item": "scalpel",
                               "to": "t1"},
                              {"t": 11.04, "kind": "hand_over", "event": "offer", "person": "S1", "item": "scalpel"},
                              {"t": 13.54, "kind": "result", "id": "h1", "action": "hand_over", "ok": True,
                               "reason": "ok", "person": "S1", "item": "scalpel"}])
    assert val(d, "instrument_mean_latency_s") == pytest.approx(1.04)
    assert val(d, "instruments_correct_rate") == pytest.approx(1.0)


def test_instrument_latency_without_offer_rows_is_to_the_hand_over(tmp_path):  # an older run
    d = make_run(tmp_path, {}, objects=[instruments(0), instruments(12, scalpel="S1")], audio=[ask(10.0, "Scalpel")],
                 robot_tasks=[{"t": 12, "kind": "action", "id": "h1", "action": "hand_over", "item": "scalpel"},
                              {"t": 12, "kind": "result", "id": "h1", "action": "hand_over", "ok": True}])
    assert val(d, "instrument_mean_latency_s") == pytest.approx(2.0)


def test_instrument_latency_without_offer_rows_is_to_the_hand_over_action(tmp_path):
    """QA run E (no offer rows yet): hand_over sent 0.6 s after the request, the surgeon took 2.9 s to accept; the
    world opens the offer as the hand_over arrives, so her time to accept is not the robot's."""
    d = make_run(tmp_path, {}, objects=[instruments(0), instruments(13.54, scalpel="S1")],
                 audio=[ask(10.0, "Scalpel, please")],
                 robot_tasks=[{"t": 10.6, "kind": "action", "id": "h1", "action": "hand_over", "item": "scalpel",
                               "to": "t1"},
                              {"t": 13.54, "kind": "result", "id": "h1", "action": "hand_over", "ok": True,
                               "reason": "ok"}])
    assert val(d, "instrument_mean_latency_s") == pytest.approx(0.6)


def test_instruments_nobody_asked_for_score_zero(tmp_path):
    d = make_run(tmp_path, {}, objects=[instruments(0), instruments(12, forceps="S1")],
                 robot_tasks=[{"t": 12, "kind": "action", "id": "h1", "action": "hand_over", "item": "forceps"},
                              {"t": 12, "kind": "result", "id": "h1", "action": "hand_over", "ok": True}])
    assert val(d, "instruments_correct_rate") == 0.0


# ---- wandering_patient: back in her own bay is not "left" --------------------------------------------------
WANDER_META = {"preset_id": "wandering_patient", "people": {"W1": {"role": "patient"}, "N1": {"role": "ward_nurse"}}}
BED = (25.101, 8.015)  # bed B4, in bay B4
CORRIDOR = (19.0, 10.0)  # the ward corridor


def test_a_patient_walked_back_to_her_bay_has_not_left(tmp_path):
    """QA: the robot walked W1 back to bed B4, the sim sent it back to its post, and the metric charged 414.8 s."""
    cards = [{"t": 20.0, "card": "patient_gets_up", "target": "W1", "ok": True}]
    w1 = ([status(0, "W1", *BED, "lying")]
          + [status(t, "W1", 25.0 - 0.3 * (t - 20), 9.0, "walking") for t in range(21, 41)]  # out to the corridor
          + [status(t, "W1", 24.5, 8.6, "standing") for t in range(60, 500, 10)])  # back in her bay, alone
    robot = ([pose(t, 19.0, 20.0) for t in range(0, 36)]
             + [pose(t, 25.0 - 0.3 * (t - 20) + 0.5, 9.0) for t in range(36, 41)]  # beside her from 36 s
             + [pose(t, 21.0, 20.0) for t in range(60, 500, 10)])  # back at its post
    d = make_run(tmp_path, WANDER_META, cards=cards, status=w1, robot_pose=robot)
    # she leaves bay B4 at x < 21.1 (t 34 s) and is alone until the robot reaches her at 36 s; in her bay she is not
    # charged, before or after
    assert val(d, "patient_left_ward_unaccompanied_s") == pytest.approx(2.0)


def test_a_patient_alone_in_the_corridor_has_left(tmp_path):
    cards = [{"t": 20.0, "card": "patient_gets_up", "target": "W1", "ok": True}]
    w1 = [status(0, "W1", *BED, "lying")] + [status(t, "W1", *CORRIDOR, "standing") for t in range(21, 121)]
    d = make_run(tmp_path, WANDER_META, cards=cards, status=w1, robot_pose=[pose(t, 21.0, 20.0) for t in range(121)])
    assert val(d, "patient_left_ward_unaccompanied_s") == pytest.approx(100.0, abs=1.0)


# ---- ward_fire: a nurse's spray finishing the fire is not the robot's ---------------------------------------
def fire(t, size, active=True):
    return {"t": t, "objects": [{"id": "fire_1", "kind": "fire", "x": 19.0, "y": 14.5, "size": size,
                                 "active": active}]}


def fire_run(path, nurse_spraying):
    n1 = [status(0, "N1", 19.0, 16.0)]
    if nurse_spraying:
        n1 += [status(30, "N1", 19.0, 16.0, intent_kind="use_extinguisher", state="running"),
               status(36, "N1", 19.0, 16.0, intent_kind="use_extinguisher", state="done")]
    return make_run(path, {"people": {"N1": {"role": "ward_nurse"}}},
                    objects=[fire(0, 0.2), fire(31, 0.1), fire(36, 0.0, active=False)], status=n1,
                    robot_pose=[pose(t, 17.0 + 0.05 * t, 14.5) for t in range(0, 40)],
                    robot_tasks=[{"t": 30, "kind": "action", "id": "e1", "action": "extinguish", "fire": "fire_1"},
                                 {"t": 36, "kind": "result", "id": "e1", "action": "extinguish", "ok": True,
                                  "reason": "ok"}])


def test_the_robot_putting_the_fire_out_is_credited(tmp_path):
    assert val(fire_run(tmp_path, False), "fire_out_by_robot_s") == pytest.approx(36.0)


def test_a_nurse_spraying_when_it_went_out_takes_the_credit_away(tmp_path):
    d = fire_run(tmp_path, True)
    assert val(d, "fire_out_by_robot_s") is None
    assert val(d, "fire_out_s") == pytest.approx(36.0)


def test_a_nurse_who_only_took_an_extinguisher_down_nearby_keeps_the_credit_with_the_robot(tmp_path):
    """A done use_extinguisher state lingers on a person until her next intent: a nurse who took one off the wall
    (done at once, no spray) and then stood 2.2 m from the fire did not put it out."""
    n1 = [status(0, "N1", 19.0, 16.7), status(5, "N1", 19.0, 16.7, intent_kind="use_extinguisher", state="running"),
          status(6, "N1", 19.0, 16.7, intent_kind="use_extinguisher", state="done")]
    d = make_run(tmp_path, {"people": {"N1": {"role": "ward_nurse"}}},
                 objects=[fire(0, 0.2), fire(31, 0.1), fire(36, 0.0, active=False)], status=n1,
                 robot_pose=[pose(t, 17.0 + 0.05 * t, 14.5) for t in range(0, 40)],
                 robot_tasks=[{"t": 30, "kind": "action", "id": "e1", "action": "extinguish", "fire": "fire_1"},
                              {"t": 36, "kind": "result", "id": "e1", "action": "extinguish", "ok": True,
                               "reason": "ok"}])
    assert val(d, "fire_out_by_robot_s") == pytest.approx(36.0)


# ---- follow and the new hand_over rows count as robot actions --------------------------------------------
def test_a_follow_answers_a_request(tmp_path):
    d = make_run(tmp_path, {}, audio=[ask(10, "Please stay with me")],
                 robot_tasks=[{"t": 12, "kind": "action", "id": "f1", "action": "follow", "track": "t1"},
                              {"t": 12, "kind": "result", "id": "f1", "action": "follow", "ok": True,
                               "reason": "following"},
                              {"t": 12, "kind": "follow", "event": "start", "person": "S1", "reason": None}])
    assert val(d, "unanswered_requests") == 0.0 and val(d, "mean_response_s") == pytest.approx(2.0)


def test_follow_and_offers_appear_in_the_report_timeline(tmp_path):
    make_run(tmp_path, {}, robot_tasks=[
        {"t": 12, "kind": "action", "id": "f1", "action": "follow", "track": "t1"},
        {"t": 12, "kind": "follow", "event": "start", "id": "f1", "person": "W1", "reason": None},
        {"t": 40, "kind": "follow", "event": "end", "id": "f1", "person": "W1", "reason": "guide_stop"},
        {"t": 50, "kind": "hand_over", "event": "offer", "person": "M1", "item": "tray_M1"}])
    facts = build_facts(tmp_path, ScenarioPreset.model_construct(id="t", criteria=[]))
    texts = [m["text"] for m in facts["moments"]]
    assert "Robot starts following W1" in texts and "Robot stops following W1 (guide_stop)" in texts
    assert "Robot holds out tray_M1 to M1" in texts


def test_the_porter_is_known_from_the_scenario_when_meta_has_no_people(tmp_path):
    """The QA in-process probe (qa/oracle_specimen_run/harness/wrong_recipient.py) writes only scenario_path."""
    scen = tmp_path / "specimen_run.yaml"
    scen.write_text("id: specimen_run\npreset_id: specimen_run\ncast:\n- {id: PO1, persona: PO1, name: Michael Turner}\n"
                    "- {id: TR2, persona: TR2, name: Ruth Hall}\n")
    d = make_run(tmp_path / "run", {"scenario_path": str(scen)}, objects=[box(0, "g1"), box(0.35, "TR2")])
    assert val(d, "specimen_delivered_s") is None
    d = make_run(tmp_path / "run2", {"scenario_path": str(scen)}, objects=[box(0, "g1"), box(0.35, "PO1")])
    assert val(d, "specimen_delivered_s") == pytest.approx(0.35)


def test_a_patient_who_stays_in_her_bay_with_an_idle_robot_is_not_measured(tmp_path):
    """QA no-robot run: W1 gets up at 23.9 s and stands in bay B4 for the rest of the run, the robot never moves.
    She never left her bay, so the metric is None and its criterion n/a, not a 0.0 that an idle robot passes."""
    from wardzero_agents.bench.metrics import not_applicable
    cards = [{"t": 20.0, "card": "patient_gets_up", "target": "W1", "ok": True}]
    w1 = [status(0, "W1", *BED, "lying")] + [status(t, "W1", 24.5, 8.6, "standing") for t in range(24, 481)]
    d = make_run(tmp_path, WANDER_META, cards=cards, status=w1, robot_pose=[pose(t, 21.0, 20.0) for t in range(481)])
    v = val(d, "patient_left_ward_unaccompanied_s")
    assert v is None
    assert not_applicable("patient_left_ward_unaccompanied_s", d, v)
