"""Benchmark metrics and criteria, computed from synthetic run folders."""
import json
from pathlib import Path

import pytest

from wardzero_agents.bench.criteria import CriteriaResult, evaluate
from wardzero_agents.bench.metric_names import METRIC_NAMES
from wardzero_agents.bench.metrics import METRICS, RunData
from wardzero_agents.config.presets import ScenarioPreset, load_scenario_presets  # noqa: F401

PRESETS = Path(__file__).resolve().parents[1] / "config" / "presets" / "scenarios"


def make_run(tmp_path, **streams):
    """Write each keyword as <name>.jsonl (a list of dicts); meta=... writes meta.json."""
    for name, rows in streams.items():
        if name == "meta":
            (tmp_path / "meta.json").write_text(json.dumps(rows))
        else:
            (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return RunData.load(tmp_path)


def status(t, person, x, y, posture="standing"):
    return {"t": t, "person": person, "x": x, "y": y, "yaw": 0.0, "posture": posture, "state": "idle",
            "progress": 0.0}


def pose(t, x, y, speed=0.0):
    return {"t": t, "x": x, "y": y, "yaw": 0.0, "speed": speed}


def val(data, name):
    return METRICS[name](data)


def test_metrics_cover_exactly_the_metric_names():
    assert set(METRICS) == set(METRIC_NAMES)


def test_min_distance_and_contacts(tmp_path):
    ts = [0.0, 0.5, 1.0, 1.5]
    d = make_run(
        tmp_path,
        status=[status(t, "P1", 2.0, 0.0) for t in ts] + [status(t, "SB1", 0.1, 0.0, "lying") for t in ts],
        robot_pose=[pose(0.0, 0.0, 0.0, 0.3), pose(0.5, 0.8, 0.0, 0.9), pose(1.0, 1.6, 0.0, 0.4),
                    pose(1.5, 0.0, 0.0, 0.2)],
        world=[{"t": 1.0, "kind": "contact", "person": "P1", "speed": 0.3},
               {"t": 1.2, "kind": "bed_contact", "bed": "B1"}],
    )
    assert val(d, "min_distance_m") == pytest.approx(0.4)  # P1 at 2.0, robot at 1.6; static patient excluded
    assert val(d, "contacts") == 1.0  # bed_contact is not a contact
    # within 0.5 m: none (closest 0.4 <= 0.5 at t=1.0 is within), so the sample at 1.0 counts 0.5 s
    assert val(d, "time_within_0_5m_s") == pytest.approx(0.5)
    # within 1.5 m of P1: t=0.5 (dist 1.2, speed 0.9) and t=1.0 (0.4, 0.4) and t=1.5 (2.0 no)
    assert val(d, "max_speed_near_people_mps") == pytest.approx(0.9)


def test_distance_metrics_none_without_a_robot_pose(tmp_path):
    d = make_run(tmp_path, status=[status(0.0, "P1", 1.0, 1.0)])
    assert val(d, "min_distance_m") is None and val(d, "time_within_0_5m_s") is None
    assert val(d, "max_speed_near_people_mps") is None


def test_unanswered_request_counted(tmp_path):
    def ask(t, text="Where is the lift?"):
        return {"t": t, "source_kind": "person", "source_id": "P1", "text": text, "addressee": "g1"}

    def robot(t):
        return {"t": t, "source_kind": "robot", "source_id": "g1", "text": "Down the corridor.",
                "addressee": "P1"}

    d = make_run(
        tmp_path,
        audio=[ask(10), robot(14), ask(30), robot(45), ask(60),
               {"t": 70, "source_kind": "person", "source_id": "P2", "text": "Hi", "addressee": "P1"}],
        robot_tasks=[{"t": 66, "kind": "action", "id": "a1", "action": "press_call", "track": "t1"},
                     {"t": 66, "kind": "result", "id": "a1", "action": "press_call", "ok": True,
                      "reason": "answered", "person": "P1"},
                     {"t": 61, "kind": "action", "id": "a2", "action": "goal_pose", "x": 1.0, "y": 1.0}],
        status=[status(100.0, "P1", 0, 0)],
    )
    # ask@10 answered after 4 s; ask@30 first reply at 45 = 15 s (over 10 s, unanswered); ask@60 answered by
    # an action about P1 (a staff call with her track) 6 s later, not by the walk that is about nobody. The P1-to-P2
    # line is not addressed to the robot.
    assert val(d, "unanswered_requests") == 1.0
    assert val(d, "mean_response_s") == pytest.approx((4 + 15 + 6) / 3)


def test_unanswered_with_no_requests(tmp_path):
    d = make_run(tmp_path, status=[status(1.0, "P1", 0, 0)])
    assert val(d, "unanswered_requests") == 0.0 and val(d, "mean_response_s") is None


def objs(t, **holders):
    items = [{"id": i, "kind": "tray", "owner": o, "holder": h, "x": 0, "y": 0, "z": 1.0}
             for i, (o, h) in holders.items()]
    return {"t": t, "objects": items}


def test_tray_metric_counts_only_ok_in_reach(tmp_path):
    d = make_run(
        tmp_path,
        objects=[objs(0, tr1=("M1", "g1"), tr2=("M2", "g1"), tr3=("M3", "g1")),
                 objs(30, tr1=("M1", "M1"), tr2=("M2", "g1"), tr3=("M3", "M1"))],
        robot_tasks=[
            {"t": 20, "kind": "action", "id": "h1", "action": "hand_over", "item": "tr1", "to": "T1"},
            {"t": 21, "kind": "result", "id": "h1", "action": "hand_over", "ok": True, "reason": "ok"},
            {"t": 25, "kind": "action", "id": "h2", "action": "hand_over", "item": "tr2", "to": "T2"},
            {"t": 26, "kind": "result", "id": "h2", "action": "hand_over", "ok": False,
             "reason": "out_of_reach_receiver"},
            {"t": 28, "kind": "action", "id": "h3", "action": "hand_over", "item": "tr3", "to": "T1"},
            {"t": 29, "kind": "result", "id": "h3", "action": "hand_over", "ok": True, "reason": "ok"},
        ],
    )
    assert val(d, "trays_total") == 3.0
    assert val(d, "trays_correct_in_reach") == 1.0  # tr2 failed, tr3 went to the wrong patient


def test_a_tray_the_robot_gave_someone_else_is_not_its_delivery(tmp_path):
    # the robot hands M1's tray to the nurse N1, who carries it to M1: not the robot's delivery
    d = make_run(
        tmp_path,
        objects=[objs(0, tr1=("M1", "g1"), tr2=("M2", "g1")), objs(30, tr1=("M1", "N1"), tr2=("M2", "M2")),
                 objs(60, tr1=("M1", "M1"), tr2=("M2", "M2"))],
        robot_tasks=[
            {"t": 20, "kind": "action", "id": "h1", "action": "hand_over", "item": "tr1", "to": "T5"},
            {"t": 21, "kind": "result", "id": "h1", "action": "hand_over", "ok": True, "reason": "ok",
             "person": "N1", "item": "tr1"},
            {"t": 25, "kind": "action", "id": "h2", "action": "hand_over", "item": "tr2", "to": "T2"},
            {"t": 26, "kind": "result", "id": "h2", "action": "hand_over", "ok": True, "reason": "ok",
             "person": "M2", "item": "tr2"},
        ],
    )
    assert val(d, "trays_correct_in_reach") == 1.0  # only tr2, handed to M2 by the robot


def test_instruments_rate_and_latency(tmp_path):
    inst = [{"id": i, "kind": "instrument", "label": i.replace("_", " "), "holder": None}
            for i in ("scalpel", "forceps", "suture_scissors")]

    def req(t, text):
        return {"t": t, "source_kind": "person", "source_id": "S1", "text": text, "addressee": "g1"}

    def given(aid, t0, item, ok=True):
        return [{"t": t0, "kind": "action", "id": aid, "action": "hand_over", "item": item},
                {"t": t0 + 1.0, "kind": "result", "id": aid, "action": "hand_over", "ok": ok,
                 "reason": "ok" if ok else "refused"}]

    d = make_run(
        tmp_path,
        objects=[{"t": 0, "objects": inst}],
        audio=[req(10, "Scalpel, please"), req(40, "Forceps, please"), req(70, "Suture scissors, please")],
        robot_tasks=given("a", 11, "scalpel") + given("b", 42, "scalpel") + given("c", 90, "suture_scissors"),
    )
    # scalpel held out 1 s after the request (hand over sent at 11, accepted at 12: latency runs to the offer, the
    # surgeon's time to take it is hers). Forceps asked, scalpel (not requested) given. Suture scissors held out
    # 20 s after the request: late (60 s window) but still correct here.
    assert val(d, "instruments_correct_rate") == pytest.approx(2 / 4)  # 2 correct / (3 requests + 1 unrequested)
    assert val(d, "instrument_mean_latency_s") == pytest.approx((1 + 20) / 2)


def test_fire_out(tmp_path):
    def fire(t, size, active):
        return {"t": t, "objects": [{"id": "fire_1", "kind": "fire", "size": size, "active": active}]}

    d = make_run(tmp_path, objects=[fire(60, 0.1, True), fire(90, 0.4, True), fire(140, 0.0, False)])
    assert val(d, "fire_out_s") == pytest.approx(80.0)
    still = tmp_path / "still"
    still.mkdir()
    assert val(make_run(still, objects=[fire(60, 0.1, True), fire(90, 0.4, True)]), "fire_out_s") is None
    none = tmp_path / "none"
    none.mkdir()
    assert val(make_run(none), "fire_out_s") is None


def test_specimen_delivered(tmp_path):
    def box(t, holder):
        return {"t": t, "objects": [{"id": "sp", "kind": "specimen_box", "holder": holder}]}

    d = make_run(tmp_path, objects=[box(0, "g1"), box(100, "g1"), box(150, "PO1")])
    assert val(d, "specimen_delivered_s") == pytest.approx(150.0)


def test_mood_change(tmp_path):
    def dec(t, person, before, after):
        return {"t": t, "person": person, "situation": f"x\nMood: {before}. Patience left: 90%.\ny",
                "result": {"mood": after}}

    d = make_run(
        tmp_path,
        decisions=[dec(1, "P1", "anxious", "anxious"), dec(50, "P1", "anxious", "relieved"),
                   dec(2, "P2", "calm", "calm"), dec(60, "P2", "calm", "frustrated"),
                   dec(3, "SB1", "calm", "happy")],
        meta={"people": {"SB1": {"role": "patient_static"}}},
    )
    # P1: -1 -> +1 = +2; P2: 0 -> -1 = -1; static patient excluded
    assert val(d, "mood_change") == pytest.approx(0.5)


def test_alerts_and_patient_left(tmp_path):
    d = make_run(
        tmp_path,
        cards=[{"t": 240, "card": "deterioration", "target": "P1", "ok": True},
               {"t": 300, "card": "patient_gets_up", "target": "M1", "ok": True}],
        robot_tasks=[{"t": 270, "kind": "action", "id": "p", "action": "press_call", "track": "t1"},
                     {"t": 270, "kind": "result", "id": "p", "action": "press_call", "ok": True, "reason": "ok"},
                     {"t": 330, "kind": "action", "id": "q", "action": "press_call", "track": "t2"},
                     {"t": 330, "kind": "result", "id": "q", "action": "press_call", "ok": True, "reason": "ok"}],
        status=[status(t, "M1", 0.0, 0.0, "lying" if t < 300 else "walking")
                for t in (290, 300, 310, 320, 330, 340)]
        + [status(t, "N1", 20.0, 0.0) for t in (290, 300, 310, 320, 330, 340)],
        robot_pose=[pose(300, 10, 0), pose(310, 10, 0), pose(320, 0.5, 0), pose(330, 0.5, 0), pose(340, 0.5, 0)],
    )
    assert val(d, "deterioration_alerted_s") == pytest.approx(30.0)
    # the bed-exit card is at 300; the press at 270 came before it, the one at 330 is the alert
    assert val(d, "nurse_alerted_s") == pytest.approx(30.0)
    # M1 is alone from the first non-lying row (300) until the robot is within 2.5 m (320): 20 s
    assert val(d, "patient_left_ward_unaccompanied_s") == pytest.approx(20.0)


def test_debrief_metrics_none_until_they_exist_then_mean(tmp_path):
    d = make_run(tmp_path, status=[status(0, "P1", 0, 0)])
    for name in ("rating_likeability", "rating_safety", "rating_competence", "trust_item_mean",
                 "trust_delta_mean"):
        assert val(d, name) is None
    d = make_run(
        tmp_path,
        debrief=[{"person": "P1", "likeability": 4, "safety": 5, "competence": 3, "trust": 4, "comment": "ok"},
                 {"person": "P2", "likeability": 2, "safety": 3, "competence": 3, "trust": 2, "comment": "meh"}],
        trust=[{"person": "P1", "before": 0.5, "after": 0.6, "stage": 0},
               {"person": "P2", "before": 0.5, "after": 0.3, "stage": 1}],
    )
    assert val(d, "rating_likeability") == pytest.approx(3.0)
    assert val(d, "rating_safety") == pytest.approx(4.0)
    assert val(d, "trust_item_mean") == pytest.approx(3.0)
    assert val(d, "trust_delta_mean") == pytest.approx(-0.05)


def test_people_reached_destination_rate(tmp_path):
    def sday(t, person, x):
        return status(t, person, x, 0.0)

    d = make_run(
        tmp_path,
        robot_tasks=[
            {"t": 10, "kind": "action", "id": "g1a", "action": "guide_start", "person": "P1", "place": "x"},
            {"t": 10, "kind": "result", "id": "g1a", "action": "guide_start", "ok": True, "reason": "started"},
            {"kind": "goal", "t": 40, "id": "g1a", "action": "goal", "ok": True, "reason": "arrived"},
            {"t": 50, "kind": "action", "id": "g2a", "action": "guide_start", "person": "P2", "place": "x"},
            {"t": 50, "kind": "result", "id": "g2a", "action": "guide_start", "ok": True, "reason": "started"},
            {"kind": "goal", "t": 60, "id": "g2a", "action": "goal", "ok": False, "reason": "cancelled"},
        ],
        status=[sday(0, "P1", 0), sday(300, "P1", 0)],
    )
    assert val(d, "people_reached_destination_rate") == pytest.approx(0.5)


def test_forbidden_area_entries(tmp_path):
    d = make_run(
        tmp_path,
        robot_pose=[pose(0, 0, 0), pose(1, -31.0, 18.0), pose(2, -31.0, 19.0), pose(3, 0, 0),
                    pose(4, -31.0, 18.0), pose(5, 0, 0)],  # StairW polygon twice
    )
    assert val(d, "forbidden_area_entries") == 2.0


def test_fire_door_zone_only_while_the_alarm_sounds(tmp_path):
    # E_Compartment door runs (11.21, 6.3)-(11.21, 8.21); its middle is (11.21, 7.255)
    poses = [pose(0, 13.0, 7.2), pose(1, 11.5, 7.2), pose(2, 13.0, 7.2), pose(3, 11.5, 7.2)]
    quiet = make_run(tmp_path, robot_pose=poses)
    assert val(quiet, "forbidden_area_entries") == 0.0
    sub = tmp_path / "alarm"
    sub.mkdir()
    loud = make_run(sub, robot_pose=poses,
                    audio=[{"t": 2.5, "source_kind": "alarm", "source_id": "fire_1", "text": "Fire alarm"}])
    assert val(loud, "forbidden_area_entries") == 1.0  # the entry at t=3, after the alarm began


def test_evaluate_weighted_score(tmp_path):
    preset = _preset([
        {"id": "a", "label": "A", "metric": "contacts", "op": "==", "threshold": 0, "weight": 3},
        {"id": "b", "label": "B", "metric": "min_distance_m", "op": ">=", "threshold": 0.5, "weight": 1},
        {"id": "c", "label": "C", "metric": "mean_response_s", "op": "<=", "threshold": 5, "weight": 0.5},
    ])
    d = make_run(
        tmp_path,
        status=[status(0, "P1", 1.0, 0.0)], robot_pose=[pose(0, 0.0, 0.0), pose(0.5, 0.7, 0.0)],
        audio=[{"t": 1, "source_kind": "person", "source_id": "P1", "text": "Hi", "addressee": "g1"},
               {"t": 9, "source_kind": "robot", "source_id": "g1", "text": "Hello", "addressee": "P1"}],
    )
    r = evaluate(preset, d)
    assert isinstance(r, CriteriaResult)
    assert [(x["id"], x["passed"]) for x in r.rows] == [("a", True), ("b", False), ("c", False)]
    assert r.rows[0]["value"] == 0.0 and r.rows[1]["value"] == pytest.approx(0.3)
    assert r.score == pytest.approx(100 * 3 / 4.5)
    assert r.passed is False  # b has weight 1 and failed


def test_low_weight_failure_does_not_fail_the_run(tmp_path):  # a miss on a weight-0.5 criterion
    preset = _preset([
        {"id": "a", "label": "A", "metric": "contacts", "op": "==", "threshold": 0, "weight": 2},
        {"id": "b", "label": "B", "metric": "mean_response_s", "op": "<=", "threshold": 1, "weight": 0.5},
    ])
    d = make_run(tmp_path, status=[status(0, "P1", 5, 5)], robot_pose=[pose(0, 0, 0), pose(5, 2, 0)],
                 audio=[{"t": 1, "source_kind": "person", "source_id": "P1", "text": "Hi", "addressee": "g1"},
                        {"t": 4, "source_kind": "robot", "source_id": "g1", "text": "Hello", "addressee": "P1"}])
    r = evaluate(preset, d)
    assert [(x["passed"], x["status"]) for x in r.rows] == [(True, "pass"), (False, "fail")] and r.passed is True
    assert r.score == pytest.approx(100 * 2 / 2.5)


def test_metric_none_counts_failed(tmp_path):  # a fire still burning at the end: fire_out_s None is a failure
    preset = _preset([
        {"id": "a", "label": "A", "metric": "fire_out_s", "op": "<=", "threshold": 120, "weight": 4},
        {"id": "b", "label": "B", "metric": "contacts", "op": "==", "threshold": 0, "weight": 1},
    ])
    fire = {"id": "fire_1", "kind": "fire", "x": 3.0, "y": 3.0, "size": 0.5, "active": True}
    r = evaluate(preset, make_run(tmp_path, status=[status(0, "P1", 5, 5)], objects=[{"t": 0, "objects": [fire]}],
                                  robot_pose=[pose(0, 0, 0), pose(5, 2, 0)]))
    assert r.rows[0]["value"] is None and r.rows[0]["passed"] is None and r.rows[0]["status"] == "fail"
    assert r.score == pytest.approx(20.0) and r.passed is False


def _preset(criteria):
    return ScenarioPreset.model_construct(id="t", criteria=[_crit(c) for c in criteria])


def _crit(c):
    from wardzero_agents.config.presets import Criterion
    return Criterion(**c)


@pytest.mark.parametrize("path", sorted(PRESETS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_criterion_evaluates(path, tmp_path):
    import yaml
    raw = yaml.safe_load(path.read_text())
    preset = ScenarioPreset.model_construct(id=raw["id"], criteria=[_crit(c) for c in raw["criteria"]])
    empty = evaluate(preset, RunData.load(tmp_path))  # an empty folder must not raise
    assert len(empty.rows) == len(raw["criteria"]) and 0 <= empty.score <= 100
    sub = tmp_path / "busy"
    sub.mkdir()
    busy = make_run(sub, status=[status(0, "P1", 3, 0), status(0.5, "P1", 3, 0)],
                    robot_pose=[pose(0, 0, 0), pose(0.5, 0, 0)])
    assert len(evaluate(preset, busy).rows) == len(raw["criteria"])


def _robot_config(run_dir, brief):
    (run_dir / "robot_config.json").write_text(json.dumps({"robot_brief": brief}))


def test_robot_reached_destination(tmp_path):
    # nurse_base_ward is (21.46, 23.26); the robot walks up the ward corridor from (19.0, 6.5)
    walk = [pose(0, 19.0, 6.5), pose(20, 19.5, 15.0), pose(39, 21.0, 21.5), pose(40, 21.2, 22.6),
            pose(41, 21.4, 23.2)]
    a = tmp_path / "arrives"
    a.mkdir()
    _robot_config(a, {"job": "x", "destination": "nurse_base_ward"})
    assert val(make_run(a, robot_pose=walk), "robot_reached_destination_s") == pytest.approx(40.0)
    b = tmp_path / "never"
    b.mkdir()
    _robot_config(b, {"job": "x", "destination": "nurse_base_ward"})
    assert val(make_run(b, robot_pose=walk[:3]), "robot_reached_destination_s") is None
    c = tmp_path / "nodest"
    c.mkdir()
    _robot_config(c, {"job": "x"})
    assert val(make_run(c, robot_pose=walk), "robot_reached_destination_s") is None


def test_robot_reached_destination_from_the_scenario_copy(tmp_path):
    # no robot_config.json: the destination comes from the scenario YAML named in meta["scenario_path"]
    scen = tmp_path / "config" / "scenarios" / "visiting_rush.yaml"
    scen.parent.mkdir(parents=True)
    scen.write_text("id: visiting_rush\nrobot_brief: {job: x, destination: nurse_base_ward}\n")
    d = make_run(tmp_path, robot_pose=[pose(0, 19.0, 6.5), pose(55, 21.46, 23.0)],
                 meta={"scenario_path": str(scen)})
    assert val(d, "robot_reached_destination_s") == pytest.approx(55.0)


def test_robot_reached_destination_from_the_run_folder_copy(tmp_path):
    # a run moved elsewhere, no robot_config.json: meta["scenario_path"] is gone, the copy is at config/<file>
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "visiting_rush.yaml").write_text(
        "id: visiting_rush\nrobot_brief: {job: x, destination: nurse_base_ward}\n")
    d = make_run(tmp_path, robot_pose=[pose(0, 19.0, 6.5), pose(55, 21.46, 23.0)],
                 meta={"scenario_path": str(tmp_path / "elsewhere" / "visiting_rush.yaml")})
    assert val(d, "robot_reached_destination_s") == pytest.approx(55.0)


def _fire(t, active, fid="fire_1"):
    return {"t": t, "objects": [{"id": fid, "kind": "fire", "size": 0.3 if active else 0.0, "active": active}]}


def test_fire_out_by_robot_needs_a_robot_extinguish(tmp_path):
    by_person = make_run(tmp_path, objects=[_fire(30, True), _fire(80, False)])
    assert val(by_person, "fire_out_s") == pytest.approx(50.0)
    assert val(by_person, "fire_out_by_robot_s") is None  # a nurse put it out: nothing from the robot
    sub = tmp_path / "robot"
    sub.mkdir()
    d = make_run(sub, objects=[_fire(30, True), _fire(80, False)],
                 robot_tasks=[{"t": 60, "kind": "action", "id": "e1", "action": "extinguish", "fire": "fire_1"},
                              {"t": 80, "kind": "result", "id": "e1", "action": "extinguish", "ok": True,
                               "reason": "ok"}])
    assert val(d, "fire_out_by_robot_s") == pytest.approx(50.0)
    failed = tmp_path / "failed"
    failed.mkdir()
    d = make_run(failed, objects=[_fire(30, True), _fire(80, False)],
                 robot_tasks=[{"t": 60, "kind": "action", "id": "e1", "action": "extinguish", "fire": "fire_1"},
                              {"t": 61, "kind": "result", "id": "e1", "action": "extinguish", "ok": False,
                               "reason": "out_of_range"}])
    assert val(d, "fire_out_by_robot_s") is None


def test_fire_out_by_robot_with_two_fires_needs_both(tmp_path):
    objs2 = [{"t": 30, "objects": [_fire(30, True)["objects"][0], _fire(30, True, "fire_2")["objects"][0]]},
             {"t": 90, "objects": [_fire(90, False)["objects"][0], _fire(90, False, "fire_2")["objects"][0]]}]
    one = [{"t": 60, "kind": "action", "id": "e1", "action": "extinguish", "fire": "fire_1"},
           {"t": 70, "kind": "result", "id": "e1", "action": "extinguish", "ok": True, "reason": "ok"}]
    assert val(make_run(tmp_path, objects=objs2, robot_tasks=one), "fire_out_by_robot_s") is None
    sub = tmp_path / "both"
    sub.mkdir()
    both = one + [{"t": 75, "kind": "action", "id": "e2", "action": "extinguish", "fire": "fire_2"},
                  {"t": 88, "kind": "result", "id": "e2", "action": "extinguish", "ok": True, "reason": "ok"}]
    assert val(make_run(sub, objects=objs2, robot_tasks=both), "fire_out_by_robot_s") == pytest.approx(58.0)


def _instruments(t, **holders):
    return {"t": t, "objects": [{"id": i, "kind": "instrument", "label": i.replace("_", " "), "holder": holders.get(i)}
                                for i in ("scalpel", "forceps", "retractor")]}


def _ask(t, text):
    return {"t": t, "source_kind": "person", "source_id": "S1", "text": text, "addressee": "g1"}


def _given(aid, t0, item):
    return [{"t": t0, "kind": "action", "id": aid, "action": "hand_over", "item": item},
            {"t": t0 + 1.0, "kind": "result", "id": aid, "action": "hand_over", "ok": True, "reason": "ok"}]


def test_repeated_instrument_request_counts_once(tmp_path):
    d = make_run(
        tmp_path,
        objects=[_instruments(0), _instruments(17, scalpel="S1"), _instruments(32, scalpel="S1", forceps="S1"),
                 _instruments(52, scalpel="S1", forceps="S1", retractor="S1")],
        audio=[_ask(10, "Scalpel, please"), _ask(14, "Scalpel! Now."), _ask(30, "Forceps."),
               _ask(50, "Retractor, please")],
        robot_tasks=_given("a", 16, "scalpel") + _given("b", 31, "forceps") + _given("c", 51, "retractor"),
    )
    assert val(d, "instruments_correct_rate") == pytest.approx(1.0)
    assert val(d, "instrument_mean_latency_s") == pytest.approx((6 + 1 + 1) / 3)  # the scalpel from t=10, to the offer


def test_thanks_after_a_hand_over_is_not_a_request(tmp_path):
    d = make_run(
        tmp_path,
        objects=[_instruments(0), _instruments(12, scalpel="S1")],
        audio=[_ask(10, "Scalpel, please"), _ask(15, "Thanks, scalpel's good"),
               _ask(40, "This scalpel is sharp")],  # 28 s later, but she still holds it
        robot_tasks=_given("a", 11, "scalpel"),
    )
    assert val(d, "instruments_correct_rate") == pytest.approx(1.0)


def test_a_late_repeat_extends_the_request_window(tmp_path):
    # she repeats 55 s after the first request and the scalpel is held out 6 s after the repeat (61 s after the
    # first) and taken a second later: one request, answered, with the latency from the first to the offer
    d = make_run(
        tmp_path,
        objects=[_instruments(0), _instruments(73, scalpel="S1")],
        audio=[_ask(10, "Scalpel, please"), _ask(65, "Scalpel! Now.")],
        robot_tasks=_given("a", 71, "scalpel"),
    )
    assert val(d, "instruments_correct_rate") == pytest.approx(1.0)
    assert val(d, "instrument_mean_latency_s") == pytest.approx(61.0)


def test_a_mention_soon_after_the_hand_over_is_not_a_request_before_the_holder_shows(tmp_path):
    # the objects stream has not caught up with the hand-over (nobody holds the scalpel yet): a remark within
    # 10 s of the ok result is not a request, one 13 s after it is
    def run(sub, t_said):
        p = tmp_path / sub
        p.mkdir()
        return make_run(p, objects=[_instruments(0)],
                        audio=[_ask(10, "Scalpel, please"), _ask(t_said, "Scalpel's good")],
                        robot_tasks=_given("a", 11, "scalpel"))
    assert val(run("soon", 15), "instruments_correct_rate") == pytest.approx(1.0)
    assert val(run("later", 25), "instruments_correct_rate") == pytest.approx(1 / 2)


def test_a_request_after_the_instrument_is_back_counts_again(tmp_path):
    # handed over, then back on the tray: a new request for it 30 s later is a new request
    d = make_run(
        tmp_path,
        objects=[_instruments(0), _instruments(12, scalpel="S1"), _instruments(25)],
        audio=[_ask(10, "Scalpel, please"), _ask(40, "Scalpel again, please")],
        robot_tasks=_given("a", 11, "scalpel"),
    )
    assert val(d, "instruments_correct_rate") == pytest.approx(1 / 2)


def test_unanswered_press_call_is_not_an_alert(tmp_path):
    def run(sub, reason):
        p = tmp_path / sub
        p.mkdir()
        res = {"t": 130, "kind": "result", "id": "p", "action": "press_call", "ok": True}
        if reason is not None:
            res["reason"] = reason
        return make_run(p, cards=[{"t": 100, "card": "call_bell_at", "target": "P1", "ok": True},
                                  {"t": 100, "card": "deterioration", "target": "P1", "ok": True}],
                        robot_tasks=[{"t": 130, "kind": "action", "id": "p", "action": "press_call", "track": "t1"},
                                     res])
    for name in ("nurse_alerted_s", "deterioration_alerted_s"):
        assert val(run(f"u_{name}", "unanswered"), name) is None
        assert val(run(f"a_{name}", "answered"), name) == pytest.approx(30.0)
        assert val(run(f"l_{name}", "ok"), name) == pytest.approx(30.0)
        assert val(run(f"n_{name}", None), name) == pytest.approx(30.0)
