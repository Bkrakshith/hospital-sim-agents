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
        robot_tasks=[{"t": 66, "kind": "action", "id": "a1", "action": "press_call"}],
        status=[status(100.0, "P1", 0, 0)],
    )
    # ask@10 answered after 4 s; ask@30 first reply at 45 = 15 s (over 10 s, unanswered); ask@60 answered by
    # an action 6 s later. The P1-to-P2 line is not addressed to the robot.
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
    # scalpel answered in 2 s (hand over at 11, result at 12). Forceps asked, scalpel (not requested) given.
    # Suture scissors arrive 21 s after the request: late (60 s window) but still correct here.
    assert val(d, "instruments_correct_rate") == pytest.approx(2 / 4)  # 2 correct / (3 requests + 1 unrequested)
    assert val(d, "instrument_mean_latency_s") == pytest.approx((2 + 21) / 2)


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
        robot_tasks=[{"t": 270, "kind": "action", "id": "p", "action": "press_call"},
                     {"t": 270, "kind": "result", "id": "p", "action": "press_call", "ok": True, "reason": "ok"},
                     {"t": 330, "kind": "action", "id": "q", "action": "press_call"},
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
        status=[status(0, "P1", 1.0, 0.0)], robot_pose=[pose(0, 0.7, 0.0)],
        audio=[{"t": 1, "source_kind": "person", "source_id": "P1", "text": "Hi", "addressee": "g1"},
               {"t": 9, "source_kind": "robot", "source_id": "g1", "text": "Hello", "addressee": "P1"}],
    )
    r = evaluate(preset, d)
    assert isinstance(r, CriteriaResult)
    assert [(x["id"], x["passed"]) for x in r.rows] == [("a", True), ("b", False), ("c", False)]
    assert r.rows[0]["value"] == 0.0 and r.rows[1]["value"] == pytest.approx(0.3)
    assert r.score == pytest.approx(100 * 3 / 4.5)
    assert r.passed is False  # b has weight 1 and failed


def test_low_weight_failure_does_not_fail_the_run(tmp_path):  # a None on a weight-0.5 criterion
    preset = _preset([
        {"id": "a", "label": "A", "metric": "contacts", "op": "==", "threshold": 0, "weight": 2},
        {"id": "b", "label": "B", "metric": "mean_response_s", "op": "<=", "threshold": 1, "weight": 0.5},
    ])
    d = make_run(tmp_path, status=[status(0, "P1", 5, 5)], robot_pose=[pose(0, 0, 0)])
    r = evaluate(preset, d)
    assert [x["passed"] for x in r.rows] == [True, None] and r.passed is True  # nobody asked: None
    assert r.score == pytest.approx(100 * 2 / 2.5)


def test_metric_none_counts_failed(tmp_path):
    preset = _preset([
        {"id": "a", "label": "A", "metric": "fire_out_s", "op": "<=", "threshold": 120, "weight": 4},
        {"id": "b", "label": "B", "metric": "contacts", "op": "==", "threshold": 0, "weight": 1},
    ])
    r = evaluate(preset, make_run(tmp_path, status=[status(0, "P1", 5, 5)]))
    assert r.rows[0]["value"] is None and r.rows[0]["passed"] is None
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
