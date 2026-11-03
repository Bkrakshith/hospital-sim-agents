"""Replay node services without ROS."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Intent, Say, Status, WorldEvent
from wardzero_agents.panel.server import BadRequest
from wardzero_agents.replay import IntentScript
from wardzero_agents.ros.replay_node import ExactReplay, ReplayAPI, timeline_from_cards
from wardzero_agents.runlog import RunLog

from tests.test_director_service import TOP

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def st(pid, t, posture="seated"):
    return Status(person=pid, t=t, x=1.0, y=2.0, yaw=0, posture=posture, progress=0, state="idle")


def recording(tmp_path):
    log = RunLog(tmp_path)
    for t in (0.0, 0.5, 1.0):
        log.write("status", st("P1", t).model_dump(mode="json", by_alias=True))
    log.write("audio", {"schema": "wz.v1", "id": "a1", "t": 0.7, "source_kind": "cough", "source_id": "P1",
                        "x": 1, "y": 2, "z": 1.6, "loudness_db": 68, "text": None, "addressee": None})
    log.write("cards", {"t": 0.6, "id": "storyboard-1", "card": "clinic_delay", "target": None, "params": {},
                        "source": "storyboard", "ok": True, "error": None})
    log.write("cards", {"t": 0.9, "id": "panel-1", "card": "name_called", "target": "P1", "params": {},
                        "source": "panel", "ok": False, "error": "target P1 is not in the room"})
    log.write_meta({"run_id": "r1", "cost_usd": 0.5, "scenario_path": str(V1)})
    return tmp_path


def test_timeline_from_cards():
    rows = timeline_from_cards([
        {"t": 1.0, "id": "storyboard-1", "card": "clinic_delay", "target": None, "ok": True, "error": None},
        {"t": 2.0, "id": "panel-1", "card": "name_called", "target": "P1", "ok": False, "error": "gone"},
    ])
    assert rows == [
        {"id": "storyboard-1", "card": "clinic_delay", "target": None, "state": "fired", "fired_t": 1.0},
        {"id": "panel-1", "card": "name_called", "target": "P1", "state": "rejected", "fired_t": 2.0,
         "reason": "gone"},
    ]


def test_exact_replay_steps_through_the_recording(tmp_path):
    r = ExactReplay(recording(tmp_path))
    assert r.end_t == 1.0
    first = r.step(0.6)
    assert [s for s, _ in first] == ["status", "status", "cards"]
    assert r.statuses["P1"].t == 0.5 and [x["card"] for x in r.timeline()] == ["clinic_delay"]
    rest = r.step(5.0)
    assert [s for s, _ in rest] == ["audio", "cards", "status"] and r.done(5.0)


def test_replay_api_is_read_only_and_has_the_panel_shape(tmp_path):
    r = ExactReplay(recording(tmp_path))
    api = ReplayAPI(r, load_scenario(V1), run_id="r1", meta={"cost_usd": 0.5})
    r.step(1.0)
    api.refresh(1.0)
    s = api.state()
    json.dumps(s)
    assert set(s) == TOP and s["recording"] is False and s["cost_usd"] == 0.5
    assert [c["id"] for c in s["cast"]] == ["P1"] and len(s["timeline"]) == 2
    assert s["map"]["seats"] and s["deck"]
    with pytest.raises(BadRequest):
        api.control("pause")
    with pytest.raises(BadRequest):
        api.set_chaos(True, 6.0)
    with pytest.raises(BadRequest):
        api.beat("skip", "b1")
    assert api.fire("clinic_delay", None, {})["ok"] is False
    assert api.snapshot("x")["ok"] is False


def test_replay_api_without_a_scenario(tmp_path):
    r = ExactReplay(recording(tmp_path))
    api = ReplayAPI(r, None, run_id="r1", meta={})
    api.refresh(0.0)
    s = api.state()
    assert set(s) == TOP and s["deck"] == [] and set(s["map"]) == {"bounds", "seats", "landmarks", "signs", "robot"}


def test_intent_script_items_for_the_intents_mode(tmp_path):
    log = RunLog(tmp_path)
    log.write("intents", Intent(id="i", person="P5", t=2.0, kind="walk_to", target="toilets").model_dump(
        mode="json", by_alias=True))
    log.write("says", Say(person="P5", t=2.0, text="Excuse me").model_dump(mode="json", by_alias=True))
    log.write("world", WorldEvent(id="w", t=1.0, kind="pa_announcement", text="x").model_dump(
        mode="json", by_alias=True))
    s = IntentScript(tmp_path)
    assert [type(x).__name__ for x in s.due(2.0)] == ["WorldEvent", "Intent", "Say"]


def test_intent_replay_records_what_it_sends_so_the_replay_is_replayable(tmp_path):
    from wardzero_agents.ros.replay_node import IntentReplay
    src, new = tmp_path / "src", tmp_path / "new"
    log = RunLog(src)
    i = Intent(id="i", person="P5", t=2.0, kind="walk_to", target="toilets")
    say = Say(person="P5", t=2.0, text="Excuse me")
    w = WorldEvent(id="w", t=1.0, kind="arrival", persona="ward_visitor")
    for stream, m in (("intents", i), ("says", say), ("world", w)):
        log.write(stream, m.model_dump(mode="json", by_alias=True))
    r = IntentReplay(src, RunLog(new))
    assert [type(x).__name__ for x in r.step(1.5)] == ["WorldEvent"]
    assert [type(x).__name__ for x in r.step(3.0)] == ["Intent", "Say"]
    again = IntentScript(new)  # the new folder replays the same items
    assert again.due(10.0) == [w, i, say]
    assert IntentReplay(src).step(10.0) == [w, i, say]  # without a recorder it only reads
