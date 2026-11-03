"""DirectorService: the director node without ROS, and its DirectorAPI for the panel."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control, Nudge, ReflexOverride, Roster, RosterEntry, Status, WorldEvent
from wardzero_agents.panel.server import BadRequest
from wardzero_agents.ros.director_node import DirectorService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


class Out:
    def __init__(self):
        self.msgs = []

    def publish(self, kind, obj):
        self.msgs.append((kind, obj))

    def of(self, kind):
        return [o for k, o in self.msgs if k == kind]


def make(tmp_path, **kw):
    out = Out()
    loaded = load_scenario(V1)
    snaps = []
    s = DirectorService(loaded, RunLog(tmp_path), out, run_id="run1", snapshot_fn=snaps.append, **kw)
    s.on_roster(Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name,
                                                seat=i.seat, companion_of=i.companion_of)
                                    for i in loaded.instances]))
    for i in loaded.instances:
        s.on_status(status(i.person_id, t=0.0))
    return s, out, snaps


def status(pid, posture="seated", t=0.0, x=-5.0, y=0.2):
    return Status(person=pid, t=t, x=x, y=y, yaw=0, posture=posture, progress=0, state="idle")


def lines(tmp_path, stream):
    p = tmp_path / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_seated_since_tracks_posture_transitions(tmp_path):
    s, _, _ = make(tmp_path)
    assert s.seated_since["P4"] == 0.0
    s.on_status(status("P4", t=10.0))
    assert s.seated_since["P4"] == 0.0  # still seated: unchanged
    s.on_status(status("P4", "standing", t=20.0))
    assert "P4" not in s.seated_since
    s.on_status(status("P4", "walking", t=21.0))
    s.on_status(status("P4", "seated", t=30.0))
    assert s.seated_since["P4"] == 30.0
    s.on_status(status("P4", "gone", t=40.0))
    assert "P4" not in s.seated_since and "P4" not in s.view(41.0).present


def test_storyboard_beat_fires_and_is_recorded(tmp_path):
    s, out, _ = make(tmp_path)
    s.tick(299.0)
    assert out.of("world") == []
    s.tick(300.0)  # b1 clinic_delay at t+05:00
    w = out.of("world")
    assert len(w) == 1 and isinstance(w[0], WorldEvent) and w[0].kind == "pa_announcement"
    cards = lines(tmp_path, "cards")
    assert cards[0]["card"] == "clinic_delay" and cards[0]["ok"] is True and cards[0]["source"] == "storyboard"
    assert lines(tmp_path, "world")[0]["kind"] == "pa_announcement"
    assert any("clinic_delay" in e["text"] for e in s.state()["events"])


def test_coughing_fit_sends_reflex_and_companion_nudge(tmp_path):
    s, out, _ = make(tmp_path)
    r = s.fire("coughing_fit", "P2", {})
    assert r["ok"] is True and r["firing_id"].startswith("panel-")
    assert [x.person for x in out.of("reflex")] == ["P2"] and isinstance(out.of("reflex")[0], ReflexOverride)
    assert [x.person for x in out.of("nudge")] == ["P3"] and isinstance(out.of("nudge")[0], Nudge)
    assert lines(tmp_path, "reflex")[0]["person"] == "P2" and lines(tmp_path, "nudges")[0]["person"] == "P3"


def test_rejections_are_recorded_with_error(tmp_path):
    s, out, _ = make(tmp_path)
    s.on_status(status("P1", "gone", t=5.0))
    r = s.fire("name_called", "P1", {})
    assert r == {"ok": False, "error": "target P1 is not in the room"}
    assert s.fire("no_such_card", None, {})["ok"] is False
    rec = lines(tmp_path, "cards")
    assert rec[0]["ok"] is False and rec[0]["error"] == "target P1 is not in the room"
    assert {k for k, _ in out.msgs} == {"event"}  # nothing reached the people or the world


def test_control_is_published_and_tracked(tmp_path):
    s, out, _ = make(tmp_path)
    s.control("pause")
    assert out.of("control")[0] == Control(action="pause") and s.state()["paused"] is True
    s.control("resume")
    assert s.state()["paused"] is False
    s.on_control(Control(action="stop"))
    assert s.stopped and s.state()["running"] is False


def test_beats_skip_rerun_and_unknown(tmp_path):
    s, _, _ = make(tmp_path)
    s.beat("skip", "b1")
    assert s.state()["timeline"][0]["state"] == "skipped"
    s.beat("rerun", "b1")
    assert s.state()["timeline"][0]["state"] == "pending"
    with pytest.raises(BadRequest):
        s.beat("skip", "nope")


def test_chaos_toggle(tmp_path):
    s, out, _ = make(tmp_path)
    s.set_chaos(True, 3600.0)
    assert s.state()["chaos"] == {"enabled": True, "rate_per_hour": 3600.0}
    for k in range(1, 30):
        s.tick(float(k))
    assert any(c["source"] == "chaos" for c in lines(tmp_path, "cards"))
    s.set_chaos(False, 3600.0)
    n = len(lines(tmp_path, "cards"))
    for k in range(30, 60):
        s.tick(float(k))
    assert len(lines(tmp_path, "cards")) == n


def test_snapshot_is_handed_off(tmp_path):
    s, _, snaps = make(tmp_path)
    assert s.snapshot("before-lunch")["ok"] is True and snaps == ["before-lunch"]
    with pytest.raises(BadRequest):
        s.snapshot("../evil")


def test_fairness_violations_are_recorded(tmp_path, caplog):
    s, _, _ = make(tmp_path)
    s.record_fairness([{"node": "spy", "topic": "/humans/P1/intent"}], 3.0)
    s.record_fairness([], 33.0)
    assert lines(tmp_path, "fairness") == [{"t": 3.0, "node": "spy", "topic": "/humans/P1/intent"}]
    assert "spy" in caplog.text


def test_run_tail_feeds_cast_and_cost(tmp_path):
    s, _, _ = make(tmp_path)
    s.ingest("decisions", {"t": 1.0, "person": "P5", "result": {"intent": {"kind": "approach", "target": "g1"},
                                                                 "mood": "anxious"}, "reason": "lost", "error": None})
    s.ingest("says", {"t": 1.0, "person": "P5", "text": "Excuse me"})
    s.ingest("tokens", {"t": 1.0, "cost_usd": 0.25})
    s.ingest("tokens", {"t": 2.0, "cost_usd": None})
    s.tick(1.0)
    p5 = {c["id"]: c for c in s.state()["cast"]}["P5"]
    assert p5["mood"] == "anxious" and p5["last_decision"] == "approach g1"
    assert p5["reason"] == "lost" and p5["last_said"] == "Excuse me"
    assert s.state()["cost_usd"] == 0.25


TOP = {"t", "running", "paused", "run_id", "recording", "cost_usd", "budget_usd", "chaos", "timeline",
       "cast", "events", "deck", "map"}


def test_state_matches_the_panel_shape(tmp_path):
    s, _, _ = make(tmp_path)
    s.tick(2.0)
    st = s.state()
    json.dumps(st)  # serialisable
    assert set(st) == TOP
    assert isinstance(st["t"], float) and st["run_id"] == "run1" and st["recording"] is True
    assert set(st["chaos"]) == {"enabled", "rate_per_hour"}
    assert all({"id", "card", "target", "state", "fired_t"} <= set(r) for r in st["timeline"])
    assert len(st["cast"]) == 5
    assert all(set(c) == {"id", "display_name", "persona", "x", "y", "posture", "mood", "last_decision",
                          "reason", "last_said"} for c in st["cast"])
    assert all(set(e) == {"t", "text"} for e in st["events"])
    assert {d["id"] for d in st["deck"]} == set(s.loaded.deck)
    assert all(set(d) == {"id", "label", "target_kind", "tests"} for d in st["deck"])
    m = st["map"]
    assert set(m) == {"bounds", "seats", "landmarks", "signs", "robot"}
    xmin, xmax, ymin, ymax = m["bounds"]
    assert xmin <= -11.5 and xmax >= 11.2 and ymin <= -2.3 and ymax >= 7.25
    assert all(set(x) == {"id", "x", "y"} for x in m["seats"]) and len(m["seats"]) == 12
    assert all(set(x) == {"id", "label", "x", "y"} for x in m["landmarks"])
    assert all(set(x) == {"id", "text", "x", "y"} for x in m["signs"])
    assert set(m["robot"]) == {"x", "y", "yaw"}


def test_state_is_a_cached_snapshot(tmp_path):
    s, _, _ = make(tmp_path)
    s.tick(2.0)
    a = s.state()
    s.on_status(status("P1", "standing", t=2.5))
    assert s.state() == a  # statuses alone do not rebuild it; the tick does
    s.tick(3.0)
    assert {c["id"]: c for c in s.state()["cast"]}["P1"]["posture"] == "standing"
