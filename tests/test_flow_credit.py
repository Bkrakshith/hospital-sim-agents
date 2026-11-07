"""Flows credit only what the robot really did: the receiver and item of a hand-over, the person it follows, speech
heard by the people around it, actions after the event, and a whole guide (not its first second)."""
from pathlib import Path

import pytest

from wardzero_agents.config.loader import _load_deck
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Roster, RosterEntry, Status
from wardzero_agents.flows.engine import FlowEngine, FlowView, holds
from wardzero_agents.flows.spec import ROBOT_ACTIONS, Flow, FlowError, fit_scenario, flows_dir, load_flow
from wardzero_agents.site import load_site

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
DECK = _load_deck(CONFIG / "cards" / "studio.yaml")
SITE = load_site(CONFIG)


def st(pid, x=0.0, y=0.0, posture="standing"):
    return Status(person=pid, t=0.0, posture=posture, state="idle", progress=0.0, x=x, y=y, yaw=0.0)


def _director(tmp_path, preset="ae_waiting", statuses=True):
    from wardzero_agents.ros.director_node import DirectorService
    from wardzero_agents.runlog import RunLog

    class Out:
        def publish(self, kind, obj):
            pass

    loaded = preset_to_scenario(PRESETS[preset], CONFIG, tmp_path / "staged")
    d = DirectorService(loaded, RunLog(tmp_path / "run"), Out(), run_id="r1")
    d.on_roster(Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                    for i in loaded.instances]))
    if statuses:
        for c in loaded.scenario.cast:
            x, y = (c.x, c.y) if getattr(c, "x", None) is not None else (0.0, 0.0)
            d.on_status(Status(person=c.id, t=0.0, x=x, y=y, yaw=0, posture="standing", progress=0, state="idle"))
    return d


def _node(tree: dict, label: str) -> dict:
    if tree["label"] == label:
        return tree
    for c in tree["children"]:
        try:
            return _node(c, label)
        except KeyError:
            pass
    raise KeyError(label)


def _labels(flow) -> list:
    return [n.label for n in FlowEngine(flow)._ctx.nodes]


def _run(pid, *, statuses=None, view_kw=lambda t: {}, until=None, dt=0.5):
    """Tick a preset's flow; statuses(t) gives where people are (default: where they start)."""
    p = PRESETS[pid]
    flow = load_flow(flows_dir(CONFIG) / f"{pid}.yaml")
    start = {c.id: st(c.id, c.x, c.y) for c in p.characters}
    e = FlowEngine(flow, DECK)
    fired, t = [], 0.0
    rs = (p.robot_start.x, p.robot_start.y)
    while t <= (until or p.duration_min * 60) and not e.finished:
        people = statuses(t) if statuses else start
        kw = {"robot_xy": rs, **view_kw(t)}
        fired += [(f.t, f.card, f.target) for f in e.tick(FlowView(t=t, statuses=people, robot_start=rs,
                                                                   present=set(people), cast=set(start),
                                                                   room_at=SITE.room_at, **kw))]
        t += dt
    return e, fired


def _said(at: dict, t: float) -> dict:
    out = {k: [x for x in v if x <= t] for k, v in at.items()}
    return {"robot_said_to": {k for k, v in out.items() if v}, "robot_said_at": {k: v for k, v in out.items() if v}}


# ---- the director records who and what -------------------------------------------------------------------------
def test_the_director_records_the_receiver_and_item_of_a_hand_over(tmp_path):
    d = _director(tmp_path, "meal_round")
    d.ingest("robot_tasks", {"kind": "hand_over", "event": "offer", "person": "M3", "item": "tray_M3", "t": 4.0})
    assert d.robot_did == []  # an offer is not a hand-over
    d.ingest("robot_tasks", {"kind": "result", "id": "a1", "action": "hand_over", "ok": True, "reason": "ok",
                             "person": "M3", "item": "tray_M2", "t": 5.0})
    d.ingest("robot_tasks", {"kind": "result", "id": "a2", "action": "take", "ok": True, "reason": "ok",
                             "item": "tray_M2", "t": 6.0})
    v = d.view(7.0)
    assert holds({"robot_did": {"action": "hand_over", "about": "M3"}}, v)
    assert holds({"robot_did": {"action": "hand_over", "item": "tray_M2"}}, v)
    assert not holds({"robot_did": {"action": "hand_over", "about": "M3", "item": "tray_M3"}}, v)
    assert not holds({"robot_did": {"action": "hand_over", "about": "M2"}}, v)
    assert holds({"robot_did": {"action": "take", "item": "tray_M2"}}, v)
    # a failed hand-over is not credited
    d.ingest("robot_tasks", {"kind": "result", "id": "a3", "action": "hand_over", "ok": False, "reason": "refused",
                             "person": "M1", "item": "tray_M1", "t": 8.0})
    assert not holds({"robot_did": {"action": "hand_over", "about": "M1"}}, d.view(9.0))


def test_the_director_records_whom_the_robot_follows(tmp_path):
    assert "follow" in ROBOT_ACTIONS
    d = _director(tmp_path, "wandering_patient")
    d.ingest("robot_tasks", {"kind": "follow", "event": "start", "person": "W1", "reason": "following", "t": 3.0})
    d.ingest("robot_tasks", {"kind": "result", "id": "a1", "action": "follow", "ok": True, "reason": "following",
                             "t": 3.0})
    d.ingest("robot_tasks", {"kind": "follow", "event": "end", "person": "W1", "reason": "guide_stop", "t": 9.0})
    assert [(a, p) for _, a, p, _ in d.robot_did] == [("follow", "W1")]
    assert holds({"robot_did": {"action": "follow", "about": "W1", "since_step": True}}, d.view(4.0), t0=3.0)


def test_robot_did_item_is_checked_by_the_schema():
    Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": {"robot_did": {"action": "hand_over",
                                                                                  "about": "P1", "item": "tray_1"}},
                                               "timeout_s": 5}}])
    with pytest.raises(ValueError):
        Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": {"robot_did": {"action": "hand_over",
                                                                                      "item": 3}}, "timeout_s": 5}}])
    f = Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": {"robot_did": {"action": "hand_over",
                                                                                      "item": "nope"}},
                                                   "timeout_s": 5}}])
    fit_scenario(f, people={"P1"}, cards=DECK, rooms=SITE.rooms)  # items unknown to the caller: not checked
    with pytest.raises(FlowError, match="no item 'nope'"):
        fit_scenario(f, people={"P1"}, cards=DECK, rooms=SITE.rooms, items={"tray_1"})


def test_robot_did_reads_old_entries_with_three_or_two_fields():
    v = FlowView(t=10.0, statuses={}, robot_xy=(0, 0), robot_did=[("guide", "P1"), (4.0, "tell", "P1"),
                                                                  (5.0, "hand_over", "P2", "tray_2")])
    assert holds({"robot_did": {"action": "guide", "about": "P1"}}, v)
    assert holds({"robot_did": {"action": "tell", "about": "P1", "since_step": True}}, v, t0=4.0)
    assert holds({"robot_did": {"action": "hand_over", "about": "P2", "item": "tray_2"}}, v)
    assert not holds({"robot_did": {"action": "tell", "item": "tray_2"}}, v)


# ---- speech the people around the robot heard ------------------------------------------------------------------
def _speech_director(tmp_path, where):
    d = _director(tmp_path, "reception_guide", statuses=False)
    for pid, (x, y) in where.items():
        d.on_status(st(pid, x, y))
    d.robot_xy = (0.0, 0.0)
    return d


def test_unaddressed_speech_counts_for_people_who_heard_it_within_4_m(tmp_path):
    d = _speech_director(tmp_path, {"V1": (3.3, 0.0), "H1": (6.0, 0.0), "V2": (1.0, 0.0)})
    d.ingest("audio", {"source_kind": "robot", "addressee": None, "text": "Hello, welcome", "x": 0.0, "y": 0.0,
                       "heard_by": ["V1", "H1"], "t": 1.0})
    assert d.robot_said_to == {"V1"}  # V2 is near but did not hear it; H1 heard it from 6 m away
    d.ingest("audio", {"source_kind": "robot", "addressee": None, "text": "Everyone, this way", "x": 0.0, "y": 0.0,
                       "heard_by": ["V1", "H1"], "to_room": True, "t": 2.0})
    assert d.robot_said_to == {"V1", "H1"}  # a line to the room counts for everyone who heard it
    assert d.robot_said_at["H1"] == [2.0]


def test_room_toward_value_and_old_rows_without_heard_by(tmp_path):
    d = _speech_director(tmp_path, {"V1": (3.3, 0.0), "H1": (2.0, 0.0), "V2": (12.0, 0.0)})
    d.ingest("audio", {"source_kind": "robot", "addressee": "room", "text": "Everyone", "x": 0.0, "y": 0.0,
                       "heard_by": ["V1", "V2"], "t": 1.0})
    assert d.robot_said_to == {"V1", "V2"} and "room" not in d.robot_said_to
    d = _speech_director(tmp_path / "old", {"V1": (3.3, 0.0), "H1": (2.0, 0.0)})
    d.ingest("audio", {"source_kind": "robot", "addressee": None, "text": "Hello", "t": 1.0})
    assert d.robot_said_to == {"H1"}  # no heard_by (an old run): within 2.5 m, as before
    d.ingest("audio", {"source_kind": "robot", "addressee": "V1", "text": "Hello", "heard_by": ["V1"], "t": 2.0})
    assert d.robot_said_to == {"H1", "V1"}


# ---- a targeted card at the very start ------------------------------------------------------------------------
def test_a_targeted_card_on_the_first_tick_waits_for_its_target_to_appear():
    flow = Flow(id="x", title="x", steps=[{"fire": {"id": "b0", "card": "deterioration", "target": "P1"}},
                                         {"wait": {"seconds": 1}}])
    e = FlowEngine(flow)
    assert e.tick(FlowView(t=0.0, statuses={}, robot_xy=(0, 0), present=set(), cast={"P1"})) == []
    assert e.timeline()[0]["state"] == "pending"
    out = e.tick(FlowView(t=0.5, statuses={"P1": st("P1")}, robot_xy=(0, 0), present={"P1"}, cast={"P1"}))
    assert [(f.card, f.target) for f in out] == [("deterioration", "P1")]
    # someone who never shows up is skipped after a short grace, and a person who has gone at once
    e = FlowEngine(flow)
    t = 0.0
    while t < 30 and e.timeline()[0]["state"] == "pending":
        e.tick(FlowView(t=t, statuses={}, robot_xy=(0, 0), present=set(), cast={"P1"}))
        t += 0.5
    assert e.timeline()[0]["state"] == "skipped" and t < 30
    e = FlowEngine(flow)
    e.tick(FlowView(t=0.0, statuses={"P1": st("P1", posture="gone")}, robot_xy=(0, 0), present=set(), cast={"P1"}))
    assert e.timeline()[0]["state"] == "skipped"


def test_specimen_run_plays_the_corridor_beat_when_the_first_tick_comes_before_any_status(tmp_path):
    d = _director(tmp_path, "specimen_run", statuses=False)
    d.tick(0.0)  # the director's first tick: no person's status has arrived yet
    loaded_cast = [c.id for c in d.loaded.scenario.cast]
    p = {c.id: c for c in PRESETS["specimen_run"].characters}
    for pid in loaded_cast:
        c = p.get(pid)
        d.on_status(st(pid, c.x if c else 0.0, c.y if c else 0.0))
    d.tick(0.5)
    b0 = [r for r in d.engine.timeline() if r["id"] == "b0"][0]
    assert b0["state"] == "fired", b0


# ---- the flow tree --------------------------------------------------------------------------------------------
def test_a_single_step_parallel_branch_has_no_extra_in_order_row():
    flow = load_flow(flows_dir(CONFIG) / "evacuation_guide.yaml")
    tree = FlowEngine(flow).tree()
    par = tree["children"][1]
    assert par["kind"] == "parallel"
    assert [c["kind"] for c in par["children"]] == ["fire"] * 5  # each branch is one card: no "In order" row
    everyone = tree["children"][2]
    assert everyone["children"][0]["kind"] == "sequence"  # a branch of several steps keeps its row
