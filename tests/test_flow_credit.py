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


# ---- each preset flow credits only the right thing --------------------------------------------------------------
def test_specimen_run_credits_only_the_box_handed_to_the_porter():
    label = "The robot hands the box to the porter"
    near = lambda t: {"robot_xy": (9.6, 0.66) if t >= 20 else (19.0, 6.0)}  # noqa: E731
    e, _ = _run("specimen_run", view_kw=lambda t: {**near(t), "robot_did": [(30.0, "hand_over", "TR2", "specimen_1")]
                                                    if t >= 30 else []})
    assert _node(e.tree(), label)["state"] == "timed_out"
    e, _ = _run("specimen_run", view_kw=lambda t: {**near(t), "robot_did": [(30.0, "hand_over", "PO1", "specimen_1")]
                                                    if t >= 30 else []})
    assert _node(e.tree(), label)["state"] == "met"


MEAL_TRAY = "The robot delivers a first tray"
MEAL_BELL = "The robot answers M3 (goes to them or fetches a nurse)"


def _meal(did_at, robot_xy_at=lambda t: (19.0, 6.0), said_at=None):
    return _run("meal_round", view_kw=lambda t: {"robot_xy": robot_xy_at(t), "robot_did": did_at(t),
                                                 **_said(said_at or {}, t)})


def test_meal_round_credits_only_the_right_tray_to_the_right_patient():
    e, _ = _meal(lambda t: [(40.0, "hand_over", "M3", "tray_M2")] if t >= 40 else [])  # George's tray to Joan
    assert _node(e.tree(), MEAL_TRAY)["state"] == "timed_out"
    e, _ = _meal(lambda t: [(40.0, "hand_over", "M3", "tray_M3")] if t >= 40 else [])
    assert _node(e.tree(), MEAL_TRAY)["state"] == "met"


def test_meal_round_bell_needs_an_answer_after_it_rings():
    m3 = (25.109, 16.204 - 1.0)
    tray = lambda t: [(40.0, "hand_over", "M3", "tray_M3")] if t >= 40 else []  # noqa: E731
    # the robot is still beside Joan when her bell rings, having spoken to her when it brought the tray
    e, fired = _meal(lambda t: tray(t) + ([(45.0, "press_call", None, None)] if t >= 45 else []),
                     robot_xy_at=lambda t: m3 if t >= 35 else (19.0, 6.0), said_at={"M3": [39.0]})
    bell_t = next(t for t, c, _ in fired if c == "ring_bell")
    assert bell_t > 45.0
    assert _node(e.tree(), MEAL_BELL)["state"] == "timed_out"
    # it speaks to her after the bell rings
    e, _ = _meal(tray, robot_xy_at=lambda t: m3 if t >= 35 else (19.0, 6.0), said_at={"M3": [39.0, bell_t + 3]})
    assert _node(e.tree(), MEAL_BELL)["state"] == "met"
    # or it fetches a nurse after the bell rings
    e, _ = _meal(lambda t: tray(t) + ([(bell_t + 5, "press_call", None, None)] if t >= bell_t + 5 else []))
    assert _node(e.tree(), MEAL_BELL)["state"] == "met"


def test_night_call_bell_needs_answers_after_the_bell():
    p1 = next(c for c in PRESETS["night_call_bell"].characters if c.id == "P1")
    kw = lambda t: {"robot_xy": (p1.x + 1.0, p1.y), "robot_did": [(5.0, "press_call", None, None)],  # noqa: E731
                    **_said({"P1": [6.0]}, t)}
    e, _ = _run("night_call_bell", view_kw=kw, until=200)
    assert _node(e.tree(), "The robot reaches P1 and reassures them")["state"] == "timed_out"
    assert _node(e.tree(), "The robot fetches the nurse")["state"] == "timed_out"


def test_wandering_patient_does_not_credit_a_call_before_she_gets_up():
    e, _ = _run("wandering_patient", view_kw=lambda t: {"robot_did": [(7.0, "press_call", None, None)]}, until=200)
    assert _node(e.tree(), "The robot calls a nurse")["state"] == "timed_out"
    e, _ = _run("wandering_patient", view_kw=lambda t: {"robot_did": [(30.0, "press_call", None, None)]
                                                         if t >= 30 else []}, until=200)
    assert _node(e.tree(), "The robot calls a nurse")["state"] == "met"


def test_wandering_patient_credits_a_follow():
    w1 = next(c for c in PRESETS["wandering_patient"].characters if c.id == "W1")
    e, _ = _run("wandering_patient", view_kw=lambda t: {"robot_xy": (w1.x + 1.2, w1.y),
                                                         "robot_did": [(25.0, "follow", "W1", None)]
                                                         if t >= 25 else []}, until=200)
    assert _node(e.tree(), "The robot goes to W1")["state"] == "met"


def test_ward_fire_starts_the_alarm_and_the_evacuation_with_the_fire():
    _, fired = _run("ward_fire", until=120)
    at = {c: t for t, c, _ in fired}
    assert at["fire_start"] == at["alarm"] == at["staff_evacuate"]


def test_ward_fire_credits_only_a_spray_after_the_fire_started():
    label = "The robot puts the fire out"
    e, _ = _run("ward_fire", view_kw=lambda t: {"fires": {"f1": True}, "robot_did": [(1.0, "extinguish", None, None)]})
    assert _node(e.tree(), label)["state"] == "timed_out"


AE_GEORGE = "The robot checks George in (a desk note, a walk to a seat or a word with staff)"
AE_WAITS = "The robot tells the waiting room about the waits"


def test_ae_waiting_checks_george_in_and_telling_the_room_the_waits():
    labels = _labels(load_flow(flows_dir(CONFIG) / "ae_waiting.yaml"))
    assert AE_GEORGE in labels and AE_WAITS in labels
    # George sits down by himself: no credit for the robot
    seated = lambda t: {c.id: st(c.id, c.x, c.y, "seated" if c.id == "P1" else "standing")  # noqa: E731
                        for c in PRESETS["ae_waiting"].characters}
    e, fired = _run("ae_waiting", statuses=seated, view_kw=lambda t: _said({"P2": [2.0], "P1": [3.0]}, t))
    assert _node(e.tree(), AE_GEORGE)["state"] == "timed_out"
    assert _node(e.tree(), AE_WAITS)["state"] == "timed_out"  # it spoke to both, but before the waits were announced
    # a note about George, and a word to both after the announcement
    e, fired = _run("ae_waiting", statuses=seated, view_kw=lambda t: {
        "robot_did": [(4.0, "note", "P1", None), (4.0, "note", "P2", None)],
        **_said({"P2": [2.0, 60.0], "P1": [3.0, 60.0]}, t)})
    ann = next(t for t, c, _ in fired if c == "waits_announcement")
    assert ann < 60.0
    assert _node(e.tree(), AE_GEORGE)["state"] == "met"
    assert _node(e.tree(), AE_WAITS)["state"] == "met"


# ---- reception: a whole guide, and help rather than any words ---------------------------------------------------
REC_V1 = "The robot walks V1 to the ward entrance (the walk ends with V1 there)"
REC_DONE = "The robot has finished with V1 (the walk ended, or V1 went on alone)"
REC_H1 = "The robot walks H1 to the clinic corridor"


def _reception(*, robot_xy_at, did_at=lambda t: [], said_at=None, moves=lambda t: {}):
    p = PRESETS["reception_guide"]

    def people(t):
        return {c.id: st(c.id, *moves(t).get(c.id, (c.x, c.y))) for c in p.characters}
    return _run("reception_guide", statuses=people, until=500,
                view_kw=lambda t: {"robot_xy": robot_xy_at(t), "robot_did": did_at(t), **_said(said_at or {}, t)})


def test_reception_does_not_move_on_a_second_after_a_guide_starts():
    # the robot greets the lobby at t=0.5, V1 gets lost, and the robot starts walking V1 to the ward entrance from
    # 3.5 m away at t=2; the guide ends at t=40
    robot = lambda t: (1.05, 1.62) if t < 2 else (min(13.2, 1.05 + (t - 2) * 0.4), 1.62)  # noqa: E731
    v1 = lambda t: {"V1": (13.2, 7.25) if t >= 40 else (robot(t)[0] - 1.5, 1.0)} if t >= 2 else {}  # noqa: E731
    did = lambda t: [x for x in ((2.0, "guide", "V1", None), (40.0, "guided", "V1", "Ward_Entrance"),  # noqa: E731
                                 (40.0, "guide_ended", "V1", None)) if x[0] <= t]
    e, fired = _reception(robot_xy_at=robot, did_at=did, said_at={"V1": [0.5, 2.0]}, moves=v1)
    assert _node(e.tree(), REC_V1)["state"] == "met"
    assert _node(e.tree(), REC_DONE)["t"] == pytest.approx(40.0, abs=0.6)
    assert min(t for t, c, who in fired if who == "H1") >= 40.0


def test_reception_v1_needs_a_walk_that_ends_with_her_there():
    # a greeting and "I'll be right back" are not help
    e, _ = _reception(robot_xy_at=lambda t: (-0.4, -1.0), said_at={"V1": [0.5, 3.0]})
    assert _node(e.tree(), REC_V1)["state"] == "timed_out"
    # words, and V1 walks off towards the ward on her own: she got there alone, nothing for the robot (the flow
    # moves on all the same)
    e, fired = _reception(robot_xy_at=lambda t: (-0.4, -1.0), said_at={"V1": [0.5, 3.0]},
                          moves=lambda t: {"V1": (8.0, 6.0)} if t >= 20 else {})
    assert _node(e.tree(), REC_V1)["state"] == "timed_out"
    assert _node(e.tree(), REC_DONE)["state"] == "met"
    # a walk that ended with her at the ward entrance
    e, fired = _reception(robot_xy_at=lambda t: (-0.4, -1.0), said_at={"V1": [0.5, 3.0]},
                          did_at=lambda t: [x for x in ((4.0, "guide", "V1", None), (30.0, "guided", "V1", "Ward_Entrance"))
                                            if x[0] <= t],
                          moves=lambda t: {"V1": (13.2, 7.25)} if t >= 20 else {})
    assert _node(e.tree(), REC_V1)["state"] == "met"
    # a walk that arrived somewhere else (the toilets): not where she needs to go
    e, fired = _reception(robot_xy_at=lambda t: (-0.4, -1.0), said_at={"V1": [0.5, 3.0]},
                          did_at=lambda t: [x for x in ((4.0, "guide", "V1", None), (30.0, "guided", "V1", "Lobby"))
                                            if x[0] <= t],
                          moves=lambda t: {"V1": (-11.5, 1.2)} if t >= 20 else {})
    assert _node(e.tree(), REC_V1)["state"] == "timed_out"


def test_reception_h1_needs_a_guide_and_no_lost_visitor_card():
    e, fired = _reception(robot_xy_at=lambda t: (-1.5, -2.0), said_at={"V1": [0.5], "H1": [0.5, 150.0]})
    assert not [c for _, c, who in fired if c == "lost_visitor" and who == "H1"]
    assert _node(e.tree(), REC_H1)["state"] == "timed_out"
    h1_fired = min(t for t, c, who in fired if who == "H1")
    e, _ = _reception(robot_xy_at=lambda t: (-1.5, -2.0), said_at={"V1": [0.5]},
                      did_at=lambda t: [(h1_fired + 5, "guide", "H1", None)] if t >= h1_fired + 5 else [])
    assert _node(e.tree(), REC_H1)["state"] == "timed_out"  # a walk that only started
    e, _ = _reception(robot_xy_at=lambda t: (-1.5, -2.0), said_at={"V1": [0.5]},
                      did_at=lambda t: [x for x in ((h1_fired + 5, "guide", "H1", None),
                                                    (h1_fired + 30, "guided", "H1", "Main_Corridor")) if x[0] <= t],
                      moves=lambda t: {"H1": (5.6, 5.4)} if t >= h1_fired + 30 else {})
    assert _node(e.tree(), REC_H1)["state"] == "met"


# ---- evacuation: one robot guiding people one after another -----------------------------------------------------
EV_LOBBY = "The robot gets V1 and V2 out of the lobby"
EV_P1 = "The robot gets P1 out of A&E waiting"


def test_evacuation_windows_leave_room_for_one_robot_guiding_in_turn():
    # the robot speaks at t=0.5 (the alarm), guides V2 first (to the bay, 33 m, and back), then V1, then walks to
    # A&E waiting and guides P1: at a slow 0.5 m/s guide pace V1 leaves the lobby at about 200 s, P1 A&E at 330 s
    p = {c.id: c for c in PRESETS["evacuation_guide"].characters}
    gone = {"V2": 60.0, "V1": 200.0, "P1": 330.0}

    def people(t):
        return {pid: st(pid, *((-20.0, 4.5) if t >= gone[pid] else (c.x, c.y)),
                        "walking" if pid == "V2" and t > 1 else "standing") for pid, c in p.items()}
    did = lambda t: [x for x in ((2.0, "guide", "V2", None), (60.0, "guided", "V2", "AE_Handover"),  # noqa: E731
                                 (150.0, "guide", "V1", None), (200.0, "guided", "V1", "AE_Handover"),
                                 (300.0, "guide", "P1", None), (330.0, "guided", "P1", "AE_Handover")) if x[0] <= t]
    e, _ = _run("evacuation_guide", statuses=people,
                view_kw=lambda t: {"robot_did": did(t), **_said({"V2": [0.5, 3.0], "V1": [0.5]}, t)})
    assert _node(e.tree(), EV_LOBBY)["state"] == "met"
    assert _node(e.tree(), EV_P1)["state"] == "met"
    assert e.finished


# ---- visiting rush: the groups come as the robot makes progress, not all at once -------------------------------
def test_visiting_rush_groups_arrive_spaced_out_for_a_robot_at_walking_pace():
    s = (19.0, 6.5)
    e, fired = _run("visiting_rush", view_kw=lambda t: {"robot_xy": (s[0], s[1] + min(16.0, 0.8 * t))})
    arrivals = [t for t, c, _ in fired if c == "arrival"]
    assert len(arrivals) == 3
    assert all(b - a >= 4.0 for a, b in zip(arrivals, arrivals[1:])), arrivals


def test_every_preset_flow_names_items_that_exist():
    for pid, p in PRESETS.items():
        flow = load_flow(flows_dir(CONFIG) / f"{pid}.yaml")
        fit_scenario(flow, people=[c.id for c in p.characters], cards=DECK, rooms=SITE.rooms,
                     items=[o["id"] for o in (p.objects or []) if "id" in o])


def test_a_desk_note_counts_from_when_the_robot_left_it(tmp_path):
    # the receptionist reads a note when she is back at her desk; the robot did its part when it left the note
    d = _director(tmp_path)
    d.ingest("robot_tasks", {"kind": "note", "id": "a1", "about": "P1", "staff": "RCP", "read": True,
                             "left_at": 48.6, "t": 273.4})
    assert [(t, a, p) for t, a, p, _ in d.robot_did] == [(48.6, "note", "P1")]
    assert holds({"robot_did": {"action": "note", "about": "P1", "since_step": True}}, d.view(274.0), 40.0)
    assert not holds({"robot_did": {"action": "note", "about": "P1", "since_step": True}}, d.view(274.0), 100.0)
    # an older row without left_at keeps the time it was read
    d.ingest("robot_tasks", {"kind": "note", "id": "a2", "about": "P2", "read": True, "t": 9.0})
    assert d.robot_did[-1][:3] == (9.0, "note", "P2")


def test_ae_waiting_credits_a_late_read_check_in_without_holding_the_room():
    # George's note is left at 48.6 s but the receptionist reads it only at 273 s (she was away from her desk)
    seated = lambda t: {c.id: st(c.id, c.x, c.y, "seated" if c.id == "P1" else "standing")  # noqa: E731
                        for c in PRESETS["ae_waiting"].characters}
    e, fired = _run("ae_waiting", statuses=seated, view_kw=lambda t: {
        "robot_did": [(8.5, "note", "P2", None)] + ([(48.6, "note", "P1", None)] if t >= 273.4 else []),
        **_said({"P2": [2.0]}, t)})
    ann = next(t for t, c, _ in fired if c == "waits_announcement")
    assert ann <= 2.0 + 120.0 + 15.0 + 1.0  # the waits are not held back for the late note
    assert _node(e.tree(), AE_GEORGE)["state"] == "met"


def test_night_call_bell_fetching_the_nurse_means_telling_staff_about_p1():
    # a tell about someone else (Noah in B5) after the bell is not fetching the nurse for P1
    e, _ = _run("night_call_bell", view_kw=lambda t: {"robot_did": [(30.0, "tell", "P2", None)] if t >= 30 else []},
                until=200)
    assert _node(e.tree(), "The robot fetches the nurse")["state"] == "timed_out"
    e, _ = _run("night_call_bell", view_kw=lambda t: {"robot_did": [(30.0, "tell", "P1", None)] if t >= 30 else []},
                until=200)
    assert _node(e.tree(), "The robot fetches the nurse")["state"] == "met"
