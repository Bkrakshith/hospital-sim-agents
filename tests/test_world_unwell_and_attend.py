"""World cues the robot needs and staff who reach the patient (second fix round).

* A patient who gets worse looks unwell: slumped, a track the robot's camera marks unwell within 4 m, and a quiet
  cough every 20 to 40 s of sim time from the world itself (not from the person's model).
* Staff the robot sends to a patient stand within 1.5 m of them even when the robot stands still at the bedside
  (they may come to 0.8 m of a still robot), and the bed's call light goes off when they get there.
"""
import math
import typing
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, Roster, RosterEntry, Status, WorldEvent
from wardzero_agents.director.cards import CardExecutor
from wardzero_agents.director.storyboard import CardFiring, WorldView
from wardzero_agents.world.perception import bbox_3d, perceive_people
from wardzero_agents.world.stub_world import StubWorld, _Phase

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            cache[preset_id] = preset_to_scenario(PRESETS[preset_id], CONFIG, tmp_path_factory.mktemp(preset_id))
        return cache[preset_id]
    return get


def _kinds(model, field):
    return typing.get_args(model.model_fields[field].annotation)


def fire_card(loaded, card, target=None, t=10.0):
    ex = CardExecutor(loaded.deck, {i.person_id: i for i in loaded.instances}, loaded.personas, loaded.scenario.seed)
    present = {i.person_id for i in loaded.instances}
    roster = Roster(t=t, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                 for i in loaded.instances])
    view = WorldView(t=t, statuses={}, robot_xy=(0.0, 0.0), present=present)
    return ex.execute(CardFiring(id=f"{card}-1", t=t, card=card, target=target, params={}, source="storyboard"),
                      view, roster)


def _robot_facing(w, pid, dist):
    """Put the robot dist metres from pid, on free floor, facing them."""
    p = w.people[pid]
    for k in range(36):
        a = 2 * math.pi * k / 36
        x, y = p.x + dist * math.cos(a), p.y + dist * math.sin(a)
        if w.router.is_free(x, y) and w._los((x, y), (p.x, p.y)):
            w.robot.teleport(x, y, math.atan2(p.y - y, p.x - x))
            return
    raise AssertionError("no free spot")


def _track_of(w, pid):
    tid = w.tracks.track(pid)
    return next(d for d in perceive_people(w, w.robot) if d["track"] == tid)


def _det_of(w, pid, t):
    tid = w.tracks.track(pid)
    return next(d for d in bbox_3d(w, w.robot, t)["detections"] if d["id"] == tid)


# ---------- looks unwell (contract 14) ----------

def test_deterioration_makes_george_look_unwell_and_cough_on_his_own(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    _robot_facing(w, "P1", 3.0)
    assert not _track_of(w, "P1").get("unwell")
    w.set_unwell("P1", 10.0)
    w.step(10.1)
    assert _track_of(w, "P1").get("unwell") is True
    assert _det_of(w, "P1", 10.1).get("unwell") is True
    coughs, t = [], 10.1
    while t < 50.0:  # the people's minds are not running: any cough is the world's own
        t = round(t + 0.1, 3)
        w.step(t)
        coughs += [e for e in w.drain_audio() if e.source_kind == "cough" and e.source_id == "P1"]
    assert coughs and coughs[0].t <= 50.0
    assert w.robot_heard(coughs[0]) is not None, "the robot 3 m away hears it"
    gaps = [b.t - a.t for a, b in zip(coughs, coughs[1:])]
    assert all(20.0 - 1e-6 <= g <= 40.0 + 1e-6 for g in gaps)


def test_unwell_is_seen_only_within_4_m(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    w.set_unwell("P1", 0.0)
    _robot_facing(w, "P1", 5.0)
    assert not _track_of(w, "P1").get("unwell")
    assert not _det_of(w, "P1", 0.0).get("unwell")


def test_unwell_coughs_are_quiet_and_regular(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    w.set_unwell("P1", 0.0)
    out, t = [], 0.0
    while t < 200.0:
        t = round(t + 0.5, 3)
        w.step(t)
        out += [e for e in w.drain_audio() if e.source_kind == "cough"]
    assert 5 <= len(out) <= 10
    assert all(e.loudness_db <= 62 for e in out)


def test_unwell_status(staged):
    """The world's side once it knows (set_unwell): the status carries unwell and posture slumped where the contract
    has them; the world's own mechanics keep the chair."""
    w = StubWorld(staged("ae_waiting"), 1)
    w.set_unwell("P1", 10.0)
    assert w.people["P1"].unwell
    st = {s.person: s for s in w.statuses()}["P1"]
    if "unwell" in Status.model_fields:
        assert st.unwell is True
    if "slumped" in _kinds(Status, "posture"):
        assert st.posture == "slumped"
    assert w.people["P1"].posture == "seated", "still in the chair as far as the world's mechanics go"
    rows = [r for s, r in w.drain_log() if s == "world" and r.get("kind") == "unwell"]
    assert rows and rows[0]["person"] == "P1"


_NO_UNWELL_EVENT = "unwell" not in _kinds(WorldEvent, "kind")


@pytest.mark.xfail(_NO_UNWELL_EVENT, strict=True,
                   reason="contracts.WorldEvent.kind has no 'unwell' yet: the card's world effect cannot reach the sim")
def test_the_fired_card_reaches_the_world(staged):
    """The deterioration card as a run delivers it: the director's outputs go to the sim the way the sim node takes
    them (world events only; the sim does not get a person's nudge), never by calling the world's setters."""
    loaded = staged("ae_waiting")
    w = StubWorld(loaded, 1)
    _robot_facing(w, "P1", 3.0)
    for o in fire_card(loaded, "deterioration", target="P1", t=10.0):
        if isinstance(o, WorldEvent):
            w.apply_world_event(o)
    w.step(10.1)
    assert w.people["P1"].unwell
    assert _track_of(w, "P1").get("unwell") is True
    coughs, t = [], 10.1
    while t < 50.0:
        t = round(t + 0.1, 3)
        w.step(t)
        coughs += [e for e in w.drain_audio() if e.source_kind == "cough" and e.source_id == "P1"]
    assert coughs and w.robot_heard(coughs[0]) is not None


def test_the_deterioration_card_has_the_unwell_world_effect():
    import yaml
    deck = {c["id"]: c for c in yaml.safe_load((CONFIG / "cards" / "studio.yaml").read_text())["cards"]}
    card = deck["deterioration"]
    assert "world" in card["channels"] and card["world"]["kind"] == "unwell"


# ---------- attending staff (contract 16) ----------

LAYOUTS = [  # the re-QA's oracle runs: where the robot stood at the bedside, still, and the patient it called for
    ("night_call_bell", (23.993, 19.098, 0.0027), "P1"),
    ("meal_round", (23.99, 16.204, -0.0034), "M3"),
    # where the scripted robot of the in-process e2e stood still at the bedside: on the near side, in the one gap of
    # free floor beside the bed, so the nurse has to go round to the bed's other side to be within 1.5 m
    ("night_call_bell", (23.74, 19.44, 0.0), "P1"),
    ("meal_round", (24.03, 16.12, 0.0), "M3"),
]


def _walk_until_done(w, pid, t0=0.0, limit=60.0):
    t = t0
    while t < t0 + limit and w.people[pid].state == "running":
        t = round(t + 0.1, 3)
        w.step(t)
    return t


@pytest.mark.parametrize("preset_id,robot,patient", LAYOUTS)
def test_a_nurse_called_to_the_patient_reaches_them_past_a_still_robot(staged, preset_id, robot, patient):
    w = StubWorld(staged(preset_id), 1)
    w.robot.teleport(*robot)
    w.apply_intent(Intent(id="a", person="N1", t=0.0, kind="approach", target=patient))
    _walk_until_done(w, "N1")
    n, p = w.people["N1"], w.people[patient]
    assert n.state == "done"
    assert math.hypot(n.x - p.x, n.y - p.y) <= 1.5
    assert math.hypot(n.x - w.robot.x, n.y - w.robot.y) >= 0.8 - 1e-6


@pytest.mark.parametrize("preset_id,robot,patient", LAYOUTS)
def test_attend_also_reaches_the_patient(staged, preset_id, robot, patient):
    w = StubWorld(staged(preset_id), 1)
    w.robot.teleport(*robot)
    w.apply_intent(Intent(id="a", person="N1", t=0.0, kind="attend", target=patient))
    _walk_until_done(w, "N1")
    n, p = w.people["N1"], w.people[patient]
    assert math.hypot(n.x - p.x, n.y - p.y) <= 1.5


def test_a_visitor_still_keeps_1_2_m_off_a_still_robot(staged):
    w = StubWorld(staged("night_call_bell"), 1)
    w.robot.teleport(23.993, 19.098, 0.0027)
    n = w.people["N1"]
    n.inst = n.inst.model_copy(update={"role": "ward_visitor"})
    w.apply_intent(Intent(id="a", person="N1", t=0.0, kind="approach", target="P1"))
    _walk_until_done(w, "N1")
    assert math.hypot(n.x - w.robot.x, n.y - w.robot.y) >= 1.2 - 1e-6


def test_staff_keep_1_2_m_off_a_robot_on_the_move(staged):
    w = StubWorld(staged("night_call_bell"), 1)
    w.robot.teleport(23.993, 19.098, 0.0027)
    w.robot.vx = 0.5  # walking
    goal = w._walk_dest(w.people["N1"], _Phase("walk", follow="P1"))
    assert math.hypot(goal[0] - w.robot.x, goal[1] - w.robot.y) >= 1.2 - 1e-6


@pytest.mark.parametrize("preset_id,robot,patient", LAYOUTS)
def test_the_call_light_goes_off_when_the_nurse_arrives(staged, preset_id, robot, patient):
    w = StubWorld(staged(preset_id), 1)
    w.robot.teleport(*robot)
    w.apply_world_event(WorldEvent(id="r", t=0.0, kind="ring_bell", person=patient))
    bed = w.people[patient].bed
    assert w.objects.bells[bed].active
    w.step(1.0)
    assert w.objects.bells[bed].active, "nobody there yet"
    w.apply_intent(Intent(id="a", person="N1", t=1.0, kind="approach", target=patient))
    _walk_until_done(w, "N1", t0=1.0)
    n, p = w.people["N1"], w.people[patient]
    assert n.state == "done" and math.hypot(n.x - p.x, n.y - p.y) <= 1.5
    w.step(w.last_t + 0.1)
    assert not w.objects.bells[bed].active
    snaps = [r for s, r in w.drain_log() if s == "objects"]
    last = {o["id"]: o for o in snaps[-1]["objects"]}
    assert last[f"bell_{bed}"]["active"] is False


@pytest.mark.parametrize("preset_id,robot,patient", LAYOUTS)
def test_the_nurse_settles_on_one_spot(staged, preset_id, robot, patient):
    """The spot she walks to is kept while the patient stays put: she does not hop between spots as she moves."""
    w = StubWorld(staged(preset_id), 1)
    w.robot.teleport(*robot)
    w.apply_intent(Intent(id="a", person="N1", t=0.0, kind="approach", target=patient))
    goals, t = set(), 0.0
    while t < 60.0 and w.people["N1"].state == "running":
        t = round(t + 0.1, 3)
        w.step(t)
        g = w.walk_goal("N1")
        if g is not None:
            goals.add((round(g[0], 2), round(g[1], 2)))
    assert w.people["N1"].state == "done"
    assert len(goals) <= 1, goals


def test_a_call_light_stays_on_while_only_the_robot_is_there(staged):
    w = StubWorld(staged("night_call_bell"), 1)
    w.robot.teleport(23.993, 19.098, 0.0027)
    w.apply_world_event(WorldEvent(id="r", t=0.0, kind="ring_bell", person="P1"))
    for k in range(1, 50):
        w.step(k * 0.2)
    assert w.objects.bells["B2"].active


@pytest.mark.parametrize("preset_id,robot,patient", LAYOUTS)
def test_an_isaac_body_route_keeps_the_attending_spot(staged, preset_id, robot, patient):
    """The Isaac host walks a character to walk_goal by route_for, which moves a goal off the robot: for a nurse going
    to the patient it must not push her back out past 1.5 m."""
    w = StubWorld(staged(preset_id), 1)
    w.robot.teleport(*robot)
    w.body_driven.add("N1")
    w.apply_intent(Intent(id="a", person="N1", t=0.0, kind="approach", target=patient))
    goal = w.walk_goal("N1")
    p = w.people[patient]
    assert math.hypot(goal[0] - p.x, goal[1] - p.y) <= 1.5
    path = w.route_for("N1", goal)
    end = path[-1] if path else goal
    assert math.hypot(end[0] - p.x, end[1] - p.y) <= 1.5
    assert math.hypot(end[0] - w.robot.x, end[1] - w.robot.y) >= 0.8 - 1e-6
