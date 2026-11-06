"""Scenario mechanics the presets depend on: the call bell, hand-overs, fire and arrivals (Quick 2D world)."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, Roster, RosterEntry, WorldEvent
from wardzero_agents.director.cards import CardExecutor
from wardzero_agents.director.storyboard import CardFiring, WorldView
from wardzero_agents.world.perception import perceive_objects
from wardzero_agents.world.stub_world import StubWorld

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


def world(staged, preset_id, seed=1):
    return StubWorld(staged(preset_id), seed)


def fire_card(loaded, card, target=None, t=10.0, params=None):
    ex = CardExecutor(loaded.deck, {i.person_id: i for i in loaded.instances}, loaded.personas, loaded.scenario.seed)
    present = {i.person_id for i in loaded.instances}
    roster = Roster(t=t, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                 for i in loaded.instances])
    view = WorldView(t=t, statuses={}, robot_xy=(0.0, 0.0), present=present)
    return ex.execute(CardFiring(id=f"{card}-1", t=t, card=card, target=target, params=params or {},
                                 source="storyboard"), view, roster)


# ---------- the call bell ----------

def test_the_robot_hears_a_bell_rung_down_the_ward_corridor(staged):
    w = world(staged, "night_call_bell")  # the robot stands in the corridor, 7 m from bed B2
    events = w.apply_intent(Intent(id="b", person="P1", t=1.0, kind="call_bell"))
    bed = w.site.beds["B2"]
    assert any(math.hypot(e.x - bed.x, e.y - bed.y) < 0.5 for e in events), "the chime sounds at the bed"
    heard = [h for h in (w.robot_heard(e) for e in events) if h is not None]
    assert heard and any(not h.partial for h in heard)
    w.robot_pose = type(w.robot_pose)(19.0, 26.0, 0.0)  # the far end of the corridor
    assert any(w.robot_heard(e) is not None for e in events)


def test_a_lit_call_light_is_seen_from_the_corridor_outside_the_room(staged):
    w = world(staged, "night_call_bell")
    w.objects.ring("B2", 0.0)
    door = (20.83, 19.5)
    x, y = 19.5, 17.0  # in the corridor: the bed itself is behind the bedroom wall from here
    assert not w._los((x, y), (w.site.beds["B2"].x, w.site.beds["B2"].y))
    w.robot.teleport(x, y, math.atan2(door[1] - y, door[0] - x))
    lights = perceive_objects(w, w.robot)["call_lights"]
    assert [c["bed"] for c in lights] == ["B2"]


def test_the_ring_bell_card_rings_the_patients_bell(staged):
    loaded = staged("night_call_bell")
    outs = fire_card(loaded, "ring_bell", target="P1")
    ev = next(o for o in outs if isinstance(o, WorldEvent))
    assert ev.kind == "ring_bell" and ev.person == "P1"
    w = StubWorld(loaded, 1)
    audio = w.apply_world_event(ev)
    assert w.objects.bells["B2"].active and audio
    assert any(w.robot_heard(e) is not None for e in audio)


def test_ring_bell_is_ignored_for_someone_not_in_bed(staged):
    w = world(staged, "night_call_bell")
    assert w.apply_world_event(WorldEvent(id="r", t=1.0, kind="ring_bell", person="N1")) == []
    assert not any(b.active for b in w.objects.bells.values())


def test_the_robot_is_told_it_heard_a_call_bell():
    from wardzero_agents.robots.claude_robot import RobotMind
    from wardzero_agents.site.model import load_site
    import asyncio
    mind = RobotMind(load_site(CONFIG), {"job": "x"}, None, lambda: 0.0)
    asyncio.run(mind.on_heard({"source_kind": "person", "source_id": "unknown", "text": None,
                               "received_db": 40.0, "partial": False, "addressed_to_me": False}, 1.0))
    assert mind._speech  # a chime is worth a look at once
    assert "call bell" in mind._situation(2.0)


def test_people_hear_a_call_bell_as_a_bell():
    from wardzero_agents.contracts import Heard
    from wardzero_agents.mind.prompts import _heard_line
    h = Heard(source_kind="person", source_id="bell_B2", text=None, received_db=40.0, partial=False,
              addressed_to_me=False)
    assert "call bell" in _heard_line(h) and "B2" in _heard_line(h)


# ---------- hand-overs ----------

def sim(staged, preset_id, tmp_path):
    from wardzero_agents.ros.stub_sim_node import StubSimService
    from wardzero_agents.runlog import RunLog
    return StubSimService(staged(preset_id), RunLog(tmp_path / "run"), wait_for=())


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


def results(outs, aid):
    return [r for o in outs for r in o.task_results if r["id"] == aid]


def track(s, pid):
    return s.world.tracks.track(pid)


def test_approaching_a_patient_in_bed_ends_at_the_bed_side(staged, tmp_path):
    s = sim(staged, "meal_round", tmp_path)
    w = s.world
    w.robot.teleport(20.0, 16.0, 0.0)  # in the corridor outside bedroom B3
    s.on_task_action({"id": "a1", "action": "approach", "track": track(s, "M3")})
    outs = advance(s, 20.0)
    goals = [r for o in outs for r in o.task_results if r.get("kind") == "goal" and r["id"] == "a1"]
    assert goals and goals[0]["reason"] == "arrived"
    b = w.site.beds["B3"]
    along = (w.robot.x - b.x) * math.cos(b.yaw) + (w.robot.y - b.y) * math.sin(b.yaw)
    across = -(w.robot.x - b.x) * math.sin(b.yaw) + (w.robot.y - b.y) * math.cos(b.yaw)
    assert abs(along) < 0.3 and 0.8 <= abs(across) <= 1.2  # beside the bed's middle, not off its end


def test_a_tray_reaches_a_patient_who_sits_up_when_offered(staged, tmp_path):
    s = sim(staged, "meal_round", tmp_path)
    w = s.world
    w.robot.teleport(20.0, 16.0, 0.0)
    s.on_task_action({"id": "a1", "action": "approach", "track": track(s, "M3")})
    advance(s, 20.0)
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "tray_M3", "to": track(s, "M3")})
    outs = advance(s, 0.5)
    assert not results(outs, "h1"), "the offer stays open while she sits up"
    nudges = [n for o in outs for n in o.nudges if n.person == "M3"]
    assert nudges and "sit_up_in_bed" in nudges[0].message
    w.apply_intent(Intent(id="u", person="M3", t=s.t, kind="sit_up_in_bed"))
    w.apply_intent(Intent(id="y", person="M3", t=s.t + 0.1, kind="accept_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "h1")] == [(True, "ok")]
    assert w.objects.items["tray_M3"].holder == "M3"


def test_a_hand_over_from_too_far_fails_with_a_clear_reason(staged, tmp_path):
    s = sim(staged, "meal_round", tmp_path)
    s.world.robot.teleport(20.0, 16.0, 0.0)
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "tray_M3", "to": track(s, "M3")})
    outs = advance(s, 0.2)
    assert [(r["ok"], r["reason"]) for r in results(outs, "h1")] == [(False, "out_of_reach_giver")]


def test_the_robot_approaches_the_surgeon_from_beside_her_and_can_pass_an_instrument(staged, tmp_path):
    s = sim(staged, "scrub_nurse", tmp_path)
    w = s.world
    sc = w.objects.items["scalpel"]
    sc.holder = "g1"
    s.on_task_action({"id": "a1", "action": "approach", "track": track(s, "S1")})
    outs = advance(s, 40.0)  # round the operating table, out of the theatre and back in
    goals = [r for o in outs for r in o.task_results if r.get("kind") == "goal" and r["id"] == "a1"]
    assert goals and goals[0]["reason"] == "arrived"
    q = w.people["S1"]
    fwd = (w.robot.x - q.x) * math.cos(q.yaw) + (w.robot.y - q.y) * math.sin(q.yaw)
    assert fwd > -0.2, "not behind her"
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "scalpel", "to": track(s, "S1")})
    w.apply_intent(Intent(id="y", person="S1", t=s.t, kind="accept_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "h1")] == [(True, "ok")]
