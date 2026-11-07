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
    start = (w.robot.x, w.robot.y)
    s.on_task_action({"id": "k1", "action": "take", "item": "scalpel"})  # the tray is beside it, on her side
    s.on_task_action({"id": "a1", "action": "approach", "track": track(s, "S1")})
    outs = advance(s, 0.2)
    assert [(r["ok"], r["reason"]) for r in results(outs, "k1")] == [(True, "ok")]
    assert [(r["ok"], r["reason"]) for r in results(outs, "a1")] == [(True, "already_there")]
    assert math.hypot(w.robot.x - start[0], w.robot.y - start[1]) < 0.05, "no walk round the operating table"
    q = w.people["S1"]
    fwd = (w.robot.x - q.x) * math.cos(q.yaw) + (w.robot.y - q.y) * math.sin(q.yaw)
    assert fwd > -0.2, "not behind her"
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "scalpel", "to": track(s, "S1")})
    w.apply_intent(Intent(id="y", person="S1", t=s.t, kind="accept_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "h1")] == [(True, "ok")]
    assert w.objects.items["scalpel"].holder == "S1"


# ---------- fire ----------

def _near_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy))


def _clear_of(path, p, r):
    return all(_near_segment(p, a, b) >= r for a, b in zip(path, path[1:]))


def test_routes_keep_clear_of_a_hazard_disc_and_never_end_in_one():
    from wardzero_agents.site import Router
    from wardzero_agents.site.model import load_site
    r = Router(load_site(CONFIG).grid)
    fire = (19.0, 14.5)
    assert not _clear_of(r.route((19.0, 9.0), (19.0, 20.0)), fire, 0.9)  # the corridor's straight way
    r.set_hazards([(fire[0], fire[1], 1.0)])
    path = r.route((19.0, 9.0), (19.0, 20.0))
    assert path is not None and _clear_of(path, fire, 0.9)
    assert r.route((19.0, 9.0), (19.2, 14.6)) is None  # into the fire
    assert r.route((19.3, 14.5), (19.0, 9.0)) is not None  # out of it
    r.set_hazards([])
    assert not _clear_of(r.route((19.0, 9.0), (19.0, 20.0)), fire, 0.9)


def test_a_fire_keeps_people_and_the_robot_away_until_it_is_out(staged, tmp_path):
    s = sim(staged, "ward_fire", tmp_path)
    w = s.world
    fire = (19.0, 14.5)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=fire[0], y=fire[1]))
    w.apply_intent(Intent(id="n", person="N1", t=0.0, kind="walk_to", point=[19.0, 9.5]))  # N1 is at (19, 20)
    n1 = w.people["N1"]
    assert n1.state == "running" and _clear_of([(n1.x, n1.y)] + list(n1.plan[-1].path), fire, 0.9)
    s.tasks.goal_pose(19.0, 20.5, 0.0, 0.0)  # the robot, at (19, 9), to the far side
    assert _clear_of([(w.robot.x, w.robot.y)] + list(w.robot._path), fire, 0.9)
    w.objects.fires["fire_1"].reduce(1.0)
    advance(s, 0.2)
    assert w.router.route((19.0, 9.0), (19.2, 14.6)) is not None


def test_extinguish_out_of_range_walks_up_to_the_fire_then_puts_it_out(staged, tmp_path):
    s = sim(staged, "ward_fire", tmp_path)
    w = s.world
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=19.0, y=14.5))
    s.on_task_action({"id": "x1", "action": "extinguish", "fire": "fire_1"})  # from (19, 9): 5.5 m away
    outs = advance(s, 40.0)
    res = results(outs, "x1")
    assert [(r.get("kind", "action"), r["ok"], r["reason"]) for r in res if r.get("kind") != "goal"] == [
        ("action", True, "ok")]
    assert w.objects.fires["fire_1"].out
    assert 1.0 <= math.hypot(w.robot.x - 19.0, w.robot.y - 14.5) <= 2.0


def test_a_fire_that_starts_mid_walk_sends_walkers_round_it(staged, tmp_path):
    s = sim(staged, "ward_fire", tmp_path)
    w = s.world
    fire = (19.0, 14.5)
    w.apply_intent(Intent(id="n", person="N1", t=0.0, kind="walk_to", point=[19.0, 9.5]))
    s.tasks.goal_pose(19.0, 20.5, 0.0, 0.0)
    advance(s, 0.5)
    w.apply_world_event(WorldEvent(id="f", t=s.t, kind="fire_start", x=fire[0], y=fire[1]))
    advance(s, 0.2)
    n1 = w.people["N1"]
    assert n1.state == "running" and _clear_of([(n1.x, n1.y)] + list(n1.plan[-1].path[n1.plan[-1].i:]), fire, 0.9)
    assert _clear_of([(w.robot.x, w.robot.y)] + list(w.robot._path), fire, 0.9)
    seen = []
    for o in advance(s, 30.0):
        seen.append(min(math.hypot(n1.x - fire[0], n1.y - fire[1]), math.hypot(w.robot.x - fire[0], w.robot.y - fire[1])))
    assert min(seen) >= 0.85


# ---------- arrivals ----------

STAFF = {"staff", "ward_nurse", "receptionist", "surgeon", "porter", "companion", "custom"}


def test_ward_presets_bring_arrivals_in_at_their_own_entrance(staged):
    # visiting_rush: the groups come down the ward corridor towards the robot (QA: arriving behind it, they never met it)
    assert PRESETS["visiting_rush"].arrival_entrance == "Ward_Corridor_centre"
    assert PRESETS["specimen_run"].arrival_entrance == "ward_entrance"
    assert PRESETS["reception_guide"].arrival_entrance is None
    assert staged("visiting_rush").scenario.arrival_entrance == "Ward_Corridor_centre"
    assert staged("reception_guide").scenario.arrival_entrance is None


def test_an_unknown_arrival_entrance_is_a_config_error(tmp_path):
    from wardzero_agents.config.loader import ConfigError
    with pytest.raises(ConfigError, match="arrival_entrance"):
        preset_to_scenario(PRESETS["visiting_rush"], CONFIG, tmp_path / "s",
                           scenario_extra={"arrival_entrance": "narnia"})


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_arrivals_are_only_visitors_and_patients_up_and_about(staged, preset_id):
    loaded = staged(preset_id)
    in_bed = {i.persona_id for i in loaded.instances if i.start is not None and i.start.bed is not None}
    for k in range(12):
        ev = next(o for o in fire_card(loaded, "arrival", t=10.0 + k) if isinstance(o, WorldEvent))
        if ev.persona is None:  # nobody fit to arrive in this scene
            continue
        p = loaded.personas[ev.persona]
        assert p.role not in STAFF and ev.persona not in in_bed, (preset_id, ev.persona, p.role)


def test_meal_round_has_nobody_to_arrive(staged):
    ev = next(o for o in fire_card(staged("meal_round"), "arrival") if isinstance(o, WorldEvent))
    assert ev.persona is None


def test_arrivals_spawn_at_the_scenario_entrance_spread_out_on_free_floor(staged):
    w = world(staged, "visiting_rush")
    lm = w.site.landmarks[PRESETS["visiting_rush"].arrival_entrance]
    persona = next(pid for pid, p in w.loaded.personas.items() if p.role == "ward_visitor")
    ids = [w.spawn(persona, 1.0) for _ in range(4)]
    pts = [(w.people[i].x, w.people[i].y) for i in ids]
    for x, y in pts:
        assert math.hypot(x - lm.x, y - lm.y) <= 2.0 and w.router.is_free(x, y)
    assert min(math.hypot(a[0] - b[0], a[1] - b[1]) for k, a in enumerate(pts) for b in pts[k + 1:]) >= 0.5


def test_an_arrival_with_no_persona_fit_to_arrive_is_ignored(staged):
    w = world(staged, "meal_round")
    n = len(w.people)
    w.apply_world_event(WorldEvent(id="a", t=1.0, kind="arrival"))
    assert len(w.people) == n


# ---------- getting out of bed ----------

@pytest.mark.parametrize("kind,target", [("walk_to", "Ward_Entrance"), ("stand_up", None)])
def test_a_patient_getting_out_of_bed_never_stands_inside_it(staged, kind, target):
    w = world(staged, "wandering_patient")
    w.apply_intent(Intent(id="g", person="W1", t=0.0, kind=kind, target=target))
    w1 = w.people["W1"]
    for k in range(1, 200):
        w.step(k * 0.1)
        if w1.posture in ("standing", "walking"):
            assert not w.site.grid.blocked(w1.x, w1.y), (k, w1.x, w1.y)
    assert w1.posture in ("standing", "walking") and w1.bed is None
