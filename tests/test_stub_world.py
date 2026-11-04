import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Intent, WorldEvent
from wardzero_agents.world.audio import route
from wardzero_agents.world.oracle import Pose
from wardzero_agents.world.stub_world import StubWorld

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def make(seed=7):
    return StubWorld(load_scenario(V1), seed)


def step_until(w, T, dt=0.1):
    n = int(round(T / dt))
    for i in range(1, n + 1):
        w.step(i * dt)


def st(w, pid):
    return {s.person: s for s in w.statuses()}[pid]


def inst(w, pid):
    return w.people[pid].inst


def I(kind, person="P5", id="i", **kw):
    return Intent(id=id, person=person, t=0, kind=kind, **kw)


def test_initial_seated():
    w = make()
    assert {s.person: s.posture for s in w.statuses()}["P1"] == "seated"
    assert st(w, "P5").x == pytest.approx(-5.103)


def test_walk_speed_after_stand_up():
    w = make()
    s0 = st(w, "P5")
    w.apply_intent(I("walk_to", target="clinic_corridor"))
    step_until(w, 5.0)
    s1 = st(w, "P5")
    moved = math.hypot(s1.x - s0.x, s1.y - s0.y)
    i = inst(w, "P5")
    assert moved == pytest.approx((5.0 - i.sit_to_stand_s) * i.walk_speed, rel=0.05)
    assert s1.posture == "walking" and s1.state == "running"
    assert s1.intent_id == "i" and 0 < s1.progress < 1


def test_walk_to_arrives_at_landmark_and_seat():
    w = make()
    w.apply_intent(I("walk_to", target="reception_front"))
    step_until(w, 60.0)
    s = st(w, "P5")
    assert s.state == "done" and s.posture == "standing"
    assert (s.x, s.y) == pytest.approx((1.0, 0.57))
    w.apply_intent(I("walk_to", target="sign_wards", id="j"))
    step_until(w, 200.0)
    assert (st(w, "P5").x, st(w, "P5").y) == pytest.approx((10.5, 6.0))


def test_approach_stops_short_of_robot():
    w = make()
    w.apply_intent(I("approach", target="g1"))
    step_until(w, 60.0)
    s = st(w, "P5")
    r = w.loaded.scenario.robot
    assert math.hypot(s.x - r.x, s.y - r.y) == pytest.approx(1.0, abs=0.1)
    assert s.state == "done"
    assert math.cos(s.yaw - math.atan2(r.y - s.y, r.x - s.x)) > 0.99


def test_approach_person_reaims_moving_target():
    w = make()
    w.apply_intent(I("walk_to", person="P4", target="reception_front", id="a"))
    w.apply_intent(I("approach", person="P5", target="P4", id="b"))
    t = 0.0
    while st(w, "P5").state == "running" and t < 120:
        t += 0.1
        w.step(t)
    a, b = st(w, "P4"), st(w, "P5")
    assert b.state == "done" and t < 120
    assert math.hypot(a.x - b.x, a.y - b.y) == pytest.approx(1.0, abs=0.1)


def test_seat_occupied_fails():
    w = make()
    w.apply_intent(I("sit_down", target="B2"))
    s = st(w, "P5")
    assert s.state == "failed" and s.reason == "seat occupied"


def test_seat_claimed_by_walker_is_occupied():
    w = make()
    w.apply_intent(I("walk_to", person="P1", target="reception_front", id="a"))
    step_until(w, 30.0)
    w.apply_intent(I("sit_down", person="P1", target="A1", id="b"))
    w.apply_intent(I("sit_down", target="A1", id="c"))
    assert st(w, "P5").reason == "seat occupied"


def test_sit_down_and_stand_up():
    w = make()
    w.apply_intent(I("sit_down", person="P1", target="A1", id="a"))  # P1 leaves B2
    step_until(w, 60.0)
    s = st(w, "P1")
    assert s.state == "done" and s.posture == "seated"
    assert (s.x, s.y) == pytest.approx((-8.356, 0.222))
    # B2 is free now
    w.apply_intent(I("sit_down", target="B2", id="b"))
    assert st(w, "P5").state == "running"
    step_until(w, 200.0)
    assert st(w, "P5").posture == "seated" and st(w, "P5").state == "done"
    w.apply_intent(I("stand_up", id="c"))
    assert st(w, "P5").posture == "seated"
    w.step(w.last_t + inst(w, "P5").sit_to_stand_s + 0.2)
    assert st(w, "P5").posture == "standing" and st(w, "P5").state == "done"


def test_unknown_target_fails():
    w = make()
    w.apply_intent(I("walk_to", target="nowhere"))
    s = st(w, "P5")
    assert s.state == "failed" and s.reason == "unknown target"
    w.apply_intent(I("approach", target="ZZ", id="k"))
    assert st(w, "P5").reason == "unknown target"


def test_leave_goes_gone_and_leaves_roster():
    w = make()
    w.apply_intent(I("leave", target="entrance"))
    step_until(w, 120.0)
    s = st(w, "P5")
    assert s.posture == "gone" and s.state == "done"
    assert "P5" not in [p.id for p in w.roster(120).people]
    assert all(x.id != "P5" for x in w.observation("P1", 120, []).visible)


def test_overlays_do_not_replace_base_or_move():
    w = make()
    w.apply_intent(I("walk_to", target="clinic_corridor", id="base"))
    step_until(w, 3.0)
    evs = w.apply_intent(I("cough", id="c"))
    assert len(evs) == 1 and evs[0].source_kind == "cough" and evs[0].z == 1.6
    assert w.apply_intent(I("fidget", id="f")) == []
    assert st(w, "P5").intent_id == "base" and st(w, "P5").state == "running"


def test_look_at_sets_yaw():
    w = make()
    w.apply_intent(I("look_at", target="g1"))
    s = st(w, "P5")
    r = w.loaded.scenario.robot
    assert s.yaw == pytest.approx(math.atan2(r.y - s.y, r.x - s.x))
    assert (s.x, s.y) == pytest.approx((-5.103, 1.112))


def test_talk_returns_event_and_heard_addressed():
    w = make()
    evs = w.apply_intent(I("talk", text="Excuse me", addressee="g1"))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.source_kind == "person" and ev.z == 1.6 and ev.loudness_db == 60 and ev.addressee == "g1"
    h = w.robot_heard(ev)
    # the robot hears a person, not who it was: the microphone cannot tell P5 from anyone else
    assert h is not None and h.addressed_to_me and h.source_kind == "person" and h.source_id == "unknown"
    assert h.text == "Excuse me"
    assert w.observation("P5", 0, [ev]).heard == []  # own speech


def test_robot_say_heard_by_distance_and_hearing():
    w = make()
    ev = w.robot_say("Hello", 1.0)
    assert ev.source_kind == "robot" and ev.source_id == "g1" and ev.z == 1.25 and ev.loudness_db == 72
    r = w.loaded.scenario.robot
    assert (ev.x, ev.y) == (r.x, r.y)
    assert len(ev.id) == 32
    for pid in ["P1", "P5"]:
        s = st(w, pid)
        h = w.observation(pid, 1.0, [ev]).heard
        exp = route(ev, Pose(s.x, s.y, s.yaw), pid, inst(w, pid).hearing_loss_db, w.rng)
        assert (len(h) == 1) == (exp is not None)
    p = w.people["P4"]
    p.inst = p.inst.model_copy(update={"hearing_loss_db": 60.0})
    assert w.observation("P4", 1.0, [ev]).heard == []


def test_observation_sees_people_robot_seats_and_signs():
    w = make()
    w.apply_intent(I("look_at", target="P1"))  # P5 at D2 faces north; turn west towards P1 at B2
    o = w.observation("P5", 0.0, [])
    kinds = {(s.kind, s.id): s for s in o.visible}
    assert ("person", "P1") in kinds
    assert kinds[("person", "P1")].detail == "seated"
    seats = {s.id: s.detail for s in o.visible if s.kind == "seat"}
    assert seats["B2"] == "taken" and seats["D3"] == "free"
    assert all(s.distance <= 6.0 for s in o.visible if s.kind == "seat")
    assert ("person", "P5") not in kinds


def test_wet_floor_adds_sign_visible_from_any_side():
    w = make()
    w.apply_intent(I("walk_to", person="P5", target="reception_front"))
    step_until(w, 60.0)
    assert w.apply_world_event(WorldEvent(id="w", t=0, kind="wet_floor", x=2.0, y=0.57)) == []
    sign = [s for s in w.signs if s.text == "Caution: wet floor"]
    assert len(sign) == 1 and sign[0].id == "wet_floor_1" and sign[0].z == 0.6 and sign[0].placeholder
    seen = [s.id for s in w.observation("P5", 60, []).visible]  # P5 stands to the west, looking east
    assert "wet_floor_1" in seen
    # viewer on the far side also reads it
    w.apply_world_event(WorldEvent(id="w2", t=0, kind="wet_floor", x=-2, y=0))
    assert any(s.id == "wet_floor_2" for s in w.signs)


def test_pa_and_alarm_events():
    w = make()
    pa = w.apply_world_event(WorldEvent(id="a", t=1, kind="pa_announcement", text="Dr Smith to ward A"))
    al = w.apply_world_event(WorldEvent(id="b", t=2, kind="alarm"))
    assert pa[0].source_kind == "pa" and pa[0].text == "Dr Smith to ward A"
    assert al[0].source_kind == "alarm"
    assert w.observation("P1", 3, pa + al).heard[0].source_kind == "pa"


def test_spawn_arrival_and_person_exit():
    w = make()
    pid = w.spawn("anxious_first_timer", 10.0)
    assert pid == "N1" and w.roster(10).people[-1].id == "N1"
    s = st(w, "N1")
    assert (s.x, s.y) == pytest.approx((-1.24, -2.3)) and s.posture == "standing"
    assert w.spawn("companion", 11.0) == "N2"
    assert inst(w, "N1").seed == 7 * 1000 + 100 + 1
    w.apply_world_event(WorldEvent(id="e", t=12, kind="arrival", persona="companion"))
    assert w.roster(12).people[-1].id == "N3"
    w.apply_world_event(WorldEvent(id="x", t=13, kind="person_exit", person="N1"))
    step_until(w, 5.0)
    assert st(w, "N1").posture == "gone"


def test_determinism():
    def run(seed):
        w = make(seed)
        out = []
        for k in range(5):
            ev = w.robot_say("please take a seat and wait for the nurse to call you", k)
            out.append([h.model_dump_json() for pid in ("P1", "P2", "P3") for h in w.observation(pid, k, [ev]).heard])
        return out
    assert run(3) == run(3)


def test_unknown_persona_tolerated_in_event_raises_in_spawn(caplog):
    w = make()
    with caplog.at_level("WARNING"):
        assert w.apply_world_event(WorldEvent(id="e", t=1, kind="arrival", persona="nope")) == []
    assert "nope" in caplog.text
    assert [p.id for p in w.roster(1).people] == ["P1", "P2", "P3", "P4", "P5"]
    with pytest.raises(ValueError, match="unknown persona 'nope'"):
        w.spawn("nope", 1.0)
    assert w.spawn("companion", 2.0) == "N1"  # counter not consumed by failures


def test_own_seat_not_in_own_observation():
    w = make()
    ids = {s.id for s in w.observation("P5", 0.0, []).visible if s.kind == "seat"}
    assert "D2" not in ids
    w.apply_intent(I("look_at", target="P1"))
    w.apply_intent(I("sit_down", person="P1", target="A1", id="a"))
    w.step(0.1)  # P1 claims A1
    seats = {s.id: s.detail for s in w.observation("P5", 0.2, []).visible if s.kind == "seat"}
    assert seats.get("A1", "taken") == "taken"
    assert "A1" not in {s.id for s in w.observation("P1", 0.2, []).visible if s.kind == "seat"}


def test_backwards_time_ignored():
    w = make()
    w.apply_intent(I("walk_to", target="clinic_corridor"))
    step_until(w, 4.0)
    a = st(w, "P5")
    w.step(2.0)
    assert w.last_t == pytest.approx(4.0)
    w.step(4.0)
    b = st(w, "P5")
    assert (a.x, a.y) == (b.x, b.y)
    w.step(5.0)
    c = st(w, "P5")
    assert math.hypot(c.x - b.x, c.y - b.y) == pytest.approx(inst(w, "P5").walk_speed, rel=0.01)


def test_interrupted_stand_up_keeps_seat():
    w = make()
    w.apply_intent(I("stand_up", id="a"))
    w.step(0.1)
    w.apply_intent(I("sit_idle", id="b"))
    w.step(10.0)
    s = st(w, "P5")
    assert s.posture == "seated" and s.state == "done"
    seats = {x.id: x.detail for x in w.observation("P1", 10.0, []).visible if x.kind == "seat"}
    w.apply_intent(I("sit_down", person="P4", target="D2", id="c"))
    assert st(w, "P4").reason == "seat occupied"
    assert seats.get("D2", "taken") == "taken"


def test_unknown_person_raises_value_error():
    w = make()
    with pytest.raises(ValueError, match="unknown person 'ZZ'"):
        w.observation("ZZ", 0, [])
    with pytest.raises(ValueError, match="unknown person 'ZZ'"):
        w.say("ZZ", "hi", None, 0)
    with pytest.raises(ValueError, match="unknown person 'ZZ'"):
        w.cough("ZZ", "mild", 0)
    with pytest.raises(ValueError, match="unknown person 'ZZ'"):
        w.apply_intent(I("walk_to", person="ZZ", target="entrance"))


def test_person_exit_unknown_or_gone_is_tolerated(caplog):
    w = make()
    with caplog.at_level("WARNING"):
        assert w.apply_world_event(WorldEvent(id="a", t=1, kind="person_exit", person="P9")) == []
    assert "P9" in caplog.text
    w.apply_intent(I("leave", target="entrance"))
    step_until(w, 120.0)
    assert st(w, "P5").posture == "gone"
    assert w.apply_world_event(WorldEvent(id="b", t=2, kind="person_exit", person="P5")) == []


def test_spawned_instance_matches_the_world_spawn():
    from wardzero_agents.world.stub_world import spawned_instance
    w = make()
    w.spawn("anxious_first_timer", 1.0)
    w.spawn("companion", 2.0)
    assert spawned_instance(w.loaded, "N2", "companion", 2.0) == inst(w, "N2")
    assert spawned_instance(w.loaded, "N1", "anxious_first_timer", 1.0) == inst(w, "N1")
    assert inst(w, "N2").arrive_at_s == 2.0  # arrival time is when they appeared, not 0
    with pytest.raises(ValueError):
        spawned_instance(w.loaded, "P9", "companion")


# ---- whole-hospital site --------------------------------------------------------------------
import shutil

import yaml

from wardzero_agents.mind.prompts import persona_card, render_situation
from wardzero_agents.contracts import Observation


def site_world(tmp_path, patch=None, seed=7):
    cfg = tmp_path / "config"
    shutil.copytree(V1.parents[1], cfg)
    p = cfg / "scenarios" / "waiting_area_v1.yaml"
    data = yaml.safe_load(p.read_text())
    data["cast"][0].pop("seat")
    data["cast"][0]["start"] = {"bed": "B2"}
    data.update({"static_patients": True, **(patch or {})})
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return StubWorld(load_scenario(p), seed)


def run(w, secs):
    t0 = w.last_t
    for i in range(1, int(secs / 0.1) + 1):
        w.step(t0 + i * 0.1)


def test_walk_follows_route_not_straight_line():
    w = make()
    ls = w.loaded
    w.apply_intent(I("walk_to", target="ward_doors"))
    grid = ls.site.grid
    seen = []
    for i in range(1, 900):
        w.step(i * 0.1)
        s = st(w, "P5")
        seen.append((s.x, s.y))
        if s.state == "done":
            break
    assert st(w, "P5").state == "done"
    assert all(not grid.blocked(x, y) for x, y in seen[45:])
    dx, dy = ls.site.landmarks["ward_doors"].x, ls.site.landmarks["ward_doors"].y
    assert math.hypot(st(w, "P5").x - dx, st(w, "P5").y - dy) < 0.5


def test_unreachable_intent_fails():
    w = make()
    sealed = next((x, y) for x in range(-48, 40) for y in range(-24, 38)
                  if not w.loaded.site.grid.blocked(x, y) and w.loaded.site.check_place(x, y) == "not reachable from the entrance")
    w.apply_intent(I("walk_to", point=list(map(float, sealed))))
    s = st(w, "P5")
    assert s.state == "failed" and s.reason == "no route"


def test_static_patients_fill_free_beds(tmp_path):
    w = site_world(tmp_path)
    statics = {pid: p for pid, p in w.people.items() if p.kind == "static"}
    assert set(statics) == {"SB1", "SB3", "SB4", "SB5", "SB6"}
    assert all(p.posture == "lying" for p in statics.values())
    assert st(w, "P1").posture == "lying"
    r = w.roster(0.0)
    assert {e.id: e.role for e in r.people if e.id.startswith("S")} == {k: "patient_static" for k in statics}
    # still when time passes, and not audible
    run(w, 2.0)
    assert all(st(w, k).posture == "lying" for k in statics)
    ev = w.cough("P2", "severe", 0.0)
    assert audio_route_none(w, ev, "SB3")


def audio_route_none(w, ev, pid):
    p = w.people[pid]
    return route(ev, Pose(p.x, p.y, p.yaw), pid, 0.0, w.rng) is None or p.kind == "static"


def test_static_patient_seen_in_bed(tmp_path):
    w = site_world(tmp_path)
    b = w.loaded.site.beds["B3"]
    w.people["P1"].x, w.people["P1"].y = b.side_point("right")
    w.people["P1"].yaw = math.atan2(b.y - w.people["P1"].y, b.x - w.people["P1"].x)
    obs = w.observation("P1", 1.0, [])
    sb3 = [s for s in obs.visible if s.id == "SB3"]
    assert sb3 and sb3[0].detail == "a patient lying in bed B3"
    # nobody sees through the wall from the lobby
    w2 = make()
    assert not [s for s in w2.observation("P1", 1.0, []).visible if s.id.startswith("S")]


def test_static_patients_off(tmp_path):
    w = site_world(tmp_path, {"static_patients": False})
    assert not [p for p in w.people if p.startswith("S")]


def test_prompt_says_hospital_and_room():
    w = make()
    ls = w.loaded
    inst5 = inst(w, "P5")
    obs = Observation(person="P5", t=1.0)
    text = render_situation(inst5, obs, st(w, "P5"), [], [], 1.0, area=ls.area, site=ls.site)
    assert "Main lobby and reception" in text or "A&E walk-in reception" in text
    from wardzero_agents.mind.prompts import SYSTEM_FIXED
    assert "a UK hospital" in SYSTEM_FIXED and "main-entrance waiting area" not in SYSTEM_FIXED
    card = persona_card(inst5, ls.area, site=ls.site)
    assert "Seat ids" in card and "AE_Triage" not in card


def test_spawn_never_overwrites_an_existing_n_id():
    from wardzero_agents.config.loader import load_scenario
    from wardzero_agents.world.stub_world import StubWorld
    from pathlib import Path
    loaded = load_scenario(Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml")
    w = StubWorld(loaded, 1)
    w.people["N1"] = w.people["P1"]
    w.people["N2"] = w.people["P2"]
    pid = w.spawn("coughing_adult", 1.0)
    assert pid == "N3" and w.people["N1"] is w.people["P1"] and w.people["N2"] is w.people["P2"]
