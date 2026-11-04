import math
import shutil
from pathlib import Path

import pytest
import yaml

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import BASIC_INTENTS, ROLE_INTENTS
from wardzero_agents.contracts import Intent, WorldEvent
from wardzero_agents.mind.prompts import decision_schema
from wardzero_agents.world.objects import Fire, Item, WorldObjects
from wardzero_agents.world.oracle import Pose
from wardzero_agents.world.reach import giver_reach_ok, receiver_zone_ok
from wardzero_agents.world.stub_world import StubWorld

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def site_world(tmp_path, seed=7):
    cfg = tmp_path / "config"
    shutil.copytree(V1.parents[1], cfg)
    p = cfg / "scenarios" / "waiting_area_v1.yaml"
    data = yaml.safe_load(p.read_text())
    data["cast"][0].pop("seat")
    data["cast"][0]["start"] = {"bed": "B2"}
    data.update({"static_patients": True})
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return StubWorld(load_scenario(p), seed)


def I(kind, person="P1", id="i", t=0.0, **kw):
    return Intent(id=id, person=person, t=t, kind=kind, **kw)


def run(w, secs, dt=0.1):
    t0 = w.last_t
    for i in range(1, int(round(secs / dt)) + 1):
        w.step(t0 + i * dt)


def bedside(w, pid="P1", dist=0.5):
    """A point `dist` m beside the person's bed."""
    p = w.people[pid]
    b = w.site.beds[p.bed]
    sx, sy = b.side_point("right")
    d = math.hypot(sx - b.x, sy - b.y)
    return (b.x + (sx - b.x) / d * dist, b.y + (sy - b.y) / d * dist)


def robot_with_tray(w, place_xy):
    w.robot_pose = Pose(place_xy[0] + 0.4, place_xy[1], math.pi)
    w.objects.add_item(Item(id="T1", kind="tray", label="meal tray", holder="g1", x=0, y=0, z=1.0))


def test_tray_in_reach_sitting_up_in_bed(tmp_path):
    w = site_world(tmp_path)
    w.apply_intent(I("sit_up_in_bed"))
    assert w.people["P1"].sitting_up
    xy = bedside(w)
    robot_with_tray(w, xy)
    w.apply_intent(I("accept_item", t=1.0))
    ok, why = w.hand_over("g1", "P1", "T1", (xy[0], xy[1], 0.85), 2.0)
    assert (ok, why) == (True, "ok")
    assert w.objects.items["T1"].holder == "P1"


def test_tray_lying_flat_not_in_reach(tmp_path):
    w = site_world(tmp_path)
    xy = bedside(w)
    robot_with_tray(w, xy)
    w.apply_intent(I("accept_item", t=1.0))
    assert w.hand_over("g1", "P1", "T1", (xy[0], xy[1], 0.85), 2.0) == (False, "out_of_reach_receiver")


def test_tray_too_high_standing():
    f = receiver_zone_ok
    # standing receiver at origin facing +x: 0.5 m ahead
    assert f("standing", False, False, (0, 0), 0.0, None, (0.5, 0.0, 1.1))
    assert not f("standing", False, False, (0, 0), 0.0, None, (0.5, 0.0, 1.6))
    assert not f("standing", False, False, (0, 0), 0.0, None, (0.9, 0.0, 1.1))
    assert not f("standing", False, False, (0, 0), 0.0, None, (-0.5, 0.0, 1.1))  # behind
    assert f("seated", False, False, (0, 0), 0.0, None, (0.45, 0.0, 0.8))
    assert not f("seated", False, False, (0, 0), 0.0, None, (0.45, 0.0, 1.3))
    assert not f("lying", True, False, (0, 0), 0.0, None, (0.3, 0.0, 0.9), item_kind="tray")
    assert not f("fallen", False, False, (0, 0), 0.0, None, (0.3, 0.0, 0.9))


def test_giver_reach_and_not_holding(tmp_path):
    assert giver_reach_ok((0, 0), (0.5, 1.4), (0.5, 0.0, 1.0))
    assert not giver_reach_ok((0, 0), (0.5, 1.4), (1.0, 0.0, 1.0))
    assert not giver_reach_ok((0, 0), (0.5, 1.4), (0.5, 0.0, 1.6))
    w = site_world(tmp_path)
    xy = bedside(w)
    robot_with_tray(w, xy)
    w.robot_pose = Pose(w.robot_pose.x + 3.0, w.robot_pose.y, math.pi)
    w.apply_intent(I("sit_up_in_bed"))
    assert w.hand_over("g1", "P1", "T1", (xy[0], xy[1], 0.85), 2.0) == (False, "out_of_reach_giver")
    assert w.hand_over("P1", "g1", "T1", (xy[0], xy[1], 0.85), 2.0) == (False, "not_holding")


def test_refused_without_acceptance(tmp_path):
    w = site_world(tmp_path)
    w.apply_intent(I("sit_up_in_bed"))
    xy = bedside(w)
    robot_with_tray(w, xy)
    place = (xy[0], xy[1], 0.85)
    assert w.hand_over("g1", "P1", "T1", place, 2.0) == (False, "refused")  # never answered
    w.apply_intent(I("refuse_item", t=1.0))
    assert w.hand_over("g1", "P1", "T1", place, 2.0) == (False, "refused")
    w.apply_intent(I("accept_item", t=3.0))
    assert w.hand_over("g1", "P1", "T1", place, 20.0) == (False, "refused")  # answer too old (10 s)
    assert w.hand_over("g1", "P1", "T1", place, 8.0)[0] is True


def test_surgeon_window_is_5s():
    from wardzero_agents.world.objects import ACCEPT_WINDOW_S, SURGEON_WINDOW_S
    assert (ACCEPT_WINDOW_S, SURGEON_WINDOW_S) == (10.0, 5.0)


def test_fire_grows_and_is_put_out(tmp_path):
    w = site_world(tmp_path)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=3.0, y=4.0))
    fire = next(iter(w.objects.fires.values()))
    s0 = fire.size
    run(w, 10.0)
    assert fire.size == pytest.approx(s0 + 0.1, abs=1e-6)
    fire.size = 0.2
    w.robot_pose = Pose(3.5, 4.0, 0.0)
    for i in range(120):
        w.extinguish("g1", w.last_t, 0.1)
    assert fire.out and fire.size == 0.0
    run(w, 5.0)
    assert fire.out and fire.size == 0.0
    # too far: nothing
    w.apply_world_event(WorldEvent(id="f2", t=w.last_t, kind="fire_start", x=30.0, y=4.0))
    f2 = w.objects.fires["fire_2"]
    before = f2.size
    w.extinguish("g1", w.last_t, 1.0)
    assert f2.size == before


def test_person_extinguisher_rate(tmp_path):
    w = site_world(tmp_path)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=3.0, y=4.0))
    fire = w.objects.fires["fire_1"]
    fire.size, fire.grow_per_s = 0.5, 0.0
    p = w.people["P1"]
    p.x, p.y = 3.5, 4.0
    w.apply_intent(I("use_extinguisher"))
    run(w, 1.0)
    assert fire.size == pytest.approx(0.5 - 0.08, abs=1e-6)


def test_fire_reaches_full_raises_alarm(tmp_path):
    w = site_world(tmp_path)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=3.0, y=4.0))
    fire = w.objects.fires["fire_1"]
    fire.size = 0.995
    run(w, 1.0)
    assert fire.size == 1.0
    evs = w.drain_audio()
    assert [e.source_kind for e in evs] == ["alarm"]
    run(w, 1.0)
    assert w.drain_audio() == []  # once


def test_call_bell_only_from_bed(tmp_path):
    w = site_world(tmp_path)
    p2 = w.people["P2"]
    w.apply_intent(I("call_bell", person="P2"))
    assert p2.state == "failed" and p2.reason == "not in bed"
    assert not w.objects.bells
    evs = w.apply_intent(I("call_bell", person="P1", t=3.0))
    bell = w.objects.bells["B2"]
    assert bell.active and bell.since == 3.0
    assert w.people["P1"].state == "done"
    assert len(evs) == 1 and evs[0].loudness_db == 55
    snap = {o.id: o for o in w.objects.snapshot()}
    assert snap["bell_B2"].kind == "call_bell" and snap["bell_B2"].active is True


def test_surgeon_cannot_walk():
    kinds = decision_schema("surgeon")["properties"]["intent"]["properties"]["kind"]["enum"]
    assert "walk_to" not in kinds and "leave" not in kinds
    assert {"request_instrument", "reject_item", "accept_item", "take_instrument_self"} <= set(kinds)


def test_schema_mood_enum_widened():
    moods = decision_schema("patient")["properties"]["mood"]["enum"]
    assert set(moods) == {"calm", "happy", "sad", "anxious", "stressed", "excited", "relieved",
                          "frustrated", "distressed"}


def test_role_intents_table():
    assert len(BASIC_INTENTS) == 14
    assert {"call_bell", "sit_up_in_bed", "lie_back", "accept_item", "refuse_item"} <= ROLE_INTENTS["patient"]
    assert {"attend", "give_item", "take_item", "call_for_help", "escort", "evacuate"} <= ROLE_INTENTS["ward_nurse"]
    assert ROLE_INTENTS["custom"] == frozenset(BASIC_INTENTS)
    for r in ("staff", "visitor", "companion", "receptionist", "porter"):
        assert BASIC_INTENTS <= ROLE_INTENTS[r] or r == "porter"


def test_surgeon_speech_and_reject(tmp_path):
    w = site_world(tmp_path)
    evs = w.apply_intent(I("request_instrument", person="P2", text="scalpel", addressee="g1"))
    assert evs[0].text == "scalpel, please" and evs[0].addressee == "g1"
    evs = w.apply_intent(I("reject_item", person="P2", t=5.0))
    assert evs[0].text == "No, not that"
    assert w.experience and w.experience[-1]["kind"] == "failure" and w.experience[-1]["person"] == "P2"


def test_take_instrument_self(tmp_path):
    w = site_world(tmp_path)
    p = w.people["P2"]
    w.objects.add_item(Item(id="X1", kind="instrument", label="scalpel", x=p.x + 0.5, y=p.y, z=0.9))
    w.apply_intent(I("take_instrument_self", person="P2", text="scalpel"))
    assert w.objects.items["X1"].holder == "P2" and p.state == "done"
    w.apply_intent(I("take_instrument_self", person="P2", id="j", text="forceps"))
    assert p.state == "failed"


def test_give_item_nurse_to_patient(tmp_path):
    w = site_world(tmp_path)
    xy = bedside(w, dist=0.5)
    n = w.people["P2"]
    n.x, n.y = xy[0] + 0.5, xy[1]
    n.yaw = math.atan2(w.people["P1"].y - n.y, w.people["P1"].x - n.x)
    w.objects.add_item(Item(id="T1", kind="tray", label="tray", holder="P2", x=n.x, y=n.y, z=1.0))
    w.apply_intent(I("sit_up_in_bed"))
    w.apply_intent(I("give_item", person="P2", id="g", target="P1", text="T1", point=(xy[0], xy[1])))
    assert w.people["P2"].state == "running"
    w.apply_intent(I("accept_item", person="P1", id="a", t=1.0))
    run(w, 1.0)
    assert w.objects.items["T1"].holder == "P1" and w.people["P2"].state == "done"


def test_objects_stream_logged(tmp_path):
    w = site_world(tmp_path)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=3.0, y=4.0))
    recs = w.drain_log()
    assert recs and recs[0][0] == "objects"
    assert w.drain_log() == []
    from wardzero_agents.runlog import STREAMS
    assert "objects" in STREAMS
