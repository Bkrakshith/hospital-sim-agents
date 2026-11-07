"""Robot body, camera-like perception and the robot topic contract (stub test double)."""
import json
import math
import shutil
from pathlib import Path

import pytest
import yaml

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.models import RobotConfig
from wardzero_agents.contracts import Intent
from wardzero_agents.ros import topics
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.world.objects import Item
from wardzero_agents.world.perception import bbox_3d, perceive_objects, perceive_people
from wardzero_agents.world.robot_body import RobotBody
from wardzero_agents.world.stub_world import StubWorld

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def loaded_scenario(tmp_path, static_patients=True, bed=True):
    cfg = tmp_path / "config"
    shutil.copytree(V1.parents[1], cfg)
    p = cfg / "scenarios" / "waiting_area_v1.yaml"
    data = yaml.safe_load(p.read_text())
    if bed:
        data["cast"][0].pop("seat")
        data["cast"][0]["start"] = {"bed": "B2"}
    data["static_patients"] = static_patients
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_scenario(p)


def world(tmp_path, **kw):
    return StubWorld(loaded_scenario(tmp_path, **kw), 7)


def place_robot(w, x, y, yaw):
    from wardzero_agents.world.oracle import Pose
    w.robot_pose = Pose(x, y, yaw)


def put(w, pid, x, y, yaw=0.0, posture="standing"):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.bed = x, y, yaw, posture, None, None


def free_wall_spot(w):
    """A free cell with a free, hidden cell behind a wall to its east within 5 m: (x, y, far_x)."""
    g = w.site.grid
    for iy in range(5, g.occ.shape[0] - 5):
        for ix in range(5, g.occ.shape[1] - 60):
            if g.occ[iy, ix] or any(g.occ[iy, ix - j] for j in range(5)):
                continue
            for k in range(3, 20):
                if w._sight[iy, ix + k] and not g.occ[iy, ix + k + 4]:
                    x, y = g.to_world(iy, ix)
                    fx, _ = g.to_world(iy, ix + k + 4)
                    if not w._los((x, y), (fx, y)) and fx - x < 5.5 and not any(g.occ[iy, ix + j] for j in range(1, 3)):
                        return x, y, fx
    raise AssertionError("no wall spot")


def run_body(body, grid, secs, dt=0.05):
    for _ in range(int(secs / dt)):
        body.step(dt, grid)


def test_config_defaults():
    r = RobotConfig()
    assert (r.profile, r.ns, tuple(r.reach), r.radius, r.max_speed, r.fov_deg, r.range_m) == \
        ("g1_claude", "robot", (0.6, 1.3), 0.3, 1.0, 87.0, 6.0)


def test_robot_never_enters_blocked_cell(tmp_path):
    w = world(tmp_path)
    x, y, _ = free_wall_spot(w)
    b = RobotBody(x, y, 0.0, radius=0.3, max_speed=1.0)
    b.set_cmd_vel(1.0, 0.0)
    for _ in range(100):  # 5 s into the wall
        b.step(0.05, w.site.grid)
        assert not w.site.grid.blocked(b.pose[0], b.pose[1])
    assert b.speed == 0.0


def test_goal_follows_route(tmp_path):
    w = world(tmp_path)
    a, g = w.site.resolve("landmark:Lobby_centre") or w.site.resolve("Lobby_centre"), w.site.resolve("ward_doors")
    route = w.router.route(a, g)
    length = sum(math.dist(p, q) for p, q in zip(route, route[1:]))
    b = RobotBody(a[0], a[1], 0.0, radius=0.3, max_speed=1.0, router=w.router)
    assert b.set_goal(g[0], g[1]) is True
    for _ in range(int((length / 1.0 + 2.0) / 0.05)):
        b.step(0.05, w.site.grid)
    assert math.dist(b.pose[:2], g) < 0.2 and not b.has_goal


def test_goal_unreachable_returns_false(tmp_path):
    w = world(tmp_path)
    b = RobotBody(0.46, 1.26, 0.0, router=w.router)
    assert b.set_goal(1e6, 1e6) is False


def test_contact_logged_when_too_close(tmp_path):
    """The robot drives up to a standing person: a contact, logged once. (A contact is the robot's only when its own
    motion closed the gap, so the robot moves here; a person put beside a standing robot is not one.)"""
    w = world(tmp_path, bed=False)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 0.46 + 0.9, 1.26)
    w.robot.set_cmd_vel(0.5, 0.0)
    for k in range(1, 15):
        w.step(0.1 * k)
    recs = [r for s, r in w.drain_log() if s == "world" and r.get("kind") == "contact"]
    assert len(recs) == 1 and recs[0]["person"] == "P1"  # logged once, not every step


def test_no_contact_when_a_person_steps_up_to_a_standing_robot(tmp_path):
    w = world(tmp_path, bed=False)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 0.46 + 0.4, 1.26)
    w.step(0.1)
    w.step(0.2)
    assert not [r for s, r in w.drain_log() if r.get("kind") == "contact"]


def test_no_contact_when_far(tmp_path):
    w = world(tmp_path, bed=False)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 0.46 + 2.0, 1.26)
    w.step(0.1)
    assert not [r for s, r in w.drain_log() if r.get("kind") == "contact"]


def test_perception_hides_identity(tmp_path):
    w = world(tmp_path, bed=False)
    place_robot(w, 0.46, 1.26, 0.0)
    for i, pid in enumerate(("P1", "P2", "P3")):
        put(w, pid, 0.46 + 2.0 + i * 0.3, 1.26 + (i - 1) * 0.5)
    w.objects.add_item(Item(id="T1", kind="tray", label="meal tray", owner="P2", holder="P1"))
    people = perceive_people(w, w.robot)
    assert len(people) >= 2
    blob = json.dumps([people, perceive_objects(w, w.robot), bbox_3d(w, w.robot, 1.0)]).lower()
    for p in w.people.values():
        for secret in (p.inst.person_id, p.inst.display_name, p.inst.persona_id, p.inst.backstory):
            if secret:
                assert secret.lower() not in blob, secret
    for key in ("mood", "persona", "person", "name", "goals", "anxiety"):
        assert f'"{key}":' not in blob
    for d in people:
        assert set(d) == {"track", "x", "y", "posture", "moving", "holding", "bed"}
        assert d["track"].startswith("t") and d["track"][1:].isdigit()
    held = [d for d in people if d["holding"]]
    assert held and held[0]["holding"] == "tray"


def test_tracks_are_stable_per_person(tmp_path):
    w = world(tmp_path, bed=False)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 2.5, 1.26)
    put(w, "P2", 3.0, 1.9)
    first = {(d["x"], d["y"]): d["track"] for d in perceive_people(w, w.robot)}
    put(w, "P1", 2.6, 1.3)
    again = perceive_people(w, w.robot)
    ids = [d["track"] for d in again]
    assert len(set(ids)) == len(ids)
    assert first[(2.5, 1.26)] in ids


def test_perception_respects_fov_and_walls(tmp_path):
    w = world(tmp_path, bed=False)
    for pid in ("P2", "P3", "P4", "P5"):
        put(w, pid, 40.0, -20.0)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 0.46 + 2.0, 1.26)
    assert len(perceive_people(w, w.robot)) == 1
    put(w, "P1", 0.46 - 2.0, 1.26)  # behind the robot
    assert perceive_people(w, w.robot) == []
    put(w, "P1", 0.46 + 7.0, 1.26)  # beyond range
    assert perceive_people(w, w.robot) == []
    # behind a wall: find a person spot in view distance with no line of sight
    x, y, fx = free_wall_spot(w)
    place_robot(w, x, y, 0.0)
    put(w, "P1", fx, y)
    assert not w._los((x, y), (fx, y))
    assert perceive_people(w, w.robot) == []


def test_bbox_3d_shape_and_sizes(tmp_path):
    w = world(tmp_path)
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P1", 2.5, 1.26, posture="standing")
    put(w, "P2", 2.5, 2.2, posture="seated")
    put(w, "P3", 2.5, 0.2, posture="lying")
    msg = bbox_3d(w, w.robot, 3.5)
    assert msg["header"] == {"stamp": 3.5, "frame_id": "map"}
    by_z = sorted(d["bbox"]["size"]["z"] for d in msg["detections"])
    assert by_z[0] == pytest.approx(0.5, abs=0.1) and by_z[-1] == pytest.approx(1.7, abs=0.1)
    assert any(d["bbox"]["size"]["x"] == pytest.approx(1.8, abs=0.1) for d in msg["detections"])
    assert any(d["bbox"]["size"]["z"] == pytest.approx(1.2, abs=0.1) for d in msg["detections"])
    d0 = msg["detections"][0]
    assert d0["results"] == [{"class_id": "person", "score": 1.0}]
    assert set(d0["bbox"]["center"]) == {"x", "y", "z", "yaw"}


def test_perceive_objects(tmp_path):
    w = world(tmp_path)
    place_robot(w, 0.46, 1.26, 0.0)
    w.objects.add_item(Item(id="T1", kind="tray", label="meal tray", owner="P1", x=2.0, y=1.26, z=0.8))
    w.objects.start_fire(3.0, 1.26)
    w.objects.add_item(Item(id="X9", kind="instrument", x=-2.0, y=1.26))  # behind
    o = perceive_objects(w, w.robot)
    assert [i["id"] for i in o["items"]] == ["T1"] and "owner" not in o["items"][0]
    assert o["fires"][0]["size"] == pytest.approx(0.1)
    place_robot(w, 23.0, 19.5, 0.0)
    o = perceive_objects(w, w.robot)
    assert any(b["id"] == "B2" and "occupied" in b for b in o["beds"]) or o["beds"] is not None


# ---- the service: task actions, topics, aliases ------------------------------------------------
def service(tmp_path, **kw):
    loaded = loaded_scenario(tmp_path)
    run = tmp_path / "run"
    return StubSimService(loaded, RunLog(run), wait_for=(), **kw), run


def advance(s, secs, dt=0.1):
    outs = []
    for _ in range(int(round(secs / dt))):
        outs.append(s.step(dt))
    return outs


def results(outs):
    return [r for o in outs for r in o.task_results]


def test_robot_static_unless_commanded(tmp_path):
    s, _ = service(tmp_path)
    start = s.world.robot.pose
    advance(s, 3)
    assert s.world.robot.pose == start


def test_cmd_vel_and_odom_at_20hz(tmp_path):
    s, _ = service(tmp_path)
    place_robot(s.world, 0.46, 1.26, 0.0)
    s.on_cmd_vel((0.5, 0.0))
    outs = advance(s, 1.0, 0.05)
    odoms = [o.odom for o in outs if o.odom]
    assert 18 <= len(odoms) <= 22
    assert odoms[-1]["x"] > 0.46 + 0.3 and odoms[-1]["vx"] == pytest.approx(0.5)
    peo = [o for o in outs if o.detections is not None]
    assert 4 <= len(peo) <= 6  # 5 Hz


def bedside_robot(s, item_kind="tray"):
    """P1 propped up in bed B2, the robot beside the bed holding an item; returns (track, place)."""
    w = s.world
    s.on_intent(Intent(id="up", person="P1", t=0, kind="sit_up_in_bed"))
    p = w.people["P1"]
    b = w.site.beds[p.bed]
    sx, sy = b.side_point("right")
    d = math.hypot(sx - b.x, sy - b.y)
    bx, by = b.x + (sx - b.x) / d * 0.5, b.y + (sy - b.y) / d * 0.5
    place_robot(w, bx + 0.4, by, math.atan2(b.y - by, b.x - bx - 0.4))
    w.objects.add_item(Item(id="T1", kind=item_kind, label="meal tray", holder="g1"))
    s.step(0.1)
    track = next(d["track"] for d in perceive_people(w, w.robot) if d["bed"] == "B2")
    return track, [bx, by, 0.85]


def test_hand_over_result_only_when_offer_resolves(tmp_path):
    s, run = service(tmp_path)
    track, place = bedside_robot(s)
    s.on_task_action({"id": "a1", "action": "hand_over", "item": "T1", "to": track, "place": place})
    assert results(advance(s, 0.5)) == []  # the offer is still open
    s.on_intent(Intent(id="acc2", person="P1", t=0, kind="accept_item"))
    got = results(advance(s, 1.0))
    assert [(r["id"], r["ok"], r["reason"]) for r in got] == [("a1", True, "ok")]
    assert s.world.objects.items["T1"].holder == "P1"
    rows = [json.loads(x) for x in (run / "robot_tasks.jsonl").read_text().splitlines()]
    assert rows[-1]["id"] == "a1" and rows[-1]["ok"] is True and rows[-1]["kind"] == "result"


def test_hand_over_no_answer_when_window_expires(tmp_path):
    s, _ = service(tmp_path)
    track, place = bedside_robot(s)
    s.on_task_action({"id": "a2", "action": "hand_over", "item": "T1", "to": track, "place": place})
    got = results(advance(s, 12.0))
    assert [(r["id"], r["ok"], r["reason"]) for r in got] == [("a2", False, "no_answer")]
    assert s.world.objects.items["T1"].holder == "g1"


def test_hand_over_out_of_reach(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    place_robot(w, 0.46, 1.26, 0.0)
    put(w, "P2", 2.0, 1.26)
    w.objects.add_item(Item(id="T1", kind="tray", holder="g1"))
    s.step(0.1)
    track = next(d["track"] for d in perceive_people(w, w.robot))
    # receiver is 1.5 m away: default place is next to them, beyond the robot's 0.8 m arm
    s.on_task_action({"id": "a3", "action": "hand_over", "item": "T1", "to": track})
    got = results(advance(s, 0.2))
    assert got and got[0]["ok"] is False and got[0]["reason"].startswith("out_of_reach")


def test_take_item_within_reach(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    place_robot(w, 0.46, 1.26, 0.0)
    w.objects.add_item(Item(id="T1", kind="tray", x=1.0, y=1.26, z=0.9))
    w.objects.add_item(Item(id="T2", kind="tray", x=3.0, y=1.26, z=0.9))
    s.on_task_action({"id": "k1", "action": "take", "item": "T1"})
    s.on_task_action({"id": "k2", "action": "take", "item": "T2"})
    got = {r["id"]: r for r in results(advance(s, 0.2))}
    assert got["k1"]["ok"] and w.objects.items["T1"].holder == "g1"
    assert not got["k2"]["ok"]


def test_extinguish_and_press_call(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    place_robot(w, 0.46, 1.26, 0.0)
    f = w.objects.start_fire(1.2, 1.26, size=0.2)
    s.on_task_action({"id": "e1", "action": "extinguish", "fire": f.id})
    got = results(advance(s, 5.0))
    assert [(r["id"], r["ok"]) for r in got] == [("e1", True)] and f.out
    s.on_task_action({"id": "c1", "action": "press_call"})
    outs = advance(s, 0.2)
    assert [r for r in results(outs) if r["id"] == "c1"][0]["ok"]
    assert any(e.source_kind == "person" and str(e.source_id).startswith("bell") for o in outs for e in o.audio)


def test_unknown_action_and_track(tmp_path):
    s, _ = service(tmp_path)
    s.on_task_action({"id": "z1", "action": "dance"})
    s.on_task_action({"id": "z2", "action": "hand_over", "item": "T1", "to": "t99"})
    got = {r["id"]: r for r in results(advance(s, 0.2))}
    assert not got["z1"]["ok"] and got["z1"]["reason"] == "unknown_action"
    assert not got["z2"]["ok"] and got["z2"]["reason"] == "unknown_track"


def test_guide_start_moves_robot_and_nudges_the_person(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    put(w, "P2", 3.0, 1.26)
    place_robot(w, 0.46, 1.26, 0.0)
    s.step(0.1)
    track = next(d["track"] for d in perceive_people(w, w.robot))
    s.on_task_action({"id": "g1", "action": "guide_start", "track": track, "place": "ward_doors"})
    outs = advance(s, 0.3)
    nudges = [n for o in outs for n in o.nudges]
    assert len(nudges) == 1 and nudges[0].person == "P2"
    assert nudges[0].message.startswith("The robot is offering to walk you to ")
    assert w.robot.has_goal
    s.on_task_action({"id": "g2", "action": "guide_stop"})
    got = {r["id"]: r for r in results(advance(s, 0.2))}
    assert got["g2"]["ok"] and not w.robot.has_goal


def test_task_brief_none_without_scenario_brief(tmp_path):
    s, _ = service(tmp_path)
    assert s.robot_brief is None


def test_robot_topics_table():
    t = topics.robot_topics("robot")
    assert t["odom"] == "/robot/odom" and t["bbox_3d"] == "/robot/camera/bbox_3d"
    assert t["task_action"] == "/robot/task/action" and t["clock"] == "/clock"
    assert set(topics.STUB_LIVE) <= set(t)
    assert topics.robot_topics("g1")["heard"] == "/g1/audio/heard"
    assert topics.uses_g1_alias("robot", "g1_claude") and topics.uses_g1_alias("g1", "other")
    assert not topics.uses_g1_alias("robot", "other")
    for name in t.values():
        assert not any(name.startswith(p) for p in topics.FORBIDDEN_FOR_ROBOT)


# ---- fix round 1 ----------------------------------------------------------------------------
def test_second_hand_over_supersedes_the_open_one(tmp_path):
    s, run = service(tmp_path)
    track, place = bedside_robot(s)
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "T1", "to": track, "place": place})
    advance(s, 0.3)
    s.on_task_action({"id": "h2", "action": "hand_over", "item": "T1", "to": track, "place": place})
    got = results(advance(s, 0.3))
    assert [(r["id"], r["ok"], r["reason"]) for r in got] == [("h1", False, "superseded")]
    s.on_intent(Intent(id="acc", person="P1", t=0, kind="accept_item"))
    got = results(advance(s, 1.0))
    assert [(r["id"], r["ok"], r["reason"]) for r in got] == [("h2", True, "ok")]


def test_robot_moves_away_during_the_window(tmp_path):
    s, _ = service(tmp_path)
    track, place = bedside_robot(s)
    s.on_task_action({"id": "m1", "action": "hand_over", "item": "T1", "to": track, "place": place})
    advance(s, 0.3)
    r = s.world.robot_pose
    place_robot(s.world, r.x + 3.0, r.y, r.yaw)
    s.on_intent(Intent(id="acc", person="P1", t=0, kind="accept_item"))
    got = results(advance(s, 1.0))
    assert [(r["id"], r["ok"], r["reason"]) for r in got] == [("m1", False, "out_of_reach_giver")]


def test_receiver_moves_away_during_the_window(tmp_path):
    s, _ = service(tmp_path)
    track, place = bedside_robot(s)
    s.on_task_action({"id": "m2", "action": "hand_over", "item": "T1", "to": track, "place": place})
    advance(s, 0.3)
    s.on_intent(Intent(id="down", person="P1", t=0, kind="lie_back"))
    s.on_intent(Intent(id="acc", person="P1", t=0, kind="accept_item"))
    got = results(advance(s, 1.0))
    assert len(got) == 1 and got[0]["id"] == "m2" and got[0]["reason"] == "out_of_reach_receiver"


def test_goal_arrived_result_and_log(tmp_path):
    s, run = service(tmp_path)
    w = s.world
    place_robot(w, 0.46, 1.26, 0.0)
    s.on_goal(3.0, 1.26, None)
    got = [r for r in results(advance(s, 8.0, 0.05)) if r.get("kind") == "goal"]
    assert [(r["ok"], r["reason"]) for r in got] == [(True, "arrived")]
    rows = [json.loads(x) for x in (run / "robot_tasks.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "goal" and r["reason"] == "arrived" for r in rows)


def test_goal_no_route_result(tmp_path):
    s, _ = service(tmp_path)
    s.on_goal(1e6, 1e6, None)
    got = [r for r in results(advance(s, 0.2)) if r.get("kind") == "goal"]
    assert [(r["ok"], r["reason"]) for r in got] == [(False, "no_route")]


def test_goal_stuck_result(tmp_path):
    s, run = service(tmp_path)
    w = s.world
    place_robot(w, 0.46, 1.26, 0.0)
    s.on_goal(5.0, 1.26, None)
    w.robot.max_speed = 0.0  # the robot cannot make progress
    got = [r for r in results(advance(s, 7.0, 0.05)) if r.get("kind") == "goal"]
    assert [(r["ok"], r["reason"]) for r in got] == [(False, "stuck")]
    assert not w.robot.has_goal
    rows = [json.loads(x) for x in (run / "robot_tasks.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "goal" and r["reason"] == "stuck" for r in rows)


def test_passing_a_bed_logs_no_contact(tmp_path):
    w = world(tmp_path)
    b = w.site.beds[w.people["P1"].bed]
    # drive past the patient's bed 0.6 m from the patient
    place_robot(w, b.x - 1.5, b.y + 0.6, 0.0)
    w.robot.set_cmd_vel(0.5, 0.0)
    for i in range(1, 61):
        w.step(i * 0.05)
    assert not [r for s_, r in w.drain_log() if r.get("kind") == "contact"]


def test_bed_contact_logged_when_robot_reaches_the_bed_centre_line(tmp_path):
    w = world(tmp_path)
    b = w.site.beds[w.people["P1"].bed]
    w.site.grid.occ[:] = 0  # an open floor, so the robot can be driven onto the bed's centre
    place_robot(w, b.x - 1.0, b.y, 0.0)
    w.robot.set_cmd_vel(0.5, 0.0)
    for i in range(1, 61):
        w.step(i * 0.05)
    log = [r for s_, r in w.drain_log()]
    recs = [r for r in log if r.get("kind") == "bed_contact"]
    assert len(recs) == 1 and recs[0]["bed"] == b.id


# ---- fix round 2 ----------------------------------------------------------------------------
def goal_results(outs):
    return [(r["id"], r["ok"], r["reason"]) for r in results(outs) if r.get("kind") == "goal"]


def test_goal_pose_ids_are_unique_and_a_replaced_goal_is_superseded(tmp_path):
    s, _ = service(tmp_path)
    place_robot(s.world, 0.46, 1.26, 0.0)
    s.on_goal(6.0, 1.26, None)
    advance(s, 0.3)
    s.on_goal(2.0, 1.26, None)
    got = goal_results(advance(s, 0.2))
    assert len(got) == 1 and got[0][1:] == (False, "superseded") and got[0][0].startswith("goal_pose-")
    s.on_goal(6.0, 1.26, None)
    got2 = goal_results(advance(s, 0.2))
    assert got2[0][0] != got[0][0]  # a different goal id


def test_guide_start_replaced_by_goal_pose_is_superseded(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    put(w, "P2", 3.0, 1.26)
    place_robot(w, 0.46, 1.26, 0.0)
    s.step(0.1)
    track = next(d["track"] for d in perceive_people(w, w.robot))
    s.on_task_action({"id": "gs", "action": "guide_start", "track": track, "place": "ward_doors"})
    advance(s, 0.3)
    s.on_goal(1.5, 1.26, None)
    assert ("gs", False, "superseded") in goal_results(advance(s, 0.2))


def test_cmd_vel_cancels_a_goal(tmp_path):
    s, _ = service(tmp_path)
    place_robot(s.world, 0.46, 1.26, 0.0)
    s.on_goal(6.0, 1.26, None)
    advance(s, 0.3)
    s.on_cmd_vel((0.2, 0.0))
    got = goal_results(advance(s, 0.2))
    assert len(got) == 1 and got[0][1:] == (False, "cancelled")


def test_guide_stop_cancels_the_guide_goal(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    put(w, "P2", 3.0, 1.26)
    place_robot(w, 0.46, 1.26, 0.0)
    s.step(0.1)
    track = next(d["track"] for d in perceive_people(w, w.robot))
    s.on_task_action({"id": "gs", "action": "guide_start", "track": track, "place": "ward_doors"})
    advance(s, 0.3)
    s.on_task_action({"id": "gx", "action": "guide_stop"})
    outs = advance(s, 0.2)
    assert goal_results(outs) == [("gs", False, "cancelled")]
    assert ("gx", True, "ok") in [(r["id"], r["ok"], r["reason"]) for r in results(outs)]


def test_human_give_item_supersedes_the_robots_open_offer(tmp_path):
    s, _ = service(tmp_path)
    w = s.world
    track, place = bedside_robot(s)
    s.on_task_action({"id": "rh", "action": "hand_over", "item": "T1", "to": track, "place": place})
    advance(s, 0.3)
    # a nurse (P2) stands by the bed holding another item and offers it to the same patient
    p1 = w.people["P1"]
    put(w, "P2", place[0], place[1] - 0.3)
    w.objects.add_item(Item(id="T2", kind="tray", holder="P2"))
    s.on_intent(Intent(id="gv", person="P2", t=0, kind="give_item", target="P1", text="T2",
                       point=(place[0], place[1])))
    got = results(advance(s, 0.3))
    assert ("rh", False, "superseded") in [(r["id"], r["ok"], r["reason"]) for r in got]
