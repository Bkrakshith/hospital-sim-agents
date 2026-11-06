"""Collisions: the robot keeps clear of furniture and of people, and people are never sent to stand inside furniture."""
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.bench import metrics as M
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.site import load_site
from wardzero_agents.site.routing import Router
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt
from wardzero_agents.world.perception import perceive_people

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
SITE = load_site(CONFIG)


def clearance(grid, x, y, r=1.0):
    """Distance from (x, y) to the nearest blocked cell centre, up to r (r when none is nearer)."""
    iy, ix = grid.to_cell(x, y)
    n = int(math.ceil(r / grid.res))
    best = r
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            if not grid.in_grid(iy + dy, ix + dx) or grid.occ[iy + dy, ix + dx]:
                cx, cy = grid.to_world(iy + dy, ix + dx)
                best = min(best, math.hypot(cx - x, cy - y))
    return best


def along(route, step=0.1):
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(k + 1):
            yield x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k


# ---- routes keep clear of furniture ----------------------------------------------------------------------------------
def test_a_hard_band_keeps_a_route_clear_of_furniture_where_the_map_allows():
    a, b = (1.05, 1.62), (-8.36, -0.63)  # the reception robot's post to the front of seat A1
    soft = Router(SITE.grid, clearance=0.3).route(a, b)
    hard = Router(SITE.grid, clearance=0.3, hard=0.5).route(a, b)
    assert soft is not None and hard is not None
    inner = [(x, y) for x, y in along(hard) if math.hypot(x - a[0], y - a[1]) > 1.0 and math.hypot(x - b[0], y - b[1]) > 1.0]
    assert inner and min(clearance(SITE.grid, x, y) for x, y in inner) >= 0.45


def test_a_hard_band_still_finds_a_way_through_a_tight_spot():
    r = Router(SITE.grid, clearance=0.3, hard=0.5)
    c = SITE.landmarks["reception_counter"]  # close to the desk
    assert r.route((1.05, 1.62), (c.x, c.y)) is not None
    r.block(-2.0, 0.0, 0.1)  # the band is rebuilt with the hard part too
    assert r.route((1.05, 1.62), (-3.0, -1.0)) is not None


# ---- the world ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reception(tmp_path_factory):
    preset = PRESETS["reception_guide"]
    chars_ = auto_staff.with_auto(list(preset.characters), preset)
    return preset_to_scenario(preset, CONFIG, tmp_path_factory.mktemp("staged"), characters=chars_)


def svc(loaded, tmp_path):
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def put(w, pid, x, y, yaw=0.0):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.plan = x, y, yaw, "standing", None, []


def track_of(s, pid):
    w = s.world
    s.step(0.1)
    return next(d["track"] for d in perceive_people(w, w.robot) if w.tracks.person(d["track"]) == pid)


def gap(w, pid):
    p = w.people[pid]
    return math.hypot(p.x - w.robot.x, p.y - w.robot.y)


def test_the_robot_stops_for_someone_in_its_way_then_goes_round_them(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    for pid in ("V2", "H1"):
        put(w, pid, -9.0, -1.0)
    put(w, "V1", -0.4, 0.8)  # standing on the robot's way west
    s.on_goal(-3.5, -1.0, None)
    gaps, arrived = [], False
    for _ in range(600):
        o = s.step(0.1)
        gaps.append(gap(w, "V1"))
        if any(r.get("kind") == "goal" and r["ok"] for r in o.task_results):
            arrived = True
            break
    assert arrived
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.05


def test_the_robot_waits_while_someone_crosses_in_front_of_it(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", 0.3, 0.9)
    s.on_goal(-3.5, -1.0, None)
    s.step(0.1)
    start = (w.robot.x, w.robot.y)
    for _ in range(10):
        s.step(0.1)
    assert w.robot.speed_scale == 0.0 or gap(w, "V1") >= rt.PERSON_KEEP_M
    assert gap(w, "V1") >= rt.PERSON_KEEP_M - 0.05 and math.hypot(w.robot.x - start[0], w.robot.y - start[1]) < 0.6


def test_the_robot_steps_back_when_someone_stands_too_close(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", w.robot.x - 0.45, w.robot.y - 0.1)
    for _ in range(80):
        s.step(0.1)
    assert gap(w, "V1") >= rt.BACK_OFF_M
    assert clearance(SITE.grid, w.robot.x, w.robot.y) >= 0.3


def test_showing_a_seat_the_robot_stands_aside_and_keeps_clear_of_people_and_chairs(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V2", 1.6, 0.0)
    tr = track_of(s, "V2")
    s.on_task_action({"id": "gd", "action": "guide_start", "track": tr, "place": "AE_Waiting", "seat": True})
    s.step(0.1)
    s.on_intent(Intent(id="f1", person="V2", t=0.0, kind="follow", target="g1"))
    gaps, clear, seat_id = [], [], None
    for _ in range(1500):
        o = s.step(0.1)
        gaps.append(gap(w, "V2"))
        if w.robot.speed > 0.05:
            clear.append(clearance(SITE.grid, w.robot.x, w.robot.y))
        for n in o.nudges:
            if n.card == rt.SEAT_CARD:
                seat_id = n.message.split("sit_down ")[1].split(")")[0]
                s.on_intent(Intent(id="s1", person="V2", t=0.0, kind="sit_down", target=seat_id))
        if w.people["V2"].posture == "seated":
            break
    assert seat_id is not None and w.people["V2"].seat == seat_id
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.1
    # 0.38 m from the nearest furniture cell centre: the A&E walkway is too narrow for the hard band, so the soft one holds
    assert clear and min(clear) >= 0.38


# ---- people ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ae(tmp_path_factory):
    preset = PRESETS["ae_waiting"]
    chars_ = auto_staff.with_auto(list(preset.characters), preset)
    return preset_to_scenario(preset, CONFIG, tmp_path_factory.mktemp("staged_ae"), characters=chars_)


def test_walking_up_to_someone_seated_in_a_row_ends_on_free_floor(ae, tmp_path):
    s = svc(ae, tmp_path)
    w = s.world
    a2 = w.site.seats["A2"]
    p2 = w.people["P2"]
    p2.x, p2.y, p2.yaw, p2.posture, p2.seat, p2.plan = a2.x, a2.y, a2.yaw, "seated", "A2", []
    put(w, "N1", -7.5, 3.0)  # behind the B row: the straight line to Ava runs through the chairs
    s.on_intent(Intent(id="a1", person="N1", t=0.0, kind="approach", target="P2"))
    for _ in range(300):
        s.step(0.1)
        if w.people["N1"].state == "done":
            break
    n1 = w.people["N1"]
    assert n1.state == "done"
    assert clearance(SITE.grid, n1.x, n1.y) >= 0.25
    assert math.hypot(n1.x - a2.x, n1.y - a2.y) <= 1.8


def test_a_spot_next_to_the_robot_keeps_a_metre_from_it(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V2", w.robot.x - 0.5, w.robot.y - 0.7)  # someone standing right by the robot
    put(w, "V1", -2.5, -1.5)
    s.on_intent(Intent(id="a1", person="V1", t=0.0, kind="approach", target="V2"))
    for _ in range(300):
        s.step(0.1)
        if w.people["V1"].state == "done":
            break
    assert gap(w, "V1") >= 0.9


def test_reading_a_sign_does_not_cancel_sitting_down(ae, tmp_path):
    s = svc(ae, tmp_path)
    w = s.world
    put(w, "P1", -4.35, -1.4)
    s.on_intent(Intent(id="s1", person="P1", t=0.0, kind="sit_down", target="C3"))
    p = w.people["P1"]
    for _ in range(200):
        s.step(0.1)
        if p.plan and p.plan[0].kind == "sit":
            break
    assert p.plan and p.plan[0].kind == "sit"
    s.on_intent(Intent(id="r1", person="P1", t=0.0, kind="read_sign", target="sign_wait"))
    assert p.intent_id == "r1" and p.state == "done"  # the mind hears its read_sign is done
    for _ in range(100):
        s.step(0.1)
    assert p.posture == "seated" and p.seat == "C3"


def test_walking_off_while_sitting_down_starts_from_the_front_of_the_seat(ae, tmp_path):
    s = svc(ae, tmp_path)
    w = s.world
    put(w, "P1", -4.35, -1.4)
    s.on_intent(Intent(id="s1", person="P1", t=0.0, kind="sit_down", target="C3"))
    p = w.people["P1"]
    for _ in range(200):
        s.step(0.1)
        if p.plan and p.plan[0].kind == "sit":
            break
    s.on_intent(Intent(id="w1", person="P1", t=0.0, kind="walk_to", target="entrance"))
    first = p.plan[0]
    assert first.kind == "walk"
    pts = [(p.x, p.y)] + list(first.path[first.i:])
    assert all(clearance(SITE.grid, x, y) > 0.0 for x, y in list(along(pts))[3:])


# ---- the report counts what is left -------------------------------------------------------------------------------------
def _run(tmp_path, status, poses):
    d = tmp_path / "r"
    d.mkdir()
    (d / "status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in status))
    (d / "robot_pose.jsonl").write_text("".join(json.dumps(r) + "\n" for r in poses))
    (d / "meta.json").write_text(json.dumps({"people": {"P1": {"role": "visitor"}}}))
    return M.RunData.load(d)


def test_the_report_counts_people_inside_furniture_and_the_robot_against_furniture(tmp_path):
    a2 = SITE.seats["A2"]
    status = [{"t": 0.0, "person": "P1", "x": 0.0, "y": 0.0, "posture": "standing"},
              {"t": 2.0, "person": "P1", "x": a2.x, "y": a2.y, "posture": "standing"},  # inside the chair row
              {"t": 5.0, "person": "P1", "x": a2.x, "y": a2.y, "posture": "seated"},  # sitting there is fine
              {"t": 8.0, "person": "P1", "x": 0.0, "y": 0.0, "posture": "standing"}]
    poses = [{"t": float(t), "x": -0.4, "y": 1.0 if t < 4 else 1.8, "speed": 0.5} for t in range(10)]
    d = _run(tmp_path, status, poses)
    assert M.people_in_furniture_s(d) == pytest.approx(3.0)
    assert M.robot_near_furniture_s(d) == pytest.approx(6.0)  # 1.8 is within 0.3 m of the desk's cells
    assert "people_in_furniture_s" in M.METRIC_NAMES and "robot_near_furniture_s" in M.METRIC_NAMES


def test_the_robot_and_a_person_walking_towards_each_other_keep_apart(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    put(w, "V1", -2.5, -1.0)
    s.on_goal(-2.0, -0.8, None)  # the robot heads for the person ...
    s.on_intent(Intent(id="a1", person="V1", t=0.0, kind="approach", target="g1"))  # ... who walks up to it
    gaps = []
    for _ in range(200):
        s.step(0.1)
        gaps.append(gap(w, "V1"))
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.1 and gaps[-1] >= 0.9


def test_held_with_someone_too_close_it_steps_back_then_carries_on(reception, tmp_path):
    s = svc(reception, tmp_path)
    w = s.world
    for pid in ("V2", "H1"):
        put(w, pid, -9.0, -1.0)
    s.on_goal(-3.5, -1.0, None)
    s.step(0.1)
    p = w.people["V1"]
    put(w, "V1", w.robot.x + 0.5 * math.cos(w.robot.yaw), w.robot.y + 0.5 * math.sin(w.robot.yaw))  # right in front
    gaps, arrived = [], False
    for _ in range(600):
        o = s.step(0.1)
        gaps.append(gap(w, "V1"))
        if any(r.get("kind") == "goal" and r["ok"] for r in o.task_results):
            arrived = True
            break
    assert max(gaps[:40]) >= rt.BACK_OFF_M  # it stepped back first
    assert arrived  # and then got where it was going (the same goal, not given up)


# ---- decisions people make that the world used to refuse --------------------------------------------------------
def test_following_a_person_walks_with_them_instead_of_being_refused():
    from wardzero_agents.mind.graph import check_decision
    loaded = preset_to_scenario(PRESETS["ae_waiting"], CONFIG)
    data = {"intent": {"kind": "follow", "target": "N1", "text": None}, "mood": "anxious", "patience_delta": 0,
            "reason": "the nurse is taking me to triage", "say": None, "addressee": None}
    d, err = check_decision(data, loaded.area, loaded.signs, ["N1", "P1", "P2"], role="ae_patient", site=loaded.site)
    assert err is None and d["intent"] == {"kind": "approach", "target": "N1", "text": None}
    data["intent"]["target"] = "g1"
    d, err = check_decision(data, loaded.area, loaded.signs, ["N1"], role="ae_patient", site=loaded.site)
    assert err is None and d["intent"]["kind"] == "follow"  # the robot still leads as before


def test_sitting_down_without_a_seat_takes_the_nearest_free_one(ae, tmp_path):
    from wardzero_agents.mind.graph import check_decision
    data = {"intent": {"kind": "sit_down", "target": None, "text": None}, "mood": "anxious", "patience_delta": 0,
            "reason": "I need to sit", "say": None, "addressee": None}
    d, err = check_decision(data, ae.area, ae.signs, ["P1"], role="ae_patient", site=ae.site)
    assert err is None and d["intent"]["target"] is None
    s = svc(ae, tmp_path)
    w = s.world
    put(w, "P2", -6.8, -1.2)
    s.on_intent(Intent(id="s1", person="P2", t=0.0, kind="sit_down", target=None))
    for _ in range(200):
        s.step(0.1)
        if w.people["P2"].posture == "seated":
            break
    seat = w.people["P2"].seat
    assert w.people["P2"].posture == "seated" and seat is not None and not w.site.seats[seat].staff_only
    assert math.hypot(w.site.seats[seat].x + 6.8, w.site.seats[seat].y + 1.2) < 3.0  # a seat near her
