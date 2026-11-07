"""The robot's own actions reach people: tell_staff finds the nurse, press_call is heard and answered, a hand-over
offer is announced, a route block reaches every router and the robot, people need an extinguisher, and the robot
hears where a sound came from (Quick 2D world)."""
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, WorldEvent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt
from wardzero_agents.world.objects import Item
from wardzero_agents.world.oracle import Pose
from wardzero_agents.world.perception import perceive_objects
from wardzero_agents.world.stub_world import ROUTE_BLOCK_RADIUS_M, StubWorld

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id, auto=False):
        key = (preset_id, auto)
        if key not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p) if auto else None
            cache[key] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[key]
    return get


def svc(loaded, tmp_path):
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def rows(run, stream):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def action_results(outs, aid):
    return [r for o in outs for r in o.task_results if r.get("id") == aid and r.get("kind") != "goal"]


def nudges(outs, card=None):
    return [n for o in outs for n in o.nudges if card is None or n.card == card]


def bedside(s, pid):
    """A spot beside the person's bed where the robot can stand (the one the approach action picks)."""
    w = s.world
    spots = s.tasks._reach_spots(w.people[pid], "tray")
    assert spots, f"no bedside spot for {pid}"
    return spots[0]


# ---- tell_staff on a ward ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("start", ["robot_start", "M1", "M2"])
def test_ward_tell_staff_reaches_the_nurse(staged, tmp_path, start):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    if start != "robot_start":
        x, y = bedside(s, start)
        w.robot.teleport(x, y, 0.0)
    tr = w.tracks.track("M2")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr, "message": "M2 is asleep."})
    outs, closest = [], math.inf
    n1 = w.people["N1"]
    for _ in range(250):
        outs.append(s.step(0.1))
        closest = min(closest, math.hypot(w.robot.x - n1.x, w.robot.y - n1.y))
        if action_results(outs, "tl"):
            break
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "tl")] == [(True, "told")]
    got = nudges(outs, rt.TELL_CARD)
    assert [n.person for n in got] == ["N1"] and "person id M2" in got[0].message
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]
    assert closest >= 0.8


def test_tell_staff_finds_the_night_nurse_in_b6(staged, tmp_path):
    s = svc(staged("night_call_bell"), tmp_path)
    w = s.world
    x, y = bedside(s, "P1")
    w.robot.teleport(x, y, 0.0)
    tr = w.tracks.track("P1")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr})
    outs = []
    for _ in range(600):
        outs.append(s.step(0.1))
        if action_results(outs, "tl"):
            break
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "tl")] == [(True, "told")]
    assert [n.person for n in nudges(outs, rt.TELL_CARD)] == ["N1"]
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]
    # then it walks back to its post on its own ("<id>-return")
    post, back = s.tasks.post, []
    for _ in range(600):
        o = s.step(0.1)
        back += [r for r in o.task_results if r.get("kind") == "goal" and r.get("id") == "tl-return"]
        if back:
            break
    assert back and back[0]["ok"] and math.hypot(w.robot.x - post[0], w.robot.y - post[1]) <= 0.5


def test_tell_staff_walks_back_to_post(staged, tmp_path):
    s = svc(staged("ae_waiting", auto=True), tmp_path)
    w = s.world
    post = s.tasks.post
    tr = w.tracks.track("P2")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr, "message": "Ava Thompson is here."})
    outs, at_told = [], None
    for _ in range(400):
        outs.append(s.step(0.1))
        if at_told is None and action_results(outs, "tl"):
            at_told = (w.robot.x, w.robot.y, s.t)
            break
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "tl")] == [(True, "told")]
    c = w.site.landmarks["reception_counter"]
    assert math.hypot(at_told[0] - c.x, at_told[1] - c.y) < 0.2
    back = None
    while s.t < at_told[2] + 35.0:
        s.step(0.1)
        if math.hypot(w.robot.x - post[0], w.robot.y - post[1]) <= 0.5:
            back = s.t
            break
    assert back is not None, "the robot did not walk back to its post"


# ---- press_call --------------------------------------------------------------------------------------------------
def test_press_call_in_ae_chimes_at_triage_and_is_answered(staged, tmp_path):
    s = svc(staged("ae_waiting"), tmp_path)
    w = s.world
    s.on_task_action({"id": "pc", "action": "press_call"})
    outs = [s.step(0.1) for _ in range(3)]
    chimes = [e for o in outs for e in o.audio if e.source_id == "bell_robot"]
    assert chimes and all(w.site.room_at(e.x, e.y) == "AE_Triage" for e in chimes)
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "pc")] == [(True, "answered")]
    got = nudges(outs, rt.CALL_CARD)
    assert [n.person for n in got] == ["N1"]
    room = w.site.rooms["AE_Waiting"].label.split(",")[0]
    assert room in got[0].message and "go to them" in got[0].message


def test_press_call_names_the_lit_bed(staged, tmp_path):
    s = svc(staged("night_call_bell"), tmp_path)
    s.on_world_event(WorldEvent(id="rb", t=0.0, kind="ring_bell", person="P1"))
    tr = s.world.tracks.track("P1")
    s.on_task_action({"id": "pc", "action": "press_call", "track": tr})
    outs = [s.step(0.1) for _ in range(3)]
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "pc")] == [(True, "answered")]
    got = nudges(outs, rt.CALL_CARD)
    assert [n.person for n in got] == ["N1"] and "B2" in got[0].message and "P1" in got[0].message


def test_press_call_with_no_staff_is_unanswered_but_ok(staged, tmp_path):
    s = svc(staged("evacuation_guide"), tmp_path)
    s.on_task_action({"id": "pc", "action": "press_call"})
    outs = [s.step(0.1) for _ in range(3)]
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "pc")] == [(True, "unanswered")]
    assert nudges(outs) == []


def test_press_call_does_not_reach_staff_out_of_earshot(staged, tmp_path):
    s = svc(staged("scrub_nurse"), tmp_path)
    w = s.world
    bx, by = s.tasks.call_point()
    n1 = w.people["N1"]
    assert math.hypot(n1.x - bx, n1.y - by) > 20.0
    s.on_task_action({"id": "pc", "action": "press_call"})
    outs = [s.step(0.1) for _ in range(3)]
    assert "N1" not in [n.person for n in nudges(outs, rt.CALL_CARD)]
    assert [r["ok"] for r in action_results(outs, "pc")] == [True]


def test_robot_own_chime_has_no_bearing(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    w.robot_pose = Pose(-8.6, -1.4, 0.0)
    chime = w._event(1.0, "person", "bell_robot", -6.6, -1.4, 2.0, 80.0)
    h = w.robot_heard(chime)
    assert h is not None and h.bearing_deg is None


# ---- hand-over offers --------------------------------------------------------------------------------------------
def test_standing_receiver_is_told_and_can_accept(staged, tmp_path):
    s = svc(staged("specimen_run"), tmp_path)
    w = s.world
    po = w.people["PO1"]
    w.robot.teleport(8.88, 1.72, math.atan2(po.y - 1.72, po.x - 8.88))
    tr = w.tracks.track("PO1")
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "specimen_1", "to": tr})
    outs = [s.step(0.1) for _ in range(5)]
    got = nudges(outs, rt.HAND_OVER_CARD)
    assert [n.person for n in got] == ["PO1"]
    assert "accept_item" in got[0].message and "refuse_item" in got[0].message
    s.on_intent(Intent(id="acc", person="PO1", t=0.0, kind="accept_item"))
    outs = [s.step(0.1) for _ in range(20)]
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "ho")] == [(True, "ok")]
    assert w.objects.items["specimen_1"].holder == "PO1"


def test_surgeon_offer_nudge_says_reject_item(staged, tmp_path):
    s = svc(staged("scrub_nurse"), tmp_path)
    w = s.world
    s1 = w.people["S1"]
    x, y = s1.x + 0.8 * math.cos(s1.yaw), s1.y + 0.8 * math.sin(s1.yaw)
    w.robot.teleport(x, y, math.atan2(s1.y - y, s1.x - x))
    w.objects.items["scalpel"].holder = "g1"
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "scalpel", "to": w.tracks.track("S1")})
    outs = [s.step(0.1) for _ in range(3)]
    got = nudges(outs, rt.HAND_OVER_CARD)
    assert [n.person for n in got] == ["S1"]
    assert "reject_item" in got[0].message and "refuse_item" not in got[0].message


def test_one_accept_sits_a_bed_patient_up(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    tr = w.tracks.track("M3")
    s.on_task_action({"id": "ap", "action": "approach", "track": tr})
    outs = []
    for _ in range(400):
        outs.append(s.step(0.1))
        if any(r.get("kind") == "goal" and r["id"] == "ap" for o in outs for r in o.task_results):
            break
    assert any(r.get("kind") == "goal" and r["id"] == "ap" and r["ok"] for o in outs for r in o.task_results)
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "tray_M3", "to": tr})
    outs = [s.step(0.1) for _ in range(5)]
    assert action_results(outs, "ho") == []  # the offer is open
    got = nudges(outs, rt.HAND_OVER_CARD)
    assert [n.person for n in got] == ["M3"] and "accept_item" in got[0].message
    s.on_intent(Intent(id="acc", person="M3", t=0.0, kind="accept_item"))
    outs = [s.step(0.1) for _ in range(10)]
    assert [(r["ok"], r["reason"]) for r in action_results(outs, "ho")] == [(True, "ok")]
    assert w.people["M3"].sitting_up is True


# ---- route blocks ------------------------------------------------------------------------------------------------
def _route_clear_of(route, start, cx, cy, r):
    pts = [start] + [tuple(p) for p in route]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
        for i in range(k + 1):
            if math.hypot(x0 + (x1 - x0) * i / k - cx, y0 + (y1 - y0) * i / k - cy) < r:
                return False
    return True


def test_route_blocked_blocks_every_router_and_is_announced(staged, tmp_path):
    s = svc(staged("evacuation_guide"), tmp_path)
    w = s.world
    lm = w.site.landmarks["clinic_corridor"]
    cx, cy = lm.x, lm.y
    a, b = (2.5, -0.5), (8.0, 8.0)
    before = w.robot.router.route(a, b)
    assert before is not None and not _route_clear_of(before, a, cx, cy, ROUTE_BLOCK_RADIUS_M - 0.15)
    # a walk already under way through the corridor is re-planned round the block
    s.on_goal(*b)
    s.step(0.1)
    s.on_world_event(WorldEvent(id="rb", t=0.0, kind="route_blocked", x=cx, y=cy))
    out = s.step(0.1)
    assert w.robot_router.version > 0
    after = w.robot.router.route(a, b)
    assert after is None or _route_clear_of(after, a, cx, cy, ROUTE_BLOCK_RADIUS_M - 0.15)
    if after is not None:  # the walk under way is kept and re-planned, not dropped
        assert w.robot.has_goal
        assert _route_clear_of(list(w.robot._path), (w.robot.x, w.robot.y), cx, cy, ROUTE_BLOCK_RADIUS_M - 0.15)
    pa = [h for h in out.robot_heard if h.source_kind == "pa"]
    # it names the landmark at the block, not the whole room the block is in (that room may hold the other way out)
    assert pa and "blocked" in pa[0].text and lm.label.split(",")[0] in pa[0].text


def test_blocks_at_one_time_and_place_are_announced_once(staged):
    w = StubWorld(staged("evacuation_guide"), 1)
    pas = []
    for i, x in enumerate((-4.2, -1.2, 1.8)):
        pas += w.apply_world_event(WorldEvent(id=f"rb{i}", t=5.0, kind="route_blocked", x=x, y=-2.6))
    assert len([e for e in pas if e.source_kind == "pa"]) == 1


def test_leave_to_a_blocked_exit_fails(staged):
    w = StubWorld(staged("evacuation_guide"), 1)
    e = w.site.landmarks["entrance"]
    # the block's edge is just past the entrance: the goal must not be moved out of it to free floor
    w.apply_world_event(WorldEvent(id="rb", t=0.0, kind="route_blocked", x=e.x, y=e.y + 1.2))
    w.apply_intent(Intent(id="lv", person="V1", t=0.0, kind="leave", target="entrance"))
    v1 = w.people["V1"]
    for i in range(1, 300):
        w.step(i * 0.1)
    assert v1.state == "failed" and v1.reason == "no route" and v1.posture != "gone"


def test_a_person_needs_an_extinguisher(staged):
    w = StubWorld(staged("evacuation_guide"), 1)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=-2.0, y=2.5))
    fire = next(iter(w.objects.fires.values()))
    fire.size, fire.grow_per_s = 0.5, 0.0
    v1 = w.people["V1"]
    v1.x, v1.y = -2.0, 1.0  # 1.5 m from the fire, nothing in hand
    w.apply_intent(Intent(id="u1", person="V1", t=0.0, kind="use_extinguisher"))
    assert v1.state == "failed" and "extinguisher" in v1.reason
    w.objects.add_item(Item(id="ext_1", kind="extinguisher", label="fire extinguisher", x=-2.5, y=1.0, z=0.5))
    w.apply_intent(Intent(id="u2", person="V1", t=0.0, kind="use_extinguisher"))
    assert v1.state == "running" and w.objects.items["ext_1"].holder == "V1"
    for i in range(1, 11):
        w.step(i * 0.1)
    assert fire.size < 0.5



def test_staff_can_fetch_the_wall_extinguisher_then_spray(staged):
    """ward_fire: ext_1 hangs about 4 m from the fire, so a nurse takes it off the wall first, then walks up."""
    w = StubWorld(staged("ward_fire"), 1)
    w.apply_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=19.0, y=14.5))
    fire = next(iter(w.objects.fires.values()))
    fire.size, fire.grow_per_s = 0.5, 0.0
    n1, ext = w.people["N1"], w.objects.items["ext_1"]
    assert math.hypot(ext.x - 19.0, ext.y - 14.5) > 3.0
    n1.x, n1.y = ext.x + 0.3, ext.y + 0.3  # at the wall unit, no fire in reach
    w.apply_intent(Intent(id="u1", person="N1", t=0.0, kind="use_extinguisher"))
    assert n1.state == "done" and ext.holder == "N1"
    n1.x, n1.y = 19.0, 13.0  # walked up to the fire with it
    w.step(0.1)
    w.apply_intent(Intent(id="u2", person="N1", t=0.1, kind="use_extinguisher"))
    assert n1.state == "running"
    for i in range(2, 12):
        w.step(i * 0.1)
    assert fire.size < 0.5

# ---- the robot's ears and eyes -------------------------------------------------------------------------------------
def test_heard_cough_has_a_bearing(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    w.robot_pose = Pose(-8.6, -1.4, 0.3)
    a = 0.3 + math.radians(30)
    cough = w._event(1.0, "cough", "P1", -8.6 + 2.0 * math.cos(a), -1.4 + 2.0 * math.sin(a), 1.6, 68.0)
    h = w.robot_heard(cough)
    assert h is not None and h.source_id == "unknown" and abs(h.bearing_deg - 30.0) <= 15.0
    pa = w._event(1.0, "pa", "pa", 0.0, 0.0, 2.5, 80.0, "Waits are about two hours")
    assert w.robot_heard(pa).bearing_deg is None


def test_blocked_sign_is_perceived_in_view(staged):
    w = StubWorld(staged("evacuation_guide"), 1)
    lm = w.site.landmarks["clinic_corridor"]
    w.apply_world_event(WorldEvent(id="rb", t=0.0, kind="route_blocked", x=lm.x, y=lm.y))
    # a free spot a few metres from the sign that can see it
    spot = None
    for d in (3.0, 3.5, 4.0, 2.5):
        for k in range(16):
            a = 2 * math.pi * k / 16
            x, y = lm.x + d * math.cos(a), lm.y + d * math.sin(a)
            if w.router.is_free(x, y) and w._los((x, y), (lm.x, lm.y)):
                spot = (x, y)
                break
        if spot:
            break
    assert spot is not None
    face = math.atan2(lm.y - spot[1], lm.x - spot[0])
    w.robot.teleport(spot[0], spot[1], face)
    blocked = perceive_objects(w, w.robot)["blocked"]
    assert [b["id"] for b in blocked] == ["route_blocked_1"] and "text" in blocked[0]
    w.robot.teleport(spot[0], spot[1], face + math.pi)
    assert perceive_objects(w, w.robot)["blocked"] == []
