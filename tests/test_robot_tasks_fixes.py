"""Robot tasks and world mechanics found wanting by the oracle runs: the speed cap before the move, actions planned on
the step, tell spots clear of the furniture, hand-over offers and their reasons, the follow action, guide mechanics,
sleep, blocked routes and what the logs carry (Quick 2D world)."""
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


def svc(loaded, tmp_path, time_scale=1.0):
    return StubSimService(loaded, RunLog(tmp_path / "run"), time_scale, wait_for=())


def rows(run, stream):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def results(outs, aid=None, kind=None):
    out = [r for o in outs for r in o.task_results]
    if aid is not None:
        out = [r for r in out if r.get("id") == aid]
    if kind == "goal":
        out = [r for r in out if r.get("kind") == "goal"]
    elif kind == "action":
        out = [r for r in out if r.get("kind") != "goal"]
    return out


def put(w, pid, x, y, yaw=0.0, posture="standing"):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.plan = x, y, yaw, posture, None, []


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


def near_people(w):
    return min((math.hypot(q.x - w.robot.x, q.y - w.robot.y) for q in w.people.values()
                if q.posture not in ("gone", "lying") and q.kind != "static"), default=math.inf)


# ---- the near-person speed cap is set before the robot moves ------------------------------------------------------
@pytest.mark.parametrize("scale,wall", [(1.0, 0.1), (10.0, 0.5)])
def test_speed_cap_holds_on_the_step_the_robot_comes_near(staged, tmp_path, scale, wall):
    s = svc(staged("visiting_rush"), tmp_path, scale)
    w = s.world
    for pid in list(w.people):
        if pid != "V1":
            w.people[pid].posture = "gone"
    put(w, "V1", 19.9, 11.0)  # beside the corridor, 0.9 m off the robot's line up it
    s.on_goal(19.0, 16.0, None)
    fast_near = []
    for _ in range(int(60 / (scale * wall))):
        s.step(wall)
        d = near_people(w)
        if d <= rt.NEAR_PERSON_M and w.robot.speed > rt.NEAR_PERSON_SPEED + 1e-6:
            fast_near.append((round(s.t, 2), round(d, 2), round(w.robot.speed, 2)))
        if not w.robot.has_goal:
            break
    assert math.hypot(w.robot.x - 19.0, w.robot.y - 16.0) < 0.3
    assert not fast_near


# ---- a tell spot is clear of the furniture -------------------------------------------------------------------------
@pytest.mark.parametrize("scale,wall", [(1.0, 0.1), (3.0, 0.1), (10.0, 0.1)])
def test_tell_staff_in_b6_stops_clear_and_the_robot_can_walk_back_to_the_patient(staged, tmp_path, scale, wall):
    """night_call_bell at x3 froze the robot for 430 s in B6: it stopped to tell the nurse in the furniture's
    clearance band, and no walk could start from there."""
    s = svc(staged("night_call_bell", auto=True), tmp_path, scale)
    w = s.world
    x, y = s.tasks._reach_spots(w.people["P1"], "tray")[0]
    w.robot.teleport(x, y, 0.0)
    tr = w.tracks.track("P1")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr})
    told = None
    for _ in range(int(120 / (scale * wall))):
        o = s.step(wall)
        if [r for r in o.task_results if r.get("id") == "tl" and r.get("kind") != "goal"]:
            told = (w.robot.x, w.robot.y)
            break
    assert told is not None and w.robot.router.is_clear(*told)
    s.on_task_action({"id": "ap", "action": "approach", "track": tr})
    goals = []
    for _ in range(int(120 / (scale * wall))):
        o = s.step(wall)
        goals += [(r["ok"], r["reason"]) for r in o.task_results if r.get("id") == "ap" and r.get("kind") == "goal"]
        if goals:
            break
    assert goals == [(True, "arrived")]


# ---- route searches run on the step, not in the message callback ---------------------------------------------------
class _CountingRouter:
    """Wraps the robot's router and counts the route searches."""

    def __init__(self, base):
        self.base, self.calls = base, 0

    def route(self, *a, **k):
        self.calls += 1
        return self.base.route(*a, **k)

    def __getattr__(self, name):
        return getattr(self.base, name)


def test_no_route_search_runs_while_a_command_is_received(staged, tmp_path):
    s = svc(staged("night_call_bell", auto=True), tmp_path, 10.0)
    w = s.world
    x, y = s.tasks._reach_spots(w.people["P1"], "tray")[0]
    w.robot.teleport(x, y, 0.0)
    s.step(0.01)
    w.robot.router = counting = _CountingRouter(w.robot.router)
    tr = w.tracks.track("P1")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr})
    s.on_task_action({"id": "ap", "action": "approach", "track": tr})
    s.on_goal(19.0, 15.0, None)
    assert counting.calls == 0
    o = s.step(0.01)
    assert counting.calls > 0
    # carried out in the order they came: the tell walk, the approach (already there), then the goal replaced the walk
    assert sorted((r["id"], r["reason"]) for r in o.task_results if r.get("kind") != "goal") == [
        ("ap", "already_there"), ("tl", "superseded")]
    assert s.tasks.goal_id == "goal_pose-1" and w.robot.has_goal


def test_a_tell_walk_that_gets_stuck_reports_stuck_not_no_route(staged, tmp_path):
    s = svc(staged("night_call_bell", auto=True), tmp_path)
    w = s.world
    x, y = s.tasks._reach_spots(w.people["P1"], "tray")[0]
    w.robot.teleport(x, y, 0.0)
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": w.tracks.track("P1")})
    advance(s, 1.0)
    assert w.robot.has_goal and s.tasks.goal_id == "tl"
    w.robot.cancel_goal()
    w.robot.goal_event = "stuck"  # the body gave the walk up
    outs = advance(s, 0.2)
    assert [(r["ok"], r["reason"]) for r in results(outs, "tl", "action")] == [(False, "stuck")]


def test_the_first_goal_at_x10_is_not_stuck_after_one_long_step(staged, tmp_path):
    """specimen_run at x10: one capped 5 s step right after a goal_pose ended it 'stuck' with no movement."""
    s = svc(staged("specimen_run", auto=True), tmp_path, 10.0)
    w = s.world
    s.step(0.01)
    start = (w.robot.x, w.robot.y)
    s.on_goal(10.0, 2.0, None)
    outs = [s.step(0.5)] + advance(s, 0.3, 0.05)
    assert not [r for r in results(outs, kind="goal") if r["reason"] == "stuck"]
    assert math.hypot(w.robot.x - start[0], w.robot.y - start[1]) > 1.0


# ---- hand-over: the offer row, the receiver and item on the result, no_answer, one-step hand-over ------------------
def _robot_rows(tmp_path, kind=None):
    return [r for r in rows(tmp_path / "run", "robot_tasks") if kind is None or r.get("kind") == kind]


def _face_porter(s):
    w = s.world
    po = w.people["PO1"]
    w.robot.teleport(8.88, 1.72, math.atan2(po.y - 1.72, po.x - 8.88))
    return w.tracks.track("PO1")


def test_an_accepted_hand_over_logs_the_offer_and_names_receiver_and_item(staged, tmp_path):
    s = svc(staged("specimen_run"), tmp_path)
    tr = _face_porter(s)
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "specimen_1", "to": tr})
    advance(s, 0.3)
    s.on_intent(Intent(id="acc", person="PO1", t=s.t, kind="accept_item"))
    outs = advance(s, 1.0)
    res = results(outs, "ho", "action")
    assert [(r["ok"], r["reason"], r["person"], r["item"]) for r in res] == [(True, "ok", "PO1", "specimen_1")]
    offer = [r for r in _robot_rows(tmp_path, "hand_over")]
    assert [(r["event"], r["person"], r["item"], r["id"]) for r in offer] == [("offer", "PO1", "specimen_1", "ho")]
    logged = [r for r in _robot_rows(tmp_path, "result") if r.get("id") == "ho"]
    assert logged and logged[0]["person"] == "PO1" and logged[0]["item"] == "specimen_1"
    assert offer[0]["t"] <= logged[0]["t"]
    # the robot's own copy of the result carries no person id
    assert "person" not in rt.wire_result(res[0]) and rt.wire_result(res[0])["item"] == "specimen_1"


def test_a_declined_offer_is_refused_and_an_unanswered_one_is_no_answer(staged, tmp_path):
    s = svc(staged("specimen_run"), tmp_path)
    tr = _face_porter(s)
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "specimen_1", "to": tr})
    advance(s, 0.3)
    s.on_intent(Intent(id="no", person="PO1", t=s.t, kind="refuse_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "h1", "action")] == [(False, "refused")]
    s.on_task_action({"id": "h2", "action": "hand_over", "item": "specimen_1", "to": tr})
    outs = advance(s, 12.0)
    assert [(r["ok"], r["reason"], r["person"]) for r in results(outs, "h2", "action")] == [
        (False, "no_answer", "PO1")]
    assert s.world.objects.items["specimen_1"].holder == "g1"


def test_an_unanswered_offer_to_a_patient_lying_flat_is_no_answer_not_out_of_reach(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    tr = w.tracks.track("M1")
    x, y = s.tasks._reach_spots(w.people["M1"], "tray")[0]
    b = w.site.beds[w.people["M1"].bed]
    w.robot.teleport(x, y, math.atan2(b.y - y, b.x - x))
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "tray_M1", "to": tr})
    outs = advance(s, 12.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "ho", "action")] == [(False, "no_answer")]


def test_a_hand_over_takes_an_item_within_reach_first(staged, tmp_path):
    """One decision passes one instrument: the robot takes it off the tray and holds it out in the same action."""
    s = svc(staged("scrub_nurse"), tmp_path)
    w = s.world
    sc = w.objects.items["scalpel"]
    assert sc.holder is None
    assert math.hypot(sc.x - w.robot.x, sc.y - w.robot.y) <= rt.TAKE_REACH_M
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "scalpel", "to": w.tracks.track("S1")})
    advance(s, 0.2)
    assert sc.holder == "g1"
    assert [r["event"] for r in _robot_rows(tmp_path, "hand_over")] == ["offer"]
    s.on_intent(Intent(id="acc", person="S1", t=s.t, kind="accept_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"], r["person"], r["item"]) for r in results(outs, "ho", "action")] == [
        (True, "ok", "S1", "scalpel")]
    assert sc.holder == "S1"
    # an item beyond its reach is not taken: not_holding, as before
    far = next(i for i in w.objects.items.values() if i.holder is None
               and math.hypot(i.x - w.robot.x, i.y - w.robot.y) > rt.TAKE_REACH_M) if any(
        i.holder is None and math.hypot(i.x - w.robot.x, i.y - w.robot.y) > rt.TAKE_REACH_M
        for i in w.objects.items.values()) else None
    if far is not None:
        s.on_task_action({"id": "h2", "action": "hand_over", "item": far.id, "to": w.tracks.track("S1")})
        assert [r["reason"] for r in results(advance(s, 0.2), "h2", "action")] == ["not_holding"]


def test_a_take_result_names_the_item(staged, tmp_path):
    s = svc(staged("scrub_nurse"), tmp_path)
    s.on_task_action({"id": "k1", "action": "take", "item": "scalpel"})
    res = results(advance(s, 0.2), "k1", "action")
    assert [(r["ok"], r["item"]) for r in res] == [(True, "scalpel")]
