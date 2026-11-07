"""Robot tasks and world mechanics found wanting by the oracle runs: the speed cap before the move, actions planned on
the step, tell spots clear of the furniture, hand-over offers and their reasons, the follow action, guide mechanics,
sleep, blocked routes and what the logs carry (Quick 2D world)."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, Nudge, WorldEvent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt
from wardzero_agents.world import stub_world as sw
from wardzero_agents.world.perception import perceive_people

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


# ---- follow: the robot stays with a person ---------------------------------------------------------------------------
def _walk(s, pid, x, y):
    s.on_intent(Intent(id=f"w-{pid}-{x}-{y}", person=pid, t=s.t, kind="walk_to", point=(x, y)))


def test_follow_keeps_the_robot_with_a_walking_person_then_stops_on_guide_stop(staged, tmp_path):
    s = svc(staged("wandering_patient"), tmp_path)
    w = s.world
    put(w, "W1", 19.0, 13.5, yaw=1.57)  # 2 m up the corridor from the robot
    s.step(0.1)
    tr = w.tracks.track("W1")
    s.on_task_action({"id": "fo", "action": "follow", "track": tr})
    outs = advance(s, 0.2)
    assert [(r["ok"], r["reason"]) for r in results(outs, "fo", "action")] == [(True, "following")]
    _walk(s, "W1", 19.0, 20.0)  # up the ward corridor
    gaps, outs = [], []
    for _ in range(300):
        o = s.step(0.1)
        outs.append(o)
        gaps.append(math.hypot(w.people["W1"].x - w.robot.x, w.people["W1"].y - w.robot.y))
    p = w.people["W1"]
    assert math.hypot(p.x - 19.0, p.y - 20.0) < 0.5  # she got there
    assert 0.9 <= gaps[-1] <= 1.6  # and the robot is with her
    assert min(gaps) >= 0.8
    assert not [r for r in results(outs, kind="goal")]  # its own walks while following are not reported
    s.on_task_action({"id": "gs", "action": "guide_stop"})
    advance(s, 0.2)
    log = _robot_rows(tmp_path, "follow")
    assert [(r["event"], r["person"]) for r in log] == [("start", "W1"), ("end", "W1")]
    assert log[-1]["reason"] == "stopped"
    _walk(s, "W1", 19.0, 13.0)
    advance(s, 10)
    assert math.hypot(w.people["W1"].x - w.robot.x, w.people["W1"].y - w.robot.y) > 3.0  # no longer following
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]


def test_follow_an_unknown_track_is_no_person_and_a_goal_ends_it(staged, tmp_path):
    s = svc(staged("wandering_patient"), tmp_path)
    w = s.world
    s.on_task_action({"id": "f0", "action": "follow", "track": "t99"})
    assert [(r["ok"], r["reason"]) for r in results(advance(s, 0.2), "f0", "action")] == [(False, "no_person")]
    put(w, "W1", 19.5, 15.0, yaw=1.57)
    s.step(0.1)
    s.on_task_action({"id": "f1", "action": "follow", "track": w.tracks.track("W1")})
    advance(s, 2.0)
    s.on_goal(19.0, 8.0, None)
    advance(s, 0.2)
    ends = [r for r in _robot_rows(tmp_path, "follow") if r["event"] == "end"]
    assert [(r["id"], r["reason"]) for r in ends] == [("f1", "superseded")]
    w.people["W1"].posture = "gone"
    s.on_task_action({"id": "f2", "action": "follow", "track": w.tracks.track("W1")})
    assert [(r["ok"], r["reason"]) for r in results(advance(s, 0.2), "f2", "action")] == [(False, "no_person")]


# ---- guide mechanics ---------------------------------------------------------------------------------------------------
def _follow_robot(s, pid):
    s.on_intent(Intent(id=f"f-{pid}-{s.t}", person=pid, t=s.t, kind="follow", target="g1"))


def _gap(w, pid):
    p = w.people[pid]
    return math.hypot(p.x - w.robot.x, p.y - w.robot.y)


def test_a_guide_to_where_she_stands_keeps_clear_of_her(staged, tmp_path):
    """ae_waiting run A: guide_start to AE_Waiting with Ava 0.43 m from the room centre; the guided person was exempt
    from the robot's avoidance and the robot drove into her (contact at 0.3 m/s)."""
    s = svc(staged("ae_waiting", auto=True), tmp_path)
    w = s.world
    w.robot.teleport(-8.6, -1.4, 1.5708)
    put(w, "P2", -8.9, -0.3)
    s.step(0.1)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("P2"), "place": "AE_Waiting",
                      "seat": True})
    s.step(0.1)
    _follow_robot(s, "P2")
    gaps = []
    for _ in range(400):
        s.step(0.1)
        gaps.append(_gap(w, "P2"))
    assert not [r for r in rows(tmp_path / "run", "world") if r.get("kind") == "contact"]
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.1
    start = [r for r in _robot_rows(tmp_path, "guide") if r["event"] == "start"][0]
    assert math.hypot(start["x"] + 8.9, start["y"] + 0.3) >= 1.0  # the walk's goal is placed clear of her


def test_the_guided_person_standing_in_front_holds_the_robot(staged, tmp_path):
    s = svc(staged("reception_guide", auto=True), tmp_path)
    w = s.world
    r = w.robot
    put(w, "V2", r.x, r.y - 0.8)  # right in front of it (it faces -y), on its way to the ward entrance
    s.step(0.1)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("V2"), "place": "Lobby"})
    gaps = []
    for _ in range(30):
        s.step(0.1)
        gaps.append(_gap(w, "V2"))
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.05


def _guide_to_arrival(s, pid, place, **opts):
    w = s.world
    s.step(0.1)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track(pid), "place": place, **opts})
    s.step(0.1)
    _follow_robot(s, pid)
    outs = []
    for _ in range(900):
        o = s.step(0.1)
        outs.append(o)
        if s.tasks._guide is None and any(r.get("kind") == "goal" and r["id"] == "gd" for r in results(outs)):
            break
    outs += advance(s, 3.0)
    return outs


def test_return_to_post_false_leaves_the_robot_where_the_guide_ended(staged, tmp_path):
    s = svc(staged("reception_guide", auto=True), tmp_path)
    w = s.world
    put(w, "V2", 1.6, 0.0)
    outs = _guide_to_arrival(s, "V2", "ward_entrance", return_to_post=False)
    ends = [r for r in _robot_rows(tmp_path, "guide") if r["event"] == "end"]
    assert ends and ends[0]["reason"] == "arrived"
    assert s.tasks.goal_id != "gd-return" and not w.robot.has_goal
    post = s.tasks.post
    assert math.hypot(w.robot.x - post[0], w.robot.y - post[1]) > 5.0


def test_no_return_to_post_while_the_fire_alarm_sounds(staged, tmp_path):
    s = svc(staged("reception_guide", auto=True), tmp_path)
    w = s.world
    put(w, "V2", 1.6, 0.0)
    s.on_world_event(WorldEvent(id="al", t=0.0, kind="alarm", text="Fire alarm"))
    outs = _guide_to_arrival(s, "V2", "ward_entrance")  # return_to_post defaults to true
    assert [r["reason"] for r in _robot_rows(tmp_path, "guide") if r["event"] == "end"] == ["arrived"]
    assert s.tasks.goal_id != "gd-return" and not w.robot.has_goal


def test_a_person_released_from_a_guide_does_not_follow_the_robot_again(staged, tmp_path):
    s = svc(staged("reception_guide", auto=True), tmp_path)
    w = s.world
    put(w, "V2", 1.6, 0.0)
    _guide_to_arrival(s, "V2", "ward_entrance")
    _follow_robot(s, "V2")  # the mind still has "follow g1" in hand
    advance(s, 0.3)
    p = w.people["V2"]
    assert p.intent_kind == "follow" and p.state == "failed" and not s.world.is_trailing("V2")
    at = (p.x, p.y)
    advance(s, 30)  # it first edges past them (they stand behind it, on its way back) at the near-person speed
    assert _gap(w, "V2") > 3.0  # the robot walked back to its post alone
    assert math.hypot(p.x - at[0], p.y - at[1]) < 0.5
    # a new guide for them: they may follow again
    s.on_task_action({"id": "g2", "action": "guide_start", "track": w.tracks.track("V2"), "place": "Lobby"})
    s.step(0.1)
    _follow_robot(s, "V2")
    s.step(0.1)
    assert p.state == "running" and s.world.is_trailing("V2")


# ---- asleep ---------------------------------------------------------------------------------------------------------
def _beside_bed(s, pid):
    w = s.world
    x, y = s.tasks._reach_spots(w.people[pid], "tray")[0]
    b = w.site.beds[w.people[pid].bed]
    w.robot.teleport(x, y, math.atan2(b.y - y, b.x - x))
    return w.tracks.track(pid)


def _fall_asleep(s, pid):
    s.world.apply_nudge(Nudge(person=pid, t=s.t, card="asleep", message="You are deeply asleep."))


def _track_of(w, pid):
    return next((d for d in perceive_people(w, w.robot) if d["track"] == w.tracks.track(pid)), None)


def test_the_asleep_card_puts_her_to_sleep_and_the_robot_sees_it_within_4_m(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    _beside_bed(s, "M2")
    assert not [st for st in w.statuses() if st.person == "M2" and st.asleep]
    assert not _track_of(w, "M2").get("asleep")
    _fall_asleep(s, "M2")
    assert [st.asleep for st in w.statuses() if st.person == "M2"] == [True]
    assert [st.asleep for st in w.statuses() if st.person == "M1"] == [False]
    assert _track_of(w, "M2")["asleep"] is True
    # from further than 4 m the camera cannot tell
    q = w.people["M2"]
    for d in (4.6, 5.0, 5.5):
        for k in range(24):
            a = 2 * math.pi * k / 24
            x, y = q.x + d * math.cos(a), q.y + d * math.sin(a)
            if w.robot.router.is_clear(x, y) and w._los((x, y), (q.x, q.y)):
                w.robot.teleport(x, y, math.atan2(q.y - y, q.x - x))
                seen = _track_of(w, "M2")
                if seen is not None:
                    assert not seen.get("asleep")
                    return
    pytest.fail("no spot 4.6-5.5 m from M2 that can see her")


def test_a_hand_over_to_someone_asleep_is_no_answer_at_once(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    tr = _beside_bed(s, "M2")
    _fall_asleep(s, "M2")
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "tray_M2", "to": tr})
    outs = advance(s, 0.3)
    assert [(r["ok"], r["reason"], r["person"]) for r in results(outs, "ho", "action")] == [
        (False, "no_answer", "M2")]
    assert not [n for o in outs for n in o.nudges if n.person == "M2"]  # she is not asked: she is asleep
    assert s.world.objects.items["tray_M2"].holder == "g1"
    assert [r["event"] for r in _robot_rows(tmp_path, "hand_over")] == ["offer"]


def test_a_line_addressed_to_her_wakes_her_and_one_to_nobody_does_not(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    tr = _beside_bed(s, "M2")
    _fall_asleep(s, "M2")
    s.on_robot_say("Lunch is here.", addressee=None)
    advance(s, 0.2)
    assert w.people["M2"].asleep
    s.on_robot_say("George, your lunch is here.", addressee="M2")
    advance(s, 0.2)
    assert not w.people["M2"].asleep
    s.on_task_action({"id": "ho", "action": "hand_over", "item": "tray_M2", "to": tr})
    advance(s, 0.3)
    s.on_intent(Intent(id="acc", person="M2", t=s.t, kind="accept_item"))
    outs = advance(s, 1.0)
    assert [(r["ok"], r["reason"]) for r in results(outs, "ho", "action")] == [(True, "ok")]


def test_her_own_waking_intent_wakes_her_and_an_idle_one_does_not(staged, tmp_path):
    s = svc(staged("meal_round"), tmp_path)
    w = s.world
    _fall_asleep(s, "M2")
    s.on_intent(Intent(id="i1", person="M2", t=s.t, kind="sit_idle"))
    s.on_intent(Intent(id="i2", person="M2", t=s.t, kind="cough", overlay=True))
    advance(s, 0.2)
    assert w.people["M2"].asleep
    s.on_intent(Intent(id="i3", person="M2", t=s.t, kind="sit_up_in_bed"))
    advance(s, 0.2)
    assert not w.people["M2"].asleep


def test_start_asleep_on_the_cast_entry(staged, tmp_path):
    loaded = staged("meal_round")
    insts = [i.model_copy(update={"start_asleep": True}) if i.person_id == "M2" else i for i in loaded.instances]
    w = sw.StubWorld(dataclasses.replace(loaded, instances=insts), 1)
    assert w.people["M2"].asleep and not w.people["M1"].asleep


# ---- route blocks: the seal, what the PA says, and the robot's guide router ------------------------------------------
ENTRANCE_BLOCKS = ((-4.2, -2.6), (-1.2, -2.6), (1.8, -2.6))  # evacuation_guide's three blocks across the main entrance


def _crosses_entrance(path, start):
    """True when a route crosses the main entrance's line (y = -2.8, between x -6.5 and 3.5)."""
    pts = [start] + [tuple(p) for p in path]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 + 2.8) * (y1 + 2.8) < 0:
            x = x0 + (x1 - x0) * (-2.8 - y0) / (y1 - y0)
            if -6.5 <= x <= 3.5:
                return True
    return False


def _block_entrance(s, t=0.0):
    pas = []
    for i, (x, y) in enumerate(ENTRANCE_BLOCKS):
        pas += s.world.apply_world_event(WorldEvent(id=f"rb{i}", t=t, kind="route_blocked", x=x, y=y))
    return pas


def test_the_entrance_blocks_seal_the_whole_doorway_for_people(staged, tmp_path):
    """The three discs left a 0.2 m gap at the door's west end (x -5.8): people walked out through it."""
    s = svc(staged("evacuation_guide", auto=True), tmp_path)
    _block_entrance(s)
    w = s.world
    for start in ((-3.0, 0.0), (1.0, -0.9), (-6.5, 0.0)):
        for goal in ((-1.24, -4.5), (-6.0, -4.0), (-1.2, -5.0)):
            path = w.router.route(start, goal)
            assert path is None or not _crosses_entrance(path, start), (start, goal, path)


def test_the_route_blocked_pa_names_the_blocked_way_not_the_whole_room(staged, tmp_path):
    s = svc(staged("evacuation_guide", auto=True), tmp_path)
    pas = [e for e in _block_entrance(s) if e.source_kind == "pa"]
    assert len(pas) == 1
    assert "Main entrance" in pas[0].text and "Main lobby" not in pas[0].text


def test_route_blocked_reaches_the_robots_guide_router(staged, tmp_path):
    """The A&E guide went through the Lobby past the fire: a guide's walk must keep out of the blocks and the fire."""
    s = svc(staged("evacuation_guide", auto=True), tmp_path)
    w = s.world
    s.on_world_event(WorldEvent(id="f", t=0.0, kind="fire_start", x=-1.24, y=-2.3))
    for i, (x, y) in enumerate(ENTRANCE_BLOCKS):
        s.on_world_event(WorldEvent(id=f"rb{i}", t=0.0, kind="route_blocked", x=x, y=y))
    s.step(0.1)
    put(w, "V2", 1.6, -0.6)
    s.step(0.1)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("V2"), "place": "AE_Handover"})
    s.step(0.1)
    assert s.tasks._guide is not None
    path = [(w.robot.x, w.robot.y)] + [tuple(p) for p in w.robot._path]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
        for i in range(k + 1):
            x, y = x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k
            for bx, by in ENTRANCE_BLOCKS:
                assert math.hypot(x - bx, y - by) >= sw.ROUTE_BLOCK_RADIUS_M - 0.15
            assert math.hypot(x + 1.24, y + 2.3) >= sw.FIRE_KEEP_M - 0.1


# ---- what the robot sees within reach ----------------------------------------------------------------------------------
def test_the_instrument_tray_behind_the_scrub_robot_is_in_its_view(staged, tmp_path):
    """scrub_nurse: the tray is on the trolley just behind the robot, within reach; its situation said "Objects in
    view: nothing notable"."""
    from wardzero_agents.world.perception import perceive_objects
    s = svc(staged("scrub_nurse"), tmp_path)
    w = s.world
    ids = {i["id"] for i in perceive_objects(w, w.robot)["items"]}
    assert {"scalpel", "forceps", "retractor", "suture_scissors", "needle_holder"} <= ids
    # an item behind it out of reach is still out of view
    w.robot.teleport(w.robot.x, w.robot.y + 1.5, w.robot.yaw)
    assert not {i["id"] for i in perceive_objects(w, w.robot)["items"]} & {"scalpel"}


# ---- what the logs carry: robot nudges, bells ------------------------------------------------------------------------
def test_robot_nudges_reach_nudges_jsonl(staged, tmp_path):
    s = svc(staged("night_call_bell"), tmp_path)
    s.on_world_event(WorldEvent(id="rb", t=0.0, kind="ring_bell", person="P1"))
    s.on_task_action({"id": "pc", "action": "press_call", "track": s.world.tracks.track("P1")})
    outs = advance(s, 0.3)
    sent = [n for o in outs for n in o.nudges]
    assert sent and sent[0].card == rt.CALL_CARD
    logged = rows(tmp_path / "run", "nudges")
    assert [(r["person"], r["card"], r["message"]) for r in logged] == [(n.person, n.card, n.message) for n in sent]


def test_the_robot_hears_what_a_bell_is_and_the_repeater_has_no_bearing(staged, tmp_path):
    s = svc(staged("night_call_bell"), tmp_path)
    w = s.world
    bed = w.people["P1"].bed
    chimes = w.apply_world_event(WorldEvent(id="rb", t=0.0, kind="ring_bell", person="P1"))
    assert len(chimes) == 2 and all(e.text is None for e in chimes)  # people's own lines for a bell are unchanged
    at_bed, repeater = chimes
    for ev in chimes:  # stand the robot 3 m from each chime, so it hears it
        w.robot.teleport(ev.x + 3.0, ev.y, math.pi)
        if not w.robot.router.is_free(w.robot.x, w.robot.y):
            w.robot.teleport(ev.x, ev.y + 3.0, -math.pi / 2)
        h = w.robot_heard(ev)
        assert h is not None and h.text == f"call bell, bed {bed}"
        assert (h.bearing_deg is None) == (ev is repeater)
    own = w._event(1.0, "person", "bell_robot", w.robot.x + 1, w.robot.y, 2.0, 55.0)
    assert w.robot_heard(own).text == "staff call chime"
