"""The person being guided does not block the guide (Quick 2D world, and a body that holds its heading while held, as
the walking G1 in Isaac does).

Isaac reception_guide (re-QA): guide rb2 stopped 0.9 m short with V1 0.88 m ahead of the robot's heading but behind it
on its way; the two walks back to post were blocked by her; H1's guide never moved because H1 stood between the robot
and the goal. Here: the robot tests who is in its way along its planned path, not its current heading; the person it
guides (or has just guided, on its walk back) is waited for, routed round and let go first instead of ending the walk
``blocked_by_person`` after the usual give-up time; a guide whose person is already at the place ends ``arrived``.
Bystanders still block as before."""
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_body as rb
from wardzero_agents.world import robot_tasks as rt

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p)
            cache[preset_id] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[preset_id]
    return get


@pytest.fixture(params=["turns", "holds_heading"])
def body(request, monkeypatch):
    """The Quick 2D body turns towards its route while the task layer holds it; the walking G1 in Isaac does not (a
    held LeggedBody stands still, facing where it faced). Both must work."""
    if request.param == "holds_heading":
        orig = rb.RobotBody._follow

        def follow(self, dt):
            v, w = orig(self, dt)
            return (0.0, 0.0) if self._path and self.speed_scale <= 0 else (v, w)
        monkeypatch.setattr(rb.RobotBody, "_follow", follow)
    return request.param


def svc(loaded, tmp_path):
    return StubSimService(loaded, RunLog(tmp_path / "run"), 1.0, wait_for=())


def rows(tmp_path, stream="robot_tasks", kind=None):
    p = tmp_path / "run" / f"{stream}.jsonl"
    out = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return [r for r in out if kind is None or r.get("kind") == kind]


def contacts(tmp_path):
    return [r for r in rows(tmp_path, "world") if r.get("kind") == "contact"]


def alone_with(w, keep):
    for pid, p in w.people.items():
        if pid not in keep:
            p.posture = "gone"


def put(w, pid, x, y, yaw=0.0):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.plan = x, y, yaw, "standing", None, []


def along(w, goal, dist):
    """The point dist metres along the robot's route to goal."""
    r = w.robot
    route = r.router.route((r.x, r.y), goal)
    pts = [(r.x, r.y)] + [tuple(p) for p in route[1:]]
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L >= dist:
            return a[0] + (b[0] - a[0]) * dist / L, a[1] + (b[1] - a[1]) * dist / L
        dist -= L
    return pts[-1]


def route_m(w, goal):
    r = w.robot
    pts = [(r.x, r.y)] + [tuple(p) for p in r.router.route((r.x, r.y), goal)[1:]]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def follow_robot(s, pid):
    s.on_intent(Intent(id=f"f-{pid}-{s.t}", person=pid, t=s.t, kind="follow", target="g1"))


def gap(w, pid):
    p = w.people[pid]
    return math.hypot(p.x - w.robot.x, p.y - w.robot.y)


def guide(s, pid, place, secs=120.0, **opts):
    """Start a guide for pid (who follows), run until it ends (and secs more at most), then 40 s for the walk back."""
    w = s.world
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track(pid), "place": place, **opts})
    outs = [s.step(0.1)]
    follow_robot(s, pid)
    gaps = []
    for i in range(int(secs / 0.1)):
        outs.append(s.step(0.1))
        gaps.append(gap(w, pid))
        if s.tasks._guide is None and i > 3:
            break
    return outs, gaps


def guide_end(tmp_path):
    return [r for r in rows(tmp_path, kind="guide") if r.get("event") == "end"]


# ---- along the path, not the heading ------------------------------------------------------------------------------
def test_someone_behind_the_robot_on_its_way_is_not_in_its_way(staged, tmp_path):
    """Isaac rb2 at t=42: the robot faced away from the place (yaw -179) with V1 0.88 m ahead of that heading, so V1
    held it although its way to the goal led the other way."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V2"})
    r = w.robot
    r.teleport(12.34, 6.87, math.radians(-179))
    assert s.tasks._set_goal("g", 13.2, 7.25, None, 0.0)
    put(w, "V2", 12.34 - 0.88, 6.87)  # in front of its heading, behind it on its way
    assert s.tasks._blocker() is None
    put(w, "V2", 12.34 + 0.8, 6.87 + 0.35)  # on its way, though behind its heading
    assert s.tasks._blocker() == "V2"


def test_only_a_turn_left_holds_for_nobody(staged, tmp_path):
    """Its route walked, turning on the spot to the goal's yaw moves it into nobody."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V2"})
    r = w.robot
    assert s.tasks._set_goal("g", r.x, r.y, r.yaw + 2.0, 0.0)
    r._path = []  # the route is walked; only the final turn is left
    put(w, "V2", r.x + 0.6 * math.cos(r.yaw), r.y + 0.6 * math.sin(r.yaw))
    assert s.tasks._blocker() is None


# ---- the person it guides stands in its way -----------------------------------------------------------------------
def test_the_guided_person_in_front_at_the_start_does_not_stop_the_guide(staged, tmp_path, body):
    """Guidee 0.8 m in front of the robot on its route at guide start: the guide still arrives, with no contact."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1"})
    goal = w._resolve_place("ward_entrance")
    put(w, "V1", *along(w, goal, 0.8))
    s.step(0.1)
    outs, gaps = guide(s, "V1", "ward_entrance")
    assert [e["reason"] for e in guide_end(tmp_path)] == ["arrived"], rows(tmp_path)
    assert not contacts(tmp_path)
    assert min(gaps) >= rt.PERSON_KEEP_M - 0.1


def test_the_guided_person_between_the_robot_and_the_place_does_not_stop_the_guide(staged, tmp_path, body):
    """Isaac rb17: the robot by the ward entrance, facing the clinic corridor with H1 about 1.1 m ahead of it on its
    way there (he walked up to it to ask): the guide never moved and ended blocked_by_person after 16.7 s."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"H1"})
    w.robot.teleport(12.85, 6.88, math.radians(171))
    goal = w._resolve_place("clinic_corridor")
    put(w, "H1", *along(w, goal, 1.1))
    s.step(0.1)
    outs, gaps = guide(s, "H1", "clinic_corridor")
    assert [e["reason"] for e in guide_end(tmp_path)] == ["arrived"], rows(tmp_path)
    assert not contacts(tmp_path)


def test_the_person_just_guided_on_the_way_back_does_not_block_the_walk_back(staged, tmp_path, body):
    """After the guide the robot turns back for its post; the person it brought stands 0.9 m along its way back. The
    walk back still reaches the post (or ends cleanly after a long wait), with no contact."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1"})
    put(w, "V1", 1.6, 0.0)
    s.step(0.1)
    guide(s, "V1", "ward_entrance")
    assert [e["reason"] for e in guide_end(tmp_path)] == ["arrived"]
    post = s.tasks.post
    put(w, "V1", *along(w, post[:2], 0.9))  # she stops on its way back
    t0 = s.t
    outs = [s.step(0.1) for _ in range(900)]
    back = [r for o in outs for r in o.task_results if r.get("id") == "gd-return"]
    assert back and back[-1]["reason"] in ("arrived", "blocked_by_person"), back
    assert back[-1]["reason"] == "arrived" or back[-1]["t"] - t0 >= 25.0, back  # given up only after a long wait
    assert not contacts(tmp_path)


def test_the_guided_person_holds_the_robot_longer_than_a_bystander_does(staged, tmp_path, monkeypatch):
    """With no way round (routes neither planned nor re-planned round people), the person it guides standing in its
    way does not end the guide within the usual give-up time; a bystander in the same spot ends the walk
    blocked_by_person as before."""
    monkeypatch.setattr(rt, "REPLAN_S", 1e9)
    monkeypatch.setattr(rt.RobotTasks, "_avoid", lambda self, exempt: [])
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1", "V2"})
    goal = w._resolve_place("ward_entrance")
    spot = along(w, goal, 0.85)
    put(w, "V1", *spot)
    put(w, "V2", -3.0, -3.0)
    s.step(0.1)
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("V1"), "place": "ward_entrance"})
    for _ in range(int((rt.GIVE_UP_S + 4.0) / 0.1)):
        s.step(0.1)
    assert s.tasks._guide is not None and not guide_end(tmp_path)  # still waiting for her
    assert gap(w, "V1") >= rt.PERSON_KEEP_M - 0.05

    s2 = svc(staged("reception_guide"), tmp_path / "b")
    w2 = s2.world
    alone_with(w2, {"V1", "V2"})
    r2 = w2.robot
    put(w2, "V1", 2 * r2.x - spot[0], 2 * r2.y - spot[1])  # close behind it (not following: it is not held for her)
    put(w2, "V2", *spot)
    s2.step(0.1)
    s2.on_task_action({"id": "gd", "action": "guide_start", "track": w2.tracks.track("V1"), "place": "ward_entrance"})
    for _ in range(int((rt.GIVE_UP_S + 4.0) / 0.1)):
        s2.step(0.1)
    assert [e["reason"] for e in guide_end(tmp_path / "b")] == ["blocked_by_person"]


def test_a_guide_whose_person_is_already_at_the_place_ends_arrived(staged, tmp_path, monkeypatch, body):
    """She walked on ahead and stands at the place, in the robot's way: the guide is done, not blocked."""
    monkeypatch.setattr(rt, "REPLAN_S", 1e9)
    monkeypatch.setattr(rt.RobotTasks, "_avoid", lambda self, exempt: [])
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1"})
    goal = w._resolve_place("clinic_corridor")
    w.robot.teleport(*along(w, goal, route_m(w, goal) - 2.5), 0.0)  # 2.5 m short of it on its way there
    d0 = math.hypot(w.robot.x - goal[0], w.robot.y - goal[1])
    assert 1.5 < d0 < 3.0, d0
    put(w, "V1", *along(w, goal, 0.85))
    assert math.hypot(w.people["V1"].x - goal[0], w.people["V1"].y - goal[1]) <= rt.NEAR_PLACE_M
    s.step(0.1)
    outs, _ = guide(s, "V1", "clinic_corridor", secs=rt.GIVE_UP_S + 6)
    assert [e["reason"] for e in guide_end(tmp_path)] == ["arrived"], rows(tmp_path)
    goal_rows = [r for o in outs for r in o.task_results if r.get("kind") == "goal" and r["id"] == "gd"]
    assert goal_rows and goal_rows[0]["ok"] and goal_rows[0]["reason"] == "arrived"


# ---- what it says at the end of a guide --------------------------------------------------------------------------
def robot_lines(outs):
    return [a.text for o in outs for a in o.audio if a.source_kind == "robot"]


def test_no_goodbye_when_it_stays_with_them(staged, tmp_path):
    """wandering_patient t=59.92: told to stay (return_to_post false), it still said goodbye."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V2"})
    put(w, "V2", 1.6, 0.0)
    s.step(0.1)
    outs, _ = guide(s, "V2", "ward_entrance", return_to_post=False)
    outs += [s.step(0.1) for _ in range(30)]
    assert [e["reason"] for e in guide_end(tmp_path)] == ["arrived"]
    said = robot_lines(outs)
    assert rt.GOODBYE not in said and rt.STAY_SAY in said
    assert "goodbye" not in rt.STAY_SAY.lower() and rt.STAY_SAY.startswith("Here we are")


def test_goodbye_when_it_walks_back(staged, tmp_path):
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V2"})
    put(w, "V2", 1.6, 0.0)
    s.step(0.1)
    outs, _ = guide(s, "V2", "ward_entrance")
    assert rt.GOODBYE in robot_lines(outs)


# ---- the near-person speed cap during a guide ------------------------------------------------------------------------
def test_the_speed_cap_exempts_only_the_guided_person_behind_the_robot(staged, tmp_path):
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1", "V2"})
    r = w.robot
    r.teleport(5.0, 5.0, 0.0)
    put(w, "V2", -3.0, -3.0)
    put(w, "V1", 3.8, 5.0)  # 1.2 m behind it
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("V1"), "place": "ward_entrance"})
    s.step(0.1)
    assert s.tasks._guide is not None
    r.teleport(5.0, 5.0, 0.0)
    put(w, "V1", 3.8, 5.0)
    assert s.tasks._near_cap() is None  # the person it guides, following behind it: no cap
    put(w, "V2", 6.2, 5.0)  # a bystander 1.2 m ahead
    assert s.tasks._near_cap() == rt.NEAR_PERSON_SPEED
    put(w, "V2", 5.0, 3.8)  # ... or beside it
    assert s.tasks._near_cap() == rt.NEAR_PERSON_SPEED
    put(w, "V2", -3.0, -3.0)
    put(w, "V1", 6.2, 5.0)  # the person it guides, ahead of it: capped like anyone
    assert s.tasks._near_cap() == rt.NEAR_PERSON_SPEED
    s.tasks._end_guide(s.t, "stopped")
    put(w, "V1", 3.8, 5.0)  # no guide: behind it or not, she caps it
    assert s.tasks._near_cap() == rt.NEAR_PERSON_SPEED


# ---- where the guide's walk ends, when the person stands at the place ------------------------------------------------
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_goal_moved_clear_of_the_guided_person_stays_on_the_robots_side_of_them(staged, tmp_path, side):
    """She stands at the place: the walk's goal is moved GUIDE_CLEAR_M from her, on the side the robot comes from, not
    beyond her (it would walk past her, or be held by her, to get there)."""
    s = svc(staged("reception_guide"), tmp_path)
    w = s.world
    alone_with(w, {"V1"})
    goal = w._resolve_place("clinic_corridor")
    put(w, "V1", *goal)
    w.robot.teleport(goal[0] + side * 4.0, goal[1], 0.0)
    if not w.robot.router.route((w.robot.x, w.robot.y), goal):
        pytest.skip("no route from that side")
    s.on_task_action({"id": "gd", "action": "guide_start", "track": w.tracks.track("V1"), "place": "clinic_corridor"})
    s.step(0.1)
    start = [r for r in rows(tmp_path, kind="guide") if r.get("event") == "start"][0]
    q, r = w.people["V1"], w.robot
    gx, gy = start["x"] - q.x, start["y"] - q.y
    assert math.hypot(gx, gy) >= rt.GUIDE_CLEAR_M - 1e-6
    assert gx * (w.robot.x - q.x) + gy * (w.robot.y - q.y) > 0.0, (start, (r.x, r.y))
