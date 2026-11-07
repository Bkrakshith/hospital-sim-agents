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
