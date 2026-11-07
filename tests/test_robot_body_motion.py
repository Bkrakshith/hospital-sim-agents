"""The stub robot body at any step size: no goal fails "stuck" because one sim step was long (x10 runs cap a stalled wall
step at 0.5 s, so 5 s of sim), a body next to furniture is never frozen by its route's first leg, and a contact is the
robot's only when its own motion closed the gap."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import DEFAULT_CONFIG_ROOT, load_scenario_presets, preset_to_scenario
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world.robot_body import RobotBody, STUCK_S
from wardzero_agents.world.stub_world import StubWorld

_PRESETS = {}


def preset(pid, tmp_path):
    if not _PRESETS:
        _PRESETS.update(load_scenario_presets(DEFAULT_CONFIG_ROOT))
    p = _PRESETS[pid]
    return preset_to_scenario(p, DEFAULT_CONFIG_ROOT, tmp_path / "s", characters=auto_staff.with_auto(list(p.characters), p))


def goal_results(out):
    return [(r.get("id"), r.get("reason")) for r in out.task_results if r.get("kind") == "goal"]


# ---- bug 5: a long step spent turning is not "stuck" --------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_one_capped_step_after_a_goal_is_not_stuck(tmp_path, scale):
    """specimen_run: the robot faces away from the goal; the next wall step stalls (MAX_WALL_DT, so 0.5 s x scale)."""
    s = StubSimService(preset("specimen_run", tmp_path), RunLog(tmp_path / "run"), time_scale=scale, wait_for=())
    s.step(0.01)
    x0, y0 = s.world.robot.x, s.world.robot.y
    s.on_goal(10.0, 2.0, None)
    got = goal_results(s.step(0.5))
    for _ in range(int(3.0 / (0.05 * scale))):
        got += goal_results(s.step(0.05))
    assert ("goal_pose-1", "stuck") not in got
    assert math.hypot(s.world.robot.x - x0, s.world.robot.y - y0) > 1.0


def test_a_body_step_longer_than_stuck_s_turns_then_walks(tmp_path):
    w = StubWorld(preset("specimen_run", tmp_path), 7)
    b = w.robot
    b.teleport(15.0, 7.5, 0.0)  # facing east; the goal is behind it to the west
    assert b.set_goal(10.0, 2.0)
    b.step(STUCK_S + 0.5, w.site.grid)
    assert b.take_goal_event() != "stuck"
    assert math.hypot(b.x - 15.0, b.y - 7.5) > 2.0  # it turned round and walked most of that step


def test_turning_toward_the_route_counts_as_progress(tmp_path):
    """A slow-turning body (a whole STUCK_S spent turning on the spot towards its route) is not stuck."""
    import wardzero_agents.world.robot_body as rb
    w = StubWorld(preset("specimen_run", tmp_path), 7)
    b = w.robot
    b.teleport(15.0, 7.5, 0.0)
    assert b.set_goal(10.0, 2.0)
    old = rb.TURN_RATE
    rb.TURN_RATE = 0.3  # 6.5 s on the spot before it may walk (heading error under 1.2 rad)
    try:
        for _ in range(int(STUCK_S / 0.05) + 10):
            b.step(0.05, w.site.grid)
            assert b.goal_event != "stuck"
    finally:
        rb.TURN_RATE = old


def test_a_blocked_body_still_ends_stuck(tmp_path):
    w = StubWorld(preset("specimen_run", tmp_path), 7)
    b = w.robot
    b.teleport(15.0, 7.5, math.pi)
    assert b.set_goal(10.0, 2.0)
    b.max_speed = 0.0  # cannot move at all
    events = []
    for _ in range(int((STUCK_S + 1.0) / 0.5)):
        b.step(0.5, w.site.grid)
        events.append(b.take_goal_event())
    assert "stuck" in events


def test_night_call_bell_tell_staff_at_x10_does_not_stick(tmp_path):
    """QA harness_tell4: approach P1, then tell_staff; the step after the action is a capped one (5 s at x10)."""
    s = StubSimService(preset("night_call_bell", tmp_path), RunLog(tmp_path / "run"), 10.0, wait_for=())
    w = s.world
    tr = w.tracks.track("P1")
    s.on_task_action({"id": "ap", "action": "approach", "track": tr})
    for _ in range(1200):
        out = s.step(0.01)
        if any(r.get("id") == "ap" and r.get("kind") == "goal" for r in out.task_results):
            break
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr,
                      "message": "The patient in bed B2 rang her call bell and needs a nurse, please."})
    res = list(s.step(0.5).task_results)
    for _ in range(600):
        res += s.step(0.01).task_results
        if any(r.get("id") == "tl" and r.get("kind") != "goal" for r in res):
            break
    assert ("tl", "stuck") not in [(r.get("id"), r.get("reason")) for r in res], res


# ---- bug 6: trapped beside furniture ------------------------------------------------------------------------------
TRAP = (14.878, 24.163, -0.5)  # night_call_bell, bed bay B6: QA harness_trap.py


def _segments_clear(grid, pts, step=0.02):
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for k in range(n + 1):
            if grid.blocked(ax + (bx - ax) * k / n, ay + (by - ay) * k / n):
                return False
    return True


def test_the_route_from_the_trapped_pose_never_cuts_a_blocked_cell(tmp_path):
    w = StubWorld(preset("night_call_bell", tmp_path), 7)
    route = w.robot.router.route(TRAP[:2], (19.0, 15.0))
    assert route is not None and _segments_clear(w.site.grid, route)


def test_the_body_at_the_trapped_pose_walks_away(tmp_path):
    w = StubWorld(preset("night_call_bell", tmp_path), 7)
    b = w.robot
    b.teleport(*TRAP)
    assert b.set_goal(19.0, 15.0, 1.2)
    for _ in range(int(30 / 0.05)):
        b.step(0.05, w.site.grid)
        if not b.has_goal:
            break
    assert b.take_goal_event() == "arrived"
    assert math.hypot(b.x - 19.0, b.y - 15.0) < 0.3


def test_a_body_inside_a_blocked_cell_walks_out_on_its_route(tmp_path):
    w = StubWorld(preset("night_call_bell", tmp_path), 7)
    g = w.site.grid
    # a blocked cell with free floor within a metre
    spot = None
    for iy in range(g.occ.shape[0]):
        for ix in range(g.occ.shape[1]):
            if g.occ[iy, ix]:
                x, y = g.to_world(iy, ix)
                if not g.blocked(x + 0.4, y) and not g.blocked(x + 0.8, y) and 10 < x < 25 and 15 < y < 25:
                    spot = (x, y)
                    break
        if spot:
            break
    assert spot is not None
    b = w.robot
    b.teleport(spot[0], spot[1], 0.0)
    assert b.set_goal(19.0, 15.0)
    for _ in range(int(40 / 0.05)):
        b.step(0.05, g)
        if not b.has_goal:
            break
    assert b.take_goal_event() == "arrived"


# ---- contract item 6: a contact is the robot's only when its own motion closed the gap -----------------------------
def test_a_person_walking_into_a_standing_robot_is_not_a_robot_contact():
    class Grid:
        def blocked(self, x, y):
            return False
    b = RobotBody(0.0, 0.0, 0.0)
    assert b.step(0.1, Grid(), [("P1", 2.0, 0.0)]) == []
    assert b.step(0.1, Grid(), [("P1", 0.4, 0.0)]) == []  # P1 walked up to 0.4 m
    b.set_cmd_vel(-0.5, 0.0)  # backing away is not a contact either
    assert b.step(0.1, Grid(), [("P1", 0.3, 0.0)]) == []


def test_a_robot_driving_into_a_person_is_a_contact_once():
    class Grid:
        def blocked(self, x, y):
            return False
    b = RobotBody(0.0, 0.0, 0.0)
    b.set_cmd_vel(1.0, 0.0)
    got = []
    for _ in range(10):
        got += b.step(0.1, Grid(), [("P1", 1.2, 0.0)])
    assert got == ["P1"]


def test_a_robot_that_drives_on_into_a_person_who_walked_up_is_a_contact():
    class Grid:
        def blocked(self, x, y):
            return False
    b = RobotBody(0.0, 0.0, 0.0)
    assert b.step(0.1, Grid(), [("P1", 0.5, 0.0)]) == []  # P1 stepped in close: not the robot's doing
    b.set_cmd_vel(0.5, 0.0)
    assert b.step(0.1, Grid(), [("P1", 0.5, 0.0)]) == ["P1"]  # the robot closes in on her: its contact


def test_a_replan_keeps_the_avoid_discs_the_goal_was_given():
    """Blocked on its route for REPLAN_S, the body plans again from where it stands round the same people (avoid
    discs) its goal was set with, not straight through them."""
    from wardzero_agents.world.robot_body import REPLAN_S

    class Grid:
        def blocked(self, x, y):
            return x > 0.5  # a wall ahead: the body cannot move along its first leg

    class Router:
        def __init__(self):
            self.calls = []

        def route(self, a, b, avoid=()):
            self.calls.append(tuple(avoid))
            return [a, (2.0, 0.0), b]

    r = Router()
    b = RobotBody(0.4, 0.0, 0.0, router=r)
    disc = (3.0, 1.0, 0.6)
    assert b.set_goal(4.0, 0.0, avoid=[disc])
    for _ in range(int(REPLAN_S / 0.1) + 2):
        b.step(0.1, Grid())
    assert len(r.calls) >= 2  # it planned again
    assert all(c == (disc,) for c in r.calls), r.calls
