"""Route follower: heading loop against a planner that stops turns short, slow zones, arrived / stuck / fell,
the 2D unicycle adapter and the GEAR-SONIC command adapter."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.sonic_fakes import fake_sessions
from wardzero_agents.isaac_host.sonic import DEFAULT_ANGLES, Phase, SonicController, SonicRunner
from wardzero_agents.isaac_host.sonic import params as P
from wardzero_agents.site.model import OccGrid
from wardzero_agents.site.routing import Router
from wardzero_agents.world.robot_body import RobotBody
from wardzero_agents.world import route_follower as rf
from wardzero_agents.world.route_follower import FollowerCommand, RouteFollower, UnicycleDrive, wrap


def two_rooms() -> OccGrid:
    """12 x 6 m, 0.1 m cells, a wall at x = 6 m with a 1.2 m doorway at y = 2.4..3.6 m."""
    occ = np.zeros((60, 120), np.uint8)
    occ[[0, -1], :] = 1
    occ[:, [0, -1]] = 1
    occ[:, 60] = 1
    occ[24:36, 60] = 0
    return OccGrid(occ, 0.1, (0.0, 0.0))


class SonicLike:
    """A walking robot that behaves like SONIC's planner: an in-place turn to a new facing stops at 90 % of it
    (81 deg for 90 deg), yaw moves at 1.5 rad/s, and walking reaches ~0.73 of the commanded speed."""

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self._face = None
        self._goal = yaw

    def apply(self, cmd: FollowerCommand, dt: float):
        if cmd.face_yaw is not None and (self._face is None or abs(wrap(cmd.face_yaw - self._face)) > 1e-9):
            self._face = cmd.face_yaw
            self._goal = self.yaw + 0.9 * wrap(cmd.face_yaw - self.yaw)
        e = wrap(self._goal - self.yaw)
        self.yaw = wrap(self.yaw + max(-1.5 * dt, min(1.5 * dt, e)))
        if cmd.mode == rf.SLOW_WALK:
            v = 0.73 * cmd.speed
            self.x += v * dt * math.cos(self.yaw)
            self.y += v * dt * math.sin(self.yaw)

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)


def drive(f: RouteFollower, robot: SonicLike, seconds: float, dt: float = 0.1, **kw):
    t, cmds = 0.0, []
    while t < seconds:
        c = f.update(t, robot.pose, **kw)
        cmds.append(c)
        if c.status in rf.RESULTS:
            return c, t, cmds
        robot.apply(c, dt)
        t += dt
    return c, t, cmds


def test_heading_loop_closes_the_planner_turn_undershoot():
    robot = SonicLike(0.0, 0.0, 0.0)
    f = RouteFollower()
    f.set_route([(0.0, 0.0)], final_yaw=math.pi / 2)
    c, t, cmds = drive(f, robot, 20.0)
    assert c.status == rf.ARRIVED and t < 6.0
    assert abs(wrap(robot.yaw - math.pi / 2)) <= rf.YAW_TOL
    assert all(x.mode == rf.IDLE for x in cmds)  # in place, IDLE with a facing
    assert f.corrections >= 1  # re-issued from the measured yaw after the short turn
    assert max(x.face_yaw for x in cmds if x.face_yaw is not None) > math.pi / 2  # asked past 90 to get 90
    assert f.take_event() == rf.ARRIVED and f.take_event() is None


def test_without_the_heading_loop_the_turn_stops_short_and_reports_stuck():
    robot = SonicLike(0.0, 0.0, 0.0)
    f = RouteFollower(correct_gain=1.0)  # re-issuing the same facing changes nothing
    f.set_route([(0.0, 0.0)], final_yaw=math.pi / 2)
    c, t, _ = drive(f, robot, 20.0)
    assert math.degrees(robot.yaw) == pytest.approx(81.0, abs=0.5)
    assert c.status == rf.STUCK and 5.0 <= t < 7.0  # 5 s of sim time without progress


def test_walks_a_route_with_turns_and_ends_facing_the_final_yaw():
    robot = SonicLike(1.0, 1.0, 0.0)
    f = RouteFollower()
    f.set_route([(4.0, 1.0), (4.0, 4.0), (1.0, 4.0)], final_yaw=math.pi)
    c, t, cmds = drive(f, robot, 60.0)
    assert c.status == rf.ARRIVED, (robot.pose, t)
    assert math.hypot(robot.x - 1.0, robot.y - 4.0) < rf.ARRIVE_M + 0.05
    assert abs(wrap(robot.yaw - math.pi)) <= rf.YAW_TOL
    walking = [x for x in cmds if x.mode == rf.SLOW_WALK]
    assert walking and all(x.speed == rf.OPEN_SPEED and x.move_yaw == x.face_yaw for x in walking)
    assert any(x.status == rf.TURNING for x in cmds)  # 90 deg corners turn in place first


def test_slow_near_people_only_not_doors():
    f = RouteFollower()
    f.set_route([(10.0, 0.0)])
    assert f.update(0.0, (0.0, 0.0, 0.0)).speed == 0.6
    assert f.update(0.1, (0.0, 0.0, 0.0), doors=[(1.4, 0.3)]).speed == 0.6  # door spike: 0.3 m/s stalls at a leaf
    c = f.update(0.2, (0.0, 0.0, 0.0), people=[(0.0, -1.5)])
    assert c.speed == 0.3 and c.slow
    assert f.update(0.3, (0.0, 0.0, 0.0), doors=[(1.6, 0.0)], people=[(-1.2, 1.0)]).speed == 0.6
    g = RouteFollower(slow_near_doors=True)  # the old behaviour, still available
    g.set_route([(10.0, 0.0)])
    assert g.update(0.0, (0.0, 0.0, 0.0), doors=[(1.4, 0.3)]).speed == 0.3


def test_a_waypoint_behind_turns_in_place_before_walking():
    f = RouteFollower()
    f.set_route([(-3.0, 0.0)])
    c = f.update(0.0, (0.0, 0.0, 0.0))
    assert c.mode == rf.IDLE and c.status == rf.TURNING and abs(wrap(c.face_yaw - math.pi)) < 1e-9
    c = f.update(0.1, (0.0, 0.0, math.radians(150)))  # still 30 deg off: keeps turning (hysteresis)
    assert c.mode == rf.IDLE
    c = f.update(0.2, (0.0, 0.0, math.radians(165)))
    assert c.mode == rf.SLOW_WALK


def test_fell_and_no_goal():
    f = RouteFollower()
    assert f.update(0.0, (0, 0, 0)).status == rf.NO_GOAL
    f.set_route([(5.0, 0.0)])
    f.update(0.0, (0, 0, 0))
    c = f.update(0.1, (0.1, 0, 0), fallen=True)
    assert c.status == rf.FELL and c.mode == rf.IDLE and not f.active
    assert f.update(0.2, (0.1, 0, 0)).status == rf.FELL and f.take_event() == rf.FELL
    f.set_route([(5.0, 0.0)])
    f.cancel()
    assert f.update(0.3, (0, 0, 0)).status == rf.NO_GOAL and f.take_event() is None


def test_router_goal_through_a_doorway_with_the_2d_body():
    grid = two_rooms()
    router = Router(grid)
    body = RobotBody(2.0, 1.5, 0.0, router=router)
    drive2d = UnicycleDrive(body)
    assert drive2d.set_goal(10.0, 4.5, yaw=math.pi / 2)
    t, dt, statuses = 0.0, 0.05, []
    door = [(6.0, 3.0)]
    while t < 60.0:
        drive2d.step(t, dt, grid, people=[], doors=door)
        statuses.append((drive2d.last.status, drive2d.last.speed, (body.x, body.y)))
        if drive2d.last.status in rf.RESULTS:
            break
        t += dt
    assert drive2d.last.status == rf.ARRIVED, body.pose
    assert math.hypot(body.x - 10.0, body.y - 4.5) < rf.ARRIVE_M + 0.05
    assert abs(wrap(body.yaw - math.pi / 2)) <= rf.YAW_TOL
    through = [s for s in statuses if abs(s[2][0] - 6.0) < 0.5 and s[0] == rf.FOLLOWING]
    assert through and all(s[1] == rf.OPEN_SPEED for s in through)  # through the doorway at walking speed


def test_a_guided_person_does_not_slow_the_2d_body_but_others_do():
    grid = two_rooms()
    body = RobotBody(2.0, 1.5, 0.0, router=Router(grid))
    d = UnicycleDrive(body)
    d.follower.set_route([(5.0, 1.5)])
    d.step(0.0, 0.05, grid, people=[("G", 1.5, 1.5)], exempt={"G"})
    assert d.last.speed == rf.OPEN_SPEED
    d.step(0.05, 0.05, grid, people=[("G", 1.5, 1.5), ("P2", 2.5, 2.5)], exempt={"G"})
    assert d.last.speed == rf.SLOW_SPEED


def test_stuck_against_a_wall_after_five_sim_seconds():
    grid = two_rooms()
    body = RobotBody(5.0, 1.0, 0.0, router=Router(grid))
    d = UnicycleDrive(body)
    d.follower.set_route([(8.0, 1.0)])  # straight through the wall, no doorway
    t, last_move, x = 0.0, 0.0, body.x
    while t < 20.0 and (d.last is None or d.last.status not in rf.RESULTS):
        d.step(t, 0.05, grid)
        if body.x != x:
            last_move, x = t, body.x
        t += 0.05
    assert d.last.status == rf.STUCK and 5.5 < body.x < 6.0
    assert 5.0 <= t - last_move <= 5.2  # 5 s of sim time after it stopped


def test_no_route_means_no_goal():
    grid = two_rooms()
    f = RouteFollower(Router(grid))
    assert not f.set_goal((2.0, 1.5), (30.0, 3.0)) and not f.active


def test_sonic_runner_turns_follower_commands_into_planner_frame_commands():
    c = SonicController(*fake_sessions())
    r = SonicRunner(c, ramp_s=0.0, settle_s=0.0, idle_s=0.0)
    psi = math.radians(40)  # the robot faces 40 deg in the world when the planner starts
    quat = np.array([math.cos(psi / 2), 0, 0, math.sin(psi / 2)])
    for k in range(3):
        out = r.tick(0.02 * k, DEFAULT_ANGLES, np.zeros(29), quat, np.zeros(3), 0.78)
    assert out.phase == Phase.READY
    cmd = r.follow(FollowerCommand(rf.SLOW_WALK, 0.3, psi + math.radians(91), psi + math.radians(91), rf.FOLLOWING))
    assert cmd.mode == P.SLOW_WALK and cmd.speed == 0.3
    assert math.degrees(math.atan2(cmd.facing[1], cmd.facing[0])) == pytest.approx(91.0)
    snapped = r.follow(FollowerCommand(rf.SLOW_WALK, 0.3, psi + math.radians(91), psi + math.radians(91),
                                       rf.FOLLOWING), quantum_rad=math.radians(5))
    assert math.degrees(math.atan2(snapped.facing[1], snapped.facing[0])) == pytest.approx(90.0)
    assert cmd.movement == cmd.facing
    idle = r.follow(FollowerCommand(rf.IDLE, 0.0, None, None, rf.ARRIVED))
    assert idle.mode == P.IDLE and idle.movement == (0.0, 0.0, 0.0) and idle.facing == snapped.facing


def test_the_follower_imports_no_simulator():
    code = ("import sys, wardzero_agents.world.route_follower; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'numpy', 'onnxruntime', 'omni', 'pxr', 'rclpy'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]"


def test_a_route_through_a_swinging_door_passes_straight_through_one_leaf():
    door = ((6.0, 2.0), (6.0, 4.0))  # a 2 m double door on x = 6
    route = [(2.0, 3.0), (10.0, 3.0)]  # the router's line goes through the meeting stiles at y = 3
    out = rf.through_leaves(route, [door])
    assert out[0] == (2.0, 3.0) and out[-1] == (10.0, 3.0)
    pre, post = out[1], out[2]
    assert pre[0] == pytest.approx(4.0) and post[0] == pytest.approx(7.5)  # 2 m before, 1.5 m after
    assert pre[1] == pytest.approx(post[1]) and pre[1] in (pytest.approx(2.5), pytest.approx(3.5))  # a leaf's centre
    back = rf.through_leaves([(10.0, 2.6), (2.0, 2.6)], [door])
    assert back[1] == (pytest.approx(8.0), pytest.approx(2.5)) and back[2][0] == pytest.approx(4.5)
    single = rf.through_leaves([(2.0, 3.0), (10.0, 3.2)], [((6.0, 2.5), (6.0, 3.5))])
    assert single[1][1] == pytest.approx(3.0)  # one leaf: its centre
    assert rf.through_leaves(route, []) == route
    blocked = rf.through_leaves(route, [door], free=lambda x, y: x > 4.6)
    assert blocked[1][0] >= 4.6  # shortened when the approach point is blocked


def test_people_do_not_slow_the_robot_while_it_pushes_through_a_door():
    f = RouteFollower()
    f.door_lines = [((6.0, 2.0), (6.0, 4.0))]
    f.set_route([(10.0, 2.5)])
    assert f.update(0.0, (5.2, 2.5, 0.0), people=[(4.5, 3.5)]).speed == rf.OPEN_SPEED  # in the door zone
    assert f.update(0.1, (2.0, 2.5, 0.0), people=[(2.5, 3.5)]).speed == rf.SLOW_SPEED  # away from it
