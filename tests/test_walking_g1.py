"""The walking G1 in the Isaac host, the CPU parts: the frame / physics split, the joint settings, the profile switch,
the LeggedBody (goals in, follower commands out, arrived / stuck / fell back on the goal-result path) and its glue to
SONIC's planner frame with a fake controller, the headed follow camera, the scripted go_to hook and the wiring order."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wardzero_agents.isaac_host import walking as W
from wardzero_agents.isaac_host.sonic import params as P
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile, g1_walks_in_isaac, profile_for_sim
from wardzero_agents.site.model import OccGrid
from wardzero_agents.site.routing import Router
from wardzero_agents.world import route_follower as rf
from wardzero_agents.world.legged_body import LeggedBody, door_points
from wardzero_agents.world.robot_tasks import RobotTasks

ROOT = Path(__file__).resolve().parents[1]


# ---- frames and physics ------------------------------------------------------------------------------------------
def test_a_walking_robot_gets_200_hz_physics_in_whole_steps_a_frame():
    assert W.frame_plan(None, walking=True) == (25, 200)
    assert W.steps_per_frame(25, 200) == 8 and W.DECIMATION == 4
    assert W.frame_plan(40, walking=True) == (40, 200)
    with pytest.raises(ValueError, match="divide 200"):
        W.frame_plan(30, walking=True)
    assert W.frame_plan(None, walking=False) == (30, None)  # nothing changes without a walking robot
    assert W.frame_plan(24, walking=False) == (24, None)


def test_the_host_args_follow_the_robot(monkeypatch):
    from wardzero_agents.isaac_host import app as A
    a = A.parse_args(["--run-dir", "r", "--scenario", "s"])
    assert a.fps == 30 and not a.fps_given and a.physics_hz is None
    legged = profile_for_sim_forced(monkeypatch, True)
    A.apply_frame_plan(a, legged)
    assert (a.fps, a.physics_hz) == (25, 200)
    b = A.parse_args(["--run-dir", "r", "--scenario", "s", "--fps", "50"])
    A.apply_frame_plan(b, legged)
    assert (b.fps, b.physics_hz) == (50, 200)
    c = A.parse_args(["--run-dir", "r", "--scenario", "s"])
    A.apply_frame_plan(c, G1_CLAUDE)  # the static G1
    assert (c.fps, c.physics_hz) == (30, None)
    d = A.parse_args(["--run-dir", "r", "--scenario", "s", "--fps", "30"])
    with pytest.raises(ValueError):
        A.apply_frame_plan(d, legged)


def test_one_sonic_tick_every_fourth_physics_step():
    s = W.StepSchedule()
    ticks = []
    for _ in range(16):
        ticks.append(s.tick_now())
        s.advance()
    assert [i for i, t in enumerate(ticks) if t] == [0, 4, 8, 12]
    assert s.t == pytest.approx(16 * 0.005)


def test_joint_settings_zero_gain_body_joints_with_our_effort_and_stiff_fingers():
    names = list(P.MJ_JOINTS) + ["left_thumb_metacarpal_joint", "right_index_proximal_joint"]
    st = {s.name: s for s in W.joint_settings(names)}
    body = [st[n] for n in P.MJ_JOINTS]
    assert all(s.stiffness == 0.0 and s.damping == 0.0 for s in body)  # the PD is the walker's
    assert st["left_ankle_pitch_joint"].max_force == 35.0 and st["waist_pitch_joint"].max_force == 35.0
    assert st["left_hip_pitch_joint"].max_force == 88.0
    assert st["left_hip_pitch_joint"].armature == pytest.approx(P.ARM_7520_14)
    assert st["left_knee_joint"].armature == pytest.approx(P.ARM_7520_22)
    finger = st["left_thumb_metacarpal_joint"]
    assert finger.stiffness > 0 and finger.max_force is None and finger.armature is None
    assert W.is_hand_link("left_index_distal_link") and W.is_leg_link("right_knee_link")
    assert not W.is_hand_link("left_knee_link") and not W.is_leg_link("torso_link")


# ---- the planner frame -------------------------------------------------------------------------------------------
def _ang(v):
    return math.atan2(v[1], v[0])


def test_follower_commands_go_to_the_planner_frame_of_the_start_heading():
    yaw0 = math.radians(-90)  # the G1 faces south at reception when the planner starts
    walk = rf.FollowerCommand(rf.SLOW_WALK, 0.6, 0.0, 0.0, rf.FOLLOWING)  # walk east, facing east
    c = W.planner_command(walk, yaw0)
    assert c.mode == P.SLOW_WALK and c.speed == 0.6
    assert math.degrees(_ang(c.facing)) == pytest.approx(90.0) and c.movement == c.facing
    idle = W.planner_command(rf.FollowerCommand(rf.IDLE, 0.0, None, None, rf.ARRIVED), yaw0, hold_facing=c.facing)
    assert idle.mode == P.IDLE and idle.movement == (0.0, 0.0, 0.0) and idle.facing == c.facing  # keeps facing
    turn = W.planner_command(rf.FollowerCommand(rf.IDLE, 0.0, None, math.pi, rf.TURNING), yaw0)
    assert turn.mode == P.IDLE and math.degrees(_ang(turn.facing)) == pytest.approx(-90.0)  # 180 - (-90) = 270
    assert W.planner_command(walk, yaw0) == c  # the same command twice: no re-plan


# ---- the profile switch ------------------------------------------------------------------------------------------
def profile_for_sim_forced(monkeypatch, walks: bool) -> RobotProfile:
    monkeypatch.setenv("WARDZERO_G1_WALKS_IN_ISAAC", "1" if walks else "0")
    return profile_for_sim(G1_CLAUDE, "isaac")


def _fake_weights(root: Path) -> None:
    from wardzero_agents.isaac_host.sonic import weights as Wt
    for f in Wt.FILES:
        p = root / f.rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.truncate(f.size)  # sparse: the size is what the quick check looks at


def test_the_g1_walks_in_isaac_only_with_the_sonic_weights(tmp_path, monkeypatch):
    from wardzero_agents.robots.claude_robot import actions_for, system_prompt
    monkeypatch.delenv("WARDZERO_G1_WALKS_IN_ISAAC", raising=False)
    monkeypatch.setenv("WARDZERO_SONIC_DIR", str(tmp_path / "none"))
    assert not g1_walks_in_isaac()
    assert profile_for_sim(G1_CLAUDE, "isaac") is G1_CLAUDE  # static, as before
    assert "go_to" not in actions_for(profile_for_sim(G1_CLAUDE, "isaac"))
    _fake_weights(tmp_path / "w")
    monkeypatch.setenv("WARDZERO_SONIC_DIR", str(tmp_path / "w"))
    assert g1_walks_in_isaac()
    p = profile_for_sim(G1_CLAUDE, "isaac")
    assert p.can_move and p.base == "legged" and p.max_speed == 0.6 and p.name == G1_CLAUDE.name
    assert {"go_to", "guide"} <= set(actions_for(p)) and "You can walk" in system_prompt(p)
    assert profile_for_sim(G1_CLAUDE, "stub").can_move  # Quick 2D: the G1 moves there whatever the weights
    other = RobotProfile("mybot", False, "static", 0.3, (0.5, 1.0), 1.0, "mybot", ())
    assert profile_for_sim(other, "isaac") is other
    monkeypatch.setenv("WARDZERO_G1_WALKS_IN_ISAAC", "0")
    assert profile_for_sim(G1_CLAUDE, "isaac") is G1_CLAUDE


def test_the_robot_config_carries_the_simulator(tmp_path):
    from wardzero_agents.config.loader import load_scenario
    from wardzero_agents.robots.view import load_robot_view, write_robot_config
    loaded = load_scenario(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")
    assert load_robot_view(write_robot_config(tmp_path, loaded, domain=None, budget_usd=1.0, sim="isaac")).sim == "isaac"
    assert load_robot_view(write_robot_config(tmp_path, loaded, domain=None, budget_usd=1.0)).sim == "stub"


def test_the_walking_g1_gets_a_floating_base_ready_usd(tmp_path, monkeypatch):
    from wardzero_agents.robots import ready_usd as ru
    legged = profile_for_sim_forced(monkeypatch, True)
    assert ru.floating_base(legged) and not ru.floating_base(G1_CLAUDE)
    assert ru.ready_usd_path(legged, tmp_path, True).name == "robot_floating.usd"
    assert ru.ready_usd_path(G1_CLAUDE, tmp_path).name == "robot.usd"
    cmd = ru.convert_command(Path("g.urdf"), Path("s.yaml"), Path("o.usd"), python="py", floating=True)
    assert cmd[-1] == "--floating-base" and "--floating-base" not in ru.convert_command(
        Path("g.urdf"), Path("s.yaml"), Path("o.usd"), python="py")


# ---- the LeggedBody and a fake SONIC ----------------------------------------------------------------------------------
def corridor() -> OccGrid:
    """16 x 6 m, 0.1 m cells, walls round it and a desk block at x 5..6, y 0..2.4."""
    occ = np.zeros((60, 160), np.uint8)
    occ[[0, -1], :] = 1
    occ[:, [0, -1]] = 1
    occ[0:24, 50:60] = 1
    return OccGrid(occ, 0.1, (0.0, 0.0))


class FakeSonic:
    """Stands in for SONIC + PhysX: takes planner-frame commands (as the walker sends them), turns towards the facing
    by 90 % of each new facing step (the planner's undershoot) and walks at ~0.75 of the commanded speed."""

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self.yaw0 = yaw
        self._face = None
        self._goal = yaw
        self.commands = []

    def apply(self, cmd, dt):
        if not self.commands or self.commands[-1] != cmd:
            self.commands.append(cmd)
        face = self.yaw0 + _ang(cmd.facing)
        if self._face is None or abs(rf.wrap(face - self._face)) > 1e-9:
            self._face = face
            self._goal = self.yaw + 0.9 * rf.wrap(face - self.yaw)
        e = rf.wrap(self._goal - self.yaw)
        self.yaw = rf.wrap(self.yaw + max(-1.5 * dt, min(1.5 * dt, e)))
        if cmd.mode == P.SLOW_WALK:
            move = self.yaw0 + _ang(cmd.movement)
            v = 0.75 * cmd.speed
            self.x += v * dt * math.cos(move)
            self.y += v * dt * math.sin(move)


def run_walk(body: LeggedBody, sonic: FakeSonic, grid, seconds: float, dt: float = 0.04, tasks=None, people=()):
    """The host's frame: the world steps the body (follower at 10 Hz), the walker converts and applies the command,
    and the measured pose goes back into the body (read_back)."""
    t, results = 0.0, []
    facing = (1.0, 0.0, 0.0)
    while t < seconds:
        body.step(dt, grid, list(people))
        cmd = W.planner_command(body.command, sonic.yaw0, facing)
        facing = cmd.facing
        sonic.apply(cmd, dt)
        body.x, body.y, body.yaw = sonic.x, sonic.y, sonic.yaw
        if tasks is not None:
            results += tasks.step(t, dt).results
        t += dt
        if tasks is None and body.goal_event is not None:
            break
    return t, results


def make_body(x=1.0, y=4.0, yaw=0.0, ready=True):
    grid = corridor()
    body = LeggedBody(x, y, yaw, router=Router(grid, clearance=rf.G1_CLEARANCE_M), doors=[(10.0, 2.0)])
    body.ready = ready
    return body, FakeSonic(x, y, yaw), grid


def test_a_goal_walks_past_the_desk_and_arrives_facing_the_goal_yaw():
    body, sonic, grid = make_body()
    assert body.set_goal(14.0, 1.0, yaw=-math.pi / 2) and body.has_goal
    t, _ = run_walk(body, sonic, grid, 120.0)
    assert body.take_goal_event() == rf.ARRIVED, (sonic.x, sonic.y, t)
    assert math.hypot(sonic.x - 14.0, sonic.y - 1.0) < rf.ARRIVE_M + 0.05
    assert abs(rf.wrap(sonic.yaw + math.pi / 2)) <= rf.YAW_TOL
    walking = [c for c in sonic.commands if c.mode == P.SLOW_WALK]
    assert walking and all(c.speed == pytest.approx(0.6) for c in walking)  # doors do not slow it (door spike)
    assert len(sonic.commands) < 200  # the facing is held steady: the planner is not asked again every tick


def test_the_g1_route_from_reception_keeps_clear_of_the_desk_end():
    """The gate test's G1 walked in place against the end of the reception desk on the way back from the ward doors
    (a 0.3 m clearance route). With the G1's band, every point of both legs, away from the post itself (which is
    behind the desk), stays at least 0.4 m from anything the map blocks."""
    from wardzero_agents.site.model import load_site
    site = load_site(ROOT / "config")
    g = site.grid
    oy, ox = np.nonzero(g.occ.astype(bool))
    blocked = np.array([g.to_world(int(a), int(b)) for a, b in zip(oy, ox)])
    post, doors = (1.05, 1.62), site.resolve("ward_doors")

    def mid_clearance(path):
        pts, s = [], 0.0
        for a, b in zip(path, path[1:]):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1))
            pts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
        far = [p for p in pts if math.hypot(p[0] - post[0], p[1] - post[1]) > 0.8
               and math.hypot(p[0] - doors[0], p[1] - doors[1]) > 0.8]
        return min(float(np.min(np.hypot(blocked[:, 0] - x, blocked[:, 1] - y))) - g.res / 2 for x, y in far)

    wide, narrow = Router(g, clearance=rf.G1_CLEARANCE_M), Router(g)
    for a, b in ((post, doors), (doors, post)):
        assert mid_clearance(wide.route(a, b)) >= 0.33
    assert mid_clearance(wide.route(doors, post)) > mid_clearance(narrow.route(doors, post)) + 0.1


def test_cross_track_correction_brings_a_pushed_robot_back_to_the_line():
    f = rf.RouteFollower()
    f.set_route([(10.0, 0.0)])
    f.update(0.0, (0.0, 0.0, 0.0))
    c = f.update(0.1, (2.0, 0.4, 0.0))  # pushed 0.4 m to the left of the line
    assert c.mode == rf.SLOW_WALK and math.degrees(c.move_yaw) < -15  # steers back, not along the old bearing
    assert math.degrees(rf.wrap(math.atan2(-0.4, 8.0))) > -5  # what aiming at the waypoint alone would give


def test_goal_results_reach_the_task_layer_and_a_fall_ends_the_robots_part():
    body, sonic, grid = make_body()
    world = SimpleNamespace(robot=body, robot_results=[], objects=SimpleNamespace(fires={}), people={})
    tasks = RobotTasks(world)
    tasks.goal_pose(8.0, 4.0, None, 0.0)
    _, results = run_walk(body, sonic, grid, 40.0, tasks=tasks)
    goals = [r for r in results if r.get("kind") == "goal"]
    assert goals and goals[0]["ok"] and goals[0]["reason"] == "arrived" and goals[0]["id"] == "goal_pose-1"
    tasks.goal_pose(2.0, 4.0, None, 50.0)
    body.step(0.04, grid, [])
    body.report_fall("tilted 70 deg (> 60 deg)")
    out = tasks.step(50.1, 0.1).results
    assert out == [{"kind": "goal", "id": "goal_pose-2", "action": "goal", "ok": False, "reason": "fell", "t": 50.1}]
    assert body.fallen and body.command.mode == rf.IDLE and not body.has_goal
    tasks.goal_pose(5.0, 4.0, None, 51.0)  # a fallen robot takes no more walks: the brain is told why
    assert tasks.step(51.1, 0.1).results[0]["reason"] == "fell"


def test_stuck_after_five_sim_seconds_without_progress_but_not_while_starting_up():
    body, sonic, grid = make_body(ready=False)
    body.set_goal(14.0, 4.0)
    for _ in range(300):  # 12 s before the walker is ready: held, never stuck
        body.step(0.04, grid, [])
    assert body.goal_event is None and body.command.status == "holding" and body.has_goal
    body.ready = True
    t = 0.0
    while body.goal_event is None and t < 20:  # the legs never move it (a fake that is wedged)
        body.step(0.04, grid, [])
        t += 0.04
    assert body.take_goal_event() == rf.STUCK and 5.0 <= t <= 5.5


def test_people_and_doors_slow_the_robot_but_not_the_person_it_guides():
    body, sonic, grid = make_body()
    body.set_goal(8.0, 4.0)
    body.step(0.04, grid, [("G", 1.5, 4.5)])
    assert body.command.speed == 0.3
    body.exempt = {"G"}
    body._next_follow = 0.0
    body.step(0.04, grid, [("G", 1.5, 4.5)])
    assert body.command.speed == 0.6
    assert door_points(SimpleNamespace(doors=[{"a": [0, 0], "b": [2, 0]}, {"bad": 1}])) == [(1.0, 0.0)]


def test_contacts_and_measured_speed_come_from_the_read_back_pose():
    body, sonic, grid = make_body()
    assert body.step(0.04, grid, [("P", 1.3, 4.0)]) == []  # within reach, but the robot did not close the gap
    body.x += 0.02  # Isaac moved the body towards her
    assert body.step(0.04, grid, [("P", 1.3, 4.0)]) == ["P"]
    assert body.step(0.04, grid, [("P", 1.3, 4.0)]) == []  # once per touch
    body.x += 0.02
    body.step(0.04, grid, [])
    assert body.speed == pytest.approx(0.5)
    x, y = body.x, body.y
    body.set_cmd_vel(1.0, 0.0)
    body.step(1.0, grid, [])
    assert (body.x, body.y) == (x, y)  # it never moves itself


# ---- the headed follow camera ------------------------------------------------------------------------------------
def test_the_follow_camera_is_behind_and_above_and_smoothed():
    from wardzero_agents.isaac_host import recording as R
    eye, target = R.chase_view((2.0, 3.0, 0.0))
    assert eye[0] < 2.0 and eye[1] == pytest.approx(3.0) and eye[2] > 2.0 and target[0] > 2.0
    cam = R.ChaseCam()
    e0, _ = cam.update((0.0, 0.0, 0.0))
    e1, _ = cam.update((1.0, 0.0, 0.0))  # the robot steps 1 m: the camera follows part of the way
    assert 0.0 < e1[0] - e0[0] < 0.5
    for _ in range(200):
        e, t = cam.update((1.0, 0.0, math.pi / 2))
    assert e == pytest.approx(R.chase_view((1.0, 0.0, math.pi / 2))[0], abs=1e-3)
    c2 = R.ChaseCam()
    c2.update((0, 0, math.radians(179)))
    c2.update((0, 0, math.radians(-179)))
    assert abs(rf.wrap(c2.yaw - math.radians(179.1))) < math.radians(1)  # the short way round the wrap


# ---- the scripted go_to hook -------------------------------------------------------------------------------------
def test_a_scripted_robot_go_to_sends_a_goal_and_post_brings_it_back(tmp_path, monkeypatch):
    from wardzero_agents.config.loader import load_scenario
    from wardzero_agents.ros.sim_service import SimService
    from wardzero_agents.runlog import RunLog
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"t": 0.5, "who": "robot", "do": "go_to", "place": "ward_doors"},
                                  {"t": 1.0, "who": "robot", "do": "go_to", "place": "post"}]))
    monkeypatch.setenv("WARDZERO_TEST_SCRIPTED_SAY", str(script))
    svc = SimService(load_scenario(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"), RunLog(tmp_path / "r"),
                     wait_for=())
    goals = []
    svc.on_goal = lambda x, y, yaw=None: goals.append((round(x, 2), round(y, 2), yaw))
    for _ in range(60):
        svc.step(1 / 40)
    r = svc.loaded.scenario.robot
    ward = svc.world.site.resolve("ward_doors")
    assert goals == [(round(ward[0], 2), round(ward[1], 2), None), (round(r.x, 2), round(r.y, 2), r.yaw)]


# ---- the wiring order --------------------------------------------------------------------------------------------
def test_the_walker_is_authored_in_the_robots_pass_and_started_after_physx_has_it():
    from wardzero_agents.isaac_host import robot_isaac as RI

    log = []

    class Walker:
        def start(self, root):
            log.append(f"start {root}")

    class Fake(RI.IsaacRobot):
        def __init__(self):
            self.app = SimpleNamespace(update=lambda: log.append("update"))
            self.cfg = SimpleNamespace(x=1.0, y=2.0, yaw=0.0)
            self.profile = profile_for_sim_forced_static()
            self.downgraded, self.art_root, self.camera_path = False, "/World/Robot/g1", None
            self.graph = SimpleNamespace(nodes=[])
            self.walking, self.walker = True, None

        def _check_ready(self):
            log.append("check")

        def _reference(self):
            log.append("reference")

        def _locate(self):
            log.append("locate")

        def _make_walker(self, world):
            log.append("author")
            self.walker = Walker()

        def _build_graph(self):
            log.append("graph")

        def _articulation(self):
            log.append("articulation")

    r = Fake()
    r.add(SimpleNamespace())
    assert log[:5] == ["check", "reference", "locate", "author", "graph"]  # one authoring pass
    assert log[5:] == ["update"] * RI.WALKER_AFTER_UPDATES + ["start /World/Robot/g1"] + \
        ["update"] * RI.SETTLE_UPDATES + ["articulation"]
    assert r._proxy.move_m == W.WALK_PROXY_MOVE_M and r._proxy.every_s == W.WALK_PROXY_EVERY_S  # throttled
    assert RI.walking_g1(SimpleNamespace(controller="builtin_claude", base="legged", can_move=True))
    assert not RI.walking_g1(G1_CLAUDE)


def profile_for_sim_forced_static():
    import dataclasses
    return dataclasses.replace(G1_CLAUDE, can_move=True, base="legged")


def test_the_bodies_read_back_writes_the_walker_pose_into_the_legged_body():
    from wardzero_agents.isaac_host.bodies import IsaacBodies
    body, _, _ = make_body()
    world = SimpleNamespace(robot=body, people={})
    b = IsaacBodies(robot=lambda: (3.0, 4.5, 0.25))
    b.read_back(world)
    assert (body.x, body.y, body.yaw) == (3.0, 4.5, 0.25)


def test_door_leaves_block_only_the_walking_g1s_router():
    from wardzero_agents.world.legged_body import block_segments
    grid = corridor()
    mine, people = Router(grid, clearance=rf.G1_CLEARANCE_M), Router(grid)
    straight = mine.route((8.0, 5.0), (8.0, 3.0))
    assert len(straight) == 2  # nothing in the way
    n = block_segments(mine, [((6.5, 4.0), (9.5, 4.0))], rf.G1_CLEARANCE_M)  # a leaf standing across the way
    assert n >= 15
    around = mine.route((8.0, 5.0), (8.0, 3.0))
    assert around is not None and len(around) > 2  # goes round the leaf
    assert all(not (6.4 < x < 9.6 and abs(y - 4.0) < 0.1) for x, y in around)
    assert len(people.route((8.0, 5.0), (8.0, 3.0))) == 2  # other routers are untouched


def test_a_walking_g1_route_keeps_out_of_a_forbidden_area_with_its_band_and_leaves():
    from wardzero_agents.config.models import ForbiddenArea
    from wardzero_agents.site.areas import robot_router
    from wardzero_agents.world.legged_body import block_segments
    grid = corridor()
    base = Router(grid, clearance=rf.G1_CLEARANCE_M)
    block_segments(base, [((12.0, 2.5), (12.0, 5.9))], rf.G1_CLEARANCE_M)  # a leaf standing across the east end
    area = ForbiddenArea(id="a1", name="wet floor", x0=7.0, y0=3.0, x1=9.0, y1=5.9)
    r = robot_router(base, [area])
    body = LeggedBody(2.0, 4.0, 0.0, router=r)
    body.ready = True
    assert body.set_goal(13.5, 4.0, avoid=[(4.0, 4.0, 0.3)])  # avoid discs pass through the area router
    path = [(2.0, 4.0)] + list(body.follower.path)
    pts = [(a[0] + (b[0] - a[0]) * k / 50, a[1] + (b[1] - a[1]) * k / 50) for a, b in zip(path, path[1:])
           for k in range(51)]
    assert not any(area.contains(x, y) for x, y in pts)  # never in the area
    assert not any(abs(x - 12.0) < 0.1 and 2.5 <= y <= 5.9 for x, y in pts)  # nor through the leaf
    blocked = r._router()
    assert blocked.clearance == rf.G1_CLEARANCE_M  # the copy keeps the G1's band ...
    iy, ix = grid.to_cell(12.0, 4.0)
    assert blocked._raw[iy][ix]  # ... and the leaf
    assert body.set_goal(8.0, 4.0) is False  # a goal inside the area is refused


def test_at_its_post_the_follow_camera_eases_to_the_desk_view_and_back():
    from wardzero_agents.isaac_host import recording as R
    post = (1.05, 1.62, -math.pi / 2)
    cam = R.ChaseCam(post=post)
    eye, target = cam.update(post)
    assert (eye, target) == R.follow_view(post, R.START_VIEW_FRAMING)  # starts on the post view: robot and visitor
    assert eye[1] < post[1]  # in front of the robot (it faces the lobby, -y), not behind the counter
    for _ in range(10):
        cam.update((post[0] + 2.0, post[1], 0.0))  # walking away
    assert 0 < cam.w < 1
    for _ in range(40):
        eye, target = cam.update((post[0] + 2.0, post[1], 0.0))
    assert cam.w == 0.0 and eye == pytest.approx(R.chase_view((cam.x, cam.y, cam.yaw))[0])
    assert R.ChaseCam().update((0, 0, 0)) == R.chase_view((0, 0, 0))  # no post: the plain chase camera


def test_a_task_layer_speed_cap_does_not_slow_the_g1_in_a_door():
    body, sonic, grid = make_body(x=4.0, y=4.0)
    body.follower.door_lines = [((5.0, 3.0), (5.0, 5.0))]
    body.follower.set_route([(8.0, 4.0)])  # already lined up on the leaf, 1 m before the door line
    body.speed_cap = 0.3  # reception's task layer: a person near
    body.step(0.04, grid, [])
    assert body.command.mode == rf.SLOW_WALK and body.command.speed == pytest.approx(0.6)
    body.x = 1.5
    body._next_follow = 0.0
    body.follower.set_route([(3.0, 4.0)])
    body.step(0.04, grid, [])
    assert body.command.speed == pytest.approx(0.3)
