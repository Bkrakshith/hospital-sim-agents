"""SimService, the backend-neutral sim, with its Bodies seam; SimRos, its shared wz.v1 wiring."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control, Heard, Intent
from wardzero_agents.isaac_host.bodies import IsaacBodies
from wardzero_agents.ros import stub_sim_node, topics
from wardzero_agents.ros.sim_service import SimOutputs, SimRos, SimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.world.bodies import Bodies, StubBodies

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def make(tmp_path, **kw):
    kw.setdefault("wait_for", ())
    return SimService(load_scenario(V1), RunLog(tmp_path), **kw)


def test_stub_sim_service_is_the_sim_service_with_stub_bodies(tmp_path):
    assert issubclass(stub_sim_node.StubSimService, SimService)
    s = stub_sim_node.StubSimService(load_scenario(V1), RunLog(tmp_path), wait_for=())
    assert isinstance(s.bodies, StubBodies)
    assert isinstance(make(tmp_path).bodies, StubBodies)


class Frozen(Bodies):
    """Bodies that never move anyone: proves the service moves people only through its Bodies."""
    name = "frozen"

    def __init__(self):
        self.calls = []

    def step(self, world, t):
        self.calls.append(t)


def test_the_service_moves_people_only_through_its_bodies(tmp_path):
    frozen = Frozen()
    s = make(tmp_path, bodies=frozen)
    stub = make(tmp_path / "b")
    for svc in (s, stub):
        svc.step(0.01)
        svc.on_intent(Intent(id="w", person="P5", t=0.0, kind="walk_to", target="entrance"))
        for _ in range(80):
            svc.step(0.1)
    p_frozen, p_stub = s.world.people["P5"], stub.world.people["P5"]
    start = load_scenario(V1).site.seats["D2"]
    assert (p_frozen.x, p_frozen.y) == pytest.approx((start.x, start.y))
    assert (p_stub.x, p_stub.y) != pytest.approx((start.x, start.y))
    assert frozen.calls[-1] == pytest.approx(s.t) and len(frozen.calls) == 81


def test_isaac_bodies_write_read_back_poses_into_the_world(tmp_path):
    poses = {"P1": (3.0, 4.0, 0.5)}
    robot = {"pose": (-1.0, -0.5, 1.2)}
    bodies = IsaacBodies(people={"P1": lambda: poses["P1"]}, robot=lambda: robot["pose"])
    s = make(tmp_path, bodies=bodies)
    s.step(0.1)
    p1 = s.world.people["P1"]
    assert (p1.x, p1.y, p1.yaw) == (3.0, 4.0, 0.5)
    assert s.world.robot.pose == (-1.0, -0.5, 1.2)
    st = {x.person: x for x in s.world.statuses()}["P1"]
    assert (st.x, st.y) == (3.0, 4.0)
    poses["P1"] = (3.5, 4.0, 0.0)
    s.step(0.1)
    assert s.world.people["P1"].x == 3.5


def test_isaac_bodies_start_empty_and_leave_the_rest_to_the_world(tmp_path):
    bodies = IsaacBodies()
    assert bodies.people == {} and bodies.robot is None
    s = make(tmp_path, bodies=bodies)
    s.step(0.01)
    s.on_intent(Intent(id="w", person="P5", t=0.0, kind="walk_to", target="entrance"))
    for _ in range(80):
        s.step(0.1)
    start = load_scenario(V1).site.seats["D2"]
    assert (s.world.people["P5"].x, s.world.people["P5"].y) != pytest.approx((start.x, start.y))
    bodies.add_person("P5", lambda: (0.0, 0.0, 0.0))
    s.step(0.1)
    assert (s.world.people["P5"].x, s.world.people["P5"].y) == (0.0, 0.0)


def test_a_missing_read_back_leaves_the_pose(tmp_path):
    s = make(tmp_path, bodies=IsaacBodies(people={"P1": lambda: None, "ghost": lambda: (1.0, 1.0, 0.0)}))
    before = (s.world.people["P1"].x, s.world.people["P1"].y)
    s.step(0.1)
    assert (s.world.people["P1"].x, s.world.people["P1"].y) == before


def test_held_says_when_the_clock_must_not_move(tmp_path):
    s = make(tmp_path, wait_for=("agents",), lockstep=True)
    assert s.held
    s.ready("agents")
    assert not s.held
    s.control(Control(action="pause"))
    assert s.held
    s.control(Control(action="resume"))
    s.set_pending(1)
    assert s.held
    s.set_pending(0)
    assert not s.held
    s.t = s.duration_s
    assert s.held


# ---- SimRos: the same topics for the stub node and the Isaac host -------------------------
class Auto:
    """A stand-in ROS message: any attribute path can be set."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = Auto()
        object.__setattr__(self, name, v)
        return v


class FakePub:
    def __init__(self, topic, sink):
        self.topic, self.sink = topic, sink

    def publish(self, msg):
        self.sink.append((self.topic, msg))


class FakeBridge:
    def __init__(self):
        self.subs, self.sent, self.pubs = {}, [], {}

    def subscribe_json(self, cls, topic, handler, qos=None):
        self.subs.setdefault(topic, handler)

    def subscribe(self, msg_type, topic, handler, qos=None, convert=None):
        self.subs.setdefault(topic, (lambda m, h=handler, c=convert: h(c(m) if c else m)))

    def publisher(self, topic, msg_type=None, qos=None):
        return self.pubs.setdefault(topic, FakePub(topic, self.sent))

    def publish_json(self, topic, model, qos=None):
        self.sent.append((topic, model))

    def publish_text(self, topic, text, qos=None):
        self.sent.append((topic, text))


MSGS = SimpleNamespace(Clock=Auto, Odometry=Auto, Twist=Auto, PoseStamped=Auto, Int32=Auto, String=Auto,
                       QOS="qos", QOS_LATCHED="latched")


def test_sim_ros_subscribes_the_wz_v1_inputs(tmp_path):
    s = make(tmp_path)
    ros = FakeBridge()
    SimRos(s, ros, msgs=MSGS)
    rt = topics.robot_topics(s.loaded.scenario.robot.ns)
    for pid in s.world.people:
        assert topics.intent(pid) in ros.subs and topics.say(pid) in ros.subs
    for t in (topics.CONTROL, topics.WORLD_EVENT, topics.PENDING, topics.TIMELINE, rt["cmd_vel"], rt["say"],
              rt["goal_pose"], rt["task_action"]):
        assert t in ros.subs
    ros.subs[topics.CONTROL](Control(action="stop"))
    assert s.step(0.01).stop


def test_sim_ros_publishes_clock_on_every_domain_and_the_outputs(tmp_path):
    s = make(tmp_path)
    ros, robot_dom = FakeBridge(), FakeBridge()
    sr = SimRos(s, ros, msgs=MSGS, clock_bridges=[robot_dom])
    out = s.step(0.25)
    sr.publish(out)
    clocks = [m for t, m in ros.sent if t == topics.CLOCK]
    assert len(clocks) == 1 and clocks[0].clock.sec == 0 and clocks[0].clock.nanosec == 250_000_000
    assert [t for t, _ in robot_dom.sent] == [topics.CLOCK]
    sent = [t for t, _ in ros.sent]
    assert topics.ROSTER in sent and topics.status("P1") in sent and topics.observation("P1") in sent
    sr.publish_stop()
    assert ros.sent[-1][0] == topics.CONTROL and ros.sent[-1][1].action == "stop"


def stamp(m):
    return m.clock.sec * 1_000_000_000 + m.clock.nanosec


def test_clock_stamps_never_go_back_at_a_whole_second(tmp_path):
    """t accumulates 1/30 s steps in floating point: 0.9999999999999 must not become sec 0 / nanosec 0."""
    s = make(tmp_path)
    ros = FakeBridge()
    sr = SimRos(s, ros, msgs=MSGS)
    t, stamps = 0.0, []
    for _ in range(3000):
        t += 1.0 / 30.0
        sr.publish(SimOutputs(clock=t))
    stamps = [stamp(m) for tp, m in ros.sent if tp == topics.CLOCK]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert 0 <= stamps[-1] / 1e9 - t < 1e-6


def test_the_robot_domain_clock_carries_isaacs_own_time_base(tmp_path):
    """Isaac stamps tf, joints and camera with its simulation time, which runs ahead of svc.t by a constant (the frames
    before the loop) and cannot be reset without stopping the timeline. The robot domain's /clock is on that base, so
    everything the robot hears agrees; the private domain's /clock stays svc.t."""
    s = make(tmp_path)
    private, robot_dom = FakeBridge(), FakeBridge()
    sr = SimRos(s, private, msgs=MSGS, clock_bridges=[robot_dom], robot_clock_offset=2.8333)
    sr.publish(SimOutputs(clock=0.25))
    assert stamp([m for t, m in private.sent if t == topics.CLOCK][0]) == 250_000_000
    assert stamp([m for t, m in robot_dom.sent if t == topics.CLOCK][0]) == 3_083_300_000


def test_sim_ros_subscribes_people_who_join(tmp_path):
    s = make(tmp_path)
    ros = FakeBridge()
    sr = SimRos(s, ros, msgs=MSGS)
    out = SimOutputs(clock=0.0, roster=s.world.roster(0.0).model_copy(update={"people": []}))
    sr.publish(out)
    newcomer = s.world.roster(0.0).people[0].model_copy(update={"id": "N9"})
    sr.publish(SimOutputs(clock=0.0, roster=s.world.roster(0.0).model_copy(update={"people": [newcomer]})))
    assert topics.intent("N9") in ros.subs


# ---- the robot's own domain (Isaac host) ----------------------------------------------------
def test_robot_topics_can_live_on_the_robots_bridge(tmp_path):
    s = make(tmp_path)
    private, robot = FakeBridge(), FakeBridge()
    sr = SimRos(s, private, msgs=MSGS, robot_ros=robot, graph_owned=("bbox_3d", "odom"))
    rt = topics.robot_topics(s.loaded.scenario.robot.ns)
    for key in ("say", "cmd_vel", "goal_pose", "task_action"):
        assert rt[key] in robot.subs and rt[key] not in private.subs
    for t in (topics.CONTROL, topics.WORLD_EVENT, topics.PENDING, topics.intent("P1")):
        assert t in private.subs and t not in robot.subs  # the people's side stays private
    out = s.step(0.25)
    out.detections, out.objects = {"detections": []}, {"items": []}
    out.odom = {"x": 1.0, "y": 2.0, "yaw": 0.0, "vx": 0.0, "wz": 0.0}
    out.task_results = [{"id": "a1", "action": "press_call", "ok": True, "reason": "ok", "t": 0.2}]
    sr.publish(out)
    ns = f"/{s.loaded.scenario.robot.ns}/"
    assert not [t for t, _ in private.sent if t.startswith(ns)]  # nothing of the robot's on the private domain
    on_robot = {t for t, _ in robot.sent}
    assert {rt["objects"], rt["task_result"]} <= on_robot
    assert rt["bbox_3d"] not in on_robot and rt["odom"] not in on_robot  # the Isaac graph publishes these
    assert rt["bbox_3d"] not in robot.pubs and rt["odom"] not in robot.pubs


def test_robot_speech_heard_goes_to_the_robots_bridge(tmp_path):
    s = make(tmp_path)
    private, robot = FakeBridge(), FakeBridge()
    sr = SimRos(s, private, msgs=MSGS, robot_ros=robot)
    heard = Heard(source_kind="person", source_id="unknown", text="hello", received_db=60.0, partial=False, addressed_to_me=True)
    sr.publish(SimOutputs(clock=0.0, robot_heard=[heard]))
    assert [t for t, _ in robot.sent if t.endswith("/audio/heard")]
    assert not [t for t, _ in private.sent if t.endswith("/audio/heard")]


def test_default_is_one_bridge_for_everything(tmp_path):
    s = make(tmp_path)
    ros = FakeBridge()
    SimRos(s, ros, msgs=MSGS)
    rt = topics.robot_topics(s.loaded.scenario.robot.ns)
    assert rt["say"] in ros.subs and rt["bbox_3d"] in ros.pubs


def test_isaac_bodies_call_the_robot_driver_after_the_poses_are_read_back(tmp_path):
    seen = []

    class Driver:
        def update(self, world, t, dt):
            seen.append((t, dt, world.robot.pose))

    s = make(tmp_path)
    bodies = IsaacBodies(robot=lambda: (-1.0, -0.5, 1.2))
    bodies.robot_driver = Driver()
    bodies.step(s.world, 0.5)
    bodies.step(s.world, 0.6)
    assert seen[0][2] == (-1.0, -0.5, 1.2)  # the driver sees this frame's pose, not last frame's
    assert seen[1][0] == 0.6 and seen[1][1] == pytest.approx(0.1)


def test_task_results_on_the_wire_never_name_a_person(tmp_path):
    """A hand_over result names its receiver for the run log and the benchmark; the robot gets it without the id."""
    import json as _json
    s = make(tmp_path)
    ros = FakeBridge()
    sr = SimRos(s, ros, msgs=MSGS)
    r = {"id": "a1", "action": "hand_over", "ok": True, "reason": "ok", "t": 1.0, "person": "P1", "item": "T1"}
    sr.publish(SimOutputs(clock=1.0, task_results=[r]))
    sent = [_json.loads(m) for tp, m in ros.sent if tp == sr.rt["task_result"]]
    assert sent == [{k: v for k, v in r.items() if k != "person"}]
