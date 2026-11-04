"""SimService, the backend-neutral sim, with its Bodies seam; SimRos, its shared wz.v1 wiring."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control, Intent
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


def test_sim_ros_subscribes_people_who_join(tmp_path):
    s = make(tmp_path)
    ros = FakeBridge()
    sr = SimRos(s, ros, msgs=MSGS)
    out = SimOutputs(clock=0.0, roster=s.world.roster(0.0).model_copy(update={"people": []}))
    sr.publish(out)
    newcomer = s.world.roster(0.0).people[0].model_copy(update={"id": "N9"})
    sr.publish(SimOutputs(clock=0.0, roster=s.world.roster(0.0).model_copy(update={"people": [newcomer]})))
    assert topics.intent("N9") in ros.subs
