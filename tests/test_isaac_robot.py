"""The Isaac robot's pure logic: when the obstacle cylinder moves, which arm pose the world calls for, how targets
ease towards it, and the base pose read from a transform. No Kit."""
import math
from types import SimpleNamespace

import pytest

from wardzero_agents.isaac_host import robot as rb


def test_a_static_robot_never_moves_its_proxy():
    m = rb.ProxyMover(1.05, 1.62)
    assert all(m.update(1.05, 1.62, t * 0.033) is None for t in range(200))


def test_the_proxy_follows_after_a_fifth_of_a_metre():
    m = rb.ProxyMover(0.0, 0.0)
    assert m.update(0.15, 0.0, 0.1) is None
    assert m.update(0.21, 0.0, 0.2) == (0.21, 0.0)
    assert m.update(0.30, 0.0, 0.3) is None  # 9 cm from where the proxy now is


def test_a_slow_robot_still_gets_its_proxy_every_half_second():
    m = rb.ProxyMover(0.0, 0.0)
    assert m.update(0.05, 0.0, 0.2) is None  # moved, but neither 0.2 m nor 0.5 s yet
    assert m.update(0.06, 0.0, 0.55) == (0.06, 0.0)
    assert m.update(0.06, 0.0, 2.0) is None  # standing still: no move however long it has been
    assert m.update(0.065, 0.0, 2.1) is None  # a hair's breadth is not a move


def test_proxy_moves_are_rare_enough_for_the_frame_budget():
    m = rb.ProxyMover(0.0, 0.0)
    moves = sum(1 for k in range(300) if m.update(k * 0.033 * 1.0, 0.0, k * 0.033) is not None)  # 1 m/s for 10 s
    assert moves <= 10 / 0.2 + 1  # a metre per second is one move per 0.2 m at most (5 a second, not 30)
    assert moves < 300 / 4


def world(offers=()):
    return SimpleNamespace(_offers={pid: {"giver": g} for pid, g in offers})


def test_the_arm_offers_while_the_robot_has_an_open_hand_over():
    assert rb.arm_pose_name(world()) == "rest"
    assert rb.arm_pose_name(world([("P1", "g1")])) == "offer"
    assert rb.arm_pose_name(world([("P1", "SB3")])) == "rest"  # somebody else's offer is not the robot's


def test_arm_targets_ease_to_a_pose_and_then_stop_writing():
    arm = rb.ArmTargets({"rest": {}, "offer": {"right_shoulder_pitch_joint": -1.0, "right_elbow_joint": 1.0}}, rate=1.0)
    assert arm.step("rest", 0.1) is None  # already there: nothing to write
    first = arm.step("offer", 0.1)
    assert first == {"right_shoulder_pitch_joint": pytest.approx(-0.1), "right_elbow_joint": pytest.approx(0.1)}
    for _ in range(9):  # 0.1 rad a step: one second in all
        last = arm.step("offer", 0.1)
    assert last == {"right_shoulder_pitch_joint": pytest.approx(-1.0), "right_elbow_joint": pytest.approx(1.0)}
    assert arm.step("offer", 0.1) is None
    back = arm.step("rest", 0.5)
    assert back["right_shoulder_pitch_joint"] == pytest.approx(-0.5)  # rest means every joint back to zero


def test_the_g1_poses_use_real_joint_names():
    assert set(rb.G1_ARM_POSES) == {"rest", "offer"}
    names = {j for pose in rb.G1_ARM_POSES.values() for j in pose}
    assert names and all(n.endswith("_joint") for n in names)
    assert all(n.startswith(("right_", "left_")) for n in names)


def test_base_pose_from_a_row_vector_matrix():
    yaw = math.radians(-90.0)
    c, s = math.cos(yaw), math.sin(yaw)
    m = ((c, s, 0, 0), (-s, c, 0, 0), (0, 0, 1, 0), (1.05, 1.62, 0.8, 1))
    x, y, a = rb.pose_from_matrix(m)
    assert (x, y) == (1.05, 1.62) and a == pytest.approx(yaw)


# ---- the detection link: raw boxes in on the private bridge, the robot's bbox_3d out on the robot's -------------
class Bridge:
    def __init__(self):
        self.subs, self.sent = {}, []

    def subscribe(self, msg_type, topic, handler, qos=None, convert=None):
        self.subs[topic] = (msg_type, lambda m, h=handler, c=convert: h(c(m) if c else m))

    def publish_text(self, topic, text, qos=None):
        self.sent.append((topic, text))


class FakeWorld:
    def __init__(self):
        self.people = {"P1": SimpleNamespace(x=1.0, y=0.2, posture="standing"),
                       "P2": SimpleNamespace(x=6.0, y=6.0, posture="gone")}
        self.seen = []
        self.tracks = SimpleNamespace(track=self._track)

    def _track(self, pid):
        if pid not in self.seen:
            self.seen.append(pid)
        return f"t{self.seen.index(pid) + 1}"


def raw_message(t, boxes):
    import json
    from wardzero_agents.robots import bbox_adapter as ba
    return json.dumps({"stamp": t, "boxes": boxes}), ba


def link(world=None):
    private, robot = Bridge(), Bridge()
    ln = rb.DetectionLink(world or FakeWorld(), "robot", private, robot, msgs=SimpleNamespace(
        Detection3DArray="D3A", String="Str", QOS="q"), parse=lambda m: m)
    return ln, private, robot


def test_the_link_listens_on_the_private_domain_and_speaks_on_the_robots():
    ln, private, robot = link()
    assert set(private.subs) == {"/wz/robot/bbox_3d_raw", "/wz/robot/semantic_labels_raw"}
    assert not robot.subs  # nothing is read from the robot's domain


def test_a_raw_message_becomes_a_robot_bbox_3d():
    import json
    ln, private, robot = link()
    private.subs["/wz/robot/semantic_labels_raw"][1](SimpleNamespace(data='{"0": {"class": "person"}}'))
    private.subs["/wz/robot/bbox_3d_raw"][1](
        (3.0, [{"class_id": "0", "center": (1.0, 0.2, 0.0), "orientation": (0, 0, 0, 1), "size": (0.5, 0.5, 1.7)}]))
    (topic, text), = robot.sent
    assert topic == "/robot/camera/bbox_3d"
    msg = json.loads(text)
    assert msg["detections"][0]["id"] == "t1" and "P1" not in text


def test_an_empty_view_is_published_when_the_camera_goes_quiet():
    import json
    ln, private, robot = link()
    ln.update(0.2)
    assert robot.sent == []  # not quiet for long yet
    ln.update(0.6)
    (topic, text), = robot.sent
    assert topic == "/robot/camera/bbox_3d" and json.loads(text)["detections"] == []
    ln.update(0.7)
    assert len(robot.sent) == 1  # then once per half second, not every frame
    private.subs["/wz/robot/semantic_labels_raw"][1](SimpleNamespace(data='{"0": {"class": "person"}}'))
    private.subs["/wz/robot/bbox_3d_raw"][1]((1.0, []))
    ln.update(1.0)
    assert len(robot.sent) == 2  # a real message (even an empty one) resets the quiet timer


def test_people_who_left_are_not_matched():
    ln, private, robot = link()
    private.subs["/wz/robot/semantic_labels_raw"][1](SimpleNamespace(data='{"0": {"class": "person"}}'))
    private.subs["/wz/robot/bbox_3d_raw"][1](
        (3.0, [{"class_id": "0", "center": (6.0, 6.0, 0.0), "orientation": (0, 0, 0, 1), "size": (0.5, 0.5, 1.7)}]))
    import json
    assert json.loads(robot.sent[-1][1])["detections"] == []  # P2 is gone: a box at their spot matches nobody


def test_camera_resolution_comes_from_the_sensor():
    from wardzero_agents.robots.profile import G1_CLAUDE
    from wardzero_agents.robots.sensors import SensorSpec
    assert rb.camera_resolution(G1_CLAUDE.sensors[0]) == (1280, 720)  # D455 catalogue row
    assert rb.camera_resolution(SensorSpec("c", "rgb", None, "l", params={"resolution": [800, 600]})) == (800, 600)
    assert rb.camera_resolution(SensorSpec("c", "rgb", None, "l")) == (640, 480)
