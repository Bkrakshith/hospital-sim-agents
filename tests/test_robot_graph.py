"""The robot's OmniGraph as data: nodes, values and connections, built without Kit. The Isaac side applies it with
og.Controller.edit (wardzero_agents.isaac_host.robot_isaac); the smoke test runs it."""
import pytest

from wardzero_agents.isaac_host import robot_graph as rg
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile
from wardzero_agents.robots.sensors import SensorSpec
from wardzero_agents.ros import topics

ART = "/World/Robot/Geometry"
RP = "/Render/RenderProduct_1"


def g1_graph(**kw):
    args = dict(profile=G1_CLAUDE, art_root=ART, robot_domain=41, private_domain=42, render_product=RP, fps=30)
    return rg.build_robot_graph(**{**args, **kw})


def types(g):
    return {n: t for n, t in g.nodes}


def value(g, key):
    return dict(g.values)[key]


def connected(g, a, b):
    return (a, b) in g.connects


def test_static_g1_graph_has_the_s1_nodes_and_no_drive():
    g = g1_graph()
    t = types(g)
    assert t["tick"] == "omni.graph.action.OnPlaybackTick"
    for n, ty in (("clock", "ROS2PublishClock"), ("tf", "ROS2PublishTransformTree"), ("js_pub", "ROS2PublishJointState"),
                  ("js_sub", "ROS2SubscribeJointState"), ("cam_rgb", "ROS2CameraHelper"), ("cam_depth", "ROS2CameraHelper"),
                  ("cam_info", "ROS2CameraInfoHelper"), ("cam_bbox", "ROS2CameraHelper")):
        assert t[n] == f"isaacsim.ros2.bridge.{ty}"
    assert t["artctl"] == "isaacsim.core.nodes.IsaacArticulationController"
    assert not {"odom_pub", "twist_sub", "diff", "odom_calc"} & set(t)  # a static robot has no base to drive


def test_topics_come_from_the_table():
    g = g1_graph()
    rt = topics.robot_topics(G1_CLAUDE.ns)
    for key, node, attr in (("clock", "clock", "topicName"), ("tf", "tf", "topicName"), ("joint_states", "js_pub", "topicName"),
                            ("joint_command", "js_sub", "topicName"), ("camera_rgb", "cam_rgb", "topicName"),
                            ("camera_depth", "cam_depth", "topicName"), ("camera_info", "cam_info", "topicName")):
        assert value(g, f"{node}.inputs:{attr}") == rt[key]
        assert g.topics[key] == rt[key]
        assert value(g, f"{node}.inputs:nodeNamespace") == ""  # names are absolute, so the table is the only truth
    assert set(g.topics) == set(topics.isaac_graph_keys("static", False, ["rgbd"]) ) - {"bbox_3d"}


def test_raw_detections_go_to_the_private_domain_not_the_robot_s():
    g = g1_graph()
    raw = topics.raw_detection_topics(G1_CLAUDE.ns)
    assert value(g, "cam_bbox.inputs:topicName") == raw["bbox_3d"]
    assert value(g, "cam_bbox.inputs:semanticLabelsTopicName") == raw["labels"]
    assert value(g, "cam_bbox.inputs:enableSemanticLabels") is True
    assert value(g, "ctx_robot.inputs:domain_id") == 41 and value(g, "ctx_private.inputs:domain_id") == 42
    assert connected(g, "ctx_private.outputs:context", "cam_bbox.inputs:context")
    for n in ("clock", "tf", "js_pub", "js_sub", "cam_rgb", "cam_depth", "cam_info"):
        assert connected(g, "ctx_robot.outputs:context", f"{n}.inputs:context")
    assert not connected(g, "ctx_robot.outputs:context", "cam_bbox.inputs:context")  # the robot never sees the raw boxes
    assert g.topics_private == raw


def test_one_context_when_the_domains_are_the_same():
    g = g1_graph(private_domain=41)
    assert "ctx_private" not in types(g)
    assert connected(g, "ctx_robot.outputs:context", "cam_bbox.inputs:context")


def test_no_second_clock_when_the_robot_shares_the_private_domain():
    g = g1_graph(private_domain=41)
    assert "clock" not in types(g) and "clock" not in g.topics  # SimRos already publishes /clock there
    assert "clock" in types(g1_graph())


def test_ros_contexts_never_follow_the_environment():
    g = g1_graph()
    for ctx in ("ctx_robot", "ctx_private"):
        assert value(g, f"{ctx}.inputs:useDomainIDEnvVar") is False


def test_everything_is_ticked_and_stamped_with_sim_time():
    g = g1_graph()
    for n in ("clock", "tf", "js_pub", "js_sub", "cam_rgb", "cam_depth", "cam_info", "cam_bbox"):
        assert connected(g, "tick.outputs:tick", f"{n}.inputs:execIn")
    for n in ("clock", "tf", "js_pub"):
        assert connected(g, "simtime.outputs:simulationTime", f"{n}.inputs:timeStamp")
    assert connected(g, "js_sub.outputs:execOut", "artctl.inputs:execIn")
    assert connected(g, "js_sub.outputs:jointNames", "artctl.inputs:jointNames")
    assert connected(g, "js_sub.outputs:positionCommand", "artctl.inputs:positionCommand")


def test_prim_targets_are_marked_as_paths():
    g = g1_graph()
    for key in ("tf.inputs:targetPrims", "js_pub.inputs:targetPrim", "artctl.inputs:targetPrim"):
        (p,) = value(g, key)
        assert isinstance(p, rg.PrimPath) and p == ART
    assert value(g, "cam_rgb.inputs:renderProductPath") == RP


def test_camera_frames_and_rate():
    g = g1_graph()
    for n in ("cam_rgb", "cam_depth", "cam_info", "cam_bbox"):
        assert value(g, f"{n}.inputs:frameId") == "d435_link"  # the sensor's link, which the TF tree publishes
        assert f"{n}.inputs:frameSkipCount" not in dict(g.values)  # deprecated: the rate is the camera prim's tick rate
    assert value(g, "cam_rgb.inputs:type") == "rgb" and value(g, "cam_depth.inputs:type") == "depth"
    assert value(g, "cam_bbox.inputs:type") == "bbox_3d"
    assert g.camera_hz == 5.0 and g1_graph(camera_hz=15.0).camera_hz == 15.0
    assert g1_graph(profile=RobotProfile(**{**G1_CLAUDE.__dict__, "sensors": ()}), render_product=None).camera_hz == 0.0


def test_no_camera_no_camera_nodes():
    bare = RobotProfile(**{**G1_CLAUDE.__dict__, "sensors": ()})
    g = g1_graph(profile=bare, render_product=None)
    assert not {"cam_rgb", "cam_depth", "cam_info", "cam_bbox", "ctx_private"} & set(types(g))
    with pytest.raises(ValueError, match="render product"):
        g1_graph(render_product=None)  # the G1 has a camera, so it needs one


def test_rgb_only_camera_has_no_depth():
    rgb = RobotProfile(**{**G1_CLAUDE.__dict__, "sensors": (SensorSpec("cam", "rgb", None, "head_link"),)})
    g = g1_graph(profile=rgb)
    assert "cam_depth" not in types(g) and "cam_rgb" in types(g)


def wheeled(**kw):
    return RobotProfile(name="cart", can_move=True, base="wheeled", radius=0.3, reach=(0.3, 1.0), max_speed=1.0, ns="cart",
                        sensors=(SensorSpec("cam", "rgbd", "realsense_d455", "camera_link"),), controller="external", **kw)


DRIVE = {"wheel_joints": ["left_wheel_joint", "right_wheel_joint"], "wheel_radius": 0.1, "wheel_distance": 0.4}


def test_wheeled_base_gets_odometry_and_a_twist_drive():
    p = wheeled()
    g = rg.build_robot_graph(p, ART, 41, 42, RP, 30, drive=DRIVE)
    t = types(g)
    assert t["odom_calc"] == "isaacsim.core.nodes.IsaacComputeOdometry"
    assert t["odom_pub"] == "isaacsim.ros2.bridge.ROS2PublishOdometry"
    assert t["twist_sub"] == "isaacsim.ros2.bridge.ROS2SubscribeTwist"
    assert t["diff"] == "isaacsim.robot.wheeled_robots.DifferentialController"
    assert t["artctl_drive"] == "isaacsim.core.nodes.IsaacArticulationController"
    rt = topics.robot_topics("cart")
    assert value(g, "odom_pub.inputs:topicName") == rt["odom"] == g.topics["odom"]
    assert value(g, "twist_sub.inputs:topicName") == rt["cmd_vel"] == g.topics["cmd_vel"]
    assert value(g, "odom_pub.inputs:odomFrameId") == "map" and value(g, "odom_pub.inputs:chassisFrameId") == "base_link"
    assert value(g, "diff.inputs:wheelRadius") == 0.1 and value(g, "diff.inputs:wheelDistance") == 0.4
    assert value(g, "artctl_drive.inputs:jointNames") == ["left_wheel_joint", "right_wheel_joint"]
    assert connected(g, "twist_sub.outputs:linearVelocity", "diff.inputs:linearVelocity")
    assert connected(g, "twist_sub.outputs:angularVelocity", "diff.inputs:angularVelocity")
    assert connected(g, "diff.outputs:velocityCommand", "artctl_drive.inputs:velocityCommand")
    assert connected(g, "twist_sub.outputs:execOut", "diff.inputs:execIn")
    assert connected(g, "diff.outputs:execOut", "artctl_drive.inputs:execIn")
    for o, i in (("position", "position"), ("orientation", "orientation"), ("linearVelocity", "linearVelocity"),
                 ("angularVelocity", "angularVelocity")):
        assert connected(g, f"odom_calc.outputs:{o}", f"odom_pub.inputs:{i}")
    assert connected(g, "odom_calc.outputs:execOut", "odom_pub.inputs:execIn")
    assert set(topics.isaac_graph_keys("wheeled", True, ["rgbd"])) - {"bbox_3d"} == set(g.topics)


def test_wheeled_base_needs_its_wheels():
    with pytest.raises(ValueError, match="wheel"):
        rg.build_robot_graph(wheeled(), ART, 41, 42, RP, 30)


def test_a_wheeled_robot_that_cannot_move_has_no_drive():
    p = RobotProfile(**{**wheeled().__dict__, "can_move": False})
    assert "twist_sub" not in types(rg.build_robot_graph(p, ART, 41, 42, RP, 30))


def test_topic_key_rules():
    assert topics.isaac_graph_keys("static", False, []) == ("clock", "tf", "joint_states", "joint_command")
    assert "camera_depth" not in topics.isaac_graph_keys("static", False, ["rgb"])
    assert "odom" not in topics.isaac_graph_keys("legged", True, ["rgbd"])  # SONIC walking is a later task
    assert topics.raw_detection_topics("g1")["bbox_3d"] == "/wz/g1/bbox_3d_raw"
    assert not any(v.startswith(p) for v in topics.robot_topics("robot").values() for p in ("/wz/",))
