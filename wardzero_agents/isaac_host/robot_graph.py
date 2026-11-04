"""The robot's OmniGraph on its ROS domain, as data (spike S1 Q3/Q4). No Kit here: ``build_robot_graph`` returns the
nodes, values and connections, and ``robot_isaac`` applies them with ``og.Controller.edit``.

What it publishes and listens on (names from ``ros.topics.robot_topics``, all absolute, so the table is the one truth):
/tf, ``joint_states``, ``joint_command`` (-> the Articulation Controller), and for a camera ``camera/rgb``,
``camera/depth`` (a depth camera), ``camera/camera_info``. A wheeled base that can move also gets ``odom`` and ``cmd_vel``
(Twist -> differential drive -> wheel joint velocities).

The camera's bbox_3d helper is the one thing that does not go to the robot: Isaac's boxes are world-frame, foot-centred
and have numeric classes, so they go to the **private** domain (``topics.raw_detection_topics``) where the host's
adapter (``robots.bbox_adapter``) turns them into the robot's anonymous bbox_3d. Two ROS2Context nodes, one per domain
(one when the domains are the same), both with ``useDomainIDEnvVar`` off: ROS_DOMAIN_ID would silently override them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..robots.profile import RobotProfile
from ..ros import topics

BRIDGE = "isaacsim.ros2.bridge"
CAMERA_HZ = 5.0  # the camera's tick rate (omni:sensor:tickRate on its prim, set by robot_isaac): it throttles the
# render product itself, which is what a 1280x720 image every frame costs. rgb, depth, camera_info and the boxes follow it.
GRAPH_PATH = "/World/RobotGraph"
SIMTIME_ATTR = f"{GRAPH_PATH}/simtime.outputs:simulationTime"
IMPULSE_ATTR = f"{GRAPH_PATH}/tick.state:enableImpulse"  # set True once per frame: one evaluation, one message per topic


class PrimPath(str):
    """A value that is a USD prim path (the Isaac side turns it into an Sdf.Path)."""


@dataclass
class GraphSpec:
    path: str = GRAPH_PATH
    nodes: list = field(default_factory=list)  # (name, type)
    values: list = field(default_factory=list)  # ("node.inputs:attr", value)
    connects: list = field(default_factory=list)  # ("a.outputs:x", "b.inputs:y")
    topics: dict = field(default_factory=dict)  # robot_topics key -> topic, on the robot domain
    topics_private: dict = field(default_factory=dict)  # raw_detection_topics, on the private domain
    camera_hz: float = 0.0  # tick rate for the camera prim (0: no camera)


def build_robot_graph(profile: RobotProfile, art_root: str, robot_domain: int, private_domain: int,
                      render_product: Optional[str], fps: int, drive: Optional[dict] = None,
                      camera_hz: float = CAMERA_HZ) -> GraphSpec:
    """``drive``: {"wheel_joints": [left, right], "wheel_radius": m, "wheel_distance": m}, required for a wheeled base
    that can move."""
    rt = topics.robot_topics(profile.ns)
    g = GraphSpec()
    cams = [s for s in profile.sensors if s.kind in ("rgb", "rgbd")]
    has_depth = any(s.kind == "rgbd" for s in cams)
    wheeled = profile.base == "wheeled" and profile.can_move
    if cams and render_product is None:
        raise ValueError("the robot has a camera, so it needs a render product")
    if wheeled and not (drive and drive.get("wheel_joints") and len(drive["wheel_joints"]) == 2):
        raise ValueError("a wheeled base needs its two wheel joints, wheel_radius and wheel_distance")

    def node(name, kind):
        g.nodes.append((name, kind if "." in kind else f"{BRIDGE}.{kind}"))

    def val(name, attr, value):
        g.values.append((f"{name}.inputs:{attr}", value))

    def link(a, b):
        g.connects.append((a, b))

    def tick_and_context(names, ctx="ctx_robot"):
        for n in names:
            link("tick.outputs:execOut", f"{n}.inputs:execIn")
            link(f"{ctx}.outputs:context", f"{n}.inputs:context")

    def absolute(name, topic, extra=()):
        val(name, "topicName", topic)
        val(name, "nodeNamespace", "")
        for k, v in extra:
            val(name, k, v)

    node("tick", "omni.graph.action.OnImpulseEvent")  # the host fires it once per frame (IsaacRobot.fire)
    node("simtime", "isaacsim.core.nodes.IsaacReadSimulationTime")
    node("ctx_robot", "ROS2Context")
    val("ctx_robot", "domain_id", int(robot_domain))
    val("ctx_robot", "useDomainIDEnvVar", False)

    # TF and joints. There is no clock node: SimRos publishes /clock on every domain (the robot domain's on Isaac's own
    # simulation-time base, which is what this graph and the camera helpers stamp with; see IsaacHost.calibrate_robot_clock).
    node("tf", "ROS2PublishTransformTree")
    node("js_pub", "ROS2PublishJointState")
    node("js_sub", "ROS2SubscribeJointState")
    node("artctl", "isaacsim.core.nodes.IsaacArticulationController")
    root = PrimPath(art_root)
    absolute("tf", rt["tf"], [("targetPrims", [root])])
    absolute("js_pub", rt["joint_states"], [("targetPrim", [root])])
    absolute("js_sub", rt["joint_command"])
    val("artctl", "targetPrim", [root])
    tick_and_context(["tf", "js_pub", "js_sub"])
    for n in ("tf", "js_pub"):
        link("simtime.outputs:simulationTime", f"{n}.inputs:timeStamp")
    link("js_sub.outputs:execOut", "artctl.inputs:execIn")
    for out in ("jointNames", "positionCommand", "velocityCommand", "effortCommand"):
        link(f"js_sub.outputs:{out}", f"artctl.inputs:{out}")
    g.topics.update({k: rt[k] for k in ("tf", "joint_states", "joint_command")})

    # camera: rgb, depth and camera_info to the robot; raw boxes to the private domain
    if cams:
        frame = cams[0].link
        names = ["cam_rgb", "cam_info"] + (["cam_depth"] if has_depth else [])
        for n, ty in (("cam_rgb", "rgb"), ("cam_depth", "depth")):
            if n in names:
                node(n, "ROS2CameraHelper")
                absolute(n, rt["camera_rgb" if ty == "rgb" else "camera_depth"],
                         [("type", ty), ("renderProductPath", render_product), ("frameId", frame)])
        node("cam_info", "ROS2CameraInfoHelper")
        absolute("cam_info", rt["camera_info"], [("renderProductPath", render_product), ("frameId", frame)])
        tick_and_context(names)
        g.topics.update({"camera_rgb": rt["camera_rgb"], "camera_info": rt["camera_info"]})
        if has_depth:
            g.topics["camera_depth"] = rt["camera_depth"]
        raw = topics.raw_detection_topics(profile.ns)
        priv = "ctx_robot"
        if int(private_domain) != int(robot_domain):
            node("ctx_private", "ROS2Context")
            val("ctx_private", "domain_id", int(private_domain))
            val("ctx_private", "useDomainIDEnvVar", False)
            priv = "ctx_private"
        node("cam_bbox", "ROS2CameraHelper")
        absolute("cam_bbox", raw["bbox_3d"],
                 [("type", "bbox_3d"), ("renderProductPath", render_product), ("frameId", frame),
                  ("enableSemanticLabels", True), ("semanticLabelsTopicName", raw["labels"])])
        tick_and_context(["cam_bbox"], priv)
        g.topics_private = dict(raw)
        g.camera_hz = float(camera_hz)

    # a wheeled base that can move: odometry out, Twist in
    if wheeled:
        node("odom_calc", "isaacsim.core.nodes.IsaacComputeOdometry")
        node("odom_pub", "ROS2PublishOdometry")
        node("twist_sub", "ROS2SubscribeTwist")
        node("diff", "isaacsim.robot.wheeled_robots.DifferentialController")
        node("artctl_drive", "isaacsim.core.nodes.IsaacArticulationController")
        val("odom_calc", "chassisPrim", [root])
        absolute("odom_pub", rt["odom"], [("odomFrameId", "map"), ("chassisFrameId", "base_link")])
        absolute("twist_sub", rt["cmd_vel"])
        val("diff", "wheelRadius", float(drive["wheel_radius"]))
        val("diff", "wheelDistance", float(drive["wheel_distance"]))
        val("artctl_drive", "targetPrim", [root])
        val("artctl_drive", "jointNames", list(drive["wheel_joints"]))
        link("tick.outputs:execOut", "odom_calc.inputs:execIn")
        link("odom_calc.outputs:execOut", "odom_pub.inputs:execIn")
        for n in ("odom_pub", "twist_sub"):
            link("ctx_robot.outputs:context", f"{n}.inputs:context")
        link("tick.outputs:execOut", "twist_sub.inputs:execIn")
        link("simtime.outputs:simulationTime", "odom_pub.inputs:timeStamp")
        for o in ("position", "orientation", "linearVelocity", "angularVelocity"):
            link(f"odom_calc.outputs:{o}", f"odom_pub.inputs:{o}")
        link("twist_sub.outputs:execOut", "diff.inputs:execIn")
        link("twist_sub.outputs:linearVelocity", "diff.inputs:linearVelocity")
        link("twist_sub.outputs:angularVelocity", "diff.inputs:angularVelocity")
        link("diff.outputs:execOut", "artctl_drive.inputs:execIn")
        link("diff.outputs:velocityCommand", "artctl_drive.inputs:velocityCommand")
        g.topics.update({"odom": rt["odom"], "cmd_vel": rt["cmd_vel"]})
    return g
