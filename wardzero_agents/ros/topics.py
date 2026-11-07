"""ROS 2 topic names for the sim boundary. Pure strings and functions, no rclpy."""

ROSTER = "/humans/roster"

AUDIO_EVENTS = "/sim/audio/events"
G1_HEARD = "/g1/audio/heard"
G1_SAY = "/g1/speech/say"
WORLD_EVENT = "/sim/world_event"
TIMELINE = "/scenario/timeline"
SCEN_EVENTS = "/scenario/events"
CONTROL = "/scenario/control"
PENDING = "/agents/pending"

CLOCK = "/clock"
G1_ALIAS_NS = "g1"

FORBIDDEN_FOR_ROBOT = ("/humans/", "/scenario/", "/sim/audio/events", "/sim/world_event", "/agents/pending")


def status(pid: str) -> str:
    return f"/humans/{pid}/status"


def intent(pid: str) -> str:
    return f"/humans/{pid}/intent"


def observation(pid: str) -> str:
    return f"/humans/{pid}/observation"


def say(pid: str) -> str:
    return f"/humans/{pid}/say"


def nudge(pid: str) -> str:
    return f"/humans/{pid}/nudge"


def reflex(pid: str) -> str:
    return f"/humans/{pid}/reflex"


TOWARD_ROOM = "room"  # speech/toward value: the line is for everyone in the robot's room (no addressee)


def robot_topics(ns: str = "robot") -> dict[str, str]:
    """Every robot-side topic for namespace ns, keyed by a short name.

    Standard Isaac ROS 2 bridge names plus WardZero's own. The stub sim publishes only clock, odom,
    cmd_vel, goal_pose, bbox_3d, audio/heard, speech/say and task/*; the rest are listed for the
    Studio's topic table and belong to the Isaac host.
    """
    ns = ns.strip("/")
    p = f"/{ns}" if ns else ""
    return {
        "clock": CLOCK,
        "tf": "/tf",
        "joint_states": f"{p}/joint_states",
        "joint_command": f"{p}/joint_command",
        "cmd_vel": f"{p}/cmd_vel",
        "odom": f"{p}/odom",
        "camera_rgb": f"{p}/camera/rgb",
        "camera_depth": f"{p}/camera/depth",
        "camera_info": f"{p}/camera/camera_info",
        "bbox_3d": f"{p}/camera/bbox_3d",
        "lidar": f"{p}/lidar/point_cloud",
        "objects": f"{p}/perception/objects",
        "heard": f"{p}/audio/heard",
        "say": f"{p}/speech/say",
        # optional: who the next speech/say is for (String), sent just before it: a track id, or TOWARD_ROOM
        "say_toward": f"{p}/speech/toward",
        "goal_pose": f"{p}/goal_pose",
        "task_action": f"{p}/task/action",
        "task_result": f"{p}/task/result",
        "task_brief": f"{p}/task/brief",
    }


def raw_detection_topics(ns: str = "robot") -> dict[str, str]:
    """Isaac's own camera detections, before the host turns them into the robot's bbox_3d. Private domain only:
    world-frame boxes with numeric classes are the host's input, not a robot topic."""
    ns = ns.strip("/") or "robot"
    return {"bbox_3d": f"/wz/{ns}/bbox_3d_raw", "labels": f"/wz/{ns}/semantic_labels_raw"}


def isaac_graph_keys(base: str, can_move: bool, sensor_kinds=()) -> tuple[str, ...]:
    """The robot_topics keys the Isaac robot graph publishes or listens on for this robot.

    Always the TF tree and the joint topics (never the clock: the host publishes /clock from its own time). A colour camera gives camera_rgb, camera_info and bbox_3d
    (the host's adapter publishes the last), a depth camera (kind rgbd) camera_depth too. A wheeled base that can
    move gets odom and cmd_vel. Lidar point clouds are not part of the graph yet."""
    kinds = set(sensor_kinds)
    keys = ["tf", "joint_states", "joint_command"]
    if kinds & {"rgb", "rgbd"}:
        keys += ["camera_rgb", "camera_info", "bbox_3d"]
    if "rgbd" in kinds:
        keys.append("camera_depth")
    if base == "wheeled" and can_move:
        keys += ["odom", "cmd_vel"]
    return tuple(keys)


STUB_LIVE = ("clock", "odom", "cmd_vel", "goal_pose", "bbox_3d", "objects", "heard", "say", "say_toward",
             "task_action", "task_result", "task_brief")


def uses_g1_alias(ns: str, profile: str) -> bool:
    """/g1/audio/heard and /g1/speech/say stay live for the g1 namespace and the g1_claude profile."""
    return ns.strip("/") == G1_ALIAS_NS or profile == "g1_claude"
