"""Small ROS 2 helpers run as their own short process: publish the stop control, and look at the graph.

    python -m wardzero_agents.studio.ros_probe stop
    python -m wardzero_agents.studio.ros_probe probe NS      # prints one JSON line

Run in a subprocess so the Studio server never imports rclpy or holds a ROS context.
"""
from __future__ import annotations

import json
import sys
import time

INBOUND = ("camera/rgb", "camera/depth", "camera/camera_info", "camera/bbox_3d", "lidar/point_cloud",
           "perception/objects", "audio/heard", "odom", "joint_states", "task/brief")
COMMANDS = ("cmd_vel", "joint_command", "goal_pose", "speech/say", "task/action")


def _stop() -> int:
    import rclpy
    from std_msgs.msg import String
    from ..contracts import Control, to_json
    from ..ros.rosutil import QOS
    from ..ros.topics import CONTROL
    rclpy.init()
    node = rclpy.create_node("studio_stop")
    pub = node.create_publisher(String, CONTROL, QOS)
    msg = String()
    msg.data = to_json(Control(action="stop"))
    end = time.monotonic() + 1.5  # let discovery find the subscribers, then send a few times
    while time.monotonic() < end:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    return 0


def _probe(ns: str) -> int:
    import rclpy
    rclpy.init()
    node = rclpy.create_node("studio_probe")
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    p = f"/{ns.strip('/')}" if ns.strip("/") else ""
    others = [n for n in node.get_node_names() if n != "studio_probe"]
    subs = any(node.get_subscriptions_info_by_topic(f"{p}/{t}") for t in INBOUND)
    cmds = any(node.get_publishers_info_by_topic(f"{p}/{t}") for t in COMMANDS)
    print(json.dumps({"controller_seen": bool(others), "sensors_subscribed": bool(subs), "commands_seen": bool(cmds)}))
    node.destroy_node()
    rclpy.shutdown()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["stop"]:
        return _stop()
    if argv[:1] == ["probe"]:
        return _probe(argv[1] if len(argv) > 1 else "robot")
    return 2


if __name__ == "__main__":
    sys.exit(main())
