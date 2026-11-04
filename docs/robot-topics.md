# Robot topics (section 7 of the interface contract)

The robot under test talks to the sim only through these topics. `<ns>` is the robot's
namespace (default `robot`, set per robot in the Studio); the built-in `g1` namespace keeps
`/g1/audio/heard` and `/g1/speech/say` as aliases. The names come from `robot_topics(ns)` in
`wardzero_agents/ros/topics.py`.

## Fairness boundary

The robot runs on its own ROS 2 domain (`--robot-domain N`, or the Studio's robot page). The
sim host (the Isaac host and the 2D stub alike) publishes only the robot topics below and `/clock` on that domain. The people's
topics (`/humans/...`, `/scenario/...`, `/sim/audio/events`, `/sim/world_event`,
`/agents/pending`) stay on the private domain, where the robot cannot see them. The robot
never learns who a person is: detections are anonymous tracks (`t1`, `t2`), and the microphone
reports `source_id: "unknown"`. `hhumans audit` still flags any non-`wardzero_` node that
subscribes to a forbidden topic on the private domain.

The lab's own nodes on the robot domain (the host's bridge, the camera graph) carry the `wardzero_` prefix, which the
audit allows; the robot's nodes must not start with it.

## Topics

Direction is from the robot's side. Sensors are optional: a topic exists only if the robot has
the matching sensor (a camera for the camera topics, a microphone for `audio/heard`, a speaker
for `speech/say`, a mobile base for `cmd_vel` and `goal_pose`).

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | in | Sim time. Use it (`use_sim_time`). |
| `/tf` | `tf2_msgs/TFMessage` | in | Isaac host only. |
| `<ns>/joint_states` | `sensor_msgs/JointState` | in | Isaac host. Every joint of the articulation. |
| `<ns>/joint_command` | `sensor_msgs/JointState` | out | Isaac host. Position targets by joint name; a robot with its own controller drives its arms and hands this way. The built-in G1 moves its own arm for hand-overs, so leave it alone. |
| `<ns>/cmd_vel` | `geometry_msgs/Twist` | out | Linear x and angular z. Mobile robots only. |
| `<ns>/odom` | `nav_msgs/Odometry` | in | 20 Hz, frame `map`. |
| `<ns>/camera/rgb`, `camera/depth`, `camera/camera_info` | `sensor_msgs/Image`, `CameraInfo` | in | Isaac host only. 5 Hz, from the robot's camera mount (frame = the sensor's link). `depth` only for a depth camera (kind `rgbd`). |
| `<ns>/camera/bbox_3d` | `std_msgs/String` (JSON) | in | 5 Hz. Anonymous person tracks (see below). |
| `<ns>/lidar/point_cloud` | `sensor_msgs/PointCloud2` | in | Isaac host only. |
| `<ns>/perception/objects` | `std_msgs/String` (JSON) | in | 5 Hz. Items, held items, beds, fires, call lights in view. |
| `<ns>/audio/heard` | `std_msgs/String` (JSON) | in | What the microphone picks up. `source_id` is always `unknown`. |
| `<ns>/speech/say` | `std_msgs/String` | out | Plain text. 72 dB at 1 m. |
| `<ns>/goal_pose` | `geometry_msgs/PoseStamped` | out | Walk to a point on the map. Mobile robots only. |
| `<ns>/task/action` | `std_msgs/String` (JSON) | out | One task action (below). |
| `<ns>/task/result` | `std_msgs/String` (JSON) | in | One result per action, and one per goal. |
| `<ns>/task/brief` | `std_msgs/String` (JSON) | in | Latched. `{role, job}`: the robot's job for this scenario. |

The stub sim publishes only: clock, odom, bbox_3d, objects, audio/heard, task/result and
task/brief, and listens on cmd_vel, goal_pose, speech/say and task/action. The Isaac host adds
the rest, from the robot's OmniGraph on the robot domain: `/clock`, `/tf`, joint states and
commands, the camera, and for a wheeled base that can move `odom` and `cmd_vel` as a Twist into a
differential drive. Its `camera/bbox_3d` is the same JSON as the stub's: the host turns Isaac's
camera boxes (world frame, numeric classes) into box centres in the `map` frame with anonymous
tracks (5 Hz). Isaac drops 3D boxes cut by the image border, so people closer than about 1.5 m, who do not fit the camera's 65 degree vertical view, are added by the host from the world (same shape, `"truncated": true`, within 1.8 m, in the view cone and line of sight). Lidar point clouds and walking legged bases (`cmd_vel` for the G1) are not wired yet.

## Task actions (`task/action`)

Each message is a JSON object with a unique `id` and an `action`:

| `action` | Fields | What it does |
|---|---|---|
| `hand_over` | `item`, `to` (a track), optional `place` `[x, y, z]` | Holds an item out to a person; the result comes when it resolves. |
| `take` | `item` | Picks up an unheld item within 0.8 m. |
| `extinguish` | `fire` | Sprays until the fire is out or the robot moves away. |
| `press_call` | none | Chimes at the nearest nurse base. |
| `guide_start` | `track`, `place` | Heads to the place; the person is offered a walk. |
| `guide_stop` | none | Cancels guiding. |

An unknown action or a malformed message gets a result with `ok: false` and reason
`unknown_action` or `bad_request`.

## Results (`task/result`)

Every action ends in exactly one result:

```json
{"id": "rb3", "action": "press_call", "ok": true, "reason": "ok", "t": 41.2}
```

A goal (from `goal_pose`, or the walking part of `guide_start`) reports its outcome as:

```json
{"kind": "goal", "id": "goal_pose-1", "action": "goal", "ok": false, "reason": "stuck", "t": 63.0}
```

`reason` for a goal is one of `arrived`, `stuck`, `no_route`, `superseded` (a newer goal
replaced it) or `cancelled`. Goal ids are `goal_pose-N` for `goal_pose` and the action id for
a guide. Action results carry no `kind`, so a consumer must key results on the pair
`(id, kind)`: an action and a goal can share an id.

## Message shapes

`camera/bbox_3d`: `{"header": {"stamp", "frame_id": "map"}, "detections": [{"id": "t1",
"results": [{"class_id": "person", "score": 1.0}], "bbox": {"center": {x, y, z, yaw}, "size":
{x, y, z}}, "truncated": true}]}` (`truncated` appears only on the Isaac host, for a person too close to fit the
camera image: the box is the world's estimate, not Isaac's). Posture is read from the box: about 1.7 m tall is standing, about 1.2 m is
seated, a low long box is a person lying down.

`audio/heard`: `{"source_kind": "person"|"robot"|"pa"|"cough"|"alarm", "source_id": "unknown",
"text": "...", "received_db": 58.0, "partial": false, "addressed_to_me": true}`.
