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
| `<ns>/camera/bbox_3d` | `std_msgs/String` (JSON) | in | 5 Hz. Anonymous person tracks (see below); a track within 4 m of someone asleep carries `"asleep": true`, and of someone who looks unwell (slumped) `"unwell": true`. |
| `<ns>/lidar/point_cloud` | `sensor_msgs/PointCloud2` | in | Isaac host only. |
| `<ns>/perception/objects` | `std_msgs/String` (JSON) | in | 5 Hz. Items, held items, beds, fires, call lights, seats (`id`, `x`, `y`, `free`; never a staff chair) and route blocked signs (`blocked`: `id`, `x`, `y`, `text`) in view. |
| `<ns>/audio/heard` | `std_msgs/String` (JSON) | in | What the microphone picks up. `source_id` is always `unknown`. A call bell or the robot's own staff call comes as `source_kind: "bell"`. |
| `<ns>/speech/say` | `std_msgs/String` | out | Plain text. 72 dB at 1 m. |
| `<ns>/speech/toward` | `std_msgs/String` | out | Optional. Who the next `speech/say` line is for: a track id (`t3`), or `room` for everyone in the robot's room (no one person). See [Addressing a line](#addressing-a-line-speechtoward). |
| `<ns>/goal_pose` | `geometry_msgs/PoseStamped` | out | Walk to a point on the map. Mobile robots only. |
| `<ns>/task/action` | `std_msgs/String` (JSON) | out | One task action (below). |
| `<ns>/task/result` | `std_msgs/String` (JSON) | in | One result per action, and one per goal. |
| `<ns>/task/brief` | `std_msgs/String` (JSON) | in | Latched. `{role, job}`: the robot's job for this scenario, plus `forbidden_areas` when the scenario has any (below), `desk_staffed` (whether a receptionist is on duty, so `note_desk` can work) and any other fields the scenario's brief sets (for example `first_contact: false` when greeting people is not part of the job). |

The stub sim publishes only: clock, odom, bbox_3d, objects, audio/heard, task/result and
task/brief, and listens on cmd_vel, goal_pose, speech/say, speech/toward and task/action. The Isaac host adds
the rest, from the robot's OmniGraph on the robot domain: `/clock`, `/tf`, joint states and
commands, the camera, and for a wheeled base that can move `odom` and `cmd_vel` as a Twist into a
differential drive. Its `camera/bbox_3d` is the same JSON as the stub's: the host turns Isaac's
camera boxes (world frame, numeric classes) into box centres in the `map` frame with anonymous
tracks (5 Hz). Isaac drops 3D boxes cut by the image border, so people closer than about 1.5 m, who do not fit the camera's 65 degree vertical view, are added by the host from the world (same shape, `"truncated": true`, within 1.8 m, in the view cone and line of sight). Lidar point clouds and walking legged bases (`cmd_vel` for the G1) are not wired yet.

## Addressing a line (`speech/toward`)

`speech/toward` and `speech/say` pair in order, first in first out: each `toward` goes with the next `say` line, so two
lines sent one after the other each reach the person meant. A `toward` that waits more than 2 s of sim time with no
`say` after it is dropped.

- A track id (`t3`): the line is addressed to that person when they are within 4 m of the robot and can hear it (in
  the camera's view or not: someone the robot has just walked somewhere may stand behind it). Otherwise (farther
  away, gone, or walls in the way) the line is said to nobody in particular; it never goes to anyone else.
- `room`: the line is for everyone in the robot's room, with no one addressee (a call to the whole waiting room, for
  example). Everyone who hears it hears it as said to the room.
- No `toward`: the line is addressed to the last person whose speech the robot heard, said to the robot or to nobody
  in particular (not a line said to someone else), if that was within the last 20 s and they are still within 3.0 m
  of it (once: the next line without a `toward` does not go to them again); otherwise it
  is said to nobody in particular, however near anyone stands.

Whoever is within hearing hears every line, addressed or not (the audio model: distance, and walls between rooms). The
run record lists, for every robot line, who heard it (`heard_by` in `audio.jsonl`, with `addressee` for the
addressee and `to_room: true` for a `room` line), and for every sound a person made whether the robot's microphone
got it (`robot_heard: true` or `false`). A flow step or a score that asks whether the robot spoke to someone counts a line
addressed to them, a `room` line they heard, and a line to nobody in particular that they heard within 4 m of the
robot.

## Forbidden areas (`task/brief`)

A Studio project can mark rectangles the robot must keep out of (Scene page, "Forbidden area"). They
come in the latched task brief as polygons in the `map` frame, metres, corners counter-clockwise:

```json
{"role": "...", "job": "...",
 "forbidden_areas": [{"id": "fa1", "name": "Resus", "polygon": [[-1.0, 1.0], [2.0, 1.0], [2.0, 3.0], [-1.0, 3.0]]}]}
```

The key is absent when there are none. Each time the robot's pose goes from outside to inside one, the
report's `forbidden_area_entries` counts it and lists it with the area's name and time. The built-in G1
routes round them and is told their names. Isaac does not draw them.

## Task actions (`task/action`)

Each message is a JSON object with a unique `id` and an `action`:

| `action` | Fields | What it does |
|---|---|---|
| `hand_over` | `item`, `to` (a track), optional `place` `[x, y, z]` | Holds an item out to a person; the result comes when it resolves (below). An item the robot does not hold yet but can take (unheld, within 0.8 m) is taken first and offered, in one action. |
| `take` | `item` | Picks up an unheld item within 0.8 m. |
| `follow` | `track` | Walks to the person and stays 1.0 to 1.5 m from them as they move, re-planning as it goes and keeping its clearance from everyone. Result `ok` with reason `following` at the start, or `ok: false` with `no_person` for an unknown track. It goes on until `guide_stop`, a new `goal_pose`, `cmd_vel`, another walking action, or the person leaves. |
| `extinguish` | `fire` | Sprays until the fire is out or the robot moves away. |
| `press_call` | optional `track` | Chimes at the call point serving the robot's area (the nearest nurse base or triage, in its own compartment first). Every staff member who hears it is told where the robot is, which call lights are on and, with `track`, who needs them. |
| `guide_start` | `track`, `place`, optional `seat` (true), optional `return_to_post` (default true) | Walks to the place at the person's pace; the person is offered a walk and can follow. With `seat`, it ends beside the nearest free seat the robot can see there and shows it to them. With `return_to_post: false` the robot stays at the place when the guide ends. |
| `approach` | `track` | Walks up to the person (1.2 m from them) and faces them. Result `started`, `already_there`, `no_route`, `unknown_track` or `cannot_walk`; the walk reports as a goal. |
| `guide_stop` | none | Cancels guiding, or following. |
| `tell_staff` | `about` (a track), optional `message` (text) | At reception, walks to the counter, faces the desk and says `message` to the member of staff there. On a ward, it is led to the nearest ward staff member (it speaks from where it stands when she is within 5 m in sight). Then it walks back to its post. |
| `note_desk` | `about` (a track), `message` (text) | Leaves a note about the person on the reception desk computer. The robot does not move or speak. |

`tell_staff`, for example:

```json
{"id": "rb4", "action": "tell_staff", "about": "t2", "message": "The lady by the entrance needs a nurse, please."}
```

The staff member hears it only if they are within hearing (the audio model, and within 5 m): they are often away from
the desk. If they heard it, they also learn who the person is and where, and decide for themselves whether to go and
help. The result is `ok` with reason `told`, or `ok: false` with `no_staff_to_tell` (no receptionist or ward staff
in this scenario), `staff_not_at_desk` (nobody heard it), `no_route` (no way there), `stuck` (there was a way, but the
walk could not get on), `blocked_by_person` (someone stayed in the way and the staff member was not within 5 m),
`cancelled`, `unknown_track` or `superseded` (another goal replaced the walk there). The robot is never told where the staff member is. A robot that cannot walk speaks from
where it stands. Without a `message` the robot says "Excuse me, someone here needs help from a nurse."

`press_call` always ends `ok`, with reason `answered` when at least one member of staff heard the chime and
`unanswered` when nobody did. Scores and flows count only an answered call as an alert.

`note_desk` is the quiet way to involve the desk: the receptionist is often busy at the computer, and a note
interrupts nobody while the robot stays with the person. The result comes at once: `ok` with reason `noted`, or
`ok: false` with `no_desk_staff` (no receptionist in this scenario) or `unknown_track`. The receptionist reads the note
when she is at her chair, straight away or when she gets back, and decides for herself whether to act on it.

While guiding, the robot waits when the person is more than 2.5 m behind it and walks on when they are within 2 m.
Anyone within 1.5 m slows it to 0.3 m/s, except the person it guides while they walk behind it (see Speed near people
below); it keeps its clearance from the person it guides as from anyone else (the guide's goal is placed clear of
them). At the place it waits up to 20 s for the
person, says goodbye and walks back to where it started (a goal with id `<action id>-return`), unless the guide was
started with `return_to_post: false` or a fire alarm is sounding: then it stays where it is. If the person never comes
within 2.5 m for 30 s the guide is given up (its goal ends `cancelled`). The person counts as guided only when they
themselves reach the place. A person the guide has ended for no longer follows the robot.

A `hand_over` result is `ok` (reason `ok`) once the person takes the item, or `ok: false` with `refused` (the
person declined it), `no_answer` (the offer stayed open with no reply, or the person is asleep), `out_of_reach_receiver`
(the robot can reach the hand-over point but the person cannot), `out_of_reach_giver` (the robot cannot reach it),
`not_holding`, `unknown_track` or `superseded` (a newer offer to the same person replaced it). A patient lying flat who
accepts sits up to take it. The run record logs the moment the robot holds the item out (`robot_tasks.jsonl`, kind
`hand_over`, event `offer`), so a hand-over's timing is measured to the offer, not to the person's reply.

A person asleep (a patient at night, say) carries `"asleep": true` on their track once the robot is within 4 m. They
wake when the robot speaks to them (a line addressed to their track) or when they wake by themselves; until then an
offer to them ends `no_answer`.

An unknown action or a malformed message gets a result with `ok: false` and reason
`unknown_action` or `bad_request`.

## Results (`task/result`)

Every action ends in exactly one result:

```json
{"id": "rb3", "action": "press_call", "ok": true, "reason": "answered", "t": 41.2}
```

A goal (from `goal_pose`, or the walking part of `guide_start`) reports its outcome as:

```json
{"kind": "goal", "id": "goal_pose-1", "action": "goal", "ok": false, "reason": "stuck", "t": 63.0}
```

`reason` for a goal is one of `arrived`, `stuck` (a route existed but the robot could not get on along it),
`no_route`, `blocked_by_person` (someone stayed in its way until it gave up), `superseded` (a newer goal replaced it)
or `cancelled`. Goal ids are `goal_pose-N` for `goal_pose`, the action id for
a guide or for the walk of a `tell_staff`, and `<id>-return` for the walk back after a guide or a `tell_staff`. Action results carry no `kind`, so a consumer must key results on the pair
`(id, kind)`: an action and a goal can share an id.

## Message shapes

`camera/bbox_3d`: `{"header": {"stamp", "frame_id": "map"}, "detections": [{"id": "t1",
"results": [{"class_id": "person", "score": 1.0}], "bbox": {"center": {x, y, z, yaw}, "size":
{x, y, z}}, "truncated": true}]}` (`truncated` appears only on the Isaac host, for a person too close to fit the
camera image: the box is the world's estimate, not Isaac's). Posture is read from the box: about 1.7 m tall is standing, about 1.2 m is
seated, a low long box is a person lying down.

`audio/heard`: `{"source_kind": "person"|"robot"|"pa"|"cough"|"alarm"|"bell", "source_id": "unknown",
"text": "...", "received_db": 58.0, "partial": false, "addressed_to_me": true, "bearing_deg": -32.0}`. `bearing_deg`
(degrees in the robot frame, positive to the left, about 10 degrees of noise) comes only with a voice, a cough or a
call bell at the bed.

The call system arrives as `source_kind: "bell"`, never as a person speaking, with a text that says what it is: a
bed's call bell is `"call bell, bed B3"`, the robot's own staff call from `press_call` is `"staff call chime"`. A bell
chimes twice: at the bed, with a bearing, and on the repeater at the nurse base, without one (the repeater is not
where the patient is). The staff call chime has no bearing either.

Speed near people: within 1.5 m of anyone the robot is held to 0.3 m/s, and the limit is applied before it moves
each step. The one exception is the person it is guiding, while a guide is under way and they are behind the robot
(relative to its heading): they do not slow it. Anyone else within 1.5 m does, and so does the person it guides when
they are beside or ahead of it. The report's `max_speed_near_people_mps` applies the same rule and measures the
robot's own motion: moments when a person closed the gap or pushed the robot are left out. A contact counts against
the robot only when its own motion closed the gap: a person who walks into a robot standing still or backing away is
not a robot contact.
