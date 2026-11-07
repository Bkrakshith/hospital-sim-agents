"""SimService: the backend-neutral sim, and SimRos: its wz.v1 topics on ROS 2.

One service for both backends. It owns the sim clock, the StubWorld logic (objects, reach, audio,
robot tasks, observations), the run streams and the robot's stand-in topics; a Bodies object
(``wardzero_agents.world.bodies``) moves the bodies. ``ros/stub_sim_node`` runs it with StubBodies on a wall-clock loop;
``isaac_host`` runs it with IsaacBodies, stepping it by the physics step.

Each step advances sim time by time_scale x dt (not while held: paused, waiting for the other nodes
to come up, in lockstep while any decision is pending, or at the end), then returns what to publish:
/clock, the roster (on change), statuses (10 Hz sim), observations (2 Hz sim, and at once when there
is new audio), every audio event and what the robot's microphone hears. With a robot it also moves
the body (cmd_vel, goal_pose), runs the task actions and returns odom (20 Hz), camera/bbox_3d,
perception/objects (5 Hz) and task/result.

Nothing here imports rclpy at module level: SimRos imports the message types when it is built.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from ..config.loader import LoadedScenario
from ..contracts import (
    AudioEvent, Control, Heard, Intent, Nudge, Observation, Roster, Say, Status, WorldEvent,
)
from ..runlog import RunLog
from ..world import audio
from ..world.bodies import Bodies, StubBodies
from ..world.oracle import Pose
from ..world.perception import bbox_3d, perceive_objects
from ..world.robot_tasks import RECEPTION_ROLES, RobotTasks, wire_result
from ..world.stub_world import ROBOT_ID as ROBOT_ADDRESSEE, StubWorld
from . import topics

log = logging.getLogger("wardzero.sim")

STEP_HZ = 100.0
STATUS_PERIOD_S = 0.1
STATUS_RECORD_PERIOD_S = 0.5
OBS_PERIOD_S = 0.5
ODOM_PERIOD_S = 0.05  # 20 Hz
PERCEPTION_PERIOD_S = 0.2  # 5 Hz
OBS_WINDOW_S = 2.0
MAX_WALL_DT = 0.5  # a stalled loop does not make the sim jump further than this
REPLY_WINDOW_S = 20.0  # with no toward, a robot line goes to whoever it last heard speak, if that was this recent ...
LAST_LISTENER_M = 3.0  # ... and is still this close to the robot; else it has no addressee
# No longer used for addressing (a line with no toward and no recent speaker near has no addressee); kept because
# stub_sim_node re-exports it.
ROBOT_ADDRESS_M = 2.5
TOWARD_WINDOW_S = 2.0  # a speech/toward waits this long (sim time) for its speech/say, then is dropped
TOWARD_M = 4.0  # a named track is the addressee when that person is this close to the robot and hears the line
READY_TIMEOUT_S = 600.0  # the first knowledge base build can be slow
PENDING_SILENT_S = 5.0  # wall seconds without a pending heartbeat before lockstep stops waiting
WAIT_TOKENS = ("agents", "director")


@dataclass
class SimOutputs:
    clock: float
    roster: Optional[Roster] = None
    statuses: list[Status] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    audio: list[AudioEvent] = field(default_factory=list)
    robot_heard: list[Heard] = field(default_factory=list)
    odom: Optional[dict] = None  # robot pose and twist, 20 Hz
    detections: Optional[dict] = None  # camera/bbox_3d message, 5 Hz
    objects: Optional[dict] = None  # perception/objects message, 5 Hz
    task_results: list[dict] = field(default_factory=list)
    nudges: list[Nudge] = field(default_factory=list)  # to the people the robot offers to guide
    stop: bool = False


_AUTO = object()  # on_robot_say: work the addressee out


def _scripted_lines() -> list:
    """WARDZERO_TEST_SCRIPTED_SAY=<json file>: [{"t": 3.0, "who": "robot" | person id, "text": "...", "to": id}].
    An entry with "do" instead of "text" moves a person: {"t", "who": person id, "do": intent kind, "target": id,
    "point": [x, y], "speed": m/s}. A person moved by the script ignores its own mind from then on.
    For the robot, {"t", "who": "robot", "do": "go_to", "place": landmark or room id | "post", "yaw": rad} sends it a
    goal_pose as its brain would ("post": back to where it started, facing its start heading).
    Unset in every real run; never read by the default fake LLM."""
    path = os.environ.get("WARDZERO_TEST_SCRIPTED_SAY")
    if not path:
        return []
    log.warning("TEST HOOK ACTIVE: WARDZERO_TEST_SCRIPTED_SAY injects scripted speech into this run. "
                "It is not a real run of any robot or person.")
    return sorted(json.loads(Path(path).read_text(encoding="utf-8")), key=lambda d: float(d["t"]))


class SimService:
    """The sim without ROS: feed it messages, call step(dt), publish what it returns.

    ``step(wall_dt)`` advances sim time by time_scale x wall_dt; the Isaac host passes the physics
    step with time_scale 1, so its clock follows the physics, not the wall. Bodies move only in
    ``bodies.step`` (StubBodies by default: the StubWorld's own kinematics).
    """

    def __init__(self, loaded: LoadedScenario, runlog: RunLog, time_scale: float = 1.0,
                 lockstep: bool = False, wait_for: Iterable[str] = WAIT_TOKENS, seed: Optional[int] = None,
                 wall: Callable[[], float] = time.monotonic, bodies: Optional[Bodies] = None):
        if not (time_scale > 0 and math.isfinite(time_scale)):
            raise ValueError("time_scale must be a positive number")
        self.loaded = loaded
        self.runlog = runlog
        self.time_scale = float(time_scale)
        self.lockstep = lockstep
        self.world = StubWorld(loaded, loaded.scenario.seed if seed is None else seed)
        if getattr(loaded.scenario, "forbidden_areas", None):  # the robot routes round the user's no-go areas
            from ..site.areas import robot_router
            self.world.robot.router = robot_router(self.world.robot_router, loaded.scenario.forbidden_areas)
        self.bodies: Bodies = bodies if bodies is not None else StubBodies()
        self.robot = loaded.scenario.robot.enabled
        self.tasks = RobotTasks(self.world)
        self._next_odom = 0.0
        self._next_percept = 0.0
        self.duration_s = loaded.scenario.duration_min * 60.0
        self.t = 0.0
        self.paused = False
        self.pending = 0
        self.wall = wall
        self._pending_at = wall()
        self._pending_silent_logged = False
        self.waiting: set[str] = set(wait_for)
        self._stop = False
        self._roster_key: Optional[tuple] = None
        self._next_status = 0.0
        self._next_status_record = 0.0
        self._next_obs = 0.0
        self._obs_now = False
        self._audio_out: list[AudioEvent] = []
        self._heard_out: list[Heard] = []
        # (emit time, event, {person: what they heard}) for the observation window
        self._recent: deque[tuple[float, AudioEvent, dict[str, Heard]]] = deque()
        self._last_spoken: dict[str, tuple[float, str]] = {}
        # Sim-side bookkeeping only: who the robot's microphone last picked up speaking. The robot
        # itself sees source "unknown"; the sim uses the true id to route the reply.
        self._last_robot_listener: Optional[tuple[float, str]] = None
        # <ns>/speech/toward, a FIFO of (sim time, track or "room" or "" for no name): each is the next say's
        self._towards: deque[tuple[float, str]] = deque()
        self._toward_rng = np.random.default_rng(0)  # only for checking a named person can hear the line
        # Robot lines wait for the next step: over ROS the say callback can run before the toward sent just ahead
        # of it, so a line takes the toward that has arrived by then.
        self._robot_lines: list[str] = []
        self._scripted = _scripted_lines()
        self._scripted_people: set[str] = set()  # test hook: people the script moves; their minds' intents are ignored

    # ---- inputs --------------------------------------------------------------------------
    def ready(self, what: str) -> None:
        if what in self.waiting:
            self.waiting.discard(what)
            log.info("%s ready", what)

    def set_pending(self, n: int) -> None:
        self.pending = max(0, int(n))
        self._pending_at = self.wall()
        self._pending_silent_logged = False

    def _pending_now(self) -> int:
        """The pending count, or 0 once the agent node's 1 Hz heartbeat has been silent for 5 s:
        a dead agent node must not hang a lockstep clock."""
        if self.pending and self.wall() - self._pending_at > PENDING_SILENT_S:
            if not self._pending_silent_logged:
                log.warning("no pending heartbeat for %.0f s; lockstep no longer waits for decisions",
                            PENDING_SILENT_S)
                self._pending_silent_logged = True
            return 0
        return self.pending

    def control(self, ctl: Control) -> None:
        if ctl.action == "pause":
            self.paused = True
        elif ctl.action in ("resume", "start"):
            self.paused = False
        elif ctl.action == "stop":
            self._stop = True
        elif ctl.action == "reset":
            log.warning("reset is not supported during a run; stop it and start a new one")

    def _walls_for(self, ev):
        """The world's walls_between for this event (None: walls cost it nothing), see ``StubWorld.walls_for``."""
        fn = getattr(self.world, "walls_for", None)
        return None if fn is None else fn(ev)

    def _emit(self, events: Iterable[AudioEvent], to_room: bool = False) -> None:
        """Route each event to everyone who can hear it and log it. A robot line's row also carries ``heard_by``
        (who got it) and, for a line to the whole room, ``to_room``; a person's row carries ``robot_heard`` (whether
        the robot's microphone got it)."""
        for ev in events:
            heard: dict[str, Heard] = {}
            for pid, p in self.world.people.items():
                if p.posture == "gone" or p.kind == "static":
                    continue  # patients lying in beds hear nothing
                h = audio.route(ev, Pose(p.x, p.y, p.yaw), pid, p.inst.hearing_loss_db, self.world.rng,
                                walls_between=self._walls_for(ev))
                if h is not None:
                    heard[pid] = h
            self._recent.append((self.t, ev, heard))
            self._audio_out.append(ev)
            rh = self.world.robot_heard(ev) if self.robot else None
            row = ev.model_dump(mode="json", by_alias=True)
            if ev.source_kind == "robot":
                row["heard_by"] = sorted(heard)
                if to_room:
                    row["to_room"] = True
            elif ev.source_kind == "person":
                row["robot_heard"] = rh is not None
            self.runlog.write("audio", row)
            if rh is not None:
                self._heard_out.append(rh)
                if ev.source_kind == "person" and ev.text and ev.addressee in (None, ROBOT_ADDRESSEE):
                    self._last_robot_listener = (self.t, ev.source_id)  # not speech the robot merely overheard
            self._obs_now = True

    def _spoken_before(self, person: str, t: float, text: str) -> bool:
        """A talk intent and a Say for the same utterance arrive separately: speak it once."""
        key = (t, text)
        if self._last_spoken.get(person) == key:
            return True
        self._last_spoken[person] = key
        return False

    def on_intent(self, intent: Intent, scripted: bool = False) -> None:
        if intent.person not in self.world.people:
            log.warning("intent for unknown person %s ignored", intent.person)
            return
        if intent.person in self._scripted_people and not scripted:
            log.info("intent %s for %s ignored: the test script moves this person", intent.kind, intent.person)
            return
        if intent.kind == "talk" and self._spoken_before(intent.person, intent.t, intent.text or ""):
            return
        try:
            events = self.world.apply_intent(intent.model_copy(update={"t": self.t}))
        except ValueError as exc:
            log.warning("intent %s for %s ignored: %s", intent.kind, intent.person, exc)
            return
        self.bodies.on_intent(self.world, intent)
        self._emit(events)

    def on_say(self, say: Say) -> None:
        if say.person not in self.world.people or self.world.people[say.person].posture == "gone":
            log.warning("speech from unknown or absent person %s ignored", say.person)
            return
        if self._spoken_before(say.person, say.t, say.text):
            return
        self._emit([self.world.say(say.person, say.text, say.addressee, self.t)])

    def on_robot_toward(self, track: str) -> None:
        """<ns>/speech/toward: queued for the next speech/say (a FIFO: towards and says pair in order). A track id, or
        ``room`` for a line to everyone in the robot's room; dropped if no say comes within TOWARD_WINDOW_S."""
        self._towards.append((self.t, (track or "").strip()))

    def _drop_stale_towards(self) -> None:
        while self._towards and self.t - self._towards[0][0] > TOWARD_WINDOW_S:
            t, track = self._towards.popleft()
            log.info("speech/toward %s had no speech/say within %.0f s; dropped", track or "(none)", TOWARD_WINDOW_S)

    def _toward_person(self, track: str, ev: AudioEvent) -> Optional[str]:
        """The person the robot named for this line, when they can be spoken to: present, within TOWARD_M of the robot
        (in the camera's view or not: someone it has just walked somewhere may stand behind it) and within hearing of
        the line (no walls in the way). None otherwise."""
        pid = self.world.tracks.person(track)
        p = self.world.people.get(pid) if pid else None
        body = self.world.robot
        if p is None or p.posture == "gone" or p.kind == "static" \
                or math.hypot(p.x - body.x, p.y - body.y) > TOWARD_M:
            return None
        # its own random stream, so asking does not shift the world's draws
        if audio.route(ev, Pose(p.x, p.y, p.yaw), pid, p.inst.hearing_loss_db, self._toward_rng,
                       walls_between=self._walls_for(ev)) is None:
            return None
        return pid

    def on_robot_say(self, text: str, addressee: "Optional[str]" = _AUTO) -> None:
        """``addressee``: left out, the robot's speech goes to the track named by its speech/toward (if near and within
        hearing, else to nobody), to the whole room for the toward ``room``, and with no toward to who it last heard
        (within LAST_LISTENER_M), else to nobody; a person id, or None for nobody (test hook only: nobody then
        answers)."""
        text = (text or "").strip()
        if not text:
            return
        if not self.robot:
            log.warning("robot speech ignored: this scenario has no robot")
            return
        if addressee is not _AUTO:
            self._last_robot_listener = None
            self._emit([self.world.robot_say(text, self.t).model_copy(update={"addressee": addressee})])
            return
        self._robot_lines.append(text)  # sent at the next step, with the toward that has come by then

    def _flush_robot_speech(self) -> None:
        self._drop_stale_towards()
        lines, self._robot_lines = self._robot_lines, []
        for text in lines:
            toward = self._towards.popleft()[1] if self._towards else ""
            self._robot_say_now(text, toward)

    def _robot_say_now(self, text: str, toward: str = "") -> None:
        ev = self.world.robot_say(text, self.t)
        r = self.world.robot_pose
        if toward == topics.TOWARD_ROOM:  # everyone in the room: no addressee, and not an answer to anyone
            self._emit([ev], to_room=True)
            return
        if toward:  # a line for one named person: them, or nobody (never someone else)
            self._last_robot_listener = None
            named = self._toward_person(toward, ev)
            self._emit([ev.model_copy(update={"addressee": named})])
            return
        # No toward: the reply goes to the person the robot just heard, if still near; else to nobody in particular
        # (not to whoever stands nearest).
        last, self._last_robot_listener = self._last_robot_listener, None
        if last is not None and self.t - last[0] <= REPLY_WINDOW_S:
            p = self.world.people.get(last[1])
            if p is not None and p.posture != "gone" and math.hypot(p.x - r.x, p.y - r.y) <= LAST_LISTENER_M:
                self._emit([ev.model_copy(update={"addressee": last[1]})])
                return
        self._emit([ev])

    # ---- the robot's body and tasks --------------------------------------------------------
    @property
    def robot_brief(self) -> Optional[dict]:
        """The task brief to publish latched on task/brief: the scenario's robot_brief, plus its forbidden
        areas as polygons (forbidden_areas) when it has any."""
        from ..site.areas import task_brief
        brief = getattr(self.loaded.scenario, "robot_brief", None)
        if hasattr(brief, "model_dump"):
            brief = brief.model_dump(mode="json")
        out = task_brief(brief, getattr(self.loaded.scenario, "forbidden_areas", None))
        if out is not None:  # deployment knowledge: is anyone at the reception desk computer to read a note?
            out["desk_staffed"] = any(p.kind != "static" and p.posture != "gone" and p.inst.role in RECEPTION_ROLES
                                      for p in self.world.people.values())
        return out

    def on_cmd_vel(self, cmd: "tuple[float, float]") -> None:
        if self.robot:
            self.tasks.cmd_vel(cmd, self.t)

    def on_goal(self, x: float, y: float, yaw: Optional[float] = None) -> None:
        if self.robot:
            self.tasks.goal_pose(x, y, yaw, self.t)

    def on_task_action(self, msg: dict) -> None:
        if not self.robot or not isinstance(msg, dict):
            return
        self.runlog.write("robot_tasks", {"t": self.t, "kind": "action", **{k: v for k, v in msg.items()
                                                                           if k != "t"}})
        self.tasks.handle(msg, self.t)

    def _robot_step(self, dt: float, out: SimOutputs) -> None:
        res = self.tasks.step(self.t, dt)
        self._emit(res.audio)
        out.nudges = list(res.nudges)
        out.task_results = res.results
        for r in res.results:
            self.runlog.write("robot_tasks", {"kind": "result", **r})
        if self.t >= self._next_odom - 1e-9:
            x, y, yaw = self.world.robot.pose
            out.odom = {"t": self.t, "x": x, "y": y, "yaw": yaw, "vx": self.world.robot.vx,
                        "wz": self.world.robot.wz}
            self._next_odom = self.t + ODOM_PERIOD_S
        if self.t >= self._next_percept - 1e-9:
            out.detections = bbox_3d(self.world, self.world.robot, self.t)
            out.objects = {"t": self.t, **perceive_objects(self.world, self.world.robot)}
            self._next_percept = self.t + PERCEPTION_PERIOD_S

    def on_world_event(self, ev: WorldEvent) -> None:
        self._emit(self.world.apply_world_event(ev.model_copy(update={"t": self.t})))

    def _flush_world(self) -> None:
        """Audio the world raised itself (fire alarm) goes out like any audio; object changes are logged."""
        self._emit(self.world.drain_audio())
        for stream, rec in self.world.drain_log():
            self.runlog.write(stream, rec)

    # ---- stepping ------------------------------------------------------------------------
    def _observation(self, pid: str) -> Observation:
        obs = self.world.observation(pid, self.t, [])
        heard = [h[pid] for _, _, h in self._recent if pid in h]
        return obs.model_copy(update={"heard": heard})

    def _speak_scripted(self) -> None:
        """Test only (WARDZERO_TEST_SCRIPTED_SAY): lines said at fixed sim times, so a recording has speech on screen
        without a model. A line for a person who is not in the scene yet waits for them (up to 20 s)."""
        while self._scripted and self._scripted[0]["t"] <= self.t:
            line = self._scripted[0]
            who = str(line.get("who", ""))
            if who == "robot" and line.get("do") == "go_to":
                self._scripted_go_to(line)
            elif who == "robot":
                if "to" in line:  # "to": null (or "") has the line heard by nobody in particular, so nobody answers it
                    self.on_robot_say(str(line["text"]), line["to"] or None)
                else:
                    self.on_robot_say(str(line["text"]))
            elif "do" in line and who in self.world.people and self.world.people[who].posture != "gone":
                self._scripted_people.add(who)
                pt = line.get("point")
                self.on_intent(Intent(id=f"scripted-{line['do']}-{who}-{line['t']}", person=who, t=self.t,
                                      kind=line["do"], target=line.get("target"),
                                      point=tuple(pt) if pt else None, speed=line.get("speed")), scripted=True)
            elif who in self.world.people and self.world.people[who].posture != "gone":
                self.on_say(Say(person=who, t=self.t, text=str(line["text"]), addressee=line.get("to")))
            elif self.t - line["t"] < 20.0:
                return
            self._scripted.pop(0)

    def _scripted_go_to(self, line: dict) -> None:
        r = self.loaded.scenario.robot
        place = str(line.get("place", ""))
        yaw = line.get("yaw")
        if place == "post":
            xy, yaw = (r.x, r.y), (r.yaw if yaw is None else yaw)
        else:
            xy = self.world._resolve_place(place)
        self.runlog.write("robot_tasks", {"t": self.t, "kind": "scripted_goal", "place": place,
                                          "xy": list(xy) if xy else None})
        if xy is not None:
            self.on_goal(float(xy[0]), float(xy[1]), None if yaw is None else float(yaw))

    @property
    def held(self) -> bool:
        """The clock must not move: paused, waiting for the other nodes, a lockstep decision pending,
        or the scenario is over. The Isaac host pauses its timeline while this is true."""
        return (self.paused or bool(self.waiting) or (self.lockstep and self._pending_now() > 0)
                or self.t >= self.duration_s)

    def step(self, wall_dt: float) -> SimOutputs:
        wall_dt = min(max(0.0, wall_dt), MAX_WALL_DT)
        self._flush_robot_speech()  # before the clock moves: the lines keep the time they were said
        held = self.held
        t0 = self.t
        if not held and not self._stop:
            self.t = min(self.duration_s, self.t + self.time_scale * wall_dt)
            self.bodies.step(self.world, self.t)
        self._speak_scripted()
        self._flush_robot_speech()
        self._flush_world()
        out = SimOutputs(clock=self.t)
        if self.robot:
            self._robot_step(self.t - t0, out)
            self._flush_world()

        roster = self.world.roster(self.t)
        key = tuple((p.id, p.persona) for p in roster.people)
        if key != self._roster_key:
            self._roster_key = key
            out.roster = roster

        if self.t >= self._next_status:
            out.statuses = self.world.statuses()
            self._next_status = self.t + STATUS_PERIOD_S
            if self.t >= self._next_status_record:
                for st in out.statuses:
                    self.runlog.write("status", st.model_dump(mode="json", by_alias=True))
                if self.robot:  # the robot's pose at the same instants, for the benchmark metrics
                    x, y, yaw = self.world.robot.pose
                    self.runlog.write("robot_pose", {"t": self.t, "x": round(x, 3), "y": round(y, 3),
                                                     "yaw": round(yaw, 4),
                                                     "speed": round(self.world.robot.speed, 3)})
                self._next_status_record = self.t + STATUS_RECORD_PERIOD_S

        while self._recent and self._recent[0][0] < self.t - OBS_WINDOW_S:
            self._recent.popleft()
        if self._obs_now or self.t >= self._next_obs:
            out.observations = [self._observation(pid) for pid, p in self.world.people.items()
                                if p.posture != "gone" and p.kind != "static"]
            self._obs_now = False
            self._next_obs = self.t + OBS_PERIOD_S

        out.audio, self._audio_out = self._audio_out, []
        out.robot_heard, self._heard_out = self._heard_out, []
        out.stop = self._stop or self.t >= self.duration_s
        return out


# ---- ROS ------------------------------------------------------------------------------------
def default_msgs():
    """The ROS message types and QoS SimRos uses (imports rclpy)."""
    from types import SimpleNamespace

    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Int32, String

    from .rosutil import QOS, QOS_LATCHED
    return SimpleNamespace(Clock=Clock, Odometry=Odometry, Twist=Twist, PoseStamped=PoseStamped, Int32=Int32,
                           String=String, QOS=QOS, QOS_LATCHED=QOS_LATCHED)


class SimRos:
    """A SimService's topics on a RosBridge: wz.v1 on the private domain plus the robot's topics (audio/heard,
    speech/say, goal_pose, task/*, perception/objects and the stand-ins for what a sensor would give).

    ``clock_bridges`` get /clock too. ``robot_ros`` is the robot's own bridge (the Isaac host's robot domain): every
    robot topic goes there and the people's stay on ``ros``; without it all topics share ``ros``. ``graph_owned`` are
    robot_topics keys (``bbox_3d``, ``odom``) that the Isaac robot graph publishes, so the service does not."""

    def __init__(self, svc: SimService, ros: Any, clock_bridges: Iterable[Any] = (), msgs: Any = None,
                 robot_ros: Any = None, graph_owned: Iterable[str] = (), robot_clock_offset: float = 0.0):
        import json

        self.svc, self.ros = svc, ros
        self.robot_ros = rr = robot_ros if robot_ros is not None else ros
        self.graph_owned = frozenset(graph_owned)
        self.m = m = msgs if msgs is not None else default_msgs()
        self._json = json
        self.clock_pubs = [ros.publisher(topics.CLOCK, m.Clock, m.QOS)]
        # the robot domains' /clock: svc.t plus a constant, the Isaac host's own simulation time (see ``stamp``)
        self.robot_clock_offset = float(robot_clock_offset)
        self.robot_clock_pubs = [b.publisher(topics.CLOCK, m.Clock, m.QOS) for b in clock_bridges]
        for pid in svc.world.people:
            self.subscribe_person(pid)
        rcfg = svc.loaded.scenario.robot
        self.rt = rt = topics.robot_topics(rcfg.ns)
        say_topics = {rt["say"]}
        self.heard_topics = [rt["heard"]]
        if topics.uses_g1_alias(rcfg.ns, rcfg.profile):  # the reception bot and old tests keep working
            say_topics.add(topics.G1_SAY)
            self.heard_topics.append(topics.G1_HEARD)
        for name in sorted(say_topics):
            rr.subscribe(m.String, name, svc.on_robot_say, m.QOS, convert=lambda msg: msg.data)
        rr.subscribe(m.String, rt["say_toward"], svc.on_robot_toward, m.QOS, convert=lambda msg: msg.data)
        rr.subscribe(m.Twist, rt["cmd_vel"], svc.on_cmd_vel, m.QOS,
                     convert=lambda msg: (msg.linear.x, msg.angular.z))
        rr.subscribe(m.PoseStamped, rt["goal_pose"], self._on_goal, m.QOS)
        rr.subscribe(m.String, rt["task_action"], svc.on_task_action, m.QOS, convert=self._parse_action)
        self.odom_pub = None if "odom" in self.graph_owned else rr.publisher(rt["odom"], m.Odometry, m.QOS)
        for key in ("bbox_3d", "objects", "task_result"):
            if key not in self.graph_owned:
                rr.publisher(rt[key], m.String, m.QOS)
        if svc.robot_brief is not None:
            rr.publish_text(rt["task_brief"], json.dumps(svc.robot_brief), m.QOS_LATCHED)
        ros.subscribe_json(WorldEvent, topics.WORLD_EVENT, svc.on_world_event)
        ros.subscribe_json(Control, topics.CONTROL, svc.control)
        ros.subscribe(m.Int32, topics.PENDING, self._on_pending, m.QOS, convert=lambda msg: int(msg.data))
        ros.subscribe(m.String, topics.TIMELINE, lambda _msg: svc.ready("director"), m.QOS)

    def subscribe_person(self, pid: str) -> None:
        self.ros.subscribe_json(Intent, topics.intent(pid), self.svc.on_intent)
        self.ros.subscribe_json(Say, topics.say(pid), self.svc.on_say)

    def _on_goal(self, msg) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.svc.on_goal(msg.pose.position.x, msg.pose.position.y, yaw)

    def _parse_action(self, msg) -> dict:
        try:
            d = self._json.loads(msg.data)
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}

    def _on_pending(self, n: int) -> None:
        self.svc.ready("agents")
        self.svc.set_pending(n)

    def _clock_msg(self, t: float):
        """A Clock message for t seconds. One rounding, then divmod: sec and nanosec cannot disagree at a whole second."""
        msg = self.m.Clock()
        msg.clock.sec, msg.clock.nanosec = divmod(int(round(t * 1e9)), 1_000_000_000)
        return msg

    def publish(self, out: SimOutputs) -> None:
        m, ros, rt = self.m, self.ros, self.rt
        msg = self._clock_msg(out.clock)
        for pub in self.clock_pubs:
            pub.publish(msg)
        robot_msg = msg
        if self.robot_clock_pubs:
            robot_msg = self._clock_msg(out.clock + self.robot_clock_offset) if self.robot_clock_offset else msg
            for pub in self.robot_clock_pubs:
                pub.publish(robot_msg)
        if out.roster is not None:
            for p in out.roster.people:
                self.subscribe_person(p.id)
            ros.publish_json(topics.ROSTER, out.roster, m.QOS_LATCHED)
        for st in out.statuses:
            ros.publish_json(topics.status(st.person), st)
        for ev in out.audio:
            ros.publish_json(topics.AUDIO_EVENTS, ev)
        for obs in out.observations:
            ros.publish_json(topics.observation(obs.person), obs)
        for h in out.robot_heard:
            for name in self.heard_topics:
                self.robot_ros.publish_json(name, h)
        if out.odom is not None and self.odom_pub is not None:
            o = m.Odometry()
            o.header.frame_id, o.child_frame_id = "map", "base_link"
            o.header.stamp = robot_msg.clock
            o.pose.pose.position.x, o.pose.pose.position.y = out.odom["x"], out.odom["y"]
            o.pose.pose.orientation.z = math.sin(out.odom["yaw"] / 2)
            o.pose.pose.orientation.w = math.cos(out.odom["yaw"] / 2)
            o.twist.twist.linear.x, o.twist.twist.angular.z = out.odom["vx"], out.odom["wz"]
            self.odom_pub.publish(o)
        if out.detections is not None:
            if "bbox_3d" not in self.graph_owned:
                self.robot_ros.publish_text(rt["bbox_3d"], self._json.dumps(out.detections))
            self.robot_ros.publish_text(rt["objects"], self._json.dumps(out.objects))
        for r in out.task_results:
            self.robot_ros.publish_text(rt["task_result"], self._json.dumps(wire_result(r)))
        for n in out.nudges:
            ros.publish_json(topics.nudge(n.person), n)

    def publish_stop(self) -> None:
        """Tell the other nodes the run is over."""
        self.ros.publish_json(topics.CONTROL, Control(action="stop"))
