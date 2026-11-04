"""Stub sim node (wardzero_stub_sim): the kinematic StubWorld on ROS, standing in for Isaac.

It owns the world and the sim clock. Each wall step it advances sim time by time_scale x wall dt
(not while paused, not while it waits for the other nodes to come up, and in lockstep not while
any decision is pending), then publishes /clock, the roster (on change), statuses (10 Hz sim),
observations (2 Hz sim, and at once when there is new audio), every audio event and what the
robot's microphone hears. With a robot it also moves the body (cmd_vel, goal_pose), runs the task
actions and publishes odom (20 Hz), camera/bbox_3d, perception/objects (5 Hz) and task/result.

    python -m wardzero_agents.ros.stub_sim_node --run-dir DIR --scenario FILE [--time-scale F]
        [--lockstep] [--wait-for agents,director] [--ready-timeout 600]
"""
from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from ..config.loader import ConfigError, LoadedScenario, load_scenario
from ..contracts import (
    AudioEvent, Control, Heard, Intent, Nudge, Observation, Roster, Say, Status, WorldEvent,
)
from ..runlog import RunLog
from ..world import audio
from ..world.oracle import Pose
from ..world.perception import bbox_3d, perceive_objects
from ..world.robot_tasks import RobotTasks
from ..world.stub_world import StubWorld

log = logging.getLogger("wardzero.stub_sim")

STEP_HZ = 100.0
STATUS_PERIOD_S = 0.1
STATUS_RECORD_PERIOD_S = 0.5
OBS_PERIOD_S = 0.5
ODOM_PERIOD_S = 0.05  # 20 Hz
PERCEPTION_PERIOD_S = 0.2  # 5 Hz
OBS_WINDOW_S = 2.0
MAX_WALL_DT = 0.5  # a stalled loop does not make the sim jump further than this
REPLY_WINDOW_S = 20.0  # a robot reply goes to whoever it last heard speak, if that was this recent
ROBOT_ADDRESS_M = 2.5  # robot speech is addressed to the nearest person within this distance
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


class StubSimService:
    """The stub sim without ROS: feed it messages, call step(wall_dt), publish what it returns."""

    def __init__(self, loaded: LoadedScenario, runlog: RunLog, time_scale: float = 1.0,
                 lockstep: bool = False, wait_for: Iterable[str] = WAIT_TOKENS, seed: Optional[int] = None,
                 wall: Callable[[], float] = time.monotonic):
        if not (time_scale > 0 and math.isfinite(time_scale)):
            raise ValueError("time_scale must be a positive number")
        self.loaded = loaded
        self.runlog = runlog
        self.time_scale = float(time_scale)
        self.lockstep = lockstep
        self.world = StubWorld(loaded, loaded.scenario.seed if seed is None else seed)
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

    def _emit(self, events: Iterable[AudioEvent]) -> None:
        for ev in events:
            heard: dict[str, Heard] = {}
            for pid, p in self.world.people.items():
                if p.posture == "gone" or p.kind == "static":
                    continue  # patients lying in beds hear nothing
                h = audio.route(ev, Pose(p.x, p.y, p.yaw), pid, p.inst.hearing_loss_db, self.world.rng)
                if h is not None:
                    heard[pid] = h
            self._recent.append((self.t, ev, heard))
            self._audio_out.append(ev)
            self.runlog.write("audio", ev.model_dump(mode="json", by_alias=True))
            rh = self.world.robot_heard(ev) if self.robot else None
            if rh is not None:
                self._heard_out.append(rh)
                if ev.source_kind == "person" and ev.text:
                    self._last_robot_listener = (self.t, ev.source_id)
            self._obs_now = True

    def _spoken_before(self, person: str, t: float, text: str) -> bool:
        """A talk intent and a Say for the same utterance arrive separately: speak it once."""
        key = (t, text)
        if self._last_spoken.get(person) == key:
            return True
        self._last_spoken[person] = key
        return False

    def on_intent(self, intent: Intent) -> None:
        if intent.person not in self.world.people:
            log.warning("intent for unknown person %s ignored", intent.person)
            return
        if intent.kind == "talk" and self._spoken_before(intent.person, intent.t, intent.text or ""):
            return
        try:
            events = self.world.apply_intent(intent.model_copy(update={"t": self.t}))
        except ValueError as exc:
            log.warning("intent %s for %s ignored: %s", intent.kind, intent.person, exc)
            return
        self._emit(events)

    def on_say(self, say: Say) -> None:
        if say.person not in self.world.people or self.world.people[say.person].posture == "gone":
            log.warning("speech from unknown or absent person %s ignored", say.person)
            return
        if self._spoken_before(say.person, say.t, say.text):
            return
        self._emit([self.world.say(say.person, say.text, say.addressee, self.t)])

    def on_robot_say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self.robot:
            log.warning("robot speech ignored: this scenario has no robot")
            return
        ev = self.world.robot_say(text, self.t)
        r = self.world.robot_pose
        # The reply goes to the person the robot just heard, not to whoever stands nearest.
        last, self._last_robot_listener = self._last_robot_listener, None
        if last is not None and self.t - last[0] <= REPLY_WINDOW_S:
            p = self.world.people.get(last[1])
            if p is not None and p.posture != "gone":
                ev = ev.model_copy(update={"addressee": last[1]})
                self._emit([ev])
                return
        near = [(math.hypot(p.x - r.x, p.y - r.y), pid) for pid, p in self.world.people.items()
                if p.posture != "gone"]
        near = [x for x in near if x[0] <= ROBOT_ADDRESS_M]
        if near:
            ev = ev.model_copy(update={"addressee": min(near)[1]})
        self._emit([ev])

    # ---- the robot's body and tasks --------------------------------------------------------
    @property
    def robot_brief(self) -> Optional[dict]:
        """The task brief to publish latched on task/brief, from the scenario's robot_brief."""
        brief = getattr(self.loaded.scenario, "robot_brief", None)
        if brief is None:
            return None
        return brief.model_dump(mode="json") if hasattr(brief, "model_dump") else dict(brief)

    def on_cmd_vel(self, cmd: "tuple[float, float]") -> None:
        if self.robot:
            self.world.robot.set_cmd_vel(*cmd)

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

    def step(self, wall_dt: float) -> SimOutputs:
        wall_dt = min(max(0.0, wall_dt), MAX_WALL_DT)
        held = (self.paused or bool(self.waiting) or (self.lockstep and self._pending_now() > 0)
                or self.t >= self.duration_s)
        t0 = self.t
        if not held and not self._stop:
            self.t = min(self.duration_s, self.t + self.time_scale * wall_dt)
            self.world.step(self.t)
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


# ---- ROS wrapper ---------------------------------------------------------------------------
def _parse(argv):
    from .rosutil import base_args
    p = base_args("Kinematic stub sim on ROS 2 (stands in for the Isaac sim host).")
    p.add_argument("--scenario", required=True)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--lockstep", action="store_true")
    p.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_S,
                   help="wall seconds to wait for the other nodes before giving up")
    p.add_argument("--wait-for", default=",".join(WAIT_TOKENS),
                   help="comma list of nodes to wait for before the clock starts: agents, director, or none")
    return p.parse_args(argv)


async def _run(args) -> int:
    import json

    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Int32, String

    from . import topics
    from .rosutil import QOS, QOS_LATCHED, RosBridge, install_stop_signals, wait_or_sleep

    try:
        loaded = load_scenario(args.scenario)
    except ConfigError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    wait_for = [] if args.wait_for.strip() in ("", "none") else [w.strip() for w in args.wait_for.split(",")]
    unknown = [w for w in wait_for if w not in WAIT_TOKENS]
    if unknown:
        print(f"--wait-for: unknown {', '.join(unknown)} (use agents, director or none)", file=sys.stderr)
        return 2
    lockstep = args.lockstep or loaded.scenario.lockstep
    svc = StubSimService(loaded, RunLog(args.run_dir), args.time_scale, lockstep, wait_for)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("wardzero_stub_sim", loop, use_sim_time=False)
    clock_pub = ros.publisher("/clock", Clock, QOS)

    def subscribe_person(pid: str) -> None:
        ros.subscribe_json(Intent, topics.intent(pid), svc.on_intent)
        ros.subscribe_json(Say, topics.say(pid), svc.on_say)

    for pid in svc.world.people:
        subscribe_person(pid)
    rcfg = loaded.scenario.robot
    rt = topics.robot_topics(rcfg.ns)
    say_topics = {rt["say"]}
    heard_topics = [rt["heard"]]
    if topics.uses_g1_alias(rcfg.ns, rcfg.profile):  # the reception bot and old tests keep working
        say_topics.add(topics.G1_SAY)
        heard_topics.append(topics.G1_HEARD)
    for name in sorted(say_topics):
        ros.subscribe(String, name, svc.on_robot_say, QOS, convert=lambda m: m.data)
    ros.subscribe(Twist, rt["cmd_vel"], svc.on_cmd_vel, QOS,
                  convert=lambda m: (m.linear.x, m.angular.z))

    def on_goal(m) -> None:
        q = m.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        svc.on_goal(m.pose.position.x, m.pose.position.y, yaw)

    ros.subscribe(PoseStamped, rt["goal_pose"], on_goal, QOS)

    def parse_action(m):
        try:
            d = json.loads(m.data)
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}

    ros.subscribe(String, rt["task_action"], svc.on_task_action, QOS, convert=parse_action)
    odom_pub = ros.publisher(rt["odom"], Odometry, QOS)
    for key in ("bbox_3d", "objects", "task_result"):
        ros.publisher(rt[key], String, QOS)
    if svc.robot_brief is not None:
        ros.publish_text(rt["task_brief"], json.dumps(svc.robot_brief), QOS_LATCHED)
    ros.subscribe_json(WorldEvent, topics.WORLD_EVENT, svc.on_world_event)
    ros.subscribe_json(Control, topics.CONTROL, svc.control)

    def on_pending(n: int) -> None:
        svc.ready("agents")
        svc.set_pending(n)

    ros.subscribe(Int32, topics.PENDING, on_pending, QOS, convert=lambda m: int(m.data))
    ros.subscribe(String, topics.TIMELINE, lambda _m: svc.ready("director"), QOS)
    ros.start()
    log.info("stub sim up: %s, time scale %.1f%s, waiting for %s", loaded.scenario.id, args.time_scale,
             ", lockstep" if lockstep else "", ", ".join(sorted(svc.waiting)) or "nothing")

    period = 1.0 / STEP_HZ
    last = time.monotonic()
    started = last
    last_wait_log = last
    code = 0
    try:
        while not stop.is_set():
            now = time.monotonic()
            out = svc.step(now - last)
            last = now
            msg = Clock()
            msg.clock.sec = int(out.clock)
            msg.clock.nanosec = int(round((out.clock - int(out.clock)) * 1e9)) % 1_000_000_000
            clock_pub.publish(msg)
            if out.roster is not None:
                for p in out.roster.people:
                    subscribe_person(p.id)
                ros.publish_json(topics.ROSTER, out.roster, QOS_LATCHED)
            for st in out.statuses:
                ros.publish_json(topics.status(st.person), st)
            for ev in out.audio:
                ros.publish_json(topics.AUDIO_EVENTS, ev)
            for obs in out.observations:
                ros.publish_json(topics.observation(obs.person), obs)
            for h in out.robot_heard:
                for name in heard_topics:
                    ros.publish_json(name, h)
            if out.odom is not None:
                o = Odometry()
                o.header.frame_id, o.child_frame_id = "map", "base_link"
                o.header.stamp = msg.clock
                o.pose.pose.position.x, o.pose.pose.position.y = out.odom["x"], out.odom["y"]
                o.pose.pose.orientation.z = math.sin(out.odom["yaw"] / 2)
                o.pose.pose.orientation.w = math.cos(out.odom["yaw"] / 2)
                o.twist.twist.linear.x, o.twist.twist.angular.z = out.odom["vx"], out.odom["wz"]
                odom_pub.publish(o)
            if out.detections is not None:
                ros.publish_text(rt["bbox_3d"], json.dumps(out.detections))
                ros.publish_text(rt["objects"], json.dumps(out.objects))
            for r in out.task_results:
                ros.publish_text(rt["task_result"], json.dumps(r))
            for n in out.nudges:
                ros.publish_json(topics.nudge(n.person), n)
            if out.stop:
                log.info("scenario over at t=%.1f s; stopping the run", out.clock)
                ros.publish_json(topics.CONTROL, Control(action="stop"))
                await asyncio.sleep(0.5)  # let the stop reach the other nodes
                break
            if svc.waiting:
                if now - started > args.ready_timeout:
                    log.error("gave up waiting for %s after %.0f s", ", ".join(sorted(svc.waiting)),
                              args.ready_timeout)
                    code = 3
                    break
                if now - last_wait_log > 10.0:
                    log.info("waiting for %s", ", ".join(sorted(svc.waiting)))
                    last_wait_log = now
            await wait_or_sleep(stop, max(0.0, period - (time.monotonic() - now)))
    finally:
        ros.close()
    return code


def main(argv=None) -> int:
    from .rosutil import setup_logging
    setup_logging()
    return asyncio.run(_run(_parse(argv)))


if __name__ == "__main__":
    sys.exit(main())
