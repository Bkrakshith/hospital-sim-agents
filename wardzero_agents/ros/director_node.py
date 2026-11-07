"""Director node (wardzero_director): storyboard, cards, chaos, the control panel and the fairness audit.

Every 0.5 s of sim time it builds a WorldView from the people's statuses, lets the storyboard and
chaos fire cards, and turns each firing into nudges, reflex overrides and world events. Panel
actions run on the same asyncio loop, so they are serialised with the tick. The panel reads a
snapshot of the state that the tick (and each action) rebuilds, so state() never blocks.

Mood, last decision and last line per person, and the running cost, come from the agent node's
run log streams (decisions, says, tokens) in the shared run folder.

    python -m wardzero_agents.ros.director_node --run-dir DIR --scenario FILE [--no-panel]
        [--panel-port 8765]
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from collections import deque
from typing import Callable, Optional, Protocol

from ..config.loader import LoadedScenario
from ..contracts import Control, Nudge, ReflexOverride, Roster, Status, WorldEvent
from ..director.cards import CardExecutor, CardRejected
from ..director.chaos import Chaos
from ..director.storyboard import CardFiring, StoryboardEngine, WorldView
from ..memory.snapshots import check_user_name
from ..panel.server import CONTROL_ACTIONS, SNAPSHOT_RE, BadRequest
from ..runlog import RunLog

log = logging.getLogger("wardzero.director")

TICK_S = 0.5
TIMELINE_PERIOD_S = 1.0
AUDIT_PERIOD_S = 30.0  # wall seconds: the ROS graph does not change with sim time
AUDIT_FIRST_S = 2.0
MAX_EVENTS = 100
TAIL_STREAMS = ("decisions", "says", "tokens", "audio", "robot_tasks", "robot_pose", "objects")
SPOKE_NEAR_M = 2.5  # an old run (no heard_by on the audio rows): an unaddressed line counts for anyone this close
HEARD_NEAR_M = 4.0  # an unaddressed line counts for the people who heard it within this distance of the robot
ROBOT_ACTIONS = ("press_call", "hand_over", "take", "extinguish", "approach")  # result rows the flow counts
FIRST_STATUS_GRACE_S = 10.0  # the flow waits this long at most for the cast's first statuses before its first tick


class Outbox(Protocol):
    def publish(self, kind: str, obj) -> None: ...
    # kinds: "nudge", "reflex", "world", "control" (contract models), "event", "timeline" (dicts)


def _rec_t(rec: dict, default: float) -> float:
    """A run log record's sim time (the director's own clock when it has none)."""
    t = rec.get("t")
    return float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) else default


def _dump(m) -> dict:
    return m.model_dump(mode="json", by_alias=True)


DEFAULT_BOUNDS = [-10, 12, -3, 8]


def area_map(loaded: Optional[LoadedScenario]) -> dict:
    """The panel's static map: seats, landmarks, signs and the robot, with bounds around them."""
    if loaded is None:
        return {"bounds": list(DEFAULT_BOUNDS), "seats": [], "landmarks": [], "signs": [],
                "robot": {"x": 0.0, "y": 0.0, "yaw": 0.0}}
    a, r = loaded.area, loaded.scenario.robot
    seats = [{"id": s.id, "x": s.x, "y": s.y} for s in a.seats]
    landmarks = [{"id": k, "label": v.label, "x": v.x, "y": v.y} for k, v in a.landmarks.items()]
    signs = [{"id": s.id, "text": s.text, "x": s.x, "y": s.y} for s in loaded.signs]
    robot = {"x": r.x, "y": r.y, "yaw": r.yaw}
    xs = [p["x"] for p in seats + landmarks + signs] + [robot["x"]]
    ys = [p["y"] for p in seats + landmarks + signs] + [robot["y"]]
    bounds = [math.floor(min(xs) - 1), math.ceil(max(xs) + 1), math.floor(min(ys) - 1),
              math.ceil(max(ys) + 1)]
    return {"bounds": bounds, "seats": seats, "landmarks": landmarks, "signs": signs, "robot": robot}


def deck_list(loaded: Optional[LoadedScenario]) -> list[dict]:
    if loaded is None:
        return []
    return [{"id": c.id, "label": c.label, "target_kind": c.target_kind, "tests": c.tests}
            for c in loaded.deck.values()]


class DirectorService:
    """The director's state and logic, and the panel's DirectorAPI."""

    def __init__(self, loaded: LoadedScenario, runlog: RunLog, out: Outbox, run_id: str,
                 snapshot_fn: Optional[Callable[[str], None]] = None):
        self.loaded = loaded
        self.runlog = runlog
        self.out = out
        self.run_id = run_id
        self.snapshot_fn = snapshot_fn
        sc = loaded.scenario
        if getattr(loaded, "flow", None) is not None:  # the scenario's flow (a behaviour tree) drives the cards
            from ..flows.engine import FlowEngine
            self.engine = FlowEngine(loaded.flow, loaded.deck)
        else:
            self.engine = StoryboardEngine(loaded.storyboard, loaded.deck)
        self.cast = {c.id for c in sc.cast}
        self.robot_xy = (sc.robot.x, sc.robot.y)  # live, from robot_pose
        self.robot_said_at: dict[str, list[float]] = {}  # person -> when the robot spoke to them
        self.said_to_robot: set[str] = set()
        self.robot_did: list[tuple[float, str, Optional[str], Optional[str]]] = []  # (t, action, person, item)
        self.fires: dict[str, bool] = {}
        self._flow_started = getattr(loaded, "flow", None) is None  # a storyboard runs on the clock from the start
        self._t_first: Optional[float] = None
        self.executor = CardExecutor(loaded.deck, {i.person_id: i for i in loaded.instances},
                                     loaded.personas, sc.seed)
        self.seed = sc.seed
        self.chaos_enabled = sc.chaos.enabled
        self.chaos_rate = float(sc.chaos.rate_per_hour)
        self.chaos = Chaos(self.chaos_rate, loaded.deck, self.seed)
        self.robot = sc.robot
        self.t = 0.0
        self.paused = False
        self.stopped = False
        self.roster = Roster(t=0.0, people=[])
        self.static_ids: set[str] = set()
        self.statuses: dict[str, Status] = {}
        self.seated_since: dict[str, float] = {}
        self.minds: dict[str, dict] = {}
        self.cost_usd = 0.0
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._n_panel = 0
        self._next_timeline = 0.0
        self._map = area_map(loaded)
        self._deck = deck_list(loaded)
        self._state: dict = {}
        self._refresh()

    @property
    def robot_said_to(self) -> set[str]:
        """Everyone the robot has spoken to in the run."""
        return set(self.robot_said_at)

    # ---- inputs --------------------------------------------------------------------------
    def on_roster(self, roster: Roster) -> None:
        self.roster = roster
        self.static_ids = {p.id for p in roster.people if p.role == "patient_static"}

    def on_status(self, st: Status) -> None:
        prev = self.statuses.get(st.person)
        if st.posture == "seated":
            if prev is None or prev.posture != "seated" or st.person not in self.seated_since:
                self.seated_since[st.person] = st.t
        else:
            self.seated_since.pop(st.person, None)
        self.statuses[st.person] = st

    def on_control(self, ctl: Control) -> None:
        if ctl.action == "pause":
            self.paused = True
        elif ctl.action in ("resume", "start"):
            self.paused = False
        elif ctl.action == "stop":
            self.stopped = True
        self._refresh()

    def on_scenario_event(self, ev: dict) -> None:
        """An event another node posted on /scenario/events (the agent service's AI provider
        warnings). The director's own events are already in the list."""
        if not isinstance(ev, dict) or ev.get("source") != "agents" or not isinstance(ev.get("text"), str):
            return
        t = ev.get("t")
        t = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else self.t
        self.events.append({"t": t, "text": ev["text"][:300]})
        self._refresh()

    def ingest(self, stream: str, rec: dict) -> None:
        """One record from the run log (the agent node's streams, and the sim's robot and object streams)."""
        if stream in ("audio", "robot_tasks", "robot_pose", "objects"):
            self._ingest_world(stream, rec)
            return
        pid = rec.get("person")
        if stream == "tokens":
            c = rec.get("cost_usd")
            if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c):
                self.cost_usd += float(c)
            return
        if not isinstance(pid, str):
            return
        m = self.minds.setdefault(pid, {"mood": None, "last_decision": None, "reason": None,
                                         "last_said": None})
        if stream == "says":
            m["last_said"] = rec.get("text")
            if rec.get("addressee") in ("g1", "robot"):
                self.said_to_robot.add(pid)
        elif stream == "decisions":
            res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            if rec.get("error"):
                m["last_decision"] = f"(no decision: {rec['error']})"
            else:
                intent = res.get("intent") if isinstance(res.get("intent"), dict) else {}
                kind, target = intent.get("kind"), intent.get("target")
                m["last_decision"] = " ".join(str(x) for x in (kind, target) if x) or None
            if res.get("mood"):
                m["mood"] = res["mood"]
            m["reason"] = rec.get("reason")

    def _ingest_world(self, stream: str, rec: dict) -> None:
        """What the flow's conditions need: where the robot is, whom it spoke to, what it did, the fires."""
        if stream == "robot_pose":
            x, y = rec.get("x"), rec.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                self.robot_xy = (float(x), float(y))
        elif stream == "audio":
            if rec.get("source_kind") != "robot":
                return
            t = _rec_t(rec, self.t)
            for pid in self._spoken_to(rec):
                self.robot_said_at.setdefault(pid, []).append(t)
        elif stream == "robot_tasks":
            kind, t = rec.get("kind"), _rec_t(rec, self.t)
            if kind == "guide" and rec.get("event") in ("start", "seat"):
                self._did(t, "guide" if rec["event"] == "start" else "seat", rec.get("person"))
            elif kind == "guide" and rec.get("event") == "end":
                if rec.get("reason") == "arrived":
                    self._did(t, "guided", rec.get("person"))  # they reached the place it walked them to
                self._did(t, "guide_ended", rec.get("person"))  # the walk is over, arrived or not
            elif kind == "follow" and rec.get("event") == "start":
                self._did(t, "follow", rec.get("person"))
            elif kind == "tell" and rec.get("heard"):
                self._did(t, "tell", rec.get("about"))
            elif kind == "note":
                self._did(t, "note", rec.get("about"))
            elif kind == "result" and rec.get("ok") and rec.get("action") in ROBOT_ACTIONS:
                if rec["action"] == "press_call" and rec.get("reason") == "unanswered":
                    return  # the call reached no member of staff: nobody was alerted
                # hand_over: the receiver took the item; other actions name a person where the sim knows one
                self._did(t, rec["action"], rec.get("person") or rec.get("about"), rec.get("item"))
        elif stream == "objects":
            for o in rec.get("objects") or []:
                if isinstance(o, dict) and o.get("kind") == "fire" and isinstance(o.get("id"), str):
                    size = o.get("size")
                    self.fires[o["id"]] = (o.get("active") is False) or (isinstance(size, (int, float)) and size <= 0)

    def _did(self, t: float, action: str, person, item=None) -> None:
        self.robot_did.append((t, action, person if isinstance(person, str) and person else None,
                               item if isinstance(item, str) and item else None))

    def _spoken_to(self, rec: dict) -> set[str]:
        """The people a robot line counts as spoken to: its addressee; with no addressee, the people who heard it
        within HEARD_NEAR_M of the robot, or all who heard it for a line to the room; an old row with no heard_by:
        anyone within SPOKE_NEAR_M."""
        to = rec.get("addressee")
        to_room = rec.get("to_room") is True or to == "room"
        if isinstance(to, str) and to and not to_room:
            return {to}
        rx, ry = self.robot_xy
        x, y = rec.get("x"), rec.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            rx, ry = float(x), float(y)  # where the robot was when it spoke

        def within(pid: str, m: float) -> bool:
            st = self.statuses.get(pid)
            return st is not None and st.posture != "gone" and math.hypot(st.x - rx, st.y - ry) <= m

        heard = rec.get("heard_by")
        if isinstance(heard, list):
            heard = {p for p in heard if isinstance(p, str) and p}
            return heard if to_room else {p for p in heard if within(p, HEARD_NEAR_M)}
        return {pid for pid in self.statuses if within(pid, SPOKE_NEAR_M)}

    # ---- tick ----------------------------------------------------------------------------
    def _flow_ready(self, t: float) -> bool:
        """A flow's first tick waits for every cast member's first status (a card aimed at someone at the start would
        otherwise find nobody there), for FIRST_STATUS_GRACE_S at most."""
        if not self._flow_started:
            if self._t_first is None:
                self._t_first = t
            self._flow_started = self.cast <= set(self.statuses) or t - self._t_first >= FIRST_STATUS_GRACE_S
        return self._flow_started

    def view(self, t: float) -> WorldView:
        present = {pid for pid, st in self.statuses.items()
                   if st.posture != "gone" and pid not in self.static_ids}
        if getattr(self.loaded, "flow", None) is not None:
            from ..flows.engine import FlowView
            site = self.loaded.site
            return FlowView(t=t, statuses=dict(self.statuses), robot_xy=self.robot_xy,
                            seated_since=dict(self.seated_since), present=present,
                            robot_start=(self.robot.x, self.robot.y), cast=set(self.cast),
                            robot_said_to=set(self.robot_said_at),
                            robot_said_at={p: list(ts) for p, ts in self.robot_said_at.items()},
                            said_to_robot=set(self.said_to_robot),
                            robot_did=list(self.robot_did), fires=dict(self.fires),
                            room_at=site.room_at if site is not None else (lambda x, y: None))
        return WorldView(t=t, statuses=dict(self.statuses), robot_xy=(self.robot.x, self.robot.y),
                         seated_since=dict(self.seated_since), present=present)

    def tick(self, t: float) -> None:
        self.t = t
        if not self.stopped and self._flow_ready(t):
            view = self.view(t)
            firings = self.engine.tick(view)
            if self.chaos_enabled:
                firings += self.chaos.tick(t, view)
            for f in firings:
                self._execute(f)
            tree = getattr(self.engine, "tree", None)
            if tree is not None:  # the flow's states, for the Studio's flow view: written when they change
                snap = tree()
                key = json.dumps(snap, sort_keys=True, default=str)
                if key != getattr(self, "_flow_key", None):
                    self._flow_key = key
                    self.runlog.write("flow", {"t": t, "tree": snap})
        if t >= self._next_timeline:
            self.publish_timeline()
            self._next_timeline = t + TIMELINE_PERIOD_S
        self._refresh()

    def publish_timeline(self) -> None:
        self.out.publish("timeline", {"schema": "wz.v1", "t": self.t, "beats": self.engine.timeline()})

    def _event(self, text: str, kind: str) -> None:
        ev = {"t": self.t, "text": text}
        self.events.append(ev)
        self.out.publish("event", {"schema": "wz.v1", "kind": kind, **ev})

    def _execute(self, f: CardFiring) -> Optional[str]:
        """Run one firing. Returns the rejection text, or None when it ran."""
        rec = {"t": f.t, "id": f.id, "card": f.card, "target": f.target, "params": f.params,
               "source": f.source, "ok": True, "error": None}
        try:
            outs = self.executor.execute(f, self.view(f.t), self.roster)
        except CardRejected as exc:
            rec.update(ok=False, error=str(exc))
            self.runlog.write("cards", rec)
            self._event(f"{f.card} ({f.source}) rejected: {exc}", "card_rejected")
            log.info("card %s rejected: %s", f.card, exc)
            return str(exc)
        self.runlog.write("cards", rec)
        for o in outs:
            if isinstance(o, Nudge):
                self.out.publish("nudge", o)
                self.runlog.write("nudges", _dump(o))
            elif isinstance(o, ReflexOverride):
                self.out.publish("reflex", o)
                self.runlog.write("reflex", _dump(o))
            elif isinstance(o, WorldEvent):
                self.out.publish("world", o)
                self.runlog.write("world", _dump(o))
        who = f" at {f.target}" if f.target else ""
        self._event(f"{f.card} fired ({f.source}){who}", "card")
        log.info("card %s fired (%s)%s", f.card, f.source, who)
        return None

    def record_fairness(self, violations: list[dict], t: float) -> None:
        if not violations:
            log.info("fairness audit at t=%.1f s: no robot-side node reads the people's topics", t)
        for v in violations:
            self.runlog.write("fairness", {"t": t, "node": v["node"], "topic": v["topic"]})
            log.warning("fairness: node %s subscribes to %s, which the robot may not read",
                        v["node"], v["topic"])
            self._event(f"fairness: {v['node']} reads {v['topic']}", "fairness")

    # ---- panel state ---------------------------------------------------------------------
    def _cast(self) -> list[dict]:
        names = {p.id: p for p in self.roster.people}
        insts = {i.person_id: i for i in self.loaded.instances}
        out = []
        for pid in sorted(set(names) | set(self.statuses)):
            st = self.statuses.get(pid)
            entry = names.get(pid)
            inst = insts.get(pid)
            m = self.minds.get(pid, {})
            out.append({
                "id": pid,
                "display_name": entry.display_name if entry else (inst.display_name if inst else pid),
                "persona": entry.persona if entry else (inst.persona_id if inst else ""),
                "x": st.x if st else None, "y": st.y if st else None,
                "posture": st.posture if st else None,
                "mood": m.get("mood"), "last_decision": m.get("last_decision"),
                "reason": m.get("reason"), "last_said": m.get("last_said"),
            })
        return out

    def _refresh(self) -> None:
        self._state = {
            "t": float(self.t), "running": not self.stopped, "paused": self.paused,
            "run_id": self.run_id, "recording": True,
            "cost_usd": round(self.cost_usd, 6), "budget_usd": float(self.loaded.scenario.budget_usd),
            "chaos": {"enabled": self.chaos_enabled, "rate_per_hour": self.chaos_rate},
            "timeline": self.engine.timeline(), "cast": self._cast(), "events": list(self.events),
            "deck": self._deck, "map": self._map,
        }

    # ---- DirectorAPI (called by the panel on the loop thread; quick and non-blocking) -----
    def state(self) -> dict:
        return self._state

    def control(self, action: str) -> None:
        if action not in CONTROL_ACTIONS:
            raise BadRequest(f"unknown action '{action}'")
        self.out.publish("control", Control(action=action))
        self._event(f"operator: {action}", "control")
        self.on_control(Control(action=action))

    def fire(self, card: str, target: Optional[str], params: dict) -> dict:
        self._n_panel += 1
        f = CardFiring(f"panel-{self._n_panel}", self.t, card, target or None, dict(params), "panel")
        err = self._execute(f)
        self._refresh()
        return {"ok": False, "error": err} if err else {"ok": True, "firing_id": f.id}

    def set_chaos(self, enabled: bool, rate_per_hour: float) -> None:
        rate = float(rate_per_hour)
        if rate != self.chaos_rate or (enabled and not self.chaos_enabled):
            self.chaos = Chaos(rate, self.loaded.deck, self.seed, t0=self.t)
        self.chaos_enabled, self.chaos_rate = enabled, rate
        self._event(f"chaos {'on' if enabled else 'off'} at {rate:g} per hour", "chaos")
        self._refresh()

    def beat(self, action: str, beat_id: str) -> None:
        try:
            if action == "skip":
                self.engine.skip(beat_id)
            elif action == "rerun":
                self.engine.rerun(beat_id)
            else:
                raise BadRequest("'action' must be skip or rerun.")
        except KeyError:
            raise BadRequest(f"There is no beat '{beat_id}'.") from None
        self._event(f"operator: {action} {beat_id}", "beat")
        self._refresh()

    def snapshot(self, name: str) -> dict:
        if not SNAPSHOT_RE.match(name or ""):
            raise BadRequest("'name' must be 1 to 64 letters, digits, dot, dash or underscore.")
        try:
            check_user_name(name)
        except ValueError as exc:  # run-* and pre-restore-* are saved automatically
            raise BadRequest(str(exc)) from None
        if self.snapshot_fn is None:
            return {"ok": False, "error": "snapshots are not available here"}
        self.snapshot_fn(name)
        self._event(f"saving memory snapshot {name}", "snapshot")
        self._refresh()
        return {"ok": True, "name": name, "status": "saving"}


# ---- ROS wrapper ---------------------------------------------------------------------------
def _parse(argv):
    from .rosutil import base_args
    p = base_args("Scenario director: storyboard, cards, chaos, panel and fairness audit.")
    p.add_argument("--scenario", required=True)
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--panel-port", type=int, default=8765)
    return p.parse_args(argv)


async def _run(args) -> int:
    from pathlib import Path

    from std_msgs.msg import String

    from ..config.loader import ConfigError, load_scenario
    from ..memory import data_dir, memory_path
    from ..memory.snapshots import save_snapshot
    from ..panel.server import run_panel
    from ..runlog import RunTail
    from . import topics
    from .fairness import live_audit
    from .rosutil import QOS_LATCHED, RosBridge, install_stop_signals, wait_or_sleep

    try:
        loaded = load_scenario(args.scenario)
    except ConfigError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir).resolve()
    runlog = RunLog(run_dir)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("wardzero_director", loop)

    class _Out:
        def publish(self, kind: str, obj) -> None:
            if kind == "nudge":
                ros.publish_json(topics.nudge(obj.person), obj)
            elif kind == "reflex":
                ros.publish_json(topics.reflex(obj.person), obj)
            elif kind == "world":
                ros.publish_json(topics.WORLD_EVENT, obj)
            elif kind == "control":
                ros.publish_json(topics.CONTROL, obj)
            elif kind in ("event", "timeline"):
                ros.publish_text(topics.SCEN_EVENTS if kind == "event" else topics.TIMELINE,
                                 json.dumps(obj, default=str))

    def snapshot_fn(name: str) -> None:
        async def save():
            try:
                await asyncio.to_thread(save_snapshot, memory_path(data_dir()), name)
                svc._event(f"memory snapshot {name} saved", "snapshot")
            except Exception as exc:
                log.error("snapshot %s failed: %s", name, type(exc).__name__)
                svc._event(f"memory snapshot {name} failed ({type(exc).__name__})", "snapshot")
        loop.create_task(save())

    svc = DirectorService(loaded, runlog, _Out(), run_dir.name, snapshot_fn)

    def subscribe_person(pid: str) -> None:
        ros.subscribe_json(Status, topics.status(pid), svc.on_status)
        ros.publisher(topics.nudge(pid))
        ros.publisher(topics.reflex(pid))

    def on_roster(r: Roster) -> None:
        svc.on_roster(r)
        for p in r.people:
            subscribe_person(p.id)

    def on_control(c: Control) -> None:
        svc.on_control(c)
        if c.action == "stop":
            stop.set()

    for i in loaded.instances:
        subscribe_person(i.person_id)
    ros.subscribe_json(Roster, topics.ROSTER, on_roster, QOS_LATCHED)
    ros.subscribe_json(Control, topics.CONTROL, on_control)

    def on_event_text(text: str) -> None:
        try:
            svc.on_scenario_event(json.loads(text))
        except ValueError:
            pass

    ros.subscribe(String, topics.SCEN_EVENTS, on_event_text, convert=lambda m: m.data)
    for t in (topics.WORLD_EVENT, topics.CONTROL, topics.TIMELINE, topics.SCEN_EVENTS):
        ros.publisher(t, String)
    ros.start()

    runner = None
    if not args.no_panel:
        try:
            runner = await run_panel(svc, "127.0.0.1", args.panel_port)
            print(f"Control panel: http://127.0.0.1:{args.panel_port}/", flush=True)
        except OSError as exc:
            log.error("the panel could not start on port %d (%s); carrying on without it",
                      args.panel_port, type(exc).__name__)

    tail = RunTail(run_dir, TAIL_STREAMS)
    log.info("director up: %s, %d beats, chaos %s", loaded.scenario.id, len(loaded.storyboard.beats),
             "on" if svc.chaos_enabled else "off")
    last_tick = float("-inf")
    next_audit = loop.time() + AUDIT_FIRST_S  # once the other nodes have joined the graph
    last_timeline_wall = 0.0

    async def audit() -> None:
        try:
            svc.record_fairness(await asyncio.to_thread(live_audit, ros.node), ros.now())
        except Exception as exc:
            log.error("fairness audit failed: %s", type(exc).__name__)

    try:
        while not stop.is_set():
            t = ros.now()
            for stream, rec in tail.poll():
                svc.ingest(stream, rec)
            if t - last_tick >= TICK_S or t < last_tick:
                last_tick = t
                try:
                    svc.tick(t)
                except Exception as exc:
                    log.error("director tick failed: %s", type(exc).__name__)
                    log.debug("director tick traceback", exc_info=True)
            wall = loop.time()
            if wall - last_timeline_wall >= TIMELINE_PERIOD_S:  # also a heartbeat while time stands still
                svc.publish_timeline()
                last_timeline_wall = wall
            if wall >= next_audit:
                await audit()
                next_audit = wall + AUDIT_PERIOD_S
            await wait_or_sleep(stop, 0.01)
        await audit()  # a last look, so short runs still audit every node that joined
    finally:
        if runner is not None:
            await runner.cleanup()
        ros.close()
    return 0


def main(argv=None) -> int:
    from .rosutil import setup_logging
    setup_logging()
    return asyncio.run(_run(_parse(argv)))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # stopped with the run (Stop, or another process failed): no traceback in the log
        sys.exit(130)
