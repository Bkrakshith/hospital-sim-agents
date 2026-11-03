"""Agent service node (wardzero_agents): one LangGraph mind and one reflex engine per person.

Start-up, in order: log redaction, scenario, key check (skipped only with
WARDZERO_TEST_FAKE_LLM=1), run lock, long-term memory (restore the scenario's snapshot, seed every
person, begin the run, save run-<id>-start), knowledge base (rebuilt when missing or stale), then
one Mind per person. Every 0.1 s of sim time each person's reflexes run, reflex actions go out as
overlay intents, and the mind gets the latest status and observation plus any nudges. On stop
(SIGINT, SIGTERM or a "stop" on /scenario/control) the minds are drained, run-<id>-end is saved,
the run meta is written and the lock released.

    python -m wardzero_agents.ros.agent_node --run-dir DIR --scenario FILE [--embedder hash]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from ..config.loader import LoadedScenario
from ..config.models import PersonInstance
from ..contracts import Control, Heard, Intent, Nudge, Observation, ReflexOverride, Roster, Say, Status
from ..llm.gateway import LLMRequest, LLMResult
from ..reflex import ReflexEngine
from ..runlog import RunLog
from ..world.stub_world import spawned_instance

log = logging.getLogger("wardzero.agents")

TICK_S = 0.1
SPATIAL_SOURCES = ("robot", "person", "cough")  # a cough's source is the person who coughed
# The sim finishes these at once, but the person carries on doing them until told otherwise.
DURABLE = ("sit_idle", "use_phone", "wait_standing")
DRAIN_TIMEOUT_S = 60.0
ERROR_LOG_EVERY_S = 30.0  # per person: a failing tick is reported at most this often
PENDING_HEARTBEAT_S = 1.0
_SNAP_BAD = re.compile(r"[^A-Za-z0-9_.-]")


def snapshot_name(run_id: str, which: str) -> str:
    """run-<id>-<which>, cleaned to the snapshot name rules and at most 64 characters."""
    rid = _SNAP_BAD.sub("_", run_id)
    room = 64 - len("run-") - len(which) - 1
    return f"run-{rid[:room]}-{which}"


class Publisher(Protocol):
    def publish_intent(self, intent: Intent) -> None: ...
    def publish_say(self, say: Say) -> None: ...


def _heard_key(h: Heard) -> str:
    return json.dumps(h.model_dump(mode="json", by_alias=True), sort_keys=True)


class _Person:
    def __init__(self, inst: PersonInstance, mind: Any, reflex: Any):
        self.inst = inst
        self.mind = mind
        self.reflex = reflex
        self.status: Optional[Status] = None
        self.obs: Optional[Observation] = None
        self.obs_new = False
        self.prev_heard: Counter = Counter()
        self.nudges: list[Nudge] = []
        self.bouts: list[ReflexOverride] = []
        self.last_base: Optional[tuple[str, tuple]] = None  # (intent id, (kind, target, text))


class AgentService:
    """The agent node's state and logic. It is also the minds' Effects."""

    def __init__(self, loaded: LoadedScenario, runlog: RunLog, publisher: Publisher,
                 mind_factory: Callable[[PersonInstance, "AgentService"], Any],
                 reflex_factory: Optional[Callable[[PersonInstance, int], Any]] = None,
                 now: Callable[[], float] = lambda: 0.0, wall: Callable[[], float] = time.monotonic):
        self.loaded = loaded
        self.runlog = runlog
        self.out = publisher
        self.mind_factory = mind_factory
        self.reflex_factory = reflex_factory or ReflexEngine
        self.now = now
        self.wall = wall
        self._tick_errors: dict[str, list] = {}  # person -> [wall time last reported, suppressed since]
        self.cost_usd = 0.0
        self.people: dict[str, _Person] = {}
        for inst in loaded.instances:
            self._add(inst)

    @property
    def minds(self) -> dict[str, Any]:
        return {pid: p.mind for pid, p in self.people.items()}

    def _add(self, inst: PersonInstance) -> None:
        self.people[inst.person_id] = _Person(inst, self.mind_factory(inst, self),
                                              self.reflex_factory(inst, inst.seed))

    # ---- inputs --------------------------------------------------------------------------
    def on_roster(self, roster: Roster) -> list[str]:
        """Create minds for people we have not met. Returns their ids."""
        new = []
        for entry in roster.people:
            if entry.id in self.people:
                continue
            try:
                inst = spawned_instance(self.loaded, entry.id, entry.persona)
            except ValueError as exc:
                log.warning("roster: cannot create a mind for %s: %s", entry.id, exc)
                continue
            self.loaded.instances.append(inst)  # so later decisions may name the newcomer
            self._add(inst)
            new.append(entry.id)
            log.info("new person %s (%s)", entry.id, entry.persona)
        return new

    def _get(self, pid: str) -> Optional[_Person]:
        p = self.people.get(pid)
        if p is None:
            log.debug("message for unknown person %s ignored", pid)
        return p

    def on_status(self, st: Status) -> None:
        p = self._get(st.person)
        if p is not None:
            p.status = st

    def on_observation(self, obs: Observation) -> None:
        p = self._get(obs.person)
        if p is not None:
            p.obs, p.obs_new = obs, True

    def on_nudge(self, n: Nudge) -> None:
        p = self._get(n.person)
        if p is not None:
            p.nudges.append(n)

    def on_reflex(self, r: ReflexOverride) -> None:
        p = self._get(r.person)
        if p is not None:
            p.bouts.append(r)

    # ---- tick ----------------------------------------------------------------------------
    def _new_heard(self, p: _Person) -> list[Heard]:
        """Observations overlap (each holds the last 2 s of audio): pass on each sound once."""
        if not p.obs_new or p.obs is None:
            return []
        p.obs_new = False
        cur = Counter(_heard_key(h) for h in p.obs.heard)
        budget = cur - p.prev_heard
        p.prev_heard = cur
        out = []
        for h in p.obs.heard:
            k = _heard_key(h)
            if budget[k] > 0:
                budget[k] -= 1
                out.append(h)
        # the newest copies are the new ones; order within identical copies does not matter
        return out

    async def tick(self, t: float) -> None:
        for pid in sorted(self.people):
            try:
                await self._tick_person(pid, self.people[pid], t)
            except Exception as exc:  # one person failing must not starve the others
                self._tick_error(pid, exc)

    def _tick_error(self, pid: str, exc: Exception) -> None:
        """Log a failed tick at most once per ERROR_LOG_EVERY_S per person, counting the rest."""
        now = self.wall()
        seen = self._tick_errors.get(pid)
        if seen is not None and now - seen[0] < ERROR_LOG_EVERY_S:
            seen[1] += 1
            return
        more = f" ({seen[1]} more since the last report)" if seen is not None and seen[1] else ""
        log.error("%s: tick failed: %s%s", pid, type(exc).__name__, more)
        log.debug("%s: tick traceback", pid, exc_info=True)
        self._tick_errors[pid] = [now, 0]

    async def _tick_person(self, pid: str, p: _Person, t: float) -> None:
        st = p.status
        if st is None or st.posture == "gone":
            p.nudges, p.bouts = [], []
            return
        heard = self._new_heard(p)
        bouts, p.bouts = p.bouts, []
        for b in bouts:
            p.reflex.start_bout(t, b.duration_s, b.severity)
        idle = st.state != "running"
        spatial = [h for h in heard if h.source_kind in SPATIAL_SOURCES]
        for a in p.reflex.tick(t, st, spatial, idle):
            target = a.target if a.kind == "look_at" else None
            if a.kind == "look_at" and target is None:
                continue
            self.publish_intent(Intent(id=uuid.uuid4().hex, person=pid, t=t, kind=a.kind,
                                       target=target, overlay=True))
        base = p.obs if p.obs is not None else Observation(person=pid, t=t)
        obs = base.model_copy(update={"heard": heard})
        nudges, p.nudges = p.nudges, []
        await p.mind.on_tick(t, obs, st, nudges)

    # ---- effects -------------------------------------------------------------------------
    def publish_intent(self, intent: Intent) -> None:
        p = self.people.get(intent.person)
        if not intent.overlay and p is not None:
            key = (intent.kind, intent.target, intent.text)
            st = p.status
            ongoing = st is not None and (
                st.state == "running" or (st.state == "done" and st.intent_kind in DURABLE))
            if (p.last_base is not None and p.last_base[1] == key and ongoing
                    and st.intent_id == p.last_base[0]):
                # Re-sending the intent that is still under way would restart it and, for the
                # durable ones, end in another "intent done" and another decision, forever.
                log.debug("%s: %s %s is already running; not sent again", intent.person, intent.kind,
                          intent.target)
                return
            p.last_base = (intent.id, key)
        self.out.publish_intent(intent)
        self.runlog.write("intents", intent.model_dump(mode="json", by_alias=True))

    def publish_say(self, say: Say) -> None:
        self.out.publish_say(say)
        self.runlog.write("says", say.model_dump(mode="json", by_alias=True))

    def on_usage(self, req: LLMRequest, res: LLMResult) -> None:
        self.cost_usd += res.cost_usd or 0.0
        self.runlog.write("tokens", {
            "t": self.now(), "person": req.person, "role": req.role, "model": res.model, "ok": res.ok,
            "error": res.error, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cache_read_tokens": res.cache_read_tokens, "cache_write_tokens": res.cache_write_tokens,
            "cost_usd": res.cost_usd, "latency_s": res.latency_s})

    def pending_count(self) -> int:
        return sum(1 for p in self.people.values() if p.mind.pending)

    async def aclose(self) -> None:
        for p in self.people.values():
            try:
                await p.mind.aclose()
            except Exception as exc:
                log.error("%s: closing the mind failed: %s", p.inst.person_id, type(exc).__name__)


# ---- ROS wrapper ---------------------------------------------------------------------------
def _parse(argv):
    from .rosutil import base_args
    p = base_args("Persona agent service: one mind per simulated person.")
    p.add_argument("--scenario", required=True)
    p.add_argument("--embedder", default=os.environ.get("WARDZERO_EMBEDDER", "hash"),
                   help="hash (default) or fastembed; must match the memory db and kb index")
    return p.parse_args(argv)


def _hg1_root() -> Path:
    env = os.environ.get("HG1_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[3]


def _key_check(loaded: LoadedScenario) -> Optional[int]:
    """None when the key works (or in test mode), else the exit code after printing the fix."""
    from .. import keys
    if os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1":
        print("TEST MODE: no AI provider", flush=True)
        return None
    provider = loaded.scenario.provider
    try:
        client = keys.make_client(provider)
        asyncio.run(keys.check(provider, client, loaded.scenario.models.decide))
    except (keys.KeyMissing, keys.KeyRejected, keys.ModelUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # network, provider outage: never print the exception text
        print(f"The {provider} key check failed ({type(exc).__name__}). Check the network and try "
              "again.", file=sys.stderr)
        return 2
    return None


def _prepare_memory(loaded: LoadedScenario, run_id: str, embedder):
    from ..memory import data_dir, memory_path
    from ..memory.longterm import LongTermMemory
    from ..memory.snapshots import restore_snapshot, save_snapshot
    mem = memory_path(data_dir())
    if loaded.scenario.memory_snapshot:
        restore_snapshot(mem, loaded.scenario.memory_snapshot)
        log.info("memory restored from snapshot %s", loaded.scenario.memory_snapshot)
    ltm = LongTermMemory(mem, embedder)
    for inst in loaded.instances:
        ltm.ensure_seed(inst.memory_key, loaded.personas[inst.persona_id].seed_memories)
    run_index = ltm.begin_run(run_id)
    snap_in = snapshot_name(run_id, "start")
    save_snapshot(mem, snap_in)
    return ltm, mem, run_index, snap_in


def _prepare_kb(loaded: LoadedScenario, embedder):
    from ..kb.index import KnowledgeBase, build_default_kb, default_kb_sha256
    from ..memory import data_dir
    kb_dir = data_dir() / "kb"
    hg1 = _hg1_root()
    want = default_kb_sha256(hg1, loaded)
    kb = None
    try:
        kb = KnowledgeBase.load(kb_dir, embedder)
    except FileNotFoundError:
        log.info("no knowledge base index yet; building it")
    except (ValueError, KeyError, OSError) as exc:
        log.warning("knowledge base index unusable (%s); rebuilding it", type(exc).__name__)
    if kb is None or kb.sources_sha256 != want:
        kb = build_default_kb(hg1, loaded, embedder)
        kb.save(kb_dir)
        log.info("knowledge base built: %d chunks", len(kb.chunks))
    return kb


def _write_meta(runlog: RunLog, run_dir: Path, update: dict) -> None:
    meta: dict = {}
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    meta.update(update)
    runlog.write_meta(meta)


async def _serve(args, loaded: LoadedScenario, run_dir: Path, run_id: str, test_mode: bool) -> int:
    from ..kb.embed import get_embedder
    from ..llm.gateway import Budget, make_gateway
    from ..memory.snapshots import save_snapshot
    from ..mind.runtime import Mind, MindDeps, open_shortterm
    from . import topics
    from .rosutil import QOS_LATCHED, RosBridge, install_stop_signals, wait_or_sleep
    from std_msgs.msg import Int32

    embedder = get_embedder(args.embedder)
    ltm, mem, run_index, snap_in = await asyncio.to_thread(_prepare_memory, loaded, run_id, embedder)
    kb = await asyncio.to_thread(_prepare_kb, loaded, embedder)
    runlog = RunLog(run_dir)
    runlog.copy_configs(loaded)
    models = loaded.scenario.models
    started = datetime.now(timezone.utc).isoformat()
    base_meta = {
        "run_id": run_id, "scenario_id": loaded.scenario.id, "scenario_path": str(loaded.scenario_path),
        "seed": loaded.scenario.seed, "provider": loaded.scenario.provider,
        "models": models.model_dump(), "test_mode": test_mode, "embedder": embedder.name,
        "run_index": run_index, "memory_snapshot_restored": loaded.scenario.memory_snapshot,
        "snapshot_in": snap_in, "snapshot_out": None, "budget_usd": loaded.scenario.budget_usd,
        "cost_usd": 0.0, "started_at": started, "kb_sources_sha256": kb.sources_sha256,
    }
    _write_meta(runlog, run_dir, {"mode": "live", **base_meta} if not (run_dir / "meta.json").exists()
                else base_meta)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("wardzero_agents", loop)

    class _Out:
        def publish_intent(self, i: Intent) -> None:
            ros.publish_json(topics.intent(i.person), i)

        def publish_say(self, s: Say) -> None:
            ros.publish_json(topics.say(s.person), s)

    holder: dict[str, AgentService] = {}
    budget = Budget(loaded.scenario.budget_usd)
    gateway = make_gateway(loaded.scenario.provider, models, budget,
                           lambda req, res: holder["svc"].on_usage(req, res))

    async with open_shortterm(run_dir) as saver:
        def mind_factory(inst: PersonInstance, effects: AgentService) -> Mind:
            return Mind(MindDeps(inst=inst, loaded=loaded, gateway=gateway, kb=kb, ltm=ltm, run_id=run_id,
                                 run_index=run_index, effects=effects, runlog=runlog, run_dir=run_dir,
                                 checkpointer=saver))

        svc = AgentService(loaded, runlog, _Out(), mind_factory, now=ros.now)
        holder["svc"] = svc

        def subscribe_person(pid: str) -> None:
            ros.subscribe_json(Status, topics.status(pid), svc.on_status)
            ros.subscribe_json(Observation, topics.observation(pid), svc.on_observation)
            ros.subscribe_json(Nudge, topics.nudge(pid), svc.on_nudge)
            ros.subscribe_json(ReflexOverride, topics.reflex(pid), svc.on_reflex)
            ros.publisher(topics.intent(pid))
            ros.publisher(topics.say(pid))

        def on_roster(r: Roster) -> None:
            for pid in svc.on_roster(r):
                subscribe_person(pid)

        def on_control(c: Control) -> None:
            if c.action == "stop":
                log.info("stop received")
                stop.set()

        for pid in svc.people:
            subscribe_person(pid)
        ros.subscribe_json(Roster, topics.ROSTER, on_roster, QOS_LATCHED)
        ros.subscribe_json(Control, topics.CONTROL, on_control)
        pending_pub = ros.publisher(topics.PENDING, Int32)
        ros.start()
        log.info("agent service up: %d people, run %s (index %d)", len(svc.people), run_id, run_index)

        last_tick = float("-inf")
        last_pending: Optional[int] = None
        last_beat = 0.0
        try:
            while not stop.is_set():
                t = ros.now()
                if t - last_tick >= TICK_S or t < last_tick:
                    last_tick = t
                    try:
                        await svc.tick(t)
                    except Exception as exc:  # one bad tick must not end the run
                        log.error("tick failed: %s", type(exc).__name__)
                        log.debug("tick traceback", exc_info=True)
                n = svc.pending_count()
                wall = loop.time()
                if n != last_pending or wall - last_beat >= PENDING_HEARTBEAT_S:
                    msg = Int32()
                    msg.data = n
                    pending_pub.publish(msg)
                    last_pending, last_beat = n, wall
                await wait_or_sleep(stop, 0.01)
        finally:
            log.info("stopping: draining %d minds", len(svc.people))
            try:
                await asyncio.wait_for(svc.aclose(), DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.warning("some decisions were still running after %.0f s; not waiting", DRAIN_TIMEOUT_S)
            snap_out = snapshot_name(run_id, "end")
            try:
                await asyncio.to_thread(save_snapshot, mem, snap_out)
            except Exception as exc:
                log.error("saving snapshot %s failed: %s", snap_out, type(exc).__name__)
                snap_out = None
            _write_meta(runlog, run_dir, {
                "snapshot_out": snap_out, "cost_usd": round(svc.cost_usd, 6),
                "ended_at": datetime.now(timezone.utc).isoformat()})
            ros.close()
    return 0


def main(argv=None) -> int:
    from ..config.loader import ConfigError, load_scenario
    from ..memory import data_dir
    from ..memory.runlock import RunActive, RunLock
    from .rosutil import setup_logging

    setup_logging()
    args = _parse(argv)
    try:
        loaded = load_scenario(args.scenario)
    except ConfigError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    code = _key_check(loaded)
    if code is not None:
        return code
    test_mode = os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1"
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    try:
        with RunLock(data_dir(), run_id):
            return asyncio.run(_serve(args, loaded, run_dir, run_id, test_mode))
    except RunActive as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:  # bad snapshot name, embedder mismatch
        print(f"Cannot start the agents: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
