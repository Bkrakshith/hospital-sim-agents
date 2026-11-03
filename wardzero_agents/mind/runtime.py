"""Runs one person's mind: gates ticks into decisions, keeps one decision in flight, owns the checkpointer."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, Protocol

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config.loader import LoadedScenario
from ..config.models import PersonInstance
from ..contracts import Heard, Intent, Nudge, Observation, Say, Status
from ..kb.index import KnowledgeBase
from ..llm.gateway import Gateway
from ..memory.longterm import LongTermMemory
from .graph import build_graph
from .triggers import TriggerGate

log = logging.getLogger(__name__)

SHORTTERM_DB = "shortterm.sqlite"


def _held_while_pending(h: Heard) -> bool:
    """Speech that must still reach the next decision: anything the robot says, and speech to me."""
    return h.source_kind == "robot" or (h.source_kind == "person" and h.addressed_to_me)


class Effects(Protocol):
    def publish_intent(self, intent: Intent) -> None: ...
    def publish_say(self, say: Say) -> None: ...


class RunLogLike(Protocol):
    """What the mind needs from the run log (the RunLog class arrives in Task 9)."""

    def write(self, stream: str, record: dict) -> None: ...


@dataclass
class MindDeps:
    inst: PersonInstance
    loaded: LoadedScenario
    gateway: Gateway
    kb: Optional[KnowledgeBase]
    ltm: Optional[LongTermMemory]
    run_id: str
    run_index: int
    effects: Effects
    runlog: RunLogLike
    # Short-term state: a shared checkpointer if given, else <run_dir>/shortterm.sqlite, else in memory.
    run_dir: Optional[Path] = None
    checkpointer: Any = None
    robot_id: str = "g1"
    # The sim clock; intents and speech are stamped with it when sent. None: the decision's start.
    clock: Optional[Callable[[], float]] = None


@asynccontextmanager
async def open_shortterm(run_dir: "str | Path") -> AsyncIterator[AsyncSqliteSaver]:
    """One async SQLite checkpointer on <run_dir>/shortterm.sqlite, to share between minds."""
    async with AsyncSqliteSaver.from_conn_string(str(Path(run_dir) / SHORTTERM_DB)) as saver:
        yield saver


class Mind:
    def __init__(self, deps: MindDeps, gate: Optional[TriggerGate] = None):
        self.deps = deps
        self.gate = gate or TriggerGate()
        self.person = deps.inst.person_id
        self._config = {"configurable": {"thread_id": self.person}}
        self._task: Optional[asyncio.Task] = None
        self._held: list[Nudge] = []
        self._held_heard: list[Heard] = []
        self._latest: Optional[tuple[float, Observation, Status]] = None  # last tick seen while pending
        self._last_obs: Optional[Observation] = None
        self._graph = None
        self._graph_lock = asyncio.Lock()
        self._owned_conn: Optional[aiosqlite.Connection] = None

    @property
    def pending(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _ensure_graph(self):
        async with self._graph_lock:
            if self._graph is None:
                saver = self.deps.checkpointer
                if saver is None and self.deps.run_dir is not None:
                    self._owned_conn = aiosqlite.connect(str(Path(self.deps.run_dir) / SHORTTERM_DB))
                    saver = AsyncSqliteSaver(self._owned_conn)
                if saver is None:
                    log.warning("%s: no run_dir or checkpointer; short-term memory is kept in memory only",
                                self.person)
                    saver = MemorySaver()
                self._graph = build_graph(self.deps, saver)
            return self._graph

    async def on_tick(self, t: float, obs: Observation, status: Status, nudges: list[Nudge]) -> None:
        """Start a decision if something warrants one. Never waits for the decision itself."""
        if self.pending:
            # One decision in flight: hold nudges and speech for the next one.
            self._held.extend(nudges)
            self._held_heard.extend(h for h in obs.heard if _held_while_pending(h))
            rest = [h for h in obs.heard if not _held_while_pending(h)]
            self._latest = (t, obs.model_copy(update={"heard": rest}), status)
            return
        self._consider(t, obs, status, nudges)

    def _consider(self, t: float, obs: Observation, status: Status, nudges: list[Nudge]) -> None:
        nudges = self._held + list(nudges)
        if self._held_heard:
            obs = obs.model_copy(update={"heard": self._held_heard + list(obs.heard)})
        triggers = self.gate.check(t, obs, status, nudges, self._last_obs)
        if not triggers:
            return  # held speech stays held until the gate lets a decision through
        self._held, self._held_heard = [], []
        self.gate.mark(t)
        self._last_obs = obs
        frame = {
            "obs": obs.model_dump(mode="json", by_alias=True),
            "status": status.model_dump(mode="json", by_alias=True),
            "nudges": [n.model_dump(mode="json", by_alias=True) for n in nudges],
            "clock_min": max(0.0, t - self.deps.inst.arrive_at_s) / 60.0,
        }
        self._task = asyncio.create_task(self._run(t, triggers, frame))

    async def _run(self, t: float, triggers: list[str], frame: dict) -> None:
        try:
            graph = await self._ensure_graph()
            await graph.ainvoke({"person": self.person, "t": t, "triggers": triggers, "frame": frame},
                                self._config, durability="exit")
        except Exception as exc:  # a broken decision must not take the person down
            log.error("%s: decision step failed: %s", self.person, type(exc).__name__)
            log.debug("%s: decision step traceback", self.person, exc_info=True)
            try:
                self.deps.runlog.write("decisions", {
                    "t": t, "person": self.person, "triggers": triggers, "situation": None,
                    "result": None, "error": "mind_error", "exception": type(exc).__name__,
                    "reason": None, "latency_s": None, "model": None, "prompt_sha256": None})
            except Exception as log_exc:
                log.error("%s: run log write failed: %s", self.person, type(log_exc).__name__)
        # Speech or nudges that arrived meanwhile get their decision now, not on some later tick.
        latest, self._latest = self._latest, None
        if latest is not None and (self._held or self._held_heard):
            self._consider(*latest, [])

    async def drain(self) -> None:
        """Wait until no decision is in flight, including follow-ups started by held speech or nudges."""
        while self._task is not None and not self._task.done():
            await asyncio.shield(self._task)

    async def state(self) -> dict:
        """The person's current short-term state (mood, patience, intent, recent events)."""
        graph = await self._ensure_graph()
        snap = await graph.aget_state(self._config)
        return dict(snap.values)

    async def aclose(self) -> None:
        await self.drain()
        if self._owned_conn is not None:
            await self._owned_conn.close()
            self._owned_conn = None
