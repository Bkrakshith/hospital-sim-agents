"""Runs one person's mind: gates ticks into decisions, keeps one decision in flight, owns the checkpointer."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Protocol

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config.loader import LoadedScenario
from ..config.models import PersonInstance
from ..contracts import Intent, Nudge, Observation, Say, Status
from ..kb.index import KnowledgeBase
from ..llm.gateway import Gateway
from ..memory.longterm import LongTermMemory
from .graph import build_graph
from .triggers import TriggerGate

log = logging.getLogger(__name__)

SHORTTERM_DB = "shortterm.sqlite"


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
                    saver = MemorySaver()
                self._graph = build_graph(self.deps, saver)
            return self._graph

    async def on_tick(self, t: float, obs: Observation, status: Status, nudges: list[Nudge]) -> None:
        """Start a decision if something warrants one. Never waits for the decision itself."""
        if self.pending:
            self._held.extend(nudges)  # kept for the next decision
            return
        nudges = self._held + list(nudges)
        triggers = self.gate.check(t, obs, status, nudges, self._last_obs)
        if not triggers:
            return
        self._held = []
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
            self.deps.runlog.write("decisions", {
                "t": t, "person": self.person, "triggers": triggers, "situation": None, "result": None,
                "error": "mind_error", "reason": None, "latency_s": None, "model": None,
                "prompt_sha256": None})

    async def drain(self) -> None:
        """Wait for the decision in flight, if any."""
        if self._task is not None:
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
