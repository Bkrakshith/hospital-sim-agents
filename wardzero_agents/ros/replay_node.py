"""Replay node (wardzero_replay): play a recorded run back.

--intents  publishes the recorded intents, says and world events on the people's topics and
           /sim/world_event as sim time passes. Run it with the stub sim (and the robot), without
           the agents or the director: the world is simulated again, so a new robot meets the same
           people doing the same things. It publishes /agents/pending = 0 as its ready signal.
--exact    republishes the recorded statuses and audio and a timeline built from the cards, and
           is its own clock (/clock at --time-scale). No stub, no agents. It serves the panel
           read-only (every action is refused) unless --no-panel.

    python -m wardzero_agents.ros.replay_node --run-dir RECORDED_RUN (--intents | --exact)
        [--time-scale F] [--scenario FILE] [--no-panel] [--panel-port 8765]
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from ..config.loader import LoadedScenario
from ..contracts import Control, Intent, Say, Status, WorldEvent
from ..panel.server import BadRequest
from ..replay import ExactScript, IntentScript, load_meta
from .director_node import area_map, deck_list

log = logging.getLogger("wardzero.replay")

READ_ONLY = "This is a replay of a recorded run; it cannot be changed."
TIMELINE_PERIOD_S = 1.0
STEP_HZ = 100.0


def timeline_from_cards(cards: list[dict]) -> list[dict]:
    rows = []
    for c in cards:
        row = {"id": c.get("id"), "card": c.get("card"), "target": c.get("target"),
               "state": "fired" if c.get("ok", True) else "rejected", "fired_t": c.get("t")}
        if not c.get("ok", True) and c.get("error"):
            row["reason"] = c["error"]
        rows.append(row)
    return rows


class ExactReplay:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.script = ExactScript(self.run_dir)
        self.end_t = self.script.last_t
        self.statuses: dict[str, Status] = {}
        self.cards: list[dict] = []

    def step(self, t: float) -> list[tuple[str, dict]]:
        items = self.script.due(t)
        for stream, rec in items:
            if stream == "status":
                try:
                    st = Status.model_validate(rec)
                except ValueError:
                    continue
                self.statuses[st.person] = st
            elif stream == "cards":
                self.cards.append(rec)
        return items

    def done(self, t: float) -> bool:
        return t >= self.end_t and self.script.exhausted

    def timeline(self) -> list[dict]:
        return timeline_from_cards(self.cards)


class ReplayAPI:
    """DirectorAPI for an exact replay: shows the recording, refuses every action."""

    def __init__(self, replay: ExactReplay, loaded: Optional[LoadedScenario], run_id: str, meta: dict):
        self.replay = replay
        self.loaded = loaded
        self.run_id = run_id
        self.meta = meta
        self._map = area_map(loaded)
        self._deck = deck_list(loaded)
        self._names = {i.person_id: (i.display_name, i.persona_id) for i in (loaded.instances if loaded else [])}
        self._state: dict = {}
        self.refresh(0.0)

    def refresh(self, t: float) -> None:
        cast = []
        for pid in sorted(self.replay.statuses):
            st = self.replay.statuses[pid]
            name, persona = self._names.get(pid, (pid, ""))
            cast.append({"id": pid, "display_name": name, "persona": persona, "x": st.x, "y": st.y,
                         "posture": st.posture, "mood": None, "last_decision": None, "reason": None,
                         "last_said": None})
        timeline = self.replay.timeline()
        events = [{"t": r["fired_t"], "text": f"{r['card']} {r['state']}" + (f" at {r['target']}" if r["target"] else "")}
                  for r in timeline]
        cost = self.meta.get("cost_usd")
        budget = self.meta.get("budget_usd", self.loaded.scenario.budget_usd if self.loaded else 0.0)
        self._state = {
            "t": float(t), "running": not self.replay.done(t), "paused": False, "run_id": self.run_id,
            "recording": False, "cost_usd": float(cost) if isinstance(cost, (int, float)) else 0.0,
            "budget_usd": float(budget or 0.0), "chaos": {"enabled": False, "rate_per_hour": 0.0},
            "timeline": timeline, "cast": cast, "events": events[-100:], "deck": self._deck, "map": self._map,
        }

    def state(self) -> dict:
        return self._state

    def control(self, action: str) -> None:
        raise BadRequest(READ_ONLY)

    def fire(self, card: str, target: Optional[str], params: dict) -> dict:
        return {"ok": False, "error": READ_ONLY}

    def set_chaos(self, enabled: bool, rate_per_hour: float) -> None:
        raise BadRequest(READ_ONLY)

    def beat(self, action: str, beat_id: str) -> None:
        raise BadRequest(READ_ONLY)

    def snapshot(self, name: str) -> dict:
        return {"ok": False, "error": READ_ONLY}


# ---- ROS wrapper ---------------------------------------------------------------------------
def _parse(argv):
    from .rosutil import base_args
    p = base_args("Replay a recorded run (--run-dir is the recording).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--intents", action="store_true", help="re-send the people's intents to a live stub sim")
    mode.add_argument("--exact", action="store_true", help="republish the recorded world as it was")
    p.add_argument("--time-scale", type=float, default=1.0, help="--exact only: playback speed")
    p.add_argument("--scenario", help="--exact only: scenario for the panel map (default: from meta.json)")
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--panel-port", type=int, default=8765)
    return p.parse_args(argv)


async def _intents(args, run_dir: Path) -> int:
    from std_msgs.msg import Int32

    from . import topics
    from .rosutil import RosBridge, install_stop_signals, wait_or_sleep

    script = IntentScript(run_dir)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("wardzero_replay", loop)
    ros.subscribe_json(Control, topics.CONTROL, lambda c: stop.set() if c.action == "stop" else None)
    ready = ros.publisher(topics.PENDING, Int32)
    ros.start()
    log.info("replaying intents, says and world events from %s", run_dir)
    last_beat = 0.0
    try:
        while not stop.is_set():
            for item in script.due(ros.now()):
                if isinstance(item, Intent):
                    ros.publish_json(topics.intent(item.person), item)
                elif isinstance(item, Say):
                    ros.publish_json(topics.say(item.person), item)
                elif isinstance(item, WorldEvent):
                    ros.publish_json(topics.WORLD_EVENT, item)
            if loop.time() - last_beat >= 1.0:  # the stub sim waits for this before starting the clock
                msg = Int32()
                msg.data = 0
                ready.publish(msg)
                last_beat = loop.time()
            await wait_or_sleep(stop, 0.01)
    finally:
        ros.close()
    return 0


def _load_for_panel(args, meta: dict) -> Optional[LoadedScenario]:
    from ..config.loader import ConfigError, load_scenario
    path = args.scenario or meta.get("scenario_path")
    if not path:
        return None
    try:
        return load_scenario(path)
    except (ConfigError, OSError) as exc:
        log.warning("panel map unavailable: %s", exc)
        return None


async def _exact(args, run_dir: Path) -> int:
    from rosgraph_msgs.msg import Clock

    from ..panel.server import run_panel
    from . import topics
    from .rosutil import QOS, RosBridge, install_stop_signals, wait_or_sleep

    if not args.time_scale > 0:
        print("--time-scale must be above 0", file=sys.stderr)
        return 2
    try:
        meta = load_meta(run_dir)
    except (OSError, ValueError):
        meta = {}
    replay = ExactReplay(run_dir)
    api = ReplayAPI(replay, _load_for_panel(args, meta), run_dir.name, meta)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge("wardzero_replay", loop, use_sim_time=False)  # it is the clock here
    clock_pub = ros.publisher("/clock", Clock, QOS)
    ros.start()
    runner = None
    if not args.no_panel:
        try:
            runner = await run_panel(api, "127.0.0.1", args.panel_port)
            print(f"Control panel (read-only replay): http://127.0.0.1:{args.panel_port}/", flush=True)
        except OSError as exc:
            log.error("the panel could not start on port %d (%s)", args.panel_port, type(exc).__name__)
    log.info("exact replay of %s: %.1f s of sim time at x%.1f", run_dir, replay.end_t, args.time_scale)
    t, last, last_tl = 0.0, time.monotonic(), float("-inf")
    finished = False
    try:
        while not stop.is_set():
            now = time.monotonic()
            if not finished:
                t = min(replay.end_t, t + args.time_scale * (now - last))
            last = now
            msg = Clock()
            msg.clock.sec = int(t)
            msg.clock.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
            clock_pub.publish(msg)
            for stream, rec in replay.step(t):
                text = json.dumps(rec)
                if stream == "status" and isinstance(rec.get("person"), str):
                    ros.publish_text(topics.status(rec["person"]), text)
                elif stream == "audio":
                    ros.publish_text(topics.AUDIO_EVENTS, text)
            if t - last_tl >= TIMELINE_PERIOD_S or (replay.done(t) and not finished):
                ros.publish_text(topics.TIMELINE, json.dumps(
                    {"schema": "wz.v1", "t": t, "beats": replay.timeline()}))
                last_tl = t
            api.refresh(t)
            if replay.done(t) and not finished:
                finished = True
                log.info("replay finished at t=%.1f s", t)
                if runner is None:
                    break
                log.info("the panel stays up; press Ctrl+C to quit")
            await wait_or_sleep(stop, 1.0 / STEP_HZ)
    finally:
        if runner is not None:
            await runner.cleanup()
        ros.close()
    return 0


def main(argv=None) -> int:
    from .rosutil import setup_logging
    setup_logging()
    args = _parse(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"No run folder at {run_dir}", file=sys.stderr)
        return 2
    return asyncio.run(_intents(args, run_dir) if args.intents else _exact(args, run_dir))


if __name__ == "__main__":
    sys.exit(main())
