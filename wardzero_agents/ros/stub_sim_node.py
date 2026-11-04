"""Stub sim node (wardzero_stub_sim): the kinematic StubWorld on ROS, standing in for Isaac.

The sim itself is ``sim_service.SimService`` with StubBodies (the same service the Isaac host runs
with IsaacBodies) and its topics are ``sim_service.SimRos``. This node steps it on the wall clock:
each wall step advances sim time by time_scale x wall dt.

    python -m wardzero_agents.ros.stub_sim_node --run-dir DIR --scenario FILE [--time-scale F]
        [--lockstep] [--wait-for agents,director] [--ready-timeout 600]
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time

from ..config.loader import ConfigError, load_scenario
from ..runlog import RunLog
from ..world.bodies import StubBodies
from .sim_service import (  # re-exported: tests and older callers import them from here
    MAX_WALL_DT, OBS_PERIOD_S, OBS_WINDOW_S, ODOM_PERIOD_S, PENDING_SILENT_S, PERCEPTION_PERIOD_S,
    READY_TIMEOUT_S, REPLY_WINDOW_S, ROBOT_ADDRESS_M, STATUS_PERIOD_S, STATUS_RECORD_PERIOD_S, STEP_HZ,
    WAIT_TOKENS, SimOutputs, SimRos, SimService,
)

__all__ = [
    "MAX_WALL_DT", "OBS_PERIOD_S", "OBS_WINDOW_S", "ODOM_PERIOD_S", "PENDING_SILENT_S", "PERCEPTION_PERIOD_S",
    "READY_TIMEOUT_S", "REPLY_WINDOW_S", "ROBOT_ADDRESS_M", "STATUS_PERIOD_S", "STATUS_RECORD_PERIOD_S", "STEP_HZ",
    "WAIT_TOKENS", "SimOutputs", "SimRos", "SimService", "StubSimService", "main",
]
log = logging.getLogger("wardzero.stub_sim")


class StubSimService(SimService):
    """The sim service with the stand-in kinematics (StubBodies), the default for ``--sim stub``."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bodies", StubBodies())
        super().__init__(*args, **kwargs)


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
    from .rosutil import RosBridge, install_stop_signals, wait_or_sleep

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
    sim_ros = SimRos(svc, ros)
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
            sim_ros.publish(out)
            if out.stop:
                log.info("scenario over at t=%.1f s; stopping the run", out.clock)
                sim_ros.publish_stop()
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
