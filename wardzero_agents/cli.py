"""hhumans: run, rerun and replay persona simulations; manage memory, the knowledge base and keys.

    hhumans run SCENARIO [--sim isaac|stub] [--headed|--headless] [--record] [--full-site]
                [--time-scale F] [--lockstep] [--no-robot] [--no-panel]
                [--panel-port 8765] [--bot claude|reception|none] [--ready-timeout 600]
                [--repeat N] [--carry-memory] [--baseline] [--no-report]
    hhumans rerun RUN_DIR [same flags]
    hhumans debrief RUN_DIR [--force]
    hhumans voice RUN_DIR [--models-dir DIR]
    hhumans report RUN_DIR|BATCH_DIR [--no-narrative]
    hhumans replay RUN_DIR (--intents | --exact) [--time-scale F]
    hhumans memory reset (--person ID --scenario S | --persona NAME --scenario S | --all) [--yes]
    hhumans memory snapshot save|restore|list [NAME]
    hhumans kb build [--embedder hash|fastembed] [--scenario S]
    hhumans keys set|clear|status claude|openai
    hhumans studio [--port 8770] [--no-browser]
    hhumans audit

`run` checks the key in this process, copies the scenario and its configs into a new run folder
(the copy is self-contained) and starts the ROS 2 nodes as child processes, each in its own
process group, all loading that copy: the sim (the Isaac host with --sim isaac, the default when
Isaac Sim is installed at ~/isaacsim; the kinematic stub sim with --sim stub), the agent service,
the director and, unless
--bot none, the robot: the built-in Claude robot when a Claude key is set, else the placeholder
reception robot. The run ends when any child exits (normally the sim,
at the scenario's duration); the rest then get SIGINT, SIGTERM after 10 s, SIGKILL after a
further grace (longer for the agent node, which drains its minds and saves the memory snapshot).
Ctrl+C does the same; press it again to hurry. Run folders go to $WARDZERO_RUNS_DIR, default
<hospital-sim-agents>/runs.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import importlib.util
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import keys

AGENTS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = AGENTS_ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_ROOT") or "~/isaacsim").expanduser()
ISAAC_HOST_SH = AGENTS_ROOT / "scripts" / "isaac_host.sh"
ISAAC_KILL_AFTER_S = 30.0  # Kit can take a while to shut down
SIMS = ("isaac", "stub")
FASTEMBED_ADVICE = ("The fastembed embedder needs the optional extra. Install it with:\n"
                    f"  pip install -e '{AGENTS_ROOT}[embed]'")
DEFAULT_PANEL_PORT = 8765
DEFAULT_READY_TIMEOUT_S = 600.0
TERM_AFTER_S = 10.0  # SIGINT, then SIGTERM after this long
STOP_MARKER = "Stopping the run"  # printed when the orderly stop of all children begins
KILL_AFTER_S = 5.0  # SIGTERM, then SIGKILL after this long
ISAAC_EARLY_EXIT_GRACE_S = 90.0  # the agents and director stop on the host's stop message; Kit then takes a while to exit
AGENT_KILL_AFTER_S = 80.0  # the agent node drains its minds for up to 60 s, then saves the snapshot
EARLY_EXIT_GRACE_S = 5.0  # a clean exit of another child this long before the run's end is an early exit
KEY_LABELS = ("stored in keyring", "from key file", "from ant profile or environment", "not set")
KEY_UNVERIFIED = "ant profile or ANTHROPIC_API_KEY (not verified)"


class CliError(Exception):
    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def runs_dir() -> Path:
    """$WARDZERO_RUNS_DIR, default <hospital-sim-agents>/runs (from the package, not the cwd)."""
    env = os.environ.get("WARDZERO_RUNS_DIR")
    return Path(env).expanduser() if env else AGENTS_ROOT / "runs"


def ros_available() -> bool:
    return bool(os.environ.get("ROS_DISTRO")) and importlib.util.find_spec("rclpy") is not None


def isaac_installed() -> bool:
    return ISAAC_ROOT.is_dir()


def default_sim() -> str:
    """$WARDZERO_SIM when set, else isaac when Isaac Sim is installed, else the stub."""
    env = os.environ.get("WARDZERO_SIM", "").strip()
    if env in SIMS:
        return env
    return "isaac" if isaac_installed() else "stub"


def _resolve_sim(args) -> None:
    """Pick the backend and check the flags that only the Isaac host takes."""
    args.sim = getattr(args, "sim", None) or default_sim()
    if args.sim == "isaac":
        if not ISAAC_ROOT.is_dir():
            raise CliError(f"Isaac Sim was not found at {ISAAC_ROOT} (set ISAAC_SIM_ROOT), or use --sim stub.")
        return
    only = [f for f, on in (("--headed", getattr(args, "headed", None)), ("--record", getattr(args, "record", False)),
                            ("--preview", getattr(args, "preview", False)),
                            ("--full-site", getattr(args, "full_site", False))) if on]
    if only:
        raise CliError(f"{', '.join(only)} need{'s' if len(only) == 1 else ''} the Isaac host (--sim isaac).")


def _need_ros() -> None:
    if not ros_available():
        raise CliError("ROS 2 is not sourced. Run `source $HG1_ROOT/env.sh` first.")


def _load(path) -> "LoadedScenario":  # noqa: F821
    from .config.loader import ConfigError, load_scenario
    try:
        return load_scenario(path)
    except ConfigError as exc:
        raise CliError(f"Scenario error: {exc}") from None


def _positive(value: float, flag: str) -> float:
    if not (value > 0 and math.isfinite(value)):
        raise CliError(f"{flag} must be a positive number")
    return float(value)


# ---- child processes -----------------------------------------------------------------------
@dataclass
class Child:
    name: str
    argv: list[str]
    kill_after_s: Optional[float] = None  # SIGTERM to SIGKILL grace; None: the supervise default
    ends_run: bool = False  # its clean exit is the normal end of the run (the stub sim)
    env: Optional[dict] = None  # this child's environment; None: the env given to supervise


def _module(mod: str, *args: str) -> list[str]:
    return [sys.executable, "-m", mod, *args]


# Credentials only the agent node and the Claude robot need. Every other child, and the placeholder
# reception robot, runs without them.
KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE")


def _child_env(with_keys: bool = False) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if not with_keys:
        for k in KEY_ENV:
            env.pop(k, None)
    # the children import the same wardzero_agents as this process
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(AGENTS_ROOT), env.get("PYTHONPATH", "")) if p)
    return env


def _signal(p: subprocess.Popen, sig: int) -> None:
    """Signal the child's process group (it leads it). Only while the child is unreaped, so the
    pid and group id cannot have been reused."""
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            p.send_signal(sig)
        except ProcessLookupError:
            pass


def _code(rc: int) -> int:
    return 128 - rc if rc < 0 else rc


def supervise(children: list[Child], *, term_after_s: float = TERM_AFTER_S, kill_after_s: float = KILL_AFTER_S,
              poll_s: float = 0.1, early_exit_grace_s: float = EARLY_EXIT_GRACE_S, env: Optional[dict] = None,
              on_start: Optional[Callable[[dict], None]] = None) -> int:
    """Start the children, wait until the run ends or Ctrl+C, then stop the rest.

    The run ends when a child marked ends_run exits (0 is a clean end), when any child exits
    non-zero, or when another child exits 0 and no ends_run child follows within
    early_exit_grace_s (at a normal end the agents and director may exit on the stop message
    just before the stub sim does). With no child marked ends_run, any exit ends the run.

    Returns the first non-zero exit code, 130 on Ctrl+C/SIGTERM, 1 when a child other than the
    run's end exited early, and 1 after a clean end when a child failed or had to be killed while
    stopping (its memory snapshot or run meta may be missing); otherwise 0. Each child runs in
    its own process group and is only ever signalled by pid or process group.
    """
    signals: list[int] = []
    handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old = {s: signal.signal(s, lambda signum, frame: signals.append(signum)) for s in handled}
    procs: dict[str, subprocess.Popen] = {}
    by_name = {c.name: c for c in children}
    enders = {c.name for c in children if c.ends_run}
    code: Optional[int] = None
    clean = False
    early: Optional[tuple[str, float]] = None  # a child that exited 0 before the run's end, and when
    try:
        for c in children:
            try:
                procs[c.name] = subprocess.Popen(c.argv, env=c.env if c.env is not None else env,
                                                 start_new_session=True)
            except OSError as exc:
                err(f"could not start {c.name}: {exc}")
                code = 2
                break
        if code is None and on_start is not None:
            on_start(procs)
        while code is None:
            if signals:
                print(STOP_MARKER + " (Ctrl+C again to hurry).", flush=True)
                code = 130
                break
            exited = {n: p.returncode for n, p in procs.items() if p.poll() is not None}
            failed = [(n, rc) for n, rc in exited.items() if rc != 0]
            if failed:
                name, rc = failed[0]
                code = _code(rc)
                err(f"{name} exited with code {code}; stopping the run")
                break
            if exited and (not enders or enders & exited.keys()):
                code, clean = 0, True
                break
            if exited:
                now = time.monotonic()
                if early is None:
                    early = (next(iter(exited)), now)
                elif now - early[1] >= early_exit_grace_s:
                    err(f"{early[0]} stopped on its own before the scenario finished, so the scenario did "
                        "not finish; stopping the run")
                    code = 1
                    break
            time.sleep(poll_s)
        if not signals:  # a signal already printed it above
            print(STOP_MARKER + " (orderly stop).", flush=True)  # the Studio watches for this line
        problems = _stop_all(procs, by_name, signals, term_after_s, kill_after_s, poll_s)
        for name, why in problems:
            if clean or code == 130:
                err(f"{name} {why} while the run was stopping; its memory snapshot or run record may be "
                    "incomplete")
        if clean and problems:
            code = 1
    finally:
        for p in procs.values():  # only reached alive on an unexpected error
            _signal(p, signal.SIGKILL)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for s, h in old.items():
            signal.signal(s, h)
    return code


def _stop_all(procs: dict, by_name: dict, signals: list, term_after_s: float, kill_after_s: float,
              poll_s: float) -> list[tuple[str, str]]:
    """SIGINT, SIGTERM, SIGKILL the children still running. Returns (child, what went wrong)
    for each that had to be killed or exited non-zero."""
    live = {n: p for n, p in procs.items() if p.poll() is None}
    for p in live.values():
        _signal(p, signal.SIGINT)
    t0 = time.monotonic()
    seen = len(signals)
    termed: dict[str, float] = {}
    killed: list[str] = []
    while any(p.poll() is None for p in live.values()):
        now = time.monotonic()
        extra = len(signals) - seen  # Ctrl+C again: SIGTERM now; a third time: SIGKILL now
        for name, p in live.items():
            if p.poll() is not None:
                continue
            if name not in termed:
                if now - t0 >= term_after_s or extra >= 1:
                    _signal(p, signal.SIGTERM)
                    termed[name] = now
            else:
                grace = by_name[name].kill_after_s if by_name[name].kill_after_s is not None else kill_after_s
                if (now - termed[name] >= grace or extra >= 2) and name not in killed:
                    err(f"{name} did not stop; killing it")
                    _signal(p, signal.SIGKILL)
                    killed.append(name)
        time.sleep(poll_s)
    problems = []
    for name, p in live.items():
        if name in killed:
            problems.append((name, "did not stop and was killed"))
        elif p.returncode != 0:
            problems.append((name, f"exited with code {_code(p.returncode)}"))
    return problems


# ---- run, rerun ----------------------------------------------------------------------------
def _key_check(loaded) -> None:
    from .ros.agent_node import _key_check as check
    code = check(loaded)  # prints TEST MODE, or the fix-it message; never the key or the exception
    if code is not None:
        raise CliError("", code)


def _refuse_if_active() -> None:
    from .memory import data_dir
    from .memory.runlock import active_run
    other = active_run(data_dir())
    if other is not None:
        raise CliError(f"Run {other} is active; stop it first.")


def _stage(loaded, root: Path, run_name: str, overrides: dict, meta: dict) -> tuple[Path, "LoadedScenario"]:  # noqa: F821
    """A new run folder holding a self-contained copy of the configs, and the loaded copy."""
    from .runlog import RunLog, new_run_dir
    run_dir = new_run_dir(root, run_name, datetime.now())
    try:
        runlog = RunLog(run_dir)
        staged = _load(runlog.copy_configs(loaded, overrides))
        runlog.write_meta(meta)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir, staged


def _robot_override(loaded, args) -> dict:
    if not getattr(args, "no_robot", False):
        return {}
    return {"robot": {**loaded.scenario.robot.model_dump(), "enabled": False}}


def _claude_key_present() -> bool:
    """Whether a Claude credential is configured (not whether it works; the run checks that)."""
    if os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1":
        return False  # tests and demos without a provider keep the placeholder robot
    return key_source("claude") != KEY_LABELS[3]


def _bot(args, staged, claude_default: bool = True) -> str:
    if args.no_robot or not staged.scenario.robot.enabled:
        if args.bot in ("reception", "claude"):
            err(f"This run has no robot, so the {args.bot} bot is not started.")
        return "none"
    if args.bot:
        return args.bot
    return "claude" if claude_default and _claude_key_present() else "reception"


def _bot_children(bot: str, sp: str, rd: str) -> list:
    if bot == "reception":
        return [Child("reception_bot", _module("wardzero_agents.examples.reception_bot", "--scenario", sp))]
    if bot == "claude":  # the one robot process that gets the key, like the agent node
        return [Child("robot_claude", _module("wardzero_agents.robots.claude_robot", "--scenario", sp,
                                              "--run-dir", rd), env=_child_env(with_keys=True))]
    return []


def _start_live(args, loaded, overrides: dict, meta: dict, out: Optional[dict] = None) -> int:
    ts = _positive(args.time_scale, "--time-scale")
    ready = _positive(args.ready_timeout, "--ready-timeout")
    _key_check(loaded)
    _need_ros()
    _refuse_if_active()
    overrides = {**overrides, **_robot_override(loaded, args)}
    sim = getattr(args, "sim", None) or "stub"
    meta = {**meta, "sim": sim, "time_scale": ts, "lockstep": bool(args.lockstep or loaded.scenario.lockstep or (sim == "isaac" and args.record)),
            "panel": not args.no_panel}
    if sim == "isaac":
        meta.update(headed=bool(args.headed), record=bool(args.record), full_site=bool(args.full_site))
        if args.record:
            meta["record_res"] = args.record_res
    for key, val in (("robot_domain", getattr(args, "robot_domain", None)),
                     ("project_id", getattr(args, "project_id", None))):
        if val is not None:  # the Isaac host also gets robot_domain; the Studio lists runs by project_id
            meta[key] = val
    run_dir, staged = _stage(loaded, runs_dir(), loaded.scenario.id, overrides, meta)
    bot = _bot(args, staged)
    rd, sp = str(run_dir), str(staged.scenario_path)
    children = [
        _sim_child(args, rd, sp, ts, ready),
        Child("agents", _module("wardzero_agents.ros.agent_node", "--run-dir", rd, "--scenario", sp),
              kill_after_s=AGENT_KILL_AFTER_S, env=_child_env(with_keys=True)),
        Child("director", _module("wardzero_agents.ros.director_node", "--run-dir", rd, "--scenario", sp,
                                  *(["--no-panel"] if args.no_panel else ["--panel-port", str(args.panel_port)]))),
    ]
    children += _bot_children(bot, sp, rd)
    if out is not None:
        out["run_dir"] = run_dir
    print(f"Run folder: {run_dir}", flush=True)
    print(f"Scenario {staged.scenario.id}: {staged.scenario.duration_min:g} min of sim time at x{ts:g}, "
          f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}, robot: {bot}, sim: {_sim_label(args)}", flush=True)
    if not args.no_panel:
        print(f"Control panel: http://127.0.0.1:{args.panel_port}/", flush=True)
    grace = ISAAC_EARLY_EXIT_GRACE_S if sim == "isaac" else EARLY_EXIT_GRACE_S
    code = supervise(children, env=_child_env(), early_exit_grace_s=grace)
    print(f"Run finished (exit code {code}). Run folder: {run_dir}", flush=True)
    if sim == "isaac" and args.record and not getattr(args, "no_voice", False) and code in (0, 130):
        _post_voice(run_dir)
    return code


def _sim_child(args, rd: str, sp: str, ts: float, ready: float) -> Child:
    """The process that owns the world and ends the run: the Isaac host or the stub sim."""
    lock = ["--lockstep"] if args.lockstep else []
    if getattr(args, "sim", None) == "isaac":
        argv = [str(ISAAC_HOST_SH), "--run-dir", rd, "--scenario", sp, "--time-scale", str(ts),
                "--ready-timeout", str(ready), "--wait-for", "agents,director",
                "--private-domain", os.environ.get("ROS_DOMAIN_ID", "").strip() or "0",
                *lock, "--headed" if args.headed else "--headless"]
        argv += ["--record", "--record-res", args.record_res] if args.record else []
        argv += ["--preview"] if getattr(args, "preview", False) else []
        argv += ["--bag-images"] if getattr(args, "bag_images", False) else []
        argv += ["--full-site"] if args.full_site else []
        if getattr(args, "robot_domain", None) is not None:
            argv += ["--robot-domain", str(args.robot_domain)]
        return Child("isaac_host", argv, kill_after_s=ISAAC_KILL_AFTER_S, ends_run=True)  # no key in its env
    return Child("stub_sim", _module("wardzero_agents.ros.stub_sim_node", "--run-dir", rd, "--scenario", sp,
                                     "--time-scale", str(ts), "--ready-timeout", str(ready),
                                     "--wait-for", "agents,director", *lock), ends_run=True)


def _sim_label(args) -> str:
    if getattr(args, "sim", None) != "isaac":
        return "stub"
    extra = [w for w, on in (("record", args.record), ("full site", args.full_site)) if on]
    return "isaac (" + ", ".join(["headed" if args.headed else "headless", *extra]) + ")"


def _post_run(run_dir: Path, report: bool = True) -> Optional[Path]:
    """Debrief a finished run and write its report, in this process with the user's key. A failure here
    never fails the run: the folder is kept and the fix-it command is printed."""
    from .bench.report import post_run
    try:
        path = asyncio.run(post_run(run_dir, report=report))
    except (keys.KeyMissing, keys.KeyRejected) as exc:
        err(f"No debrief or report: {keys.redact(str(exc))[:300]}")
    except Exception as exc:
        err(f"The debrief or report failed ({type(exc).__name__}: {keys.redact(str(exc))[:200]}). "
            f"Run `hhumans debrief {run_dir}` and `hhumans report {run_dir}` to try again.")
    else:
        if path is not None:
            print(f"Report: {path}", flush=True)
        return path
    return None


def _post_voice(run_dir: Path) -> None:
    """Spoken voices for a recorded run, after the simulator has exited (CPU only). Never fails the run."""
    try:
        from .voice.post import build_voice
        build_voice(run_dir, say=lambda m: print(m, flush=True))
    except Exception as exc:
        err(f"No voices ({type(exc).__name__}: {keys.redact(str(exc))[:200]}). Run `hhumans voice {run_dir}` to try again.")


def cmd_voice(args) -> int:
    from .voice.post import build_voice
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise CliError(f"No folder at {run_dir}")
    res = build_voice(run_dir, models_dir=Path(args.models_dir) if args.models_dir else None)
    return 0 if res["wav"] is not None or res["warning"] is None else 1


def _ran_for_real(run_dir: Path) -> bool:
    """The agent node wrote its run meta (a run id), so there is something to debrief."""
    try:
        return bool(json.loads((run_dir / "meta.json").read_text(encoding="utf-8")).get("run_id"))
    except (OSError, ValueError):
        return False


def _cmd_batch(args, loaded) -> int:
    from .bench.batch import run_batch
    from .config.presets import preset_for_run
    if args.repeat < 1:
        raise CliError("--repeat must be at least 1")
    preset = preset_for_run({"preset_id": loaded.scenario.preset_id, "scenario_id": loaded.scenario.id})
    meta = {"mode": "live", "scenario_source": str(loaded.scenario_path)}

    def runner(seed: int, profile: Optional[str]) -> Path:
        overrides: dict = {"seed": seed}
        if profile:
            overrides["robot"] = {**loaded.scenario.robot.model_dump(), "profile": profile}
        out: dict = {}
        code = _start_live(args, loaded, overrides, meta, out=out)
        if code != 0 or "run_dir" not in out:
            raise CliError(f"A run of the batch ended with code {code}; the batch stops here.", code or 1)
        if _ran_for_real(out["run_dir"]):
            _post_run(out["run_dir"], report=not args.no_report)
        return out["run_dir"]

    batch_dir = run_batch(loaded, args.repeat, args.carry_memory, args.baseline, runner=runner, preset=preset,
                          log=lambda m: print(m, flush=True))
    print(f"Batch (repeatability): {batch_dir / 'report.html'}", flush=True)
    return 0


def cmd_run(args) -> int:
    _resolve_sim(args)
    loaded = _load(args.scenario)
    if args.repeat > 1 or args.baseline:
        return _cmd_batch(args, loaded)
    out: dict = {}
    code = _start_live(args, loaded, {}, {"mode": "live", "scenario_source": str(loaded.scenario_path)}, out=out)
    # 130 is a stop by the user (the Studio's Stop, Ctrl+C): the run is still debriefed and reported.
    # Any other non-zero code is a crash and gets neither.
    if code in (0, 130) and not args.no_report and "run_dir" in out and _ran_for_real(out["run_dir"]):
        _mark_stopped_early(out["run_dir"], loaded, force=code == 130)
        _post_run(out["run_dir"])
    return code


def _mark_stopped_early(run_dir: Path, loaded, force: bool = False) -> None:
    """meta.json "stopped_early": true when the run ended before the scenario's length (or by SIGINT)."""
    from .runlog import RunLog
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        end = 0.0
        try:
            for line in (run_dir / "status.jsonl").read_text(encoding="utf-8").splitlines():
                end = max(end, float(json.loads(line).get("t") or 0.0))
        except (OSError, ValueError):
            pass
        # no status records at all says nothing about when the run ended: only a user stop (force) counts
        if force or (end > 0 and end < 0.95 * loaded.scenario.duration_min * 60):
            RunLog(run_dir).write_meta({**meta, "stopped_early": True})
    except (OSError, ValueError):
        pass


def cmd_debrief(args) -> int:
    from .bench.report import post_run
    run_dir = Path(args.run_dir).resolve()
    _recorded_meta(run_dir)
    _refuse_if_active()
    try:
        asyncio.run(post_run(run_dir, report=False, force_debrief=args.force))
    except (keys.KeyMissing, keys.KeyRejected) as exc:
        raise CliError(keys.redact(str(exc))[:300], 1) from None
    print(f"Debrief: {run_dir / 'debrief.jsonl'} and {run_dir / 'trust.jsonl'}")
    return 0


def cmd_report(args) -> int:
    from .bench.batch import report_batch
    from .bench.report import build_report_async
    from .config.presets import preset_for_run
    from .campaign.debrief import make_post_gateway
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise CliError(f"No folder at {run_dir}")
    if (run_dir / "batch.json").is_file():
        print(f"Batch report (repeatability): {report_batch(run_dir)}")
        return 0
    meta = _recorded_meta(run_dir)
    try:
        gateway = None if args.no_narrative else make_post_gateway(run_dir)
        path = asyncio.run(build_report_async(run_dir, preset_for_run(meta), gateway))
    except (keys.KeyMissing, keys.KeyRejected) as exc:
        raise CliError(keys.redact(str(exc))[:300] + "\nUse --no-narrative for a report without the AI text.", 1) from None
    print(f"Report: {path}")
    return 0


def _recorded_meta(run_dir: Path) -> dict:
    from .replay import load_meta
    if not run_dir.is_dir():
        raise CliError(f"No run folder at {run_dir}")
    try:
        return load_meta(run_dir)
    except (OSError, ValueError):
        raise CliError(f"{run_dir} has no readable meta.json; is it a run folder?") from None


def _load_recorded(run_dir: Path, meta: dict, tmp: Path):
    """The run's copied scenario, loaded from a scratch copy of its config/ folder.

    Copies made before the copy was self-contained name the storyboard and deck by their
    original relative paths; those are pointed at the copied files here.
    """
    cfg = run_dir / "config"
    if not cfg.is_dir():
        raise CliError(f"{run_dir} has no config/ folder to rerun from")
    name = Path(meta.get("scenario_path") or "").name
    if not name or not (cfg / name).is_file():
        name = ""
        for f in sorted(cfg.glob("*.yaml")):
            try:
                raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(raw, dict) and "cast" in raw and "storyboard" in raw:
                name = f.name
                break
    if not name:
        raise CliError(f"no scenario file in {cfg}")
    work = tmp / "config"
    shutil.copytree(cfg, work)
    scen = work / name
    raw = yaml.safe_load(scen.read_text(encoding="utf-8"))
    for field in ("storyboard", "deck"):
        base = Path(str(raw.get(field, ""))).name
        if base and (work / base).is_file():
            raw[field] = base
    scen.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return _load(scen)


def cmd_rerun(args) -> int:
    from .memory import data_dir
    _resolve_sim(args)
    src = Path(args.run_dir).resolve()
    meta = _recorded_meta(src)
    snap = meta.get("snapshot_in")
    if not isinstance(snap, str) or not snap:
        raise CliError(f"{src} records no start snapshot (snapshot_in), so it cannot be rerun.")
    if not (data_dir() / "snapshots" / f"{snap}.sqlite").is_file():
        raise CliError(f"Memory snapshot {snap} is missing from {data_dir() / 'snapshots'}; "
                       "it cannot be rerun with the memories it started with.")
    with tempfile.TemporaryDirectory(prefix="hhumans-rerun-") as tmp:
        loaded = _load_recorded(src, meta, Path(tmp))
        source = meta.get("scenario_source") or meta.get("scenario_path")
        return _start_live(args, loaded, {"memory_snapshot": snap},
                           {"mode": "rerun", "rerun_of": src.name, "scenario_source": source})


# ---- replay --------------------------------------------------------------------------------
def cmd_replay(args) -> int:
    src = Path(args.run_dir).resolve()
    meta = _recorded_meta(src)
    ts = _positive(args.time_scale, "--time-scale")
    _need_ros()
    rd = str(src)
    if args.exact:
        scen = []
        copy = src / "config" / Path(meta.get("scenario_path") or "").name
        if copy.is_file():
            scen = ["--scenario", str(copy)]
        panel = ["--no-panel"] if args.no_panel else ["--panel-port", str(args.panel_port)]
        children = [Child("replay", _module("wardzero_agents.ros.replay_node", "--run-dir", rd, "--exact",
                                            "--time-scale", str(ts), *scen, *panel), ends_run=True)]
        print(f"Replaying {src.name} exactly at x{ts:g}.", flush=True)
        if not args.no_panel:
            print(f"Control panel (read-only): http://127.0.0.1:{args.panel_port}/", flush=True)
        return supervise(children, env=_child_env())

    ready = _positive(args.ready_timeout, "--ready-timeout")
    with tempfile.TemporaryDirectory(prefix="hhumans-replay-") as tmp:
        loaded = _load_recorded(src, meta, Path(tmp))
        out_dir, staged = _stage(loaded, runs_dir(), f"{loaded.scenario.id}-replay", _robot_override(loaded, args),
                                 {"mode": "replay-intents", "replay_of": src.name, "time_scale": ts})
    bot = _bot(args, staged, claude_default=False)  # a replay stays free of API calls unless asked
    sp = str(staged.scenario_path)
    children = [
        Child("stub_sim", _module("wardzero_agents.ros.stub_sim_node", "--run-dir", str(out_dir), "--scenario", sp,
                                  "--time-scale", str(ts), "--ready-timeout", str(ready), "--wait-for", "agents"),
              ends_run=True),
        Child("replay", _module("wardzero_agents.ros.replay_node", "--run-dir", rd, "--intents",
                                "--record-dir", str(out_dir))),
    ]
    children += _bot_children(bot, sp, str(out_dir))
    print(f"Replaying the intents of {src.name} at x{ts:g}, robot: {bot}. Recording to {out_dir}", flush=True)
    return supervise(children, env=_child_env())


# ---- memory --------------------------------------------------------------------------------
def _db_keys(mem: Path) -> list[str]:
    if not mem.exists():
        return []
    try:
        with closing(sqlite3.connect(f"file:{mem}?mode=ro", uri=True)) as c:
            return [r[0] for r in c.execute("SELECT DISTINCT person_key FROM memories ORDER BY person_key")]
    except sqlite3.Error:
        return []


def _personas(loaded):
    if loaded is not None:
        return loaded.personas
    return _load(DEFAULT_SCENARIO).personas


def cmd_memory_reset(args) -> int:
    from .kb.embed import get_embedder
    from .memory import data_dir, memory_path
    from .memory.runlock import RunActive
    from .memory.snapshots import guarded_reset

    if (args.person or args.persona) and not args.scenario:
        raise CliError("--person and --persona need --scenario to know who that is")
    try:
        _refuse_if_active()  # before asking; guarded_reset checks again
    except CliError as exc:
        raise CliError(f"Refused: {exc}") from None
    loaded = _load(args.scenario) if args.scenario else None
    who: dict[str, str] = {}  # memory key -> who it is
    seeds: dict[str, list[str]] = {}
    if args.all:
        loaded = loaded or _load(DEFAULT_SCENARIO)
        personas = loaded.personas
        for k in _db_keys(memory_path(data_dir())):
            who[k] = ""
        for i in loaded.instances:
            who[i.memory_key] = f"{i.person_id}, {i.persona_id}"
        for k in who:
            persona = personas.get(k.split("#", 1)[0])
            if persona is not None:
                seeds[k] = list(persona.seed_memories)
    else:
        if args.person:
            chosen = [i for i in loaded.instances if i.person_id == args.person]
            if not chosen:
                raise CliError(f"No person {args.person} in {loaded.scenario.id} "
                               f"(cast: {', '.join(i.person_id for i in loaded.instances)})")
        else:
            chosen = [i for i in loaded.instances if i.persona_id == args.persona]
            if not chosen:
                raise CliError(f"No one with persona {args.persona} in {loaded.scenario.id}")
        for i in chosen:
            who[i.memory_key] = f"{i.person_id}, {i.persona_id}"
            seeds[i.memory_key] = list(loaded.personas[i.persona_id].seed_memories)
    print("This resets the long-term memory of:" if not args.all else "This resets ALL long-term memory:")
    for k in sorted(who):
        print(f"  {k}" + (f" ({who[k]})" if who[k] else ""))
    print("Each goes back to their persona's seed memories.")
    if not args.yes:
        try:
            answer = input("Type yes to reset these memories: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "yes":
            print("Nothing was reset.")
            return 1
    try:
        guarded_reset(data_dir(), sorted(who), seeds, get_embedder(args.embedder))
    except RunActive as exc:
        raise CliError(f"Refused: {exc}") from None
    except ValueError as exc:  # embedder mismatch
        raise CliError(f"Cannot reset: {exc}") from None
    print(f"Reset {len(who)} memories.")
    return 0


def cmd_snapshot(args) -> int:
    from .memory import data_dir, memory_path
    from .memory.runlock import RunActive
    from .memory.snapshots import check_user_name, guarded_restore, list_snapshots, save_snapshot

    d = data_dir()
    if args.action == "list":
        snaps = list_snapshots(d)
        if not snaps:
            print("No snapshots yet.")
        for s in snaps:
            print(f"{s['name']:<40} {s.get('created', '?'):<34} {s.get('rows', '?')} memories")
        return 0
    if not args.name:
        raise CliError(f"memory snapshot {args.action} needs a NAME")
    try:
        if args.action == "save":
            check_user_name(args.name)
            _refuse_if_active()
            path = save_snapshot(memory_path(d), args.name)
            print(f"Saved snapshot {args.name}: {path}")
        else:
            guarded_restore(d, args.name, keep_as=f"pre-restore-manual-{datetime.now():%Y%m%d-%H%M%S}")
            print(f"Restored the long-term memory from snapshot {args.name}.")
    except RunActive as exc:
        raise CliError(f"Refused: {exc}") from None
    except (FileNotFoundError, ValueError) as exc:
        raise CliError(str(exc)) from None
    return 0


# ---- kb ------------------------------------------------------------------------------------
def cmd_kb_build(args) -> int:
    from .kb.embed import get_embedder
    from .kb.index import build_default_kb
    from .memory import data_dir
    from .ros.agent_node import _hg1_root

    if args.embedder == "fastembed" and importlib.util.find_spec("fastembed") is None:
        raise CliError(FASTEMBED_ADVICE, 1)
    try:
        embedder = get_embedder(args.embedder)
    except (ImportError, RuntimeError):
        raise CliError(FASTEMBED_ADVICE, 1) from None
    loaded = _load(args.scenario or DEFAULT_SCENARIO)
    out = data_dir() / "kb"
    kb = build_default_kb(_hg1_root(), loaded, embedder)
    kb.save(out)
    print(f"Knowledge base: {len(kb.chunks)} chunks, embedder {embedder.name}, saved to {out}")
    if args.embedder != "hash":
        print("Runs use the hash embedder unless WARDZERO_EMBEDDER is set to match.")
    return 0


# ---- keys ----------------------------------------------------------------------------------
def _claude_ambient() -> Optional[bool]:
    """Whether the Claude SDK finds a credential by itself (environment or `ant auth login` profile).

    True for ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN, or when the SDK resolves a profile into
    `client.credentials` (anthropic 2.x resolves env, profile and federation credentials at
    construction and leaves it None when there are none). None when the SDK cannot tell.
    Nothing here checks that the credential works.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        import anthropic
        client = anthropic.Anthropic()
        if not hasattr(client, "credentials"):
            return None
        return bool(client.credentials is not None or getattr(client, "api_key", None)
                    or getattr(client, "auth_token", None))
    except Exception:
        return None


def key_source(provider: str) -> str:
    import keyring
    import keyring.errors
    try:
        if keyring.get_password(keys.SERVICE, provider):
            return KEY_LABELS[0]
    except keyring.errors.KeyringError:
        pass
    try:
        if keys._file_key(provider):
            return KEY_LABELS[1]
    except keys.KeyMissing as exc:
        err(str(exc))  # the key file is unusable; the message never holds its contents
    if provider == "claude":
        ambient = _claude_ambient()
        if ambient is None:
            return KEY_UNVERIFIED
    else:
        ambient = bool(os.environ.get("OPENAI_API_KEY"))
    return KEY_LABELS[2] if ambient else KEY_LABELS[3]


def cmd_keys(args) -> int:
    import keyring.errors
    provider = args.provider
    if args.action == "set":
        secret = getpass.getpass(f"Paste your {provider} API key (input is hidden): ").strip()
        if not secret:
            print("No key entered; nothing stored.")
            return 1
        try:
            keys.set_key(provider, secret)
        except keyring.errors.KeyringError as exc:
            raise CliError(f"The OS keyring is not available ({type(exc).__name__}). Put the key in a mode-600 "
                           f'JSON file ({{"{provider}": "..."}}) and set WARDZERO_KEY_FILE to its path.', 1) from None
        print(f"Stored the {provider} key in the keyring (service {keys.SERVICE}).")
        return 0
    if args.action == "clear":
        keys.clear_key(provider)
        print(f"Removed the {provider} key from the keyring.")
        return 0
    label = key_source(provider)
    print(f"{provider}: {label}")
    return 0 if label != KEY_LABELS[3] else 1


# ---- audit ---------------------------------------------------------------------------------
def cmd_audit(args) -> int:
    _need_ros()
    import rclpy

    from .ros.fairness import live_audit

    rclpy.init()
    node = rclpy.create_node("wardzero_audit")
    try:
        end = time.monotonic() + args.wait
        while time.monotonic() < end:  # let discovery see the graph
            rclpy.spin_once(node, timeout_sec=0.1)
        names = node.get_node_names()
        violations = live_audit(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not violations:
        print(f"Fairness audit: no violations ({len(names)} nodes seen, "
              f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}).")
        return 0
    for v in violations:
        print(f"VIOLATION: {v['node']} subscribes to {v['topic']}")
    return 1


# ---- parser --------------------------------------------------------------------------------
def _record_res(text: str) -> str:
    from .isaac_host.recording import parse_res
    try:
        w, h = parse_res(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return f"{w}x{h}"


def _live_flags() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sim", choices=SIMS, default=None,
                   help="isaac (the Isaac Sim host; the default when Isaac Sim is installed) or stub "
                        "(the kinematic stand-in, for tests)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--headed", dest="headed", action="store_true", default=None,
                      help="Isaac: show the Kit window to watch the run")
    mode.add_argument("--headless", dest="headed", action="store_false", help="Isaac: no window (the default)")
    p.add_argument("--record", action="store_true",
                   help="Isaac: record the run to <run>/video and the robot domain to <run>/robot_bag (lockstep)")
    p.add_argument("--bag-images", action="store_true",
                   help="Isaac --record: also put the robot's rgb, depth and point cloud in the rosbag (large)")
    p.add_argument("--no-voice", action="store_true",
                   help="Isaac --record: no spoken voices in the recorded videos (voices are added after the run, on the CPU)")
    p.add_argument("--record-res", default="1280x720", metavar="WxH", type=_record_res,
                   help="Isaac: resolution of the recorded videos (1920x1080 for a film)")
    p.add_argument("--preview", action="store_true",
                   help="Isaac, headless: a live overhead preview for the Studio (<run>/preview/latest.jpg)")
    p.add_argument("--full-site", action="store_true",
                   help="Isaac: load the whole hospital instead of the scenario's areas (needs a large GPU)")
    p.add_argument("--time-scale", type=float, default=1.0, help="sim seconds per wall second")
    p.add_argument("--lockstep", action="store_true", help="hold the sim clock while a decision is pending")
    p.add_argument("--no-robot", action="store_true", help="run the scenario without the robot")
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--panel-port", type=int, default=DEFAULT_PANEL_PORT)
    p.add_argument("--bot", choices=("claude", "reception", "none"), default=None,
                   help="the robot to run: claude (the built-in Claude robot; the default when a Claude key is "
                        "set), reception (the placeholder) or none")
    p.add_argument("--robot-domain", type=int, default=None, metavar="N",
                   help="ROS_DOMAIN_ID of the developer's robot (Isaac: /clock is published there too; "
                        "the robot graph follows)")
    p.add_argument("--project-id", default=None, help=argparse.SUPPRESS)  # set by the Studio
    p.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S,
                   help="wall seconds the sim waits for the agents and director to come up")
    return p


def cmd_studio(args) -> int:
    from .studio.server import serve
    return serve(args.port, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hhumans", description="WardZero persona agents: simulated hospital visitors.")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    live = _live_flags()

    r = sub.add_parser("run", parents=[live], help="run a scenario")
    r.add_argument("scenario")
    r.add_argument("--repeat", type=int, default=1, metavar="N", help="run N seeds back to back as one batch")
    r.add_argument("--carry-memory", action="store_true",
                   help="in a batch, carry memory from run to run (default: every repeat starts from the same snapshot)")
    r.add_argument("--baseline", action="store_true",
                   help="also run the same seeds with the built-in robot from the same start memory")
    r.add_argument("--no-report", action="store_true", help="skip the AI-written report after the run")
    r.set_defaults(func=cmd_run)

    db = sub.add_parser("debrief", help="ask the people about the robot and update their trust")
    db.add_argument("run_dir")
    db.add_argument("--force", action="store_true", help="debrief again, applying trust a second time")
    db.set_defaults(func=cmd_debrief)

    vo = sub.add_parser("voice", help="speak a recorded run's lines: audio/voice.wav and a voice track in its videos")
    vo.add_argument("run_dir")
    vo.add_argument("--models-dir", default=None, help="folder of Piper voice models (default <data>/voices)")
    vo.set_defaults(func=cmd_voice)

    rpt = sub.add_parser("report", help="write the report of a run, or of a batch folder")
    rpt.add_argument("run_dir", metavar="RUN_DIR|BATCH_DIR")
    rpt.add_argument("--no-narrative", action="store_true", help="no AI text, only the figures")
    rpt.set_defaults(func=cmd_report)

    rr = sub.add_parser("rerun", parents=[live], help="run a recorded run's scenario again from its start memory")
    rr.add_argument("run_dir")
    rr.set_defaults(func=cmd_rerun)

    rp = sub.add_parser("replay", help="play a recorded run back")
    rp.add_argument("run_dir")
    mode = rp.add_mutually_exclusive_group(required=True)
    mode.add_argument("--intents", action="store_true", help="re-send the intents to a live stub sim and robot")
    mode.add_argument("--exact", action="store_true", help="republish the recorded world; read-only panel")
    rp.add_argument("--time-scale", type=float, default=1.0)
    rp.add_argument("--no-robot", action="store_true")
    rp.add_argument("--bot", choices=("claude", "reception", "none"), default=None)
    rp.add_argument("--no-panel", action="store_true")
    rp.add_argument("--panel-port", type=int, default=DEFAULT_PANEL_PORT)
    rp.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S)
    rp.set_defaults(func=cmd_replay)

    m = sub.add_parser("memory", help="long-term memory: reset and snapshots")
    msub = m.add_subparsers(dest="memory_command", metavar="ACTION")
    reset = msub.add_parser("reset", help="reset people to their seed memories")
    who = reset.add_mutually_exclusive_group(required=True)
    who.add_argument("--person", help="a person id in --scenario (e.g. P2)")
    who.add_argument("--persona", help="everyone with this persona in --scenario")
    who.add_argument("--all", action="store_true", help="everyone in the memory db")
    reset.add_argument("--scenario")
    reset.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    reset.add_argument("--embedder", default=os.environ.get("WARDZERO_EMBEDDER", "hash"))
    reset.set_defaults(func=cmd_memory_reset)
    snap = msub.add_parser("snapshot", help="save, restore or list memory snapshots")
    snap.add_argument("action", choices=("save", "restore", "list"))
    snap.add_argument("name", nargs="?")
    snap.set_defaults(func=cmd_snapshot)

    k = sub.add_parser("kb", help="knowledge base")
    ksub = k.add_subparsers(dest="kb_command", metavar="ACTION")
    kb = ksub.add_parser("build", help="build the knowledge base index")
    kb.add_argument("--embedder", choices=("hash", "fastembed"), default="hash")
    kb.add_argument("--scenario")
    kb.set_defaults(func=cmd_kb_build)

    ky = sub.add_parser("keys", help="API keys (stored in the OS keyring)")
    ky.add_argument("action", choices=("set", "clear", "status"))
    ky.add_argument("provider", choices=("claude", "openai"))
    ky.set_defaults(func=cmd_keys)

    st = sub.add_parser("studio", help="open the Scenario Studio in the browser")
    st.add_argument("--port", type=int, default=8770)
    st.add_argument("--no-browser", action="store_true")
    st.set_defaults(func=cmd_studio)

    a = sub.add_parser("audit", help="run the robot fairness audit on the live ROS graph once")
    a.add_argument("--wait", type=float, default=2.0, help="seconds of discovery before the audit")
    a.set_defaults(func=cmd_audit)
    return p


def main(argv=None) -> int:
    keys.install_log_redaction()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help is 0, usage errors 2
        return exc.code if isinstance(exc.code, int) else 2
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return func(args)
    except CliError as exc:
        if str(exc):
            err(str(exc))
        return exc.code
    except KeyboardInterrupt:
        err("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
