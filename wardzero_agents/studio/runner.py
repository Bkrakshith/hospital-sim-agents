"""Run control for the Studio: start `hhumans run` as a subprocess and read its run folder for live state."""
from __future__ import annotations

import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..runlog import RunTail
from .compile import compile_project, robot_profile_for
from .project import Project

TAIL_STREAMS = ("status", "says", "audio", "tokens", "robot_pose", "objects")
SAYS_TAIL = 20
RUN_FOLDER = re.compile(r"^Run folder: (.+)$", re.M)
AGENTS_ROOT = Path(__file__).resolve().parents[2]


class RunnerBusy(Exception):
    pass


def default_command(project: Project, staged: Path, bot: str, robot_domain: Optional[int]) -> list[str]:
    argv = [sys.executable, "-m", "wardzero_agents.cli", "run", str(staged), "--sim", "stub", "--bot", bot,
            "--no-panel", "--project-id", project.id]
    if robot_domain is not None:
        argv += ["--robot-domain", str(robot_domain)]
    if project.repeat > 1 or project.baseline:
        argv += ["--repeat", str(project.repeat)]
        if project.carry_memory:
            argv.append("--carry-memory")
        if project.baseline:
            argv.append("--baseline")
    return argv


class Runner:
    def __init__(self, data: Path, config_root: Path, runs_dir: Path, *,
                 command: Callable[..., list[str]] = default_command,
                 stop_publisher: Optional[Callable[[], bool]] = None,
                 key_present: Optional[Callable[[], bool]] = None):
        self.data, self.config_root, self.runs_dir = Path(data), Path(config_root), Path(runs_dir)
        self._command = command
        self._stop_publisher = stop_publisher or publish_stop
        self._key_present = key_present or _claude_key_present
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._log_fh = None
        self._run_id: Optional[str] = None
        self._project: Optional[Project] = None
        self._reset_view()

    # ---- control ---------------------------------------------------------------------------
    def start(self, project: Project, robot_domain: Optional[int] = None) -> str:
        from ..memory.runlock import active_run
        with self._lock:
            if self._alive():
                raise RunnerBusy("A run is already going. Stop it first.")
            other = active_run(self.data)
            if other is not None:
                raise RunnerBusy(f"Run {other} is active; stop it first.")
            profile = robot_profile_for(project.robot_profile, self.data)
            staged = compile_project(project, self.config_root, self.data / "studio" / "projects" / project.id
                                     / "staged", self.data)
            if profile.controller == "builtin_claude":
                bot = "claude" if self._key_present() else "reception"
            else:
                bot = "none"  # the developer's own robot (external), or no robot
            argv = self._command(project, staged, bot, robot_domain)
            log_dir = self.data / "studio" / "projects" / project.id
            self._log_path = log_dir / "last_run.log"
            self._close_log()
            self._log_fh = open(self._log_path, "wb")
            env = dict(os.environ, PYTHONUNBUFFERED="1", WARDZERO_RUNS_DIR=str(self.runs_dir),
                       WARDZERO_DATA_DIR=str(self.data))
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(AGENTS_ROOT), env.get("PYTHONPATH", "")) if p)
            self._proc = subprocess.Popen(argv, stdout=self._log_fh, stderr=subprocess.STDOUT, env=env,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            self._run_id = "pending-" + uuid.uuid4().hex[:8]
            self._project = project
            self._reset_view()
            return self._run_id

    def stop(self) -> dict:
        """Stop the run. The stop goes out on the scenario control topic, like the panel's stop; if that
        cannot be sent, the run process gets SIGINT. A stop during the debrief or the report is refused
        politely: both finish by themselves in moments, and cutting them off would lose the trust update."""
        with self._lock:
            if not self._alive():
                return {"ok": True, "stopped": False, "note": "No run is going."}
            phase = self._phase()
            if phase in ("debrief", "report"):
                return {"ok": True, "stopped": False,
                        "note": "The run is already over; the debrief and report are finishing."}
            sent = False
            try:
                sent = bool(self._stop_publisher())
            except Exception:
                sent = False
            if not sent:
                self._signal(signal.SIGINT)
            return {"ok": True, "stopped": True, "via": "control" if sent else "sigint"}

    def shutdown(self) -> None:
        """The Studio is closing: do not leave a paid run behind, but never cut a debrief short."""
        with self._lock:
            if self._alive() and self._phase() == "running":
                self._signal(signal.SIGINT)
        deadline = time.monotonic() + 15
        while self._alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        self._close_log()

    def _signal(self, sig: int) -> None:
        try:
            self._proc.send_signal(sig)
        except (OSError, AttributeError, ProcessLookupError):
            pass

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    # ---- live state ------------------------------------------------------------------------
    def _reset_view(self) -> None:
        self._dir: Optional[Path] = None
        self._tail: Optional[RunTail] = None
        self._people: dict[str, dict] = {}
        self._says: collections.deque = collections.deque(maxlen=SAYS_TAIL)
        self._tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._cost = 0.0
        self._robot: dict = {}
        self._objects: list = []
        self._t = 0.0

    def _log_text(self) -> str:
        try:
            return self._log_path.read_text(errors="replace") if self._log_path else ""
        except OSError:
            return ""

    def _follow(self, log: str) -> None:
        """Point the tail at the newest run folder named in the log and read what was appended."""
        found = RUN_FOLDER.findall(log)
        if found and Path(found[-1].strip()) != self._dir:
            keep = self._project
            self._reset_view()
            self._project = keep
            self._dir = Path(found[-1].strip())
            self._tail = RunTail(self._dir, TAIL_STREAMS)
        if self._tail is None:
            return
        for stream, rec in self._tail.poll():
            if stream == "status" and isinstance(rec.get("person"), str):
                self._people[rec["person"]] = {k: rec.get(k) for k in ("x", "y", "yaw", "posture", "state")}
                self._people[rec["person"]]["id"] = rec["person"]
                self._t = max(self._t, float(rec.get("t") or 0.0))
            elif stream == "says":
                self._says.append({"t": rec.get("t"), "person": rec.get("person"), "text": rec.get("text"),
                                   "addressee": rec.get("addressee")})
            elif stream == "audio" and rec.get("source_kind") == "robot" and rec.get("text"):
                self._says.append({"t": rec.get("t"), "person": "robot", "text": rec.get("text"),
                                   "addressee": rec.get("addressee")})
            elif stream == "tokens":
                self._cost += float(rec.get("cost_usd") or 0.0)
                for k, f in (("input", "input_tokens"), ("output", "output_tokens"),
                             ("cache_read", "cache_read_tokens"), ("cache_write", "cache_write_tokens")):
                    self._tokens[k] += int(rec.get(f) or 0)
            elif stream == "robot_pose":
                self._robot = {k: rec.get(k) for k in ("x", "y", "yaw", "speed")}
                self._t = max(self._t, float(rec.get("t") or 0.0))
            elif stream == "objects":
                objs = rec.get("objects")
                if isinstance(objs, (list, dict)):
                    self._objects = objs

    def _phase(self, log: Optional[str] = None) -> str:
        if self._proc is None:
            return "idle"
        log = self._log_text() if log is None else log
        if not self._alive():
            return "done"
        m = list(RUN_FOLDER.finditer(log))
        after = log[m[-1].end():] if m else log
        if "Run finished" not in after:
            return "running"
        if self._dir is not None and (self._dir / "trust.jsonl").exists():
            return "report"
        return "debrief"

    def state(self) -> dict:
        with self._lock:
            log = self._log_text()
            self._follow(log)
            phase = self._phase(log)
            rd = self._dir
            out = {
                "running": self._alive(),
                "run_id": rd.name if rd is not None else self._run_id,
                "project_id": self._project.id if self._project else None,
                "t": round(self._t, 2), "cost_usd": round(self._cost, 6),
                "budget_usd": self._project.budget_usd if self._project else None,
                "tokens": dict(self._tokens), "people": list(self._people.values()),
                "robot": dict(self._robot), "objects": self._objects, "says_tail": list(self._says),
                "phase": phase,
                "has_report": bool(rd is not None and (rd / "report.html").is_file()),
            }
            if phase == "done" and self._proc is not None:
                out["exit_code"] = self._proc.returncode
                if self._proc.returncode not in (0, None) and rd is None:
                    out["error"] = "\n".join(log.strip().splitlines()[-6:])[-800:]
            return out


def _claude_key_present() -> bool:
    from ..cli import _claude_key_present as present
    return present()


def publish_stop() -> bool:
    """Publish Control(stop) on the scenario control topic from a short ROS process. False when ROS is not
    available or the call failed, so the caller falls back to SIGINT."""
    from ..cli import ros_available
    if not ros_available():
        return False
    try:
        r = subprocess.run([sys.executable, "-m", "wardzero_agents.studio.ros_probe", "stop"], timeout=6,
                           capture_output=True, env=dict(os.environ, PYTHONPATH=str(AGENTS_ROOT)))
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0
