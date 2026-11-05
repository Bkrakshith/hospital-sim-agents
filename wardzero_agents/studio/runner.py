"""Run control for the Studio: start `hhumans run` as a subprocess and read its run folder for live state."""
from __future__ import annotations

import collections
import json
import os
import re
import secrets
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
from .project import Project, ProjectError

TAIL_STREAMS = ("status", "says", "audio", "tokens", "robot_pose", "objects")
SAYS_TAIL = 20
RUN_FOLDER = re.compile(r"^Run folder: (.+)$")
STOP_GRACE_S = 10.0  # after the stop message the run must have begun its orderly stop, else SIGINT
STOP_HARD_S = 120.0  # once it has begun, it gets this long in all (agents drain for up to 60 s) before SIGINT
STOP_MARKER = "Stopping the run"  # the line the CLI prints when its orderly stop begins
AGENTS_ROOT = Path(__file__).resolve().parents[2]
KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE")


class RunnerBusy(Exception):
    pass


PREVIEW_FILE = Path("preview") / "latest.jpg"  # the Isaac host's 5 fps overhead frame, in the run folder
PREVIEW_TOKEN_BYTES = 24  # 192 bits


def studio_sim(project: Project) -> str:
    """The project's simulator; $WARDZERO_STUDIO_SIM (isaac or stub) overrides it."""
    env = os.environ.get("WARDZERO_STUDIO_SIM", "").strip()
    return env if env in ("isaac", "stub") else project.sim


def default_command(project: Project, staged: Path, bot: str, robot_domain: Optional[int]) -> list[str]:
    sim = studio_sim(project)
    argv = [sys.executable, "-m", "wardzero_agents.cli", "run", str(staged), "--sim", sim, "--bot", bot,
            "--no-panel", "--project-id", project.id]
    if sim == "isaac":
        # a window is its own live view; headless gets the preview the Run page streams
        argv += ["--headed"] if project.headed else ["--headless", "--preview"]
        if project.record:
            argv += ["--record", "--record-res", project.record_res]
            if not project.voice:
                argv.append("--no-voice")
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
                 stop_publisher: Optional[Callable[[str], bool]] = None,
                 stop_grace_s: float = STOP_GRACE_S, stop_hard_s: float = STOP_HARD_S,
                 key_present: Optional[Callable[[], bool]] = None):
        self.data, self.config_root, self.runs_dir = Path(data), Path(config_root), Path(runs_dir)
        self._command = command
        self._stop_publisher = stop_publisher or publish_stop
        self._grace, self._hard = stop_grace_s, stop_hard_s
        self._hard_timer_armed = False
        self._stop_requested = False
        self._timer: Optional[threading.Timer] = None
        self._run_domain = os.environ.get("ROS_DOMAIN_ID", "0")
        self._key_present = key_present or _claude_key_present
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._log_fh = None
        self._run_id: Optional[str] = None
        self._project: Optional[Project] = None
        self._folder: Optional[str] = None
        self._log_pos, self._log_partial, self._finished, self._stopping = 0, b"", False, False
        self._log_lines: collections.deque = collections.deque(maxlen=8)
        self._diag: Optional[tuple] = None  # (the finished process, what _failure found)
        self._cost_before = 0.0  # what earlier run folders of this Start spent
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
            try:
                profile = robot_profile_for(project.robot_profile, self.data)
            except (KeyError, ValueError, OSError):
                raise ProjectError(f"Unknown robot '{project.robot_profile}'. Save it under Robots first.") from None
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
            self._cancel_timer()
            self._stop_requested = False
            self._log_fh = open(self._log_path, "wb")
            env = dict(os.environ, PYTHONUNBUFFERED="1", WARDZERO_RUNS_DIR=str(self.runs_dir),
                       WARDZERO_DATA_DIR=str(self.data))
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(AGENTS_ROOT), env.get("PYTHONPATH", "")) if p)
            self._run_domain = env.get("ROS_DOMAIN_ID", "0")
            self._proc = subprocess.Popen(argv, stdout=self._log_fh, stderr=subprocess.STDOUT, env=env,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            self._run_id = "pending-" + uuid.uuid4().hex[:8]
            self._project = project
            self._reset_view()
            self._cost_before = 0.0
            self._folder = None
            self._log_pos, self._log_partial, self._finished, self._stopping = 0, b"", False, False
            self._log_lines = collections.deque(maxlen=8)
            return self._run_id

    def stop(self) -> dict:
        """Stop the run. The stop goes out on the scenario control topic, like the panel's stop; if nobody
        took it, the run process gets SIGINT. A stop during the debrief or the report is refused politely:
        both finish by themselves in moments, and cutting them off would lose the trust update.

        After a control stop the CLI prints a marker line as soon as it begins its orderly stop. The run
        is interrupted only if that line has not appeared within the grace period (it never began
        stopping), or if it is still running long after (the stop itself hangs). Once it has begun
        stopping, the agents may be draining their memory for a minute: no signal in between."""
        with self._lock:
            if not self._alive():
                return {"ok": True, "stopped": False, "note": "No run is going."}
            self._read_log()
            if self._phase() in ("debrief", "report"):
                return {"ok": True, "stopped": False,
                        "note": "The run is already over; the debrief and report are finishing."}
            if self._stop_requested or self._stopping:  # a double click or retry: never signal a draining run
                return {"ok": True, "stopped": True, "via": "control", "note": "already stopping"}
            self._stop_requested = True
            domain = self._run_domain
        sent = False
        try:
            sent = bool(self._stop_publisher(domain))  # True only when a subscriber got it; not under the lock
        except Exception:
            sent = False
        with self._lock:
            if not sent:
                self._signal(signal.SIGINT)
                return {"ok": True, "stopped": True, "via": "sigint"}
            self._hard_timer_armed = False
            self._arm_timer(self._grace)
            return {"ok": True, "stopped": True, "via": "control"}

    def _arm_timer(self, seconds: float) -> None:
        self._cancel_timer()
        self._timer = threading.Timer(seconds, self._escalate)
        self._timer.daemon = True
        self._timer.start()

    def _escalate(self) -> None:
        with self._lock:
            self._read_log()
            if not self._alive() or self._phase() != "running":
                return  # over, or already debriefing: leave it alone
            if self._stopping and not self._hard_timer_armed:
                self._hard_timer_armed = True  # it began stopping: wait out the drain, then one last look
                self._arm_timer(max(0.0, self._hard - self._grace))
                return
            self._signal(signal.SIGINT)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def shutdown(self) -> None:
        """The Studio is closing: do not leave a paid run behind, but never cut a debrief short."""
        self._cancel_timer()
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
        self._preview_token: Optional[str] = None  # per run; it is the only way to the preview stream
        self._people: dict[str, dict] = {}
        self._says: collections.deque = collections.deque(maxlen=SAYS_TAIL)
        self._tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._cost = 0.0  # this run folder's
        self._robot: dict = {}
        self._objects: list = []
        self._t = 0.0

    def _read_log(self) -> None:
        """Read what the run process printed since the last call: the run folder it names, whether the
        run has finished, and the last lines (for an error message)."""
        if self._log_path is None:
            return
        try:
            with open(self._log_path, "rb") as f:
                f.seek(self._log_pos)
                data = f.read()
        except OSError:
            return
        self._log_pos += len(data)
        data = self._log_partial + data
        end = data.rfind(b"\n")
        if end < 0:
            self._log_partial = data
            return
        self._log_partial = data[end + 1:]
        for raw in data[:end].splitlines():
            line = raw.decode(errors="replace")
            self._log_lines.append(line)
            m = RUN_FOLDER.match(line)
            if m:
                self._folder, self._finished = m.group(1).strip(), False
            elif line.startswith(STOP_MARKER):
                self._stopping = True
            elif line.startswith("Run finished"):
                self._finished = True

    def _follow(self) -> None:
        """Point the tail at the newest run folder named in the log and read what was appended."""
        if self._folder and Path(self._folder) != self._dir:
            keep = self._project
            if self._tail is not None:  # the last of this folder (its debrief and report) counts before the next
                self._consume(self._tail.poll())
            self._cost_before += self._cost  # one Start can be many runs: the cost bar and the cap are for all
            self._reset_view()
            self._project = keep
            self._dir = Path(self._folder)
            self._tail = RunTail(self._dir, TAIL_STREAMS)
            self._preview_token = secrets.token_urlsafe(PREVIEW_TOKEN_BYTES)
        if self._tail is None:
            return
        self._consume(self._tail.poll())

    def _consume(self, records) -> None:
        for stream, rec in records:
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

    def _phase(self) -> str:
        if self._proc is None:
            return "idle"
        if not self._alive():
            return "done"
        if not self._finished:
            return "running"
        if self._dir is not None and (self._dir / "trust.jsonl").exists():
            return "report"
        return "debrief"

    def _preview_path(self) -> Optional[Path]:
        """The preview frame of the live run, or None: no run folder yet, the run is over (or debriefing), or the host
        writes no preview."""
        if self._dir is None or self._preview_token is None or self._phase() != "running":
            return None
        f = self._dir / PREVIEW_FILE
        return f if f.is_file() else None

    def preview_file(self, token: str) -> Optional[Path]:
        """The frame file for this token, only while the run it belongs to is running. Constant-time comparison."""
        with self._lock:
            self._read_log()
            ok = self._preview_token is not None and secrets.compare_digest(
                token.encode("utf-8", "replace"), self._preview_token.encode())
            return self._preview_path() if ok else None

    def state(self) -> dict:
        with self._lock:
            self._read_log()
            self._follow()
            phase = self._phase()
            rd = self._dir
            out = {
                "running": self._alive(),
                "run_id": rd.name if rd is not None else self._run_id,
                "project_id": self._project.id if self._project else None,
                "t": round(self._t, 2), "cost_usd": round(self._cost_before + self._cost, 6),
                "budget_usd": self._project.budget_usd if self._project else None,
                "tokens": dict(self._tokens), "people": list(self._people.values()),
                "robot": dict(self._robot), "objects": self._objects, "says_tail": list(self._says),
                "phase": phase,
                "has_report": bool(rd is not None and (rd / "report.html").is_file()),
            }
            if self._preview_path() is not None:
                out["preview_url"] = f"/preview/{self._preview_token}.mjpg"
            if phase == "done" and self._proc is not None:
                out["exit_code"] = self._proc.returncode
                if self._proc.returncode not in (0, None, 130):  # 130 is a stop by the user; anything else failed
                    out.update(self._failure())
            return out

    def _failure(self) -> dict:
        """Why the finished run failed and how to fix it, read once from the whole run log (the last lines are often
        only the other processes being stopped)."""
        if self._diag is not None and self._diag[0] is self._proc:
            return self._diag[1]
        from ..diagnose import diagnose, format_problems, read_log
        from ..keys import redact
        rc = self._proc.returncode
        lines = read_log(self._log_path) if self._log_path is not None else list(self._log_lines)
        found = diagnose(lines, rc, self._log_path)
        text = f"The run stopped with exit code {rc}.\n\n" + redact(format_problems(found))
        if any(p.key == "unknown" for p in found):
            tail = redact("\n".join(list(self._log_lines)[-6:]))[-800:]
            if tail:
                text += "\n\nLast lines of the log:\n" + tail
        info = {"error": text,
                "problems": [{"title": redact(p.title), "fix": [redact(f) for f in p.fix]} for p in found],
                "log_path": str(self._log_path) if self._log_path is not None else None}
        self._diag = (self._proc, info)
        return info

    def busy(self) -> bool:
        """A run is going: starting, running, in its debrief, voice step or report. Resets, project deletes and
        character edits wait until it is over."""
        with self._lock:
            self._read_log()
            return self._phase() not in ("idle", "done")


def _claude_key_present() -> bool:
    from ..cli import _claude_key_present as present
    return present()


def publish_stop(domain: str = "0") -> bool:
    """Publish Control(stop) on the scenario control topic, on the run's ROS domain, from a short ROS
    process. True only when that process saw a subscriber and published; False when ROS is not available,
    nobody listens or the call failed, so the caller falls back to SIGINT."""
    from ..cli import ros_available
    if not ros_available():
        return False
    try:
        r = subprocess.run([sys.executable, "-m", "wardzero_agents.studio.ros_probe", "stop"], timeout=8,
                           capture_output=True,
                           env={**{k: v for k, v in os.environ.items() if k not in KEY_ENV},
                                "PYTHONPATH": str(AGENTS_ROOT), "ROS_DOMAIN_ID": str(domain)})
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0
