"""Why a run failed, in plain words, and what to do about it.

A run is many processes (the Isaac host or the stub sim, the people, the director, the robot). When one fails the
others are stopped, and their stop noise (a KeyboardInterrupt traceback, "There was an error running python") lands
after the line that matters. ``diagnose`` reads the whole run log, finds the lines that say what went wrong, and
turns each into a ``Problem``: a title and the steps that fix it. The Studio shows these instead of the log's last
lines, and ``hhumans diagnose`` prints them for a log file.

Every rule here is matched against text the code prints itself (or Isaac/Kit prints), so the catalogue stays in
step with the messages: tests/test_diagnose.py feeds each rule the real message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .isaac_assets import BUILD_SCRIPT, COLLISION_ENV, STAGE_ENV

KIT_LOG_TAIL_BYTES = 2_000_000
TITLE_MAX = 300


@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    fix: tuple[str, ...]
    line: str = ""  # the log line that showed it (redacted by the caller before display)


@dataclass(frozen=True)
class Rule:
    key: str
    pattern: re.Pattern
    title: str  # str.format(**match.groupdict())
    fix: tuple[str, ...]
    in_kit_log: bool = False  # also looked for in Isaac's own log (only rules that cannot fire on routine Kit noise)


def _r(key: str, pattern: str, title: str, *fix: str, kit: bool = False, flags: int = 0) -> Rule:
    return Rule(key, re.compile(pattern, flags), title, tuple(fix), kit)


SWITCH_TO_2D = "Or choose Quick 2D on the Run page: it needs no Isaac Sim and shows the same people."
RESTART_STUDIO = ("Then stop the Studio (Ctrl+C in its terminal) and start it again from that terminal: "
                  "`hhumans studio`.")
PROVIDER_FIX = {
    "auth": ("The AI provider refused the key.",
             "Store a working key: `hhumans keys set claude` (or `ant auth login`), and check it with "
             "`hhumans keys status claude`."),
    "bad_request": ("The AI provider refused the requests as invalid. The usual cause is an account with no "
                    "credit left.",
                    "Check the credit balance and spend limit at console.anthropic.com (Plans & Billing), top up, "
                    "then start again."),
    "model_not_found": ("A model named in the scenario is not available to this account.",
                        "Use a model the account has (the scenario's `models:` section), or enable it in the Console."),
    "too_large": ("A request was too large for the model.",
                  "Shorten the persona or role prompts in the Studio's character editor, then start again."),
    "rate_limited": ("The AI provider is rate-limiting this account.",
                     "Wait a minute and start again, or lower the number of people in the scene. Rate limits are "
                     "raised in the Console as the account's usage tier grows."),
    "provider_error": ("The AI provider had an outage or an overload.",
                       "Check status.anthropic.com and start again when it is green."),
}

RULES: tuple[Rule, ...] = (
    # ---- the robot -------------------------------------------------------------------------------
    _r("robot_no_urdf", r"robot '(?P<robot>[^']+)' has no URDF",
       "The robot '{robot}' has no URDF, so Isaac Sim cannot build it.",
       "On the Robot page, pick the built-in G1, or give your robot's URDF folder and press Check the URDF.",
       SWITCH_TO_2D),
    _r("robot_error", r"^Robot error: (?P<msg>.+)",
       "Isaac Sim could not prepare the robot: {msg}",
       "On the Robot page press Check the URDF and fix what it lists (missing meshes, no base link), then start again.",
       "If you changed the URDF, the robot is converted again on the next start; that takes a minute or two.",
       SWITCH_TO_2D),
    _r("robot_config", r"Robot config error: (?P<msg>.+)",
       "The robot's settings are not valid: {msg}",
       "Open the Robot page, correct the field it names and save, then start again."),
    # ---- scenario and files ------------------------------------------------------------------------
    _r("scenario", r"Scenario error: (?P<msg>.+)",
       "The scenario has a mistake: {msg}",
       "In the Studio, open the Scene page and save the project again (it rewrites the scenario).",
       "If you wrote the YAML by hand, fix the field the message names."),
    _r("stage_missing", r"Stage not found: (?P<path>\S+)|the hospital stage is missing: (?P<path2>\S+?)\.?(?:\s|$)",
       "The hospital stage (USD) was not found.",
       "It comes with the hospital-sim-env checkout next to hospital-sim-agents. Check that checkout is there and "
       "complete.",
       f"If it lives elsewhere, set {STAGE_ENV}=/path/to/hospital_g1.usd before starting the Studio."),
    _r("collision_missing", r"collision USD (?:\S+ )?is missing",
       "The collision USD (the walls people and robots bump into) is missing.",
       f"Build it once: `~/isaacsim/python.sh {BUILD_SCRIPT}` from the Hospital_G1 folder. It takes a few minutes.",
       f"If it lives elsewhere, set {COLLISION_ENV}=/path/to/hospital_collision.usd before starting the Studio."),
    _r("isaac_not_found", r"Isaac Sim (?:was )?not found|Isaac Sim's python\.sh is not in|no python\.sh\)",
       "Isaac Sim was not found on this computer.",
       "Install Isaac Sim 6.0 in ~/isaacsim, or set ISAAC_SIM_ROOT to where it is installed, then restart the Studio.",
       SWITCH_TO_2D),
    # ---- keys and the AI provider -------------------------------------------------------------------
    _r("key_missing", r"No working (?P<provider>Claude|OpenAI) key",
       "There is no working {provider} key on this computer.",
       "Store one: `hhumans keys set claude` (or `ant auth login`). Check with `hhumans keys status claude`.",
       "A Claude Pro or Max subscription is not an API key: the key comes from console.anthropic.com.",
       "To try the Studio without a key, start it with `WARDZERO_TEST_FAKE_LLM=1 hhumans studio` (nothing is billed)."),
    _r("model_unavailable", r"Model (?P<model>\S+) is not available to this",
       "The model {model} is not available to this account.",
       "Use a model the account has (the scenario's `models:` section), or enable it in the Console."),
    _r("key_check_failed", r"key check failed \((?P<exc>\w+)\)",
       "The key could not be checked: the AI provider did not answer ({exc}).",
       "Check this computer's internet connection and status.anthropic.com, then start again."),
    _r("provider", r"AI provider (?:is )?reject(?:ing|ed) (?:a request|requests|the robot's requests)\S*:? \(?"
                   r"(?P<kind>auth|bad_request|model_not_found|too_large|rate_limited|provider_error)",
       "{kind}", ""),  # title and fix come from PROVIDER_FIX
    _r("keyring", r"OS keyring is not available",
       "The OS keyring is not available, so the key cannot be read.",
       "On a desktop, unlock the login keyring. On a server, put the key in a file (mode 600) and set "
       "WARDZERO_KEY_FILE to its path."),
    _r("brotli", r"[Bb]rotli.*(?:version|1\.0\.9|has no attribute)",
       "The installed Brotli library is too old for the AI provider's client.",
       "Run `pip install \"Brotli>=1.2\"` with the venv active (`source env.sh`), then start again."),
    # ---- the computer ------------------------------------------------------------------------------
    _r("glibcxx", r"GLIBCXX_[\d.]+'? not found|libstdc\+\+\.so\.6: version",
       "A conda library is shadowing the system one (libstdc++), so Isaac Sim or ROS cannot load.",
       "In the terminal you start the Studio from, run `conda deactivate` and `source ~/ntu_dev/Hospital_G1/env.sh`.",
       RESTART_STUDIO, kit=True),
    _r("ros_missing", r"ROS 2 is not sourced|^ModuleNotFoundError: No module named 'rclpy'",
       "ROS 2 is not set up in the terminal the Studio was started from.",
       "Run `source ~/ntu_dev/Hospital_G1/env.sh` in that terminal.", RESTART_STUDIO),
    _r("module_missing", r"^ModuleNotFoundError: No module named '(?P<mod>(?!rclpy)[\w.]+)'",
       "A Python package is missing: {mod}.",
       "Run `source ~/ntu_dev/Hospital_G1/env.sh`, then `pip install -e .` in hospital-sim-agents.", RESTART_STUDIO),
    _r("gpu_oom", r"ERROR_OUT_OF_DEVICE_MEMORY|CUDA(?: error)?:? out of memory|[Oo]ut of (?:GPU|device|video) memory",
       "The graphics card ran out of memory.",
       "Close other programs using the GPU, and any other Isaac Sim (`pgrep -x kit` lists them; stop each by its "
       "number with `kill`).",
       "On the Run page choose headless and a smaller recording size (1280x720), then start again.", kit=True),
    _r("flow_crash", r"'lastCommand' = 'FlowCreatePrim",
       "Isaac Sim crashed while adding the fire effect: the graphics card ran out of memory.",
       "Start again: the fire is now drawn as a simple stand-in when graphics memory is short.",
       "For the full fire, choose headless on the Run page (a window costs graphics memory) and close other GPU "
       "programs.",
       "To always use the simple fire, start the Studio with `WARDZERO_FIRE=proxy hhumans studio`.", kit=True),
    _r("isaac_crash_cmd", r"'lastCommand' = '(?P<cmd>[A-Za-z_]+)",
       "Isaac Sim crashed (its last command was {cmd}).",
       "Graphics memory is the usual cause on an 8 GB card: choose headless and a smaller recording size, close other "
       "GPU programs, and start again.",
       "If it crashes at the same moment every time, keep the Isaac log named in the run log for a bug report.",
       kit=True),
    _r("isaac_crash", r"Segmentation fault|\[crash\] A crash has occurred",
       "Isaac Sim crashed.",
       "Graphics memory is the usual cause on an 8 GB card: choose headless and a smaller recording size, close other "
       "GPU programs, and start again.",
       "If it crashes at the same moment every time, keep the Isaac log named in the run log for a bug report."),
    _r("no_gpu", r"Failed to create any GPU devices|No device could be created|vkCreateInstance failed|"
                 r"NVIDIA driver .*(?:not|too old)",
       "Isaac Sim could not use the graphics card.",
       "Check the NVIDIA driver with `nvidia-smi`. If it fails, reinstall the driver and reboot.", kit=True),
    _r("display", r"(?:[Cc]annot|[Ff]ailed to) open (?:the )?display|Failed to create (?:the )?(?:GLFW )?window|"
                  r"GLFW error.*(?:display|X11|Wayland)",
       "Isaac Sim could not open a window (there is no desktop display here).",
       "On the Run page choose headless (the live preview still shows in the Studio), then start again.", kit=True),
    _r("ram_killed", r"exited with code (?:-9|137)\b|Out of memory: Killed process|oom-kill",
       "A run process was killed, most likely because the computer ran out of RAM.",
       "Close memory-hungry programs and start again. Isaac's limit is WARDZERO_ISAAC_MEMORY_MAX (40G by default)."),
    _r("port_in_use", r"[Pp]ort (?P<port>\d+) is already in use",
       "Port {port} is already in use.",
       "Close the other Studio or panel using it, or start with another port: `hhumans studio --port 8771`."),
    _r("run_active", r"[Rr]un (?P<other>\S+) is active; stop it first",
       "Another run ({other}) is still going, and only one can run at a time.",
       "Stop it on the Run page (or Ctrl+C in its terminal) and wait for its report, then start again."),
    _r("isaac_busy", r"[Aa]nother Isaac Sim is running",
       "Another Isaac Sim is already running and the graphics card cannot hold two.",
       "Close it, or list them with `pgrep -x kit` and stop each by its number with `kill`."),
    # ---- the run itself ---------------------------------------------------------------------------
    _r("gave_up_waiting", r"gave up waiting for (?P<who>[\w, ]+?) after",
       "The simulator waited for {who} to come up and gave up.",
       "The real reason is usually a key or ROS problem earlier in the log: fix that first.",
       "On a slow first start (Isaac compiling shaders), start again: the second start is much faster."),
    _r("agents_cannot_start", r"Cannot start the agents: (?P<msg>.+)",
       "The people could not start: {msg}",
       "If it names a memory snapshot, restore or pick an existing one (`hhumans memory snapshot list`).",
       "If it names an embedder, rebuild the knowledge base: `hhumans kb build`."),
    _r("stopped_on_its_own", r"(?P<who>\S+) stopped on its own before the scenario finished",
       "{who} stopped on its own before the scenario finished.",
       "The lines just above it in the full log say why."),
    _r("could_not_start", r"could not start (?P<who>\S+): (?P<msg>.+)",
       "The run could not start {who}: {msg}",
       "Check the program exists and is executable; for Isaac Sim, check ISAAC_SIM_ROOT."),
    _r("no_voices", r"No voices \(",
       "The recording has no voices (the run itself is fine).",
       "Install the voice models once: `pip install '.[voice]'` and `python scripts/fetch_voices.py`.",
       "Then add the voices to that run: `hhumans voice RUN_FOLDER`."),
    _r("no_report", r"The debrief or report failed|No debrief or report:",
       "The run finished but its debrief or report could not be written.",
       "Fix the key or network problem it names, then write the report again: `hhumans report RUN_FOLDER`."),
)

# a specific problem hides the general one it explains
SUPERSEDES = {"flow_crash": ("isaac_crash_cmd", "isaac_crash"), "isaac_crash_cmd": ("isaac_crash",),
              "gpu_oom": ("isaac_crash_cmd", "isaac_crash")}
AFTER_RUN = ("no_voices", "no_report", "gave_up_waiting", "stopped_on_its_own")

_EXITED = re.compile(r"(?P<who>\S+) exited with code (?P<code>-?\d+); stopping the run")
_KIT_LOG = re.compile(r"Logging to file: (?P<path>\S+\.log)")


def _title(rule: Rule, m: re.Match) -> tuple[str, tuple[str, ...]]:
    groups = {k: (v or "") for k, v in m.groupdict().items()}
    if rule.key == "provider":
        title, *fix = PROVIDER_FIX.get(groups["kind"], PROVIDER_FIX["provider_error"])
        return title, tuple(fix)
    title = rule.title.format(**groups).strip()
    if len(title) > TITLE_MAX:
        title = title[:TITLE_MAX - 1] + "…"
    return title, rule.fix


_LOGGED = re.compile(r"\s(?:DEBUG|INFO|WARNING)\s|\[(?:Info|Warning|Verbose)\]")
_EXC_END = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)\b")


def _without_logged_tracebacks(lines: Iterable[str]) -> Iterable[str]:
    """Drop tracebacks that a library logged at DEBUG, INFO or WARNING (Isaac's ROS bridge logs a harmless
    "No module named 'em'" one on every start): they are not why a run failed."""
    prev_logged = quiet = False
    for raw in lines:
        line = raw.rstrip()
        if quiet:
            if _EXC_END.match(line):
                quiet = False
            continue
        if prev_logged and line.startswith("Traceback (most recent call last)"):
            quiet = True
            continue
        prev_logged = bool(_LOGGED.search(line))
        yield line


def _match(lines: Iterable[str], rules: Iterable[Rule], seen: set[tuple[str, str]], out: list[Problem]) -> None:
    rules = tuple(rules)
    for line in _without_logged_tracebacks(lines):
        for rule in rules:
            m = rule.pattern.search(line)
            if m is None:
                continue
            title, fix = _title(rule, m)
            if (rule.key, title) not in seen:
                seen.add((rule.key, title))
                out.append(Problem(rule.key, title, fix, line.strip()))
            break  # one rule per line: the specific ones are listed before the general ones


def kit_log_path(lines: Iterable[str]) -> Optional[Path]:
    """The Isaac (Kit) log file the run log names, if any."""
    found = None
    for line in lines:
        m = _KIT_LOG.search(line)
        if m:
            found = Path(m.group("path"))
    return found


def _tail(path: Path, n: int = KIT_LOG_TAIL_BYTES) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def diagnose(lines: Iterable[str], exit_code: Optional[int] = None, log_path: Optional[Path] = None,
             read_kit_log: bool = True) -> list[Problem]:
    """The problems a run log shows, most likely cause first. With a failing ``exit_code`` and nothing recognised,
    one general problem that says where to look. An exit code of 0 or 130 (a stop by the user) adds nothing."""
    lines = list(lines)
    out: list[Problem] = []
    seen: set[tuple[str, str]] = set()
    _match(lines, RULES, seen, out)
    failed = exit_code not in (None, 0, 130)
    kit = kit_log_path(lines)
    if failed and read_kit_log and kit is not None:
        _match(_tail(kit), (r for r in RULES if r.in_kit_log), seen, out)
    hidden = {k for p in out for k in SUPERSEDES.get(p.key, ())}
    out = [p for p in out if p.key not in hidden]
    # after-run notes and knock-on effects go after a cause of the failure
    out.sort(key=lambda p: p.key in AFTER_RUN)
    if failed and not [p for p in out if p.key not in ("no_voices", "no_report")]:
        who = next((m.group("who") for m in map(_EXITED.search, lines) if m), None)
        fix = []
        if log_path is not None:
            fix.append(f"Read the full run log: {log_path}")
        if kit is not None:
            fix.append(f"Isaac Sim started; its own log is {kit} (look for [Error] near the end).")
        fix.append("Run `hhumans doctor` in a terminal: it checks Isaac Sim, the files, the key, ROS and the GPU.")
        what = f"{who} stopped with an error" if who else f"The run stopped with exit code {exit_code}"
        out.append(Problem("unknown", f"{what}, and the log does not say why in a way the Studio recognises.",
                           tuple(fix)))
    return out


def format_problems(problems: list[Problem], show_lines: bool = False) -> str:
    """Plain text: each problem, then its numbered fix steps."""
    parts = []
    for p in problems:
        block = [p.title]
        block += [f"  {i}. {step}" for i, step in enumerate(p.fix, 1)]
        if show_lines and p.line:
            block.append(f"  (log: {p.line[:240]})")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def read_log(path: Path, max_bytes: int = 4_000_000) -> list[str]:
    """The last ``max_bytes`` of a run log, as lines."""
    return _tail(Path(path), max_bytes)


__all__ = ["Problem", "RULES", "diagnose", "format_problems", "kit_log_path", "read_log"]
