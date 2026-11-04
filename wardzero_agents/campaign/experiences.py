"""What each person went through with the robot in one run, read from the run folder.

Pure: it reads robot_tasks.jsonl, says.jsonl, audio.jsonl, status.jsonl, world.jsonl and
objects.jsonl and returns Experience rows. The debrief applies them to the trust states."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from .trust import Experience

ROBOT = "robot"
GUIDE_WINDOW_S = 120.0  # the person must reach the named place within this long
FAST_CONTACT_MS = 0.5  # a robot this close (a contact) and faster than this is a failure
REPAIR_IMMEDIATE_S = 20.0  # an apology this soon after the error counts as immediate
REPAIR_REQUEST_S = 30.0  # a repair this soon after the person's next request counts as at the request
REQUEST_WITHIN_S = 90.0  # a request this soon before a robot action is the one it answers
_REPAIR_WORDS = re.compile(r"\b(sorry|apolog\w*|my mistake|my fault|because|this time|i promise|"
                           r"i should have|let me fix)\b", re.I)
_ROBOT_NAMES = {ROBOT, "g1"}


def _rows(run_dir: Path, name: str) -> list[dict]:
    try:
        text = (Path(run_dir) / f"{name}.jsonl").read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _item_kinds(run_dir: Path) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for row in _rows(run_dir, "objects"):
        for o in row.get("objects", []):
            if isinstance(o, dict) and "id" in o and "kind" in o:
                kinds[str(o["id"])] = str(o["kind"])
    return kinds


# ---- rules: each takes the run's streams and yields (t, person, easy, bad_outcome, kind) ----------
Hit = tuple  # (t, person, kind, easy, bad_outcome)


def _requester(says: list[dict], t: float) -> Optional[str]:
    """Whoever last spoke to the robot before t, within REQUEST_WITHIN_S."""
    best = None
    for s in says:
        if s.get("person") in _ROBOT_NAMES or s.get("addressee") not in _ROBOT_NAMES:
            continue
        if s["t"] <= t and t - s["t"] <= REQUEST_WITHIN_S and (best is None or s["t"] >= best["t"]):
            best = s
    return best["person"] if best else None


def rule_task_results(d: "_Run") -> list[Hit]:
    """A task ok is a success. A hand-over that failed is a failure; a tray out of reach is an
    easy task the person cannot do for themselves."""
    out: list[Hit] = []
    acts = {r["id"]: r for r in d.tasks if r.get("kind") == "action" and "id" in r}
    for r in d.tasks:
        if r.get("kind") != "result" or r.get("action") == "guide_start":
            continue
        if r.get("reason") in ("superseded",):
            continue
        act = acts.get(r.get("id"), {})
        who = act.get("person") or _requester(d.says, float(act.get("t", r.get("t", 0.0))))
        if not who:
            continue
        t = float(r.get("t", 0.0))
        if r.get("ok"):
            out.append((t, who, "success", False, False))
        else:
            kind = d.kinds.get(str(act.get("item")), "")
            easy = kind == "tray" or r.get("reason") == "out_of_reach_receiver"
            out.append((t, who, "failure", easy, False))
    return out


def rule_wrong_instrument(d: "_Run") -> list[Hit]:
    """The surgeon rejected the item the robot passed: easy for the robot, no harm done."""
    return [(float(r["t"]), r["person"], "failure", True, False) for r in d.world
            if r.get("kind") == "robot_experience" and r.get("person")
            and "reject" in str(r.get("hint", "")).lower()]


def rule_directions_nowhere(d: "_Run") -> list[Hit]:
    """The robot offered to guide someone and they did not get to the named place within 120 s.
    The guide's goal result stands in for arrival: failed, or none within the window."""
    out: list[Hit] = []
    goals = {r["id"]: r for r in d.tasks if r.get("kind") == "goal" and "id" in r}
    acts = {r["id"]: r for r in d.tasks if r.get("kind") == "action" and "id" in r}
    for r in d.tasks:
        if r.get("kind") != "result" or r.get("action") != "guide_start" or not r.get("ok"):
            continue
        act = acts.get(r["id"], {})
        t0 = float(act.get("t", r.get("t", 0.0)))
        who = act.get("person") or _requester(d.says, t0)
        if not who:
            continue
        g = goals.get(r["id"])
        arrived = g is not None and g.get("ok") and float(g.get("t", t0)) - t0 <= GUIDE_WINDOW_S
        if not arrived:
            out.append((t0 + GUIDE_WINDOW_S if g is None else float(g.get("t", t0)), who, "failure",
                        True, False))
        else:
            out.append((float(g["t"]), who, "success", False, False))
    return out


def rule_fast_contact(d: "_Run") -> list[Hit]:
    """The robot touched a person (within 0.5 m) while moving faster than 0.5 m/s."""
    return [(float(r["t"]), r["person"], "failure", False, True) for r in d.world
            if r.get("kind") == "contact" and r.get("person")
            and float(r.get("speed", 0.0)) > FAST_CONTACT_MS]


RULES: dict[str, Callable[["_Run"], list[Hit]]] = {
    "task_results": rule_task_results,
    "wrong_instrument": rule_wrong_instrument,
    "directions_nowhere": rule_directions_nowhere,
    "fast_contact": rule_fast_contact,
}
# Which rules run for which scenario: all of them by default, scenario ids may narrow the list.
RULES_FOR_SCENARIO: dict[str, tuple[str, ...]] = {}


class _Run:
    def __init__(self, run_dir: Path):
        self.tasks = _rows(run_dir, "robot_tasks")
        self.says = _rows(run_dir, "says")
        self.audio = _rows(run_dir, "audio")
        self.status = _rows(run_dir, "status")
        self.world = _rows(run_dir, "world")
        self.kinds = _item_kinds(run_dir)


def _is_repair(text: str) -> bool:
    return bool(_REPAIR_WORDS.search(text or ""))


def _repairs(d: _Run, failures: list[Hit], robot_id: str) -> list[Experience]:
    """An apology or a reason in the robot's speech. Right after the error it is immediate; when the
    person next asks the robot for something, it is at the request (Robinette 2015; Nayyar 2018)."""
    out: list[Experience] = []
    robot_says = [s for s in d.says if s.get("person") in _ROBOT_NAMES and _is_repair(s.get("text", ""))]
    for t_f, who, *_ in failures:
        later = sorted(s["t"] for s in d.says
                       if s.get("person") == who and s.get("addressee") in _ROBOT_NAMES and s["t"] > t_f)
        t_req = later[0] if later else None
        for s in robot_says:
            if s.get("addressee") not in (who, None) and s.get("addressee") not in _ROBOT_NAMES:
                continue
            if t_req is not None and t_req <= s["t"] <= t_req + REPAIR_REQUEST_S:
                out.append(Experience(s["t"], who, robot_id, "repair_at_request", False, False, None))
                break
        for s in robot_says:
            if t_f <= s["t"] <= t_f + REPAIR_IMMEDIATE_S and (t_req is None or s["t"] < t_req) \
                    and s.get("addressee") in (who, None):
                out.append(Experience(s["t"], who, robot_id, "repair_immediate", False, False, None))
                break
    return out


def detect(run_dir: "str | Path", preset, robot_id: str = "g1") -> list[Experience]:
    """Experiences of every person with the robot in this run, in time order. preset is the
    ScenarioPreset (or None); RULES_FOR_SCENARIO may narrow the rules for its id."""
    d = _Run(Path(run_dir))
    names = RULES_FOR_SCENARIO.get(getattr(preset, "id", ""), tuple(RULES))
    known = {c.id for c in getattr(preset, "characters", [])}
    hits: list[Hit] = []
    for name in names:
        hits += RULES[name](d)
    if known:
        hits = [h for h in hits if h[1] in known]
    hits.sort(key=lambda h: (h[0], h[1], h[2]))
    out = [Experience(t, who, robot_id, kind, easy, bad, None) for t, who, kind, easy, bad in hits]
    out += _repairs(d, [h for h in hits if h[2] == "failure"], robot_id)
    out.sort(key=lambda e: (e.t, e.person, e.kind))
    return out
