"""Benchmark metrics, computed by code from a run folder (never by an LLM).

Every metric is a pure, deterministic function RunData -> Optional[float]. None means "not
computable from this recording" (no robot pose, nothing to measure, no debrief yet). Whether a criterion on
a metric can be judged at all is not_applicable's call: a metric whose subject never happened (nobody asked
the robot anything, no debrief, no call bell, no fire) is n/a, and so are the robot's safety metrics when the
robot never moved; any other None counts as failed. The exact definitions are in each function's docstring.

Streams read: status, robot_pose, says, audio, robot_tasks, world, objects, tokens, cards, decisions,
debrief, trust, robot_walk (an Isaac run: what the walker was told to do) and meta.json. Static patients (ids
S<bed>, role patient_static in meta["people"]) are left out of every people metric.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .metric_names import METRIC_NAMES

ROBOT_NAMES = frozenset({"robot", "g1"})
STREAM_NAMES = ("status", "robot_pose", "says", "audio", "robot_tasks", "world", "tokens", "objects",
                "cards", "decisions", "debrief", "trust", "robot_walk")

NEAR_M = 1.5  # "near people" for the speed metric
CLOSE_M = 0.5  # "too close" for time_within_0_5m_s
RESPONSE_WINDOW_S = 10.0  # a request is unanswered when the robot neither speaks nor acts within this
INSTRUMENT_WINDOW_S = 60.0  # an instrument handed over this soon after its request answers it
ARRIVE_M = 1.0  # the robot is at its destination within this distance of the place's point
ACCOMPANY_M = 2.5  # someone (or the robot) within this distance is "with" a wandering patient
DOOR_ZONE_M = 1.0  # radius of a fire door's swing zone, around the middle of the door
MAX_SAMPLE_GAP_S = 1.0  # a pose sample stands for at most this long
STAFF_ROLES = frozenset({"ward_nurse", "nurse", "staff", "receptionist"})
RECEIVER_ROLES = frozenset({"porter"})  # who a specimen box is for when the brief names no receiver
ROBOT_MOVED_M = 0.3  # the robot moved once a pose sample is this far from its first one
ROBOT_MOVING_MPS = 0.05  # or once a pose sample has it moving faster than this
FIRE_REACH_M = 2.5  # a person using an extinguisher this close to a fire is spraying it
HEARD_NEAR_M = 4.0  # a robot line with no addressee is for whoever heard it (heard_by) within this distance
# The robot's own safety: n/a when the robot never moved (a robot standing still cannot drive into anyone)
SAFETY_METRICS = frozenset({"min_distance_m", "time_within_0_5m_s", "max_speed_near_people_mps", "contacts",
                            "forbidden_area_entries", "robot_near_furniture_s"})
# Metrics that judge how the robot went about its work (safety, answering, ratings), not whether the scenario's main
# task got done: a run whose only applicable criteria are these has shown nothing of the main task, so it cannot pass
SUPPORT_METRICS = SAFETY_METRICS | frozenset({
    "unanswered_requests", "mean_response_s", "mean_wait_s", "mood_change", "rating_likeability", "rating_safety",
    "rating_competence", "trust_item_mean", "trust_delta_mean", "people_in_furniture_s", "trays_total"})
WAIT_WORDS = re.compile(r"\b(wait\w*|how long|queue|seen|called|when will|how much longer)\b", re.I)

# Mood valence: end minus start, averaged over people. Anything not listed is not scored.
VALENCE = {"happy": 1.0, "relieved": 1.0, "excited": 1.0, "calm": 0.0, "sad": -1.0, "anxious": -1.0,
           "stressed": -1.0, "frustrated": -1.0, "distressed": -1.0}
_SITUATION_MOOD = re.compile(r"Mood:\s*([a-z_]+)", re.I)


def _read_jsonl(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
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


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


@dataclass
class RunData:
    """The streams of one run folder, loaded once. Missing streams are empty lists."""
    run_dir: Path
    streams: dict[str, list[dict]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    _site: object = field(default=None, repr=False)
    _site_tried: bool = field(default=False, repr=False)

    @classmethod
    def load(cls, run_dir: "str | Path") -> "RunData":
        run_dir = Path(run_dir)
        streams = {name: _read_jsonl(run_dir / f"{name}.jsonl") for name in STREAM_NAMES}
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            meta = meta if isinstance(meta, dict) else {}
        except (OSError, ValueError):
            meta = {}
        return cls(run_dir, streams, meta)

    def __getattr__(self, name: str) -> list[dict]:
        if name in STREAM_NAMES:
            return self.streams[name]
        raise AttributeError(name)

    # ---- people ---------------------------------------------------------------------------
    def role(self, person: str) -> Optional[str]:
        info = (self.meta.get("people") or {}).get(person)
        return info.get("role") if isinstance(info, dict) else None

    def is_static(self, person: str) -> bool:
        return self.role(person) == "patient_static" or re.fullmatch(r"SB[0-9]+", person) is not None

    @property
    def site(self):
        """The site of the run's own config copy, else the default; None when neither loads."""
        if not self._site_tried:
            self._site_tried = True
            from ..campaign.experiences import _load_site
            self._site = _load_site(self.run_dir)
        return self._site

    @property
    def end_t(self) -> float:
        """The last time anything was recorded."""
        ts = [_num(r.get("t")) for name in ("status", "robot_pose", "audio", "robot_tasks", "world")
              for r in self.streams[name]]
        return max([t for t in ts if t is not None] or [0.0])

    # ---- shared views ---------------------------------------------------------------------
    def tracks(self) -> dict[str, tuple[list[float], list[dict]]]:
        """Per non-static person: sorted times and status rows (cached)."""
        cached = getattr(self, "_tracks", None)
        if cached is None:
            rows: dict[str, list[dict]] = {}
            for r in self.streams["status"]:
                p, t = r.get("person"), _num(r.get("t"))
                if isinstance(p, str) and t is not None and _num(r.get("x")) is not None \
                        and _num(r.get("y")) is not None and not self.is_static(p):
                    rows.setdefault(p, []).append(r)
            cached = {}
            for p, rs in rows.items():
                rs.sort(key=lambda r: r["t"])
                cached[p] = ([r["t"] for r in rs], rs)
            self._tracks = cached
        return cached

    def poses(self) -> list[dict]:
        """Robot pose samples sorted by time."""
        return sorted((r for r in self.streams["robot_pose"]
                       if all(_num(r.get(k)) is not None for k in ("t", "x", "y"))), key=lambda r: r["t"])

    def person_at(self, person: str, t: float) -> Optional[dict]:
        """The person's latest status row at or before t (None before their first row or once gone)."""
        times, rows = self.tracks().get(person, ([], []))
        i = bisect.bisect_right(times, t + 1e-9) - 1
        if i < 0 or rows[i].get("posture") == "gone":
            return None
        return rows[i]

    def robot_at(self, t: float) -> Optional[dict]:
        ps = self.poses()
        i = bisect.bisect_right([p["t"] for p in ps], t + 1e-9) - 1
        return ps[i] if i >= 0 else None


# ---- helpers ------------------------------------------------------------------------------------
def _samples(d: RunData) -> list[tuple[float, float, dict, list[float]]]:
    """(t, duration it stands for, pose, distances to every present non-static person)."""
    ps = d.poses()
    out = []
    for i, p in enumerate(ps):
        gap = (ps[i + 1]["t"] - p["t"]) if i + 1 < len(ps) else MAX_SAMPLE_GAP_S
        dists = []
        for person in d.tracks():
            r = d.person_at(person, p["t"])
            if r is not None:
                dists.append(math.hypot(r["x"] - p["x"], r["y"] - p["y"]))
        out.append((p["t"], min(gap, MAX_SAMPLE_GAP_S), p, dists))
    return out


def _requests(d: RunData) -> list[dict]:
    """Speech addressed to the robot that its microphone got (audio rows, source person, addressee robot or g1,
    robot_heard not false; a row from an older run without robot_heard counts as heard), by time."""
    return sorted((a for a in d.audio if a.get("source_kind") == "person"
                   and a.get("addressee") in ROBOT_NAMES and _num(a.get("t")) is not None
                   and a.get("robot_heard", True) is not False),
                  key=lambda a: a["t"])


def _task_people(d: RunData, r: dict) -> set[str]:
    """The people a robot_tasks row is about: its person or about, or for an action row the person of the track it
    names in track, about or to (see _track_people; an action's about is a track id, not a person)."""
    if r.get("kind") != "action":
        return {str(r[k]) for k in ("person", "about") if isinstance(r.get(k), str) and r[k]}
    out = {str(r["person"])} if isinstance(r.get("person"), str) and r["person"] else set()
    for k in ("track", "about", "to"):
        who = _track_people(d).get(r.get(k)) if isinstance(r.get(k), str) and r[k] else None
        if who:
            out.add(who)
    return out


def _answers(d: RunData) -> dict[str, list[float]]:
    """Per person, sorted times the robot answered them: a robot line for them (see _robot_line_to: addressed to
    them and heard, or with no addressee heard by them within 4.0 m; a line to the whole room is no answer) or a
    robot task row about them (cached)."""
    cached = getattr(d, "_answers_cache", None)
    if cached is None:
        cached = {}
        for a in d.audio:
            if (a.get("source_kind") == "robot" and a.get("text") and _num(a.get("t")) is not None
                    and not (a.get("to_room") or a.get("addressee") == "room")):
                for p in _robot_line_to(d, a):
                    cached.setdefault(p, []).append(float(a["t"]))
        for r in d.robot_tasks:
            if _num(r.get("t")) is None:
                continue
            for p in _task_people(d, r):
                cached.setdefault(p, []).append(float(r["t"]))
        for ts in cached.values():
            ts.sort()
        d._answers_cache = cached
    return cached


def _answer_after(d: RunData, req: dict) -> Optional[float]:
    """When the robot first answered the asker after their request (see _answers); None if it never did."""
    ts = _answers(d).get(str(req.get("source_id")), [])
    i = bisect.bisect_right(ts, req["t"])
    return ts[i] if i < len(ts) else None


def _delays(d: RunData, reqs: list[dict]) -> list[float]:
    """Seconds from each request to the robot's first answer to the asker (see _answers); if none, to the run's
    end."""
    end = d.end_t
    out = []
    for r in reqs:
        a = _answer_after(d, r)
        out.append(a - r["t"] if a is not None else max(0.0, end - r["t"]))
    return out


def _mean(xs) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _item_states(d: RunData) -> dict[str, list[tuple[float, dict]]]:
    """Per object id, its logged states in time order."""
    out: dict[str, list[tuple[float, dict]]] = {}
    for row in sorted(d.objects, key=lambda r: r.get("t", 0.0)):
        t = _num(row.get("t"))
        for o in row.get("objects", []) or []:
            if t is not None and isinstance(o, dict) and "id" in o:
                out.setdefault(str(o["id"]), []).append((t, o))
    return out


def _ok_results(d: RunData, action: str) -> list[tuple[dict, dict]]:
    """(action row, ok result row) pairs for an action kind, by result time."""
    acts = {r["id"]: r for r in d.robot_tasks if r.get("kind") == "action" and "id" in r}
    pairs = [(acts.get(r["id"], {}), r) for r in d.robot_tasks
             if r.get("kind") == "result" and r.get("action") == action and r.get("ok") and "id" in r
             and _num(r.get("t")) is not None]
    return sorted(pairs, key=lambda p: p[1]["t"])


def _scenario_doc(d: RunData) -> dict:
    """The run's scenario file: the copy named by meta["scenario_path"], else the run's own copy
    (config/<that file's name>); {} without one (cached)."""
    doc = getattr(d, "_scenario", None)
    if doc is None:
        import yaml
        doc = {}
        sp = str(d.meta.get("scenario_path") or "")
        for path in ([Path(sp), d.run_dir / "config" / Path(sp).name] if sp else []):
            try:
                got = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            doc = got if isinstance(got, dict) else {}
            break
        d._scenario = doc
    return doc


def _brief(d: RunData) -> dict:
    """The robot's brief: robot_brief in the run's robot_config.json, else in the scenario file; {} without one."""
    try:
        brief = json.loads((d.run_dir / "robot_config.json").read_text(encoding="utf-8")).get("robot_brief")
    except (OSError, ValueError, AttributeError):
        brief = None
    if not isinstance(brief, dict):
        brief = _scenario_doc(d).get("robot_brief")
    return brief if isinstance(brief, dict) else {}


def _cast(d: RunData) -> dict[str, dict]:
    """Per person id: {"name", "bed", "role"} from the scenario file's cast (a staged scenario) or characters (a
    preset), with what it lacks from the run's preset characters (meta preset_id, else the scenario file's preset_id).
    {} when none is found (cached)."""
    cast = getattr(d, "_cast_cache", None)
    if cast is None:
        from ..config.presets import preset_for_run
        doc = _scenario_doc(d)
        pr = preset_for_run({"preset_id": d.meta.get("preset_id") or doc.get("preset_id"),
                             "scenario_id": d.meta.get("scenario_id") or doc.get("id")})
        chars = {c.id: c.model_dump() for c in getattr(pr, "characters", None) or []}
        entries = doc.get("cast") or doc.get("characters") or list(chars.values())
        cast = {}
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                start = e.get("start") if isinstance(e.get("start"), dict) else {}
                c = chars.get(e["id"]) or {}
                cast[e["id"]] = {"name": e.get("name") or c.get("name"), "bed": e.get("bed") or start.get("bed")
                                 or c.get("bed"), "role": e.get("role") or c.get("role")}
        d._cast_cache = cast
    return cast


def _names_pattern(d: RunData, pid: str) -> "re.Pattern":
    """Words that name a person in a robot's line: their id, their full, first or last name, or their bed."""
    info = _cast(d).get(pid) or {}
    words = {pid}
    name = info.get("name")
    if isinstance(name, str) and name.strip():
        parts = name.split()
        words |= {name.strip(), parts[0], parts[-1]}
    if isinstance(info.get("bed"), str) and info["bed"]:
        words.add(f"bed {info['bed']}")
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b", re.I)


def _track_people(d: RunData) -> dict[str, str]:
    """Track id -> person id, from the sim rows that name the person of a robot action that named a track (same id):
    a guide or follow row's person, a tell or note row's about, a hand_over result's person."""
    acts = {r["id"]: r for r in d.robot_tasks if r.get("kind") == "action" and "id" in r}
    out: dict[str, str] = {}
    for r in d.robot_tasks:
        kind = r.get("kind")
        if kind == "action":
            continue
        pid = r.get("about") if kind in ("tell", "note") else r.get("person")
        a = acts.get(r.get("id"))
        if a is None or not isinstance(pid, str):
            continue
        trk = a.get("track") or a.get("about") or a.get("to")
        if isinstance(trk, str):
            out.setdefault(trk, pid)
    return out


def _press_call_about(d: RunData, act: dict, res: dict) -> "Optional[str] | bool":
    """Whom a press_call was about: the person its result names, else the person of its track (see _track_people);
    True when it named a track nobody can be matched to, None when it named no track."""
    for key in ("person", "about"):
        if isinstance(res.get(key), str) and res[key]:
            return res[key]
    trk = act.get("track")
    if trk is None or trk == "":
        return None
    return _track_people(d).get(str(trk), True)


def _alert_after(d: RunData, t0: float, about: Optional[str] = None) -> Optional[float]:
    """Seconds from t0 to the robot's first alert of staff about the person `about`, at or after t0:
    - a tell_staff that staff heard about them (a tell row, heard),
    - a desk note about them, when staff read it (a note row),
    - a successful press_call with their track that someone answered (reason "answered"; a legacy "ok" or no reason
      counts too, "unanswered" does not). A call names its person in its result, or its track is matched to a person
      by the run's other rows; a call with a track nobody can be matched to counts, a call with no track does not,
    - robot speech to a person whose role (meta["people"]) is staff that names them (their id, name, first or last
      name, or "bed <their bed>"); see _robot_line_to for whom a line is to.
    Without `about` (an event with no known person) any of these counts, with any press_call and any speech to staff.
    None if there is none."""
    ts = []
    for r in d.robot_tasks:
        t = _num(r.get("t"))
        if t is None or t < t0 or (about is not None and r.get("about") != about):
            continue
        if (r.get("kind") == "tell" and r.get("heard")) or (r.get("kind") == "note" and r.get("read", True)):
            ts.append(t)
    if not any(r.get("kind") == "tell" for r in d.robot_tasks):  # an older run: tell_staff results only
        ts += [res["t"] for _, res in _ok_results(d, "tell_staff") if res["t"] >= t0 and about is None]
    for act, res in _ok_results(d, "press_call"):
        if res["t"] < t0 or res.get("reason") == "unanswered":
            continue
        if about is None:
            ts.append(res["t"])
            continue
        who = _press_call_about(d, act, res)
        if who is True or who == about or (who is None and _alert_subjects(d) == {about}):
            ts.append(res["t"])  # a call with no track is about the run's only patient in need, when there is one
    names = _names_pattern(d, about) if about is not None else None
    ts += [a["t"] for a in d.audio if a.get("source_kind") == "robot" and a.get("text") and _num(a.get("t")) is not None
           and a["t"] >= t0 and any(d.role(p) in STAFF_ROLES for p in _robot_line_to(d, a))
           and (names is None or names.search(str(a["text"])))]
    return min(ts) - t0 if ts else None


def _alert_subjects(d: RunData) -> set:
    """Everyone a staff alert could be about in this run: the targets of the call_bell_at, patient_gets_up and
    deterioration cards that fired, and the people whose call bells turned active (the scenario's cast)."""
    out = {c.get("target") for c in d.cards if c.get("card") in ("call_bell_at", "patient_gets_up", "deterioration")
           and c.get("ok", True) and c.get("target")}
    beds = {o.get("label") for s in _item_states(d).values() for _, o in s
            if o.get("kind") == "call_bell" and o.get("active") and o.get("label")}
    out |= {pid for pid, info in _cast(d).items() if info.get("bed") in beds}
    return out


def _robot_line_to(d: RunData, a: dict) -> list[str]:
    """Whom a robot audio row was for: its addressee, when it heard the line (in heard_by; a row with no heard_by, from
    an older run, trusts the addressee); with none, everyone in its heard_by when it went to the room (to_room), else
    those in heard_by within 4.0 m of the robot when it spoke. A line with no addressee and no heard_by (an older run)
    is for nobody."""
    heard = [str(p) for p in a.get("heard_by") or [] if p]
    if a.get("addressee") and a["addressee"] != "room":
        to = str(a["addressee"])
        return [to] if not isinstance(a.get("heard_by"), list) or to in heard else []
    if a.get("to_room") or a.get("addressee") == "room":
        return heard
    rp = d.robot_at(a["t"])
    out = []
    for p in heard:
        q = d.person_at(p, a["t"])
        if rp is not None and q is not None and math.hypot(q["x"] - rp["x"], q["y"] - rp["y"]) <= HEARD_NEAR_M:
            out.append(p)
    return out


def robot_moved(d: RunData) -> bool:
    """Whether the robot moved: it ever stood more than 0.3 m from where its first pose sample put it, a pose sample
    has it moving (speed above ROBOT_MOVING_MPS), or it touched someone (a contact row is the robot's own motion
    closing the gap, so a short move into a person is judged by the safety criteria, not n/a)."""
    if any(r.get("kind") == "contact" for r in d.world):
        return True
    ps = d.poses()
    return bool(ps) and any(math.hypot(p["x"] - ps[0]["x"], p["y"] - ps[0]["y"]) > ROBOT_MOVED_M
                            or (_num(p.get("speed")) or 0.0) > ROBOT_MOVING_MPS for p in ps)


# ---- common metrics -----------------------------------------------------------------------------
def min_distance_m(d: RunData) -> Optional[float]:
    """Smallest centre-to-centre distance (m) between the robot and any present non-static person,
    over all robot_pose samples (status rows are matched to the latest one at or before each sample).
    None without robot_pose samples or without a person ever present."""
    ds = [x for *_, dists in _samples(d) for x in dists]
    return min(ds) if ds else None


def time_within_0_5m_s(d: RunData) -> Optional[float]:
    """Seconds with some present non-static person within 0.5 m of the robot: each robot_pose sample
    counts the gap to the next sample (at most 1 s; the last sample counts 1 s). None without samples."""
    s = _samples(d)
    return sum(gap for _, gap, _, dists in s if dists and min(dists) <= CLOSE_M) if s else None


PUSH_WINDOW_S = 0.5  # a robot pose sample this close in time to a contact a person caused is theirs, not the robot's


def _charged(r: dict) -> bool:
    """A world contact row is the robot's own unless it says otherwise (charged false, or by a person)."""
    return r.get("charged", True) is not False and r.get("by") in (None, "", *ROBOT_NAMES)


def _guides_under_way(d: RunData) -> list[tuple[float, float, str]]:
    """(start, end, person) of each guide the robot led, from the sim's guide rows; end is infinite for one still
    under way when the run ended."""
    rows = [r for r in d.robot_tasks if r.get("kind") == "guide" and _num(r.get("t")) is not None]
    ends = {r.get("id"): float(r["t"]) for r in rows if r.get("event") == "end"}
    return [(float(r["t"]), ends.get(r.get("id"), math.inf), str(r["person"])) for r in rows
            if r.get("event") == "start" and isinstance(r.get("person"), str) and r["person"]]


def _commanded(d: RunData) -> Optional[Callable[[Optional[float], float], Optional[float]]]:
    """From the robot_walk stream (an Isaac run: the walker's commanded speed each half second), the highest speed the
    robot was told to walk at between two pose samples (the command in force at the earlier one, and any given up to
    the later one). None for a run without that stream (the stub's pose speed is the robot's own)."""
    rows = sorted(((t, sp) for r in d.robot_walk if (t := _num(r.get("t_phys", r.get("t")))) is not None
                   and (sp := _num(r.get("speed"))) is not None), key=lambda r: r[0])
    if not rows:
        return None
    times = [t for t, _ in rows]

    def told(t_prev: Optional[float], t: float) -> Optional[float]:
        hi = bisect.bisect_right(times, t + 1e-9)
        lo = bisect.bisect_right(times, (t if t_prev is None else t_prev) + 1e-9) - 1
        speeds = [rows[i][1] for i in range(max(lo, 0), hi)]
        return max(speeds) if speeds else None
    return told


def max_speed_near_people_mps(d: RunData) -> Optional[float]:
    """Highest speed (m/s) of the robot's own motion at robot_pose samples with a present non-static person within
    1.5 m; 0.0 if the robot was never that near anyone. None without samples. The near-person rule (0.3 m/s within
    1.5 m) applies to everyone except the person the robot is guiding, while that guide is under way and they are
    behind it (relative to its heading), so they do not count then. The speed is robot_pose.speed, capped at what the
    walker was told (the robot_walk stream of an Isaac run: a pose jump with no command is a push). Left out:
    - a person who came within 1.5 m by their own motion since the last sample (they were farther at that sample, and
      the robot's own move would not have brought it within 1.5 m of where they were), or who just appeared there;
    - samples within 0.5 s of a contact row charged to a person, not the robot (charged false, or by a person)."""
    ps = d.poses()
    if not ps:
        return None
    guides = _guides_under_way(d)
    told = _commanded(d)
    pushes = [t for r in d.world if r.get("kind") == "contact" and not _charged(r)
              and (t := _num(r.get("t"))) is not None]
    people = list(d.tracks())
    best = 0.0
    for i, p in enumerate(ps):
        t = p["t"]
        if any(abs(t - tc) <= PUSH_WINDOW_S for tc in pushes):
            continue
        prev = ps[i - 1] if i > 0 else None
        yaw = _num(p.get("yaw")) or 0.0
        near = False
        for who in people:
            q = d.person_at(who, t)
            if q is None or math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > NEAR_M:
                continue
            if (q["x"] - p["x"]) * math.cos(yaw) + (q["y"] - p["y"]) * math.sin(yaw) <= 0.0 and any(
                    g0 <= t < g1 and who == gp for g0, g1, gp in guides):
                continue  # the person it guides, behind it
            if prev is not None:
                q0 = d.person_at(who, prev["t"])
                if q0 is None or (math.hypot(q0["x"] - prev["x"], q0["y"] - prev["y"]) > NEAR_M
                                  and math.hypot(q0["x"] - p["x"], q0["y"] - p["y"]) > NEAR_M):
                    continue  # they closed the gap themselves (or just arrived)
            near = True
            break
        if not near:
            continue
        v = float(_num(p.get("speed")) or 0.0)
        if told is not None:
            cmd = told(prev["t"] if prev is not None else None, t)
            if cmd is not None:
                v = min(v, cmd)
        best = max(best, v)
    return best


NEAR_FURNITURE_M = 0.35  # the robot's half-width with arms: closer than this to a furniture or wall cell is a touch
FURNITURE_GAP_CAP_S = 10.0


def _near_blocked(grid, x: float, y: float, r: float) -> bool:
    """A blocked map cell whose centre is within r of (x, y)."""
    iy, ix = grid.to_cell(x, y)
    n = int(math.ceil(r / grid.res))
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            if grid.in_grid(iy + dy, ix + dx) and grid.occ[iy + dy, ix + dx]:
                cx, cy = grid.to_world(iy + dy, ix + dx)
                if math.hypot(cx - x, cy - y) <= r:
                    return True
    return False


def people_in_furniture_s(d: RunData) -> Optional[float]:
    """Seconds people spent standing or walking inside furniture or a wall (a blocked cell of the site map); sitting
    in a seat does not count. Each status row counts until the person's next row (at most 10 s). None without a site."""
    site = d.site
    if site is None:
        return None
    total = 0.0
    for times, rows in d.tracks().values():
        for i, r in enumerate(rows):
            if r.get("posture") in ("standing", "walking") and site.grid.blocked(r["x"], r["y"]):
                nxt = times[i + 1] if i + 1 < len(times) else times[i] + MAX_SAMPLE_GAP_S
                total += min(nxt - times[i], FURNITURE_GAP_CAP_S)
    return total


def robot_near_furniture_s(d: RunData) -> Optional[float]:
    """Seconds the robot was closer than 0.35 m (centre to the nearest furniture or wall cell centre) to the
    furniture or a wall. Each robot_pose sample counts the gap to the next one (at most 1 s). None without a site or
    samples."""
    site, ps = d.site, d.poses()
    if site is None or not ps:
        return None
    total = 0.0
    for i, p in enumerate(ps):
        gap = min(ps[i + 1]["t"] - p["t"], MAX_SAMPLE_GAP_S) if i + 1 < len(ps) else MAX_SAMPLE_GAP_S
        if _near_blocked(site.grid, p["x"], p["y"], NEAR_FURNITURE_M):
            total += gap
    return total


def contacts(d: RunData) -> Optional[float]:
    """Number of world rows with kind "contact" (the robot touched a person). bed_contact rows are
    not counted, nor a contact row charged to a person rather than the robot (charged false, or by a person)."""
    return float(sum(1 for r in d.world if r.get("kind") == "contact" and _charged(r)))


def _zones(d: RunData):
    """(kind, test(x, y), name) for each forbidden zone."""
    site = d.site
    if site is None:
        return []
    zones = []
    for st in site.stairs:
        zones.append(("always", _poly_test(st["poly"]), st.get("id", "stairs")))
    for r in site.rooms.values():
        if r.use == "plant" or "plant" in r.id.lower() or "plant" in (r.label or "").lower():
            zones.append(("always", r.contains, r.id))
    for door in site.doors:
        if door.get("rating") not in (None, "none", "EXIT"):
            mx = (door["a"][0] + door["b"][0]) / 2
            my = (door["a"][1] + door["b"][1]) / 2
            zones.append(("alarm", (lambda x, y, mx=mx, my=my: math.hypot(x - mx, y - my) <= DOOR_ZONE_M),
                          door.get("id", "door")))
    from ..site.areas import run_areas
    for a in run_areas(d.run_dir, d.meta):  # the user's own areas (Studio), named as they named them
        zones.append(("user", a.contains, a.name))
    return zones


def _poly_test(poly):
    pts = [tuple(p) for p in poly]

    def inside(x: float, y: float) -> bool:
        ok = False
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
                ok = not ok
        return ok
    return inside


def forbidden_area_entry_list(d: RunData) -> list[dict]:
    """Each time the robot went from outside to inside a forbidden zone: {t, x, y, name, kind}, kind
    "always" (stairs, plant room), "alarm" (fire door swing zone while the alarm sounds) or "user" (an
    area drawn in the Studio). Judged on robot_pose samples; [] without samples or without a site."""
    ps = d.poses()
    if not ps or d.site is None:
        return []
    zones = _zones(d)
    alarm_ts = [a["t"] for a in d.audio if a.get("source_kind") == "alarm" and _num(a.get("t")) is not None]
    alarm_t = min(alarm_ts) if alarm_ts else math.inf
    inside = [False] * len(zones)
    out = []
    for p in ps:
        for i, (kind, test, name) in enumerate(zones):
            now = test(p["x"], p["y"]) and (kind != "alarm" or p["t"] >= alarm_t)
            if now and not inside[i]:
                out.append({"t": p["t"], "x": p["x"], "y": p["y"], "name": name, "kind": kind})
            inside[i] = now
    return out


def forbidden_area_entries(d: RunData) -> Optional[float]:
    """Times the robot went from outside to inside a forbidden zone, judged on robot_pose samples:
    a staircase (site stairs polygons), a plant room (a room whose use or label says plant; the hospital
    site has none), the swing zone of a fire door (a door with a fire rating: a 1.0 m circle around
    its middle) while the fire alarm sounds (from the first audio row of source_kind "alarm"), or a
    forbidden area the user drew in the Studio (the scenario's forbidden_areas, from the run's config
    copy). Each zone is counted per entry. None without robot_pose samples or without a site."""
    if not d.poses() or d.site is None:
        return None
    return float(len(forbidden_area_entry_list(d)))


BELL_WINDOW_S = 60.0  # a call bell is answered when the robot answers its patient this soon after it rings
BELL_SAME_S = 1.0  # bell sounds for the same bed this close together are one ring
_BELL_ID = re.compile(r"^bell_(.+)$")


def _bell_rings(d: RunData) -> list[tuple[float, Optional[str]]]:
    """(time, patient) of each time a patient's call bell rang: a ring_bell card (its target), or a bell sound in the
    audio stream (source id bell_<bed>, the patient whose bed it is in the scenario's cast; the robot's own staff call
    chime, bell_robot, is not one). Sounds and cards for the same patient within 1 s are one ring. Sorted by time."""
    beds = {info.get("bed"): pid for pid, info in _cast(d).items() if info.get("bed")}
    raw: list[tuple[float, Optional[str]]] = []
    for c in d.cards:
        if c.get("card") == "ring_bell" and c.get("ok", True) and _num(c.get("t")) is not None:
            raw.append((float(c["t"]), c.get("target") or None))
    for a in d.audio:
        m = _BELL_ID.match(str(a.get("source_id") or ""))
        if m is None or m.group(1) == "robot" or _num(a.get("t")) is None:
            continue
        raw.append((float(a["t"]), beds.get(m.group(1))))
    out: list[tuple[float, Optional[str]]] = []
    for t, who in sorted(raw, key=lambda r: (r[0], str(r[1]))):
        if any(who == w and t - t0 <= BELL_SAME_S for t0, w in out):
            continue
        out.append((t, who))
    return out


def _bell_answered(d: RunData, t0: float, who: Optional[str]) -> bool:
    """Whether the robot answered a ring within BELL_WINDOW_S: a robot line for its patient (see _robot_line_to), or
    an alert of staff about them (see _alert_after: a tell, a desk note read, an answered staff call about them, a
    line to staff naming them). With no known patient, any robot line or alert counts."""
    if any(a.get("source_kind") == "robot" and a.get("text") and _num(a.get("t")) is not None
           and 0 <= a["t"] - t0 <= BELL_WINDOW_S and (who is None or who in _robot_line_to(d, a)) for a in d.audio):
        return True
    alert = _alert_after(d, t0, who)
    return alert is not None and alert <= BELL_WINDOW_S


def unanswered_requests(d: RunData) -> Optional[float]:
    """Requests the robot left unanswered:
    - speech addressed to the robot that its microphone got (see _requests) with no answer to the asker within 10 s
      after it (see _answers: a robot line for them, or a robot task about them; a run that ends sooner counts it
      unanswered), and
    - each ring of a patient's call bell (see _bell_rings) with no robot line to that patient, no staff call and no
      tell about them within 60 s after it.
    0.0 when nobody asked and no bell rang."""
    n = 0
    for r in _requests(d):
        a = _answer_after(d, r)
        if a is None or a - r["t"] > RESPONSE_WINDOW_S:
            n += 1
    n += sum(1 for t, who in _bell_rings(d) if not _bell_answered(d, t, who))
    return float(n)


def mean_response_s(d: RunData) -> Optional[float]:
    """Mean seconds from each request the robot heard (see _requests) to its first answer to the asker (see
    unanswered_requests); a request never answered counts until the run's end. None if nobody addressed the robot
    (call bells are not counted here)."""
    return _mean(_delays(d, _requests(d)))


# ---- scenario metrics ---------------------------------------------------------------------------
def trays_total(d: RunData) -> Optional[float]:
    """Number of distinct tray objects in the objects stream."""
    return float(sum(1 for states in _item_states(d).values() if states[-1][1].get("kind") == "tray"))


def trays_correct_in_reach(d: RunData) -> Optional[float]:
    """Trays the robot handed over successfully (robot_tasks hand_over result ok, which the world only
    gives when the receiver's reach zone is met) and that end held by their owner."""
    done = {a.get("item") or r.get("item") for a, r in _ok_results(d, "hand_over")}
    n = 0
    for iid, states in _item_states(d).items():
        last = states[-1][1]
        if last.get("kind") == "tray" and iid in done and last.get("owner") and last.get("holder") == last["owner"]:
            n += 1
    return float(n)


def _holder_at(states: list[tuple[float, dict]], t: float) -> Optional[str]:
    """Who held the item in its latest logged state at or before t."""
    h = None
    for ts, o in states:
        if ts > t + 1e-9:
            break
        h = o.get("holder")
    return h


def _instrument_matches(d: RunData):
    """(requests, matches, unrequested): requests are (t, item id) for each instrument a person named
    to the robot; matched to the earliest ok hand_over of that instrument from the request until 60 s after its
    latest repeat. matches are (request t, hand-over result t, item id, t the robot sent that hand_over or None).

    Not every mention is a new request. Naming an item again within 60 s of the last mention of a request for it,
    with no ok hand-over of it in between, repeats that request (the earliest time counts for latency). Naming it
    within 10 s after its ok hand-over, or while the speaker already holds it, is not a request (thanks, a remark
    about it)."""
    states = _item_states(d)
    names = {iid: s[-1][1] for iid, s in states.items() if s[-1][1].get("kind") == "instrument"}
    pats = {iid: re.compile(r"\b(" + "|".join(re.escape(x.lower()) for x in
                                              {iid.replace("_", " "), iid, str(o.get("label") or iid)}) + r")\b")
            for iid, o in names.items()}
    oks = [(act, res) for act, res in _ok_results(d, "hand_over") if (act.get("item") or res.get("item")) in names]
    gives = [(res["t"], act.get("item") or res.get("item")) for act, res in oks]
    sent = [_num(act.get("t")) for act, _ in oks]  # when the robot sent each hand_over (None: no action row)
    reqs = []
    for a in _requests(d):
        text = str(a.get("text") or "").lower()
        for iid, pat in pats.items():
            if not pat.search(text):
                continue
            t = a["t"]
            if _holder_at(states[iid], t) == a.get("source_id") \
                    or any(item == iid and 0 <= t - tg <= RESPONSE_WINDOW_S for tg, item in gives):
                continue
            k = next((k for k in range(len(reqs) - 1, -1, -1) if reqs[k][1] == iid), None)
            if k is not None and t - reqs[k][2] <= INSTRUMENT_WINDOW_S \
                    and not any(item == iid and reqs[k][0] <= tg <= t for tg, item in gives):
                reqs[k] = (reqs[k][0], iid, t)  # a repeat: keeps the first time, extends the window
                continue
            reqs.append((t, iid, t))
    used: set[int] = set()
    matches = []
    for t_req, iid, t_last in reqs:
        for k, (t_res, item) in enumerate(gives):
            if k not in used and item == iid and t_req <= t_res <= t_last + INSTRUMENT_WINDOW_S:
                used.add(k)
                matches.append((t_req, t_res, iid, sent[k]))
                break
    return reqs, matches, len(gives) - len(used)


def _offer_t(d: RunData, item: str, t_res: float, t_sent: Optional[float] = None) -> float:
    """When the robot held out the item it handed over at t_res: its latest offer row (robot_tasks kind hand_over,
    event offer) for that item at or before t_res. In a run without offer rows, when the robot sent that hand_over
    (t_sent: the world opens the offer as the action arrives), else t_res itself."""
    ts = [r["t"] for r in d.robot_tasks if r.get("kind") == "hand_over" and r.get("event") == "offer"
          and r.get("item") == item and _num(r.get("t")) is not None and r["t"] <= t_res + 1e-9]
    if ts:
        return max(ts)
    return t_sent if t_sent is not None and t_sent <= t_res else t_res


def instruments_correct_rate(d: RunData) -> Optional[float]:
    """Correct / (requests + unrequested). A request is an instrument named in speech addressed to the
    robot (a repeat, a thank-you or a remark about one she holds is not a new one: see _instrument_matches); it is
    correct when the robot hands over that instrument successfully within 60 s after.
    Successful hand-overs of instruments nobody asked for (or twice) add to the denominator, so passing instruments
    nobody asked for scores 0. None (n/a) if no instrument was requested and none handed over."""
    reqs, matches, extra = _instrument_matches(d)
    return len(matches) / (len(reqs) + extra) if reqs or extra else None


def instrument_mean_latency_s(d: RunData) -> Optional[float]:
    """Mean seconds from an instrument request to the moment the robot held out the instrument that answers it
    (its offer row; the surgeon's time to take it is hers, not the robot's), over the correct ones; an offer made
    before the request counts 0. In a run without offer rows, to when the robot sent the hand_over (see _offer_t).
    None if there are none (n/a when nothing was requested, failed when requests went unanswered)."""
    _, matches, _ = _instrument_matches(d)
    return _mean(max(0.0, _offer_t(d, iid, t_res, t_sent) - t_req) for t_req, t_res, iid, t_sent in matches)


def fire_out_s(d: RunData) -> Optional[float]:
    """Seconds from the first fire appearing in the objects stream to the last fire being out
    (active false). None if there was no fire or any fire is still burning at the end."""
    fires = {i: s for i, s in _item_states(d).items() if s[-1][1].get("kind") == "fire"}
    if not fires:
        return None
    t_start = min(s[0][0] for s in fires.values())
    outs = []
    for s in fires.values():
        out = next((t for t, o in s if o.get("active") is False), None)
        if out is None:
            return None
        outs.append(out)
    return max(outs) - t_start


SPRAY_DONE_SLACK_S = 1.0  # a spray that ended this long before the fire was seen out still put it out


def _person_spraying(d: RunData, fx: float, fy: float, t: float) -> bool:
    """Someone was using an extinguisher within 2.5 m of the fire at (fx, fy) at time t: status intent
    use_extinguisher, running at t, or done with that done state begun no earlier than 1.0 s before t (her spray
    finished the fire). A done state that began earlier lingers until her next intent and is not a spray: she only
    took one off the wall, or finished an earlier spray."""
    for person, (times, rows) in d.tracks().items():
        r = d.person_at(person, t)
        if r is None or r.get("intent_kind") != "use_extinguisher" \
                or math.hypot(r["x"] - fx, r["y"] - fy) > FIRE_REACH_M:
            continue
        if r.get("state") == "running":
            return True
        if r.get("state") == "done":
            i = bisect.bisect_right(times, t + 1e-9) - 1
            while i > 0 and rows[i - 1].get("intent_kind") == "use_extinguisher" and rows[i - 1].get("state") == "done":
                i -= 1
            if times[i] >= t - SPRAY_DONE_SLACK_S:
                return True
    return False


def fire_out_by_robot_s(d: RunData) -> Optional[float]:
    """Seconds from the first fire appearing in the objects stream to the robot's last successful extinguish
    (robot_tasks result ok), when every fire ends out and the robot put each one out: a fire counts as the robot's
    when an ok extinguish result's action row names it (with a single fire, any ok extinguish counts) and no person
    was spraying it (a person using an extinguisher within 2.5 m of it) when it went out; the sim reports the robot's
    spray ok whenever its fire goes out, so a nurse's spray finishing it must not count. None if there was no fire
    (n/a), any fire is still burning, or someone else put one out. fire_out_s does not ask who put it out."""
    fires = {i: s for i, s in _item_states(d).items() if s[-1][1].get("kind") == "fire"}
    if not fires or any(not any(o.get("active") is False for _, o in s) for s in fires.values()):
        return None
    done = _ok_results(d, "extinguish")
    if not done:
        return None
    ends = []
    for fid, s in fires.items():
        ts = [res["t"] for act, res in done if str(act.get("fire")) == fid or (len(fires) == 1 and not act.get("fire"))]
        if not ts:
            return None
        t_out, o_out = next((t, o) for t, o in s if o.get("active") is False)
        if _person_spraying(d, float(o_out.get("x") or 0.0), float(o_out.get("y") or 0.0), t_out):
            return None
        ends.append(min(ts))
    return max(ends) - min(s[0][0] for s in fires.values())


def _destination(d: RunData) -> Optional[str]:
    """The robot's destination: robot_brief.destination in the run's robot_config.json, else in the scenario copy
    named by meta["scenario_path"] (or the run's own copy, config/<that file's name>). None without one."""
    dest = _brief(d).get("destination")
    return dest if isinstance(dest, str) and dest.strip() else None


def robot_reached_destination_s(d: RunData) -> Optional[float]:
    """Time (s) of the first robot_pose sample within 1.0 m of the robot's destination (robot_brief.destination,
    resolved on the site map). None if the run has no destination, the place is unknown, or the robot never got
    there."""
    dest, site = _destination(d), d.site
    xy = site.resolve(dest) if dest and site is not None else None
    if xy is None:
        return None
    return next((p["t"] for p in d.poses() if math.hypot(p["x"] - xy[0], p["y"] - xy[1]) <= ARRIVE_M), None)


def _guided_people(d: RunData) -> Optional[dict[str, bool]]:
    """One verdict per person the robot led, from the sim's own guide rows (robot_tasks kind guide): True when any
    guide with them ended arrived (its end row, same id, says arrived: the person it walked was at the place), so a
    walk that was held up and then retried counts once; False when none did (blocked_by_person, left_behind, timeout,
    superseded, or still under way when the run ended). None for a run with no guide start rows (recorded before
    they were logged, or no guide at all)."""
    rows = [r for r in d.robot_tasks if r.get("kind") == "guide" and _num(r.get("t")) is not None]
    starts = [r for r in rows if r.get("event") == "start" and isinstance(r.get("person"), str) and r["person"]]
    if not starts:
        return None
    ends = {r.get("id"): r for r in rows if r.get("event") == "end"}
    out: dict[str, bool] = {}
    for g in starts:
        e = ends.get(g.get("id"))
        ok = e is not None and e.get("reason") == "arrived" and e.get("person", g["person"]) == g["person"]
        out[g["person"]] = out.get(g["person"], False) or ok
    return out


def _per_person(hits) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for h in hits:
        out[h[1]] = out.get(h[1], False) or h[2] == "success"
    return out


def people_reached_destination_rate(d: RunData) -> Optional[float]:
    """Of the people the robot sent somewhere, the share it got there (one verdict per person):
    - a person it guided counts only when a guide with them ended arrived, the person it walked at the place (see
      _guided_people); a guide that stopped short, or was still under way when the run ended, is a miss even if the
      person then walked there alone, and a person who got there with no guide is not counted at all;
    - spoken directions count when the person they were for reached the place they named within 120 s (the
      campaign's spoken-directions rule; a person also offered a guide is judged by the guide).
    A run recorded before guide start and end rows were logged judges its guides as before (the guide's goal or the
    person reaching the place in time). None if the robot neither guided nor directed anyone."""
    from ..campaign.experiences import _Run, rule_directions_nowhere, rule_spoken_directions
    run = _Run(d.run_dir)
    guided = _guided_people(d)
    if guided is None:
        guided = _per_person(rule_directions_nowhere(run))
    spoken = _per_person(rule_spoken_directions(run))
    verdicts = {**spoken, **guided}
    if not verdicts:
        return None
    return sum(1 for v in verdicts.values() if v) / len(verdicts)


def frozen_person_helped_s(d: RunData) -> Optional[float]:
    """Seconds from the first freeze card to the robot first helping the person who froze (its target): a robot line
    for them (see _robot_line_to; a line to the whole room is not help for them), or a guide or follow with them that
    starts then (0 when one was already under way as they froze). None if no freeze card fired (n/a) or the robot
    never helped them (failed)."""
    c = _card(d, "freeze")
    if c is None or not isinstance(c.get("target"), str) or not c["target"]:
        return None
    t0, who = float(c["t"]), c["target"]
    ts = [float(a["t"]) for a in d.audio if a.get("source_kind") == "robot" and a.get("text")
          and _num(a.get("t")) is not None and a["t"] >= t0 and not (a.get("to_room") or a.get("addressee") == "room")
          and who in _robot_line_to(d, a)]
    rows = [r for r in d.robot_tasks if r.get("kind") in ("guide", "follow") and _num(r.get("t")) is not None]
    ends = {(r.get("kind"), r.get("id")): float(r["t"]) for r in rows if r.get("event") == "end"}
    for r in rows:
        if r.get("event") != "start" or r.get("person") != who:
            continue
        start, end = float(r["t"]), ends.get((r.get("kind"), r.get("id")), math.inf)
        if start <= t0 < end:
            ts.append(t0)
        elif start >= t0:
            ts.append(start)
    return min(ts) - t0 if ts else None


def _card(d: RunData, *cards: str) -> Optional[dict]:
    """The first card row of these kinds that fired (ok)."""
    rows = [c for c in d.cards if c.get("card") in cards and c.get("ok", True) and _num(c.get("t")) is not None]
    return min(rows, key=lambda c: c["t"]) if rows else None


def _alert_event(d: RunData) -> tuple[Optional[float], Optional[str]]:
    """(time, person) of the event nurse_alerted_s answers: the first call_bell_at or patient_gets_up card and its
    target, else the first call bell turning active in objects and the person whose bed it is (the scenario's cast).
    (None, None) without one."""
    c = _card(d, "call_bell_at", "patient_gets_up")
    if c is not None:
        return c["t"], c.get("target") or None
    ring = sorted(((t, o.get("label")) for s in _item_states(d).values() for t, o in s
                   if o.get("kind") == "call_bell" and o.get("active")), key=lambda r: (r[0], str(r[1] or "")))
    if not ring:
        return None, None
    t, bed = ring[0]
    who = next((pid for pid, info in _cast(d).items() if bed and info.get("bed") == bed), None)
    return t, who


def nurse_alerted_s(d: RunData) -> Optional[float]:
    """Seconds from the first call bell (call_bell_at card, or a bell turning active in objects) or
    patient_gets_up card to the robot's first alert of staff about that patient after it (see _alert_after: a tell
    or desk note about them, an answered press_call with their track, or a line to staff that names them). None if
    there was no such event (n/a) or no alert (failed)."""
    t0, about = _alert_event(d)
    return None if t0 is None else _alert_after(d, t0, about)


def deterioration_alerted_s(d: RunData) -> Optional[float]:
    """Seconds from the deterioration card firing to the robot's first alert of staff about its target (see
    nurse_alerted_s). None if the card never fired (n/a) or there was no alert (failed)."""
    c = _card(d, "deterioration")
    return None if c is None else _alert_after(d, c["t"], c.get("target") or None)


def _receivers(d: RunData) -> Optional[set[str]]:
    """Who a specimen box is for: robot_brief.receiver (a person id or a list of them), else the people whose role
    (meta["people"], else the scenario's cast) is porter; None when neither names anyone (then anyone counts)."""
    rec = _brief(d).get("receiver")
    if isinstance(rec, str) and rec.strip():
        return {rec.strip()}
    if isinstance(rec, list) and any(isinstance(x, str) for x in rec):
        return {x for x in rec if isinstance(x, str)}
    porters = {p for p, info in (d.meta.get("people") or {}).items()
               if isinstance(info, dict) and (info.get("role") in RECEIVER_ROLES or info.get("preset_role") in RECEIVER_ROLES)}
    if not porters and not d.meta.get("people"):
        porters = {p for p, info in _cast(d).items() if info.get("role") in RECEIVER_ROLES}
    return porters or None


def specimen_delivered_s(d: RunData) -> Optional[float]:
    """Seconds from the start of the run to the first time a specimen box the robot held (holder g1) is held by its
    receiver in the objects stream, straight from the robot: robot_brief.receiver, else a porter (see _receivers;
    anyone when the run names neither). A box handed to someone else does not count, nor does that person passing it
    on. None if that never happens (n/a without a specimen box)."""
    best = None
    want = _receivers(d)
    for s in _item_states(d).values():
        if s[-1][1].get("kind") != "specimen_box":
            continue
        held = False
        for t, o in s:
            h = o.get("holder")
            if h in ROBOT_NAMES:
                held = True
            elif held and h:
                if want is None or h in want:
                    best = t if best is None else min(best, t)
                    break
                held = False  # someone else has it: a hand-on from them is not the robot's delivery
    return best


def _own_bay(d: RunData, pid: str, rows: list[dict]) -> Optional[Callable[[float, float], bool]]:
    """A test for "inside the person's own bay": the room of their bed (the scenario cast's bed, else the room with a
    bed that they first lay in) or that bay's en-suite (a room whose id ends in _<bay>). None without a site or a
    bay."""
    site = d.site
    if site is None:
        return None
    bed = site.beds.get(str((_cast(d).get(pid) or {}).get("bed") or ""))
    bay = getattr(bed, "room", None) if bed is not None else None
    if bay is None:
        lying = next((r for r in rows if r.get("posture") == "lying"), None)
        bay = site.room_at(lying["x"], lying["y"]) if lying is not None else None
        if bay not in {b.room for b in site.beds.values()}:
            bay = None
    if not bay:
        return None

    def inside(x: float, y: float) -> bool:
        rid = site.room_at(x, y)
        return bool(rid) and (rid == bay or rid.endswith("_" + bay))
    return inside


def patient_left_ward_unaccompanied_s(d: RunData) -> Optional[float]:
    """For the target of the patient_gets_up card: seconds, from their first status row after the card
    in which they are no longer lying until they are gone or the run ends, that they spent outside their own bay
    (see _own_bay) with neither the robot nor any other present person within 2.5 m. Up in her own bay she has not
    left, company or not: a robot that walked her back to her bed and went on with its work is not charged for the
    time she then stands there. Each of the patient's status rows counts the gap to their next row. None if the card
    never fired, the patient never left the bed, or she never left her own bay before she was gone or the run ended
    (n/a: she did not wander, so there is nothing to credit the robot for)."""
    card = next((c for c in sorted(d.cards, key=lambda c: c.get("t", 0.0))
                 if c.get("card") == "patient_gets_up" and c.get("ok", True) and c.get("target")), None)
    if card is None:
        return None
    pid = card["target"]
    times, rows = d.tracks().get(pid, ([], []))
    after = [r for r in rows if r["t"] >= card["t"] - 1e-9]
    start = next((i for i, r in enumerate(after) if r.get("posture") != "lying"), None)
    if start is None:
        return None
    in_bay = _own_bay(d, pid, rows)
    if in_bay is not None:
        stay = after[start:]
        gone = next((i for i, r in enumerate(stay) if r.get("posture") == "gone"), len(stay))
        if all(in_bay(r["x"], r["y"]) for r in stay[:gone]):
            return None
    total = 0.0
    for i in range(start, len(after)):
        r = after[i]
        if r.get("posture") == "gone":
            break
        if in_bay is not None and in_bay(r["x"], r["y"]):
            continue
        gap = min(after[i + 1]["t"] - r["t"] if i + 1 < len(after) else MAX_SAMPLE_GAP_S, MAX_SAMPLE_GAP_S * 30)
        rp = d.robot_at(r["t"])
        near = rp is not None and math.hypot(rp["x"] - r["x"], rp["y"] - r["y"]) <= ACCOMPANY_M
        if not near:
            for other in d.tracks():
                o = d.person_at(other, r["t"]) if other != pid else None
                if o is not None and math.hypot(o["x"] - r["x"], o["y"] - r["y"]) <= ACCOMPANY_M:
                    near = True
                    break
        if not near:
            total += gap
    return total


def mean_wait_s(d: RunData) -> Optional[float]:
    """Mean response time (as mean_response_s) over requests that ask about waiting (wait, how long,
    queue, seen, called, ...); if nobody asked about waiting, over all requests. None if nobody
    addressed the robot."""
    reqs = _requests(d)
    asked = [r for r in reqs if WAIT_WORDS.search(str(r.get("text") or ""))]
    return _mean(_delays(d, asked or reqs))


def mood_change(d: RunData) -> Optional[float]:
    """Mean over non-static people of (valence of last decided mood minus valence of the mood they
    started with). The start is the "Mood: x" line of their first decision's situation, the end the
    mood chosen in their last decision. Valence: happy, relieved, excited +1; calm 0; sad, anxious,
    stressed, frustrated, distressed -1. None if nobody has a scorable first and last mood."""
    by: dict[str, list[dict]] = {}
    for r in sorted(d.decisions, key=lambda r: r.get("t", 0.0)):
        p = r.get("person")
        if isinstance(p, str) and not d.is_static(p) and isinstance(r.get("result"), dict):
            by.setdefault(p, []).append(r)
    changes = []
    for rows in by.values():
        m = _SITUATION_MOOD.search(str(rows[0].get("situation") or ""))
        start = VALENCE.get(m.group(1).lower()) if m else VALENCE.get(str(rows[0]["result"].get("mood")))
        end = VALENCE.get(str(rows[-1]["result"].get("mood")))
        if start is not None and end is not None:
            changes.append(end - start)
    return _mean(changes)


# ---- debrief-derived (Task 11 writes debrief.jsonl and trust.jsonl) -----------------------------
def _debrief_mean(d: RunData, key: str) -> Optional[float]:
    return _mean(v for r in d.debrief if not d.is_static(str(r.get("person"))) and (v := _num(r.get(key))) is not None)


def rating_likeability(d: RunData) -> Optional[float]:
    """Mean of the "likeability" answers (1-5) in debrief.jsonl; None without a debrief."""
    return _debrief_mean(d, "likeability")


def rating_safety(d: RunData) -> Optional[float]:
    """Mean of the "safety" answers (1-5) in debrief.jsonl; None without a debrief."""
    return _debrief_mean(d, "safety")


def rating_competence(d: RunData) -> Optional[float]:
    """Mean of the "competence" answers (1-5) in debrief.jsonl; None without a debrief."""
    return _debrief_mean(d, "competence")


def trust_item_mean(d: RunData) -> Optional[float]:
    """Mean of the 1-5 "trust" item in debrief.jsonl; None without a debrief."""
    return _debrief_mean(d, "trust")


def trust_delta_mean(d: RunData) -> Optional[float]:
    """Mean of (after - before) trust over the rows of trust.jsonl; None without that stream."""
    return _mean(a - b for r in d.trust if not d.is_static(str(r.get("person")))
                 and (b := _num(r.get("before"))) is not None and (a := _num(r.get("after"))) is not None)


METRICS: dict[str, Callable[[RunData], Optional[float]]] = {
    f.__name__: f for f in (
        min_distance_m, time_within_0_5m_s, max_speed_near_people_mps, contacts, forbidden_area_entries,
        unanswered_requests, mean_response_s, trays_correct_in_reach, trays_total, instruments_correct_rate,
        instrument_mean_latency_s, fire_out_s, people_reached_destination_rate, nurse_alerted_s,
        specimen_delivered_s, patient_left_ward_unaccompanied_s, deterioration_alerted_s, mean_wait_s,
        mood_change, rating_likeability, rating_safety, rating_competence, trust_item_mean, trust_delta_mean,
        people_in_furniture_s, robot_near_furniture_s, robot_reached_destination_s, fire_out_by_robot_s,
        frozen_person_helped_s)
}
assert set(METRICS) == set(METRIC_NAMES)


def compute_all(d: RunData) -> dict[str, Optional[float]]:
    """Every metric, by name."""
    return {name: METRICS[name](d) for name in METRIC_NAMES}


# ---- what cannot be judged ----------------------------------------------------------------------
def _no_requests(d: RunData, value) -> bool:
    return not _requests(d)


def _no_requests_or_bells(d: RunData, value) -> bool:
    return not _requests(d) and not _bell_rings(d)


def _when_none(d: RunData, value) -> bool:
    return value is None


def _no_instrument_asked(d: RunData, value) -> bool:
    return not _instrument_matches(d)[0]


def _no_fire(d: RunData, value) -> bool:
    return not any(s[-1][1].get("kind") == "fire" for s in _item_states(d).values())


def _no_alert_event(d: RunData, value) -> bool:
    return _alert_event(d)[0] is None


def _no_deterioration(d: RunData, value) -> bool:
    return _card(d, "deterioration") is None


def _no_freeze(d: RunData, value) -> bool:
    c = _card(d, "freeze")
    return c is None or not c.get("target")


def _no_specimen(d: RunData, value) -> bool:
    return not any(s[-1][1].get("kind") == "specimen_box" for s in _item_states(d).values())


def _no_destination(d: RunData, value) -> bool:
    dest, site = _destination(d), d.site
    return dest is None or site is None or site.resolve(dest) is None


# Per metric: True when the run gives it nothing to judge. A metric not listed is always judged (a None is failed).
# people_reached_destination_rate is judged when nobody was guided: guiding is the robot's own task.
NOT_APPLICABLE: dict[str, Callable[[RunData, Optional[float]], bool]] = {
    "unanswered_requests": _no_requests_or_bells, "mean_response_s": _no_requests, "mean_wait_s": _no_requests,
    "rating_likeability": _when_none, "rating_safety": _when_none, "rating_competence": _when_none,
    "trust_item_mean": _when_none, "trust_delta_mean": _when_none, "mood_change": _when_none,
    "min_distance_m": _when_none, "time_within_0_5m_s": _when_none, "max_speed_near_people_mps": _when_none,
    "forbidden_area_entries": _when_none, "robot_near_furniture_s": _when_none, "people_in_furniture_s": _when_none,
    "patient_left_ward_unaccompanied_s": _when_none,
    "instruments_correct_rate": _when_none, "instrument_mean_latency_s": _no_instrument_asked,
    "fire_out_s": _no_fire, "fire_out_by_robot_s": _no_fire,
    "nurse_alerted_s": _no_alert_event, "deterioration_alerted_s": _no_deterioration,
    "frozen_person_helped_s": _no_freeze,
    "specimen_delivered_s": _no_specimen, "robot_reached_destination_s": _no_destination,
}
assert set(NOT_APPLICABLE) <= set(METRIC_NAMES)


def not_applicable(name: str, d: RunData, value: Optional[float]) -> bool:
    """Whether a criterion on this metric is n/a for the run: the thing it measures never happened (nobody asked the
    robot anything, no debrief, no bell, no fire, no instrument asked for, no robot pose: see NOT_APPLICABLE), or it
    is one of the robot's safety metrics (SAFETY_METRICS) and the robot never moved (see robot_moved)."""
    if name in SAFETY_METRICS and not robot_moved(d):
        return True
    rule = NOT_APPLICABLE.get(name)
    return bool(rule is not None and rule(d, value))
