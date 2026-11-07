"""Benchmark metrics, computed by code from a run folder (never by an LLM).

Every metric is a pure, deterministic function RunData -> Optional[float]. None means "not
computable from this recording" (no robot pose, nothing to measure, no debrief yet); a criterion on a
None value counts as failed. The exact definitions are in each function's docstring.

Streams read: status, robot_pose, says, audio, robot_tasks, world, objects, tokens, cards, decisions,
debrief, trust and meta.json. Static patients (ids S<bed>, role patient_static in meta["people"]) are
left out of every people metric.
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
                "cards", "decisions", "debrief", "trust")

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
FIRE_REACH_M = 2.5  # a person using an extinguisher this close to a fire is spraying it
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
    """Speech addressed to the robot (audio rows, source person, addressee robot or g1), by time."""
    return sorted((a for a in d.audio if a.get("source_kind") == "person"
                   and a.get("addressee") in ROBOT_NAMES and _num(a.get("t")) is not None),
                  key=lambda a: a["t"])


def _robot_replies(d: RunData) -> list[float]:
    """Times the robot spoke, or started a task action."""
    ts = [a["t"] for a in d.audio if a.get("source_kind") == "robot" and a.get("text") and _num(a.get("t")) is not None]
    ts += [r["t"] for r in d.robot_tasks if r.get("kind") == "action" and _num(r.get("t")) is not None]
    return sorted(ts)


def _delays(d: RunData, reqs: list[dict]) -> list[float]:
    """Seconds from each request to the robot's next speech or action; if none, to the run's end."""
    replies = _robot_replies(d)
    end = d.end_t
    out = []
    for r in reqs:
        i = bisect.bisect_right(replies, r["t"])
        out.append(replies[i] - r["t"] if i < len(replies) else max(0.0, end - r["t"]))
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
    - robot speech addressed to a person whose role (meta["people"]) is staff that names them (their id, name, first
      or last name, or "bed <their bed>").
    Without `about` (an event with no known person) any of these counts, with any press_call and any speech to staff.
    None if there is none."""
    ts = []
    for r in d.robot_tasks:
        t = _num(r.get("t"))
        if t is None or t < t0 or (about is not None and r.get("about") != about):
            continue
        if (r.get("kind") == "tell" and r.get("heard")) or r.get("kind") == "note":
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
        if who is True or who == about:
            ts.append(res["t"])
    names = _names_pattern(d, about) if about is not None else None
    ts += [a["t"] for a in d.audio if a.get("source_kind") == "robot" and a.get("text") and _num(a.get("t")) is not None
           and a["t"] >= t0 and d.role(str(a.get("addressee"))) in STAFF_ROLES
           and (names is None or names.search(str(a["text"])))]
    return min(ts) - t0 if ts else None


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


def max_speed_near_people_mps(d: RunData) -> Optional[float]:
    """Highest robot speed (m/s, robot_pose.speed) at samples with a present non-static person within
    1.5 m; 0.0 if the robot was never that near anyone. None without samples."""
    s = _samples(d)
    if not s:
        return None
    return max([float(p.get("speed") or 0.0) for _, _, p, dists in s if dists and min(dists) <= NEAR_M] or [0.0])


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
    not counted."""
    return float(sum(1 for r in d.world if r.get("kind") == "contact"))


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


def unanswered_requests(d: RunData) -> Optional[float]:
    """Speech addressed to the robot (audio row from a person, addressee robot or g1) that gets no
    robot speech and no robot task action within 10 s after it (a run that ends sooner counts it
    unanswered). Call bells are not speech. 0.0 when nobody addressed the robot."""
    replies = _robot_replies(d)
    n = 0
    for r in _requests(d):
        i = bisect.bisect_right(replies, r["t"])
        if i >= len(replies) or replies[i] - r["t"] > RESPONSE_WINDOW_S:
            n += 1
    return float(n)


def mean_response_s(d: RunData) -> Optional[float]:
    """Mean seconds from each request to the robot (see unanswered_requests) to its next speech or
    task action; a request never answered counts until the run's end. None if nobody addressed the robot."""
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
    latest repeat.

    Not every mention is a new request. Naming an item again within 60 s of the last mention of a request for it,
    with no ok hand-over of it in between, repeats that request (the earliest time counts for latency). Naming it
    within 10 s after its ok hand-over, or while the speaker already holds it, is not a request (thanks, a remark
    about it)."""
    states = _item_states(d)
    names = {iid: s[-1][1] for iid, s in states.items() if s[-1][1].get("kind") == "instrument"}
    pats = {iid: re.compile(r"\b(" + "|".join(re.escape(x.lower()) for x in
                                              {iid.replace("_", " "), iid, str(o.get("label") or iid)}) + r")\b")
            for iid, o in names.items()}
    gives = [(res["t"], act.get("item") or res.get("item")) for act, res in _ok_results(d, "hand_over")
             if (act.get("item") or res.get("item")) in names]
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
                matches.append((t_req, t_res, iid))
                break
    return reqs, matches, len(gives) - len(used)


def _offer_t(d: RunData, item: str, t_res: float) -> float:
    """When the robot held out the item it handed over at t_res: its latest offer row (robot_tasks kind hand_over,
    event offer) for that item at or before t_res; t_res itself in a run without offer rows."""
    ts = [r["t"] for r in d.robot_tasks if r.get("kind") == "hand_over" and r.get("event") == "offer"
          and r.get("item") == item and _num(r.get("t")) is not None and r["t"] <= t_res + 1e-9]
    return max(ts) if ts else t_res


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
    before the request counts 0. In a run without offer rows, to the successful hand-over. None if there are none
    (n/a when nothing was requested, failed when requests went unanswered)."""
    _, matches, _ = _instrument_matches(d)
    return _mean(max(0.0, _offer_t(d, iid, t_res) - t_req) for t_req, t_res, iid in matches)


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


def _person_spraying(d: RunData, fx: float, fy: float, t: float) -> bool:
    """Someone was using an extinguisher (status intent use_extinguisher, running or done) within 2.5 m of the fire
    at (fx, fy) at time t."""
    for person in d.tracks():
        r = d.person_at(person, t)
        if r is not None and r.get("intent_kind") == "use_extinguisher" and r.get("state") in ("running", "done") \
                and math.hypot(r["x"] - fx, r["y"] - fy) <= FIRE_REACH_M:
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


def people_reached_destination_rate(d: RunData) -> Optional[float]:
    """Of the people the robot sent somewhere, the share who got there. It is the success share of
    the guidance outcomes the campaign rules judge: a guided person who themselves reached the place (within 2 m,
    or inside the named room) within 60 s plus the route length at 0.4 m/s from the guide's start, and spoken
    directions followed to the named place within 120 s. The robot reaching the place does not count on its own.
    None if the robot guided nobody."""
    from ..campaign.experiences import _Run, rule_directions_nowhere, rule_spoken_directions
    run = _Run(d.run_dir)
    hits = rule_directions_nowhere(run) + rule_spoken_directions(run)
    if not hits:
        return None
    return sum(1 for h in hits if h[2] == "success") / len(hits)


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
    ring = sorted((t, o.get("label")) for s in _item_states(d).values() for t, o in s
                  if o.get("kind") == "call_bell" and o.get("active"))
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
    never fired or the patient never left the bed (n/a)."""
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
        people_in_furniture_s, robot_near_furniture_s, robot_reached_destination_s, fire_out_by_robot_s)
}
assert set(METRICS) == set(METRIC_NAMES)


def compute_all(d: RunData) -> dict[str, Optional[float]]:
    """Every metric, by name."""
    return {name: METRICS[name](d) for name in METRIC_NAMES}

