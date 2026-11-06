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
ACCOMPANY_M = 2.5  # someone (or the robot) within this distance is "with" a wandering patient
DOOR_ZONE_M = 1.0  # radius of a fire door's swing zone, around the middle of the door
MAX_SAMPLE_GAP_S = 1.0  # a pose sample stands for at most this long
STAFF_ROLES = frozenset({"ward_nurse", "nurse", "staff", "receptionist"})
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


def _card_t(d: RunData, *cards: str) -> Optional[float]:
    ts = [c["t"] for c in d.cards if c.get("card") in cards and c.get("ok", True) and _num(c.get("t")) is not None]
    return min(ts) if ts else None


def _alert_after(d: RunData, t0: float) -> Optional[float]:
    """Seconds from t0 to the robot's first alert of staff: a successful press_call, or robot speech
    addressed to a person whose role (meta["people"]) is staff. None if there is none."""
    ts = [res["t"] for _, res in _ok_results(d, "press_call") if res["t"] >= t0]
    ts += [a["t"] for a in d.audio if a.get("source_kind") == "robot" and a.get("text") and a["t"] >= t0
           and d.role(str(a.get("addressee"))) in STAFF_ROLES]
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
    done = {a.get("item") for a, _ in _ok_results(d, "hand_over")}
    n = 0
    for iid, states in _item_states(d).items():
        last = states[-1][1]
        if last.get("kind") == "tray" and iid in done and last.get("owner") and last.get("holder") == last["owner"]:
            n += 1
    return float(n)


def _instrument_matches(d: RunData):
    """(requests, matches, unrequested): requests are (t, item id) for each instrument a person named
    to the robot; matched to the earliest ok hand_over of that instrument within 60 s after it."""
    states = _item_states(d)
    names = {iid: s[-1][1] for iid, s in states.items() if s[-1][1].get("kind") == "instrument"}
    pats = {iid: re.compile(r"\b(" + "|".join(re.escape(x.lower()) for x in
                                              {iid.replace("_", " "), iid, str(o.get("label") or iid)}) + r")\b")
            for iid, o in names.items()}
    reqs = []
    for a in _requests(d):
        text = str(a.get("text") or "").lower()
        for iid, pat in pats.items():
            if pat.search(text):
                reqs.append((a["t"], iid))
    gives = [(res["t"], act.get("item")) for act, res in _ok_results(d, "hand_over") if act.get("item") in names]
    used: set[int] = set()
    matches = []
    for t_req, iid in reqs:
        for k, (t_res, item) in enumerate(gives):
            if k not in used and item == iid and t_req <= t_res <= t_req + INSTRUMENT_WINDOW_S:
                used.add(k)
                matches.append((t_req, t_res))
                break
    return reqs, matches, len(gives) - len(used)


def instruments_correct_rate(d: RunData) -> Optional[float]:
    """Correct / (requests + unrequested). A request is an instrument named in speech addressed to the
    robot; it is correct when the robot hands over that instrument successfully within 60 s after.
    Successful hand-overs of instruments nobody asked for (or twice) add to the denominator. None if
    no instrument was requested."""
    reqs, matches, extra = _instrument_matches(d)
    return len(matches) / (len(reqs) + extra) if reqs else None


def instrument_mean_latency_s(d: RunData) -> Optional[float]:
    """Mean seconds from an instrument request to the successful hand-over that answers it, over the
    correct ones. None if there are none."""
    _, matches, _ = _instrument_matches(d)
    return _mean(t_res - t_req for t_req, t_res in matches)


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


def people_reached_destination_rate(d: RunData) -> Optional[float]:
    """Of the people the robot sent somewhere, the share who got there. It is the success share of
    the guidance outcomes the campaign rules judge (a guide offer whose goal arrived within 120 s, and
    spoken directions followed to the named place within 120 s). None if the robot guided nobody."""
    from ..campaign.experiences import _Run, rule_directions_nowhere, rule_spoken_directions
    run = _Run(d.run_dir)
    hits = rule_directions_nowhere(run) + rule_spoken_directions(run)
    if not hits:
        return None
    return sum(1 for h in hits if h[2] == "success") / len(hits)


def nurse_alerted_s(d: RunData) -> Optional[float]:
    """Seconds from the first call bell (call_bell_at card, or a bell turning active in objects) or
    patient_gets_up card to the robot's first alert of staff after it (a successful press_call, or robot
    speech addressed to a staff-role person). None if there was no such event or no alert."""
    t0 = _card_t(d, "call_bell_at", "patient_gets_up")
    if t0 is None:
        ring = [t for s in _item_states(d).values() for t, o in s if o.get("kind") == "call_bell" and o.get("active")]
        t0 = min(ring) if ring else None
    return None if t0 is None else _alert_after(d, t0)


def deterioration_alerted_s(d: RunData) -> Optional[float]:
    """Seconds from the deterioration card firing to the robot's first alert of staff (see nurse_alerted_s).
    None if the card never fired or there was no alert."""
    t0 = _card_t(d, "deterioration")
    return None if t0 is None else _alert_after(d, t0)


def specimen_delivered_s(d: RunData) -> Optional[float]:
    """Seconds from the start of the run to the first time a specimen box the robot held (holder g1) is
    held by a person in the objects stream. None if that never happens."""
    best = None
    for s in _item_states(d).values():
        if s[-1][1].get("kind") != "specimen_box":
            continue
        held = False
        for t, o in s:
            h = o.get("holder")
            if h in ROBOT_NAMES:
                held = True
            elif held and h:
                best = t if best is None else min(best, t)
                break
    return best


def patient_left_ward_unaccompanied_s(d: RunData) -> Optional[float]:
    """For the target of the patient_gets_up card: seconds, from their first status row after the card
    in which they are no longer lying until they are gone or the run ends, that neither the robot nor
    any other present person is within 2.5 m. Each of the patient's status rows counts the gap to their
    next row. None if the card never fired or the patient never left the bed."""
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
    total = 0.0
    for i in range(start, len(after)):
        r = after[i]
        if r.get("posture") == "gone":
            break
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
        mood_change, rating_likeability, rating_safety, rating_competence, trust_item_mean, trust_delta_mean)
}
assert set(METRICS) == set(METRIC_NAMES)


def compute_all(d: RunData) -> dict[str, Optional[float]]:
    """Every metric, by name."""
    return {name: METRICS[name](d) for name in METRIC_NAMES}
