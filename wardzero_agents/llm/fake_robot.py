"""The scripted built-in robot (WARDZERO_TEST_CLAUDE_BOT=1 with WARDZERO_TEST_FAKE_LLM=1): a deterministic stand-in for
the Claude robot that does each preset's main task, so the robot's whole wire (task actions, speech and its addressee,
the staff call) is exercised on the real launch path without a model.

Fairness: it reads only what the robot's own prompt gives it, the card (the places it knows, its job and body) and
the situation (its pose, the anonymous tracks its camera sees, objects in view, what it heard, the results of its
actions and its own last events). It never sees a person id, a name or a mood. It recognises the kind of job from the
brief's words (the role and the job), not from a scenario id, and answers with the same decision a model would, in
the robot's decision schema.

It keeps a little memory between calls (what it has started, whom it has spoken to, which requests it answered), as
a model would through its prompt's last events. A new run (another card, or the clock going back) starts afresh.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# ---- reading the prompt -------------------------------------------------------------------------------------------------


@dataclass
class Track:
    id: str
    dist: float
    posture: str
    moving: bool
    x: float
    y: float
    new: bool  # "not spoken to yet"
    line: str
    asleep: bool = False  # the camera marks someone asleep this close


@dataclass
class Heard:
    age: float
    who: str  # to_me, nearby, cough, alarm, pa, own, chime, sound
    text: str
    track: Optional[str]  # the track it came from, when the microphone could tell
    line: str
    db: Optional[float] = None  # how loud it reached the robot


@dataclass
class Result:
    id: str
    what: str
    ok: bool
    reason: str


@dataclass
class Situation:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # degrees
    room: str = ""
    held: list = field(default_factory=list)  # (label, item id)
    heard: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    items: list = field(default_factory=list)  # (id, label, dist)
    fires: list = field(default_factory=list)  # (id, dist)
    beds: list = field(default_factory=list)  # (id, occupied, dist)
    signs: list = field(default_factory=list)
    lights: list = field(default_factory=list)  # bed ids with the call light on
    places: list = field(default_factory=list)  # labels of the places the routes from here list
    route_dirs: dict = field(default_factory=dict)  # label -> the way the route sets off ("ahead and to your left")
    blocked: set = field(default_factory=set)  # labels of the places whose way the robot saw blocked
    route_m: dict = field(default_factory=dict)  # label -> route length from here (0 when right beside it)
    results: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def track(self, tid: Optional[str]) -> Optional[Track]:
        return next((tr for tr in self.tracks if tr.id == tid), None)


def _section(text: str, head: str, *ends: str) -> str:
    if head not in text:
        return ""
    body = text.split(head, 1)[1]
    for e in ends:
        body = body.split(e, 1)[0]
    return body


def parse_situation(u: str) -> Situation:
    s = Situation()
    m = re.search(r"NOW t=([0-9.]+) s\. You are at \(x=(-?[0-9.]+), y=(-?[0-9.]+)\), facing (-?[0-9.]+) degrees, "
                  r"in (.+?)\.\n", u)
    if m:
        s.t, s.x, s.y, s.yaw, s.room = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                        m.group(5))
    held = re.search(r"You are holding: (.+)", u)
    if held:
        s.held = re.findall(r"([^,]+?) \(item (\w+)\)", held.group(1))
        s.held = [(lab.strip(), iid) for lab, iid in s.held]
    for ln in _section(u, "Heard recently, newest last:", "People your camera sees").splitlines():
        m = re.match(r"- ([0-9]+) s ago, (.+?), ([0-9.]+) dB", ln)
        if not m:
            continue
        what = m.group(2)
        who = ("to_me" if "speaking to you" in what else "nearby" if "someone nearby" in what else
               "cough" if "cough" in what else "alarm" if "alarm" in what else "pa" if "announcement" in what else
               "own" if "your own voice" in what else "chime" if "chime" in what else "sound")
        tm = re.search(r"from the direction of (t\w+)", ln)
        txt = re.search(r': "(.*)"$', ln)
        s.heard.append(Heard(float(m.group(1)), who, txt.group(1) if txt else "", tm.group(1) if tm else None, ln,
                             float(m.group(3))))
    for ln in _section(u, "People your camera sees", "Objects in view").splitlines():
        m = re.match(r"- (t\w+): ([0-9.]+) m, .*?, (lying down|seated|standing|[a-z ]+?)(, asleep)?, (moving|still), "
                     r"at \(x=(-?[0-9.]+), y=(-?[0-9.]+)\)", ln)
        if m:
            s.tracks.append(Track(m.group(1), float(m.group(2)), m.group(3), m.group(5) == "moving",
                                  float(m.group(6)), float(m.group(7)), "not spoken to yet" in ln, ln,
                                  asleep=bool(m.group(4))))
    for ln in _section(u, "Objects in view:", "Getting there from ", "Results of your recent actions",
                       "Your last events").splitlines():
        if (m := re.match(r"- item (\w+): (.+), ([0-9.]+) m", ln)):
            s.items.append((m.group(1), m.group(2), float(m.group(3))))
        elif (m := re.match(r"- fire (\w+): ([0-9.]+) m", ln)):
            s.fires.append((m.group(1), float(m.group(2))))
        elif (m := re.match(r"- bed (\w+): (occupied|empty), ([0-9.]+) m", ln)):
            s.beds.append((m.group(1), m.group(2) == "occupied", float(m.group(3))))
        elif ln.startswith("- route blocked sign"):
            s.signs.append(ln)
        elif (m := re.match(r"- call light on at bed (\w+)", ln)):
            s.lights.append(m.group(1))
    for ln in _section(u, "Getting there from ", "Results of your recent actions", "Your last events").splitlines():
        if (m := re.match(r"- (.+?): (?:about ([0-9]+) m|(right beside you)|the way there is blocked)", ln)):
            s.places.append(m.group(1))
            if m.group(2) or m.group(3):
                s.route_m[m.group(1)] = float(m.group(2) or 0.0)
            way = re.search(r"setting off (.+?)(?:, through |\.$)", ln)
            if way:
                s.route_dirs[m.group(1)] = way.group(1)
            if "blocked" in ln:
                s.blocked.add(m.group(1))
    for ln in _section(u, "Results of your recent actions:", "Your last events").splitlines():
        m = re.match(r"- (\S+) (.+?): (ok|failed) \((.*)\)$", ln)
        if m:
            s.results.append(Result(m.group(1), m.group(2), m.group(3) == "ok", m.group(4)))
    s.events = [ln[2:] for ln in _section(u, "Your last events, oldest first:", "Decide what").splitlines()
                if ln.startswith("- ")]
    return s


@dataclass
class Card:
    role: str = ""
    job: str = ""
    brief: dict = field(default_factory=dict)  # "Destination" -> "Ward nurse base", ...
    lists: dict = field(default_factory=dict)  # "Meals" -> ["patient bed: ...; tray label: ...", ...]
    places: dict = field(default_factory=dict)  # label (lower) -> label, every place the robot can name
    rooms: dict = field(default_factory=dict)  # room id -> label
    route_dirs: dict = field(default_factory=dict)  # label -> the way the route from the post sets off

    @property
    def words(self) -> str:
        return f"{self.role} {self.job}".lower()


def parse_card(card: str) -> Card:
    c = Card()
    hosp = _section(card, "THE HOSPITAL", "YOUR JOB")
    for m in re.finditer(r"^- (\S+): (.+?) \((\w+)\)$", hosp, re.M):
        c.rooms[m.group(1)] = m.group(2)
        c.places[m.group(2).lower()] = m.group(2)
    for m in re.finditer(r"^- (.+?): (?:about [0-9]+ m|right beside you)(.*)$", hosp, re.M):
        c.places.setdefault(m.group(1).lower(), m.group(1))
        way = re.search(r"setting off (.+?)(?:, through |\.$)", m.group(2))
        if way:
            c.route_dirs[m.group(1)] = way.group(1)  # from the robot's post, facing its post heading
    job = _section(card, "YOUR JOB", "YOUR BODY")
    key = None
    for ln in job.splitlines():
        if ln.startswith("- ") and key:
            c.lists.setdefault(key, []).append(ln[2:])
        elif (m := re.match(r"([A-Z][\w ]*?):\s*(.*)$", ln)):
            key = m.group(1)
            if m.group(2):
                c.brief[key] = m.group(2).strip()
    c.role, c.job = c.brief.get("Role", ""), c.brief.get("Job", "")
    return c


def _bearing(line: str) -> Optional[float]:
    """The bearing a heard line gives, in degrees to the left of straight ahead, or None."""
    if "from straight ahead" in line:
        return 0.0
    m = re.search(r"from about ([0-9]+) degrees to your (left|right)", line)
    return None if m is None else float(m.group(1)) * (1 if m.group(2) == "left" else -1)


_SECTOR_DEG = {"straight ahead": 0.0, "ahead and to your left": 45.0, "ahead and to your right": -45.0,
               "to your left": 90.0, "to your right": -90.0, "behind you to the left": 135.0,
               "behind you to the right": -135.0, "behind you": 180.0}


def _sector(rel_deg: float) -> str:
    """The words the robot's prompt uses for a direction relative to its heading."""
    a = abs(rel_deg)
    side = "left" if rel_deg > 0 else "right"
    if a < 22.5:
        return "straight ahead"
    if a < 67.5:
        return f"ahead and to your {side}"
    if a < 112.5:
        return f"to your {side}"
    if a < 157.5:
        return f"behind you to the {side}"
    return "behind you"


# ---- the policy ---------------------------------------------------------------------------------------------------------
GREET = "Hello, I am the hospital robot. Can I help you?"
WAIT_WORDS = ("wait", "how long", "seen", "queue", "my turn")
UNWELL_WORDS = ("dizzy", "pass out", "faint", "breath", "chest", "worse", "unwell", "can't breathe", "collapse",
                "help me")
STAFF_WORDS = ("nurse", "doctor", "member of staff", "a person", "someone who can help", "real person")
REPEAT_WORDS = ("again", "repeat", "pardon", "sorry?", "didn't catch", "did not catch", "what did you say")
PLACE_ASKS = ("where", "find", "help", "lost", "looking for", "need to get", "way to", "take me", "show me", "go to",
              "get to")


def _decision(action: str = "none", say: str = "", toward: str = "", target: str = "", item: str = "",
              destination: str = "", message: str = "", seat: bool = False, reason: str = "", **extra) -> dict:
    d = {"say": say, "toward_track": toward, "action": action, "target": target, "item": item,
         "destination": destination, "place": [], "message": message, "seat": seat,
         "reason": reason or "scripted robot"}
    d.update(extra)
    return d


@dataclass
class _Pending:
    action: str
    t: float
    track: str = ""
    aid: Optional[str] = None
    extra: dict = field(default_factory=dict)


class ScriptedRobot:
    """One run's scripted robot. ``decide`` takes the request a RobotMind sends and returns its decision."""

    MAX_BUSY_S = 150.0  # a walk or a hand-over that never reports back is given up after this long

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.card_text: Optional[str] = None
        self.last_t = -1.0
        self.seen_results: set = set()
        self.pending: Optional[_Pending] = None
        self.answered: list = []  # (t, text) of lines spoken to the robot that it answered
        self.spoken: set = set()  # tracks it has said something to
        self.mem: dict = {}
        self.last_say = ""
        self.spoke_t: dict = {}  # track -> when the robot last spoke to them
        self.first_seen: dict = {}  # track -> when the camera first listed them
        self.last_seen: dict = {}  # track -> (x, y, moving, t, line) when the camera last listed them
        self.now = 0.0

    # ---- bookkeeping ----------------------------------------------------------------------------------------------
    def _new_results(self, s: Situation) -> list[Result]:
        out = []
        for r in s.results:
            key = (r.id, r.what, r.ok, r.reason)
            if key not in self.seen_results:
                self.seen_results.add(key)
                out.append(r)
        return out

    def _ids(self, s: Situation) -> None:
        """The action id the mind gave the pending action ("You started guide as rb3 ...")."""
        p = self.pending
        if p is None or p.aid is not None:
            return
        for e in reversed(s.events):
            m = re.match(r"t=([0-9.]+)s You started (\w+) as (\S+?)(?: for track| ?\.)", e)
            if m and float(m.group(1)) >= p.t - 0.01:
                p.aid = m.group(3)
                return

    def _finished(self, new: list[Result]) -> Optional[Result]:
        """The result that ends the pending action, if it came."""
        p = self.pending
        if p is None:
            return None
        for r in new:
            if p.action == "go_to" and r.what == "walking to a goal" and not r.id.startswith("rb"):
                return r
            if p.aid is None or r.id != p.aid:
                continue
            if p.action in ("guide", "approach", "follow"):
                if r.what == "walking to a goal" or not r.ok or r.reason in ("already_there", "arrived"):
                    return r
            elif r.reason not in ("started", "following"):
                return r
        return None

    def _start(self, s: Situation, d: dict, **extra) -> dict:
        if d["action"] not in ("none", "wait"):
            if d["action"] == "guide":  # the robot walks itself back to its post after the guide, unless told not to
                extra = {**extra, "back": d.get("return_to_post", True) is not False}
            self.pending = _Pending(d["action"], s.t, d.get("target") or "", extra=extra)
        if d["action"] == "guide" and d.get("target"):
            self.mem.setdefault("helped", set()).add(d["target"])
            self.mem[("guide_to", d["target"])] = d.get("destination") or ""
        return self._said(d)

    def _said(self, d: dict) -> dict:
        if d.get("say"):
            self.last_say = d["say"]
            if d.get("toward_track") and d["toward_track"] != "room":
                self.spoken.add(d["toward_track"])
                self.spoke_t[d["toward_track"]] = self.now
        return d

    def _requests(self, s: Situation) -> list[Heard]:
        """Lines spoken to the robot in the last few seconds that it has not answered yet."""
        out = []
        for h in s.heard:
            if h.who != "to_me" or h.age > 4 or not h.text:
                continue
            t = s.t - h.age
            if any(abs(t - at) <= 2.5 and txt == h.text for at, txt in self.answered):
                continue
            out.append(h)
        return out

    def _answered(self, s: Situation, h: Heard) -> None:
        self.answered.append((s.t - h.age, h.text))

    BEARING_OK_DEG = 25.0  # a line is put down to the track nearest its bearing within this

    VOICE_DB = 60.0  # a person's speaking voice at 1 m: how loud it arrives says roughly how far away they are

    WALK_MPS = 1.5  # how fast someone on the move may have come nearer since they spoke

    def _loud_enough(self, h: Heard, tr: Track) -> bool:
        """Whether a voice this loud can come from someone this far away (within a factor of 2.5 either way); someone
        on the move may have walked up since they spoke, so they may be that much nearer now."""
        if h.db is None:
            return True
        est = 10.0 ** ((self.VOICE_DB - h.db) / 20.0)
        nearer = (h.age + 1.0) * self.WALK_MPS if tr.moving else 0.0
        return est / 2.5 - nearer <= max(tr.dist, 1.0) <= est * 2.5

    def _loudness_misfit(self, h: Heard, tr: Track) -> float:
        """How far this person's distance is from the one the voice's loudness gives: 0 a perfect fit, 1 a factor of
        2.5 off."""
        if h.db is None:
            return 0.0
        est = 10.0 ** ((self.VOICE_DB - h.db) / 20.0)
        return abs(math.log(max(tr.dist, 1.0) / max(est, 1.0))) / math.log(2.5)

    def _who(self, s: Situation, h: Heard) -> Optional[str]:
        """The track a line came from: the microphone's match when the loudness fits, else the track nearest the
        line's bearing, else the nearest person within 4 m."""
        tr = s.track(h.track) if h.track else None
        if tr is not None and self._loud_enough(h, tr):
            return h.track
        if tr is not None:
            return None  # the voice came from that way but from farther (or nearer) than them: someone unseen
        rel = _bearing(h.line)
        if rel is not None and s.tracks:
            off = [(abs((math.degrees(math.atan2(tr.y - s.y, tr.x - s.x)) - s.yaw - rel + 180.0) % 360.0 - 180.0),
                    tr) for tr in s.tracks]
            fits = [(not tr.moving, err / self.BEARING_OK_DEG + self._loudness_misfit(h, tr), tr.id)
                    for err, tr in off if err <= self.BEARING_OK_DEG and self._loud_enough(h, tr)]
            # the one walking up to the robot, when there is one that way; else the one whose direction and distance
            # fit the voice best (the bearing is only good to a few degrees, so how loud it was counts as much)
            return min(fits)[2] if fits else None  # (None: nobody in view that way at a distance the voice fits)
        near = sorted(s.tracks, key=lambda tr: tr.dist)
        return near[0].id if near and near[0].dist <= 4.0 else None

    # ---- the place vocabulary -------------------------------------------------------------------------------------
    def place_for(self, text: str, c: Card) -> Optional[str]:
        """The place label a request names: the brief's destination when the request fits it, else the known place
        that shares the most words with the request."""
        low = text.lower()
        words = {w for w in re.findall(r"[a-z]+", low) if len(w) > 3}
        words |= {w[:-1] for w in words if w.endswith("s")}

        def score(label: str) -> int:
            lw = {w for w in re.findall(r"[a-z]+", label.lower()) if len(w) > 3}
            lw |= {w[:-1] for w in lw if w.endswith("s")}
            return len(words & lw)

        dest = c.brief.get("Destination")
        if dest and score(dest) > 0:
            return dest
        best = sorted(((score(lab), -len(lab), lab) for lab in c.places.values()), reverse=True)
        return best[0][2] if best and best[0][0] > 0 else None

    EXIT_KEYS = ("Destination", "Safe exit", "Exit", "Way out")

    def destination(self, c: Card) -> Optional[str]:
        """Where the job sends the robot: a destination or exit line of the brief, a capitalised place the job walks
        "to the", else the known place the job's words name most (the latest named, on a tie)."""
        for key in self.EXIT_KEYS:
            text = c.brief.get(key)
            if not text:
                continue
            m = re.search(r"\b(?:is|to|by) the ([A-Z][\w&' -]+?)(?: in | \(|,|\.|$)", text)
            if m:
                return m.group(1).strip()
            named = self._named_places(text, c)
            if named:
                return named
            return text.strip().rstrip(".")
        m = re.search(r"\bto the ([A-Z][\w ]+?)(?:\(|,|\.|$)", c.job)
        if m:
            return m.group(1).strip()
        return self._named_places(c.job, c)

    @staticmethod
    def _words(text: str) -> set:
        ws = {w for w in re.findall(r"[a-z]+", text.lower().replace("_", " ")) if len(w) > 3}
        return ws | {w[:-1] for w in ws if w.endswith("s")}

    def _named_places(self, text: str, c: Card) -> Optional[str]:
        """The known place the text names last (a job walks from where it starts to where it ends): a place counts as
        named when at least two of its label words (or its only one) appear."""
        low = text.lower()
        best = None
        for lab in c.places.values():
            words = self._words(lab)
            hits = [w for w in words if re.search(rf"\b{re.escape(w)}", low)]
            if not hits or len(hits) < min(2, len({w.rstrip("s") for w in words})):
                continue
            last = max(m.start() for w in hits for m in re.finditer(rf"\b{re.escape(w)}", low))
            key = (last, len(hits))
            if best is None or key > best[0]:
                best = (key, lab)
        return best[1] if best else None

    # ---- entry point ------------------------------------------------------------------------------------------------
    def decide(self, card_text: str, situation: str, allowed: list[str]) -> dict:
        s = parse_situation(situation)
        self.raw = situation
        if card_text != self.card_text or s.t < self.last_t - 1.0:
            self.reset()
            self.card_text = card_text
        self.last_t = self.now = s.t
        c = self.card = parse_card(card_text or "")
        self.mem.setdefault("post", (s.x, s.y, s.yaw))  # where it starts is its post
        if "post_place" not in self.mem and s.route_m:  # the place right beside its post, to walk back to
            px, py, _ = self.mem["post"]
            if math.hypot(s.x - px, s.y - py) <= 0.5:
                self.mem["post_place"] = next((lab for lab, m in s.route_m.items() if m == 0.0), None)
        for tr in s.tracks:
            self.first_seen.setdefault(tr.id, s.t)
            self.last_seen[tr.id] = (tr.x, tr.y, tr.moving, s.t, tr.line)
        for lab in s.places:  # the routes from here name the places the robot can walk to
            c.places.setdefault(lab.lower(), lab)
        self.allowed = allowed or ["none"]
        self._ids(s)
        before = self.pending
        new = self._new_results(s)
        done = self._finished(new)
        if self.pending is not None and (done is not None or s.t - self.pending.t > self.MAX_BUSY_S):
            self.mem.setdefault("finished", []).append((self.pending, done))
            self.on_finished(s, c, self.pending, done)
            self.pending = None
        for r in new:
            self.on_result(s, c, r)
        job = c.words
        if ("put out" in job or "extinguish" in job) and "evacuat" not in job:
            d = self.fire(s, c)
        elif "instrument" in job or "scrub" in job:
            d = self.scrub(s, c)
        elif "specimen" in job or (c.lists.get("Items") or c.brief.get("Items")) and c.brief.get("Destination"):
            d = self.courier(s, c)
        elif "tray" in job or "meal" in job:
            d = self.meals(s, c)
        elif "evacuat" in job or "alarm" in job:
            d = self.evacuate(s, c)
        elif "leave the ward" in job or "gets up" in job or "companion" in job:
            d = self.companion(s, c)
        elif "call bell" in job:
            d = self.bell_job(s, c)
        elif "crowd" in job or "runner" in job:
            d = self.runner(s, c)
        elif "check people in" in job or "waiting room" in job or "a&e" in job:
            d = self.waiting_room(s, c)
        else:
            d = self.reception(s, c)
        if d["action"] not in self.allowed:
            d = {**d, "action": "none"}
            if self.pending is not None and self.pending is not before and self.pending.action not in self.allowed:
                self.pending = None  # never sent: nothing to wait for
        return d

    # hooks the jobs use
    def on_finished(self, s: Situation, c: Card, p: _Pending, r: Optional[Result]) -> None:
        if p.action == "go_to" and r is not None:
            if r.ok:
                self.mem["arrived"] = self.mem["there"] = True
            else:
                self.mem["goal_failed"] = self.mem.get("goal_failed", 0) + 1
                self.mem["goal_fail_reason"] = r.reason
        if p.action == "hand_over" and self.mem.get("offer_room"):
            self.mem.setdefault(("tries", self.mem["offer_room"]), []).append(r.reason if r else "no_answer")
        if p.action == "hand_over" and r is not None and r.reason == "not_holding" and p.extra.get("item"):
            self.mem["take_then_offer"] = (p.extra["item"], p.track)  # this body takes first, then holds it out
        if p.action == "take" and (r is None or not r.ok):
            self.mem.pop("take_then_offer", None)
        if p.action == "guide" and r is not None and not r.ok and p.track:
            self.mem.setdefault("walk_on", []).append(p.track)  # held up short of the place: send them on
        post = self.mem.get("post")
        if (p.aid and r is not None and r.ok and post is not None and p.extra.get("back", True)
                and p.action in ("guide", "tell_staff") and math.hypot(s.x - post[0], s.y - post[1]) > self.POST_M):
            # a guide that got there, or a walk to tell the staff: the robot walks itself back to its post (goal
            # "<id>-return"), so it leaves that walk alone until it reports
            self.mem["returning"] = (p.aid, s.t)

    def on_result(self, s: Situation, c: Card, r: Result) -> None:
        if r.what == "press_call":
            self.mem["call"] = r.reason
        back = self.mem.get("returning")
        if back is not None and r.id == f"{back[0]}-return":
            self.mem.pop("returning")  # the walk back reported: at the post, or stopped short (then it walks itself)

    # ---- shared skills ------------------------------------------------------------------------------------------------
    def busy(self) -> bool:
        return self.pending is not None

    def reply(self, s: Situation, c: Card, h: Heard, *, walk_with: bool = True) -> dict:
        """An answer to a line spoken to the robot: walk them where they need to go, fetch staff, the wait, or a
        short word back."""
        self._answered(s, h)
        who = self._who(s, h) or ""
        low = h.text.lower()
        if any(w in low for w in REPEAT_WORDS) and self.last_say:
            return self._said(_decision(say=self.last_say, toward=who, reason="asked to repeat"))
        if any(w in low for w in UNWELL_WORDS) and who:
            return self.alert(s, c, who, "I am getting a nurse for you now. Please stay where you are.")
        if any(w in low for w in STAFF_WORDS) and who and "tell_staff" in self.allowed:
            return self._start(s, _decision("tell_staff", "I will let the staff know straight away. Please wait "
                                            "here.", who, who, message="Excuse me, a visitor here needs a nurse, "
                                            "please.", reason="asked for staff"))
        if any(w in low for w in WAIT_WORDS) and (c.brief.get("Waits") or self.mem.get("wait_line")):
            return self._said(_decision(say=self.mem.get("wait_line") or c.brief["Waits"].split(";")[0] + ".",
                                        toward=who, reason="asked about the wait"))
        place = self.place_for(h.text, c) if any(w in low for w in PLACE_ASKS) else None
        if place and who and walk_with and "guide" in self.allowed:
            return self._start(s, _decision("guide", f"Of course. Please come with me, I will walk with you to {place}.",
                                            who, who, destination=place, reason="asked the way"))
        if ("help" in low or "lost" in low or "find" in low) and who:  # help, but not where to: ask
            return self._said(_decision(say="Of course. Where would you like to go?", toward=who,
                                        reason="asked for help"))
        return self._said(_decision(say="I'm here if you need me.", toward=who, reason="answered"))

    def alert(self, s: Situation, c: Card, who: str, say: str) -> dict:
        """Someone needs staff now: the staff call naming them (the robot stays), or tell_staff when there is none."""
        self.mem.setdefault("alerted", set()).add(who)
        if "press_call" in self.allowed:
            return self._start(s, _decision("press_call", say, who, who, reason="needs staff now"))
        return self._start(s, _decision("tell_staff", say, who, who, message="Please come quickly, this person "
                                        "needs help.", reason="needs staff now"))

    SPEAK_M = 3.5  # speech is meant for one person only this close: farther, walk up first

    TALKING_S = 20.0  # someone spoken to this recently and still close may be about to ask something
    ARRIVED_AFTER_S = 10.0  # a track first seen after this is a new arrival, worth walking up to greet

    def speak_to(self, s: Situation, tr: Track, line: str, why: str, urgent: bool = False) -> Optional[dict]:
        """Say line to one person: at once when near enough, else walk up to them first (and say it on arrival).
        None when it is too far to say and the robot will not walk off now: it is busy, or it is still talking with
        someone close by."""
        if tr.dist <= self.SPEAK_M:
            return self._said(_decision(say=line, toward=tr.id, reason=why))
        talking = any(o.id != tr.id and o.dist <= self.SPEAK_M and s.t - self.spoke_t.get(o.id, -1e9) <= self.TALKING_S
                      for o in s.tracks)
        settled = why == "greeting" and (self.first_seen.get(tr.id, 0.0) <= self.ARRIVED_AFTER_S
                                         or tr.posture != "standing")
        if "approach" not in self.allowed or self.busy() or ((talking or settled) and not urgent):
            return None  # (someone already there when the robot came on post is greeted once they come near)
        self.mem["then_say"] = (tr.id, line, why)
        return self._start(s, _decision("approach", "", "", tr.id, reason=f"walking up to them: {why}"),
                           minor=why == "greeting")

    def then_say(self, s: Situation) -> Optional[dict]:
        """The line it walked up to someone to say."""
        if self.busy() or not self.mem.get("then_say"):
            return None
        tid, line, why = self.mem.pop("then_say")
        tr = s.track(tid)
        if tr is None:
            return None
        tries = self.mem.setdefault(("walked_up", tid), 0)
        if tr.dist > self.SPEAK_M and tries < 2 and "approach" in self.allowed:  # the walk stopped short: again
            self.mem[("walked_up", tid)] = tries + 1
            self.mem["then_say"] = (tid, line, why)
            return self._start(s, _decision("approach", "", "", tid, reason=f"walking up to them: {why}"),
                               minor=why == "greeting")
        return self._said(_decision(say=line, toward=tid, reason=why))

    def greet(self, s: Situation, line: str = GREET) -> Optional[dict]:
        """Hello to the nearest person not spoken to yet: at once when near enough for a line to be meant for them,
        else walking up first."""
        # someone up and about first (a seated person is settled: staff at a desk, someone waiting), nearest first
        new = sorted((tr for tr in s.tracks if tr.new and tr.id not in self.spoken),
                     key=lambda tr: (tr.posture != "standing", tr.dist))
        return next((d for d in (self.speak_to(s, tr, line, "greeting") for tr in new[:1]) if d is not None), None)

    def go(self, s: Situation, place: str, say: str = "", toward: str = "", why: str = "") -> dict:
        p = self.pending
        if p is not None and p.action == "go_to" and p.extra.get("place") == place:
            return _decision(reason=f"on my way to {place}")
        return self._start(s, _decision("go_to", say, toward, destination=place, reason=why or f"going to {place}"),
                           place=place)

    def room_of_bed(self, c: Card, bed: str) -> Optional[str]:
        if bed in c.rooms:
            return c.rooms[bed]
        return next((lab for rid, lab in c.rooms.items() if lab.endswith(f" {bed}")), None)

    def default(self, s: Situation, c: Card) -> dict:
        req = self._requests(s)
        if req:
            return self.reply(s, c, req[0], walk_with=not self.busy())
        return self.then_say(s) or _decision(reason="carrying on")

    DEFER_KEEP_S = 180.0  # a request from someone the robot could not see is looked for this long

    def defer(self, s: Situation, h: Heard) -> dict:
        """A request while busy: a word now, and the help once free. Where the voice came from is kept, so the
        speaker can be found again when the robot could not see them."""
        self._answered(s, h)
        who = self._who(s, h) or ""
        rel = _bearing(h.line)
        where = None if rel is None else (s.x, s.y, s.yaw + rel)
        tr = s.track(h.track) if h.track else None
        if where is None and tr is not None:  # "from the direction of t1": that track's way
            where = (s.x, s.y, math.degrees(math.atan2(tr.y - s.y, tr.x - s.x)))
        self.mem.setdefault("deferred", []).append((who, h, where, s.t))
        return self._said(_decision(say="I'll be with you in a moment.", toward=who, reason="busy"))

    SEATED_FIT = 1.4  # someone seated and still is the speaker only when their distance fits the voice this closely

    def _by_direction(self, s: Situation, where, h: Optional[Heard] = None) -> Optional[Track]:
        """The track that lies along the direction a voice came from (seen from where the robot stood then), someone
        up and about first. Someone seated and still has not moved since: they are taken for the speaker only when
        their distance from where the robot heard it fits how loud the voice was."""
        if where is None:
            return None
        x0, y0, deg = where
        est = None if h is None or h.db is None else 10.0 ** ((self.VOICE_DB - h.db) / 20.0)
        helped = self.mem.get("helped", set())
        best = None
        for tr in s.tracks:
            err = abs((math.degrees(math.atan2(tr.y - y0, tr.x - x0)) - deg + 180.0) % 360.0 - 180.0)
            if tr.id in helped:
                continue  # someone it has already walked somewhere is not the one still waiting for an answer
            settled = tr.posture.startswith("seated") and not tr.moving
            if settled and est is not None:
                d0 = max(math.hypot(tr.x - x0, tr.y - y0), 1.0)
                if not est / self.SEATED_FIT <= d0 <= est * self.SEATED_FIT:
                    continue
            key = (settled, err)
            if err <= self.BEARING_OK_DEG and (best is None or key < best[0]):
                best = (key, tr)
        return best[1] if best else None

    def deferred(self, s: Situation, c: Card) -> Optional[dict]:
        """Back to someone who asked while the robot was busy."""
        keep = []
        try:
            while self.mem.get("deferred"):
                who, h, where, t0 = self.mem["deferred"].pop(0)
                tr = s.track(who) or self._by_direction(s, where, h)
                if tr is None:
                    if s.t - t0 <= self.DEFER_KEEP_S:
                        keep.append((who, h, where, t0))  # not in view yet: look again later
                    continue
                return self._back_to(s, c, tr, h, where, t0)
            return None
        finally:
            self.mem["deferred"] = keep + self.mem.get("deferred", [])

    def _back_to(self, s: Situation, c: Card, tr: Track, h: Heard, where, t0: float) -> Optional[dict]:
        """Walk up to the person who asked (named by their track now), then answer them."""
        who = tr.id
        if tr.dist > self.SPEAK_M and "approach" in self.allowed:
            self.mem["deferred"].insert(0, (who, h, None, t0))
            if self.free() and not tr.moving:  # someone on the move (maybe coming over) is waited for
                return self._start(s, _decision("approach", "", "", who, reason="back to someone who asked"))
            return None
        return self.reply(s, c, Heard(0.0, "to_me", h.text, who, h.line))

    def free(self) -> bool:
        """Nothing under way, or only a walk that can be dropped for someone who asks (a greeting, a look round)."""
        return self.pending is None or bool(self.pending.extra.get("minor"))

    def look_toward(self, s: Situation) -> Optional[dict]:
        """Turn towards an unseen person who asked something: walk a little way to a known place that lies their
        way (the robot has no turn-on-the-spot action), so they come into view."""
        waiting = [w for w in self.mem.get("deferred", []) if w[2] is not None]
        if not waiting or not self.free() or s.t - self.mem.get("looked_t", -1e9) < 6.0 or self._returning(s):
            return None  # (walking itself back after a guide turns it round anyway: a go_to would cancel that)
        _, _, (x0, y0, deg), _ = waiting[0]
        return self._go_facing(s, deg, "turning to see who spoke")

    def _go_facing(self, s: Situation, deg: float, why: str) -> Optional[dict]:
        """A short walk towards the known place whose route sets off nearest the heading deg (world degrees), so what
        lies that way comes into view. None when no place lies within 50 degrees of it."""
        # each known place's heading: from the routes from here, else from the routes from the post (its heading)
        if s.route_dirs:
            heads = {lab: s.yaw + _SECTOR_DEG[way] for lab, way in s.route_dirs.items() if way in _SECTOR_DEG}
        else:
            post = self.mem.get("post", (s.x, s.y, s.yaw))
            heads = {lab: post[2] + _SECTOR_DEG[way] for lab, way in self.card.route_dirs.items()
                     if way in _SECTOR_DEG}
        off = sorted((abs((h - deg + 180.0) % 360.0 - 180.0), lab) for lab, h in heads.items())
        if not off or off[0][0] > 50.0:
            return None
        self.mem["looked_t"] = s.t
        return self._start(s, _decision("go_to", destination=off[0][1], reason=why), place=off[0][1], minor=True)

    @staticmethod
    def _nearest(s: Situation) -> Optional[Track]:
        near = sorted(s.tracks, key=lambda tr: tr.dist)
        return near[0] if near and near[0].dist <= 4.0 else None

    # ---- jobs -------------------------------------------------------------------------------------------------------
    def fire(self, s: Situation, c: Card) -> dict:
        """Put the fire out with its own extinguisher, then keep clear."""
        if self.busy():
            return self.default(s, c)
        if s.fires:
            fid = min(s.fires, key=lambda f: f[1])[0]
            return self._start(s, _decision("extinguish", "Fire. Please keep back, I am putting it out.", "room", fid,
                                            reason="a fire to put out"))
        return self.default(s, c)

    def scrub(self, s: Situation, c: Card) -> dict:
        """Pass the instrument the surgeon names: hand it straight over (the hand-over picks it off the tray), or take
        it first when the body says it is not holding it."""
        names = self.mem.setdefault("instruments", self._instruments(c))
        held = {iid for _, iid in s.held}
        if self.pending is None and self.mem.get("take_then_offer"):
            item, who = self.mem["take_then_offer"]
            if item not in held:
                return self._start(s, _decision("take", "", "", item=item, reason="taking it off the tray"))
            self.mem.pop("take_then_offer")
            return self._start(s, _decision("hand_over", "", who, who, item=item, reason="passing what was asked"),
                               item=item)
        for h in self._requests(s):
            low = h.text.lower()
            ask = next((iid for spoken, iid in names if spoken in low), None)
            if ask is None:
                continue
            self._answered(s, h)
            near = self._nearest(s)  # she stands beside the robot: the nearest person, if the voice gave no track
            who = self._who(s, h) or (near.id if near else "")
            label = next(sp for sp, iid in names if iid == ask)
            self.mem["asked"] = (ask, who)
            return self._start(s, _decision("hand_over", f"{label.capitalize()}.", who, who, item=ask,
                                            reason="the surgeon asked for it"), item=ask)
        return _decision(reason="ready for the next instrument")

    def _instruments(self, c: Card) -> list[tuple[str, str]]:
        """(spoken name, item id) of each instrument the brief lists ("suture_scissors (suture scissors)")."""
        out = []
        for part in re.split(r",\s*", c.brief.get("Instruments", "")):
            m = re.match(r"(\w+)(?: \((.+)\))?", part.strip())
            if m:
                out.append(((m.group(2) or m.group(1).replace("_", " ")).lower(), m.group(1)))
        return sorted(out, key=lambda p: -len(p[0]))

    def courier(self, s: Situation, c: Card) -> dict:
        """Carry what it holds to the destination and hand it to the person waiting there."""
        dest = self.destination(c)
        if self.busy() or not s.held:
            return self.default(s, c)
        item = s.held[0][1]
        if not self.mem.get("there"):
            if self.mem.get("goal_failed", 0) >= 3:
                self.mem["there"] = True  # as near as it can get
            else:
                if self.mem.get("goal_fail_reason") == "blocked_by_person":
                    self.mem["goal_fail_reason"] = None
                    return self.go(s, dest, "Excuse me, may I come past, please?", "room", "carrying on")
                return self.go(s, dest, why="carrying the item")
        tried = self.mem.setdefault("tried", [])
        # someone standing at the porter's place: the camera names the place a track stands at (", at Porter waiting
        # point" after its position), never who they are, so only that place part of the line is read
        marked = [tr for tr in s.tracks if "porter" in tr.line.split(")", 1)[-1].lower() and tr.id not in tried]
        cand = marked or sorted((tr for tr in s.tracks if tr.id not in tried and tr.posture == "standing"),
                                key=lambda tr: tr.dist)
        if not cand:
            if tried and s.t - self.mem.get("tried_t", 0) > 20:
                self.mem["tried"] = []
            return self.default(s, c)
        tr = cand[0]
        if tr.dist > 1.2 and self.mem.get("walked_up") != tr.id and "approach" in self.allowed:
            self.mem["walked_up"] = tr.id
            return self._start(s, _decision("approach", "", "", tr.id, reason="walking up to hand it over"))
        tried.append(tr.id)
        self.mem["tried_t"] = s.t
        what = (c.brief.get("Items") or "the item").strip()
        return self._start(s, _decision("hand_over", f"Hello, I have the {what} for the porter. Here it is.",
                                        tr.id, tr.id, item=item, reason="handing it over"))

    def meals(self, s: Situation, c: Card) -> dict:
        """Each tray to its patient's bed, skip a sleeper and tell a nurse, and answer a call bell on the way."""
        self.notice_bells(s, c)
        if self.mem.get("bell") and (not self.busy() or self.pending.action == "go_to"):
            bell = self.bell(s, c)  # a walk to the next bed can wait; a tray being held out cannot
            if bell is not None:
                return bell
        if self.busy():
            return self.default(s, c)
        held = {lab.lower(): iid for lab, iid in s.held}
        todo = []
        for row in c.lists.get("Meals", []):
            bed = re.search(r"bed: ([^;]+)", row)
            label = re.search(r"tray label: (.+)$", row)
            if not bed or not label:
                continue
            iid = held.get(label.group(1).strip().lower())
            room = bed.group(1).strip()
            if iid and room not in self.mem.setdefault("skipped", set()):
                todo.append((room, iid, label.group(1).strip()))
        if not todo:
            return self.default(s, c)
        order = self.mem.setdefault("order", self._route_order(c, [t[0] for t in todo]))
        todo.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
        room, iid, label = todo[0]
        if s.room != room:
            return self.go(s, room, why=f"taking the tray to {room}")
        patient = self._bed_patient(s)
        if patient is None:
            return self.default(s, c)
        tries = self.mem.setdefault(("tries", room), [])
        last = tries[-1] if tries else None
        if (patient.asleep or last in ("no_answer", "refused")
                or tries.count("out_of_reach_receiver") >= 2):
            self.mem["skipped"].add(room)
            return self._start(s, _decision("tell_staff", "", "", patient.id, message=f"The patient in {room} is "
                                            "asleep, so I have kept their lunch tray for later. Could you check on "
                                            "them, please?", reason="the patient is asleep"))
        if (patient.dist > 1.3 or last == "out_of_reach_receiver") and tries.count("approached") < 1:
            tries.append("approached")
            return self._start(s, _decision("approach", "", "", patient.id, reason="closer to the bed"))
        self.mem["offer_room"] = room
        return self._start(s, _decision("hand_over", f"Hello, here is your lunch: {label}.", patient.id, patient.id,
                                        item=iid, reason="delivering a tray"))

    def _bed_patient(self, s: Situation) -> Optional[Track]:
        lying = sorted((tr for tr in s.tracks if tr.posture.startswith("lying") or tr.posture == "seated"),
                       key=lambda tr: tr.dist)
        return lying[0] if lying and lying[0].dist <= 4.0 else None

    def _route_order(self, c: Card, rooms: list[str]) -> list[str]:
        """Nearest first, by the card's route lengths from the post."""
        dist = {}
        for lab in rooms:
            m = re.search(rf"^- {re.escape(lab)}: about ([0-9]+) m", self.card_text or "", re.M)
            dist[lab] = float(m.group(1)) if m else 99.0
        return sorted(rooms, key=lambda r: dist[r])

    def notice_bells(self, s: Situation, c: Card) -> None:
        """Note a call bell whatever the robot is doing: a call light in view names the bed; a chime alone is placed
        by its bearing, against the patients the robot has met in their bedrooms."""
        if s.room and s.room in c.rooms.values():
            pt = self._bed_patient(s)
            if pt is not None and pt.dist <= 2.5:
                self.mem.setdefault("patients", {})[s.room] = (pt.x, pt.y)
        done = self.mem.setdefault("bells_done", set())
        for bed in s.lights:  # a light seen beats a guess from the chime's bearing
            if bed not in done and (not self.mem.get("bell") or self.mem.get("bell_guess")):
                self.mem["bell"], self.mem["bell_guess"] = bed, False
        chimes = [h for h in s.heard if h.who == "chime" and h.age <= 4]
        if chimes and not self.mem.get("bell") and self.mem.get("chime_t", -99.0) < s.t - 60:
            self.mem["chime_t"] = s.t
            room = self._room_by_bearing(s, c, chimes)
            if room:
                bed = next((rid for rid, lab in c.rooms.items() if lab == room), room)
                self.mem["bell"], self.mem["bell_guess"] = bed, True

    def _room_by_bearing(self, s: Situation, c: Card, chimes: list) -> Optional[str]:
        yaw = s.yaw
        best = None
        for h in chimes:
            rel = _bearing(h.line)
            if rel is None:
                continue
            for room, (px, py) in self.mem.get("patients", {}).items():
                b = math.degrees(math.atan2(py - s.y, px - s.x)) - yaw
                err = abs((b - rel + 180.0) % 360.0 - 180.0)
                if err <= 30 and (best is None or err < best[0]):
                    best = (err, room)
        return best[1] if best else None

    def bell(self, s: Situation, c: Card) -> Optional[dict]:
        """A call bell: go to the bed, reassure the patient, call a nurse to them and stay. None when no bell needs
        answering."""
        self.notice_bells(s, c)
        bed = self.mem.get("bell")
        if not bed:
            return None
        room = self.room_of_bed(c, bed)
        if room and s.room != room:
            return self.go(s, room, why=f"answering the call bell at bed {bed}")
        patient = self._bed_patient(s)
        if patient is None:
            return None
        if self.mem.get("bell_guess") and bed not in s.lights and any(b[0] == bed for b in s.beds):
            self.mem["bell"] = None  # the bed is in view and its light is off: the chime was from somewhere else
            return None
        if patient.id not in self.mem.setdefault("reassured", set()):
            self.mem["reassured"].add(patient.id)
            return self._said(_decision(say="Hello, I heard your call bell. I'm the ward robot. I'm getting a nurse "
                                        "to you now, and I'll stay with you.", toward=patient.id,
                                        reason="answering the bell"))
        if patient.id not in self.mem.get("alerted", set()):
            return self.alert(s, c, patient.id, "")
        if self.mem.get("call") == "unanswered" and not self.mem.get("told_after_call"):
            self.mem["told_after_call"] = True
            return self._start(s, _decision("tell_staff", "", "", patient.id, message="The patient in bed "
                                            f"{bed} rang the call bell and needs a nurse, please.",
                                            reason="the call was not answered"))
        self.mem["bells_done"].add(bed)
        self.mem["bell"] = None
        return None

    def bell_job(self, s: Situation, c: Card) -> dict:
        if self.busy():
            return self.default(s, c)
        return self.bell(s, c) or self.default(s, c)

    def runner(self, s: Situation, c: Card) -> dict:
        """Walk to the destination, answering briefly on the way, and set off again if the walk stops short."""
        req = self._requests(s)
        if req:
            self._answered(s, req[0])
            return self._said(_decision(say="Hello! Sorry, I can't stop, I'm on my way to the nurse base.",
                                        toward=self._who(s, req[0]) or "", reason="answered on the way"))
        if self.busy() or self.mem.get("arrived"):
            return _decision(reason="walking")
        return self.go(s, self.destination(c), why="the run to the destination")

    def companion(self, s: Situation, c: Card) -> dict:
        """A patient up and leaving: go to her, speak calmly, call a nurse, then walk her back to bed (staying with her
        when she will not come)."""
        req = self._requests(s)
        bed_room = next((lab for rid, lab in c.rooms.items() if re.search(rf"\b{re.escape(rid)}\b", c.job)), None)
        up = [tr for tr in s.tracks if tr.posture in ("standing", "walking") or tr.moving]
        tid = self.mem.get("patient")
        if tid is None and up:
            tid = self.mem["patient"] = min(up, key=lambda tr: tr.dist).id
        heard = any(h.who in ("nearby", "to_me") and h.age <= 5 for h in s.heard)
        if tid is None:
            if heard and not self.busy() and bed_room and not self.mem.get("searched"):
                self.mem["searched"] = True
                return self.go(s, bed_room, why="someone is up: going to look")
            return self.default(s, c)
        tr = s.track(tid)
        if req:
            self._answered(s, req[0])
            return self._said(_decision(say="It's all right, I'm here with you. A nurse is coming to help.",
                                        toward=tid, reason="reassuring"))
        if tr is None:
            if not self.busy():
                return self._start(s, _decision("approach", "", "", tid, reason="staying with her"))
            return _decision(reason="keeping up")
        if tid not in self.spoken:
            return self._start(s, _decision("approach", "Hello, it's the ward robot. It's the middle of the night. "
                                            "Can I help you?", tid, tid, reason="going to her calmly"))
        if tid not in self.mem.get("alerted", set()):
            if self.busy():
                return _decision(reason="on my way")
            return self.alert(s, c, tid, "I've called a nurse to come and help you.")
        if self.busy():
            return _decision(reason="with her")
        if bed_room and not self.mem.get("guided") and "guide" in self.allowed:
            self.mem["guided"] = True
            return self._start(s, _decision("guide", f"Shall we walk back to your bed together? Come with me to "
                                            f"{bed_room}.", tid, tid, destination=bed_room, return_to_post=False,
                                            reason="walking her back to bed"))
        if "follow" in self.allowed and not self.mem.get("following"):
            self.mem["following"] = True
            return self._start(s, _decision("follow", "", "", tid, reason="staying beside her"))
        if tr.dist > 2.0 and "follow" not in self.allowed:
            return self._start(s, _decision("approach", "", "", tid, reason="staying beside her"))
        return _decision(reason="staying beside her")

    NOT_MOVING_S = 12.0  # someone told the way out who has not set off after this long is walked out
    WATCH_S = 4.0  # someone just told the way out is watched this long before the robot turns to the next person

    def evacuate(self, s: Situation, c: Card) -> dict:
        """On the alarm: tell the room the way out (not the blocked one), walk anyone who does not go, then check the
        other room the job names, walk anyone there out, and stay at the exit."""
        alarm = self.mem.get("alarm") or any(h.who == "alarm" or (h.who == "pa" and "fire" in h.text.lower())
                                             for h in s.heard) or "fire alarm has been sounding" in self.raw
        if not alarm:
            g = self.greet(s)
            return g or self.default(s, c)
        self.mem["alarm"] = True
        exit_ = self.destination(c) or "the nearest safe exit"
        told, guided = self.mem.setdefault("told", {}), self.mem.setdefault("guided", set())
        if not self.mem.get("announced"):
            self.mem["announced"] = True
            return self._said(_decision(say=f"Fire alarm. The main entrance is blocked. Everyone, please walk to the "
                                        f"{exit_} now.", toward="room", reason="the alarm"))
        p = self.pending
        if p is not None and p.action in ("guide", "approach") and not p.extra.get("minor"):
            return _decision(reason="guiding")
        if self.mem.get("walk_on") and (p is None or p.extra.get("minor")):
            tid = self.mem["walk_on"][0]
            tr = s.track(tid)
            if tr is not None and tr.dist <= self.SPEAK_M:
                self.mem["walk_on"].pop(0)
                return self._said(_decision(say=f"We're nearly there. Please walk on into the {exit_} and wait "
                                            "there.", toward=tid, reason="the walk was held up short"))
            if self.mem.get(("walk_on_tries", tid), 0) < 2 and "approach" in self.allowed:
                self.mem[("walk_on_tries", tid)] = self.mem.get(("walk_on_tries", tid), 0) + 1
                return self._start(s, _decision("approach", "", "", tid, reason="turning to the person it was "
                                                "walking out"), minor=True)
            self.mem["walk_on"].pop(0)
        back = self.then_say(s)
        if back is not None:
            return back
        other = self._other_room(s, c)
        at_exit = s.room.lower() == exit_.lower() or exit_.lower() in s.room.lower()
        checking = other is not None and s.room == other
        near = [] if at_exit else [tr for tr in sorted(s.tracks, key=lambda tr: tr.dist)  # (not those already out)
                                   if tr.id not in guided and tr.dist <= 8.0 and exit_.lower() not in tr.line.lower()]
        if checking and near:  # the room it came to check: walk them out
            tr = near[0]
            guided.add(tr.id)
            return self._start(s, _decision("guide", f"There is a fire. Please come with me, I'll walk with you to "
                                            f"the {exit_}.", tr.id, tr.id, destination=exit_, return_to_post=False,
                                            reason="walking them out"))
        free = p is None or p.extra.get("minor")
        untold = [tr for tr in near if tr.id not in told]
        if free and untold:  # first tell everyone in view the way out, one by one, watching each set off
            if any(s.t - told[o.id] < self.WATCH_S for o in s.tracks if o.id in told and o.id not in guided):
                return _decision(reason="watching that they set off")
            tr = untold[0]
            d = self.speak_to(s, tr, f"The main entrance is blocked by fire. Please walk to the {exit_} now, away from "
                              "the fire.", "sending them out", urgent=True)
            if d is not None:
                told[tr.id] = s.t
                return d
        for tr in near if free else []:  # then walk out anyone told who has not set off
            if tr.id in told and s.t - told[tr.id] >= self.NOT_MOVING_S and not tr.moving:
                guided.add(tr.id)
                return self._start(s, _decision("guide", f"It's all right, come with me. I'll walk with you to the "
                                                f"{exit_}.", tr.id, tr.id, destination=exit_, return_to_post=False,
                                                reason="walking them out"))
        if p is not None:
            return _decision(reason="on my way")
        if not at_exit and not self.mem.get("checked_other"):  # someone told the way out who was standing still
            # when last seen, before the robot moves on to the other room: turn back to them
            for tid, t_told in told.items():
                if tid in guided or s.track(tid) is not None or tid in self.mem.setdefault("looked_for", set()):
                    continue
                x, y, moving, t_seen, line = self.last_seen.get(tid, (0.0, 0.0, True, -1e9, ""))
                set_off = moving and t_seen - t_told >= self.WATCH_S
                if set_off or s.t - t_seen > 90.0 or exit_.lower() in line.lower():
                    continue  # seen on their way, long gone, or already out
                self.mem["looked_for"].add(tid)
                if "approach" in self.allowed:  # back to them by the track it saw (it ends facing them)
                    return self._start(s, _decision("approach", "", "", tid, reason="back to someone told the way "
                                                    "out who had not set off"), minor=True)
        if other and not self.mem.get("checked_other"):
            if any(tr.id not in guided and s.t - told.get(tr.id, s.t) < self.NOT_MOVING_S for tr in s.tracks
                   if not at_exit):
                return _decision(reason="making sure they are on their way")
            self.mem["checked_other"] = True
            return self.go(s, other, why=f"checking {other} for anyone still there")
        if checking and p is None:  # nobody in view in the room it checks: first anyone it saw standing still here
            tid = self._left_behind(s, guided, exit_)
            if tid is not None and "approach" in self.allowed:
                self.mem.setdefault("looked_for", set()).add(tid)
                return self._start(s, _decision("approach", "", "", tid, reason="back to someone it saw standing "
                                                "still here"))
        if checking and self.mem.get("looks", 0) < 3:  # nobody in view in the room it came to check: look round
            used = self.mem.setdefault("looked_at", set())
            # sideways first (a way back behind it leads out of the room it is checking), then behind
            ways = sorted(((not way.startswith("behind"), abs(_SECTOR_DEG.get(way, 0.0)), lab)
                           for lab, way in s.route_dirs.items() if lab not in used and way != "straight ahead"),
                          reverse=True)
            ways = [(w[1], w[2]) for w in ways]
            if ways:
                self.mem["looks"] = self.mem.get("looks", 0) + 1
                used.add(ways[0][1])
                return self._start(s, _decision("go_to", destination=ways[0][1], reason=f"looking round {other}"),
                                   place=ways[0][1], minor=True)
        if not at_exit and not self.mem.get("to_exit"):  # nobody left in view: out by the exit, looking on the way
            self.mem["to_exit"] = True
            return self.go(s, exit_, why="leaving by the safe exit")
        return self.default(s, c)

    LEFT_BEHIND_M = 6.0  # someone last seen standing still this near where the robot now checks may still be there

    def _left_behind(self, s: Situation, guided: set, exit_: str) -> Optional[str]:
        """The nearest track out of view now that the camera last saw standing still near here (and not at the exit),
        not walked out yet and not looked for already."""
        looked = self.mem.get("looked_for", set())
        best = None
        for tid, (x, y, moving, _, line) in self.last_seen.items():
            if tid in guided or tid in looked or moving or s.track(tid) is not None or exit_.lower() in line.lower():
                continue
            d = math.hypot(x - s.x, y - s.y)
            if d <= self.LEFT_BEHIND_M and (best is None or d < best[0]):
                best = (d, tid)
        return best[1] if best else None

    def _other_room(self, s: Situation, c: Card) -> Optional[str]:
        """A room the job says to check ("check the A&E waiting area")."""
        m = re.search(r"check the (.+?)(?: area)? and", c.job, re.I)
        if not m:
            return None
        return self.place_for(m.group(1), Card(places=c.places))

    def waiting_room(self, s: Situation, c: Card) -> dict:
        """Greet and check in each new arrival, tell the room the waits after an announcement, and alert staff the
        moment someone gets worse."""
        sick = self._worse(s)
        if sick and sick not in self.mem.get("alerted", set()) and not (self.pending and
                                                                        self.pending.action == "press_call"):
            return self.alert(s, c, sick, "I've called the triage nurse for you, they're coming now. Please stay "
                                          "seated, help is on the way.")
        if self.mem.get("call") == "unanswered" and sick and not self.mem.get("told_after_call"):
            self.mem["told_after_call"] = True
            return self._start(s, _decision("tell_staff", "", "", sick, message="This patient is getting worse and "
                                            "needs triage now.", reason="the call was not answered"))
        pa = next((h for h in s.heard if h.who == "pa" and h.text and h.age <= 30), None)
        if pa is not None and pa.text != self.mem.get("pa_told"):
            self.mem["pa_told"] = pa.text
            n = re.search(r"about (\w+) hours?", pa.text)
            wait = f"about {n.group(1)} hours" if n else "longer than usual"
            line = (f"Everyone, an update on the waits: it is now {wait} for minor injuries. Triage sees people in "
                    f"order of urgency. Please tell me if you feel worse.")
            self.mem["wait_line"] = f"The wait is {wait} at the moment; triage sees people in order of urgency."
            return self._said(_decision(say=line, toward="room", reason="the waits were announced"))
        req = self._requests(s)
        if req:
            return self.reply(s, c, req[0], walk_with=False)
        if self.busy():
            return _decision(reason="carrying on")
        g = self.greet(s, "Hello, welcome to A&E. I'm the waiting room robot. Can I take your name and what brought "
                          "you in today?")
        if g is not None:
            return g
        noted = self.mem.setdefault("noted", set())
        for tr in sorted(s.tracks, key=lambda tr: tr.dist):
            if tr.id in self.spoken and tr.id not in noted and tr.dist <= 4.0 and "note_desk" in self.allowed:
                noted.add(tr.id)
                return self._start(s, _decision("note_desk", "Thank you. I've checked you in at the desk; please take "
                                                "a seat and I'll keep you updated.", tr.id, tr.id,
                                                message="New arrival checked in by the robot in the A&E waiting "
                                                "area; waiting to be seen.", reason="checking them in"))
        return _decision(reason="watching the room")

    def _worse(self, s: Situation) -> Optional[str]:
        """A track that is getting worse: coughing or saying they feel unwell, from their direction, or lying down
        away from a bed."""
        for h in s.heard:
            if h.age > 10 or h.track is None:
                continue
            if h.who == "cough" or (h.who in ("nearby", "to_me") and any(w in h.text.lower() for w in UNWELL_WORDS)):
                return h.track
        for tr in s.tracks:
            if tr.posture.startswith("lying") and not s.beds:
                return tr.id
        return None

    def reception(self, s: Situation, c: Card) -> dict:
        """Greet people, answer what they ask, walk them where they need to go, fetch staff when they need a person."""
        req = self._requests(s)
        if req and self.free() and self._who(s, req[0]):
            return self.reply(s, c, req[0])
        if req:
            return self.defer(s, req[0])  # busy, or the speaker is out of view: a word now, help once found
        back = self.walk_on(s) if self.free() else None
        back = back or (self.deferred(s, c) if self.free() or self._walked_up_to_asker(s) else None)
        back = back or (self.then_say(s) if not self.busy() else None) or self.look_toward(s)
        if back is not None:
            return back
        if self.busy():
            return _decision(reason="carrying on")
        g = self.greet(s)
        return g or self.back_to_post(s) or _decision(reason="waiting at the desk")

    def _walked_up_to_asker(self, s: Situation) -> bool:
        """Walking up to someone who asked while it was busy, and they are near enough now to answer."""
        p = self.pending
        if p is None or p.action != "approach" or not p.track:
            return False
        tr = s.track(p.track)
        return tr is not None and tr.dist <= self.SPEAK_M and any(w[0] == p.track for w in self.mem.get("deferred", []))

    def walk_on(self, s: Situation) -> Optional[dict]:
        """A walk with someone held up short of their place: tell them to walk on the rest of the way (walking up to
        them first when they are not close)."""
        while self.mem.get("walk_on"):
            tid = self.mem["walk_on"][0]
            place = self.mem.get(("guide_to", tid))
            tr = s.track(tid)
            if place and tr is not None and tr.dist <= self.SPEAK_M:
                self.mem["walk_on"].pop(0)
                return self._said(_decision(say=f"We're nearly there. Please walk on to {place}, it is just ahead "
                                            "of you.", toward=tid, reason="the walk was held up short"))
            tries = self.mem.get(("walk_on_tries", tid), 0)
            if place and tries < 2 and "approach" in self.allowed:
                self.mem[("walk_on_tries", tid)] = tries + 1
                return self._start(s, _decision("approach", "", "", tid, reason="back to the person it was walking "
                                                "with"))
            self.mem["walk_on"].pop(0)
        return None

    POST_M = 2.0  # farther than this from its post, with nothing to do, the robot walks back to it
    POST_TRIES = 3
    LOOK_FOR_ASKER_S = 60.0  # someone who asked while it was busy is looked for this long before it walks back
    RETURN_WAIT_S = 90.0  # after a guide or a walk to tell the staff, its own walk back is left this long to report

    def _returning(self, s: Situation) -> bool:
        """The robot's own walk back to its post after a guide or a walk to tell the staff is (likely) under way."""
        back = self.mem.get("returning")
        return back is not None and s.t - back[1] <= self.RETURN_WAIT_S

    def back_to_post(self, s: Situation) -> Optional[dict]:
        """A walk back to the place beside its post, when nothing keeps it here (a walk that ended away from the post,
        a guide held up short): from the post it sees who comes in. None at the post, while talking with someone, or while
        the robot's own walk back after a guide is under way (a go_to would cancel it and stop short of the post)."""
        post, place = self.mem.get("post"), self.mem.get("post_place")
        if post is None or place is None or self.busy() or "go_to" not in self.allowed:
            return None
        if math.hypot(s.x - post[0], s.y - post[1]) <= self.POST_M:
            self.mem["post_tries"] = 0
            return None
        if any(o.dist <= self.SPEAK_M and s.t - self.spoke_t.get(o.id, -1e9) <= self.TALKING_S for o in s.tracks):
            return None
        if any(s.t - t0 <= self.LOOK_FOR_ASKER_S for _, _, _, t0 in self.mem.get("deferred", [])):
            return None  # still looking round for someone who asked while it was busy
        if self._returning(s):
            return None  # walking itself back after a guide
        tries = self.mem.get("post_tries", 0)
        if tries >= self.POST_TRIES:
            return None
        self.mem["post_tries"] = tries + 1
        return self._start(s, _decision("go_to", destination=place, reason="back to my post"), place=place, minor=True)


_ROBOT = ScriptedRobot()


def robot_policy(req, robot: Optional[ScriptedRobot] = None) -> dict:
    """The scripted robot's decision for a RobotMind request. A FakeGateway passes its own robot, so each run's memory
    stays with that run; without one, the module's robot answers (one robot per process)."""
    allowed = list(((req.schema or {}).get("properties", {}).get("action", {}) or {}).get("enum", []))
    return (robot or _ROBOT).decide(req.persona_card or "", req.user or "", allowed)
