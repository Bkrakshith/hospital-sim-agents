"""Kinematic stand-in for the Isaac sim host.

Moves the simulated people in straight lines, seats them, and turns speech, coughs and world
events into audio events. It speaks the same wz.v1 contracts as the real host, so a ROS node can
wrap it now and Isaac can replace it in Plan 2.

Deviations from the task brief's signatures:
  * apply_intent returns list[AudioEvent] (cough and talk produce audio, all others return []).
  * Wet-floor signs are omni-directional: the sign is stored with facing_yaw 0, and
    observation() re-faces every wet-floor sign towards the viewer, so it is readable from any
    side without changing the shared Sign schema.
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wardzero_agents.config.loader import LoadedScenario
from wardzero_agents.config.models import CastEntry, PersonInstance, Sign
from wardzero_agents.config.sampling import sample_instance
from wardzero_agents.contracts import (
    AudioEvent,
    Heard,
    Intent,
    Nudge,
    Observation,
    Roster,
    SeenObject,
    RosterEntry,
    Status,
    WorldEvent,
)
from wardzero_agents.site import Router
from wardzero_agents.world import audio
from wardzero_agents.world.objects import (
    ACCEPT_WINDOW_S, EXTINGUISH_RANGE_M, PERSON_EXTINGUISH_PER_S, ROBOT_EXTINGUISH_PER_S, SURGEON_WINDOW_S, CallBell, Item, WorldObjects,
)
from wardzero_agents.world.reach import giver_reach_ok, receiver_zone_ok
from wardzero_agents.world.oracle import Entity, Pose, bearing_deg, visible, wrap_deg
from wardzero_agents.world.perception import Tracks, call_light_points
from wardzero_agents.world.robot_body import CONTACT_MARGIN_M, RobotBody
from wardzero_agents.world.route_follower import ROBOT_HARD_M

ROBOT_ID = "g1"
ROUTE_BLOCK_RADIUS_M = 1.5  # a route_blocked event seals this radius for routing
DOOR_END_SEAL_M = 0.4  # ... and, where it reaches the end of a doorway, this far on along the door's line
DOOR_SEAL_R_M = 0.25  # ... in discs this wide
DOOR_LANDMARK_M = 1.5  # the PA names a blocked doorway by a landmark this close to its middle
BLOCK_LANDMARK_M = 3.0  # ... or a block off any doorway by a landmark this close to it
FIRE_KEEP_M = 1.0  # no route (a person's or the robot's) comes closer than this to a burning fire
APPROACH_GAP_M = 1.0
SEAT_VIEW_RANGE_M = 6.0
PERSON_Z = 1.6
ROBOT_Z = 1.25
OVERLAYS = ("cough", "fidget", "look_at")
INSTANT = ("sit_idle", "use_phone", "wait_standing", "read_sign", "gesture")
EPS = 1e-9
ARRIVE_M = 0.05
REROUTE_M = 0.5  # an approach re-plans its route once the person it follows has moved this far
LOS_STEP_M = 0.1
BEARING_NOISE_DEG = 10.0  # the direction of a voice or cough the robot hears is off by up to this
STATIC_ROLE = "patient_static"
STATIC_PERSONA = "static_patient"

PERSON_REACH_Z = (0.5, 1.6)
GIVE_RANGE_M = 1.5
TAKE_RANGE_M = 1.2
EXTINGUISHER_PICK_M = 1.0  # a person picks up a free extinguisher this close before using it
BELL_DB = 65.0  # carries the length of the ward corridor (heard in full to about 28 m)
ALARM_DB = 100.0
APPROACH_KINDS = ("approach", "attend")
# where a person is sent to stand (walking up to someone, following the robot) is moved to free floor: out of the
# furniture's clearance band, this far from the robot and from other people, within SPOT_SEARCH_M of the first choice
SPOT_ROBOT_M = 1.2  # an Isaac character may stop up to 0.4 m short of or past its spot: still 0.8 m off the robot
SPOT_PEOPLE_M = 0.6
SPOT_SEARCH_M = 1.6
KEEP_SIT = ("sit_idle", "use_phone", "read_sign", "gesture")  # quick things that do not cancel sitting down
ESCORT_GAP_M = 1.0  # the escorted person trails this far behind the one leading
ADJACENT_M = 0.6  # compartments whose boxes are this close count as neighbours
FOLLOW_NEAR_M = 1.0  # a person following the robot keeps between these two distances behind it
FOLLOW_FAR_M = 1.5
FOLLOW_GAP_M = 1.25  # ... walking up to this far behind when they have fallen back
FOLLOW_DONE_M = 1.75  # the robot has stopped and the follower is this close: the follow is done
ROBOT_AVOID_M = 0.75  # people's routes keep this far from where the robot stands, when they can
ARRIVAL_JITTER_M = 0.8  # people arriving mid-run appear up to this far from the entrance, so they do not stack
ROBOT_SUBSTEP_S = 0.1  # the robot moves in steps no longer than this, its speed cap set before each one
STAFF_ROLES = frozenset({"receptionist", "ward_nurse", "staff", "nurse", "porter", "surgeon"})
ASLEEP_CARD = "asleep"  # a fired card with this id puts its person to sleep
# what a person asleep may do and stay asleep (anything else their own mind chooses wakes them); a reflex (an overlay
# intent such as a cough) never wakes them
ASLEEP_KEEP = ("sit_idle", "wait_standing", "lie_back", "cough", "fidget")
ROLE_KINDS = ("give_item", "take_item", "call_for_help", "call_bell", "accept_item", "refuse_item",
              "request_instrument", "reject_item", "take_instrument_self", "use_extinguisher",
              "sit_up_in_bed", "lie_back")

log = logging.getLogger(__name__)


def spawned_instance(loaded: LoadedScenario, pid: str, persona_id: str,
                     arrive_at_s: float = 0.0) -> PersonInstance:
    """The instance of spawned person N<n>, exactly as StubWorld.spawn samples it.

    arrive_at_s is the sim time the person appeared (the roster's t for another process).
    Other processes (the agent service) rebuild a newcomer from its roster entry with this.
    """
    m = re.fullmatch(r"N([1-9][0-9]*)", pid)
    if m is None:
        raise ValueError(f"'{pid}' is not a spawned person id (N1, N2, ...)")
    if persona_id not in loaded.personas:
        raise ValueError(f"unknown persona '{persona_id}'")
    n = int(m.group(1))
    seed = loaded.scenario.seed * 1000 + 100 + n
    cast = CastEntry(id=pid, persona=persona_id, seed=seed, arrive_at_s=max(0.0, float(arrive_at_s)))
    return sample_instance(loaded.personas[persona_id], cast, loaded.scenario.seed, n)


def _starts_asleep(inst: PersonInstance, scenario) -> bool:
    """start_asleep: true on the person's scenario entry (the instance, or its cast entry)."""
    if getattr(inst, "start_asleep", False):
        return True
    for c in getattr(scenario, "cast", None) or ():
        if getattr(c, "id", None) == inst.person_id and getattr(c, "start_asleep", False):
            return True
    return False


def _seg_point(x: float, y: float, a, b) -> float:
    """Distance from (x, y) to the segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / L2))
    return math.hypot(x - (a[0] + u * dx), y - (a[1] + u * dy))


@dataclass
class _Phase:
    kind: str  # "stand" | "walk" | "sit"
    left: float = 0.0  # seconds, for stand and sit
    dest: Optional[tuple[float, float]] = None  # walk destination (fixed)
    follow: Optional[str] = None  # walk: id of a person or "g1" to approach
    speed: float = 1.0
    path: list = field(default_factory=list)  # walk: waypoints from the Router
    i: int = 0  # index of the waypoint being walked to
    goal: Optional[tuple[float, float]] = None  # the point the path was routed to
    lazy: bool = False  # walk: re-route from wherever the walker is when the phase starts
    carry: Optional[str] = None  # walk: id of a person who is led (escorted) along
    trail: bool = False  # walk: follow the robot (follow is "g1"), 1.0-1.5 m behind it, until it stops


@dataclass
class _Person:
    inst: PersonInstance
    x: float
    y: float
    yaw: float
    posture: str
    seat: Optional[str] = None  # seat currently occupied
    claim: Optional[str] = None  # seat being walked to
    plan: list[_Phase] = field(default_factory=list)
    intent_id: Optional[str] = None
    intent_kind: Optional[str] = None
    state: str = "idle"
    reason: Optional[str] = None
    progress: float = 0.0
    total_est: float = 1.0
    leaving: bool = False
    final_yaw: Optional[float] = None
    kind: str = "mind"  # "static" for a patient lying in a bed with no mind
    bed: Optional[str] = None
    sitting_up: bool = False  # in a bed: propped up (posture stays "lying")
    asleep: bool = False  # asleep until spoken to by address (a robot line to them) or their own mind acts


class StubWorld:
    def __init__(self, loaded: LoadedScenario, seed: int):
        self.loaded = loaded
        self.rng = np.random.default_rng(seed)
        self._ear_rng = np.random.default_rng([seed, 1])  # the noise on the direction of what the robot hears
        self._repeaters: set[str] = set()  # ids of call-bell chimes repeated at a nurse base (no bearing to the bed)
        self._sleep_rng = np.random.default_rng([seed, 2])  # whether a sleeper's ears caught a line to them
        self._pa_blocked: set = set()  # (t, place) of route-block announcements already made
        self.signs: list[Sign] = list(loaded.signs)
        self.last_t = 0.0
        self._wet = 0
        self._blocked = 0
        self._spawned = 0
        self._omni: set[str] = set()
        self.site = loaded.site
        self.router = Router(self.site.grid)
        # the robot's own router: the same map with a hard band, so its routes keep its body off the furniture
        self.robot_router = Router(self.site.grid, hard=ROBOT_HARD_M)
        self._sight = self._wall_cells()
        self._seats = dict(self.site.seats)
        r = loaded.scenario.robot
        self.robot = RobotBody(r.x, r.y, r.yaw, r.radius, r.max_speed, self.robot_router)
        self.robot_reach = tuple(r.reach)
        self.tracks = Tracks()
        self._bed_touch: set[str] = set()
        self.robot_results: list[dict] = []  # resolved robot hand-over offers, drained by the sim node
        self.objects = WorldObjects()
        self.experience: list[dict] = []  # failure hints for the robot's experience log
        self._outbox: list[AudioEvent] = []  # audio the world raised by itself (fire alarm)
        self._log: list[tuple[str, dict]] = []  # (run stream, record) not yet drained
        self._responses: dict[str, tuple[str, float]] = {}  # person -> (accept_item|refuse_item, t)
        self._offers: dict[str, dict] = {}  # receiver -> open give_item offer
        self._fighting: dict[str, str] = {}  # person -> fire id they are putting out
        self._lay_back: dict[str, float] = {}  # person -> when they last lay back in bed
        self._route_blocks: list[tuple[float, float, float]] = []  # (x, y, r) discs route_blocked sealed
        self._logged_objects = None
        self.people: dict[str, _Person] = {}
        self.body_driven: set[str] = set()  # people whose phases a body ends (see end_phase)
        self.named: set[str] = set()  # people who have said their own first name aloud
        self.trail_released: set[str] = set()  # followers whose guide is over: their follow ends now
        self.guide_over: set[str] = set()  # people whose robot guide ended: they do not follow it again unless guided
        self.alarm_active = False  # a fire alarm has sounded (it stays on for the rest of the run)
        # set by the task layer: the robot's speed cap near people (None for no cap), worked out from where everyone
        # is just before the robot moves, so it is never a step late
        self.robot_speed_cap = None
        # set by the task layer: carries out the robot's commands that came since the last step (route searches
        # included), at the start of the step, before anyone moves
        self.robot_commands = None
        taken_beds: set[str] = set()
        for inst in loaded.instances:
            if inst.seat is not None:
                s = self._seats[inst.seat]
                self.people[inst.person_id] = _Person(inst, s.x, s.y, s.yaw, "seated", seat=inst.seat)
            elif inst.start is not None and inst.start.bed is not None:
                b = self.site.beds[inst.start.bed]
                taken_beds.add(b.id)
                self.people[inst.person_id] = _Person(inst, b.x, b.y, b.yaw, "lying", bed=b.id)
            elif inst.start is not None:
                self.people[inst.person_id] = _Person(inst, inst.start.x, inst.start.y, inst.start.yaw, "standing")
            else:
                self.people[inst.person_id] = self._at_entrance(inst)
        for inst in loaded.instances:
            if _starts_asleep(inst, loaded.scenario):
                self.people[inst.person_id].asleep = True
        self._seed_objects(loaded.scenario.objects)
        if loaded.scenario.static_patients:
            for b in self.site.beds.values():
                if re.fullmatch(r"B[1-9][0-9]*", b.id) and b.id not in taken_beds:
                    self.people[f"S{b.id}"] = self._static_patient(b)

    # ---- helpers -------------------------------------------------------------------------
    def _seed_objects(self, specs: list[dict]) -> None:
        """Items and call bells the scenario starts with. Fires start from a card, not from here."""
        for i, o in enumerate(specs):
            kind = o.get("kind")
            if kind == "call_bell":
                bed = o["bed"]
                self.objects.bells.setdefault(bed, CallBell(bed))
            elif kind in ("tray", "specimen_box", "instrument", "extinguisher"):
                self.objects.add_item(Item(
                    id=o.get("id", f"{kind}_{i + 1}"), kind=kind, label=o.get("label", ""),
                    owner=o.get("owner"), holder=o.get("holder"), x=float(o.get("x", 0.0)),
                    y=float(o.get("y", 0.0)), z=float(o.get("z", 0.0))))

    @property
    def robot_pose(self) -> Pose:
        return Pose(*self.robot.pose)

    @robot_pose.setter
    def robot_pose(self, pose: Pose) -> None:
        self.robot.teleport(pose.x, pose.y, pose.yaw)

    def _static_patient(self, bed) -> _Person:
        """A patient lying in a bed: no mind, a fixed status, seen but never heard."""
        sid = f"S{bed.id}"
        inst = PersonInstance(
            person_id=sid, persona_id=STATIC_PERSONA, seed=0, memory_key=sid,
            display_name=f"Patient in bed {bed.id}", age=70, sex="unknown", cough_per_hour=0.0,
            cough_severity="mild", breathless=False, walk_speed=0.0, sit_to_stand_s=0.0, hearing_loss_db=0.0,
            patience_min=0.0, anxiety=0.0, sociability=0.0, phone_use=0.0, attitude_to_robots="neutral",
            english_fluency="native", mobility_aid="none", goals=[], backstory="", condition="in bed",
            role="patient", seat=None, companion_of=None, arrive_at_s=0.0, entrance=None)
        return _Person(inst, bed.x, bed.y, bed.yaw, "lying", bed=bed.id, kind="static")

    def _at_entrance(self, inst: PersonInstance) -> _Person:
        lm = self.site.landmarks[inst.entrance or self.loaded.area.entrance]
        return _Person(inst, lm.x, lm.y, 0.0, "standing")

    def _get(self, person: str) -> _Person:
        p = self.people.get(person)
        if p is None:
            raise ValueError(f"unknown person '{person}'")
        return p

    def _pos(self, ident: str) -> Optional[tuple[float, float]]:
        if ident == ROBOT_ID:
            return self.robot_pose.x, self.robot_pose.y
        p = self.people.get(ident)
        if p is not None and p.posture != "gone":
            return p.x, p.y
        return None

    def _resolve_place(self, target: Optional[str]) -> Optional[tuple[float, float]]:
        if target is None:
            return None
        place = self.site.resolve(target)
        if place is not None:
            return place
        for s in self.signs:
            if s.id == target:
                return s.x, s.y
        return None

    def _nearest_free_seat(self, p: _Person, pid: str, within: float = 6.0) -> Optional[str]:
        staff = p.inst.role in STAFF_ROLES
        free = [s for s in self._seats.values() if (staff or not s.staff_only) and not self._seat_taken(s.id, pid)
                and math.hypot(s.x - p.x, s.y - p.y) <= within]
        return min(free, key=lambda s: math.hypot(s.x - p.x, s.y - p.y)).id if free else None

    def _seat_taken(self, seat: str, by: str) -> bool:
        return any(
            pid != by and q.posture != "gone" and (q.seat == seat or q.claim == seat)
            for pid, q in self.people.items()
        )

    def _present(self) -> list[tuple[str, _Person]]:
        return [(pid, q) for pid, q in self.people.items() if q.posture != "gone"]

    # ---- audio ---------------------------------------------------------------------------
    def _event(self, t, kind, sid, x, y, z, db, text=None, addressee=None) -> AudioEvent:
        return AudioEvent(
            id=uuid.uuid4().hex, t=t, source_kind=kind, source_id=sid, x=x, y=y, z=z,
            loudness_db=db, text=text, addressee=addressee,
        )

    def say(self, person: str, text: str, addressee: Optional[str], t: float) -> AudioEvent:
        p = self._get(person)
        first = (p.inst.display_name or "").split(" ")[0]
        if first and re.search(r"\b" + re.escape(first) + r"\b", text or "", re.I):
            self.named.add(person)
        return self._event(t, "person", person, p.x, p.y, PERSON_Z, audio.SPEECH_DB, text, addressee)

    def robot_say(self, text: str, t: float) -> AudioEvent:
        r = self.robot_pose
        return self._event(t, "robot", ROBOT_ID, r.x, r.y, ROBOT_Z, audio.ROBOT_SPEECH_DB, text)

    def cough(self, person: str, severity: str, t: float) -> AudioEvent:
        p = self._get(person)
        return self._event(t, "cough", person, p.x, p.y, PERSON_Z, audio.COUGH_DB[severity])

    def robot_heard(self, ev: AudioEvent) -> Optional[Heard]:
        """What the robot's microphone hears. A microphone cannot tell who spoke, so the source
        id (P1, N2, ...) is replaced by "unknown"; the kind of sound is kept. A person's voice or cough also
        carries the direction it came from (bearing_deg in the robot's frame, positive = left), give or take
        BEARING_NOISE_DEG, so the robot can tell which of the people it sees made it."""
        self.hear_robot_line(ev)  # every sound in the world passes here: a line to someone asleep wakes them
        r = self.robot_pose
        h = audio.route(ev, r, ROBOT_ID, 0.0, self.rng)
        if h is None:
            return None
        upd: dict = {"source_id": "unknown"}
        if ev.source_kind == "person" and not ev.text and ev.source_id.startswith("bell_"):
            # a chime: the robot can tell what it is (a bed's call bell, or the staff call it pressed itself)
            bed = ev.source_id[len("bell_"):]
            upd["text"] = "staff call chime" if bed == "robot" else f"call bell, bed {bed}"
        if ev.source_kind in ("person", "cough") and ev.source_id != "bell_robot" and ev.id not in self._repeaters:
            # not its own call chime, nor a bell's repeater at the nurse base (that points away from the bed)
            # its own random stream: the world's draws (and so the rest of a seeded run) are unchanged
            noise = float(self._ear_rng.uniform(-BEARING_NOISE_DEG, BEARING_NOISE_DEG))
            upd["bearing_deg"] = round(wrap_deg(bearing_deg(r, ev.x, ev.y) + noise), 1)
        return h.model_copy(update=upd)

    # ---- sleep -----------------------------------------------------------------------------
    def apply_nudge(self, n: Nudge) -> None:
        """A card fired at a person: the "asleep" card puts them to sleep (other cards only reach their mind)."""
        if n.card != ASLEEP_CARD:
            return
        p = self.people.get(n.person)
        if p is None or p.posture == "gone" or p.kind == "static" or p.asleep:
            return
        p.asleep = True
        self._log.append(("world", {"t": n.t, "kind": "asleep", "event": "start", "person": n.person}))

    def _wake(self, pid: str, t: float, reason: str) -> None:
        p = self.people.get(pid)
        if p is None or not p.asleep:
            return
        p.asleep = False
        self._log.append(("world", {"t": t, "kind": "asleep", "event": "end", "person": pid, "reason": reason}))

    def hear_robot_line(self, ev: AudioEvent) -> None:
        """A robot line addressed to someone asleep wakes them, when it reaches their ears."""
        if ev.source_kind != "robot" or not ev.addressee:
            return
        p = self.people.get(ev.addressee)
        if p is None or not p.asleep or p.posture == "gone":
            return
        if audio.route(ev, Pose(p.x, p.y, p.yaw), ev.addressee, p.inst.hearing_loss_db, self._sleep_rng) is not None:
            self._wake(ev.addressee, ev.t, "spoken_to")

    # ---- intents -------------------------------------------------------------------------
    def _fail(self, p: _Person, intent: Intent, reason: str) -> None:
        p.plan = []
        p.claim = None
        p.leaving = False
        p.intent_id, p.intent_kind = intent.id, intent.kind
        p.state, p.reason = "failed", reason
        if p.posture == "walking":
            p.posture = "standing"

    def _begin(self, p: _Person, intent: Intent) -> None:
        p.plan = []
        p.claim = None
        p.leaving = False
        p.final_yaw = None
        p.intent_id, p.intent_kind = intent.id, intent.kind
        p.state, p.reason, p.progress = "running", None, 0.0
        if p.posture == "walking":
            p.posture = "standing"

    def _stand_phase(self, p: _Person) -> list[_Phase]:
        if p.posture in ("seated", "lying"):
            # the seat is released when the stand phase completes (see _advance), so an
            # interrupted stand-up leaves the person seated and the seat taken
            return [_Phase("stand", left=p.inst.sit_to_stand_s)]
        return []

    def _start(self, p: _Person, plan: list[_Phase]) -> None:
        p.plan = plan
        p.total_est = max(self._remaining(p), 1e-6)
        if not plan:
            self._finish(p)

    def _robot_disc(self, p: Optional[_Person] = None) -> tuple:
        """Where people's routes keep clear of: the robot standing in the way (when the scenario has one). Someone
        already close to it gets a smaller disc that leaves them out, down to the robot's contact distance."""
        if not self.loaded.scenario.robot.enabled:
            return ()
        r = ROBOT_AVOID_M
        if p is not None:
            r = min(r, math.hypot(p.x - self.robot.x, p.y - self.robot.y) - 0.05)
        return ((self.robot.x, self.robot.y, r),) if r >= self.robot.radius + 0.3 else ()

    def _route(self, p: _Person, goal: tuple[float, float], speed: float,
               follow: Optional[str] = None) -> Optional[_Phase]:
        path = self.router.route((p.x, p.y), goal, self._robot_disc(p))
        if path is None:
            return None
        if self.site.grid.blocked(p.x, p.y) and tuple(path[0]) != (p.x, p.y):
            path = [(p.x, p.y)] + list(path)  # in a chair: step out to the route's free start first, not across the row
        return _Phase("walk", dest=None if follow else goal, follow=follow, speed=speed, path=path, i=1, goal=goal)

    def apply_intent(self, intent: Intent) -> list[AudioEvent]:
        p = self._get(intent.person)
        if p.posture == "gone":
            return []
        if p.kind == "static":
            self._fail(p, intent, "patient is in bed")
            return []
        kind = intent.kind
        if p.asleep and not intent.overlay and kind not in ASLEEP_KEEP:
            self._wake(intent.person, intent.t, "own_intent")
        self._fighting.pop(intent.person, None)
        if kind in ROLE_KINDS:
            return self._role_action(p, intent)
        if kind in OVERLAYS:
            if kind == "cough":
                return [self.cough(intent.person, p.inst.cough_severity, intent.t)]
            if kind == "look_at" and intent.target is not None:
                pos = self._pos(intent.target) or self._resolve_place(intent.target)
                if pos is not None:
                    p.yaw = math.atan2(pos[1] - p.y, pos[0] - p.x)
                if not intent.overlay:
                    # the mind's own look_at is its current activity: it ends at once, so the mind
                    # gets "intent done" and decides again (a reflex glance is an overlay and does not)
                    self._begin(p, intent)
                    self._finish(p)
            return []
        if kind == "talk":
            return [self.say(intent.person, intent.text or "", intent.addressee, intent.t)]

        speed = intent.speed if intent.speed else p.inst.walk_speed
        if kind in KEEP_SIT and p.claim is not None and any(ph.kind == "sit" for ph in p.plan):
            # sitting down: a quick look at a sign or a phone does not cancel it (in Isaac a cancelled sit left the
            # character standing inside the chair). The mind hears this intent is done; the sit carries on.
            p.intent_id, p.intent_kind = intent.id, intent.kind
            p.state, p.reason, p.progress = "done", None, 1.0
            return []
        if kind in INSTANT:
            self._begin(p, intent)
            if kind == "read_sign" and intent.target is not None:
                pos = self._resolve_place(intent.target)
                if pos is not None:
                    p.yaw = math.atan2(pos[1] - p.y, pos[0] - p.x)
            self._finish(p)
        elif kind == "stand_up":
            self._begin(p, intent)
            self._start(p, self._stand_phase(p))
        elif kind == "walk_to":
            dest = self._resolve_place(intent.target) or (tuple(intent.point) if intent.point else None)
            if dest is None:
                self._fail(p, intent, "unknown target")
                return []
            walk = self._route(p, dest, speed)
            if walk is None:
                self._fail(p, intent, "no route")
                return []
            self._begin(p, intent)
            self._start(p, self._stand_phase(p) + [walk])
        elif kind in ("escort", "evacuate"):
            q = self.people.get(intent.target or "")
            if q is None or q is p or q.posture == "gone":
                self._fail(p, intent, "unknown target")
                return []
            if kind == "escort":
                dest = self._resolve_place(intent.text) or (tuple(intent.point) if intent.point else None)
            else:
                dest = self._safe_point(q)
            if dest is None:
                self._fail(p, intent, "unknown target" if kind == "escort" else "no safe compartment")
                return []
            lead_path = self.router.route((q.x, q.y), dest)
            walk = self._route(p, self._walk_dest(p, _Phase("walk", follow=intent.target)), speed, follow=intent.target)
            if lead_path is None or walk is None:
                self._fail(p, intent, "no route")
                return []
            lead = _Phase("walk", dest=dest, speed=speed, path=lead_path, i=1, goal=dest, lazy=True,
                          carry=intent.target)
            self._begin(p, intent)
            self._start(p, self._stand_phase(p) + [walk, lead])
        elif kind == "follow":
            if not self.loaded.scenario.robot.enabled or (intent.target or ROBOT_ID) not in (ROBOT_ID, "robot"):
                self._fail(p, intent, "unknown target")
                return []
            if intent.person in self.guide_over:
                # the robot's guide for them is over: they do not trail it again (it would cap the robot at 0.3 m/s
                # and walk them back with it) unless it offers them a new guide
                self._fail(p, intent, "the robot is not guiding you now")
                return []
            self._begin(p, intent)
            self.trail_released.discard(intent.person)
            self._start(p, self._stand_phase(p) + [_Phase("walk", follow=ROBOT_ID, speed=speed, trail=True)])
        elif kind in APPROACH_KINDS:
            tgt = intent.target
            if tgt is None or tgt == intent.person or self._pos(tgt) is None:
                self._fail(p, intent, "unknown target")
                return []
            walk = self._route(p, self._walk_dest(p, _Phase("walk", follow=tgt)), speed, follow=tgt)
            if walk is None:
                self._fail(p, intent, "no route")
                return []
            self._begin(p, intent)
            self._start(p, self._stand_phase(p) + [walk])
        elif kind == "sit_down":
            if intent.target is None:  # "sit down here": the nearest free seat they may use, within a few metres
                intent = intent.model_copy(update={"target": self._nearest_free_seat(p, intent.person)})
            seat = self._seats.get(intent.target or "")
            if seat is None:
                self._fail(p, intent, "unknown target")
                return []
            if p.seat == seat.id:
                self._begin(p, intent)
                self._finish(p)
                return []
            if self._seat_taken(seat.id, intent.person):
                self._fail(p, intent, "seat occupied")
                return []
            if seat.staff_only and p.inst.role not in STAFF_ROLES:
                self._fail(p, intent, "that chair is for staff only")
                return []
            walk = self._route(p, (seat.x, seat.y), speed)
            if walk is None:
                self._fail(p, intent, "no route")
                return []
            self._begin(p, intent)
            p.claim = seat.id
            self._start(p, self._stand_phase(p) + [walk, _Phase("sit", left=p.inst.sit_to_stand_s)])
        elif kind == "leave":
            name = intent.target or self.loaded.area.entrance
            dest = self._resolve_place(name)
            if dest is None:
                self._fail(p, intent, "unknown target")
                return []
            if any(math.hypot(dest[0] - bx, dest[1] - by) <= br for bx, by, br in self._route_blocks):
                # the way out itself is blocked: no walking to the free floor beside it and out of the building
                self._fail(p, intent, "no route")
                return []
            walk = self._route(p, dest, speed)
            if walk is None:
                self._fail(p, intent, "no route")
                return []
            self._begin(p, intent)
            p.leaving = True
            self._start(p, self._stand_phase(p) + [walk])
        return []

    # ---- kinematics ----------------------------------------------------------------------
    def _walk_dest(self, p: _Person, ph: _Phase) -> Optional[tuple[float, float]]:
        if ph.follow is None:
            return ph.dest
        tp = self._pos(ph.follow)
        if tp is None:
            return None
        if ph.trail:  # behind the robot: stay put while within the band, else walk up to FOLLOW_GAP_M from it
            d = math.hypot(p.x - tp[0], p.y - tp[1])
            if d <= FOLLOW_FAR_M:
                return p.x, p.y
            k = FOLLOW_GAP_M / d
            return self.free_spot(tp[0] + (p.x - tp[0]) * k, tp[1] + (p.y - tp[1]) * k, p.inst.person_id)
        d = math.hypot(p.x - tp[0], p.y - tp[1])
        if d <= APPROACH_GAP_M:
            return p.x, p.y
        k = APPROACH_GAP_M / d
        return self.free_spot(tp[0] + (p.x - tp[0]) * k, tp[1] + (p.y - tp[1]) * k, p.inst.person_id)

    def _spot_ok(self, x: float, y: float, pid: str) -> bool:
        if not self.router.is_free(x, y):
            return False
        if self.loaded.scenario.robot.enabled and math.hypot(x - self.robot.x, y - self.robot.y) < SPOT_ROBOT_M:
            return False
        return all(math.hypot(x - q.x, y - q.y) >= SPOT_PEOPLE_M for qid, q in self.people.items()
                   if qid != pid and q.posture != "gone")

    def free_spot(self, x: float, y: float, pid: str) -> tuple[float, float]:
        """(x, y) if a person can stand there (free floor clear of furniture, of the robot and of other people), else
        the nearest such spot within SPOT_SEARCH_M that can be seen from (x, y); else (x, y) unchanged."""
        if self._spot_ok(x, y, pid):
            return x, y
        key = (round(x, 1), round(y, 1), pid)
        cache = self.__dict__.setdefault("_spots", {})
        if key in cache:
            return cache[key]
        cache[key] = self._search_spot(x, y, pid)
        return cache[key]

    def _search_spot(self, x: float, y: float, pid: str) -> tuple[float, float]:
        res = self.site.grid.res
        best, bd = None, math.inf
        n = int(SPOT_SEARCH_M / res)
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                d = math.hypot(dx, dy) * res
                if d >= bd or d > SPOT_SEARCH_M:
                    continue
                cx, cy = x + dx * res, y + dy * res
                if self._spot_ok(cx, cy, pid) and self._los((x, y), (cx, cy)):
                    best, bd = (cx, cy), d
        return best if best is not None else (x, y)

    def _lead(self, p: _Person, ph: _Phase) -> None:
        """The person being escorted (or wheeled, in a bed) trails the leader."""
        q = self.people.get(ph.carry or "")
        if q is not None:
            q.x, q.y, q.yaw = p.x - math.cos(p.yaw) * ESCORT_GAP_M, p.y - math.sin(p.yaw) * ESCORT_GAP_M, p.yaw

    def _compartment_at(self, x: float, y: float) -> Optional[str]:
        rid = self.site.room_at(x, y)
        if rid is not None and self.site.rooms[rid].compartment:
            return self.site.rooms[rid].compartment
        for c in self.site.compartments:
            xs, ys = [q[0] for q in c["poly"]], [q[1] for q in c["poly"]]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return c["id"]
        return None

    def _safe_point(self, q: _Person) -> Optional[tuple[float, float]]:
        """The nearest reachable point (by route) in a compartment next to the one q is in."""
        here = self._compartment_at(q.x, q.y)
        boxes = {c["id"]: (min(a[0] for a in c["poly"]), max(a[0] for a in c["poly"]),
                           min(a[1] for a in c["poly"]), max(a[1] for a in c["poly"]))
                 for c in self.site.compartments}
        if here not in boxes:
            return None
        h = boxes[here]
        near = {cid for cid, b in boxes.items() if cid != here
                and b[0] - ADJACENT_M <= h[1] and h[0] - ADJACENT_M <= b[1]
                and b[2] - ADJACENT_M <= h[3] and h[2] - ADJACENT_M <= b[3]}
        best, best_len = None, math.inf
        for r in self.site.rooms.values():
            if r.compartment not in near:
                continue
            c = r.centre
            if self.site.grid.blocked(*c):
                continue
            path = self.router.route((q.x, q.y), c)
            if path is None or math.hypot(path[-1][0] - c[0], path[-1][1] - c[1]) > ARRIVE_M * 4:
                continue
            ln = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
            if ln < best_len:
                best, best_len = c, ln
        return best

    def _remaining(self, p: _Person) -> float:
        total = 0.0
        for ph in p.plan:
            if ph.kind == "walk":
                pts = [(p.x, p.y)] + list(ph.path[ph.i:])
                if ph.follow is not None:
                    dest = self._walk_dest(p, ph)
                    pts = pts + ([dest] if dest is not None and ph.i >= len(ph.path) else [])
                total += sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) / max(ph.speed, 1e-6)
            else:
                total += ph.left
        return total

    def _finish(self, p: _Person) -> None:
        p.plan = []
        if p.leaving:
            p.posture = "gone"
        elif p.claim is not None:
            p.seat = p.claim
        p.claim = None
        if p.posture == "walking":
            p.posture = "standing"
        p.state, p.progress = "done", 1.0

    def _end_timed(self, p: _Person, ph: _Phase) -> None:
        """A stand or sit phase has run its time: the person is standing (seat and bed freed) or seated."""
        if ph.kind == "stand":
            from_bed = p.bed is not None
            p.posture = "standing"
            p.seat = None
            p.bed = None
            if from_bed and self.site.grid.blocked(p.x, p.y):
                self._out_of_bed(p)
        else:
            seat = self._seats[p.claim]
            p.x, p.y, p.yaw, p.posture = seat.x, seat.y, seat.yaw, "seated"
            if p.state != "running":  # the sit outlived its intent (a quick intent came meanwhile): the seat is theirs
                p.seat, p.claim = p.claim, None

    def _out_of_bed(self, p: _Person) -> None:
        """A patient who gets up stands beside the bed, not inside it: where the walk that follows leaves the bed (the
        route's first free point), else on the nearest free floor."""
        nxt = p.plan[0] if p.plan and p.plan[0].kind == "walk" and not p.plan[0].lazy else None
        if nxt is not None and nxt.i < len(nxt.path) and not self.site.grid.blocked(*nxt.path[nxt.i]):
            p.x, p.y = nxt.path[nxt.i]
            nxt.i += 1
            return
        p.x, p.y = self.free_spot(p.x, p.y, p.inst.person_id)

    def _settle(self, p: _Person) -> None:
        """After a phase ends: the intent is done when the plan is empty, else its progress moves on."""
        if p.state == "running":
            if not p.plan:
                self._finish(p)
            else:
                p.progress = max(p.progress, min(0.999, 1.0 - self._remaining(p) / p.total_est))

    # ---- body-driven people (the Isaac host) -------------------------------------------------
    # A person in ``body_driven`` is not walked by step(): a body (Isaac's character) does the walking, sitting
    # and standing, and tells the world when each phase of the plan has ended. The plan, the validation and the
    # intent's state machine stay exactly the same.
    def route_for(self, pid: str, goal: tuple[float, float]) -> Optional[list]:
        """A body's route from where the person is to goal, keeping clear of the robot as the world's own walks do (the
        Isaac characters walk it waypoint by waypoint). A goal inside the robot's contact distance is moved out of it."""
        p = self._get(pid)
        if not self.loaded.scenario.robot.enabled:
            return None
        rx, ry = self.robot.x, self.robot.y
        keep = max(self.robot.radius + CONTACT_MARGIN_M + 0.15, SPOT_ROBOT_M)
        d = math.hypot(goal[0] - rx, goal[1] - ry)
        moved = 1e-6 < d < keep
        if moved:
            goal = (rx + (goal[0] - rx) * keep / d, ry + (goal[1] - ry) * keep / d)
        dx, dy = goal[0] - p.x, goal[1] - p.y
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((rx - p.x) * dx + (ry - p.y) * dy) / L2))
        if not moved and math.hypot(rx - (p.x + u * dx), ry - (p.y + u * dy)) >= keep:
            return None  # the straight walk keeps clear of the robot: no route needed
        return self.router.route((p.x, p.y), goal, self._robot_disc(p))

    def walk_goal(self, pid: str) -> Optional[tuple[float, float]]:
        """Where the head walk phase of this person's plan is going now (a followed person moves), or None."""
        p = self._get(pid)
        if not p.plan or p.plan[0].kind != "walk":
            return None
        ph = p.plan[0]
        return self._walk_dest(p, ph) if ph.follow is not None else (ph.goal or ph.dest)

    def seat(self, seat_id: str):
        return self._seats[seat_id]

    def look_target(self, ident: Optional[str]) -> Optional[tuple[float, float, float]]:
        """(x, y, head height) of a person, the robot, a place or a sign, for a character to look at."""
        pos = self._pos(ident) if ident else None
        if pos is None:
            pos = self._resolve_place(ident)
        if pos is None:
            return None
        z = 1.5 if ident in self.people or ident == ROBOT_ID else 1.2
        return pos[0], pos[1], z

    def carry_of(self, pid: str) -> Optional[str]:
        """The person this person is leading (escort, evacuate, wheeling a bed) in the head walk phase, or None."""
        p = self.people.get(pid)
        if p is None or not p.plan or p.plan[0].kind != "walk":
            return None
        c = p.plan[0].carry
        return c if c in self.people else None

    def lead(self, pid: str) -> None:
        """Move the led person to their trail point behind pid (what step() does for a world-walked leader)."""
        p = self._get(pid)
        if p.plan and p.plan[0].kind == "walk":
            self._lead(p, p.plan[0])

    def end_phase(self, pid: str) -> None:
        """The body finished the head phase of the plan."""
        p = self._get(pid)
        if not p.plan:
            return
        ph = p.plan.pop(0)
        if ph.kind == "walk":
            if ph.follow is not None:
                tp = self._pos(ph.follow)
                if tp is not None:
                    p.yaw = math.atan2(tp[1] - p.y, tp[0] - p.x)
            p.posture = "standing"
        else:
            self._end_timed(p, ph)
        self._settle(p)

    def fail_intent(self, pid: str, reason: str) -> None:
        """The body could not do the head phase (a walk blocked for good)."""
        p = self._get(pid)
        p.plan = []
        p.claim = None
        p.leaving = False
        p.state, p.reason = "failed", reason
        if p.posture == "walking":
            p.posture = "standing"

    def _advance(self, pid: str, p: _Person, dt: float) -> None:
        rem = dt
        while p.plan and rem > EPS:
            ph = p.plan[0]
            if ph.kind in ("stand", "sit"):
                use = min(rem, ph.left)
                ph.left -= use
                rem -= use
                if ph.left <= EPS:
                    p.plan.pop(0)
                    self._end_timed(p, ph)
                continue
            if ph.trail:  # following the robot: one move per step, and the world says when it is over
                self._trail(p, ph, rem)
                rem = 0.0
                continue
            if ph.follow is not None:
                dest = self._walk_dest(p, ph)
                if dest is None:
                    p.plan = []
                    p.claim = None
                    p.state, p.reason = "failed", "target gone"
                    p.posture = "standing"
                    return
                if ph.goal is None or math.hypot(dest[0] - ph.goal[0], dest[1] - ph.goal[1]) > REROUTE_M:
                    path = self.router.route((p.x, p.y), dest, self._robot_disc(p))
                    if path is None:
                        p.plan = []
                        p.claim = None
                        p.state, p.reason = "failed", "no route"
                        p.posture = "standing"
                        return
                    ph.path, ph.i, ph.goal = path, 1, dest
            if ph.lazy:
                ph.lazy = False
                path = self.router.route((p.x, p.y), ph.goal, self._robot_disc(p))
                if path is None:
                    p.plan = []
                    p.state, p.reason = "failed", "no route"
                    p.posture = "standing"
                    return
                ph.path, ph.i = path, 1
                q = self.people.get(ph.carry or "")
                if q is not None:
                    q.plan, q.claim = [], None
                    if q.posture in ("seated", "walking"):
                        q.posture, q.seat = "standing", None
            if ph.i >= len(ph.path):
                wp = ph.goal if ph.goal is not None else (p.x, p.y)
                last = True
            else:
                wp = ph.path[ph.i]
                last = ph.i == len(ph.path) - 1
            d = math.hypot(wp[0] - p.x, wp[1] - p.y)
            if d > 1e-6:
                p.yaw = math.atan2(wp[1] - p.y, wp[0] - p.x)
            reach = ph.speed * rem
            if reach >= d:
                p.x, p.y = wp
                self._lead(p, ph)
                rem -= d / max(ph.speed, 1e-6)
                if not last:
                    ph.i += 1
                    p.posture = "walking"
                    continue
                if ph.follow is not None:  # the person followed may have moved on while we walked
                    nd = self._walk_dest(p, ph)
                    if nd is not None and math.hypot(nd[0] - p.x, nd[1] - p.y) > ARRIVE_M:
                        path = self.router.route((p.x, p.y), nd, self._robot_disc(p))
                        if path is not None and math.hypot(path[-1][0] - p.x, path[-1][1] - p.y) > ARRIVE_M:
                            ph.path, ph.i, ph.goal = path, 1, nd
                            p.posture = "walking"
                            continue
                p.plan.pop(0)
                if ph.follow is not None:
                    tp = self._pos(ph.follow)
                    if tp is not None:
                        p.yaw = math.atan2(tp[1] - p.y, tp[0] - p.x)
                p.posture = "standing"
            else:
                p.x += (wp[0] - p.x) / d * reach
                p.y += (wp[1] - p.y) / d * reach
                self._lead(p, ph)
                p.posture = "walking"
                rem = 0.0
        self._settle(p)

    # ---- following the robot ------------------------------------------------------------------
    def trail_done(self, pid: str) -> bool:
        """A person following the robot is done: the robot's guide for them is over (at the place, or given up), or
        the robot has stopped (no goal) and they have caught up with it. Also done when there is no robot."""
        p = self.people.get(pid)
        if p is None or not self.loaded.scenario.robot.enabled or pid in self.trail_released:
            return True
        return not self.robot.has_goal and math.hypot(p.x - self.robot.x, p.y - self.robot.y) <= FOLLOW_DONE_M

    def is_trailing(self, pid: str) -> bool:
        """The head of this person's plan is following the robot."""
        p = self.people.get(pid)
        return p is not None and bool(p.plan) and p.plan[0].kind == "walk" and p.plan[0].trail

    def _trail(self, p: _Person, ph: _Phase, dt: float) -> None:
        """Keep 1.0-1.5 m behind the robot: stand while within FOLLOW_FAR_M, walk up to FOLLOW_GAP_M when further."""
        r = self.robot
        d = math.hypot(p.x - r.x, p.y - r.y)
        if self.trail_done(p.inst.person_id):
            self.trail_released.discard(p.inst.person_id)
            p.plan.pop(0)
            p.posture = "standing"
            p.yaw = math.atan2(r.y - p.y, r.x - p.x)
            return
        if d <= FOLLOW_GAP_M or (p.posture != "walking" and d <= FOLLOW_FAR_M):
            p.posture = "standing"
            p.yaw = math.atan2(r.y - p.y, r.x - p.x)
            ph.goal = None
            return
        dest = self._walk_dest(p, ph) if d > FOLLOW_FAR_M else (
            r.x + (p.x - r.x) * FOLLOW_GAP_M / d, r.y + (p.y - r.y) * FOLLOW_GAP_M / d)
        if ph.goal is None or ph.i >= len(ph.path) or math.hypot(dest[0] - ph.goal[0], dest[1] - ph.goal[1]) > REROUTE_M:
            path = self.router.route((p.x, p.y), dest, self._robot_disc(p))
            if path is None:
                p.plan = []
                p.state, p.reason = "failed", "no route"
                p.posture = "standing"
                return
            ph.path, ph.i, ph.goal = path, 1, dest
        left = ph.speed * dt
        while left > EPS and ph.i < len(ph.path):
            wp = ph.path[ph.i]
            seg = math.hypot(wp[0] - p.x, wp[1] - p.y)
            if seg > 1e-6:
                p.yaw = math.atan2(wp[1] - p.y, wp[0] - p.x)
            if seg <= left:
                p.x, p.y = wp
                left -= seg
                ph.i += 1
            else:
                p.x += (wp[0] - p.x) / seg * left
                p.y += (wp[1] - p.y) / seg * left
                left = 0.0
            if math.hypot(p.x - r.x, p.y - r.y) <= FOLLOW_GAP_M:
                break
        p.posture = "walking"

    def step(self, t: float) -> None:
        self.__dict__.pop("_spots", None)  # free spots depend on where everyone is: found again each step
        if t <= self.last_t:
            return  # time never runs backwards; last_t is unchanged
        dt = t - self.last_t
        self.last_t = t
        if self.robot_commands is not None:
            self.robot_commands(t)
        for pid, p in self.people.items():
            if p.plan and pid not in self.body_driven:
                self._advance(pid, p, dt)
        if self.loaded.scenario.robot.enabled:
            # people lying in a bed are never "contacts"; passing a bed is bed_contact (below)
            near = [(pid, p.x, p.y) for pid, p in self.people.items() if p.posture not in ("gone", "lying")]
            n = max(1, math.ceil(dt / ROBOT_SUBSTEP_S - 1e-9))
            for _ in range(n):  # people have moved: the cap for where they are now, before each part of the move
                if self.robot_speed_cap is not None:
                    self.robot.speed_cap = self.robot_speed_cap(dt / n) if self.robot.has_goal else None
                for pid in self.robot.step(dt / n, self.site.grid, near):
                    self._log.append(("world", {"t": t, "kind": "contact", "person": pid,
                                                "x": round(self.robot.x, 3), "y": round(self.robot.y, 3),
                                                "speed": round(self.robot.speed, 3)}))
            touching = set()
            if self.robot.speed > 0.05:
                touching = {b.id for b in self.site.beds.values()
                            if math.hypot(b.x - self.robot.x, b.y - self.robot.y) <= self.robot.radius + 0.1}
            for bid in sorted(touching - self._bed_touch):
                self._log.append(("world", {"t": t, "kind": "bed_contact", "bed": bid,
                                            "x": round(self.robot.x, 3), "y": round(self.robot.y, 3)}))
            self._bed_touch = touching
        self._step_objects(t, dt)

    def _wall_cells(self) -> np.ndarray:
        """Cells the building's wall segments pass through (doorways are already gaps in them).
        Furniture and beds block walking but not sight, so sight uses walls only."""
        g = self.site.grid
        cells = np.zeros(g.occ.shape, dtype=bool)
        for x0, y0, x1, y1 in self.site.walls:
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (g.res / 4)))
            for k in range(n + 1):
                iy, ix = g.to_cell(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
                if g.in_grid(iy, ix):
                    cells[iy, ix] = True
        return cells

    def _los(self, a, b) -> bool:
        """True when no wall lies between a and b."""
        g = self.site.grid
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        for k in range(1, int(d / LOS_STEP_M)):
            f = k * LOS_STEP_M / d
            iy, ix = g.to_cell(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            if g.in_grid(iy, ix) and self._sight[iy, ix]:
                return False
        return True

    # ---- outputs -------------------------------------------------------------------------
    def statuses(self) -> list[Status]:
        return [
            Status(
                person=pid, t=self.last_t, x=p.x, y=p.y, yaw=p.yaw, posture=p.posture,
                intent_id=p.intent_id, intent_kind=p.intent_kind, progress=p.progress,
                state=p.state, reason=p.reason, asleep=p.asleep,
            )
            for pid, p in self.people.items()
        ]

    def roster(self, t: float) -> Roster:
        return Roster(
            t=t,
            people=[
                RosterEntry(
                    id=pid, persona=p.inst.persona_id, display_name=p.inst.display_name,
                    seat=p.inst.seat, companion_of=p.inst.companion_of,
                    role=STATIC_ROLE if p.kind == "static" else None,
                )
                for pid, p in self._present()
            ],
        )

    def observation(self, person: str, t: float, recent_audio: list[AudioEvent]) -> Observation:
        p = self._get(person)
        if p.posture == "gone":
            return Observation(person=person, t=t)
        if p.kind == "static":
            return Observation(person=person, t=t)
        me = Pose(p.x, p.y, p.yaw)
        ents = [Entity("person", pid, q.x, q.y, q.posture) for pid, q in self._present()
                if pid != person and q.kind != "static"]
        if self.loaded.scenario.robot.enabled:
            ents.append(Entity("robot", ROBOT_ID, self.robot_pose.x, self.robot_pose.y, None))
        # wet-floor signs are readable from any side: face them towards this viewer
        signs = [
            s.model_copy(update={"facing_yaw": math.atan2(p.y - s.y, p.x - s.x)}) if s.id in self._omni else s
            for s in self.signs
        ]
        los = self._los
        seen = visible(me, ents, signs, los=los)
        seats = [
            Entity("seat", sid, s.x, s.y, "taken" if self._seat_taken(sid, "") else "free")
            for sid, s in self._seats.items()
            if sid not in (p.seat, p.claim)  # the viewer's own seat is not an observation
            and (not s.staff_only or p.inst.role in STAFF_ROLES)  # a staff chair is not a seat for visitors
        ]
        seen += visible(me, seats, [], range_m=SEAT_VIEW_RANGE_M, los=los)
        beds = [Entity("person", pid, q.x, q.y, f"a patient lying in bed {q.bed}")
                for pid, q in self.people.items() if q.kind == "static"]
        seen += visible(me, beds, [], los=self._los)
        objs = self._seen_objects(me, person)
        heard = []
        for ev in recent_audio:
            h = audio.route(ev, me, person, p.inst.hearing_loss_db, self.rng)
            if h is not None:
                heard.append(h)
        return Observation(person=person, t=t, visible=seen, objects=objs, heard=heard)

    def _seen_objects(self, me: Pose, viewer: str) -> list[SeenObject]:
        """Items, fires and lit call-bell lights the viewer can see (same view and wall rules as people)."""
        ents, text = [], {}
        for it in self.objects.items.values():
            pos = (it.x, it.y) if it.holder is None else self._pos(it.holder)
            if pos is None:
                continue
            ents.append(Entity("seat", f"item:{it.id}", pos[0], pos[1]))
            held = "" if it.holder is None else f", held by {it.holder}"
            text[f"item:{it.id}"] = ("item", f"a {it.kind.replace('_', ' ')} (id {it.id}, {it.label or it.kind}){held}")
        for f in self.objects.fires.values():
            if not f.out:
                ents.append(Entity("seat", f"fire:{f.id}", f.x, f.y))
                size = "small" if f.size < 0.34 else "medium" if f.size < 0.67 else "large"
                text[f"fire:{f.id}"] = ("fire", f"a {size} fire (id {f.id})")
        for b in self.objects.bells.values():
            if not b.active:
                continue
            for k, (lx, ly) in enumerate(call_light_points(self.site, b.bed_id)):
                ents.append(Entity("seat", f"bell:{b.bed_id}:{k}", lx, ly))
                text[f"bell:{b.bed_id}:{k}"] = ("call_light", f"call light on at bed {b.bed_id}")
        out, lit = [], set()
        for s in sorted(visible(me, ents, [], los=self._los), key=lambda s: s.distance):
            kind, detail = text[s.id]
            if kind == "call_light":  # one line per bell, however many of its lights are in view
                bed_id = s.id.split(":")[1]
                if bed_id in lit:
                    continue
                lit.add(bed_id)
            out.append(SeenObject(kind=kind, id=s.id.split(":")[1], distance=s.distance,
                                  bearing_deg=s.bearing_deg, detail=detail))
        return out

    # ---- world events and spawning -------------------------------------------------------
    def spawn(self, persona_id: str, t: float) -> str:
        if persona_id not in self.loaded.personas:
            raise ValueError(f"unknown persona '{persona_id}'")
        self._spawned += 1
        while f"N{self._spawned}" in self.people:  # never overwrite a cast person who has an N id
            self._spawned += 1
        pid = f"N{self._spawned}"
        p = self._at_entrance(spawned_instance(self.loaded, pid, persona_id, t))
        if self.loaded.scenario.arrival_entrance is not None and p.inst.entrance is None:
            lm = self.site.landmarks[self.loaded.scenario.arrival_entrance]
            p.x, p.y = lm.x, lm.y
        # a spot of its own near the entrance: its own random stream, so the rest of the run's draws are unchanged
        rng = np.random.default_rng((self.loaded.scenario.seed, self._spawned))
        a, r = rng.uniform(0.0, 2 * math.pi), ARRIVAL_JITTER_M * math.sqrt(rng.uniform())
        x, y = p.x + r * math.cos(a), p.y + r * math.sin(a)
        if self.router.is_free(x, y) and self._los((p.x, p.y), (x, y)):
            p.x, p.y = x, y
        self.people[pid] = p
        p.x, p.y = self.free_spot(p.x, p.y, pid)
        return pid

    def apply_world_event(self, ev: WorldEvent) -> list[AudioEvent]:
        x = 0.0 if ev.x is None else ev.x
        y = 0.0 if ev.y is None else ev.y
        if ev.kind == "pa_announcement":
            return [self._event(ev.t, "pa", "pa", x, y, 2.5, 80.0, ev.text)]
        if ev.kind == "alarm":
            self.alarm_active = True
            return [self._event(ev.t, "alarm", "alarm", x, y, 2.5, 100.0, ev.text)]
        if ev.kind == "wet_floor":
            self._wet += 1
            # facing_yaw is a placeholder: omni signs are re-faced towards each viewer in observation()
            sign = Sign(
                id=f"wet_floor_{self._wet}", text="Caution: wet floor", x=x, y=y, z=0.6,
                facing_yaw=0.0, placeholder=True,
            )
            self.signs.append(sign)
            self._omni.add(sign.id)
            return []
        if ev.kind == "route_blocked":
            # sealed for everyone: the people's router, the robot's, and any the robot's was swapped for (one that
            # wraps another, like the forbidden-area router, follows the one it wraps)
            self._blocked += 1
            self._route_blocks.append((x, y, ROUTE_BLOCK_RADIUS_M))
            for sx, sy, sr in [(x, y, ROUTE_BLOCK_RADIUS_M)] + self._door_end_seals(x, y, ROUTE_BLOCK_RADIUS_M):
                self._seal(sx, sy, sr)
            sign = Sign(id=f"route_blocked_{self._blocked}", text=ev.text or "Route blocked: do not use",
                        x=x, y=y, z=1.2, facing_yaw=0.0, placeholder=True)
            self.signs.append(sign)
            self._omni.add(sign.id)
            # announced on the PA, so everyone (the robot too) hears which way is blocked: the doorway or landmark
            # it blocks, not the whole room (the room may hold the only other way out)
            where = self._blocked_name(x, y, ROUTE_BLOCK_RADIUS_M)
            if (ev.t, where) in self._pa_blocked:  # several blocks at once in one place: one announcement
                return []
            self._pa_blocked.add((ev.t, where))
            return [self._event(ev.t, "pa", "pa", x, y, 2.5, 80.0, f"Fire: the route through {where} is blocked")]
        if ev.kind == "fire_start":
            self.objects.start_fire(x, y)
            self._fire_hazards()
            self._log_objects(ev.t)
            return []
        if ev.kind == "arrival":
            persona = ev.persona
            if persona is None and "anxious_first_timer" in self.loaded.personas:
                persona = "anxious_first_timer"
            if persona is None:
                from wardzero_agents.director.cards import arrival_persona
                in_bed = {i.persona_id for i in self.loaded.instances if i.start is not None and i.start.bed}
                persona = arrival_persona(self.loaded.personas, self.loaded.scenario.seed, ev.id, in_bed)
            if persona is None:
                log.warning("arrival ignored: nobody in this scenario could walk in")
                return []
            try:
                self.spawn(persona, ev.t)
            except ValueError:
                log.warning("arrival ignored: unknown persona '%s'", persona)
            return []
        if ev.kind == "ring_bell":
            # the scenario rings the bell itself, so it does not hang on the patient's model choosing to ring
            q = self.people.get(ev.person or "")
            if q is None or q.posture != "lying" or q.bed is None:
                log.warning("ring_bell ignored: '%s' is not a patient in bed", ev.person)
                return []
            return self._ring_bell(q, ev.t)
        if ev.kind == "person_exit" and ev.person:
            q = self.people.get(ev.person)
            if q is None or q.posture == "gone":
                log.warning("person_exit ignored: '%s' is unknown or already gone", ev.person)
                return []
            self.apply_intent(
                Intent(id=f"exit_{ev.id}", person=ev.person, t=ev.t, kind="leave", target=self.loaded.area.entrance)
            )
        return []

    def _seal(self, x: float, y: float, r: float) -> None:
        """Seal a disc for everyone: the people's router, the robot's, and any the robot's was swapped for (one that
        wraps another, like the forbidden-area router, follows the one it wraps)."""
        routers = [rt for rt in self._all_routers() if hasattr(rt, "block")]
        ids = {id(rt) for rt in routers}
        for rt in routers:
            if id(getattr(rt, "base", None)) not in ids:
                rt.block(x, y, r)

    def _door_end_seals(self, x: float, y: float, r: float) -> list[tuple[float, float, float]]:
        """Extra discs that carry a block over the end of a doorway it reaches: the map's wall beside a door frame can
        leave a gap a cell wide, which the disc alone does not close (people walked out past the main entrance's west
        end). Each end within the disc is sealed DOOR_END_SEAL_M further along the door's line."""
        out = []
        for d in self.site.doors:
            a, b = d["a"], d["b"]
            n = math.hypot(b[0] - a[0], b[1] - a[1])
            if n < 1e-6:
                continue
            for (ex, ey), (ox, oy) in ((a, b), (b, a)):
                if math.hypot(ex - x, ey - y) > r:
                    continue
                ux, uy = (ex - ox) / n, (ey - oy) / n  # along the door's line, outward past this end
                k = 0.0
                while k <= DOOR_END_SEAL_M + 1e-9:
                    out.append((ex + ux * k, ey + uy * k, DOOR_SEAL_R_M))
                    k += 0.1
        return out

    def _blocked_name(self, x: float, y: float, r: float) -> str:
        """What the PA calls a blocked spot: the doorway the block covers (by the landmark at it, else the door's
        name), else the nearest landmark within BLOCK_LANDMARK_M, else the room."""
        lms = [lm for lid, lm in self.site.landmarks.items() if not lid.endswith("_centre") and lm.label]

        def landmark_near(px, py, within):
            best = min(lms, key=lambda lm: math.hypot(lm.x - px, lm.y - py), default=None)
            if best is not None and math.hypot(best.x - px, best.y - py) <= within:
                return best.label.split(",")[0]
            return None

        doors = []
        for d in self.site.doors:
            a, b = d["a"], d["b"]
            dist = _seg_point(x, y, a, b)
            if dist <= r:
                doors.append((dist, d))
        if doors:
            d = min(doors, key=lambda e: e[0])[1]
            mx, my = (d["a"][0] + d["b"][0]) / 2, (d["a"][1] + d["b"][1]) / 2
            name = landmark_near(mx, my, DOOR_LANDMARK_M)
            if name:
                return name
            if d.get("id"):
                return str(d["id"]).replace("_", " ").capitalize()
        name = landmark_near(x, y, BLOCK_LANDMARK_M)
        if name:
            return name
        rid = self.site.room_at(x, y)
        return self.site.rooms[rid].label.split(",")[0] if rid else "this area"

    # ---- objects, hand-overs and role actions --------------------------------------------
    def _log_objects(self, t: float) -> None:
        """Queue an objects.jsonl record when something changed (fire size in 5 % steps)."""
        snap = self.objects.snapshot()
        key = [(o.id, o.holder, o.active, None if o.size is None else round(o.size * 20), o.x, o.y, o.z)
               for o in snap]
        if key != self._logged_objects:
            self._logged_objects = key
            self._log.append(("objects", {"t": t, "objects": [o.model_dump(exclude={"schema_"}) for o in snap]}))

    def drain_log(self) -> list[tuple[str, dict]]:
        """(run stream, record) pairs raised since the last call; the host writes them to the RunLog."""
        out, self._log = self._log, []
        return out

    def drain_audio(self) -> list[AudioEvent]:
        """Audio the world raised by itself since the last call (a fire alarm)."""
        out, self._outbox = self._outbox, []
        return out

    def _step_objects(self, t: float, dt: float) -> None:
        for f in self.objects.step(dt):
            self.alarm_active = True
            self._outbox.append(self._event(t, "alarm", f.id, f.x, f.y, 2.5, ALARM_DB, "Fire alarm"))
        for pid, fid in list(self._fighting.items()):
            p, f = self.people.get(pid), self.objects.fires.get(fid)
            if p is None or f is None or p.posture == "gone":
                self._fighting.pop(pid, None)
            elif f.out:
                self._fighting.pop(pid)
                p.state, p.progress = "done", 1.0
            elif math.hypot(f.x - p.x, f.y - p.y) > EXTINGUISH_RANGE_M:
                self._fighting.pop(pid)
                p.state, p.reason = "failed", "out of range"
            else:
                f.reduce(PERSON_EXTINGUISH_PER_S * dt)
                if f.out:
                    self._fighting.pop(pid)
                    p.state, p.progress = "done", 1.0
        self._resolve_offers(t)
        self._fire_hazards()
        self._log_objects(t)

    def _all_routers(self) -> list:
        """The people's router, the robot's, and the one the robot's body uses (it may have been swapped), each once."""
        rs = (self.router, self.robot_router, getattr(self.robot, "router", None))
        return list({id(r): r for r in rs if r is not None}.values())

    def _fire_hazards(self) -> None:
        """Every router (people's, the robot's and any it was swapped for) keeps out of the burning fires."""
        discs = tuple((f.x, f.y, FIRE_KEEP_M) for f in self.objects.fires.values() if not f.out)
        if discs == getattr(self, "_hazards", ()):
            return
        self._hazards = discs
        for r in self._all_routers():
            if hasattr(r, "set_hazards"):
                r.set_hazards(discs)
        for p in self.people.values():  # walks planned before the fire: round it now (later phases when they start)
            if p.posture == "gone" or p.inst.person_id in self.body_driven:
                continue
            for k, ph in enumerate(p.plan):
                if ph.kind != "walk" or ph.trail:
                    continue
                if k > 0:
                    ph.lazy = ph.lazy or ph.goal is not None
                    continue
                goal = self._walk_dest(p, ph) if ph.follow is not None else (ph.goal or ph.dest)
                path = self.router.route((p.x, p.y), goal, self._robot_disc(p)) if goal is not None else None
                if path is None:
                    p.plan, p.claim = [], None
                    p.state, p.reason = "failed", "no route"
                    p.posture = "standing"
                    break
                ph.path, ph.i, ph.goal = path, 1, goal

    def extinguish(self, by: str, t: float, dt: float) -> bool:
        """by (the robot or a person) sprays the nearest fire within 2 m for dt seconds."""
        pos = self._pos(by)
        if pos is None:
            return False
        f = self.objects.nearest_fire(*pos)
        if f is None:
            return False
        f.reduce((ROBOT_EXTINGUISH_PER_S if by == ROBOT_ID else PERSON_EXTINGUISH_PER_S) * dt)
        self._log_objects(t)
        return True

    def _check_hand_over(self, giver: str, receiver: str, item_id: str,
                         place: tuple[float, float, float]) -> str:
        item = self.objects.items.get(item_id)
        giver = ROBOT_ID if giver == "robot" else giver
        receiver = ROBOT_ID if receiver == "robot" else receiver
        if item is None or item.holder != giver:
            return "not_holding"
        gpos = self._pos(giver)
        reach = self.robot_reach if giver == ROBOT_ID else PERSON_REACH_Z
        if gpos is None or not giver_reach_ok(gpos, reach, place):
            return "out_of_reach_giver"
        if receiver != ROBOT_ID:
            r = self._get(receiver)
            if r.posture == "gone" or not receiver_zone_ok(
                    r.posture, r.bed is not None and r.posture == "lying", r.sitting_up, (r.x, r.y),
                    r.yaw, self.site.beds.get(r.bed or ""), place, item_kind=item.kind):
                return "out_of_reach_receiver"
        return "ok"

    def _window(self, receiver: str) -> float:
        r = self.people.get(receiver)
        return SURGEON_WINDOW_S if r is not None and r.inst.role == "surgeon" else ACCEPT_WINDOW_S

    def _accepted(self, receiver: str, t: float) -> bool:
        r = self.people.get(receiver)
        if receiver in (ROBOT_ID, "robot") or (r is not None and r.kind == "static"):
            return True
        resp = self._responses.get(receiver)
        return resp is not None and resp[0] == "accept_item" and resp[1] >= t - self._window(receiver)

    def hand_over(self, giver: str, receiver: str, item_id: str, place: tuple[float, float, float],
                  t: float) -> tuple[bool, str]:
        """Move item_id from giver to receiver at place. Reason: ok, out_of_reach_giver,
        out_of_reach_receiver, refused or not_holding. The receiver accepted when their latest
        accept_item/refuse_item decision is an accept_item at most 10 s (surgeon 5 s) before t;
        mindless receivers (bed patients) always accept. giver may be "robot" (or g1)."""
        giver = ROBOT_ID if giver == "robot" else giver
        receiver = ROBOT_ID if receiver == "robot" else receiver
        why = self._check_hand_over(giver, receiver, item_id, place)
        if why != "ok":
            return False, why
        if not self._accepted(receiver, t):
            return False, "refused"
        item = self.objects.items[item_id]
        item.holder, item.x, item.y, item.z = receiver, place[0], place[1], place[2]
        self._responses.pop(receiver, None)
        self._log_objects(t)
        return True, "ok"

    def _resolve_offers(self, t: float) -> None:
        for receiver, o in list(self._offers.items()):
            giver = self.people.get(o["giver"])
            resp = self._responses.get(receiver)
            answered = resp is not None and resp[1] >= o["t"]
            if not answered and t - o["t"] <= self._window(receiver):
                continue
            del self._offers[receiver]
            q = self.people.get(receiver)
            mindless = q is not None and q.kind == "static"
            if o["giver"] == ROBOT_ID and not answered and not mindless:
                ok, why = False, "no_answer"  # the window ran out with no reply: not a refusal, not a reach problem
            elif o["giver"] == ROBOT_ID and answered and resp[0] != "accept_item":
                ok, why = False, "refused"  # they declined: said before any check of where it was held out
            else:
                ok, why = self.hand_over(o["giver"], receiver, o["item"], o["place"], max(t, resp[1] if resp else t))
            if o["giver"] == ROBOT_ID:
                self.robot_results.append({"id": o["intent"], "action": "hand_over", "ok": ok,
                                           "reason": why, "person": receiver, "item": o["item"]})
            if giver is not None and giver.intent_id == o["intent"]:
                giver.state, giver.reason = ("done", None) if ok else ("failed", why)
                giver.progress = 1.0 if ok else giver.progress

    def _nurse_base(self, x: float, y: float) -> tuple[float, float]:
        bases = [r.centre for r in self.site.rooms.values() if r.use == "nurse_base"]
        return min(bases, key=lambda c: math.hypot(c[0] - x, c[1] - y)) if bases else (x, y)

    def _ring_bell(self, p: _Person, t: float) -> list[AudioEvent]:
        """Light p's call bell and chime it at the bed and at the nearest nurse base, so whoever is on the ward
        (the robot included) can hear it, not only the nurses at their base."""
        self.objects.ring(p.bed, t)
        bed = self.site.beds.get(p.bed or "")
        bx, by = (bed.x, bed.y) if bed is not None else (p.x, p.y)
        nx, ny = self._nurse_base(p.x, p.y)
        self._log_objects(t)
        rep = self._event(t, "person", f"bell_{p.bed}", nx, ny, 2.0, BELL_DB)
        self._repeaters.add(rep.id)
        return [self._event(t, "person", f"bell_{p.bed}", bx, by, 1.0, BELL_DB), rep]

    def _held(self, who: str) -> list:
        return [i for i in self.objects.items.values() if i.holder == who]

    def _role_action(self, p: _Person, intent: Intent) -> list[AudioEvent]:
        pid, kind, t = intent.person, intent.kind, intent.t
        if p.kind == "static":
            self._fail(p, intent, "patient is in bed")
            return []
        out: list[AudioEvent] = []
        in_bed = p.bed is not None and p.posture == "lying"
        if kind in ("sit_up_in_bed", "lie_back", "call_bell") and not in_bed:
            self._fail(p, intent, "not in bed")
            return []
        if kind == "use_extinguisher":
            f = self.objects.nearest_fire(p.x, p.y)
            # an extinguisher in hand, or a free one within reach (picked up first, e.g. off the wall)
            ext = next((i for i in self._held(pid) if i.kind == "extinguisher"), None)
            if ext is None:
                free = [i for i in self.objects.items.values() if i.kind == "extinguisher" and i.holder is None
                        and math.hypot(i.x - p.x, i.y - p.y) <= EXTINGUISHER_PICK_M]
                if free:
                    ext = min(free, key=lambda i: math.hypot(i.x - p.x, i.y - p.y))
                    ext.holder, ext.x, ext.y = pid, p.x, p.y
                    self._log_objects(t)
                    if f is None:  # took it off the wall; she sprays once she has walked up to the fire
                        self._begin(p, intent)
                        self._finish(p)
                        return []
            if f is None:
                self._fail(p, intent, "no fire in reach")
                return []
            if ext is None:
                self._fail(p, intent, "no extinguisher in hand")
                return []
            self._begin(p, intent)
            self._fighting[pid] = f.id
            return []
        if kind in ("give_item", "take_item"):
            return self._item_transfer(p, intent)
        if kind == "take_instrument_self":
            near = [i for i in self.objects.items.values()
                    if i.kind == "instrument" and i.holder is None
                    and math.hypot(i.x - p.x, i.y - p.y) <= TAKE_RANGE_M
                    and (not intent.text or i.label == intent.text or i.id == intent.text)]
            if not near:
                self._fail(p, intent, "no instrument in reach")
                return []
            self._begin(p, intent)
            near[0].holder = pid
            near[0].x, near[0].y = p.x, p.y
            self._finish(p)
            self._log_objects(t)
            return []
        self._begin(p, intent)
        if kind == "sit_up_in_bed":
            p.sitting_up = True
        elif kind == "lie_back":
            p.sitting_up = False
            self._lay_back[pid] = t
        elif kind == "call_bell":
            out += self._ring_bell(p, t)
        elif kind in ("accept_item", "refuse_item"):
            o = self._offers.get(pid)
            if kind == "accept_item" and in_bed and not p.sitting_up and o is not None and o["giver"] == ROBOT_ID \
                    and self._lay_back.get(pid, -math.inf) < o["t"]:
                p.sitting_up = True  # taking the robot's tray: she sits up for it (one answer is enough)
            self._responses[pid] = (kind, t)
        elif kind == "call_for_help":
            out.append(self.say(pid, intent.text or "Help, I need a nurse here!", intent.addressee, t))
        elif kind == "request_instrument":
            out.append(self.say(pid, f"{intent.text or 'instrument'}, please", intent.addressee or ROBOT_ID, t))
        elif kind == "reject_item":
            self._responses[pid] = ("refuse_item", t)
            out.append(self.say(pid, "No, not that", intent.addressee or ROBOT_ID, t))
            self.experience.append({"t": t, "person": pid, "kind": "failure",
                                    "hint": "the surgeon rejected the item that was passed"})
            self._log.append(("world", {"t": t, "kind": "robot_experience", **self.experience[-1]}))
        self._finish(p)
        return out

    def _item_transfer(self, p: _Person, intent: Intent) -> list[AudioEvent]:
        pid, t = intent.person, intent.t
        other = self.people.get(intent.target or "")
        if other is None or other.posture == "gone":
            self._fail(p, intent, "unknown target")
            return []
        if math.hypot(other.x - p.x, other.y - p.y) > (GIVE_RANGE_M if intent.kind == "give_item" else TAKE_RANGE_M):
            self._fail(p, intent, "too far")
            return []
        src = pid if intent.kind == "give_item" else intent.target
        held = {i.id: i for i in self._held(src)}
        item = held.get(intent.text or "") or next(iter(held.values()), None)
        if item is None:
            self._fail(p, intent, "not holding")
            return []
        if intent.kind == "take_item":
            self._begin(p, intent)
            item.holder, item.x, item.y = pid, p.x, p.y
            self._finish(p)
            self._log_objects(t)
            return []
        if intent.point is not None:
            place = (intent.point[0], intent.point[1], 0.85 if other.bed else 1.1)
        else:  # half a metre from the receiver, towards the giver
            d = max(math.hypot(p.x - other.x, p.y - other.y), 1e-6)
            place = (other.x + (p.x - other.x) / d * 0.5, other.y + (p.y - other.y) / d * 0.5,
                     0.85 if other.bed else 1.1)
        why = self._check_hand_over(pid, intent.target, item.id, place)
        if why != "ok":
            self._fail(p, intent, why)
            return []
        self._begin(p, intent)
        old = self._offers.get(intent.target)
        if old is not None and old["giver"] == ROBOT_ID:  # the robot's offer is replaced: tell it
            self.robot_results.append({"id": old["intent"], "action": "hand_over", "ok": False,
                                       "reason": "superseded", "person": intent.target, "item": old["item"]})
        self._offers[intent.target] = {"giver": pid, "item": item.id, "place": place, "t": t,
                                       "intent": intent.id}
        return []
