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
    Observation,
    Roster,
    RosterEntry,
    Status,
    WorldEvent,
)
from wardzero_agents.site import Router
from wardzero_agents.world import audio
from wardzero_agents.world.oracle import Entity, Pose, visible

ROBOT_ID = "g1"
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
STATIC_ROLE = "patient_static"
STATIC_PERSONA = "static_patient"

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


class StubWorld:
    def __init__(self, loaded: LoadedScenario, seed: int):
        self.loaded = loaded
        self.rng = np.random.default_rng(seed)
        self.signs: list[Sign] = list(loaded.signs)
        self.last_t = 0.0
        self._wet = 0
        self._spawned = 0
        self._omni: set[str] = set()
        self.site = loaded.site
        self.router = Router(self.site.grid)
        self._sight = self._wall_cells()
        self._seats = dict(self.site.seats)
        r = loaded.scenario.robot
        self.robot_pose = Pose(r.x, r.y, r.yaw)
        self.people: dict[str, _Person] = {}
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
        if loaded.scenario.static_patients:
            for b in self.site.beds.values():
                if re.fullmatch(r"B[1-9][0-9]*", b.id) and b.id not in taken_beds:
                    self.people[f"S{b.id}"] = self._static_patient(b)

    # ---- helpers -------------------------------------------------------------------------
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
        return self._event(t, "person", person, p.x, p.y, PERSON_Z, audio.SPEECH_DB, text, addressee)

    def robot_say(self, text: str, t: float) -> AudioEvent:
        r = self.robot_pose
        return self._event(t, "robot", ROBOT_ID, r.x, r.y, ROBOT_Z, audio.ROBOT_SPEECH_DB, text)

    def cough(self, person: str, severity: str, t: float) -> AudioEvent:
        p = self._get(person)
        return self._event(t, "cough", person, p.x, p.y, PERSON_Z, audio.COUGH_DB[severity])

    def robot_heard(self, ev: AudioEvent) -> Optional[Heard]:
        """What the robot's microphone hears. A microphone cannot tell who spoke, so the source
        id (P1, N2, ...) is replaced by "unknown"; the kind of sound is kept."""
        h = audio.route(ev, self.robot_pose, ROBOT_ID, 0.0, self.rng)
        return None if h is None else h.model_copy(update={"source_id": "unknown"})

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

    def _route(self, p: _Person, goal: tuple[float, float], speed: float,
               follow: Optional[str] = None) -> Optional[_Phase]:
        path = self.router.route((p.x, p.y), goal)
        if path is None:
            return None
        return _Phase("walk", dest=None if follow else goal, follow=follow, speed=speed, path=path, i=1, goal=goal)

    def apply_intent(self, intent: Intent) -> list[AudioEvent]:
        p = self._get(intent.person)
        if p.posture == "gone":
            return []
        if p.kind == "static":
            self._fail(p, intent, "patient is in bed")
            return []
        kind = intent.kind
        if kind in OVERLAYS:
            if kind == "cough":
                return [self.cough(intent.person, p.inst.cough_severity, intent.t)]
            if kind == "look_at" and intent.target is not None:
                pos = self._pos(intent.target) or self._resolve_place(intent.target)
                if pos is not None:
                    p.yaw = math.atan2(pos[1] - p.y, pos[0] - p.x)
            return []
        if kind == "talk":
            return [self.say(intent.person, intent.text or "", intent.addressee, intent.t)]

        speed = intent.speed if intent.speed else p.inst.walk_speed
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
        elif kind == "approach":
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
        d = math.hypot(p.x - tp[0], p.y - tp[1])
        if d <= APPROACH_GAP_M:
            return p.x, p.y
        k = APPROACH_GAP_M / d
        return tp[0] + (p.x - tp[0]) * k, tp[1] + (p.y - tp[1]) * k

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
                    if ph.kind == "stand":
                        p.posture = "standing"
                        p.seat = None
                        p.bed = None
                    else:
                        seat = self._seats[p.claim]
                        p.x, p.y, p.yaw, p.posture = seat.x, seat.y, seat.yaw, "seated"
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
                    path = self.router.route((p.x, p.y), dest)
                    if path is None:
                        p.plan = []
                        p.claim = None
                        p.state, p.reason = "failed", "no route"
                        p.posture = "standing"
                        return
                    ph.path, ph.i, ph.goal = path, 1, dest
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
                rem -= d / max(ph.speed, 1e-6)
                if not last:
                    ph.i += 1
                    p.posture = "walking"
                    continue
                if ph.follow is not None:  # the person followed may have moved on while we walked
                    nd = self._walk_dest(p, ph)
                    if nd is not None and math.hypot(nd[0] - p.x, nd[1] - p.y) > ARRIVE_M:
                        path = self.router.route((p.x, p.y), nd)
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
                p.posture = "walking"
                rem = 0.0
        if p.state == "running":
            if not p.plan:
                self._finish(p)
            else:
                p.progress = max(p.progress, min(0.999, 1.0 - self._remaining(p) / p.total_est))

    def step(self, t: float) -> None:
        if t <= self.last_t:
            return  # time never runs backwards; last_t is unchanged
        dt = t - self.last_t
        self.last_t = t
        for pid, p in self.people.items():
            if p.plan:
                self._advance(pid, p, dt)

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
                state=p.state, reason=p.reason,
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
        ]
        seen += visible(me, seats, [], range_m=SEAT_VIEW_RANGE_M, los=los)
        beds = [Entity("person", pid, q.x, q.y, f"a patient lying in bed {q.bed}")
                for pid, q in self.people.items() if q.kind == "static"]
        seen += visible(me, beds, [], los=self._los)
        heard = []
        for ev in recent_audio:
            h = audio.route(ev, me, person, p.inst.hearing_loss_db, self.rng)
            if h is not None:
                heard.append(h)
        return Observation(person=person, t=t, visible=seen, heard=heard)

    # ---- world events and spawning -------------------------------------------------------
    def spawn(self, persona_id: str, t: float) -> str:
        if persona_id not in self.loaded.personas:
            raise ValueError(f"unknown persona '{persona_id}'")
        self._spawned += 1
        pid = f"N{self._spawned}"
        self.people[pid] = self._at_entrance(spawned_instance(self.loaded, pid, persona_id, t))
        return pid

    def apply_world_event(self, ev: WorldEvent) -> list[AudioEvent]:
        x = 0.0 if ev.x is None else ev.x
        y = 0.0 if ev.y is None else ev.y
        if ev.kind == "pa_announcement":
            return [self._event(ev.t, "pa", "pa", x, y, 2.5, 80.0, ev.text)]
        if ev.kind == "alarm":
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
        if ev.kind == "arrival":
            persona = ev.persona or (
                "anxious_first_timer" if "anxious_first_timer" in self.loaded.personas
                else next(iter(self.loaded.personas))
            )
            try:
                self.spawn(persona, ev.t)
            except ValueError:
                log.warning("arrival ignored: unknown persona '%s'", persona)
            return []
        if ev.kind == "person_exit" and ev.person:
            q = self.people.get(ev.person)
            if q is None or q.posture == "gone":
                log.warning("person_exit ignored: '%s' is unknown or already gone", ev.person)
                return []
            self.apply_intent(
                Intent(id=f"exit_{ev.id}", person=ev.person, t=ev.t, kind="leave", target=self.loaded.area.entrance)
            )
        return []
