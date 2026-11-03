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

log = logging.getLogger(__name__)


def spawned_instance(loaded: LoadedScenario, pid: str, persona_id: str) -> PersonInstance:
    """The instance of spawned person N<n>, exactly as StubWorld.spawn samples it.

    Other processes (the agent service) rebuild a newcomer from its roster entry with this.
    """
    m = re.fullmatch(r"N([1-9][0-9]*)", pid)
    if m is None:
        raise ValueError(f"'{pid}' is not a spawned person id (N1, N2, ...)")
    if persona_id not in loaded.personas:
        raise ValueError(f"unknown persona '{persona_id}'")
    n = int(m.group(1))
    seed = loaded.scenario.seed * 1000 + 100 + n
    cast = CastEntry(id=pid, persona=persona_id, seed=seed)
    return sample_instance(loaded.personas[persona_id], cast, loaded.scenario.seed, n)


@dataclass
class _Phase:
    kind: str  # "stand" | "walk" | "sit"
    left: float = 0.0  # seconds, for stand and sit
    dest: Optional[tuple[float, float]] = None  # walk destination (fixed)
    follow: Optional[str] = None  # walk: id of a person or "g1" to approach
    speed: float = 1.0


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


class StubWorld:
    def __init__(self, loaded: LoadedScenario, seed: int):
        self.loaded = loaded
        self.rng = np.random.default_rng(seed)
        self.signs: list[Sign] = list(loaded.signs)
        self.last_t = 0.0
        self._wet = 0
        self._spawned = 0
        self._omni: set[str] = set()
        self._seats = {s.id: s for s in loaded.area.seats}
        r = loaded.scenario.robot
        self.robot_pose = Pose(r.x, r.y, r.yaw)
        self.people: dict[str, _Person] = {}
        for inst in loaded.instances:
            if inst.seat is not None:
                s = self._seats[inst.seat]
                self.people[inst.person_id] = _Person(inst, s.x, s.y, s.yaw, "seated", seat=inst.seat)
            else:
                self.people[inst.person_id] = self._at_entrance(inst)

    # ---- helpers -------------------------------------------------------------------------
    def _at_entrance(self, inst: PersonInstance) -> _Person:
        lm = self.loaded.area.landmarks[inst.entrance or self.loaded.area.entrance]
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
        lm = self.loaded.area.landmarks.get(target)
        if lm is not None:
            return lm.x, lm.y
        seat = self._seats.get(target)
        if seat is not None:
            return seat.x, seat.y
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
        return audio.route(ev, self.robot_pose, ROBOT_ID, 0.0, self.rng)

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
        if p.posture == "seated":
            # the seat is released when the stand phase completes (see _advance), so an
            # interrupted stand-up leaves the person seated and the seat taken
            return [_Phase("stand", left=p.inst.sit_to_stand_s)]
        return []

    def _start(self, p: _Person, plan: list[_Phase]) -> None:
        p.plan = plan
        p.total_est = max(self._remaining(p), 1e-6)
        if not plan:
            self._finish(p)

    def apply_intent(self, intent: Intent) -> list[AudioEvent]:
        p = self._get(intent.person)
        if p.posture == "gone":
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
            self._begin(p, intent)
            self._start(p, self._stand_phase(p) + [_Phase("walk", dest=dest, speed=speed)])
        elif kind == "approach":
            tgt = intent.target
            if tgt is None or tgt == intent.person or self._pos(tgt) is None:
                self._fail(p, intent, "unknown target")
                return []
            self._begin(p, intent)
            self._start(p, self._stand_phase(p) + [_Phase("walk", follow=tgt, speed=speed)])
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
            self._begin(p, intent)
            p.claim = seat.id
            self._start(
                p,
                self._stand_phase(p)
                + [_Phase("walk", dest=(seat.x, seat.y), speed=speed), _Phase("sit", left=p.inst.sit_to_stand_s)],
            )
        elif kind == "leave":
            name = intent.target or self.loaded.area.entrance
            dest = self._resolve_place(name)
            if dest is None:
                self._fail(p, intent, "unknown target")
                return []
            self._begin(p, intent)
            p.leaving = True
            self._start(p, self._stand_phase(p) + [_Phase("walk", dest=dest, speed=speed)])
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
                dest = self._walk_dest(p, ph)
                if dest is not None:
                    total += math.hypot(dest[0] - p.x, dest[1] - p.y) / max(ph.speed, 1e-6)
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
                    else:
                        seat = self._seats[p.claim]
                        p.x, p.y, p.yaw, p.posture = seat.x, seat.y, seat.yaw, "seated"
                continue
            dest = self._walk_dest(p, ph)
            if dest is None:
                p.plan = []
                p.claim = None
                p.state, p.reason = "failed", "target gone"
                p.posture = "standing"
                return
            d = math.hypot(dest[0] - p.x, dest[1] - p.y)
            heading = math.atan2(dest[1] - p.y, dest[0] - p.x) if d > EPS else p.yaw
            if d > 1e-6:
                p.yaw = heading
            reach = ph.speed * rem
            if reach >= d:
                p.x, p.y = dest
                rem -= d / max(ph.speed, 1e-6)
                p.plan.pop(0)
                if ph.follow is not None:
                    tp = self._pos(ph.follow)
                    if tp is not None:
                        p.yaw = math.atan2(tp[1] - p.y, tp[0] - p.x)
                p.posture = "standing"
            else:
                p.x += (dest[0] - p.x) / d * reach
                p.y += (dest[1] - p.y) / d * reach
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
                )
                for pid, p in self._present()
            ],
        )

    def observation(self, person: str, t: float, recent_audio: list[AudioEvent]) -> Observation:
        p = self._get(person)
        if p.posture == "gone":
            return Observation(person=person, t=t)
        me = Pose(p.x, p.y, p.yaw)
        ents = [Entity("person", pid, q.x, q.y, q.posture) for pid, q in self._present() if pid != person]
        if self.loaded.scenario.robot.enabled:
            ents.append(Entity("robot", ROBOT_ID, self.robot_pose.x, self.robot_pose.y, None))
        # wet-floor signs are readable from any side: face them towards this viewer
        signs = [
            s.model_copy(update={"facing_yaw": math.atan2(p.y - s.y, p.x - s.x)}) if s.id in self._omni else s
            for s in self.signs
        ]
        seen = visible(me, ents, signs)
        seats = [
            Entity("seat", sid, s.x, s.y, "taken" if self._seat_taken(sid, "") else "free")
            for sid, s in self._seats.items()
            if sid not in (p.seat, p.claim)  # the viewer's own seat is not an observation
        ]
        seen += visible(me, seats, [], range_m=SEAT_VIEW_RANGE_M)
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
        self.people[pid] = self._at_entrance(spawned_instance(self.loaded, pid, persona_id))
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
