"""Robot task actions for the stub world (the test double for the Isaac host's task layer).

An action is a JSON dict {"id", "action", ...} from <ns>/task/action:
  hand_over {item, to: track, place?: [x,y,z]}   an open offer; the result comes when it resolves. An item it does not
                                                  hold but has within TAKE_REACH_M is taken first, in the same action.
                                                  Reasons: ok, refused (declined), no_answer (no reply in the offer
                                                  window, or the person is asleep), out_of_reach_receiver (geometry),
                                                  out_of_reach_giver, not_holding, unknown_track, superseded
  take {item}                                     an unheld item within 0.8 m
  extinguish {fire}                               sprays until the fire is out (or the robot leaves); out of range, it
                                                  first walks to free floor SPRAY_FROM_M from the fire
  press_call {track?}                             chimes at the call point serving the robot's area (the nearest
                                                  nurse base or triage, in its own compartment first); every staff
                                                  member who hears it is told where the robot is, which call lights
                                                  are on and, with track, who needs them. ok, with reason "answered"
                                                  (someone heard it) or "unanswered" (nobody did)
  guide_start {track, place, seat?, return_to_post?} / guide_stop {}
                                                  walks to the place at the person's pace; the person is offered a walk;
                                                  with seat, it ends beside a free seat there and shows it to them;
                                                  return_to_post (default true) false leaves it where the guide ended
  follow {track}                                  walks up to the person and stays 1.0-1.5 m from them as they move,
                                                  stepping back when they come nearer, until guide_stop, a goal,
                                                  cmd_vel, another walking task it takes on (one refused leaves the
                                                  follow on), or they are gone; result ok/following at the start
                                                  (no_person: unknown track)
  approach {track}                                walks up to the person (APPROACH_M from them) and faces them; to a
                                                  patient in bed, or with an item in hand, it ends where it can hand
                                                  the item over (beside the bed; in front of or beside them)
  tell_staff {about: track, message?}             walks to the reception counter, or on a ward to the nearest ward
                                                  staff member (a staff locator: the robot is led to where she is),
                                                  tells them this person needs help, then walks back to its post
                                                  (not while a fire alarm sounds)
  note_desk {about: track, message}               a note on the reception desk computer: the robot does not move or
                                                  speak; the receptionist reads it when she is at the desk
Each action ends in one result {"id", "action", "ok", "reason", "t"}; nothing here uses person ids
on the robot's side of the wire: people are addressed by track.

Guiding: the robot holds while the person it guides is more than 2.5 m behind (and walks on once they are within
2 m); that person, following behind it, does not slow it, but it keeps clear of them like anyone else, and its walk
ends at least GUIDE_CLEAR_M from them (on its own side of them when they stand at the place). Other people within
1.5 m slow it to 0.3 m/s. At the place it waits up to 20 s for them to arrive, says goodbye and walks back to its post
(goal id "<id>-return"), unless return_to_post is false (it says STAY_SAY and stays with them) or a fire alarm has
sounded. It gives up if they never come within 2.5 m for 30 s. Once a guide is over, the person does not take up
following the robot again unless it guides them again.

A guide to a seat: when the robot reaches the place it picks the free seat nearest to it, among those its camera could
see by turning its head (within its range, in line of sight; never a staff chair). It walks to a spot beside that seat,
says SEAT_SAY and the person is told the seat (a nudge naming sit_down <seat>); they have arrived when they are near
the seat. With no free seat in sight it says NO_SEAT_SAY and the guide ends as a plain guide to the place.

Telling the staff: the staff member hears it by the audio model (and must be within 5 m); when they do, they get a
structured message (a nudge) with the person's id, their name if they said it, and where they are. At reception the
robot walks to the counter. On a ward it tells the nearest ward staff member: from where it stands when she is within
5 m in sight, else it walks to free floor 1.0-1.5 m from her that can see her (else to the nurse base nearest her),
and speaks as soon as she is within 1.5 m in sight, or when people hold it up but she is within 5 m. This is a staff
locator: the robot's brain is never told where she is, the task layer finds her. After a tell it walked for, the robot
walks back to its post (goal id "<id>-return"); on a ward that takes it away from the patient, so a brain that
should stay with them sends approach again once the tell result arrives.

A hand-over to someone who can take things (their role has accept_item) is announced to them (a nudge naming
accept_item, and refuse_item or, for a surgeon, reject_item). A patient lying flat is asked to accept it: accepting the
robot's offer sits her up for it.

The desk note: the result ("noted") comes at once, since the computer takes the note. The receptionist gets it as a
nudge (what is on her screen) when she is within 1.5 m of her chair, at once or when she gets back, and only once.

The sim-side run log (robot_tasks.jsonl, never sent to the robot) gets a "guide" row at the start and end of each
guide, with the person's id, a "follow" row at the start and end of each follow, a "hand_over" offer row (person,
item) when the robot holds an item out, the hand_over result row with the receiver (person) and item, and a "tell" row for each tell_staff and a "note" row for each desk note when it is read (and when it is left unread at
the end of the run, nothing: unread notes are not logged).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from wardzero_agents.config.presets import ROLE_INTENTS
from wardzero_agents.contracts import AudioEvent, Nudge
from wardzero_agents.world import audio
from wardzero_agents.world.objects import EXTINGUISH_RANGE_M, ROBOT_EXTINGUISH_PER_S
from wardzero_agents.world.oracle import Pose
from wardzero_agents.world.reach import giver_reach_ok, receiver_zone_ok

TAKE_REACH_M = 0.8
CHIME_DB = 55.0
ACTIONS = ("hand_over", "take", "extinguish", "press_call", "guide_start", "guide_stop", "tell_staff", "note_desk",
           "approach", "follow")
WALKING = ("approach", "guide_start", "tell_staff", "extinguish", "follow")  # these end a follow under way
GUIDE_CARD = "robot_guide"
TELL_CARD = "robot_tell_staff"
CALL_CARD = "robot_staff_call"
CALL_POINT_USES = ("nurse_base", "triage")  # rooms with a staff call point
NOTE_CARD = "robot_desk_note"
SEAT_CARD = "robot_seat"
SEAT_SAY = "This seat is free. Please take a seat."
NO_SEAT_SAY = "I'm sorry, I can't see a free seat here."
SEAT_BESIDE_M = 0.9  # the robot stands this far from the seat it shows
APPROACH_M = 1.2  # how close the robot comes to a person it walks up to
BEDSIDE_M = (1.0, 0.9, 1.1)  # walking up to a patient in bed: this far from the bed's middle, across it (the bed itself
# is blocked out to about 0.7 m)
REACH_SPOT_M = (0.85, 0.75, 0.95)  # walking up to someone with an item in hand: this far from them, where it can pass it
PLACE_GRID_M = 0.1  # the hand-over places tried around the receiver, on this grid ...
PLACE_SEARCH_M = 0.8  # ... out to this far from them
HAND_OVER_CARD = "robot_hand_over"
SPRAY_FROM_M = 1.5  # out of range of a fire, extinguish first walks to free floor this far from it
SEAT_ASIDE_M = 1.0  # showing a seat, the robot stands this far to the side of the seat's front, out of the way in
# keeping clear of people while it walks (any goal, its own or a task's):
PERSON_KEEP_M = 0.7  # centre to centre: body radius 0.3 + a person's 0.25 + 0.15 m
STOP_CLOSE_M = 0.9  # anyone in front closer than this holds it at once (the walking G1 needs room to stop)
TARGET_KEEP_M = 0.45  # the person it walks up to or leads: it may come this close (a hand-over needs it)
STOP_AHEAD_M = 1.2  # someone this close ahead of it (and nearer than its goal) holds it
LATERAL_M = 0.6  # ... "ahead" is within this of its line of travel
REPLAN_S = 2.0  # held this long: it routes round the people in the way
AVOID_R_M = 0.9  # the discs its new route keeps out of
GIVE_UP_S = 8.0  # held this long in all: the walk is given up (a seat is shown from where it stands)
BACK_OFF_M = 0.6  # standing still, someone this close for BACK_OFF_S: it steps back
BACK_OFF_S = 0.5
BACK_OFF_STEP_M = 0.7
BACK_OFF_COOLDOWN_S = 4.0
SAFETY = "safety"  # goal ids of its own safety moves: their results are not reported to the brain
FOLLOW_GOAL = "follow"  # goal ids of its own walks while it follows someone: not reported either
FOLLOW_NEAR_M = 1.0  # following, the robot keeps between these two distances from the person ...
FOLLOW_FAR_M = 1.5
FOLLOW_GAP_M = 1.25  # ... walking up to this far from them when they get further than FOLLOW_FAR_M
FOLLOW_REPLAN_M = 0.5  # its walk is re-planned once the spot it walks to has moved this far
FOLLOW_RETRY_S = 2.0  # a walk that failed (blocked, stuck) is tried again after this long
DESK_M = 1.5  # the receptionist reads the desk computer when she is this close to her chair

GUIDE_HOLD_M = 2.5  # the robot waits while the guided person is further behind than this ...
GUIDE_RESUME_M = 2.0  # ... and walks on once they are this close again
GUIDE_GIVE_UP_S = 30.0  # held this long without them coming within GUIDE_HOLD_M: the guide is given up
GUIDE_ARRIVE_WAIT_S = 20.0  # at the place, wait this long for the person to arrive
RETURN_CLEAR_M = 1.5  # after a guide the robot starts back to its post once nobody is this close ...
RETURN_WAIT_S = 8.0  # ... or after this long
NEAR_PLACE_M = 2.0  # a person this close to the place (or in the named room) has arrived
GUIDE_CLEAR_M = 1.2  # the guide's walk ends at least this far from the person it guides (moved off the place if needed)
NEAR_PERSON_M = 1.5  # another person this close slows the robot ...
NEAR_PERSON_SPEED = 0.3  # ... to this (m/s)
CAP_MARGIN_M = 0.02  # coming up to NEAR_PERSON_M of someone, the robot keeps this short of it at full speed
PERSON_AVOID_M = 0.7  # the robot's routes keep this far from people standing in the way, when they can
TELL_RANGE_M = 5.0  # a staff member further than this from the robot is "not at the desk"
TELL_NEAR_M = 1.5  # walking to tell a staff member, it stops and speaks once she is this close and in sight
TELL_SPOT_M = (1.0, 1.25, 1.5)  # ... walking to free floor this far from her
TELL_SPOTS_TRIED = 8  # the nearest few such spots are tried, then the nurse base nearest her
TELL_DEFAULT = "Excuse me, someone here needs help from a nurse."
GOODBYE = "Here we are. Goodbye, and take care."
STAY_SAY = "Here we are. I'll stay with you."  # the end of a guide with return_to_post false: it stays with them
COUNTER = "reception_counter"  # where the robot stands to talk to the receptionist
STAFF_SPOT = "reception_staff"  # where the receptionist sits (the robot faces it)
STAFF_SEAT = "R1"  # the receptionist's chair, where the desk computer is
RECEPTION_ROLES = ("receptionist",)
WARD_STAFF_ROLES = ("ward_nurse", "nurse", "staff")


SIM_ONLY_KEYS = ("person",)  # result keys for the run log only: the robot addresses people by track, never by id


RETURN_SUFFIX = "-return"  # the goal id of a walk back to post after a guide or a tell: "<action id>-return"


def wire_result(r: dict) -> dict:
    """A task result as the robot may see it on <ns>/task/result: without the sim-only keys (the receiver's person id
    on a hand_over result is for the run log and the benchmark)."""
    return {k: v for k, v in r.items() if k not in SIM_ONLY_KEYS}


def _seg_dist(x: float, y: float, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / L2))
    return math.hypot(x - (a[0] + u * dx), y - (a[1] + u * dy))


@dataclass
class TaskOutputs:
    results: list[dict] = field(default_factory=list)
    audio: list[AudioEvent] = field(default_factory=list)
    nudges: list[Nudge] = field(default_factory=list)


class RobotTasks:
    def __init__(self, world):
        self.world = world
        self._spray: dict[str, str] = {}  # action id -> fire id
        self._to_fire: dict[str, str] = {}  # action id -> fire id, walking up to it before spraying
        self._guide: Optional[dict] = None
        self._return_after: Optional[tuple] = None  # (goal id, t): the walk back to post after a guide, waiting to start
        self._following: Optional[dict] = None  # the follow action under way
        self._tell: Optional[dict] = None
        self._notes: list[dict] = []  # desk notes not read yet
        self._goal_xy: Optional[tuple] = None  # (x, y, yaw, exempt) of the goal in flight, for a re-route
        self._held_s = 0.0
        self._replanned_at = -1e9
        self._close_s = 0.0
        self._backed_at = -1e9
        self._n_safety = 0
        self.goal_id = ""
        self._goal_n = 0
        self._out = TaskOutputs()
        # False when the robot's body cannot walk (the Isaac G1 before its walking controller): tell_staff then
        # speaks from where the robot stands (the staff locator does not walk it to the staff member), and nothing
        # turns it or sends it back to its post
        self.mobile = True
        rc = getattr(getattr(getattr(world, "loaded", None), "scenario", None), "robot", None)
        self.post = (rc.x, rc.y, rc.yaw) if rc is not None else tuple(world.robot.pose)
        self._hazards_seen = (getattr(world, "_hazards", ()), self._router_version())
        world.robot_speed_cap = self._near_cap  # the world sets the cap just before it moves the robot
        self._inbox: list[tuple[str, object, float]] = []  # commands not carried out yet: (kind, payload, t)
        world.robot_commands = self.apply_pending  # the world carries them out at the start of its step

    def _result(self, aid, action, ok, reason, t, **extra) -> None:
        self._out.results.append({"id": aid, "action": action, "ok": ok, "reason": reason, "t": t, **extra})

    def _log(self, rec: dict) -> None:
        """A sim-side row for robot_tasks.jsonl (the benchmark's view; never published to the robot)."""
        self.world._log.append(("robot_tasks", rec))

    # ---- commands in: kept for the step (a route search never runs in a message callback) -----------------------
    def handle(self, msg: dict, t: float) -> None:
        """A task action from <ns>/task/action. It is carried out on the world's next step (``apply_pending``), in the
        order the commands came, so the route searches it may need run there, not in the message callback."""
        self._inbox.append(("action", dict(msg), t))

    def goal_pose(self, x, y, yaw, t) -> None:
        """A goal_pose command (carried out on the next step); no route is reported as a goal result."""
        self._inbox.append(("goal", (x, y, yaw), t))

    def cmd_vel(self, cmd, t) -> None:
        """A cmd_vel command (carried out on the next step, after any command that came before it)."""
        self._inbox.append(("cmd", cmd, t))

    def apply_pending(self, t: float) -> None:
        """Carry out the commands that came since the last step. The world calls this at the start of its step, before
        anyone moves; ``step`` calls it too, for a clock that is held."""
        while self._inbox:  # each keeps the time it came (a reply to it may come before this step)
            kind, payload, t0 = self._inbox.pop(0)
            t0 = min(t0, t)
            if kind == "action":
                self._handle_now(payload, t0)
            elif kind == "goal":
                self._goal_pose_now(*payload, t0)
            else:
                self._cmd_vel_now(payload, t0)

    def _handle_now(self, msg: dict, t: float) -> None:
        aid = str(msg.get("id") or f"a{len(self._out.results)}")
        action = msg.get("action")
        if action not in ACTIONS:
            self._result(aid, str(action), False, "unknown_action", t)
            return
        f, n0 = self._following, len(self._out.results)
        try:
            getattr(self, f"_{action}")(aid, msg, t)
        except (KeyError, TypeError, ValueError, IndexError):
            self._result(aid, action, False, "bad_request", t)
        refused = any(r.get("id") == aid and r.get("ok") is False for r in self._out.results[n0:])
        if action in WALKING and f is not None and self._following is f and not refused:
            self._end_follow(t, "superseded")  # another walking task was taken on (one refused leaves the follow on)

    # ---- actions -------------------------------------------------------------------------
    def _person(self, track) -> Optional[str]:
        pid = self.world.tracks.person(str(track))
        p = self.world.people.get(pid or "")
        return pid if p is not None and p.posture != "gone" else None

    def _ho_result(self, aid, ok, reason, t, pid: Optional[str], item_id) -> None:
        """A hand_over result; the run log row names the receiver (person) and the item. ``wire_result`` drops the
        person before the robot gets it."""
        self._out.results.append({"id": aid, "action": "hand_over", "ok": ok, "reason": reason, "t": t,
                                  "person": pid, "item": None if item_id is None else str(item_id)})

    def _hand_over(self, aid, msg, t) -> None:
        w = self.world
        item_id = str(msg["item"])
        pid = self._person(msg["to"])
        if pid is None:
            return self._ho_result(aid, False, "unknown_track", t, None, item_id)
        item = w.objects.items.get(item_id)
        r = w.robot
        if item is not None and item.holder is None and \
                math.hypot(item.x - r.x, item.y - r.y) <= TAKE_REACH_M:
            # within its reach and not held yet: it takes the item and holds it out, in one action
            item.holder, item.x, item.y = "g1", r.x, r.y
            w._log_objects(t)
        if item is None or item.holder != "g1":
            return self._ho_result(aid, False, "not_holding", t, pid, item_id)
        q = w.people[pid]
        if getattr(q, "asleep", False):
            # held out to someone asleep: nobody answers (she is not asked, and nothing is put down)
            self._log({"t": t, "kind": "hand_over", "event": "offer", "id": aid, "person": pid, "item": item.id})
            return self._ho_result(aid, False, "no_answer", t, pid, item.id)
        if msg.get("place") is not None:
            place = tuple(float(v) for v in msg["place"])
            if len(place) != 3:
                raise ValueError("place")
        else:
            place, why = self.hand_over_place((r.x, r.y), q, item.kind)
            if place is None:
                return self._ho_result(aid, False, why, t, pid, item.id)
        why = w._check_hand_over("g1", pid, item.id, place)
        lying = q.bed is not None and q.posture == "lying" and not q.sitting_up
        name = self._item_name(item)
        if why == "out_of_reach_receiver" and lying and self._receiver_ok(q, place, item.kind, sitting_up=True):
            # lying flat (a tray never counts then): the offer stays open and she is asked for it; accepting sits her
            # up, and the hand-over is checked again when she answers
            self._out.nudges.append(Nudge(person=pid, t=t, card=HAND_OVER_CARD, message=(
                f"The robot is holding out the {name} for you. You are lying flat: if you want it, accept it "
                f"(accept_item) and you sit up in bed to take it, with no separate sit_up_in_bed needed; if not, "
                f"refuse it (refuse_item).")))
        elif why != "ok":
            return self._ho_result(aid, False, why, t, pid, item.id)
        elif q.kind != "static" and "accept_item" in ROLE_INTENTS.get(q.inst.role, ()):
            # someone up and about: nothing else tells them an offer is open, and it closes after a few seconds
            no = "reject_item" if "reject_item" in ROLE_INTENTS.get(q.inst.role, ()) else "refuse_item"
            self._out.nudges.append(Nudge(person=pid, t=t, card=HAND_OVER_CARD, message=(
                f"The robot is holding out the {name} to you. If you want it, take it (accept_item); if not, refuse "
                f"it ({no}).")))
        old = w._offers.get(pid)
        if old is not None and old["giver"] == "g1":  # one result per action: close the old offer
            self._ho_result(old["intent"], False, "superseded", t, pid, old["item"])
        w._offers[pid] = {"giver": "g1", "item": item.id, "place": place, "t": t, "intent": aid}
        # the moment the robot holds it out (a latency is measured to here, not to the receiver's answer)
        self._log({"t": t, "kind": "hand_over", "event": "offer", "id": aid, "person": pid, "item": item.id})

    @staticmethod
    def _item_name(item) -> str:
        """What a person is told the robot holds out: an instrument by its name, else the kind and its label."""
        what = item.kind.replace("_", " ")
        if item.kind == "instrument" and item.label:
            return item.label
        return f"{what} ({item.label})" if item.label else what

    def _receiver_ok(self, q, place, kind, sitting_up=None) -> bool:
        w = self.world
        in_bed = q.bed is not None and q.posture == "lying"
        return receiver_zone_ok(q.posture, in_bed, q.sitting_up if sitting_up is None else sitting_up, (q.x, q.y),
                                q.yaw, w.site.beds.get(q.bed or ""), place, item_kind=kind)

    def hand_over_place(self, giver_xy, q, kind) -> tuple[Optional[tuple], str]:
        """(place, "ok"): where the robot standing at giver_xy holds an item out to person q, the point in reach of
        both nearest to half a metre from them towards the robot. A patient lying in bed is taken as sitting up (she is
        asked to). (None, out_of_reach_giver) when no such point is in the robot's reach, (None,
        out_of_reach_receiver) when the robot could reach but she could not."""
        w = self.world
        z = 0.85 if q.bed else 1.1
        d = max(math.hypot(giver_xy[0] - q.x, giver_xy[1] - q.y), 1e-6)
        ax, ay = q.x + (giver_xy[0] - q.x) / d * 0.5, q.y + (giver_xy[1] - q.y) / d * 0.5
        n = int(round(PLACE_SEARCH_M / PLACE_GRID_M))
        best, bd, giver_ok = None, math.inf, False
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                p = (q.x + i * PLACE_GRID_M, q.y + j * PLACE_GRID_M, z)
                if not giver_reach_ok(giver_xy, w.robot_reach, p):
                    continue
                giver_ok = True
                if not self._receiver_ok(q, p, kind, sitting_up=True if q.bed is not None else None):
                    continue
                e = math.hypot(p[0] - ax, p[1] - ay)
                if e < bd:
                    best, bd = p, e
        if best is not None:
            return best, "ok"
        return None, "out_of_reach_receiver" if giver_ok else "out_of_reach_giver"

    def _reach_spots(self, q, kind: Optional[str]) -> list[tuple[float, float]]:
        """Where the robot can stand to hand something to q, best first: beside the middle of the bed for a patient in
        bed; with an item in hand, in front of or beside someone up and about (never behind); else []."""
        w, r = self.world, self.world.robot
        free = getattr(r.router, "is_free", None)
        out = []
        if q.bed is not None and q.posture == "lying":
            b = w.site.beds.get(q.bed)
            if b is None:
                return []
            for dist in BEDSIDE_M:
                for side in (1.0, -1.0):
                    out.append((b.x - side * dist * math.sin(b.yaw), b.y + side * dist * math.cos(b.yaw)))
        elif kind is not None:
            for dist in REACH_SPOT_M:
                for deg in (0, 30, -30, 60, -60, 90, -90):
                    a = q.yaw + math.radians(deg)
                    out.append((q.x + dist * math.cos(a), q.y + dist * math.sin(a)))
        ok = [(x, y) for x, y in out if not w.site.grid.blocked(x, y) and (free is None or free(x, y))
              and self.hand_over_place((x, y), q, kind or "tray")[0] is not None]
        # the nearer side first, among spots equally good
        return sorted(ok, key=lambda s: (out.index(s) // 2 if q.bed is not None else 0,
                                         math.hypot(s[0] - r.x, s[1] - r.y)))

    def _take(self, aid, msg, t) -> None:
        w = self.world
        item_id = str(msg["item"])
        item = w.objects.items.get(item_id)
        if item is None or item.holder is not None:
            return self._result(aid, "take", False, "unknown_item", t, item=item_id)
        if math.hypot(item.x - w.robot.x, item.y - w.robot.y) > TAKE_REACH_M:
            return self._result(aid, "take", False, "out_of_reach", t, item=item_id)
        item.holder, item.x, item.y = "g1", w.robot.x, w.robot.y
        w._log_objects(t)
        self._result(aid, "take", True, "ok", t, item=item_id)

    def _extinguish(self, aid, msg, t) -> None:
        w = self.world
        f = w.objects.fires.get(str(msg["fire"]))
        if f is None or f.out:
            return self._result(aid, "extinguish", False, "no_such_fire", t)
        if math.hypot(f.x - w.robot.x, f.y - w.robot.y) <= EXTINGUISH_RANGE_M:
            self._spray[aid] = f.id
            return
        if not self.mobile:
            return self._result(aid, "extinguish", False, "out_of_range", t)
        # walk up to it first (the goal is its own: a result for aid comes when the spraying ends)
        r = w.robot
        free = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        spots = []
        for k in range(16):
            a = 2 * math.pi * k / 16
            x, y = f.x + SPRAY_FROM_M * math.cos(a), f.y + SPRAY_FROM_M * math.sin(a)
            if not w.site.grid.blocked(x, y) and (free is None or free(x, y)) and w._los((x, y), (f.x, f.y)):
                spots.append((math.hypot(x - r.x, y - r.y), x, y))
        for _, x, y in sorted(spots)[:6]:
            if self._set_goal(aid, x, y, math.atan2(f.y - y, f.x - x), t):
                self._end_guide(t, "superseded")
                self._to_fire[aid] = f.id
                return
        self._result(aid, "extinguish", False, "no_route", t)

    def _press_call(self, aid, msg, t) -> None:
        w = self.world
        about = self._person(msg["track"]) if msg.get("track") is not None else None
        bx, by = self.call_point()
        ev = w._event(t, "person", "bell_robot", bx, by, 2.0, CHIME_DB)
        self._out.audio.append(ev)
        text = self.call_message(about)
        heard = []
        for sid, q in w.people.items():
            if q.posture == "gone" or q.kind == "static" or q.inst.role not in WARD_STAFF_ROLES + RECEPTION_ROLES:
                continue
            if audio.route(ev, Pose(q.x, q.y, q.yaw), sid, q.inst.hearing_loss_db, w.rng,
                           walls_between=w.walls_for(ev)) is not None:
                heard.append(sid)  # even faintly: a call chime is known when heard
                self._out.nudges.append(Nudge(person=sid, t=t, card=CALL_CARD, message=text))
        self._result(aid, "press_call", True, "answered" if heard else "unanswered", t)

    def call_point(self) -> tuple[float, float]:
        """Where the robot's staff call chimes: the nearest nurse base or triage room, in the robot's own compartment
        when there is one there (so A&E calls triage, not a ward)."""
        w, r = self.world, self.world.robot
        rooms = [rm for rm in w.site.rooms.values() if rm.use in CALL_POINT_USES]
        if not rooms:
            return r.x, r.y
        mine = w._compartment_at(r.x, r.y)
        same = [rm for rm in rooms if mine is not None and rm.compartment == mine]
        rm = min(same or rooms, key=lambda rm: math.hypot(rm.centre[0] - r.x, rm.centre[1] - r.y))
        return rm.centre

    def _room_of(self, x: float, y: float, default: str = "the building") -> str:
        w = self.world
        rid = w.site.room_at(x, y)
        return w.site.rooms[rid].label.split(",")[0] if rid else default

    def call_message(self, pid: Optional[str]) -> str:
        """What a staff member who hears the robot's staff call is told: where the robot is, which call lights are on,
        and who needs them when the robot said."""
        w = self.world
        out = f"The robot pressed the staff call from {self._room_of(w.robot.x, w.robot.y)}"
        lit = sorted(b.bed_id for b in w.objects.bells.values() if b.active)
        if lit:
            out += f"; call light on at bed {', '.join(lit)}"
        p = w.people.get(pid or "")
        if p is not None:
            name = p.inst.display_name if pid in w.named else "they have not said their name"
            return (f"{out}. The person who needs a nurse is person id {pid} ({name}), in {self._room_of(p.x, p.y)} "
                    f"at x {p.x:.1f}, y {p.y:.1f}: go to them (approach {pid}), help, then go back to your work.")
        return f"{out}. Someone there needs a nurse: go to them, help, then go back to your work."

    def _approach(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["track"])
        if pid is None:
            return self._result(aid, "approach", False, "unknown_track", t)
        if not self.mobile:
            return self._result(aid, "approach", False, "cannot_walk", t)
        p, r = w.people[pid], w.robot
        d = math.hypot(r.x - p.x, r.y - p.y)
        held = next((i.kind for i in w.objects.items.values() if i.holder == "g1"), None)
        spots = self._reach_spots(p, held)
        if spots:
            # a patient in bed, or someone it carries something for: it ends where it can hand it over
            if self.hand_over_place((r.x, r.y), p, held or "tray")[0] is not None and (p.bed is None or any(
                    math.hypot(r.x - x, r.y - y) < 0.3 for x, y in spots)):
                r.turn_to(math.atan2(p.y - r.y, p.x - r.x))
                return self._result(aid, "approach", True, "already_there", t)
            for x, y in spots:
                if self._set_goal(aid, x, y, math.atan2(p.y - y, p.x - x), t, exempt=pid):
                    self._end_guide(t, "superseded")
                    return self._result(aid, "approach", True, "started", t)
        if d <= APPROACH_M + 0.2:
            r.turn_to(math.atan2(p.y - r.y, p.x - r.x))
            return self._result(aid, "approach", True, "already_there", t)
        ux, uy = (r.x - p.x) / d, (r.y - p.y) / d
        x, y = p.x + ux * APPROACH_M, p.y + uy * APPROACH_M
        if not self._set_goal(aid, x, y, math.atan2(-uy, -ux), t, exempt=pid):
            return self._result(aid, "approach", False, "no_route", t)
        self._end_guide(t, "superseded")
        self._result(aid, "approach", True, "started", t)

    # ---- guiding -------------------------------------------------------------------------
    def _guide_start(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["track"])
        if pid is None:
            return self._result(aid, "guide_start", False, "unknown_track", t)
        place = str(msg["place"])
        xy = w._resolve_place(place)
        if xy is None:
            return self._result(aid, "guide_start", False, "unknown_place", t)
        back = msg.get("return_to_post", True)
        if not isinstance(back, bool):
            raise TypeError("return_to_post")
        room = place if place in w.site.rooms else None
        xy = self._clear_of_person(xy, w.people[pid], room)
        # the person it guides is not exempt from its keeping clear: a walk ending where they stand drove into them
        if not self._set_goal(aid, xy[0], xy[1], None, t):
            return self._result(aid, "guide_start", False, "no_route", t)
        self._end_guide(t, "superseded")  # an earlier guide (perhaps waiting at its place) is over
        lm = w.site.landmarks.get(place)
        label = (lm.label.split(",")[0] if lm is not None and lm.label else place.replace("_", " "))
        r = w.robot
        pts = [(r.x, r.y)] + list(r._path)
        route_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        self._guide = {"id": aid, "pid": pid, "place": place, "goal": xy, "room": room, "phase": "walking",
                       "held": False, "held_s": 0.0, "t0": t, "seat": bool(msg.get("seat")), "seat_id": None,
                       "return": back}
        w.guide_over.discard(pid)  # guided again: they may follow the robot
        self._log({"t": t, "kind": "guide", "event": "start", "id": aid, "person": pid, "place": place,
                   "x": round(xy[0], 3), "y": round(xy[1], 3), "room": room, "route_m": round(route_m, 2)})
        self._out.nudges.append(Nudge(person=pid, t=t, card=GUIDE_CARD, message=(
            f"The robot is offering to walk you to {label}. If you want to go there, follow it (follow g1): "
            f"walk a little behind it and it will wait for you.")))
        self._result(aid, "guide_start", True, "started", t)

    def _clear_of_person(self, xy, q, room: Optional[str]) -> tuple[float, float]:
        """Where a guide's walk ends: the place itself, or when the person it guides stands within GUIDE_CLEAR_M of it,
        the nearest free floor (clear for the robot's body, in the place's room when it names one) that far from
        them, on the robot's side of them if there is any (beyond them it would have to walk past them, and they would
        hold it), the spot nearest the robot first. The place unchanged when there is none."""
        w = self.world
        if math.hypot(xy[0] - q.x, xy[1] - q.y) >= GUIDE_CLEAR_M:
            return xy
        r = w.robot
        clear = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        others = [(p.x, p.y) for p in w.people.values() if p is not q and p.posture not in ("gone",) and
                  p.kind != "static"]
        for near_side in (True, False):
            for k in range(1, 9):
                dist = 0.25 * k
                ring = []
                for j in range(16):
                    a = 2 * math.pi * j / 16
                    x, y = xy[0] + dist * math.cos(a), xy[1] + dist * math.sin(a)
                    if math.hypot(x - q.x, y - q.y) < GUIDE_CLEAR_M or w.site.grid.blocked(x, y) \
                            or (near_side and (x - q.x) * (r.x - q.x) + (y - q.y) * (r.y - q.y) <= 0.0) \
                            or (clear is not None and not clear(x, y)) \
                            or any(math.hypot(x - ox, y - oy) < PERSON_KEEP_M for ox, oy in others) \
                            or (room is not None and w.site.room_at(x, y) != room):
                        continue
                    ring.append((math.hypot(x - r.x, y - r.y), x, y))
                if ring:
                    _, x, y = min(ring)
                    return x, y
        return xy

    def _end_guide(self, t, reason: str) -> None:
        g, self._guide = self._guide, None
        self.world.robot.speed_scale = 1.0
        if g is not None:
            p = self.world.people.get(g["pid"])
            if p is not None and p.plan and p.plan[0].kind == "walk" and p.plan[0].trail:
                self.world.trail_released.add(g["pid"])  # following ends with the guide
            self.world.guide_over.add(g["pid"])  # and they do not take it up again unless guided again
            self._log({"t": t, "kind": "guide", "event": "end", "id": g["id"], "person": g["pid"], "reason": reason})

    def _arrived(self, g: dict) -> bool:
        p = self.world.people.get(g["pid"])
        if p is None or p.posture == "gone":
            return False
        if g.get("seat_id"):  # shown a seat: arrived once they sit in it (or in any seat), or after the wait
            return p.posture == "seated"
        if math.hypot(p.x - g["goal"][0], p.y - g["goal"][1]) <= NEAR_PLACE_M:
            return True
        return g["room"] is not None and self.world.site.room_at(p.x, p.y) == g["room"]

    def _guide_step(self, t: float, dt: float) -> None:
        g, w = self._guide, self.world
        p = w.people.get(g["pid"])
        if g["phase"] == "walking":
            if p is None or p.posture == "gone":
                w.robot.speed_scale = 1.0
                return
            d = math.hypot(p.x - w.robot.x, p.y - w.robot.y)
            if g["held"] and d <= GUIDE_RESUME_M:
                g["held"], g["held_s"] = False, 0.0
            elif not g["held"] and d > GUIDE_HOLD_M:
                g["held"] = True
            if g["held"]:
                g["held_s"] += dt
                if g["held_s"] >= GUIDE_GIVE_UP_S:  # they never came: stop where it is
                    if w.robot.has_goal:
                        self._goal(self.goal_id, False, "cancelled", t)
                    w.robot.cancel_goal()
                    self._end_guide(t, "left_behind")
                    return
            w.robot.speed_scale = 0.0 if g["held"] else 1.0
            return
        # waiting at the place
        arrived = self._arrived(g)
        if arrived or t - g["t_arrive"] >= GUIDE_ARRIVE_WAIT_S:
            # back to its post unless told to stay, and never while the fire alarm sounds (its post may be by the fire)
            back = self.mobile and g.get("return", True) and not getattr(w, "alarm_active", False)
            # told to stay with them: no goodbye
            line = GOODBYE if g.get("return", True) else STAY_SAY
            self._out.audio.append(w.robot_say(line, t).model_copy(update={"addressee": g["pid"]}))
            self._end_guide(t, "arrived" if arrived else "timeout")
            if back:
                # it lets the person it brought (and anyone else close by) go on past first: turning back at once in a
                # doorway walks it into them
                self._return_after = (f"{g['id']}-return", t)

    def free_seat(self, room: Optional[str]) -> Optional[str]:
        """The free seat nearest the robot that its camera could see from here, preferring one in the room."""
        w, r = self.world, self.world.robot
        rng = w.loaded.scenario.robot.range_m
        taken = {q.seat for q in w.people.values() if q.posture != "gone" and q.seat}
        taken |= {q.claim for q in w.people.values() if q.posture != "gone" and getattr(q, "claim", None)}
        seen = [c for c in w.site.seats.values()
                if not c.staff_only and c.id not in taken and math.hypot(c.x - r.x, c.y - r.y) <= rng
                and w._los((r.x, r.y), (c.x, c.y))]
        if not seen:
            return None
        inside = [c for c in seen if room is not None and w.site.room_at(c.x, c.y) == room]
        return min(inside or seen, key=lambda c: math.hypot(c.x - r.x, c.y - r.y)).id

    def aside_of_seat(self, c, pid: Optional[str]) -> tuple[float, float]:
        """Where the robot stands to show seat c: beside the seat's front, out of the way the person walks in (the
        free spot furthest from the line from them to the seat's front), never in the chairs."""
        w = self.world
        fx, fy = c.x + 0.85 * math.cos(c.yaw), c.y + 0.85 * math.sin(c.yaw)
        px, py = -math.sin(c.yaw), math.cos(c.yaw)
        q = w.people.get(pid or "")
        r_ = w.robot.router
        for free in (getattr(r_, "is_clear", None), getattr(r_, "is_free", None)):  # strict first, then the soft band
            if free is None:
                continue
            spot = self._aside(c, q, fx, fy, px, py, free)
            if spot is not None:
                return spot
        # nowhere beside it: back off along the line from the seat through the robot, past the seat's front, to a spot
        # its body is clear at; else it shows the seat from where it stands
        d = max(math.hypot(w.robot.x - c.x, w.robot.y - c.y), 1e-6)
        for free in (getattr(r_, "is_clear", None), getattr(r_, "is_free", None)):
            for k in (1.0, 1.5, 0.5, 2.0):
                x = c.x + (w.robot.x - c.x) / d * (SEAT_BESIDE_M + k)
                y = c.y + (w.robot.y - c.y) / d * (SEAT_BESIDE_M + k)
                if not w.site.grid.blocked(x, y) and (free is None or free(x, y)):
                    return x, y
        return w.robot.x, w.robot.y

    def _aside(self, c, q, fx, fy, px, py, free) -> Optional[tuple[float, float]]:
        w = self.world
        best, score = None, -math.inf
        for out in (0.4, 0.9, 1.4):
            for aside in (SEAT_ASIDE_M, 1.5 * SEAT_ASIDE_M):
                for side in (1.0, -1.0):
                    x = fx + out * math.cos(c.yaw) + side * aside * px
                    y = fy + out * math.sin(c.yaw) + side * aside * py
                    if w.site.grid.blocked(x, y) or not free(x, y):
                        continue
                    lane = _seg_dist(x, y, (q.x, q.y), (fx, fy)) if q is not None else 1.0
                    them = math.hypot(x - q.x, y - q.y) if q is not None else 2.0
                    if them < 1.0:
                        continue  # where they stand now
                    sc = min(lane, 1.5) + 0.5 * min(them, 2.0) - 0.2 * math.hypot(x - w.robot.x, y - w.robot.y) / 5.0
                    if sc > score and self._clean_route(x, y):
                        best, score = (x, y), sc
        return best

    def _clean_route(self, x: float, y: float) -> bool:
        """The robot's route to (x, y) exists and stays out of the clearance band past its first 0.3 m."""
        w = self.world
        r = w.robot
        route = r.router.route((r.x, r.y), (x, y))
        soft = getattr(r.router, "is_free", None)
        if route is None or soft is None:
            return route is not None
        pts = [(r.x, r.y)] + [tuple(q) for q in route]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            k = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.1))
            for i in range(k + 1):
                px, py = x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k
                if math.hypot(px - r.x, py - r.y) > 0.3 and not soft(px, py):
                    return False
        return True

    def _to_seat(self, g: dict, t: float) -> bool:
        """At the place of a guide to a seat: pick a free seat and step aside of it first (so the person's way to it is
        clear), then show it (``_show_seat``, when that walk ends). False when no free seat is in sight."""
        w = self.world
        sid = self.free_seat(g["room"])
        if sid is None:
            ev = w.robot_say(NO_SEAT_SAY, t).model_copy(update={"addressee": g["pid"]})
            self._out.audio.append(ev)
            return False
        c = w.site.seats[sid]
        g["seat_id"], g["goal"], g["room"], g["shown"] = sid, (c.x, c.y), None, False
        x, y = self.aside_of_seat(c, g["pid"])
        if not self._set_goal(g["id"], x, y, math.atan2(c.y - y, c.x - x), t):
            w.robot.turn_to(math.atan2(c.y - w.robot.y, c.x - w.robot.x))  # nowhere to stand aside: show it from here
            self._show_seat(g, t)
            g["phase"], g["t_arrive"] = "waiting", t
        return True

    def _show_seat(self, g: dict, t: float) -> None:
        w = self.world
        sid = g["seat_id"]
        g["shown"] = True
        self._out.audio.append(w.robot_say(SEAT_SAY, t).model_copy(update={"addressee": g["pid"]}))
        self._out.nudges.append(Nudge(person=g["pid"], t=t, card=SEAT_CARD, message=(
            f"The robot is showing you a free seat, seat {sid}. If you want it, sit down there (sit_down {sid}).")))
        self._log({"t": t, "kind": "guide", "event": "seat", "id": g["id"], "person": g["pid"], "seat": sid})

    def _guide_stop(self, aid, msg, t) -> None:
        if self.world.robot.has_goal and self.world.robot._had_goal:
            self._goal(self.goal_id, False, "cancelled", t)
        self.world.robot.cancel_goal()
        self._end_guide(t, "stopped")
        self._end_follow(t, "stopped")
        self._result(aid, "guide_stop", True, "ok", t)

    # ---- following a person ---------------------------------------------------------------
    def _follow(self, aid, msg, t) -> None:
        """Stay with a person: walk up to them and keep FOLLOW_NEAR_M-FOLLOW_FAR_M from them as they move, until
        guide_stop, a goal_pose, cmd_vel, another walking task, or they are gone."""
        w = self.world
        pid = self._person(msg["track"])
        if pid is None:
            return self._result(aid, "follow", False, "no_person", t)
        if not self.mobile:
            return self._result(aid, "follow", False, "cannot_walk", t)
        r = w.robot
        if r.has_goal and r._had_goal:  # a walk under way (its own or a task's) is over
            self._goal(self.goal_id, False, "superseded", t)
            if self._tell is not None:
                self._result(self._tell["id"], "tell_staff", False, "superseded", t)
                self._tell = None
            r.cancel_goal()
        self._end_guide(t, "superseded")
        self._end_follow(t, "superseded")  # following someone else (or the same person afresh)
        self._following = {"id": aid, "pid": pid, "spot": None, "retry_at": -math.inf}
        self._log({"t": t, "kind": "follow", "event": "start", "id": aid, "person": pid})
        self._result(aid, "follow", True, "following", t)
        self._follow_step(t)

    def _end_follow(self, t: float, reason: str) -> None:
        f, self._following = self._following, None
        if f is None:
            return
        r = self.world.robot
        if str(self.goal_id).startswith(FOLLOW_GOAL + "-") and r.has_goal and r._had_goal:
            r.cancel_goal()
        self._log({"t": t, "kind": "follow", "event": "end", "id": f["id"], "person": f["pid"], "reason": reason})

    def _follow_spot(self, p) -> Optional[tuple[float, float]]:
        """Free floor FOLLOW_GAP_M (or a little nearer or further) from the person, on the robot's side of them, clear
        for its body: the one nearest the robot."""
        w, r = self.world, self.world.robot
        clear = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        base = math.atan2(r.y - p.y, r.x - p.x)
        best, bd = None, math.inf
        for dist in (FOLLOW_GAP_M, FOLLOW_NEAR_M + 0.1, FOLLOW_FAR_M - 0.1):
            for deg in (0, 30, -30, 60, -60, 90, -90):
                a = base + math.radians(deg)
                x, y = p.x + dist * math.cos(a), p.y + dist * math.sin(a)
                if w.site.grid.blocked(x, y) or (clear is not None and not clear(x, y)) \
                        or not w._los((x, y), (p.x, p.y)):
                    continue
                d = math.hypot(x - r.x, y - r.y) + 0.3 * abs(deg) / 90.0
                if d < bd:
                    best, bd = (x, y), d
            if best is not None:
                return best
        return None

    def _follow_step(self, t: float) -> None:
        f, w = self._following, self.world
        r = w.robot
        p = w.people.get(f["pid"])
        if p is None or p.posture == "gone":
            return self._end_follow(t, "person_gone")
        walking = r.has_goal and r._had_goal and str(self.goal_id).startswith(FOLLOW_GOAL + "-")
        d = math.hypot(p.x - r.x, p.y - r.y)
        # nearer than FOLLOW_NEAR_M (she came towards it): it steps back to the gap, on its side of her
        if d >= FOLLOW_NEAR_M and (d <= FOLLOW_GAP_M or (not walking and d <= FOLLOW_FAR_M)):
            if walking:
                r.cancel_goal()  # close enough: stop here
                f["spot"] = None
            if not r.has_goal:
                face = math.atan2(p.y - r.y, p.x - r.x)
                if abs(math.atan2(math.sin(face - r.yaw), math.cos(face - r.yaw))) > 0.35:
                    r.turn_to(face)
            return
        if t < f["retry_at"]:
            return
        spot = self._follow_spot(p)
        if spot is None:
            f["retry_at"] = t + FOLLOW_RETRY_S
            return
        if walking and f["spot"] is not None and math.hypot(spot[0] - f["spot"][0], spot[1] - f["spot"][1]) \
                < FOLLOW_REPLAN_M:
            return  # the walk under way still goes to about the right place
        gid = f"{FOLLOW_GOAL}-{f['id']}"
        if self._set_goal(gid, spot[0], spot[1], math.atan2(p.y - spot[1], p.x - spot[0]), t):
            f["spot"] = spot
        else:
            f["retry_at"] = t + FOLLOW_RETRY_S

    # ---- telling the staff ---------------------------------------------------------------
    def _staff_to_tell(self) -> tuple[Optional[str], list]:
        """(staff person, [(x, y, yaw), ...] where the robot may stand to tell them, best first): the receptionist at
        the counter; else the ward staff member nearest the robot, from where it stands ([]) when she is within
        TELL_RANGE_M in sight, else from free floor near her, else from the nurse base nearest her. (None, []) when
        the scenario has neither."""
        w, r = self.world, self.world.robot
        present = [(pid, q) for pid, q in w.people.items() if q.posture != "gone" and q.kind != "static"]
        desk = [pid for pid, q in present if q.inst.role in RECEPTION_ROLES]
        c, s = w.site.landmarks.get(COUNTER), w.site.landmarks.get(STAFF_SPOT)
        if desk and c is not None:
            yaw = math.atan2(s.y - c.y, s.x - c.x) if s is not None else None
            return desk[0], [(c.x, c.y, yaw)]
        ward = [(pid, q) for pid, q in present if q.inst.role in WARD_STAFF_ROLES]
        if not ward:
            return None, []
        pid, q = min(ward, key=lambda e: math.hypot(e[1].x - r.x, e[1].y - r.y))
        if self._in_earshot(q):
            return pid, []
        bx, by = w._nurse_base(q.x, q.y)
        return pid, self._spots_near(q) + [(bx, by, None)]

    def _clear_here(self) -> bool:
        """The robot stands where its body is clear of the furniture (out of its router's clearance band)."""
        r = self.world.robot
        clear = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        return clear is None or bool(clear(r.x, r.y))

    def _in_earshot(self, q, within: float = TELL_RANGE_M) -> bool:
        """q is within this of the robot, with no wall between them."""
        w, r = self.world, self.world.robot
        return math.hypot(q.x - r.x, q.y - r.y) <= within and w._los((r.x, r.y), (q.x, q.y))

    def _spots_near(self, q) -> list[tuple]:
        """Free floor TELL_SPOT_M from q that can see her, facing her: the nearest to the robot first."""
        w, r = self.world, self.world.robot
        free = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        spots = []
        for dist in TELL_SPOT_M:
            for k in range(16):
                a = 2 * math.pi * k / 16
                x, y = q.x + dist * math.cos(a), q.y + dist * math.sin(a)
                if not w.site.grid.blocked(x, y) and (free is None or free(x, y)) and w._los((x, y), (q.x, q.y)):
                    spots.append((math.hypot(x - r.x, y - r.y), x, y))
        return [(x, y, math.atan2(q.y - y, q.x - x)) for _, x, y in sorted(spots)[:TELL_SPOTS_TRIED]]

    def _tell_staff(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["about"])
        if pid is None:
            return self._result(aid, "tell_staff", False, "unknown_track", t)
        message = msg.get("message")
        if message is not None and not isinstance(message, str):
            raise TypeError("message")
        message = (message or "").strip()[:400] or TELL_DEFAULT
        staff, spots = self._staff_to_tell()
        if staff is None:
            self._log({"t": t, "kind": "tell", "id": aid, "about": pid, "staff": None, "heard": False,
                       "reason": "no_staff_to_tell"})
            return self._result(aid, "tell_staff", False, "no_staff_to_tell", t)
        if self._tell is not None:
            self._result(self._tell["id"], "tell_staff", False, "superseded", t)
        self._tell = {"id": aid, "about": pid, "staff": staff, "message": message, "yaw0": w.robot.yaw,
                      "walked": False}
        if not self.mobile or not spots or math.hypot(w.robot.x - spots[0][0], w.robot.y - spots[0][1]) <= 0.3:
            return self._speak_tell(t)
        for x, y, yaw in spots:
            if self._set_goal(aid, x, y, yaw, t):
                self._tell["walked"] = True
                return
        self._tell = None
        return self._result(aid, "tell_staff", False, "no_route", t)

    def _speak_tell(self, t: float) -> None:
        tl, self._tell = self._tell, None
        w = self.world
        ev = w.robot_say(tl["message"], t).model_copy(update={"addressee": tl["staff"]})
        self._out.audio.append(ev)
        q, p = w.people.get(tl["staff"]), w.people.get(tl["about"])
        heard = False
        if q is not None and q.posture != "gone" and math.hypot(q.x - w.robot.x, q.y - w.robot.y) <= TELL_RANGE_M:
            h = audio.route(ev, Pose(q.x, q.y, q.yaw), tl["staff"], q.inst.hearing_loss_db, w.rng,
                            walls_between=w.walls_for(ev))
            heard = h is not None and not h.partial
        if heard and p is not None:
            self._out.nudges.append(Nudge(person=tl["staff"], t=t, card=TELL_CARD,
                                          message=self.tell_message(tl["message"], tl["about"], q)))
        reason = "told" if heard else "staff_not_at_desk"
        self._log({"t": t, "kind": "tell", "id": tl["id"], "about": tl["about"], "staff": tl["staff"],
                   "heard": heard, "reason": reason})
        self._result(tl["id"], "tell_staff", heard, reason, t)
        if self.mobile and tl.get("walked") and not getattr(w, "alarm_active", False):
            # walked to tell them: back to its post, as after a guide (never while the fire alarm sounds)
            x, y, yaw = self.post
            if self._set_goal(f"{tl['id']}-return", x, y, yaw, t):
                return
        if self.mobile:  # face the way it faced before, back towards the visitors
            w.robot.turn_to(tl["yaw0"])

    # ---- the desk computer ---------------------------------------------------------------
    def _desk_staff(self) -> Optional[str]:
        w = self.world
        return next((pid for pid, q in w.people.items()
                     if q.posture != "gone" and q.kind != "static" and q.inst.role in RECEPTION_ROLES), None)

    def _note_desk(self, aid, msg, t) -> None:
        pid = self._person(msg["about"])
        if pid is None:
            return self._result(aid, "note_desk", False, "unknown_track", t)
        message = msg["message"]
        if not isinstance(message, str) or not message.strip():
            raise TypeError("message")
        staff = self._desk_staff()
        if staff is None:
            return self._result(aid, "note_desk", False, "no_desk_staff", t)
        self._notes.append({"id": aid, "about": pid, "staff": staff, "message": message.strip()[:400], "t": t})
        self._result(aid, "note_desk", True, "noted", t)
        self._read_notes(t)

    def _at_desk(self, q) -> bool:
        seat = self.world.site.seats.get(q.inst.seat or STAFF_SEAT)
        if seat is None:
            return False
        return math.hypot(q.x - seat.x, q.y - seat.y) <= DESK_M

    def _read_notes(self, t: float) -> None:
        w, left = self.world, []
        for n in self._notes:
            q, p = w.people.get(n["staff"]), w.people.get(n["about"])
            if q is None or q.posture == "gone" or p is None:
                continue  # nobody will read it
            if not self._at_desk(q):
                left.append(n)
                continue
            self._out.nudges.append(Nudge(person=n["staff"], t=t, card=NOTE_CARD,
                                          message=self.note_message(n["message"], n["about"])))
            self._log({"t": t, "kind": "note", "id": n["id"], "about": n["about"], "staff": n["staff"], "read": True,
                       "left_at": n["t"]})
        self._notes = left

    def note_message(self, message: str, pid: str) -> str:
        """What the receptionist sees on her screen."""
        w = self.world
        p = w.people[pid]
        name = p.inst.display_name if pid in w.named else "they have not said their name"
        rid = w.site.room_at(p.x, p.y)
        room = w.site.rooms[rid].label.split(",")[0] if rid else "the building"
        return (f"A note from the robot appears on your computer screen: \"{message}\" It is about person id "
                f"{pid} ({name}), now in {room}. The robot is dealing with them. You do not need to get up unless the "
                f"note says they need a member of staff; carry on with your work.")

    def tell_message(self, message: str, pid: str, staff) -> str:
        """What the staff member is told, besides hearing it: who needs help, where, and what to do."""
        w = self.world
        p = w.people[pid]
        name = p.inst.display_name if pid in w.named else "they have not said their name"
        rid = w.site.room_at(p.x, p.y)
        room = w.site.rooms[rid].label.split(",")[0] if rid else "the building"
        if staff.inst.seat:
            then = f"help them, then go back to your chair (seat {staff.inst.seat})"
        else:
            then = "help, then go back to your work"
        return (f"The robot tells you: \"{message}\" The person who needs help is person id {pid} "
                f"({name}), in {room} at x {p.x:.1f}, y {p.y:.1f}. If you can help, go to them (approach {pid}), "
                f"{then}.")

    # ---- keeping clear of people -------------------------------------------------------------
    def _start_return(self, t: float) -> None:
        """The walk back to post after a guide starts once nobody is within RETURN_CLEAR_M of the robot, or after
        RETURN_WAIT_S. A new walk, a follow or guide, cmd_vel or the fire alarm in the meantime calls it off."""
        ra = self._return_after
        if ra is None:
            return
        w = self.world
        r = w.robot
        walking = r.has_goal and r._had_goal  # a turn on the spot (facing someone) is no new walk
        if walking and str(self.goal_id).startswith(SAFETY):
            return  # a step back from someone too close: the walk back waits for it to end
        if walking or self._guide is not None or self._following is not None or getattr(w, "alarm_active", False) \
                or abs(getattr(r, "_v", 0.0)) > 1e-6 or abs(getattr(r, "_w", 0.0)) > 1e-6:
            self._return_after = None
            return
        near = any(math.hypot(q.x - r.x, q.y - r.y) < RETURN_CLEAR_M for _, q in self._others())
        if near and t - ra[1] < RETURN_WAIT_S:
            return
        self._return_after = None
        x, y, yaw = self.post
        self._set_goal(ra[0], x, y, yaw, t)

    def _others(self) -> list:
        w = self.world
        return [(pid, q) for pid, q in w.people.items() if q.posture != "gone" and q.kind != "static"]

    def _blocker(self) -> Optional[str]:
        """Someone ahead of the walking robot, nearer than its goal, or anyone inside PERSON_KEEP_M in front."""
        w = self.world
        r = w.robot
        c, s_ = math.cos(r.yaw), math.sin(r.yaw)
        left = math.inf
        if self._goal_xy is not None:
            left = math.hypot(self._goal_xy[0] - r.x, self._goal_xy[1] - r.y)
        for pid, q in self._others():
            dx, dy = q.x - r.x, q.y - r.y
            ahead, lat = dx * c + dy * s_, abs(-dx * s_ + dy * c)
            if ahead <= 0.0:
                continue
            d = math.hypot(dx, dy)
            in_front = ahead >= 0.5 * d  # within 60 degrees of its heading: someone beside it is not in its way
            before_goal = ahead < left + 0.3  # someone beyond the end of its walk is not in its way
            exempt = self._goal_xy[3] if self._goal_xy is not None else None
            if pid == exempt:
                # the person it walks up to (a hand-over may need it close): only when really close
                if d < TARGET_KEEP_M and in_front:
                    return pid
                continue
            if in_front and (d < PERSON_KEEP_M or (d < STOP_CLOSE_M and before_goal)):
                return pid
            if ahead < STOP_AHEAD_M and lat < LATERAL_M and before_goal:
                return pid
        return None

    def _give_up(self, t: float) -> None:
        """Held too long by someone who stays in the way: stop. A guide stepping aside to show a seat shows it from
        here; any other walk ends ``blocked_by_person`` so the brain can decide what to do."""
        w = self.world
        self._held_total, self._held_s = 0.0, 0.0
        w.robot.cancel_goal()
        w.robot.speed_scale = 1.0
        g = self._guide
        if g is not None and g.get("seat_id") and not g.get("shown"):
            c = w.site.seats[g["seat_id"]]
            w.robot.turn_to(math.atan2(c.y - w.robot.y, c.x - w.robot.x))
            self._show_seat(g, t)
            g["phase"], g["t_arrive"] = "waiting", t
            return
        gid = self.goal_id
        if not str(gid).startswith(SAFETY):
            self._goal(gid, False, "blocked_by_person", t)
        if g is not None:
            self._end_guide(t, "blocked_by_person")
        if self._tell is not None:
            q = w.people.get(self._tell["staff"])
            if q is not None and q.posture != "gone" and \
                    math.hypot(q.x - w.robot.x, q.y - w.robot.y) <= TELL_RANGE_M:
                self._speak_tell(t)  # held up near her: tell her from here
            else:
                self._result(self._tell["id"], "tell_staff", False, "blocked_by_person", t)
                self._tell = None

    def _safety(self, t: float, dt: float) -> None:
        """Every step: hold the robot for someone in its way, route round them if they stay, and step back from
        someone standing too close to it. Pure geometry on the world's people: the same in Quick 2D and Isaac."""
        w = self.world
        r = w.robot
        tl = self._tell
        if tl is not None and self.mobile and r.has_goal and r._had_goal and self.goal_id == tl["id"]:
            q = w.people.get(tl["staff"])
            if q is not None and q.posture != "gone" and self._in_earshot(q, TELL_NEAR_M) and self._clear_here():
                # close to the staff member it walks to: stop and tell her now (no walking into her), but only where
                # its body is clear of the furniture (stopped in the clearance band it could not walk off again)
                r.cancel_goal()
                r.speed_scale, self._holding, self._held_s, self._held_total = 1.0, False, 0.0, 0.0
                self._backing = False  # a step back under way ends here too
                self._goal(tl["id"], True, "arrived", t)
                self._speak_tell(t)
                return
        held_by_guide = self._guide is not None and self._guide["phase"] == "walking" and self._guide["held"]
        moving = r.has_goal and r._had_goal and self.mobile
        hold = moving and self._blocker() is not None
        if hold:
            r.speed_scale, self._holding = 0.0, True
        elif getattr(self, "_holding", False):  # release only a hold of its own (a guide holds the robot too)
            self._holding = False
            r.speed_scale = 0.0 if held_by_guide else 1.0
        if hold:
            self._held_s += dt
            self._held_total = getattr(self, "_held_total", 0.0) + dt
            if self._held_total >= GIVE_UP_S:
                self._give_up(t)
                return
            if self._held_s >= REPLAN_S and t - self._replanned_at >= REPLAN_S and self._goal_xy is not None:
                x, y, yaw, exempt = self._goal_xy
                discs = [(q.x, q.y, AVOID_R_M) for _, q in self._others() if math.hypot(q.x - r.x, q.y - r.y) < 4.0]
                self._replanned_at, self._held_s = t, 0.0
                r.set_goal(x, y, yaw, discs)  # the same goal (and goal id), a new way round
        else:
            self._held_s = 0.0
            if not moving:
                self._held_total = 0.0
        if not self.mobile or getattr(self, "_backing", False) or (moving and not hold):
            self._close_s = 0.0
            return
        near = [(math.hypot(q.x - r.x, q.y - r.y), q) for pid, q in self._others()]
        near = [(d, q) for d, q in near if d < BACK_OFF_M]
        self._close_s = self._close_s + dt if near else 0.0
        if not near or self._close_s < BACK_OFF_S or t - self._backed_at < BACK_OFF_COOLDOWN_S:
            return
        d, q = min(near, key=lambda e: e[0])
        ux, uy = ((r.x - q.x) / d, (r.y - q.y) / d) if d > 1e-6 else (math.cos(r.yaw + math.pi), math.sin(r.yaw + math.pi))
        checks = [c for c in (getattr(r.router, "is_clear", None), getattr(r.router, "is_free", None)) if c] or [None]
        for free in checks:  # clear of the furniture's hard band if it can, else of the soft one (by a desk)
            for k in (1.0, 0.7, 0.45):
                for turn in (0.0, 0.6, -0.6, 1.2, -1.2):
                    a = math.atan2(uy, ux) + turn
                    x, y = r.x + math.cos(a) * BACK_OFF_STEP_M * k, r.y + math.sin(a) * BACK_OFF_STEP_M * k
                    if math.hypot(x - q.x, y - q.y) < d + 0.25 or (free is not None and not free(x, y)) \
                            or w.site.grid.blocked(x, y):
                        continue  # it must really get further from them
                    self._n_safety += 1
                    self._backed_at, self._close_s = t, 0.0
                    face = math.atan2(q.y - y, q.x - x)
                    if moving:  # held on its way somewhere: step back, then carry on with the same goal (and id)
                        self._backing = True
                        self._holding = False
                        r.speed_scale = 1.0
                        r.set_goal(x, y, face, ())
                    else:
                        self._set_goal(f"{SAFETY}-{self._n_safety}", x, y, face, t, exempt=None)
                    return

    # ---- goals ---------------------------------------------------------------------------
    def _router_version(self):
        """The robot's router's version (it goes up when a route is blocked); a wrapping router reports its base's."""
        rr = getattr(self.world.robot, "router", None)
        v = getattr(rr, "version", None)
        return v if v is not None else getattr(getattr(rr, "base", None), "version", 0)

    def _goal_pose_now(self, x, y, yaw, t) -> None:
        self._goal_n += 1
        self._end_follow(t, "superseded")
        gid = f"goal_pose-{self._goal_n}"
        if not self._set_goal(gid, x, y, yaw, t):
            self._goal(gid, False, "no_route", t)
        else:
            self._end_guide(t, "superseded")

    def _avoid(self, exempt: Optional[str]) -> list:
        """People standing in the way of a new route (the person being guided is behind the robot: left out)."""
        return [(q.x, q.y, PERSON_AVOID_M) for pid, q in self.world.people.items()
                if pid != exempt and q.posture not in ("gone", "lying") and q.kind != "static"]

    def _set_goal(self, gid, x, y, yaw, t, exempt: Optional[str] = None) -> bool:
        """Start a goal; one still in flight is reported as superseded (and a tell_staff walking to it with it)."""
        w = self.world
        in_flight = w.robot.has_goal and w.robot._had_goal
        if not w.robot.set_goal(x, y, yaw, self._avoid(exempt)):
            return False
        self._goal_xy, self._held_s = (x, y, yaw, exempt), 0.0
        if in_flight:
            self._goal(self.goal_id, False, "superseded", t)
        self.goal_id = gid  # before the follow ends: ending it must not cancel this new walk
        if self._following is not None and not str(gid).startswith((SAFETY, FOLLOW_GOAL + "-")):
            self._end_follow(t, "superseded")  # another walk: the follow is over
        if self._tell is not None and self._tell["id"] != gid:
            self._result(self._tell["id"], "tell_staff", False, "superseded", t)
            self._tell = None
        return True

    def _cmd_vel_now(self, cmd, t) -> None:
        if self.world.robot.has_goal and self.world.robot._had_goal:
            self._goal(self.goal_id, False, "cancelled", t)
        if self._tell is not None:
            self._result(self._tell["id"], "tell_staff", False, "cancelled", t)
            self._tell = None
        self._end_guide(t, "cancelled")
        self._end_follow(t, "cancelled")
        self.world.robot.set_cmd_vel(*cmd)

    def _goal(self, gid, ok, reason, t) -> None:
        if str(gid).startswith(FOLLOW_GOAL + "-"):
            return  # one of its own walks while following: the follow action reports, not each walk
        self._out.results.append({"kind": "goal", "id": gid, "action": "goal", "ok": ok,
                                  "reason": reason, "t": t})

    # ---- time ----------------------------------------------------------------------------
    def _near_cap(self, dt: float = 0.0) -> Optional[float]:
        """0.3 m/s while someone is within 1.5 m (the person it guides only when not behind it), else no cap. With dt (the move about
        to be made), someone just outside 1.5 m caps it too, so that it reaches 1.5 m from them no faster than it
        could stop to 0.3 m/s there: it never comes within 1.5 m of anyone faster than 0.3 m/s."""
        w = self.world
        r = w.robot
        exempt = self._guide["pid"] if self._guide is not None else None
        nearest = math.inf
        for pid, q in w.people.items():
            if q.posture == "gone" or q.kind == "static":
                continue
            if pid == exempt and (q.x - r.x) * math.cos(r.yaw) + (q.y - r.y) * math.sin(r.yaw) <= 0.0:
                continue  # the person it guides, following behind it: the robot walks away from them
            nearest = min(nearest, math.hypot(q.x - r.x, q.y - r.y))
        if nearest <= NEAR_PERSON_M:
            return NEAR_PERSON_SPEED
        if dt > 0 and nearest - NEAR_PERSON_M < w.robot.max_speed * dt:
            return max(NEAR_PERSON_SPEED, (nearest - NEAR_PERSON_M - CAP_MARGIN_M) / dt)
        return None

    def step(self, t: float, dt: float) -> TaskOutputs:
        w = self.world
        self.apply_pending(t)  # commands the world did not step for (a held clock)
        for aid, fid in list(self._spray.items()):
            f = w.objects.fires.get(fid)
            if f is None or math.hypot(f.x - w.robot.x, f.y - w.robot.y) > EXTINGUISH_RANGE_M:
                del self._spray[aid]
                self._result(aid, "extinguish", False, "out_of_range", t)
                continue
            f.reduce(ROBOT_EXTINGUISH_PER_S * dt)
            if f.out:
                del self._spray[aid]
                self._result(aid, "extinguish", True, "ok", t)
        ev = w.robot.take_goal_event()
        if ev is not None and getattr(self, "_backing", False):  # the step back is done: back on its way
            self._backing = False
            if self._goal_xy is not None:
                x, y, yaw, exempt = self._goal_xy
                w.robot.set_goal(x, y, yaw, self._avoid(exempt))
            ev = None
        if ev is not None:
            gid = self.goal_id
            if not str(gid).startswith(SAFETY):
                self._goal(gid, ev == "arrived", ev, t)
            fid = self._to_fire.pop(gid, None)
            if fid is not None:
                f = w.objects.fires.get(fid)
                if ev != "arrived":
                    self._result(gid, "extinguish", False, ev, t)
                elif f is None or f.out:
                    self._result(gid, "extinguish", False, "no_such_fire", t)
                else:
                    self._spray[gid] = fid  # in range now: the spraying above reports how it ends
            g = self._guide
            if g is not None and g["id"] == gid and g["phase"] == "walking":
                if ev == "arrived" and g["seat"] and g["seat_id"] is None and self._to_seat(g, t):
                    pass  # stepping aside of the seat it picked
                elif ev == "arrived" and g.get("seat_id") and not g.get("shown"):
                    self._show_seat(g, t)  # aside now: show the seat
                    g["phase"], g["t_arrive"] = "waiting", t
                    w.robot.speed_scale = 1.0
                elif ev == "arrived":
                    g["phase"], g["t_arrive"] = "waiting", t
                    w.robot.speed_scale = 1.0
                else:
                    self._end_guide(t, ev)
            if self._tell is not None and self._tell["id"] == gid:
                if ev == "arrived":
                    self._speak_tell(t)
                else:
                    self._result(gid, "tell_staff", False, ev, t)  # a route existed: "stuck" says what happened
                    self._tell = None
        hz = (getattr(w, "_hazards", ()), self._router_version())
        if hz != self._hazards_seen:  # a fire started or went out, or a route was blocked: the walk under way goes round
            self._hazards_seen = hz
            if w.robot.has_goal and w.robot._had_goal and self._goal_xy is not None:
                x, y, yaw, exempt = self._goal_xy
                if not w.robot.set_goal(x, y, yaw, self._avoid(exempt)):
                    w.robot.cancel_goal()
                    self._goal(self.goal_id, False, "no_route", t)
                    if self._tell is not None and self._tell["id"] == self.goal_id:
                        self._result(self.goal_id, "tell_staff", False, "no_route", t)
                        self._tell = None
                    if self._guide is not None and self._guide["id"] == self.goal_id:
                        self._end_guide(t, "no_route")
        self._start_return(t)
        if getattr(w, "alarm_active", False) and str(self.goal_id).endswith(RETURN_SUFFIX) and \
                w.robot.has_goal and w.robot._had_goal:
            w.robot.cancel_goal()  # the alarm sounded on its way back to post: it stays put (its post may be by the fire)
            self._goal(self.goal_id, False, "alarm", t)
        for aid in list(self._to_fire):  # its walk was replaced, cancelled or given up: no spraying
            if self.goal_id != aid or not (w.robot.has_goal and w.robot._had_goal):
                del self._to_fire[aid]
                self._result(aid, "extinguish", False, "cancelled", t)
        if self._guide is not None:
            self._guide_step(t, dt)
        if self._following is not None:
            if ev is not None and ev != "arrived" and str(self.goal_id).startswith(FOLLOW_GOAL + "-"):
                self._following["retry_at"] = t + FOLLOW_RETRY_S  # its walk failed: try again in a moment
            self._follow_step(t)
        if self._notes:
            self._read_notes(t)
        self._safety(t, dt)
        w.robot.speed_cap = self._near_cap() if w.robot.has_goal else None
        for r in w.robot_results:
            self._out.results.append({**r, "t": t})
        w.robot_results.clear()
        for n in self._out.nudges:  # what the robot's actions told people goes in nudges.jsonl, like every nudge
            w._log.append(("nudges", n.model_dump(mode="json", by_alias=True)))
        out, self._out = self._out, TaskOutputs()
        return out
