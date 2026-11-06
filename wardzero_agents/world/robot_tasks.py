"""Robot task actions for the stub world (the test double for the Isaac host's task layer).

An action is a JSON dict {"id", "action", ...} from <ns>/task/action:
  hand_over {item, to: track, place?: [x,y,z]}   an open offer; the result comes when it resolves
  take {item}                                     an unheld item within 0.8 m
  extinguish {fire}                               sprays until the fire is out (or the robot leaves)
  press_call {}                                   chimes at the nearest nurse base
  guide_start {track, place, seat?} / guide_stop {}
                                                  walks to the place at the person's pace; the person is offered a walk;
                                                  with seat, it ends beside a free seat there and shows it to them
  approach {track}                                walks up to the person (APPROACH_M from them) and faces them
  tell_staff {about: track, message?}             walks to the reception counter (or the ward nurse base) and tells
                                                  the staff there that this person needs help
  note_desk {about: track, message}               a note on the reception desk computer: the robot does not move or
                                                  speak; the receptionist reads it when she is at the desk
Each action ends in one result {"id", "action", "ok", "reason", "t"}; nothing here uses person ids
on the robot's side of the wire: people are addressed by track.

Guiding: the robot holds while the person it guides is more than 2.5 m behind (and walks on once they are within
2 m); that person never slows it, other people within 1.5 m slow it to 0.3 m/s. At the place it waits up to 20 s
for them to arrive, says goodbye and walks back to its post (goal id "<id>-return"). It gives up if they never come
within 2.5 m for 30 s.

A guide to a seat: when the robot reaches the place it picks the free seat nearest to it, among those its camera could
see by turning its head (within its range, in line of sight; never a staff chair). It walks to a spot beside that seat,
says SEAT_SAY and the person is told the seat (a nudge naming sit_down <seat>); they have arrived when they are near
the seat. With no free seat in sight it says NO_SEAT_SAY and the guide ends as a plain guide to the place.

Telling the staff: the staff member hears it by the audio model (and must be within 5 m); when they do, they get a
structured message (a nudge) with the person's id, their name if they said it, and where they are. The robot is not
told where the staff member is.

The desk note: the result ("noted") comes at once, since the computer takes the note. The receptionist gets it as a
nudge (what is on her screen) when she is within 1.5 m of her chair, at once or when she gets back, and only once.

The sim-side run log (robot_tasks.jsonl, never sent to the robot) gets a "guide" row at the start and end of each
guide, with the person's id, and a "tell" row for each tell_staff and a "note" row for each desk note when it is read (and when it is left unread at
the end of the run, nothing: unread notes are not logged).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from wardzero_agents.contracts import AudioEvent, Nudge
from wardzero_agents.world import audio
from wardzero_agents.world.objects import EXTINGUISH_RANGE_M, ROBOT_EXTINGUISH_PER_S
from wardzero_agents.world.oracle import Pose

TAKE_REACH_M = 0.8
CHIME_DB = 55.0
ACTIONS = ("hand_over", "take", "extinguish", "press_call", "guide_start", "guide_stop", "tell_staff", "note_desk",
           "approach")
GUIDE_CARD = "robot_guide"
TELL_CARD = "robot_tell_staff"
NOTE_CARD = "robot_desk_note"
SEAT_CARD = "robot_seat"
SEAT_SAY = "This seat is free. Please take a seat."
NO_SEAT_SAY = "I'm sorry, I can't see a free seat here."
SEAT_BESIDE_M = 0.9  # the robot stands this far from the seat it shows
APPROACH_M = 1.2  # how close the robot comes to a person it walks up to
SEAT_ASIDE_M = 1.0  # showing a seat, the robot stands this far to the side of the seat's front, out of the way in
# keeping clear of people while it walks (any goal, its own or a task's):
PERSON_KEEP_M = 0.7  # centre to centre: body radius 0.3 + a person's 0.25 + 0.15 m
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
DESK_M = 1.5  # the receptionist reads the desk computer when she is this close to her chair

GUIDE_HOLD_M = 2.5  # the robot waits while the guided person is further behind than this ...
GUIDE_RESUME_M = 2.0  # ... and walks on once they are this close again
GUIDE_GIVE_UP_S = 30.0  # held this long without them coming within GUIDE_HOLD_M: the guide is given up
GUIDE_ARRIVE_WAIT_S = 20.0  # at the place, wait this long for the person to arrive
NEAR_PLACE_M = 2.0  # a person this close to the place (or in the named room) has arrived
NEAR_PERSON_M = 1.5  # another person this close slows the robot ...
NEAR_PERSON_SPEED = 0.3  # ... to this (m/s)
PERSON_AVOID_M = 0.7  # the robot's routes keep this far from people standing in the way, when they can
TELL_RANGE_M = 5.0  # a staff member further than this from the robot is "not at the desk"
TELL_DEFAULT = "Excuse me, someone here needs help from a nurse."
GOODBYE = "Here we are. Goodbye, and take care."
COUNTER = "reception_counter"  # where the robot stands to talk to the receptionist
STAFF_SPOT = "reception_staff"  # where the receptionist sits (the robot faces it)
STAFF_SEAT = "R1"  # the receptionist's chair, where the desk computer is
RECEPTION_ROLES = ("receptionist",)
WARD_STAFF_ROLES = ("ward_nurse", "nurse", "staff")


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
        self._guide: Optional[dict] = None
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
        # speaks from where the robot stands, and nothing turns it or sends it back to its post
        self.mobile = True
        rc = getattr(getattr(getattr(world, "loaded", None), "scenario", None), "robot", None)
        self.post = (rc.x, rc.y, rc.yaw) if rc is not None else tuple(world.robot.pose)

    def _result(self, aid, action, ok, reason, t) -> None:
        self._out.results.append({"id": aid, "action": action, "ok": ok, "reason": reason, "t": t})

    def _log(self, rec: dict) -> None:
        """A sim-side row for robot_tasks.jsonl (the benchmark's view; never published to the robot)."""
        self.world._log.append(("robot_tasks", rec))

    def handle(self, msg: dict, t: float) -> None:
        aid = str(msg.get("id") or f"a{len(self._out.results)}")
        action = msg.get("action")
        if action not in ACTIONS:
            self._result(aid, str(action), False, "unknown_action", t)
            return
        try:
            getattr(self, f"_{action}")(aid, msg, t)
        except (KeyError, TypeError, ValueError, IndexError):
            self._result(aid, action, False, "bad_request", t)

    # ---- actions -------------------------------------------------------------------------
    def _person(self, track) -> Optional[str]:
        pid = self.world.tracks.person(str(track))
        p = self.world.people.get(pid or "")
        return pid if p is not None and p.posture != "gone" else None

    def _hand_over(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["to"])
        if pid is None:
            return self._result(aid, "hand_over", False, "unknown_track", t)
        item = w.objects.items.get(str(msg["item"]))
        if item is None or item.holder != "g1":
            return self._result(aid, "hand_over", False, "not_holding", t)
        q, r = w.people[pid], w.robot
        if msg.get("place") is not None:
            place = tuple(float(v) for v in msg["place"])
            if len(place) != 3:
                raise ValueError("place")
        else:  # half a metre from the receiver, towards the robot
            d = max(math.hypot(r.x - q.x, r.y - q.y), 1e-6)
            place = (q.x + (r.x - q.x) / d * 0.5, q.y + (r.y - q.y) / d * 0.5, 0.85 if q.bed else 1.1)
        why = w._check_hand_over("g1", pid, item.id, place)
        if why != "ok":
            return self._result(aid, "hand_over", False, why, t)
        old = w._offers.get(pid)
        if old is not None and old["giver"] == "g1":  # one result per action: close the old offer
            self._result(old["intent"], "hand_over", False, "superseded", t)
        w._offers[pid] = {"giver": "g1", "item": item.id, "place": place, "t": t, "intent": aid}

    def _take(self, aid, msg, t) -> None:
        w = self.world
        item = w.objects.items.get(str(msg["item"]))
        if item is None or item.holder is not None:
            return self._result(aid, "take", False, "unknown_item", t)
        if math.hypot(item.x - w.robot.x, item.y - w.robot.y) > TAKE_REACH_M:
            return self._result(aid, "take", False, "out_of_reach", t)
        item.holder, item.x, item.y = "g1", w.robot.x, w.robot.y
        w._log_objects(t)
        self._result(aid, "take", True, "ok", t)

    def _extinguish(self, aid, msg, t) -> None:
        w = self.world
        f = w.objects.fires.get(str(msg["fire"]))
        if f is None or f.out:
            return self._result(aid, "extinguish", False, "no_such_fire", t)
        if math.hypot(f.x - w.robot.x, f.y - w.robot.y) > EXTINGUISH_RANGE_M:
            return self._result(aid, "extinguish", False, "out_of_range", t)
        self._spray[aid] = f.id

    def _press_call(self, aid, msg, t) -> None:
        w = self.world
        bx, by = w._nurse_base(w.robot.x, w.robot.y)
        self._out.audio.append(w._event(t, "person", "bell_robot", bx, by, 2.0, CHIME_DB))
        self._result(aid, "press_call", True, "ok", t)

    def _approach(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["track"])
        if pid is None:
            return self._result(aid, "approach", False, "unknown_track", t)
        if not self.mobile:
            return self._result(aid, "approach", False, "cannot_walk", t)
        p, r = w.people[pid], w.robot
        d = math.hypot(r.x - p.x, r.y - p.y)
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
        if not self._set_goal(aid, xy[0], xy[1], None, t, exempt=pid):
            return self._result(aid, "guide_start", False, "no_route", t)
        self._end_guide(t, "superseded")  # an earlier guide (perhaps waiting at its place) is over
        lm = w.site.landmarks.get(place)
        label = (lm.label.split(",")[0] if lm is not None and lm.label else place.replace("_", " "))
        room = place if place in w.site.rooms else None
        r = w.robot
        pts = [(r.x, r.y)] + list(r._path)
        route_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        self._guide = {"id": aid, "pid": pid, "place": place, "goal": xy, "room": room, "phase": "walking",
                       "held": False, "held_s": 0.0, "t0": t, "seat": bool(msg.get("seat")), "seat_id": None}
        self._log({"t": t, "kind": "guide", "event": "start", "id": aid, "person": pid, "place": place,
                   "x": round(xy[0], 3), "y": round(xy[1], 3), "room": room, "route_m": round(route_m, 2)})
        self._out.nudges.append(Nudge(person=pid, t=t, card=GUIDE_CARD, message=(
            f"The robot is offering to walk you to {label}. If you want to go there, follow it (follow g1): "
            f"walk a little behind it and it will wait for you.")))
        self._result(aid, "guide_start", True, "started", t)

    def _end_guide(self, t, reason: str) -> None:
        g, self._guide = self._guide, None
        self.world.robot.speed_scale = 1.0
        if g is not None:
            p = self.world.people.get(g["pid"])
            if p is not None and p.plan and p.plan[0].kind == "walk" and p.plan[0].trail:
                self.world.trail_released.add(g["pid"])  # following ends with the guide
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
            self._out.audio.append(w.robot_say(GOODBYE, t).model_copy(update={"addressee": g["pid"]}))
            self._end_guide(t, "arrived" if arrived else "timeout")
            if self.mobile:
                x, y, yaw = self.post
                self._set_goal(f"{g['id']}-return", x, y, yaw, t)

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
        # nowhere beside it: back off along the line from the seat through the robot, past the seat's front
        d = max(math.hypot(w.robot.x - c.x, w.robot.y - c.y), 1e-6)
        return c.x + (w.robot.x - c.x) / d * (SEAT_BESIDE_M + 1.0), c.y + (w.robot.y - c.y) / d * (SEAT_BESIDE_M + 1.0)

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
        if not self._set_goal(g["id"], x, y, math.atan2(c.y - y, c.x - x), t, exempt=g["pid"]):
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
        self._result(aid, "guide_stop", True, "ok", t)

    # ---- telling the staff ---------------------------------------------------------------
    def _staff_to_tell(self) -> tuple[Optional[str], Optional[tuple]]:
        """(staff person, (x, y, yaw) where the robot stands to tell them): the receptionist at the counter, else a
        ward staff member at the nearest nurse base. (None, None) when the scenario has neither."""
        w = self.world
        present = [(pid, q) for pid, q in w.people.items() if q.posture != "gone" and q.kind != "static"]
        desk = [pid for pid, q in present if q.inst.role in RECEPTION_ROLES]
        c, s = w.site.landmarks.get(COUNTER), w.site.landmarks.get(STAFF_SPOT)
        if desk and c is not None:
            yaw = math.atan2(s.y - c.y, s.x - c.x) if s is not None else None
            return desk[0], (c.x, c.y, yaw)
        ward = [(pid, q) for pid, q in present if q.inst.role in WARD_STAFF_ROLES]
        if ward:
            bx, by = w._nurse_base(w.robot.x, w.robot.y)
            pid = min(ward, key=lambda e: math.hypot(e[1].x - bx, e[1].y - by))[0]
            return pid, (bx, by, None)
        return None, None

    def _tell_staff(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["about"])
        if pid is None:
            return self._result(aid, "tell_staff", False, "unknown_track", t)
        message = msg.get("message")
        if message is not None and not isinstance(message, str):
            raise TypeError("message")
        message = (message or "").strip()[:400] or TELL_DEFAULT
        staff, spot = self._staff_to_tell()
        if staff is None:
            self._log({"t": t, "kind": "tell", "id": aid, "about": pid, "staff": None, "heard": False,
                       "reason": "no_staff_to_tell"})
            return self._result(aid, "tell_staff", False, "no_staff_to_tell", t)
        if self._tell is not None:
            self._result(self._tell["id"], "tell_staff", False, "superseded", t)
        self._tell = {"id": aid, "about": pid, "staff": staff, "message": message, "yaw0": w.robot.yaw}
        x, y, yaw = spot
        if not self.mobile or math.hypot(w.robot.x - x, w.robot.y - y) <= 0.3:
            return self._speak_tell(t)
        if not self._set_goal(aid, x, y, yaw, t):
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
            h = audio.route(ev, Pose(q.x, q.y, q.yaw), tl["staff"], q.inst.hearing_loss_db, w.rng)
            heard = h is not None and not h.partial
        if heard and p is not None:
            self._out.nudges.append(Nudge(person=tl["staff"], t=t, card=TELL_CARD,
                                          message=self.tell_message(tl["message"], tl["about"], q)))
        reason = "told" if heard else "staff_not_at_desk"
        self._log({"t": t, "kind": "tell", "id": tl["id"], "about": tl["about"], "staff": tl["staff"],
                   "heard": heard, "reason": reason})
        self._result(tl["id"], "tell_staff", heard, reason, t)
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
        return (f"A note from the reception robot appears on your computer screen: \"{message}\" It is about person id "
                f"{pid} ({name}), now in {room}. The robot is dealing with them. You do not need to get up unless the "
                f"note says they need a member of staff; carry on with your work.")

    def tell_message(self, message: str, pid: str, staff) -> str:
        """What the staff member is told, besides hearing it: who needs help, where, and what to do."""
        w = self.world
        p = w.people[pid]
        name = p.inst.display_name if pid in w.named else "they have not said their name"
        rid = w.site.room_at(p.x, p.y)
        room = w.site.rooms[rid].label.split(",")[0] if rid else "the building"
        back = f"go back to your chair (seat {staff.inst.seat})" if staff.inst.seat else "go back to your post"
        return (f"The reception robot tells you: \"{message}\" The person who needs help is person id {pid} "
                f"({name}), in {room} at x {p.x:.1f}, y {p.y:.1f}. If you can help, go to them (approach {pid}), "
                f"help them, then {back}.")

    # ---- keeping clear of people -------------------------------------------------------------
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
            if self._guide is not None and pid == self._guide["pid"]:  # the person it leads: only when really close
                if math.hypot(dx, dy) < PERSON_KEEP_M:
                    return pid
                continue
            if math.hypot(dx, dy) < PERSON_KEEP_M or (ahead < STOP_AHEAD_M and lat < LATERAL_M and ahead < left + 0.3):
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
            self._result(self._tell["id"], "tell_staff", False, "blocked_by_person", t)
            self._tell = None

    def _safety(self, t: float, dt: float) -> None:
        """Every step: hold the robot for someone in its way, route round them if they stay, and step back from
        someone standing too close to it. Pure geometry on the world's people: the same in Quick 2D and Isaac."""
        w = self.world
        r = w.robot
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
        if moving or not self.mobile or self._guide is not None or self._tell is not None:
            self._close_s = 0.0
            return
        near = [(math.hypot(q.x - r.x, q.y - r.y), q) for _, q in self._others()]
        near = [(d, q) for d, q in near if d < BACK_OFF_M]
        self._close_s = self._close_s + dt if near else 0.0
        if not near or self._close_s < BACK_OFF_S or t - self._backed_at < BACK_OFF_COOLDOWN_S:
            return
        d, q = min(near, key=lambda e: e[0])
        ux, uy = ((r.x - q.x) / d, (r.y - q.y) / d) if d > 1e-6 else (math.cos(r.yaw + math.pi), math.sin(r.yaw + math.pi))
        free = getattr(r.router, "is_clear", None) or getattr(r.router, "is_free", None)
        for k in (1.0, 0.7, 0.45):
            for turn in (0.0, 0.6, -0.6, 1.2, -1.2):
                a = math.atan2(uy, ux) + turn
                x, y = r.x + math.cos(a) * BACK_OFF_STEP_M * k, r.y + math.sin(a) * BACK_OFF_STEP_M * k
                if (free is None or free(x, y)) and not w.site.grid.blocked(x, y):
                    self._n_safety += 1
                    self._backed_at, self._close_s = t, 0.0
                    face = math.atan2(q.y - y, q.x - x)
                    self._set_goal(f"{SAFETY}-{self._n_safety}", x, y, face, t, exempt=None)
                    return

    # ---- goals ---------------------------------------------------------------------------
    def goal_pose(self, x, y, yaw, t) -> None:
        """A goal_pose command; no route is reported as a goal result."""
        self._goal_n += 1
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
        if self._tell is not None and self._tell["id"] != gid:
            self._result(self._tell["id"], "tell_staff", False, "superseded", t)
            self._tell = None
        self.goal_id = gid
        return True

    def cmd_vel(self, cmd, t) -> None:
        if self.world.robot.has_goal and self.world.robot._had_goal:
            self._goal(self.goal_id, False, "cancelled", t)
        if self._tell is not None:
            self._result(self._tell["id"], "tell_staff", False, "cancelled", t)
            self._tell = None
        self._end_guide(t, "cancelled")
        self.world.robot.set_cmd_vel(*cmd)

    def _goal(self, gid, ok, reason, t) -> None:
        self._out.results.append({"kind": "goal", "id": gid, "action": "goal", "ok": ok,
                                  "reason": reason, "t": t})

    # ---- time ----------------------------------------------------------------------------
    def _near_cap(self) -> Optional[float]:
        """0.3 m/s while someone other than the guided person is within 1.5 m, else no cap."""
        w = self.world
        exempt = self._guide["pid"] if self._guide is not None else None
        for pid, q in w.people.items():
            if pid != exempt and q.posture != "gone" and q.kind != "static" \
                    and math.hypot(q.x - w.robot.x, q.y - w.robot.y) <= NEAR_PERSON_M:
                return NEAR_PERSON_SPEED
        return None

    def step(self, t: float, dt: float) -> TaskOutputs:
        w = self.world
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
        if ev is not None:
            gid = self.goal_id
            if not str(gid).startswith(SAFETY):
                self._goal(gid, ev == "arrived", ev, t)
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
                    self._result(gid, "tell_staff", False, "no_route" if ev == "stuck" else ev, t)
                    self._tell = None
        if self._guide is not None:
            self._guide_step(t, dt)
        if self._notes:
            self._read_notes(t)
        self._safety(t, dt)
        w.robot.speed_cap = self._near_cap() if w.robot.has_goal else None
        for r in w.robot_results:
            self._out.results.append({**r, "t": t})
        w.robot_results.clear()
        out, self._out = self._out, TaskOutputs()
        return out
