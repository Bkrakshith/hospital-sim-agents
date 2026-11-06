"""Robot task actions for the stub world (the test double for the Isaac host's task layer).

An action is a JSON dict {"id", "action", ...} from <ns>/task/action:
  hand_over {item, to: track, place?: [x,y,z]}   an open offer; the result comes when it resolves
  take {item}                                     an unheld item within 0.8 m
  extinguish {fire}                               sprays until the fire is out (or the robot leaves)
  press_call {}                                   chimes at the nearest nurse base
  guide_start {track, place} / guide_stop {}      walks to the place at the person's pace; the person is offered a walk
  tell_staff {about: track, message?}             walks to the reception counter (or the ward nurse base) and tells
                                                  the staff there that this person needs help
Each action ends in one result {"id", "action", "ok", "reason", "t"}; nothing here uses person ids
on the robot's side of the wire: people are addressed by track.

Guiding: the robot holds while the person it guides is more than 2.5 m behind (and walks on once they are within
2 m); that person never slows it, other people within 1.5 m slow it to 0.3 m/s. At the place it waits up to 20 s
for them to arrive, says goodbye and walks back to its post (goal id "<id>-return"). It gives up if they never come
within 2.5 m for 30 s.

Telling the staff: the staff member hears it by the audio model (and must be within 5 m); when they do, they get a
structured message (a nudge) with the person's id, their name if they said it, and where they are. The robot is not
told where the staff member is.

The sim-side run log (robot_tasks.jsonl, never sent to the robot) gets a "guide" row at the start and end of each
guide, with the person's id, and a "tell" row for each tell_staff: the benchmark reads arrival from these.
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
ACTIONS = ("hand_over", "take", "extinguish", "press_call", "guide_start", "guide_stop", "tell_staff")
GUIDE_CARD = "robot_guide"
TELL_CARD = "robot_tell_staff"

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
RECEPTION_ROLES = ("receptionist",)
WARD_STAFF_ROLES = ("ward_nurse", "nurse", "staff")


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
                       "held": False, "held_s": 0.0, "t0": t}
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
            self._goal(gid, ev == "arrived", ev, t)
            g = self._guide
            if g is not None and g["id"] == gid and g["phase"] == "walking":
                if ev == "arrived":
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
        w.robot.speed_cap = self._near_cap() if w.robot.has_goal else None
        for r in w.robot_results:
            self._out.results.append({**r, "t": t})
        w.robot_results.clear()
        out, self._out = self._out, TaskOutputs()
        return out
