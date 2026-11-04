"""Robot task actions for the stub world (the test double for the Isaac host's task layer).

An action is a JSON dict {"id", "action", ...} from <ns>/task/action:
  hand_over {item, to: track, place?: [x,y,z]}   an open offer; the result comes when it resolves
  take {item}                                     an unheld item within 0.8 m
  extinguish {fire}                               sprays until the fire is out (or the robot leaves)
  press_call {}                                   chimes at the nearest nurse base
  guide_start {track, place} / guide_stop {}      heads to the place; the person is offered a walk
Each action ends in one result {"id", "action", "ok", "reason", "t"}; nothing here uses person ids
on the robot's side of the wire: people are addressed by track.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from wardzero_agents.contracts import AudioEvent, Nudge
from wardzero_agents.world.objects import EXTINGUISH_RANGE_M, ROBOT_EXTINGUISH_PER_S

TAKE_REACH_M = 0.8
CHIME_DB = 55.0
ACTIONS = ("hand_over", "take", "extinguish", "press_call", "guide_start", "guide_stop")
GUIDE_CARD = "robot_guide"


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
        self.goal_id = "goal_pose"
        self._out = TaskOutputs()

    def _result(self, aid, action, ok, reason, t) -> None:
        self._out.results.append({"id": aid, "action": action, "ok": ok, "reason": reason, "t": t})

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

    def _guide_start(self, aid, msg, t) -> None:
        w = self.world
        pid = self._person(msg["track"])
        if pid is None:
            return self._result(aid, "guide_start", False, "unknown_track", t)
        place = str(msg["place"])
        xy = w._resolve_place(place)
        if xy is None:
            return self._result(aid, "guide_start", False, "unknown_place", t)
        if not w.robot.set_goal(*xy):
            return self._result(aid, "guide_start", False, "no_route", t)
        self.goal_id = aid
        lm = w.site.landmarks.get(place)
        label = (lm.label if lm is not None and lm.label else place.replace("_", " "))
        self._guide = {"id": aid, "goal": xy}
        self._out.nudges.append(Nudge(person=pid, t=t, card=GUIDE_CARD,
                                      message=f"The robot is offering to walk you to {label}"))
        self._result(aid, "guide_start", True, "started", t)

    def goal_pose(self, x, y, yaw, t) -> None:
        """A goal_pose command; no route is reported as a goal result."""
        if self.world.robot.set_goal(x, y, yaw):
            self.goal_id = "goal_pose"
        else:
            self._goal("goal_pose", False, "no_route", t)

    def _goal(self, gid, ok, reason, t) -> None:
        self._out.results.append({"kind": "goal", "id": gid, "action": "goal", "ok": ok,
                                  "reason": reason, "t": t})

    def _guide_stop(self, aid, msg, t) -> None:
        self.world.robot.cancel_goal()
        self._guide = None
        self._result(aid, "guide_stop", True, "ok", t)

    # ---- time ----------------------------------------------------------------------------
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
            self._goal(self.goal_id, ev == "arrived", ev, t)
            self._guide = None
        for r in w.robot_results:
            self._out.results.append({**r, "t": t})
        w.robot_results.clear()
        out, self._out = self._out, TaskOutputs()
        return out
