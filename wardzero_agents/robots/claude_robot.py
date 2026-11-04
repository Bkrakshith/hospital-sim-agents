"""The built-in robot: a Claude mind in the G1's seat, with the hospital as its context.

Fairness: this process is the robot. It never reads the people's topics. It knows only what its body
gives it: its pose (odom), what the camera sees as anonymous tracks (camera/bbox_3d, whose box size
tells lying from seated from standing), objects in view (perception/objects), what the microphone
hears, the results of its own actions and the task brief. No person id, name, mood or persona text
reaches its prompt, and it speaks in place labels, never in internal ids.

    python -m wardzero_agents.robots.claude_robot --robot-config FILE|--scenario FILE --run-dir DIR [--domain N]

The node is named robot_claude (not wardzero_*), so the fairness audit treats it as the robot.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from ..llm.gateway import PERMANENT_ERRORS, LLMRequest, LLMResult
from .profile import G1_CLAUDE, G1_NAME, RobotProfile, get_profile

log = logging.getLogger("robot_claude")

NODE_NAME = "robot_claude"
BRIEF_WAIT_S = 5.0  # wait this long for the latched task brief, then start with an empty one
HEARTBEAT_S = 10.0
MIN_GAP_S = 2.0  # between calls, unless someone is speaking to the robot
NEAR_SPEECH_M = 3.0
EVENTS_KEPT = 20
HEARD_KEPT_S = 60.0
LLM_OFF_AFTER = 3  # permanent failures in a row (bad key, bad model) before the mind stops calling
ROBOT_BUDGET_SHARE = 0.25  # of the scenario's budget_usd: the robot has its own budget
MAX_SAY_CHARS = 400

Action = Literal["none", "go_to", "guide", "hand_over", "take", "extinguish", "press_call", "wait"]
ACTIONS = ("none", "go_to", "guide", "hand_over", "take", "extinguish", "press_call", "wait")
MOVING_ACTIONS = ("go_to", "guide")
NEEDS = {"hand_over": ("item", "target"), "take": ("item",), "extinguish": ("target",),
         "guide": ("target", "destination"), "go_to": ("destination",)}

SYSTEM_RULES = """You are the robot described under YOUR BODY, working in a UK hospital and doing the job in your task brief. People around you are real people with their own worries. You act through a few simple actions and by speaking.

How you work
- Each turn you are shown what you can perceive right now: your position, who your camera sees (as anonymous tracks such as t1, t2 with a distance, a direction and a posture), objects in view, what your microphone heard, the results of your recent actions and your own last events. You know nothing else about the people. Never guess a name, an age or how someone feels; you can only see what the camera shows and hear what is said.
- A box that is low and long means a person lying down. A box about 1.2 m tall means someone seated. About 1.7 m tall means standing. Someone lying on the floor, away from a bed, may need urgent help.
- Speak in short, plain, warm sentences, the way a good receptionist or ward helper would. Use the place names as given in the lists (for example "Main entrance"), never codes, identifiers or words joined by underscores.
- Be truthful. Do not claim to have done something you have not done, and do not promise what your body cannot do. Your abilities are listed in your briefing; stay inside them.
- When you are unsure, or someone seems unwell, in distress or in danger, fetch a human: press the call button for staff and tell the person you have done so.
- Safety comes first: keep clear of people, never block an escape route or a doorway, and do not touch anyone. Do not use the extinguisher unless a fire is in range and putting it out is part of your job.
- Do not repeat yourself. If you have just said something and nothing has changed, choose "none" or "wait" and say nothing.

How you answer
Reply with one JSON object and nothing else.
- say: what you say aloud now, or an empty string to stay silent.
- toward_track: the track you are speaking to or attending to, or an empty string.
- action: one of the allowed actions in the schema. "none" means do nothing new; "wait" means stand still and stop any movement.
- target: the track for hand_over, the fire for extinguish, or the track for guide; otherwise an empty string.
- item: the item id for hand_over or take; otherwise an empty string.
- destination: a place name from your lists for go_to or guide; otherwise an empty string.
- place: for hand_over only, optionally [x, y, z] in metres where you hold the item out (z inside your reach); otherwise an empty list.
- reason: one short sentence on why, for the record."""

NO_WALK_RULE = ("You are standing at a fixed post and cannot walk in this build; to help someone get "
                "somewhere, give clear directions or press the call button for staff.")
WALK_RULE = ("You can walk. To take someone somewhere, use guide with their track and the place name; to go "
             "somewhere yourself, use go_to.")


def system_prompt(profile: RobotProfile) -> str:
    return SYSTEM_RULES + ("\n\nYou are a Unitree G1 humanoid." if profile.name == G1_NAME else "") + "\n\nYour mobility\n" + (WALK_RULE if profile.can_move else NO_WALK_RULE)


def actions_for(profile: RobotProfile) -> tuple[str, ...]:
    return ACTIONS if profile.can_move else tuple(a for a in ACTIONS if a not in MOVING_ACTIONS)


def decision_schema(profile: RobotProfile) -> dict:
    s = {"type": "string"}
    return {
        "type": "object",
        "properties": {"say": s, "toward_track": s, "action": {"type": "string", "enum": list(actions_for(profile))},
                       "target": s, "item": s, "destination": s,
                       "place": {"type": "array", "items": {"type": "number"}}, "reason": s},
        "required": ["say", "toward_track", "action", "target", "item", "destination", "place", "reason"],
        "additionalProperties": False,
    }


@dataclass
class RobotDecision:
    say: Optional[str]
    toward_track: Optional[str]
    action: Action
    target: Optional[str]
    item: Optional[str]
    place: Optional[list[float]]
    reason: str
    destination: Optional[str] = None  # a place label for go_to and guide (an addition to the brief's fields)


@dataclass
class Outputs:
    say: Optional[str] = None
    action: Optional[dict] = None  # JSON for <ns>/task/action
    goal: Optional[tuple[float, float, float]] = None  # x, y, yaw for <ns>/goal_pose
    stop: bool = False  # zero cmd_vel


# ---- places in words ------------------------------------------------------------------------
def _clean_label(text: str) -> str:
    text = re.sub(r"\s*\(.*?\)\s*", " ", text or "").strip()
    return text.split(", ")[0].strip()


def place_label(site, pid: str) -> str:
    """The words for a place id: the room or landmark label, else a bed, else the id spaced out."""
    if pid in site.rooms:
        return _clean_label(site.rooms[pid].label) or pid.replace("_", " ")
    if pid in site.landmarks and site.landmarks[pid].label:
        return _clean_label(site.landmarks[pid].label)
    if pid in site.beds:
        return f"bed {pid}"
    return pid.replace("_", " ")


def _direction_words(rel_deg: float) -> str:
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


def _rel_deg(pose, x, y) -> float:
    d = math.degrees(math.atan2(y - pose[1], x - pose[0]) - pose[2])
    return (d + 180.0) % 360.0 - 180.0


_ROUTE_CACHE: dict = {}


def route_words(site, start: tuple[float, float, float], place_ids, router=None) -> str:
    """Words-only routes from the robot's post to each known place: how far, which way, which rooms."""
    key = (id(site), tuple(round(v, 2) for v in start), tuple(place_ids))
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]
    if router is None:
        from ..site.routing import Router
        router = Router(site.grid)
    here = site.room_at(start[0], start[1])
    lines = [f"Getting there from your post ({place_label(site, here) if here else 'where you stand'}):"]
    for pid in place_ids:
        xy = site.resolve(pid)
        if xy is None:
            continue
        label = place_label(site, pid)
        route = router.route((start[0], start[1]), xy)
        if not route:
            lines.append(f"- {label}: no walkable route found.")
            continue
        pts = [(start[0], start[1]), *route]
        length, rooms = 0.0, []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            length += seg
            for k in range(max(1, int(seg / 0.5))):
                f = k / max(1, int(seg / 0.5))
                rid = site.room_at(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
                if rid and (not rooms or rooms[-1] != rid):
                    rooms.append(rid)
        first = route[0]
        way = _direction_words(_rel_deg(start, first[0], first[1]))
        names = [place_label(site, r) for r in rooms]
        names = [n for i, n in enumerate(names) if n and (i == 0 or n != names[i - 1])]
        via = f", through {', then '.join(names[:5])}" if len(names) > 1 else ""
        if length < 1.5:
            lines.append(f"- {label}: right beside you.")
            continue
        lines.append(f"- {label}: about {length:.0f} m, setting off {way}{via}.")
    text = "\n".join(lines)
    _ROUTE_CACHE[key] = text
    return text


def posture_from_box(size: dict) -> str:
    z, x = float(size.get("z", 1.7)), float(size.get("x", 0.5))
    if z < 0.9 or (z < 1.0 and x > 1.0):
        return "lying down"
    if z < 1.45:
        return "seated"
    return "standing"


# ---- decisions ------------------------------------------------------------------------------
def _opt_str(data: dict, key: str) -> Optional[str]:
    v = data.get(key, "")
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{key} must be a string")
    return v.strip() or None


def parse_decision(data: Any, profile: RobotProfile, mind: "Optional[RobotMind]" = None) -> RobotDecision:
    """Validate a model reply against the decision schema. Raises ValueError when it does not fit."""
    if not isinstance(data, dict):
        raise ValueError("not an object")
    action = data.get("action")
    if action not in actions_for(profile):
        raise ValueError(f"action {action!r} is not allowed")
    reason = data.get("reason")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    place = data.get("place", [])
    if place is None:
        place = []
    if not isinstance(place, list) or len(place) not in (0, 3) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in place):
        raise ValueError("place must be empty or [x, y, z]")
    say = _opt_str(data, "say")
    d = RobotDecision(say=say[:MAX_SAY_CHARS] if say else None, toward_track=_opt_str(data, "toward_track"),
                      action=action, target=_opt_str(data, "target"), item=_opt_str(data, "item"),
                      place=[float(v) for v in place] or None, reason=reason.strip(),
                      destination=_opt_str(data, "destination"))
    for need in NEEDS.get(action, ()):
        if getattr(d, need) is None:
            raise ValueError(f"{action} needs {need}")
    if d.destination is not None and mind is not None and mind.place_id(d.destination) is None:
        raise ValueError("unknown destination")
    return d


_UNDERSCORED = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")


def speakable(text: str, site=None) -> str:
    """Replace any internal id in the words the robot speaks by its label (or spaces)."""
    if site is not None:
        def label(m):
            pid = m.group(0)
            for table in (site.rooms, site.landmarks):
                for key in table:
                    if key.lower() == pid.lower():
                        return place_label(site, key)
            return pid.replace("_", " ")
        return _UNDERSCORED.sub(label, text)
    return _UNDERSCORED.sub(lambda m: m.group(0).replace("_", " "), text)


def decision_outputs(d: RobotDecision, mind: "RobotMind", aid: str) -> Outputs:
    """What a decision puts on the wire: speech/say, task/action, goal_pose or a stop."""
    out = Outputs(say=speakable(d.say, mind.site) if d.say else None)
    a = d.action
    if a == "hand_over":
        out.action = {"id": aid, "action": "hand_over", "item": d.item, "to": d.target}
        if d.place:
            out.action["place"] = list(d.place)
    elif a == "take":
        out.action = {"id": aid, "action": "take", "item": d.item}
    elif a == "extinguish":
        out.action = {"id": aid, "action": "extinguish", "fire": d.target}
    elif a == "press_call":
        out.action = {"id": aid, "action": "press_call"}
    elif a == "guide":
        out.action = {"id": aid, "action": "guide_start", "track": d.target, "place": mind.place_id(d.destination)}
    elif a == "go_to":
        xy = mind.site.resolve(mind.place_id(d.destination))
        x, y, _ = mind.pose
        out.goal = (xy[0], xy[1], math.atan2(xy[1] - y, xy[0] - x))
    elif a == "wait":
        out.stop = True
    if out.action or out.goal:
        mind.note(f"You started {a} as {aid}.")
    return out


# ---- the mind -------------------------------------------------------------------------------
@dataclass
class _Track:
    x: float
    y: float
    posture: str
    t: float
    moving: bool = False


class RobotMind:
    """Decides when to think, builds the prompt, calls the gateway, validates the answer."""

    def __init__(self, site, brief: Optional[dict], gateway, clock: Callable[[], float], *,
                 profile: RobotProfile = G1_CLAUDE, known_places=(), pose=(0.0, 0.0, 0.0), router=None):
        self.site, self.gateway, self.clock, self.profile = site, gateway, clock, profile
        self.brief = brief
        self.known_places = list(known_places)
        self.pose = tuple(float(v) for v in pose)
        self._start = self.pose
        self._router = router
        self._places = {place_label(site, p).lower(): p for p in self.known_places if site.resolve(p) is not None}
        self._card: Optional[str] = None
        self._tracks: dict[str, _Track] = {}
        self._seen: set[str] = set()
        self._objects: dict = {}
        self._heard: deque = deque(maxlen=40)
        self._results: dict[tuple, dict] = {}
        self._recent_results: deque = deque(maxlen=6)
        self._events: deque = deque(maxlen=EVENTS_KEPT)
        self._now = 0.0
        self._t0: Optional[float] = None
        self._last_call: Optional[float] = None
        self._addressed = self._speech = self._new_track = self._new_result = False
        self._permanent = 0
        self._n_ids = 0
        self.llm_off = False

    # ---- inputs --------------------------------------------------------------------------
    def _t(self, t: Optional[float]) -> float:
        return float(self.clock() if t is None else t)

    def note(self, text: str) -> None:
        self._events.append(f"t={self._now:.1f}s {text}")

    def on_brief(self, brief: Optional[dict]) -> None:
        if brief is not None and brief != self.brief:
            self.brief, self._card = brief, None

    def on_pose(self, x: float, y: float, yaw: float) -> None:
        self.pose = (float(x), float(y), float(yaw))

    def place_id(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        low = text.strip().lower()
        if low in self._places:
            return self._places[low]
        hits = [p for lbl, p in self._places.items() if low in lbl or lbl in low]
        if len(hits) == 1:
            return hits[0]
        return text.strip() if text.strip() in self._places.values() else None

    def on_people(self, detections, t: Optional[float] = None) -> None:
        """camera/bbox_3d detections (the list, or the whole message). Tracks are anonymous."""
        t = self._t(t)
        if isinstance(detections, dict):
            detections = detections.get("detections", [])
        cur: dict[str, _Track] = {}
        for d in detections or []:
            try:
                if not any(r.get("class_id") == "person" for r in d.get("results", [])):
                    continue
                c, size = d["bbox"]["center"], d["bbox"]["size"]
                tid = str(d["id"])
                tr = _Track(float(c["x"]), float(c["y"]), posture_from_box(size), t)
            except (KeyError, TypeError, ValueError):
                continue
            prev = self._tracks.get(tid)
            if prev is not None and t - prev.t >= 0.3:
                tr.moving = math.hypot(tr.x - prev.x, tr.y - prev.y) / (t - prev.t) > 0.2
            elif prev is not None:
                tr.moving, tr.t = prev.moving, prev.t
                tr.x, tr.y = (prev.x, prev.y) if t - prev.t < 0.3 else (tr.x, tr.y)
            cur[tid] = tr
            if tid not in self._seen:
                self._seen.add(tid)
                self._new_track = True
        self._tracks = cur

    def on_objects(self, objs) -> None:
        self._objects = {"items": objs} if isinstance(objs, list) else dict(objs or {})

    def on_task_result(self, r: dict, t: Optional[float] = None) -> None:
        if not isinstance(r, dict) or "id" not in r:
            return
        key = (str(r["id"]), str(r.get("kind", "action")))
        if self._results.get(key) == r:
            return
        self._results[key] = r
        self._recent_results.append(r)
        self._new_result = True

    def _min_person_dist(self) -> float:
        return min((math.hypot(tr.x - self.pose[0], tr.y - self.pose[1]) for tr in self._tracks.values()),
                   default=math.inf)

    async def on_heard(self, h: dict, t: Optional[float] = None) -> None:
        t = self._t(t)
        if not isinstance(h, dict):
            return
        self._heard.append((t, h))
        if h.get("addressed_to_me"):
            self._addressed = True
        elif h.get("source_kind") == "person" and h.get("text") and self._min_person_dist() <= NEAR_SPEECH_M:
            self._speech = True

    # ---- prompt --------------------------------------------------------------------------
    def card(self) -> str:
        """The cached block: the site, the way to each place, the job and the robot's body."""
        if self._card is None:
            b = self.brief or {}
            rooms = self.site.summary_for_prompt(self.known_places)
            lines = ["THE HOSPITAL"] + ([rooms, ""] if not rooms.startswith("You do not know") else [])
            lines += [route_words(self.site, self._start, self.known_places, self._router), "", "YOUR JOB"]
            if self.brief is None:
                lines.append("No task brief was given. Be a polite, helpful presence and ask people what they need.")
            else:
                for k, v in b.items():
                    if isinstance(v, str) and v:
                        if k == "destination" and self.place_id(v):
                            v = place_label(self.site, self.place_id(v))
                        lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")
            lines += ["", "YOUR BODY", self.profile.abilities()]
            self._card = "\n".join(lines)
        return self._card

    def _situation(self, t: float) -> str:
        x, y, yaw = self.pose
        rid = self.site.room_at(x, y)
        L = [f"NOW t={t:.1f} s. You are at (x={x:.2f}, y={y:.2f}), facing {math.degrees(yaw):.0f} degrees, "
             f"in {place_label(self.site, rid) if rid else 'an unnamed space'}."]
        held = self._objects.get("held") or []
        L.append("You are holding: " + (", ".join(f"{i.get('label') or i.get('kind')} (item {i['id']})" for i in held)
                                        if held else "nothing."))
        L.append("Heard recently, newest last:")
        recent = [(ht, h) for ht, h in self._heard if t - ht <= HEARD_KEPT_S][-8:]
        for ht, h in recent or []:
            who = "someone speaking to you" if h.get("addressed_to_me") else {
                "person": "someone nearby", "cough": "a cough", "alarm": "an alarm", "pa": "an announcement",
                "robot": "your own voice"}.get(h.get("source_kind"), "a sound")
            text = f': "{h["text"]}"' if h.get("text") else ""
            part = " (only part heard)" if h.get("partial") else ""
            L.append(f"- {t - ht:.0f} s ago, {who}, {float(h.get('received_db', 0)):.0f} dB{part}{text}")
        if not recent:
            L.append("- nothing")
        L.append("People your camera sees (anonymous tracks):")
        for tid, tr in sorted(self._tracks.items()):
            d = math.hypot(tr.x - x, tr.y - y)
            L.append(f"- {tid}: {d:.1f} m, {_direction_words(_rel_deg(self.pose, tr.x, tr.y))}, {tr.posture}, "
                     f"{'moving' if tr.moving else 'still'}, at (x={tr.x:.2f}, y={tr.y:.2f})")
        if not self._tracks:
            L.append("- nobody in view")
        L.append("Objects in view:")
        n0 = len(L)
        for i in self._objects.get("items", []):
            L.append(f"- item {i['id']}: {i.get('label') or i.get('kind')}, {math.hypot(i['x'] - x, i['y'] - y):.1f} m, "
                     f"{_direction_words(_rel_deg(self.pose, i['x'], i['y']))}")
        for f in self._objects.get("fires", []):
            L.append(f"- fire {f['id']}: {math.hypot(f['x'] - x, f['y'] - y):.1f} m, "
                     f"{_direction_words(_rel_deg(self.pose, f['x'], f['y']))}, size {f.get('size', 0):.1f}")
        for b in self._objects.get("beds", []):
            L.append(f"- bed {b['id']}: {'occupied' if b.get('occupied') else 'empty'}, "
                     f"{math.hypot(b['x'] - x, b['y'] - y):.1f} m, {_direction_words(_rel_deg(self.pose, b['x'], b['y']))}")
        for c in self._objects.get("call_lights", []):
            L.append(f"- call light on at bed {c['bed']}, {math.hypot(c['x'] - x, c['y'] - y):.1f} m, "
                     f"{_direction_words(_rel_deg(self.pose, c['x'], c['y']))}")
        if len(L) == n0:
            L.append("- nothing notable")
        if self._recent_results:
            L.append("Results of your recent actions:")
            for r in self._recent_results:
                what = r.get("action", "action") if r.get("kind") != "goal" else "walking to a goal"
                L.append(f"- {r.get('id')} {what}: {'ok' if r.get('ok') else 'failed'} ({r.get('reason')})")
        L.append("Your last events, oldest first:")
        L += [f"- {e}" for e in self._events] or ["- none yet"]
        L.append("Decide what to do now.")
        return "\n".join(L)

    # ---- the tick ------------------------------------------------------------------------
    def _due(self, t: float) -> bool:
        if self.llm_off:
            return False
        if self.brief is None and t - (t if self._t0 is None else self._t0) < BRIEF_WAIT_S:
            return False
        gap = math.inf if self._last_call is None else t - self._last_call
        if self._addressed:
            return True
        if (self._speech or self._new_track or self._new_result) and gap >= MIN_GAP_S:
            return True
        return t - (self._last_call if self._last_call is not None else self._t0) >= HEARTBEAT_S

    def next_id(self) -> str:
        self._n_ids += 1
        return f"rb{self._n_ids}"

    async def tick(self, t: Optional[float] = None) -> Optional[RobotDecision]:
        t = self._now = self._t(t)
        if self._t0 is None:
            self._t0 = t
        if not self._due(t):
            return None
        self._addressed = self._speech = self._new_track = self._new_result = False
        self._last_call = t
        req = LLMRequest(role="robot", person="robot", system_fixed=system_prompt(self.profile),
                         persona_card=self.card(), user=self._situation(t),
                         schema=decision_schema(self.profile))
        res: LLMResult = await self.gateway.complete(req)
        if not res.ok:
            self._permanent = self._permanent + 1 if res.error in PERMANENT_ERRORS else 0
            if self._permanent >= LLM_OFF_AFTER and not self.llm_off:
                self.llm_off = True
                log.error("The AI provider is rejecting the robot's requests (%s). The robot stops thinking "
                          "for this run; check the key and model.", res.error)
            else:
                log.warning("robot call failed: %s", res.error)
            self.note(f"Your thinking failed ({res.error}).")
            return None
        self._permanent = 0
        try:
            d = parse_decision(res.data, self.profile, self)
            self.note(f"You decided: {d.action} ({d.reason})"
                      + (f"; you said: \"{speakable(d.say, self.site)}\"" if d.say else ""))
            return d
        except ValueError as exc:
            log.warning("dropped an invalid robot decision: %s", exc)
            self.note("Your last answer was invalid and was dropped.")
            return None


# ---- tokens ---------------------------------------------------------------------------------
def make_usage_logger(runlog, now: Callable[[], float]) -> Callable[[LLMRequest, LLMResult], None]:
    """The on_usage the agent node uses, writing the same tokens.jsonl row with person "robot"."""
    def on_usage(req: LLMRequest, res: LLMResult) -> None:
        runlog.write("tokens", {
            "t": now(), "person": req.person, "role": req.role, "model": res.model, "ok": res.ok,
            "error": res.error, "status": res.status, "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "cache_read_tokens": res.cache_read_tokens,
            "cache_write_tokens": res.cache_write_tokens, "cost_usd": res.cost_usd,
            "latency_s": res.latency_s})
    return on_usage


# ---- the ROS node ---------------------------------------------------------------------------
def _parse(argv):
    import argparse
    p = argparse.ArgumentParser(description="Built-in Claude robot (the robot side of the ROS contract only).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--robot-config", help="the robot's own config (robot_config.json in the run folder)")
    src.add_argument("--scenario", help="a scenario file (worked example; the robot reads only its robot part)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--domain", type=int, default=None,
                   help="the ROS domain the robot lives on (the run's robot domain); default ROS_DOMAIN_ID")
    return p.parse_args(argv)


def _robot_view(args):
    """The robot's view of the run: from robot_config.json, or from a scenario file (worked example)."""
    from ..config.loader import ConfigError, default_models, load_scenario
    from .view import RobotView, load_robot_view
    if args.robot_config:
        try:
            return load_robot_view(args.robot_config)
        except (OSError, ValueError, KeyError, ConfigError) as exc:
            raise ConfigError(f"{args.robot_config}: {exc}") from None
    loaded = load_scenario(args.scenario)
    sc = loaded.scenario
    return RobotView(root=loaded.root, site_name=sc.site, area=sc.area, robot=sc.robot,
                     known_places=list(sc.known_places), models=sc.models or default_models(sc.provider),
                     provider=sc.provider, budget_usd=sc.budget_usd * ROBOT_BUDGET_SHARE, brief=sc.robot_brief,
                     domain=None, site=loaded.site, waiting_area=loaded.area, signs=loaded.signs)


async def _run(args) -> int:
    from pathlib import Path

    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry

    from .. import keys
    from ..config.loader import ConfigError
    from ..llm.gateway import Budget, make_gateway
    from ..ros import topics
    from ..ros.rosutil import QOS, QOS_LATCHED, RosBridge, install_stop_signals, wait_or_sleep
    from ..runlog import RunLog

    try:
        view = _robot_view(args)
    except ConfigError as exc:
        print(f"Robot config error: {exc}", file=sys.stderr)
        return 2
    cfg = view.robot
    profile = get_profile(cfg.profile)
    rt = topics.robot_topics(cfg.ns)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    install_stop_signals(loop, stop)
    ros = RosBridge(NODE_NAME, loop, domain_id=args.domain)
    runlog = RunLog(Path(args.run_dir))
    budget = Budget(view.budget_usd)
    try:
        gateway = make_gateway("claude", view.models, budget, make_usage_logger(runlog, ros.now))
    except keys.KeyMissing as exc:
        print(str(exc), file=sys.stderr)
        ros.close()
        return 2
    mind = RobotMind(view.site, None, gateway, ros.now, profile=profile,
                     known_places=view.known_places, pose=(cfg.x, cfg.y, cfg.yaw))
    t_start = None

    def jload(m):
        try:
            return json.loads(m.data)
        except ValueError:
            return None

    def on_brief(b):
        if isinstance(b, dict):
            mind.on_brief(b)

    ros.subscribe(String, rt["heard"], lambda h: asyncio.ensure_future(mind.on_heard(h)), QOS, convert=jload)
    ros.subscribe(String, rt["bbox_3d"], lambda m: mind.on_people(m), QOS, convert=jload)
    ros.subscribe(String, rt["objects"], lambda m: mind.on_objects(m), QOS, convert=jload)
    ros.subscribe(String, rt["task_result"], lambda m: mind.on_task_result(m), QOS, convert=jload)
    ros.subscribe(String, rt["task_brief"], on_brief, QOS_LATCHED, convert=jload)

    def on_odom(m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        mind.on_pose(m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    ros.subscribe(Odometry, rt["odom"], on_odom, QOS)
    for key in ("say", "task_action"):
        ros.publisher(rt[key], String, QOS)
    ros.publisher(rt["goal_pose"], PoseStamped, QOS)
    ros.publisher(rt["cmd_vel"], Twist, QOS)
    ros.start()
    log.info("%s up on domain %s (%s, %s)", NODE_NAME, "default" if args.domain is None else args.domain,
             profile.name, "mobile" if profile.can_move else "fixed post")

    def publish(out: Outputs) -> None:
        if out.say:
            ros.publish_text(rt["say"], out.say)
        if out.action:
            ros.publish_text(rt["task_action"], json.dumps(out.action))
        if out.goal:
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.pose.position.x, msg.pose.position.y = out.goal[0], out.goal[1]
            msg.pose.orientation.z, msg.pose.orientation.w = math.sin(out.goal[2] / 2), math.cos(out.goal[2] / 2)
            ros.publisher(rt["goal_pose"], PoseStamped, QOS).publish(msg)
        if out.stop:
            ros.publisher(rt["cmd_vel"], Twist, QOS).publish(Twist())

    try:
        while not stop.is_set():
            try:
                now = ros.now()
                t_start = now if t_start is None else t_start
                if mind.brief is None and view.brief and now - t_start >= BRIEF_WAIT_S:
                    mind.on_brief(view.brief)  # no latched brief arrived: use the one in the robot's own config
                d = await mind.tick(now)
                if d is not None:
                    publish(decision_outputs(d, mind, mind.next_id()))
            except Exception as exc:  # one bad tick must not end the run
                log.error("robot tick failed: %s", type(exc).__name__)
                log.debug("tick traceback", exc_info=True)
            await wait_or_sleep(stop, 0.1)
    finally:
        ros.close()
    return 0


def main(argv=None) -> int:
    from ..ros.rosutil import setup_logging
    setup_logging()
    return asyncio.run(_run(_parse(argv)))


if __name__ == "__main__":
    sys.exit(main())
