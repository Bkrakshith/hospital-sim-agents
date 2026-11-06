"""Deterministic offline gateway for tests. No network, no key, zero usage."""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from .gateway import LLMRequest, LLMResult

LANDMARKS = ["entrance", "clinic_corridor", "ward_doors", "toilets", "reception_front", "nurse_call_point"]


class FakeRefusal(Exception):
    """Raise from a policy to simulate a model refusal."""


# The placeholder robot speaks the human-readable landmark labels, not the ids.
LABELS = {"outpatient clinics": "clinic_corridor", "wards": "ward_doors", "toilets": "toilets",
          "main entrance": "entrance"}


def _landmark(text: str) -> Optional[str]:
    low = text.lower()
    return next((lm for lm in LANDMARKS if lm in text), None) or next(
        (lm for label, lm in LABELS.items() if label in low), None)


ROBOT_GREETING = "Hello, I am the hospital robot. Can I help you?"


def _person_in_view(situation: str) -> bool:
    """The robot's situation lists tracks under 'People your camera sees' and says 'nobody in view' when none."""
    if "People your camera sees" not in situation:
        return False
    block = situation.split("People your camera sees", 1)[1].split("Objects in view", 1)[0]
    rows = [ln for ln in block.splitlines() if ln.startswith("- ")]
    return bool(rows) and "nobody in view" not in block


def _decision(kind: str, target: str = "", say: str = "", addressee: str = "") -> dict:
    return {"intent": {"kind": kind, "target": target, "text": ""}, "say": say, "addressee": addressee,
            "mood": "calm", "patience_delta": 0.0, "reason": "fake policy"}


ROBOT_TELL_SAY = "I will let the receptionist know straight away. Please wait here."
ROBOT_TELL_MESSAGE = "Excuse me, a visitor here needs a nurse, please."
ROBOT_GUIDE_SAY = "Of course. Please come with me, I will walk with you."
STAFF_HELLO = "Hello, I am from reception. A nurse is on the way to you."


def _robot_tracks(situation: str) -> list[tuple[float, str]]:
    """(distance, track) of every track the robot's camera lists, nearest first."""
    block = situation.split("People your camera sees", 1)[-1].split("Objects in view", 1)[0]
    return sorted((float(m.group(2)), m.group(1)) for m in re.finditer(r"^- (t\w+): ([0-9.]+) m", block, re.M))


def _robot_reacts(req: LLMRequest, props: dict) -> Optional[dict]:
    """The fake robot reacts to two requests spoken to it, once each: a nurse (tell_staff about the nearest person
    it has not helped yet) and the ward (guide them to the brief's destination). Otherwise None."""
    u = req.user
    asked = " ".join(re.findall(r"someone speaking to you[^:]*: \"([^\"]*)\"", u)).lower()
    allowed = props.get("action", {}).get("enum", [])
    done = set(re.findall(r"for track (t\w+)", u))
    tracks = [t for _, t in _robot_tracks(u) if t not in done]
    if not asked or not tracks:
        return None
    base = {"say": "", "toward_track": tracks[0], "action": "none", "target": tracks[0], "item": "",
            "destination": "", "place": [], "message": "", "reason": "fake policy"}
    dest = re.search(r"^Destination: (.+)$", req.persona_card or "", re.M)
    if "ward" in asked and "guide" in allowed and "You started guide" not in u and dest:
        return {**base, "say": ROBOT_GUIDE_SAY, "action": "guide", "destination": dest.group(1).strip()}
    if "nurse" in asked and "tell_staff" in allowed and "You started tell_staff" not in u:
        return {**base, "say": ROBOT_TELL_SAY, "action": "tell_staff", "message": ROBOT_TELL_MESSAGE}
    return None


def _staff_errand(u: str) -> Optional[dict]:
    """A member of staff the robot told about a person: go to them, say hello, then back to the chair."""
    seat = re.search(r"sat at seat (\w+)", u)
    nudge = re.search(r"NUDGE: The robot tells you:.*?person id (\w+)", u)
    if nudge:
        return _decision("approach", nudge.group(1))
    cur = re.search(r"Current activity: (approach|sit_down) (\w+) \((\w+)\)", u)
    if cur is None or seat is None:
        return None
    kind, target, state = cur.groups()
    if kind == "approach" and state in ("sent", "running"):
        return _decision("approach", target)
    if kind == "approach" and state == "done":
        return _decision("sit_down", seat.group(1), STAFF_HELLO, target)
    if kind == "sit_down" and state in ("sent", "running"):
        return _decision("sit_down", target)
    return None


def default_policy(req: LLMRequest) -> dict:
    props = (req.schema or {}).get("properties", {})
    if req.role == "report" and "flow_yaml" in props:  # adapting a scenario flow: keep the preset's
        u = req.user
        i, j = u.find("<current_flow>"), u.find("</current_flow>")
        return {"flow_yaml": u[i + len("<current_flow>"):j] if 0 <= i < j else "", "notes": "The scripted model kept "
                "the preset's flow."}
    if req.role == "report":
        return {"narrative": "", "suggestions": []}
    if req.role == "summarise" and "likeability" in props:  # the after-run questionnaire
        return {"likeability": 3, "safety": 3, "competence": 3, "trust": 3, "comment": "It was fine."}
    if req.role == "summarise" and "items" in props:  # the word-of-mouth classifier
        return {"items": []}
    if req.role == "summarise":
        return {"memories": []}
    if req.role == "robot":  # the built-in robot waits, and greets once when a person is in view
        acted = _robot_reacts(req, props)
        if acted is not None:
            return acted
        say = ROBOT_GREETING if _person_in_view(req.user) and "you said:" not in req.user else ""
        return {"say": say, "toward_track": "", "action": "none", "target": "", "item": "",
                "destination": "", "place": [], "reason": "fake policy"}
    u = req.full_user
    if "NUDGE: The robot is offering to walk you" in u or re.search(r"Current activity: follow \S+ \((sent|running)\)", u):
        return _decision("follow", "g1")  # the natural reply to a guide offer, and keep following
    if "Role: receptionist." in (req.persona_card or "") or "NUDGE: The robot tells you" in u:
        errand = _staff_errand(u + "\n" + (req.persona_card or ""))
        if errand is not None:
            return errand
    if "ROBOT SAID TO YOU" in u:
        lm = _landmark(u.split("ROBOT SAID TO YOU", 1)[1].split("\n", 1)[0])
        return _decision("walk_to", lm, "Thank you") if lm else _decision("wait_standing", "", "Thank you")
    if "NUDGE:" in u:
        nudge = u.split("NUDGE:", 1)[1]
        lm = _landmark(nudge)
        if lm:
            return _decision("walk_to", lm)
        if "help" in nudge:
            return _decision("approach", "g1", "Excuse me, can you help us?")
        return _decision("wait_standing")
    return _decision("sit_idle")


class FakeGateway:
    def __init__(self, policy: Optional[Callable[[LLMRequest], dict]] = None, on_usage=None):
        self.policy = policy or default_policy
        self.on_usage = on_usage

    async def complete(self, req: LLMRequest) -> LLMResult:
        t0 = time.monotonic()
        model = "fake"
        try:
            res = LLMResult(True, self.policy(req), None, model, cost_usd=0.0)
        except FakeRefusal:
            res = LLMResult(False, None, "refusal", model, cost_usd=0.0)
        except Exception:
            res = LLMResult(False, None, "provider_error", model, cost_usd=0.0)
        res.latency_s = time.monotonic() - t0
        if self.on_usage:
            self.on_usage(req, res)
        return res
