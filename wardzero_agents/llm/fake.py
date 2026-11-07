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
    nudge = re.search(r"NUDGE: The robot (?:tells you:|pressed the staff call).*?person id (\w+)", u)
    if nudge:
        return _decision("approach", nudge.group(1))
    if "NUDGE: The robot pressed the staff call" in u:  # nobody named: the robot is with whoever needs a nurse
        return _decision("approach", "g1")
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


FLOW_NOTE = "The scripted model kept the preset's flow and added the prompt to its summary."


def _adapted_flow(u: str) -> str:
    """The preset's flow with the user's prompt added to its summary: a change the Studio shows, so adapting a flow
    can be checked end to end without a model."""
    i, j = u.find("<current_flow>"), u.find("</current_flow>")
    text = u[i + len("<current_flow>"):j] if 0 <= i < j else ""
    m = re.search(r"The scenario as the user now describes it \(write the flow for this\):\n(.*?)\n\n", u, re.S)
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return text
    if not isinstance(data, dict) or m is None:
        return text
    prompt = " ".join(m.group(1).split())[:200]
    data["summary"] = f"{str(data.get('summary') or '').strip()} Adapted to: \"{prompt}\"".strip()
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


# ---- the people ---------------------------------------------------------------------------------------------------
def _kinds(req: LLMRequest) -> list[str]:
    """The intent kinds this person may choose (the schema's enum), or the basics."""
    try:
        return list(req.schema["properties"]["intent"]["properties"]["kind"]["enum"])
    except (KeyError, TypeError):
        return []


def _idle(req: LLMRequest) -> dict:
    """Waiting quietly, in a way this person's role allows (a surgeon cannot sit_idle)."""
    kinds = _kinds(req)
    return _decision(next((k for k in ("sit_idle", "wait_standing") if not kinds or k in kinds), kinds[0]))


def _card_places(card: str) -> list[tuple[str, str]]:
    """(label, id) of the places a person knows, from their card's "Places you know" lines, longest label first."""
    out = []
    block = card.split("Places you know", 1)[1] if "Places you know" in card else ""
    for m in re.finditer(r"^- (\w+): (.+)$", block, re.M):
        label = re.split(r"[,+(]", m.group(2).replace("(where you are)", ""), 1)[0].strip().lower()
        if len(label) >= 4:
            out.append((label, m.group(1)))
    return sorted(out, key=lambda p: -len(p[0]))


_GO = re.compile(r"\b(?:walk|go|come|head|move|make your way|leave|get out)\b[^.]*?\b(?:to|towards|by|via|through)\b",
                 re.I)


def _place(text: str, card: str) -> Optional[str]:
    """A place named in text: the one a "go to ..." points at when the words say where to go (so "the entrance is
    blocked, walk to the handover bay" means the bay), else a waiting-area landmark, else a place on the card."""
    low = text.lower()
    found: list[tuple[int, str]] = []
    for label, pid in [*((lm, lm) for lm in LANDMARKS), *LABELS.items(), *_card_places(card)]:
        for m in re.finditer(re.escape(label.lower()), low):
            found.append((m.start(), pid))
    go = _GO.search(text)
    if go is not None:
        after = sorted(f for f in found if f[0] >= go.end())
        if after:
            return after[0][1]
    lm = _landmark(text)
    if lm:
        return lm
    return next((pid for label, pid in _card_places(card) if label in low), None)


def _own_name(card: str) -> str:
    m = re.match(r"You are (.+?) \(person id", card or "")
    return m.group(1) if m else ""


def _asleep(u: str) -> bool:
    """Fell asleep (a nudge said so) and nobody has spoken to them since."""
    i = max(u.rfind("You are deeply asleep"), u.rfind("You are asleep"))
    if i < 0:
        return False
    return max(u.rfind("the robot said to me"), u.rfind("ROBOT SAID TO YOU")) < i


_OFFER = re.compile(r"NUDGE: The robot is holding out the (.+?) (?:to you|for you)\.")
_BATCH = re.compile(r"ask for these instruments, one at a time and in this order, by name: ([^.\n]+)")


def _instruments(u: str) -> tuple[list[str], int, int]:
    """(the instruments asked for, in order; how many were accepted; how many requests were made)."""
    names = [n.strip().lower() for b in _BATCH.findall(u) for n in b.split(",") if n.strip()]
    return names, u.count("I chose accept_item"), u.count("I chose request_instrument")


def _wants(item: str, req: LLMRequest, u: str) -> bool:
    """Whether the item the robot holds out is this person's: a tray labelled with their name, the instrument the
    surgeon asked for last, or the kind of thing their own goals say they are waiting for."""
    card = req.persona_card or ""
    low = item.lower()
    name = _own_name(card)
    if name and name.lower() in low:
        return True
    if re.search(r"\(([A-Z][a-z]+ ){1,2}[A-Z][a-z'-]+ - ", item):
        return False  # labelled for someone else
    if "request_instrument" in _kinds(req):
        names, accepted, _ = _instruments(u)
        return accepted < len(names) and low == names[accepted]
    kind = low.split(" (", 1)[0].strip()
    own = "\n".join(ln for ln in card.splitlines() if "written to the robot" not in ln).lower()
    return bool(kind) and kind in own


def _offer_reply(req: LLMRequest, u: str) -> Optional[dict]:
    m = _OFFER.search(u)
    if m is None:
        return None
    kinds = _kinds(req)
    if _wants(m.group(1), req, u):
        return _decision("accept_item")
    no = next((k for k in ("reject_item", "refuse_item") if k in kinds), None)
    return _decision(no) if no else _idle(req)


def _surgeon(req: LLMRequest, u: str) -> Optional[dict]:
    """Ask the robot for the next instrument on the list once the last one is in hand."""
    if "request_instrument" not in _kinds(req):
        return None
    names, accepted, asked = _instruments(u)
    if accepted < len(names) and asked <= accepted:
        d = _decision("request_instrument", "", "", "g1")
        d["intent"]["text"] = names[accepted]
        return d
    return None


def _help_line(nudge: str) -> str:
    m = re.search(r"where (.+?) is\b", nudge)
    what = f" I can't find {m.group(1)}." if m else ""
    return f"Excuse me, can you help us?{what}"


def _carry_on(u: str) -> Optional[dict]:
    """Keep walking where they were going: a new sound or face is no reason to stop half way."""
    m = re.search(r"Current activity: (walk_to|approach) (\S+) \((sent|running)\)", u)
    return _decision(m.group(1), m.group(2)) if m else None


def person_policy(req: LLMRequest) -> dict:
    u = req.full_user
    card = req.persona_card or ""
    if _asleep(u):  # a sleeping person does not answer, whatever is held out or said
        return _idle(req)
    offered = _offer_reply(req, u)
    if offered is not None:
        return offered
    if "NUDGE: The robot is offering to walk you" in u or re.search(r"Current activity: follow \S+ \((sent|running)\)", u):
        return _decision("follow", "g1")  # the natural reply to a guide offer, and keep following
    if "Role: receptionist." in card or "NUDGE: The robot tells you" in u or "NUDGE: The robot pressed the staff" in u:
        errand = _staff_errand(u + "\n" + card)
        if errand is not None:
            return errand
    asked = _surgeon(req, u)
    if asked is not None:
        return asked
    if "ROBOT SAID TO YOU" in u:
        lm = _place(u.split("ROBOT SAID TO YOU", 1)[1].split("\n", 1)[0], card)
        return _decision("walk_to", lm, "Thank you") if lm else _decision("wait_standing", "", "Thank you")
    if "NUDGE:" in u:
        nudge = u.split("NUDGE:", 1)[1]
        lm = _place(nudge, card)
        if "help" in nudge:  # someone who wants help asks the robot for it, in so many words
            return _decision("approach", "g1", _help_line(nudge), "g1")
        if lm:
            return _decision("walk_to", lm)
        return _decision("wait_standing") if "wait_standing" in (_kinds(req) or ["wait_standing"]) else _idle(req)
    return _carry_on(u) or _idle(req)


def default_policy(req: LLMRequest) -> dict:
    props = (req.schema or {}).get("properties", {})
    if req.role == "report" and "flow_yaml" in props:  # adapting a scenario flow: the preset's, visibly marked
        return {"flow_yaml": _adapted_flow(req.user), "notes": FLOW_NOTE}
    if req.role == "report":
        return {"narrative": "", "suggestions": []}
    if req.role == "summarise" and "likeability" in props:  # the after-run questionnaire: a pass, so a test-mode
        # run can meet the 3.5 rating criteria
        return {"likeability": 4, "safety": 4, "competence": 4, "trust": 4, "comment": "It was fine."}
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
    return person_policy(req)


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
