"""Deterministic offline gateway for tests. No network, no key, zero usage."""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from .fake_robot import GREET as ROBOT_GREETING  # noqa: F401  (what the scripted robot says to someone new)
from .fake_robot import robot_policy
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


def _decision(kind: str, target: str = "", say: str = "", addressee: str = "") -> dict:
    return {"intent": {"kind": kind, "target": target, "text": ""}, "say": say, "addressee": addressee,
            "mood": "calm", "patience_delta": 0.0, "reason": "fake policy"}


STAFF_HELLO = "Hello, I am from reception. A nurse is on the way to you."


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
    if not kinds:
        return _decision("sit_idle")
    return _decision(next((k for k in ("sit_idle", "wait_standing") if k in kinds), kinds[0]))


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


def _own_id(card: str) -> str:
    m = re.match(r"You are .+? \(person id (\w+)\)", card or "")
    return m.group(1) if m else ""


def _asleep(u: str) -> bool:
    """Fell asleep (a nudge said so) and nobody has spoken to them since."""
    i = max(u.rfind("You are deeply asleep"), u.rfind("You are asleep"))
    if i < 0:
        return False
    return max(u.rfind("the robot said to me"), u.rfind("ROBOT SAID TO YOU")) < i


_OFFER = re.compile(r"NUDGE: The robot is holding out the (.+?) (?:to you|for you)\.")
_BATCH = re.compile(r"ask for these instruments, one at a time and in this order, by name: ([^.\n]+)")


def _instruments(u: str, me: str = "") -> tuple[list[str], int, int]:
    """(the instruments of the latest batch asked for, in order; how many of them are in hand; how many requests
    were made since that batch). What is in hand is read off the objects in view, so it holds after the day so far
    has been summarised."""
    batches = list(_BATCH.finditer(u))
    if not batches:
        return [], 0, 0
    names = [n.strip().lower() for n in batches[-1].group(1).split(",") if n.strip()]
    after = u[batches[-1].end():]
    held = {lab.strip().lower() for lab in re.findall(r"\(id \w+, ([^)]+)\), held by " + re.escape(me or "\0"), u)}
    in_hand = max(after.count("I chose accept_item"), sum(1 for n in names if n in held))
    return names, in_hand, after.count("I chose request_instrument")


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
        names, accepted, _ = _instruments(u, _own_id(card))
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
    names, accepted, asked = _instruments(u, _own_id(req.persona_card or ""))
    if accepted < len(names) and asked <= accepted:
        d = _decision("request_instrument", "", "", "g1")
        d["intent"]["text"] = names[accepted]
        return d
    return None


_SKIP_PLACES = ("where the g1 stands", "in front of the reception desk", "where a nurse is called from", "where you are")
_LOST = ("do not know the way", "don't know the way", "go and ask it", "ask the robot", "cannot find", "can't find")


def _goal_text(card: str) -> str:
    """The person's own reasons and goals (their background and goal lines), without the robot's job quoted to them."""
    keep = [ln for ln in card.splitlines() if ln.startswith(("Background:", "Role:", "- ")) and
            "written to the robot" not in ln and not re.match(r"- \w+: ", ln)]
    return "\n".join(keep).lower()


def _my_place(card: str) -> Optional[str]:
    """Where this person is going, in words: the place on their card their goals name, else the one their role
    names (a ward visitor: the ward entrance). None when nothing says."""
    places = [(label, pid) for label, pid in _card_places(card) if not any(k in label for k in _SKIP_PLACES)]
    goals = _goal_text(card)
    words = lambda t: {w for w in re.findall(r"[a-z]+", t) if len(w) > 4}  # noqa: E731
    gw = words(goals)
    scored = sorted(((len(words(label) & gw), -len(label), label) for label, _ in places), reverse=True)
    if scored and scored[0][0] >= 2:
        return scored[0][2].split(",")[0].strip()
    role = re.search(r"^Role: (\w+)", card, re.M)
    for w in (role.group(1).split("_") if role else []):
        hit = next((label for label, _ in places if label.startswith(w)), None)
        if hit and len(w) >= 4:
            return hit.split(",")[0].strip()
    return None


def _lost(card: str) -> bool:
    return any(k in _goal_text(card) for k in _LOST)


def _help_line(nudge: str, card: str = "") -> str:
    m = re.search(r"where (.+?) is\b", nudge)
    place = _my_place(card)
    what = f" I need to get to the {place}." if place else f" I can't find {m.group(1)}." if m else ""
    return f"Excuse me, can you help us?{what}"


def _promised_staff(u: str) -> bool:
    """They were told a nurse or staff are coming to them (by the robot, or by staff on their way): they wait for
    them rather than set off somewhere."""
    said = [m.group(1).lower() for m in re.finditer(r"(?:the robot said to me|ROBOT SAID TO YOU:) \"?([^\"\n]*)", u)]
    return any(("staff" in t or "nurse" in t) and "wait" in t for t in said) or STAFF_HELLO in u


def _asked_the_way(u: str) -> bool:
    return bool(re.search(r"I chose approach g1 and said \"Excuse me", u))


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
        line = u.split("ROBOT SAID TO YOU", 1)[1].split("\n", 1)[0]
        mine = _my_place(card)
        if "where would you like to go" in line.lower() or ("where" in line.lower() and "?" in line):
            if mine:  # asked where they are going: they say
                return _decision("wait_standing", "", f"I need to get to the {mine}, please.", "g1")
        lm = _place(line, card)
        return _decision("walk_to", lm, "Thank you") if lm else _decision("wait_standing", "", "Thank you")
    if "NUDGE:" in u:
        nudge = u.split("NUDGE:", 1)[1]
        lm = _place(nudge, card)
        if "help" in nudge or (_lost(card) and not _asked_the_way(u)):
            if _promised_staff(u):  # the robot already sent for staff: wait for them
                return _decision("wait_standing")
            # someone who wants help, or does not know their way, asks the robot in so many words
            return _decision("approach", "g1", _help_line(nudge, card), "g1")
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
    if req.role == "robot":  # the scripted built-in robot: does the brief's main task (see fake_robot)
        return robot_policy(req)
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
