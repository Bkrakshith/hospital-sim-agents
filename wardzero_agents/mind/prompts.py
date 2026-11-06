"""Prompt text for the persona mind: the fixed instructions, the persona card, the situation and the schemas."""
from __future__ import annotations

from typing import Optional, get_args

import copy
import math

from ..config.models import PersonInstance, WaitingArea
from ..config.presets import BASIC_INTENTS, ROLE_INTENTS
from ..contracts import Heard, IntentKind, Nudge, Observation, Seen, Status

ALL_INTENT_KINDS: list[str] = list(get_args(IntentKind))
INTENT_KINDS: list[str] = [k for k in ALL_INTENT_KINDS if k in BASIC_INTENTS]  # the basics every role has
MOODS: list[str] = ["calm", "happy", "sad", "anxious", "stressed", "excited", "relieved", "frustrated",
                    "distressed"]
MAX_MEMORIES = 3
SEAT_MATCH_M = 0.6  # a seated person this close to a seat is "at" that seat
NEAR_LANDMARK_M = 1.5

KB_HEADER = (
    "Background on how people in this situation typically behave (for your behaviour only; you do "
    "not know these as facts and never quote figures or studies):"
)

_INTENT_HELP = {
    "sit_idle": "sit and wait quietly",
    "use_phone": "look at your phone",
    "fidget": "shift about, restless",
    "look_at": "look at someone or something (target needed)",
    "read_sign": "go and read a sign (target: a sign id)",
    "stand_up": "stand up from your seat",
    "sit_down": "sit down (target: a seat id)",
    "walk_to": "walk to a place, seat, sign or person (target needed)",
    "approach": "go up to a person or the robot to talk (target needed)",
    "wait_standing": "stand and wait where you are",
    "talk": "say something to someone (text needed)",
    "gesture": "wave, point or nod",
    "cough": "cough",
    "leave": "leave the hospital (target defaults to the entrance)",
    "attend": "go to a person to attend to them (target: a person id)",
    "give_item": "hand an item you hold to a person (target: a person id; text: the item id)",
    "take_item": "take an item that a person next to you is holding (target: that person's id)",
    "call_for_help": "call out for help or for a nurse (text: what you say)",
    "escort": "walk with a person to a place (target: a person id; text: the place id)",
    "evacuate": "take a person to safety in the next fire compartment (target: a person id)",
    "call_bell": "press your call bell from your bed",
    "accept_item": "accept the item being handed to you",
    "refuse_item": "refuse the item being handed to you",
    "request_instrument": "ask the robot for an instrument (text: its name)",
    "reject_item": "push back: this is not the item you asked for",
    "take_instrument_self": "take an instrument from the tray yourself (text: its name)",
    "use_extinguisher": "put out a fire close to you with an extinguisher",
    "sit_up_in_bed": "sit up in your bed",
    "lie_back": "lie back down flat in your bed",
    "follow": "follow the robot when it offers to walk you somewhere (target: g1); you stop when it stops there",
}

SYSTEM_FIXED = (
    "You are one person in a UK hospital. Each turn you are given your "
    "persona card and the situation you are in right now, and you choose what you do next.\n"
    "Rules:\n"
    "- Choose only from the listed intents:\n"
    + "".join(f"  - {k}: {_INTENT_HELP[k]}\n" for k in INTENT_KINDS)
    + "- A target must be an id you were given: a seat id, a place id, a sign id, a person id, "
    "or g1 for the reception robot. Use null when no target is needed.\n"
    "- Stay in character. Act as this person would, with their health, mood, patience and goals.\n"
    "- Say short lines in British English, one or two sentences at most, or say nothing (null). "
    "Most of the time people in a hospital say nothing.\n"
    "- You only know what the situation tells you. Do not invent things you have not seen, heard "
    "or remembered.\n"
    "- Never quote statistics, studies or figures. Speak as an ordinary member of the public.\n"
    "- Lines starting ROBOT SAID TO YOU are the reception robot speaking to you. Lines starting "
    "NUDGE are a feeling or need you have right now; act on them in character.\n"
    "- mood is how you feel after this turn; patience_delta (between -1 and 1) is how much your "
    "patience changes, negative when you are getting fed up.\n"
    "- reason is one short sentence on why you chose this, in the third person."
)

SUMMARY_FIXED = (
    "You keep the long-term memory of one person in a UK hospital. Given their "
    f"recent events, write at most {MAX_MEMORIES} short first-person memories worth keeping for a later "
    "visit (what they learned, how they were treated, what the robot did). importance is between "
    "0 and 1. Return an empty list if nothing is worth keeping."
)


def _nullable() -> dict:
    """A string or null. anyOf, not a type array: Claude structured outputs and OpenAI strict
    mode both accept anyOf."""
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def role_kinds(role: Optional[str]) -> list[str]:
    """The intent kinds a role may choose, in IntentKind order. Unknown or no role: the basics."""
    allowed = ROLE_INTENTS.get(role or "custom", BASIC_INTENTS)
    return [k for k in ALL_INTENT_KINDS if k in allowed]


def system_fixed(role: Optional[str] = None) -> str:
    """SYSTEM_FIXED, plus one help line per intent this role has beyond the basics (and a note
    when it lacks some of the basics). Identical to SYSTEM_FIXED when the role has the basics only."""
    kinds = role_kinds(role)
    extra = [k for k in kinds if k not in BASIC_INTENTS]
    lost = [k for k in INTENT_KINDS if k not in kinds]
    if not extra and not lost:
        return SYSTEM_FIXED
    out = SYSTEM_FIXED
    if extra:
        out += "\nYour role also lets you choose:\n" + "".join(f"  - {k}: {_INTENT_HELP[k]}\n" for k in extra)
    if lost:
        out += "\nYour role does not let you choose: " + ", ".join(lost) + ".\n"
    return out


def decision_schema(role: Optional[str] = None) -> dict:
    """DECISION_SCHEMA with the intent.kind enum filtered to the role's intents."""
    schema = copy.deepcopy(DECISION_SCHEMA)
    schema["properties"]["intent"]["properties"]["kind"]["enum"] = role_kinds(role)
    return schema


DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(INTENT_KINDS)},
                "target": _nullable(),
                "text": _nullable(),
            },
            "required": ["kind", "target", "text"],
            "additionalProperties": False,
        },
        "say": _nullable(),
        "addressee": _nullable(),
        "mood": {"type": "string", "enum": MOODS},
        "patience_delta": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["intent", "say", "addressee", "mood", "patience_delta", "reason"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "importance": {"type": "number"}},
                "required": ["text", "importance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memories"],
    "additionalProperties": False,
}


def _level(x: float) -> str:
    return "low" if x < 0.35 else "moderate" if x < 0.65 else "high"


TRUST_HEADER = "What you think of this robot, from experience:"
TRUST_USE = ("Whether you use it depends on how much you trust it against how sure you are of doing "
             "this yourself, how urgent things are, and whether anyone else can help.")
WOM_YES = "If it comes up naturally, you may tell others what you think of the robot."
WOM_NO = "Keep your opinion of the robot to yourself unless asked."


def persona_card(inst: PersonInstance, area: WaitingArea, site=None, known_places=(), *,
                 trust_stage: int = 0, ladder: "dict[int, str] | None" = None,
                 robot_memories: bool = False, word_of_mouth: "bool | None" = None) -> str:
    """Stable description of one person and the places they can name. It is the cached prompt block.

    The trust section is added only for a campaign person who has moved off stage 0 or remembers the
    robot; word_of_mouth is None outside campaigns, so existing scenarios keep their exact prompt."""
    lines = [
        f"You are {inst.display_name} (person id {inst.person_id}), {inst.age}, {inst.sex}.",
        f"Role: {inst.role}. Why you are here: {inst.condition}",
        f"Background: {inst.backstory.strip()}",
        "Your goals today:",
        *[f"- {g}" for g in inst.goals],
        "Health and mobility:",
    ]
    if inst.cough_per_hour > 0:
        lines.append(f"- You have a {inst.cough_severity} cough, about {inst.cough_per_hour:.0f} times an hour.")
    if inst.breathless:
        lines.append("- You get breathless easily.")
    aid = {"none": "You walk without an aid.", "stick": "You walk with a stick.",
           "frame": "You walk with a walking frame.", "wheelchair": "You use a wheelchair."}
    lines.append(f"- {aid[inst.mobility_aid]} Getting up from a chair takes you about "
                 f"{inst.sit_to_stand_s:.0f} seconds.")
    if inst.hearing_loss_db >= 25:
        lines.append("- You are hard of hearing and often miss what is said.")
    lines += [
        f"English: {inst.english_fluency}.",
        "Temperament:",
        f"- You get restless after about {inst.patience_min:.0f} minutes of waiting.",
        f"- Anxiety {_level(inst.anxiety)}, sociability {_level(inst.sociability)}, "
        f"phone use {_level(inst.phone_use)}.",
        f"- Feelings about robots: {inst.attitude_to_robots}.",
    ]
    if inst.companion_of:
        lines.append(f"You are here with {inst.companion_of} and are looking after them.")
    lines.append(f"You arrived and sat at seat {inst.seat}." if inst.seat else "You arrived without a seat.")
    if site is None:
        lines.append("Places in the waiting area (id: description):")
        lines += [f"- {lid}: {lm.label}" for lid, lm in area.landmarks.items()]
        lines.append("Seat ids: " + ", ".join(s.id for s in area.seats) + ".")
    else:
        lines += _known_places(inst, site, known_places)
    lines.append("The reception robot is g1.")
    if trust_stage > 0 or robot_memories:
        lines.append(TRUST_HEADER)
        if ladder and ladder.get(trust_stage):
            lines.append(ladder[trust_stage])
        lines.append(TRUST_USE)
    if word_of_mouth is not None:
        lines.append(WOM_YES if word_of_mouth else WOM_NO)
    return "\n".join(lines)


def _start_xy(inst: PersonInstance, site, default_entrance: str = "entrance"):
    if inst.seat is not None and inst.seat in site.seats:
        s = site.seats[inst.seat]
        return s.x, s.y
    if inst.start is not None:
        if inst.start.bed is not None and inst.start.bed in site.beds:
            b = site.beds[inst.start.bed]
            return b.x, b.y
        return inst.start.x, inst.start.y
    lm = site.landmarks.get(inst.entrance or default_entrance)
    return (lm.x, lm.y) if lm else None


def _neighbours(site, rid: str) -> list[str]:
    """Rooms whose bounding box touches this room's (within half a metre)."""
    def box(r):
        xs, ys = [q[0] for q in r.poly], [q[1] for q in r.poly]
        return min(xs), max(xs), min(ys), max(ys)
    x0, x1, y0, y1 = box(site.rooms[rid])
    out = []
    for oid, o in site.rooms.items():
        if oid == rid:
            continue
        a0, a1, b0, b1 = box(o)
        if a0 <= x1 + 0.5 and a1 >= x0 - 0.5 and b0 <= y1 + 0.5 and b1 >= y0 - 0.5:
            out.append(oid)
    return out


def _known_places(inst: PersonInstance, site, known_places) -> list[str]:
    """The places this person can name: landmarks of the room they start in, the rooms next to it,
    and the scenario's known_places. Not every id on the site."""
    xy = _start_xy(inst, site)
    here = site.room_at(*xy) if xy else None
    rooms = [here] if here else []
    rooms += _neighbours(site, here) if here else []
    lines = ["Places you know (id: description):"]
    seen: set[str] = set()
    for lid, lm in site.landmarks.items():
        if lid.endswith("_centre") or lid in seen:
            continue
        if here and site.room_at(lm.x, lm.y) == here:
            lines.append(f"- {lid}: {lm.label}")
            seen.add(lid)
    for rid in rooms:
        if rid not in seen:
            r = site.rooms[rid]
            lines.append(f"- {rid}: {r.label}" + (" (where you are)" if rid == here else ""))
            seen.add(rid)
    for name in known_places:
        if name in seen:
            continue
        key = name.partition(":")[2] if ":" in name else name
        lm = site.landmarks.get(key)
        r = site.rooms.get(key)
        label = lm.label if lm and not name.startswith("room:") else (r.label if r else key)
        lines.append(f"- {key}: {label}")
        seen.add(name)
        seen.add(key)
    seats = [q.id for q in site.seats.values() if site.room_at(q.x, q.y) in rooms]
    if seats:
        lines.append("Seat ids: " + ", ".join(seats) + ".")
    return lines


def _direction(bearing: float) -> str:
    b = abs(bearing)
    if b <= 20:
        return "ahead"
    side = "left" if bearing > 0 else "right"
    if b <= 70:
        return f"ahead to your {side}"
    if b <= 120:
        return f"to your {side}"
    return f"behind you to the {side}"


def _seen_line(s: Seen) -> str:
    where = f"{s.distance:.1f} m {_direction(s.bearing_deg)}"
    if s.kind == "sign":
        return f"- sign {s.id}: \"{s.detail or ''}\" ({where})"
    if s.kind == "robot":
        return f"- the reception robot {s.id} ({where})"
    if s.kind == "person":
        return f"- person {s.id}{f' ({s.detail})' if s.detail else ''} ({where})"
    return f"- seat {s.id} ({where})"


def _heard_line(h: Heard) -> str:
    part = " (you only caught part of it)" if h.partial else ""
    if h.source_kind == "robot":
        if not h.text:
            return "The robot said something you could not make out."
        prefix = "ROBOT SAID TO YOU" if h.addressed_to_me else "ROBOT SAID NEARBY"
        return f"{prefix}: {h.text}{part}"
    if h.source_kind == "person":
        if not h.text:
            return f"{h.source_id} said something you could not make out."
        to = " to you" if h.addressed_to_me else ""
        return f"{h.source_id} said{to}: \"{h.text}\"{part}"
    if h.source_kind == "pa":
        return f"PA announcement: {h.text or '(could not make it out)'}{part}"
    if h.source_kind == "cough":
        return f"{h.source_id} coughed nearby."
    return "An alarm is sounding."


def _where(status: Status, area: Optional[WaitingArea], site=None) -> str:
    """_where_pos plus, with a site, the room the person is in."""
    base = _where_pos(status, area)
    rid = site.room_at(status.x, status.y) if site is not None and area is not None else None
    if rid is None:
        return base
    return f"{base[:-1]}, in {site.rooms[rid].label}."


def _where_pos(status: Status, area: Optional[WaitingArea]) -> str:
    """Where the person is now, from Status: a seat id when seated at a known seat, else a position."""
    if area is None:
        return f"You are {status.posture}."
    if status.posture == "seated":
        seat = min(area.seats + area.staff_seats, key=lambda s: math.hypot(s.x - status.x, s.y - status.y), default=None)
        if seat is not None and math.hypot(seat.x - status.x, seat.y - status.y) <= SEAT_MATCH_M:
            return f"You are seated at seat {seat.id}."
    near = min(area.landmarks.items(), key=lambda kv: math.hypot(kv[1].x - status.x, kv[1].y - status.y),
               default=None)
    out = f"You are {status.posture} at ({status.x:.1f}, {status.y:.1f})"
    if near is not None and math.hypot(near[1].x - status.x, near[1].y - status.y) <= NEAR_LANDMARK_M:
        out += f", near {near[0]} ({near[1].label})"
    return out + "."


def render_situation(inst: PersonInstance, obs: Observation, status: Status, triggers: list[str],
                     nudges: list[Nudge], clock_min: float, *, mood: str = "calm",
                     patience_left: float = 1.0, current_intent: Optional[dict] = None,
                     area: Optional[WaitingArea] = None, site=None) -> str:
    """What this person perceives now. Nudges come last, one per line."""
    m = round(clock_min)
    lines = [f"You have been waiting about {m} minute{'' if m == 1 else 's'}."]
    lines.append(_where(status, area, site))
    if current_intent:
        target = f" {current_intent['target']}" if current_intent.get("target") else ""
        state = current_intent.get("state") or status.state
        lines.append(f"Current activity: {current_intent['kind']}{target} ({state}).")
    elif status.intent_kind:
        lines.append(f"Current activity: {status.intent_kind} ({status.state}).")
    else:
        lines.append("Current activity: none.")
    if status.reason:
        lines.append(f"Last activity note: {status.reason}.")
    lines.append(f"Mood: {mood}. Patience left: {round(patience_left * 100)}%.")
    if obs.visible:
        lines.append("You can see:")
        order = {"robot": 0, "person": 1, "sign": 2, "seat": 3}
        for s in sorted(obs.visible, key=lambda s: (order[s.kind], s.distance)):
            lines.append(_seen_line(s))
    else:
        lines.append("You see nothing of note.")
    if obs.objects:
        lines.append("You can also see:")
        lines += [f"- {o.detail} ({o.distance:.1f} m away)" for o in sorted(obs.objects, key=lambda o: o.distance)]
    if obs.heard:
        lines.append("You hear:")
        lines += [_heard_line(h) for h in obs.heard]
    lines.append("Why you are deciding now: " + (", ".join(triggers) if triggers else "nothing new"))
    lines += [f"NUDGE: {n.message}" for n in nudges]
    return "\n".join(lines)
