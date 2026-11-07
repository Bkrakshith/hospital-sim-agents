"""wz.v1 JSON contracts for every message that crosses the sim boundary."""
from __future__ import annotations

import json
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA = "wz.v1"

IntentKind = Literal[
    "sit_idle",
    "use_phone",
    "fidget",
    "look_at",
    "read_sign",
    "stand_up",
    "sit_down",
    "walk_to",
    "approach",
    "wait_standing",
    "talk",
    "gesture",
    "cough",
    "leave",
    "attend",
    "give_item",
    "take_item",
    "call_for_help",
    "escort",
    "evacuate",
    "call_bell",
    "accept_item",
    "refuse_item",
    "request_instrument",
    "reject_item",
    "take_instrument_self",
    "use_extinguisher",
    "sit_up_in_bed",
    "lie_back",
    "follow",  # target: the robot (g1). Keep 1.0-1.5 m behind it until it stops at the place it is guiding to
]


class ContractError(ValueError):
    """Raised when a wz.v1 payload is not valid JSON or fails validation."""


class _Envelope(BaseModel):
    """Common base: every payload carries "schema": "wz.v1" and rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["wz.v1"] = Field(SCHEMA, alias="schema")


class Intent(_Envelope):
    id: str
    person: str
    t: float
    kind: IntentKind
    target: Optional[str] = None
    point: Optional[tuple[float, float]] = None
    speed: Optional[float] = None
    style: Optional[str] = None
    text: Optional[str] = None
    addressee: Optional[str] = None
    overlay: bool = False


class Status(_Envelope):
    person: str
    t: float
    x: float
    y: float
    yaw: float
    posture: Literal["seated", "standing", "walking", "lying", "fallen", "gone"]
    intent_id: Optional[str] = None
    intent_kind: Optional[IntentKind] = None
    progress: float
    state: Literal["running", "done", "failed", "idle"]
    reason: Optional[str] = None
    asleep: bool = False  # fell asleep (an "asleep" card, or start_asleep) and not woken yet


class Seen(_Envelope):
    kind: Literal["person", "robot", "sign", "seat"]
    id: str
    distance: float
    bearing_deg: float
    detail: Optional[str] = None


class Heard(_Envelope):
    source_kind: Literal["robot", "person", "pa", "cough", "alarm"]
    source_id: str
    text: Optional[str] = None
    received_db: float
    partial: bool
    addressed_to_me: bool
    # where the sound came from, in degrees in the listener's frame (positive = left); set only for what the robot
    # hears from a person or a cough, else None
    bearing_deg: Optional[float] = None


class SeenObject(_Envelope):
    """A world object a person can see: an item, a fire, or a lit call-bell light."""
    kind: Literal["item", "fire", "call_light"]
    id: str
    distance: float
    bearing_deg: float
    detail: str


class Observation(_Envelope):
    person: str
    t: float
    visible: list[Seen] = Field(default_factory=list)
    objects: list[SeenObject] = Field(default_factory=list)
    heard: list[Heard] = Field(default_factory=list)


class AudioEvent(_Envelope):
    id: str
    t: float
    source_kind: str
    source_id: str
    x: float
    y: float
    z: float
    loudness_db: float
    text: Optional[str] = None
    addressee: Optional[str] = None


class Say(_Envelope):
    person: str
    t: float
    text: str
    addressee: Optional[str] = None


class Nudge(_Envelope):
    person: str
    t: float
    card: str
    message: str


class ReflexOverride(_Envelope):
    person: str
    t: float
    kind: Literal["cough_bout"]
    duration_s: float
    severity: Literal["mild", "moderate", "severe"]


class WorldEvent(_Envelope):
    id: str
    t: float
    kind: Literal["pa_announcement", "alarm", "wet_floor", "arrival", "person_exit", "fire_start",
                  "route_blocked", "ring_bell"]
    text: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    persona: Optional[str] = None
    person: Optional[str] = None


class ObjectState(_Envelope):
    """One world object (item, fire or call bell) as logged on the objects stream."""
    id: str
    kind: Literal["tray", "specimen_box", "instrument", "extinguisher", "fire", "call_bell"]
    label: str = ""
    owner: Optional[str] = None
    holder: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    size: Optional[float] = None  # fire: 0..1
    active: Optional[bool] = None  # fire: burning; call bell: ringing


class RosterEntry(_Envelope):
    id: str
    persona: str
    display_name: str
    seat: Optional[str] = None
    companion_of: Optional[str] = None
    role: Optional[str] = None  # "patient_static" for a bed patient with no mind


class Roster(_Envelope):
    t: float
    people: list[RosterEntry]


class Control(_Envelope):
    action: Literal["start", "pause", "resume", "stop", "reset"]


_M = TypeVar("_M", bound=_Envelope)


def to_json(model: _Envelope) -> str:
    """Serialise a contract model to JSON text, using the "schema" key."""
    return model.model_dump_json(by_alias=True)


def from_json(cls: type[_M], text: str) -> _M:
    """Parse and validate JSON text as cls. Raises ContractError on any failure."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ContractError(f"invalid JSON for {cls.__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"{cls.__name__} payload must be a JSON object")
    if "schema" not in data:
        raise ContractError(f"{cls.__name__} payload is missing 'schema' (expected {SCHEMA!r})")
    if data["schema"] != SCHEMA:
        raise ContractError(
            f"{cls.__name__} payload has schema {data['schema']!r}, expected {SCHEMA!r}"
        )
    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        raise ContractError(f"invalid {cls.__name__}: {exc}") from exc
