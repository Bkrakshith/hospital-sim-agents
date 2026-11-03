"""Pydantic v2 schemas for personas, scenarios, the waiting area and sampled person instances."""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _ordered(r: tuple[float, float]) -> tuple[float, float]:
    if r[0] > r[1]:
        raise ValueError(f"range {list(r)} must have low <= high")
    return r


Range = Annotated[tuple[float, float], AfterValidator(_ordered)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Symptoms(_Model):
    cough_per_hour: Range = (0, 0)
    cough_severity: Literal["mild", "moderate", "severe"] = "mild"
    breathless: bool = False
    fever: bool = False


class Mobility(_Model):
    aid: Literal["none", "stick", "frame", "wheelchair"] = "none"
    walk_speed: Range
    sit_to_stand_s: Range


class Persona(_Model):
    id: str
    label: str
    role: Literal["patient", "companion", "visitor", "staff"]
    age: Range
    sex: list[str]
    condition: str
    symptoms: Symptoms
    mobility: Mobility
    hearing_loss_db: Range = (0, 0)
    english_fluency: Literal["native", "fluent", "limited"] = "native"
    patience_min: Range
    anxiety: Range
    sociability: Range
    phone_use: Range
    attitude_to_robots: Literal["curious", "neutral", "wary"]
    goals: list[str]
    backstory: str
    seed_memories: list[str] = Field(default_factory=list)
    sources: list[str]


class CastEntry(_Model):
    id: str
    persona: str
    seat: Optional[str] = None
    arrive_at_s: float = Field(default=0, ge=0)
    entrance: Optional[str] = None
    companion_of: Optional[str] = None
    seed: Optional[int] = None


class ModelRoles(_Model):
    decide: str
    summarise: str
    vision: str
    effort_decide: Literal["low", "medium", "high"] = "low"


class Chaos(_Model):
    enabled: bool = False
    rate_per_hour: float = 6.0


class RobotConfig(_Model):
    enabled: bool = True
    x: float = 1.05
    y: float = 1.62
    yaw: float = -1.5708


class Scenario(_Model):
    id: str
    area: str
    duration_min: float = Field(gt=0, le=60)
    seed: int
    perception: Literal["oracle", "full"] = "oracle"
    provider: Literal["claude", "openai"]
    models: Optional[ModelRoles] = None
    budget_usd: float = 20.0
    storyboard: str
    deck: str
    chaos: Chaos = Field(default_factory=Chaos)
    memory_snapshot: Optional[str] = None
    lockstep: bool = False
    robot: RobotConfig = Field(default_factory=RobotConfig)
    cast: list[CastEntry]


class Seat(_Model):
    id: str
    x: float
    y: float
    yaw: float
    row: str


class Landmark(_Model):
    x: float
    y: float
    label: str


class WaitingArea(_Model):
    seats: list[Seat]
    landmarks: dict[str, Landmark]
    entrance: str
    exits: list[str]
    nurse_call_point: str


class Sign(_Model):
    id: str
    text: str
    x: float
    y: float
    z: float
    facing_yaw: float
    read_distance_m: float = 6.0
    placeholder: bool = True


class PersonInstance(_Model):
    person_id: str
    persona_id: str
    seed: int
    memory_key: str
    display_name: str
    age: int
    sex: str
    cough_per_hour: float
    cough_severity: Literal["mild", "moderate", "severe"]
    breathless: bool
    walk_speed: float
    sit_to_stand_s: float
    hearing_loss_db: float
    patience_min: float
    anxiety: float
    sociability: float
    phone_use: float
    attitude_to_robots: Literal["curious", "neutral", "wary"]
    english_fluency: Literal["native", "fluent", "limited"]
    mobility_aid: Literal["none", "stick", "frame", "wheelchair"]
    goals: list[str]
    backstory: str
    condition: str
    role: Literal["patient", "companion", "visitor", "staff"]
    seat: Optional[str]
    companion_of: Optional[str]
    arrive_at_s: float
    entrance: Optional[str]


