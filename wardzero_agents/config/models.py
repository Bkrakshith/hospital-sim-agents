"""Pydantic v2 schemas for personas, scenarios, the waiting area and sampled person instances."""
from __future__ import annotations

import re
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


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
    role: Literal["patient", "companion", "visitor", "staff", "ward_nurse", "receptionist", "surgeon", "porter", "custom", "bed_patient", "ward_visitor"]
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


class StartPose(_Model):
    """A non-seated start. With bed set the person starts lying in that bed (x, y, yaw are the bed's)."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    bed: Optional[str] = None


class CastEntry(_Model):
    id: str
    persona: str
    seat: Optional[str] = None
    start: Optional[StartPose] = None
    arrive_at_s: float = Field(default=0, ge=0)
    entrance: Optional[str] = None
    companion_of: Optional[str] = None
    seed: Optional[int] = None
    name: Optional[str] = None  # overrides the sampled display name


class ModelRoles(_Model):
    decide: str
    summarise: str
    vision: str
    robot: str = "claude-sonnet-5-5"  # the built-in Claude robot
    report: str = "claude-sonnet-5-5"  # the after-run report narrative
    effort_decide: Literal["low", "medium", "high"] = "low"


class Chaos(_Model):
    enabled: bool = False
    rate_per_hour: float = 6.0


class RobotConfig(_Model):
    enabled: bool = True
    profile: str = "g1_claude"
    ns: str = "robot"
    x: float = 1.05
    y: float = 1.62
    yaw: float = -1.5708
    reach: tuple[float, float] = (0.6, 1.3)  # [zmin, zmax] the hands can put things at, metres
    radius: float = Field(default=0.3, gt=0)
    max_speed: float = Field(default=1.0, gt=0)
    fov_deg: float = Field(default=87.0, gt=0, le=360)
    range_m: float = Field(default=6.0, gt=0)


class Scenario(_Model):
    id: str
    site: str = "hospital"
    area: Optional[str] = None
    static_patients: bool = True
    known_places: list[str] = Field(default_factory=list)
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
    robot_brief: Optional[dict] = None  # the job told to the robot; published latched on task/brief
    objects: list[dict] = Field(default_factory=list)  # items and call bells present at the start
    preset_id: Optional[str] = None  # the Studio scenario preset this scenario was built from
    campaign: bool = False  # trust carries across runs and word of mouth is rolled (Studio sets it)
    ladder: dict[int, str] = Field(default_factory=dict)  # what each trust stage looks like (0..4)
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
    role: Literal["patient", "companion", "visitor", "staff", "ward_nurse", "receptionist", "surgeon", "porter", "custom", "bed_patient", "ward_visitor"]
    seat: Optional[str]
    companion_of: Optional[str]
    arrive_at_s: float
    entrance: Optional[str]
    start: Optional[StartPose] = None


AT_RE = re.compile(r"^t\+\d{2}:\d{2}$")


class Condition(_Model):
    kind: Literal["robot_within", "waited_longer_than", "posture_is"]
    person: str
    value: float | str


class Beat(_Model):
    id: str
    at: Optional[str] = None
    when: Optional[Condition] = None
    after: Optional[str] = None
    card: str
    target: Optional[str] = None
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_trigger(self):
        if sum(x is not None for x in (self.at, self.when, self.after)) != 1:
            raise ValueError("exactly one of at, when, after")
        if self.at is not None and not AT_RE.match(self.at):
            raise ValueError(f"at '{self.at}' must look like t+MM:SS")
        return self


class Storyboard(_Model):
    id: str
    beats: list[Beat]


class CardDef(_Model):
    id: str
    label: str
    channels: list[Literal["nudge", "reflex", "world"]]
    target_kind: Literal["person", "none", "new", "all"]
    nudge: Optional[str] = None
    companion_nudge: Optional[str] = None
    reflex: Optional[dict] = None
    world: Optional[dict] = None
    tests: str
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _channels_have_content(self):
        if "nudge" in self.channels and self.nudge is None and self.companion_nudge is None:
            raise ValueError("channel 'nudge' needs a nudge or companion_nudge")
        if "reflex" in self.channels and not self.reflex:
            raise ValueError("channel 'reflex' needs a reflex")
        if "world" in self.channels and not (self.world and "kind" in self.world):
            raise ValueError("channel 'world' needs a world with a kind")
        return self
