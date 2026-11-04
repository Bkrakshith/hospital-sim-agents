"""Role presets: which intents each role may choose (spec 8.1)."""
from __future__ import annotations

BASIC_INTENTS: frozenset[str] = frozenset({
    "sit_idle", "use_phone", "fidget", "look_at", "read_sign", "stand_up", "sit_down", "walk_to",
    "approach", "wait_standing", "talk", "gesture", "cough", "leave",
})

_STAFF_EXTRA = {"attend", "give_item", "take_item", "call_for_help", "escort", "evacuate",
                "use_extinguisher"}
_BED = {"sit_up_in_bed", "lie_back", "call_bell"}

# A surgeon is limited to the table: no walking, sitting or leaving.
_SURGEON = {"talk", "look_at", "gesture", "wait_standing", "fidget", "cough", "request_instrument",
            "accept_item", "reject_item", "take_instrument_self"}

# The legacy roles (patient, companion, visitor, staff) keep exactly the basics, so the waiting-area
# prompts and schema are unchanged. Only the newer role ids carry extras.
_LEGACY = {r: frozenset(BASIC_INTENTS) for r in ("patient", "companion", "visitor", "staff")}

ROLE_INTENTS: dict[str, frozenset[str]] = {
    "ward_nurse": frozenset(BASIC_INTENTS | _STAFF_EXTRA),
    "receptionist": frozenset(BASIC_INTENTS | {"call_for_help"}),
    "bed_patient": frozenset(BASIC_INTENTS | _BED | {"accept_item", "refuse_item"}),
    "ward_visitor": frozenset(BASIC_INTENTS | {"escort", "call_for_help"}),
    "surgeon": frozenset(_SURGEON),
    "porter": frozenset(BASIC_INTENTS | {"take_item", "give_item", "accept_item", "refuse_item",
                                         "call_for_help"}),
    "custom": frozenset(BASIC_INTENTS),
    **_LEGACY,
}


# ---- Studio presets: roles, moods and scenarios (spec 8 and 9) ------------------------------------
import re  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal, Optional  # noqa: E402

import yaml  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # noqa: E402

from ..bench.metric_names import METRIC_NAMES  # noqa: E402
from .models import Beat  # noqa: E402


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RolePreset(_P):
    id: str
    label: str
    persona_role: str
    prompt: str
    default_rooms: list[str] = Field(default_factory=list)
    intents: list[str]
    self_confidence: float = Field(ge=0, le=1)
    trust_start: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _intents_match_persona_role(self):
        if self.persona_role not in ROLE_INTENTS:
            raise ValueError(f"persona_role '{self.persona_role}' has no intents")
        if set(self.intents) != set(ROLE_INTENTS[self.persona_role]):
            raise ValueError(f"intents differ from ROLE_INTENTS['{self.persona_role}']")
        return self


class MoodPreset(_P):
    id: str
    label: str
    prompt: str
    walk_speed_x: float = Field(gt=0)
    patience_x: float = Field(gt=0)
    anxiety_shift: float
    sociability_shift: float
    start_mood: str
    sources: list[str] = Field(default_factory=list)


class CharacterSpec(_P):
    id: str
    name: str
    sex: Literal["female", "male"]
    age_band: Literal["18-30", "31-50", "51-70", "71+"]
    role: str  # a RolePreset id
    role_prompt: Optional[str] = None  # None = use the preset text
    mood: str  # a MoodPreset id
    mood_prompt: Optional[str] = None
    room: str
    x: float
    y: float
    yaw: float = 0.0
    bed: Optional[str] = None


class Criterion(_P):
    id: str
    label: str
    metric: str
    op: Literal[">=", "<=", "=="]
    threshold: float
    weight: float = Field(gt=0)

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, v: str) -> str:
        if v not in METRIC_NAMES:
            raise ValueError(f"unknown metric '{v}'")
        return v


class RobotStart(_P):
    x: float
    y: float
    yaw: float = 0.0


class ScenarioPreset(_P):
    id: str
    title: str
    prompt: str
    focus_rooms: list[str]
    known_places: list[str] = Field(default_factory=list)
    robot_brief: dict
    robot_start: RobotStart
    characters: list[CharacterSpec]
    objects: list[dict] = Field(default_factory=list)
    timeline: list[Beat] = Field(default_factory=list)
    criteria: list[Criterion]
    ladder: dict[int, str]
    duration_min: float = Field(default=10, gt=0, le=60)
    sources: list[str] = Field(default_factory=list)

    @field_validator("ladder")
    @classmethod
    def _ladder_0_to_4(cls, v: dict[int, str]) -> dict[int, str]:
        if sorted(v) != [0, 1, 2, 3, 4]:
            raise ValueError("ladder needs exactly the stages 0, 1, 2, 3 and 4")
        return v


def _read(path: Path):
    from .loader import ConfigError
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None


def _build(model, data, path: Path):
    from .loader import ConfigError
    from pydantic import ValidationError
    try:
        return model.model_validate(data)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors())
        raise ConfigError(f"{path.name}: {msg}") from None


def load_roles(root: "str | Path") -> dict[str, RolePreset]:
    """config/presets/roles.yaml. A role's intents come from ROLE_INTENTS, the single source of truth."""
    path = Path(root) / "presets" / "roles.yaml"
    out: dict[str, RolePreset] = {}
    for entry in (_read(path) or {}).get("roles", []):
        entry = dict(entry)
        pr = entry.get("persona_role")
        entry.setdefault("intents", sorted(ROLE_INTENTS.get(pr, ())))
        r = _build(RolePreset, entry, path)
        out[r.id] = r
    return out


def load_moods(root: "str | Path") -> dict[str, MoodPreset]:
    path = Path(root) / "presets" / "moods.yaml"
    out: dict[str, MoodPreset] = {}
    for entry in (_read(path) or {}).get("moods", []):
        m = _build(MoodPreset, entry, path)
        out[m.id] = m
    return out


def load_scenario_presets(root: "str | Path") -> dict[str, ScenarioPreset]:
    """Every config/presets/scenarios/*.yaml, cross-checked against the roles, moods and the studio deck."""
    from .loader import ConfigError
    root = Path(root)
    roles, moods = load_roles(root), load_moods(root)
    deck = {c["id"] for c in (_read(root / "cards" / "studio.yaml") or {}).get("cards", [])}
    out: dict[str, ScenarioPreset] = {}
    for path in sorted((root / "presets" / "scenarios").glob("*.yaml")):
        p = _build(ScenarioPreset, _read(path), path)
        if p.id != path.stem:
            raise ConfigError(f"{path.name}: id '{p.id}' must match the file name")
        ids = [c.id for c in p.characters]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"{path.name}: duplicate character id")
        for c in p.characters:
            if c.role not in roles:
                raise ConfigError(f"{path.name}: character {c.id}: unknown role '{c.role}'")
            if c.mood not in moods:
                raise ConfigError(f"{path.name}: character {c.id}: unknown mood '{c.mood}'")
        for b in p.timeline:
            if b.card not in deck:
                raise ConfigError(f"{path.name}: beat {b.id}: card '{b.card}' is not in cards/studio.yaml")
            if b.target is not None and b.target not in ids:
                raise ConfigError(f"{path.name}: beat {b.id}: unknown target '{b.target}'")
        out[p.id] = p
    return out


# Sampling ranges per age band: (walk speed m/s, sit-to-stand s, age years)
_BAND = {"18-30": ((1.3, 1.5), (1.2, 2.0), (18, 30)), "31-50": ((1.2, 1.4), (1.5, 2.5), (31, 50)),
         "51-70": ((1.0, 1.2), (2.0, 3.5), (51, 70)), "71+": ((0.7, 0.9), (3.0, 5.0), (71, 88))}
_ROLE_BASE = {  # patience_min, anxiety, sociability
    "ward_nurse": ((10, 25), (0.1, 0.3), (0.3, 0.6)), "receptionist": ((15, 30), (0.1, 0.3), (0.5, 0.8)),
    "patient": ((30, 60), (0.3, 0.5), (0.3, 0.7)), "visitor": ((20, 45), (0.3, 0.6), (0.3, 0.7)),
    "surgeon": ((10, 20), (0.1, 0.3), (0.2, 0.4)), "porter": ((15, 30), (0.1, 0.3), (0.3, 0.6)),
    "custom": ((20, 40), (0.2, 0.5), (0.3, 0.7))}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _persona_dict(c: CharacterSpec, role: RolePreset, mood: MoodPreset) -> dict:
    walk, sit, age = _BAND[c.age_band]
    pat, anx, soc = _ROLE_BASE.get(role.id, _ROLE_BASE["custom"])
    role_text = (c.role_prompt if c.role_prompt is not None else role.prompt).strip()
    mood_text = (c.mood_prompt if c.mood_prompt is not None else mood.prompt).strip()
    aid = "stick" if c.age_band == "71+" and c.bed is None else "none"
    return {
        "id": c.id, "label": f"{role.label} ({c.name})", "role": role.persona_role, "age": list(age),
        "sex": [c.sex], "condition": f"{role.label}, {mood.label.lower()} today.",
        "symptoms": {"cough_per_hour": [0, 0], "cough_severity": "mild", "breathless": False, "fever": False},
        "mobility": {"aid": aid, "walk_speed": [round(v * mood.walk_speed_x, 3) for v in walk],
                     "sit_to_stand_s": list(sit)},
        "patience_min": [round(v * mood.patience_x, 3) for v in pat],
        "anxiety": [round(_clamp(v + mood.anxiety_shift), 3) for v in anx],
        "sociability": [round(_clamp(v + mood.sociability_shift), 3) for v in soc],
        "phone_use": [0.2, 0.6], "attitude_to_robots": "neutral",
        "goals": [ln.rsplit("[", 1)[0].strip() for ln in role_text.splitlines() if ln.strip()][:3] or ["Go about your day."],
        "backstory": " ".join(x for x in (role_text.replace("\n", " "), mood_text.replace("\n", " ")) if x) or "A person.",
        "sources": list(role.sources) + list(mood.sources) or ["none"],
    }


def preset_to_scenario(preset: ScenarioPreset, root: "str | Path", dest: "str | Path | None" = None):
    """Stage a loadable scenario folder for a preset and load it. Minimal: Studio compiles richer ones."""
    from .loader import load_scenario
    root = Path(root)
    dest = Path(dest) if dest is not None else Path(tempfile.mkdtemp(prefix=f"preset_{preset.id}_"))
    roles, moods = load_roles(root), load_moods(root)
    for sub in ("cards", "storyboards", "scenarios"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    if not (dest / "site").exists():
        (dest / "site").symlink_to((root / "site").resolve())
    (dest / "signs.yaml").write_text((root / "signs.yaml").read_text())
    (dest / "cards" / "studio.yaml").write_text((root / "cards" / "studio.yaml").read_text())
    from ..site import load_site
    site = load_site(root)
    personas, cast = {}, []
    for c in preset.characters:
        personas[c.id] = _persona_dict(c, roles[c.role], moods[c.mood])
        entry: dict = {"id": c.id, "persona": c.id, "name": c.name}
        seat = next((s.id for s in site.seats.values() if abs(s.x - c.x) < 0.05 and abs(s.y - c.y) < 0.05), None)
        if c.bed:
            entry["start"] = {"bed": c.bed}
        elif seat:
            entry["seat"] = seat
        else:
            entry["start"] = {"x": c.x, "y": c.y, "yaw": c.yaw}
        cast.append(entry)
    (dest / "personas.yaml").write_text(yaml.safe_dump({"personas": personas}, sort_keys=False))
    (dest / "storyboards" / f"{preset.id}.yaml").write_text(yaml.safe_dump(
        {"id": preset.id, "beats": [b.model_dump(mode="json", exclude_none=True, exclude_defaults=False)
                                    for b in preset.timeline]}, sort_keys=False))
    scenario = {
        "id": preset.id, "known_places": preset.known_places, "duration_min": preset.duration_min, "seed": 7,
        "provider": "claude", "storyboard": f"../storyboards/{preset.id}.yaml", "deck": "../cards/studio.yaml",
        "robot": {"x": preset.robot_start.x, "y": preset.robot_start.y, "yaw": preset.robot_start.yaw},
        "robot_brief": preset.robot_brief, "objects": preset.objects, "cast": cast}
    path = dest / "scenarios" / f"{preset.id}.yaml"
    path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return load_scenario(path)
