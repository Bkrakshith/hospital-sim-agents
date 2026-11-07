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
    "bed_patient": frozenset(BASIC_INTENTS | _BED | {"accept_item", "refuse_item", "follow"}),
    "ward_visitor": frozenset(BASIC_INTENTS | {"escort", "call_for_help", "follow"}),
    "ae_patient": frozenset(BASIC_INTENTS | {"follow", "accept_item", "refuse_item", "call_for_help"}),
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
    mobility_aid: Optional[Literal["none", "stick", "frame", "wheelchair"]] = None  # None: by age (a stick at 71+)

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
    # The same mood for a member of staff at work (None: staff get the prompt above). The public text tells a person
    # to ignore a robot asking for anything; staff must still answer the robot when it calls them to a patient.
    staff_prompt: Optional[str] = None


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
    start_seated: bool = False  # on a free A&E waiting seat at the start (the Studio picks the seat)
    start_asleep: bool = False  # a patient asleep from the start (the cast entry's start_asleep, when the sim has it)
    auto: bool = False  # added by the Studio itself (the receptionist in a reception scene); the user may remove it


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
    # What the staff in the scene are told about the robot's job, written to them (never shown to the robot). The
    # robot's own job text is written to the robot and must not be quoted to staff as if it were theirs.
    staff_line: Optional[str] = None
    robot_start: RobotStart
    characters: list[CharacterSpec]
    objects: list[dict] = Field(default_factory=list)
    timeline: list[Beat] = Field(default_factory=list)
    criteria: list[Criterion]
    ladder: dict[int, str]
    duration_min: float = Field(default=10, gt=0, le=60)
    arrival_entrance: Optional[str] = None  # a landmark: where the arrival card brings people in (None: main entrance)
    sources: list[str] = Field(default_factory=list)
    # False: shown last in the Studio, and hidden when WARDZERO_STUDIO_PUBLIC_CAPTURE=1 (demo films and screenshots)
    listed_in_public_capture: bool = True

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
    "patient": ((30, 60), (0.3, 0.5), (0.3, 0.7)), "ae_patient": ((20, 45), (0.4, 0.7), (0.3, 0.7)), "visitor": ((20, 45), (0.3, 0.6), (0.3, 0.7)),
    "surgeon": ((10, 20), (0.1, 0.3), (0.2, 0.4)), "porter": ((15, 30), (0.1, 0.3), (0.3, 0.6)),
    "wheelchair_patient": ((30, 60), (0.3, 0.5), (0.3, 0.7)),
    "custom": ((20, 40), (0.2, 0.5), (0.3, 0.7))}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


STAFF_PRESET_ROLES = frozenset({"ward_nurse", "receptionist", "porter", "surgeon"})


def robot_duty_line(staff_line: Optional[str], robot_role: str = "") -> str:
    """What a member of staff is told about the robot's job, so the robot does the scenario's main tasks, not them.
    staff_line is written to staff (the preset's staff_line); the robot's own job text is written to the robot and is
    never quoted to them. One line, with no '[' but the source tag (the goal line is cut at its last '[')."""
    what = " ".join((staff_line or "").split()).replace("[", "(").replace("]", ")")
    if not what:
        what = f"The robot on this shift ({robot_role}) has a job of its own." if robot_role else \
            "The robot on this shift has a job of its own."
    return (f"{what} Leave that work to the robot and get on with your own; your habit of doing things yourself is for "
            f"your own work, not the robot's. Step in only when the robot calls you or asks you to, or when someone is "
            f"in danger and the robot cannot help them. [scenario brief]")


def condition_line(role: RolePreset, mood: MoodPreset) -> str:
    """Why a Studio character is here, in words: the role and the mood, leaving out a "Custom" one (its text is the
    character's own prompt, and the word itself means nothing to the person)."""
    who = role.label if role.id != "custom" else ""
    how = f"{mood.label.lower()} today" if mood.id != "custom" else ""
    if who and how:
        return f"{who}, {how}."
    if who or how:
        return f"{(who or how)[0].upper()}{(who or how)[1:]}."
    return "Here today; see your background."


def _persona_dict(c: CharacterSpec, role: RolePreset, mood: MoodPreset, staff_line: Optional[str] = None,
                  robot_role: Optional[str] = None) -> dict:
    """The persona for a Studio character. robot_role is None when the scenario has no robot brief (then staff get no
    duty line); staff_line is what staff are told about the robot's job."""
    walk, sit, age = _BAND[c.age_band]
    pat, anx, soc = _ROLE_BASE.get(role.id, _ROLE_BASE["custom"])
    is_staff = role.id in STAFF_PRESET_ROLES
    role_text = (c.role_prompt if c.role_prompt is not None else role.prompt).strip()
    if is_staff and (staff_line or robot_role is not None):
        role_text = robot_duty_line(staff_line, robot_role or "") + "\n" + role_text  # first: among the goals
    if c.mood_prompt is not None:
        mood_text = c.mood_prompt.strip()
    else:  # staff at work get the staff variant of a mood written for the public
        mood_text = ((mood.staff_prompt if is_staff and mood.staff_prompt else None) or mood.prompt).strip()
    aid = role.mobility_aid or ("stick" if c.age_band == "71+" and c.bed is None else "none")
    label = role.label if role.id != "custom" else "Character"
    return {
        "id": c.id, "label": f"{label} ({c.name})", "role": role.persona_role, "age": list(age),
        "sex": [c.sex], "condition": condition_line(role, mood),
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


def preset_to_scenario(preset: ScenarioPreset, root: "str | Path", dest: "str | Path | None" = None, *,
                       characters: "list[CharacterSpec] | None" = None, scenario_extra: "dict | None" = None,
                       cast_extra: "dict[str, dict] | None" = None, robot_extra: "dict | None" = None,
                       flow=None):
    """Stage a loadable scenario folder for a preset and load it. The Studio compiler passes its own
    characters (default: the preset's), top-level scenario fields (scenario_extra), per-cast-entry fields
    such as memory_key (cast_extra, by character id) and robot block fields (robot_extra). flow: the scenario flow
    to stage (default: the preset's own, config/presets/flows/<id>.yaml), pruned to the cast."""
    from .loader import load_scenario
    from .models import CastEntry
    root = Path(root)
    dest = Path(dest) if dest is not None else Path(tempfile.mkdtemp(prefix=f"preset_{preset.id}_"))
    roles, moods = load_roles(root), load_moods(root)
    for sub in ("cards", "storyboards", "scenarios", "flows"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    if not (dest / "site").exists():
        try:
            (dest / "site").symlink_to((root / "site").resolve())
        except OSError:  # a filesystem without symlinks (exFAT, some Windows setups): a copy
            import shutil
            shutil.copytree((root / "site").resolve(), dest / "site")
    (dest / "signs.yaml").write_text((root / "signs.yaml").read_text())
    (dest / "cards" / "studio.yaml").write_text((root / "cards" / "studio.yaml").read_text())
    from ..site import load_site
    site = load_site(root)
    personas, cast = {}, []
    for c in (preset.characters if characters is None else characters):
        brief = preset.robot_brief
        personas[c.id] = _persona_dict(c, roles[c.role], moods[c.mood], preset.staff_line,
                                       None if brief is None else str(brief.get("role") or ""))
        entry: dict = {"id": c.id, "persona": c.id, "name": c.name}
        seat = next((s.id for s in site.seats.values() if abs(s.x - c.x) < 0.05 and abs(s.y - c.y) < 0.05), None)
        if c.bed:
            entry["start"] = {"bed": c.bed}
        elif seat:
            entry["seat"] = seat
        else:
            entry["start"] = {"x": c.x, "y": c.y, "yaw": c.yaw}
        if c.start_asleep and "start_asleep" in CastEntry.model_fields:  # a sim without it uses the asleep card
            entry["start_asleep"] = True
        entry.update((cast_extra or {}).get(c.id, {}))
        cast.append(entry)
    (dest / "personas.yaml").write_text(yaml.safe_dump({"personas": personas}, sort_keys=False))
    (dest / "storyboards" / f"{preset.id}.yaml").write_text(yaml.safe_dump(
        {"id": preset.id, "beats": [b.model_dump(mode="json", exclude_none=True, exclude_defaults=False)
                                    for b in preset.timeline]}, sort_keys=False))
    scenario = {
        "id": preset.id, "preset_id": preset.id, "known_places": preset.known_places, "duration_min": preset.duration_min, "seed": 7,
        "provider": "claude", "storyboard": f"../storyboards/{preset.id}.yaml", "deck": "../cards/studio.yaml",
        "robot": {"x": preset.robot_start.x, "y": preset.robot_start.y, "yaw": preset.robot_start.yaw,
                  **(robot_extra or {})},
        "robot_brief": preset.robot_brief, "objects": preset.objects,
        "ladder": {int(k): v for k, v in preset.ladder.items()}, "cast": cast}
    from ..flows.spec import dump_flow, preset_flow, prune_flow
    if flow is None:
        flow = preset_flow(root, preset.id)
    if flow is not None:
        flow = prune_flow(flow, [e["id"] for e in cast])
        (dest / "flows" / f"{preset.id}.yaml").write_text(dump_flow(flow))
        scenario["flow"] = f"../flows/{preset.id}.yaml"
    if preset.arrival_entrance is not None:
        scenario["arrival_entrance"] = preset.arrival_entrance
    scenario.update(scenario_extra or {})
    path = dest / "scenarios" / f"{preset.id}.yaml"
    path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return load_scenario(path)


DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"


def preset_for_run(meta: dict, root: "str | Path | None" = None) -> Optional[ScenarioPreset]:
    """The scenario preset a run was made from: meta["preset_id"], else the scenario id when a preset
    has that id. None for a legacy scenario (or when the presets cannot be loaded)."""
    try:
        presets = load_scenario_presets(root or DEFAULT_CONFIG_ROOT)
    except Exception:
        return None
    for key in ("preset_id", "scenario_id"):
        pid = meta.get(key)
        if isinstance(pid, str) and pid in presets:
            return presets[pid]
    return None


def people_meta(loaded, preset: Optional[ScenarioPreset] = None) -> dict:
    """meta["people"]: for each sampled person, {role: persona role, preset_role: studio role id or None,
    static: False}. Static patients are made by the world, not sampled, and are named by id."""
    if preset is None:
        pid = getattr(loaded.scenario, "preset_id", None)
        preset = preset_for_run({"preset_id": pid}) if pid else None
    studio = {c.id: c.role for c in preset.characters} if preset else {}
    return {i.person_id: {"role": i.role, "preset_role": studio.get(i.person_id), "static": False}
            for i in loaded.instances}
