"""Load a scenario file and its sibling configs, cross-check them and sample the cast."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from .models import ModelRoles, Persona, PersonInstance, Scenario, Sign, WaitingArea
from .sampling import sample_instance


class ConfigError(Exception):
    """A config file is invalid. Messages read '<field path>: <problem>'."""


@dataclass
class LoadedScenario:
    scenario: Scenario
    personas: dict[str, Persona]
    area: WaitingArea
    signs: list[Sign]
    storyboard: Optional[Any]  # Storyboard model arrives in Task 11
    deck: Optional[dict]  # dict[str, CardDef] arrives in Task 11
    instances: list[PersonInstance]
    root: Path = field(default_factory=Path)
    scenario_path: Path = field(default_factory=Path)


def default_models(provider: str) -> ModelRoles:
    if provider == "claude":
        return ModelRoles(decide="claude-opus-5-5", summarise="claude-haiku-5-5", vision="claude-opus-5-5")
    raise ConfigError("models: OpenAI needs model names for decide, summarise and vision")


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None


def _validate(model, data, path: Path):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"{loc}: {err['msg']}")
        raise ConfigError(f"{path.name}: " + "; ".join(lines)) from None


def _find_root(scenario_path: Path) -> Path:
    for parent in scenario_path.parents:
        if (parent / "personas.yaml").exists():
            return parent
    raise ConfigError(f"{scenario_path}: no personas.yaml found in any parent directory")


def load_scenario(path: "str | Path") -> LoadedScenario:
    path = Path(path).resolve()
    root = _find_root(path)
    scenario = _validate(Scenario, _read_yaml(path), path)

    raw_personas = (_read_yaml(root / "personas.yaml") or {}).get("personas", {})
    personas = {pid: _validate(Persona, p, root / "personas.yaml") for pid, p in raw_personas.items()}
    area = _validate(WaitingArea, _read_yaml(root / f"{scenario.area}.yaml"), root / f"{scenario.area}.yaml")
    raw_signs = (_read_yaml(root / "signs.yaml") or {}).get("signs", [])
    signs = [_validate(Sign, s, root / "signs.yaml") for s in raw_signs]

    if scenario.models is None:
        scenario = scenario.model_copy(update={"models": default_models(scenario.provider)})

    _cross_check(scenario, personas, area)

    # Task 11 adds the Storyboard and CardDef models; until then the loader leaves them None.
    storyboard = None
    deck = None

    instances = [sample_instance(personas[c.persona], c, scenario.seed, i) for i, c in enumerate(scenario.cast)]
    return LoadedScenario(scenario, personas, area, signs, storyboard, deck, instances, root, path)


def _cross_check(scenario: Scenario, personas: dict[str, Persona], area: WaitingArea) -> None:
    seat_ids = {s.id for s in area.seats}
    person_ids = [c.id for c in scenario.cast]
    taken: dict[str, str] = {}
    for i, c in enumerate(scenario.cast):
        if person_ids.index(c.id) != i:
            raise ConfigError(f"cast[{i}].id: duplicate person id '{c.id}'")
        if c.persona not in personas:
            raise ConfigError(f"cast[{i}].persona: unknown persona '{c.persona}'")
        if c.seat is not None:
            if c.seat not in seat_ids:
                raise ConfigError(f"cast[{i}].seat: unknown seat '{c.seat}'")
            if c.seat in taken:
                raise ConfigError(f"cast[{i}].seat: seat '{c.seat}' already taken by {taken[c.seat]}")
            taken[c.seat] = c.id
        if c.companion_of is not None and c.companion_of not in person_ids:
            raise ConfigError(f"cast[{i}].companion_of: unknown person '{c.companion_of}'")
        if c.entrance is not None and c.entrance not in area.landmarks:
            raise ConfigError(f"cast[{i}].entrance: unknown landmark '{c.entrance}'")
    for name in [area.entrance, area.nurse_call_point, *area.exits]:
        if name not in area.landmarks:
            raise ConfigError(f"{scenario.area}: unknown landmark '{name}'")
