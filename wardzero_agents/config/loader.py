"""Load a scenario file and its sibling configs, cross-check them and sample the cast."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from .models import CardDef, ModelRoles, Persona, PersonInstance, Scenario, Sign, Storyboard, WaitingArea
from .sampling import sample_instance


class ConfigError(Exception):
    """A config file is invalid. Messages read '<field path>: <problem>'."""


@dataclass
class LoadedScenario:
    scenario: Scenario
    personas: dict[str, Persona]
    area: WaitingArea
    signs: list[Sign]
    storyboard: Optional[Storyboard]  # None when the scenario's storyboard file is missing
    deck: Optional[dict[str, CardDef]]  # None when the scenario's deck file is missing
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

    deck = _load_deck(path.parent / scenario.deck)
    storyboard = _load_storyboard(path.parent / scenario.storyboard, deck, scenario)

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


def _load_deck(path: Path) -> Optional[dict[str, CardDef]]:
    """Load the card deck. A missing file gives None so partial configs still load."""
    if not path.exists():
        return None
    raw = (_read_yaml(path) or {}).get("cards", [])
    deck: dict[str, CardDef] = {}
    for i, entry in enumerate(raw):
        card = _validate(CardDef, entry, path)
        if card.id in deck:
            raise ConfigError(f"deck.cards[{i}].id: duplicate card id '{card.id}'")
        deck[card.id] = card
    return deck


def _load_storyboard(path: Path, deck: Optional[dict[str, CardDef]], scenario: Scenario) -> Optional[Storyboard]:
    """Load and cross-check the storyboard. A missing file gives None."""
    if not path.exists():
        return None
    raw = _read_yaml(path)
    for i, b in enumerate((raw or {}).get("beats", []) or []):
        if isinstance(b, dict) and sum(b.get(k) is not None for k in ("at", "when", "after")) != 1:
            raise ConfigError(f"storyboard.beats[{i}]: exactly one of at, when, after")
        if isinstance(b, dict) and b.get("at") is not None and not re.match(r"^t\+\d{2}:\d{2}$", str(b["at"])):
            raise ConfigError(f"storyboard.beats[{i}].at: '{b['at']}' must look like t+MM:SS")
    sb = _validate(Storyboard, raw, path)
    people = {c.id for c in scenario.cast}
    seen: dict[str, int] = {}
    for i, b in enumerate(sb.beats):
        if b.id in seen:
            raise ConfigError(f"storyboard.beats[{i}].id: duplicate beat id '{b.id}'")
        if b.target is not None and b.target not in people:
            raise ConfigError(f"storyboard.beats[{i}].target: unknown person '{b.target}'")
        if b.when is not None and b.when.person not in people:
            raise ConfigError(f"storyboard.beats[{i}].when.person: unknown person '{b.when.person}'")
        if deck is not None:
            if b.card not in deck:
                raise ConfigError(f"storyboard.beats[{i}].card: unknown card '{b.card}'")
            if deck[b.card].target_kind == "person" and b.target is None:
                raise ConfigError(f"storyboard.beats[{i}].target: card '{b.card}' needs a target person")
        if b.after is not None and b.after not in seen:
            raise ConfigError(f"storyboard.beats[{i}].after: '{b.after}' must name an earlier beat")
        seen[b.id] = i
    return sb
