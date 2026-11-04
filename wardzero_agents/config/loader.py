"""Load a scenario file and its sibling configs, cross-check them and sample the cast."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from .models import (AT_RE, CardDef, Landmark, ModelRoles, Persona, PersonInstance, Scenario, Seat, Sign, Storyboard,
                     WaitingArea)
from .sampling import sample_instance


class ConfigError(Exception):
    """A config file is invalid. Messages read '<field path>: <problem>'."""


@dataclass
class LoadedScenario:
    scenario: Scenario
    personas: dict[str, Persona]
    area: WaitingArea
    signs: list[Sign]
    storyboard: Storyboard
    deck: dict[str, CardDef]
    instances: list[PersonInstance]
    site: Any = None  # wardzero_agents.site.Site
    root: Path = field(default_factory=Path)
    scenario_path: Path = field(default_factory=Path)


def default_models(provider: str) -> ModelRoles:
    if provider == "claude":
        return ModelRoles(decide="claude-sonnet-5-5", summarise="claude-haiku-5-5", vision="claude-sonnet-5-5")
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
    site, area = _load_site_and_area(root, scenario)
    raw_signs = (_read_yaml(root / "signs.yaml") or {}).get("signs", [])
    signs = [_validate(Sign, s, root / "signs.yaml") for s in raw_signs]

    if scenario.models is None:
        scenario = scenario.model_copy(update={"models": default_models(scenario.provider)})

    _cross_check(scenario, personas, area, site)

    deck = _load_deck(path.parent / scenario.deck)
    storyboard = _load_storyboard(path.parent / scenario.storyboard, deck, scenario)

    instances = [sample_instance(personas[c.persona], c, scenario.seed, i) for i, c in enumerate(scenario.cast)]
    return LoadedScenario(scenario, personas, area, signs, storyboard, deck, instances, site, root, path)


def _load_site_and_area(root: Path, scenario: Scenario):
    """The site, with the scenario's optional area merged in, and the WaitingArea view of both.

    The area's seats and landmarks are added to the site (the area wins on a clash). With no area
    file the WaitingArea is built from the site's own seats and landmarks."""
    from ..site import load_site
    if scenario.site != "hospital":
        raise ConfigError(f"site: unknown site '{scenario.site}' (only 'hospital' exists)")
    try:
        site = load_site(root)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        raise ConfigError(f"site: cannot load the site map: {e}") from None
    if scenario.area is not None:
        apath = root / f"{scenario.area}.yaml"
        area = _validate(WaitingArea, _read_yaml(apath), apath)
        from ..site import Landmark as SLandmark, Seat as SSeat
        for lid, lm in area.landmarks.items():
            site.landmarks[lid] = SLandmark(lid, lm.x, lm.y, lm.label)
        for st in area.seats:
            site.seats[st.id] = SSeat(st.id, st.x, st.y, st.yaw)
        return site, area
    area = WaitingArea(
        seats=[Seat(id=q.id, x=q.x, y=q.y, yaw=q.yaw, row=q.id[:1]) for q in site.seats.values()],
        landmarks={k: Landmark(x=m.x, y=m.y, label=m.label or k) for k, m in site.landmarks.items()},
        entrance="entrance", exits=["entrance"], nurse_call_point="nurse_call_point")
    return site, area


def _cross_check(scenario: Scenario, personas: dict[str, Persona], area: WaitingArea, site=None) -> None:
    seat_ids = {s.id for s in area.seats}
    occupied_beds: dict[str, str] = {}
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
        if c.start is not None:
            if c.seat is not None:
                raise ConfigError(f"cast[{i}].start: a person with a seat cannot also have a start")
            if c.start.bed is not None:
                bed = site.beds.get(c.start.bed) if site is not None else None
                if bed is None:
                    raise ConfigError(f"cast[{i}].start.bed: unknown bed '{c.start.bed}'")
                if c.start.bed in occupied_beds:
                    raise ConfigError(f"cast[{i}].start.bed: bed '{c.start.bed}' already has a patient "
                                      f"({occupied_beds[c.start.bed]})")
                occupied_beds[c.start.bed] = c.id
            elif site is not None:
                why = site.check_place(c.start.x, c.start.y, occupied_beds=tuple(occupied_beds))
                if why is not None:
                    raise ConfigError(f"cast[{i}].start: ({c.start.x}, {c.start.y}) is {why}")
        if c.companion_of is not None and c.companion_of not in person_ids:
            raise ConfigError(f"cast[{i}].companion_of: unknown person '{c.companion_of}'")
        if c.entrance is not None and c.entrance not in area.landmarks:
            raise ConfigError(f"cast[{i}].entrance: unknown landmark '{c.entrance}'")
    for name in [area.entrance, area.nurse_call_point, *area.exits]:
        if name not in area.landmarks:
            raise ConfigError(f"{scenario.area or 'site'}: unknown landmark '{name}'")
    for k, name in enumerate(scenario.known_places):
        if site is not None and site.resolve(name) is None:
            raise ConfigError(f"known_places[{k}]: unknown place '{name}'")


def _load_deck(path: Path) -> dict[str, CardDef]:
    """Load and check the card deck. A missing file is an error."""
    if not path.exists():
        raise ConfigError(f"deck: file not found: {path}")
    raw = (_read_yaml(path) or {}).get("cards", [])
    deck: dict[str, CardDef] = {}
    for i, entry in enumerate(raw):
        card = _validate(CardDef, entry, path)
        if card.id in deck:
            raise ConfigError(f"deck.cards[{i}].id: duplicate card id '{card.id}'")
        _check_card(card)
        deck[card.id] = card
    return deck


def _load_storyboard(path: Path, deck: dict[str, CardDef], scenario: Scenario) -> Storyboard:
    """Load and cross-check the storyboard. A missing file is an error."""
    if not path.exists():
        raise ConfigError(f"storyboard: file not found: {path}")
    raw = _read_yaml(path)
    for i, b in enumerate((raw or {}).get("beats", []) or []):
        if isinstance(b, dict) and sum(b.get(k) is not None for k in ("at", "when", "after")) != 1:
            raise ConfigError(f"storyboard.beats[{i}]: exactly one of at, when, after")
        if isinstance(b, dict) and b.get("at") is not None and not AT_RE.match(str(b["at"])):
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
        if b.card not in deck:
            raise ConfigError(f"storyboard.beats[{i}].card: unknown card '{b.card}'")
        if deck[b.card].target_kind == "person" and b.target is None:
            raise ConfigError(f"storyboard.beats[{i}].target: card '{b.card}' needs a target person")
        if b.when is not None:
            v, kind = b.when.value, b.when.kind
            if kind == "posture_is":
                if v not in POSTURES:
                    raise ConfigError(f"storyboard.beats[{i}].when.value: '{v}' is not a posture")
            elif isinstance(v, (str, bool)):
                raise ConfigError(f"storyboard.beats[{i}].when.value: {kind} needs a number")
        if b.after is not None and b.after not in seen:
            raise ConfigError(f"storyboard.beats[{i}].after: '{b.after}' must name an earlier beat")
        seen[b.id] = i
    return sb


POSTURES = ("seated", "standing", "walking", "lying", "fallen", "gone")
_KEY_RE = re.compile(r"\{([^{}]*)\}")


def _check_card(card: CardDef) -> None:
    if card.reflex is not None:
        d, sev = card.reflex.get("duration_s"), card.reflex.get("severity")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
            raise ConfigError(f"deck.{card.id}.reflex: duration_s must be a number above 0")
        if sev not in ("mild", "moderate", "severe"):
            raise ConfigError(f"deck.{card.id}.reflex: severity must be mild, moderate or severe")
    for text in (card.nudge, card.companion_nudge, (card.world or {}).get("text")):
        for key in _KEY_RE.findall(text or ""):
            if key not in ("name", "companion") and not (key.startswith("params.") and len(key) > 7):
                raise ConfigError(f"deck.{card.id}: unknown template key '{key}'")
