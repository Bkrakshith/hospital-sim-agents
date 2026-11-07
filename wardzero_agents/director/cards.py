"""Scenario cards: turn a firing into nudges, a reflex override and world events.

Cards never address the robot. The robot only meets the consequences.
"""
from __future__ import annotations

import hashlib
import logging
import re
import typing
from typing import Optional

from wardzero_agents.config.models import CardDef, Persona, PersonInstance
from wardzero_agents.contracts import Nudge, ReflexOverride, Roster, WorldEvent

from .storyboard import CardFiring, WorldView

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
log = logging.getLogger(__name__)
PERSON_EVENTS = ("person_exit", "ring_bell", "asleep", "unwell")  # world events about the card's target person
# the world event kinds this contract carries: a card whose world kind it lacks (unwell, before the contract had it)
# still nudges, and leaves the world event out
_EVENT_KINDS = set(typing.get_args(WorldEvent.model_fields["kind"].annotation))


# who can walk in mid-run: visitors and patients, never staff, porters or companions (who come with someone)
ARRIVAL_ROLES = ("patient", "ae_patient", "bed_patient", "visitor", "ward_visitor")


def arrival_persona(personas: dict[str, Persona], seed: int, firing_id: str,
                    in_bed: "set[str] | frozenset[str]" = frozenset()) -> Optional[str]:
    """The persona for an arrival that names none: chosen from the run seed and the firing id among the visitor and
    patient personas, less those of patients who start in a bed (in_bed: persona ids). Same seed, same choice; None
    when nobody in the scene could walk in."""
    ids = sorted(pid for pid, p in personas.items() if p.role in ARRIVAL_ROLES and pid not in in_bed)
    if not ids:
        return None
    h = hashlib.sha256(f"{seed}:{firing_id}".encode("utf-8")).digest()
    return ids[int.from_bytes(h[:8], "big") % len(ids)]


class CardRejected(Exception):
    """The card cannot run, for example because its target has left the room."""


class CardExecutor:
    def __init__(self, deck: dict[str, CardDef], instances_by_id: dict[str, PersonInstance],
                 personas: Optional[dict[str, Persona]] = None, seed: int = 0):
        self.deck = deck
        self.instances = instances_by_id
        self.personas = personas  # with seed, picks the persona of an arrival that names none
        self.seed = seed

    def _names(self, roster: Roster) -> dict[str, str]:
        names = {pid: i.display_name for pid, i in self.instances.items()}
        names.update({p.id: p.display_name for p in roster.people})
        return names

    @staticmethod
    def _render(template: str, values: dict[str, str], params: dict) -> str:
        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key.startswith("params."):
                pk = key[len("params."):]
                if pk not in params:
                    raise CardRejected(f"template key '{key}' is missing")
                return str(params[pk])
            if key in values:
                return values[key]
            raise CardRejected(f"template key '{key}' is missing")
        return _PLACEHOLDER.sub(sub, template)

    def execute(self, firing: CardFiring, view: WorldView, roster: Roster) -> list:
        card = self.deck.get(firing.card)
        if card is None:
            raise CardRejected(f"unknown card '{firing.card}'")
        target = firing.target
        if card.target_kind == "person":
            if target is None:
                raise CardRejected(f"card '{card.id}' needs a target person")
            if target not in view.present:
                raise CardRejected(f"target {target} is not in the room")
        names = self._names(roster)
        companions = sorted(p.id for p in roster.people if target is not None and p.companion_of == target)
        # Companions who are not in the room cannot be nudged.
        companions = [c for c in companions if c in view.present]
        params = firing.params

        def values_for(person: Optional[str]) -> dict[str, str]:
            vals: dict[str, str] = {}
            if person is not None:
                vals["name"] = names.get(person, person)
            if companions:
                vals["companion"] = names.get(companions[0], companions[0])
            return vals

        outs: list = []
        if "reflex" in card.channels and card.reflex and target is not None:
            r = {**card.reflex, **{k: params[k] for k in ("duration_s", "severity") if k in params}}
            d, sev = r.get("duration_s"), r.get("severity")
            if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
                raise CardRejected("reflex duration_s must be a number above 0")
            if sev not in ("mild", "moderate", "severe"):
                raise CardRejected("reflex severity must be mild, moderate or severe")
            outs.append(ReflexOverride(person=target, t=firing.t, kind="cough_bout",
                                       duration_s=float(d), severity=sev))
        if "nudge" in card.channels:
            if card.target_kind == "all":
                if card.nudge:
                    for pid in sorted(view.present):
                        outs.append(Nudge(person=pid, t=firing.t, card=card.id,
                                          message=self._render(card.nudge, values_for(pid), params)))
            else:
                if card.nudge and target is not None:
                    outs.append(Nudge(person=target, t=firing.t, card=card.id,
                                      message=self._render(card.nudge, values_for(target), params)))
                if card.companion_nudge:
                    for c in companions:
                        outs.append(Nudge(person=c, t=firing.t, card=card.id,
                                          message=self._render(card.companion_nudge, values_for(target), params)))
        if "world" in card.channels and card.world:
            w = {**card.world, **{k: params[k] for k in ("kind", "text", "x", "y", "persona") if k in params}}
            text = w.get("text")
            if text is not None:
                text = self._render(str(text), values_for(target), params)
            kind = w["kind"]
            if kind not in _EVENT_KINDS:
                log.warning("card '%s': world event kind '%s' is not in this contract; left out", card.id, kind)
                return outs
            persona = w.get("persona") if kind == "arrival" else None
            if kind == "arrival" and persona is None and self.personas:
                in_bed = {i.persona_id for i in self.instances.values() if i.start is not None and i.start.bed}
                persona = arrival_persona(self.personas, self.seed, firing.id, in_bed)
            outs.append(WorldEvent(
                id=firing.id, t=firing.t, kind=kind, text=text, x=w.get("x"), y=w.get("y"),
                persona=persona,
                person=target if kind in PERSON_EVENTS else None))
        return outs
