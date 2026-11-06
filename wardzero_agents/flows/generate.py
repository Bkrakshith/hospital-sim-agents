"""Adapt a scenario's flow to a prompt the user wrote in the Studio, with a model, before a run starts.

The model gets the flow format (``flows.spec``), the scenario's people, cards, rooms and the preset's own flow, and
returns a new flow as YAML. It is checked like a preset flow (schema, then every person, card and room it names); on a
mistake the model is told what was wrong and tries again, up to ``TRIES`` times. The result carries the prompt's hash
(``prompt_sha``), so the Studio knows it was made for this prompt and only adapts again when the prompt changes.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import yaml
from pydantic import ValidationError

from ..llm.gateway import LLMRequest
from . import spec
from .spec import Flow, FlowError, dump_flow, fit_scenario

TRIES = 3
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["flow_yaml", "notes"],
          "properties": {"flow_yaml": {"type": "string"}, "notes": {"type": "string"}}}

SYSTEM = """You write scenario flows for a hospital robot test lab. A scenario puts a robot among simulated people; a
flow says what should happen, in order, and when each storyboard card fires. The director runs the flow as a
behaviour tree. A step that waits for something holds the flow until it happens or until its time limit, so the
scenario always moves on.

Rules:
- Use only the people, cards, rooms and robot actions listed for this scenario. Never invent ids.
- Every wait_until and expect needs a time limit (timeout_s). Keep the whole flow inside the run's length.
- Start with something that lets the scene get going (a short wait, or waiting for the robot to speak to someone),
  so nothing happens before the people and the robot have had a chance to act.
- Fire a card when the scene is ready for it (the robot is near enough to notice, the person is where it matters),
  with a time limit as the fallback. Use expect steps for what the robot should do in reply, so the run's report
  can show whether it did.
- World cards that happen at a place (fire_start, route_blocked, alarm) take params {x, y}: a point in the room.
- Labels are short plain English, written for a person watching the run.
- Answer with the whole flow as YAML in flow_yaml (keys: id, title, summary, steps), and one or two sentences in
  notes on what you changed and why.

The flow format:
""" + (spec.__doc__ or "")


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(" ".join((prompt or "").split()).encode()).hexdigest()[:16]


def _context(*, scenario_id: str, base: Flow, preset_prompt: str, prompt: str, people: list[dict], deck: dict,
             rooms: dict, duration_min: float) -> str:
    cards = "\n".join(f"- {c.id}: {c.label}. Target: {c.target_kind}. {c.tests}" for c in deck.values())
    who = "\n".join(f"- {p['id']}: {p['name']}, {p['role']}" + (f", starts in {p['room']}" if p.get("room") else "")
                    for p in people)
    places = "\n".join(f"- {rid}: {getattr(r, 'label', rid)}, centre ({r.centre[0]:.1f}, {r.centre[1]:.1f})"
                       for rid, r in rooms.items())
    return (f"Scenario: {scenario_id}. The run lasts {duration_min:g} minutes.\n\n"
            f"The scenario as the preset describes it:\n{preset_prompt.strip()}\n\n"
            f"The scenario as the user now describes it (write the flow for this):\n{prompt.strip()}\n\n"
            f"People (ids to use; 'anyone' and 'newcomer' also work where a person is asked for):\n{who}\n\n"
            f"Cards:\n{cards}\n\nRooms:\n{places}\n\n"
            f"Robot actions for robot_did: {', '.join(spec.ROBOT_ACTIONS)}\n\n"
            f"The preset's flow, to adapt:\n<current_flow>\n{dump_flow(base)}</current_flow>")


def check(text: str, *, scenario_id: str, people, deck, rooms) -> Flow:
    """A model's YAML as a Flow for this scenario, or FlowError with what is wrong."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise FlowError(f"the YAML does not parse: {e}") from None
    if not isinstance(data, dict):
        raise FlowError("the flow must be a mapping with id, title, summary and steps")
    data.setdefault("id", scenario_id)
    data.pop("prompt_sha", None)
    try:
        flow = Flow.model_validate(data)
    except ValidationError as e:
        raise FlowError("; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])) \
            from None
    fit_scenario(flow, people=people, cards=deck, rooms=rooms)
    return flow


async def adapt_flow(gateway, *, scenario_id: str, base: Flow, preset_prompt: str, prompt: str, people: list[dict],
                     deck: dict, rooms: dict, duration_min: float) -> tuple[Flow, str]:
    """(the flow for prompt, the model's notes). FlowError when no try gave a valid flow."""
    user = _context(scenario_id=scenario_id, base=base, preset_prompt=preset_prompt, prompt=prompt, people=people,
                    deck=deck, rooms=rooms, duration_min=duration_min)
    ids = [p["id"] for p in people]
    last: Optional[str] = None
    for _ in range(TRIES):
        msg = user if last is None else f"{user}\n\nYour last answer was not usable: {last}\nFix it and answer again."
        res = await gateway.complete(LLMRequest(role="report", person="flow", system_fixed=SYSTEM, persona_card="",
                                                user=msg, schema=SCHEMA))
        if not res.ok or not isinstance(res.data, dict):
            last = f"the call failed ({res.error})"
            if res.error in ("auth", "model_not_found", "budget"):
                break
            continue
        try:
            flow = check(str(res.data.get("flow_yaml", "")), scenario_id=scenario_id, people=ids, deck=deck,
                         rooms=rooms)
        except FlowError as e:
            last = str(e)
            continue
        return flow.model_copy(update={"prompt_sha": prompt_sha(prompt)}), str(res.data.get("notes", ""))
    raise FlowError(f"the scenario flow could not be adapted to the prompt: {last}")
