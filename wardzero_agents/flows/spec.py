"""Scenario flows: what should happen in a scenario, in order, as data a person or a model can write.

A flow is a list of steps (the root is a sequence). The director builds a py_trees behaviour tree from it
(``flows.engine``) and ticks it every 0.5 s of sim time: a step that waits for something in the world holds the flow
until it happens or its time limit runs out, so a scenario always moves on; a ``fire`` step plays a storyboard card.

Steps (each is a one-key mapping):
  wait:       {seconds, label?}
  wait_until: {when, timeout_s, label}           hold until ``when`` holds (or the time limit)
  expect:     {when, timeout_s, label}           the same, for what the robot should do; recorded met or missed
  fire:       {id, card, target?, params?, label?}
  parallel:   {label, branches: [[steps], ...]}  all branches at once; done when all are
  unless:     {when, then: [steps], label}       run ``then`` unless ``when`` comes true first (then it is not needed)

Conditions (``when``, a one-key mapping):
  robot_spoke_to: <person id | anyone | newcomer>     the robot said something to them
  spoke_to_robot: <person id | anyone>                they said something to the robot
  robot_near:     {person, within_m}
  near:           {person, to, within_m}              one person is near another
  posture:        {person, is}                        standing | walking | seated | lying | gone
  in_room:        {person, room}
  not_in_room:    {person, room}
  robot_did:      {action, about?, at_least?}          guide | seat | tell | note | press_call | hand_over | take |
                                                       extinguish | approach   (about: person id, where it is known;
                                                       at_least: how many times, default 1)
  robot_moved:    <metres from where it started>
  fire_out:       true                                 every fire in the scene is out
  elapsed:        <seconds since the run started>
  all: [conditions]   any: [conditions]   not: <condition>
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

STEP_KINDS = ("wait", "wait_until", "expect", "fire", "parallel", "unless")
COND_KINDS = ("robot_spoke_to", "spoke_to_robot", "robot_near", "near", "posture", "in_room", "not_in_room", "robot_did",
              "robot_moved", "fire_out", "elapsed", "all", "any", "not")
ROBOT_ACTIONS = ("guide", "seat", "tell", "note", "press_call", "hand_over", "take", "extinguish", "approach")
POSTURES = ("standing", "walking", "seated", "lying", "gone")
PEOPLE_WORDS = ("anyone", "newcomer")
MAX_TIMEOUT_S = 3600.0


class FlowError(ValueError):
    """A flow that does not fit the schema or the scenario (unknown card, person, room)."""


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Flow(_M):
    id: str
    title: str
    summary: str = ""
    steps: list[dict] = Field(min_length=1)
    prompt_sha: Optional[str] = None  # the scenario prompt this flow was written for (a generated flow)

    @field_validator("steps")
    @classmethod
    def _steps(cls, v):
        for s in v:
            check_step(s)
        return v


def _one(d: Any, kinds, what: str) -> tuple[str, Any]:
    if not isinstance(d, dict) or len(d) != 1:
        raise ValueError(f"a {what} must be a mapping with exactly one key, got {d!r}"[:200])
    k, v = next(iter(d.items()))
    if k not in kinds:
        raise ValueError(f"unknown {what} '{k}' (one of {', '.join(kinds)})")
    return k, v


def _num(v, what: str, lo: float = 0.0, hi: float = MAX_TIMEOUT_S) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
        raise ValueError(f"{what} must be a number from {lo:g} to {hi:g}")
    return float(v)


def _keys(v, need, may, what: str) -> dict:
    if not isinstance(v, dict):
        raise ValueError(f"{what} needs a mapping")
    missing = [k for k in need if k not in v]
    extra = [k for k in v if k not in need and k not in may]
    if missing or extra:
        raise ValueError(f"{what}: " + "; ".join(([f"missing {', '.join(missing)}"] if missing else [])
                                                  + ([f"unknown {', '.join(extra)}"] if extra else [])))
    return v


def check_cond(c) -> None:
    k, v = _one(c, COND_KINDS, "condition")
    if k in ("all", "any"):
        if not isinstance(v, list) or not v:
            raise ValueError(f"{k} needs a non-empty list of conditions")
        for x in v:
            check_cond(x)
    elif k == "not":
        check_cond(v)
    elif k in ("robot_spoke_to", "spoke_to_robot"):
        if not isinstance(v, str) or not v:
            raise ValueError(f"{k} needs a person id, anyone or newcomer")
    elif k == "robot_near":
        v = _keys(v, ("person", "within_m"), (), k)
        _num(v["within_m"], "within_m", 0.1, 50.0)
    elif k == "near":
        v = _keys(v, ("person", "to", "within_m"), (), k)
        _num(v["within_m"], "within_m", 0.1, 50.0)
    elif k == "posture":
        v = _keys(v, ("person", "is"), (), k)
        if v["is"] not in POSTURES:
            raise ValueError(f"posture 'is' must be one of {', '.join(POSTURES)}")
    elif k in ("in_room", "not_in_room"):
        _keys(v, ("person", "room"), (), k)
    elif k == "robot_did":
        v = _keys(v, ("action",), ("about", "at_least"), k)
        if "at_least" in v:
            _num(v["at_least"], "at_least", 1, 100)
        if v["action"] not in ROBOT_ACTIONS:
            raise ValueError(f"robot_did action must be one of {', '.join(ROBOT_ACTIONS)}")
    elif k == "robot_moved":
        _num(v, "robot_moved", 0.0, 500.0)
    elif k == "fire_out":
        if v is not True:
            raise ValueError("fire_out takes true")
    elif k == "elapsed":
        _num(v, "elapsed")


def check_step(s) -> None:
    k, v = _one(s, STEP_KINDS, "step")
    if k == "wait":
        v = _keys(v, ("seconds",), ("label",), k)
        _num(v["seconds"], "seconds")
    elif k in ("wait_until", "expect"):
        v = _keys(v, ("when", "timeout_s", "label"), (), k)
        check_cond(v["when"])
        _num(v["timeout_s"], "timeout_s", 1.0)
    elif k == "fire":
        v = _keys(v, ("id", "card"), ("target", "params", "label"), k)
        if v.get("params") is not None and not isinstance(v["params"], dict):
            raise ValueError("fire params must be a mapping")
    elif k == "parallel":
        v = _keys(v, ("label", "branches"), (), k)
        if not isinstance(v["branches"], list) or len(v["branches"]) < 2:
            raise ValueError("parallel needs at least two branches")
        for b in v["branches"]:
            if not isinstance(b, list) or not b:
                raise ValueError("each parallel branch is a non-empty list of steps")
            for x in b:
                check_step(x)
    elif k == "unless":
        v = _keys(v, ("when", "then", "label"), (), k)
        check_cond(v["when"])
        if not isinstance(v["then"], list) or not v["then"]:
            raise ValueError("unless needs a non-empty 'then' list of steps")
        for x in v["then"]:
            check_step(x)


# ---- the parts a scenario names --------------------------------------------------------------------------------
def walk_steps(steps):
    """Every step, depth first, as (kind, body)."""
    for s in steps:
        k, v = next(iter(s.items()))
        yield k, v
        if k == "parallel":
            for b in v["branches"]:
                yield from walk_steps(b)
        elif k == "unless":
            yield from walk_steps(v["then"])


def walk_conds(c):
    k, v = next(iter(c.items()))
    yield k, v
    if k in ("all", "any"):
        for x in v:
            yield from walk_conds(x)
    elif k == "not":
        yield from walk_conds(v)


def fit_scenario(flow: Flow, *, people, cards, rooms) -> None:
    """FlowError unless every card, person and room the flow names exists in the scenario; fire ids are unique."""
    people, cards, rooms = set(people), set(cards), set(rooms)
    errs, ids = [], set()

    def person(p, where):
        if p not in people and p not in PEOPLE_WORDS:
            errs.append(f"{where}: no person '{p}' in this scenario")

    for k, v in walk_steps(flow.steps):
        if k == "fire":
            if v["id"] in ids:
                errs.append(f"fire id '{v['id']}' is used twice")
            ids.add(v["id"])
            if v["card"] not in cards:
                errs.append(f"fire {v['id']}: no card '{v['card']}'")
            if v.get("target") is not None:
                person(v["target"], f"fire {v['id']}")
        if k in ("wait_until", "expect", "unless"):
            for ck, cv in walk_conds(v["when"]):
                if ck in ("robot_spoke_to", "spoke_to_robot"):
                    person(cv, ck)
                elif ck in ("robot_near", "posture", "in_room", "not_in_room", "near"):
                    person(cv["person"], ck)
                    if ck == "near":
                        person(cv["to"], ck)
                    if ck in ("in_room", "not_in_room") and cv["room"] not in rooms:
                        errs.append(f"{ck}: no room '{cv['room']}'")
                elif ck == "robot_did" and cv.get("about") is not None:
                    person(cv["about"], ck)
    if errs:
        raise FlowError("; ".join(errs))


def _names_missing(c: dict, people: set) -> bool:
    for k, v in walk_conds(c):
        names = []
        if k in ("robot_spoke_to", "spoke_to_robot"):
            names = [v]
        elif k in ("robot_near", "posture", "in_room", "not_in_room"):
            names = [v["person"]]
        elif k == "near":
            names = [v["person"], v["to"]]
        elif k == "robot_did" and v.get("about") is not None:
            names = [v["about"]]
        if any(n not in people and n not in PEOPLE_WORDS for n in names):
            return True
    return False


def prune_flow(flow: Flow, people) -> Flow:
    """The flow for a cast without some of its people (the user removed them): cards aimed at a missing person are
    dropped and a wait on a missing person no longer waits (it is met at once). No model is needed."""
    people = set(people)

    def steps(ss: list) -> list:
        out = []
        for s in ss:
            k, v = next(iter(s.items()))
            if k == "fire" and v.get("target") is not None and v["target"] not in people \
                    and v["target"] not in PEOPLE_WORDS:
                continue
            if k in ("wait_until", "expect") and _names_missing(v["when"], people):
                continue  # what it waited for cannot happen without them
            if k == "unless":
                then = steps(v["then"])
                if not then:
                    continue
                if _names_missing(v["when"], people):
                    out += then  # nobody can make it unnecessary: the steps just run
                    continue
                out.append({"unless": {**v, "then": then}})
                continue
            if k == "parallel":
                branches = [b for b in (steps(b) for b in v["branches"]) if b]
                if len(branches) >= 2:
                    out.append({"parallel": {**v, "branches": branches}})
                elif branches:
                    out += branches[0]
                continue
            out.append(s)
        return out

    kept = steps(flow.steps) or [{"wait": {"seconds": 1, "label": "Nothing left to wait for"}}]
    return flow.model_copy(update={"steps": kept})


def load_flow(path: "str | Path") -> Flow:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return Flow.model_validate(data)
    except (OSError, yaml.YAMLError) as e:
        raise FlowError(f"{path}: {e}") from None
    except ValidationError as e:
        raise FlowError(f"{Path(path).name}: {e.errors()[0]['msg']}") from None


def dump_flow(flow: Flow) -> str:
    return yaml.safe_dump(flow.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True,
                          width=110)


def flows_dir(config_root: "str | Path") -> Path:
    return Path(config_root) / "presets" / "flows"


def preset_flow(config_root: "str | Path", preset_id: str) -> Optional[Flow]:
    p = flows_dir(config_root) / f"{preset_id}.yaml"
    return load_flow(p) if p.exists() else None
