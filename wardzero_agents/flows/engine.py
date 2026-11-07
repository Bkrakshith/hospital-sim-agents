"""Run a scenario flow as a py_trees behaviour tree: the director's replacement for the clock-driven storyboard.

``FlowEngine`` has the storyboard engine's interface (``tick``, ``timeline``, ``skip``, ``rerun``), so the director, the
control panel and the Studio's Run page use it unchanged. Each tick the director hands it a ``FlowView`` (what has
happened so far in the run); the tree decides which cards fire now. ``tree()`` is the tree with each step's state, for
the Studio's flow view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import py_trees
from py_trees.common import Status as S

from ..director.storyboard import CardFiring, WorldView
from .spec import Flow

PENDING, RUNNING, MET, TIMED_OUT, FIRED, SKIPPED, NOT_NEEDED, DONE = (
    "pending", "running", "met", "timed_out", "fired", "skipped", "not_needed", "done")
TARGET_GRACE_S = 10.0  # a card aimed at someone in the cast whose first status has not arrived yet waits this long


@dataclass
class FlowView(WorldView):
    """WorldView plus what the flow's conditions look at."""
    robot_start: tuple[float, float] = (0.0, 0.0)
    cast: set[str] = field(default_factory=set)  # the people there at the start (an arrival is a newcomer)
    robot_said_to: set[str] = field(default_factory=set)
    robot_said_at: dict[str, list[float]] = field(default_factory=dict)  # person -> when the robot spoke to them
    said_to_robot: set[str] = field(default_factory=set)
    # (t, action, person or None, item or None); older callers give (t, action, person) or (action, person)
    robot_did: list[tuple] = field(default_factory=list)
    fires: dict[str, bool] = field(default_factory=dict)  # fire id -> out
    room_at: Callable[[float, float], Optional[str]] = lambda x, y: None


# ---- conditions ---------------------------------------------------------------------------------------------------
def _since(a, t0: Optional[float]) -> Optional[float]:
    """The time from which a condition counts events: the step's start with ``since_step: true``, else None (all)."""
    return t0 if isinstance(a, dict) and a.get("since_step") is True and t0 is not None else None


def _did(v: FlowView):
    """The robot's actions as (t, action, person, item); t is None for an entry without a time (an old caller)."""
    for r in v.robot_did:
        if len(r) >= 4:
            yield r[0], r[1], r[2], r[3]
        elif len(r) == 3:
            yield r[0], r[1], r[2], None
        else:
            yield None, r[0], r[1], None


def holds(c: dict, v: FlowView, t0: Optional[float] = None) -> bool:
    """Whether the condition holds. ``t0`` is when the step began; ``since_step`` conditions count from it."""
    k, a = next(iter(c.items()))
    st = v.statuses
    if k == "all":
        return all(holds(x, v, t0) for x in a)
    if k == "any":
        return any(holds(x, v, t0) for x in a)
    if k == "not":
        return not holds(a, v, t0)
    if k in ("robot_spoke_to", "spoke_to_robot"):
        said = v.robot_said_to if k == "robot_spoke_to" else v.said_to_robot
        since = _since(a, t0)
        if isinstance(a, dict):
            a = a["person"]
        if since is not None:  # only words at or after the step began
            said = {p for p, ts in v.robot_said_at.items() if any(t >= since for t in ts)}
        if a == "anyone":
            return bool(said)
        if a == "newcomer":
            return any(p not in v.cast for p in said)
        return a in said
    if k == "robot_near":
        s = st.get(a["person"])
        return s is not None and s.posture != "gone" and \
            math.hypot(s.x - v.robot_xy[0], s.y - v.robot_xy[1]) <= float(a["within_m"])
    if k == "near":
        s, o = st.get(a["person"]), st.get(a["to"])
        return s is not None and o is not None and "gone" not in (s.posture, o.posture) and \
            math.hypot(s.x - o.x, s.y - o.y) <= float(a["within_m"])
    if k == "posture":
        s = st.get(a["person"])
        return s is not None and s.posture == a["is"]
    if k in ("in_room", "not_in_room"):
        s = st.get(a["person"])
        inside = s is not None and s.posture != "gone" and v.room_at(s.x, s.y) == a["room"]
        return inside if k == "in_room" else (s is not None and not inside)
    if k == "robot_did":
        since = _since(a, t0)
        n = sum(1 for t, act, who, item in _did(v)
                if act == a["action"] and (a.get("about") is None or who == a["about"])
                and (a.get("item") is None or item == a["item"])
                and (since is None or (t is not None and t >= since)))
        return n >= int(a.get("at_least", 1))
    if k == "robot_moved":
        return math.hypot(v.robot_xy[0] - v.robot_start[0], v.robot_xy[1] - v.robot_start[1]) >= float(a)
    if k == "fire_out":
        return bool(v.fires) and all(v.fires.values())
    if k == "elapsed":
        return v.t >= float(a)
    raise ValueError(k)


# ---- the tree -------------------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self):
        self.view: Optional[FlowView] = None
        self.firings: list[CardFiring] = []
        self.n = 0
        self.nodes: list["_Step"] = []  # every step node, in tree order


class _Step(py_trees.behaviour.Behaviour):
    """A leaf step. ``state`` and ``t`` are what the timeline and the flow view show."""

    kind = ""

    def __init__(self, ctx: _Ctx, body: dict, label: str):
        super().__init__(name=label)
        self.ctx, self.body, self.label = ctx, body, label
        self.state, self.t, self.t0, self.reason = PENDING, None, None, None
        ctx.nodes.append(self)

    def initialise(self):
        self.t0 = self.ctx.view.t
        if self.state == PENDING:
            self.state = RUNNING


class _Wait(_Step):
    kind = "wait"

    def update(self):
        if self.ctx.view.t - self.t0 >= float(self.body["seconds"]):
            self.state, self.t = DONE, self.ctx.view.t
            return S.SUCCESS
        return S.RUNNING


class _Until(_Step):
    kind = "wait_until"

    def update(self):
        v = self.ctx.view
        if holds(self.body["when"], v, self.t0):
            self.state, self.t = MET, v.t
            return S.SUCCESS
        if v.t - self.t0 >= float(self.body["timeout_s"]):
            self.state, self.t = TIMED_OUT, v.t
            self.reason = f"not seen within {float(self.body['timeout_s']):g} s: the flow moved on"
            return S.SUCCESS
        return S.RUNNING


class _Expect(_Until):
    kind = "expect"


class _Fire(_Step):
    kind = "fire"

    def update(self):
        v = self.ctx.view
        if self.state in (SKIPPED, NOT_NEEDED):
            return S.SUCCESS
        if self.state != FIRED:
            target = self.body.get("target")
            if target is not None and target not in v.present:
                if target in v.cast and target not in v.statuses and v.t - self.t0 < TARGET_GRACE_S:
                    return S.RUNNING  # in the cast, but their first status has not arrived yet (the first ticks)
                self.state, self.reason = SKIPPED, f"target {target} is not in the scene"
                return S.SUCCESS
            self.ctx.n += 1
            self.ctx.firings.append(CardFiring(f"flow-{self.ctx.n}", v.t, self.body["card"], target,
                                               dict(self.body.get("params") or {}), "storyboard"))
            self.state, self.t = FIRED, v.t
        return S.SUCCESS


class _Check(py_trees.behaviour.Behaviour):
    """The condition of an ``unless``: SUCCESS when it holds (the ``then`` branch is not needed)."""

    def __init__(self, ctx: _Ctx, when: dict, owner: "_Unless"):
        super().__init__(name="unless")
        self.ctx, self.when, self.owner = ctx, when, owner
        self.t0: Optional[float] = None  # when the unless began (it is checked afresh every tick)

    def update(self):
        if self.t0 is None:
            self.t0 = self.ctx.view.t
        if holds(self.when, self.ctx.view, self.t0):
            self.owner.resolved(self.ctx.view.t)
            return S.SUCCESS
        return S.FAILURE


class _Unless(py_trees.composites.Selector):
    def __init__(self, ctx: _Ctx, body: dict, then):
        super().__init__(name=body["label"], memory=False)
        self.label, self.body, self.then = body["label"], body, then
        self.add_children([_Check(ctx, body["when"], self), then])
        self.state, self.t = PENDING, None

    def resolved(self, t: float) -> None:
        if self.state == PENDING:
            self.state, self.t = MET, t
            for n in _leaves(self.then):
                if n.state in (PENDING, RUNNING):
                    n.state, n.reason = NOT_NEEDED, f"not needed: {self.label}"


def _leaves(node):
    if isinstance(node, _Step):
        yield node
    for c in getattr(node, "children", []):
        yield from _leaves(c)


def _build(ctx: _Ctx, steps: list[dict], label: str):
    seq = py_trees.composites.Sequence(name=label, memory=True)
    for s in steps:
        seq.add_child(_node(ctx, s))
    return seq


def _node(ctx: _Ctx, s: dict):
    k, v = next(iter(s.items()))
    if k == "wait":
        return _Wait(ctx, v, v.get("label") or f"Wait {float(v['seconds']):g} s")
    if k == "wait_until":
        return _Until(ctx, v, v["label"])
    if k == "expect":
        return _Expect(ctx, v, v["label"])
    if k == "fire":
        return _Fire(ctx, v, v.get("label") or f"Play {v['card']}" + (f" ({v['target']})" if v.get("target") else ""))
    if k == "parallel":
        par = py_trees.composites.Parallel(name=v["label"], policy=py_trees.common.ParallelPolicy.SuccessOnAll())
        for i, b in enumerate(v["branches"]):
            par.add_child(_build(ctx, b, f"{v['label']} ({i + 1})"))
        return par
    if k == "unless":
        return _Unless(ctx, v, _build(ctx, v["then"], v["label"]))
    raise ValueError(k)


class FlowEngine:
    """The storyboard engine's interface over a flow's behaviour tree."""

    def __init__(self, flow: Flow, deck: Optional[dict] = None):
        self.flow = flow
        self.deck = deck or {}
        self._ctx = _Ctx()
        self.root = _build(self._ctx, flow.steps, flow.title)
        self.root.setup_with_descendants()
        self.finished = False

    def tick(self, view: FlowView) -> list[CardFiring]:
        if not isinstance(view, FlowView):  # a plain WorldView (tests, an old caller): nothing extra seen
            view = FlowView(**vars(view))
        ctx = self._ctx
        ctx.view, ctx.firings = view, []
        if not self.finished:
            self.root.tick_once()
            self.finished = self.root.status == S.SUCCESS
        for n in ctx.nodes:  # rerun: a fired card asked to fire again
            if getattr(n, "_again", False):
                n._again = False
                ctx.n += 1
                ctx.firings.append(CardFiring(f"flow-{ctx.n}", view.t, n.body["card"], n.body.get("target"),
                                              dict(n.body.get("params") or {}), "storyboard"))
                n.state, n.t = FIRED, view.t
        return ctx.firings

    def _fire_node(self, beat_id: str) -> "_Fire":
        for n in self._ctx.nodes:
            if isinstance(n, _Fire) and n.body["id"] == beat_id:
                return n
        raise KeyError(beat_id)

    def skip(self, beat_id: str) -> None:
        n = self._fire_node(beat_id)
        if n.state in (PENDING, RUNNING):
            n.state, n.reason = SKIPPED, "skipped by operator"

    def rerun(self, beat_id: str) -> None:
        n = self._fire_node(beat_id)
        n._again = True

    def timeline(self) -> list[dict]:
        """The cards (and the robot's expected actions) in flow order, for the panel and the Run page."""
        rows = []
        for n in self._ctx.nodes:
            if isinstance(n, _Fire):
                row = {"id": n.body["id"], "card": n.body["card"], "target": n.body.get("target"),
                       "state": {FIRED: "fired", SKIPPED: "skipped", NOT_NEEDED: "skipped"}.get(n.state, "pending"),
                       "fired_t": n.t if n.state == FIRED else None}
                if n.reason:
                    row["reason"] = n.reason
                rows.append(row)
        return rows

    def tree(self) -> dict:
        """The tree with each step's state: {label, kind, state, t, reason, children}. A parallel branch of one step
        is shown as that step (no "in order" row around it)."""
        def walk(node) -> dict:
            if isinstance(node, py_trees.composites.Parallel):
                kids = [c.children[0] if isinstance(c, py_trees.composites.Sequence) and len(c.children) == 1 else c
                        for c in node.children]
                return {"label": node.name, "kind": "parallel", "children": [walk(c) for c in kids],
                        "state": getattr(node, "state", None), "t": getattr(node, "t", None), "reason": None}
            if isinstance(node, _Step):
                return {"label": node.label, "kind": node.kind, "state": node.state, "t": node.t,
                        "reason": node.reason, "children": []}
            kind = ("unless" if isinstance(node, _Unless) else
                    "parallel" if isinstance(node, py_trees.composites.Parallel) else "sequence")
            kids = [c for c in node.children if not isinstance(c, _Check)]
            if isinstance(node, _Unless):
                kids = kids[0].children if kids else []
            out = {"label": node.name, "kind": kind, "children": [walk(c) for c in kids],
                   "state": getattr(node, "state", None), "t": getattr(node, "t", None), "reason": None}
            if isinstance(node, _Unless):
                out["when"] = node.body["when"]
            return out
        return walk(self.root)
