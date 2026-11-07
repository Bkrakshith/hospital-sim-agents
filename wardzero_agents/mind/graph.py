"""The persona mind as a LangGraph graph: observe, recall, decide, act or keep, remember, summarise."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..contracts import Intent, Nudge, Observation, Say, Status
from ..llm.gateway import LLMRequest
from ..memory.longterm import render_memory, robot_tag
from .prompts import (
    INTENT_KINDS, KB_HEADER, MAX_MEMORIES, MOODS, SUMMARY_FIXED, SUMMARY_SCHEMA,
    SYSTEM_FIXED, decision_schema, persona_card, render_situation, robot_role_of, role_kinds, system_fixed)

if TYPE_CHECKING:
    from ..config.models import Sign, WaitingArea
    from .runtime import MindDeps

log = logging.getLogger(__name__)

SUMMARY_AT_LINES = 30  # summarise once short-term memory holds this many lines
SHORT_TERM_CAP = 60  # hard cap while summaries keep failing
SHORT_TERM_KEEP = 5
SUMMARY_EVERY_S = 300.0
SUMMARY_RETRY_S = 60.0  # back-off after a failed summary call
KB_K = 4
LTM_K = 5
NEEDS_TARGET = {"walk_to", "approach", "sit_down", "read_sign", "look_at", "attend", "give_item",
                "take_item", "escort", "evacuate"}


class MindState(TypedDict, total=False):
    person: str
    t: float
    triggers: list[str]
    frame: dict  # this tick's observation, status, nudges and clock_min, as JSON data
    situation: str
    kb_hits: list[str]
    ltm_hits: list[str]
    short_term: list[str]
    current_intent: Optional[dict]
    mood: str
    patience_left: float
    decision: Optional[dict]
    error: Optional[str]
    last_summary_t: float
    summary_retry_t: float  # no summary before this time (set after a failed summary call)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError
    v = v.strip()
    return v or None


def check_decision(data: Any, area: "WaitingArea", signs: Iterable["Sign"], roster: Iterable[str],
                   robot_id: str = "g1", role: Optional[str] = None, site: Any = None) -> tuple[Optional[dict], Optional[str]]:
    """Normalise a decision; role limits which intent kinds are valid (None: the basics). Returns (decision, None) or (None, "invalid_target" | "invalid_decision").

    With the site, any place the site resolves is a valid target too: the persona card names rooms
    ("AE_Waiting") and the world walks to them."""
    try:
        intent = data["intent"]
        kind = intent["kind"]
        target = _str_or_none(intent.get("target"))
        text = _str_or_none(intent.get("text"))
        say = _str_or_none(data.get("say"))
        addressee = _str_or_none(data.get("addressee"))
        mood = data["mood"]
        delta = data["patience_delta"]
        reason = data.get("reason") or ""
    except (KeyError, TypeError, AttributeError):
        return None, "invalid_decision"
    if kind not in role_kinds(role) or mood not in MOODS or not isinstance(reason, str):
        return None, "invalid_decision"
    if not _is_number(delta):
        return None, "invalid_decision"
    known = ({s.id for s in area.seats + area.staff_seats} | set(area.landmarks) | {s.id for s in signs}
             | set(roster) | {robot_id})
    if kind == "leave" and target is None:
        target = area.entrance
    if kind == "follow":  # the robot leads a follow; following a person is walking with them (approach)
        target = robot_id if target in (None, "robot", robot_id) else target
        if target != robot_id:
            if target not in set(roster):
                return None, "invalid_target"
            kind = "approach"
    if kind == "sit_down" and target is None:
        pass  # "sit down here": the world picks the nearest free seat
    elif kind in NEEDS_TARGET and target is None:
        return None, "invalid_target"
    if target is not None and target not in known and not (site is not None and site.resolve(target) is not None):
        return None, "invalid_target"
    if kind == "talk":
        text = text or say
        if not text:
            return None, "invalid_decision"
        say = say or text
    if kind == "request_instrument":
        if not text:
            return None, "invalid_decision"
        addressee = addressee or robot_id
    if addressee is not None and addressee not in known:
        addressee = None  # a stray addressee should not cost the whole decision
    return {"intent": {"kind": kind, "target": target, "text": text}, "say": say,
            "addressee": addressee, "mood": mood, "patience_delta": max(-1.0, min(1.0, float(delta))),
            "reason": reason}, None


def validate_decision(data: Any, area: "WaitingArea", signs: Iterable["Sign"], roster: Iterable[str],
                      robot_id: str = "g1", role: Optional[str] = None) -> Optional[dict]:
    """The normalised decision, or None if it names an unknown target or is malformed."""
    return check_decision(data, area, signs, roster, robot_id, role)[0]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _event_line(t: float, frame: dict, decision: Optional[dict], error: Optional[str]) -> str:
    """One short-term memory line. Plain words, never the prompt markers."""
    parts = [f"t={t:.0f}s"]
    for h in frame["obs"].get("heard", []):
        text = h.get("text")
        if h["source_kind"] == "robot" and text:
            parts.append(f"the robot said {'to me' if h['addressed_to_me'] else 'nearby'} \"{text}\"")
        elif h["source_kind"] == "person" and text:
            parts.append(f"{h['source_id']} said \"{text}\"")
        elif h["source_kind"] == "pa" and text:
            parts.append(f"the PA said \"{text}\"")
    for n in frame["nudges"]:
        parts.append(f"I felt: {n['message']}")
    if decision:
        i = decision["intent"]
        act = f"I chose {i['kind']}" + (f" {i['target']}" if i["target"] else "")
        if decision["say"]:
            act += f" and said \"{decision['say']}\""
        parts.append(act + f" (mood {decision['mood']})")
    else:
        parts.append(f"I hesitated and carried on ({error})")
    return "; ".join(parts)


_ROBOT_WORDS = re.compile(r"\b(robot|g1)\b", re.I)


def _robot_tag_for(text: str, robot_id: str) -> Optional[str]:
    """A memory that talks about the robot is tagged with it, so a different robot is not credited."""
    return robot_tag(robot_id) if _ROBOT_WORDS.search(text) else None


def build_graph(deps: "MindDeps", checkpointer: Any = None):
    """Compile the mind graph for one person. thread_id in the run config must be the person id."""
    inst, loaded = deps.inst, deps.loaded
    robot_role = robot_role_of(loaded.scenario)
    card = persona_card(inst, loaded.area, loaded.site, loaded.scenario.known_places, robot_role=robot_role,
                        trust_stage=deps.trust_stage, ladder=loaded.scenario.ladder or None,
                        robot_memories=deps.robot_memories,
                        word_of_mouth=deps.word_of_mouth if loaded.scenario.campaign else None)
    person = inst.person_id
    n_persona = sum(1 for c in deps.kb.chunks if c.file == "<persona>") if deps.kb is not None else 0

    async def observe(state: MindState) -> dict:
        frame = state["frame"]
        obs = Observation.model_validate(frame["obs"])
        status = Status.model_validate(frame["status"])
        nudges = [Nudge.model_validate(n) for n in frame["nudges"]]
        current = state.get("current_intent")
        if current and status.intent_id == current.get("id"):
            current = {**current, "state": status.state}
        mood = state.get("mood", "calm")
        patience = state.get("patience_left", 1.0)
        situation = render_situation(inst, obs, status, state["triggers"], nudges, frame["clock_min"],
                                     mood=mood, patience_left=patience, current_intent=current,
                                     area=loaded.area, site=loaded.site)
        return {"situation": situation, "current_intent": current, "mood": mood,
                "patience_left": patience, "short_term": state.get("short_term", []),
                "last_summary_t": state.get("last_summary_t", state["t"]),
                "decision": None, "error": None, "kb_hits": [], "ltm_hits": []}

    async def recall(state: MindState) -> dict:
        q = state["situation"]
        kb_hits: list[str] = []
        if deps.kb is not None:
            # persona chunks are left out: one's own card is already in the prompt, other people's are private
            hits = await asyncio.to_thread(deps.kb.search, q, KB_K + n_persona)
            kb_hits = [c.text for c, _ in hits if c.file != "<persona>"][:KB_K]
        ltm_hits: list[str] = []
        if deps.ltm is not None:
            mem = await asyncio.to_thread(deps.ltm.search, inst.memory_key, q, LTM_K,
                                          run_index=deps.run_index)
            ltm_hits = [render_memory(h, deps.memory_robot_id) for h in mem]
        return {"kb_hits": kb_hits, "ltm_hits": ltm_hits}

    def history_blocks(state: MindState) -> list[str]:
        """The day so far: append-only, so the cached prefix grows by one block a call."""
        if not state.get("short_term"):
            return []
        return ["What has happened so far today (oldest first):", *[f"- {h}" for h in state["short_term"]]]

    def user_text(state: MindState) -> str:
        out = []
        if state.get("kb_hits"):
            out.append(KB_HEADER)
            out += ["- " + h.strip().replace("\n", "\n  ") for h in state["kb_hits"]]
        if state.get("ltm_hits"):
            out.append("Things you remember:")
            out += [f"- {h}" for h in state["ltm_hits"]]
        out.append("Decide what you do next. The situation now:")
        out.append(state["situation"])
        return "\n".join(out)

    sysfixed, schema = system_fixed(inst.role, robot_role), decision_schema(inst.role)

    async def decide(state: MindState) -> dict:
        user = user_text(state)
        history = history_blocks(state)
        req = LLMRequest(role="decide", person=person, system_fixed=sysfixed, persona_card=card,
                         user=user, schema=schema, history=history)
        res = await deps.gateway.complete(req)
        decision, error = None, res.error
        if res.ok:
            # the roster as it is now: people who arrived after this graph was built count too
            roster = [i.person_id for i in loaded.instances]
            decision, error = check_decision(res.data, loaded.area, loaded.signs, roster, deps.robot_id,
                                             inst.role, site=loaded.site)
        data = res.data if isinstance(res.data, dict) else None
        deps.runlog.write("decisions", {
            "t": state["t"], "person": person, "triggers": list(state["triggers"]),
            "situation": state["situation"], "result": data, "error": error,
            "reason": data.get("reason") if data else None, "latency_s": res.latency_s,
            "model": res.model,
            "prompt_sha256": hashlib.sha256(
                "\0".join([sysfixed, card, "\n".join([*history, user])]).encode("utf-8")).hexdigest()[:16],
        })
        return {"decision": decision, "error": error}

    async def act(state: MindState) -> dict:
        # Stamp with the sim time of sending, not of the decision's start: the decision took the
        # LLM's latency, and a replay plays each intent at its stamp. One value for the intent
        # and its speech, because the sim speaks a talk intent and its Say once by (t, text).
        d = state["decision"]
        t = deps.clock() if deps.clock is not None else state["t"]
        i = d["intent"]
        intent = Intent(id=uuid.uuid4().hex, person=person, t=t, kind=i["kind"], target=i["target"],
                        text=i["text"], addressee=d["addressee"])
        in_effect = deps.effects.publish_intent(intent)
        if d["say"]:
            deps.effects.publish_say(Say(person=person, t=t, text=d["say"], addressee=d["addressee"]))
        current = {"id": intent.id, "kind": intent.kind, "target": intent.target, "t": t, "state": "sent"}
        if isinstance(in_effect, str) and in_effect != intent.id:
            # Not sent: it repeats the intent still running, which stays the current one. Recording the unsent id
            # would read "(sent)" for ever and hide that intent's end.
            prev = state.get("current_intent") or {}
            status = state["frame"]["status"]
            current = (prev if prev.get("id") == in_effect
                       else {**current, "id": in_effect, "t": status.get("t", t), "state": status.get("state")})
        return {"current_intent": current, "mood": d["mood"],
                "patience_left": _clamp01(state["patience_left"] + d["patience_delta"])}

    async def keep(state: MindState) -> dict:
        log.info("%s keeps its current intent (decision error: %s)", person, state.get("error"))
        return {}

    async def remember(state: MindState) -> dict:
        line = _event_line(state["t"], state["frame"], state.get("decision"), state.get("error"))
        return {"short_term": (state.get("short_term", []) + [line])[-SHORT_TERM_CAP:]}

    async def summarise(state: MindState) -> dict:
        t, lines = state["t"], state.get("short_term", [])
        if not lines:
            return {"last_summary_t": t}
        req = LLMRequest(role="summarise", person=person, system_fixed=SUMMARY_FIXED, persona_card=card,
                         user="Recent events (oldest first):\n" + "\n".join(f"- {x}" for x in lines),
                         schema=SUMMARY_SCHEMA)
        res = await deps.gateway.complete(req)
        mems = res.data.get("memories") if res.ok and isinstance(res.data, dict) else None
        record = {"t": t, "person": person, "lines": len(lines), "memories": [], "error": res.error,
                  "exception": None, "model": res.model, "latency_s": res.latency_s}
        if not isinstance(mems, list):
            # The summary call failed: keep the events and try again after a back-off.
            record["error"] = res.error or "invalid_summary"
            deps.runlog.write("summaries", record)
            return {"last_summary_t": t, "summary_retry_t": t + SUMMARY_RETRY_S}
        try:
            for m in mems[:MAX_MEMORIES]:
                if not isinstance(m, dict) or not isinstance(m.get("text"), str) or not m["text"].strip():
                    continue
                text = m["text"].strip()
                imp = _clamp01(float(m["importance"])) if _is_number(m.get("importance")) else 0.5
                if deps.ltm is not None:
                    tag = _robot_tag_for(text, deps.memory_robot_id)
                    extra = {"tag": tag} if tag else {}
                    await asyncio.to_thread(deps.ltm.add, inst.memory_key, deps.run_id, t, "episode",
                                            text, imp, **extra)
                record["memories"].append({"text": text, "importance": imp})
        except Exception as exc:
            # The summary itself worked, so trim anyway; retrying would only repeat the same call.
            log.error("%s: writing long-term memory failed: %s", person, type(exc).__name__)
            record["error"], record["exception"] = "ltm_write_failed", type(exc).__name__
        deps.runlog.write("summaries", record)
        return {"short_term": lines[-SHORT_TERM_KEEP:], "last_summary_t": t}

    def after_decide(state: MindState) -> str:
        return "act" if state.get("decision") else "keep"

    def after_remember(state: MindState) -> str:
        t = state["t"]
        if t < state.get("summary_retry_t", float("-inf")):
            return END
        due = (len(state.get("short_term", [])) >= SUMMARY_AT_LINES
               or t - state.get("last_summary_t", t) >= SUMMARY_EVERY_S)
        return "summarise" if due else END

    g = StateGraph(MindState)
    for name, fn in [("observe", observe), ("recall", recall), ("decide", decide), ("act", act),
                     ("keep", keep), ("remember", remember), ("summarise", summarise)]:
        g.add_node(name, fn)
    g.add_edge(START, "observe")
    g.add_edge("observe", "recall")
    g.add_edge("recall", "decide")
    g.add_conditional_edges("decide", after_decide, {"act": "act", "keep": "keep"})
    g.add_edge("act", "remember")
    g.add_edge("keep", "remember")
    g.add_conditional_edges("remember", after_remember, {"summarise": "summarise", END: END})
    g.add_edge("summarise", END)
    return g.compile(checkpointer=checkpointer)
