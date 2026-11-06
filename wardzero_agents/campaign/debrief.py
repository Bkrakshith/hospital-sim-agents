"""After a run: the people answer a short questionnaire about the robot, hearsay is read from what
they said to each other, and the experiences of the run move each person's trust.

The numbers that change trust come from code (campaign.experiences rules, seeded dice, campaign.trust).
The language model only answers the questionnaire in character and says which lines of talk were
about the robot. Writes debrief.jsonl and trust.jsonl in the run folder; metrics.py reads both."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..llm.gateway import Gateway, LLMRequest
from .dice import roll_d20
from .experiences import detect
from .trust import Experience, TrustStore, apply_event, end_of_run, stage_for

log = logging.getLogger(__name__)

ROBOT_ID = "g1"  # as campaign.ROBOT_ID; the id trust is kept under comes from the run (meta robot_id)
ROBOT_NAMES = frozenset({"robot", "g1"})
MAX_TRANSCRIPT_LINES = 40
MAX_HEARSAY_LINES = 80
HEARSAY_MIN_VALENCE = 0.2  # a mention that mild is not an opinion

QUESTIONNAIRE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {k: {"type": "integer"}  # 1 to 5, clamped on reading (structured outputs reject minimum/maximum)
                   for k in ("likeability", "safety", "competence", "trust")}
    | {"comment": {"type": "string"}},
    "required": ["likeability", "safety", "competence", "trust", "comment"]}
HEARSAY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"line": {"type": "integer"}, "speaker": {"type": "string"},
                       "listener": {"type": "string"}, "valence": {"type": "number"}},
        "required": ["line", "speaker", "listener", "valence"]}}},
    "required": ["items"]}

QUESTIONNAIRE_SYSTEM = (
    "You are one person who spent time near a robot in a hospital. Answer a short questionnaire about the "
    "robot, in character, using only what the card and the account below tell you. Rate each item from 1 "
    "(very poor) to 5 (very good): likeability (did you like it), safety (did you feel safe near it), "
    "competence (did it do its job well) and trust (would you rely on it again). Then give one sentence of "
    "comment in your own voice. Do not invent events that the account does not mention.")
HEARSAY_SYSTEM = (
    "Below are lines that people in a hospital said to each other, numbered. Pick only the lines in which "
    "the speaker tells the listener what they think of the robot or what it did. For each give the line "
    "number, the speaker, the listener and a valence from -1 (the speaker warns against the robot) to +1 "
    "(the speaker recommends it). Lines that do not mention the robot are left out. If there are none, "
    "return an empty list.")


def clock(t: float) -> str:
    """Seconds as mm:ss."""
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


@dataclass
class DebriefResult:
    ratings: list[dict] = field(default_factory=list)
    trust: list[dict] = field(default_factory=list)
    hearsay: list[Experience] = field(default_factory=list)
    experiences: list[Experience] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # people whose questionnaire could not be used
    already: bool = False
    ok: bool = True  # False when no questionnaire could be used: nothing was written or applied


def _rows(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _load_instances(run_dir: Path, meta: dict) -> dict:
    """person id -> PersonInstance from the run's own config copy; empty when it cannot be loaded."""
    try:
        from ..config.loader import load_scenario
        cfg = run_dir / "config"
        name = Path(meta.get("scenario_path") or "").name
        path = cfg / name if name and (cfg / name).is_file() else next(iter(sorted(cfg.glob("*.yaml"))), None)
        # the copy also holds the deck and storyboard; the scenario file is the one with a cast
        for cand in ([path] if path else []) + sorted(cfg.glob("*.yaml")):
            try:
                return {i.person_id: i for i in load_scenario(cand).instances}
            except Exception:
                continue
    except Exception:
        pass
    return {}


def run_people(meta: dict, data) -> list[str]:
    """Non-static people of the run: meta["people"], else whoever has a status row."""
    people = meta.get("people")
    if isinstance(people, dict) and people:
        return [p for p, info in people.items() if not (isinstance(info, dict) and info.get("static"))
                and not data.is_static(p)]
    return sorted(p for p in {r.get("person") for r in data.status} if isinstance(p, str)
                  and p not in ROBOT_NAMES and not data.is_static(p))


def _account(pid: str, says: list[dict], exps: list[Experience]) -> str:
    lines = []
    for s in says:
        who, to = s.get("person"), s.get("addressee")
        if pid not in (who, to) or not s.get("text"):
            continue
        if who in ROBOT_NAMES or to in ROBOT_NAMES:
            who_s = "You" if who == pid else "The robot"
            to_s = "the robot" if to in ROBOT_NAMES else str(to or "everyone")
            lines.append((s["t"], f"[{clock(s['t'])}] {who_s} said to {to_s}: {s['text']}"))
    texts = {"success": "The robot did what you needed.", "failure": "The robot failed at something you needed.",
             "repair_immediate": "The robot apologised or explained straight away.",
             "repair_at_request": "The robot apologised or explained when you next asked it for something."}
    for e in exps:
        if e.person == pid and e.kind in texts:
            lines.append((e.t, f"[{clock(e.t)}] {texts[e.kind]}"))
    lines.sort(key=lambda x: x[0])
    body = "\n".join(t for _, t in lines[:MAX_TRANSCRIPT_LINES]) or "You had no dealings with the robot."
    return body


async def _questionnaire(gateway: Gateway, pid: str, card: str, account: str) -> Optional[dict]:
    req = LLMRequest(role="summarise", person=pid, system_fixed=QUESTIONNAIRE_SYSTEM, persona_card=card,
                     user=f"WHAT HAPPENED BETWEEN YOU AND THE ROBOT:\n{account}\n\nAnswer the questionnaire.",
                     schema=QUESTIONNAIRE_SCHEMA)
    res = await gateway.complete(req)
    if not res.ok or not isinstance(res.data, dict):
        return None
    out = {}
    for k in ("likeability", "safety", "competence", "trust"):
        v = res.data.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out[k] = int(min(5, max(1, round(v))))
    out["comment"] = str(res.data.get("comment") or "").strip()
    return out


async def _hearsay(gateway: Gateway, says: list[dict], people: set[str], robot_id: str = ROBOT_ID) -> list[Experience]:
    talk = [s for s in sorted(says, key=lambda s: s.get("t", 0.0))
            if s.get("person") in people and s.get("addressee") in people and s.get("text")][:MAX_HEARSAY_LINES]
    if not talk:
        return []
    body = "\n".join(f"L{i} [{clock(s['t'])}] {s['person']} to {s['addressee']}: {s['text']}"
                     for i, s in enumerate(talk))
    res = await gateway.complete(LLMRequest(
        role="summarise", person="hearsay", system_fixed=HEARSAY_SYSTEM, persona_card="People: " + ", ".join(sorted(people)),
        user=body, schema=HEARSAY_SCHEMA))
    if not res.ok or not isinstance(res.data, dict):
        return []
    out = []
    for it in res.data.get("items") or []:
        if not isinstance(it, dict):
            continue
        try:
            line = talk[int(it["line"])]
            v = float(it["valence"])
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        v = max(-1.0, min(1.0, v))
        if abs(v) < HEARSAY_MIN_VALENCE:
            continue
        out.append(Experience(float(line["t"]), str(line["addressee"]), robot_id, "hearsay", False, v < 0,
                              str(line["person"])))
    return out


async def debrief(run_dir: "str | Path", gateway: Gateway, *, preset=None, roles: Optional[dict] = None,
                  db_path: "str | Path | None" = None, force: bool = False) -> DebriefResult:
    """Run the debrief of a finished run. Trust is applied to the campaign db once: a run already
    debriefed (marked in the db in the same transaction as the trust) is returned as it was; force re-applies it from the states it started from."""
    from ..bench.metrics import RunData
    from ..config.presets import DEFAULT_CONFIG_ROOT, load_roles, preset_for_run
    from ..memory import data_dir, memory_path
    from ..runlog import RunLog
    from . import role_defaults

    run_dir = Path(run_dir)
    d = RunData.load(run_dir)
    meta = d.meta
    store = TrustStore(db_path or memory_path(data_dir()))
    run_id = str(meta.get("run_id") or run_dir.name)
    stored_before = store.debriefed_before(run_id)  # the db is the authority on whether trust was applied
    if stored_before is not None and not force:
        return DebriefResult(ratings=_rows(run_dir / "debrief.jsonl"), trust=_rows(run_dir / "trust.jsonl"),
                             already=True)
    preset = preset if preset is not None else preset_for_run(meta)
    if roles is None:
        try:
            roles = load_roles(DEFAULT_CONFIG_ROOT)
        except Exception:
            roles = {}
    people = run_people(meta, d)
    instances = _load_instances(run_dir, meta)
    chars = {c.id: c for c in getattr(preset, "characters", [])}
    says = d.says
    robot_id = str(meta.get("robot_id") or ROBOT_ID)  # per robot; runs made before this have the G1's
    exps = detect(run_dir, preset, robot_id)

    # 1. the questionnaire, one call per person
    result = DebriefResult()
    for pid in people:
        inst = instances.get(pid)
        name = inst.display_name if inst else (chars[pid].name if pid in chars else pid)
        backstory = inst.backstory if inst else ""
        card = f"You are {name} ({pid}). {backstory}".strip()
        row = await _questionnaire(gateway, pid, card, _account(pid, says, exps))
        if row is None:
            result.skipped.append(pid)
            log.warning("debrief: no usable questionnaire for %s", pid)
            continue
        result.ratings.append({"person": pid, **row})
    if people and not result.ratings:
        log.error("debrief: no questionnaire could be used, so nothing was written or applied")
        result.ok = False
        return result
    _write_jsonl(run_dir / "debrief.jsonl", result.ratings)

    # 2. hearsay, only when people talked about the robot to each other in this run
    if meta.get("word_of_mouth"):
        result.hearsay = await _hearsay(gateway, says, set(people), robot_id)
    events = sorted([*exps, *result.hearsay], key=lambda e: (e.t, e.person, e.kind))
    result.experiences = events

    # 3. trust: events in time order, a seeded d20 for each, then the end of the run
    seed = meta.get("project_seed", meta.get("seed", 0))
    run_index = int(meta.get("run_index") or 0)
    after_states, before_states, trust_rows = [], [], []
    for pid in people:
        inst = instances.get(pid)
        key = inst.memory_key if inst else pid
        info = (meta.get("people") or {}).get(pid) or {}
        role = info.get("preset_role") or info.get("role") or (inst.role if inst else "visitor")
        # with --force the run starts again from the states it started from the first time
        state = (stored_before or {}).get(key) or store.get(key, robot_id, role_defaults(roles, role))
        before = state
        before_states.append(state)
        k = 0
        mine = [e for e in events if e.person == pid]
        for e in mine:
            k += 1
            state = apply_event(state, e, roll_d20(int(seed), run_index, pid, k))
        state = end_of_run(state, clean=not any(e.kind == "failure" for e in mine))
        after_states.append(state)
        trust_rows.append({"person": pid, "robot_id": robot_id, "before": round(before.trust, 4),
                           "after": round(state.trust, 4), "stage_before": stage_for(before.trust),
                           "stage_after": state.stage})
    store.put_many(after_states, mark_run=run_id, before=before_states)  # trust and marker, one transaction
    result.trust = trust_rows
    _write_jsonl(run_dir / "trust.jsonl", trust_rows)
    RunLog(run_dir).write_meta({**meta, "debriefed": True})
    return result


POST_BUDGET_USD = 2.0  # the most a debrief and report may spend when the run did not set its own (post_budget_usd)


def make_post_gateway(run_dir: "str | Path") -> Gateway:
    """The gateway for the debrief and the report, in the CLI process (the user's key from the keyring).
    Models and provider are the run's own. Each call is logged to the run's tokens.jsonl as person
    "debrief" (questionnaire, word of mouth) or "report" (narrative), with the subject in "subject"."""
    from ..bench.metrics import RunData
    from ..config.loader import default_models
    from ..config.models import ModelRoles
    from ..llm.gateway import Budget, make_gateway
    from ..runlog import RunLog

    run_dir = Path(run_dir)
    d = RunData.load(run_dir)
    provider = d.meta.get("provider") or "claude"
    models = d.meta.get("models")
    roles = ModelRoles(**models) if isinstance(models, dict) and models else default_models(provider)
    if provider == "openai":
        roles = roles.model_copy(update={"report": roles.decide})
    runlog, t_end = RunLog(run_dir), d.end_t

    def on_usage(req, res) -> None:
        runlog.write("tokens", {
            "t": t_end, "person": "report" if req.role == "report" else "debrief", "subject": req.person,
            "role": req.role, "model": res.model, "ok": res.ok, "error": res.error, "status": res.status,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cache_read_tokens": res.cache_read_tokens, "cache_write_tokens": res.cache_write_tokens,
            "cost_usd": res.cost_usd, "latency_s": res.latency_s})

    return make_gateway(provider, roles, Budget(float(d.meta.get("post_budget_usd") or POST_BUDGET_USD)), on_usage)
