import asyncio
import json
import sqlite3
from pathlib import Path
from typing import get_args

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Heard, IntentKind, Nudge, Observation, Seen, Status
from wardzero_agents.kb.chunker import Chunk
from wardzero_agents.kb.embed import HashEmbedder
from wardzero_agents.kb.index import KnowledgeBase
from wardzero_agents.llm.fake import FakeGateway, FakeRefusal, default_policy
from wardzero_agents.llm.gateway import LLMRequest, LLMResult
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.mind.graph import validate_decision
from wardzero_agents.mind.prompts import (
    DECISION_SCHEMA, SUMMARY_SCHEMA, SYSTEM_FIXED, persona_card, render_situation)
from wardzero_agents.mind.runtime import Mind, MindDeps
from wardzero_agents.mind.triggers import TriggerGate

AGENTS_ROOT = Path(__file__).resolve().parents[1]
V1 = AGENTS_ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"
LOADED = load_scenario(V1)
INST = {i.person_id: i for i in LOADED.instances}
P1, P5 = INST["P1"], INST["P5"]
RUN = "run-test"


# ---------- helpers ----------

def assert_strict(schema, path="$"):
    t = schema.get("type")
    types = t if isinstance(t, list) else [t]
    if "object" in types:
        props = schema["properties"]
        assert schema.get("additionalProperties") is False, path
        assert sorted(schema["required"]) == sorted(props), path
        for k, v in props.items():
            assert_strict(v, f"{path}.{k}")
    if "array" in types:
        assert_strict(schema["items"], f"{path}[]")
    for i, alt in enumerate(schema.get("anyOf", [])):
        assert_strict(alt, f"{path}|{i}")


def heard(kind, src, text, addressed=False, db=60.0):
    return Heard(source_kind=kind, source_id=src, text=text, received_db=db, partial=False,
                 addressed_to_me=addressed)


def seen(kind, id_, dist=3.0, bearing=0.0, detail=None):
    return Seen(kind=kind, id=id_, distance=dist, bearing_deg=bearing, detail=detail)


def obs(person="P5", t=0.0, visible=(), heard=()):
    return Observation(person=person, t=t, visible=list(visible), heard=list(heard))


def status(person="P5", t=0.0, posture="seated", state="idle", intent_id=None, intent_kind=None):
    return Status(person=person, t=t, x=0.0, y=0.0, yaw=0.0, posture=posture, intent_id=intent_id,
                  intent_kind=intent_kind, progress=0.0, state=state)


def nudge(person, t, message, card="test_card"):
    return Nudge(person=person, t=t, card=card, message=message)


def obs_addressed(text, person="P5", t=0.0):
    return obs(person, t, visible=[seen("robot", "g1", 1.5)], heard=[heard("robot", "g1", text, True)])


class RecordingEffects:
    def __init__(self):
        self.intents, self.says = [], []

    def publish_intent(self, intent):
        self.intents.append(intent)

    def publish_say(self, say):
        self.says.append(say)


class RecordingRunLog:
    def __init__(self):
        self.records = []

    def write(self, stream, record):
        self.records.append((stream, record))

    def stream(self, name):
        return [r for s, r in self.records if s == name]

    def last(self, name):
        return self.stream(name)[-1]


def decision(kind, target=None, say=None, text=None, addressee=None, mood="calm", delta=0.0):
    return {"intent": {"kind": kind, "target": target, "text": text}, "say": say,
            "addressee": addressee, "mood": mood, "patience_delta": delta, "reason": "test"}


def make_kb():
    texts = ["Outpatients usually wait 15 to 45 minutes after checking in at reception.",
             "Anxious first-time attenders often ask staff where to check in."]
    chunks = [Chunk(f"f#{i}", "f", [], t, []) for i, t in enumerate(texts)]
    return KnowledgeBase.build(chunks, HashEmbedder())


@pytest.fixture
def ltm(tmp_path):
    m = LongTermMemory(tmp_path / "memory" / "longterm.sqlite", HashEmbedder())
    m.begin_run(RUN)
    return m


@pytest.fixture
def build(ltm):
    """Build a Mind with recording effects and runlog. Tests close it with `await m.aclose()`."""

    def _build(inst, gateway, run_dir=None, checkpointer="memory", memory=None, effects=None,
               runlog=None):
        effects, runlog = effects or RecordingEffects(), runlog or RecordingRunLog()
        deps = MindDeps(inst=inst, loaded=LOADED, gateway=gateway, kb=make_kb(), ltm=memory or ltm,
                        run_id=RUN, run_index=1, effects=effects, runlog=runlog, run_dir=run_dir,
                        checkpointer=MemorySaver() if checkpointer == "memory" else checkpointer)
        return Mind(deps), effects, runlog

    return _build


SEATED = status()


# ---------- prompts ----------

def test_schema_is_strict():
    assert_strict(DECISION_SCHEMA)
    assert_strict(SUMMARY_SCHEMA)


def test_schema_enums():
    props = DECISION_SCHEMA["properties"]
    assert props["intent"]["properties"]["kind"]["enum"] == list(get_args(IntentKind))
    assert len(props["intent"]["properties"]["kind"]["enum"]) == 14
    assert props["mood"]["enum"] == ["calm", "anxious", "frustrated", "distressed"]
    assert props["patience_delta"]["type"] == "number"
    for nullable in (props["say"], props["addressee"], props["intent"]["properties"]["target"],
                     props["intent"]["properties"]["text"]):
        assert nullable == {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert "type" not in nullable  # no type arrays anywhere
    assert '"type": [' not in json.dumps(DECISION_SCHEMA) + json.dumps(SUMMARY_SCHEMA)
    mem = SUMMARY_SCHEMA["properties"]["memories"]["items"]["properties"]
    assert mem["text"]["type"] == "string" and mem["importance"]["type"] == "number"


def test_system_fixed_instructions():
    s = SYSTEM_FIXED.lower()
    for phrase in ("uk hospital", "main-entrance", "waiting area", "british english", "in character"):
        assert phrase in s


def test_persona_card_describes_person_and_places():
    card = persona_card(P5, LOADED.area)
    assert P5.display_name in card
    assert P5.goals[0] in card
    assert "clinic_corridor" in card and "Corridor to the outpatient clinics" in card
    assert P5.seat in card
    # stable across calls (it is the cached prompt block)
    assert card == persona_card(P5, LOADED.area)


def test_situation_marks_robot_speech():
    s = render_situation(P5, obs(heard=[heard("robot", "g1", "Clinics are down the corridor", addressed=True)]),
                         SEATED, ["addressed_speech"], [], 12.0)
    assert "ROBOT SAID TO YOU: Clinics are down the corridor" in s


def test_situation_lists_everything():
    o = obs(visible=[seen("person", "P3", 2.0, 30.0, "Margaret Hall"), seen("robot", "g1", 4.0),
                     seen("sign", "sign_clinics", 5.0, -10.0, "Outpatient clinics 1-6 →")],
            heard=[heard("robot", "g1", "Please take a seat", addressed=False),
                   heard("person", "P3", "Is that a robot?"), heard("cough", "P1", None)])
    s = render_situation(P5, o, SEATED, ["speech_nearby", "nudge"],
                         [nudge("P5", 0, "You need the toilet")], 12.0,
                         mood="anxious", patience_left=0.4, current_intent={"kind": "use_phone"})
    lines = s.splitlines()
    assert "ROBOT SAID NEARBY: Please take a seat" in lines
    assert "NUDGE: You need the toilet" in lines
    assert "P3" in s and "Margaret Hall" in s
    assert "sign_clinics" in s and "Outpatient clinics 1-6 →" in s
    assert "g1" in s
    assert "Is that a robot?" in s
    assert "cough" in s.lower()
    assert "seated" in s and "use_phone" in s
    assert "12 minutes" in s
    assert "anxious" in s and "40%" in s
    # nudges come last so nothing after them is read as part of the nudge
    assert lines[-1].startswith("NUDGE:")


# ---------- triggers ----------

def test_gate_debounce():
    g = TriggerGate()
    assert "new_person" in g.check(10, obs(visible=[seen("person", "P3")]), SEATED, [], None)
    g.mark(10)
    last = obs(visible=[seen("person", "P3")])
    assert g.check(12, obs(visible=[seen("person", "P3"), seen("person", "P4")]), SEATED, [], last) == []
    r = g.check(12, obs_addressed("Hello"), SEATED, [], last)
    assert "addressed_speech" in r
    assert "nudge" in g.check(12, obs(), SEATED, [nudge("P5", 12, "x")], last)
    # after the interval the new person counts again
    assert "new_person" in g.check(15, obs(visible=[seen("person", "P3"), seen("person", "P4")]),
                                   SEATED, [], last)


def test_gate_reasons():
    g = TriggerGate()
    g.mark(0)
    last = obs(visible=[seen("robot", "g1", 5.0)])
    assert "robot_near" in g.check(10, obs(visible=[seen("robot", "g1", 1.5)]), SEATED, [], last)
    assert "robot_near" not in g.check(10, obs(visible=[seen("robot", "g1", 1.5)]), SEATED, [],
                                       obs(visible=[seen("robot", "g1", 1.8)]))
    assert "new_sign" in g.check(10, obs(visible=[seen("sign", "sign_wait")]), SEATED, [], obs())
    assert "cough_nearby" in g.check(10, obs(heard=[heard("cough", "P1", None)]), SEATED, [], obs())
    assert "speech_nearby" in g.check(10, obs(heard=[heard("person", "P3", "hi")]), SEATED, [], obs())
    assert g.check(10, obs(), SEATED, [], obs()) == []
    done = status(state="done", intent_id="abc", intent_kind="walk_to")
    assert "intent_done" in g.check(10, obs(), done, [], obs())
    g.mark(10)
    assert "intent_done" not in g.check(20, obs(), done, [], obs())  # reported once per intent
    failed = status(state="failed", intent_id="def", intent_kind="walk_to")
    assert "intent_failed" in g.check(20, obs(), failed, [], obs())
    g.mark(20)
    assert g.check(199, obs(), SEATED, [], obs()) == []
    assert g.check(200, obs(), SEATED, [], obs()) == ["patience_timer"]


# ---------- validation ----------

ROSTER = [i.person_id for i in LOADED.instances]


def v(d):
    return validate_decision(d, LOADED.area, LOADED.signs, ROSTER)


def test_validate_decision():
    assert v(decision("walk_to", "clinic_corridor"))["intent"]["target"] == "clinic_corridor"
    assert v(decision("sit_down", "D2")) is not None
    assert v(decision("read_sign", "sign_clinics")) is not None
    assert v(decision("approach", "g1")) is not None
    assert v(decision("look_at", "P3")) is not None
    assert v(decision("walk_to", "narnia")) is None
    assert v(decision("walk_to", None)) is None
    assert v(decision("approach", "")) is None
    assert v(decision("talk", "P3", text="")) is None
    assert v(decision("talk", "P3", text="Hello")) is not None
    assert v(decision("leave"))["intent"]["target"] == "entrance"
    assert v(decision("fly")) is None
    assert v(decision("sit_idle", mood="ecstatic")) is None
    assert v({"intent": "sit_idle"}) is None
    # empty strings from models are treated as null
    out = v(decision("sit_idle", target="", say="", addressee=""))
    assert out["intent"]["target"] is None and out["say"] is None and out["addressee"] is None


# ---------- runtime ----------

async def test_robot_speech_leads_to_walk_and_thanks(build):
    m, effects, runlog = build(P5, FakeGateway())
    await m.on_tick(30, obs_addressed("clinic_corridor is down there", t=30), SEATED, [])
    await m.drain()
    assert effects.intents[-1].kind == "walk_to"
    assert effects.intents[-1].target == "clinic_corridor"
    assert effects.says[-1].text == "Thank you"
    i = effects.intents[-1]
    assert i.person == "P5" and i.t == 30 and len(i.id) == 32
    rec = runlog.last("decisions")
    assert rec["error"] is None and rec["person"] == "P5" and rec["t"] == 30
    assert "addressed_speech" in rec["triggers"]
    assert "ROBOT SAID TO YOU" in rec["situation"]
    assert rec["result"]["intent"]["kind"] == "walk_to"
    assert len(rec["prompt_sha256"]) == 16 and rec["model"] == "fake"
    assert {"reason", "latency_s"} <= set(rec)
    await m.aclose()


async def test_failed_decision_keeps_intent(build):
    def refuse(req):
        raise FakeRefusal()

    m, effects, runlog = build(P1, FakeGateway(policy=refuse))
    await m.on_tick(5, obs(person="P1", visible=[seen("person", "P2")]), status(person="P1"), [])
    await m.drain()
    assert effects.intents == [] and effects.says == []
    assert runlog.last("decisions")["error"] == "refusal"
    await m.aclose()


async def test_invalid_target_rejected(build):
    m, effects, runlog = build(P1, FakeGateway(policy=lambda r: decision("walk_to", "narnia")))
    await m.on_tick(5, obs(person="P1"), status(person="P1"), [nudge("P1", 5, "go somewhere")])
    await m.drain()
    assert effects.intents == []
    rec = runlog.last("decisions")
    assert rec["error"] == "invalid_target" and rec["result"]["intent"]["target"] == "narnia"
    await m.aclose()


async def test_nudge_triggers_decision_inside_debounce(build):
    m, effects, runlog = build(P5, FakeGateway())
    await m.on_tick(10, obs(visible=[seen("person", "P3")]), SEATED, [])
    await m.drain()
    assert len(runlog.stream("decisions")) == 1
    await m.on_tick(11, obs(visible=[seen("person", "P3"), seen("person", "P4")]), SEATED, [])
    await m.drain()
    assert len(runlog.stream("decisions")) == 1  # debounced
    await m.on_tick(12, obs(visible=[seen("person", "P3")]), SEATED,
                    [nudge("P5", 12, "You want help finding where to check in")])
    await m.drain()
    assert len(runlog.stream("decisions")) == 2
    assert effects.intents[-1].kind == "approach" and effects.intents[-1].target == "g1"
    assert effects.says[-1].text == "Excuse me, can you help us?"
    await m.aclose()


async def wait_for(cond, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "timed out"
        await asyncio.sleep(0.005)


class GatedGateway:
    """Holds every decide call until released, recording the requests it saw."""

    def __init__(self):
        self.release = asyncio.Event()
        self.requests = []

    async def complete(self, req: LLMRequest) -> LLMResult:
        self.requests.append(req)
        if req.role == "decide":
            await self.release.wait()
        return LLMResult(True, default_policy(req), None, "gated")


async def test_pending_ignores_triggers_but_keeps_nudges(build):
    gw = GatedGateway()
    m, effects, runlog = build(P5, gw)
    await m.on_tick(10, obs_addressed("Hello there", t=10), SEATED, [])
    await wait_for(lambda: gw.requests)  # the first decision is now waiting on the gateway
    assert m.pending
    await m.on_tick(11, obs_addressed("Hello again", t=11), SEATED, [nudge("P5", 11, "Ask for help")])
    assert len([r for r in gw.requests if r.role == "decide"]) == 1  # still one in flight
    gw.release.set()
    await m.drain()
    assert not m.pending
    # the held nudge and robot speech drive a follow-up decision as soon as the first one ends
    decides = [r for r in gw.requests if r.role == "decide"]
    assert len(decides) == 2
    assert "NUDGE: Ask for help" in decides[-1].user
    assert "ROBOT SAID TO YOU: Hello again" in decides[-1].user
    assert [r["t"] for r in runlog.stream("decisions")] == [10, 11]
    await m.aclose()


async def test_robot_speech_during_pending_is_not_lost(build):
    gw = GatedGateway()
    m, effects, runlog = build(P5, gw)
    await m.on_tick(10, obs(), SEATED, [nudge("P5", 10, "You feel restless")])
    await wait_for(lambda: gw.requests)  # the first decision is now waiting on the gateway
    assert m.pending
    await m.on_tick(11, obs_addressed("clinic_corridor is down there", t=11), SEATED, [])
    await m.on_tick(12, obs(), SEATED, [])  # the speech is gone from later ticks
    gw.release.set()
    await m.drain()
    rec = runlog.last("decisions")
    assert "ROBOT SAID TO YOU:" in rec["situation"] and "addressed_speech" in rec["triggers"]
    assert effects.intents[-1].kind == "walk_to" and effects.intents[-1].target == "clinic_corridor"
    assert effects.says[-1].text == "Thank you"
    await m.aclose()


async def test_nearby_robot_speech_held_through_debounce(build):
    gw = GatedGateway()
    m, effects, runlog = build(P5, gw)
    await m.on_tick(10, obs(), SEATED, [nudge("P5", 10, "x")])
    await wait_for(lambda: gw.requests)  # the first decision is now waiting on the gateway
    await m.on_tick(11, obs(heard=[heard("robot", "g1", "Please take a seat")]), SEATED, [])
    gw.release.set()
    await m.drain()
    assert len(runlog.stream("decisions")) == 1  # nearby speech waits out the 4 s debounce
    await m.on_tick(13, obs(), SEATED, [])
    assert not m.pending
    await m.on_tick(14, obs(), SEATED, [])
    await m.drain()
    rec = runlog.last("decisions")
    assert len(runlog.stream("decisions")) == 2 and rec["t"] == 14
    assert "ROBOT SAID NEARBY: Please take a seat" in rec["situation"]
    await m.aclose()


async def test_mood_and_patience_clamped(build):
    seq = iter([decision("sit_idle", mood="frustrated", delta=-0.7),
                decision("sit_idle", mood="distressed", delta=-0.7),
                decision("sit_idle", mood="calm", delta=5.0)])
    m, effects, runlog = build(P1, FakeGateway(policy=lambda r: next(seq)))
    st = status(person="P1")
    for t, p in [(10, 0.3), (20, 0.0), (30, 1.0)]:
        await m.on_tick(t, obs(person="P1"), st, [nudge("P1", t, "time passes")])
        await m.drain()
        s = await m.state()
        assert s["patience_left"] == pytest.approx(p)
    assert s["mood"] == "calm"
    assert s["current_intent"]["kind"] == "sit_idle"
    await m.aclose()


async def test_remember_and_recall_feed_the_prompt(build, ltm):
    ltm.add(P5.memory_key, RUN, 0.0, "fact", "Reception is where you check in for outpatient clinics", 0.9)
    seen_users = []
    seen_reqs = []

    def policy(req):
        seen_users.append(req.full_user)
        seen_reqs.append(req)
        return decision("use_phone")

    m, effects, runlog = build(P5, FakeGateway(policy=policy))
    await m.on_tick(10, obs(), SEATED, [nudge("P5", 10, "Where do you check in?")])
    await m.drain()
    await m.on_tick(20, obs(), SEATED, [nudge("P5", 20, "Still waiting")])
    await m.drain()
    assert "Reception is where you check in" in seen_users[0]
    assert "checking in at reception" in seen_users[0]  # kb hit
    s = await m.state()
    assert len(s["short_term"]) == 2 and "use_phone" in s["short_term"][0]
    assert "use_phone" in seen_users[1]  # recent events reach the next prompt
    assert seen_reqs[0].history == []
    assert seen_reqs[1].history[0].startswith("What has happened so far")
    assert "What has happened so far" not in seen_reqs[1].user  # history is its own (cached) blocks
    await m.aclose()


async def test_summarise_writes_ltm(build, ltm):
    summaries = []

    def policy(req):
        if req.role == "summarise":
            summaries.append(req)
            return {"memories": [{"text": "Waited a long time and fidgeted a lot near the robot",
                                  "importance": 0.7}]}
        return decision("fidget")

    m, effects, runlog = build(P1, FakeGateway(policy=policy))
    st = status(person="P1")
    for i in range(30):
        t = 10.0 + i  # well inside 300 s, so the 30-line cap is what fires it
        await m.on_tick(t, obs(person="P1"), st, [nudge("P1", t, f"event {i}")])
        await m.drain()
    assert len(summaries) == 1 and summaries[0].schema == SUMMARY_SCHEMA
    hits = await asyncio.to_thread(ltm.search, P1.memory_key, "waited fidgeted robot", 5, run_index=1)
    assert any(h.kind == "episode" and "fidgeted" in h.text for h in hits)
    s = await m.state()
    assert len(s["short_term"]) == 5
    assert s["last_summary_t"] == pytest.approx(39.0)
    await m.aclose()


async def test_summarise_on_timer(build, ltm):
    calls = []

    def policy(req):
        if req.role == "summarise":
            calls.append(req.user)
            return {"memories": [{"text": "Sat quietly for five minutes", "importance": 2.0}]}
        return decision("sit_idle")

    m, effects, runlog = build(P1, FakeGateway(policy=policy))
    st = status(person="P1")
    await m.on_tick(10, obs(person="P1"), st, [nudge("P1", 10, "a")])
    await m.drain()
    assert calls == []
    await m.on_tick(320, obs(person="P1"), st, [nudge("P1", 320, "b")])
    await m.drain()
    assert len(calls) == 1
    hits = await asyncio.to_thread(ltm.search, P1.memory_key, "sat quietly", 5, run_index=1)
    assert any(h.kind == "episode" and h.importance == 1.0 for h in hits)  # importance clamped
    await m.aclose()


async def test_sqlite_checkpointer_persists_short_term(build, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    m, effects, runlog = build(P5, FakeGateway(), run_dir=run_dir, checkpointer=None)
    await m.on_tick(10, obs(), SEATED, [nudge("P5", 10, "x")])
    await m.drain()
    await m.aclose()
    db = run_dir / "shortterm.sqlite"
    assert db.exists()
    with sqlite3.connect(db) as c:
        threads = {r[0] for r in c.execute("select distinct thread_id from checkpoints")}
    assert threads == {"P5"}
    # a fresh mind on the same run folder picks the short-term state back up
    m2, _, _ = build(P5, FakeGateway(), run_dir=run_dir, checkpointer=None)
    await m2.on_tick(20, obs(), SEATED, [nudge("P5", 20, "y")])
    await m2.drain()
    s = await m2.state()
    assert len(s["short_term"]) == 2
    await m2.aclose()


# ---------- fix round 1 ----------

class FailingAddMemory(LongTermMemory):
    def add(self, *a, **kw):
        raise sqlite3.OperationalError("database is locked")


def summarising(memories=None, fail=False):
    calls = []

    def policy(req):
        if req.role == "summarise":
            calls.append(req)
            if fail:
                raise FakeRefusal()
            return {"memories": memories or [{"text": "Waited and fidgeted", "importance": 0.5}]}
        return decision("fidget")

    return calls, policy


async def run_decisions(m, person, times):
    for t in times:
        await m.on_tick(t, obs(person=person), status(person=person), [nudge(person, t, f"event {t}")])
        await m.drain()


async def test_ltm_write_failure_still_trims_and_does_not_repeat(build, tmp_path):
    bad = FailingAddMemory(tmp_path / "bad.sqlite", HashEmbedder())
    bad.begin_run(RUN)
    calls, policy = summarising()
    m, effects, runlog = build(P1, FakeGateway(policy=policy), memory=bad)
    await run_decisions(m, "P1", [10.0 + i for i in range(30)])
    assert len(calls) == 1
    rec = runlog.last("summaries")
    assert rec["error"] == "ltm_write_failed" and rec["exception"] == "OperationalError"
    assert "locked" not in str(rec)  # exception type only
    assert len((await m.state())["short_term"]) == 5
    before = len(runlog.stream("decisions"))
    await run_decisions(m, "P1", [41.0])
    assert len(calls) == 1  # no second summarise call
    assert len(runlog.stream("decisions")) == before + 1
    assert all(r["error"] != "mind_error" for r in runlog.stream("decisions"))
    await m.aclose()


async def test_summary_llm_failure_keeps_events_and_backs_off(build):
    calls, policy = summarising(fail=True)
    m, effects, runlog = build(P1, FakeGateway(policy=policy))
    await run_decisions(m, "P1", [10.0 + i for i in range(30)])  # line 30 at t=39
    assert len(calls) == 1 and runlog.last("summaries")["error"] == "refusal"
    s = await m.state()
    assert len(s["short_term"]) == 30 and s["last_summary_t"] == pytest.approx(39.0)
    await run_decisions(m, "P1", [40.0 + i for i in range(40)])  # inside the 60 s back-off
    assert len(calls) == 1
    assert len((await m.state())["short_term"]) == 60  # capped
    await run_decisions(m, "P1", [99.0])  # back-off over: retried
    assert len(calls) == 2
    await m.aclose()


async def test_summary_retry_success_trims(build):
    state = {"fail": True}
    calls = []

    def policy(req):
        if req.role == "summarise":
            calls.append(req)
            if state["fail"]:
                raise FakeRefusal()
            return {"memories": [{"text": "a", "importance": 0.5}, {"text": "b", "importance": 0.5},
                                 {"text": "c", "importance": 0.5}, {"text": "d", "importance": 0.5}]}
        return decision("fidget")

    m, effects, runlog = build(P1, FakeGateway(policy=policy))
    await run_decisions(m, "P1", [10.0 + i for i in range(30)])
    state["fail"] = False
    await run_decisions(m, "P1", [100.0])
    assert len(calls) == 2
    assert len(runlog.last("summaries")["memories"]) == 3  # MAX_MEMORIES
    assert len((await m.state())["short_term"]) == 5
    await m.aclose()


def test_summary_limit_consistent():
    from wardzero_agents.mind.prompts import MAX_MEMORIES, SUMMARY_FIXED
    assert MAX_MEMORIES == 3 and "at most 3 " in SUMMARY_FIXED


async def test_background_is_behaviour_only(build):
    users = []

    def policy(req):
        users.append(req.user)
        return decision("sit_idle")

    m, effects, runlog = build(P5, FakeGateway(policy=policy))
    await run_decisions(m, "P5", [10.0])
    assert ("Background on how people in this situation typically behave (for your behaviour only; "
            "you do not know these as facts and never quote figures or studies):") in users[0]
    assert "Things you know" not in users[0]
    assert "Never quote statistics, studies or figures" in SYSTEM_FIXED
    assert "ordinary member of the public" in SYSTEM_FIXED
    await m.aclose()


def test_situation_reports_position_from_status():
    d2 = next(s for s in LOADED.area.seats if s.id == "D2")
    a1 = next(s for s in LOADED.area.seats if s.id == "A1")
    at_a1 = Status(person="P5", t=0, x=a1.x + 0.1, y=a1.y, yaw=0, posture="seated", progress=0, state="idle")
    s = render_situation(P5, obs(), at_a1, [], [], 1.0, area=LOADED.area)
    assert "seated at seat A1" in s and "D2" not in s
    rf = LOADED.area.landmarks["reception_front"]
    standing = Status(person="P5", t=0, x=rf.x + 0.2, y=rf.y, yaw=0, posture="standing", progress=0,
                      state="idle")
    s = render_situation(P5, obs(), standing, [], [], 1.0, area=LOADED.area)
    assert "standing at (1.2, 0.6)" in s and "reception_front" in s
    assert P5.seat == "D2" and d2 is not None


def test_persona_card_has_only_starting_seat():
    card = persona_card(P5, LOADED.area)
    assert "You arrived and sat at seat D2." in card
    assert "Your seat" not in card


class FlakyEffects(RecordingEffects):
    def __init__(self):
        super().__init__()
        self.fail_next = True

    def publish_intent(self, intent):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("publisher gone")
        super().publish_intent(intent)


async def test_node_exception_logs_one_mind_error(build):
    eff = FlakyEffects()
    m, effects, runlog = build(P5, FakeGateway(), effects=eff)
    await run_decisions(m, "P5", [10.0])
    errs = [r for r in runlog.stream("decisions") if r["error"] == "mind_error"]
    assert len(errs) == 1 and errs[0]["exception"] == "RuntimeError" and not m.pending
    await run_decisions(m, "P5", [20.0])
    assert len(effects.intents) == 1
    assert len([r for r in runlog.stream("decisions") if r["error"] == "mind_error"]) == 1
    await m.aclose()


class BrokenRunLog(RecordingRunLog):
    def write(self, stream, record):
        raise OSError("disk full")


async def test_runlog_failure_does_not_break_mind(build):
    calls = []

    def policy(req):
        calls.append(req)
        return decision("sit_idle")

    m, effects, runlog = build(P5, FakeGateway(policy=policy), runlog=BrokenRunLog())
    await run_decisions(m, "P5", [10.0])
    assert not m.pending
    await run_decisions(m, "P5", [20.0])
    assert len(calls) == 2 and not m.pending
    await m.aclose()


async def test_memory_saver_fallback_warns(build, caplog):
    m, effects, runlog = build(P5, FakeGateway(), checkpointer=None)
    with caplog.at_level("WARNING", logger="wardzero_agents.mind.runtime"):
        await run_decisions(m, "P5", [10.0, 20.0])
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1 and "P5" in warns[0].getMessage()
    await m.aclose()


# ---------- final review fixes ----------

async def test_newcomer_can_be_targeted_and_addressed(build):
    """A person who arrives after the mind was built is a valid target and addressee."""
    from wardzero_agents.world.stub_world import spawned_instance
    loaded = load_scenario(V1)  # a private copy: this test adds a newcomer to it
    gw = FakeGateway(policy=lambda r: decision("approach", "N1", say="Hello", addressee="N1"))
    effects, runlog = RecordingEffects(), RecordingRunLog()
    deps = MindDeps(inst=loaded.instances[0], loaded=loaded, gateway=gw, kb=None, ltm=None, run_id=RUN,
                    run_index=1, effects=effects, runlog=runlog, checkpointer=MemorySaver())
    m = Mind(deps)
    await m.on_tick(5, obs(person="P1"), status(person="P1"), [nudge("P1", 5, "say hello")])
    await m.drain()
    assert runlog.last("decisions")["error"] == "invalid_target"  # N1 is not here yet
    loaded.instances.append(spawned_instance(loaded, "N1", "ward_visitor", arrive_at_s=40.0))
    await m.on_tick(45, obs(person="P1", t=45, visible=[seen("person", "N1")]), status(person="P1"),
                    [nudge("P1", 45, "say hello")])
    await m.drain()
    assert runlog.last("decisions")["error"] is None
    assert effects.intents[-1].target == "N1" and effects.intents[-1].addressee == "N1"
    assert effects.says[-1].addressee == "N1"
    await m.aclose()


async def test_intent_stamped_with_clock_at_release(build):
    gw = GatedGateway()
    clock = {"t": 10.0}
    effects, runlog = RecordingEffects(), RecordingRunLog()
    deps = MindDeps(inst=P5, loaded=LOADED, gateway=gw, kb=None, ltm=None, run_id=RUN, run_index=1,
                    effects=effects, runlog=runlog, checkpointer=MemorySaver(), clock=lambda: clock["t"])
    m = Mind(deps)
    await m.on_tick(10, obs_addressed("clinic_corridor is down there", t=10), SEATED, [])
    await wait_for(lambda: gw.requests)
    clock["t"] = 13.5  # the LLM took 3.5 s of sim time
    gw.release.set()
    await m.drain()
    assert effects.intents[-1].t == 13.5 and effects.says[-1].t == 13.5
    assert runlog.last("decisions")["t"] == 10  # the decision record keeps its start time
    st = await m.state()
    assert st["current_intent"]["t"] == 13.5
    await m.aclose()
