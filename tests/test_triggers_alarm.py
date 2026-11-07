"""People perceive a fire alarm and PA announcements: they are urgent and are never lost while a decision runs."""
import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Heard, Observation, Status
from wardzero_agents.llm.fake import default_policy
from wardzero_agents.llm.gateway import LLMRequest, LLMResult
from wardzero_agents.mind.runtime import Mind, MindDeps, _held_while_pending
from wardzero_agents.mind.triggers import TriggerGate

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
LOADED = load_scenario(V1)
P5 = next(i for i in LOADED.instances if i.person_id == "P5")
SEATED = Status(person="P5", t=0.0, x=0.0, y=0.0, yaw=0.0, posture="seated", progress=0.0, state="idle")
ALARM_TEXT = "Fire alarm. Please leave by the nearest safe exit."


def heard(kind, src, text=None):
    return Heard(source_kind=kind, source_id=src, text=text, received_db=90.0, partial=False, addressed_to_me=False)


def obs(t=0.0, heard_=()):
    return Observation(person="P5", t=t, heard=list(heard_))


ALARM = heard("alarm", "alarm", ALARM_TEXT)
PA = heard("pa", "pa", "Fire in the main lobby. Do not use the main entrance.")


class Effects:
    def __init__(self):
        self.intents, self.says = [], []

    def publish_intent(self, intent):
        self.intents.append(intent)

    def publish_say(self, say):
        self.says.append(say)


class RunLog:
    def __init__(self):
        self.records = []

    def write(self, stream, record):
        self.records.append((stream, record))

    def decisions(self):
        return [r for s, r in self.records if s == "decisions"]


class GatedGateway:
    """Holds every decide call until released."""

    def __init__(self, gated=True):
        self.release = asyncio.Event()
        if not gated:
            self.release.set()
        self.requests = []

    async def complete(self, req: LLMRequest) -> LLMResult:
        self.requests.append(req)
        if req.role == "decide":
            await self.release.wait()
        return LLMResult(True, default_policy(req), None, "gated")


def mind(gateway):
    runlog = RunLog()
    deps = MindDeps(inst=P5, loaded=LOADED, gateway=gateway, kb=None, ltm=None, run_id="r", run_index=1,
                    effects=Effects(), runlog=runlog, checkpointer=MemorySaver())
    return Mind(deps), runlog


async def wait_for(cond, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "timed out"
        await asyncio.sleep(0.005)


# ---- the gate ------------------------------------------------------------------------------
def test_an_alarm_is_an_urgent_reason():
    g = TriggerGate()
    g.mark(0.0)
    # 1 s after a decision, inside the 4 s debounce: the alarm still gets a decision at once
    assert g.check(1.0, obs(1.0, [ALARM]), SEATED, [], obs()) == ["alarm"]


def test_a_pa_announcement_is_urgent_and_keeps_its_speech_reason():
    g = TriggerGate()
    g.mark(0.0)
    assert g.check(1.0, obs(1.0, [PA]), SEATED, [], obs()) == ["speech_nearby"]
    # other speech nearby still waits out the debounce
    assert g.check(1.0, obs(1.0, [heard("person", "P3", "hi")]), SEATED, [], obs()) == []


def test_alarm_and_pa_are_held_while_a_decision_runs():
    assert _held_while_pending(ALARM)
    assert _held_while_pending(PA)
    assert not _held_while_pending(heard("person", "P3", "hi"))
    assert not _held_while_pending(heard("cough", "P1"))


# ---- the mind ------------------------------------------------------------------------------
async def test_alarm_during_a_pending_decision_reaches_the_next_one():
    """QA: the alarm sounded while a decision was in flight and no later decision ever said so."""
    gw = GatedGateway()
    m, runlog = mind(gw)
    await m.on_tick(0.9, obs(0.9), SEATED, [])  # "start": a decision goes out and waits on the gateway
    await wait_for(lambda: gw.requests)
    assert m.pending
    await m.on_tick(1.2, obs(1.2, [ALARM]), SEATED, [])  # heard once (the agent node passes each sound once)
    await m.on_tick(1.3, obs(1.3), SEATED, [])
    gw.release.set()
    await m.drain()
    decs = runlog.decisions()
    assert len(decs) == 2
    assert "alarm" in decs[-1]["triggers"]
    assert "An alarm is sounding." in decs[-1]["situation"]
    await m.aclose()


async def test_alarm_inside_the_debounce_gets_a_decision_at_once():
    gw = GatedGateway(gated=False)
    m, runlog = mind(gw)
    await m.on_tick(0.1, obs(0.1), SEATED, [])
    await m.drain()
    await m.on_tick(1.2, obs(1.2, [ALARM]), SEATED, [])
    await m.drain()
    decs = runlog.decisions()
    assert [d["t"] for d in decs] == [0.1, 1.2]
    assert "An alarm is sounding." in decs[-1]["situation"]
    await m.aclose()


async def test_pa_during_a_pending_decision_reaches_the_next_one():
    gw = GatedGateway()
    m, runlog = mind(gw)
    await m.on_tick(0.9, obs(0.9), SEATED, [])
    await wait_for(lambda: gw.requests)
    await m.on_tick(1.2, obs(1.2, [PA]), SEATED, [])
    await m.on_tick(1.3, obs(1.3), SEATED, [])
    gw.release.set()
    await m.drain()
    decs = runlog.decisions()
    assert len(decs) == 2
    assert "PA announcement: Fire in the main lobby." in decs[-1]["situation"]
    await m.aclose()
