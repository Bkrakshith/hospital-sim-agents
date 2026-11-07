"""An intent the agent node does not send (it repeats the one still running) is not recorded as the mind's current
intent: the mind keeps the running one, so it sees it end."""
import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Intent, Nudge, Observation, Status
from wardzero_agents.llm.gateway import LLMRequest, LLMResult
from wardzero_agents.mind.runtime import Mind, MindDeps
from wardzero_agents.ros.agent_node import AgentService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


class Out:
    def __init__(self):
        self.intents, self.says = [], []

    def publish_intent(self, i):
        self.intents.append(i)

    def publish_say(self, s):
        self.says.append(s)


class QuietReflex:
    def __init__(self, inst, seed):
        pass

    def tick(self, t, status, heard, idle):
        return []

    def start_bout(self, t, d, sev):
        pass


def status(pid, state="idle", intent_id=None, kind=None, posture="standing", t=0.0):
    return Status(person=pid, t=t, x=0, y=0, yaw=0, posture=posture, intent_id=intent_id, intent_kind=kind,
                  progress=0.0, state=state)


def intent(pid, kind, target=None, id="i1"):
    return Intent(id=id, person=pid, t=1.0, kind=kind, target=target)


def test_publish_intent_returns_the_intent_in_effect(tmp_path):
    s = AgentService(load_scenario(V1), RunLog(tmp_path), Out(), lambda inst, eff: None, reflex_factory=QuietReflex)
    assert s.publish_intent(intent("P5", "walk_to", "toilets", id="a")) == "a"
    s.on_status(status("P5", "running", "a", "walk_to", "walking"))
    assert s.publish_intent(intent("P5", "walk_to", "toilets", id="b")) == "a"  # dropped: "a" goes on
    assert s.publish_intent(intent("P5", "walk_to", "entrance", id="c")) == "c"


async def test_a_dropped_repeat_is_not_the_current_intent(tmp_path):
    """QA (evacuation_guide): a person who chose "follow g1" again while following was recorded as following under
    the dropped intent's id, which never ran; their situation read "follow g1 (sent)" for ever, so they never saw the
    follow end and chose it again after the guide was over."""
    loaded = load_scenario(V1)
    target = next(iter(loaded.area.landmarks))
    requests = []

    class Gateway:
        async def complete(self, req: LLMRequest) -> LLMResult:
            requests.append(req)
            return LLMResult(True, {"intent": {"kind": "walk_to", "target": target, "text": None}, "say": None,
                                    "addressee": None, "mood": "calm", "patience_delta": 0.0, "reason": "t"},
                             None, "fake")

    def mind_factory(inst, effects):
        return Mind(MindDeps(inst=inst, loaded=loaded, gateway=Gateway(), kb=None, ltm=None, run_id="r",
                             run_index=1, effects=effects, runlog=effects.runlog, checkpointer=MemorySaver()))

    out = Out()
    svc = AgentService(loaded, RunLog(tmp_path), out, mind_factory, reflex_factory=QuietReflex)
    p5 = svc.people["P5"].mind
    poke = [Nudge(person="P5", t=0.0, card="c", message="You feel restless")]

    svc.on_status(status("P5", t=0.0))
    await p5.on_tick(0.0, Observation(person="P5", t=0.0), svc.people["P5"].status, poke)
    await p5.drain()
    sent = [i for i in out.intents if i.person == "P5"]
    assert len(sent) == 1
    first = sent[0].id

    running = status("P5", "running", first, "walk_to", "walking", t=10.0)
    svc.on_status(running)
    await p5.on_tick(10.0, Observation(person="P5", t=10.0), running, poke)
    await p5.drain()
    assert [i.id for i in out.intents if i.person == "P5"] == [first]  # the repeat was not sent
    assert (await p5.state())["current_intent"]["id"] == first

    done = status("P5", "done", first, "walk_to", "standing", t=20.0)
    svc.on_status(done)
    await p5.on_tick(20.0, Observation(person="P5", t=20.0), done, poke)
    await p5.drain()
    assert f"Current activity: walk_to {target} (done)." in requests[-1].user
    await svc.aclose()


async def test_effects_without_a_return_value_still_record_the_new_intent(tmp_path):
    """Effects that return nothing (older callers, tests) mean the intent was sent."""
    loaded = load_scenario(V1)
    inst = next(i for i in loaded.instances if i.person_id == "P5")
    from wardzero_agents.llm.fake import FakeGateway

    class Effects:
        def __init__(self):
            self.intents = []

        def publish_intent(self, i):
            self.intents.append(i)

        def publish_say(self, s):
            pass

    class Log:
        def write(self, stream, rec):
            pass

    eff = Effects()
    m = Mind(MindDeps(inst=inst, loaded=loaded, gateway=FakeGateway(), kb=None, ltm=None, run_id="r", run_index=1,
                      effects=eff, runlog=Log(), checkpointer=MemorySaver()))
    await m.on_tick(0.0, Observation(person="P5", t=0.0), status("P5"), [])
    await m.drain()
    assert eff.intents and (await m.state())["current_intent"]["id"] == eff.intents[-1].id
    await m.aclose()
