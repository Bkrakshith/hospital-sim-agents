"""AgentService: the agent node without ROS."""
import json
from pathlib import Path

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Heard, Intent, Nudge, Observation, ReflexOverride, Roster, RosterEntry, Say, Status
from wardzero_agents.llm.gateway import LLMRequest, LLMResult
from wardzero_agents.ros.agent_node import AgentService, snapshot_name
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


class FakeMind:
    def __init__(self, inst, effects=None):
        self.inst = inst
        self.ticks = []
        self.pending = False
        self.closed = False

    async def on_tick(self, t, obs, status, nudges):
        self.ticks.append((t, obs, status, list(nudges)))

    async def drain(self):
        pass

    async def aclose(self):
        self.closed = True


class Out:
    def __init__(self):
        self.intents, self.says = [], []

    def publish_intent(self, i):
        self.intents.append(i)

    def publish_say(self, s):
        self.says.append(s)


class QuietReflex:
    def __init__(self, inst, seed):
        self.bouts = []
        self.heard = []

    def tick(self, t, status, heard, idle):
        self.heard.append(list(heard))
        return []

    def start_bout(self, t, d, sev):
        self.bouts.append((t, d, sev))


def make(tmp_path, reflex=QuietReflex):
    out = Out()
    loaded = load_scenario(V1)
    s = AgentService(loaded, RunLog(tmp_path), out, FakeMind, reflex_factory=reflex)
    return s, out


def lines(tmp_path, stream):
    p = tmp_path / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def status(pid, state="idle", intent_id=None, kind=None, posture="seated", t=0.0):
    return Status(person=pid, t=t, x=0, y=0, yaw=0, posture=posture, intent_id=intent_id,
                  intent_kind=kind, progress=0.0, state=state)


def heard(kind, src, text=None, db=60.0):
    return Heard(source_kind=kind, source_id=src, text=text, received_db=db, partial=False,
                 addressed_to_me=False)


def intent(pid, kind, target=None, text=None, id="i1", overlay=False):
    return Intent(id=id, person=pid, t=1.0, kind=kind, target=target, text=text, overlay=overlay)


def test_one_mind_per_cast_member_and_newcomers(tmp_path):
    s, _ = make(tmp_path)
    assert sorted(s.minds) == ["P1", "P2", "P3", "P4", "P5"]
    people = [RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
              for i in s.loaded.instances]
    new = s.on_roster(Roster(t=5, people=people + [RosterEntry(id="N1", persona="companion", display_name="X")]))
    assert new == ["N1"] and s.minds["N1"].inst.persona_id == "companion"
    assert any(i.person_id == "N1" for i in s.loaded.instances)
    assert s.on_roster(Roster(t=6, people=people)) == []


def test_dedupe_running_base_intent(tmp_path):
    s, out = make(tmp_path)
    s.publish_intent(intent("P5", "walk_to", "toilets", id="a"))
    s.on_status(status("P5", "running", "a", "walk_to", "walking"))
    s.publish_intent(intent("P5", "walk_to", "toilets", id="b"))   # same and running: dropped
    s.publish_intent(intent("P5", "walk_to", "entrance", id="c"))  # different target: sent
    assert [i.id for i in out.intents] == ["a", "c"]
    s.on_status(status("P5", "running", "c", "walk_to", "walking"))
    s.publish_intent(intent("P5", "cough", overlay=True, id="o1"))  # overlays never deduped
    s.publish_intent(intent("P5", "cough", overlay=True, id="o2"))
    s.on_status(status("P5", "done", "c", "walk_to", "standing"))
    s.publish_intent(intent("P5", "walk_to", "entrance", id="d"))  # finished: may run again
    assert [i.id for i in out.intents] == ["a", "c", "o1", "o2", "d"]
    assert [r["id"] for r in lines(tmp_path, "intents")] == ["a", "c", "o1", "o2", "d"]


def test_dedupe_compares_text_too(tmp_path):
    s, out = make(tmp_path)
    s.publish_intent(intent("P1", "approach", "g1", id="a"))
    s.on_status(status("P1", "running", "a", "approach", "walking"))
    s.publish_intent(intent("P1", "approach", "g1", text="hurry", id="b"))
    assert [i.id for i in out.intents] == ["a", "b"]


def test_says_are_published_and_recorded(tmp_path):
    s, out = make(tmp_path)
    s.publish_say(Say(person="P3", t=2.0, text="Excuse me", addressee="g1"))
    assert out.says[0].text == "Excuse me" and lines(tmp_path, "says")[0]["person"] == "P3"


async def test_tick_feeds_minds_and_skips_people_without_status(tmp_path):
    s, _ = make(tmp_path)
    s.on_status(status("P1"))
    s.on_nudge(Nudge(person="P1", t=0.5, card="c", message="Go"))
    await s.tick(1.0)
    assert len(s.minds["P1"].ticks) == 1 and s.minds["P2"].ticks == []
    t, obs, st, nudges = s.minds["P1"].ticks[0]
    assert obs.person == "P1" and obs.heard == [] and [n.message for n in nudges] == ["Go"]
    await s.tick(1.1)
    assert s.minds["P1"].ticks[1][3] == []  # nudges are delivered once
    s.on_status(status("P1", posture="gone"))
    await s.tick(1.2)
    assert len(s.minds["P1"].ticks) == 2


async def test_heard_is_delivered_once_across_overlapping_observations(tmp_path):
    s, _ = make(tmp_path)
    s.on_status(status("P3"))
    hi = heard("person", "P2", "hi")
    c = heard("cough", "P2")
    s.on_observation(Observation(person="P3", t=1.0, heard=[hi]))
    await s.tick(1.0)
    await s.tick(1.1)  # no new observation: nothing heard again
    s.on_observation(Observation(person="P3", t=1.5, heard=[hi, c]))
    await s.tick(1.5)
    s.on_observation(Observation(person="P3", t=2.0, heard=[hi, c, c]))  # a second, identical cough
    await s.tick(2.0)
    got = [[h.source_kind for h in tick[1].heard] for tick in s.minds["P3"].ticks]
    assert got == [["person"], [], ["cough"], ["cough"]]


async def test_look_at_only_spatial_sources(tmp_path):
    s, out = make(tmp_path, reflex=None)  # the real ReflexEngine
    s.on_status(status("P4"))
    s.on_observation(Observation(person="P4", t=1.0, heard=[
        heard("pa", "pa", "Clinics are late"), heard("alarm", "alarm")]))
    await s.tick(1.0)
    assert not [i for i in out.intents if i.kind == "look_at"]
    s.on_observation(Observation(person="P4", t=5.0, heard=[heard("pa", "pa", "x"), heard("robot", "g1", "hi")]))
    await s.tick(5.0)
    looks = [i for i in out.intents if i.kind == "look_at"]
    assert [(i.target, i.overlay) for i in looks] == [("g1", True)]


async def test_reflex_override_starts_a_bout_of_coughs(tmp_path):
    s, out = make(tmp_path, reflex=None)
    s.on_status(status("P2"))
    s.on_reflex(ReflexOverride(person="P2", t=60.0, kind="cough_bout", duration_s=40, severity="severe"))
    for k in range(0, 100):
        await s.tick(60.0 + k * 0.1)
    coughs = [i for i in out.intents if i.person == "P2" and i.kind == "cough"]
    assert len(coughs) >= 3 and all(i.overlay and i.text is None for i in coughs)
    assert all(r["overlay"] for r in lines(tmp_path, "intents") if r["kind"] == "cough")


async def test_idle_flag_follows_running_base_intent(tmp_path):
    seen = []

    class Spy(QuietReflex):
        def tick(self, t, status, heard, idle):
            seen.append(idle)
            return []

    s, _ = make(tmp_path, reflex=Spy)
    s.on_status(status("P1", "running", "a", "walk_to", "walking"))
    await s.tick(1.0)
    s.on_status(status("P1", "done", "a", "walk_to"))
    await s.tick(1.1)
    assert seen == [False, True]


def test_pending_count(tmp_path):
    s, _ = make(tmp_path)
    s.minds["P1"].pending = True
    s.minds["P4"].pending = True
    assert s.pending_count() == 2


def test_usage_goes_to_tokens(tmp_path):
    s, _ = make(tmp_path)
    s.now = lambda: 12.5
    req = LLMRequest(role="decide", person="P1", system_fixed="", persona_card="", user="", schema={})
    s.on_usage(req, LLMResult(True, {}, None, "claude-opus-5-5", input_tokens=10, output_tokens=5,
                              cost_usd=0.01, latency_s=0.2))
    r = lines(tmp_path, "tokens")[0]
    assert r["t"] == 12.5 and r["person"] == "P1" and r["role"] == "decide" and r["cost_usd"] == 0.01
    assert r["model"] == "claude-opus-5-5" and r["input_tokens"] == 10


async def test_aclose_closes_every_mind(tmp_path):
    s, _ = make(tmp_path)
    await s.aclose()
    assert all(m.closed for m in s.minds.values())


def test_snapshot_name_is_valid_and_bounded():
    assert snapshot_name("20261103-120000_waiting_area_v1", "start") == "run-20261103-120000_waiting_area_v1-start"
    long = snapshot_name("x" * 100 + " bad/chars", "end")
    assert long.startswith("run-") and long.endswith("-end") and len(long) <= 64
    assert all(c.isalnum() or c in "_.-" for c in long)


def test_dedupe_durable_intent_that_is_still_the_latest(tmp_path):
    # sit_idle, use_phone and wait_standing finish at once in the sim but the person carries on
    # doing them: re-sending one would only end it again and start another decision.
    s, out = make(tmp_path)
    s.publish_intent(intent("P1", "sit_idle", id="a"))
    s.on_status(status("P1", "done", "a", "sit_idle"))
    s.publish_intent(intent("P1", "sit_idle", id="b"))
    assert [i.id for i in out.intents] == ["a"]
    s.publish_intent(intent("P1", "use_phone", id="c"))
    s.on_status(status("P1", "done", "c", "use_phone"))
    s.publish_intent(intent("P1", "sit_idle", id="d"))
    assert [i.id for i in out.intents] == ["a", "c", "d"]
    s.publish_intent(intent("P1", "walk_to", "toilets", id="e"))
    s.on_status(status("P1", "done", "e", "walk_to", "standing"))
    s.publish_intent(intent("P1", "walk_to", "toilets", id="f"))  # a finished walk is not durable
    assert [i.id for i in out.intents][-2:] == ["e", "f"]
