"""Scenario flows: the schema, the behaviour-tree engine that drives the storyboard, and the preset flows."""
from pathlib import Path

import pytest

from wardzero_agents.contracts import Status
from wardzero_agents.flows.engine import FlowEngine, FlowView, holds
from wardzero_agents.flows.spec import Flow, FlowError, fit_scenario

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def st(pid, x=0.0, y=0.0, posture="standing"):
    return Status(person=pid, t=0.0, posture=posture, state="idle", progress=0.0, x=x, y=y, yaw=0.0)


def view(t, people=None, **kw):
    people = people or {"P1": st("P1", 5.0, 0.0), "P2": st("P2", 1.0, 0.0)}
    kw.setdefault("cast", {"P1", "P2"})
    return FlowView(t=t, statuses=people, robot_xy=kw.pop("robot_xy", (0.0, 0.0)), present=set(people), **kw)


FLOW = Flow(id="demo", title="Demo", steps=[
    {"wait_until": {"label": "the robot greets P2", "when": {"robot_spoke_to": "P2"}, "timeout_s": 30}},
    {"fire": {"id": "b1", "card": "waits_announcement"}},
    {"wait": {"seconds": 10}},
    {"unless": {"label": "a nurse comes", "when": {"robot_near": {"person": "P1", "within_m": 1.0}},
                "then": [{"wait": {"seconds": 20}}, {"fire": {"id": "b2", "card": "call_bell_at", "target": "P1"}}]}},
    {"expect": {"label": "the robot tells staff", "when": {"robot_did": {"action": "tell", "about": "P1"}},
                "timeout_s": 60}},
])


def test_a_wait_is_met_as_soon_as_its_condition_holds():
    e = FlowEngine(FLOW)
    assert e.tick(view(0.0)) == []
    assert [f.card for f in e.tick(view(3.0, robot_said_to={"P2"}))] == ["waits_announcement"]
    assert e.timeline()[0]["state"] == "fired" and e.timeline()[0]["fired_t"] == 3.0


def test_a_wait_times_out_so_the_scenario_always_moves_on():
    e = FlowEngine(FLOW)
    out = []
    t = 0.0
    while t < 200:
        out += [(f.t, f.card) for f in e.tick(view(t))]
        t += 0.5
    assert [c for _, c in out] == ["waits_announcement", "call_bell_at"]
    assert out[0][0] == pytest.approx(30.0, abs=0.6) and out[1][0] == pytest.approx(60.0, abs=1.1)
    tree = e.tree()
    assert tree["children"][0]["state"] == "timed_out" and "moved on" in tree["children"][0]["reason"]
    assert tree["children"][-1]["state"] == "timed_out"  # the robot never told staff: recorded as missed
    assert e.finished


def test_unless_skips_its_steps_when_the_condition_comes_first():
    e = FlowEngine(FLOW)
    t = 0.0
    fired = []
    while t < 120:
        near = t >= 20  # the robot reaches P1 before the bell would ring (about 31 s)
        v = view(t, robot_said_to={"P2"}, robot_xy=(4.5, 0.0) if near else (0.0, 0.0))
        fired += [f.card for f in e.tick(v)]
        t += 0.5
    assert fired == ["waits_announcement"]
    b2 = [r for r in e.timeline() if r["id"] == "b2"][0]
    assert b2["state"] == "skipped" and b2["reason"].startswith("not needed")


def test_skip_and_rerun_work_as_on_the_storyboard():
    e = FlowEngine(FLOW)
    e.skip("b1")
    assert e.tick(view(1.0, robot_said_to={"P2"})) == []
    assert e.timeline()[0]["state"] == "skipped"
    e.rerun("b1")
    assert [f.card for f in e.tick(view(2.0))] == ["waits_announcement"]


def test_a_card_whose_target_has_gone_is_skipped():
    e = FlowEngine(Flow(id="x", title="x", steps=[{"fire": {"id": "b1", "card": "deterioration", "target": "P9"}}]))
    assert e.tick(view(0.0)) == [] and e.timeline()[0]["state"] == "skipped"


def test_conditions():
    v = view(10.0, people={"P1": st("P1", 1.0, 0.0, "seated"), "N9": st("N9", 3.0, 0.0)},
             cast={"P1"}, robot_said_to={"N9"}, said_to_robot={"P1"}, robot_did=[("guide", "P1"), ("press_call", None)],
             fires={"f1": True}, room_at=lambda x, y: "AE_Waiting" if x < 2 else "Lobby", robot_start=(-5.0, 0.0))
    assert holds({"robot_spoke_to": "newcomer"}, v) and holds({"robot_spoke_to": "anyone"}, v)
    assert not holds({"robot_spoke_to": "P1"}, v) and holds({"spoke_to_robot": "P1"}, v)
    assert holds({"posture": {"person": "P1", "is": "seated"}}, v)
    assert holds({"in_room": {"person": "P1", "room": "AE_Waiting"}}, v)
    assert holds({"not_in_room": {"person": "N9", "room": "AE_Waiting"}}, v)
    assert holds({"robot_did": {"action": "guide", "about": "P1"}}, v) and holds({"robot_did": {"action": "press_call"}}, v)
    assert not holds({"robot_did": {"action": "tell"}}, v)
    assert holds({"robot_moved": 4.0}, v) and holds({"fire_out": True}, v) and holds({"elapsed": 10}, v)
    assert holds({"all": [{"elapsed": 5}, {"not": {"elapsed": 20}}]}, v)
    assert holds({"any": [{"elapsed": 50}, {"robot_near": {"person": "P1", "within_m": 1.5}}]}, v)


def test_the_schema_refuses_bad_steps_and_names_outside_the_scenario():
    with pytest.raises(ValueError):
        Flow(id="x", title="x", steps=[{"wiat": {"seconds": 1}}])
    with pytest.raises(ValueError):
        Flow(id="x", title="x", steps=[{"wait_until": {"label": "l", "when": {"elapsed": 5}}}])  # no timeout
    with pytest.raises(ValueError):
        Flow(id="x", title="x", steps=[{"wait_until": {"label": "l", "when": {"robot_did": {"action": "fly"}},
                                                         "timeout_s": 5}}])
    f = Flow(id="x", title="x", steps=[{"fire": {"id": "b1", "card": "nope", "target": "Q1"}},
                                       {"expect": {"label": "l", "when": {"in_room": {"person": "P1", "room": "Mars"}},
                                                   "timeout_s": 5}}])
    with pytest.raises(FlowError) as e:
        fit_scenario(f, people={"P1"}, cards={"arrival"}, rooms={"Lobby"})
    assert "no card 'nope'" in str(e.value) and "no person 'Q1'" in str(e.value) and "no room 'Mars'" in str(e.value)


# ---- the preset flows -------------------------------------------------------------------------------------------
import yaml  # noqa: E402

from wardzero_agents.config.loader import _load_deck  # noqa: E402
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario  # noqa: E402
from wardzero_agents.flows.spec import flows_dir, load_flow, prune_flow  # noqa: E402
from wardzero_agents.site import load_site  # noqa: E402

PRESETS = load_scenario_presets(CONFIG)
DECK = _load_deck(CONFIG / "cards" / "studio.yaml")
SITE = load_site(CONFIG)


@pytest.mark.parametrize("pid", sorted(PRESETS))
def test_every_preset_has_a_flow_that_fits_it_and_finishes_even_if_the_robot_does_nothing(pid):
    p = PRESETS[pid]
    flow = load_flow(flows_dir(CONFIG) / f"{pid}.yaml")
    fit_scenario(flow, people=[c.id for c in p.characters], cards=DECK, rooms=SITE.rooms)
    people = {c.id: st(c.id, c.x, c.y) for c in p.characters}
    e = FlowEngine(flow, DECK)
    cards, t = [], 0.0
    while t <= p.duration_min * 60 and not e.finished:
        cards += [f.card for f in e.tick(FlowView(t=t, statuses=people, robot_xy=(p.robot_start.x, p.robot_start.y),
                                                  present=set(people), cast=set(people)))]
        t += 0.5
    assert e.finished, f"{pid}: the flow did not finish within the run's {p.duration_min} min"
    fired = [r for r in e.timeline() if r["state"] == "fired"]
    assert len(fired) == len(e.timeline()), [r for r in e.timeline() if r["state"] != "fired"]
    assert {b.card for b in p.timeline} <= set(cards)  # it plays every card the old timeline did


def test_a_flow_without_some_of_its_people_drops_what_concerns_them():
    flow = load_flow(flows_dir(CONFIG) / "ae_waiting.yaml")
    pruned = prune_flow(flow, ["P2", "V1", "N1"])  # George removed
    fit_scenario(pruned, people=["P2", "V1", "N1"], cards=DECK, rooms=SITE.rooms)
    text = yaml.safe_dump(pruned.model_dump())
    assert "P1" not in text and "deterioration" not in text and "waits_announcement" in text


def test_a_preset_scenario_is_staged_with_its_flow(tmp_path):
    loaded = preset_to_scenario(PRESETS["ae_waiting"], CONFIG, tmp_path)
    assert loaded.flow is not None and loaded.flow.id == "ae_waiting"
    assert (tmp_path / "flows" / "ae_waiting.yaml").exists() and loaded.scenario.flow == "../flows/ae_waiting.yaml"


def test_the_document_is_up_to_date_and_leaves_out_the_theatre_scenario():
    import subprocess
    import sys
    assert subprocess.run([sys.executable, str(ROOT / "scripts" / "flows_doc.py"), "--check"]).returncode == 0, \
        "run python scripts/flows_doc.py"
    doc = (ROOT / "docs" / "scenario-flows.md").read_text()
    assert "ae_waiting" in doc and "scrub_nurse" not in doc and "instrument" not in doc and "```mermaid" in doc


# ---- the director runs the flow ------------------------------------------------------------------------------------
def test_the_director_fires_cards_when_the_flow_says_and_writes_the_tree(tmp_path):
    from wardzero_agents.contracts import Roster, RosterEntry
    from wardzero_agents.ros.director_node import DirectorService
    from wardzero_agents.runlog import RunLog

    class Out:
        def __init__(self):
            self.msgs = []

        def publish(self, kind, obj):
            self.msgs.append((kind, obj))

    loaded = preset_to_scenario(PRESETS["ae_waiting"], CONFIG, tmp_path / "staged")
    out = Out()
    d = DirectorService(loaded, RunLog(tmp_path / "run"), out, run_id="r1")
    d.on_roster(Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                    for i in loaded.instances]))
    for c in PRESETS["ae_waiting"].characters:
        d.on_status(Status(person=c.id, t=0.0, x=c.x, y=c.y, yaw=0, posture="standing", progress=0, state="idle"))
    d.tick(1.0)
    assert not [o for k, o in out.msgs if k == "world"]
    # the robot greets Ava, then checks her in with a desk note; George sits down
    d.ingest("audio", {"source_kind": "robot", "addressee": "P2", "text": "Hello", "t": 2.0})
    d.ingest("robot_tasks", {"kind": "note", "about": "P2", "read": True, "t": 3.0})
    d.on_status(Status(person="P1", t=3.0, x=-7.58, y=0.22, yaw=0, posture="seated", progress=0, state="idle"))
    t = 2.0
    while t < 40:
        d.tick(t)
        t += 0.5
    fired = [r for r in d.engine.timeline() if r["state"] == "fired"]
    # the waits come once the room has settled; the robot starts in sight of George, so he gets worse soon after
    assert [r["card"] for r in fired] == ["waits_announcement", "deterioration"]
    assert fired[0]["fired_t"] < 25 and fired[1]["fired_t"] - fired[0]["fired_t"] == pytest.approx(10.5, abs=1.1)
    rows = [json.loads(x) for x in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert rows and rows[-1]["tree"]["children"][0]["state"] == "met"


# ---- adapting a flow to a prompt ----------------------------------------------------------------------------------
import asyncio  # noqa: E402
import json  # noqa: E402

from wardzero_agents.flows import generate as G  # noqa: E402
from wardzero_agents.llm.gateway import LLMResult  # noqa: E402


class ScriptedGateway:
    def __init__(self, answers):
        self.answers, self.reqs = list(answers), []

    async def complete(self, req):
        self.reqs.append(req)
        return LLMResult(True, self.answers.pop(0), None, "fake", cost_usd=0.0)


def _adapt(gw, prompt="Ava comes in with a cut hand and George is very anxious."):
    p = PRESETS["ae_waiting"]
    base = load_flow(flows_dir(CONFIG) / "ae_waiting.yaml")
    people = [{"id": c.id, "name": c.name, "role": c.role, "room": c.room} for c in p.characters]
    return asyncio.run(G.adapt_flow(gw, scenario_id="ae_waiting", base=base, preset_prompt=p.prompt, prompt=prompt,
                                    people=people, deck=DECK, rooms=SITE.rooms, duration_min=p.duration_min))


def test_an_adapted_flow_is_checked_and_a_bad_answer_is_tried_again_with_the_reason():
    good = yaml.safe_dump({"id": "ae_waiting", "title": "A&E", "summary": "s", "steps": [
        {"wait": {"seconds": 10}}, {"fire": {"id": "b1", "card": "deterioration", "target": "P2"}}]})
    gw = ScriptedGateway([{"flow_yaml": "title: A&E\nsteps: [{fire: {id: b1, card: nope}}]", "notes": ""},
                          {"flow_yaml": good, "notes": "Ava gets worse instead of George."}])
    flow, notes = _adapt(gw)
    assert flow.steps[1]["fire"]["target"] == "P2" and notes.startswith("Ava")
    assert flow.prompt_sha == G.prompt_sha("Ava comes in with a cut hand and George is very anxious.")
    assert "no card 'nope'" in gw.reqs[1].user  # told what was wrong
    assert "P1: George" in gw.reqs[0].user and "<current_flow>" in gw.reqs[0].user


def test_adapting_gives_up_with_a_clear_error_after_three_bad_answers():
    gw = ScriptedGateway([{"flow_yaml": "steps: []", "notes": ""}] * 3)
    with pytest.raises(FlowError, match="could not be adapted"):
        _adapt(gw)


def test_the_scripted_model_keeps_the_preset_flow(monkeypatch):
    from wardzero_agents.llm.fake import FakeGateway
    flow, _ = _adapt(FakeGateway())
    assert flow.steps == load_flow(flows_dir(CONFIG) / "ae_waiting.yaml").steps
