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
    # the robot greets Ava, then checks her and George in with desk notes; George sits down
    d.ingest("audio", {"source_kind": "robot", "addressee": "P2", "text": "Hello", "t": 2.0})
    d.ingest("robot_tasks", {"kind": "note", "about": "P2", "read": True, "t": 3.0})
    d.ingest("robot_tasks", {"kind": "note", "about": "P1", "read": True, "t": 3.0})
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


def test_schemas_sent_to_claude_have_no_keywords_structured_outputs_reject():
    from wardzero_agents.campaign.debrief import HEARSAY_SCHEMA, QUESTIONNAIRE_SCHEMA
    from wardzero_agents.llm.anthropic_backend import UNSUPPORTED_SCHEMA_KEYS, api_schema
    s = api_schema({"type": "object", "properties": {"a": {"type": "integer", "minimum": 1, "maximum": 5},
                                                     "b": {"type": "array", "minItems": 2, "maxItems": 3,
                                                           "items": {"type": "string", "maxLength": 4}},
                                                     "c": {"type": "string", "enum": ["minimum"]}},
                    "required": ["a", "b", "c"]})
    assert s["properties"]["a"] == {"type": "integer"} and s["properties"]["b"] == {"type": "array",
                                                                                      "items": {"type": "string"}}
    assert s["properties"]["c"]["enum"] == ["minimum"] and s["required"] == ["a", "b", "c"]
    text = json.dumps([QUESTIONNAIRE_SCHEMA, HEARSAY_SCHEMA, G.SCHEMA])
    assert not any(f'"{k}"' in text for k in UNSUPPORTED_SCHEMA_KEYS)


def test_the_flow_request_never_sends_an_empty_text_block():
    gw = ScriptedGateway([{"flow_yaml": yaml.safe_dump({"id": "ae_waiting", "title": "t", "steps": [
        {"wait": {"seconds": 1}}]}), "notes": ""}])
    _adapt(gw)
    r = gw.reqs[0]
    assert r.system_fixed.strip() and r.persona_card.strip() and r.user.strip()


def test_a_run_folder_copy_keeps_the_flow_so_the_run_loads_it(tmp_path):
    from wardzero_agents.config.loader import load_scenario
    from wardzero_agents.runlog import RunLog
    loaded = preset_to_scenario(PRESETS["ae_waiting"], CONFIG, tmp_path / "staged")
    copied = RunLog(tmp_path / "run").copy_configs(loaded)
    again = load_scenario(copied)
    assert again.flow is not None and again.flow.steps == loaded.flow.steps
    assert (tmp_path / "run" / "config" / "ae_waiting.yaml").exists() or again.scenario.flow


# ---- steps credit only what the robot did after the step began -----------------------------------------------------
def _node_by_label(tree: dict, label: str) -> dict:
    if tree["label"] == label:
        return tree
    for c in tree["children"]:
        try:
            return _node_by_label(c, label)
        except KeyError:
            pass
    raise KeyError(label)


def test_since_step_ignores_speech_before_the_step():
    flow = Flow(id="x", title="x", steps=[
        {"wait": {"seconds": 20}},
        {"expect": {"label": "new words to P1", "when": {"robot_spoke_to": {"person": "P1", "since_step": True}},
                    "timeout_s": 60}}])
    e = FlowEngine(flow)
    t = 0.0
    while t < 25:
        said = {"P1": [10.0]} if t >= 10 else {}
        e.tick(view(t, robot_said_to=set(said), robot_said_at=said))
        t += 0.5
    assert _node_by_label(e.tree(), "new words to P1")["state"] == "running"  # the words at t=10 came before it
    e.tick(view(25.0, robot_said_to={"P1"}, robot_said_at={"P1": [10.0, 25.0]}))
    node = _node_by_label(e.tree(), "new words to P1")
    assert node["state"] == "met" and node["t"] == 25.0
    # without since_step any words count, as before
    v = view(30.0, robot_said_to={"P1"}, robot_said_at={"P1": [10.0]})
    assert holds({"robot_spoke_to": "P1"}, v) and holds({"robot_spoke_to": {"person": "P1"}}, v)
    assert holds({"robot_spoke_to": {"person": "P1", "since_step": True}}, v, t0=5.0)
    assert not holds({"robot_spoke_to": {"person": "P1", "since_step": True}}, v, t0=20.0)
    assert not holds({"robot_spoke_to": {"person": "anyone", "since_step": True}}, v, t0=20.0)
    assert holds({"robot_spoke_to": "anyone"}, v)


def _evacuation(robot_did_at):
    """The evacuation flow with V2 walking from the start; robot_did_at(t) is what the robot has done by then."""
    p = PRESETS["evacuation_guide"]
    flow = load_flow(flows_dir(CONFIG) / "evacuation_guide.yaml")
    people = {c.id: st(c.id, c.x, c.y, "walking" if c.id == "V2" else "standing") for c in p.characters}
    e = FlowEngine(flow, DECK)
    fired_t = {}
    t = 0.0
    while t < 120:
        for f in e.tick(FlowView(t=t, statuses=people, robot_xy=(p.robot_start.x, p.robot_start.y), present=set(people),
                                 cast=set(people), robot_said_to={"V1"} if t >= 10 else set(),
                                 robot_said_at={"V1": [10.0]} if t >= 10 else {},
                                 robot_did=robot_did_at(t))):
            fired_t[f.card] = f.t
        t += 0.5
    return e, fired_t


def test_since_step_robot_did_counts_only_new_actions():
    e, fired = _evacuation(lambda t: [(2.0, "guide", "V2")] if t >= 2.0 else [])
    assert "freeze" in fired and fired["freeze"] > 2.0
    assert _node_by_label(e.tree(), "The robot helps V2")["state"] == "timed_out"  # the guide came before the freeze
    e, fired = _evacuation(lambda t: [(2.0, "guide", "V2")] + ([(t, "guide", "V2")] if t >= 60 else []))
    node = _node_by_label(e.tree(), "The robot helps V2")
    assert node["state"] == "met" and node["t"] == 60.0


def test_robot_did_keeps_reading_entries_without_a_time():
    v = view(10.0, robot_did=[("guide", "P1"), (4.0, "tell", "P1")])
    assert holds({"robot_did": {"action": "guide", "about": "P1"}}, v)
    assert not holds({"robot_did": {"action": "guide", "about": "P1", "since_step": True}}, v, t0=0.0)
    assert holds({"robot_did": {"action": "tell", "about": "P1", "since_step": True}}, v, t0=4.0)
    assert not holds({"robot_did": {"action": "tell", "about": "P1", "since_step": True}}, v, t0=4.5)


def test_since_step_is_checked_by_the_schema():
    ok = {"robot_did": {"action": "guide", "about": "V1", "since_step": True}}
    Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": ok, "timeout_s": 5}}])
    Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": {"robot_spoke_to": {"person": "V1", "since_step": True}},
                                               "timeout_s": 5}}])
    for bad in ({"robot_did": {"action": "guide", "since_step": "yes"}},
                {"robot_spoke_to": {"person": "V1", "since_step": 1}},
                {"robot_spoke_to": {"since_step": True}},
                {"spoke_to_robot": {"person": "V1", "since_step": True}},
                {"robot_near": {"person": "V1", "within_m": 2, "since_step": True}}):
        with pytest.raises(ValueError):
            Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": bad, "timeout_s": 5}}])
    f = Flow(id="x", title="x", steps=[{"expect": {"label": "l", "when": {"robot_spoke_to": {"person": "Q9"}},
                                                   "timeout_s": 5}}])
    with pytest.raises(FlowError, match="no person 'Q9'"):
        fit_scenario(f, people={"P1"}, cards=DECK, rooms=SITE.rooms)
    assert prune_flow(f, ["P1"]).steps[0] != f.steps[0]  # a wait on a missing person is dropped


def _director(tmp_path, preset="ae_waiting"):
    from wardzero_agents.contracts import Roster, RosterEntry
    from wardzero_agents.ros.director_node import DirectorService
    from wardzero_agents.runlog import RunLog

    class Out:
        def publish(self, kind, obj):
            pass

    loaded = preset_to_scenario(PRESETS[preset], CONFIG, tmp_path / "staged")
    d = DirectorService(loaded, RunLog(tmp_path / "run"), Out(), run_id="r1")
    d.on_roster(Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                    for i in loaded.instances]))
    for c in PRESETS[preset].characters:
        d.on_status(Status(person=c.id, t=0.0, x=c.x, y=c.y, yaw=0, posture="standing", progress=0, state="idle"))
    return d


def test_unanswered_press_call_is_not_an_alert(tmp_path):
    call = {"robot_did": {"action": "press_call"}}
    d = _director(tmp_path / "a")
    d.ingest("robot_tasks", {"kind": "result", "id": "a1", "action": "press_call", "ok": True, "reason": "unanswered",
                             "t": 5.0})
    assert not holds(call, d.view(6.0))
    for i, reason in enumerate(("answered", "ok", None)):
        d = _director(tmp_path / f"b{i}")
        rec = {"kind": "result", "id": "a1", "action": "press_call", "ok": True, "t": 5.0}
        if reason is not None:
            rec["reason"] = reason
        d.ingest("robot_tasks", rec)
        assert holds(call, d.view(6.0)), reason
        assert holds({"robot_did": {"action": "press_call", "since_step": True}}, d.view(6.0), t0=5.0)
        assert not holds({"robot_did": {"action": "press_call", "since_step": True}}, d.view(6.0), t0=5.5)


def test_guided_arrival_is_recorded(tmp_path):
    d = _director(tmp_path)
    d.ingest("robot_tasks", {"t": 3.0, "kind": "guide", "event": "start", "id": "a1", "person": "P2", "place": "x"})
    d.ingest("robot_tasks", {"t": 9.0, "kind": "guide", "event": "end", "id": "a1", "person": "P2",
                             "reason": "arrived"})
    d.ingest("robot_tasks", {"t": 12.0, "kind": "guide", "event": "end", "id": "a2", "person": "P1",
                             "reason": "left_behind"})
    assert [(a, p) for _, a, p, _ in d.robot_did] == [("guide", "P2"), ("guided", "P2"), ("guide_ended", "P2"),
                                                      ("guide_ended", "P1")]
    assert d.robot_did[1][0] == 9.0
    v = d.view(10.0)
    assert holds({"robot_did": {"action": "guided", "about": "P2"}}, v)
    assert not holds({"robot_did": {"action": "guided", "about": "P1"}}, v)


def test_the_director_keeps_when_the_robot_spoke_to_each_person(tmp_path):
    d = _director(tmp_path)
    d.ingest("audio", {"source_kind": "robot", "addressee": "P2", "text": "Hello", "t": 2.0})
    d.ingest("audio", {"source_kind": "robot", "addressee": "P2", "text": "This way", "t": 8.0})
    assert d.robot_said_to == {"P2"}  # the set view old callers read
    assert d.robot_said_at == {"P2": [2.0, 8.0]}
    v = d.view(9.0)
    assert v.robot_said_to == {"P2"} and v.robot_said_at == {"P2": [2.0, 8.0]}
    assert holds({"robot_spoke_to": {"person": "P2", "since_step": True}}, v, t0=8.0)
    assert not holds({"robot_spoke_to": {"person": "P2", "since_step": True}}, v, t0=8.5)


def _run_preset(pid, *, view_kw=lambda t: {}, until=None):
    p = PRESETS[pid]
    flow = load_flow(flows_dir(CONFIG) / f"{pid}.yaml")
    people = {c.id: st(c.id, c.x, c.y) for c in p.characters}
    e = FlowEngine(flow, DECK)
    fired, t = [], 0.0
    while t <= (until or p.duration_min * 60) and not e.finished:
        start = (p.robot_start.x, p.robot_start.y)
        fired += [(f.t, f.card) for f in e.tick(FlowView(t=t, statuses=people, robot_xy=start, robot_start=start,
                                                         present=set(people), cast=set(people), **view_kw(t)))]
        t += 0.5
    return e, fired


def test_specimen_run_fires_the_escort_at_the_start():
    e, fired = _run_preset("specimen_run", until=5.0)
    assert fired == [(0.0, "escort_patient")]  # nothing to wait for: the box starts in the robot's hand


def test_ward_fire_expect_needs_the_robot_to_extinguish():
    label = next(n.label for n in FlowEngine(load_flow(flows_dir(CONFIG) / "ward_fire.yaml"))._ctx.nodes
                 if n.kind == "expect")
    e, _ = _run_preset("ward_fire", view_kw=lambda t: {"fires": {"f1": True}})  # a nurse put it out
    assert _node_by_label(e.tree(), label)["state"] == "timed_out"
    e, _ = _run_preset("ward_fire", view_kw=lambda t: {"fires": {"f1": True},
                                                        "robot_did": [(70.0, "extinguish", None)] if t >= 70 else []})
    assert _node_by_label(e.tree(), label)["state"] == "met"


def test_a_check_in_in_the_same_turn_as_the_greeting_counts(tmp_path):
    # the robot greets Ava and sends the desk note in one decision: both rows carry the same time, and the director
    # sees the greeting (and starts the check-in step) on a later tick
    d = _director(tmp_path)
    d.tick(12.0)
    d.ingest("audio", {"source_kind": "robot", "addressee": "P2", "text": "Welcome, I have let the desk know",
                       "t": 12.3})
    d.ingest("robot_tasks", {"kind": "note", "about": "P2", "read": True, "t": 12.3})
    assert _node_by_label(d.engine.tree(), "The robot greets Ava")["state"] == "running"
    t = 12.5
    while t < 20:
        d.tick(t)
        t += 0.5
    node = _node_by_label(d.engine.tree(), "The robot checks Ava in (a desk note, a walk to a seat or a word with staff)")
    assert node["state"] == "met"


def _reception(robot_xy_at, robot_did_at=lambda t: [], until=200.0):
    """The reception flow: the robot greets V1 at t=4.8 and speaks to V1 again at t=10, after V1 gets lost."""
    p = PRESETS["reception_guide"]
    flow = load_flow(flows_dir(CONFIG) / "reception_guide.yaml")
    people = {c.id: st(c.id, c.x, c.y) for c in p.characters}
    e = FlowEngine(flow, DECK)
    fired, t = {}, 0.0
    while t <= until:
        said = {"V1": [x for x in (4.8, 10.0) if x <= t]}
        said = {k: v for k, v in said.items() if v}
        for f in e.tick(FlowView(t=t, statuses=people, robot_xy=robot_xy_at(t), present=set(people), cast=set(people),
                                 robot_said_to=set(said), robot_said_at=said, robot_did=robot_did_at(t),
                                 room_at=SITE.room_at)):
            fired.setdefault(f.card + (f.target or ""), f.t)
        t += 0.5
    return e, fired


def test_reception_waits_until_the_robot_has_finished_with_v1():
    v1 = (-1.2 + 0.8, -1.0)  # the robot stands next to V1
    # it talks to V1 until t=70, then walks off; V1 stays in the lobby: the wait runs to its limit
    e, fired = _reception(lambda t: v1 if t < 70 else (5.0, 5.0), until=300.0)
    assert fired["lost_visitorV1"] == 5.0
    help_t = _node_by_label(e.tree(), "The robot walks V1 where they need to go (the walk ends with V1 there)")["t"]
    assert fired["hard_of_hearingH1"] == pytest.approx(help_t + 120.0, abs=0.6)
    # the end of a guide ends the wait, even with the robot still beside V1
    e, fired = _reception(lambda t: v1, robot_did_at=lambda t: [(8.0, "guide", "V1")] +
                          ([(40.0, "guided", "V1"), (40.0, "guide_ended", "V1")] if t >= 40 else []))
    assert fired["hard_of_hearingH1"] == 40.0


def _evacuation_outcomes(*, moves, robot_did_at, said_at):
    """The evacuation flow; moves(t) gives where people have gone by then. The alarm comes at t=5."""
    p = PRESETS["evacuation_guide"]
    flow = load_flow(flows_dir(CONFIG) / "evacuation_guide.yaml")
    e = FlowEngine(flow, DECK)
    t = 0.0
    while t <= p.duration_min * 60 and not e.finished:
        people = {c.id: st(c.id, *moves(t).get(c.id, (c.x, c.y))) for c in p.characters}
        said = {k: [x for x in v if x <= t] for k, v in said_at.items()}
        said = {k: v for k, v in said.items() if v}
        e.tick(FlowView(t=t, statuses=people, robot_xy=(p.robot_start.x, p.robot_start.y), present=set(people),
                        cast=set(people), robot_said_to=set(said), robot_said_at=said, robot_did=robot_did_at(t),
                        room_at=SITE.room_at))
        t += 0.5
    return e


LOBBY = "The robot gets V1 and V2 out of the lobby"
P1_OUT = "The robot gets P1 out of A&E waiting"


def test_evacuation_credits_only_people_the_robot_got_out():
    out = {"V1": (3.0, 10.0), "V2": (3.5, 10.0), "P1": (0.0, 5.0)}  # Main_Corridor, Main_Corridor, Lobby
    # everyone walks out on their own; the robot only spoke to V1 before the alarm
    e = _evacuation_outcomes(moves=lambda t: out if t >= 30 else {}, robot_did_at=lambda t: [],
                             said_at={"V1": [4.8]})
    assert _node_by_label(e.tree(), LOBBY)["state"] == "timed_out"
    assert _node_by_label(e.tree(), P1_OUT)["state"] == "timed_out"
    # after the alarm the robot walks V1, V2 and P1 out, and each walk ends with them there
    did = lambda t: [x for x in ((20.0, "guide", "V1"), (30.0, "guided", "V1"), (31.0, "guide", "V2"),  # noqa: E731
                                 (45.0, "guided", "V2"), (46.0, "guide", "P1"), (60.0, "guided", "P1")) if x[0] <= t]
    e = _evacuation_outcomes(moves=lambda t: {k: v for k, v in out.items() if t >= (60 if k == "P1" else 30)},
                             robot_did_at=did, said_at={"V1": [4.8], "V2": [25.0]})
    assert _node_by_label(e.tree(), LOBBY)["state"] == "met"
    assert _node_by_label(e.tree(), P1_OUT)["state"] == "met"
    # everyone leaves; the robot spoke to V2 and P1 and started walks it never finished: none of it is credited
    e = _evacuation_outcomes(moves=lambda t: {k: v for k, v in out.items() if t >= 30},
                             robot_did_at=lambda t: [x for x in ((20.0, "guided", "V1"), (21.0, "guide", "V2"),
                                                                 (22.0, "guide", "P1"), (40.0, "guide_ended", "P1"))
                                                     if x[0] <= t],
                             said_at={"V1": [4.8], "V2": [25.0], "P1": [26.0]})
    assert _node_by_label(e.tree(), LOBBY)["state"] == "timed_out"
    assert _node_by_label(e.tree(), P1_OUT)["state"] == "timed_out"
