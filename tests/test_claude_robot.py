import json
import re
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.models import ModelRoles
from wardzero_agents.llm.fake import FakeGateway
from wardzero_agents.robots.claude_robot import RobotMind, Outputs, decision_outputs, parse_decision
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile, get_profile

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
LS = load_scenario(V1)
BRIEF = {"role": "reception guide", "job": "Greet people who come to reception and give short clear directions."}
NONE = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
        "place": [], "reason": "nothing to do"}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Recorder:
    """A gateway that records every request and answers with a scripted decision."""

    def __init__(self, answer=None):
        self.reqs, self.answer = [], answer if answer is not None else dict(NONE)

    async def complete(self, req):
        from wardzero_agents.llm.gateway import LLMResult
        self.reqs.append(req)
        data = self.answer(req) if callable(self.answer) else self.answer
        return LLMResult(True, data, None, "fake", cost_usd=0.0)


def mind(gw=None, brief=BRIEF, profile=G1_CLAUDE):
    clock = Clock()
    gw = gw or Recorder()
    m = RobotMind(LS.site, brief, gw, clock, profile=profile, known_places=LS.scenario.known_places,
                  pose=(1.05, 1.62, -1.5708))
    return m, gw, clock


def heard(text="Excuse me, where is the toilet?", addressed=True, kind="person", db=60.0):
    return {"source_kind": kind, "source_id": "unknown", "text": text, "received_db": db,
            "partial": False, "addressed_to_me": addressed}


def det(track, x, y, z=1.7, sx=0.5):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": sx, "y": 0.5, "z": z}}}


async def test_addressed_question_triggers_call():
    m, gw, clock = mind()
    await m.tick(0.0)
    await m.on_heard(heard(), 0.5)
    d = await m.tick(0.5)
    assert len(gw.reqs) == 1 and d is not None and d.action == "none"
    assert "Excuse me, where is the toilet?" in gw.reqs[0].user


async def test_addressed_bypasses_two_second_gap_but_others_do_not():
    m, gw, clock = mind()
    await m.tick(0.0)
    await m.on_heard(heard(), 0.1)
    await m.tick(0.1)
    await m.on_heard(heard("Hello?"), 0.5)
    await m.tick(0.5)
    assert len(gw.reqs) == 2  # addressed twice within 2 s: both go through
    m.on_people([det("t1", 1.0, 0.5)], 0.6)  # a new track, but only 0.1 s since the last call
    await m.tick(0.6)
    assert len(gw.reqs) == 2
    await m.tick(2.6)
    assert len(gw.reqs) == 3


async def test_speech_within_three_metres_triggers_but_far_speech_does_not():
    m, gw, _ = mind()
    await m.tick(0.0)
    m.on_people([det("t1", 1.05, 5.0)], 1.0)  # 3.4 m away
    await m.tick(1.0)
    n = len(gw.reqs)
    await m.on_heard(heard("blah", addressed=False), 2.5)
    await m.tick(2.5)
    assert len(gw.reqs) == n
    m.on_people([det("t1", 1.05, 3.0)], 3.0)  # 1.4 m away
    await m.tick(3.0)
    n = len(gw.reqs)
    await m.on_heard(heard("blah", addressed=False), 6.0)
    await m.tick(6.0)
    assert len(gw.reqs) == n + 1


async def test_heartbeat_every_10s_only():
    m, gw, _ = mind()
    for t in [0.0, 1.0, 5.0, 9.9]:
        await m.tick(t)
    assert gw.reqs == []
    await m.tick(10.0)
    assert len(gw.reqs) == 1
    await m.tick(15.0)
    assert len(gw.reqs) == 1
    await m.tick(20.0)
    assert len(gw.reqs) == 2


async def test_new_track_and_task_result_trigger():
    m, gw, _ = mind()
    await m.tick(0.0)
    m.on_people([det("t1", 1.0, 0.5)], 3.0)
    await m.tick(3.0)
    assert len(gw.reqs) == 1
    m.on_people([det("t1", 1.0, 0.6)], 6.0)  # same track: nothing new
    await m.tick(6.0)
    assert len(gw.reqs) == 1
    m.on_task_result({"id": "rb1", "action": "press_call", "ok": True, "reason": "ok", "t": 6.5})
    await m.tick(7.0)
    assert len(gw.reqs) == 2
    m.on_task_result({"id": "rb1", "action": "press_call", "ok": True, "reason": "ok", "t": 6.5})  # repeat
    await m.tick(17.0)
    assert len(gw.reqs) == 3  # the 10 s heartbeat, not the repeated result
    assert "press_call" in gw.reqs[-1].user and gw.reqs[-1].user.count("press_call") >= 1


async def test_waits_for_brief_then_starts_without_after_five_seconds():
    m, gw, _ = mind(brief=None)
    await m.tick(0.0)
    await m.tick(4.0)
    m.on_people([det("t1", 1.0, 0.5)], 4.0)
    await m.tick(4.5)
    assert gw.reqs == []
    await m.tick(5.1)
    assert len(gw.reqs) == 1 and "No task brief was given" in gw.reqs[0].persona_card
    m2, gw2, _ = mind(brief=None)
    await m2.tick(0.0)
    m2.on_brief(BRIEF)
    await m2.on_heard(heard(), 1.0)
    await m2.tick(1.0)
    assert len(gw2.reqs) == 1 and BRIEF["job"] in gw2.reqs[0].persona_card


def ctx():
    m, _, _ = mind()
    m.on_task_result({"id": "x", "action": "x", "ok": True, "reason": "ok", "t": 0})
    return m


def test_decision_maps_to_topics():
    m, _, _ = mind(profile=RobotProfile(**{**G1_CLAUDE.__dict__, "can_move": True}))
    m.on_people([det("t1", 1.0, 0.5)], 1.0)
    hand = parse_decision({**NONE, "action": "hand_over", "target": "t1", "item": "item1",
                           "place": [1.0, 0.9, 1.1], "say": "Here you are"}, m.profile, m)
    out = decision_outputs(hand, m, "rb1")
    assert out.say == "Here you are"
    assert out.action == {"id": "rb1", "action": "hand_over", "item": "item1", "to": "t1",
                          "place": [1.0, 0.9, 1.1]}
    guide = parse_decision({**NONE, "action": "guide", "target": "t1", "destination": "Main entrance"},
                           m.profile, m)
    out = decision_outputs(guide, m, "rb2")
    assert out.action == {"id": "rb2", "action": "guide_start", "track": "t1", "place": "entrance"}
    assert out.goal is None
    go = decision_outputs(parse_decision({**NONE, "action": "go_to", "destination": "Main entrance"},
                                         m.profile, m), m, "rb3")
    assert go.goal is not None and go.action is None and len(go.goal) == 3
    wait = decision_outputs(parse_decision({**NONE, "action": "wait"}, m.profile, m), m, "rb4")
    assert wait.stop is True and wait.action is None
    simple = {"take": ({"item": "item1"}, {"id": "rb5", "action": "take", "item": "item1"}),
              "extinguish": ({"target": "fire1"}, {"id": "rb5", "action": "extinguish", "fire": "fire1"}),
              "press_call": ({}, {"id": "rb5", "action": "press_call"})}
    for name, (extra, want) in simple.items():
        out = decision_outputs(parse_decision({**NONE, "action": name, **extra}, m.profile, m), m, "rb5")
        assert out.action == want


def test_fixed_post_profile_cannot_walk():
    assert G1_CLAUDE.can_move is False and G1_CLAUDE.base == "static" and G1_CLAUDE.ns == "robot"
    assert G1_CLAUDE.reach == (0.6, 1.3) and G1_CLAUDE.radius == 0.3 and G1_CLAUDE.max_speed == 1.0
    assert get_profile("g1_claude") is G1_CLAUDE
    m, _, _ = mind()
    for bad in ("go_to", "guide"):
        with pytest.raises(ValueError):
            parse_decision({**NONE, "action": bad, "destination": "Main entrance", "target": "t1"},
                           G1_CLAUDE, m)
    assert parse_decision({**NONE, "action": "press_call"}, G1_CLAUDE, m).action == "press_call"


async def test_fixed_post_prompt_and_schema():
    m, gw, _ = mind()
    await m.on_heard(heard(), 0.1)
    await m.tick(0.1)
    r = gw.reqs[0]
    assert ("You are standing at a fixed post and cannot walk in this build; to help someone get "
            "somewhere, give clear directions or press the call button for staff.") in r.system_fixed
    enum = r.schema["properties"]["action"]["enum"]
    assert "go_to" not in enum and "guide" not in enum and "hand_over" in enum
    assert r.role == "robot" and r.person == "robot"
    movable = RobotProfile(**{**G1_CLAUDE.__dict__, "can_move": True})
    m2, gw2, _ = mind(profile=movable)
    await m2.on_heard(heard(), 0.1)
    await m2.tick(0.1)
    enum = gw2.reqs[0].schema["properties"]["action"]["enum"]
    assert "go_to" in enum and "guide" in enum
    assert "cannot walk" not in gw2.reqs[0].system_fixed


async def test_prompt_has_site_and_brief_no_minds():
    m, gw, _ = mind()
    m.on_people([det("t1", 1.0, 0.5), det("t2", 2.0, 1.0, z=0.5, sx=1.8)], 0.2)
    await m.on_heard(heard(), 0.3)
    await m.tick(0.3)
    r = gw.reqs[0]
    text = "\n".join([r.system_fixed, r.persona_card, *r.history, r.user])
    assert BRIEF["job"] in r.persona_card
    room = LS.site.rooms[LS.scenario.known_places[0]] if LS.scenario.known_places[0] in LS.site.rooms else None
    labels = [LS.site.landmarks[k].label.split(",")[0] for k in LS.scenario.known_places if k in LS.site.landmarks]
    assert any(lbl.lower() in text.lower() for lbl in labels)
    assert "lying" in r.user and "standing" in r.user
    banned = set()
    for inst in LS.instances:
        banned |= {inst.person_id, inst.display_name, inst.persona_id, inst.backstory[:30]}
        banned |= {w for w in inst.display_name.split() if len(w) > 2}
    for persona in LS.personas.values():
        banned |= {persona.id} if hasattr(persona, "id") else set()
    for b in banned:
        if b:
            assert not re.search(rf"\b{re.escape(b)}\b", text), b
    for mood in ("anxious", "stressed", "angry", "sad", "frightened", "impatient", "confused"):
        assert mood not in text.lower(), mood
    assert "persona" not in text.lower()


async def test_cached_prefix_is_stable_across_ticks():
    m, gw, _ = mind()
    await m.on_heard(heard(), 0.1)
    await m.tick(0.1)
    m.on_people([det("t1", 1.0, 0.5)], 3.0)
    await m.tick(3.0)
    assert gw.reqs[0].system_fixed == gw.reqs[1].system_fixed
    assert gw.reqs[0].persona_card == gw.reqs[1].persona_card
    assert gw.reqs[0].user != gw.reqs[1].user


async def test_invalid_decision_is_dropped(caplog):
    bad = [{"action": "fly"}, {**NONE, "action": "hand_over"}, "text", {**NONE, "say": 3},
           {**NONE, "action": "none", "place": [1, 2]}]
    for answer in bad:
        m, gw, _ = mind(Recorder(answer))
        await m.on_heard(heard(), 0.1)
        with caplog.at_level("WARNING"):
            d = await m.tick(0.1)
        assert d is None
    assert "dropped" in caplog.text.lower()


async def test_failed_call_returns_none_and_counts_permanent_errors():
    from wardzero_agents.llm.gateway import LLMResult

    class Failing:
        n = 0

        async def complete(self, req):
            Failing.n += 1
            return LLMResult(False, None, "auth", "m", status=401)

    m, _, _ = mind(Failing())
    for i in range(6):
        await m.on_heard(heard(), i * 1.0)
        assert await m.tick(i * 1.0) is None
    assert Failing.n == 3  # after three permanent failures the mind stops calling


async def test_speech_never_contains_an_internal_id():
    ans = {**NONE, "say": "Head to ward_doors, then Main_LiftLobby, via clinic_corridor.",
           "action": "none"}
    m, gw, _ = mind(Recorder(ans))
    await m.on_heard(heard(), 0.1)
    d = await m.tick(0.1)
    out = decision_outputs(d, m, "rb1")
    assert out.say and "_" not in out.say
    assert "ward doors" in out.say.lower() or "wards" in out.say.lower()


async def test_prompt_lists_routes_in_words_and_pose_and_recent_events():
    m, gw, _ = mind()
    await m.on_heard(heard(), 0.1)
    await m.tick(0.1)
    r = gw.reqs[0]
    assert "Getting there from your post" in r.persona_card
    await m.on_heard(heard("Thanks"), 1.0)
    d = await m.tick(1.0)
    assert "You decided" in gw.reqs[1].user and "nothing to do" in gw.reqs[1].user
    assert "(x=" in gw.reqs[1].user


async def test_tokens_use_person_robot(tmp_path):
    from wardzero_agents.robots.claude_robot import make_usage_logger
    from wardzero_agents.runlog import RunLog
    log = RunLog(tmp_path)
    gw = FakeGateway(policy=lambda req: dict(NONE), on_usage=make_usage_logger(log, lambda: 5.0))
    m, _, _ = mind(gw)
    await m.on_heard(heard(), 0.1)
    await m.tick(0.1)
    rows = [json.loads(x) for x in (tmp_path / "tokens.jsonl").read_text().splitlines()]
    assert rows[0]["person"] == "robot" and rows[0]["role"] == "robot"


def test_roles_and_gateway_model():
    roles = ModelRoles(decide="a", summarise="b", vision="c")
    assert roles.robot == "claude-sonnet-5-5"
    from wardzero_agents.llm.gateway import Budget, BaseGateway
    gw = BaseGateway(roles, Budget(1.0), lambda *_: None)
    assert gw.model_for("robot") == "claude-sonnet-5-5"


async def test_claude_backend_robot_request_layout():
    from types import SimpleNamespace as NS
    from wardzero_agents.llm.anthropic_backend import AnthropicGateway
    from wardzero_agents.llm.gateway import Budget, LLMRequest
    calls = []

    async def create(**kw):
        calls.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="{}")],
                  usage=NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                           cache_creation_input_tokens=0))

    client = NS(beta=NS(messages=NS(create=create)))
    gw = AnthropicGateway(client, ModelRoles(decide="x", summarise="y", vision="z"), Budget(5), lambda *_: None)
    res = await gw.complete(LLMRequest(role="robot", person="robot", system_fixed="S", persona_card="C",
                                       user="U", schema={"type": "object"}))
    kw = calls[0]
    assert res.model == "claude-sonnet-5-5" and kw["model"] == "claude-sonnet-5-5"
    assert kw["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert kw["max_tokens"] == 1024


def test_default_policy_answers_robot_role():
    from wardzero_agents.llm.gateway import LLMRequest
    from wardzero_agents.llm.fake import default_policy
    d = default_policy(LLMRequest(role="robot", person="robot", system_fixed="", persona_card="", user="",
                                  schema={}))
    assert d["action"] == "none"


# ---- cli ------------------------------------------------------------------------------------
def test_cli_bot_selection(monkeypatch):
    from types import SimpleNamespace as NS
    from wardzero_agents import cli
    staged = NS(scenario=NS(robot=NS(enabled=True)))
    a = lambda bot=None, no_robot=False: NS(bot=bot, no_robot=no_robot)
    monkeypatch.setattr(cli, "key_source", lambda p: cli.KEY_LABELS[0])
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)
    assert cli._bot(a(), staged) == "claude"
    assert cli._bot(a(), staged, claude_default=False) == "reception"
    assert cli._bot(a("reception"), staged) == "reception"
    assert cli._bot(a("none"), staged) == "none"
    monkeypatch.setattr(cli, "key_source", lambda p: cli.KEY_LABELS[3])
    assert cli._bot(a(), staged) == "reception"
    assert cli._bot(a("claude"), staged) == "claude"
    assert cli._bot(a(no_robot=True), staged) == "none"


def test_cli_claude_child_is_the_only_robot_with_the_key(monkeypatch):
    from wardzero_agents import cli
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    (c,) = cli._bot_children("claude", "/s.yaml", "/run")
    assert c.name == "robot_claude" and not c.name.startswith("wardzero_")
    assert c.env["ANTHROPIC_API_KEY"] and c.argv[-4:] == ["--scenario", "/s.yaml", "--run-dir", "/run"]
    assert "wardzero_agents.robots.claude_robot" in c.argv
    (r,) = cli._bot_children("reception", "/s.yaml", "/run")
    assert r.env is None  # supervised with the key-free env
    assert cli._bot_children("none", "/s.yaml", "/run") == []
