import asyncio
import json
from types import SimpleNamespace as NS

import anthropic
import httpx
import openai
import pytest

from wardzero_agents.config.models import ModelRoles
from wardzero_agents.llm.anthropic_backend import AnthropicGateway
from wardzero_agents.llm.fake import FakeGateway, FakeRefusal
from wardzero_agents.llm.gateway import Budget, LLMRequest, make_gateway
from wardzero_agents.llm.openai_backend import OpenAIGateway
from wardzero_agents.llm.pricing import Usage, cost

ROLES = ModelRoles(decide="claude-opus-5-5", summarise="claude-haiku-5-5", vision="claude-opus-5-5")
SCHEMA = {"type": "object", "properties": {}}


def req(role="decide", user="hello", images=None):
    return LLMRequest(role=role, person="p1", system_fixed="SYS", persona_card="CARD",
                      user=user, schema=SCHEMA, images=images or [])


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "https://x"))


def message(text='{"a": 1}', stop="end_turn", usage=None):
    usage = usage or NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                        cache_creation_input_tokens=10)
    return NS(stop_reason=stop, content=[NS(type="text", text=text)], usage=usage)


class FakeAnthropic:
    def __init__(self, result=None, raises=None, delay=0):
        self.calls, self.result, self.raises, self.delay = [], result or message(), raises, delay
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.result


def make_anthropic(roles=ROLES, budget=None, **kw):
    seen = []
    client = FakeAnthropic(**kw)
    gw = AnthropicGateway(client, roles, budget or Budget(10.0), lambda r, x: seen.append((r, x)),
                          timeout_s=kw.pop("timeout_s", 0.2))
    return gw, client, seen


async def test_request_shape_opus():
    gw, client, _ = make_anthropic()
    r = await gw.complete(req())
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 4096
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["system"][0] == {"type": "text", "text": "SYS"}
    assert kw["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in kw and "thinking" not in kw
    assert kw["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert r.ok and r.data == {"a": 1}
    assert (r.input_tokens, r.output_tokens, r.cache_read_tokens, r.cache_write_tokens) == (100, 20, 50, 10)
    assert r.cost_usd == pytest.approx(cost("claude-opus-5-5", Usage(100, 20, 50, 10)))


async def test_haiku_has_no_fallbacks():
    gw, client, _ = make_anthropic()
    await gw.complete(req(role="summarise"))
    kw = client.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["max_tokens"] == 1024
    assert "fallbacks" not in kw and "betas" not in kw
    assert kw["output_config"]["effort"] == "low"


async def test_decide_effort_follows_roles():
    roles = ModelRoles(decide="claude-sonnet-5-5", summarise="claude-haiku-5-5",
                       vision="claude-opus-5-5", effort_decide="high")
    gw, client, _ = make_anthropic(roles=roles)
    await gw.complete(req())
    assert client.calls[0]["output_config"]["effort"] == "high"
    assert client.calls[0]["fallbacks"] == "default"


async def test_images_before_text():
    gw, client, _ = make_anthropic()
    await gw.complete(req(role="vision", images=[b"\x89PNGxx"]))
    blocks = client.calls[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["type"] == "base64"
    assert blocks[-1] == {"type": "text", "text": "hello"}


async def test_refusal_maps_to_error():
    gw, _, seen = make_anthropic(result=message(stop="refusal", text=""))
    r = await gw.complete(req())
    assert not r.ok and r.error == "refusal" and len(seen) == 1


async def test_invalid_json():
    gw, _, _ = make_anthropic(result=message(text="not json"))
    r = await gw.complete(req())
    assert not r.ok and r.error == "invalid_json" and r.input_tokens == 100


async def test_rate_limit():
    err = anthropic.RateLimitError("slow", response=_resp(429), body=None)
    gw, _, seen = make_anthropic(raises=err)
    r = await gw.complete(req())
    assert r.error == "rate_limited" and len(seen) == 1


async def test_provider_errors():
    for err in (anthropic.InternalServerError("x", response=_resp(500), body=None),
                anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))):
        gw, _, _ = make_anthropic(raises=err)
        assert (await gw.complete(req())).error == "provider_error"


async def test_timeouts():
    gw, _, _ = make_anthropic(raises=anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")))
    assert (await gw.complete(req())).error == "timeout"
    gw, _, _ = make_anthropic(delay=1.0)
    assert (await gw.complete(req())).error == "timeout"


async def test_budget_blocks_call():
    gw, client, seen = make_anthropic(budget=Budget(0.0))
    r = await gw.complete(req())
    assert r.error == "budget" and client.calls == [] and len(seen) == 1


async def test_budget_charged_and_exhausted():
    b = Budget(0.0005)
    gw, client, seen = make_anthropic(budget=b)
    await gw.complete(req())
    assert b.spent > 0 and b.exhausted
    assert (await gw.complete(req())).error == "budget" and len(client.calls) == 1
    b.charge(None)


def test_pricing():
    assert cost("claude-opus-5-5", Usage(input=1_000_000, output=0, cache_read=0, cache_write=0)) == pytest.approx(4.0)
    assert cost("claude-sonnet-5-5", Usage(0, 1_000_000, 0, 0)) == pytest.approx(10.0)
    assert cost("claude-haiku-5-5", Usage(0, 0, 1_000_000, 0)) == pytest.approx(0.01)
    assert cost("claude-opus-5-5", Usage(0, 0, 0, 1_000_000)) == pytest.approx(8.0)
    assert cost("gpt-x", Usage(1, 1, 1, 1)) is None


class FakeOpenAI:
    def __init__(self, text='{"b": 2}', raises=None):
        self.calls, self.text, self.raises = [], text, raises
        self.responses = NS(create=self._create)

    async def _create(self, **kw):
        self.calls.append(kw)
        if self.raises:
            raise self.raises
        return NS(output_text=self.text, usage=NS(input_tokens=30, output_tokens=7,
                                                  input_tokens_details=NS(cached_tokens=12)))


async def test_openai_shape_and_usage():
    seen = []
    client = FakeOpenAI()
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    gw = OpenAIGateway(client, roles, Budget(1.0), lambda r, x: seen.append(x))
    r = await gw.complete(req())
    kw = client.calls[0]
    assert kw["model"] == "gpt-x" and kw["instructions"] == "SYS\n\nCARD" and kw["input"] == "hello"
    assert kw["text"] == {"format": {"type": "json_schema", "name": "decide", "schema": SCHEMA, "strict": True}}
    assert r.ok and r.data == {"b": 2} and r.cost_usd is None
    assert (r.input_tokens, r.output_tokens, r.cache_read_tokens) == (30, 7, 12)
    assert len(seen) == 1


async def test_openai_errors():
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    cases = [(openai.RateLimitError("x", response=_resp(429), body=None), "rate_limited"),
             (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), "timeout"),
             (openai.InternalServerError("x", response=_resp(500), body=None), "provider_error")]
    for err, want in cases:
        gw = OpenAIGateway(FakeOpenAI(raises=err), roles, Budget(1.0), lambda r, x: None)
        assert (await gw.complete(req())).error == want
    gw = OpenAIGateway(FakeOpenAI(text="nope"), roles, Budget(1.0), lambda r, x: None)
    assert (await gw.complete(req())).error == "invalid_json"
    gw = OpenAIGateway(FakeOpenAI(), roles, Budget(0.0), lambda r, x: None)
    assert (await gw.complete(req())).error == "budget"


async def test_fake_policy_robot_speech():
    r = await FakeGateway().complete(req(user="ROBOT SAID TO YOU: clinic_corridor is down there"))
    assert r.ok and r.data["intent"]["kind"] == "walk_to" and r.data["intent"]["target"] == "clinic_corridor"
    assert r.data["say"] == "Thank you" and r.cost_usd == 0.0 and r.input_tokens == 0


async def test_fake_policy_other_branches():
    g = FakeGateway()
    d = (await g.complete(req(user="ROBOT SAID TO YOU: blah"))).data
    assert d["intent"]["kind"] == "wait_standing" and d["say"] == "Thank you"
    d = (await g.complete(req(user="NUDGE: go to the toilets"))).data
    assert d["intent"] == {"kind": "walk_to", "target": "toilets", "text": ""}
    d = (await g.complete(req(user="NUDGE: ask for help"))).data
    assert d["intent"]["kind"] == "approach" and d["intent"]["target"] == "g1"
    assert d["say"] == "Excuse me, can you help us?"
    d = (await g.complete(req(user="NUDGE: relax"))).data
    assert d["intent"]["kind"] == "wait_standing"
    d = (await g.complete(req(user="nothing"))).data
    assert d["intent"]["kind"] == "sit_idle" and d["mood"] == "calm" and d["reason"] == "fake policy"
    assert (await g.complete(req(role="summarise"))).data == {"memories": []}


async def test_fake_policy_injection_and_errors():
    def refuse(_):
        raise FakeRefusal()

    def boom(_):
        raise RuntimeError("x")

    assert (await FakeGateway(refuse).complete(req())).error == "refusal"
    assert (await FakeGateway(boom).complete(req())).error == "provider_error"
    assert (await FakeGateway(lambda r: {"z": 1}).complete(req())).data == {"z": 1}


def test_make_gateway_fake(monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    gw = make_gateway("claude", ROLES, Budget(1.0), lambda r, x: None)
    assert isinstance(gw, FakeGateway)


async def test_fake_gateway_calls_on_usage_once(monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    seen = []
    gw = make_gateway("claude", ROLES, Budget(1.0), lambda r, x: seen.append(x))
    await gw.complete(req())
    assert len(seen) == 1


async def test_nothing_escapes_anthropic():
    err = anthropic.APIResponseValidationError(_resp(200), None)
    gw, _, seen = make_anthropic(raises=err)
    assert (await gw.complete(req())).error == "provider_error" and len(seen) == 1
    for content in ([], None):
        m = message()
        m.content = content
        gw, _, _ = make_anthropic(result=m)
        assert (await gw.complete(req())).error == "invalid_json"
    gw, _, seen = make_anthropic(raises=RuntimeError("secret prompt text"))
    r = await gw.complete(req())
    assert r.error == "provider_error" and not r.ok and len(seen) == 1


async def test_nothing_escapes_openai():
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    seen = []
    gw = OpenAIGateway(FakeOpenAI(raises=openai.OpenAIError("x")), roles, Budget(1.0),
                       lambda r, x: seen.append(x))
    assert (await gw.complete(req())).error == "provider_error" and len(seen) == 1
    gw = OpenAIGateway(FakeOpenAI(raises=RuntimeError("x")), roles, Budget(1.0), lambda r, x: None)
    assert (await gw.complete(req())).error == "provider_error"


async def test_openai_image_media_type():
    client = FakeOpenAI()
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    await OpenAIGateway(client, roles, Budget(1.0), lambda r, x: None).complete(
        req(role="vision", images=[b"\xff\xd8\xff"]))
    url = client.calls[0]["input"][0]["content"][0]["image_url"]
    assert url.startswith("data:image/jpeg;base64,")


async def test_fake_robot_speech_ignores_scene_context():
    u = "Scene: you are near the entrance.\nROBOT SAID TO YOU: clinic_corridor is down there\nmore entrance"
    d = (await FakeGateway().complete(req(user=u))).data
    assert d["intent"]["target"] == "clinic_corridor"


def test_make_gateway_unknown_provider(monkeypatch):
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)
    with pytest.raises(ValueError):
        make_gateway("bogus", ROLES, Budget(1.0), lambda r, x: None)


# ---------- error kinds per HTTP status (final review finding 4b) ----------

STATUS_KINDS = [(400, "bad_request"), (401, "auth"), (403, "auth"), (404, "model_not_found"),
                (413, "too_large"), (500, "provider_error"), (503, "provider_error"),
                (422, "provider_error")]


def _status_exc(mod, code):
    return mod.APIStatusError("x", response=_resp(code), body=None)


async def test_anthropic_status_kinds():
    for code, kind in STATUS_KINDS:
        gw, _, seen = make_anthropic(raises=_status_exc(anthropic, code))
        r = await gw.complete(req())
        assert (r.ok, r.error, r.status) == (False, kind, code), code
        assert seen[0][1].status == code
    # the SDK's own subclasses map the same way
    gw, _, _ = make_anthropic(raises=anthropic.AuthenticationError("x", response=_resp(401), body=None))
    assert (await gw.complete(req())).error == "auth"
    gw, _, _ = make_anthropic(raises=anthropic.NotFoundError("x", response=_resp(404), body=None))
    assert (await gw.complete(req())).error == "model_not_found"
    # connection errors and 429 keep their kinds, with no status
    gw, _, _ = make_anthropic(raises=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")))
    r = await gw.complete(req())
    assert (r.error, r.status) == ("provider_error", None)
    gw, _, _ = make_anthropic(raises=anthropic.RateLimitError("slow", response=_resp(429), body=None))
    assert (await gw.complete(req())).error == "rate_limited"


async def test_anthropic_max_tokens():
    gw, _, _ = make_anthropic(result=message(text='{"intent": {"ki', stop="max_tokens"))
    r = await gw.complete(req())
    assert not r.ok and r.error == "max_tokens" and r.data is None and r.output_tokens == 20


async def test_openai_status_kinds():
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    for code, kind in STATUS_KINDS:
        gw = OpenAIGateway(FakeOpenAI(raises=_status_exc(openai, code)), roles, Budget(1.0), lambda r, x: None)
        r = await gw.complete(req())
        assert (r.ok, r.error, r.status) == (False, kind, code), code
    gw = OpenAIGateway(FakeOpenAI(raises=openai.APIConnectionError(request=httpx.Request("POST", "https://x"))),
                       roles, Budget(1.0), lambda r, x: None)
    r = await gw.complete(req())
    assert (r.error, r.status) == ("provider_error", None)


async def test_openai_max_tokens():
    class Cut(FakeOpenAI):
        async def _create(self, **kw):
            r = await super()._create(**kw)
            r.status, r.incomplete_details = "incomplete", NS(reason="max_output_tokens")
            return r
    roles = ModelRoles(decide="gpt-x", summarise="gpt-y", vision="gpt-x")
    r = await OpenAIGateway(Cut(text='{"b"'), roles, Budget(1.0), lambda r, x: None).complete(req())
    assert not r.ok and r.error == "max_tokens"
