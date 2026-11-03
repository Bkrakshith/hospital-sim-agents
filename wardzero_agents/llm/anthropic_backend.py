"""Claude backend: beta messages with structured output, 1 h cached persona card, server fallback."""
from __future__ import annotations

import asyncio
import base64
import json

import anthropic

from .images import media_type
from .gateway import BaseGateway, LLMRequest, LLMResult
from .pricing import Usage, cost

FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class AnthropicGateway(BaseGateway):
    def __init__(self, client, roles, budget, on_usage, timeout_s: float = 30.0):
        super().__init__(roles, budget, on_usage)
        self.client, self.timeout_s = client, timeout_s

    async def _call(self, req: LLMRequest, model: str) -> LLMResult:
        decide = req.role == "decide"
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": media_type(i),
                                               "data": base64.b64encode(i).decode()}}
                  for i in req.images]
        blocks.append({"type": "text", "text": req.user})
        kwargs = dict(
            model=model,
            max_tokens=4096 if decide else 1024,
            system=[{"type": "text", "text": req.system_fixed},
                    {"type": "text", "text": req.persona_card,
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            messages=[{"role": "user", "content": blocks}],
            output_config={"effort": self.roles.effort_decide if decide else "low",
                           "format": {"type": "json_schema", "schema": req.schema}},
            **(FALLBACK if model in FALLBACK_MODELS else {}),
        )
        fail = lambda kind: LLMResult(False, None, kind, model)
        try:
            msg = await asyncio.wait_for(self.client.beta.messages.create(**kwargs), self.timeout_s)
        except (asyncio.TimeoutError, anthropic.APITimeoutError):
            return fail("timeout")
        except anthropic.RateLimitError:
            return fail("rate_limited")
        except anthropic.APIError:
            return fail("provider_error")

        u = msg.usage
        usage = Usage(input=getattr(u, "input_tokens", 0) or 0,
                      output=getattr(u, "output_tokens", 0) or 0,
                      cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
        res = LLMResult(False, None, None, model, usage.input, usage.output,
                        usage.cache_read, usage.cache_write, cost(model, usage))
        if msg.stop_reason == "refusal":
            res.error = "refusal"
            return res
        text = next((b.text for b in (msg.content or []) if getattr(b, "type", None) == "text"), None)
        try:
            res.data = json.loads(text)
            res.ok = True
        except (TypeError, ValueError):
            res.error = "invalid_json"
        return res
