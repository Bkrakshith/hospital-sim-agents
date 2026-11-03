"""OpenAI backend: Responses API with strict json_schema output. Tokens only, no price."""
from __future__ import annotations

import asyncio
import base64
import json

import openai

from .images import media_type
from .gateway import BaseGateway, LLMRequest, LLMResult, status_error


class OpenAIGateway(BaseGateway):
    def __init__(self, client, roles, budget, on_usage, timeout_s: float = 30.0):
        super().__init__(roles, budget, on_usage)
        self.client, self.timeout_s = client, timeout_s

    async def _call(self, req: LLMRequest, model: str) -> LLMResult:
        if req.images:
            content = [{"type": "input_image",
                        "image_url": "data:" + media_type(i) + ";base64," + base64.b64encode(i).decode()}
                       for i in req.images]
            content.append({"type": "input_text", "text": req.user})
            inp = [{"role": "user", "content": content}]
        else:
            inp = req.user
        fail = lambda kind: LLMResult(False, None, kind, model)
        try:
            r = await asyncio.wait_for(self.client.responses.create(
                model=model,
                instructions=req.system_fixed + "\n\n" + req.persona_card,
                input=inp,
                text={"format": {"type": "json_schema", "name": req.role,
                                 "schema": req.schema, "strict": True}},
            ), self.timeout_s)
        except (asyncio.TimeoutError, openai.APITimeoutError):
            return fail("timeout")
        except openai.RateLimitError:
            return fail("rate_limited")
        except openai.APIStatusError as exc:
            res = fail(status_error(exc.status_code))
            res.status = exc.status_code
            return res
        except openai.OpenAIError:  # connection errors and the like
            return fail("provider_error")

        u = r.usage
        details = getattr(u, "input_tokens_details", None)
        res = LLMResult(False, None, None, model, getattr(u, "input_tokens", 0) or 0,
                        getattr(u, "output_tokens", 0) or 0,
                        getattr(details, "cached_tokens", 0) or 0, 0, None)
        inc = getattr(r, "incomplete_details", None)
        if getattr(r, "status", None) == "incomplete" and getattr(inc, "reason", None) == "max_output_tokens":
            res.error = "max_tokens"  # the JSON was cut off
            return res
        try:
            res.data = json.loads(r.output_text)
            res.ok = True
        except (TypeError, ValueError):
            res.error = "invalid_json"
        return res
