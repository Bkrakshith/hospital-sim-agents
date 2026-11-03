"""Gateway types shared by every backend, the spend budget and the factory."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol

from ..config.models import ModelRoles

Role = Literal["decide", "summarise", "vision"]
ErrorKind = Literal["refusal", "invalid_json", "timeout", "rate_limited", "budget", "provider_error"]


@dataclass
class LLMRequest:
    role: Role
    person: str
    system_fixed: str
    persona_card: str
    user: str
    schema: dict
    images: list[bytes] = field(default_factory=list)


@dataclass
class LLMResult:
    ok: bool
    data: Optional[dict]
    error: Optional[ErrorKind]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_s: float = 0.0


class Gateway(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResult: ...


class Budget:
    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self._spent = 0.0

    def charge(self, usd: Optional[float]) -> None:
        if usd:
            self._spent += usd

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def exhausted(self) -> bool:
        return self._spent >= self.limit_usd


class BaseGateway:
    """Shared flow: budget check, timing, charging and exactly one on_usage call."""

    def __init__(self, roles: ModelRoles, budget: Budget,
                 on_usage: Callable[[LLMRequest, LLMResult], None]):
        self.roles, self.budget, self.on_usage = roles, budget, on_usage

    def model_for(self, role: str) -> str:
        return {"decide": self.roles.decide, "summarise": self.roles.summarise,
                "vision": self.roles.vision}[role]

    async def _call(self, req: LLMRequest, model: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    async def complete(self, req: LLMRequest) -> LLMResult:
        model = self.model_for(req.role)
        t0 = time.monotonic()
        if self.budget.exhausted:
            res = LLMResult(False, None, "budget", model)
        else:
            res = await self._call(req, model)
            self.budget.charge(res.cost_usd)
        res.latency_s = time.monotonic() - t0
        self.on_usage(req, res)
        return res


def make_gateway(provider: str, roles: ModelRoles, budget: Budget,
                 on_usage: Callable[[LLMRequest, LLMResult], None]) -> Gateway:
    if os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1":
        from .fake import FakeGateway
        return FakeGateway(on_usage=on_usage)
    from .. import keys
    client = keys.make_client(provider)
    if provider == "claude":
        from .anthropic_backend import AnthropicGateway
        return AnthropicGateway(client, roles, budget, on_usage)
    from .openai_backend import OpenAIGateway
    return OpenAIGateway(client, roles, budget, on_usage)
