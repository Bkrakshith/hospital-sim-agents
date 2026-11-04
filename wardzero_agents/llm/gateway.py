"""Gateway types shared by every backend, the spend budget and the factory."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol

from ..config.models import ModelRoles

log = logging.getLogger(__name__)

Role = Literal["decide", "summarise", "vision", "robot", "report"]
ErrorKind = Literal["refusal", "invalid_json", "timeout", "rate_limited", "budget", "provider_error",
                    "bad_request", "auth", "model_not_found", "too_large", "max_tokens", "llm_disabled"]
# Errors that retrying cannot fix: the request, the key or the model is wrong.
PERMANENT_ERRORS = ("bad_request", "auth", "model_not_found", "too_large")


def status_error(status: Optional[int]) -> ErrorKind:
    """The error kind for an HTTP error status (429 is handled by the caller as rate_limited)."""
    if status == 400:
        return "bad_request"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "model_not_found"
    if status == 413:
        return "too_large"
    return "provider_error"


@dataclass
class LLMRequest:
    role: Role
    person: str
    system_fixed: str
    persona_card: str
    user: str
    schema: dict
    images: list[bytes] = field(default_factory=list)
    # Append-only text blocks that come before `user` (a person's day so far). The Claude backend
    # puts a cache breakpoint after the last one, so each call reads the previous call's blocks.
    history: list[str] = field(default_factory=list)

    @property
    def full_user(self) -> str:
        """Everything the user turn says, as one string, for backends that do not cache by block."""
        return "\n".join([*self.history, self.user])


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
    status: Optional[int] = None  # the HTTP status of a failed call, when there was one


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
                "vision": self.roles.vision, "robot": self.roles.robot,
                "report": self.roles.report}[role]

    async def _call(self, req: LLMRequest, model: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    async def complete(self, req: LLMRequest) -> LLMResult:
        model = self.model_for(req.role)
        t0 = time.monotonic()
        if self.budget.exhausted:
            res = LLMResult(False, None, "budget", model)
        else:
            try:
                res = await self._call(req, model)
            except Exception as exc:  # last resort: nothing escapes complete()
                log.error("gateway call failed: %s", type(exc).__name__)
                res = LLMResult(False, None, "provider_error", model)
            self.budget.charge(res.cost_usd)
        res.latency_s = time.monotonic() - t0
        self.on_usage(req, res)
        return res


def make_gateway(provider: str, roles: ModelRoles, budget: Budget,
                 on_usage: Callable[[LLMRequest, LLMResult], None]) -> Gateway:
    if os.environ.get("WARDZERO_TEST_FAKE_LLM") == "1":
        from .fake import FakeGateway
        return FakeGateway(on_usage=on_usage)
    if provider not in ("claude", "openai"):
        raise ValueError(f"unknown provider {provider!r}")
    from .. import keys
    client = keys.make_client(provider)
    if provider == "claude":
        from .anthropic_backend import AnthropicGateway
        return AnthropicGateway(client, roles, budget, on_usage)
    from .openai_backend import OpenAIGateway
    return OpenAIGateway(client, roles, budget, on_usage)
