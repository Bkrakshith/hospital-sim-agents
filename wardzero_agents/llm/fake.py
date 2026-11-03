"""Deterministic offline gateway for tests. No network, no key, zero usage."""
from __future__ import annotations

import time
from typing import Callable, Optional

from .gateway import LLMRequest, LLMResult

LANDMARKS = ["entrance", "clinic_corridor", "ward_doors", "toilets", "reception_front", "nurse_call_point"]


class FakeRefusal(Exception):
    """Raise from a policy to simulate a model refusal."""


def _landmark(text: str) -> Optional[str]:
    return next((lm for lm in LANDMARKS if lm in text), None)


def _decision(kind: str, target: str = "", say: str = "") -> dict:
    return {"intent": {"kind": kind, "target": target, "text": ""}, "say": say, "addressee": "",
            "mood": "calm", "patience_delta": 0.0, "reason": "fake policy"}


def default_policy(req: LLMRequest) -> dict:
    if req.role == "summarise":
        return {"memories": []}
    u = req.user
    if "ROBOT SAID TO YOU" in u:
        lm = _landmark(u)
        return _decision("walk_to", lm, "Thank you") if lm else _decision("wait_standing", "", "Thank you")
    if "NUDGE:" in u:
        nudge = u.split("NUDGE:", 1)[1]
        lm = _landmark(nudge)
        if lm:
            return _decision("walk_to", lm)
        if "help" in nudge:
            return _decision("approach", "g1", "Excuse me, can you help us?")
        return _decision("wait_standing")
    return _decision("sit_idle")


class FakeGateway:
    def __init__(self, policy: Optional[Callable[[LLMRequest], dict]] = None, on_usage=None):
        self.policy = policy or default_policy
        self.on_usage = on_usage

    async def complete(self, req: LLMRequest) -> LLMResult:
        t0 = time.monotonic()
        model = "fake"
        try:
            res = LLMResult(True, self.policy(req), None, model, cost_usd=0.0)
        except FakeRefusal:
            res = LLMResult(False, None, "refusal", model, cost_usd=0.0)
        except Exception:
            res = LLMResult(False, None, "provider_error", model, cost_usd=0.0)
        res.latency_s = time.monotonic() - t0
        if self.on_usage:
            self.on_usage(req, res)
        return res
