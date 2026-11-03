"""Claude prices in $ per MTok: (input, output, cache read). Cache writes bill at 1.25 x input."""
from __future__ import annotations

from dataclasses import dataclass

PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_FACTOR = 1.25


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0


def cost(model: str, usage: Usage) -> float | None:
    """Dollar cost, or None for models without a price (OpenAI): tokens only."""
    p = PRICES.get(model)
    if p is None:
        return None
    pin, pout, pread = p
    return (usage.input * pin + usage.output * pout + usage.cache_read * pread
            + usage.cache_write * pin * CACHE_WRITE_FACTOR) / 1_000_000
