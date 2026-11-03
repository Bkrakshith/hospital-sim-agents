"""Chaos mode: a seeded Poisson process that draws random cards."""
from __future__ import annotations

import numpy as np

from wardzero_agents.config.models import CardDef

from .storyboard import CardFiring, WorldView

MAX_PER_TICK = 3


class Chaos:
    def __init__(self, rate_per_hour: float, deck: dict[str, CardDef], seed: int):
        self.rate = rate_per_hour
        self.deck = deck
        self.rng = np.random.default_rng(seed)
        self._n = 0
        self._next = self._gap() if rate_per_hour > 0 else float("inf")

    def _gap(self) -> float:
        return float(self.rng.exponential(3600.0 / self.rate))

    def tick(self, t: float, view: WorldView) -> list[CardFiring]:
        out: list[CardFiring] = []
        while self._next <= t and len(out) < MAX_PER_TICK:
            self._next += self._gap()
            firing = self._draw(t, view)
            if firing is not None:
                out.append(firing)
        if self._next <= t:  # cap hit: drop the backlog rather than burst later
            while self._next <= t:
                self._next += self._gap()
        return out

    def _draw(self, t: float, view: WorldView):
        present = sorted(view.present)
        cands = [c for _, c in sorted(self.deck.items()) if c.target_kind != "person" or present]
        if not cands:
            return None
        w = np.array([c.weight for c in cands], dtype=float)
        card = cands[int(self.rng.choice(len(cands), p=w / w.sum()))]
        target = None
        if card.target_kind == "person":
            target = present[int(self.rng.integers(len(present)))]
        self._n += 1
        return CardFiring(f"chaos-{self._n}", t, card.id, target, {}, "chaos")
