"""Score a run against a scenario's success criteria (spec 12). Pure: numbers come from metrics.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .metrics import METRICS, RunData

EPS = 1e-9
MUST_PASS_WEIGHT = 1.0  # a criterion with at least this weight must pass for the run to pass


@dataclass
class CriteriaResult:
    rows: list[dict] = field(default_factory=list)  # {id, label, value, threshold, op, passed, weight}
    score: float = 0.0  # 0..100: weighted share of passed criteria
    passed: bool = False


def _holds(op: str, value: float, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold - EPS
    if op == "<=":
        return value <= threshold + EPS
    return abs(value - threshold) <= EPS


def evaluate(preset, data: RunData) -> CriteriaResult:
    """Each criterion is computed from its metric. A metric that cannot be computed (None) gives
    passed=None and counts as failed. score = 100 x passed weight / total weight. passed requires
    every criterion with weight >= 1 to pass; lighter ones only move the score. No criteria: score 0,
    not passed."""
    rows = []
    total = got = 0.0
    must = True
    for c in preset.criteria:
        value: Optional[float] = METRICS[c.metric](data)
        ok = None if value is None else _holds(c.op, value, c.threshold)
        rows.append({"id": c.id, "label": c.label, "metric": c.metric, "value": value,
                     "threshold": c.threshold, "op": c.op, "weight": c.weight, "passed": ok})
        total += c.weight
        if ok:
            got += c.weight
        elif c.weight >= MUST_PASS_WEIGHT:
            must = False
    return CriteriaResult(rows, 100.0 * got / total if total else 0.0, bool(rows) and must)
