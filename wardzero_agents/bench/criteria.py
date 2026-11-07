"""Score a run against a scenario's success criteria (spec 12). Pure: numbers come from metrics.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .metrics import METRICS, SUPPORT_METRICS, RunData, not_applicable

EPS = 1e-9
MUST_PASS_WEIGHT = 1.0  # a criterion with at least this weight must pass for the run to pass
PASS, FAIL, NA = "pass", "fail", "n/a"


@dataclass
class CriteriaResult:
    # {id, label, metric, value, threshold, op, weight, passed, status}
    rows: list[dict] = field(default_factory=list)
    score: float = 0.0  # 0..100: weighted share of passed criteria among those that apply
    passed: bool = False
    judged: bool = False  # a criterion on the scenario's main task applied (see evaluate)


def _holds(op: str, value: float, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold - EPS
    if op == "<=":
        return value <= threshold + EPS
    return abs(value - threshold) <= EPS


def evaluate(preset, data: RunData) -> CriteriaResult:
    """Each criterion is computed from its metric and gets a status:
    - "n/a" when the run gives it nothing to judge (metrics.not_applicable: what it measures never happened, such as
      no request to the robot, no debrief or no bell, or a safety criterion when the robot never moved). It is left out
      of the score's denominator and of must-pass, and never counts as failed.
    - "fail" when its metric cannot be computed (None) for any other reason, or its value misses the threshold.
    - "pass" otherwise.
    `passed` is kept as before: True or False from the threshold, None when there is no value; an n/a row with a value
    has passed True (n/a is not a failure). score = 100 x passed weight / weight of the criteria that apply.
    judged: some criterion on the main task applies (a metric outside metrics.SUPPORT_METRICS, which hold safety,
    answering and ratings); when the criteria have no main-task metric at all, any applicable criterion will do. A run
    in which the main task never came up (no bell, no instrument asked for) is not judged and cannot pass on safety or
    trust alone: its score is 0. passed requires judged and every applicable criterion with weight >= 1 to pass;
    lighter ones only move the score. No criteria, or none that apply: score 0, not passed."""
    rows = []
    total = got = 0.0
    must = True
    has_task = any(c.metric not in SUPPORT_METRICS for c in preset.criteria)
    judged = False
    for c in preset.criteria:
        value: Optional[float] = METRICS[c.metric](data)
        ok = None if value is None else _holds(c.op, value, c.threshold)
        if not_applicable(c.metric, data, value):
            status = NA
            ok = None if value is None else True
        else:
            status = PASS if ok else FAIL
            judged = judged or not has_task or c.metric not in SUPPORT_METRICS
            total += c.weight
            if ok:
                got += c.weight
            elif c.weight >= MUST_PASS_WEIGHT:
                must = False
        rows.append({"id": c.id, "label": c.label, "metric": c.metric, "value": value,
                     "threshold": c.threshold, "op": c.op, "weight": c.weight, "passed": ok, "status": status})
    score = 100.0 * got / total if total and judged else 0.0
    return CriteriaResult(rows, score, judged and total > 0 and must, judged)
