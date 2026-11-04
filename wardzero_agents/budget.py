"""The spending limit of one Start.

The number a user types is the total for one Start: the people, the robot and the post-run work (debrief
and report), across every repeat and the baseline arm. Each run gets what remains divided by the runs
left, and splits that between people, robot and post-run."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PEOPLE_SHARE, ROBOT_SHARE, POST_SHARE = 0.60, 0.20, 0.20
MIN_RUN_USD = 0.05  # a run with less than this left to spend is not started


@dataclass(frozen=True)
class RunBudget:
    total: float
    people: float
    robot: float
    post: float


def split(run_usd: float) -> RunBudget:
    run_usd = max(0.0, float(run_usd))
    return RunBudget(run_usd, run_usd * PEOPLE_SHARE, run_usd * ROBOT_SHARE, run_usd * POST_SHARE)


def per_run(remaining_usd: float, runs_left: int) -> float:
    return max(0.0, float(remaining_usd)) / max(1, int(runs_left))


def cap_reached(remaining_usd: float, cap_usd: float) -> bool:
    """Whether too little is left to start another run."""
    return remaining_usd < max(MIN_RUN_USD, 0.02 * cap_usd)


def run_cost(run_dir: "str | Path") -> float:
    """Everything a run folder has spent so far: people, robot, debrief and report (tokens.jsonl)."""
    total = 0.0
    try:
        for line in (Path(run_dir) / "tokens.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("cost_usd") or 0.0)
            except (ValueError, AttributeError, TypeError):
                continue
    except OSError:
        pass
    return total
