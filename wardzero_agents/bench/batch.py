"""Repeat batches: the same scenario run N times, optionally against the built-in robot as a baseline.

Everything a batch reports is labelled "repeatability": it shows how stable a result is across seeds
and runs, not a measure of the robot beyond those runs. Without carry_memory every repeat starts from
the same memory snapshot (saved as batch-<id>-start); with it, memory carries from run to run."""
from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .criteria import evaluate
from .metric_names import METRIC_NAMES
from .metrics import RunData, compute_all

LABEL = "repeatability"
BASELINE_PROFILE = "g1_claude"
Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval (95 % by default) for k successes in n tries; (0, 1) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _spread(values: list[float]) -> Optional[dict]:
    return ({"mean": round(sum(values) / len(values), 3), "min": round(min(values), 3),
             "max": round(max(values), 3), "n": len(values)} if values else None)


def summarise_arm(runs: list[dict]) -> dict:
    k = sum(1 for r in runs if r["passed"])
    lo, hi = wilson(k, len(runs))
    metrics = {}
    for name in METRIC_NAMES:
        s = _spread([r["metrics"][name] for r in runs if r["metrics"].get(name) is not None])
        if s:
            metrics[name] = s
    return {"runs": runs, "pass_rate": {"k": k, "n": len(runs), "rate": round(k / len(runs), 3) if runs else 0.0,
                                        "lo": round(lo, 3), "hi": round(hi, 3)}, "metrics": metrics}


def _run_row(run_dir: Path, seed: int, preset) -> dict:
    from .report import _fairness_rows
    d = RunData.load(run_dir)
    res = evaluate(preset, d) if preset is not None else None
    valid = not _fairness_rows(run_dir)
    return {"run_dir": str(run_dir), "seed": seed, "valid": valid,
            "passed": bool(res and res.passed and valid), "score": round(res.score, 1) if res else 0.0,
            "metrics": {k: (round(v, 3) if v is not None else None) for k, v in compute_all(d).items()}}


def _forget_memory(mem: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(mem) + suffix).unlink(missing_ok=True)


def run_batch(scenario, n: int, carry_memory: bool = False, baseline: bool = False, *,
              runner: Callable[..., Path], preset=None, batch_root: "Path | None" = None,
              mem_path: "Path | None" = None, now: Optional[datetime] = None,
              log: Callable[[str], None] = lambda s: None) -> Path:
    """Run `n` repeats and write batch.json and report.html into a new batch-<id> folder.

    scenario is the staged scenario (anything with .scenario.seed and .scenario.robot.profile and
    .scenario.id). runner(seed=..., profile=...) performs one complete run, debrief and report
    included, and returns its run folder; profile is None for the scenario's own robot. With baseline
    the same seeds are repeated with the built-in profile from the same start memory."""
    from ..memory import data_dir, memory_path
    from ..memory.snapshots import restore_snapshot, save_snapshot
    if n < 1:
        raise ValueError("a batch needs at least one run")
    sc = scenario.scenario
    now = now or datetime.now()
    batch_id = f"{now:%Y%m%d-%H%M%S}"
    from ..cli import runs_dir
    root = Path(batch_root) if batch_root is not None else runs_dir()
    batch_dir = root / f"batch-{sc.id}-{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    mem = Path(mem_path) if mem_path is not None else memory_path(data_dir())
    snap = f"batch-{batch_id}-start"
    had_memory = mem.exists()
    if had_memory:
        save_snapshot(mem, snap)

    def reset() -> None:
        if had_memory:
            restore_snapshot(mem, snap)
        else:
            _forget_memory(mem)

    seeds = [sc.seed + i for i in range(n)]
    arms = [("primary", None)]
    skipped = None
    if baseline:
        if sc.robot.profile == BASELINE_PROFILE:
            skipped = "The scenario's robot already is the built-in one, so there is nothing to compare."
        else:
            arms.append(("baseline", BASELINE_PROFILE))
    out_arms = {}
    try:
        for arm, profile in arms:
            rows = []
            for i, seed in enumerate(seeds):
                if not carry_memory or i == 0:
                    reset()
                log(f"{arm}: run {i + 1} of {n}, seed {seed}")
                rows.append(_run_row(Path(runner(seed=seed, profile=profile)), seed, preset))
            out_arms[arm] = {"profile": profile or sc.robot.profile, **summarise_arm(rows)}
    finally:
        if not carry_memory:  # the user's campaign goes back to how it was before the batch
            reset()
    batch = {"label": LABEL, "batch_id": batch_id, "scenario_id": sc.id, "title": getattr(preset, "title", sc.id),
             "n": n, "carry_memory": carry_memory, "baseline": "baseline" in out_arms, "baseline_note": skipped,
             "seeds": seeds, "snapshot": snap if had_memory else None, "arms": out_arms}
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2, sort_keys=True), encoding="utf-8")
    (batch_dir / "report.html").write_text(render_batch_html(batch), encoding="utf-8")
    return batch_dir


def report_batch(batch_dir: "str | Path") -> Path:
    """Rewrite report.html of a batch folder from its batch.json."""
    batch_dir = Path(batch_dir)
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    out = batch_dir / "report.html"
    out.write_text(render_batch_html(batch), encoding="utf-8")
    return out


def _e(v) -> str:
    return html.escape(str(v), quote=True)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def render_batch_html(batch: dict) -> str:
    from .report import TEMPLATE
    style = re.search(r"<style>.*?</style>", TEMPLATE.read_text(encoding="utf-8"), re.S).group(0)
    arms = batch["arms"]
    names = {"primary": "Your robot", "baseline": "Built-in robot (baseline)"}
    head = "".join(f"<th>{_e(names[a])}</th>" for a in arms)
    pr = "".join(
        f'<td>{_e(arms[a]["pass_rate"]["k"])} of {_e(arms[a]["pass_rate"]["n"])} passed '
        f'({_pct(arms[a]["pass_rate"]["rate"])}); 95% Wilson interval {_pct(arms[a]["pass_rate"]["lo"])} to '
        f'{_pct(arms[a]["pass_rate"]["hi"])}'
        + (f'; <strong>{sum(1 for r in arms[a]["runs"] if not r["valid"])} INVALID (fairness audit failed)</strong>'
           if any(not r["valid"] for r in arms[a]["runs"]) else "") + "</td>" for a in arms)
    metric_names = [m for m in METRIC_NAMES if any(m in arms[a]["metrics"] for a in arms)]
    rows = ""
    for m in metric_names:
        cells = ""
        for a in arms:
            s = arms[a]["metrics"].get(m)
            cells += (f"<td class=\"n\">mean {s['mean']:g}, min {s['min']:g}, max {s['max']:g}</td>" if s
                      else "<td>not measured</td>")
        rows += f"<tr><td>{_e(m.replace('_', ' '))}</td>{cells}</tr>"
    runs = ""
    for a in arms:
        for r in arms[a]["runs"]:
            runs += (f"<tr><td>{_e(names[a])}</td><td class=\"n\">{_e(r['seed'])}</td>"
                     f"<td>{'pass' if r['passed'] else 'fail'}{'' if r['valid'] else ' (invalid)'}</td>"
                     f"<td class=\"n\">{_e(r['score'])}</td><td>{_e(Path(r['run_dir']).name)}</td></tr>")
    note = f'<p class="note">{_e(batch["baseline_note"])}</p>' if batch.get("baseline_note") else ""
    carry = "Memory carried from run to run." if batch["carry_memory"] else \
        "Every repeat started from the same memory snapshot."
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Repeatability: {_e(batch["title"])}</title>{style}</head><body><main>'
            f'<h1>Repeatability: {_e(batch["title"])}</h1>'
            f'<p class="sub">{_e(batch["n"])} runs, seeds {_e(", ".join(map(str, batch["seeds"])))}. {carry} '
            "This page shows how stable the result is across runs; it is not a verdict on the robot beyond them.</p>"
            f'{note}<h2>Pass rate</h2><div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>'
            f'<tr>{pr}</tr></tbody></table></div>'
            f'<h2>Spread of each measurement</h2><div class="wrap"><table><thead><tr><th>Measurement</th>{head}</tr>'
            f'</thead><tbody>{rows}</tbody></table></div>'
            f'<h2>Runs</h2><div class="wrap"><table><thead><tr><th>Robot</th><th>Seed</th><th>Result</th><th>Score</th>'
            f'<th>Run folder</th></tr></thead><tbody>{runs}</tbody></table></div></main></body></html>')
