"""The spending limit is the total for one Start: split between people, robot and post-run, divided over the runs."""
import asyncio
import json
from pathlib import Path

import pytest

from wardzero_agents import budget, cli
from wardzero_agents.bench import batch as batch_mod
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.llm.gateway import BaseGateway, Budget, LLMRequest, LLMResult

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")


def test_split_adds_up_to_the_run_budget():
    b = budget.split(10.0)
    assert b.people + b.robot + b.post == pytest.approx(10.0) and b.total == 10.0
    assert min(b.people, b.robot, b.post) > 0


def test_per_run_is_what_remains_over_the_runs_left():
    assert budget.per_run(6.0, 3) == 2.0 and budget.per_run(-1, 2) == 0.0 and budget.per_run(5, 0) == 5.0
    assert budget.cap_reached(0.01, 5.0) and not budget.cap_reached(1.0, 5.0)


def test_run_cost_sums_every_stream_of_the_folder(tmp_path):
    (tmp_path / "tokens.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"person": "P1", "cost_usd": 0.1}, {"person": "robot", "cost_usd": 0.2},
        {"person": "debrief", "cost_usd": None}, {"person": "report", "cost_usd": 0.05})) + "\nnot json\n")
    assert budget.run_cost(tmp_path) == pytest.approx(0.35) and budget.run_cost(tmp_path / "none") == 0.0


@pytest.fixture
def fake_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    calls = []

    def start(args, loaded, overrides, meta, out=None, *, run_usd=None, baseline=False):
        d = tmp_path / "runs" / f"run{len(calls)}"
        d.mkdir(parents=True)
        (d / "tokens.jsonl").write_text(json.dumps({"cost_usd": calls_cost[0]}) + "\n")
        out["run_dir"] = d
        calls.append({"run_usd": run_usd, "baseline": baseline, "seed": overrides.get("seed")})
        return 0

    calls_cost = [0.0]
    monkeypatch.setattr(cli, "_start_live", start)
    monkeypatch.setattr(batch_mod, "_run_row", lambda d, seed, preset: {
        "run_dir": str(d), "seed": seed, "valid": True, "passed": True, "score": 1.0, "metrics": {}})
    return calls, calls_cost


def batch_args(*extra):
    return cli.build_parser().parse_args(["run", V1, "--sim", "stub", *extra])


def with_budget(usd, profile=None):
    loaded = load_scenario(V1)
    robot = loaded.scenario.robot.model_copy(update={"profile": profile} if profile else {})
    loaded.scenario = loaded.scenario.model_copy(update={"budget_usd": usd, "robot": robot})
    return loaded


def test_each_run_gets_what_remains_divided_by_the_runs_left(fake_batch):
    calls, cost = fake_batch
    cost[0] = 1.0  # every run spends $1 of its share
    assert cli._cmd_batch(batch_args("--repeat", "3", "--baseline"), with_budget(12.0, "my_robot")) == 0
    shares = [c["run_usd"] for c in calls]
    assert shares[0] == pytest.approx(2.0)  # 12 over 6 runs
    assert shares[1] == pytest.approx(11 / 5)  # the first run spent 1 of its 2
    assert [c["baseline"] for c in calls] == [False] * 3 + [True] * 3


def test_the_batch_stops_cleanly_at_the_cap_and_reports(fake_batch, tmp_path, capsys):
    calls, cost = fake_batch
    cost[0] = 2.0  # each run spends $2 against a $5 limit with 10 repeats
    assert cli._cmd_batch(batch_args("--repeat", "10"), with_budget(5.0)) == 0
    assert len(calls) == 3  # 5 -> 3 -> 1 -> -1: no run starts once the limit is gone
    (bdir,) = list((tmp_path / "runs").glob("batch-*"))
    b = json.loads((bdir / "batch.json").read_text())
    assert "spending limit" in b["stopped"] and b["cap_usd"] == 5.0 and b["spent_usd"] == pytest.approx(6.0)
    assert len(b["arms"]["primary"]["runs"]) == 3
    assert "spending limit" in (bdir / "report.html").read_text()


def test_start_live_gives_the_run_its_share(monkeypatch, tmp_path):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    monkeypatch.setattr(cli, "supervise", lambda children, **kw: 0)
    out = {}
    assert cli._start_live(batch_args(), with_budget(10.0), {}, {}, out=out, run_usd=4.0) == 0
    meta = json.loads((out["run_dir"] / "meta.json").read_text())
    rb = budget.split(4.0)
    assert meta["post_budget_usd"] == pytest.approx(rb.post) and meta["run_budget_usd"] == 4.0
    staged = load_scenario(next((out["run_dir"] / "config").glob("waiting_area_v1.yaml")))
    assert staged.scenario.budget_usd == pytest.approx(rb.people)
    assert json.loads((out["run_dir"] / "robot_config.json").read_text())["budget_usd"] == pytest.approx(rb.robot)


# ---- in-flight calls are counted ----------------------------------------------------------------
class Slow(BaseGateway):
    def __init__(self, budget):
        super().__init__(None, budget, lambda *_: None)
        self.started = 0

    async def _call(self, req, model):
        self.started += 1
        await asyncio.sleep(0.05)
        return LLMResult(True, {}, None, model, cost_usd=0.03)

    def model_for(self, role):
        return "m"


def test_concurrent_calls_cannot_overshoot_the_limit_by_the_number_in_flight():
    async def go():
        gw = Slow(Budget(0.05))
        req = LLMRequest(role="decide", person="p", system_fixed="", persona_card="", user="", schema={})
        res = await asyncio.gather(*(gw.complete(req) for _ in range(20)))
        return gw, res
    gw, res = asyncio.run(go())
    assert gw.started <= 5 and gw.budget.spent <= 0.16  # 20 at once used to all go through ($0.60)
    assert sum(1 for r in res if r.error == "budget") >= 15
