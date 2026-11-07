"""`hhumans run --duration S` caps the run's length (re-QA bug 22): the staged copy of the scenario lasts S seconds, so
the sim ends the run at S through the normal end-of-run path (the Isaac host's video encode and the SIGINT to SIGTERM
grace are sized from that length), and a run that lasted its capped length is not "stopped early"."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from wardzero_agents import cli
from wardzero_agents.cli import main
from wardzero_agents.config.loader import load_scenario
from tests.test_cli import argv_of, fake_llm, isolated, launched, opt  # noqa: F401  (fixtures)

AGENTS_ROOT = Path(__file__).resolve().parents[1]
V1 = str(AGENTS_ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")


def _staged(children):
    return load_scenario(opt(argv_of(children, "stub_sim"), "--scenario"))


def test_duration_caps_the_staged_scenario(fake_llm, launched, capsys):
    assert load_scenario(V1).scenario.duration_min * 60 > 45
    assert main(["run", V1, "--no-panel", "--duration", "45"]) == 0
    (children,) = launched
    assert _staged(children).scenario.duration_min == pytest.approx(0.75)
    assert "0.75 min of sim time" in capsys.readouterr().out


def test_without_duration_the_scenario_keeps_its_length(fake_llm, launched):
    assert main(["run", V1, "--no-panel"]) == 0
    (children,) = launched
    assert _staged(children).scenario.duration_min == load_scenario(V1).scenario.duration_min


@pytest.mark.parametrize("bad", ["0", "-5", "3601", "nan"])
def test_a_bad_duration_is_refused(fake_llm, launched, capsys, bad):
    assert main(["run", V1, "--duration", bad]) == 2
    assert launched == []


def test_the_isaac_stop_grace_comes_from_the_capped_length(fake_llm, launched, monkeypatch):
    seen = {}
    real = cli._sim_child

    def spy(args, rd, sp, ts, ready, duration_s=0.0):
        seen["duration_s"] = duration_s
        return real(args, rd, sp, ts, ready, duration_s=duration_s)

    monkeypatch.setattr(cli, "_sim_child", spy)
    assert main(["run", V1, "--no-panel", "--duration", "120"]) == 0
    assert seen["duration_s"] == pytest.approx(120.0)


def test_a_run_that_lasted_its_capped_length_is_not_stopped_early(fake_llm, monkeypatch, tmp_path):
    def fake(children, **kw):
        run_dir = Path(opt(argv_of(children, "stub_sim"), "--run-dir"))
        (run_dir / "status.jsonl").write_text(json.dumps({"person": "P1", "t": 44.9}) + "\n")
        meta = json.loads((run_dir / "meta.json").read_text())
        (run_dir / "meta.json").write_text(json.dumps({**meta, "run_id": run_dir.name}))
        return 0

    monkeypatch.setattr(cli, "supervise", fake)
    monkeypatch.setattr(cli, "_post_run", lambda run_dir, report=True: None)
    assert main(["run", V1, "--no-panel", "--duration", "45"]) == 0
    (run_dir,) = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    assert not json.loads((run_dir / "meta.json").read_text()).get("stopped_early")


@pytest.mark.ros
@pytest.mark.skipif(os.environ.get("ROS_DISTRO") != "humble", reason="needs a sourced ROS 2 Humble")
def test_a_stub_run_ends_at_its_capped_length_through_the_normal_end(tmp_path):
    runs_root = tmp_path / "runs"
    env = dict(os.environ, WARDZERO_TEST_FAKE_LLM="1", WARDZERO_DATA_DIR=str(tmp_path / "data"),
               WARDZERO_RUNS_DIR=str(runs_root), PYTHONUNBUFFERED="1",
               ROS_DOMAIN_ID=os.environ.get("ROS_DOMAIN_ID") or str(random.randint(1, 101)))
    env.pop("WARDZERO_TEST_CLAUDE_BOT", None)
    prog = "import sys; from wardzero_agents.cli import main; sys.exit(main(sys.argv[1:]))"
    proc = subprocess.run([sys.executable, "-c", prog, "run", V1, "--sim", "stub", "--no-panel", "--no-report",
                           "--bot", "none", "--time-scale", "10", "--duration", "20"],
                          env=env, cwd=AGENTS_ROOT, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    (run,) = [p for p in runs_root.iterdir() if p.is_dir()]
    ts = [json.loads(x)["t"] for x in (run / "status.jsonl").read_text().splitlines() if x.strip()]
    assert 19.0 <= max(ts) <= 21.0
    assert "Ctrl+C again" not in proc.stdout  # the sim ended it, not a signal
    assert not json.loads((run / "meta.json").read_text()).get("stopped_early")
