"""hhumans run --sim isaac: the Isaac host child, its flags, and the default backend."""
import json
from pathlib import Path

import pytest

from wardzero_agents import cli
from wardzero_agents.cli import main

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    isaac = tmp_path / "isaacsim"
    isaac.mkdir()
    (isaac / "python.sh").write_text("#!/bin/sh\n")  # an install has python.sh
    for name, env in (("stage.usd", "WARDZERO_STAGE_USD"), ("collision.usd", "WARDZERO_COLLISION_USD")):
        (tmp_path / name).write_text("x")
        monkeypatch.setenv(env, str(tmp_path / name))
    monkeypatch.setattr(cli, "ISAAC_ROOT", isaac)


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def fake(children, **kw):
        calls.append((children, kw))
        return 0

    monkeypatch.setattr(cli, "supervise", fake)
    return calls


def opt(argv, flag):
    return argv[argv.index(flag) + 1]


def child(calls, name):
    (children, kw), = calls
    return next(c for c in children if c.name == name), children, kw


def test_live_flags_parse():
    p = cli.build_parser()
    a = p.parse_args(["run", "x.yaml", "--sim", "isaac", "--headed", "--record", "--full-site"])
    assert a.sim == "isaac" and a.headed is True and a.record and a.full_site
    a = p.parse_args(["run", "x.yaml", "--headless"])
    assert a.sim is None and a.headed is False and not a.record and not a.full_site
    assert p.parse_args(["run", "x.yaml"]).headed is None
    with pytest.raises(SystemExit):
        p.parse_args(["run", "x.yaml", "--headed", "--headless"])
    with pytest.raises(SystemExit):
        p.parse_args(["run", "x.yaml", "--sim", "unreal"])


def test_default_sim_is_isaac_when_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("WARDZERO_SIM", raising=False)
    assert cli.default_sim() == "isaac"
    monkeypatch.setattr(cli, "ISAAC_ROOT", tmp_path / "nowhere")
    assert cli.default_sim() == "stub"
    monkeypatch.setenv("WARDZERO_SIM", "stub")
    monkeypatch.setattr(cli, "ISAAC_ROOT", ROOT)
    assert cli.default_sim() == "stub"


def test_run_isaac_starts_the_isaac_host_instead_of_the_stub(launched, monkeypatch, capsys):
    monkeypatch.setenv("ROS_DOMAIN_ID", "17")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert main(["run", V1, "--sim", "isaac", "--record", "--full-site", "--robot-domain", "5",
                 "--time-scale", "2", "--lockstep", "--no-report"]) == 0
    host, children, kw = child(launched, "isaac_host")
    assert [c.name for c in children][:3] == ["isaac_host", "agents", "director"]
    assert "stub_sim" not in [c.name for c in children]
    assert host.ends_run and host.argv[0] == str(cli.ISAAC_HOST_SH)
    a = host.argv
    assert opt(a, "--private-domain") == "17" and opt(a, "--robot-domain") == "5"
    assert "--record" in a and "--full-site" in a and "--headless" in a and "--lockstep" in a
    assert opt(a, "--time-scale") == "2.0" and opt(a, "--wait-for") == "agents,director"
    run_dir = Path(opt(a, "--run-dir"))
    assert Path(opt(a, "--scenario")).parent == run_dir / "config"
    env = host.env if host.env is not None else kw["env"]
    assert "ANTHROPIC_API_KEY" not in env  # the Isaac host never gets the key
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["sim"] == "isaac" and meta["record"] is True and meta["full_site"] is True


def test_headed_flag_reaches_the_host(launched):
    assert main(["run", V1, "--sim", "isaac", "--headed", "--no-report"]) == 0
    host, _, _ = child(launched, "isaac_host")
    assert "--headed" in host.argv and "--headless" not in host.argv


def test_default_backend_follows_default_sim(launched, monkeypatch):
    monkeypatch.delenv("WARDZERO_SIM", raising=False)
    assert main(["run", V1, "--no-report"]) == 0
    child(launched, "isaac_host")


def test_stub_keeps_working_and_records_its_name(launched):
    assert main(["run", V1, "--sim", "stub", "--no-report"]) == 0
    stub, _, _ = child(launched, "stub_sim")
    meta = json.loads((Path(opt(stub.argv, "--run-dir")) / "meta.json").read_text())
    assert meta["sim"] == "stub" and "record" not in meta


def test_isaac_flags_need_isaac(launched, capsys):
    assert main(["run", V1, "--sim", "stub", "--record"]) == 2
    assert "--record" in capsys.readouterr().err


def test_isaac_missing_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ISAAC_ROOT", tmp_path / "nowhere")
    assert main(["run", V1, "--sim", "isaac"]) == 2
    assert "Isaac Sim" in capsys.readouterr().err


def test_record_res_and_preview_reach_the_host(launched):
    assert main(["run", V1, "--sim", "isaac", "--record", "--record-res", "1920x1080", "--preview",
                 "--no-report"]) == 0
    host, _, _ = child(launched, "isaac_host")
    a = host.argv
    assert opt(a, "--record-res") == "1920x1080" and "--record" in a and "--preview" in a
    meta = json.loads((Path(opt(a, "--run-dir")) / "meta.json").read_text())
    assert meta["record_res"] == "1920x1080" and meta["lockstep"] is True  # a recorded run is in lockstep


def test_record_res_defaults_and_is_checked(launched, capsys):
    assert main(["run", V1, "--sim", "isaac", "--record", "--no-report"]) == 0
    host, _, _ = child(launched, "isaac_host")
    assert opt(host.argv, "--record-res") == "1280x720" and "--preview" not in host.argv
    assert main(["run", V1, "--sim", "isaac", "--record-res", "huge"]) == 2
    assert "--record-res" in capsys.readouterr().err


def test_preview_needs_the_isaac_host(launched, capsys):
    assert main(["run", V1, "--sim", "stub", "--preview"]) == 2
    assert "--preview" in capsys.readouterr().err


def test_an_isaac_run_waits_for_kit_to_exit_before_calling_the_agents_exit_early(launched):
    """The agents and director leave on the host's stop message; Kit takes longer than 5 s to follow."""
    assert main(["run", V1, "--sim", "isaac", "--no-report"]) == 0
    (_, kw), = launched
    assert kw["early_exit_grace_s"] >= 60
    launched.clear()
    assert main(["run", V1, "--sim", "stub", "--no-report"]) == 0
    assert launched[0][1]["early_exit_grace_s"] == cli.EARLY_EXIT_GRACE_S


# ---- the Isaac host gets time to finish its videos (QA bug 21) -----------------------------------------------
import os
import signal
import sys
import time


def _py(code):
    return [sys.executable, "-c", code]


def test_a_child_with_a_longer_stop_grace_is_not_sent_sigterm_while_it_finishes(tmp_path, capsys):
    """The host keeps encoding follow.mp4 after SIGINT: SIGTERM 10 s later killed it (exit 143 in every QA run)."""
    done = tmp_path / "encoded"
    slow = ("import signal, sys, time, pathlib\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))\n"
            f"def h(*a):\n    time.sleep(1.2); pathlib.Path({str(done)!r}).write_text('x'); sys.exit(0)\n"
            "signal.signal(signal.SIGINT, h)\nwhile True: time.sleep(0.05)")
    code = cli.supervise([cli.Child("ender", _py("import time; time.sleep(0.3)"), ends_run=True),
                          cli.Child("isaac_like", _py(slow), term_after_s=10.0)],
                         term_after_s=0.3, kill_after_s=5)
    assert code == 0 and done.exists()
    assert "143" not in capsys.readouterr().err


def test_the_default_stop_grace_still_applies_to_other_children(tmp_path):
    slow = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))\n"
            "signal.signal(signal.SIGINT, lambda *a: time.sleep(3) or sys.exit(0))\nwhile True: time.sleep(0.05)")
    t0 = time.monotonic()
    code = cli.supervise([cli.Child("ender", _py("import time; time.sleep(0.3)"), ends_run=True),
                          cli.Child("agents", _py(slow))], term_after_s=0.3, kill_after_s=5)
    assert code == 1 and time.monotonic() - t0 < 3


def test_a_recording_isaac_host_gets_a_stop_grace_for_its_encode(launched):
    assert main(["run", V1, "--sim", "isaac", "--record", "--no-report"]) == 0
    host, children, _ = child(launched, "isaac_host")
    from wardzero_agents.config.loader import load_scenario
    minutes = load_scenario(V1).scenario.duration_min
    # every frame of the follow camera is encoded after the stop: at least the run's frames at the encode rate
    assert host.term_after_s >= minutes * 60 * cli.ISAAC_MAX_FPS / cli.ISAAC_ENCODE_FPS
    assert host.term_after_s > cli.TERM_AFTER_S
    others = [c for c in children if c.name != "isaac_host"]
    assert all(c.term_after_s is None for c in others)


def test_an_isaac_host_that_does_not_record_still_gets_more_than_the_default(launched):
    assert main(["run", V1, "--sim", "isaac", "--no-report"]) == 0
    host, _, _ = child(launched, "isaac_host")
    assert cli.TERM_AFTER_S < host.term_after_s <= cli.ISAAC_STOP_BASE_S


def test_a_host_problem_reaches_the_run_meta_and_the_console(monkeypatch, capsys):
    """people_not_animated in host.jsonl (QA bug 12): the run record says so, and so does the console."""
    def fake(children, **kw):
        host = next(c for c in children if c.name == "isaac_host")
        rd = Path(opt(host.argv, "--run-dir"))
        row = {"kind": "problem", "what": "people_not_animated", "missing": ["N1", "S1"], "bound": 0, "expected": 2,
               "message": "people bound: 0 of 2 characters have a Behavior agent; N1, S1 will not walk or animate"}
        (rd / "host.jsonl").write_text(json.dumps({"kind": "startup"}) + "\n" + json.dumps(row) + "\n")
        return 0

    monkeypatch.setattr(cli, "supervise", fake)
    assert main(["run", V1, "--sim", "isaac", "--no-report"]) == 0
    runs = sorted((Path(os.environ["WARDZERO_RUNS_DIR"])).iterdir())
    meta = json.loads((runs[-1] / "meta.json").read_text())
    assert meta["isaac_problems"][0]["what"] == "people_not_animated"
    assert "0 of 2" in capsys.readouterr().err


def test_a_run_without_host_problems_has_none_in_meta(launched):
    assert main(["run", V1, "--sim", "isaac", "--no-report"]) == 0
    host, _, _ = child(launched, "isaac_host")
    meta = json.loads((Path(opt(host.argv, "--run-dir")) / "meta.json").read_text())
    assert "isaac_problems" not in meta
