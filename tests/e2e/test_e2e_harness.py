"""The end-to-end harness itself (always run): it keeps everything it writes inside its work folder."""
import os
import time
from pathlib import Path

import pytest

import _e2e_harness as H


def test_the_scripted_debrief_writes_trust_under_the_run_not_the_users_data_dir(tmp_path, monkeypatch):
    home_data = tmp_path / "users_data"
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(home_data))
    run_dir = tmp_path / "work" / "run_claude"
    run_dir.mkdir(parents=True)
    seen = {}

    async def fake_debrief(rd, gateway, *, preset=None, db_path=None, **kw):
        seen["db_path"] = db_path

    import wardzero_agents.bench.metrics as metrics
    import wardzero_agents.campaign.debrief as camp
    import wardzero_agents.config.presets as presets
    monkeypatch.setattr(camp, "debrief", fake_debrief)
    monkeypatch.setattr(metrics.RunData, "load", classmethod(lambda cls, rd: type("D", (), {"meta": {}})()))
    monkeypatch.setattr(presets, "preset_for_run", lambda meta: None)
    H.debrief(run_dir)
    assert seen["db_path"] is not None
    db = Path(seen["db_path"]).resolve()
    assert db.is_relative_to((tmp_path / "work").resolve()) and not db.is_relative_to(home_data.resolve())


def test_a_run_that_times_out_is_stopped_with_the_processes_it_started(tmp_path):
    # like cli run: the launcher starts a child in a session of its own and stops it when it is told to stop
    pidfile = tmp_path / "child.pid"
    script = (f"setsid sleep 300 & c=$!; echo $c > {pidfile}; trap 'kill $c; exit 143' TERM; "
              "while true; do sleep 0.1; done")
    with pytest.raises(TimeoutError):
        H.run_bounded(["bash", "-c", script], env=dict(os.environ), timeout_s=1.0)
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and Path(f"/proc/{child}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{child}").exists(), "the launcher's child outlived the timeout"


def test_a_run_that_ends_in_time_returns_its_exit_code_and_output():
    code, out, err = H.run_bounded(["bash", "-c", "echo hi; echo oops >&2; exit 3"], env=dict(os.environ),
                                   timeout_s=10.0)
    assert code == 3 and out == "hi\n" and err.endswith("oops\n")
