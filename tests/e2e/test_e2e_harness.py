"""The end-to-end harness itself (always run): it keeps everything it writes inside its work folder."""
from pathlib import Path

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
