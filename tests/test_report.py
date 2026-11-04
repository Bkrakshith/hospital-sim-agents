"""The report: number-checked narrative, html sections, batches and the CLI."""
import asyncio
import json
import re
from pathlib import Path

import pytest
from _rundir import make_ae_run, write

from wardzero_agents import cli
from wardzero_agents.bench.batch import run_batch, summarise_arm, wilson
from wardzero_agents.bench.report import build_report, check_numbers
from wardzero_agents.campaign.debrief import debrief
from wardzero_agents.config.presets import load_scenario_presets
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.memory import data_dir, memory_path
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.kb.embed import HashEmbedder

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)
AE = PRESETS["ae_waiting"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)


def report_policy(narrative, suggestions=()):
    def policy(req):
        if req.role == "report":
            return {"narrative": narrative, "suggestions": list(suggestions)}
        return default_policy(req)
    return policy


# ---- check_numbers ----------------------------------------------------------------------------------
def test_check_numbers_drops_invented():
    rep = {"trays": {"correct": 5, "total": 9}, "pass_rate": 0.5, "mean_wait_s": 12.46, "name": "P1"}
    text = ("The robot delivered 7 of 9 trays. It delivered 5 of 9 trays [03:12]. "
            "Half were wrong, which is 50% of them. About 12.5 seconds went by. It took 12.9 seconds. "
            "P1 was upset at [10:45].\nA second paragraph with 41%.\nA closing paragraph, kept.")
    out = check_numbers(text, rep)
    assert "7 of 9" not in out and "12.9" not in out and "41%" not in out
    assert "5 of 9 trays [03:12]." in out          # the stamp is exempt, the figures are known
    assert "50%" in out                           # a share written as a percentage
    assert "12.5 seconds" in out                  # rounded to the precision shown
    assert "P1 was upset at [10:45]." in out      # digits in a name and stamps are exempt
    assert out.count("\n") == 1                   # paragraphs survive


def test_check_numbers_keeps_text_without_numbers_and_decimals_are_not_split():
    out = check_numbers("It was slow. It took 2.5 s. Then it stopped!", {"x": 2.5})
    assert out == "It was slow. It took 2.5 s. Then it stopped!"
    assert check_numbers("It took 2.6 s. Fine.", {"x": 2.5}) == "Fine."


# ---- report -----------------------------------------------------------------------------------------
def debriefed_run(tmp_path, name="r1", **meta):
    rd = make_ae_run(tmp_path / name, **meta)
    asyncio.run(debrief(rd, FakeGateway(), preset=AE))
    return rd


def test_report_html_has_sections_and_invalid_banner(tmp_path):
    rd = debriefed_run(tmp_path)
    gw = FakeGateway(report_policy("The nurse was alerted [04:10]. The wait question went unanswered [01:40].",
                                   ["Answer questions about waiting within seconds."]))
    out = build_report(rd, AE, gw)
    page = out.read_text()
    for sec in ('id="criteria"', 'id="timeline"', 'id="ratings"', 'id="trust"', 'id="narrative"', 'id="facts"'):
        assert sec in page
    assert "INVALID" not in page and "prefers-color-scheme:dark" in page
    assert not re.search(r'(src|href)="https?:', page) and "<script" not in page
    assert "The nurse was alerted [04:10]." in page and "Answer questions about waiting" in page
    assert "Triage alerted within a minute" in page and "Waits answered within 30 seconds" in page
    rep = json.loads((rd / "report.json").read_text())
    assert rep["verdict"]["label"] == "FAIL" and rep["valid"] is True
    by = {c["id"]: c for c in rep["criteria"]}
    assert by["alerted"]["passed"] is True and by["wait"]["passed"] is False
    assert rep["run"]["word_of_mouth"] is False and rep["trust"] and rep["ratings"]
    assert "Word of mouth" in page and "mm:ss" in page
    # a fairness violation makes the run invalid
    write(rd, "fairness", [{"t": 5.0, "node": "robot_x", "topic": "/person/P1/status"}])
    page = build_report(rd, AE, None).read_text()
    assert "INVALID: fairness audit failed" in page and "No AI assessment" in page
    assert json.loads((rd / "report.json").read_text())["verdict"]["label"] == "INVALID"


def test_narrative_numbers_are_checked_and_other_scenarios_never_named(tmp_path):
    rd = debriefed_run(tmp_path)
    gw = FakeGateway(report_policy("Triage was alerted [04:10]. The robot took 999 seconds to answer [01:40].",
                                   ["Reply faster, about 77 times faster.", "Say the expected wait aloud."]))
    build_report(rd, AE, gw)
    rep = json.loads((rd / "report.json").read_text())
    assert "999" not in rep["narrative"] and "[04:10]" in rep["narrative"]
    assert rep["suggestions"] == ["Say the expected wait aloud."]
    page = (rd / "report.html").read_text()
    others = [pid for pid in PRESETS if pid != "ae_waiting"]
    assert not any(pid in page for pid in others)
    assert not any(PRESETS[pid].title in page for pid in others)


def test_the_narrative_prompt_carries_the_data_and_thirty_excerpts_at_most(tmp_path):
    rd = debriefed_run(tmp_path)
    write(rd, "says", [{"t": float(i), "person": "g1", "addressee": "P2", "text": f"line {i}"} for i in range(80)])
    seen = []

    def policy(req):
        if req.role == "report":
            seen.append(req)
        return {"narrative": "", "suggestions": []}
    build_report(rd, AE, FakeGateway(policy))
    req = seen[0]
    assert req.role == "report" and "[mm:ss]" in req.system_fixed
    assert json.loads(req.persona_card.split("\n", 1)[1])["verdict"]
    assert len([ln for ln in req.user.splitlines() if ln.startswith("[")]) <= 30


def test_report_for_a_legacy_run_without_a_preset(tmp_path):
    rd = make_ae_run(tmp_path / "r1", scenario_id="waiting_area_v1", preset_id=None)
    page = build_report(rd, None, None).read_text()
    assert "No verdict" in page and "no success criteria" in page


# ---- wilson and batches ------------------------------------------------------------------------------
def test_wilson():
    lo, hi = wilson(96, 100)
    assert lo == pytest.approx(0.90, abs=0.01) and hi == pytest.approx(0.985, abs=0.01)
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5


class Staged:
    class scenario:
        id = "ae_waiting"
        seed = 100

        class robot:
            profile = "other_bot"


def db_state(mem):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(mem)) as c:
        return sorted(c.execute("SELECT person_key, text FROM memories").fetchall())


def test_batch_restores_snapshot_between_repeats(tmp_path):
    mem = memory_path(data_dir())
    ltm = LongTermMemory(mem, HashEmbedder())
    ltm.ensure_seed("P1", ["I like gardens."])
    starts, calls = [], []

    def runner(seed, profile):
        starts.append(db_state(mem))
        calls.append((seed, profile))
        ltm.ensure_seed(f"extra{seed}", ["a new memory"])  # the run changes memory
        return make_ae_run(tmp_path / f"run{len(calls)}")

    batch = run_batch(Staged, 3, False, False, runner=runner, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert starts[0] == starts[1] == starts[2] and len(starts[0]) == 1
    assert db_state(mem) == starts[0]                  # the user's memory is back as it was
    assert calls == [(100, None), (101, None), (102, None)]
    data = json.loads((batch / "batch.json").read_text())
    assert data["label"] == "repeatability" and data["snapshot"].startswith("batch-") and data["n"] == 3
    arm = data["arms"]["primary"]
    assert arm["pass_rate"]["n"] == 3 and arm["pass_rate"]["k"] == 0
    assert arm["metrics"]["min_distance_m"]["mean"] is not None and "mean_wait_s" in arm["metrics"]
    page = (batch / "report.html").read_text()
    assert "Repeatability" in page and "Wilson" in page and "Built-in robot" not in page


def test_batch_carries_memory_and_runs_the_baseline_from_the_same_start(tmp_path):
    mem = memory_path(data_dir())
    ltm = LongTermMemory(mem, HashEmbedder())
    ltm.ensure_seed("P1", ["I like gardens."])
    starts, calls = [], []

    def runner(seed, profile):
        starts.append(len(db_state(mem)))
        calls.append((seed, profile))
        ltm.ensure_seed(f"extra{len(calls)}", ["a new memory"])
        return make_ae_run(tmp_path / f"run{len(calls)}")

    batch = run_batch(Staged, 2, True, True, runner=runner, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert calls == [(100, None), (101, None), (100, "g1_claude"), (101, "g1_claude")]
    assert starts == [1, 2, 1, 2]        # carried within an arm, both arms from the same start
    data = json.loads((batch / "batch.json").read_text())
    assert set(data["arms"]) == {"primary", "baseline"} and data["arms"]["baseline"]["profile"] == "g1_claude"
    page = (batch / "report.html").read_text()
    assert "Built-in robot (baseline)" in page and "Your robot" in page


def test_batch_without_a_memory_db_starts_every_repeat_empty(tmp_path):
    mem = memory_path(data_dir())
    seen = []

    def runner(seed, profile):
        seen.append(mem.exists())
        LongTermMemory(mem, HashEmbedder()).ensure_seed("P1", ["x"])
        return make_ae_run(tmp_path / f"run{len(seen)}")
    run_batch(Staged, 2, False, False, runner=runner, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert seen == [False, False]


def test_summarise_arm_counts_passes():
    rows = [{"passed": True, "metrics": {"contacts": 0.0}}, {"passed": False, "metrics": {"contacts": 2.0}}]
    s = summarise_arm(rows)
    assert s["pass_rate"]["k"] == 1 and s["metrics"]["contacts"] == {"mean": 1.0, "min": 0.0, "max": 2.0, "n": 2}


# ---- CLI --------------------------------------------------------------------------------------------
def test_cli_debrief_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    rd = make_ae_run(tmp_path / "runs" / "r1")
    assert cli.main(["debrief", str(rd)]) == 0
    assert (rd / "debrief.jsonl").exists() and (rd / "trust.jsonl").exists()
    assert cli.main(["report", str(rd)]) == 0
    assert (rd / "report.html").exists() and "Report:" in capsys.readouterr().out
    people = {json.loads(x)["person"] for x in (rd / "tokens.jsonl").read_text().splitlines()}
    assert {"debrief", "report"} <= people
    assert cli.main(["report", str(rd), "--no-narrative"]) == 0
    assert cli.main(["debrief", str(tmp_path / "nowhere")]) == 2


def test_cli_report_of_a_batch_folder(tmp_path, monkeypatch, capsys):
    mem = memory_path(data_dir())
    LongTermMemory(mem, HashEmbedder()).ensure_seed("P1", ["x"])
    batch = run_batch(Staged, 1, False, False, runner=lambda seed, profile: make_ae_run(tmp_path / "r"),
                      preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    (batch / "report.html").unlink()
    assert cli.main(["report", str(batch)]) == 0
    assert "Repeatability" in (batch / "report.html").read_text()


def test_run_flags_exist():
    a = cli.build_parser().parse_args(["run", "x.yaml", "--repeat", "3", "--carry-memory", "--baseline", "--no-report"])
    assert (a.repeat, a.carry_memory, a.baseline, a.no_report) == (3, True, True, True)


def _fake_supervise(children, **kw):
    """Stand in for the nodes: the agent node would have written a run id into meta.json."""
    rd = Path(children[0].argv[children[0].argv.index("--run-dir") + 1])
    meta = json.loads((rd / "meta.json").read_text())
    meta.update(run_id=rd.name, people={"P1": {"role": "patient", "preset_role": "patient", "static": False}})
    (rd / "meta.json").write_text(json.dumps(meta))
    write(rd, "status", [{"t": 1.0, "person": "P1", "x": 0.0, "y": 0.0, "yaw": 0.0, "posture": "standing",
                          "state": "idle", "progress": 0.0}])
    return 0


V1 = str(Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml")


def _runs(tmp_path):
    return sorted(p for p in (tmp_path / "runs").iterdir() if p.is_dir())


def test_a_normal_run_is_debriefed_and_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    monkeypatch.setattr(cli, "supervise", _fake_supervise)
    assert cli.main(["run", V1, "--no-panel"]) == 0
    (rd,) = _runs(tmp_path)
    assert (rd / "debrief.jsonl").exists() and (rd / "report.html").exists()


def test_no_report_flag_skips_debrief_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    monkeypatch.setattr(cli, "supervise", _fake_supervise)
    assert cli.main(["run", V1, "--no-panel", "--no-report"]) == 0
    (rd,) = _runs(tmp_path)
    assert not (rd / "report.html").exists() and not (rd / "debrief.jsonl").exists()


def test_run_repeat_makes_a_batch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    monkeypatch.setattr(cli, "supervise", _fake_supervise)
    assert cli.main(["run", V1, "--no-panel", "--repeat", "2", "--no-report"]) == 0
    runs = _runs(tmp_path)
    batch = next(p for p in runs if p.name.startswith("batch-"))
    data = json.loads((batch / "batch.json").read_text())
    assert data["label"] == "repeatability" and data["seeds"] == [data["seeds"][0], data["seeds"][0] + 1]
    assert len([p for p in runs if not p.name.startswith("batch-")]) == 2
    assert "repeatability" in capsys.readouterr().out


def test_batch_puts_memory_back_after_an_exception_and_removes_a_db_it_created(tmp_path):
    mem = memory_path(data_dir())
    ltm = LongTermMemory(mem, HashEmbedder())
    ltm.ensure_seed("P1", ["I like gardens."])
    start = db_state(mem)

    def boom(seed, profile):
        ltm.ensure_seed("extra", ["changed"])
        raise RuntimeError("run failed")
    with pytest.raises(RuntimeError):
        run_batch(Staged, 2, False, False, runner=boom, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert db_state(mem) == start
    mem.unlink()

    def create(seed, profile):
        LongTermMemory(mem, HashEmbedder()).ensure_seed("P1", ["x"])
        return make_ae_run(tmp_path / "rr")
    run_batch(Staged, 1, False, False, runner=create, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert not mem.exists()


def test_batch_with_carry_memory_keeps_the_last_state(tmp_path):
    mem = memory_path(data_dir())
    ltm = LongTermMemory(mem, HashEmbedder())
    ltm.ensure_seed("P1", ["I like gardens."])

    def runner(seed, profile):
        ltm.ensure_seed(f"extra{seed}", ["new"])
        return make_ae_run(tmp_path / f"c{seed}")
    run_batch(Staged, 2, True, False, runner=runner, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert len(db_state(mem)) == 3


def test_batch_headline_counts_invalid_runs(tmp_path):
    mem = memory_path(data_dir())
    LongTermMemory(mem, HashEmbedder()).ensure_seed("P1", ["x"])

    def runner(seed, profile):
        rd = make_ae_run(tmp_path / f"i{seed}")
        write(rd, "fairness", [{"t": 1.0, "node": "n", "topic": "/t"}])
        return rd
    b = run_batch(Staged, 2, False, False, runner=runner, preset=AE, batch_root=tmp_path / "out", mem_path=mem)
    assert "2 INVALID" in (b / "report.html").read_text()


def test_number_words_and_multipliers_are_checked():
    rep = {"trays": {"correct": 5, "total": 9}, "metrics": {"contacts": 0.0}}
    out = check_numbers("Seven of nine trays were delivered. Five of nine trays were delivered. Twice the robot "
                        "stopped. Half of them waited. No one complained. The third try failed.", rep)
    assert "Seven" not in out and "Twice" not in out and "third" not in out and "Half" not in out
    assert "Five of nine trays were delivered." in out and "No one complained." in out
    assert "Twice the robot stopped." in check_numbers("Twice the robot stopped.", {"metrics": {"stops": 2}})
    assert "Half of them waited." in check_numbers("Half of them waited.", {"share": 0.5})


def test_seed_run_index_and_tokens_are_not_citable():
    rep = {"run": {"seed": 7, "run_index": 4, "duration_s": 300.0}, "tokens": {"input": 1234, "cost_usd": 0.31},
           "metrics": {"contacts": 2.0}, "narrative": "It touched 9 people."}
    out = check_numbers("The robot stopped 7 times. It was run number 4. It used 1234 tokens. It cost 0.31. "
                        "It touched 2 people. The run lasted 300 seconds.", rep)
    assert out == "It touched 2 people. The run lasted 300 seconds."


def test_narrative_text_cannot_trigger_a_placeholder(tmp_path):
    rd = debriefed_run(tmp_path)
    page = build_report(rd, AE, FakeGateway(report_policy("Look {{facts}} and {{title}} here [01:00]."))).read_text()
    assert "{{facts}}" in page and page.count('id="facts"') == 1
