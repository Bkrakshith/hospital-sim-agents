import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Intent, Say, WorldEvent
from wardzero_agents.replay import ExactScript, IntentScript, iter_stream, load_meta
from wardzero_agents.runlog import RunLog, new_run_dir


def _intent(i, t):
    return {"schema": "wz.v1", "id": i, "person": "P1", "t": t, "kind": "sit_idle"}


def test_write_and_iterate(tmp_path):
    rl = RunLog(tmp_path)
    rl.write("intents", {"t": 2.0, "person": "P1"})
    rl.write("intents", {"t": 1.0, "person": "P2"})
    assert [r["person"] for r in iter_stream(tmp_path, "intents")] == ["P2", "P1"]


def test_stable_for_equal_t(tmp_path):
    rl = RunLog(tmp_path)
    for p in "abc":
        rl.write("intents", {"t": 1.0, "person": p})
    assert [r["person"] for r in iter_stream(tmp_path, "intents")] == ["a", "b", "c"]


def test_redacts_secrets(tmp_path):
    key = "sk-ant-" + "api03-abcdefghijk"
    RunLog(tmp_path).write("decisions", {"t": 0, "note": key, "n": {"l": [key]}})
    text = (tmp_path / "decisions.jsonl").read_text()
    assert "abcdefghijk" not in text


def test_unknown_stream(tmp_path):
    with pytest.raises(ValueError):
        RunLog(tmp_path).write("decisons", {"t": 0})
    RunLog(tmp_path).write("summaries", {"t": 0})


def test_truncated_last_line(tmp_path):
    RunLog(tmp_path).write("intents", {"t": 1.0, "person": "P1"})
    with open(tmp_path / "intents.jsonl", "a") as f:
        f.write('{"t": 2.0, "per')
    assert len(list(iter_stream(tmp_path, "intents"))) == 1


def test_thread_safe(tmp_path):
    rl = RunLog(tmp_path)
    def work(k):
        for i in range(50):
            rl.write("tokens", {"t": float(i), "k": k})
    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(list(iter_stream(tmp_path, "tokens"))) == 400


def test_meta_roundtrip_redacted(tmp_path):
    rl = RunLog(tmp_path)
    rl.write_meta({"run_id": "r", "host": "sk-ant-" + "api03-zzzzzzzzzz"})
    meta = load_meta(tmp_path)
    assert meta["run_id"] == "r" and "zzzzzzzzzz" not in json.dumps(meta)
    assert not (tmp_path / "meta.json.tmp").exists()


def test_intent_script_due(tmp_path):
    rl = RunLog(tmp_path)
    rl.write("intents", _intent("i1", 1.0))
    rl.write("intents", _intent("i2", 5.0))
    s = IntentScript(tmp_path)
    assert len(s.due(2.0)) == 1 and len(s.due(2.0)) == 0 and len(s.due(6.0)) == 1


def test_intent_script_merges_types(tmp_path):
    rl = RunLog(tmp_path)
    rl.write("world", {"schema": "wz.v1", "id": "w1", "t": 3.0, "kind": "alarm"})
    rl.write("says", {"schema": "wz.v1", "person": "P1", "t": 2.0, "text": "hi"})
    rl.write("intents", _intent("i1", 1.0))
    out = IntentScript(tmp_path).due(10)
    assert [type(o) for o in out] == [Intent, Say, WorldEvent]


def test_exact_script(tmp_path):
    rl = RunLog(tmp_path)
    rl.write("status", {"t": 2.0, "x": 1})
    rl.write("audio", {"t": 1.0, "x": 2})
    rl.write("intents", {"t": 0.5})
    s = ExactScript(tmp_path)
    assert [n for n, _ in s.due(5)] == ["audio", "status"] and s.due(5) == []


def test_run_dir_name_and_clash(tmp_path):
    now = datetime(2026, 11, 3, 14, 5, 9)
    a = new_run_dir(tmp_path, "waiting_area_v1", now)
    b = new_run_dir(tmp_path, "waiting_area_v1", now)
    c = new_run_dir(tmp_path, "waiting_area_v1", now)
    assert a.name == "20261103-140509_waiting_area_v1"
    assert b.name == a.name + "-2" and c.name == a.name + "-3"


def test_copy_configs(tmp_path):
    loaded = load_scenario(Path(__file__).resolve().parent.parent / "config/scenarios/waiting_area_v1.yaml")
    RunLog(tmp_path).copy_configs(loaded)
    names = {p.name for p in (tmp_path / "config").iterdir()}
    assert {"waiting_area_v1.yaml", "personas.yaml", "waiting_area.yaml", "signs.yaml"} <= names


def test_redacts_str_of_objects_and_keys(tmp_path):
    key = "sk-ant-" + "api03-abcdefghijk"

    class Box:
        def __str__(self):
            return "box " + key

    rl = RunLog(tmp_path)
    rl.write("decisions", {"t": 0, "e": Exception(key), "b": Box(), "p": Path("/x/" + key), (1, key): 1})
    rl.write_meta({"b": Box(), "e": ValueError(key)})
    assert "abcdefghijk" not in (tmp_path / "decisions.jsonl").read_text()
    assert "abcdefghijk" not in (tmp_path / "meta.json").read_text()
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_missing_or_bad_t_is_zero(tmp_path):
    rl = RunLog(tmp_path)
    rl.write("intents", {"t": 2.0, "person": "a"})
    rl.write("intents", {"person": "b"})
    rl.write("intents", {"t": "x", "person": "c"})
    assert [r["person"] for r in iter_stream(tmp_path, "intents")] == ["b", "c", "a"]


def test_exact_script_ties_across_streams(tmp_path):
    rl = RunLog(tmp_path)
    for s in ("cards", "audio", "status"):
        rl.write(s, {"t": 1.0})
    assert [n for n, _ in ExactScript(tmp_path).due(1.0)] == ["status", "audio", "cards"]


def test_run_tail_reads_only_complete_new_lines(tmp_path):
    from wardzero_agents.runlog import RunTail
    log = RunLog(tmp_path)
    tail = RunTail(tmp_path, ("says", "tokens"))
    assert tail.poll() == []
    log.write("says", {"t": 1.0, "person": "P1", "text": "hi"})
    with open(tmp_path / "tokens.jsonl", "a") as f:
        f.write('{"t": 2.0, "cost_usd": 0.5}\n{"t": 3.0, "cost')
    got = tail.poll()
    assert got == [("says", {"t": 1.0, "person": "P1", "text": "hi"}), ("tokens", {"t": 2.0, "cost_usd": 0.5})]
    with open(tmp_path / "tokens.jsonl", "a") as f:
        f.write('_usd": 0.25}\n')
    assert tail.poll() == [("tokens", {"t": 3.0, "cost_usd": 0.25})]
    assert tail.poll() == []


def test_scripts_report_last_t_and_exhausted(tmp_path):
    log = RunLog(tmp_path)
    for t in (3.0, 1.0):
        log.write("status", {"t": t, "person": "P1"})
    s = ExactScript(tmp_path)
    assert s.last_t == 3.0 and not s.exhausted
    s.due(2.0)
    assert not s.exhausted
    s.due(3.0)
    assert s.exhausted and ExactScript(tmp_path / "none").last_t == 0.0
