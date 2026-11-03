import json
import os
import sqlite3

import pytest

from wardzero_agents.kb.embed import HashEmbedder
from wardzero_agents.memory import data_dir, memory_path
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.memory.runlock import RunActive, RunLock, active_run
from wardzero_agents.memory.snapshots import (
    guarded_reset, guarded_restore, list_snapshots, restore_snapshot, save_snapshot)

K = "anxious_first_timer#7005"
K2 = "other_person#1"


@pytest.fixture
def mem(tmp_path):
    return LongTermMemory(memory_path(tmp_path), HashEmbedder())


def count_rows(mem):
    with sqlite3.connect(mem.path) as c:
        return c.execute("select count(*) from memories").fetchone()[0]


def test_paths(tmp_path, monkeypatch):
    assert memory_path(tmp_path) == tmp_path / "memory" / "longterm.sqlite"
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "d"))
    assert data_dir() == tmp_path / "d"
    monkeypatch.delenv("WARDZERO_DATA_DIR")
    assert str(data_dir()).endswith(".local/share/wardzero-agents")


def test_seed_then_search(mem):
    mem.ensure_seed(K, ["I have never been to this hospital before."])
    hits = mem.search(K, "first visit hospital", k=1, run_index=1)
    assert hits[0].kind == "seed" and hits[0].run_index == 0


def test_ensure_seed_only_if_empty(mem):
    mem.ensure_seed(K, ["a"])
    mem.ensure_seed(K, ["b", "c"])
    assert count_rows(mem) == 1


def test_begin_run(mem):
    assert mem.begin_run("r1") == 1
    assert mem.begin_run("r1") == 1
    assert mem.begin_run("r2") == 2


def test_recency_beats_old(mem):
    mem.begin_run("r1")
    mem.add(K, "r1", 10, "fact", "The robot said clinic 3 is upstairs", 0.5)
    for i in range(2, 10):
        mem.begin_run(f"r{i}")
    mem.add(K, "r9", 10, "fact", "The robot said clinic 3 is down the corridor", 0.5)
    assert "corridor" in mem.search(K, "where is clinic 3", 1, run_index=9)[0].text


def test_newer_t_wins_tie_in_run(mem):
    mem.begin_run("r1")
    mem.add(K, "r1", 1, "fact", "same words", 0.5)
    mem.add(K, "r1", 2, "fact", "same words", 0.5)
    with sqlite3.connect(mem.path) as c:
        c.execute("update memories set text='same words'")
    hits = mem.search(K, "same words", 2, run_index=1)
    assert len(hits) == 2 and hits[0].score == pytest.approx(hits[1].score)


def test_search_scoped_to_person(mem):
    mem.ensure_seed(K, ["alpha"])
    mem.ensure_seed(K2, ["beta"])
    assert [h.text for h in mem.search(K, "beta", 5, run_index=1)] == ["alpha"]
    assert sorted(mem.all_keys()) == sorted([K, K2])


def test_reset_restores_seed(mem):
    mem.add(K, "r1", 1, "episode", "shouted at robot", 0.9)
    mem.reset([K], {K: ["seed text"]})
    assert [h.text for h in mem.search(K, "robot", 5, run_index=2)] == ["seed text"]


def test_reset_other_person_untouched(mem):
    mem.add(K, "r1", 1, "episode", "one", 0.9)
    mem.add(K2, "r1", 1, "episode", "two", 0.9)
    mem.reset([K], {K: ["seed"]})
    assert [h.text for h in mem.search(K2, "two", 5, run_index=2)] == ["two"]


def test_dim_mismatch(tmp_path):
    m1 = LongTermMemory(memory_path(tmp_path), HashEmbedder(64))
    m1.add(K, "r1", 1, "fact", "x", 0.5)
    m2 = LongTermMemory(memory_path(tmp_path), HashEmbedder(32))
    with pytest.raises(ValueError, match="dim"):
        m2.search(K, "x", 1, run_index=1)


def test_snapshot_roundtrip(mem, tmp_path):
    mem.add(K, "r1", 1, "fact", "a", 0.5)
    before = count_rows(mem)
    p = save_snapshot(mem.path, "before")
    assert p == tmp_path / "snapshots" / "before.sqlite"
    meta = json.loads((tmp_path / "snapshots" / "before.json").read_text())
    assert meta["rows"] == before and meta["keys"] == [K] and "created" in meta
    mem.add(K, "r1", 2, "fact", "b", 0.5)
    restore_snapshot(mem.path, "before")
    assert count_rows(mem) == before
    assert [s["name"] for s in list_snapshots(tmp_path)] == ["before"]


def test_restore_missing(mem):
    with pytest.raises(FileNotFoundError, match="nope"):
        restore_snapshot(mem.path, "nope")


@pytest.mark.parametrize("bad", ["../x", "", "a/b", "x" * 65, "a b"])
def test_bad_snapshot_name(mem, bad):
    with pytest.raises(ValueError):
        save_snapshot(mem.path, bad)
    with pytest.raises(ValueError):
        restore_snapshot(mem.path, bad)


def test_reset_refused_during_run(tmp_path):
    with RunLock(tmp_path, "run-1"):
        assert active_run(tmp_path) == "run-1"
        with pytest.raises(RunActive, match="run run-1 is active; stop it first"):
            guarded_reset(tmp_path, [K], {})
        with pytest.raises(RunActive, match="run-1"):
            guarded_restore(tmp_path, "x")
        with pytest.raises(RunActive):
            with RunLock(tmp_path, "run-2"):
                pass
    assert active_run(tmp_path) is None


def test_guarded_when_idle(tmp_path):
    m = LongTermMemory(memory_path(tmp_path), HashEmbedder())
    m.add(K, "r1", 1, "fact", "a", 0.5)
    save_snapshot(m.path, "s")
    guarded_reset(tmp_path, [K], {K: ["seed"]})
    assert [h.text for h in m.search(K, "a", 5, run_index=1)] == ["seed"]
    guarded_restore(tmp_path, "s")
    assert [h.text for h in m.search(K, "a", 5, run_index=1)] == ["a"]


def test_stale_lock_removed(tmp_path):
    (tmp_path / "active_run.json").write_text(json.dumps({"run_id": "old", "pid": 2**22 + 12345}))
    assert active_run(tmp_path) is None
    assert not (tmp_path / "active_run.json").exists()
    with RunLock(tmp_path, "new"):
        assert json.loads((tmp_path / "active_run.json").read_text())["pid"] == os.getpid()
