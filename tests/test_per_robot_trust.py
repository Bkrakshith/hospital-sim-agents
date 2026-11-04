"""Trust and the memories about a robot are kept per robot, and the run lock covers the whole of a run command."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from _rundir import make_ae_run

from wardzero_agents import cli
from wardzero_agents.campaign import robot_key, start_campaign
from wardzero_agents.campaign.debrief import debrief
from wardzero_agents.campaign.trust import TrustStore
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_roles
from wardzero_agents.llm.fake import FakeGateway
from wardzero_agents.memory import data_dir, memory_path
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.kb.embed import HashEmbedder
from wardzero_agents.memory.runlock import RunLock, active_run

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")
ROLES = load_roles(ROOT / "config")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)


def test_robot_key_keeps_g1_rows_working():
    assert robot_key("g1_claude") == "g1" and robot_key("acme_arm") == "acme_arm"


def test_a_second_robot_does_not_inherit_the_first_robots_trust():
    loaded = load_scenario(V1)
    key = loaded.instances[0].memory_key
    ltm = LongTermMemory(memory_path(data_dir()), HashEmbedder())
    store = TrustStore(memory_path(data_dir()))
    base = store.get(key, "robot_a", ROLES["patient"])
    from dataclasses import replace
    store.put_many([replace(base, trust=0.05, stage=4, violations=3)])  # robot A burned this person
    ltm.begin_run("r1")

    def stage_with(profile):
        scen = loaded.scenario.model_copy(update={"robot": loaded.scenario.robot.model_copy(update={"profile": profile})})
        view = SimpleNamespace(scenario=scen, instances=loaded.instances)
        stages, mems, _ = start_campaign(view, ltm, memory_path(data_dir()), 1, ROLES)
        return stages[key]

    assert stage_with("robot_a") == 4
    assert stage_with("robot_b") != 4  # B starts from the role's own default
    assert stage_with("g1_claude") != 4


def test_robot_memories_are_tagged_per_robot():
    ltm = LongTermMemory(memory_path(data_dir()), HashEmbedder())
    ltm.begin_run("r1")
    ltm.add("k", "r1", 1.0, "episode", "the robot was rude", 0.9, tag="robot:robot_a")
    assert ltm.has_robot_memories("k", "robot_a") and not ltm.has_robot_memories("k", "robot_b")


def test_debrief_writes_trust_under_the_runs_robot(tmp_path):
    rd = make_ae_run(tmp_path / "r1", robot_id="acme_arm")
    ok = {k: 4 for k in ("likeability", "safety", "competence", "trust")}
    from wardzero_agents.llm.fake import default_policy

    def policy(req):
        return {**ok, "comment": "ok"} if "likeability" in req.schema.get("properties", {}) else default_policy(req)

    asyncio.run(debrief(rd, FakeGateway(policy)))
    rows = [json.loads(x) for x in (rd / "trust.jsonl").read_text().splitlines()]
    assert rows and all(r["robot_id"] == "acme_arm" for r in rows)
    store = TrustStore(memory_path(data_dir()))
    assert store.get("P1", "acme_arm", ROLES["patient"]).runs == 1
    assert store.get("P1", "g1", ROLES["patient"]).runs == 0


# ---- the lock ---------------------------------------------------------------------------------
@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setattr(cli, "ros_available", lambda: True)
    seen = {}

    def fake_supervise(children, **kw):
        seen["during_run"] = active_run(data_dir())
        seen["parent_env"] = os.environ.get("WARDZERO_RUN_LOCK_PARENT")
        return 0

    monkeypatch.setattr(cli, "supervise", fake_supervise)
    monkeypatch.setattr(cli, "_ran_for_real", lambda d: True)
    return seen


def test_the_lock_is_held_through_voice_debrief_and_report(run_env, monkeypatch):
    def post(run_dir, report=True):
        run_env["during_debrief"] = active_run(data_dir())
        # a second run, from a terminal or another Studio, is refused while the first debriefs
        run_env["second_run"] = cli.main(["run", V1, "--no-panel"])
        run_env["reset_refused"] = False
        from wardzero_agents.memory.snapshots import guarded_reset
        from wardzero_agents.memory.runlock import RunActive
        try:
            guarded_reset(data_dir(), ["k"], {"k": []}, HashEmbedder())
        except RunActive:
            run_env["reset_refused"] = True

    monkeypatch.setattr(cli, "_post_run", post)
    assert cli.main(["run", V1, "--no-panel"]) == 0
    assert run_env["during_run"] and run_env["during_debrief"] == run_env["during_run"]
    assert run_env["second_run"] == 2 and run_env["reset_refused"]
    assert active_run(data_dir()) is None and "WARDZERO_RUN_LOCK_PARENT" not in os.environ


def test_the_batch_holds_the_lock_too(run_env, monkeypatch):
    monkeypatch.setattr(cli, "_cmd_batch", lambda args, loaded: run_env.update(batch=active_run(data_dir())) or 0)
    assert cli.main(["run", V1, "--no-panel", "--repeat", "2"]) == 0
    assert run_env["batch"] and active_run(data_dir()) is None


def test_the_agent_node_does_not_take_a_lock_its_parent_holds(monkeypatch):
    from wardzero_agents.ros.agent_node import LOCK_PARENT_ENV, _lock_for
    with RunLock(data_dir(), "cli-run"):
        monkeypatch.setenv(LOCK_PARENT_ENV, str(os.getpid()))
        with _lock_for(data_dir(), "folder"):  # would raise RunActive if it tried to take it
            assert active_run(data_dir()) == "cli-run"
        monkeypatch.delenv(LOCK_PARENT_ENV)
        from wardzero_agents.memory.runlock import RunActive
        with pytest.raises(RunActive):
            _lock_for(data_dir(), "folder").__enter__()
