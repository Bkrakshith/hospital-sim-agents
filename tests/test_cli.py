"""The hhumans CLI, without ROS: process launching is replaced by a recorder."""
import json
import os
import signal
import sys
import time
from pathlib import Path

import keyring
import keyring.backend
import pytest
import yaml

from wardzero_agents import cli, keys
from wardzero_agents.cli import main
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.memory import data_dir, memory_path
from wardzero_agents.memory.longterm import LongTermMemory
from wardzero_agents.kb.embed import HashEmbedder

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")
# Built by concatenation so this file never holds a literal key-shaped string (pre-commit scan).
SECRET = "sk-" + "ant-api03-abcdefghijk"


class MemBackend(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        self.d = {}

    def set_password(self, service, username, password):
        self.d[(service, username)] = password

    def get_password(self, service, username):
        return self.d.get((service, username))

    def delete_password(self, service, username):
        self.d.pop((service, username), None)


@pytest.fixture
def memkeyring(monkeypatch):
    old = keyring.get_keyring()
    keyring.set_keyring(MemBackend())
    monkeypatch.delenv("WARDZERO_KEY_FILE", raising=False)
    yield
    keyring.set_keyring(old)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Never touch the real data or runs folders, and never start ROS."""
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)
    monkeypatch.setattr(cli, "ros_available", lambda: True)


@pytest.fixture
def launched(monkeypatch):
    """Record what run/rerun/replay would start instead of starting it."""
    calls = []

    def fake(children, **kw):
        calls.append(children)
        return 0

    monkeypatch.setattr(cli, "supervise", fake)
    return calls


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")


async def raise_keymissing(*a, **k):
    raise keys.KeyMissing("No working Claude key. Run `hhumans keys set claude` or `ant auth login`.")


def argv_of(children, name):
    return next(c.argv for c in children if c.name == name)


def opt(argv, flag):
    return argv[argv.index(flag) + 1]


# ---- the brief's tests -------------------------------------------------------------------
def test_isaac_needs_isaac_sim_installed(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ISAAC_ROOT", tmp_path / "no-isaacsim")
    assert main(["run", V1, "--sim", "isaac"]) == 2
    assert "Isaac Sim" in capsys.readouterr().err


def test_run_without_key_refuses(monkeypatch, capsys, memkeyring, launched):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(keys, "check", raise_keymissing)
    assert main(["run", V1]) == 2
    err = capsys.readouterr().err
    assert "hhumans keys set claude" in err and "sk-" not in err
    assert launched == []


def test_keys_status_never_prints_secret(memkeyring, capsys):
    keys.set_key("claude", SECRET)
    main(["keys", "status", "claude"])
    out = capsys.readouterr().out
    assert "keyring" in out and "abcdefghijk" not in out


def test_memory_reset_needs_confirmation(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert main(["memory", "reset", "--all"]) == 1


def test_kb_build(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path))
    assert main(["kb", "build"]) == 0 and (tmp_path / "kb" / "meta.json").exists()


# ---- run ---------------------------------------------------------------------------------
def test_runs_dir_env_and_default(monkeypatch, tmp_path):
    assert cli.runs_dir() == tmp_path / "runs"
    monkeypatch.delenv("WARDZERO_RUNS_DIR")
    monkeypatch.chdir(tmp_path)
    assert cli.runs_dir() == ROOT / "runs"


def test_run_spawns_the_nodes(fake_llm, launched, capsys, tmp_path):
    assert main(["run", V1, "--time-scale", "30", "--lockstep"]) == 0
    out = capsys.readouterr().out
    assert "TEST MODE: no AI provider" in out
    assert "http://127.0.0.1:8765/" in out
    (children,) = launched
    assert [c.name for c in children] == ["stub_sim", "agents", "director", "reception_bot"]
    assert [c.name for c in children if c.ends_run] == ["stub_sim"]
    run_dir = Path(opt(argv_of(children, "stub_sim"), "--run-dir"))
    assert run_dir.parent == tmp_path / "runs" and run_dir.name.endswith("_waiting_area_v1")
    stub = argv_of(children, "stub_sim")
    assert opt(stub, "--time-scale") == "30.0" and "--lockstep" in stub and opt(stub, "--ready-timeout") == "600.0"
    for c in children:
        assert c.argv[:2] == [sys.executable, "-m"]
        if c.name == "reception_bot":  # the robot gets its own config, never the scenario with its people
            assert Path(opt(c.argv, "--robot-config")) == run_dir / "robot_config.json" and "--scenario" not in c.argv
            continue
        scen = Path(opt(c.argv, "--scenario"))
        assert scen.parent == run_dir / "config"  # every other node loads the run's own copy
    assert load_scenario(scen).storyboard.beats
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["mode"] == "live" and meta["scenario_source"] == str(Path(V1).resolve())


def test_run_no_panel_no_bot_no_robot(fake_llm, launched, capsys):
    assert main(["run", V1, "--no-panel", "--no-robot", "--ready-timeout", "30"]) == 0
    assert "http://" not in capsys.readouterr().out
    (children,) = launched
    assert [c.name for c in children] == ["stub_sim", "agents", "director"]
    assert "--no-panel" in argv_of(children, "director")
    assert opt(argv_of(children, "stub_sim"), "--ready-timeout") == "30.0"
    scen = load_scenario(opt(argv_of(children, "stub_sim"), "--scenario"))
    assert scen.scenario.robot.enabled is False


def test_run_panel_port_and_bot_none(fake_llm, launched, capsys):
    assert main(["run", V1, "--panel-port", "9001", "--bot", "none"]) == 0
    assert "http://127.0.0.1:9001/" in capsys.readouterr().out
    (children,) = launched
    assert "reception_bot" not in [c.name for c in children]
    assert opt(argv_of(children, "director"), "--panel-port") == "9001"


def test_run_refuses_bad_time_scale_and_bad_scenario(fake_llm, launched, capsys, tmp_path):
    assert main(["run", V1, "--time-scale", "0"]) == 2
    assert main(["run", str(tmp_path / "nope.yaml")]) == 2
    assert launched == []


def test_run_refuses_while_a_run_is_active(fake_llm, launched, capsys):
    d = data_dir()
    d.mkdir(parents=True)
    (d / "active_run.json").write_text(json.dumps({"run_id": "other", "pid": os.getpid()}))
    assert main(["run", V1]) == 2
    assert "other" in capsys.readouterr().err and launched == []


def test_run_without_ros(fake_llm, launched, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ros_available", lambda: False)
    assert main(["run", V1]) == 2
    assert "env.sh" in capsys.readouterr().err and launched == []


def test_run_returns_the_supervisor_code(fake_llm, monkeypatch):
    monkeypatch.setattr(cli, "supervise", lambda children, **kw: 3)
    assert main(["run", V1, "--no-panel"]) == 3


def test_key_check_failure_never_prints_the_exception(monkeypatch, capsys, memkeyring):
    async def boom(*a, **k):
        raise RuntimeError("connection reset " + SECRET)

    monkeypatch.setattr(keys, "check", boom)
    assert main(["run", V1]) == 2
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "abcdefghijk" not in err


# ---- rerun -------------------------------------------------------------------------------
def make_recorded_run(tmp_path):
    """A finished run folder as `hhumans run` leaves it (config copy + meta)."""
    main(["run", V1, "--no-panel"])
    run_dir = sorted((tmp_path / "runs").iterdir())[-1]
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(run_id=run_dir.name, scenario_path=str(run_dir / "config" / "waiting_area_v1.yaml"),
                snapshot_in=f"run-{run_dir.name}-start", snapshot_out=f"run-{run_dir.name}-end")
    (run_dir / "meta.json").write_text(json.dumps(meta))
    snaps = data_dir() / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    (snaps / f"run-{run_dir.name}-start.sqlite").write_bytes(b"")
    return run_dir


def test_rerun_uses_the_runs_config_and_start_snapshot(fake_llm, launched, tmp_path):
    src = make_recorded_run(tmp_path)
    time.sleep(1.1)  # a different timestamp for the new run folder
    assert main(["rerun", str(src), "--no-panel", "--time-scale", "5"]) == 0
    children = launched[-1]
    new_dir = Path(opt(argv_of(children, "agents"), "--run-dir"))
    assert new_dir != src and new_dir.parent == src.parent
    scen_path = Path(opt(argv_of(children, "agents"), "--scenario"))
    assert scen_path.parent == new_dir / "config"
    loaded = load_scenario(scen_path)
    assert loaded.scenario.memory_snapshot == f"run-{src.name}-start"
    meta = json.loads((new_dir / "meta.json").read_text())
    assert meta["mode"] == "rerun" and meta["rerun_of"] == src.name


def test_rerun_of_an_old_style_copy(fake_llm, launched, tmp_path):
    """Runs recorded before the config copy was made self-contained still rerun."""
    src = make_recorded_run(tmp_path)
    scen = src / "config" / "waiting_area_v1.yaml"
    raw = yaml.safe_load(scen.read_text())
    raw.update(storyboard="../storyboards/waiting_area_day.yaml", deck="../cards/starter.yaml")
    scen.write_text(yaml.safe_dump(raw, sort_keys=False))
    time.sleep(1.1)
    assert main(["rerun", str(src), "--no-panel"]) == 0
    loaded = load_scenario(opt(argv_of(launched[-1], "agents"), "--scenario"))
    assert loaded.storyboard.beats


def test_rerun_needs_a_start_snapshot(fake_llm, launched, tmp_path, capsys):
    src = make_recorded_run(tmp_path)
    meta = json.loads((src / "meta.json").read_text())
    meta.pop("snapshot_in")
    (src / "meta.json").write_text(json.dumps(meta))
    n = len(launched)
    assert main(["rerun", str(src)]) == 2
    assert "snapshot" in capsys.readouterr().err and len(launched) == n


def test_rerun_needs_the_snapshot_file(fake_llm, launched, tmp_path, capsys):
    src = make_recorded_run(tmp_path)
    (data_dir() / "snapshots" / f"run-{src.name}-start.sqlite").unlink()
    n = len(launched)
    assert main(["rerun", str(src)]) == 2
    assert f"run-{src.name}-start" in capsys.readouterr().err and len(launched) == n


# ---- replay ------------------------------------------------------------------------------
def test_replay_intents(fake_llm, launched, tmp_path):
    src = make_recorded_run(tmp_path)
    launched.clear()
    time.sleep(1.1)
    assert main(["replay", str(src), "--intents", "--time-scale", "4"]) == 0
    (children,) = launched
    assert [c.name for c in children] == ["stub_sim", "replay", "reception_bot"]
    rep = argv_of(children, "replay")
    assert opt(rep, "--run-dir") == str(src) and "--intents" in rep
    stub = argv_of(children, "stub_sim")
    assert opt(stub, "--wait-for") == "agents" and opt(stub, "--time-scale") == "4.0"
    assert Path(opt(stub, "--run-dir")) != src


def test_replay_exact(launched, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    src = make_recorded_run(tmp_path)
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM")
    launched.clear()
    assert main(["replay", str(src), "--exact", "--time-scale", "2"]) == 0  # needs no key
    (children,) = launched
    assert [c.name for c in children] == ["replay"]
    rep = children[0].argv
    assert "--exact" in rep and opt(rep, "--time-scale") == "2.0" and "--no-panel" not in rep
    assert "http://127.0.0.1:8765/" in capsys.readouterr().out


def test_replay_missing_run(launched, tmp_path, capsys):
    assert main(["replay", str(tmp_path / "nope"), "--exact"]) == 2


# ---- supervise (real child processes, no ROS) --------------------------------------------
def py(code):
    return [sys.executable, "-c", code]


SLEEPER = "import time\nwhile True: time.sleep(0.05)"
STUBBORN = ("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(0.05)")


def test_supervise_stops_the_rest_when_one_ends(tmp_path):
    marker = tmp_path / "got_sigint"
    waiter = ("import signal, sys, time, pathlib\n"
              f"def h(*a): pathlib.Path({str(marker)!r}).write_text('x'); sys.exit(0)\n"
              "signal.signal(signal.SIGINT, h)\nwhile True: time.sleep(0.05)")
    t0 = time.monotonic()
    code = cli.supervise([cli.Child("ender", py("import time; time.sleep(0.3)"), ends_run=True),
                          cli.Child("waiter", py(waiter))], term_after_s=5, kill_after_s=5)
    assert code == 0 and marker.exists() and time.monotonic() - t0 < 5


def test_supervise_returns_the_failing_childs_code():
    code = cli.supervise([cli.Child("bad", py("import sys, time; time.sleep(0.3); sys.exit(4)")),
                          cli.Child("ok", py(SLEEPER))], term_after_s=5, kill_after_s=5)
    assert code == 4


def test_supervise_escalates_to_sigkill_and_reports_it(capsys):
    """A clean end still fails when a child had to be killed: its snapshot may be unsaved."""
    t0 = time.monotonic()
    code = cli.supervise([cli.Child("ender", py("import time; time.sleep(0.2)"), ends_run=True),
                          cli.Child("stubborn", py(STUBBORN), kill_after_s=0.5)],
                         term_after_s=0.5, kill_after_s=0.5)
    assert code == 1 and time.monotonic() - t0 < 5
    err = capsys.readouterr().err
    assert "stubborn" in err and "killed" in err


def test_supervise_clean_end_but_a_child_fails_while_stopping(capsys):
    failer = ("import signal, sys, time\n"
              "signal.signal(signal.SIGINT, lambda *a: sys.exit(5))\nwhile True: time.sleep(0.05)")
    code = cli.supervise([cli.Child("ender", py("import time; time.sleep(0.3)"), ends_run=True),
                          cli.Child("agents", py(failer))], term_after_s=5, kill_after_s=5)
    assert code == 1
    err = capsys.readouterr().err
    assert "agents" in err and "code 5" in err


def test_supervise_keeps_the_first_failure_code(capsys):
    failer = ("import signal, sys, time\n"
              "signal.signal(signal.SIGINT, lambda *a: sys.exit(5))\nwhile True: time.sleep(0.05)")
    code = cli.supervise([cli.Child("ender", py("import sys, time; time.sleep(0.3); sys.exit(3)"), ends_run=True),
                          cli.Child("agents", py(failer))], term_after_s=5, kill_after_s=5)
    assert code == 3


def test_supervise_early_clean_exit_of_another_child_fails_the_run(capsys):
    code = cli.supervise([cli.Child("stub_sim", py(SLEEPER), ends_run=True),
                          cli.Child("director", py("import time; time.sleep(0.2)"))],
                         term_after_s=5, kill_after_s=5, early_exit_grace_s=0.5)
    assert code == 1
    err = capsys.readouterr().err
    assert "director" in err and "did not finish" in err


def test_supervise_others_may_stop_just_before_the_sim():
    """At a normal end the agents and director can exit on the stop message before the stub sim."""
    code = cli.supervise([cli.Child("stub_sim", py("import time; time.sleep(0.6)"), ends_run=True),
                          cli.Child("director", py("import time; time.sleep(0.2)"))],
                         term_after_s=5, kill_after_s=5, early_exit_grace_s=3)
    assert code == 0


def test_supervise_children_get_their_own_process_group():
    seen = {}

    def on_start(procs):
        seen.update({name: os.getpgid(p.pid) for name, p in procs.items()})

    cli.supervise([cli.Child("a", py("import time; time.sleep(0.2)"))], term_after_s=1, kill_after_s=1,
                  on_start=on_start)
    assert seen["a"] != os.getpgrp()


def test_supervise_ctrl_c_returns_130():
    def on_start(procs):
        os.kill(os.getpid(), signal.SIGINT)

    t0 = time.monotonic()
    code = cli.supervise([cli.Child("a", py(SLEEPER))], term_after_s=5, kill_after_s=5, on_start=on_start)
    assert code == 130 and time.monotonic() - t0 < 5


# ---- memory ------------------------------------------------------------------------------
def seeded_memory():
    loaded = load_scenario(V1)
    ltm = LongTermMemory(memory_path(data_dir()), HashEmbedder())
    for inst in loaded.instances:
        ltm.ensure_seed(inst.memory_key, loaded.personas[inst.persona_id].seed_memories)
    ltm.begin_run("r1")
    for inst in loaded.instances:
        ltm.add(inst.memory_key, "r1", 1.0, "episode", "the robot was rude", 0.9)
    return loaded, ltm


def episodes(ltm, key):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(ltm.path)) as c:
        return c.execute("SELECT COUNT(*) FROM memories WHERE person_key=? AND kind='episode'", (key,)).fetchone()[0]


def test_memory_reset_person(monkeypatch, capsys):
    loaded, ltm = seeded_memory()
    p1 = next(i for i in loaded.instances if i.person_id == "P1")
    p2 = next(i for i in loaded.instances if i.person_id == "P2")
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert main(["memory", "reset", "--person", "P1", "--scenario", V1]) == 0
    assert p1.memory_key in capsys.readouterr().out
    assert episodes(ltm, p1.memory_key) == 0 and episodes(ltm, p2.memory_key) == 1
    assert ltm.search(p1.memory_key, "x", 10, run_index=1)  # seeds are back


def test_memory_reset_persona_with_yes(monkeypatch):
    loaded, ltm = seeded_memory()
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked despite --yes"))
    assert main(["memory", "reset", "--persona", "companion", "--scenario", V1, "--yes"]) == 0
    for inst in loaded.instances:
        assert episodes(ltm, inst.memory_key) == (0 if inst.persona_id == "companion" else 1)


def test_memory_reset_all(monkeypatch):
    loaded, ltm = seeded_memory()
    assert main(["memory", "reset", "--all", "--yes"]) == 0
    assert all(episodes(ltm, i.memory_key) == 0 for i in loaded.instances)
    assert ltm.search(loaded.instances[0].memory_key, "x", 10, run_index=1)  # reseeded from personas


def test_memory_reset_unknown_person(capsys):
    assert main(["memory", "reset", "--person", "P99", "--scenario", V1, "--yes"]) == 2


def test_memory_reset_refused_during_a_run(capsys):
    loaded, ltm = seeded_memory()
    (data_dir() / "active_run.json").write_text(json.dumps({"run_id": "live1", "pid": os.getpid()}))
    assert main(["memory", "reset", "--all", "--yes"]) == 2
    assert "live1" in capsys.readouterr().err
    assert episodes(ltm, loaded.instances[0].memory_key) == 1


def test_memory_reset_checks_the_lock_before_asking(monkeypatch, capsys):
    seeded_memory()
    (data_dir() / "active_run.json").write_text(json.dumps({"run_id": "live1", "pid": os.getpid()}))
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked during an active run"))
    assert main(["memory", "reset", "--all"]) == 2


def test_memory_snapshot_save_list_restore(capsys):
    loaded, ltm = seeded_memory()
    assert main(["memory", "snapshot", "save", "before"]) == 0
    out = capsys.readouterr().out
    assert str(data_dir() / "snapshots" / "before.sqlite") in out
    ltm.reset([loaded.instances[0].memory_key], {})
    assert main(["memory", "snapshot", "list"]) == 0
    assert "before" in capsys.readouterr().out
    assert main(["memory", "snapshot", "restore", "before"]) == 0
    assert episodes(ltm, loaded.instances[0].memory_key) == 1


def test_memory_snapshot_save_refused_during_a_run(capsys):
    seeded_memory()
    (data_dir() / "active_run.json").write_text(json.dumps({"run_id": "live1", "pid": os.getpid()}))
    assert main(["memory", "snapshot", "save", "x"]) == 2
    assert main(["memory", "snapshot", "restore", "x"]) == 2
    assert not (data_dir() / "snapshots" / "x.sqlite").exists()


def test_memory_snapshot_bad_name_and_missing(capsys):
    seeded_memory()
    assert main(["memory", "snapshot", "save", "../evil"]) == 2
    assert main(["memory", "snapshot", "restore", "nope"]) == 2


# ---- kb ----------------------------------------------------------------------------------
def test_kb_build_fastembed_missing(monkeypatch, capsys):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "fastembed" else real(name, *a))
    assert main(["kb", "build", "--embedder", "fastembed"]) == 1
    assert "pip install" in capsys.readouterr().err


# ---- keys --------------------------------------------------------------------------------
def test_keys_set_uses_getpass_and_never_echoes(memkeyring, monkeypatch, capsys):
    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": SECRET)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("input() would echo the key"))
    assert main(["keys", "set", "claude"]) == 0
    assert keys.get_key("claude") == SECRET
    cap = capsys.readouterr()
    assert "abcdefghijk" not in cap.out + cap.err


def test_keys_set_empty_refused(memkeyring, monkeypatch):
    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "  ")
    assert main(["keys", "set", "openai"]) == 1 and keys.get_key("openai") is None


def test_keys_clear_and_status_labels(memkeyring, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_claude_ambient", lambda: False)
    keys.set_key("claude", SECRET)
    assert main(["keys", "clear", "claude"]) == 0 and keys.get_key("claude") is None
    capsys.readouterr()
    main(["keys", "status", "claude"])
    assert capsys.readouterr().out.strip().endswith("not set")
    f = tmp_path / "k.json"
    f.write_text(json.dumps({"openai": "oa-" + "secretvalue"}))
    f.chmod(0o600)
    monkeypatch.setenv("WARDZERO_KEY_FILE", str(f))
    main(["keys", "status", "openai"])
    out = capsys.readouterr().out
    assert "from key file" in out and "secretvalue" not in out
    monkeypatch.delenv("WARDZERO_KEY_FILE")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-" + "envvalue")
    main(["keys", "status", "openai"])
    out = capsys.readouterr().out
    assert "from ant profile or environment" in out and "envvalue" not in out


def test_entry_point_needs_a_command(capsys):
    assert main([]) == 2


def test_keys_status_claude_unverifiable(memkeyring, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_claude_ambient", lambda: None)
    main(["keys", "status", "claude"])
    assert "ant profile or ANTHROPIC_API_KEY (not verified)" in capsys.readouterr().out


def test_claude_ambient_profile_detection(monkeypatch):
    import anthropic
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    class WithProfile:
        credentials = object()

    class Without:
        credentials = None

    class Broken:
        def __init__(self):
            raise RuntimeError("no idea")

    for cls, want in ((WithProfile, True), (Without, False), (Broken, None)):
        monkeypatch.setattr(anthropic, "Anthropic", cls)
        assert cli._claude_ambient() is want
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert cli._claude_ambient() is True


# ---- final review fixes ------------------------------------------------------------------
KEY_VARS = {"ANTHROPIC_API_KEY": "sk-" + "ant-api03-zzzzzzzzzzzz", "OPENAI_API_KEY": "sk-" + "proj-yyyyyyyyyyyyyyyy",
            "ANTHROPIC_AUTH_TOKEN": "tok-" + "xxxxxxxx", "WARDZERO_KEY_FILE": "/tmp/keys.json"}


def test_child_env_strips_keys_except_for_the_agents(monkeypatch):
    for k, v in KEY_VARS.items():
        monkeypatch.setenv(k, v)
    plain = cli._child_env()
    assert not set(KEY_VARS) & set(plain) and plain["PYTHONUNBUFFERED"] == "1"
    agents = cli._child_env(with_keys=True)
    assert all(agents[k] == v for k, v in KEY_VARS.items())


def test_run_gives_the_key_only_to_the_agent_node(fake_llm, monkeypatch):
    for k, v in KEY_VARS.items():
        monkeypatch.setenv(k, v)
    seen = {}

    def fake(children, **kw):
        seen["children"], seen["env"] = children, kw.get("env")
        return 0
    monkeypatch.setattr(cli, "supervise", fake)
    assert main(["run", V1, "--no-panel"]) == 0
    default = seen["env"]
    assert default is not None and not set(KEY_VARS) & set(default)
    for c in seen["children"]:
        env = c.env if c.env is not None else default
        has = set(KEY_VARS) <= set(env)
        assert has == (c.name == "agents"), c.name


def test_supervise_uses_each_childs_own_env(tmp_path):
    out = tmp_path / "env"
    code = "import os, sys; open(sys.argv[1], 'a').write(sys.argv[2] + '=' + os.environ.get('WZ_X', '-') + '\\n')"
    kids = [cli.Child("a", py(code) + [str(out), "a"], env=dict(os.environ, WZ_X="own")),
            cli.Child("b", py(code) + [str(out), "b"], ends_run=True)]
    assert cli.supervise(kids, env=dict(os.environ, WZ_X="shared"), early_exit_grace_s=5) == 0
    assert sorted(out.read_text().split()) == ["a=own", "b=shared"]


def test_replay_intents_records_into_the_new_run_folder(fake_llm, launched, tmp_path):
    src = make_recorded_run(tmp_path)
    launched.clear()
    time.sleep(1.1)
    assert main(["replay", str(src), "--intents"]) == 0
    (children,) = launched
    rec = Path(opt(argv_of(children, "replay"), "--record-dir"))
    assert rec == Path(opt(argv_of(children, "stub_sim"), "--run-dir")) and rec != src


@pytest.mark.parametrize("name", ["run-20261103-start", "run-x-end", "pre-restore-abc"])
def test_snapshot_save_refuses_reserved_names(name, capsys):
    seeded_memory()
    assert main(["memory", "snapshot", "save", name]) == 2
    assert "reserved" in capsys.readouterr().err
    assert not (data_dir() / "snapshots" / f"{name}.sqlite").exists()


def test_snapshot_restore_keeps_the_memory_it_replaces(capsys):
    import sqlite3
    from contextlib import closing
    loaded, ltm = seeded_memory()
    assert main(["memory", "snapshot", "save", "before"]) == 0
    ltm.add(loaded.instances[0].memory_key, "r1", 2.0, "episode", "learned later", 0.9)
    capsys.readouterr()
    assert main(["memory", "snapshot", "restore", "before"]) == 0
    out = capsys.readouterr().out
    (kept,) = list((data_dir() / "snapshots").glob("pre-restore-manual-*.sqlite"))
    assert kept.stem in out
    with closing(sqlite3.connect(kept)) as c:
        assert c.execute("SELECT COUNT(*) FROM memories WHERE text='learned later'").fetchone()[0] == 1
    assert episodes(ltm, loaded.instances[0].memory_key) == 1  # the restore itself happened


def test_snapshot_restore_of_a_missing_snapshot_saves_nothing(capsys):
    seeded_memory()
    assert main(["memory", "snapshot", "restore", "nope"]) == 2
    assert not list((data_dir() / "snapshots").glob("pre-restore-*"))
