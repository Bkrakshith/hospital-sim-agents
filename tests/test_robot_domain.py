"""The robot side of a run: on the robot's own domain, with only the robot's own config, a mind in the baseline arm."""
import json
from pathlib import Path

import pytest

from wardzero_agents import cli
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.robots.view import ROBOT_CONFIG, load_robot_view

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "17")
    monkeypatch.setattr(cli, "ros_available", lambda: True)


@pytest.fixture
def launched(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "supervise", lambda children, **kw: calls.append(children) or 0)
    return calls


def opt(argv, flag):
    return argv[argv.index(flag) + 1]


def run_args(*extra):
    return cli.build_parser().parse_args(["run", V1, "--sim", "stub", "--no-panel", *extra])


def robot_children(children):
    return [c for c in children if c.name in ("robot_claude", "reception_bot")]


@pytest.mark.parametrize("bot,name", [("claude", "robot_claude"), ("reception", "reception_bot")])
def test_every_robot_child_is_on_the_robot_domain(launched, bot, name):
    assert cli.cmd_run(run_args("--bot", bot, "--robot-domain", "18")) == 0
    (children,) = launched
    (c,) = robot_children(children)
    assert c.name == name
    assert opt(c.argv, "--domain") == "18" and c.env["ROS_DOMAIN_ID"] == "18"
    assert opt(next(x for x in children if x.name == "stub_sim").argv, "--robot-domain") == "18"
    # the people's side stays on the private domain
    agents = next(x for x in children if x.name == "agents")
    assert "--domain" not in agents.argv and agents.env["ROS_DOMAIN_ID"] == "17"


def test_robot_child_carries_no_persona_data(launched):
    loaded = load_scenario(V1)
    assert cli.cmd_run(run_args("--bot", "claude", "--robot-domain", "18")) == 0
    (children,) = launched
    (c,) = robot_children(children)
    rd = Path(opt(c.argv, "--run-dir"))
    cfg_file = Path(opt(c.argv, "--robot-config"))
    assert cfg_file == rd / ROBOT_CONFIG and "--scenario" not in c.argv
    secrets = set()
    for p in loaded.personas.values():
        secrets |= {p.backstory[:40]} if getattr(p, "backstory", "") else set()
    secrets |= {i.display_name for i in loaded.instances} | set(loaded.personas)  # (ids like P3 are too short to search for)
    blob = " ".join(c.argv) + json.dumps(c.env) + cfg_file.read_text()
    assert secrets and not [s for s in secrets if s and s in blob]
    cfg = json.loads(cfg_file.read_text())
    assert not {"cast", "personas", "storyboard", "instances", "deck"} & set(cfg)
    view = load_robot_view(cfg_file)  # what the robot process loads: site, area, signs, its own settings
    assert view.domain == 18 and view.robot.ns == "robot" and view.site is not None and view.budget_usd > 0
    assert not hasattr(view, "personas") and not hasattr(view, "instances")


def test_no_robot_domain_means_the_robot_stays_on_its_default(launched):
    assert cli.cmd_run(run_args("--bot", "claude")) == 0
    (c,) = robot_children(launched[0])
    assert "--domain" not in c.argv and "ROS_DOMAIN_ID" in c.env  # the inherited one
    assert "--robot-domain" not in next(x for x in launched[0] if x.name == "stub_sim").argv


def test_baseline_arm_has_a_mind(launched, monkeypatch):
    monkeypatch.setattr(cli, "_claude_key_present", lambda: True)
    args = run_args("--bot", "none", "--robot-domain", "18")
    loaded = load_scenario(V1)
    assert cli._start_live(args, loaded, {"robot": {**loaded.scenario.robot.model_dump(), "profile": "g1_claude"}},
                           {}, baseline=True) == 0
    assert [c.name for c in robot_children(launched[0])] == ["robot_claude"]
    assert cli._start_live(args, loaded, {}, {}) == 0  # the developer's own robot: no built-in mind
    assert robot_children(launched[1]) == []


def test_baseline_without_a_key_gets_the_placeholder(launched, monkeypatch):
    monkeypatch.setattr(cli, "_claude_key_present", lambda: False)
    assert cli._start_live(run_args("--bot", "none"), load_scenario(V1), {}, {}, baseline=True) == 0
    assert [c.name for c in robot_children(launched[0])] == ["reception_bot"]


def test_batch_runs_the_baseline_arm_as_baseline(monkeypatch, launched):
    seen = []
    monkeypatch.setattr(cli, "_start_live", lambda *a, **k: seen.append(k) or 3)
    loaded = load_scenario(V1)
    scen = loaded.scenario.model_copy(update={"robot": loaded.scenario.robot.model_copy(update={"profile": "other"})})
    loaded.scenario = scen
    with pytest.raises(cli.CliError):
        cli._cmd_batch(run_args("--repeat", "1", "--baseline"), loaded)
    assert seen[0].get("baseline") is False  # the first arm is the developer's own robot
