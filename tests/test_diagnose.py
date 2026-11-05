"""Failed runs explain themselves: every rule in the catalogue is fed the message the code (or Isaac) really prints,
the cause is found under the stop noise, the Studio shows it with the fix, and an Isaac run that cannot work is refused
before it starts."""
import asyncio
import dataclasses
import sys
import textwrap
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents import cli, diagnose as dg, doctor, isaac_assets, keys, preflight
from wardzero_agents.diagnose import diagnose, format_problems
from wardzero_agents.isaac_host.objects_isaac import FLOW_MIN_FREE_MIB, UsdObjectStage, flow_fits
from wardzero_agents.isaac_host.recording import CEILING_CUT_M, start_view
from wardzero_agents.robots.profile import G1_CLAUDE


def RobotProfile(name, urdf):  # a saved robot like the Studio makes, with this URDF
    return dataclasses.replace(G1_CLAUDE, name=name, urdf=urdf, controller="external")
from wardzero_agents.studio.runner import Runner
from wardzero_agents.studio.server import Ctx, make_app

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}


def keys_of(lines, code=1, **kw):
    return [p.key for p in diagnose(lines, code, **kw)]


# Each rule, with a message as the code (or Isaac, or the OS) prints it. Built from the code's own strings where the
# code makes them, so a reworded message breaks this test instead of silently escaping the catalogue.
def samples(tmp_path, monkeypatch):
    isaac = tmp_path / "isaac"
    monkeypatch.setenv(isaac_assets.STAGE_ENV, str(tmp_path / "no_stage.usd"))
    monkeypatch.setenv(isaac_assets.COLLISION_ENV, str(tmp_path / "no_collision.usd"))
    missing = isaac_assets.missing_for_isaac(isaac)  # no python.sh, no stage, no collision USD there
    return {
        "robot_no_urdf": [f"Robot error: robot 'my_robot' has no URDF to convert"],
        "robot_error": ["Robot error: the URDF importer failed on base_link.STL"],
        "robot_config": ["Robot config error: radius must be above 0"],
        "scenario": ["Scenario error: cast[0].persona: unknown persona 'x'"],
        "stage_missing": ["Stage not found: /nowhere/hospital_g1.usd"],
        "collision_missing": [next(m for m in missing if "collision" in m)],
        "isaac_not_found": [next(m for m in missing if "python.sh" in m)],
        "key_missing": [keys._FIX["claude"]],
        "model_unavailable": ["Model claude-x is not available to this claude account."],
        "key_check_failed": ["The claude key check failed (ConnectError). Check the network and try again."],
        "provider": ["2026 robot_claude ERROR The AI provider is rejecting the robot's requests (auth). The robot "
                     "stops thinking for this run; check the key and model."],
        "keyring": ["The OS keyring is not available (NoKeyringError). Put the key in a mode-600 file"],
        "brotli": ["AttributeError: module 'brotli' has no attribute 'Decompressor' (Brotli 1.0.9)"],
        "glibcxx": ["ImportError: /lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found"],
        "ros_missing": ["ROS 2 is not sourced. Run `source $HG1_ROOT/env.sh` first."],
        "module_missing": ["ModuleNotFoundError: No module named 'aiohttp'"],
        "flow_crash": ["[Warning] [carb.crashreporter-breakpad.plugin] [crash]   'lastCommand' = "
                       "'FlowCreatePrim(prim_path=/World/WZ/Fires/fire_1/emitter,type_name=FlowEmitterSphere)'"],
        "isaac_crash_cmd": ["[crash]   'lastCommand' = 'ChangeProperty(prop_path=/World/x)'"],
        "isaac_crash": ["/home/u/isaacsim/python.sh: line 72: 4162132 Segmentation fault      (core dumped)"],
        "gpu_oom": ["[Error] [gpu.foundation.plugin] VkResult: ERROR_OUT_OF_DEVICE_MEMORY"],
        "no_gpu": ["[Error] [carb.graphics-vulkan.plugin] Failed to create any GPU devices"],
        "display": ["[Error] [carb.windowing-glfw.plugin] Failed to create the GLFW window"],
        "ram_killed": ["isaac_host exited with code -9; stopping the run"],
        "port_in_use": ["Port 8770 is already in use (is another Studio running?). Close it, or start this one"],
        "run_active": ["run waiting-101010 is active; stop it first"],
        "isaac_busy": [preflight.isaac_busy_problem([4242])],
        "gave_up_waiting": ["2026 wardzero.isaac.host ERROR gave up waiting for agents, director after 600 s"],
        "agents_cannot_start": ["Cannot start the agents: snapshot pre-x is missing"],
        "stopped_on_its_own": ["director stopped on its own before the scenario finished, so the scenario did not"],
        "could_not_start": ["could not start isaac_host: [Errno 13] Permission denied"],
        "no_voices": ["No voices (FileNotFoundError: no model). Run `hhumans voice /r` to try again."],
        "no_report": ["The debrief or report failed (APIError: x). Run `hhumans report /r`."],
    }


def test_every_rule_has_a_sample_and_finds_it(tmp_path, monkeypatch):
    s = samples(tmp_path, monkeypatch)
    assert set(s) == {r.key for r in dg.RULES}  # a new rule needs a sample here
    for key, lines in s.items():
        found = diagnose(lines, 1, read_kit_log=False)
        assert found and found[0].key == key, (key, [p.key for p in found])
        assert found[0].title and found[0].fix, key  # every problem says what to do


def test_every_provider_error_kind_has_its_own_fix():
    for kind in dg.PROVIDER_FIX:
        p = diagnose([f"AI provider rejecting requests: {kind} (400). Check the key and model; see the README."], 1)
        assert p[0].key == "provider" and p[0].title == dg.PROVIDER_FIX[kind][0]
    credit = diagnose(["AI provider rejected a request: bad_request (400)"], 1)[0]
    assert "credit" in credit.title and "console.anthropic.com" in credit.fix[0]


# The user's first failure: the cause, then Isaac's python.sh line, then the people's KeyboardInterrupt from the stop.
FIRST = """Run folder: /r
Scenario evacuation_guide: 8 min of sim time at x1, ROS_DOMAIN_ID=0, robot: none, sim: isaac (headed)
Robot error: robot 'my_robot' has no URDF to convert
There was an error running python
isaac_host exited with code 1; stopping the run
Stopping the run (orderly stop).
Traceback (most recent call last):
  File "/x/agent_node.py", line 365, in _key_check
    import anthropic
  File "<frozen importlib._bootstrap_external>", line 87, in _unpack_uint32
KeyboardInterrupt
Run finished (exit code 1). Run folder: /r""".splitlines()


def test_the_cause_is_found_under_the_stop_noise():
    found = diagnose(FIRST, 1)
    assert [p.key for p in found] == ["robot_no_urdf"]
    assert "my_robot" in found[0].title and "Robot page" in found[0].fix[0]


def test_tracebacks_a_library_logs_at_debug_are_not_failures():
    lines = ["2026 rosidl_generator_py DEBUG Failed to import modules for generating Python structures:",
             "Traceback (most recent call last):", "  File \"x\", line 1", "    import em",
             "ModuleNotFoundError: No module named 'em'",
             "[11.534s] Could not import system rclpy: No module named 'rclpy'",  # Isaac then loads its own
             "isaac_host exited with code 1; stopping the run"]
    assert keys_of(lines) == ["unknown"]
    assert keys_of(["Traceback (most recent call last):", "ModuleNotFoundError: No module named 'aiohttp'"]) == \
        ["module_missing"]  # a real one still counts


def test_a_fire_crash_hides_the_general_crash():
    lines = ["[crash]   'lastCommand' = 'FlowCreatePrim(prim_path=/World/WZ/Fires/fire_1/emitter)'",
             "python.sh: line 72: 4162132 Segmentation fault      (core dumped)"]
    assert keys_of(lines) == ["flow_crash"]


def test_knock_on_effects_come_after_the_cause():
    lines = ["gave up waiting for agents after 600 s", keys._FIX["claude"]]
    assert keys_of(lines) == ["key_missing", "gave_up_waiting"]


def test_nothing_recognised_says_where_to_look(tmp_path):
    kit = tmp_path / "kit.log"
    kit.write_text("[Info] all fine\n")
    log = tmp_path / "last_run.log"
    found = diagnose([f"[Info] [carb] Logging to file: {kit}", "isaac_host exited with code 1; stopping the run"],
                     1, log)
    assert [p.key for p in found] == ["unknown"]
    assert "isaac_host" in found[0].title and str(log) in found[0].fix[0] and str(kit) in found[0].fix[1]
    assert "hhumans doctor" in found[0].fix[-1]


def test_isaacs_own_log_is_read_for_gpu_trouble_only_when_the_run_failed(tmp_path):
    kit = tmp_path / "kit.log"
    kit.write_text("[Info] start\n[Error] [rtx] VkResult: ERROR_OUT_OF_DEVICE_MEMORY\n")
    lines = [f"2026 [Info] [carb] Logging to file: {kit}", "isaac_host exited with code 1; stopping the run"]
    assert keys_of(lines, 1) == ["gpu_oom"]
    assert keys_of(lines, 0) == [] and keys_of(lines, 130) == []  # a good run or a user stop: nothing


def test_format_numbers_the_steps():
    text = format_problems(diagnose(FIRST, 1))
    assert text.startswith("The robot 'my_robot' has no URDF") and "\n  1. On the Robot page" in text


# ---- the Studio shows the problems ------------------------------------------------------------------------
NO_URDF_RUN = textwrap.dedent('''
    import sys
    print("Run folder: /nowhere", flush=True)
    print("Robot error: robot 'my_robot' has no URDF to convert", file=sys.stderr, flush=True)
    print("isaac_host exited with code 1; stopping the run", file=sys.stderr, flush=True)
    print("KeyboardInterrupt", flush=True)
    sys.exit(1)
''')


@pytest.fixture
async def studio(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(data))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(runs))
    script = tmp_path / "fake_cli.py"
    script.write_text(NO_URDF_RUN)
    runner = Runner(data, ROOT, runs, command=lambda p, s, b, d: [sys.executable, str(script)],
                    stop_publisher=lambda d: False, key_present=lambda: False)
    problems = {"list": []}
    c = TestClient(TestServer(make_app(Ctx(data, ROOT, runs, runner, isaac_installed=lambda: True,
                                           isaac_start_problems=lambda profile: problems["list"]))))
    await c.start_server()
    c.problems = problems
    yield c
    await asyncio.to_thread(runner.shutdown)
    await c.close()


async def _project(c, **fields):
    r = await c.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting", **fields}, headers=H)
    assert r.status == 201
    return await r.json()


async def _done(c):
    for _ in range(150):
        s = await (await c.get("/api/run/state")).json()
        if s["phase"] == "done":
            return s
        await asyncio.sleep(0.1)
    raise AssertionError("the run did not finish")


async def test_the_run_page_gets_the_problem_and_its_fix(studio):
    p = await _project(studio)
    assert (await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202
    s = await _done(studio)
    assert s["exit_code"] == 1
    assert [x["title"] for x in s["problems"]] == ["The robot 'my_robot' has no URDF, so Isaac Sim cannot build it."]
    assert s["problems"][0]["fix"][0].startswith("On the Robot page")
    assert "no URDF" in s["error"] and "KeyboardInterrupt" not in s["error"]
    assert s["log_path"].endswith("last_run.log")


async def test_an_isaac_start_that_cannot_work_is_refused_with_the_fix(studio):
    p = await _project(studio, sim="isaac")
    studio.problems["list"] = ["The robot 'my_robot' has no URDF, so Isaac Sim cannot build it. " + preflight.FIX_ROBOT]
    r = await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 400 and "Robot page" in (await r.json())["error"]
    studio.problems["list"] = []
    assert (await studio.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202


# ---- the checks before a start ------------------------------------------------------------------------------
def test_robot_problem(tmp_path):
    assert "has no URDF" in preflight.robot_problem(RobotProfile(name="my_robot", urdf=None))
    gone = tmp_path / "gone.urdf"
    assert str(gone) in preflight.robot_problem(RobotProfile(name="arm", urdf=str(gone)))
    gone.write_text("<robot name='arm'/>")
    assert preflight.robot_problem(RobotProfile(name="arm", urdf=str(gone))) is None


def test_another_isaac_is_named_with_its_process():
    assert preflight.isaac_busy_problem([]) is None
    assert "kill 4242" in preflight.isaac_busy_problem([4242])


def test_the_cli_refuses_an_isaac_run_that_cannot_work(monkeypatch):
    class R:
        enabled, profile = True, "my_robot"

    class S:
        robot = R()

    class L:
        scenario = S()

    class A:
        sim, no_robot = "isaac", False

    monkeypatch.setattr("wardzero_agents.robots.profile.get_profile",
                        lambda name: RobotProfile(name=name, urdf=None))
    monkeypatch.setattr(preflight, "isaac_pids", lambda: [])
    with pytest.raises(cli.CliError, match="has no URDF"):
        cli._check_isaac_start(A(), L())
    A.sim = "stub"
    cli._check_isaac_start(A(), L())  # Quick 2D needs no URDF


# ---- hhumans doctor ---------------------------------------------------------------------------------------
def test_doctor_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/home/u/miniconda3/lib")
    assert [c.status for c in doctor.check_ros(lambda: False)] == [doctor.FIX, doctor.FIX]
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)
    assert doctor.check_key(lambda p: "not set", "not set")[0].status == doctor.FIX
    assert doctor.check_key(lambda p: "stored in keyring", "not set")[0].status == doctor.OK
    isaac = tmp_path / "isaac"
    assert doctor.check_isaac(isaac, lambda r: [], lambda: None)[0].status == doctor.NOTE  # not installed: 2D only
    isaac.mkdir()
    (isaac / "python.sh").write_text("")
    low = doctor.check_isaac(isaac, lambda r: [], lambda: None, gpu=lambda: (1500, 8192))
    assert any(c.status == doctor.FIX and "1500 MiB" in c.what for c in low)
    busy = doctor.check_isaac(isaac, lambda r: [], lambda: "another Isaac Sim is running (process 7)",
                              gpu=lambda: (7000, 8192))
    assert any(c.status == doctor.FIX and c.what.startswith("Another Isaac") for c in busy)
    assert doctor.check_voices(tmp_path / "none")[0].status == doctor.NOTE
    text = doctor.format_checks(low)
    assert "FIX" in text and "to fix before a run" in text


def test_hhumans_diagnose_reads_a_log(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("\n".join(FIRST) + "\n")
    assert cli.main(["diagnose", str(log)]) == 0
    out = capsys.readouterr().out
    assert "has no URDF" in out and "1. On the Robot page" in out
    assert cli.main(["diagnose", str(tmp_path / "missing.log")]) != 0


# ---- the fire falls back instead of crashing Isaac, and the window starts where the scene is -------------
class FakeStage:
    def DefinePrim(self, path, kind=""):
        return None


def test_the_fire_is_the_stand_in_when_graphics_memory_is_short(monkeypatch):
    assert flow_fits(None) and flow_fits(FLOW_MIN_FREE_MIB) and not flow_fits(FLOW_MIN_FREE_MIB - 1)
    drawn = []
    s = UsdObjectStage(FakeStage(), "flow", free_vram=lambda: 900)
    monkeypatch.setattr(s, "_init_flow", lambda: pytest.fail("Flow must not be authored without room"))
    monkeypatch.setattr(s, "_proxy", lambda key, x, y, params: drawn.append(key))
    s.set_fire("fire_1", 1.0, 2.0, None)
    assert s.fire == "proxy" and drawn == ["fire_1"]


def test_the_window_starts_at_the_robots_post_below_the_ceiling():
    eye, target, focal = start_view((2.5, -0.5, 3.14159))
    assert target[:2] == (2.5, -0.5)
    assert eye[0] < 2.5 - 4.0  # in front of a robot facing -x, where the people come to it
    assert eye[2] <= CEILING_CUT_M + 0.2  # under the ceiling, not above the roof
    assert start_view(None) is None
