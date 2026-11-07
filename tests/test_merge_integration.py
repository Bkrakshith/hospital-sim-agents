"""Pieces that join two fix-round workstreams: the robot's own microphone and its tell/press_call hearing checks lose
sound through walls the way people's ears do, and the Isaac LeggedBody counts a contact only when the robot's own
motion closed the gap (contract item 6)."""
import tempfile
from pathlib import Path

import pytest

from wardzero_agents.config.presets import DEFAULT_CONFIG_ROOT, load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Say
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world.legged_body import LeggedBody


@pytest.fixture(scope="module")
def ae():
    p = load_scenario_presets(DEFAULT_CONFIG_ROOT)["ae_waiting"]
    return preset_to_scenario(p, DEFAULT_CONFIG_ROOT, Path(tempfile.mkdtemp()),
                              characters=auto_staff.with_auto(list(p.characters), p))


def test_the_robots_microphone_loses_sound_through_walls(ae, tmp_path):
    s = StubSimService(ae, RunLog(tmp_path), wait_for=())
    w = s.world
    s.step(0.01)
    n1, p1 = w.people["N1"], w.people["P1"]
    assert w.walls_between((p1.x, p1.y), (n1.x, n1.y)) >= 1
    w.robot.x, w.robot.y = n1.x + 0.5, n1.y  # the robot stands in Triage, behind walls from the waiting room
    s.on_say(Say(person="P1", t=s.t, text="Excuse me, how long is the wait?", addressee=None))
    s.step(0.01)
    said = [ev for _, ev, _ in s._recent if ev.source_id == "P1" and ev.text]
    assert said and w.robot_heard(said[-1]) is None
    w.robot.x, w.robot.y = p1.x + 1.0, p1.y  # beside her, it hears her
    assert w.robot_heard(said[-1]) is not None


def test_a_person_walking_into_a_standing_legged_robot_is_not_its_contact():
    body = LeggedBody(0.0, 0.0, 0.0)
    assert body.step(0.1, None, [("P1", 3.0, 0.0)]) == []
    assert body.step(0.1, None, [("P1", 0.4, 0.0)]) == []  # she walked up; the robot stood still
    body.x = 0.0
    assert body.step(0.1, None, [("P1", 0.4, 0.0)]) == []


def test_a_legged_robot_driving_into_someone_is_its_contact():
    body = LeggedBody(0.0, 0.0, 0.0)
    assert body.step(0.1, None, [("P1", 1.0, 0.0)]) == []
    body.x = 0.5  # Isaac moved the body towards her
    assert body.step(0.1, None, [("P1", 1.0, 0.0)]) == ["P1"]
    assert body.step(0.1, None, [("P1", 1.0, 0.0)]) == []  # once per touch


def test_call_system_chimes_reach_the_staff_through_walls(tmp_path):
    """The robot's staff call sounds through the ward's call system (no wall loss): on night_call_bell the night nurse
    at the far end of the ward, three walls from the nurse base, still gets it."""
    from wardzero_agents.contracts import WorldEvent
    from wardzero_agents.world import robot_tasks as rt
    p = load_scenario_presets(DEFAULT_CONFIG_ROOT)["night_call_bell"]
    ld = preset_to_scenario(p, DEFAULT_CONFIG_ROOT, tmp_path / "s")
    s = StubSimService(ld, RunLog(tmp_path / "run"), wait_for=())
    w = s.world
    n1 = w.people["N1"]
    assert w.walls_between(s.tasks.call_point(), (n1.x, n1.y)) >= 1
    s.on_world_event(WorldEvent(id="rb", t=0.0, kind="ring_bell", person="P1"))
    s.on_task_action({"id": "pc", "action": "press_call", "track": w.tracks.track("P1")})
    outs = [s.step(0.1) for _ in range(3)]
    res = [r for o in outs for r in o.task_results if r.get("id") == "pc"]
    assert res and res[0]["reason"] == "answered"
    assert [n.person for o in outs for n in o.nudges if n.card == rt.CALL_CARD] == ["N1"]
