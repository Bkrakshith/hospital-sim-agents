"""How a person's situation names the robot they can see: "the reception robot" only in the original reception
scenarios (no robot brief), "the robot" wherever the robot has a brief of its own."""
import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway
from wardzero_agents.mind.prompts import _seen_line
from wardzero_agents.mind.runtime import Mind, MindDeps
from tests.test_mind import RecordingEffects, RecordingRunLog, decision, obs, seen, status

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def _situation(loaded, pid):
    reqs = []

    def policy(req):
        reqs.append(req)
        return decision("wait_standing")

    inst = next(i for i in loaded.instances if i.person_id == pid)

    async def go():
        m = Mind(MindDeps(inst=inst, loaded=loaded, gateway=FakeGateway(policy=policy), kb=None, ltm=None,
                          run_id="r", run_index=1, effects=RecordingEffects(), runlog=RecordingRunLog(),
                          checkpointer=MemorySaver()))
        await m.on_tick(5, obs(person=pid, t=5, visible=[seen("robot", "g1", 2.0)]),
                        status(person=pid, t=5, posture="standing"), [])
        await m.drain()
        await m.aclose()

    asyncio.run(go())
    return reqs[-1].user


def test_seen_line_keeps_the_legacy_words_without_a_robot_role():
    assert _seen_line(seen("robot", "g1", 2.0)) == "- the reception robot g1 (2.0 m ahead)"
    assert _seen_line(seen("robot", "g1", 2.0), robot_role=None) == "- the reception robot g1 (2.0 m ahead)"
    assert _seen_line(seen("robot", "g1", 2.0), robot_role="") == "- the robot g1 (2.0 m ahead)"
    assert _seen_line(seen("robot", "g1", 2.0), robot_role="ward fire responder") == "- the robot g1 (2.0 m ahead)"


def test_original_reception_scenario_sees_the_reception_robot():
    loaded = load_scenario(CONFIG / "scenarios" / "waiting_area_v1.yaml")
    assert "- the reception robot g1 (2.0 m ahead)" in _situation(loaded, "P5")


def test_a_scenario_with_a_robot_brief_sees_the_robot(tmp_path):
    loaded = preset_to_scenario(load_scenario_presets(CONFIG)["ward_fire"], CONFIG, tmp_path)
    user = _situation(loaded, "N1")
    assert "- the robot g1 (2.0 m ahead)" in user
    assert "reception robot" not in user
