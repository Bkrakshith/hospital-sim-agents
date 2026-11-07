"""People's prompts leave the robot's job to the robot: staff read the robot's instructions as the robot's, not as
their own goals, and nobody is told the robot is a reception robot when it has another job."""
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import yaml
from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.config.presets import load_roles, load_scenario_presets, preset_to_scenario, robot_duty_line
from wardzero_agents.llm.fake import FakeGateway
from wardzero_agents.mind.prompts import SYSTEM_FIXED, _seen_line, persona_card, robot_role_of, system_fixed
from wardzero_agents.mind.runtime import Mind, MindDeps
from tests.test_mind import RecordingEffects, RecordingRunLog, decision, obs, seen, status

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)


def _staff_line(pid):
    return " ".join(PRESETS[pid].staff_line.split())


def test_staff_duty_line_is_written_to_staff_not_quoted_from_the_robot(tmp_path):
    loaded = preset_to_scenario(PRESETS["ae_waiting"], CONFIG, tmp_path)
    goal = loaded.personas["N1"].goals[0]
    assert goal.startswith(_staff_line("ae_waiting")[:40])
    assert "Leave that work to the robot" in goal
    assert "written to the robot" not in goal
    assert "You are the first point of contact" not in goal


def test_duty_line_scopes_fix_it_yourself_to_own_work(tmp_path):
    loaded = preset_to_scenario(PRESETS["meal_round"], CONFIG, tmp_path)
    assert "for your own work, not the robot's" in loaded.personas["N1"].goals[0]


def test_a_multiline_staff_line_stays_one_goal(tmp_path):
    p = PRESETS["meal_round"]
    line = "The robot brings the trays.\nIt answers the bells\n  during the round.\n"
    nurse = preset_to_scenario(p.model_copy(update={"staff_line": line}), CONFIG, tmp_path).personas["N1"]
    duty = [g for g in nurse.goals if "Leave that work to the robot" in g]
    assert len(duty) == 1 and nurse.goals[0] == duty[0]
    assert duty[0].startswith("The robot brings the trays. It answers the bells during the round.")
    assert "\n" not in robot_duty_line(line).strip()


def test_without_a_staff_line_staff_get_the_robot_role_not_its_job(tmp_path):
    p = PRESETS["meal_round"]
    nurse = preset_to_scenario(p.model_copy(update={"staff_line": None}), CONFIG, tmp_path).personas["N1"]
    assert nurse.goals[0].startswith("The robot on this shift (ward meal assistant) has a job of its own.")
    assert p.robot_brief["job"][:40] not in nurse.backstory


def test_duty_line_keeps_its_source_tag_and_no_bracket_from_the_line():
    line = robot_duty_line("Fetch the [urgent] box.")
    assert line.rstrip().endswith("[scenario brief]")
    assert line.count("[") == 1


def _mind_request(loaded, pid):
    """The system prompt and persona card one person's mind sends."""
    reqs = []

    def policy(req):
        reqs.append(req)
        return decision("wait_standing")

    inst = next(i for i in loaded.instances if i.person_id == pid)

    async def go():
        deps = MindDeps(inst=inst, loaded=loaded, gateway=FakeGateway(policy=policy), kb=None, ltm=None,
                        run_id="r", run_index=1, effects=RecordingEffects(), runlog=RecordingRunLog(),
                        checkpointer=MemorySaver())
        m = Mind(deps)
        await m.on_tick(5, obs(person=pid, t=5, visible=[seen("robot", "g1", 2.0)]),
                        status(person=pid, t=5, posture="standing"), [])
        await m.drain()
        await m.aclose()

    asyncio.run(go())
    assert reqs
    return reqs[-1]


def test_people_are_told_the_robot_by_its_role(tmp_path):
    loaded = preset_to_scenario(PRESETS["ward_fire"], CONFIG, tmp_path)
    req = _mind_request(loaded, "N1")
    assert "The robot is g1 (ward fire responder)." in req.persona_card
    assert "reception robot" not in req.persona_card
    assert "reception robot" not in req.system_fixed
    assert "reception robot" not in system_fixed("ward_nurse", robot_role="ward fire responder")
    assert "reception robot" not in _seen_line(seen("robot", "g1"), robot_role="ward fire responder")
    inst = next(i for i in loaded.instances if i.person_id == "N1")
    no_role = SimpleNamespace(robot_brief={"job": "Help people."})
    assert robot_role_of(no_role) == ""  # a brief with no role
    card = persona_card(inst, loaded.area, loaded.site, robot_role=robot_role_of(no_role))
    assert "The robot is g1.\n" in card + "\n"


def test_scenarios_with_no_robot_brief_keep_their_exact_prompt():
    """The original reception scenarios have no robot brief: their pinned prompt and card stay as they were."""
    assert robot_role_of(SimpleNamespace(robot_brief=None)) is None
    for role in ("patient", "companion", "visitor", "staff"):
        assert system_fixed(role) == SYSTEM_FIXED


def test_receptionist_steps_in_only_when_the_robot_cannot_help():
    text = load_roles(CONFIG)["receptionist"].prompt
    assert "will step in for them" not in text
    line = next(ln for ln in text.splitlines() if "step in for them" in ln)
    assert re.search(r"robot asks you to|robot could not help", line)


def test_staff_evacuate_nudge_leaves_the_fire_alone():
    cards = {c["id"]: c for c in yaml.safe_load((CONFIG / "cards" / "studio.yaml").read_text())["cards"]}
    nudge = cards["staff_evacuate"]["nudge"]
    assert "whoever has an extinguisher" not in nudge
    assert "robot" in nudge and "Do not fight the fire yourself" in nudge
