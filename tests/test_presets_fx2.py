"""Second fix round, people side of the presets: bed porters who can turn down the specimen box, a surgeon who leaves
the tray to the robot and staff lines per role, a stressed patient who still answers the robot's check-in, and an
outpatient on his feet in reception (Quick 2D world)."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import (ROLE_INTENTS, CharacterSpec, load_moods, load_scenario_presets,
                                            preset_to_scenario)
from wardzero_agents.contracts import Intent
from wardzero_agents.mind.prompts import persona_card, robot_role_of
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.world.objects import Item
from wardzero_agents.world.stub_world import StubWorld

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)
MOODS = load_moods(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            cache[preset_id] = preset_to_scenario(PRESETS[preset_id], CONFIG, tmp_path_factory.mktemp(preset_id))
        return cache[preset_id]
    return get


def inst(loaded, pid):
    return next(i for i in loaded.instances if i.person_id == pid)


def card(loaded, pid):
    return persona_card(inst(loaded, pid), loaded.area, loaded.site, loaded.scenario.known_places,
                        robot_role=robot_role_of(loaded.scenario))


def persona_text(loaded, pid):
    per = loaded.personas[pid]
    return " ".join([per.condition, per.backstory, *per.goals])


# ---------- Bug 13: bed porters ----------

@pytest.mark.parametrize("pid", ["TR1", "TR2"])
def test_bed_porters_can_turn_down_an_offer_and_still_wheel_a_bed(staged, pid):
    """Staff roles answer an offer (accept_item, refuse_item). The bed porters keep the staff role that escorts a bed:
    as porters, the score would take them for the specimen porter (people whose role is porter)."""
    loaded = staged("specimen_run")
    role = inst(loaded, pid).role
    assert {"accept_item", "refuse_item", "escort"} <= ROLE_INTENTS[role]
    assert "porter" not in role


def test_an_offer_to_a_bed_porter_ends_refused_not_no_answer(staged, tmp_path):
    """The robot holds the box out to TR2: he is told of the offer (his role can answer it) and turns it down."""
    loaded = staged("specimen_run")
    s = StubSimService(loaded, RunLog(tmp_path / "run"), 1.0, wait_for=())
    w = s.world
    tr = w.people["TR2"]
    w.robot.teleport(tr.x + 0.9 * math.cos(tr.yaw), tr.y + 0.9 * math.sin(tr.yaw), tr.yaw + math.pi)  # in front
    outs = []
    s.on_task_action({"id": "h1", "action": "hand_over", "item": "specimen_1", "to": w.tracks.track("TR2")})
    o = s.step(0.1)
    outs.append(o)
    told = [n for n in o.nudges if n.person == "TR2"]
    assert told, "a porter is told the robot is holding something out to him"
    w.apply_intent(Intent(id="r", person="TR2", t=s.t, kind="refuse_item"))
    for _ in range(20):
        outs.append(s.step(0.1))
    res = [r for o in outs for r in o.task_results if r.get("id") == "h1" and r.get("reason") != "started"]
    assert res and res[-1]["reason"] == "refused"


# ---------- Bug 15: the surgeon ----------

def test_with_a_robot_scrub_nurse_the_surgeon_cannot_take_from_the_tray_herself(staged):
    loaded = staged("scrub_nurse")
    w = StubWorld(loaded, 1)
    s1 = w.people["S1"]
    w.objects.add_item(Item(id="X1", kind="instrument", label="scalpel", x=s1.x + 0.3, y=s1.y, z=1.0))
    w.apply_intent(Intent(id="i", person="S1", t=1.0, kind="take_instrument_self", text="scalpel"))
    assert w.objects.items["X1"].holder is None
    assert s1.state == "failed" and "robot" in (s1.reason or "")


def test_without_a_robot_the_surgeon_still_takes_her_own_instrument(tmp_path):
    p = PRESETS["scrub_nurse"]
    loaded = preset_to_scenario(p, CONFIG, tmp_path, robot_extra={"enabled": False})
    w = StubWorld(loaded, 1)
    s1 = w.people["S1"]
    w.objects.add_item(Item(id="X1", kind="instrument", label="scalpel", x=s1.x + 0.3, y=s1.y, z=1.0))
    w.apply_intent(Intent(id="i", person="S1", t=1.0, kind="take_instrument_self", text="scalpel"))
    assert w.objects.items["X1"].holder == "S1"


def test_the_circulating_nurse_does_not_get_the_surgeons_line(staged):
    loaded = staged("scrub_nurse")
    p = PRESETS["scrub_nurse"]
    surgeon_line = " ".join(p.staff_line_for("surgeon").split())
    nurse_line = " ".join(p.staff_line_for("ward_nurse").split())
    assert surgeon_line != nurse_line
    assert "ask" in surgeon_line.lower() and "take what it holds out to you" in surgeon_line
    assert "take what it holds out to you" not in nurse_line and "you ask" not in nurse_line
    assert surgeon_line[:60] in persona_text(loaded, "S1")
    n1 = persona_text(loaded, "N1")
    assert nurse_line[:60] in n1 and "take what it holds out to you" not in n1


def test_a_plain_staff_line_is_every_roles_line():
    p = PRESETS["meal_round"]
    assert p.staff_line_for("ward_nurse") == p.staff_line == p.staff_line_for("porter")


# ---------- Bug 16: the stressed patient ----------

def test_stressed_mood_has_a_patient_variant_that_answers_the_robot():
    m = MOODS["stressed"]
    assert m.patient_prompt
    assert "from the robot" not in m.patient_prompt and "go around it" not in m.patient_prompt
    assert "robot" in m.patient_prompt and "answer" in m.patient_prompt
    assert all("[" in ln for ln in m.patient_prompt.splitlines() if ln.strip())


def test_ae_waiting_stressed_patient_gets_the_patient_variant(staged):
    loaded = staged("ae_waiting")
    text = persona_text(loaded, "P2")
    assert "Do not stop for strangers or for small talk, from people or from the robot" not in text
    assert "If the robot slows you down even once, go around it" not in text
    assert " ".join(MOODS["stressed"].patient_prompt.split()).split(" [")[0] in text


def test_stressed_visitors_and_staff_keep_their_lines(staged):
    rec = staged("reception_guide")
    assert "go around it" in persona_text(rec, "V2")  # a stressed visitor: the public line
    ae = staged("ae_waiting")
    assert MOODS["stressed"].staff_prompt.split(".")[0] in persona_text(ae, "N1")


# ---------- Bug 20: the outpatient in reception ----------

def test_h1_is_an_outpatient_on_his_feet_who_is_glad_of_the_robots_help(staged):
    loaded = staged("reception_guide")
    c = next(c for c in PRESETS["reception_guide"].characters if c.id == "H1")
    assert c.mood != "sad"
    text = card(loaded, "H1")
    assert "Role: outpatient." in text and "bed_patient" not in text
    assert "less inclined to start conversations" not in text
    assert "follow" in ROLE_INTENTS[inst(loaded, "H1").role]
    assert inst(loaded, "H1").start is None or inst(loaded, "H1").start.bed is None


def test_a_persona_role_override_must_be_a_patient_role():
    for bad in ("astronaut", "ward_nurse"):
        with pytest.raises(ValueError):
            CharacterSpec(id="X", name="A B", sex="male", age_band="31-50", role="patient", mood="custom",
                          room="Lobby", x=0.0, y=0.0, persona_role=bad)


def test_the_override_is_dropped_when_the_character_is_moved_to_a_non_patient_role(tmp_path):
    p = PRESETS["reception_guide"]
    chars = [c.model_copy(update={"role": "visitor"}) if c.id == "H1" else c for c in p.characters]
    loaded = preset_to_scenario(p, CONFIG, tmp_path, characters=chars)
    assert inst(loaded, "H1").role == "ward_visitor"


def test_the_reception_arrivals_are_unchanged_by_h1s_role(staged):
    """Arrivals pick among the people who could walk in: an outpatient still can, so the same newcomer comes."""
    from wardzero_agents.director.cards import ARRIVAL_ROLES
    loaded = staged("reception_guide")
    assert inst(loaded, "H1").role in ARRIVAL_ROLES
