"""The scenario presets after QA: staff are told their own work and the robot's job in words written to them, moods
written for the public do not tell staff to ignore the robot, each robot brief points the robot at its own task, and
the preset timelines and starts let the scenario play out (Quick 2D world)."""
import math
from pathlib import Path

import pytest
import yaml

from wardzero_agents.config.models import CastEntry
from wardzero_agents.config.presets import (STAFF_PRESET_ROLES, load_moods, load_roles, load_scenario_presets,
                                            preset_to_scenario)
from wardzero_agents.site import load_site
from wardzero_agents.studio import auto_staff

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)
ROLES = load_roles(CONFIG)
MOODS = load_moods(CONFIG)
CARDS = {c["id"]: c for c in yaml.safe_load((CONFIG / "cards" / "studio.yaml").read_text())["cards"]}
WARD_NURSE_SCENES = ("meal_round", "night_call_bell", "wandering_patient", "specimen_run", "ward_fire")


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id):
        if preset_id not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p)
            cache[preset_id] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[preset_id]
    return get


def staff(preset_id):
    p = PRESETS[preset_id]
    return [c for c in auto_staff.with_auto(list(p.characters), p) if c.role in STAFF_PRESET_ROLES]


def persona_text(loaded, pid):
    per = loaded.personas[pid]
    return " ".join([per.condition, per.backstory, *per.goals])


# ---------- Bug 9: staff prompts ----------

def test_stressed_mood_has_a_staff_variant_that_answers_the_robot():
    m = MOODS["stressed"]
    assert m.staff_prompt
    for line in ("for a robot asking for anything", "go around it"):
        assert line not in m.staff_prompt
    assert "robot" in m.staff_prompt and "go" in m.staff_prompt


def test_public_stressed_line_does_not_ignore_a_robot_that_needs_something():
    assert "for a robot asking for anything" not in MOODS["stressed"].prompt


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_no_staff_member_is_told_to_ignore_or_go_around_the_robot(staged, preset_id):
    loaded = staged(preset_id)
    for c in staff(preset_id):
        text = persona_text(loaded, c.id)
        for bad in ("for a robot asking for anything", "go around it", "resent it", "carry the item yourself",
                    "do the step yourself"):
            assert bad not in text, (c.id, bad)


def test_ward_nurse_stock_lines_are_scoped_to_her_own_work():
    text = ROLES["ward_nurse"].prompt
    assert "resent it" not in text and "carry the item yourself" not in text
    fix = next(ln for ln in text.splitlines() if "fix it yourself" in ln)
    assert "own work" in fix
    assert "robot" in next(ln for ln in text.splitlines() if "interrupt" in ln)


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_staff_read_a_line_written_to_them_not_the_robot_job(staged, preset_id):
    p, loaded = PRESETS[preset_id], staged(preset_id)
    job = " ".join(p.robot_brief["job"].split())
    people = staff(preset_id)
    if people:
        assert p.staff_line, preset_id
    for c in people:
        text = persona_text(loaded, c.id)
        assert job[:60] not in text, c.id
        assert "written to the robot" not in text
        assert " ".join(p.staff_line.split())[:60] in text, c.id
    assert p.staff_line is None or "staff_line" not in p.robot_brief


def test_surgeon_is_not_told_the_tray_is_within_her_reach(staged):
    text = persona_text(staged("scrub_nurse"), "S1")
    assert "within reach" not in text and "just behind you" not in text
    assert "ask" in PRESETS["scrub_nurse"].staff_line.lower()


@pytest.mark.parametrize("preset_id", WARD_NURSE_SCENES)
def test_every_nurse_has_concrete_work_of_her_own(preset_id):
    nurses = [c for c in PRESETS[preset_id].characters if c.role == "ward_nurse"]
    assert nurses
    for c in nurses:
        assert c.role_prompt and len(c.role_prompt) > 80, (preset_id, c.id)
        assert c.role_prompt.rstrip().endswith("]"), (preset_id, c.id)


def test_meal_round_job_and_staff_line_name_the_call_bell():
    p = PRESETS["meal_round"]
    assert "call bell" in p.robot_brief["job"]
    assert "bell" in p.staff_line


def test_staff_evacuate_says_what_robot_failure_means():
    nudge = CARDS["staff_evacuate"]["nudge"]
    assert "unless the robot fails" not in nudge
    assert "Do not fight the fire yourself" in nudge and "robot" in nudge
    assert "asks you" in nudge and "spraying" in nudge


def test_evacuation_receptionist_leaves_the_guiding_to_the_robot(staged):
    rcp = next(c for c in auto_staff.with_auto([], PRESETS["evacuation_guide"]) if c.id == "RCP")
    assert rcp.role_prompt and "alarm" in rcp.role_prompt and "robot" in rcp.role_prompt
    assert "do not lead" in rcp.role_prompt.lower()
    goals = " ".join(staged("evacuation_guide").personas["RCP"].goals)
    assert "alarm" in goals
    other = next(c for c in auto_staff.with_auto([], PRESETS["reception_guide"]) if c.id == "RCP")
    assert other.role_prompt is None


# ---------- Bug 16: robot briefs ----------

@pytest.mark.parametrize("preset_id", ("meal_round", "night_call_bell", "wandering_patient"))
def test_ward_jobs_are_not_greeting_jobs(preset_id):
    assert PRESETS[preset_id].robot_brief.get("first_contact") is False


def test_evacuation_brief_names_the_safe_exit():
    b = PRESETS["evacuation_guide"].robot_brief
    words = " ".join(str(v) for k, v in b.items() if k != "destination")
    assert "Ambulance handover bay" in words and "safe" in words
    assert b["destination"] == "AE_Handover"  # the metric's key stays


def test_specimen_brief_matches_the_start_and_names_the_porter_spot():
    p = PRESETS["specimen_run"]
    job = p.robot_brief["job"]
    assert "ward nurse base" not in job
    assert "ward entrance" in job.lower() and "holding" in job.lower()
    assert "porter point" in job.lower()
    site = load_site(CONFIG)
    assert site.room_at(p.robot_start.x, p.robot_start.y) == "Ward_Entrance"


def test_night_call_bell_job_keeps_the_robot_with_the_patient():
    job = PRESETS["night_call_bell"].robot_brief["job"]
    assert "press_call" in job or "staff call button" in job
    assert "stay" in job.lower()
    assert job.index("call button") < job.index("tell") if "tell" in job else True


def test_reception_job_and_fetch_rule_agree():
    job = PRESETS["reception_guide"].robot_brief["job"]
    assert "note" in job and "desk" in job


# ---------- contract item 5: asleep ----------

def test_meal_round_m2_is_asleep(staged):
    p = PRESETS["meal_round"]
    m2 = next(c for c in p.characters if c.id == "M2")
    assert m2.start_asleep
    assert any(b.card == "asleep" and b.target == "M2" for b in p.timeline)
    entry = next(e for e in yaml.safe_load(Path(staged("meal_round").scenario_path).read_text())["cast"]
                 if e["id"] == "M2")
    assert entry.get("start_asleep", False) is ("start_asleep" in CastEntry.model_fields)


# ---------- Bug 19: timelines ----------

def test_ward_fire_alarm_and_evacuation_start_with_the_fire():
    tl = {b.id: b for b in PRESETS["ward_fire"].timeline}
    fire = next(b for b in tl.values() if b.card == "fire_start")
    for b in tl.values():
        assert b.at == fire.at, b.id
    assert {b.card for b in tl.values()} == {"fire_start", "alarm", "staff_evacuate"}


def test_preset_timelines_agree_with_the_flows():
    def secs(at):
        m, s = at[2:].split(":")
        return int(m) * 60 + int(s)
    wf = {b.card: secs(b.at) for b in PRESETS["ward_fire"].timeline}
    assert wf["fire_start"] == 30
    wp = {b.card: secs(b.at) for b in PRESETS["wandering_patient"].timeline}
    assert wp["patient_gets_up"] == 20


# ---------- Bug 18: reception start ----------

def test_reception_cast_starts_within_greeting_range_of_the_robot():
    p = PRESETS["reception_guide"]
    r = p.robot_start
    near = [c.id for c in p.characters if math.hypot(c.x - r.x, c.y - r.y) <= 2.5]
    assert near, "nobody within 2.5 m of the robot's post"
    assert all(math.hypot(c.x - r.x, c.y - r.y) >= 1.2 for c in p.characters)


# ---------- Bug 26 ----------

def test_wheelchair_users_are_not_bed_patients(staged):
    for preset_id, pid in (("visiting_rush", "W1"), ("specimen_run", "W1"), ("specimen_run", "W3")):
        loaded = staged(preset_id)
        inst = next(i for i in loaded.instances if i.person_id == pid)
        assert inst.role != "bed_patient", (preset_id, pid)
        assert loaded.personas[pid].mobility.aid == "wheelchair", (preset_id, pid)


def test_visiting_rush_arrivals_come_in_ahead_of_the_robot():
    p = PRESETS["visiting_rush"]
    site = load_site(CONFIG)
    lm = site.landmarks[p.arrival_entrance]
    goal = site.resolve(p.robot_brief["destination"])
    r = p.robot_start
    # ahead: between the robot and its destination, up the corridor
    assert r.y + 3.0 < lm.y < goal[1]
    assert site.room_at(lm.x, lm.y) == "Ward_Corridor"


def test_specimen_destination_is_not_where_the_porter_stands():
    p = PRESETS["specimen_run"]
    site = load_site(CONFIG)
    dest = site.resolve(p.robot_brief["destination"])
    po1 = next(c for c in p.characters if c.id == "PO1")
    d = math.hypot(po1.x - dest[0], po1.y - dest[1])
    assert 0.3 <= d <= 1.0, d  # off the spot, so the robot can stand on it, and still at it
    for c in p.characters:  # the porter is the one person standing by the porter point
        if c.id != "PO1":
            assert math.hypot(c.x - dest[0], c.y - dest[1]) > 1.3, c.id


def test_lost_visitor_card_does_not_name_the_ward():
    assert "ward" not in CARDS["lost_visitor"]["nudge"].lower()


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_custom_never_shows_in_a_condition_line(staged, preset_id):
    loaded = staged(preset_id)
    for pid, per in loaded.personas.items():
        assert "custom" not in per.condition.lower(), (preset_id, pid, per.condition)
        assert "Custom" not in per.label, (preset_id, pid, per.label)


def _sim(loaded, tmp_path):
    from wardzero_agents.ros.stub_sim_node import StubSimService
    from wardzero_agents.runlog import RunLog
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def test_specimen_robot_walking_up_to_the_porter_reaches_its_destination(staged, tmp_path):
    from wardzero_agents.bench.metrics import ARRIVE_M
    s = _sim(staged("specimen_run"), tmp_path)
    w = s.world
    dest = w.site.resolve(PRESETS["specimen_run"].robot_brief["destination"])
    ends, tries, closest = [], 0, math.inf
    for _ in range(900):
        if tries == len(ends) and tries < 4:
            tries += 1
            s.on_task_action({"id": f"a{tries}", "action": "approach", "track": w.tracks.track("PO1")})
        o = s.step(0.1)
        closest = min(closest, math.hypot(w.robot.x - dest[0], w.robot.y - dest[1]))
        ends += [r["reason"] for r in o.task_results if r.get("kind") == "goal"]
        if "arrived" in ends:
            break
    assert "arrived" in ends, ends
    assert closest <= ARRIVE_M, closest


def test_reception_robot_sees_someone_within_greeting_range_at_the_start(staged):
    from wardzero_agents.world.perception import perceive_people
    from wardzero_agents.world.stub_world import StubWorld
    w = StubWorld(staged("reception_guide"), 1)
    seen = {w.tracks.person(d["track"]) for d in perceive_people(w, w.robot)}
    near = {pid for pid in seen if math.hypot(w.people[pid].x - w.robot.x, w.people[pid].y - w.robot.y) <= 2.5}
    assert near, seen
