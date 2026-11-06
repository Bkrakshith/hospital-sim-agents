"""The scenario presets set the scene so that the robot, not the people, does each scenario's main task: who stands
where at the start, what the robot can see from its post, what the staff are told their own job is, and what the
robot is told to do (Quick 2D world)."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent
from wardzero_agents.site import Router, load_site
from wardzero_agents.studio import auto_staff
from wardzero_agents.world.perception import perceive_objects, perceive_people
from wardzero_agents.world.stub_world import ROUTE_BLOCK_RADIUS_M, StubWorld

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id, with_auto=False):
        key = (preset_id, with_auto)
        if key not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p) if with_auto else None
            cache[key] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[key]
    return get


def sim(loaded, tmp_path):
    from wardzero_agents.ros.stub_sim_node import StubSimService
    from wardzero_agents.runlog import RunLog
    return StubSimService(loaded, RunLog(tmp_path / "run"), wait_for=())


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


def results(outs, aid):
    return [r for o in outs for r in o.task_results if r["id"] == aid]


def seen_people(w):
    return {w.tracks.person(d["track"]) for d in perceive_people(w, w.robot)}


def role_text(preset_id, pid):
    return next(c for c in PRESETS[preset_id].characters if c.id == pid).role_prompt or ""


# ---------- ae_waiting ----------

def test_ae_p2_is_a_walk_in(staged):
    w = StubWorld(staged("ae_waiting"), 1)
    p2 = w.people["P2"]
    assert p2.seat is None and p2.posture == "standing", "Ava walks in: the robot checks her in and seats her"
    assert w.site.room_at(p2.x, p2.y) == "AE_Waiting"
    assert not w._seat_taken("A1", "")
    # she and the robot see each other from the start, so checking her in is the robot's doing, not luck
    assert "P2" in seen_people(w)
    assert "g1" in [v.id for v in w.observation("P2", 0.0, []).visible if v.kind == "robot"]


def test_ae_robot_reaches_george_past_the_walk_in(staged, tmp_path):
    s = sim(staged("ae_waiting", with_auto=True), tmp_path)
    s.on_task_action({"id": "a1", "action": "approach", "track": s.world.tracks.track("P1")})
    goals = [r for r in results(advance(s, 30.0), "a1") if r.get("kind") == "goal"]
    assert goals and goals[0]["reason"] in ("arrived", "already_there"), goals


def test_ae_triage_nurse_has_her_own_job(staged):
    goals = " ".join(staged("ae_waiting").personas["N1"].goals)
    assert "triage nurse" in goals
    assert "carry the item yourself" not in goals


def test_ae_brief_wait_matches_the_announcement():
    import yaml
    cards = {c["id"]: c for c in yaml.safe_load((CONFIG / "cards" / "studio.yaml").read_text())["cards"]}
    assert "three hours" in cards["waits_announcement"]["world"]["text"]
    assert "three hours" in PRESETS["ae_waiting"].robot_brief["waits"]


def test_ae_prompt_and_criterion_say_alert_staff():
    p = PRESETS["ae_waiting"]
    assert "alert triage" not in p.prompt and "alert staff" in p.prompt
    alerted = next(c for c in p.criteria if c.id == "alerted")
    assert "Staff alerted" in alerted.label


# ---------- evacuation_guide ----------

def _near_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy))


def test_evacuation_main_entrance_is_sealed_and_the_ambulance_bay_is_reachable():
    p = PRESETS["evacuation_guide"]
    site = load_site(CONFIG)
    alarm = next(b for b in p.timeline if b.card == "alarm")
    blocks = [b for b in p.timeline if b.card == "route_blocked"]
    assert len(blocks) == 3 and all(b.at == alarm.at for b in blocks)
    fire = next(b for b in p.timeline if b.card == "fire_start")
    e = site.landmarks["entrance"]
    assert fire.at == alarm.at and (fire.params["x"], fire.params["y"]) == (e.x, e.y)
    assert p.robot_brief["destination"] == "AE_Handover" and "AE_Handover" in p.known_places

    r = Router(site.grid)
    for b in blocks:
        r.block(b.params["x"], b.params["y"], ROUTE_BLOCK_RADIUS_M)
    n = int(round(1.0 / site.grid.res))
    ey, ex = site.grid.to_cell(e.x, e.y)
    free = [(ey + dy, ex + dx) for dy in range(-n, n + 1) for dx in range(-n, n + 1)
            if dy * dy + dx * dx <= n * n and site.grid.in_grid(ey + dy, ex + dx) and not r._raw[ey + dy][ex + dx]]
    assert not free, "a free cell within 1 m of the main entrance"
    chars = {c.id: c for c in p.characters}
    out = r.route((chars["V1"].x, chars["V1"].y), (e.x, e.y))
    assert out is None or math.hypot(out[-1][0] - e.x, out[-1][1] - e.y) > 1.0, "a way out through the entrance"
    goal = site.resolve("AE_Handover")
    for pid in ("V1", "V2", "P1"):
        path = r.route((chars[pid].x, chars[pid].y), goal)
        assert path is not None, pid
        for b in blocks:
            c = (b.params["x"], b.params["y"])
            assert min(_near_segment(c, a, q) for a, q in zip(path, path[1:])) > ROUTE_BLOCK_RADIUS_M - site.grid.res


def test_evacuation_brief_sends_the_robot_to_sweep_ae_waiting():
    job = PRESETS["evacuation_guide"].robot_brief["job"]
    assert "A&E waiting" in job and "blocked" in job


# ---------- wandering_patient ----------

def test_wandering_patient_does_not_walk_into_a_still_robot(staged):
    w = StubWorld(staged("wandering_patient"), 1)
    w.apply_intent(Intent(id="g", person="W1", t=0.0, kind="walk_to", target="Ward_Entrance"))
    w1, first, closest = w.people["W1"], None, math.inf
    for k in range(1, 601):
        w.step(k * 0.1)
        if w1.posture in ("standing", "walking"):
            closest = min(closest, math.hypot(w1.x - w.robot.x, w1.y - w.robot.y))
        if first is None and "W1" in seen_people(w):
            first = k * 0.1
    assert first is not None and first <= 10.0, first
    assert closest >= 2.5, "she passes the still robot: robot_near would be met without it acting"


def test_wandering_brief_keeps_the_robot_beside_her_and_on_the_ward():
    job = PRESETS["wandering_patient"].robot_brief["job"]
    assert "press the staff call button" in job and "do not walk away" in job
    assert "Never offer to take them off the ward" in job and "B4" in job


# ---------- night_call_bell ----------

def test_night_call_bell_light_in_view_at_start(staged):
    w = StubWorld(staged("night_call_bell"), 1)
    assert not perceive_people(w, w.robot)
    w.objects.ring("B2", 0.0)
    assert "B2" in [c["bed"] for c in perceive_objects(w, w.robot)["call_lights"]]


# ---------- reception_guide ----------

def test_reception_h1_is_in_view_and_off_the_route(staged, tmp_path):
    s = sim(staged("reception_guide", with_auto=True), tmp_path)
    w = s.world
    assert "H1" in seen_people(w)
    assert "appointment" in role_text("reception_guide", "H1")
    s.on_task_action({"id": "a1", "action": "approach", "track": w.tracks.track("V1")})
    outs = advance(s, 30.0)
    goals = [r for r in results(outs, "a1") if r.get("kind") == "goal"]
    assert goals and goals[0]["reason"] == "arrived", goals


# ---------- specimen_run ----------

def test_specimen_robot_reaches_the_porter_with_traffic(staged, tmp_path):
    p = PRESETS["specimen_run"]
    b0 = next(b for b in p.timeline if b.id == "b0")
    assert b0.params["place"] == "Main_Corridor"
    s = sim(staged("specimen_run"), tmp_path)
    w = s.world
    tries, ends, escorted = 0, [], False
    for _ in range(600):
        if not escorted and s.t >= 5.0:  # the bed porter sets off down the ward corridor with his patient
            escorted = True
            w.apply_intent(Intent(id="e", person="TR1", t=s.t, kind="escort", target=b0.params["patient"],
                                  text=b0.params["place"]))
        if tries == len(ends):  # walk up to the porter, and again when the bed at the lobby door held it up
            tries += 1
            s.on_task_action({"id": f"a{tries}", "action": "approach", "track": w.tracks.track("PO1")})
        o = s.step(0.1)
        ends += [r["reason"] for r in o.task_results if r.get("kind") == "goal"]
        if "arrived" in ends or tries > 3:
            break
    assert ends and ends[-1] == "arrived" and len(ends) <= 2, ends
    assert s.t <= 40.0
    po1 = w.people["PO1"]
    assert math.hypot(w.robot.x - po1.x, w.robot.y - po1.y) <= 1.1
    assert s.tasks.hand_over_place((w.robot.x, w.robot.y), po1, "specimen_box")[0] is not None


def test_specimen_bed_porters_refuse_the_box(staged):
    for pid in ("TR1", "TR2"):
        assert "not collecting any specimens" in role_text("specimen_run", pid), pid
    goals = " ".join(staged("specimen_run").personas["PO1"].goals)
    assert "go and fetch it" not in goals and "porter point" in goals
    assert PRESETS["specimen_run"].robot_brief.get("first_contact") is False


# ---------- scrub_nurse ----------

def test_scrub_robot_passes_without_walking_round(staged, tmp_path):
    p = PRESETS["scrub_nurse"]
    s = sim(staged("scrub_nurse"), tmp_path)
    w = s.world
    s.on_task_action({"id": "k1", "action": "take", "item": "scalpel"})
    outs = advance(s, 0.2)
    assert [(r["ok"], r["reason"]) for r in results(outs, "k1")] == [(True, "ok")]
    s1 = w.people["S1"]
    for o in p.objects:
        assert math.hypot(o["x"] - s1.x, o["y"] - s1.y) > 1.2, o["id"]
    assert p.robot_brief.get("first_contact") is False
    ids = p.robot_brief["instruments"]
    assert isinstance(ids, str) and all(o["id"] in ids for o in p.objects)
    assert "instrument table" not in p.prompt + p.robot_brief["job"]
    assert "instrument_table" not in p.known_places
    b2 = next(b for b in p.timeline if b.id == "b2")
    b1 = next(b for b in p.timeline if b.id == "b1")
    first = {x.strip() for x in b1.params["instruments"].split(",")}
    assert not first & {x.strip() for x in b2.params["instruments"].split(",")}


# ---------- meal_round ----------

def test_meal_m3_can_sit_up_when_asked():
    text = role_text("meal_round", "M3")
    assert "cannot sit up" not in text and "sit up in bed when someone asks" in text
    assert text.rstrip().endswith("]")
    assert "cannot sit up" not in PRESETS["meal_round"].prompt
    reach = next(c for c in PRESETS["meal_round"].criteria if c.id == "reach")
    assert reach.threshold == 2
