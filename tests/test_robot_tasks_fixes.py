"""Robot tasks and world mechanics found wanting by the oracle runs: the speed cap before the move, actions planned on
the step, tell spots clear of the furniture, hand-over offers and their reasons, the follow action, guide mechanics,
sleep, blocked routes and what the logs carry (Quick 2D world)."""
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, WorldEvent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world import robot_tasks as rt

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}

    def get(preset_id, auto=False):
        key = (preset_id, auto)
        if key not in cache:
            p = PRESETS[preset_id]
            chars = auto_staff.with_auto(list(p.characters), p) if auto else None
            cache[key] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id), characters=chars)
        return cache[key]
    return get


def svc(loaded, tmp_path, time_scale=1.0):
    return StubSimService(loaded, RunLog(tmp_path / "run"), time_scale, wait_for=())


def rows(run, stream):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def results(outs, aid=None, kind=None):
    out = [r for o in outs for r in o.task_results]
    if aid is not None:
        out = [r for r in out if r.get("id") == aid]
    if kind == "goal":
        out = [r for r in out if r.get("kind") == "goal"]
    elif kind == "action":
        out = [r for r in out if r.get("kind") != "goal"]
    return out


def put(w, pid, x, y, yaw=0.0, posture="standing"):
    p = w.people[pid]
    p.x, p.y, p.yaw, p.posture, p.seat, p.plan = x, y, yaw, posture, None, []


def advance(s, secs, dt=0.1):
    return [s.step(dt) for _ in range(int(round(secs / dt)))]


def near_people(w):
    return min((math.hypot(q.x - w.robot.x, q.y - w.robot.y) for q in w.people.values()
                if q.posture not in ("gone", "lying") and q.kind != "static"), default=math.inf)


# ---- the near-person speed cap is set before the robot moves ------------------------------------------------------
@pytest.mark.parametrize("scale,wall", [(1.0, 0.1), (10.0, 0.5)])
def test_speed_cap_holds_on_the_step_the_robot_comes_near(staged, tmp_path, scale, wall):
    s = svc(staged("visiting_rush"), tmp_path, scale)
    w = s.world
    for pid in list(w.people):
        if pid != "V1":
            w.people[pid].posture = "gone"
    put(w, "V1", 19.9, 11.0)  # beside the corridor, 0.9 m off the robot's line up it
    s.on_goal(19.0, 16.0, None)
    fast_near = []
    for _ in range(int(60 / (scale * wall))):
        s.step(wall)
        d = near_people(w)
        if d <= rt.NEAR_PERSON_M and w.robot.speed > rt.NEAR_PERSON_SPEED + 1e-6:
            fast_near.append((round(s.t, 2), round(d, 2), round(w.robot.speed, 2)))
        if not w.robot.has_goal:
            break
    assert math.hypot(w.robot.x - 19.0, w.robot.y - 16.0) < 0.3
    assert not fast_near


# ---- a tell spot is clear of the furniture -------------------------------------------------------------------------
@pytest.mark.parametrize("scale,wall", [(1.0, 0.1), (3.0, 0.1), (10.0, 0.1)])
def test_tell_staff_in_b6_stops_clear_and_the_robot_can_walk_back_to_the_patient(staged, tmp_path, scale, wall):
    """night_call_bell at x3 froze the robot for 430 s in B6: it stopped to tell the nurse in the furniture's
    clearance band, and no walk could start from there."""
    s = svc(staged("night_call_bell", auto=True), tmp_path, scale)
    w = s.world
    x, y = s.tasks._reach_spots(w.people["P1"], "tray")[0]
    w.robot.teleport(x, y, 0.0)
    tr = w.tracks.track("P1")
    s.on_task_action({"id": "tl", "action": "tell_staff", "about": tr})
    told = None
    for _ in range(int(120 / (scale * wall))):
        o = s.step(wall)
        if [r for r in o.task_results if r.get("id") == "tl" and r.get("kind") != "goal"]:
            told = (w.robot.x, w.robot.y)
            break
    assert told is not None and w.robot.router.is_clear(*told)
    s.on_task_action({"id": "ap", "action": "approach", "track": tr})
    goals = []
    for _ in range(int(120 / (scale * wall))):
        o = s.step(wall)
        goals += [(r["ok"], r["reason"]) for r in o.task_results if r.get("id") == "ap" and r.get("kind") == "goal"]
        if goals:
            break
    assert goals == [(True, "arrived")]
