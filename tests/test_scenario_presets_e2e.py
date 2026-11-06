"""Every scenario preset, end to end on the stub backend, with no ROS.

The stub sim service, the director, the agent service (real minds, fake LLM) and the built-in robot
mind (fake gateway) run in one process in lockstep, for the preset's whole duration at time scale 10. Then the
report is built from the run folder."""
import asyncio
import json
import math
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.bench.metrics import RunData, people_in_furniture_s, robot_near_furniture_s
from wardzero_agents.bench.report import build_report
from wardzero_agents.config.presets import load_scenario_presets, people_meta, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.llm.gateway import LLMResult
from wardzero_agents.mind.runtime import Mind, MindDeps
from wardzero_agents.robots.claude_robot import RobotMind, decision_outputs
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile, get_profile
from wardzero_agents.ros.agent_node import TICK_S, AgentService
from wardzero_agents.ros.director_node import DirectorService
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)
TIME_SCALE = 10.0
WALL_STEP = 0.01  # x TIME_SCALE = 0.1 sim-s per step
ROBOT_NEAR_FURNITURE_MAX_S = 10.0  # standing beside a bed or a desk to help someone is fine; brushing past is not


# A movable stand-in for a bring-your-own wheeled robot (same sensors as the built-in one).
MOVER = RobotProfile(name="test_mover", can_move=True, base="wheeled", radius=0.3, reach=(0.4, 1.2),
                     max_speed=1.0, ns="robot", sensors=G1_CLAUDE.sensors)


class RobotGateway:
    """A scripted robot: walk to the first known place, press the call button, then stay quiet."""

    def __init__(self, places):
        self.places, self.n, self.reqs = places, 0, 0

    async def complete(self, req):
        self.reqs += 1
        self.n += 1
        a = {"say": "", "toward_track": "", "action": "none", "target": "", "item": "", "destination": "",
             "place": [], "reason": "scripted"}
        if self.n == 1:
            a.update(action="go_to", destination=self.places[0].replace("_", " "))
        elif self.n == 2:
            a.update(action="press_call", say="I have called for staff.")
        return LLMResult(True, a, None, "fake", cost_usd=0.0)


class Wire:
    """The director's outbox: collects what it publishes."""

    def __init__(self):
        self.msgs = []

    def publish(self, kind, obj):
        self.msgs.append((kind, obj))


class AgentOut:
    def __init__(self):
        self.items = []

    def publish_intent(self, i):
        self.items.append(("intent", i))

    def publish_say(self, s):
        self.items.append(("say", s))


def run_scenario(preset_id, tmp_path, seconds=None, profile=None):
    preset = PRESETS[preset_id]
    seconds = preset.duration_min * 60.0 if seconds is None else seconds
    loaded = preset_to_scenario(preset, CONFIG, tmp_path / "staged")
    run_dir = tmp_path / "run"
    runlog = RunLog(run_dir)
    runlog.copy_configs(loaded)
    runlog.write_meta({"run_id": "e2e", "scenario_id": loaded.scenario.id, "preset_id": preset_id,
                       "seed": loaded.scenario.seed, "test_mode": True, "people": people_meta(loaded, preset),
                       "models": loaded.scenario.models.model_dump()})
    sim = StubSimService(loaded, runlog, TIME_SCALE, True, ())
    wire, aout = Wire(), AgentOut()
    director = DirectorService(loaded, runlog, wire, "e2e")
    gw = FakeGateway(default_policy)
    clock = {"t": 0.0}

    def mind_factory(inst, effects):
        return Mind(MindDeps(inst=inst, loaded=loaded, gateway=effects.guard(gw), kb=None, ltm=None,
                             run_id="e2e", run_index=1, effects=effects, runlog=runlog,
                             checkpointer=MemorySaver(), clock=lambda: clock["t"]))

    agents = AgentService(loaded, runlog, aout, mind_factory, now=lambda: clock["t"])
    cfg = loaded.scenario.robot
    rgw = RobotGateway(loaded.scenario.known_places)
    robot = RobotMind(loaded.site, sim.robot_brief, rgw, lambda: clock["t"], profile=profile or get_profile(cfg.profile),
                      known_places=loaded.scenario.known_places, pose=(cfg.x, cfg.y, cfg.yaw))
    stats = {"decisions": 0, "ticks": 0}

    async def drive():
        steps = int(seconds / (WALL_STEP * TIME_SCALE))
        last_agent = last_dir = -1.0
        for _ in range(steps):
            out = sim.step(WALL_STEP)
            t = clock["t"] = out.clock
            if out.roster is not None:
                agents.on_roster(out.roster)
                director.on_roster(out.roster)
            for st in out.statuses:
                agents.on_status(st)
                director.on_status(st)
            for ob in out.observations:
                agents.on_observation(ob)
            for n in out.nudges:
                agents.on_nudge(n)
            if out.odom:
                robot.on_pose(out.odom["x"], out.odom["y"], out.odom["yaw"])
            if out.detections:
                robot.on_people(out.detections, t)
                robot.on_objects(out.objects)
            for r in out.task_results:
                robot.on_task_result(r, t)
            for h in out.robot_heard:
                await robot.on_heard(h.model_dump(mode="json"), t)
            if t - last_dir >= 0.5:
                last_dir = t
                director.tick(t)
                for kind, obj in wire.msgs:
                    if kind == "nudge":
                        agents.on_nudge(obj)
                    elif kind == "reflex":
                        agents.on_reflex(obj)
                    elif kind == "world":
                        sim.on_world_event(obj)
                wire.msgs.clear()
            if t - last_agent >= TICK_S:
                last_agent = t
                await agents.tick(t)
                stats["ticks"] += 1
            d = await robot.tick(t)
            if d is not None:
                stats["decisions"] += 1
                o = decision_outputs(d, robot, robot.next_id())
                if o.say:
                    sim.on_robot_say(o.say)
                if o.action:
                    sim.on_task_action(o.action)
                if o.goal:
                    sim.on_goal(*o.goal)
            for kind, obj in aout.items:
                (sim.on_intent if kind == "intent" else sim.on_say)(obj)
            aout.items.clear()
            sim.set_pending(agents.pending_count())
            deadline = time.monotonic() + 5.0
            while agents.pending_count() and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            assert not agents.pending_count(), f"the agents never settled (t={t:.1f} s)"
            sim.set_pending(0)
        await agents.aclose()

    asyncio.run(drive())
    return preset, loaded, run_dir, sim, stats


def rows(run_dir, stream):
    p = run_dir / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def beat_seconds(at):
    mm, ss = at.split("+")[1].split(":")
    return int(mm) * 60 + int(ss)


def check_preset(preset_id, tmp_path, caplog, profile=None):
    with caplog.at_level("WARNING"):
        preset, loaded, run_dir, sim, stats = run_scenario(preset_id, tmp_path, profile=profile)
    run_s = preset.duration_min * 60.0
    bad = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert not bad, bad[:3]  # a failed tick is logged, not raised: no errors may have been logged
    assert stats["ticks"] > 100 and sim.t == pytest.approx(run_s, abs=0.2)
    assert stats["decisions"] >= 1

    # everyone with a mind takes stock as soon as they are there (at the start, or when they arrive), not only
    # when something new happens
    first = {}
    for r in rows(run_dir, "decisions"):
        first.setdefault(r["person"], r["t"])
    late = {i.person_id: first.get(i.person_id) for i in loaded.instances
            if first.get(i.person_id) is None or first[i.person_id] > i.arrive_at_s + 10.0}
    assert not late, f"no decision in the first 10 s: {late}"

    # the people did something: a sampled (not static) person spoke or acted
    moving = [r for r in rows(run_dir, "says") + rows(run_dir, "intents")
              if r.get("person") and not r["person"].startswith("SB") and r["person"] != "g1"]
    assert moving, "no person spoke or acted"
    assert rows(run_dir, "status") and rows(run_dir, "robot_pose")

    # every beat of the timeline fired
    due = list(preset.timeline)
    assert all(beat_seconds(b.at) < run_s for b in due)
    fired = [c["card"] for c in rows(run_dir, "cards") if c["ok"]]
    for b in due:
        assert b.card in fired, f"beat {b.id} ({b.card}) did not fire"

    # nobody stood inside furniture, and the robot hardly brushed it
    data = RunData.load(run_dir)
    assert people_in_furniture_s(data) == 0
    assert (robot_near_furniture_s(data) or 0.0) <= ROBOT_NEAR_FURNITURE_MAX_S

    # the preset's world objects exist (trays, specimen box, call bell, instruments, extinguisher...)
    seen = {o.get("id") for r in rows(run_dir, "objects") for o in r["objects"]}
    kinds = {o.get("kind") for r in rows(run_dir, "objects") for o in r["objects"]}
    for o in preset.objects:
        oid = o.get("id") or f"bell_{o['bed']}"  # a call bell is named after its bed
        assert oid in seen, f"object {oid} never appeared"
    if any(b.card == "fire_start" for b in due):  # ward_fire and evacuation_guide: the fire exists
        assert "fire" in kinds
    for b in due:
        if b.card == "call_bell_at":  # the patient was nudged to ring (the fake mind does not)
            assert any(n["person"] == b.target for n in rows(run_dir, "nudges"))
    if any(b.card == "route_blocked" for b in due):
        assert "route_blocked" in {w["kind"] for w in rows(run_dir, "world")}

    # the robot's scripted action was acknowledged by a result with the same id
    tasks = rows(run_dir, "robot_tasks")
    acted = [r for r in tasks if r["kind"] == "action" and r.get("action") == "press_call"]
    assert acted, "the robot attempted no task"
    for a in acted:
        assert any(r["kind"] == "result" and r["id"] == a["id"] and r["ok"] for r in tasks)
    if profile is not None and profile.can_move:  # a goal-type action: the goal reports an outcome
        goals = [r for r in tasks if r["kind"] == "goal" or r.get("action") == "goal"]
        assert goals and goals[0]["reason"] in ("arrived", "stuck", "no_route", "superseded", "cancelled")

    build_report(run_dir, preset, None)
    report = json.loads((run_dir / "report.json").read_text())
    assert (run_dir / "report.html").is_file()
    assert report["run"]["scenario_id"] == preset_id
    assert len(report["criteria"]) == len(preset.criteria) > 0
    for c in report["criteria"]:  # evaluated: a value, or an explicit None that counts as not passed
        assert "value" in c and c["label"]
        assert c["passed"] is (None if c["value"] is None else c["passed"])
        assert c["value"] is not None or c["passed"] is None
        assert c["value"] is None or isinstance(c["passed"], bool)
    assert report["verdict"]["label"] in ("PASS", "FAIL")
    assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in report["metrics"].values())
    assert RunData.load(run_dir).meta["preset_id"] == preset_id


@pytest.mark.parametrize("preset_id", sorted(p for p in PRESETS if p != "scrub_nurse"))
def test_preset_runs_end_to_end(preset_id, tmp_path, caplog):
    check_preset(preset_id, tmp_path, caplog)


def test_movable_robot_goal_result(tmp_path, caplog):
    check_preset("reception_guide", tmp_path, caplog, profile=MOVER)


def test_preset_02(tmp_path, caplog):
    ids = sorted(PRESETS)
    assert len(ids) == 10
    check_preset(next(p for p in ids if p.startswith("scrub")), tmp_path, caplog)
