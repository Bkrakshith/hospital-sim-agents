"""Every scenario preset, end to end on the stub backend, with no ROS.

The stub sim service, the director, the agent service (real minds, fake LLM) and the built-in robot
mind (fake gateway) run in one process in lockstep: 120 sim-seconds at time scale 10. Then the report
is built from the run folder."""
import asyncio
import json
import math
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.bench.metrics import RunData
from wardzero_agents.bench.report import build_report
from wardzero_agents.config.presets import load_scenario_presets, people_meta, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.llm.gateway import LLMResult
from wardzero_agents.mind.runtime import Mind, MindDeps
from wardzero_agents.robots.claude_robot import RobotMind, decision_outputs
from wardzero_agents.robots.profile import get_profile
from wardzero_agents.ros.agent_node import TICK_S, AgentService
from wardzero_agents.ros.director_node import DirectorService
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog

CONFIG = Path(__file__).resolve().parents[1] / "config"
PRESETS = load_scenario_presets(CONFIG)
RUN_S = 120.0
TIME_SCALE = 10.0
WALL_STEP = 0.01  # x TIME_SCALE = 0.1 sim-s per step


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


def run_scenario(preset_id, tmp_path, seconds=RUN_S):
    preset = PRESETS[preset_id]
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
    robot = RobotMind(loaded.site, sim.robot_brief, rgw, lambda: clock["t"], profile=get_profile(cfg.profile),
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
            sim.set_pending(agents.pending_count())
        await agents.aclose()

    asyncio.run(drive())
    return preset, loaded, run_dir, sim, stats


def rows(run_dir, stream):
    p = run_dir / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def check_preset(preset_id, tmp_path, caplog):
    with caplog.at_level("WARNING"):
        preset, loaded, run_dir, sim, stats = run_scenario(preset_id, tmp_path)
    bad = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert not bad, bad[:3]  # a failed tick is logged, not raised: no errors may have been logged
    assert stats["ticks"] > 100
    assert rows(run_dir, "decisions") or rows(run_dir, "intents"), "no person ever decided anything"
    assert sim.t == pytest.approx(RUN_S, abs=0.2)
    assert stats["decisions"] >= 1
    tasks = rows(run_dir, "robot_tasks")
    assert any(r["kind"] == "action" for r in tasks), "the robot attempted no task"
    assert rows(run_dir, "status") and rows(run_dir, "robot_pose")
    build_report(run_dir, preset, None)
    report = json.loads((run_dir / "report.json").read_text())
    assert (run_dir / "report.html").is_file()
    assert report["run"]["scenario_id"] == preset_id
    assert len(report["criteria"]) == len(preset.criteria) > 0
    assert all("passed" in c and c["label"] for c in report["criteria"])
    assert report["verdict"]["label"] in ("PASS", "FAIL")
    assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in report["metrics"].values())
    assert RunData.load(run_dir).meta["preset_id"] == preset_id


@pytest.mark.parametrize("preset_id", sorted(p for p in PRESETS if p != "scrub_nurse"))
def test_preset_runs_end_to_end(preset_id, tmp_path, caplog):
    check_preset(preset_id, tmp_path, caplog)


def test_preset_02(tmp_path, caplog):
    ids = sorted(PRESETS)
    assert len(ids) == 10
    check_preset(next(p for p in ids if p.startswith("scrub")), tmp_path, caplog)
