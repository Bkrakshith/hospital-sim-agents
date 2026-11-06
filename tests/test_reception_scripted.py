"""Reception and guide, scripted, on the stub backend (no ROS, no paid calls).

The sim service, director, agent service (real minds, fake LLM) and the built-in robot mind (fake LLM) run in one
process in lockstep, with WARDZERO_TEST_SCRIPTED_SAY putting two requests in people's mouths:
  1. Priya walks up and asks for a nurse: the robot walks to the counter and tells the receptionist, who gets up,
     goes to Priya and then back to her chair;
  2. Daniel asks for the ward: the robot guides him through the compartment doors to ward_entrance, he follows and
     arrives, and the robot walks back to its post.
The report's reached, unanswered and contacts criteria pass."""
import asyncio
import json
import math
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wardzero_agents.bench.report import build_report
from wardzero_agents.config.presets import load_scenario_presets, people_meta, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.mind.runtime import Mind, MindDeps
from wardzero_agents.robots.claude_robot import RobotMind, decision_outputs
from wardzero_agents.robots.profile import get_profile, profile_for_sim
from wardzero_agents.ros.agent_node import TICK_S, AgentService
from wardzero_agents.ros.director_node import DirectorService
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff

CONFIG = Path(__file__).resolve().parents[1] / "config"
TIME_SCALE = 10.0
WALL_STEP = 0.01
RUN_S = 150.0
SCRIPT = [
    {"t": 2.0, "who": "V1", "do": "approach", "target": "g1"},
    {"t": 9.0, "who": "V1", "text": "Excuse me, I'm Priya. I need a nurse, please.", "to": "g1"},
    {"t": 40.0, "who": "V2", "text": "Hello, could you take me to the ward, please?", "to": "g1"},
]


class Wire:
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


def run_reception(tmp_path, monkeypatch):
    script = tmp_path / "say.json"
    script.write_text(json.dumps(SCRIPT))
    monkeypatch.setenv("WARDZERO_TEST_SCRIPTED_SAY", str(script))
    preset = load_scenario_presets(CONFIG)["reception_guide"]
    chars = auto_staff.with_auto(list(preset.characters), preset)
    loaded = preset_to_scenario(preset, CONFIG, tmp_path / "staged", characters=chars)
    run_dir = tmp_path / "run"
    runlog = RunLog(run_dir)
    runlog.copy_configs(loaded)
    runlog.write_meta({"run_id": "scripted", "scenario_id": loaded.scenario.id, "preset_id": preset.id,
                       "seed": loaded.scenario.seed, "test_mode": True, "people": people_meta(loaded, preset),
                       "models": loaded.scenario.models.model_dump()})
    sim = StubSimService(loaded, runlog, TIME_SCALE, True, ())
    wire, aout = Wire(), AgentOut()
    director = DirectorService(loaded, runlog, wire, "scripted")
    gw = FakeGateway(default_policy)
    clock = {"t": 0.0}

    def mind_factory(inst, effects):
        return Mind(MindDeps(inst=inst, loaded=loaded, gateway=effects.guard(gw), kb=None, ltm=None,
                             run_id="scripted", run_index=1, effects=effects, runlog=runlog,
                             checkpointer=MemorySaver(), clock=lambda: clock["t"]))

    agents = AgentService(loaded, runlog, aout, mind_factory, now=lambda: clock["t"])
    cfg = loaded.scenario.robot
    robot = RobotMind(loaded.site, sim.robot_brief, FakeGateway(default_policy), lambda: clock["t"],
                      profile=profile_for_sim(get_profile(cfg.profile), "stub"),
                      known_places=loaded.scenario.known_places, pose=(cfg.x, cfg.y, cfg.yaw))
    trace = []

    async def drive():
        last_agent = last_dir = -1.0
        for _ in range(int(RUN_S / (WALL_STEP * TIME_SCALE))):
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
            d = await robot.tick(t)
            if d is not None:
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
            sim.set_pending(0)
            w = sim.world
            trace.append((t, {pid: (p.x, p.y, p.posture) for pid, p in w.people.items()}, w.robot.pose))
        await agents.aclose()

    asyncio.run(drive())
    return preset, run_dir, sim, trace


def rows(run_dir, stream):
    p = run_dir / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield run_reception(tmp_path_factory.mktemp("reception"), mp)
    finally:
        mp.undo()


def test_priya_asks_for_a_nurse_and_the_receptionist_comes_to_her(scripted):
    preset, run_dir, sim, trace = scripted
    tasks = rows(run_dir, "robot_tasks")
    told = [r for r in tasks if r.get("kind") == "result" and r.get("action") == "tell_staff"]
    assert told and told[0]["ok"] and told[0]["reason"] == "told", tasks
    t_told = told[0]["t"]
    tell = [r for r in tasks if r.get("kind") == "tell"]
    assert tell[0]["about"] == "V1" and tell[0]["staff"] == "RCP"
    # she gets up, reaches Priya, then goes back and sits in her chair
    reached = [t for t, people, _ in trace if t > t_told and math.dist(people["RCP"][:2], people["V1"][:2]) <= 1.6]
    assert reached, "the receptionist never reached Priya"
    r1 = sim.world.site.seats["R1"]
    back = [t for t, people, _ in trace if t > reached[0] and people["RCP"][2] == "seated"
            and math.dist(people["RCP"][:2], (r1.x, r1.y)) < 0.05]
    assert back, "the receptionist did not go back to her chair"
    hello = [a for a in rows(run_dir, "audio") if a.get("source_id") == "RCP" and a.get("addressee") == "V1"]
    assert hello


def test_daniel_is_guided_to_the_ward_entrance_and_arrives(scripted):
    preset, run_dir, sim, trace = scripted
    tasks = rows(run_dir, "robot_tasks")
    start = [r for r in tasks if r.get("kind") == "guide" and r.get("event") == "start"]
    assert start and start[0]["person"] == "V2" and start[0]["place"] == "ward_entrance", tasks
    end = [r for r in tasks if r.get("kind") == "guide" and r.get("event") == "end"]
    assert end and end[0]["reason"] == "arrived"
    we = sim.world.site.landmarks["ward_entrance"]
    assert any(math.dist(people["V2"][:2], (we.x, we.y)) <= 2.0 for _, people, _ in trace)
    # he went through the compartment doors (x 11.2) behind the robot, and the robot walked back to its post
    assert any(people["V2"][0] > 11.3 for _, people, _ in trace)
    post = sim.world.site.landmarks["reception_g1"]
    assert math.dist(trace[-1][2][:2], (post.x, post.y)) < 0.3
    assert any(r.get("id", "").endswith("-return") and r.get("ok") for r in tasks if r.get("kind") == "goal")


def test_the_report_passes_reached_unanswered_and_contacts(scripted):
    preset, run_dir, sim, trace = scripted
    build_report(run_dir, preset, None)
    report = json.loads((run_dir / "report.json").read_text())
    crit = {c["id"]: c for c in report["criteria"]}
    for cid in ("reached", "unanswered", "contacts"):
        assert crit[cid]["passed"] is True, (cid, crit[cid])
    assert report["metrics"]["people_reached_destination_rate"] == 1.0
    assert [(g["person"], g["arrived"]) for g in report["guides"]] == [("V2", True)]
