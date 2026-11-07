"""Run a scenario preset end to end with the scripted models, and judge the run as QA does.

Two ways to run one, both on the stub world at time scale 10, staged as the Studio stages a preset (with the auto
staff), with the scripted people (WARDZERO_TEST_FAKE_LLM=1):

- ``run_launch``: the real launch path, ``python -m wardzero_agents.cli run ... --sim stub --bot claude|none`` in a
  clean ROS 2 environment (the sim, director, agent and robot nodes talk over ROS, the robot on its own domain), with
  the built-in robot on the scripted model (WARDZERO_TEST_CLAUDE_BOT=1).
- ``run_inproc``: the same services in one process in lockstep, no ROS (quick, for working on a policy).

``judge`` then reads the run folder: the flow's final tree, the score after the scripted debrief, decision errors and
contacts. ``Verdict.problems`` lists in words what an ideal-robot run got wrong, and ``Verdict.credited`` what a
no-robot run was credited with.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
TIME_SCALE = 10.0
WALL_STEP = 0.01  # x TIME_SCALE = 0.1 sim-s per step (in-process)
PYTHON = os.environ.get("WARDZERO_E2E_PYTHON") or str(Path(os.sys.executable))
ROS_SETUP = "/opt/ros/humble/setup.bash"


def presets():
    from wardzero_agents.config.presets import load_scenario_presets
    return load_scenario_presets(CONFIG)


def stage(preset_id: str, out: Path):
    """The preset staged as the Studio stages it: its characters plus the auto staff. Returns the loaded scenario."""
    from wardzero_agents.config.presets import preset_to_scenario
    from wardzero_agents.studio import auto_staff
    p = presets()[preset_id]
    return preset_to_scenario(p, CONFIG, out / preset_id, characters=auto_staff.with_auto(list(p.characters), p))


# ---- the real launch path -------------------------------------------------------------------------------------------
def ros_ready() -> bool:
    return Path(ROS_SETUP).is_file()


STOP_GRACE_S = 20.0  # cli run stops its nodes within its own SIGINT, SIGTERM, SIGKILL ladder (10 s + 5 s)


def run_bounded(argv: list, *, env: dict, timeout_s: float) -> tuple[int, str, str]:
    """Run argv in a process group of its own and return (exit code, stdout, stderr). Past timeout_s it is sent
    SIGTERM, so ``cli run`` stops the nodes it started in sessions of their own (killing only the launcher would leave
    them publishing on the e2e domains), then the group is killed; raises TimeoutError."""
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout_s)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        for sig, wait in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
            try:
                out, err = p.communicate(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                out = err = ""
        raise TimeoutError(f"{' '.join(map(str, argv))[:200]} ran past {timeout_s:g} s and was stopped\n"
                           f"{(out or '')[-2000:]}\n{(err or '')[-2000:]}") from None


def run_launch(preset_id: str, work: Path, *, bot: str, domain: int, timeout_s: float = 1200.0) -> tuple[Path, str]:
    """Stage and run one preset through ``cli run`` (stub, x10). bot: claude (the built-in robot on the scripted
    model) or none. Returns (the run folder, the log). The child gets a clean environment: no session bus, no
    runtime dir, its own data and runs folders, people on domain, robot on domain + 1."""
    loaded = stage(preset_id, work / "staged")
    runs, data = work / "runs", work / "data"
    runs.mkdir(parents=True, exist_ok=True)
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": f"{Path(PYTHON).parent}:/usr/bin:/bin",
           "WARDZERO_TEST_FAKE_LLM": "1", "WARDZERO_TEST_CLAUDE_BOT": "1", "WARDZERO_DATA_DIR": str(data),
           "WARDZERO_RUNS_DIR": str(runs), "ROS_DOMAIN_ID": str(domain), "WARDZERO_SIM": "stub",
           "PYTHONPATH": str(ROOT)}
    cmd = (f"source {ROS_SETUP} >/dev/null 2>&1; export PYTHONPATH={ROOT}:$PYTHONPATH; cd {ROOT}; "
           f"exec {PYTHON} -m wardzero_agents.cli run {loaded.scenario_path} --sim stub --time-scale {TIME_SCALE:g} "
           f"--bot {bot} --no-panel --robot-domain {domain + 1}")
    before = set(runs.iterdir())
    code, out, err = run_bounded(["bash", "-c", cmd], env=env, timeout_s=timeout_s)
    log = f"exit {code}\n{out}\n{err}"
    (work / f"log_{bot}.txt").write_text(log)
    new = sorted(set(runs.iterdir()) - before)
    if not new:
        raise RuntimeError(f"{preset_id} --bot {bot}: no run folder was written\n{log[-3000:]}")
    return new[-1], log


# ---- in one process ---------------------------------------------------------------------------------------------------
class _Wire:
    def __init__(self):
        self.msgs = []

    def publish(self, kind, obj):
        self.msgs.append((kind, obj))


class _AgentOut:
    def __init__(self):
        self.items = []

    def publish_intent(self, i):
        self.items.append(("intent", i))

    def publish_say(self, s):
        self.items.append(("say", s))


def run_inproc(preset_id: str, work: Path, *, bot: str, seconds: Optional[float] = None) -> Path:
    """The sim, director, agents and (bot "claude") the built-in robot mind on the scripted model, in lockstep in this
    process. Returns the run folder."""
    from langgraph.checkpoint.memory import MemorySaver

    from wardzero_agents.config.presets import people_meta
    from wardzero_agents.llm.fake import FakeGateway
    from wardzero_agents.mind.runtime import Mind, MindDeps
    from wardzero_agents.robots.claude_robot import RobotMind, decision_outputs
    from wardzero_agents.robots.profile import get_profile, profile_for_sim
    from wardzero_agents.ros.agent_node import TICK_S, AgentService
    from wardzero_agents.ros.director_node import TAIL_STREAMS, DirectorService
    from wardzero_agents.ros.stub_sim_node import StubSimService
    from wardzero_agents.runlog import RunLog, RunTail

    preset = presets()[preset_id]
    loaded = stage(preset_id, work / "staged")
    seconds = preset.duration_min * 60.0 if seconds is None else seconds
    run_dir = work / f"run_{bot}"
    shutil.rmtree(run_dir, ignore_errors=True)
    runlog = RunLog(run_dir)
    runlog.copy_configs(loaded)
    run_id = f"e2e-{preset_id}-{bot}-{uuid.uuid4().hex[:8]}"  # the campaign db debriefs a run id once
    runlog.write_meta({"run_id": run_id, "scenario_id": loaded.scenario.id, "preset_id": preset_id,
                       "seed": loaded.scenario.seed, "test_mode": True, "people": people_meta(loaded, preset),
                       "models": loaded.scenario.models.model_dump(), "scenario_path": str(loaded.scenario_path)})
    sim = StubSimService(loaded, runlog, TIME_SCALE, True, ())
    wire, aout = _Wire(), _AgentOut()
    director = DirectorService(loaded, runlog, wire, run_id)
    tail = RunTail(run_dir, TAIL_STREAMS)  # what the director node reads from the run folder
    gw = FakeGateway()
    clock = {"t": 0.0}

    def mind_factory(inst, effects):
        return Mind(MindDeps(inst=inst, loaded=loaded, gateway=effects.guard(gw), kb=None, ltm=None,
                             run_id=run_id, run_index=1, effects=effects, runlog=runlog,
                             checkpointer=MemorySaver(), clock=lambda: clock["t"]))

    agents = AgentService(loaded, runlog, aout, mind_factory, now=lambda: clock["t"])
    cfg = loaded.scenario.robot
    robot = None
    if bot == "claude":
        robot = RobotMind(loaded.site, sim.robot_brief, FakeGateway(), lambda: clock["t"],
                          profile=profile_for_sim(get_profile(cfg.profile), "stub"),
                          known_places=loaded.scenario.known_places, pose=(cfg.x, cfg.y, cfg.yaw),
                          on_decision=lambda row: runlog.write("robot_decisions", row))

    async def drive():
        last_agent = last_dir = -1.0
        for _ in range(int(seconds / (WALL_STEP * TIME_SCALE))):
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
            if robot is not None:
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
                for stream, rec in tail.poll():
                    director.ingest(stream, rec)
                director.tick(t)
                for kind, obj in wire.msgs:
                    if kind == "nudge":
                        agents.on_nudge(obj)
                        if hasattr(sim, "on_nudge"):  # the sim node hears the director's cards too, when it can
                            sim.on_nudge(obj)
                    elif kind == "reflex":
                        agents.on_reflex(obj)
                    elif kind == "world":
                        sim.on_world_event(obj)
                wire.msgs.clear()
            if t - last_agent >= TICK_S:
                last_agent = t
                await agents.tick(t)
            if robot is not None:
                d = await robot.tick(t)
                if d is not None:
                    o = decision_outputs(d, robot, robot.next_id())
                    if o.say:
                        if getattr(o, "toward", None):
                            sim.on_robot_toward(o.toward)
                        sim.on_robot_say(o.say)
                    if o.action:
                        sim.on_task_action(o.action)
                    if o.goal:
                        sim.on_goal(*o.goal)
                    if getattr(o, "stop", False):
                        sim.on_cmd_vel((0.0, 0.0))  # the robot node sends a zero cmd_vel
            for kind, obj in aout.items:
                (sim.on_intent if kind == "intent" else sim.on_say)(obj)
            aout.items.clear()
            sim.set_pending(agents.pending_count())
            deadline = time.monotonic() + 5.0
            while agents.pending_count() and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            sim.set_pending(0)
        await agents.aclose()

    asyncio.run(drive())
    return run_dir


# ---- judging a run ----------------------------------------------------------------------------------------------------
def rows(run_dir: Path, stream: str) -> list[dict]:
    p = Path(run_dir) / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _leaves(node):
    if not node.get("children"):
        yield node
    for c in node.get("children", []):
        yield from _leaves(c)


@dataclass
class Verdict:
    preset_id: str
    run_dir: Path
    flow_done: bool = False
    expects: list = field(default_factory=list)  # (label, state, reason)
    score: float = 0.0
    passed: bool = False
    criteria: list = field(default_factory=list)  # the evaluated rows
    decision_errors: list = field(default_factory=list)
    contacts: float = 0.0

    @property
    def expects_met(self) -> list:
        return [e for e in self.expects if e[1] == "met"]

    def problems(self) -> list[str]:
        """What an ideal-robot run got wrong, in words."""
        out = []
        if not self.flow_done:
            out.append("the flow did not finish")
        out += [f"expect not met: {lab} ({state}: {why})" for lab, state, why in self.expects if state != "met"]
        if not self.passed:
            out.append(f"score {self.score:.1f} not passed")
            out += [f"criterion {c['id']} ({c['label']}): {c['metric']} = {c['value']} needs {c['op']} "
                    f"{c['threshold']}" for c in self.criteria if c.get("status", "pass" if c["passed"] else "fail")
                    == "fail"]
        out += [f"decision error: {e}" for e in self.decision_errors[:5]]
        if self.contacts:
            out.append(f"{self.contacts:g} contacts")
        return out

    def credited(self) -> list[str]:
        """What a no-robot run was credited with, in words."""
        out = [f"expect met without a robot: {lab}" for lab, state, _ in self.expects if state == "met"]
        if self.passed:
            out.append(f"score {self.score:.1f} passed without a robot")
        return out


def work_data_dir(run_dir: Path) -> Path:
    """The data folder of a run's work folder: <work>/data (the launch path's WARDZERO_DATA_DIR for the child, whose
    run folders sit in <work>/runs), never the user's own data dir."""
    run_dir = Path(run_dir)
    work = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    return work / "data"


def debrief(run_dir: Path, db_path: Optional[Path] = None) -> None:
    """The scripted debrief (as `cli run` does after a run), unless the run already has one. Its trust rows go to the
    campaign database under the run's work folder (db_path to choose another)."""
    if (Path(run_dir) / "debrief.jsonl").exists():
        return
    from wardzero_agents.bench.metrics import RunData
    from wardzero_agents.campaign import debrief as campaign
    from wardzero_agents.config import presets as preset_cfg
    from wardzero_agents.llm.fake import FakeGateway
    from wardzero_agents.memory import memory_path
    db_path = Path(db_path) if db_path is not None else memory_path(work_data_dir(run_dir))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    preset = preset_cfg.preset_for_run(RunData.load(run_dir).meta)
    asyncio.run(campaign.debrief(run_dir, FakeGateway(), preset=preset, db_path=db_path))


def judge(preset_id: str, run_dir: Path) -> Verdict:
    from wardzero_agents.bench.criteria import evaluate
    from wardzero_agents.bench.metrics import RunData, contacts
    run_dir = Path(run_dir)
    v = Verdict(preset_id, run_dir)
    flow = rows(run_dir, "flow")
    if flow:
        leaves = list(_leaves(flow[-1]["tree"]))
        v.flow_done = all(lf.get("state") not in ("pending", "running") for lf in leaves)
        v.expects = [(lf["label"], lf.get("state"), lf.get("reason")) for lf in leaves if lf["kind"] == "expect"]
    data = RunData.load(run_dir)
    res = evaluate(presets()[preset_id], data)
    v.score, v.passed, v.criteria = res.score, res.passed, res.rows
    v.decision_errors = [f"{r['person']} t={r['t']:.0f}: {r['error']}" for r in rows(run_dir, "decisions")
                         if r.get("error")]
    v.decision_errors += [f"robot t={r['t']:.0f}: {r['error']}" for r in rows(run_dir, "robot_decisions")
                          if r.get("error")]
    v.contacts = contacts(data) or 0.0
    return v
