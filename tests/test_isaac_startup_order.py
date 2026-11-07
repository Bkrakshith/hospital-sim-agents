"""The host's start-up order is the one thing that keeps a moving robot cheap next to the Behavior System (S1b):
people, NavMesh bake, the robot's obstacle cylinder, play(), and only then the robot. One sequence, used by run() and the demos."""
import argparse
from types import SimpleNamespace

from wardzero_agents.isaac_host import app as A


def host_with_recorder():
    calls = []
    h = A.IsaacHost.__new__(A.IsaacHost)
    h.robot_usd = None
    for name in ("prepare_robot", "start_app", *A.STARTUP):
        setattr(h, name, lambda n=name: calls.append(n))
    return h, calls


def test_startup_runs_the_binding_order():
    h, calls = host_with_recorder()
    h.boot_sequence()
    assert calls[:2] == ["prepare_robot", "start_app"]
    steps = calls[2:]
    for a, b in (("spawn_people", "bake_navmesh"), ("spawn_objects", "bake_navmesh"), ("bake_navmesh", "robot_proxy"),
                 ("robot_proxy", "play"), ("play", "bind_people"), ("bind_people", "add_robots")):
        assert steps.index(a) < steps.index(b), (a, b)
    assert steps[-1] == "add_robots" and steps[0] == "enable_extensions"  # extensions before any stage


def test_robots_are_added_after_play():
    assert A.STARTUP.index("add_robots") > A.STARTUP.index("play")


def test_demos_boot_through_the_host_sequence():
    from wardzero_agents.isaac_host import robot_demo

    calls = []
    host = SimpleNamespace(boot_sequence=lambda order: calls.append(tuple(order)), robot=None)
    host.__dict__.update(stage=None)
    import types
    fake_app = types.SimpleNamespace(IsaacHost=lambda *a: host, STARTUP=A.STARTUP)
    args = argparse.Namespace(scenario=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"),
                              rooms=None, full_site=False)
    orig = robot_demo.app_mod
    robot_demo.app_mod = fake_app
    try:
        robot_demo.boot(args, SimpleNamespace(order_control=False))
    finally:
        robot_demo.app_mod = orig
    assert calls == [A.STARTUP]  # exactly the host's own order, nothing re-implemented


def test_no_update_between_the_robot_reference_and_its_graph():
    """S1b: PhysX must not parse the robot half authored, so the reference, pose, camera and graph go in one pass."""
    from wardzero_agents.isaac_host.robot_isaac import IsaacRobot, SETTLE_UPDATES

    log = []

    class Fake(IsaacRobot):
        def __init__(self):
            self.app = SimpleNamespace(update=lambda: log.append("update"))
            self.cfg = SimpleNamespace(x=0, y=0, yaw=0)
            self.profile = SimpleNamespace(can_move=False, name="g1", ns="robot")
            self.downgraded, self.art_root, self.camera_path = False, "/r", None
            self.graph = SimpleNamespace(nodes=[])

        def _check_ready(self):
            log.append("check")

        def _reference(self):
            log.append("reference")

        def _locate(self):
            log.append("locate")

        def _build_graph(self):
            log.append("graph")

        def _articulation(self):
            log.append("articulation")

    Fake().add()
    assert log[:4] == ["check", "reference", "locate", "graph"]  # nothing steps the app in between
    assert log[4:] == ["update"] * SETTLE_UPDATES + ["articulation"]


# ---- every character gets a Behavior agent, or the run says so (QA bug 12) -----------------------------------
def _bind_host(tmp_path, skelroots, unbound):
    from wardzero_agents.runlog import RunLog

    h = A.IsaacHost.__new__(A.IsaacHost)
    h.startup, h._t0 = {}, 0.0
    h.args = SimpleNamespace(fps=25)
    h.runlog = RunLog(tmp_path)
    h.bodies = SimpleNamespace(controllers={})
    h.svc = SimpleNamespace(world=None)
    h.app = None

    def bind(app, bodies, world, fps=30):
        h.people.unbound = list(unbound)

    h.people = SimpleNamespace(skelroots=dict.fromkeys(skelroots, "/World/x"), bind=bind, unbound=[])
    return h


def _host_rows(tmp_path):
    import json
    return [json.loads(line) for line in (tmp_path / "host.jsonl").read_text().splitlines()] \
        if (tmp_path / "host.jsonl").exists() else []


def test_people_bound_equals_the_cast_with_bodies(tmp_path, caplog):
    h = _bind_host(tmp_path, ["N1", "S1"], [])
    h.bind_people()
    assert h.startup["people_expected"] == 2 and h.startup["people_agents"] == 2
    assert not [r for r in _host_rows(tmp_path) if r.get("kind") == "problem"]


def test_people_without_an_agent_are_a_problem_in_the_run_record(tmp_path, caplog):
    import logging
    h = _bind_host(tmp_path, ["N1", "S1"], ["N1", "S1"])
    with caplog.at_level(logging.ERROR, logger="wardzero.isaac.app"):
        h.bind_people()
    assert h.startup["people_agents"] == 0 and h.startup["people_expected"] == 2
    rows = [r for r in _host_rows(tmp_path) if r.get("kind") == "problem"]
    assert rows and rows[0]["what"] == "people_not_animated" and rows[0]["missing"] == ["N1", "S1"]
    assert any("0 of 2" in r.getMessage() and "N1, S1" in r.getMessage() for r in caplog.records)
