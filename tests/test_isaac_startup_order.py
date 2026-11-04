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


def test_robots_are_added_after_play_and_nothing_else_does_it():
    assert A.STARTUP.index("add_robots") > A.STARTUP.index("play")
    import inspect
    assert "add_robots" not in inspect.getsource(A.IsaacHost.run)  # run() has no order of its own


def test_demos_use_the_same_sequence():
    import inspect
    from wardzero_agents.isaac_host import robot_demo
    src = inspect.getsource(robot_demo.boot)
    assert "host.boot_sequence(" in src and "host.add_robots()" not in src and "host.play()" not in src
