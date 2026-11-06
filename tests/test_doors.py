"""Door physics, the pure parts: which doors swing, the closed pose, the closer's damping switch, the leaf geometry, the
capsules that follow people and the slow zone."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from wardzero_agents.isaac_host import doors as D

ROOT = Path(__file__).resolve().parents[1]


def test_spike_values():
    assert (D.MASS_KG, D.LIMIT_DEG, D.STIFFNESS, D.DAMPING_OPEN, D.DAMPING_CLOSE, D.MAX_SPEED) == \
        (40.0, 100.0, 18.2, 5.0, 30.0, 3.0)
    assert D.per_degree(18.2) == pytest.approx(0.3176, abs=1e-4)  # USD drive gains are per degree
    assert D.per_degree(30.0) == pytest.approx(0.5236, abs=1e-4)
    assert D.STIFFNESS * math.pi / 2 / 0.96 == pytest.approx(29.8, abs=0.1)  # ~30 N at the edge holds 90 deg


def test_closed_yaw_from_the_drawn_open_leaf():
    assert D.closed_yaw(0.0, -90.0) == 90.0  # E_Compartment Leaf0 (spike: closed yaw +90)
    assert D.closed_yaw(0.0, 90.0) == -90.0  # ... Leaf1 (-90)
    assert D.closed_yaw(90.0, 90.0) == 0.0  # Main_LiftLobby Leaf0: drawn rotateZ 90 at state 90
    assert D.closed_yaw(90.0, -90.0) == -180.0


def test_the_closer_is_light_opening_and_firm_closing():
    assert D.closer_damping(0.5, 1.0) == D.DAMPING_OPEN
    assert D.closer_damping(-0.5, -1.0) == D.DAMPING_OPEN
    assert D.closer_damping(0.5, -1.0) == D.DAMPING_CLOSE
    assert D.closer_damping(0.0, 0.0) == D.DAMPING_CLOSE


def test_doors_on_the_routes_of_reception_guide():
    from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
    from wardzero_agents.isaac_host.stage import cull_plan
    import tempfile
    loaded = preset_to_scenario(load_scenario_presets(ROOT / "config")["reception_guide"], (ROOT / "config").resolve(),
                                Path(tempfile.mkdtemp()))
    doors = D.route_doors(loaded.site, cull_plan(loaded).rects)
    assert {"E_Compartment", "Main_LiftLobby"} <= set(doors)
    assert len(D.route_doors(loaded.site, None)) == len(loaded.site.doors)  # full site: every door
    far = {"x": (100.0, 100.0, 101.0, 101.0)}
    assert D.route_doors(loaded.site, far) == []


def test_leaf_rectangle_distance_follows_the_swing():
    leaf = D.Leaf("E_Compartment", "Leaf0", (11.21, 6.30), 90.0)  # closed: along +y from the hinge
    assert leaf.dist(0.0, 11.21, 6.78) == pytest.approx(-0.027, abs=1e-3)  # on the leaf's mid-line: inside
    assert leaf.dist(0.0, 10.21, 6.78) == pytest.approx(1.0 - 0.027, abs=1e-3)  # 1 m west of the closed leaf
    swung = math.radians(90)  # swung west (positive) into the corridor: along -x now
    assert leaf.dist(swung, 10.73, 6.30) < 0
    assert leaf.dist(swung, 11.21, 6.78) > 0.4


def test_baked_prims_left_out():
    names = ["c1_World_Fixtures_Doors_E_Compartment_Leaf1_SignN", "c2_World_Fixtures_Doors_E_Compartment_Base_x",
             "c3_World_BuildingMods_Partitions_Junction_CompartmentSidePanel_b1", "c4_World_Walls_W1"]
    assert D.baked_off(names) == [names[0], names[2]]


def test_capsules_follow_people_and_lead_by_their_speed():
    tr = D.CapsuleTracker(3)
    tr.frame({"P1": (1.0, 2.0), "P2": None}, 0.04)
    assert tr.slot == {"P1": 0, "P2": 1}
    assert tr.targets(4, 0.005)[0] == (1.0, 2.0) and tr.targets(4, 0.005)[1] == D.PARK
    tr.frame({"P1": (1.04, 2.0), "P2": None}, 0.04)  # 1 m/s east, EMA 0.5 -> 0.5 m/s
    x, y = tr.targets(8, 0.005)[0]
    assert x == pytest.approx(1.04 + 0.5 * 0.04) and y == pytest.approx(2.0)
    tr.frame({"P1": (9.0, 2.0)}, 0.04)  # a teleport is not a speed
    assert tr.vel[0] == pytest.approx((0.25, 0.0))
    tr.frame({"P3": (0, 0), "P4": (1, 1)}, 0.04)
    assert "P4" in tr.overflow and tr.slot["P3"] == 2


def test_slow_within_a_metre_of_a_leaf():
    leaf = D.Leaf("E_Compartment", "Leaf0", (11.21, 6.30), 90.0)
    assert D.near_leaf([leaf], [0.0], 10.4, 6.8)
    assert not D.near_leaf([leaf], [0.0], 9.9, 6.8)


def test_people_slow_near_a_leaf_and_speed_up_after():
    from wardzero_agents.isaac_host.people_isaac import IraAgent
    calls = []
    agent = SimpleNamespace(set_speed=lambda v: calls.append(round(v, 2)), get_speed=lambda: 1.5)
    api = IraAgent(agent, "/World/Characters/P1/x", carb_module=SimpleNamespace(Float3=lambda *a: a))
    api.set_speed(1.3)
    api.set_door_slow(True)
    api.set_speed(1.1)  # a new phase while near the door: still capped
    api.set_door_slow(False)
    assert calls == [1.3, 0.8, 0.8, 1.1]
