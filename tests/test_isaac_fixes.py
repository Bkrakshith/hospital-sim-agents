"""Stage fixes of the Isaac host: the NavMesh exclude volumes over the bedrooms, and a door whose leaves are posed with a rotate op."""
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.isaac_host import fixes
from wardzero_agents.isaac_host.stage import CullPlan, cull_plan, room_rect

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
NAV = "/World/Hospital/Navmesh"


@pytest.fixture(scope="module")
def v1():
    return load_scenario(V1)


# the real stage's volumes (from hospital_g1.usd): one include slab and the old exclusions
VOLUMES = {
    f"{NAV}/NavMeshVolume": ("Include", ((-48.8, -4.8), (26.2, 37.2))),
    f"{NAV}/NavMeshVolume_01": ("Exclude", ((20.7, 21.35), (26.7, 31.35))),  # B1 and the nurse base WS1
    f"{NAV}/NavMeshVolume_02": ("Exclude", ((20.7, 3.5), (26.7, 17.7))),  # B3, B4, B5, WS2
    f"{NAV}/NavMeshVolume_03": ("Exclude", ((12.1, 10.6), (17.1, 25.2))),  # B6
    f"{NAV}/NavMeshVolume_04": ("Exclude", ((-23.4, 12.1), (3.6, 15.9))),  # behind the lobby
}


def test_exclude_volumes_over_bedrooms_are_switched_off(v1):
    plan = cull_plan(v1, rooms=["Ward_Corridor", "B1", "B2", "B3", "B4", "B5", "B6", "Ward_WS1", "Ward_WS2"])
    assert fixes.nav_volumes_off(VOLUMES, v1.site, plan) == [f"{NAV}/NavMeshVolume_01", f"{NAV}/NavMeshVolume_02",
                                                           f"{NAV}/NavMeshVolume_03"]


def test_volumes_over_rooms_that_are_not_kept_stay(v1):
    plan = cull_plan(v1, rooms=["Ward_Corridor", "B6"])
    assert fixes.nav_volumes_off(VOLUMES, v1.site, plan) == [f"{NAV}/NavMeshVolume_03"]


def test_the_lobby_alone_changes_nothing(v1):
    assert fixes.nav_volumes_off(VOLUMES, v1.site, cull_plan(v1, rooms=["Lobby"])) == []


def test_the_full_site_frees_every_bedroom(v1):
    off = fixes.nav_volumes_off(VOLUMES, v1.site, CullPlan())
    assert f"{NAV}/NavMeshVolume_02" in off and f"{NAV}/NavMeshVolume_01" in off
    assert f"{NAV}/NavMeshVolume" not in off  # an include volume is never touched


def test_an_exclude_that_only_touches_a_wall_stays(v1):
    b1 = room_rect(v1.site.rooms["B1"])
    sliver = {f"{NAV}/NavMeshVolume_x": ("Exclude", ((b1[2] - 0.05, b1[1]), (b1[2] + 3, b1[3])))}
    assert fixes.nav_volumes_off(sliver, v1.site, CullPlan()) == []


def test_door_leaves_posed_with_a_rotate_op_are_made_static():
    bodies = {"/World/Fixtures/Doors/Compartment/Leaf0": True,  # rotateZ on a body the joint expects unrotated: NaN pose
              "/World/Fixtures/Doors/Compartment/Leaf1": True,
              "/World/Fixtures/Doors/E_Compartment/Leaf0": False,
              "/World/Fixtures/Doors/E_Compartment/Base": False}
    assert fixes.rotated_leaves(bodies) == ["/World/Fixtures/Doors/Compartment/Leaf0", "/World/Fixtures/Doors/Compartment/Leaf1"]


def test_only_door_leaves_are_touched():
    bodies = {"/World/Fixtures/Doors/Compartment/Leaf0": True, "/World/Fixtures/Lifts/Lift1/Car": True,
              "/World/Fixtures/Calls/MainCall/fx_button": True, "/World/Fixtures/DoorsX/Leaf0": True}
    assert fixes.rotated_leaves(bodies) == ["/World/Fixtures/Doors/Compartment/Leaf0"]


def test_more_than_the_known_pair_is_a_warning(caplog, monkeypatch, v1):
    monkeypatch.setattr(fixes, "read_nav_volumes", lambda st: {})
    monkeypatch.setattr(fixes, "read_door_bodies", lambda st: {"/World/Fixtures/Doors/Other/Leaf0": True,
                                                              "/World/Fixtures/Doors/Compartment/Leaf0": True})
    import sys, types
    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(Stage=types.SimpleNamespace(Open=lambda p: None))
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    with caplog.at_level("WARNING"):
        fixes.stage_fixes(v1.site, CullPlan(), Path("x.usd"))
    assert any("Other/Leaf0" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)
