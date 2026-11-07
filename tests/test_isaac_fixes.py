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


def test_leaves_of_doors_that_do_not_swing_are_made_static():
    bodies = {"/World/Fixtures/Doors/Compartment/Leaf0": True, "/World/Fixtures/Doors/Compartment/Leaf1": True,
              "/World/Fixtures/Doors/E_Compartment/Leaf0": False, "/World/Fixtures/Doors/E_Compartment/Base": False,
              "/World/Fixtures/Doors1/E_Stair/Leaf0": False, "/World/Fixtures/Lifts/Lift1/Car": True,
              "/World/Fixtures/DoorsX/Leaf0": True}
    assert fixes.static_leaves(bodies, ["E_Compartment", "E_Stair"]) == [
        "/World/Fixtures/Doors/Compartment/Leaf0", "/World/Fixtures/Doors/Compartment/Leaf1"]
    assert "/World/Fixtures/Doors1/E_Stair/Leaf0" in fixes.static_leaves(bodies, [])


def test_baked_leaf_copies_and_the_junction_panels_leave_the_collision():
    names = ["c02528_World_Fixtures_Doors_Compartment_Leaf0_SignP", "c02600_World_Fixtures_Doors1_E_Stair_Leaf0_SignN",
             "c01698_World_BuildingMods_Partitions_Junction_CompartmentSidePanel_b0",
             "c01701_World_BuildingMods_Partitions_Junction_CompartmentSidePanel_b3", "c00001_World_Hospital_Wall"]
    assert fixes.baked_collision_off(names) == [
        "/World/BakedColliders/c01698_World_BuildingMods_Partitions_Junction_CompartmentSidePanel_b0",
        "/World/BakedColliders/c02528_World_Fixtures_Doors_Compartment_Leaf0_SignP",
        "/World/BakedColliders/c02600_World_Fixtures_Doors1_E_Stair_Leaf0_SignN"]


def test_stage_fixes_returns_the_baked_colliders_and_static_leaves(monkeypatch, v1):
    monkeypatch.setattr(fixes, "read_nav_volumes", lambda st: {})
    monkeypatch.setattr(fixes, "read_door_bodies", lambda st: {"/World/Fixtures/Doors/Other/Leaf0": True,
                                                              "/World/Fixtures/Doors/E_Compartment/Leaf0": False})
    monkeypatch.setattr(fixes, "read_baked_names", lambda p: ["c1_World_Fixtures_Doors_Other_Leaf0_SignP"])
    import sys, types
    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(Stage=types.SimpleNamespace(Open=lambda p: None))
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    off, authored, _ = fixes.stage_fixes(v1.site, CullPlan(), Path("x.usd"), Path("c.usd"), ["E_Compartment"])
    assert off == ["/World/BakedColliders/c1_World_Fixtures_Doors_Other_Leaf0_SignP"]
    assert authored == {"/World/Fixtures/Doors/Other/Leaf0": {"physics:rigidBodyEnabled": False}}


# ---- a NavMesh where the stage's include volume does not reach (the theatre wing) ---------------------------
def test_kept_rooms_outside_the_include_volume_get_an_include_of_their_own(v1):
    # scrub_nurse keeps Scrub and Theatre, south of the hospital's only include slab (y >= -4.8): with nothing else the
    # bake made no NavMesh, the Behavior System was off and the surgeon stood in a T-pose (QA bug 12)
    plan = cull_plan(v1, rooms=["Scrub", "Theatre"])
    box = fixes.nav_include_box(VOLUMES, v1.site, plan)
    assert box is not None
    (x0, y0), (x1, y1) = box
    for rid in ("Scrub", "Theatre"):
        r = room_rect(v1.site.rooms[rid])
        assert x0 <= r[0] and y0 <= r[1] and x1 >= r[2] and y1 >= r[3]
    assert y1 < -4.0  # only the wing: not a second slab over the hospital


def test_kept_rooms_inside_the_include_volume_need_nothing(v1):
    plan = cull_plan(v1, rooms=["Ward_Corridor", "B1", "Lobby"])
    assert fixes.nav_include_box(VOLUMES, v1.site, plan) is None


def test_the_full_site_covers_the_wing_too(v1):
    box = fixes.nav_include_box(VOLUMES, v1.site, CullPlan())
    assert box is not None and box[0][1] <= room_rect(v1.site.rooms["Theatre"])[1]


def test_without_any_include_volume_every_kept_room_is_covered(v1):
    plan = cull_plan(v1, rooms=["Lobby"])
    box = fixes.nav_include_box({}, v1.site, plan)
    r = room_rect(v1.site.rooms["Lobby"])
    assert box[0][0] <= r[0] and box[1][1] >= r[3]


def test_stage_fixes_returns_the_include_box(monkeypatch, v1):
    monkeypatch.setattr(fixes, "read_nav_volumes", lambda st: dict(VOLUMES))
    monkeypatch.setattr(fixes, "read_door_bodies", lambda st: {})
    monkeypatch.setattr(fixes, "read_baked_names", lambda p: [])
    import sys, types
    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(Stage=types.SimpleNamespace(Open=lambda p: None))
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    plan = cull_plan(v1, rooms=["Theatre"])
    off, authored, include = fixes.stage_fixes(v1.site, plan, Path("x.usd"), Path("c.usd"), [])
    assert include == fixes.nav_include_box(VOLUMES, v1.site, plan) and include is not None
