"""Which hospital areas the Isaac host keeps: rooms from navmesh_hints.json, placements, overrides."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.isaac_host.stage import CullPlan, cull_plan, intersects, room_rect, select_off

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


@pytest.fixture(scope="module")
def v1():
    return load_scenario(V1)


def with_scenario(loaded, **update):
    import dataclasses
    return dataclasses.replace(loaded, scenario=loaded.scenario.model_copy(update=update))


def test_preset_rooms_come_from_the_hints(v1):
    plan = cull_plan(with_scenario(v1, preset_id="visiting_rush"))
    hints = json.loads((v1.root / "site" / "navmesh_hints.json").read_text())["visiting_rush"]
    assert set(hints["focus_rooms"]) | set(hints["route_rooms"]) <= set(plan.rooms)
    assert not plan.full and plan.source == "hints"
    assert plan.rects["Ward_Corridor"] == room_rect(v1.site.rooms["Ward_Corridor"])


def test_placements_add_their_rooms(v1):
    # waiting_area_v1 has no preset: its seats and the robot at reception are all in the lobby
    plan = cull_plan(v1)
    assert "Lobby" in plan.rooms and plan.source == "placements"
    assert not plan.full
    # a ward preset still keeps the lobby the cast sits in
    ward = cull_plan(with_scenario(v1, preset_id="meal_round"))
    assert {"B1", "Ward_Corridor", "Lobby"} <= set(ward.rooms)


def test_room_rect_is_the_polygon_bounds(v1):
    r = room_rect(v1.site.rooms["Lobby"])
    assert r == pytest.approx((-5.64, -2.85, 6.75, 5.4))


def test_explicit_rooms_override_everything(v1):
    plan = cull_plan(with_scenario(v1, preset_id="visiting_rush"), rooms=["B1", "B2"])
    assert plan.rooms == ["B1", "B2"] and set(plan.rects) == {"B1", "B2"} and plan.source == "rooms"
    with pytest.raises(ValueError, match="Nowhere"):
        cull_plan(v1, rooms=["Nowhere"])


def test_full_site_keeps_everything(v1):
    plan = cull_plan(with_scenario(v1, preset_id="visiting_rush"), full_site=True)
    assert plan.full and plan.rects == {} and plan.source == "full"


def test_nothing_to_go_on_means_the_full_site(v1):
    empty = with_scenario(v1, cast=[], robot=v1.scenario.robot.model_copy(update={"enabled": False}))
    empty = __import__("dataclasses").replace(empty, instances=[])
    plan = cull_plan(empty)
    assert plan.full and plan.source == "full"


def test_a_placement_outside_every_room_gets_a_small_box(v1):
    outside = (-45.0, -20.0)  # the car park, in no room
    assert not any(r.contains(*outside) for r in v1.site.rooms.values())
    robot = v1.scenario.robot.model_copy(update={"x": outside[0], "y": outside[1]})
    plan = cull_plan(with_scenario(v1, robot=robot))
    (key,) = [k for k in plan.rects if k.startswith("near_")]
    x0, y0, x1, y1 = plan.rects[key]
    assert x0 < outside[0] < x1 and y0 < outside[1] < y1 and x1 - x0 <= 6.0


def test_intersects_uses_the_margin():
    rects = {"a": (0.0, 0.0, 10.0, 10.0)}
    assert intersects(((10.2, 2.0), (12.0, 3.0)), rects, margin=0.3)
    assert not intersects(((10.5, 2.0), (12.0, 3.0)), rects, margin=0.3)
    assert intersects(((-5.0, -5.0), (20.0, 20.0)), rects)  # a box around the whole rect


def test_select_off_culls_boxes_outside_and_keeps_inside_and_unbounded():
    rects = {"a": (0.0, 0.0, 10.0, 10.0)}
    boxes = {"/World/H/in": ((1.0, 1.0), (2.0, 2.0)), "/World/H/out": ((50.0, 50.0), (51.0, 51.0)),
             "/World/H/edge": ((9.9, 5.0), (30.0, 6.0))}
    assert select_off(boxes, rects) == ["/World/H/out"]


def test_cull_plan_is_a_plain_record(v1):
    plan = cull_plan(v1)
    assert isinstance(plan, CullPlan) and plan.rooms == sorted(plan.rooms)
