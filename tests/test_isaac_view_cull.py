"""The rooms in the robot camera's view stay in the stage: a wedge from the camera, tested against room rectangles."""
import math
from pathlib import Path

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.isaac_host import stage as st

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
loaded = load_scenario(V1)
SITE = loaded.site


def test_wedge_is_a_fan_from_the_camera():
    poly = st.view_wedge(0.0, 0.0, 0.0, 90.0, 10.0)
    assert poly[0] == (0.0, 0.0)
    far = [p for p in poly[1:]]
    assert all(math.hypot(*p) == __import__("pytest").approx(10.0) for p in far)
    assert min(p[1] for p in far) < -7.0 and max(p[1] for p in far) > 7.0  # 45 degrees either side of +x


def test_rect_wedge_intersection():
    wedge = st.view_wedge(0.0, 0.0, 0.0, 90.0, 10.0)
    assert st.wedge_hits_rect(wedge, (4.0, -1.0, 6.0, 1.0))  # dead ahead
    assert not st.wedge_hits_rect(wedge, (-6.0, -1.0, -4.0, 1.0))  # behind
    assert not st.wedge_hits_rect(wedge, (1.0, 6.0, 3.0, 8.0))  # off to the side, outside 45 degrees
    assert st.wedge_hits_rect(wedge, (9.0, 4.0, 11.0, 6.0))  # inside the arc at its edge
    assert not st.wedge_hits_rect(wedge, (11.0, -1.0, 13.0, 1.0))  # beyond the range


def test_rooms_in_view_of_the_reception_g1():
    rooms = st.rooms_in_view(SITE, (1.05, 1.62, -math.pi / 2), 90.0, 15.0)
    assert rooms == ["Lobby"]  # PreOp and the wing corridor lie south in the wedge, behind the lobby's wall
    assert "Lobby" in st.rooms_in_view(SITE, (1.05, 1.62, 0.0), 90.0, 15.0)  # facing east: still the room it stands in


def test_a_wall_hides_the_room_behind_it_but_a_doorway_does_not():
    import numpy as np
    from types import SimpleNamespace

    occ = np.zeros((40, 40), dtype=np.uint8)
    occ[:, 20] = 1  # a wall at x = 20 * 0.5 = 10 m (origin 0)
    from wardzero_agents.site.model import OccGrid
    grid = OccGrid(occ, 0.5, (0.0, 0.0))

    class Room:
        def __init__(self, x0, y0, x1, y1):
            self.poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            self.r = (x0, y0, x1, y1)

        def contains(self, x, y):
            return self.r[0] <= x <= self.r[2] and self.r[1] <= y <= self.r[3]

    site = SimpleNamespace(rooms={"here": Room(0, 0, 8, 8), "behind": Room(11, 2, 15, 6)}, grid=grid)
    assert st.rooms_in_view(site, (4.0, 4.0, 0.0), 90.0, 15.0) == ["here"]
    occ[7:12, 20] = 0  # a doorway
    assert st.rooms_in_view(site, (4.0, 4.0, 0.0), 90.0, 15.0) == ["behind", "here"]


def test_the_outdoors_in_view_is_kept_for_the_exterior_only():
    r = loaded.scenario.robot
    plan = st.cull_plan(loaded, rooms=["Lobby"])
    (name, (x0, y0, x1, y1)), = plan.exterior_rects.items()
    assert name == "robot_view" and y0 < r.y - 5.0 and x0 < r.x < x1  # the wedge south of the G1
    assert plan.rects.keys() == {"Lobby"}  # the building is not extended by it
    assert st.cull_plan(loaded, rooms=["Lobby"], robot_view=False).exterior_rects == {}


def test_exterior_prims_in_view_are_not_culled():
    boxes = {"/World/Exterior/Ground/a": ((0.0, -9.0), (4.0, -6.0)), "/World/Hospital/p": ((0.0, -9.0), (4.0, -6.0))}
    plan = st.CullPlan(["Lobby"], {"Lobby": (-5.6, -2.9, 6.8, 5.4)}, "rooms", {"robot_view": (-3.0, -12.0, 5.0, 2.0)})
    outdoors = {p: b for p, b in boxes.items() if p.startswith("/World/Exterior/")}
    assert st.select_off(outdoors, {**plan.rects, **plan.exterior_rects}) == []
    assert st.select_off({"/World/Hospital/p": boxes["/World/Hospital/p"]}, plan.rects) == ["/World/Hospital/p"]


def test_plan_gains_the_rooms_the_robot_sees_unless_full_site():
    base = st.cull_plan(loaded, rooms=["Lobby"], robot_view=False)
    assert base.rooms == ["Lobby"]
    plan = st.cull_plan(loaded, rooms=["Lobby"])
    r = loaded.scenario.robot
    seen = st.rooms_in_view(SITE, (r.x, r.y, r.yaw), st.ROBOT_VIEW_FOV_DEG, st.ROBOT_VIEW_RANGE_M)
    assert set(plan.rooms) == {"Lobby", *seen} and set(plan.rects) >= set(plan.rooms)
    assert st.cull_plan(loaded, full_site=True).full
    no_robot = loaded.scenario.model_copy(update={"robot": r.model_copy(update={"enabled": False})})
    import dataclasses
    assert st.cull_plan(dataclasses.replace(loaded, scenario=no_robot), rooms=["Lobby"]).rooms == ["Lobby"]
