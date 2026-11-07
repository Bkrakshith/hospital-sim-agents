"""The robot's router (hard band) on the hospital: it takes the corridor-to-bay link that people take in
evacuation_guide instead of a route twice as long, keeps its body clear of walls and furniture doing so, and a
route_blocked seal keeps its routes out of the blocked way."""
import math

import pytest

from wardzero_agents.bench.metrics import NEAR_FURNITURE_M, _near_blocked
from wardzero_agents.config.presets import DEFAULT_CONFIG_ROOT, load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import WorldEvent
from wardzero_agents.site.routing import Router
from wardzero_agents.studio import auto_staff
from wardzero_agents.world.stub_world import StubWorld

BAY = (-29.14, 6.66)  # Ambulance handover bay (QA evidence_routes.txt)


@pytest.fixture(scope="module")
def evac(tmp_path_factory):
    p = load_scenario_presets(DEFAULT_CONFIG_ROOT)["evacuation_guide"]
    ld = preset_to_scenario(p, DEFAULT_CONFIG_ROOT, tmp_path_factory.mktemp("evac"),
                            characters=auto_staff.with_auto(list(p.characters), p))
    return ld


def length(r):
    return sum(math.dist(a, b) for a, b in zip(r, r[1:]))


def samples(route, step=0.05):
    for (ax, ay), (bx, by) in zip(route, route[1:]):
        n = max(1, int(math.dist((ax, ay), (bx, by)) / step))
        for k in range(n + 1):
            yield ax + (bx - ax) * k / n, ay + (by - ay) * k / n


@pytest.mark.parametrize("a", [(-10.2, 2.7), (-9.0, 0.0), (1.0, -0.9)])
def test_robot_guide_routes_to_the_bay_are_as_short_as_peoples(evac, a):
    w = StubWorld(evac, 7)
    people = w.router.route(a, BAY)
    robot = w.robot.router.route(a, BAY)
    assert people is not None and robot is not None
    assert length(robot) <= 1.15 * length(people) + 1.0, (length(robot), length(people))


@pytest.mark.parametrize("a", [(-10.2, 2.7), (1.0, -0.9)])
def test_the_link_route_keeps_the_body_off_walls_and_furniture(evac, a):
    """Past its first and last metre the route stays out of the robot_near_furniture_s distance (0.35 m) and the
    body (radius 0.3 m) never overlaps a blocked cell."""
    w = StubWorld(evac, 7)
    g = w.site.grid
    route = w.robot.router.route(a, BAY)
    for x, y in samples(route):
        if math.dist((x, y), route[0]) < 1.0 or math.dist((x, y), route[-1]) < 1.0:
            continue
        assert not _near_blocked(g, x, y, NEAR_FURNITURE_M - 1e-6), (x, y)


def test_where_the_hard_band_route_is_reasonable_it_is_still_used(evac):
    """The wider berth (hard band) stays the first choice: a route the hard band can take without a long detour is
    the hard band's route."""
    w = StubWorld(evac, 7)
    rr = w.robot.router
    a, b = (2.5, -0.5), (19.0, 15.0)  # the lobby to the ward
    hard_only = Router(w.site.grid, hard=rr.hard, tight=0.0)
    assert hard_only._tight is None
    ca, cb = rr._snap(rr.grid.to_cell(*a)), rr._snap(rr.grid.to_cell(*b))
    assert hard_only._strict(ca, cb, ()) is not None  # the hard band has a route here
    assert rr.route(a, b) == hard_only.route(a, b)


def test_where_the_hard_band_has_no_route_the_tight_band_comes_before_the_soft_one(evac):
    """From the bay out to A&E the hard band has no route at all; the soft band's route (the old fallback) hugs the
    walls, the tight band's keeps the body off them."""
    w = StubWorld(evac, 7)
    g = w.site.grid
    route = w.robot.router.route((-26.2, 6.6), (-10.0, 0.0))
    assert route is not None
    for x, y in samples(route):
        if math.dist((x, y), route[0]) < 1.0 or math.dist((x, y), route[-1]) < 1.0:
            continue
        assert not _near_blocked(g, x, y, NEAR_FURNITURE_M - 1e-6), (x, y)


def test_a_route_blocked_seal_keeps_the_robot_out_of_the_blocked_way(evac):
    """evacuation_guide's three seals across the main entrance (and the fire there) reach the robot's router: a route
    from outside the entrance to the bay that went through the Lobby before the seals no longer passes within the
    sealed radius of a block or through the fire, and the guide from A&E to the bay stays out of the Lobby."""
    w = StubWorld(evac, 7)
    blocks = [(-4.2, -2.6), (-1.2, -2.6), (1.8, -2.6)]
    fire = (-1.24, -2.3)
    outside = (0.0, -6.0)  # in front of the main entrance: its unsealed route to the bay crosses the Lobby

    def near_a_block(route):
        return min(math.dist(p, b) for p in samples(route) for b in blocks)

    before = w.robot.router.route(outside, BAY)
    assert before is not None and near_a_block(before) < 1.0  # the check below is not met by chance
    w.apply_world_event(WorldEvent(id="f", t=60.0, kind="fire_start", x=fire[0], y=fire[1]))
    for i, (x, y) in enumerate(blocks):
        w.apply_world_event(WorldEvent(id=f"b{i}", t=60.0, kind="route_blocked", x=x, y=y, text="x"))
    w.step(60.1)
    rr = w.robot.router
    g = rr.grid
    for b in blocks:  # the seal is in the robot's own router, not only the people's
        iy, ix = g.to_cell(*b)
        assert rr._raw[iy][ix], b
    after = rr.route(outside, BAY)
    assert after != before
    starts = [(-10.2, 2.7), (1.0, -0.9), (2.5, -0.5), (-2.0, 1.0)] + ([outside] if after is not None else [])
    for a in starts:
        route = rr.route(a, BAY)
        assert route is not None, a
        for x, y in samples(route):
            for bx, by in blocks:
                assert math.dist((x, y), (bx, by)) > 1.5 - 0.15, (a, (x, y))
            assert math.dist((x, y), fire) > 1.5, (a, (x, y))
    guide = rr.route((-10.2, 2.7), BAY)  # the A&E-to-bay guide QA flagged
    assert all(w.site.room_at(x, y) != "Lobby" for x, y in samples(guide))


def test_a_wider_berth_router_gets_no_tight_band_unless_asked(evac):
    """The tight band (0.35 m) is for the stub robot's body (radius 0.3 m, clearance 0.3 m). A router built with a
    wider clearance (the walking G1's 0.6 m: its half-width with arms is about 0.35 m) keeps its wider berth: no tight
    band, so it does not take the 1 m corridor-to-bay link a few centimetres off the walls."""
    w = StubWorld(evac, 7)
    g = w.site.grid
    g1 = Router(g, clearance=0.6, hard=0.5)
    assert g1._tight is None
    route = g1.route((-10.2, 2.7), BAY)
    assert route is not None
    for x, y in samples(route):
        if math.dist((x, y), route[0]) < 1.0 or math.dist((x, y), route[-1]) < 1.0:
            continue
        assert not _near_blocked(g, x, y, 0.45), (x, y)
    asked = Router(g, clearance=0.6, hard=0.5, tight=0.35)  # still there for a caller that wants it
    assert asked._tight is not None
    assert w.robot.router._tight is not None  # the stub robot's router keeps it by default
