"""Data tests for the committed whole-hospital site map (config/site)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "config" / "site"


@pytest.fixture(scope="module")
def site():
    return json.loads((SITE / "hospital.json").read_text())


@pytest.fixture(scope="module")
def occ():
    return np.load(SITE / "hospital_occ.npz")


def inside(poly, x, y):
    ok = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            ok = not ok
    return ok


def test_site_json_has_core_rooms(site):
    ids = {r["id"] for r in site["rooms"]}
    for need in ["Lobby", "Ward_Entrance", "Ward_Corridor", "AE_Waiting", "Theatre",
                 "Main_LiftLobby", "B1", "B6"]:
        assert need in ids


def test_beds_inside_their_rooms(site):
    rooms = {r["id"]: r for r in site["rooms"]}
    assert {b["id"] for b in site["beds"]} == {"B1", "B2", "B3", "B4", "B5", "B6", "AE_Isolation"}
    for b in site["beds"]:
        assert inside(rooms[b["room"]]["poly"], b["x"], b["y"]), b["id"]
        # head end is 1 m or so along the yaw from the centre
        assert math.hypot(b["head"][0] - b["x"], b["head"][1] - b["y"]) > 0.9


def test_bed_head_directions(site):
    beds = {b["id"]: b for b in site["beds"]}
    for bid, sign in [("B1", 1), ("B2", -1), ("B3", 1), ("B4", -1), ("B5", 1), ("B6", -1)]:
        assert (beds[bid]["head"][1] - beds[bid]["y"]) * sign > 0.9


def test_seats_match_waiting_area_yaml(site):
    want = yaml.safe_load((ROOT / "config" / "waiting_area.yaml").read_text())
    got = {s["id"]: s for s in site["seats"]}
    assert set(got) == {s["id"] for s in want["seats"]}
    for s in want["seats"]:
        assert abs(got[s["id"]]["x"] - s["x"]) < 1e-6
        assert abs(got[s["id"]]["y"] - s["y"]) < 1e-6
    for k, v in want["landmarks"].items():
        assert abs(site["landmarks"][k]["x"] - v["x"]) < 1e-6
        assert abs(site["landmarks"][k]["y"] - v["y"]) < 1e-6


def test_occ_grid_shape_and_res(occ):
    assert float(occ["res"]) == pytest.approx(0.2)
    assert np.allclose(occ["origin"], [-49.44, -25.64], atol=0.01)
    assert occ["occ"].dtype == np.uint8
    assert set(np.unique(occ["occ"])) <= {0, 1}


def cell(occ, x, y):
    res = float(occ["res"])
    ox, oy = occ["origin"]
    return int(math.floor((y - oy) / res)), int(math.floor((x - ox) / res))


def test_landmarks_not_blocked(site, occ):
    for name, lm in site["landmarks"].items():
        iy, ix = cell(occ, lm["x"], lm["y"])
        assert occ["occ"][iy, ix] == 0, name


def test_every_room_has_a_centre_landmark(site):
    for r in site["rooms"]:
        assert f"{r['id']}_centre" in site["landmarks"]
    for need in ["porter_point", "nurse_base_ward"]:
        assert need in site["landmarks"]


def test_navmesh_hints_reference_real_rooms(site):
    hints = json.loads((SITE / "navmesh_hints.json").read_text())
    ids = {r["id"] for r in site["rooms"]}
    assert set(hints) == {"reception_guide", "scrub_nurse", "meal_round", "ward_fire", "evacuation_guide",
                          "specimen_run", "visiting_rush", "wandering_patient", "night_call_bell", "ae_waiting"}
    for h in hints.values():
        assert h["focus_rooms"]
        assert set(h["focus_rooms"]) | set(h["route_rooms"]) <= ids


def test_main_places_reachable(site, occ):
    """Entrance reaches the ward, theatre, A&E waiting and the porter point over free cells."""
    g = occ["occ"]
    start = cell(occ, *[site["landmarks"]["entrance"][k] for k in ("x", "y")])
    seen = {start}
    todo = [start]
    while todo:
        y, x = todo.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (y + dy, x + dx)
            if 0 <= n[0] < g.shape[0] and 0 <= n[1] < g.shape[1] and n not in seen and g[n] == 0:
                seen.add(n)
                todo.append(n)
    for name in ["porter_point", "nurse_base_ward", "Theatre_centre", "AE_Waiting_centre", "B1_centre",
                 "B6_centre", "Ward_Corridor_centre"]:
        lm = site["landmarks"][name]
        assert cell(occ, lm["x"], lm["y"]) in seen, name


# ---- Site model ----
from wardzero_agents.site import load_site  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_site(ROOT / "config")


def test_room_at_and_resolve(model):
    assert model.room_at(*model.resolve("bed:B1")) == "B1"
    assert model.resolve("reception_g1") == (1.05, 1.62)
    assert model.room_at(-500, 0) is None


def test_room_at_smallest_overlap(model):
    a, b = model.rooms["Ward_Corridor"], model.rooms["Ward_Entrance"]
    xs = [p[0] for r in (a, b) for p in r.poly]
    ys = [p[1] for r in (a, b) for p in r.poly]
    found = None
    for i in range(int((max(xs) - min(xs)) / 0.1)):
        for j in range(int((max(ys) - min(ys)) / 0.1)):
            x, y = min(xs) + i * 0.1, min(ys) + j * 0.1
            if a.contains(x, y) and b.contains(x, y):
                found = (x, y)
                break
        if found:
            break
    assert found, "rooms do not overlap"
    small = min((a, b), key=lambda r: r.area)
    assert model.room_at(*found) == small.id


def test_resolve_prefixes_and_order(model):
    seat = model.seats["B2"]
    assert model.resolve("B2") == (seat.x, seat.y)
    assert model.resolve("seat:B2") == (seat.x, seat.y)
    assert model.resolve("landmark:entrance") == model.resolve("entrance")
    assert model.resolve("room:B2") != model.resolve("seat:B2")
    assert model.room_at(*model.resolve("room:B2")) == "B2"
    bed = model.beds["B2"]
    assert model.resolve("bed:B2") == bed.side_point("right")
    fid = next(iter(model.furniture))
    f = model.furniture[fid]
    assert model.resolve(f"furniture:{fid}") == (f.x, f.y)
    assert model.resolve("nope") is None and model.resolve("room:nope") is None


def test_bed_side_point(model):
    b = model.beds["B1"]
    l, r = b.side_point("left"), b.side_point("right")
    assert math.hypot(l[0] - b.x, l[1] - b.y) == pytest.approx(0.6)
    assert math.hypot(r[0] - b.x, r[1] - b.y) == pytest.approx(0.6)
    assert l != r


def test_place_rejects_wall(model):
    g = model.grid
    iy, ix = [int(v[0]) for v in np.nonzero(g.occ == 1)]
    x, y = g.to_world(iy, ix)
    assert model.check_place(x, y) == "inside a wall"
    assert model.check_place(-500, 0) == "outside the building"
    assert model.check_place(*model.resolve("entrance")) is None
    bed = model.beds["B3"]
    assert model.check_place(bed.x, bed.y, occupied_beds=["B3"]) == "bed B3 already has a patient"


def test_summary_for_prompt(model):
    s = model.summary_for_prompt(["B1", "Lobby"])
    assert "B1" in s and "Lobby" in s


def test_check_place_rejects_sealed_pocket(model):
    g = model.grid
    pocket = [(int(a), int(b)) for a, b in np.argwhere((g.occ == 0) & ~model.main)]
    assert pocket, "map has no sealed pocket"
    x, y = g.to_world(*pocket[0])
    assert model.check_place(x, y) == "not reachable from the entrance"
