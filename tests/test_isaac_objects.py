"""World objects to stage prims, with a fake stage (no Isaac): where an item is drawn, when it is hidden, how a fire's
size sets the emitter, when a call light is lit, and that only changes are written."""
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.isaac_host import objects as ob
from wardzero_agents.ros.sim_service import SimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.world.objects import Item

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


class FakeStage:
    """Records what the mirror asks of the stage."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.fires: dict[str, dict] = {}
        self.bells: dict[str, dict] = {}
        self.calls: list[tuple] = []

    def add_item(self, key, kind, label):
        self.calls.append(("add_item", key, kind, label))
        self.items[key] = {"kind": kind, "label": label, "pose": None, "visible": None}

    def place_item(self, key, x, y, z, yaw):
        self.calls.append(("place_item", key))
        self.items[key]["pose"] = (x, y, z, yaw)

    def show_item(self, key, visible):
        self.calls.append(("show_item", key, visible))
        self.items[key]["visible"] = visible

    def add_bell(self, key, x, y, z, yaw):
        self.calls.append(("add_bell", key))
        self.bells[key] = {"pos": (x, y, z), "lit": None}

    def light_bell(self, key, lit):
        self.calls.append(("light_bell", key, lit))
        self.bells[key]["lit"] = lit

    def set_fire(self, key, x, y, params):
        self.calls.append(("set_fire", key, params))
        self.fires[key] = {"pos": (x, y), "params": params}

    def remove_fire(self, key):
        self.calls.append(("remove_fire", key))
        self.fires.pop(key, None)


@pytest.fixture
def world(tmp_path):
    import dataclasses
    loaded = load_scenario(V1)
    scen = loaded.scenario.model_copy(update={"objects": [
        {"kind": "tray", "id": "tray_1", "label": "meal", "owner": "P1", "holder": "g1", "x": 1.0, "y": 2.0, "z": 1.0},
        {"kind": "extinguisher", "id": "ext_1", "label": "wall extinguisher", "x": -3.0, "y": 1.0, "z": 0.9},
        {"kind": "specimen_box", "id": "spec_1", "x": -2.0, "y": 0.5, "z": 0.8},
        {"kind": "instrument", "id": "ins_1", "label": "scissors", "x": -2.5, "y": 0.5, "z": 0.8},
        {"kind": "call_bell", "bed": "B1"},
    ]})
    loaded = dataclasses.replace(loaded, scenario=scen)
    return SimService(loaded, RunLog(tmp_path), 1.0, False, []).world


def test_items_are_added_once_with_their_kind(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    m.sync()
    m.sync()
    assert [c for c in fs.calls if c[0] == "add_item"] == [("add_item", "tray_1", "tray", "meal"),
                                                           ("add_item", "ext_1", "extinguisher", "wall extinguisher"),
                                                           ("add_item", "spec_1", "specimen_box", ""),
                                                           ("add_item", "ins_1", "instrument", "scissors")]


def test_an_item_on_the_floor_is_drawn_at_its_state(world):
    fs = FakeStage()
    ob.ObjectMirror(world, fs).sync()
    x, y, z, _ = fs.items["spec_1"]["pose"]
    assert (x, y, z) == pytest.approx((-2.0, 0.5, 0.8)) and fs.items["spec_1"]["visible"] is True


def test_an_extinguisher_stands_on_the_floor(world):
    fs = FakeStage()
    ob.ObjectMirror(world, fs).sync()
    assert fs.items["ext_1"]["pose"][2] == 0.0  # its z is the grab height, not where the body sits


def test_an_item_the_robot_holds_follows_the_robot_at_hand_height(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    world.robot.x, world.robot.y, world.robot.yaw = 5.0, 6.0, math.pi / 2
    m.sync()
    x, y, z, yaw = fs.items["tray_1"]["pose"]
    assert (x, y) == pytest.approx((5.0, 6.0 + ob.CARRY_AHEAD_M), abs=1e-6) and z == pytest.approx(1.0)
    assert yaw == pytest.approx(math.pi / 2)
    world.robot.x = 8.0  # the robot walks on: the tray goes with it
    m.sync()
    assert fs.items["tray_1"]["pose"][0] == pytest.approx(8.0)


def test_an_item_a_walking_person_holds_follows_them(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    p = next(q for q in world.people.values() if q.posture == "seated")
    item = world.objects.items["tray_1"]
    item.holder = p.inst.person_id
    item.z = 0.2  # a low value: the hands are still at carrying height
    m.sync()
    x, y, z, _ = fs.items["tray_1"]["pose"]
    assert math.hypot(x - p.x, y - p.y) == pytest.approx(ob.CARRY_AHEAD_M, abs=1e-6)
    assert z == pytest.approx(ob.HANDS_MIN_Z_M)


def test_an_item_handed_to_a_patient_in_bed_stays_where_it_was_put(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    p = next(q for q in world.people.values())
    p.posture, p.bed = "lying", "B1"
    item = world.objects.items["tray_1"]
    item.holder, item.x, item.y, item.z = p.inst.person_id, 25.0, 27.0, 1.0
    m.sync()
    assert fs.items["tray_1"]["pose"][:3] == pytest.approx((25.0, 27.0, 1.0))


def test_an_item_held_by_someone_who_left_is_hidden(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    p = next(q for q in world.people.values())
    world.objects.items["tray_1"].holder = p.inst.person_id
    m.sync()
    assert fs.items["tray_1"]["visible"] is True
    p.posture = "gone"
    m.sync()
    assert fs.items["tray_1"]["visible"] is False


def test_only_changes_are_written(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    m.sync()
    n = len(fs.calls)
    m.sync()
    assert len(fs.calls) == n  # nothing moved: nothing is written
    world.objects.items["spec_1"].x = 3.0
    m.sync()
    assert [c for c in fs.calls[n:]] == [("place_item", "spec_1")]


def test_fire_size_sets_the_emitter(world):
    small, big = ob.fire_params(0.1), ob.fire_params(1.0)
    assert small.enabled and big.enabled
    assert small.radius < big.radius and small.fuel < big.fuel
    assert ob.fire_params(0.0).enabled is False


def test_a_fire_is_started_scaled_and_removed_at_zero(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    f = world.objects.start_fire(19.0, 14.5, size=0.1, grow_per_s=0.0)
    m.sync()
    assert fs.fires[f.id]["pos"] == (19.0, 14.5) and fs.fires[f.id]["params"] == ob.fire_params(0.1)
    f.size = 0.6
    m.sync()
    assert fs.fires[f.id]["params"] == ob.fire_params(0.6)
    f.reduce(1.0)  # out
    m.sync()
    assert f.id not in fs.fires and ("remove_fire", f.id) in fs.calls
    n = len(fs.calls)
    m.sync()
    assert len(fs.calls) == n  # removed once


def test_a_small_change_in_fire_size_is_not_rewritten(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    f = world.objects.start_fire(19.0, 14.5, size=0.30, grow_per_s=0.0)
    m.sync()
    n = len(fs.calls)
    f.size = 0.31
    m.sync()
    assert len(fs.calls) == n
    f.size = 0.40
    m.sync()
    assert len(fs.calls) == n + 1


def test_call_lights_follow_the_bell_state(world):
    fs = FakeStage()
    m = ob.ObjectMirror(world, fs)
    m.sync()
    assert fs.bells["B1"]["lit"] is False and fs.bells["B1"]["pos"][2] == pytest.approx(ob.BELL_Z_M)
    world.objects.ring("B1", 5.0)
    m.sync()
    assert fs.bells["B1"]["lit"] is True
    world.objects.bells["B1"].active = False
    m.sync()
    assert fs.bells["B1"]["lit"] is False


def test_a_bell_light_sits_on_the_wall_above_the_head_of_the_bed(world):
    fs = FakeStage()
    ob.ObjectMirror(world, fs).sync()
    bed = world.site.beds["B1"]
    bx, by, _ = fs.bells["B1"]["pos"]
    hx, hy = ob.bed_head(bed)
    assert math.hypot(bx - hx, by - hy) < 0.3
    assert math.hypot(hx - bed.x, hy - bed.y) == pytest.approx(ob.BED_HALF_LENGTH_M)


def test_the_asset_of_an_instrument_follows_its_label():
    assert ob.instrument_asset("Mayo scissors") == "scissors"
    assert ob.instrument_asset("needle driver") == "needle_driver"
    assert ob.instrument_asset("Adson forceps") == "adson_forceps"
    assert ob.instrument_asset("scalpel") == "scalpel_handle"
    assert ob.instrument_asset("") in ob.INSTRUMENT_ASSETS and ob.instrument_asset("mystery thing") in ob.INSTRUMENT_ASSETS


def test_the_bodies_mirror_objects_once_a_frame_after_the_poses(world):
    from wardzero_agents.isaac_host.bodies import IsaacBodies

    seen = []

    class Spy:
        def update(self):
            seen.append((world.robot.x, world.people["P1"].x if "P1" in world.people else None))

    bodies = IsaacBodies(robot=lambda: (7.0, 1.0, 0.0))
    bodies.objects = Spy()
    bodies.step(world, 0.1)
    assert seen and seen[0][0] == 7.0  # the robot's Isaac pose was already in the world when the objects were drawn


def test_the_host_takes_a_fire_mode():
    from wardzero_agents.isaac_host import app

    base = ["--run-dir", "/tmp/x", "--scenario", "s.yaml"]
    assert app.parse_args(base).fire == "flow"
    assert app.parse_args(base + ["--fire", "proxy"]).fire == "proxy"
    with pytest.raises(SystemExit):
        app.parse_args(base + ["--fire", "lava"])


def test_a_patient_propped_up_in_bed_sits_up_in_the_drawing(world):
    from wardzero_agents.isaac_host.bodies import IsaacBodies

    class Prop:
        def __init__(self):
            self.calls = []

        def set_sitting_up(self, v):
            self.calls.append(v)

    p = next(q for q in world.people.values())
    p.posture, p.bed = "lying", "B1"
    prop = Prop()
    bodies = IsaacBodies()
    bodies.props[p.inst.person_id] = prop
    bodies.step(world, 0.1)
    p.sitting_up = True
    bodies.step(world, 0.2)
    assert prop.calls == [False, True]


def test_one_failing_item_does_not_stop_the_others(world, caplog):
    class Flaky(FakeStage):
        def add_item(self, key, kind, label):
            if key == "tray_1":
                raise RuntimeError("no such asset")
            super().add_item(key, kind, label)

    fs = Flaky()
    m = ob.ObjectMirror(world, fs)
    with caplog.at_level("ERROR"):
        m.sync()
        m.sync()
    assert "tray_1" not in fs.items and {"ext_1", "spec_1", "ins_1"} <= set(fs.items)
    assert fs.bells["B1"]["lit"] is False  # the bell is still drawn
    assert sum("tray_1" in r.getMessage() for r in caplog.records) == 1  # logged once, with the exception


def test_a_failing_fire_is_skipped_and_logged_once_and_other_fires_still_draw(world, caplog):
    class Flaky(FakeStage):
        def set_fire(self, key, x, y, params):
            if key == "fire_1":
                raise RuntimeError("flow broke")
            super().set_fire(key, x, y, params)

    fs = Flaky()
    m = ob.ObjectMirror(world, fs)
    a = world.objects.start_fire(1.0, 1.0, size=0.2, grow_per_s=0.0)
    b = world.objects.start_fire(5.0, 5.0, size=0.2, grow_per_s=0.0)
    with caplog.at_level("ERROR"):
        m.sync()
        a.size = 0.5
        m.sync()
    assert b.id in fs.fires and a.id not in fs.fires
    assert sum("fire_1" in r.getMessage() for r in caplog.records) == 1
