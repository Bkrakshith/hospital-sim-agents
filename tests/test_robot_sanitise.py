"""The rules that make a converted robot USD ready to reference after play(): what is floored, what is stripped from
a mounted sensor asset, and the marker the host checks. Pure functions; the importer applies them with pxr."""
import pytest

from wardzero_agents.robots import sanitise as sn


def test_constants_are_the_s1_values():
    assert sn.ARMATURE == 0.01 and sn.MASS_FLOOR_KG == 0.005 and sn.INERTIA_FLOOR == 1e-6
    assert (sn.SOLVER_POSITION_ITERATIONS, sn.SOLVER_VELOCITY_ITERATIONS) == (16, 2)


def test_mass_and_inertia_floors():
    assert sn.floored_mass(None) == 0.005 and sn.floored_mass(0.0) == 0.005 and sn.floored_mass(0.004) == 0.005
    assert sn.floored_mass(0.2) is None  # no change needed
    assert sn.floored_inertia(None) == (1e-6, 1e-6, 1e-6)
    assert sn.floored_inertia((0.0, 2e-3, 5e-7)) == (1e-6, 2e-3, 1e-6)
    assert sn.floored_inertia((1e-3, 2e-3, 3e-3)) is None


def edits(**kw):
    base = dict(type_name="Xform", name="x", is_mesh=False, has_rigid_body=False, has_mass=False, has_collision=False,
                kind="rgbd", hide_housing=None)
    return sn.sensor_prim_edits(**{**base, **kw})


def test_a_sensor_assets_rigid_body_is_stripped_and_its_colliders_off():
    assert edits(has_rigid_body=True, has_mass=True) == ["strip_rigid_body", "strip_mass"]
    assert edits(has_mass=True) == ["strip_mass"]
    assert edits(has_collision=True) == ["disable_collision"]


def test_camera_housing_is_hidden_unless_the_sensor_says_otherwise():
    assert edits(is_mesh=True) == ["hide"]  # rgbd: the housing covers the lenses inside a head shell
    assert edits(is_mesh=True, kind="rgb") == ["hide"]
    assert edits(is_mesh=True, kind="lidar") == []
    assert edits(is_mesh=True, kind="lidar", hide_housing=True) == ["hide"]
    assert edits(is_mesh=True, hide_housing=False) == []


def test_the_assets_imu_and_depth_template_are_deactivated():
    assert edits(type_name="IsaacImuSensor") == ["deactivate"]
    assert edits(name="TemplateRenderProducts") == ["deactivate"]


def test_ready_marker_roundtrip():
    data = sn.ready_layer_data()
    assert sn.is_ready(data)
    assert not sn.is_ready({}) and not sn.is_ready(None)
    assert not sn.is_ready({**data, sn.READY_KEY: sn.SANITISE_VERSION - 1})  # an older sanitiser: convert again
    with pytest.raises(ValueError, match="convert"):
        sn.require_ready({}, "robot.usd")
    sn.require_ready(data, "robot.usd")
