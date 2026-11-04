"""Bring-your-own-robot tooling: URDF check, Gazebo sensors, sensor catalogue, profiles, import script."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wardzero_agents.robots import sensors as S
from wardzero_agents.robots.profile import G1_CLAUDE, RobotProfile, load_profile, save_profile
from wardzero_agents.robots.urdf import parse_urdf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data" / "urdf"
CATALOGUE = ROOT / "config" / "sensors" / "catalogue.yaml"


def test_parse_simple_urdf():
    info = parse_urdf(DATA / "simple")
    assert info.robot_name == "simple_bot"
    assert info.links == ["base_link", "torso", "arm", "head"]
    assert [j["name"] for j in info.joints] == ["waist", "shoulder", "neck"]
    assert info.dof == 2
    assert info.mass_kg == pytest.approx(16.5)
    assert info.height_m_est == pytest.approx(1.1)
    assert info.joints[0]["limits"] == {"lower": -1.5, "upper": 1.5, "effort": 50.0, "velocity": 2.0}
    assert info.joints[0]["parent"] == "base_link" and info.joints[0]["child"] == "torso"
    assert not [i for i in info.issues if "mesh" in i or "xacro" in i]
    assert any("Gazebo" in i for i in info.issues)  # no sensors in the file: say so


def test_gazebo_sensors_extracted():
    info = parse_urdf(DATA / "gazebo_sensors")
    by = {s.id: s for s in info.gazebo_sensors}
    assert set(by) == {"front_depth", "scan"}
    d, lid = by["front_depth"], by["scan"]
    assert (d.kind, d.link) == ("rgbd", "cam_link") and d.xyz == (0.02, 0.0, 0.0) and d.rpy == (0.0, 0.1, 0.0)
    assert d.params["fov_h"] == pytest.approx(1.5) and d.params["resolution"] == [640, 480]
    assert d.params["range"] == [0.2, 8.0]
    assert (lid.kind, lid.link) == ("lidar", "lidar_link") and lid.params["range"] == [0.1, 25.0]
    assert lid.params["samples"] == 720
    assert any("ghost_link" in i for i in info.issues)  # sensor on a link that does not exist
    assert not any("no Gazebo sensors" in i for i in info.issues)


def test_urdf_issues_reported(tmp_path):
    info = parse_urdf(DATA / "missing_mesh")
    assert "mesh not found: package://other_pkg/meshes/gone.stl" in info.issues
    assert "mesh not found: meshes/nothere.dae" in info.issues
    assert not any("mesh_pkg" in i for i in info.issues)  # package://mesh_pkg resolved by searching the folder
    assert info.dof == 1  # continuous joint

    x = parse_urdf(DATA / "xacro")
    assert "xacro found: run xacro to produce a .urdf first" in x.issues
    assert x.links == []

    for bad in (tmp_path / "nope", tmp_path):  # missing folder, empty folder
        assert parse_urdf(bad).issues
    (tmp_path / "broken.urdf").write_text("<robot><link name='a'>")
    assert any("XML" in i for i in parse_urdf(tmp_path).issues)
    (tmp_path / "broken.urdf").write_text(
        '<!DOCTYPE r [<!ENTITY a "x">]><robot name="&a;"><link name="a"/></robot>')
    assert parse_urdf(tmp_path).issues  # entity expansion refused, no exception
    (tmp_path / "broken.urdf").write_text('<robot name="r"><joint name="j" type="revolute"><limit lower="x"/></joint></robot>')
    assert parse_urdf(tmp_path).issues


def test_catalogue_entries_valid():
    cat = yaml.safe_load(CATALOGUE.read_text())["sensors"]
    assert len(cat) > 40
    ids = [e["id"] for e in cat]
    assert len(ids) == len(set(ids))
    for e in cat:
        assert e["kind"] in ("rgbd", "rgb", "lidar"), e["id"]
        assert e["usd"].endswith((".usd", ".usda", ".usdc")) and not e["usd"].startswith("/"), e["id"]
        assert e.get("fov_h") or e.get("range_max"), e["id"]
    by = {e["id"]: e for e in cat}
    for must in ("realsense_d455", "orbbec_femto_mega", "stereolabs_zed_x", "ouster_os1", "hesai_xt32"):
        assert must in by, must
    assert by["realsense_d455"]["usd"] == "Isaac/Sensors/RealSense/D455/rsd455.usd"
    assert by["realsense_d455"]["source"] == "datasheet"
    assert S.get_catalogue_entry("ouster_os1")["kind"] == "lidar"


def test_catalogue_builder_on_fake_root(tmp_path):
    base = tmp_path / "Isaac" / "Sensors"
    for rel in ("Ouster/OS1/OS1.usd", "Ouster/OS1/OS1_REV6_32ch10hz512res.usda", "Foo/cam/foo_cam.usd",
                "Foo/materials/materials.usd", "Foo/.thumbs/x.usd", "Foo/cam/foo_cam_config.usd"):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text("#usda 1.0\n")
    ids = {e["id"] for e in S.build_catalogue(tmp_path)}
    assert ids == {"ouster_os1", "foo_cam"}


def test_profile_roundtrip(tmp_path):
    p = RobotProfile(name="my_bot", can_move=True, base="wheeled", radius=0.4, reach=(0.2, 1.1), max_speed=0.8,
                     ns="my_bot", sensors=(S.SensorSpec("cam", "rgbd", "realsense_d455", "head", (0.1, 0, 0.2),
                                                         (0, 0, 0), {"fov_h": 87}),),
                     urdf=tmp_path / "desc" / "bot.urdf", controller="external")
    path = save_profile(p, tmp_path)
    assert path == tmp_path / "robots" / "my_bot" / "robot.yaml"
    q = load_profile("my_bot", tmp_path)
    assert q == p and q.sensors[0].xyz == (0.1, 0.0, 0.2)
    save_profile(G1_CLAUDE, tmp_path)
    assert load_profile("g1_claude", tmp_path) == G1_CLAUDE
    assert "RealSense D455" in G1_CLAUDE.abilities()
    assert G1_CLAUDE.controller == "builtin_claude" and G1_CLAUDE.urdf is None
    with pytest.raises(KeyError):
        load_profile("missing", tmp_path)
    with pytest.raises(ValueError):
        save_profile(RobotProfile(**{**G1_CLAUDE.__dict__, "name": "../evil"}), tmp_path)


def test_sensors_yaml_roundtrip(tmp_path):
    specs = [S.SensorSpec("a", "lidar", "ouster_os1", "base", (0, 0, 1), (0, 0, 0), {"range": [0.1, 50]}),
             S.SensorSpec("mic", "mic", None, "head", (0, 0, 0), (0, 0, 0), {})]
    f = S.write_sensors_yaml(tmp_path, specs)
    assert f.name == "wardzero_sensors.yaml"
    assert S.read_sensors_yaml(tmp_path) == specs


def test_import_script_dry_run(tmp_path):
    script = ROOT / "scripts" / "isaac_import_urdf.py"
    S.write_sensors_yaml(tmp_path, [S.SensorSpec("cam", "rgbd", "realsense_d455", "nowhere_link", (0, 0, 0), (0, 0, 0), {})])
    base = [sys.executable, str(script), "--dry-run", "--sensors", str(tmp_path / "wardzero_sensors.yaml"),
            "--out", str(tmp_path / "robot.usd")]
    r = subprocess.run(base + ["--urdf", str(DATA / "simple" / "simple.urdf")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ok"] is True and out["joints"] == 2 and out["dry_run"] is True
    assert any("nowhere_link" in w for w in out["warnings"])  # sensor on a link the URDF lacks
    r = subprocess.run(base + ["--urdf", str(tmp_path / "none.urdf")], capture_output=True, text=True)
    assert r.returncode == 1 and json.loads(r.stdout.strip().splitlines()[-1])["ok"] is False
