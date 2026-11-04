#!/usr/bin/env python3
"""URDF folder + wardzero_sensors.yaml -> robot.usd, using Isaac Sim's own URDF importer.

    ~/isaacsim/python.sh scripts/isaac_import_urdf.py --urdf PATH.urdf --sensors wardzero_sensors.yaml --out robot.usd
    python scripts/isaac_import_urdf.py --dry-run ...     # no Isaac: validates the inputs

Prints one JSON object on the last line: {ok, usd, joints, warnings, ...}. Exit code 0 when ok.
Under Isaac's python only pyyaml is used, so this file reads the sensors and catalogue YAML itself.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CATALOGUE = REPO / "config" / "sensors" / "catalogue.yaml"


def asset_root() -> Path:
    env = os.environ.get("WARDZERO_ISAAC_ASSETS")
    return Path(env).expanduser() if env else Path.home() / "isaacsim_assets" / "Assets" / "Isaac" / "6.0"


def load_sensors(path: Path) -> list:
    import yaml
    return (yaml.safe_load(Path(path).read_text()) or {}).get("sensors", [])


def load_catalogue() -> dict:
    import yaml
    return {e["id"]: e for e in yaml.safe_load(CATALOGUE.read_text())["sensors"]}


def finish(result: dict) -> None:
    print(json.dumps(result))
    sys.stdout.flush()
    os._exit(0 if result.get("ok") else 1)  # Kit can hang on interpreter shutdown


def check_inputs(urdf: Path, sensors_yaml: Path, out: Path, link_names=None):
    """Shared by dry-run and the real import: (errors, warnings, sensors, joint_count)."""
    errors, warnings, sensors, joints = [], [], [], 0
    if not urdf.is_file():
        errors.append(f"URDF not found: {urdf}")
    else:
        sys.path.insert(0, str(REPO))
        try:
            from wardzero_agents.robots.urdf import parse_urdf
            info = parse_urdf(urdf.parent)
            # the folder's first urdf may not be the one asked for; parse the asked-for one when it differs
            if info.urdf_path != urdf:
                import tempfile
                import shutil
                with tempfile.TemporaryDirectory() as t:
                    shutil.copy(urdf, Path(t) / urdf.name)
                    info = parse_urdf(Path(t))
            joints = sum(1 for j in info.joints if j["type"] in ("revolute", "continuous", "prismatic"))
            warnings += [i for i in info.issues if not i.startswith("no Gazebo sensors")]
            link_names = set(info.links)
        except ImportError:  # no wardzero deps in this python: fall back to a plain read
            import xml.etree.ElementTree as ET
            root = ET.parse(urdf).getroot()
            link_names = {l.get("name") for l in root.findall("link")}
            joints = sum(1 for j in root.findall("joint") if j.get("type") in ("revolute", "continuous", "prismatic"))
    if sensors_yaml and Path(sensors_yaml).is_file():
        try:
            sensors = load_sensors(sensors_yaml)
        except Exception as e:
            errors.append(f"sensors YAML unreadable: {e}")
        cat = load_catalogue()
        for s in sensors:
            if s.get("model"):
                e = cat.get(s["model"])
                if e is None:
                    errors.append(f"sensor {s.get('id')}: unknown model {s['model']!r}")
                elif not (asset_root() / e["usd"]).exists():
                    warnings.append(f"sensor {s.get('id')}: asset {e['usd']} not found under {asset_root()}")
            if link_names is not None and s.get("kind") not in ("mic", "speaker") and s.get("link") not in link_names:
                warnings.append(f"sensor {s.get('id')}: link {s.get('link')!r} is not in the URDF")
    elif sensors_yaml:
        errors.append(f"sensors YAML not found: {sensors_yaml}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(out.parent, os.W_OK):
            errors.append(f"output folder not writable: {out.parent}")
    except OSError as e:
        errors.append(f"output folder not usable: {e}")
    return errors, warnings, sensors, joints


# ------------------------------------------------------------------ the Isaac part
def import_robot(args, sensors: list, warnings: list) -> dict:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    for _ in range(5):
        app.update()
    import shutil
    import tempfile

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    urdf = Path(args.urdf).resolve()
    pkgs = [{"name": p, "path": str(urdf.parent.parent)}
            for p in sorted(set(re.findall(r"package://([^/]+)/", urdf.read_text(errors="ignore"))))]
    # package://pkg/... : the folder named pkg, searched near the URDF, else its parent
    resolved = []
    for p in pkgs:
        hit = next((d for d in [urdf.parent, urdf.parent.parent, *urdf.parent.rglob(p["name"])]
                    if d.name == p["name"] and d.is_dir()), None)
        resolved.append({"name": p["name"], "path": str(hit.parent) if hit else p["path"]})
    work = tempfile.mkdtemp(prefix="wz_urdf_")
    conf = URDFImporterConfig(
        urdf_path=str(urdf), usd_path=work, merge_fixed_joints=False, merge_mesh=False, collision_from_visuals=False,
        allow_self_collision=False, ros_package_paths=resolved, fix_base=not args.floating_base,
        joint_drive_type="force", joint_target_type="position",
        override_joint_stiffness=args.stiffness, override_joint_damping=args.damping, run_asset_transformer=False)
    stage = Usd.Stage.Open(URDFImporter(conf).import_urdf())

    n_joints, n_floor = 0, 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):  # near-massless sensor frames import with ~0 mass
            ma = UsdPhysics.MassAPI.Apply(p)
            m = ma.GetMassAttr().Get()
            if not m or m < 0.005:
                ma.CreateMassAttr(0.005)
                n_floor += 1
            d = ma.GetDiagonalInertiaAttr().Get()
            if d is None or min(d) < 1e-6:
                ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[max(1e-6, v) for v in (d or (0, 0, 0))]))
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
            n_joints += 1
            # armature (reflected rotor inertia): without it the articulation goes NaN within ~5 frames
            PhysxSchema.PhysxJointAPI.Apply(p).CreateArmatureAttr(args.armature)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            pa = PhysxSchema.PhysxArticulationAPI.Apply(p)
            pa.CreateSolverPositionIterationCountAttr(16)
            pa.CreateSolverVelocityIterationCountAttr(2)
    if n_floor:
        warnings.append(f"{n_floor} links raised to the 5 g mass floor")

    links = {}
    for p in stage.Traverse():
        links.setdefault(p.GetName(), p)
    cat = load_catalogue()
    placed = 0
    for s in sensors:
        link = links.get(s.get("link"))
        if link is None:
            warnings.append(f"sensor {s.get('id')} skipped: link {s.get('link')!r} not in the imported USD")
            continue
        try:
            add_sensor(stage, link, s, cat, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, warnings)
            placed += 1
        except Exception as e:  # one bad sensor must not lose the robot
            warnings.append(f"sensor {s.get('id')} failed: {type(e).__name__}: {e}")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(out))
    shutil.rmtree(work, ignore_errors=True)
    return {"ok": True, "usd": str(out), "joints": n_joints, "sensors_placed": placed, "warnings": warnings}


def add_sensor(stage, link, s, cat, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, warnings):
    import math
    kind, sid = s["kind"], re.sub(r"\W", "_", s["id"])
    path = link.GetPath().AppendChild(f"wz_{sid}")
    xyz, rpy = s.get("xyz") or [0, 0, 0], s.get("rpy") or [0, 0, 0]
    r, p, y = (math.degrees(v) for v in rpy)
    local = Gf.Matrix4d(1)
    for axis, deg in ((Gf.Vec3d(1, 0, 0), r), (Gf.Vec3d(0, 1, 0), p), (Gf.Vec3d(0, 0, 1), y)):
        local = local * Gf.Matrix4d().SetRotate(Gf.Rotation(axis, deg))  # row-vector convention: X then Y then Z
    local.SetTranslateOnly(Gf.Vec3d(*xyz))
    prim = stage.DefinePrim(path, "Xform")
    UsdGeom.Xformable(prim).AddTransformOp().Set(local)
    params = s.get("params") or {}
    prim.CreateAttribute("wardzero:kind", __import__("pxr").Sdf.ValueTypeNames.String).Set(kind)
    if kind in ("mic", "speaker"):
        return
    entry = cat.get(s.get("model")) if s.get("model") else None
    if entry:
        usd = Path(os.environ.get("WARDZERO_ISAAC_ASSETS") or asset_root()) / entry["usd"]
        child = stage.DefinePrim(path.AppendChild("model"), "Xform")
        child.GetReferences().AddReference(str(usd))
        for q in Usd.PrimRange(child):  # sensor assets are rigid bodies with their own IMU and depth template
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, PhysxSchema.PhysxRigidBodyAPI):
                    q.RemoveAPI(api)
            if q.HasAPI(UsdPhysics.MassAPI):
                q.RemoveAPI(UsdPhysics.MassAPI)
            if q.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
            if params.get("hide_housing", kind in ("rgbd", "rgb")) and q.IsA(UsdGeom.Mesh):
                UsdGeom.Imageable(q).MakeInvisible()  # the housing covers the lenses inside a robot's head shell
            if q.GetTypeName() == "IsaacImuSensor" or q.GetName() == "TemplateRenderProducts":
                q.SetActive(False)
        return
    if kind in ("rgb", "rgbd"):
        res = params.get("resolution") or [640, 480]
        fov_h = params.get("fov_h", math.radians(87))
        fov_h = fov_h if fov_h < 6.3 else math.radians(fov_h)  # URDF gives radians, the catalogue degrees
        cam = UsdGeom.Camera.Define(stage, path.AppendChild("camera"))
        ap = 3.896
        cam.CreateHorizontalApertureAttr(ap)
        cam.CreateVerticalApertureAttr(ap * res[1] / res[0])
        cam.CreateFocalLengthAttr(ap / 2 / math.tan(fov_h / 2))
        rng = params.get("range") or [0.05, 100.0]
        cam.CreateClippingRangeAttr(Gf.Vec2f(*rng))
        # link frame: x forward, z up -> USD camera looks down -Z with +Y up
        cam.AddTransformOp().Set(Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1))
    elif kind == "lidar":
        import omni.kit.commands
        ok, _ = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="rtx_lidar", parent=str(path),
                                          config="Example_Rotary", translation=(0, 0, 0))
        if not ok:
            warnings.append(f"sensor {s['id']}: RTX lidar creation failed")
    elif kind == "imu":
        import omni.kit.commands
        omni.kit.commands.execute("IsaacSensorCreateImuSensor", path="imu", parent=str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--sensors", default=None, help="wardzero_sensors.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate the inputs without Isaac")
    ap.add_argument("--floating-base", action="store_true", help="default is a fixed base (anchored to the world)")
    ap.add_argument("--armature", type=float, default=0.01)
    ap.add_argument("--stiffness", type=float, default=150.0)
    ap.add_argument("--damping", type=float, default=8.0)
    args = ap.parse_args(argv)
    errors, warnings, sensors, joints = check_inputs(Path(args.urdf), args.sensors, Path(args.out))
    if errors:
        finish({"ok": False, "usd": None, "joints": joints, "warnings": warnings, "errors": errors,
                "dry_run": args.dry_run})
    if args.dry_run:
        finish({"ok": True, "usd": str(Path(args.out)), "joints": joints, "warnings": warnings, "dry_run": True})
    try:
        res = import_robot(args, sensors, warnings)
    except Exception as e:
        import traceback
        traceback.print_exc()
        res = {"ok": False, "usd": None, "joints": 0, "warnings": warnings, "errors": [f"{type(e).__name__}: {e}"]}
    finish(res)


if __name__ == "__main__":
    main()
