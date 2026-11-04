"""Read a developer's URDF folder and say what is wrong with it, without ever raising on bad input.

parse_urdf picks the first *.urdf in the folder, lists links and joints, estimates mass and height, and extracts
Gazebo camera, depth, ray and IMU sensors as SensorSpecs. Problems (missing meshes, unresolved package:// paths,
xacro-only folders, unreadable XML, no sensors) are strings in UrdfInfo.issues.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from defusedxml import ElementTree as ET

from .sensors import SensorSpec

_MOVING = ("revolute", "continuous", "prismatic")
_CAMERA = {"camera": "rgb", "wideanglecamera": "rgb", "multicamera": "rgb", "depth": "rgbd", "depth_camera": "rgbd",
           "rgbd_camera": "rgbd", "ray": "lidar", "gpu_ray": "lidar", "lidar": "lidar", "gpu_lidar": "lidar",
           "imu": "imu"}
_MAX_DIRS = 5000


@dataclass
class UrdfInfo:
    robot_name: str = ""
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)  # {name, type, parent, child, limits}
    dof: int = 0
    mass_kg: float = 0.0
    height_m_est: float = 0.0
    issues: list = field(default_factory=list)
    gazebo_sensors: list = field(default_factory=list)
    urdf_path: Optional[Path] = None


def _floats(text, n=None, default=None):
    try:
        vals = [float(v) for v in (text or "").split()]
    except ValueError:
        return default
    return vals if n is None or len(vals) == n else default


def _f(el, tag, default=None):
    child = el.find(tag) if el is not None else None
    if child is None or child.text is None:
        return default
    try:
        return float(child.text.strip())
    except ValueError:
        return default


def _rot(rpy):
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(el):
    o = el.find("origin") if el is not None else None
    if o is None:
        return np.eye(4)
    xyz = _floats(o.get("xyz"), 3, [0.0] * 3)
    rpy = _floats(o.get("rpy"), 3, [0.0] * 3)
    t = np.eye(4)
    t[:3, :3] = _rot(rpy)
    t[:3, 3] = xyz
    return t


def _height_estimate(links, joints) -> float:
    """Zero-pose forward kinematics over the joint origins; height = highest minus lowest link origin."""
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    child_names = {j["child"] for j in joints}
    roots = [l for l in links if l not in child_names]
    pos = {r: np.eye(4) for r in roots}
    stack = list(roots)
    while stack:
        cur = stack.pop()
        for j in children.get(cur, []):
            if j["child"] in pos:
                continue
            pos[j["child"]] = pos[cur] @ j["_T"]
            stack.append(j["child"])
    zs = [t[2, 3] for t in pos.values()]
    return round(float(max(zs) - min(zs)), 4) if zs else 0.0


def _package_roots(folder: Path) -> dict:
    """Directory name -> path for every directory under folder (bounded), plus the folder itself."""
    out = {folder.name: folder}
    n = 0
    for p in folder.rglob("*"):
        if p.is_dir():
            out.setdefault(p.name, p)
            n += 1
            if n > _MAX_DIRS:
                break
    return out


def _mesh_ok(fn: str, urdf_dir: Path, folder: Path, pkgs: dict) -> bool:
    if fn.startswith("package://"):
        pkg, _, rest = fn[len("package://"):].partition("/")
        root = pkgs.get(pkg)
        return bool(root and rest and (root / rest).is_file())
    if fn.startswith("file://"):
        return Path(fn[7:]).is_file()
    p = Path(fn)
    if p.is_absolute():
        return p.is_file()
    return (urdf_dir / fn).is_file() or (folder / fn).is_file()


def _gazebo_sensors(root, link_names, issues) -> list:
    out, used = [], set()
    for gz in root.iter("gazebo"):
        ref = gz.get("reference")
        for s in gz.iter("sensor"):
            kind = _CAMERA.get((s.get("type") or "").lower())
            if kind is None:
                continue
            if not ref or ref not in link_names:
                issues.append(f"Gazebo sensor {s.get('name') or s.get('type')} is on link {ref!r}, which is not in the URDF")
                continue
            base = re.sub(r"\W+", "_", s.get("name") or f"{ref}_{kind}").strip("_") or kind
            sid, k = base, 2
            while sid in used:
                sid, k = f"{base}_{k}", k + 1
            used.add(sid)
            pose = _floats((s.findtext("pose") or "").strip(), None, []) or []
            pose = (pose + [0.0] * 6)[:6]
            params = {}
            cam = s.find("camera")
            if cam is not None and kind in ("rgb", "rgbd"):
                if (fov := _f(cam, "horizontal_fov")) is not None:
                    params["fov_h"] = fov
                w, h = _f(cam, "image/width"), _f(cam, "image/height")
                if w and h:
                    params["resolution"] = [int(w), int(h)]
                near, far = _f(cam, "clip/near"), _f(cam, "clip/far")
                if near is not None and far is not None:
                    params["range"] = [near, far]
            ray = s.find("ray")
            if ray is not None and kind == "lidar":
                hz = ray.find("scan/horizontal")
                if hz is not None:
                    if (n := _f(hz, "samples")) is not None:
                        params["samples"] = int(n)
                    lo, hi = _f(hz, "min_angle"), _f(hz, "max_angle")
                    if lo is not None and hi is not None:
                        params["angle"] = [lo, hi]
                lo, hi = _f(ray, "range/min"), _f(ray, "range/max")
                if lo is not None and hi is not None:
                    params["range"] = [lo, hi]
            if (rate := _f(s, "update_rate")) is not None:
                params["rate_hz"] = rate
            out.append(SensorSpec(sid, kind, None, ref, tuple(pose[:3]), tuple(pose[3:]), params))
    return out


def parse_urdf(folder) -> UrdfInfo:
    info = UrdfInfo()
    try:
        _parse(Path(folder), info)
    except Exception as e:  # bad input must come back as an issue, never as a crash
        info.issues.append(f"could not read the URDF: {type(e).__name__}: {e}")
    return info


def _parse(folder: Path, info: UrdfInfo) -> None:
    chosen = None
    if folder.is_file() and folder.suffix == ".urdf":  # a specific URDF: parse it in place, meshes resolve around it
        chosen, folder = folder, folder.parent
    if not folder.is_dir():
        info.issues.append(f"not a folder: {folder}")
        return
    urdfs = [chosen] if chosen else sorted(folder.rglob("*.urdf"))
    if not urdfs:
        if any(folder.rglob("*.xacro")) or any(folder.rglob("*.urdf.xacro")):
            info.issues.append("xacro found: run xacro to produce a .urdf first")
        else:
            info.issues.append("no .urdf file found in the folder")
        return
    # the first in the folder itself, else the first anywhere below it
    path = chosen or next((u for u in urdfs if u.parent == folder), urdfs[0])
    info.urdf_path = path
    try:
        root = ET.parse(str(path)).getroot()
    except Exception as e:
        info.issues.append(f"XML error in {path.name}: {type(e).__name__}: {e}")
        return
    if root.tag != "robot":
        info.issues.append(f"{path.name}: the root element is <{root.tag}>, not <robot>")
        return
    info.robot_name = root.get("name") or path.stem
    links = [l.get("name") for l in root.findall("link") if l.get("name")]
    info.links = links
    names = set(links)
    if not links:
        info.issues.append("the URDF has no links")
    mass = 0.0
    for l in root.findall("link"):
        me = l.find("inertial/mass")
        try:
            m = float(me.get("value")) if me is not None else None
        except (TypeError, ValueError):
            m = None
            info.issues.append(f"link {l.get('name')}: unreadable mass")
        mass += m or 0.0
    info.mass_kg = round(mass, 4)
    if links and mass == 0:
        info.issues.append("no link has a mass: physics needs <inertial> data")
    raw = []
    for j in root.findall("joint"):
        name, typ = j.get("name"), (j.get("type") or "")
        p = j.find("parent").get("link") if j.find("parent") is not None else None
        c = j.find("child").get("link") if j.find("child") is not None else None
        lim = {}
        le = j.find("limit")
        if le is not None:
            for k in ("lower", "upper", "effort", "velocity"):
                if le.get(k) is not None:
                    try:
                        lim[k] = float(le.get(k))
                    except ValueError:
                        info.issues.append(f"joint {name}: limit {k}={le.get(k)!r} is not a number")
        if typ in ("revolute", "prismatic") and le is None:
            info.issues.append(f"joint {name}: {typ} joint without <limit>")
        if p not in names or c not in names:
            info.issues.append(f"joint {name}: parent or child link is not defined")
            continue
        raw.append({"name": name, "type": typ, "parent": p, "child": c, "limits": lim, "_T": _origin(j)})
    info.dof = sum(1 for j in raw if j["type"] in _MOVING)
    info.height_m_est = _height_estimate(links, raw)
    info.joints = [{k: v for k, v in j.items() if k != "_T"} for j in raw]

    pkgs = _package_roots(folder)
    seen = set()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn not in seen:
            seen.add(fn)
            if not _mesh_ok(fn, path.parent, folder, pkgs):
                info.issues.append(f"mesh not found: {fn}")
    info.gazebo_sensors = _gazebo_sensors(root, names, info.issues)
    if not info.gazebo_sensors and not any("Gazebo sensor" in i for i in info.issues):
        info.issues.append("no Gazebo sensors in the URDF: add sensors from the catalogue")
