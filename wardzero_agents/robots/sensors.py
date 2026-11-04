"""Robot sensors: the SensorSpec, the catalogue of Isaac's sensor models, and wardzero_sensors.yaml.

A developer's robot gets sensors from the Gazebo tags in its URDF or by picking a model from the catalogue
(config/sensors/catalogue.yaml, generated from the Isaac asset tree by build_catalogue and hand-checked against
datasheets). The WardZero microphone and speaker are always available as kinds "mic" and "speaker".
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml

SENSORS_FILE = "wardzero_sensors.yaml"
KINDS = ("rgbd", "rgb", "lidar", "imu", "mic", "speaker")
CATALOGUE_PATH = Path(__file__).resolve().parents[2] / "config" / "sensors" / "catalogue.yaml"


def default_asset_root() -> Path:
    """The Isaac asset root (the folder holding Isaac/), $WARDZERO_ISAAC_ASSETS or ~/isaacsim_assets/Assets/Isaac/6.0."""
    env = os.environ.get("WARDZERO_ISAAC_ASSETS")
    return Path(env).expanduser() if env else Path.home() / "isaacsim_assets" / "Assets" / "Isaac" / "6.0"


@dataclass(frozen=True)
class SensorSpec:
    id: str
    kind: Literal["rgbd", "rgb", "lidar", "imu", "mic", "speaker"]
    model: Optional[str]  # catalogue id, or None for a generic sensor built from params
    link: str
    xyz: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown sensor kind {self.kind!r}")
        object.__setattr__(self, "xyz", tuple(float(v) for v in self.xyz))
        object.__setattr__(self, "rpy", tuple(float(v) for v in self.rpy))

    def label(self) -> str:
        """A short phrase for the robot's prompt."""
        text = str(self.params.get("label") or f"{self.kind} {self.model or self.id}")
        return text + " (approx.)" if self.model and self.params.get("source") == "generic" else text

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "model": self.model, "link": self.link,
                "xyz": list(self.xyz), "rpy": list(self.rpy), "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "SensorSpec":
        return cls(id=str(d["id"]), kind=d["kind"], model=d.get("model"), link=str(d.get("link", "")),
                   xyz=tuple(d.get("xyz") or (0, 0, 0)), rpy=tuple(d.get("rpy") or (0, 0, 0)),
                   params=dict(d.get("params") or {}))


def write_sensors_yaml(folder: Path, sensors) -> Path:
    path = Path(folder) / SENSORS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sensors": [s.to_dict() for s in sensors]}, sort_keys=False))
    return path


def read_sensors_yaml(folder_or_file: Path) -> list[SensorSpec]:
    p = Path(folder_or_file)
    if p.is_dir():
        p = p / SENSORS_FILE
    data = yaml.safe_load(p.read_text()) or {}
    return [SensorSpec.from_dict(d) for d in data.get("sensors", [])]


# ------------------------------------------------------------------ catalogue
@lru_cache(maxsize=1)
def load_catalogue() -> list[dict]:
    return yaml.safe_load(CATALOGUE_PATH.read_text())["sensors"]


def catalogue_entry(model_id: str) -> dict:
    """The catalogue row for a model, including `source` ("datasheet" or "generic": approximate numbers)."""
    for e in load_catalogue():
        if e["id"] == model_id:
            return e
    raise KeyError(f"unknown sensor model {model_id!r}")


get_catalogue_entry = catalogue_entry


def sensor_from_catalogue(model_id: str, sensor_id: str, link: str, xyz=(0, 0, 0), rpy=(0, 0, 0),
                          params: Optional[dict] = None) -> SensorSpec:
    """A SensorSpec for a catalogue model; its `source` is copied into params so UIs and prompts can flag approximations."""
    e = catalogue_entry(model_id)
    return SensorSpec(sensor_id, e["kind"], model_id, link, xyz, rpy, {**(params or {}), "source": e["source"]})


_SKIP_VENDORS = {"tashan", "texasinstruments"}  # tactile skin and mmWave radar: no matching sensor kind
_SKIP_WORDS = ("material", "parts", "_config", "sim_base", "_case", "case.", "profil", "resolution", "configuration",
               "texture", "thumb", "meshes", "inspector", "caaz")
_SKIP_RE = re.compile(r"\d+ch\d+hz|rev\d", re.I)
_VENDOR_KIND = {"realsense": "rgbd", "orbbec": "rgbd", "luxonis": "rgbd", "stereolabs": "rgbd",
                "leopardimaging": "rgb", "sensing": "rgb"}
_GENERIC = {  # approximations for models without a datasheet row; marked source: generic
    "rgbd": {"fov_h": 80.0, "fov_v": 60.0, "range_min": 0.3, "range_max": 8.0, "resolution": [1280, 720]},
    "rgb": {"fov_h": 90.0, "fov_v": 60.0, "resolution": [1920, 1080]},
    "lidar": {"fov_h": 360.0, "fov_v": 30.0, "range_min": 0.1, "range_max": 30.0},
}
# Hand-checked values (vendor datasheets; for RealSense D455 the asset's colour camera, focal 1.93 mm over a
# 3.896 mm aperture, gives the same 90 degrees). fov in degrees, range in metres. Livox and D435 are not in the 6.0 assets.
_DATASHEET = {
    "realsense_d455": {"fov_h": 87.0, "fov_v": 58.0, "range_min": 0.6, "range_max": 6.0, "resolution": [1280, 720],
                       "color_fov_h": 90.0, "color_fov_v": 65.0},
    "realsense_d457": {"fov_h": 87.0, "fov_v": 58.0, "range_min": 0.6, "range_max": 6.0, "resolution": [1280, 720]},
    "realsense_d555": {"fov_h": 87.0, "fov_v": 58.0, "range_min": 0.4, "range_max": 6.0, "resolution": [1280, 720]},
    "orbbec_femto_mega": {"fov_h": 75.0, "fov_v": 65.0, "range_min": 0.25, "range_max": 5.46, "resolution": [640, 576]},
    "orbbec_gemini2": {"fov_h": 91.0, "fov_v": 66.0, "range_min": 0.15, "range_max": 5.0, "resolution": [1280, 800]},
    "orbbec_gemini_335": {"fov_h": 94.0, "fov_v": 68.0, "range_min": 0.17, "range_max": 20.0, "resolution": [1280, 800]},
    "orbbec_gemini_335l": {"fov_h": 94.0, "fov_v": 68.0, "range_min": 0.17, "range_max": 20.0, "resolution": [1280, 800]},
    "stereolabs_zed_x": {"fov_h": 110.0, "fov_v": 80.0, "range_min": 0.3, "range_max": 20.0, "resolution": [1920, 1200]},
    "stereolabs_zed_x_mini": {"fov_h": 110.0, "fov_v": 80.0, "range_min": 0.1, "range_max": 15.0,
                              "resolution": [1920, 1200]},
    "ouster_os0": {"fov_h": 360.0, "fov_v": 90.0, "range_min": 0.25, "range_max": 50.0},
    "ouster_os1": {"fov_h": 360.0, "fov_v": 45.0, "range_min": 0.25, "range_max": 120.0},
    "ouster_os2": {"fov_h": 360.0, "fov_v": 22.5, "range_min": 0.25, "range_max": 240.0},
    "hesai_xt32": {"fov_h": 360.0, "fov_v": 31.0, "range_min": 0.05, "range_max": 120.0},
    "hesai_xt32_sd10": {"fov_h": 360.0, "fov_v": 31.0, "range_min": 0.05, "range_max": 120.0},
    "velodyne_vls_128": {"fov_h": 360.0, "fov_v": 40.0, "range_min": 0.5, "range_max": 245.0},
    "slamtec_rplidar_s2e": {"fov_h": 360.0, "fov_v": 0.0, "range_min": 0.05, "range_max": 30.0},
    "sick_tim781": {"fov_h": 270.0, "fov_v": 0.0, "range_min": 0.05, "range_max": 25.0},
    "sick_nanoscan3": {"fov_h": 275.0, "fov_v": 0.0, "range_min": 0.0, "range_max": 9.0},
    "sick_microscan3": {"fov_h": 275.0, "fov_v": 0.0, "range_min": 0.0, "range_max": 9.0},
    "luxonis_oak_d_pro_poe": {"fov_h": 80.0, "fov_v": 55.0, "range_min": 0.7, "range_max": 12.0,
                              "resolution": [1280, 800]},
}
_ALIASES = {  # asset stem (vendor dir lower, normalised stem) -> catalogue id
    ("realsense", "rsd455"): "realsense_d455", ("realsense", "rsd457"): "realsense_d457",
    ("realsense", "rsd555"): "realsense_d555", ("orbbec", "femtomegav10"): "orbbec_femto_mega",
    ("orbbec", "gemini2v10"): "orbbec_gemini2", ("orbbec", "gemini335"): "orbbec_gemini_335",
    ("orbbec", "gemini335l"): "orbbec_gemini_335l", ("hesai", "xt32"): "hesai_xt32",
    ("stereolabs", "zedxmini"): "stereolabs_zed_x_mini", ("stereolabs", "zedx"): "stereolabs_zed_x",
    ("slamtec", "rplidars2e"): "slamtec_rplidar_s2e",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def build_catalogue(root: Path) -> list[dict]:
    """One entry per sensor model under <root>/Isaac/Sensors/<Vendor>/. Paths in the entries are relative to root.

    Skips materials, parts, configuration layers, per-profile and per-resolution variants (the base model carries
    them), hidden files and vendors without a matching kind. Numbers come from the datasheet table, else from
    generic per-kind values (source: generic) that a human should replace.
    """
    root = Path(root)
    base = root / "Isaac" / "Sensors"
    found: dict[tuple[str, str], Path] = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix not in (".usd", ".usda", ".usdc"):
            continue
        rel = p.relative_to(base)
        vendor = rel.parts[0]
        low = str(rel).lower()
        if (len(rel.parts) < 2 or vendor.lower() in _SKIP_VENDORS or any(part.startswith(".") for part in rel.parts)
                or any(w in low for w in _SKIP_WORDS) or _SKIP_RE.search(p.stem)):
            continue
        stem = _norm(p.stem)
        stem = stem[len(_norm(vendor)):] if stem.startswith(_norm(vendor)) and len(stem) > len(_norm(vendor)) else stem
        key = (vendor.lower(), stem)
        old = found.get(key)
        rank = lambda q: (q.suffix != ".usd", len(q.parts))  # noqa: E731
        if old is None or rank(p) < rank(old):
            found[key] = p
    entries = []
    for (vendor, stem), p in sorted(found.items()):
        rel = p.relative_to(root)
        raw = re.sub(rf"^{re.escape(vendor)}[_\- ]?", "", p.stem, flags=re.I)
        sid = _ALIASES.get((vendor, stem)) or _slug(f"{vendor}_{raw}")
        name = p.relative_to(base).parts[0].lower()
        kind = _VENDOR_KIND.get(name, "rgbd" if "visionary" in p.stem.lower() else "lidar")
        e = {"id": sid, "vendor": p.relative_to(base).parts[0], "kind": kind, "usd": rel.as_posix()}
        if sid in _DATASHEET:
            e.update(_DATASHEET[sid])
            e["source"] = "datasheet"
        else:
            e.update(_GENERIC[kind])
            e["source"] = "generic"
        entries.append(e)
    ids = [e["id"] for e in entries]
    return [e for i, e in enumerate(entries) if e["id"] not in ids[:i]]


def main(argv=None) -> int:
    """python -m wardzero_agents.robots.sensors [asset_root]  regenerates config/sensors/catalogue.yaml."""
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else default_asset_root()
    entries = build_catalogue(root)
    CATALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOGUE_PATH.write_text(
        "# Generated by wardzero_agents.robots.sensors.build_catalogue; source: datasheet rows are hand-checked,\n"
        "# source: generic rows carry per-kind approximations. fov in degrees, range in metres, usd relative to the\n"
        "# Isaac asset root.\n" + yaml.safe_dump({"sensors": entries}, sort_keys=False))
    print(f"{len(entries)} models -> {CATALOGUE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
