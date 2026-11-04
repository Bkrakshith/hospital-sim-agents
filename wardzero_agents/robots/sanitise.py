"""What makes a converted robot USD ready to be referenced into a running stage (spike S1 Q4, S1b).

The robot goes in after ``timeline.play()`` in one authoring pass, so everything PhysX needs must already be in the
file: joint armature (without it a light articulation goes NaN within about 5 frames), mass and inertia floors,
solver iterations, and a mounted sensor asset reduced to its camera. ``scripts/isaac_import_urdf.py`` applies these
rules with pxr and marks the layer; the host refuses a robot USD without the mark. Pure Python: the decisions are
here so they are testable without Kit.
"""
from __future__ import annotations

from typing import Optional

SANITISE_VERSION = 1
READY_KEY = "wardzero:ready_usd"
ARMATURE = 0.01  # kg m^2, on every revolute and prismatic joint
MASS_FLOOR_KG = 0.005
INERTIA_FLOOR = 1e-6
SOLVER_POSITION_ITERATIONS = 16
SOLVER_VELOCITY_ITERATIONS = 2


def floored_mass(mass: Optional[float]) -> Optional[float]:
    """The floor when the link is (nearly) massless, else None (leave it)."""
    return MASS_FLOOR_KG if not mass or mass < MASS_FLOOR_KG else None


def floored_inertia(diag) -> Optional[tuple]:
    """The diagonal inertia with every term at least the floor, or None when it already is."""
    if diag is None:
        return (INERTIA_FLOOR,) * 3
    if min(diag) >= INERTIA_FLOOR:
        return None
    return tuple(max(INERTIA_FLOOR, float(v)) for v in diag)


def sensor_prim_edits(type_name: str, name: str, is_mesh: bool, has_rigid_body: bool, has_mass: bool,
                      has_collision: bool, kind: str, hide_housing: Optional[bool]) -> list[str]:
    """Edits for one prim under a mounted sensor asset, in the order to apply them.

    Sensor assets are rigid bodies with their own IMU and a depth render-product template, and the housing sits over
    the lenses inside a robot's head shell. A body nested in an articulation costs ~300 ms/frame once it moves, so the
    body, its mass and its colliders go; the IMU and the template are deactivated; camera housings are hidden."""
    out = []
    if has_rigid_body:
        out.append("strip_rigid_body")
    if has_mass:
        out.append("strip_mass")
    if has_collision:
        out.append("disable_collision")
    if is_mesh and (hide_housing if hide_housing is not None else kind in ("rgbd", "rgb")):
        out.append("hide")
    if type_name == "IsaacImuSensor" or name == "TemplateRenderProducts":
        out.append("deactivate")
    return out


def ready_layer_data() -> dict:
    """customLayerData for the exported robot layer."""
    return {READY_KEY: SANITISE_VERSION, "wardzero:armature": ARMATURE}


def is_ready(layer_data) -> bool:
    return bool(layer_data) and layer_data.get(READY_KEY) == SANITISE_VERSION


def require_ready(layer_data, what: str) -> None:
    if not is_ready(layer_data):
        raise ValueError(f"{what} is not a ready robot USD: convert the URDF again (scripts/isaac_import_urdf.py)")
