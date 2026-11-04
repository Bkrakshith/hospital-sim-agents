"""The Isaac side of the world's objects: props, Flow fire and call-bell lights as prims under ``/World/WZ``.

Everything here needs Kit (pxr, carb); imports are inside the functions that use them. ``UsdObjectStage`` implements the
small stage interface that ``objects.ObjectMirror`` drives (and the unit tests fake); ``IsaacObjects`` ties the two to a
StubWorld. The StubWorld stays the owner of the logic: nothing here decides anything.

- Props: a tray, an extinguisher and the instruments are referenced from existing assets (see ``PROP_ASSETS``); the
  specimen box is two cubes. Kinematic by default: any rigid body inside an asset (the instrument models are rigid
  bodies) is made kinematic with its colliders off, so props never push a person, a robot or the NavMesh.
- Fire: ``omni.flowusd`` sphere emitters, one per fire, on one shared FlowSimulate / FlowOffscreen / FlowRender layer
  (authored the way Isaac Sim's own incident extension does it, in metres and Z up). The emitter's radius, fuel and
  temperature follow the fire's size; at size 0 the emitter prim is removed and the flames burn off. If Flow cannot
  be used (``fire="proxy"``, or the extension is missing) a stack of emissive cones, scaled by the fire's size, stands in for it.
- Call lights: a small emissive lamp on the wall above the head of each bed with a bell; lit (and a light switched on)
  while the bell rings.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import objects as ob
from .people_isaac import ASSETS_ROOT

if TYPE_CHECKING:
    from ..world.stub_world import StubWorld

log = logging.getLogger("wardzero.isaac.objects")

HG1 = Path(__file__).resolve().parents[3]
OBJECTS_USD = HG1 / "hospital-sim-env" / "objects" / "usd"
PROP_ASSETS = {  # kind -> usd under the Isaac assets (instruments: hospital-sim-env/objects/usd/<name>.usd)
    "tray": ASSETS_ROOT / "Isaac/Environments/Hospital/Props/SM_UtilitiesTray_01e.usd",  # 0.53 x 0.31 x 0.09 m, 3.6 k triangles
    "extinguisher": ASSETS_ROOT / "Isaac/Environments/Office/Props/SM_Extinguisher.usd",  # 0.18 x 0.26 x 0.58 m, 1.5 k triangles
}
ROOT = "/World/WZ"
BELL_LIT = (8.0, 0.7, 0.2)  # emissive colour of a ringing call light
BELL_OFF = (0.02, 0.02, 0.02)
FLOW_LAYER = 3
FIRE_MODES = ("flow", "proxy", "none")


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", key)


class UsdObjectStage:
    """Authors and updates the prims. Methods are the ones ``objects.ObjectMirror`` calls."""

    def __init__(self, stage, fire: str = "flow"):
        if fire not in FIRE_MODES:
            raise ValueError(f"fire must be one of {', '.join(FIRE_MODES)}")
        self.stage = stage
        self.fire = fire
        self.paths: dict[str, str] = {}  # item key -> prim path
        self.bell_paths: dict[str, str] = {}
        self.fire_paths: dict[str, str] = {}
        self._flow_ready = False
        self._ops: dict[str, tuple] = {}
        for scope in ("", "/Items", "/Bells", "/Fires"):
            self.stage.DefinePrim(ROOT + scope, "Scope" if scope else "Xform")

    # ---- items -------------------------------------------------------------------------------
    def add_item(self, key: str, kind: str, label: str) -> None:
        from pxr import UsdGeom

        path = f"{ROOT}/Items/{_safe(key)}"
        if kind == "specimen_box":
            prim = self._specimen_box(path)
        else:
            name = ob.instrument_asset(label) if kind == "instrument" else None
            asset = OBJECTS_USD / f"{name}.usd" if name else PROP_ASSETS[kind]
            prim = self.stage.DefinePrim(path, "Xform")
            if not Path(asset).exists():
                log.warning("prop asset %s is missing; %s is drawn as a box", asset, key)
                prim = self._specimen_box(path)
            else:
                prim.GetReferences().AddReference(str(asset))
        self._kinematic(prim)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        self._ops[key] = (xf.AddTranslateOp(), xf.AddRotateZOp())
        self.paths[key] = path
        UsdGeom.Imageable(prim).MakeInvisible()  # shown by the first show_item once it has a pose

    def _kinematic(self, root) -> None:
        """Every rigid body in the asset becomes kinematic and every collider is switched off."""
        from pxr import Usd, UsdPhysics

        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.IsInstanceProxy():
                continue
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
                if p.HasAttribute("physxRigidBody:enableCCD"):  # PhysX rejects CCD on a kinematic body
                    p.GetAttribute("physxRigidBody:enableCCD").Set(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)

    def _specimen_box(self, path: str):
        """A white transport box with a blue lid and an orange band: 0.26 x 0.18 x 0.14 m, base at the origin."""
        from pxr import Gf, UsdGeom

        root = self.stage.DefinePrim(path, "Xform")

        def cube(name, size, z, rgb):
            c = UsdGeom.Cube.Define(self.stage, f"{path}/{name}")
            c.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

        cube("Body", (0.26, 0.18, 0.11), 0.055, (0.92, 0.93, 0.95))
        cube("Lid", (0.27, 0.19, 0.03), 0.125, (0.10, 0.30, 0.65))
        cube("Band", (0.262, 0.055, 0.112), 0.056, (0.95, 0.45, 0.05))
        return root

    def place_item(self, key: str, x: float, y: float, z: float, yaw: float) -> None:
        from pxr import Gf

        t, r = self._ops[key]
        t.Set(Gf.Vec3d(x, y, z))
        r.Set(math.degrees(yaw))

    def show_item(self, key: str, visible: bool) -> None:
        from pxr import UsdGeom

        img = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.paths[key]))
        img.MakeVisible() if visible else img.MakeInvisible()

    # ---- call lights -------------------------------------------------------------------------
    def add_bell(self, key: str, x: float, y: float, z: float, yaw: float) -> None:
        from pxr import Gf, UsdGeom, UsdLux, UsdShade

        path = f"{ROOT}/Bells/{_safe(key)}"
        root = UsdGeom.Xform.Define(self.stage, path)
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xf.AddRotateZOp().Set(math.degrees(yaw))
        plate = UsdGeom.Cube.Define(self.stage, f"{path}/Plate")  # a small backplate on the wall (local +x is towards the wall)
        plate.CreateSizeAttr(1.0)
        pxf = UsdGeom.Xformable(plate)
        pxf.AddTranslateOp().Set(Gf.Vec3d(-0.015, 0.0, 0.0))
        pxf.AddScaleOp().Set(Gf.Vec3f(0.03, 0.16, 0.10))
        plate.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.87)])
        lamp = UsdGeom.Sphere.Define(self.stage, f"{path}/Lamp")
        lamp.CreateRadiusAttr(0.035)
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(-0.06, 0.0, 0.0))
        mat = UsdShade.Material.Define(self.stage, f"{path}/Mat")
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Mat/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", _sdf("color3f")).Set(Gf.Vec3f(0.25, 0.02, 0.02))
        sh.CreateInput("emissiveColor", _sdf("color3f")).Set(Gf.Vec3f(*BELL_OFF))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(lamp.GetPrim()).Bind(mat)
        light = UsdLux.SphereLight.Define(self.stage, f"{path}/Light")
        light.CreateRadiusAttr(0.04)
        light.CreateIntensityAttr(8000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.25, 0.1))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, -0.1))
        UsdGeom.Imageable(light).MakeInvisible()
        self.bell_paths[key] = path

    def light_bell(self, key: str, lit: bool) -> None:
        from pxr import Gf, UsdGeom, UsdShade

        path = self.bell_paths[key]
        sh = UsdShade.Shader(self.stage.GetPrimAtPath(f"{path}/Mat/Surface"))
        sh.GetInput("emissiveColor").Set(Gf.Vec3f(*(BELL_LIT if lit else BELL_OFF)))
        light = UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{path}/Light"))
        light.MakeVisible() if lit else light.MakeInvisible()

    # ---- fire --------------------------------------------------------------------------------
    def set_fire(self, key: str, x: float, y: float, params: "ob.FireParams") -> None:
        if self.fire == "none":
            return
        if self.fire == "flow" and not self._flow_ready:
            self._flow_ready = self._init_flow() or False
            if not self._flow_ready:
                log.warning("Flow is not available: fires are drawn as emissive proxies")
                self.fire = "proxy"
        if self.fire == "flow":
            self._flow_emitter(key, x, y, params)
        else:
            self._proxy(key, x, y, params)

    def remove_fire(self, key: str) -> None:
        path = self.fire_paths.pop(key, None)
        if path is not None and self.stage.GetPrimAtPath(path):
            self.stage.RemovePrim(path)

    def _init_flow(self) -> bool:
        """The shared Flow layer (simulate, offscreen shadows and colormap, render), in metres and Z up."""
        try:
            import carb
            import omni.kit.commands
            from omni.flowusd import _flowusd  # noqa: F401  (the extension is loaded)
            from pxr import Sdf
        except Exception:
            return False
        st = carb.settings.get_settings()
        for name in ("enabled", "rayTracedReflectionsEnabled", "rayTracedTranslucencyEnabled", "pathTracingEnabled"):
            st.set(f"/rtx/flow/{name}", True)
        stage = self.stage

        def make(path, kind):
            ok, prim = omni.kit.commands.execute("FlowCreatePrim", prim_path=path, type_name=kind)
            return prim if ok else None

        def attrs(prim_path, table):
            prim = stage.GetPrimAtPath(prim_path)
            for name, (kind, value) in table.items():
                prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, kind)).Set(value)

        sim = make(f"{ROOT}/flowSimulate", "FlowSimulate")
        off = make(f"{ROOT}/flowOffscreen", "FlowOffscreen")
        ren = make(f"{ROOT}/flowRender", "FlowRender")
        if sim is None or off is None or ren is None:
            return False
        for rel, kind in (("flowSimulate/advection", "FlowAdvectionCombustionParams"),
                          ("flowSimulate/advection/smoke", "FlowAdvectionChannelParams"),
                          ("flowSimulate/advection/velocity", "FlowAdvectionChannelParams"),
                          ("flowSimulate/advection/divergence", "FlowAdvectionChannelParams"),
                          ("flowSimulate/advection/temperature", "FlowAdvectionChannelParams"),
                          ("flowSimulate/advection/fuel", "FlowAdvectionChannelParams"),
                          ("flowSimulate/advection/burn", "FlowAdvectionChannelParams"),
                          ("flowSimulate/vorticity", "FlowVorticityParams"), ("flowSimulate/pressure", "FlowPressureParams"),
                          ("flowSimulate/summaryAllocate", "FlowSummaryAllocateParams"),
                          ("flowOffscreen/colormap", "FlowRayMarchColormapParams"),
                          ("flowOffscreen/shadow", "FlowShadowParams"), ("flowOffscreen/debugVolume", "FlowDebugVolumeParams"),
                          ("flowRender/rayMarch", "FlowRayMarchParams"), ("flowRender/rayMarch/cloud", "FlowRayMarchCloudParams")):
            stage.DefinePrim(f"{ROOT}/{rel}", kind)  # FlowCreatePrim only makes the three roots, not their parameter prims
        attrs(f"{ROOT}/flowSimulate", {"layer": ("Int", FLOW_LAYER), "densityCellSize": ("Float", 0.05),
                                       "stepsPerSecond": ("Float", 60.0), "forceSimulate": ("Bool", False),
                                       "simulateWhenPaused": ("Bool", False)})
        attrs(f"{ROOT}/flowSimulate/advection", {
            "buoyancyPerTemp": ("Float", 2.0), "burnPerTemp": ("Float", 4.0), "coolingRate": ("Float", 1.5),
            "fuelPerBurn": ("Float", 0.25), "smokePerBurn": ("Float", 3.0), "tempPerBurn": ("Float", 5.0),
            "ignitionTemp": ("Float", 0.05), "gravity": ("Float3", (0.0, 0.0, -9.81)), "combustionEnabled": ("Bool", True)})
        for chan, (damp, fade) in {"velocity": (0.01, 0.01), "divergence": (0.01, 1.0), "temperature": (0.0, 0.0),
                                   "fuel": (0.0, 0.0), "burn": (0.0, 0.0), "smoke": (0.3, 0.65)}.items():
            attrs(f"{ROOT}/flowSimulate/advection/{chan}", {"damping": ("Float", damp), "fade": ("Float", fade)})
        attrs(f"{ROOT}/flowSimulate/vorticity", {"forceScale": ("Float", 0.6), "velocityLogScale": ("Float", 100.0),
                                                 "velocityMask": ("Float", 1.0), "enabled": ("Bool", True)})
        attrs(f"{ROOT}/flowOffscreen", {"layer": ("Int", FLOW_LAYER)})
        attrs(f"{ROOT}/flowOffscreen/shadow", {"attenuation": ("Float", 100.0), "minIntensity": ("Float", 0.02),
                                               "lightDirection": ("Float3", (-1.0, 1.0, 1.0)), "enabled": ("Bool", True)})
        attrs(f"{ROOT}/flowOffscreen/colormap", {
            "colorScale": ("Float", 2.5), "colorScalePoints": ("FloatArray", [1, 1, 1, 1, 1, 1]), "resolution": ("UInt", 32),
            "rgbaPoints": ("Float4Array", [(0.0154, 0.0177, 0.0154, 0.004902), (0.03575, 0.03575, 0.03575, 0.504902),
                                           (0.03575, 0.03575, 0.03575, 0.504902), (1, 0.1594, 0.0134, 0.8),
                                           (13.53, 2.99, 0.12599, 0.8), (78, 39, 6.1, 0.7)]),
            "xPoints": ("FloatArray", [0, 0.05, 0.15, 0.6, 0.85, 1])})
        attrs(f"{ROOT}/flowRender", {"layer": ("Int", FLOW_LAYER)})
        attrs(f"{ROOT}/flowRender/rayMarch", {"attenuation": ("Float", 50.0), "shadowFactor": ("Float", 1.0)})
        attrs(f"{ROOT}/flowRender/rayMarch/cloud", {"densityMultiplier": ("Float", 0.5)})
        log.info("Flow fire layer authored under %s/flow*", ROOT)
        return True

    def _flow_emitter(self, key: str, x: float, y: float, p: "ob.FireParams") -> None:
        from pxr import Gf, Sdf

        path = self.fire_paths.get(key)
        if path is None:
            import omni.kit.commands

            base = f"{ROOT}/Fires/{_safe(key)}"
            self.stage.DefinePrim(base, "Xform")
            ok, prim = omni.kit.commands.execute("FlowCreatePrim", prim_path=f"{base}/emitter", type_name="FlowEmitterSphere")
            if not ok:
                raise RuntimeError("could not create a Flow emitter")
            from pxr import UsdGeom

            UsdGeom.Xformable(self.stage.GetPrimAtPath(base)).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
            for name, kind, value in (("layer", "Int", FLOW_LAYER), ("position", "Float3", (0.0, 0.0, 0.15)),
                                      ("velocity", "Float3", (0.0, 0.0, 0.0)), ("radiusIsWorldSpace", "Bool", True),
                                      ("coupleRateFuel", "Float", 2.0), ("coupleRateTemperature", "Float", 10.0),
                                      ("coupleRateVelocity", "Float", 2.0), ("smoke", "Float", 0.0),
                                      ("coupleRateSmoke", "Float", 2.0)):
                prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, kind)).Set(value)
            path = self.fire_paths[key] = base
        em = self.stage.GetPrimAtPath(f"{path}/emitter")
        for name, kind, value in (("radius", "Float", p.radius), ("fuel", "Float", p.fuel),
                                  ("temperature", "Float", p.temperature), ("enabled", "Bool", p.enabled)):
            em.CreateAttribute(name, getattr(Sdf.ValueTypeNames, kind)).Set(value)

    def _proxy(self, key: str, x: float, y: float, p: "ob.FireParams") -> None:
        """Fallback: three emissive cones (outer, middle, core) that scale with the fire."""
        from pxr import Gf, UsdGeom, UsdShade

        path = self.fire_paths.get(key)
        if path is None:
            path = self.fire_paths[key] = f"{ROOT}/Fires/{_safe(key)}"
            root = UsdGeom.Xform.Define(self.stage, path)
            UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
            UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
            for i, (r, h, rgb) in enumerate(((1.0, 1.0, (60.0, 12.0, 1.0)), (0.7, 0.8, (90.0, 40.0, 3.0)), (0.4, 0.5, (120.0, 100.0, 30.0)))):
                cone = UsdGeom.Cone.Define(self.stage, f"{path}/Cone{i}")
                cone.CreateAxisAttr("Z")
                cone.CreateRadiusAttr(r)
                cone.CreateHeightAttr(h)
                UsdGeom.Xformable(cone).AddTranslateOp().Set(Gf.Vec3d(0, 0, h / 2))
                mat = UsdShade.Material.Define(self.stage, f"{path}/Mat{i}")
                sh = UsdShade.Shader.Define(self.stage, f"{path}/Mat{i}/Surface")
                sh.CreateIdAttr("UsdPreviewSurface")
                sh.CreateInput("emissiveColor", _sdf("color3f")).Set(Gf.Vec3f(*rgb))
                sh.CreateInput("opacity", _sdf("float")).Set(0.85)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(cone.GetPrim()).Bind(mat)
        scale = max(p.radius, 1e-3) * (1.0 if p.enabled else 0.0)
        UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).MakeVisible() if p.enabled else UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).MakeInvisible()
        op = UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetOrderedXformOps()[1]
        op.Set(Gf.Vec3f(scale, scale, scale * 3.0))


def _sdf(name: str):
    from pxr import Sdf

    return getattr(Sdf.ValueTypeNames, {"color3f": "Color3f", "float": "Float"}[name])


class IsaacObjects:
    """The mirror and its USD stage for one StubWorld. ``update`` runs once a frame from ``IsaacBodies.step``."""

    def __init__(self, stage, world: "StubWorld", fire: str = "flow"):
        self.ustage = UsdObjectStage(stage, fire)
        self.mirror = ob.ObjectMirror(world, self.ustage)
        self._failed = False

    def update(self) -> None:
        if self._failed:
            return
        try:
            self.mirror.sync()
        except Exception:
            self._failed = True  # the logic goes on without the drawing
            log.exception("mirroring the world's objects failed; objects are no longer drawn")
