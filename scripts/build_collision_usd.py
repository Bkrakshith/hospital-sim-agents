"""Bake the hospital's visible geometry into a collision-only USD for Isaac runs. Run it once after cloning:

    pip install usd-core numpy
    python scripts/build_collision_usd.py [--stage PATH/hospital_g1.usd] [--out PATH/hospital_collision.usd]

(or, without installing anything: uv run --no-project --with usd-core --with numpy python scripts/build_collision_usd.py)

The stage defaults to the hospital-sim-env checkout next to this repo, and the output to assets/hospital_collision.usd,
which is where `hhumans` looks (WARDZERO_COLLISION_USD overrides it). It takes a few minutes.

Every visible mesh becomes a static triangle-mesh collider (PhysX approximation "none"), with its points baked to world
space, so people and robots meet exactly the walls, doors, lifts, desks and furniture they see.

Left out, on purpose:
- /World/Wing (the operating suite, 18.8 M triangles);
- floor slabs (flat meshes whose top is within 3 cm of z = 0): the physics ground plane is the floor;
- the waiting-chair rows SM_Chair_02a* (people sit on them) and the automatic sliding entrance doors (GlassDoor*);
- with --skip-door-leaves: the hinged door leaves' meshes (/World/Fixtures/Doors*/<door>/Leaf*) and the Junction side
  panels that overlap Compartment/Leaf1's hinge. The Isaac host swings the leaves on a run's routes with physics, and a
  static copy would block them. (The host also switches these prims off in its wrapper layer, so an older
  collision USD works too.)
No textures or render materials are copied; one physics material (painted wall / vinyl floor) is bound to everything.
"""

import argparse
import os
import re
import time

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.dirname(HERE)
REPO = os.environ.get("HG1_ROOT") or os.path.dirname(AGENTS)
SKIP = re.compile(r"^/World/Wing|SM_Chair_02a|GlassDoor")
SKIP_DOORS = re.compile(r"^/World/Fixtures/Doors\d*/[^/]+/Leaf|Junction_CompartmentSidePanel_b[0-2]$")
FLOOR_TOP = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=os.path.join(REPO, "hospital-sim-env", "isaac", "usd", "hospital_g1.usd"))
    ap.add_argument("--out", default=os.path.join(AGENTS, "assets", "hospital_collision.usd"))
    ap.add_argument("--skip-door-leaves", action="store_true",
                    help="leave out the hinged door leaves (and the panels on Compartment/Leaf1's hinge)")
    a = ap.parse_args()
    t0 = time.time()
    src = Usd.Stage.Open(a.stage)
    xc = UsdGeom.XformCache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = Usd.Stage.CreateNew(a.out)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    root = UsdGeom.Xform.Define(out, "/Hospital")
    out.SetDefaultPrim(root.GetPrim())
    mat = UsdShade.Material.Define(out, "/Hospital/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.6)  # painted plaster / vinyl against clothing or shoes (typical, assumed)
    pm.CreateDynamicFrictionAttr(0.5)
    pm.CreateRestitutionAttr(0.0)
    kept = skipped_floor = skipped_named = tris = 0
    for p in src.Traverse(Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        path = str(p.GetPath())
        if SKIP.search(path) or (a.skip_door_leaves and SKIP_DOORS.search(path)):
            skipped_named += 1
            continue
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        m = UsdGeom.Mesh(p)
        pts = m.GetPointsAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        fvi = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc or not fvi:
            continue
        M = np.array(xc.GetLocalToWorldTransform(p))
        w = (np.c_[np.asarray(pts, dtype=np.float64), np.ones(len(pts))] @ M)[:, :3]
        if w[:, 2].max() < FLOOR_TOP:
            skipped_floor += 1
            continue
        name = re.sub(r"[^A-Za-z0-9_]", "_", path.strip("/"))[-120:]
        name = name if name[0].isalpha() else "m_" + name
        dst = UsdGeom.Mesh.Define(out, f"/Hospital/c{kept:05d}_{name}")
        dst.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(w.astype(np.float32)))
        dst.CreateFaceVertexCountsAttr(fvc)
        dst.CreateFaceVertexIndicesAttr(fvi)
        dst.CreatePurposeAttr(UsdGeom.Tokens.guide)  # never rendered
        UsdPhysics.CollisionAPI.Apply(dst.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(dst.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(dst.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        kept += 1
        tris += int(sum(max(c - 2, 0) for c in fvc))
    out.GetRootLayer().Save()
    print(f"{kept} collision meshes, {tris} triangles; skipped {skipped_named} (wing, chairs, entrance doors), "
          f"{skipped_floor} floor slabs -> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
