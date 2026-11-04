"""Contact sheet of the Isaac People allow-list: one still per character, in a grid, with labels.

    scripts/isaac_host.sh is not used (no host): run with Isaac's python, e.g.
    ~/isaacsim/python.sh scripts/character_sheet.py --out artifacts/screens/isaac_people/characters.png

Every character in config/isaac/characters.yaml is spawned (its ``hide`` prims deactivated) in an empty stage with
a light, in rows of seven, and photographed head on. Look at it: no hard hats, hi-vis or workwear should show.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-row", type=int, default=7)
ap.add_argument("--yaw", type=float, default=0.0, help="degrees about z, to turn the characters towards the camera")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSkel  # noqa: E402
import numpy as np  # noqa: E402

from wardzero_agents.isaac_host import characters as ch  # noqa: E402
from wardzero_agents.isaac_host.people_isaac import ASSETS_ROOT  # noqa: E402

allow = ch.load_allowlist()
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.GetSizeAttr().Set(1.0)
xf = UsdGeom.Xformable(floor)
xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
xf.AddScaleOp().Set(Gf.Vec3d(2000, 2000, 0.1))
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(900)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 20))
SP = 1.5
ROW = 60.0  # rows far apart, so a camera sees one row only
DIST = 13.0
rows = [allow.characters[i:i + args.per_row] for i in range(0, len(allow.characters), args.per_row)]
for r, row in enumerate(rows):
    for c, char in enumerate(row):
        path = f"/World/C_{r}_{c}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetPayloads().AddPayload(str(ASSETS_ROOT / char.usd))
        t = UsdGeom.Xformable(prim)
        t.ClearXformOpOrder()
        t.AddTranslateOp().Set(Gf.Vec3d(c * SP, r * ROW, 0.0))
        t.AddRotateZOp().Set(args.yaw)
for _ in range(30):
    app.update()
for r, row in enumerate(rows):  # accessories hidden as at spawn
    for c, char in enumerate(row):
        for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/C_{r}_{c}")):
            if p.IsA(UsdSkel.Root):
                for name in char.hide:
                    q = stage.GetPrimAtPath(p.GetPath().AppendChild(name))
                    if q:
                        q.SetActive(False)
for _ in range(40):
    app.update()
W, H = 1920, 600
sheet = Image.new("RGB", (W, H * len(rows)))
cx = (len(rows[0]) - 1) * SP / 2
for r, row in enumerate(rows):
    cam = UsdGeom.Camera.Define(stage, f"/World/Cam{r}")
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * H / W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(cx, r * ROW - DIST, 1.0), Gf.Vec3d(cx, r * ROW, 0.9), Gf.Vec3d(0, 0, 1)).GetInverse()
    xfc = UsdGeom.Xformable(cam)
    xfc.ClearXformOpOrder()
    xfc.AddTransformOp().Set(m)
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach([rep.create.render_product(f"/World/Cam{r}", (W, H))])
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
    img = Image.fromarray(np.asarray(an.get_data())[..., :3])
    d = ImageDraw.Draw(img)
    scale = 24.0 / 20.955 * W / DIST  # pixels per metre at the row's depth
    for c, char in enumerate(row):
        x = W / 2 + (c * SP - cx) * scale
        d.text((x - 55, H - 22), f"{char.id} {char.sex[0]} {char.outfit}", fill=(255, 255, 255))
    sheet.paste(img, (0, r * H))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.out)
print("SHEET", args.out, len(allow.characters), flush=True)
app.close()
