"""Recording and the Studio's live preview (I5).

``--record`` writes two MP4s to ``<run>/video`` with Isaac's hardware encoder (NVENC, ``omni.videoencoding``):
``overhead.mp4`` (a camera over the kept rooms) and ``follow.mp4`` (a third-person camera on the robot, when the
run has one). ``--preview`` (headless only) writes ``<run>/preview/latest.jpg`` 5 times a second, which the Studio
streams as MJPEG.

Rendering a frame for a writer costs about 300 ms on the RTX 3070 (spike S1: 3.3 fps at 2 x 720p), so a recorded run is
in lockstep: sim time per frame is fixed at 1/fps and wall time is whatever recording costs. ``Recorder.step`` replaces
``app.update`` in the host loop: it renders through ``rep.orchestrator.step`` (which advances the timeline one fixed
frame and fires the writers) and does nothing extra while the timeline is held, so video time is sim time.

The overhead camera streams from the render product's ``rgb`` annotator straight into the encoder; the encoder takes
one session at a time, so the follow camera goes through a Replicator writer (JPEG files) and is encoded in a second
pass when the run ends. Everything here that needs Kit imports it inside the function that uses it.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

from . import mp4fix

log = logging.getLogger("wardzero.isaac.recording")

DEFAULT_RES = (1280, 720)
MIN_RES, MAX_RES = (160, 90), (3840, 2160)
PREVIEW_WIDTH = 640  # the preview is a stride of the render, so the width is a divisor of the record width
PREVIEW_HZ = 5
PREVIEW_QUALITY = 70
CAMERAS_PATH = "/World/WardZeroCams"
CEILING_CUT_M = 2.3  # the overhead near plane sits this far above the floor: the ceiling is clipped, not deactivated
APERTURE_MM = 20.955  # Kit's default horizontal aperture
OVERHEAD_FOCAL_MM = 12.0
OVERHEAD_MARGIN = 1.06
FOLLOW_AHEAD_M, FOLLOW_SIDE_M, FOLLOW_HEIGHT_M, FOLLOW_TARGET_Z, FOLLOW_FOCAL_MM = 2.8, 1.0, 1.7, 1.0, 16.0
FOLLOW_DIR = "follow_frames"


def parse_res(text: str) -> tuple[int, int]:
    """``1280x720`` to (1280, 720). Even sides (the encoder wants them), between 160x90 and 3840x2160."""
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
    except ValueError:
        raise ValueError(f"--record-res must look like 1280x720, not {text!r}") from None
    if w % 2 or h % 2 or not (MIN_RES[0] <= w <= MAX_RES[0] and MIN_RES[1] <= h <= MAX_RES[1]):
        raise ValueError(f"--record-res {w}x{h}: both sides must be even, from {MIN_RES[0]}x{MIN_RES[1]} "
                         f"to {MAX_RES[0]}x{MAX_RES[1]}")
    return w, h


def preview_stride(fps: int, hz: int = PREVIEW_HZ) -> int:
    return max(1, round(fps / hz))


def preview_scale(res: tuple[int, int]) -> int:
    """The integer pixel stride that takes a record frame to about the preview width."""
    return max(1, res[0] // PREVIEW_WIDTH)


def preview_size(res: tuple[int, int]) -> tuple[int, int]:
    s = preview_scale(res)
    return res[0] // s, res[1] // s


@dataclasses.dataclass
class RecordConfig:
    res: tuple[int, int] = DEFAULT_RES
    fps: int = 30
    record: bool = False
    preview: bool = False
    video_dir: Path = Path("video")
    preview_path: Path = Path("preview/latest.jpg")
    preview_hz: int = PREVIEW_HZ

    @classmethod
    def from_args(cls, args, run_dir: Path) -> Optional["RecordConfig"]:
        """None when the run neither records nor previews (a headed run's window is its preview)."""
        preview = bool(getattr(args, "preview", False)) and not args.headed
        if not (args.record or preview):
            return None
        run_dir = Path(run_dir)
        return cls(res=tuple(args.record_res), fps=int(args.fps), record=bool(args.record), preview=preview,
                   video_dir=run_dir / "video", preview_path=run_dir / "preview" / "latest.jpg")

    @property
    def overhead_mp4(self) -> Path:
        return self.video_dir / "overhead.mp4"

    @property
    def follow_mp4(self) -> Path:
        return self.video_dir / "follow.mp4"

    def frames_to_seconds(self, frames: int) -> float:
        return frames / float(self.fps)


def encoder_args(cfg: RecordConfig) -> dict:
    """``start_encoding`` keywords: the frame count is unknown (0), the frame rate is the sim's."""
    return {"framerate": float(cfg.fps), "nframes": 0, "overwrite_video": True}


# ---- the cameras (pure) -----------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class OverheadView:
    eye: tuple
    target: tuple
    focal: float
    aperture: float
    near: float
    far: float


def overhead_view(rects: Sequence[tuple], aspect: float) -> OverheadView:
    """A camera straight above the kept rooms that sees all of them: every rect is inside the image, with a margin.
    The near plane cuts the ceiling away (S1b) so the floor is visible from above."""
    if not rects:
        raise ValueError("no rooms to frame")
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
    need_w = max(x1 - x0, (y1 - y0) * aspect) * OVERHEAD_MARGIN  # floor width the image must span
    height = max(need_w * OVERHEAD_FOCAL_MM / APERTURE_MM, CEILING_CUT_M + 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return OverheadView((cx, cy, height), (cx, cy + 0.01, 0.0), OVERHEAD_FOCAL_MM, APERTURE_MM,
                        height - CEILING_CUT_M, height + 50.0)


def follow_view(pose: tuple) -> tuple[tuple, tuple]:
    """(eye, target) of the third-person camera: in front of the robot and a little to its right, above its head,
    looking back at it. (Behind it, a reception robot stands behind its desk and the desk hides it; in front, the people
    it serves are in the frame.) It follows the robot's pose every frame."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    ex = x + FOLLOW_AHEAD_M * c + FOLLOW_SIDE_M * s
    ey = y + FOLLOW_AHEAD_M * s - FOLLOW_SIDE_M * c
    return (ex, ey, FOLLOW_HEIGHT_M), (x, y, FOLLOW_TARGET_Z)


# ---- the preview file -------------------------------------------------------------------------
class JpegSink:
    """Writes ``latest.jpg`` by writing a temporary file and renaming it, so the Studio never reads half a frame."""

    def __init__(self, path: Path, quality: int = PREVIEW_QUALITY):
        self.path, self.quality = Path(path), quality
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")

    def write(self, rgba) -> None:
        from PIL import Image

        Image.fromarray(rgba[..., :3]).save(self._tmp, format="JPEG", quality=self.quality)
        os.replace(self._tmp, self.path)

    def clear(self) -> None:
        for p in (self.path, self._tmp):
            try:
                p.unlink()
            except OSError:
                pass


# ---- Kit side ---------------------------------------------------------------------------------
def define_camera(stage, path: str, eye, target, focal: float, res: tuple[int, int], near: float = 0.05,
                  far: float = 300.0, aperture: float = APERTURE_MM) -> str:
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(aperture))
    cam.CreateVerticalApertureAttr(float(aperture) * res[1] / res[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(look_at(eye, target))
    return path


def look_at(eye, target):
    from pxr import Gf

    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()


class Recorder:
    """The writers, the preview and the per-frame step. Built after the host's start-up (the cameras are authored once
    the timeline is playing, like the robot's: they never touch physics or the NavMesh)."""

    def __init__(self, cfg: RecordConfig, app, timeline, stage, rects: Sequence[tuple], robot_pose=None):
        self.cfg, self.app, self.timeline, self.stage = cfg, app, timeline, stage
        self.rects, self.robot_pose = list(rects), robot_pose
        self.frames = 0  # frames rendered for the writers
        self.sim_frames = 0  # frames the timeline advanced while the recorder was in charge
        self.render_s = 0.0
        self.encoded = 0
        self.follow_frames = 0
        self._rep = self._enc = self._an_top = self._an_prev = self._follow_xf = self._follow_writer = None
        self._sink: Optional[JpegSink] = None
        self._stride = preview_stride(cfg.fps, cfg.preview_hz)
        self._last_rgba = None

    # ---- set up ----
    def start(self) -> None:
        import omni.replicator.core as rep

        cfg = self.cfg
        self._rep = rep
        res = cfg.res if cfg.record else preview_size(cfg.res)
        view = overhead_view(self.rects, res[0] / res[1])
        top = define_camera(self.stage, f"{CAMERAS_PATH}/overhead", view.eye, view.target, view.focal, res,
                            near=view.near, far=view.far, aperture=view.aperture)
        self._an_top = rep.AnnotatorRegistry.get_annotator("rgb")
        self._an_top.attach([rep.create.render_product(top, res, name="wz_overhead")])
        if cfg.preview:
            self._sink = JpegSink(cfg.preview_path)
        if cfg.record:
            cfg.video_dir.mkdir(parents=True, exist_ok=True)
            from video_encoding import get_video_encoding_interface

            self._enc = get_video_encoding_interface()
            ok = self._enc.start_encoding(video_filename=str(cfg.overhead_mp4), **encoder_args(cfg))
            log.info("recording: overhead %dx%d at %d fps to %s (encoder %s)", *res, cfg.fps, cfg.overhead_mp4, ok)
            self._start_follow()

    def _start_follow(self) -> None:
        pose = self.robot_pose() if self.robot_pose is not None else None
        if pose is None:
            log.info("recording: no robot, so no follow camera")
            return
        rep, cfg = self._rep, self.cfg
        eye, target = follow_view(pose)
        cam = define_camera(self.stage, f"{CAMERAS_PATH}/follow", eye, target, FOLLOW_FOCAL_MM, cfg.res)
        from pxr import UsdGeom

        self._follow_xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(cam))
        frames = cfg.video_dir / FOLLOW_DIR
        shutil.rmtree(frames, ignore_errors=True)
        self._follow_writer = rep.WriterRegistry.get("BasicWriter")
        self._follow_writer.initialize(output_dir=str(frames), rgb=True, image_output_format="jpeg")
        self._follow_writer.attach([rep.create.render_product(cam, cfg.res, name="wz_follow")])

    # ---- one frame ----
    def step(self) -> None:
        """Instead of ``app.update()``: one fixed frame, rendered for the writers when the timeline is running."""
        if not self.timeline.is_playing():
            self.app.update()  # held: nothing moves, so nothing is recorded
            return
        self.sim_frames += 1
        capture = self.cfg.record or self.sim_frames % self._stride == 0
        if not capture:
            self.app.update()
            return
        if self._follow_xf is not None:
            pose = self.robot_pose()
            if pose is not None:
                self._follow_xf.GetOrderedXformOps()[0].Set(look_at(*follow_view(pose)))
        t0 = time.monotonic()
        self._rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        import numpy as np

        img = np.asarray(self._an_top.get_data())
        self.render_s += time.monotonic() - t0
        self.frames += 1
        if img.size:
            self._last_rgba = img
        if self.cfg.record and self._last_rgba is not None:  # a frame the renderer was late with repeats the last one
            self._enc.encode_next_frame_from_buffer(np.ascontiguousarray(self._last_rgba), *self.cfg.res)
            self.encoded += 1
        if self._sink is not None and self._last_rgba is not None and self.sim_frames % self._stride == 0:
            s = preview_scale(self.cfg.res) if self.cfg.record else 1
            self._sink.write(self._last_rgba[::s, ::s])

    # ---- the end ----
    def finish(self) -> dict:
        """Finalise the overhead MP4, encode the follow frames, and return what was recorded."""
        cfg = self.cfg
        stats = {"frames": self.frames, "sim_frames": self.sim_frames, "render_s": round(self.render_s, 2),
                 "render_fps": round(self.frames / self.render_s, 2) if self.render_s else None,
                 "res": f"{cfg.res[0]}x{cfg.res[1]}", "fps": cfg.fps}
        if self._sink is not None:
            self._sink.clear()
        if not cfg.record:
            return stats
        try:
            self._rep.orchestrator.wait_until_complete()
        except Exception as exc:
            log.warning("the writers did not finish cleanly: %s", exc)
        self._enc.finalize_encoding()
        mp4fix.set_duration(cfg.overhead_mp4, self.encoded, cfg.fps)  # the encoder was told no length at the start
        stats["overhead_frames"] = self.encoded
        if self._follow_writer is not None:
            stats["follow_frames"] = self._encode_follow()
        log.info("recording done: %s", stats)
        return stats

    def _encode_follow(self) -> int:
        cfg = self.cfg
        folder = cfg.video_dir / FOLLOW_DIR
        files = sorted(p for p in folder.glob("rgb_*.jp*g"))
        if not files:
            log.warning("no follow-camera frames were written")
            return 0
        self._enc.start_encoding(video_filename=str(cfg.follow_mp4), framerate=float(cfg.fps), nframes=len(files),
                                 overwrite_video=True)
        for p in files:
            self._enc.encode_next_frame_from_file(str(p))
        self._enc.finalize_encoding()
        shutil.rmtree(folder, ignore_errors=True)
        return len(files)
