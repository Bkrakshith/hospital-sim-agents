"""Recording and the Studio's live preview (I5).

``--record`` writes two MP4s to ``<run>/video`` with Isaac's hardware encoder (NVENC, ``omni.videoencoding``):
``overhead.mp4`` (a camera over the kept rooms) and ``follow.mp4`` (a third-person camera on the robot, when the
run has one). ``--preview`` (headless only) writes ``<run>/preview/latest.jpg`` 5 times a second, which the Studio
streams as MJPEG.

Recording renders through plain ``app.update()`` and reads the render products' ``rgb`` annotators (they fill as the
timeline plays): 40 ms a frame at 720p on the RTX 3070, against 300 ms through ``rep.orchestrator.step`` (which also
crashed the Behavior System once real people were moving). A recorded run is still in lockstep, so the sim time per
frame is fixed at 1/fps whatever the cost. ``Recorder.step`` replaces ``app.update`` in the host loop and does nothing
extra while the timeline is held, so video time is sim time.

The overhead camera streams into the encoder; the encoder takes one session at a time, so the follow camera is written
as JPEG files and encoded to ``follow.mp4`` when the run ends (the files are then deleted). Everything here that needs
Kit imports it inside the function that uses it.
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
OVERHEAD_MARGIN = 1.03  # around the rooms' bounding box at the height of the wall tops the near plane leaves
FOLLOW_AHEAD_M, FOLLOW_SIDE_M, FOLLOW_HEIGHT_M, FOLLOW_TARGET_Z, FOLLOW_FOCAL_MM = 2.8, 1.0, 1.7, 1.0, 16.0
# WARDZERO_FOLLOW_FRAMING=desk: a wider, higher camera well in front of a fixed robot, so the people at its desk are in
# the frame too (the default follow camera is so close that they are cut off). (ahead, side, height, target z, focal mm)
FRAMINGS = {"follow": (FOLLOW_AHEAD_M, FOLLOW_SIDE_M, FOLLOW_HEIGHT_M, FOLLOW_TARGET_Z, FOLLOW_FOCAL_MM),
            "desk": (5.4, 0.6, 2.4, 1.0, 14.0)}
FOLLOW_DIR = "follow_frames"
FOLLOW_JPEG_QUALITY = 92
TIMING_FILE = "timing.json"


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
    rotated: bool = False  # True: the floor's y axis runs across the image (the image's top is -x), not x


ROTATE_GAIN = 1.1  # turn the camera only when that brings it at least this much closer (no flip-flop near square)


def overhead_view(rects: Sequence[tuple], aspect: float) -> OverheadView:
    """A camera straight above the kept rooms that sees all of them, as large as the frame allows: the rooms' bounding
    box (up to the wall tops at the cut height) fills the image's width or its height, with a small margin. An area
    much longer in y than in x (a ward along its corridor) would be a thin strip down the middle of a wide frame, so
    the camera then turns 90 degrees about the vertical and the long side runs across the image. The near plane cuts
    the ceiling away (S1b) so the floor is visible from above."""
    if not rects:
        raise ValueError("no rooms to frame")
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
    w, h = x1 - x0, y1 - y0
    need_w = max(w, h * aspect)  # floor width the image must span with x across it
    need_rot = max(h, w * aspect)  # ... with y across it
    rotated = need_rot * ROTATE_GAIN < need_w
    span = (need_rot if rotated else need_w) * OVERHEAD_MARGIN
    # the box is fitted at the cut height, not the floor: the outer walls' tops (the highest thing the near plane
    # leaves) are nearer the camera, so they reach further out in the image than the floor does
    height = max(span * OVERHEAD_FOCAL_MM / APERTURE_MM + CEILING_CUT_M, CEILING_CUT_M + 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # straight down; the tiny offset of the target sets which floor direction is the image's top: +y, or -x when turned
    # (then +y runs left to right)
    target = (cx - 0.01, cy, 0.0) if rotated else (cx, cy + 0.01, 0.0)
    return OverheadView((cx, cy, height), target, OVERHEAD_FOCAL_MM, APERTURE_MM, height - CEILING_CUT_M,
                        height + 50.0, rotated)


def follow_framing(env=None) -> str:
    """The framing named by WARDZERO_FOLLOW_FRAMING; anything unknown (or unset) is the default ``follow``."""
    name = (os.environ if env is None else env).get("WARDZERO_FOLLOW_FRAMING", "follow").strip().lower()
    return name if name in FRAMINGS else "follow"


def follow_view(pose: tuple, framing: str = "follow") -> tuple[tuple, tuple]:
    """(eye, target) of the third-person camera: in front of the robot and a little to its right, above its head,
    looking back at it. (Behind it, a reception robot stands behind its desk and the desk hides it; in front, the people
    it serves are in the frame.) It follows the robot's pose every frame."""
    ahead, side, height, target_z, _ = FRAMINGS.get(framing, FRAMINGS["follow"])
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    ex = x + ahead * c + side * s
    ey = y + ahead * s - side * c
    return (ex, ey, height), (x, y, target_z)


START_VIEW_FRAMING = "desk"


def start_view(robot_pose: Optional[tuple]) -> Optional[tuple[tuple, tuple, float]]:
    """(eye, target, focal mm) of the headed window when a run starts: where the scene happens, at the robot's start
    post, framed like the desk camera (in front of the robot, below the ceiling, wide enough for the people who come
    to it). None without a robot: the caller frames the kept rooms instead."""
    if robot_pose is None:
        return None
    eye, target = follow_view(robot_pose, START_VIEW_FRAMING)
    return eye, target, FRAMINGS[START_VIEW_FRAMING][4]


# ---- the headed window's follow camera (pure) ----------------------------------------------------
# behind and above the robot, looking past it: (back, height, look-ahead, target z, focal mm). At its post the G1 stands
# in front of the reception desk, so the camera is behind the counter: high and steep enough to look over it.
CHASE = (1.8, 2.4, 0.6, 0.7, 14.0)
CHASE_POS_ALPHA = 0.12  # per frame: how far the camera's anchor moves towards the robot (smoothing)
CHASE_YAW_ALPHA = 0.06  # ... and its heading towards the robot's
HEADED_CAMS = ("view_follow", "view_desk", "view_overhead")  # under CAMERAS_PATH; Kit's camera menu lists them


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


POST_RADIUS_M = 0.6  # the robot counts as at its post within this distance of it
POST_BLEND_PER_FRAME = 0.04  # how fast the camera eases between the post view and the chase view (about 1 s at 25 fps)


class ChaseCam:
    """A smoothed third-person camera behind and above a pose. ``update(pose)`` once a frame returns (eye, target).

    With a ``post`` (x, y, yaw), while the robot stands at its post the camera eases to the post view instead: the desk
    framing in front of the robot, high and wide enough for the robot and the person talking to it (from behind, the
    reception counter the G1 stands in front of hides it). Walking away, it eases back to the chase view."""

    def __init__(self, framing: tuple = CHASE, pos_alpha: float = CHASE_POS_ALPHA, yaw_alpha: float = CHASE_YAW_ALPHA,
                 post: Optional[tuple] = None):
        self.framing, self.pos_alpha, self.yaw_alpha = framing, pos_alpha, yaw_alpha
        self.x = self.y = self.yaw = None
        self.post = post
        self.w = None  # 1: post view, 0: chase view

    def at_post(self, pose: tuple) -> bool:
        return self.post is not None and math.hypot(pose[0] - self.post[0], pose[1] - self.post[1]) < POST_RADIUS_M

    def update(self, pose: tuple) -> tuple[tuple, tuple]:
        x, y, yaw = (float(v) for v in pose)
        if self.x is None:
            self.x, self.y, self.yaw = x, y, yaw
        else:
            self.x += self.pos_alpha * (x - self.x)
            self.y += self.pos_alpha * (y - self.y)
            self.yaw = _wrap(self.yaw + self.yaw_alpha * _wrap(yaw - self.yaw))
        eye, target = chase_view((self.x, self.y, self.yaw), self.framing)
        if self.post is None:
            return eye, target
        want = 1.0 if self.at_post(pose) else 0.0
        self.w = want if self.w is None else self.w + max(-POST_BLEND_PER_FRAME, min(POST_BLEND_PER_FRAME,
                                                                                       want - self.w))
        if self.w <= 0.0:
            return eye, target
        p_eye, p_target = follow_view(self.post, START_VIEW_FRAMING)
        k = self.w * self.w * (3 - 2 * self.w)  # smoothstep
        mix = lambda a, b: tuple(u + k * (v - u) for u, v in zip(a, b))  # noqa: E731
        return mix(eye, p_eye), mix(target, p_target)


def chase_view(pose: tuple, framing: tuple = CHASE) -> tuple[tuple, tuple]:
    """(eye, target): ``back`` metres behind the pose at ``height``, looking at a point ``ahead`` in front of it."""
    back, height, ahead, target_z, _ = framing
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x - back * c, y - back * s, height), (x + ahead * c, y + ahead * s, target_z)


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


class HeadedCameras:
    """The headed window's cameras: ``view_follow`` (the default: behind and above the robot, smoothed, moved every
    frame), ``view_desk`` (the fixed desk framing at the robot's post) and ``view_overhead`` (the kept rooms from
    above). All three are USD cameras, so Kit's camera menu offers them. Recording has its own cameras."""

    def __init__(self, stage, rects: Sequence[tuple], robot_pose, res: tuple[int, int] = DEFAULT_RES,
                 post: Optional[tuple] = None):
        self.stage, self.rects, self.robot_pose, self.res = stage, list(rects), robot_pose, res
        self.chase = ChaseCam(post=post)
        self.paths: dict[str, str] = {}
        self._follow_op = None
        self.updates = 0

    def start(self) -> None:
        from pxr import UsdGeom

        pose = self.robot_pose()
        if pose is None:
            raise RuntimeError("the robot has no pose yet")
        follow, desk, top = (f"{CAMERAS_PATH}/{n}" for n in HEADED_CAMS)
        eye, target = self.chase.update(pose)
        self.paths["follow"] = define_camera(self.stage, follow, eye, target, CHASE[4], self.res)
        self._follow_op = UsdGeom.Xformable(self.stage.GetPrimAtPath(follow)).GetOrderedXformOps()[0]
        d_eye, d_target, d_focal = start_view(pose)
        self.paths["desk"] = define_camera(self.stage, desk, d_eye, d_target, d_focal, self.res)
        if self.rects:
            v = overhead_view(self.rects, self.res[0] / self.res[1])
            self.paths["overhead"] = define_camera(self.stage, top, v.eye, v.target, v.focal, self.res, near=v.near,
                                                   far=v.far, aperture=v.aperture)
        self.use("follow")
        log.info("headed window: follow camera %s (also %s)", follow,
                 ", ".join(p for k, p in self.paths.items() if k != "follow"))

    def use(self, name: str) -> None:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = self.paths[name]

    def update(self) -> None:
        """Once a frame: the follow camera to the robot's latest pose (smoothed)."""
        if self._follow_op is None:
            return
        pose = self.robot_pose()
        if pose is None:
            return
        self._follow_op.Set(look_at(*self.chase.update(pose)))
        self.updates += 1


class Recorder:
    """The writers, the preview and the per-frame step. Built after the host's start-up (the cameras are authored once
    the timeline is playing, like the robot's: they never touch physics or the NavMesh)."""

    def __init__(self, cfg: RecordConfig, app, timeline, stage, rects: Sequence[tuple], robot_pose=None, clock=None):
        self.cfg, self.app, self.timeline, self.stage = cfg, app, timeline, stage
        self.rects, self.robot_pose = list(rects), robot_pose
        self.clock = clock  # the run's sim time (svc.t); stamps video frame 0 so voices can be placed on the video
        self.t_start: Optional[float] = None
        self.frames = 0  # frames rendered for the writers
        self.sim_frames = 0  # frames the timeline advanced while the recorder was in charge
        self.render_s = 0.0
        self.encoded = 0
        self.follow_frames = 0
        self._framing = "follow"
        self._rep = self._enc = self._an_top = self._an_follow = self._follow_xf = None
        self.follow_dir: Optional[Path] = None
        self._last_follow = None
        self._sink: Optional[JpegSink] = None
        self._stride = preview_stride(cfg.fps, cfg.preview_hz)
        self._last_rgba = None
        self._held = False

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
        self._framing = follow_framing()
        eye, target = follow_view(pose, self._framing)
        cam = define_camera(self.stage, f"{CAMERAS_PATH}/follow", eye, target, FRAMINGS[self._framing][4], cfg.res)
        from pxr import UsdGeom

        self._follow_xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(cam))
        self.follow_dir = cfg.video_dir / FOLLOW_DIR
        shutil.rmtree(self.follow_dir, ignore_errors=True)
        self.follow_dir.mkdir(parents=True)
        self._an_follow = rep.AnnotatorRegistry.get_annotator("rgb")
        self._an_follow.attach([rep.create.render_product(cam, cfg.res, name="wz_follow")])

    # ---- one frame ----
    def step(self) -> None:
        """Instead of ``app.update()``: one fixed frame, rendered for the writers when the timeline is running."""
        if not self.timeline.is_playing():
            self._held = True
            self.app.update()  # held: nothing moves, so nothing is recorded
            return
        self.sim_frames += 1
        if self._held:
            # the first frame after a hold renders nothing new; the clip repeats its last frame
            self._held = False
            self.app.update()
            if self.cfg.record:
                import numpy as np

                self._emit(np)
            return
        if not self.cfg.record:
            # preview alone: plain updates (the robot's camera writers fire on them; an orchestrator step only now and
            # then would never line up with its 5 Hz tick), and the overhead annotator is read every few frames
            self.app.update()
            if self.sim_frames % self._stride == 0:
                self._preview_only_frame()
            return
        if self._follow_xf is not None:
            pose = self.robot_pose()
            if pose is not None:
                self._follow_xf.GetOrderedXformOps()[0].Set(look_at(*follow_view(pose, self._framing)))
        t0 = time.monotonic()
        self.app.update()  # one fixed frame; the annotators fill as the timeline plays (capture on play)
        import numpy as np

        img = np.asarray(self._an_top.get_data())
        if img.size:
            self._last_rgba = img
        if self._an_follow is not None:
            fimg = np.asarray(self._an_follow.get_data())
            if fimg.size:
                self._last_follow = fimg
        self.render_s += time.monotonic() - t0
        self.frames += 1
        self._emit(np)
        if self._sink is not None and self._last_rgba is not None and self.sim_frames % self._stride == 0:
            self._sink.write(self._last_rgba[::preview_scale(self.cfg.res), ::preview_scale(self.cfg.res)])

    def _emit(self, np) -> None:
        """One video frame for this sim frame: a renderer that is late repeats the last image."""
        if self._last_rgba is not None:
            if self.t_start is None and self.clock is not None:
                self.t_start = float(self.clock())  # video frame 0 starts here: the service steps after this frame
            self._enc.encode_next_frame_from_buffer(np.ascontiguousarray(self._last_rgba), *self.cfg.res)
            self.encoded += 1
        if self._last_follow is not None:
            from PIL import Image

            # the follow annotator can still be empty for the first frames: the first image it gives is written for
            # each of them, so follow.mp4 and overhead.mp4 have the same frames and start together
            img = Image.fromarray(self._last_follow[..., :3])
            while True:
                img.save(self.follow_dir / f"f_{self.follow_frames:06d}.jpg", format="JPEG", quality=FOLLOW_JPEG_QUALITY)
                self.follow_frames += 1
                if self.follow_frames >= self.encoded:
                    break

    def _preview_only_frame(self) -> None:
        import numpy as np

        t0 = time.monotonic()
        img = np.asarray(self._an_top.get_data())
        self.render_s += time.monotonic() - t0
        if img.size and self._sink is not None:
            self.frames += 1
            self._sink.write(img)

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
        self._enc.finalize_encoding()
        self._fix_header(cfg.overhead_mp4, self.encoded)  # the encoder was told no length at the start
        stats["overhead_frames"] = self.encoded
        stats["t_start_s"] = round(self.t_start or 0.0, 4)
        if self._an_follow is not None:
            t0 = time.monotonic()
            stats["follow_frames"] = self._encode_follow()
            stats["follow_encode_s"] = round(time.monotonic() - t0, 1)
        try:  # writer schedules for frames that will never come (the robot camera's 5 Hz) would stall Kit's shutdown ~50 s
            self._rep.orchestrator.stop()
        except Exception as exc:
            log.warning("could not stop the Replicator orchestrator: %s", exc)
        self._write_timing(stats)
        log.info("recording done: %s", stats)
        return stats

    def _write_timing(self, stats: dict) -> None:
        """video/timing.json: the run time of video frame 0 and the frame counts, for the voice track."""
        import json

        try:
            (self.cfg.video_dir / TIMING_FILE).write_text(json.dumps({
                "fps": self.cfg.fps, "t_start_s": stats.get("t_start_s", 0.0),
                "overhead_frames": stats.get("overhead_frames", 0), "follow_frames": stats.get("follow_frames", 0)}),
                encoding="utf-8")
        except OSError as exc:
            log.warning("could not write %s: %s", TIMING_FILE, exc)

    def _encode_follow(self) -> int:
        cfg = self.cfg
        folder = self.follow_dir
        files = sorted(folder.glob("f_*.jpg"))
        if not files:
            log.warning("no follow-camera frames were written")
            return 0
        self._enc.start_encoding(video_filename=str(cfg.follow_mp4), framerate=float(cfg.fps), nframes=len(files),
                                 overwrite_video=True)
        for p in files:
            self._enc.encode_next_frame_from_file(str(p))
        self._enc.finalize_encoding()
        self._fix_header(cfg.follow_mp4, len(files))
        shutil.rmtree(folder, ignore_errors=True)
        return len(files)

    def _fix_header(self, mp4: Path, frames: int) -> None:
        if not mp4fix.set_duration(mp4, frames, self.cfg.fps):
            log.warning("could not set the header duration of %s: players may show a wrong length", mp4)
