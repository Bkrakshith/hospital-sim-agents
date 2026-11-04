"""Recording config and geometry (pure): resolution parsing, the writer config, the overhead and follow cameras,
the preview JPEG sink. The Kit side is covered by tests/isaac/test_isaac_record_smoke.py."""
import math
from pathlib import Path

import pytest

from wardzero_agents.isaac_host import recording as R
from wardzero_agents.isaac_host.app import parse_args


def test_parse_res():
    assert R.parse_res("1280x720") == (1280, 720)
    assert R.parse_res("1920X1080") == (1920, 1080)
    for bad in ("1280", "axb", "1281x720", "1280x721", "100x100", "9000x5000", "-1280x720", "1280x720x3", ""):
        with pytest.raises(ValueError):
            R.parse_res(bad)


def host_args(*extra):
    return parse_args(["--run-dir", "/tmp/r", "--scenario", "s.yaml", *extra])


def test_cli_defaults_to_720p_and_accepts_1080p():
    assert host_args().record_res == (1280, 720)
    assert host_args("--record-res", "1920x1080").record_res == (1920, 1080)
    with pytest.raises(SystemExit):
        host_args("--record-res", "big")


def test_record_config_from_args(tmp_path):
    cfg = R.RecordConfig.from_args(host_args("--record", "--record-res", "1920x1080", "--fps", "24"), tmp_path)
    assert cfg.record and not cfg.preview and cfg.res == (1920, 1080) and cfg.fps == 24
    assert cfg.video_dir == tmp_path / "video"
    assert cfg.overhead_mp4 == tmp_path / "video" / "overhead.mp4" and cfg.follow_mp4 == tmp_path / "video" / "follow.mp4"
    assert R.RecordConfig.from_args(host_args(), tmp_path) is None  # neither recording nor previewing


def test_preview_only_on_a_headless_run(tmp_path):
    cfg = R.RecordConfig.from_args(host_args("--preview"), tmp_path)
    assert cfg.preview and not cfg.record and cfg.preview_path == tmp_path / "preview" / "latest.jpg"
    assert cfg.preview_hz == 5
    assert R.RecordConfig.from_args(host_args("--preview", "--headed"), tmp_path) is None  # a window shows it live


def test_preview_stride_and_scale():
    assert R.preview_stride(30, 5) == 6 and R.preview_stride(24, 5) == 5 and R.preview_stride(4, 5) == 1
    assert R.preview_scale((1280, 720)) == 2 and R.preview_scale((1920, 1080)) == 3 and R.preview_scale((640, 360)) == 1
    assert R.preview_size((1920, 1080)) == (640, 360) and R.preview_size((1280, 720)) == (640, 360)


def test_overhead_view_frames_every_focus_rect_at_both_resolutions():
    rects = {"Lobby": (-12.0, -3.5, 11.2, 9.0), "C3": (11.2, -4.5, 27.5, 29.0)}
    for res in ((1280, 720), (1920, 1080)):
        v = R.overhead_view(list(rects.values()), res[0] / res[1])
        cx, cy = (-12 + 27.5) / 2, (-4.5 + 29) / 2
        assert (v.eye[0], v.eye[1]) == pytest.approx((cx, cy)) and v.target[2] == 0.0
        half_w = v.eye[2] * v.aperture / v.focal / 2  # what the camera sees on the floor
        half_h = half_w * res[1] / res[0]
        assert half_w >= (27.5 + 12) / 2 and half_h >= (29 + 4.5) / 2  # nothing of the rooms is cut off
        assert v.near == pytest.approx(v.eye[2] - R.CEILING_CUT_M) and v.near > 0  # the ceiling is clipped away
        assert v.far > v.eye[2]


def test_overhead_view_of_a_small_room_stays_close():
    v = R.overhead_view([(0.0, 0.0, 6.0, 4.0)], 16 / 9)
    assert v.eye[2] < 8.0 and v.eye[2] > R.CEILING_CUT_M + 0.5


def test_overhead_view_needs_a_rect():
    with pytest.raises(ValueError):
        R.overhead_view([], 16 / 9)


def test_follow_view_sits_in_front_and_above_the_robot_looking_at_it():
    eye, target = R.follow_view((2.0, 3.0, 0.0))  # facing +x
    assert target == pytest.approx((2.0, 3.0, R.FOLLOW_TARGET_Z))
    assert eye[0] > 2.0 and eye[2] > R.FOLLOW_TARGET_Z  # ahead of the robot (it looks +x) and above its chest
    assert math.hypot(eye[0] - 2.0, eye[1] - 3.0) == pytest.approx(math.hypot(R.FOLLOW_AHEAD_M, R.FOLLOW_SIDE_M))
    eye2, _ = R.follow_view((2.0, 3.0, math.pi / 2))  # facing +y: ahead is +y
    assert eye2[1] > 3.0
    eye3, _ = R.follow_view((2.0, 3.0, -math.pi / 2))  # the reception G1 faces -y
    assert eye3[1] < 3.0


def test_desk_framing_is_wider_and_higher_and_unknown_names_fall_back(monkeypatch):
    base_eye, _ = R.follow_view((2.0, 3.0, -math.pi / 2))
    eye, target = R.follow_view((2.0, 3.0, -math.pi / 2), "desk")
    assert target == pytest.approx((2.0, 3.0, R.FRAMINGS["desk"][3]))
    assert eye[1] < base_eye[1] < 3.0 and eye[2] > base_eye[2]  # further in front of the reception G1, and higher
    assert R.follow_view((2.0, 3.0, 0.0), "nonsense") == R.follow_view((2.0, 3.0, 0.0))
    monkeypatch.delenv("WARDZERO_FOLLOW_FRAMING", raising=False)
    assert R.follow_framing() == "follow"
    monkeypatch.setenv("WARDZERO_FOLLOW_FRAMING", "Desk")
    assert R.follow_framing() == "desk"
    monkeypatch.setenv("WARDZERO_FOLLOW_FRAMING", "bogus")
    assert R.follow_framing() == "follow"


def test_jpeg_sink_writes_atomically_and_replaces(tmp_path):
    np = pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    sink = R.JpegSink(tmp_path / "preview" / "latest.jpg")
    frame = np.zeros((360, 640, 4), dtype=np.uint8)
    frame[..., 0] = 200
    sink.write(frame)
    p = tmp_path / "preview" / "latest.jpg"
    img = PIL.open(p)
    assert img.size == (640, 360) and img.format == "JPEG"
    sink.write(frame[:180])  # a replacement, never a partial file; no temp file is left behind
    assert PIL.open(p).size == (640, 180)
    assert [f.name for f in p.parent.iterdir()] == ["latest.jpg"]
    sink.clear()
    assert not p.exists()


def test_writer_settings():
    cfg = R.RecordConfig(res=(1920, 1080), fps=30, record=True)
    assert cfg.frames_to_seconds(300) == pytest.approx(10.0)
    assert R.encoder_args(cfg) == {"framerate": 30.0, "nframes": 0, "overwrite_video": True}


# ---- the per-frame step, with a fake Kit -------------------------------------------------------
class FakeKit:
    def __init__(self, playing=True):
        self.playing, self.log = playing, []
        self.orchestrator = self  # stands in for rep and rep.orchestrator

    def is_playing(self):
        return self.playing

    def update(self):
        self.log.append("update")

    def step(self, **kw):  # an orchestrator step: recording must not use it (it crashed the Behavior System)
        raise AssertionError("the recorder stepped the Replicator orchestrator")

    def get_data(self):
        np = pytest.importorskip("numpy")
        return np.full((720, 1280, 4), 9, dtype=np.uint8)

    def encode_next_frame_from_buffer(self, buf, w, h):
        assert buf.flags["C_CONTIGUOUS"] and (w, h) == (1280, 720) and buf.shape == (720, 1280, 4)
        self.log.append("encode")


def recorder(kit, tmp_path, **kw):
    cfg = R.RecordConfig(video_dir=tmp_path / "video", preview_path=tmp_path / "preview" / "latest.jpg", **kw)
    rec = R.Recorder(cfg, kit, kit, None, [(0, 0, 10, 6)])
    rec._rep, rec._an_top, rec._enc = kit, kit, kit
    if cfg.preview:
        rec._sink = R.JpegSink(cfg.preview_path)
    return rec


def test_recorded_frames_render_and_encode_only_while_the_timeline_runs(tmp_path):
    pytest.importorskip("numpy")
    kit = FakeKit()
    rec = recorder(kit, tmp_path, record=True)
    rec.step()
    rec.step()
    kit.playing = False  # held (waiting for the agents, a decision pending): nothing moves, nothing is recorded
    rec.step()
    assert kit.log == ["update", "encode", "update", "encode", "update"]
    assert (rec.frames, rec.encoded, rec.sim_frames) == (2, 2, 2)


def test_the_frame_after_a_hold_is_a_plain_update_that_repeats_the_last_video_frame(tmp_path):
    """An orchestrator step on the frame the timeline resumes crashed the Behavior System in a lockstep run."""
    pytest.importorskip("numpy")
    kit = FakeKit()
    rec = recorder(kit, tmp_path, record=True)
    rec.step()
    kit.playing = False
    rec.step()
    kit.playing = True
    rec.step()  # resumed
    rec.step()
    assert kit.log == ["update", "encode", "update", "update", "encode", "update", "encode"]
    assert rec.encoded == 3 and rec.frames == 2  # the clip has a frame for every sim frame


def test_preview_alone_never_steps_the_orchestrator_and_reads_one_frame_in_six(tmp_path):
    """An orchestrator step now and then would never line up with the robot camera's 5 Hz tick: its images would stop."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    kit = FakeKit()
    rec = recorder(kit, tmp_path, preview=True)
    rec._an_top = type("A", (), {"get_data": staticmethod(lambda: __import__("numpy").full((360, 640, 4), 9, "uint8"))})()
    for _ in range(12):
        rec.step()
    assert kit.log.count("update") == 12 and "encode" not in kit.log
    assert rec.frames == 2 and (tmp_path / "preview" / "latest.jpg").stat().st_size > 0


def test_recording_and_preview_together_downscale_the_record_frame(tmp_path):
    np = pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    kit = FakeKit()
    rec = recorder(kit, tmp_path, record=True, preview=True)
    for _ in range(6):
        rec.step()
    assert kit.log.count("encode") == 6  # every frame to the video
    assert PIL.open(tmp_path / "preview" / "latest.jpg").size == (640, 360)  # a stride of the 1280x720 render


# ---- one frame, one robot tick, one time base ---------------------------------------------------
class Tl:
    def __init__(self, playing=True):
        self.playing, self.t = playing, 7.0

    def is_playing(self):
        return self.playing

    def set_current_time(self, t):
        self.t = t


def fake_host(playing=True, recorder=None):
    from types import SimpleNamespace

    from wardzero_agents.isaac_host import app as A
    log = []
    h = A.IsaacHost.__new__(A.IsaacHost)
    h.timeline = Tl(playing)
    h.app = SimpleNamespace(update=lambda: log.append("update"))
    h.robot = SimpleNamespace(fire=lambda: log.append("fire"))
    h.recorder = recorder
    return h, log


def test_the_robot_graph_fires_once_before_each_running_frame():
    h, log = fake_host()
    h.advance()
    h.advance()
    assert log == ["fire", "update", "fire", "update"]


def test_calibration_reads_isaacs_time_after_one_evaluation_and_a_robotless_run_has_none():
    from types import SimpleNamespace
    h, log = fake_host()
    h.robot = SimpleNamespace(fire=lambda: log.append("fire"), sim_time=lambda: 2.8333)
    h.startup = {}
    assert h.calibrate_robot_clock() == 2.8333 and log == ["fire", "update"] and h.startup["robot_clock_offset_s"] == 2.8333
    h.robot = None
    assert h.calibrate_robot_clock() == 0.0


def test_a_held_frame_fires_nothing():
    h, log = fake_host(playing=False)
    h.advance()
    assert log == ["update"]  # the timeline is paused: no new message, no repeated stamp


def test_the_recorder_steps_instead_of_update_but_the_graph_still_fires_once():
    from types import SimpleNamespace
    rec_log = []
    h, log = fake_host(recorder=SimpleNamespace(step=lambda: rec_log.append("step")))
    h.advance()
    assert log == ["fire"] and rec_log == ["step"]


def test_a_header_that_cannot_be_patched_is_warned_about(tmp_path, caplog):
    kit = FakeKit()
    rec = recorder(kit, tmp_path, record=True)
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"junk")
    with caplog.at_level("WARNING"):
        rec._fix_header(bad, 10)
    assert "header duration" in caplog.text


def test_the_follow_camera_writes_one_jpeg_per_sim_frame_and_repeats_when_late(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    kit = FakeKit()
    rec = recorder(kit, tmp_path, record=True)
    rec._an_follow, rec.follow_dir = kit, tmp_path / "video" / "follow_frames"
    rec.follow_dir.mkdir(parents=True)
    for _ in range(4):
        rec.step()
    kit.playing = False
    rec.step()
    kit.playing = True
    rec.step()  # the resume frame repeats the last image: the follow clip is as long as the overhead one
    assert sorted(p.name for p in rec.follow_dir.iterdir()) == [f"f_{i:06d}.jpg" for i in range(5)]
    assert rec.follow_frames == rec.encoded == 5


def test_a_follow_camera_that_starts_late_is_backfilled_so_both_clips_start_together(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    class LateFollow(FakeKit):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_data(self):  # empty for the first two frames, like an annotator that has not filled yet
            self.calls += 1
            return np.zeros((0,), np.uint8) if self.calls <= 2 else super().get_data()

    top, follow = FakeKit(), LateFollow()
    rec = recorder(top, tmp_path, record=True)
    rec._an_follow, rec.follow_dir = follow, tmp_path / "video" / "follow_frames"
    rec.follow_dir.mkdir(parents=True)
    for _ in range(5):
        rec.step()
    assert rec.follow_frames == rec.encoded == 5  # the leading frames repeat the first image
    assert len(list(rec.follow_dir.iterdir())) == 5


def test_frame_zero_is_stamped_with_the_run_clock_and_written_to_timing_json(tmp_path):
    import json

    kit, now = FakeKit(), [12.5]
    rec = recorder(kit, tmp_path, record=True)
    rec.clock = lambda: now[0]
    rec._enc.finalize_encoding = lambda: None
    rec._rep.orchestrator.stop = lambda: None
    rec.cfg.video_dir.mkdir(parents=True)
    kit.playing = False
    rec.step()  # held: no frame, no stamp
    assert rec.t_start is None
    kit.playing = True
    now[0] = 11.0
    rec.step()  # the resume frame has no image yet, so it is not video frame 0
    assert rec.t_start is None and rec.encoded == 0
    now[0] = 12.5
    rec.step()
    now[0] = 99.0
    rec.step()
    assert rec.t_start == 12.5
    rec._fix_header = lambda *a: None
    stats = rec.finish()
    t = json.loads((tmp_path / "video" / "timing.json").read_text())
    assert stats["t_start_s"] == 12.5 and t["t_start_s"] == 12.5 and t["fps"] == 30 and t["overhead_frames"] == 2 == rec.encoded
