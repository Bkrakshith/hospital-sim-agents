"""Isaac smoke test for recording (I5): the host records 10 s of sim time, headless, with 3 people and the G1 at reception.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_isaac_record_smoke.py -q -s
    WARDZERO_ISAAC=1 WARDZERO_ISAAC_HEADED=1 ...   # also the headed run (needs a display)

Runs the real host (``scripts/isaac_host.sh``, ``--record --wait-for none``) with a 10 s scenario and asserts what the
brief asks: both MP4s exist with durations of 10 s +- 0.5 s (ffprobe), the run streams are non-empty and carry sim time,
and ``robot_bag/`` has a ``metadata.yaml`` and /robot/camera/bbox_3d messages. A few frames of each MP4 go to
artifacts/screens/isaac_record/ (git-ignored). Needs Isaac Sim 6.0 at ~/isaacsim and ros2 (``env.sh``); starts one Kit
process. A 10 s clip takes a few minutes: rendering for the writers costs about 300 ms a frame.
"""
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]

ROOT = Path(__file__).resolve().parents[2]
HOST_SH = ROOT / "scripts" / "isaac_host.sh"
OUT = ROOT / "artifacts" / "screens" / "isaac_record"
BOOT_TIMEOUT_S = 1500.0
ROBOT_DOMAIN, PRIVATE_DOMAIN = 43, 44
SECONDS = 10.0
BAG_TOPIC = "/robot/camera/bbox_3d"


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_exe():
    return shutil.which("ffprobe") or next(iter(sorted(Path.home().glob("miniconda3/envs/*/bin/ffprobe"))), None)


def probe(mp4: Path) -> dict:
    """Duration in seconds and size, with ffprobe (or ffmpeg's banner when there is no ffprobe)."""
    exe = ffprobe_exe()
    if exe:
        out = subprocess.check_output([str(exe), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,nb_frames:format=duration", "-of", "json", str(mp4)])
        d = json.loads(out)
        s = d["streams"][0]
        return {"duration": float(d["format"]["duration"]), "w": int(s["width"]), "h": int(s["height"]),
                "frames": int(s.get("nb_frames") or 0)}
    err = subprocess.run([ffmpeg_exe(), "-i", str(mp4)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    w, hh = re.search(r"(\d{3,4})x(\d{3,4})", err).groups()
    return {"duration": int(h) * 3600 + int(m) * 60 + float(s), "w": int(w), "h": int(hh), "frames": 0}


def grab_frames(mp4: Path, stem: str, secs=(1.0, 5.0, 9.0)) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for t in secs:
        png = OUT / f"{stem}_{t:.0f}s.png"
        subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", str(png)],
                       check=True)
        out.append(png)
    return out


def write_scenario(tmp: Path) -> Path:
    cfg = tmp / "config"
    shutil.copytree(ROOT / "config", cfg)
    (cfg / "storyboards" / "empty.yaml").write_text("id: empty\nbeats: []\n")
    data = yaml.safe_load((cfg / "scenarios" / "waiting_area_v1.yaml").read_text())
    data["cast"] = [c for c in data["cast"] if c["id"] in ("P1", "P2", "P4")]
    data.update(id="isaac_record_smoke", static_patients=False, duration_min=SECONDS / 60.0, objects=[],
                robot={"enabled": True}, storyboard="../storyboards/empty.yaml")
    q = cfg / "scenarios" / "isaac_record_smoke.yaml"
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    return q


@pytest.fixture
def isaac_free():
    if not (Path.home() / "isaacsim" / "python.sh").exists():
        pytest.skip("no Isaac Sim at ~/isaacsim")
    if subprocess.run(["pgrep", "-x", "kit"], capture_output=True).returncode == 0:
        pytest.skip("another Isaac (kit) process is running")
    if not (ROOT.parent / "env.sh").exists() or shutil.which("bash") is None:
        pytest.skip("env.sh (system ROS) is missing")


def run_host(tmp_path: Path, res: str, headed: bool = False, flags=("--record",), watch=None) -> Path:
    """The real host on the 10 s scenario. ``watch(run_dir)`` is called every half second while it runs."""
    scen = write_scenario(tmp_path)
    run = tmp_path / "run"
    log = open(tmp_path / "host.log", "w")
    argv = [str(HOST_SH), "--run-dir", str(run), "--scenario", str(scen), "--headed" if headed else "--headless",
            *flags, "--record-res", res, "--wait-for", "none", "--rooms", "Lobby,AE_Waiting,Main_LiftLobby",
            "--fire", "none", "--private-domain", str(PRIVATE_DOMAIN), "--robot-domain", str(ROBOT_DOMAIN)]
    t0 = time.time()
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while proc.poll() is None and time.time() - t0 < BOOT_TIMEOUT_S:
            if watch is not None:
                watch(run)
            time.sleep(0.5)
        code = proc.wait(timeout=1)
        assert code == 0, (tmp_path / "host.log").read_text()[-4000:]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    print(f"\nhost wall time {time.time() - t0:.0f} s")
    return run


def stream(run: Path, name: str) -> list:
    f = run / f"{name}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []


def bag_messages(bag: Path, topic: str) -> int:
    db = next(iter(sorted(bag.glob("*.db3"))), None)
    assert db is not None, list(bag.iterdir())
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("select count(*) from messages m join topics t on t.id = m.topic_id where t.name = ?",
                          (topic,)).fetchone()
    finally:
        con.close()
    return row[0]


def bag_stamps(bag: Path, topic: str) -> list:
    """Header stamps (s) of every message of a /clock or JointState topic, read from the CDR bytes (no ROS needed)."""
    import struct
    db = next(iter(sorted(bag.glob("*.db3"))))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("select m.data from messages m join topics t on t.id = m.topic_id where t.name = ? "
                           "order by m.id", (topic,)).fetchall()
    finally:
        con.close()
    out = []
    for (data,) in rows:
        if bytes(data[:4]) == b"\x28\xb5\x2f\xfd":  # the bag is zstd per message: the CDR is inside
            import zstandard
            data = zstandard.ZstdDecompressor().decompress(bytes(data))
        sec, nsec = struct.unpack("<iI", bytes(data[4:12]))
        out.append(sec + nsec * 1e-9)
    return out


def check_one_time_base(run: Path, bag: Path) -> dict:
    """What the robot hears is on one time base: /clock (published once per frame from svc.t plus Isaac's start-up offset),
    tf-side joint stamps and camera_info stamps agree, never repeat and never go back. The run's own streams stay on svc.t."""
    run_t = [r for r in stream(run, "host") if r.get("kind") == "final"][-1]["t"]  # the service's time at the end
    off = next(r for r in stream(run, "host") if r.get("kind") == "startup")["robot_clock_offset_s"]
    frame = 1 / 30
    clock = bag_stamps(bag, "/clock")
    assert len(clock) >= SECONDS * 25, len(clock)
    # strictly increasing (the very last message repeats the final time: the scenario's end holds the clock at its length)
    assert all(b > a for a, b in zip(clock[:-1], clock[1:-1])) and clock[-1] >= clock[-2], "/clock repeats or goes back"
    assert 0 <= clock[0] - off <= 2 * frame and abs(clock[-1] - off - run_t) <= frame + 1e-6, (clock[0], clock[-1], off, run_t)
    out = {"/clock": {"n": len(clock), "first_minus_offset": round(clock[0] - off, 4),
                      "last_minus_offset_minus_run_t": round(clock[-1] - off - run_t, 4)}, "run_t": run_t, "offset": off}
    near = lambda x: min(abs(x - c) for c in clock)  # noqa: E731
    joints = bag_stamps(bag, "/robot/joint_states")
    assert len(joints) >= SECONDS * 25 and all(b > a for a, b in zip(joints, joints[1:])), "joint stamps repeat"  # once a frame
    assert max(near(x) for x in joints[:-1]) <= 1e-6 and near(joints[-1]) <= frame + 1e-6, "joint stamps are not /clock's"
    info = bag_stamps(bag, "/robot/camera/camera_info")
    assert len(info) >= SECONDS * 4 and all(b > a for a, b in zip(info, info[1:]))
    assert max(near(x) for x in info) <= frame + 1e-6, "camera stamps are on another time base"
    out["joint_n"], out["camera_info_n"] = len(joints), len(info)
    return out


def bag_mb(bag: Path) -> float:
    return sum(f.stat().st_size for f in bag.iterdir()) / 1e6


def _bag_topics(bag: Path) -> set:
    db = next(iter(sorted(bag.glob("*.db3"))))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("select name from topics")}
    finally:
        con.close()


def check_recording(run: Path, res: str) -> dict:
    w, h = (int(v) for v in res.split("x"))
    out = {}
    for name in ("overhead", "follow"):
        mp4 = run / "video" / f"{name}.mp4"
        assert mp4.is_file() and mp4.stat().st_size > 50_000, mp4
        p = probe(mp4)
        assert abs(p["duration"] - SECONDS) <= 0.5, (name, p)  # sim time, not wall time
        assert (p["w"], p["h"]) == (w, h), (name, p)
        out[name] = p
        for png in grab_frames(mp4, f"{name}_{res}"):
            assert png.stat().st_size > 20_000, png
    assert not (run / "video" / "follow_frames").exists()  # the temporary frames are gone
    # the run streams are non-empty and their timestamps are sim time (a slow encoder slows wall time, not the story)
    status = stream(run, "status")
    assert status and stream(run, "robot_pose") and stream(run, "host")
    ts = [r["t"] for r in status if "t" in r]
    assert max(ts) == pytest.approx(SECONDS, abs=0.5) and min(ts) < 1.0, (min(ts), max(ts))
    host = [r for r in stream(run, "host") if r.get("kind") == "recording"]
    assert host and host[-1]["frames"] >= SECONDS * 30 * 0.9, host
    meta_frames = host[-1]["overhead_frames"]
    assert abs(meta_frames / 30.0 - SECONDS) <= 0.5, host
    # the robot-domain bag is finalised and carries the detection link
    bag = run / "robot_bag"
    assert (bag / "metadata.yaml").is_file(), list(bag.iterdir()) if bag.exists() else "no bag"
    assert bag_messages(bag, BAG_TOPIC) > 0
    topics = _bag_topics(bag)
    assert "/robot/camera/rgb" not in topics and "/robot/camera/depth" not in topics  # no images by default
    assert "/clock" in topics and not any(t.startswith("/wz") for t in topics)  # nothing private
    out["clock"] = check_one_time_base(run, bag)
    out["bag_mb"] = round(bag_mb(bag), 1)
    out["host"] = host[-1]
    return out


def test_headless_record_10s_720p(tmp_path, isaac_free):
    run = run_host(tmp_path, "1280x720")
    r = check_recording(run, "1280x720")
    print("720p:", json.dumps(r["host"]), "clock:", json.dumps(r["clock"]), "bag MB:", r["bag_mb"])
    assert not (run / "preview").exists()  # no --preview, no preview


def test_headless_record_10s_1080p(tmp_path, isaac_free):
    run = run_host(tmp_path, "1920x1080")
    r = check_recording(run, "1920x1080")
    print("1080p:", json.dumps(r["host"]), "bag MB:", r["bag_mb"])


@pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC_HEADED") != "1", reason="set WARDZERO_ISAAC_HEADED=1 (needs a display)")
def test_headed_record_10s(tmp_path, isaac_free):
    run = run_host(tmp_path, "1280x720", headed=True)
    check_recording(run, "1280x720")


def test_headless_preview_alone_writes_a_live_jpeg_and_removes_it_at_the_end(tmp_path, isaac_free):
    from PIL import Image
    seen = {}

    def watch(run):
        f = run / "preview" / "latest.jpg"
        try:
            data = f.read_bytes()
        except OSError:
            return
        seen[hash(data)] = data

    run = run_host(tmp_path, "1280x720", flags=("--preview",), watch=watch)
    assert len(seen) >= 2, "the preview never changed while the run went on"
    assert all(d[:2] == b"\xff\xd8" and d[-2:] == b"\xff\xd9" for d in seen.values())  # whole JPEGs, never half a frame
    import io
    assert Image.open(io.BytesIO(next(iter(seen.values())))).size == (640, 360)
    assert not (run / "preview" / "latest.jpg").exists() and not (run / "video").exists() and not (run / "robot_bag").exists()
    final = [r for r in stream(run, "host") if r.get("kind") == "final"][-1]
    print("preview only: ms/frame median", final["ms_median"], "p90", final["ms_p90"], "VRAM peak", final["vram_peak_mib"])
    assert max(r["t"] for r in stream(run, "status")) == pytest.approx(SECONDS, abs=0.5)


def test_record_with_preview_writes_both(tmp_path, isaac_free):
    from PIL import Image
    seen = []

    def watch(run):
        f = run / "preview" / "latest.jpg"
        if f.is_file():
            seen.append(f.read_bytes())

    run = run_host(tmp_path, "1280x720", flags=("--record", "--preview"), watch=watch)
    check_recording(run, "1280x720")
    import io
    assert seen and Image.open(io.BytesIO(seen[-1])).size == (640, 360)  # a stride of the 1280x720 frame


def test_the_clock_is_the_runs_own_time_without_recording_too(tmp_path, isaac_free):
    """No --record: a probe bag on the robot domain (the same ros2 bag the host uses) sees the same single time base."""
    from wardzero_agents.isaac_host import bag as B
    probe = B.BagRecorder(tmp_path / "probe", ROBOT_DOMAIN, private_domain=PRIVATE_DOMAIN, images=False)
    assert probe.start()
    time.sleep(3)
    try:
        run = run_host(tmp_path, "1280x720", flags=("--preview",))
    finally:
        probe.stop()
    assert probe.finalised
    r = check_one_time_base(run, probe.out_dir)
    print("no record:", json.dumps(r))
