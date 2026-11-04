"""The Isaac host's frame loop with a fake Kit app and timeline: the clock follows the physics step."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control
from wardzero_agents.isaac_host.app import parse_args
from wardzero_agents.isaac_host.bodies import IsaacBodies
from wardzero_agents.isaac_host.service import HostLoop
from wardzero_agents.ros.sim_service import SimService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
DT = 1.0 / 30.0


class FakeTimeline:
    def __init__(self):
        self.playing, self.time, self.calls = True, 0.0, []

    def is_playing(self):
        return self.playing

    def play(self):
        self.calls.append("play")
        self.playing = True

    def pause(self):
        self.calls.append("pause")
        self.playing = False

    def stop(self):  # the host must never call this
        raise AssertionError("the host stopped the timeline")

    def get_current_time(self):
        return self.time


class FakeApp:
    def __init__(self, tl):
        self.tl, self.updates = tl, 0

    def update(self):
        self.updates += 1
        if self.tl.playing:
            self.tl.time += DT


class FakeSimRos:
    def __init__(self):
        self.published, self.stops = [], 0

    def publish(self, out):
        self.published.append(out)

    def publish_stop(self):
        self.stops += 1


class Wall:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def make(tmp_path, wait_for=(), time_scale=1.0, **kw):
    wall = Wall()
    svc = SimService(load_scenario(V1), RunLog(tmp_path), wait_for=wait_for, bodies=IsaacBodies(), wall=wall)
    tl = FakeTimeline()
    loop = HostLoop(svc, FakeSimRos(), FakeApp(tl), tl, time_scale=time_scale, wall=wall, sleep=wall.sleep,
                    vram=lambda: 1234, **kw)
    return loop, svc, tl, wall


def test_clock_advances_by_the_physics_step_not_wall_time(tmp_path):
    loop, svc, tl, wall = make(tmp_path)
    for _ in range(30):
        loop.frame()
    assert svc.t == pytest.approx(30 * DT) and svc.t == pytest.approx(tl.time)
    assert loop.sim_ros.published[-1].clock == pytest.approx(svc.t)


def test_held_pauses_the_timeline_and_ready_plays_it(tmp_path):
    loop, svc, tl, wall = make(tmp_path, wait_for=("agents",))
    for _ in range(5):
        loop.frame()
    assert svc.t == 0.0 and tl.calls == ["pause"] and not tl.playing
    svc.ready("agents")
    for _ in range(3):
        loop.frame()
    assert tl.calls == ["pause", "play"] and svc.t == pytest.approx(3 * DT)


def test_pacing_holds_the_sim_to_the_time_scale(tmp_path):
    loop, svc, tl, wall = make(tmp_path, time_scale=2.0)
    loop.frame()
    assert wall.slept and wall.slept[-1] == pytest.approx(DT / 2.0)
    loop2, *_ , wall2 = make(tmp_path / "b", time_scale=0.0)  # 0: as fast as it goes
    loop2.frame()
    assert wall2.slept == []


def test_stop_on_the_control_topic_ends_the_run_cleanly(tmp_path):
    loop, svc, tl, wall = make(tmp_path)
    n = {"k": 0}

    def stop_after_10s():
        n["k"] += 1
        if svc.t >= 10.0:
            svc.control(Control(action="stop"))
        return False

    assert loop.run(stop_requested=stop_after_10s, ready_timeout=60.0) == 0
    assert svc.t == pytest.approx(10.0, abs=2 * DT) and loop.sim_ros.stops == 1
    assert "stop" not in tl.calls


def test_scenario_end_stops_the_run(tmp_path):
    loop, svc, tl, wall = make(tmp_path)
    svc.duration_s = 1.0
    assert loop.run(stop_requested=lambda: False, ready_timeout=60.0) == 0
    assert svc.t == pytest.approx(1.0) and loop.sim_ros.stops == 1


def test_a_signal_stops_without_the_stop_message(tmp_path):
    loop, svc, tl, wall = make(tmp_path)
    calls = {"n": 0}

    def after_three():
        calls["n"] += 1
        return calls["n"] > 3

    assert loop.run(stop_requested=after_three, ready_timeout=60.0) == 0
    assert loop.sim_ros.stops == 0 and svc.t == pytest.approx(3 * DT)


def test_ready_timeout(tmp_path):
    loop, svc, tl, wall = make(tmp_path, wait_for=("agents",), time_scale=0.0)

    def tick():
        wall.now += 1.0
        return False

    assert loop.run(stop_requested=tick, ready_timeout=5.0) == 3


def test_host_stream_records_frame_times_and_vram(tmp_path):
    loop, svc, tl, wall = make(tmp_path, report_every_s=0.5)
    svc.duration_s = 1.0
    loop.run(stop_requested=lambda: False, ready_timeout=60.0)
    recs = [json.loads(x) for x in (tmp_path / "host.jsonl").read_text().splitlines()]
    assert recs[-1]["kind"] == "final" and recs[-1]["frames"] >= 30
    assert {"t", "ms_median", "ms_p90", "vram_mib", "vram_peak_mib"} <= set(recs[-1])
    assert recs[-1]["vram_peak_mib"] == 1234
    assert any(r["kind"] == "progress" for r in recs)


# ---- the host's own flags -------------------------------------------------------------------
def test_host_flags():
    a = parse_args(["--run-dir", "r", "--scenario", "s.yaml"])
    assert not a.headed and not a.record and not a.full_site and a.rooms is None and a.fps == 30
    a = parse_args(["--run-dir", "r", "--scenario", "s.yaml", "--headed", "--record", "--full-site",
                    "--rooms", "Lobby,B1", "--private-domain", "7", "--robot-domain", "9", "--wait-for", "none"])
    assert a.headed and a.record and a.full_site and a.rooms == ["Lobby", "B1"]
    assert a.private_domain == 7 and a.robot_domain == 9 and a.wait_for == []
    with pytest.raises(SystemExit):
        parse_args(["--run-dir", "r", "--scenario", "s.yaml", "--headed", "--headless"])
    with pytest.raises(SystemExit):
        parse_args(["--run-dir", "r", "--scenario", "s.yaml", "--wait-for", "agents,bogus"])


def test_the_loop_advances_through_the_recorders_step_when_there_is_one(tmp_path):
    tl, calls = FakeTimeline(), []

    def advance():
        calls.append("advance")
        tl.time += DT

    class NoUpdate:
        def update(self):
            raise AssertionError("app.update must not run when a recorder advances the frame")

    loaded = load_scenario(V1)
    svc = SimService(loaded, RunLog(tmp_path), 1.0, True, [], bodies=IsaacBodies())
    ros = type("R", (), {"publish": lambda self, out: None})()
    HostLoop(svc, ros, NoUpdate(), tl, time_scale=0.0, advance=advance).frame()
    assert calls == ["advance"] and svc.t == pytest.approx(DT)


@pytest.mark.parametrize("time_scale", [0.0, 0.5, 1.0, 4.0])
def test_the_time_scale_only_paces_the_wall_so_video_time_is_always_sim_time(tmp_path, time_scale):
    """--record voices rely on run time = video time: --time-scale never changes how far a frame advances the clock."""
    loop, svc, tl, wall = make(tmp_path, time_scale=time_scale)
    for _ in range(30):
        loop.frame()
    assert svc.t == pytest.approx(30 * DT)
