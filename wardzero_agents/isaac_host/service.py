"""HostLoop: the Isaac host's frame loop around a SimService.

Each frame:
1. run the ROS callbacks queued since the last frame (on this thread, like the stub node's loop);
2. hold or release the timeline: while the service is held (waiting for the agents and director,
   paused, a lockstep decision pending, the scenario over) the timeline is paused, never stopped
   (a stop/play would make the Behavior System track a robot's links again, S1b);
3. ``app.update()``: Kit steps physics, the people and rendering by one fixed frame;
4. advance the service by exactly the sim time the timeline moved (the physics step, not wall time),
   which runs the StubWorld logic and reads the Isaac bodies back;
5. publish the outputs (wz.v1 on the private domain, /clock on every domain);
6. sleep so the sim runs no faster than time_scale x real time (0: as fast as it goes).

Frame times and VRAM are recorded to the run's ``host`` stream every ``report_every_s`` of sim time
and once at the end.
"""
from __future__ import annotations

import logging
import statistics
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger("wardzero.isaac.host")

REPORT_EVERY_S = 5.0  # sim seconds between host stream records
STOP_GRACE_S = 0.5  # wall seconds to let the stop message reach the other nodes
MAX_FRAME_SAMPLES = 20000


def nvidia_smi_mib() -> Optional[int]:
    """Whole-GPU used memory in MiB (the desktop included), or None without nvidia-smi."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      text=True, timeout=5)
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class VramSampler:
    """Samples VRAM in a background thread (nvidia-smi takes ~50 ms; never on the frame thread)."""

    def __init__(self, read: Callable[[], Optional[int]] = nvidia_smi_mib, period_s: float = 1.0):
        self.read, self.period_s = read, period_s
        self.last: Optional[int] = None
        self.peak: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample(self) -> Optional[int]:
        v = self.read()
        if v is not None:
            self.last = v
            self.peak = v if self.peak is None else max(self.peak, v)
        return v

    def start(self) -> "VramSampler":
        def run():
            while not self._stop.is_set():
                self.sample()
                self._stop.wait(self.period_s)
        self._thread = threading.Thread(target=run, name="vram", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()


class HostLoop:
    def __init__(self, svc, sim_ros, app, timeline, bridges: Iterable[Any] = (), time_scale: float = 1.0,
                 wall: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 vram: Optional[Callable[[], Optional[int]]] = None,
                 vram_peak: Optional[Callable[[], Optional[int]]] = None, report_every_s: float = REPORT_EVERY_S):
        self.svc, self.sim_ros, self.app, self.timeline = svc, sim_ros, app, timeline
        self.bridges = list(bridges)
        self.time_scale = max(0.0, float(time_scale))
        self.wall, self.sleep = wall, sleep
        self._vram_read = vram  # the current VRAM (MiB); the host passes a VramSampler's last sample
        self._vram_peak_read = vram_peak  # its peak; without one the loop keeps the peak of what it read
        self.vram_peak: Optional[int] = None
        self.report_every_s = report_every_s
        self._last_time = float(timeline.get_current_time())
        self._ms: list[float] = []
        self.frames = 0
        self._next_report = report_every_s
        self._started = wall()

    # ---- one frame ----------------------------------------------------------------------------
    def frame(self):
        start = self.wall()
        for b in self.bridges:
            b.drain()
        held = self.svc.held
        playing = self.timeline.is_playing()
        if held and playing:
            self.timeline.pause()
        elif not held and not playing:
            self.timeline.play()
        self.app.update()
        now = float(self.timeline.get_current_time())
        dt = max(0.0, now - self._last_time)
        self._last_time = now
        out = self.svc.step(dt)
        self.sim_ros.publish(out)
        spent = self.wall() - start
        self.frames += 1
        if len(self._ms) < MAX_FRAME_SAMPLES:
            self._ms.append(spent * 1000.0)
        if self.time_scale > 0 and dt > 0:
            budget = dt / self.time_scale
            if spent < budget:
                self.sleep(budget - spent)
        if self.svc.t >= self._next_report:
            self._report("progress")
            self._next_report = self.svc.t + self.report_every_s
        return out

    # ---- the run ------------------------------------------------------------------------------
    def run(self, stop_requested: Callable[[], bool], ready_timeout: float) -> int:
        """Frames until the scenario stops (0), a signal (0) or the others never come up (3)."""
        code = 0
        last_wait_log = self.wall()
        try:
            while not stop_requested():
                out = self.frame()
                if out.stop:
                    log.info("scenario over at t=%.1f s; stopping the run", out.clock)
                    self.sim_ros.publish_stop()
                    self.sleep(STOP_GRACE_S)
                    break
                if self.svc.waiting:
                    now = self.wall()
                    if now - self._started > ready_timeout:
                        log.error("gave up waiting for %s after %.0f s", ", ".join(sorted(self.svc.waiting)),
                                  ready_timeout)
                        code = 3
                        break
                    if now - last_wait_log > 10.0:
                        log.info("waiting for %s", ", ".join(sorted(self.svc.waiting)))
                        last_wait_log = now
        finally:
            self._report("final")
        return code

    # ---- stats --------------------------------------------------------------------------------
    def _vram(self) -> Optional[int]:
        v = self._vram_read() if self._vram_read is not None else None
        if v is not None:
            self.vram_peak = v if self.vram_peak is None else max(self.vram_peak, v)
        return v

    def stats(self) -> dict:
        ms = sorted(self._ms[30:] or self._ms)  # skip the warm-up frames when there are enough
        def pct(q):
            return round(ms[min(len(ms) - 1, int(q * len(ms)))], 2) if ms else None
        mean = statistics.fmean(ms) if ms else None
        frame_dt = None
        if self.svc.t > 0 and self.frames:
            frame_dt = self.svc.t / self.frames
        return {"t": round(self.svc.t, 3), "frames": self.frames,
                "ms_median": pct(0.5), "ms_p90": pct(0.9), "ms_mean": round(mean, 2) if mean is not None else None,
                "rtf": round(frame_dt * 1000.0 / mean, 3) if mean and frame_dt else None}

    def _report(self, kind: str) -> None:
        rec = {"kind": kind, **self.stats(), "vram_mib": self._vram()}
        peak = self._vram_peak_read() if self._vram_peak_read is not None else None
        rec["vram_peak_mib"] = peak if peak is not None else self.vram_peak
        try:
            self.svc.runlog.write("host", rec)
        except OSError as exc:
            log.warning("could not write the host stream: %s", exc)
        log.info("host %s: t=%.1f s, %d frames, %s ms/frame median, %s ms p90, VRAM %s MiB (peak %s)", kind,
                 rec["t"], rec["frames"], rec["ms_median"], rec["ms_p90"], rec["vram_mib"], rec["vram_peak_mib"])
