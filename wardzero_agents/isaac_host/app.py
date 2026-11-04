"""The Isaac host process: Kit app, stage, physics, ROS, and the frame loop.

    scripts/isaac_host.sh --run-dir DIR --scenario FILE [--headed|--headless] [--record] [--full-site]
        [--rooms A,B] [--private-domain N] [--robot-domain M] [--time-scale F] [--lockstep]
        [--wait-for agents,director|none] [--ready-timeout 600] [--fps 30]

Start-up order (binding, from spike S1b; the order is what keeps a moving robot cheap next to the
Behavior System's people):
1. enable every extension (enabling one after a physics stage is open crashed Kit);
2. open the stage through the culling wrapper layer (``<run>/stage.usda``);
3. a CPU PhysX scene (``enableGPUDynamics=False``, MBP broadphase);
4. the baked collision USD;
5. the people (I2): the characters, the seat effectors and the lying patients are authored here, then the
   objects (I3): props and call lights (a fire appears when the world starts one);
6. bake the NavMesh;
7. one robot obstacle cylinder, collisions off;
8. ``timeline.play()``, then the characters' Behavior agents are bound to their controllers (I2);
9. the robots, in one authoring pass (I4).
Then the timeline is never stopped (only paused while the run is held) and the NavMesh is never
re-baked while robots are present.

ROS: Isaac's bundled rclpy, one explicit Context per domain. The private domain carries wz.v1 (the
same topics as ``stub_sim_node``, through ``SimRos``); /clock is published on the private and the
robot domain. The host never receives the Claude key: the launcher and this module drop it.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

from ..ros.sim_service import READY_TIMEOUT_S, WAIT_TOKENS
from . import stage as stage_mod

log = logging.getLogger("wardzero.isaac.app")

EXTENSIONS = (
    "isaacsim.ros2.bridge",  # the bundled Humble rclpy and the OmniGraph bridge nodes (I4)
    "isaacsim.replicator.agent.core",  # IRA: Behavior System, NavMesh, character commands (I2)
    "omni.flowusd",  # Flow fire and smoke (I3); nothing is simulated until a fire exists
)
KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WARDZERO_KEY_FILE")
CAMERA = "/OmniverseKit_Persp"
TEXTURE_MAX_HEADLESS = 1024  # pixels: the default for --headless runs (headed runs keep full textures); 0 turns it off


def max_mip_count(pixels: int) -> int:
    """The RTX setting /rtx-transient/resourcemanager/maxMipCount for a largest texture side (64 -> 7, 1024 -> 11)."""
    if pixels < 64 or pixels > 32768 or pixels & (pixels - 1):
        raise ValueError(f"--texture-max must be a power of two from 64 to 32768, not {pixels}")
    return pixels.bit_length()


def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return default


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="isaac_host", description="WardZero's world simulator on Isaac Sim.")
    p.add_argument("--run-dir", required=True, help="run folder to record into")
    p.add_argument("--scenario", required=True)
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="at most this many sim seconds per wall second (0: as fast as it goes)")
    p.add_argument("--lockstep", action="store_true")
    p.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_S)
    p.add_argument("--wait-for", default=",".join(WAIT_TOKENS),
                   help="comma list of nodes to wait for before the clock starts: agents, director, or none")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--headed", dest="headed", action="store_true", default=False, help="show the Kit window")
    mode.add_argument("--headless", dest="headed", action="store_false")
    p.add_argument("--record", action="store_true", help="record the run to <run>/video (I5)")
    p.add_argument("--full-site", action="store_true", help="load the whole hospital (no area culling)")
    p.add_argument("--rooms", default=None, help="comma list of rooms to keep, instead of the scenario's")
    p.add_argument("--private-domain", type=int,
                   default=_env_int("WARDZERO_PRIVATE_DOMAIN", "ROS_DOMAIN_ID", default=0),
                   help="ROS domain of the wz.v1 topics (the agents and director)")
    p.add_argument("--robot-domain", type=int, default=None, help="ROS domain of the robot (default: private)")
    p.add_argument("--fps", type=int, default=30, help="fixed frames per sim second")
    p.add_argument("--stage", type=Path, default=stage_mod.STAGE_USD)
    p.add_argument("--collision", type=Path, default=stage_mod.COLLISION_USD)
    p.add_argument("--renderer", default="RaytracedLighting")
    p.add_argument("--fire", choices=("flow", "proxy", "none"), default=os.environ.get("WARDZERO_FIRE", "flow"),
                   help="how a fire is drawn: Flow fire and smoke (default), emissive cones, or not at all")
    p.add_argument("--texture-max", type=int, default=None,
                   help="largest texture side in pixels (64..32768, a power of two); bigger textures are downsampled. "
                        f"Default: {TEXTURE_MAX_HEADLESS} when headless (it saves about 480 MiB per DH character), "
                        "full size with --headed; 0 keeps full size")
    a = p.parse_args(argv)
    a.wait_for = [] if a.wait_for.strip() in ("", "none") else [w.strip() for w in a.wait_for.split(",") if w.strip()]
    unknown = [w for w in a.wait_for if w not in WAIT_TOKENS]
    if unknown:
        p.error(f"--wait-for: unknown {', '.join(unknown)} (use agents, director or none)")
    a.rooms = [r.strip() for r in a.rooms.split(",") if r.strip()] if a.rooms else None
    if a.fps <= 0:
        p.error("--fps must be positive")
    if a.texture_max is None:
        a.texture_max = 0 if a.headed else TEXTURE_MAX_HEADLESS
    if a.texture_max:
        try:
            max_mip_count(a.texture_max)
        except ValueError as exc:
            p.error(str(exc))
    return a


class IsaacHost:
    """One run in one Kit process (one stage per process: a stage re-open leaks VRAM)."""

    def __init__(self, args: argparse.Namespace, loaded, plan: "stage_mod.CullPlan"):
        self.args, self.loaded, self.plan = args, loaded, plan
        self.run_dir = Path(args.run_dir)
        self.app = None
        self.timeline = None
        self.stage = None
        self.bridges: list = []
        self.runlog = None
        self.bodies = None
        self.svc = None  # the SimService (made before the people, which need its StubWorld)
        self.people = None  # IsaacPeople
        self.objects = None  # IsaacObjects: props, fire and call lights
        self.startup: dict = {}
        self._stop = False
        self._t0 = time.monotonic()

    # ---- helpers ----------------------------------------------------------------------------
    def _mark(self, step: str) -> None:
        self.startup[step] = round(time.monotonic() - self._t0, 2)
        log.info("start-up: %s at %.1f s", step, self.startup[step])

    def _updates(self, n: int) -> None:
        for _ in range(n):
            self.app.update()

    def _on_signal(self, signum, frame) -> None:
        if self._stop:  # a second signal: hurry
            os._exit(130)
        self._stop = True

    # ---- start-up, in the binding order -----------------------------------------------------
    def start_app(self) -> None:
        from isaacsim import SimulationApp

        self.app = SimulationApp({"headless": not self.args.headed, "renderer": self.args.renderer,
                                  "width": 1280, "height": 720})
        if self.args.texture_max:  # before the stage opens: textures larger than this are downsampled on load
            import carb

            carb.settings.get_settings().set("/rtx-transient/resourcemanager/maxMipCount",
                                             max_mip_count(self.args.texture_max))
        # SimulationApp's own SIGINT handler kills Kit at once; the host stops at a frame boundary instead
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self._mark("app")

    def enable_extensions(self) -> None:  # 1
        import omni.kit.app

        em = omni.kit.app.get_app().get_extension_manager()
        for ext in EXTENSIONS:
            em.set_extension_enabled_immediate(ext, True)
            self._updates(3)
        self._mark("extensions")

    def open_stage(self) -> None:  # 2
        from . import fixes

        off = stage_mod.compute_cull(self.plan, self.args.stage, self.args.collision)
        fix_off, authored = fixes.stage_fixes(self.loaded.site, self.plan, self.args.stage)
        wrapper = stage_mod.write_wrapper(self.run_dir / "stage.usda", off + fix_off, self.args.stage, authored)
        self.stage = stage_mod.open_stage(self.app, wrapper)
        self.startup["culled_prims"] = len(off)
        self.startup["kept"] = self.plan.describe()
        log.info("stage: %s, %d prims deactivated", self.plan.describe(), len(off))
        self._mark("stage")

    def physics(self) -> None:  # 3, 4
        stage_mod.add_physics_scene(self.stage)
        stage_mod.add_collision(self.stage, self.args.collision)
        self._updates(2)
        self._mark("physics")

    def make_service(self) -> None:
        """The sim service and its StubWorld are made first: the people are spawned where the world puts them."""
        from ..ros.sim_service import SimService
        from ..runlog import RunLog
        from .bodies import IsaacBodies

        a = self.args
        self.runlog = RunLog(self.run_dir)
        self.bodies = IsaacBodies()
        self.svc = SimService(self.loaded, self.runlog, 1.0, a.lockstep or self.loaded.scenario.lockstep, a.wait_for,
                              bodies=self.bodies)

    def spawn_people(self) -> None:  # 5: the IRA characters, before the NavMesh bake and before play
        from .people_isaac import IsaacPeople

        self.people = IsaacPeople(self.stage, self.loaded)
        self.people.spawn(self.svc.world, self.app)
        self.bodies.spawner = self.people.spawn_runtime
        self._updates(5)
        self._mark("people")

    def spawn_objects(self) -> None:  # 5b: props before the NavMesh bake (kinematic bodies stay out of it)
        from .objects_isaac import IsaacObjects

        self.objects = IsaacObjects(self.stage, self.svc.world, self.args.fire)
        self.objects.update()
        self.bodies.objects = self.objects
        self._updates(2)
        self._mark("objects")

    def bind_people(self) -> None:  # 8b: after play the Behavior System has the agents; they get their controllers
        self.people.bind(self.app, self.bodies, self.svc.world, fps=self.args.fps)
        self._mark("people_bound")

    def bake_navmesh(self) -> None:  # 6
        import omni.anim.navigation.core as nav

        self._updates(5)
        nav.acquire_interface().start_navmesh_baking_and_wait()
        self._mark("navmesh")

    def robot_proxy(self) -> None:  # 7
        r = self.loaded.scenario.robot
        if r.enabled:
            stage_mod.add_robot_proxy(self.stage, r.x, r.y, r.radius + 0.05)

    def play(self) -> None:  # 8
        import carb
        import omni.timeline

        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
        tl = omni.timeline.get_timeline_interface()
        tl.set_time_codes_per_second(self.args.fps)
        tl.set_end_time(self.loaded.scenario.duration_min * 60.0 + 3600.0)
        tl.set_looping(False)
        tl.set_play_every_frame(True)
        tl.play()
        self._updates(3)  # the Behavior System registers its obstacles now
        self.timeline = tl
        self._mark("play")

    def add_robots(self) -> None:  # 9: I4 references the sanitised robot USDs here, in one authoring pass
        pass

    def frame_view(self) -> None:
        if not self.args.headed:
            return
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            eye, target = stage_mod.view_of(self.plan, self.loaded.site)
            ViewportManager.set_camera_view(CAMERA, eye=list(eye), target=list(target))
        except Exception as exc:  # a nicety only
            log.warning("could not frame the viewport: %s", exc)

    # ---- the run ----------------------------------------------------------------------------
    def run(self) -> int:
        self.start_app()
        try:
            self.enable_extensions()
            self.open_stage()
            self.physics()
            self.make_service()
            self.spawn_people()
            self.spawn_objects()
            self.bake_navmesh()
            self.robot_proxy()
            self.play()
            self.bind_people()
            self.add_robots()
            self.frame_view()
            if self.args.record:
                log.warning("--record: recording arrives with the recording task (I5); this run is not recorded")
            return self.loop()
        except Exception:
            log.exception("the Isaac host failed")
            return 1

    def loop(self) -> int:
        from ..ros.rosutil import RosBridge
        from ..ros.sim_service import SimRos
        from .service import HostLoop, VramSampler

        a = self.args
        runlog, svc = self.runlog, self.svc
        private = RosBridge("wardzero_isaac_host", None, use_sim_time=False, domain_id=a.private_domain)
        self.bridges = [private]
        clocks = []
        if a.robot_domain is not None and a.robot_domain != a.private_domain:
            robot = RosBridge("wardzero_isaac_clock", None, use_sim_time=False, domain_id=a.robot_domain)
            self.bridges.append(robot)
            clocks.append(robot)
        sim_ros = SimRos(svc, private, clock_bridges=clocks)
        for b in self.bridges:
            b.start()
        vram = VramSampler().start()
        vram.sample()
        runlog.write("host", {"kind": "startup", **self.startup, "vram_mib": vram.last,
                              "headed": bool(a.headed), "private_domain": a.private_domain,
                              "robot_domain": a.robot_domain, "fps": a.fps})
        log.info("isaac host up: %s on domain %d%s, waiting for %s", self.loaded.scenario.id, a.private_domain,
                 f" (+ /clock on {a.robot_domain})" if clocks else "", ", ".join(sorted(svc.waiting)) or "nothing")
        loop = HostLoop(svc, sim_ros, self.app, self.timeline, bridges=self.bridges, time_scale=a.time_scale,
                        vram=lambda: vram.last, vram_peak=lambda: vram.peak)
        try:
            return loop.run(stop_requested=lambda: self._stop, ready_timeout=a.ready_timeout)
        finally:
            vram.close()

    def close(self, code: int) -> None:
        for b in self.bridges:
            try:
                b.close()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        if self.app is not None:
            self.app.close(exit_code=code)  # fast shutdown: may end the process here


def main(argv=None) -> int:
    args = parse_args(argv)
    for k in KEY_ENV:  # never the key in this process, whatever started it
        os.environ.pop(k, None)
    from ..keys import install_log_redaction

    install_log_redaction()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from ..config.loader import ConfigError, load_scenario

    try:
        loaded = load_scenario(args.scenario)
        plan = stage_mod.cull_plan(loaded, rooms=args.rooms, full_site=args.full_site)
    except (ConfigError, ValueError) as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 2
    if not Path(args.stage).exists():
        print(f"Stage not found: {args.stage}", file=sys.stderr)
        return 2
    host = IsaacHost(args, loaded, plan)
    code = 1
    try:
        code = host.run()
    finally:
        host.close(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
