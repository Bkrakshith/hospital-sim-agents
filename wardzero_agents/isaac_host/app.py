"""The Isaac host process: Kit app, stage, physics, ROS, and the frame loop.

    scripts/isaac_host.sh --run-dir DIR --scenario FILE [--headed|--headless] [--record] [--full-site]
        [--rooms A,B] [--private-domain N] [--robot-domain M] [--time-scale F] [--lockstep]
        [--record-res WxH] [--preview]
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
9. the robots, in one authoring pass (I4): their ready USDs (made before Kit starts, ``prepare_robot``) are referenced
   with their pose, and their OmniGraph is built on the robot domain.
Recording (``recording``, ``bag``) starts after step 9 and replaces ``app.update`` in the frame loop, so it is not a
start-up step. Then the timeline is never stopped (only paused while the run is held) and the NavMesh is never
re-baked while robots are present.

ROS: Isaac's bundled rclpy, one explicit Context per domain. The private domain carries wz.v1 (the
same topics as ``stub_sim_node``, through ``SimRos``) and /clock; the robot domain carries the robot's topics:
audio/heard, speech/say, goal_pose, task/* and perception/objects through ``SimRos`` (the StubWorld's logic), and
/clock, /tf, joint states, the camera and bbox_3d (the host's detection link) through the robot's OmniGraph. The host never receives the Claude key: the launcher and this module drop it.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from ..ros.sim_service import READY_TIMEOUT_S, WAIT_TOKENS
from . import recording
from . import stage as stage_mod
from .bag import BagRecorder

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
    p.add_argument("--record", action="store_true",
                   help="record the run to <run>/video (overhead.mp4, follow.mp4) and the robot domain to "
                        "<run>/robot_bag; recorded runs are in lockstep (sim time per frame is fixed at 1/fps)")
    p.add_argument("--bag-images", action="store_true",
                   help="with --record: also put the robot's rgb, depth and point cloud in the rosbag (large)")
    p.add_argument("--record-res", type=recording.parse_res, default=recording.DEFAULT_RES, metavar="WxH",
                   help="resolution of the recorded videos (default 1280x720; 1920x1080 for the overhead film)")
    p.add_argument("--preview", action="store_true",
                   help="headless: write <run>/preview/latest.jpg five times a second for the Studio's Run page")
    p.add_argument("--full-site", action="store_true", help="load the whole hospital (no area culling)")
    p.add_argument("--rooms", default=None, help="comma list of rooms to keep, instead of the scenario's")
    p.add_argument("--private-domain", type=int,
                   default=_env_int("WARDZERO_PRIVATE_DOMAIN", "ROS_DOMAIN_ID", default=0),
                   help="ROS domain of the wz.v1 topics (the agents and director)")
    p.add_argument("--robot-domain", type=int, default=None, help="ROS domain of the robot (default: private)")
    p.add_argument("--fps", type=int, default=None,
                   help="fixed frames per sim second (default 30; 25 with the walking G1, whose physics runs at 200 Hz "
                        "in whole steps a frame, so its fps must divide 200)")
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
    a.fps_given = a.fps is not None
    a.fps = 30 if a.fps is None else a.fps
    a.physics_hz = None  # set by apply_frame_plan once the robot is known
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


def host_profile(loaded):
    """The robot's profile as this host runs it (the built-in G1 walks here with the GEAR-SONIC weights), or None."""
    from ..robots.profile import get_profile, profile_for_sim

    r = loaded.scenario.robot
    return profile_for_sim(get_profile(r.profile), "isaac") if r.enabled else None


def apply_frame_plan(args, profile) -> None:
    """--fps and the physics rate: physics 200 Hz and a frame rate dividing it (25 by default) for the walking G1."""
    from .robot_isaac import walking_g1
    from .walking import frame_plan

    walking = profile is not None and walking_g1(profile)
    args.fps, args.physics_hz = frame_plan(args.fps if args.fps_given else None, walking)


# The start-up steps after the app is up, in the binding order (the module docstring). ``boot_sequence`` is the only
# place this order lives: ``run`` and the demos (robot_demo, whose frame-budget test guards the order) call it.
STARTUP = ("enable_extensions", "open_stage", "physics", "make_service", "spawn_people", "spawn_objects",
           "bake_navmesh", "robot_proxy", "play", "bind_people", "add_robots")


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
        self.robot = None  # IsaacRobot (None without a robot)
        self.robot_usd: Optional[Path] = None
        self.profile = None  # the robot's profile as this host runs it (host_profile)
        self.cams = None  # recording.HeadedCameras: the headed window's named cameras
        self.swinging: list[str] = []  # site doors on the run's routes: hinged with a closer (doors_isaac)
        self.doors = None  # doors_isaac.IsaacDoors
        self.recorder: Optional[recording.Recorder] = None
        self.bag: Optional[BagRecorder] = None
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

        # WARDZERO_KIT_ARGS: extra Kit settings for this process, e.g. Kit's derived-data cache and log on another disk
        # ("--/UJITSO/datastore/localCachePath=/big/disk/kitcache --/log/file=/big/disk/kit.log"); Kit reads sys.argv
        sys.argv += [a for a in os.environ.get("WARDZERO_KIT_ARGS", "").split() if a.startswith("--/")]
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
        for ext in EXTENSIONS + (("omni.videoencoding",) if self.args.record else ()):
            em.set_extension_enabled_immediate(ext, True)
            self._updates(3)
        self._mark("extensions")

    def open_stage(self) -> None:  # 2
        from . import fixes

        from .doors import route_doors

        off = stage_mod.compute_cull(self.plan, self.args.stage, self.args.collision)
        self.swinging = route_doors(self.loaded.site, None if self.plan.full else self.plan.rects)
        fix_off, authored, nav_include = fixes.stage_fixes(self.loaded.site, self.plan, self.args.stage,
                                                           self.args.collision, self.swinging)
        wrapper = stage_mod.write_wrapper(self.run_dir / "stage.usda", off + fix_off, self.args.stage, authored,
                                          nav_include=nav_include)
        self.stage = stage_mod.open_stage(self.app, wrapper)
        self.startup["culled_prims"] = len(off)
        self.startup["kept"] = self.plan.describe()
        log.info("stage: %s, %d prims deactivated", self.plan.describe(), len(off))
        self._mark("stage")

    def physics(self) -> None:  # 3, 4
        hz = getattr(self.args, "physics_hz", None)
        stage_mod.add_physics_scene(self.stage, hz)
        if hz:
            import carb

            from .walking import MIN_FRAME_RATE
            carb.settings.get_settings().set("/persistent/simulation/minFrameRate", MIN_FRAME_RATE)  # never drop substeps
            self.startup["physics_hz"] = hz

        stage_mod.add_collision(self.stage, self.args.collision)
        from .doors_isaac import IsaacDoors

        self.doors = IsaacDoors(self.stage)
        self.doors.author(self.swinging)  # before the NavMesh bake and play: closed leaves, closer, out of the NavMesh
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
        # a recorded run is in lockstep: the clock never outruns the minds, however slow the encoder is
        self.svc = SimService(self.loaded, self.runlog, 1.0, a.lockstep or a.record or self.loaded.scenario.lockstep,
                              a.wait_for, bodies=self.bodies)

    def spawn_people(self) -> None:  # 5: the IRA characters, before the NavMesh bake and before play
        from .people_isaac import IsaacPeople

        self.people = IsaacPeople(self.stage, self.loaded)
        self.people.spawn(self.svc.world, self.app)
        self.bodies.spawner = self.people.spawn_runtime
        self._updates(5)
        if getattr(self, "doors", None) is not None:
            from .doors import SPARE_CAPSULES
            self.doors.add_capsules(len(self.people.skelroots) + SPARE_CAPSULES)
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
        if getattr(self, "doors", None) is not None:
            self.doors.bind_agents(self.bodies.controllers)
        self._check_people_bound()
        self._mark("people_bound")

    def _check_people_bound(self) -> None:
        """Every character spawned at the start must have a Behavior agent: one without stays frozen (a T-pose) and
        never walks. A shortfall is logged and written to the run record (host.jsonl, kind "problem"); the launcher
        copies it into meta.json."""
        expected = sorted(self.people.skelroots)
        missing = sorted(getattr(self.people, "unbound", []) or [])
        self.startup["people_expected"] = len(expected)
        self.startup["people_agents"] = len(expected) - len(missing)
        if not missing:
            log.info("people bound: all %d characters have a Behavior agent", len(expected))
            return
        msg = (f"people bound: {len(expected) - len(missing)} of {len(expected)} characters have a Behavior agent; "
               f"{', '.join(missing)} will not walk or animate (no NavMesh over their rooms?)")
        log.error("%s", msg)
        self.runlog.write("host", {"kind": "problem", "what": "people_not_animated", "missing": missing,
                                   "bound": len(expected) - len(missing), "expected": len(expected), "message": msg})

    def bake_navmesh(self) -> None:  # 6
        import omni.anim.navigation.core as nav

        self._updates(5)
        nav.acquire_interface().start_navmesh_baking_and_wait()
        self._mark("navmesh")

    def robot_proxy(self) -> None:  # 7
        r = self.loaded.scenario.robot
        if r.enabled:
            from ..world.robot_body import CONTACT_MARGIN_M
            # the people's avoidance keeps them outside the cylinder: as wide as the world's contact distance (a person
            # within radius + 0.25 m of the robot is a contact; at radius + 0.05 the receptionist brushed the G1)
            stage_mod.add_robot_proxy(self.stage, r.x, r.y, r.radius + CONTACT_MARGIN_M + 0.05)

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
        if getattr(self, "doors", None) is not None:
            self.doors.start(1.0 / (getattr(self.args, "physics_hz", None) or 60))
        self._mark("play")

    def prepare_robot(self) -> None:
        """Before Kit starts: the robot's ready USD (converted with Isaac's URDF importer in a process of its own the
        first time, or when its URDF, sensors or the importer changed)."""
        from ..robots import ready_usd

        r = self.loaded.scenario.robot
        if r.enabled and self.profile is None:
            self.profile = host_profile(self.loaded)
        if r.enabled and self.robot_usd is None:
            self.robot_usd = ready_usd.ensure_ready_usd(self.profile)

    @property
    def robot_domain(self) -> int:
        a = self.args
        return a.private_domain if a.robot_domain is None else a.robot_domain

    def add_robots(self) -> None:  # 9: the ready robot USD goes in as one reference after play(), with its graph
        from .robot_isaac import IsaacRobot, walking_g1

        r = self.loaded.scenario.robot
        if not r.enabled:
            return
        if self.robot_usd is None:
            raise RuntimeError("the robot's ready USD is not made: call prepare_robot() before start_app()")
        if walking_g1(self.profile):
            self._legged_body()
        elif self.profile.name == "g1_claude":
            from ..robots.profile import g1_static_reason
            log.warning("%s", g1_static_reason())
        self.robot = IsaacRobot(self.stage, self.app, self.loaded, self.profile, self.robot_usd,
                                self.args.fps, self.robot_domain, self.args.private_domain, runlog=self.runlog)
        self.robot.add(self.svc.world)
        # a robot that cannot walk here (the G1 until its walking controller lands) tells the staff from where it stands
        self.svc.tasks.mobile = bool(getattr(self.robot.profile, "can_move", False))
        self.bodies.set_robot(self.robot.pose)
        self.bodies.robot_driver = self.robot
        self._mark("robot")

    def _legged_body(self) -> None:
        """The world's robot becomes a LeggedBody: Isaac moves it, a RouteFollower on a router with the G1's clearance
        turns goals into walking commands, and the walker reports falls to it."""
        from ..site.routing import Router
        from ..world.legged_body import LeggedBody, door_points
        from ..world.route_follower import G1_CLEARANCE_M, ROBOT_HARD_M

        from ..world.legged_body import block_segments
        from .walker import door_leaf_segments

        w = self.svc.world
        old = w.robot
        router = Router(w.site.grid, clearance=G1_CLEARANCE_M, hard=ROBOT_HARD_M)
        try:  # door leaves are not on the site map: the G1 routes round them as they stand
            leaves = door_leaf_segments(self.stage, skip=self.swinging)  # swinging doors are walked through
            cells = block_segments(router, leaves, G1_CLEARANCE_M)
            log.info("the G1's routes keep clear of %d door leaves (%d map cells)", len(leaves), cells)
        except Exception as exc:
            log.warning("could not read the door leaves for the G1's routes: %s", exc)
        areas = getattr(self.loaded.scenario, "forbidden_areas", None)
        if areas:  # the user's no-go areas over the G1's own router (its band, its door leaves), as SimService does
            from ..site.areas import robot_router
            router = robot_router(router, areas)
            log.info("the G1's routes keep out of %d forbidden areas", len(areas))
        w.robot = LeggedBody(old.x, old.y, old.yaw, old.radius, self.profile.max_speed, router,
                             doors=door_points(w.site))
        swing = set(self.swinging)  # the G1 pushes these open: straight through the centre of a leaf (door spike)
        w.robot.follower.door_lines = [(tuple(d["a"]), tuple(d["b"])) for d in w.site.doors if d.get("id") in swing]
        log.info("the G1 walks (GEAR-SONIC): routes with %.2f m clearance, at most %.1f m/s", G1_CLEARANCE_M,
                 self.profile.max_speed)

    def _frame_rects(self) -> list:
        """What the overhead camera fits: the scenario's places (stage.frame_rects), not every kept room."""
        try:
            return stage_mod.frame_rects(self.loaded, self.plan)
        except Exception as exc:  # a nicety only: the kept rooms, as before
            log.warning("could not work out the camera framing: %s", exc)
            return list(self.plan.rects.values())

    def frame_view(self) -> None:
        if not self.args.headed:
            return
        if self.robot is not None:  # the follow camera, behind and above the robot; desk and overhead stay selectable
            try:
                r = self.loaded.scenario.robot
                self.cams = recording.HeadedCameras(self.stage, self._frame_rects() or
                                                    [tuple(self.loaded.site.bounds[i] for i in (0, 2, 1, 3))],
                                                    self.robot.pose, (1280, 720), post=(r.x, r.y, r.yaw))
                self.cams.start()
                return
            except Exception as exc:  # a nicety only: fall back to the fixed view
                log.warning("could not set up the follow camera: %s", exc)
                self.cams = None
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            r = self.loaded.scenario.robot
            view = recording.start_view((r.x, r.y, r.yaw) if r.enabled else None)
            if view is None:  # no robot: the kept rooms from above
                eye, target = stage_mod.view_of(self.plan, self.loaded.site)
            else:  # the robot's post, below the ceiling, with the people who come to it
                eye, target, focal = view
                from pxr import UsdGeom
                cam = UsdGeom.Camera(self.stage.GetPrimAtPath(CAMERA))
                if cam:
                    cam.GetFocalLengthAttr().Set(float(focal))
            ViewportManager.set_camera_view(CAMERA, eye=list(eye), target=list(target))
            log.info("viewport framed: eye %s looking at %s", tuple(round(v, 2) for v in eye),
                     tuple(round(v, 2) for v in target))
        except Exception as exc:  # a nicety only
            log.warning("could not frame the viewport: %s", exc)

    # ---- the run ----------------------------------------------------------------------------
    def boot_sequence(self, order=STARTUP) -> None:
        """Make the robot's USD, start Kit and run every start-up step. ``order`` exists for tests (a negative control)."""
        self.prepare_robot()
        self.start_app()
        for step in order:
            getattr(self, step)()

    def run(self) -> int:
        try:
            self.prepare_robot()
        except Exception as exc:
            print(f"Robot error: {exc}", file=sys.stderr)
            return 2
        try:
            if self.args.record and self.loaded.scenario.robot.enabled:  # before Kit: it is up when the robot speaks
                self.bag = BagRecorder(self.run_dir, self.robot_domain, self.args.private_domain,
                                       ns=self.loaded.scenario.robot.ns, images=self.args.bag_images)
                self.bag.start()
            self.boot_sequence()
            self.frame_view()
            self.start_recording()
            return self.loop()
        except Exception:
            log.exception("the Isaac host failed")
            return 1

    def start_recording(self) -> None:
        """The cameras, writers and preview (after the start-up: they never touch physics or the NavMesh)."""
        cfg = recording.RecordConfig.from_args(self.args, self.run_dir)
        if cfg is None:
            return
        self.recorder = recording.Recorder(cfg, self.app, self.timeline, self.stage, self._frame_rects() or
                                           [tuple(self.loaded.site.bounds[i] for i in (0, 2, 1, 3))],
                                           robot_pose=self.robot.pose if self.robot is not None else None,
                                           clock=lambda: float(self.svc.t) if getattr(self, "svc", None) is not None else 0.0)
        self.recorder.start()
        self._updates(2)
        self._mark("recording")

    def calibrate_robot_clock(self) -> float:
        """Isaac stamps the robot's tf, joints, camera and boxes with its simulation time, which started when the timeline
        played and so runs ahead of the service's time (0 at the loop's start) by the frames of the start-up. It cannot
        be reset without stopping the timeline, so the robot domain's /clock is on Isaac's base: svc.t plus this offset
        (Isaac's time now). The people's /clock stays svc.t. One frame is spent reading it."""
        if self.robot is None or self.timeline is None:
            return 0.0
        self.robot.fire()
        self.app.update()
        offset = self.robot.sim_time()
        self.startup["robot_clock_offset_s"] = round(offset, 4)
        log.info("robot clock offset: %.4f s (Isaac simulation time at the start of the run)", offset)
        return offset

    def advance(self) -> None:
        """One frame for the loop: fire the robot's graph once (while the timeline runs), then update, or render for the
        recorder. A frame with several Kit updates (the recorder's) still gives one message per topic."""
        if self.robot is not None and self.timeline.is_playing():
            self.robot.fire()
        cams = getattr(self, "cams", None)
        if cams is not None:
            cams.update()
        walker = getattr(self.robot, "walker", None)
        if walker is not None:
            walker.begin_frame()
        (self.recorder.step if self.recorder is not None else self.app.update)()
        if walker is not None:
            walker.end_frame()
            self._check_steps(walker)
        self._doors_frame()
        self._test_viewport_shot()

    def _check_steps(self, walker) -> None:
        """Once, a few frames in: the physics steps a frame must be 200 Hz / fps, or SONIC's PD runs at the wrong rate."""
        # frames with no physics step are the timeline held while the agents start up: they say nothing about the rate
        played = [n for n in walker.frame_steps if n > 0]
        if "steps_per_frame" in self.startup or len(played) < 6:
            return
        from .walking import steps_per_frame

        want = steps_per_frame(self.args.fps, self.args.physics_hz)
        got = sorted(played[2:6])
        self.startup["steps_per_frame"] = got[len(got) // 2]
        (log.info if all(n == want for n in got) else log.error)(
            "physics steps a frame: %s (want %s: %s Hz physics at %d fps)", got, want, self.args.physics_hz, self.args.fps)

    def _test_viewport_shot(self) -> None:
        """Test only (WARDZERO_TEST_VIEWPORT_SHOTS=<dir>, headed): the window's view as a PNG every 5 s of sim time."""
        d = os.environ.get("WARDZERO_TEST_VIEWPORT_SHOTS")
        if not d or not getattr(getattr(self, "args", None), "headed", False) or getattr(self, "svc", None) is None:
            return
        n = int(self.svc.t // 5.0)
        if n == getattr(self, "_shot_n", -1):
            return
        self._shot_n = n
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            Path(d).mkdir(parents=True, exist_ok=True)
            capture_viewport_to_file(get_active_viewport(), str(Path(d) / f"viewport_{n * 5:04d}s.png"))
        except Exception as exc:
            log.warning("viewport shot failed: %s", exc)

    def finish_recording(self) -> None:
        if self.recorder is not None:
            try:
                stats = self.recorder.finish()
                self.runlog.write("host", {"kind": "recording", **stats})
            except Exception:
                log.exception("could not finish the recording")
            self.recorder = None
        if self.bag is not None:
            bag, self.bag = self.bag, None
            result = bag.stop()
            log.info("rosbag %s (%s)", result, "finalised" if bag.finalised else "no metadata.yaml")

    def loop(self) -> int:
        from ..ros.rosutil import RosBridge
        from ..ros.sim_service import SimRos
        from .service import HostLoop, VramSampler

        a = self.args
        runlog, svc = self.runlog, self.svc
        private = RosBridge("wardzero_isaac_host", None, use_sim_time=False, domain_id=a.private_domain)
        self.bridges = [private]
        robot_bridge = None
        if a.robot_domain is not None and a.robot_domain != a.private_domain:
            robot_bridge = RosBridge("wardzero_isaac_robot_side", None, use_sim_time=False, domain_id=a.robot_domain)
            self.bridges.append(robot_bridge)
        robot = self.robot
        # /clock is the service's time (svc.t) on every domain: the robot's graph publishes none
        clocks = [robot_bridge] if robot_bridge is not None else []
        offset = self.calibrate_robot_clock()
        sim_ros = SimRos(svc, private, clock_bridges=clocks, robot_ros=robot_bridge,
                         graph_owned=robot.graph_owned if robot is not None else (), robot_clock_offset=offset)
        if robot is not None:
            robot.attach_ros(svc.world, private, robot_bridge)
        for b in self.bridges:
            b.start()
        vram = VramSampler().start()
        vram.sample()
        runlog.write("host", {"kind": "startup", **self.startup, "vram_mib": vram.last,
                              "headed": bool(a.headed), "private_domain": a.private_domain,
                              "robot_domain": a.robot_domain, "fps": a.fps})
        log.info("isaac host up: %s on domain %d%s, waiting for %s", self.loaded.scenario.id, a.private_domain,
                 f" (+ the robot on {a.robot_domain})" if robot_bridge is not None else "", ", ".join(sorted(svc.waiting)) or "nothing")
        loop = HostLoop(svc, sim_ros, self.app, self.timeline, bridges=self.bridges, time_scale=a.time_scale,
                        vram=lambda: vram.last, vram_peak=lambda: vram.peak,
                        advance=self.advance)
        try:
            return loop.run(stop_requested=lambda: self._stop, ready_timeout=a.ready_timeout)
        finally:
            vram.close()
            self._write_walker()
            if getattr(self, "doors", None) is not None:
                try:
                    self.runlog.write("host", {"kind": "doors", "t": round(float(self.svc.t), 3),
                                               **self.doors.summary()})
                    log.info("doors: %s", self.doors.summary())
                except Exception:
                    log.exception("could not write the door summary")
            self.finish_recording()

    def _doors_frame(self) -> None:
        doors = getattr(self, "doors", None)
        if doors is None or getattr(self, "svc", None) is None or self.bodies is None:
            return
        w = self.svc.world
        poses = {}
        for pid, ctrl in self.bodies.controllers.items():
            p = w.people.get(pid)
            if p is None or p.posture in ("gone", "lying") or getattr(ctrl, "in_bed", False):
                poses[pid] = None
                continue
            try:
                x, y, _ = ctrl.api.pose()
                poses[pid] = (x, y)
            except Exception:
                poses[pid] = None
        doors.end_frame(poses, self.bodies.controllers, float(self.svc.t))

    def _write_walker(self) -> None:
        walker = getattr(self.robot, "walker", None)
        if walker is None:
            return
        try:
            s = walker.summary()
            self.runlog.write("host", {"kind": "walker", "t": round(float(self.svc.t), 3), **s})
            log.info("walker: %s", s)
        except Exception:
            log.exception("could not write the walker summary")

    def close(self, code: int) -> None:
        self.finish_recording()  # the loop's own finally did it; this covers a failed start-up
        if self.robot is not None and hasattr(self.robot, "close"):
            self.robot.close()
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
    try:
        profile = host_profile(loaded)
    except KeyError:
        profile = None  # prepare_robot says what is wrong with it
    try:
        apply_frame_plan(args, profile)
    except ValueError as exc:
        print(f"Frame rate error: {exc}", file=sys.stderr)
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
