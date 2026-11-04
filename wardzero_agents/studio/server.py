"""The Studio's HTTP API (aiohttp, loopback only). The request checks are the panel's: Host and Origin
must be loopback, writes need X-WardZero-Panel: 1, bodies are capped at 64 KB."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import socket
import sys
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Optional

import yaml
from aiohttp import WSMsgType, web
from pydantic import ValidationError

from ..config.presets import (DEFAULT_CONFIG_ROOT, CharacterSpec, RobotStart, load_moods, load_roles,
                              load_scenario_presets)
from ..panel.server import MAX_BODY, WS_PERIOD_S, WS_SEND_TIMEOUT_S, BadRequest, _body, _dumps, _err, _guard
from ..robots.profile import G1_NAME, PROFILES, RobotProfile, load_profile, profile_path, save_profile
from ..robots.sensors import SensorSpec, write_sensors_yaml
from ..robots.urdf import parse_urdf
from ..site import load_site
from . import characters as chars
from .project import NotFound, Project, ProjectError, ProjectStore, defaults_from_preset, valid_id, validation_message
from .runner import Runner, RunnerBusy

log = logging.getLogger("wardzero.studio")
STATIC = Path(__file__).parent / "static"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
PROBE_TIMEOUT_S = 8.0  # the probe runs in the background; requests never wait for it
PROBE_EVERY_S = 3.0
NO_LIGHTS = {"controller_seen": False, "sensors_subscribed": False, "commands_seen": False}
JOB_LOG_MAX = 40000

CTX = web.AppKey("ctx", object)
SOCKETS = web.AppKey("sockets", set)
PROBE = web.AppKey("probe", object)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Ctx:
    """What the handlers share: where things live, and the long-lived pieces."""

    def __init__(self, data: Path, config_root: Path, runs_dir: Path, runner: Runner,
                 convert_command=None, probe=None, probe_every: float = PROBE_EVERY_S):
        self.data, self.config_root, self.runs_dir, self.runner = data, config_root, runs_dir, runner
        self.projects = ProjectStore(data)
        self.roles, self.moods = load_roles(config_root), load_moods(config_root)
        self.presets = load_scenario_presets(config_root)
        self.site = load_site(config_root)
        self.jobs: dict[str, dict] = {}
        self.tasks: set = set()
        self.convert_command = convert_command or default_convert_command
        self.probe = probe or probe_graph
        self.probe_every = probe_every
        self.probe_target = (0, "robot")  # robot domain and namespace the last /api/connection asked about
        self.lights = dict(NO_LIGHTS)


# ---- helpers ----------------------------------------------------------------------------------
def j(obj: Any, status: int = 200) -> web.Response:
    return web.Response(text=_dumps(obj), content_type="application/json", status=status)


@web.middleware
async def _errors(request: web.Request, handler):
    try:
        return await handler(request)
    except ApiError as e:
        return _err(e.status, e.message)
    except NotFound as e:
        return _err(404, str(e))
    except ProjectError as e:
        return _err(400, str(e))
    except ValidationError as e:
        return _err(400, validation_message(e))
    except (BadRequest, web.HTTPException, asyncio.CancelledError):
        raise
    except Exception as e:  # never leak details
        log.error("studio API call failed: %s", type(e).__name__)
        return _err(500, "The Studio could not do that. See its terminal for the error type.")


def check_robot_profile(ctx: Ctx, project: Project) -> None:
    if project.robot_profile in PROFILES:
        return
    try:
        load_profile(project.robot_profile, ctx.data)
    except (KeyError, ValueError):
        raise ProjectError(f"Unknown robot '{project.robot_profile}'. Save it under Robots first.") from None


def _project(ctx: Ctx, request) -> Project:
    return ctx.projects.get(request.match_info["id"])


def _robot_name(request) -> str:
    return request.match_info["name"]


def profile_dict(p: RobotProfile) -> dict:
    return {"name": p.name, "builtin": p.name in PROFILES, "can_move": p.can_move, "base": p.base,
            "radius": p.radius, "reach": list(p.reach), "max_speed": p.max_speed, "ns": p.ns,
            "controller": p.controller, "urdf": str(p.urdf) if p.urdf else None,
            "sensors": [s.to_dict() for s in p.sensors]}


def saved_profiles(data: Path) -> list[RobotProfile]:
    out = []
    for f in sorted((Path(data) / "robots").glob("*/robot.yaml")):
        try:
            out.append(load_profile(f.parent.name, data))
        except (KeyError, ValueError, OSError, TypeError, yaml.YAMLError):
            continue
    return out


def find_profile(ctx: Ctx, name: str, editable: bool = False) -> RobotProfile:
    if name in PROFILES:
        if editable:
            raise ApiError(400, f"{name} is built in and cannot be changed. Save your own robot instead.")
        return PROFILES[name]
    try:
        return load_profile(name, ctx.data)
    except (KeyError, ValueError):
        raise ApiError(404, f"no robot '{name}'") from None


def _number(d: dict, key: str, lo: float, hi: float, default: float) -> float:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
        raise BadRequest(f"'{key}' must be a number from {lo:g} to {hi:g}.")
    return float(v)


def _connection_file(data: Path) -> Path:
    return Path(data) / "studio" / "connection.yaml"


def read_connection(data: Path) -> dict:
    try:
        d = yaml.safe_load(_connection_file(data).read_text()) or {}
    except (OSError, yaml.YAMLError):
        d = {}
    dom = d.get("robot_domain", os.environ.get("ROS_DOMAIN_ID", "0"))
    try:
        dom = int(dom)
    except (TypeError, ValueError):
        dom = 0
    return {"robot_domain": dom, "rmw": d.get("rmw") or os.environ.get("RMW_IMPLEMENTATION") or "default"}


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


_TYPES = {"clock": "rosgraph_msgs/Clock", "tf": "tf2_msgs/TFMessage", "joint_states": "sensor_msgs/JointState",
          "joint_command": "sensor_msgs/JointState", "cmd_vel": "geometry_msgs/Twist",
          "odom": "nav_msgs/Odometry", "camera_rgb": "sensor_msgs/Image", "camera_depth": "sensor_msgs/Image",
          "camera_info": "sensor_msgs/CameraInfo", "bbox_3d": "std_msgs/String (JSON)",
          "lidar": "sensor_msgs/PointCloud2", "objects": "std_msgs/String (JSON)",
          "heard": "std_msgs/String (JSON)", "say": "std_msgs/String (JSON)",
          "goal_pose": "geometry_msgs/PoseStamped", "task_action": "std_msgs/String (JSON)",
          "task_result": "std_msgs/String (JSON)", "task_brief": "std_msgs/String (JSON)"}
_OUT = {"joint_command", "cmd_vel", "goal_pose", "say", "task_action"}  # the robot publishes these


def topic_table(profile: RobotProfile) -> list[dict]:
    from ..ros.topics import robot_topics
    kinds = {s.kind for s in profile.sensors}
    skip = set()
    if not kinds & {"rgb", "rgbd"}:
        skip |= {"camera_rgb", "camera_info", "bbox_3d"}
    if "rgbd" not in kinds:
        skip.add("camera_depth")
    if "lidar" not in kinds:
        skip.add("lidar")
    if "mic" not in kinds:
        skip.add("heard")
    if "speaker" not in kinds:
        skip.add("say")
    if not profile.can_move:
        skip |= {"cmd_vel", "goal_pose"}
    return [{"name": name, "type": _TYPES.get(key, "std_msgs/String"), "dir": "robot_to_sim" if key in _OUT
             else "sim_to_robot"} for key, name in robot_topics(profile.ns).items() if key not in skip]


async def probe_graph(domain: int, ns: str, argv: Optional[list] = None) -> Optional[dict]:
    """Look at the ROS graph on the robot's domain from a short child process; None if ROS is not available,
    it fails or it takes too long. argv replaces the child's command (tests)."""
    from ..cli import ros_available
    if argv is None:
        if not ros_available():
            return None
        argv = [sys.executable, "-m", "wardzero_agents.studio.ros_probe", "probe", ns]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            env=dict(os.environ, ROS_DOMAIN_ID=str(domain),
                     PYTHONPATH=str(Path(__file__).resolve().parents[2])))
    except OSError:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), PROBE_TIMEOUT_S)
        return json.loads(out.decode().strip().splitlines()[-1])
    except (asyncio.TimeoutError, ValueError, IndexError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()  # reap it: no zombie
        return None


async def refresh_lights(ctx: "Ctx") -> None:
    """Keep ctx.lights fresh: one graph probe every few seconds, for the robot the page last asked about."""
    from ..cli import ros_available
    while True:
        try:
            if ros_available():
                domain, ns = ctx.probe_target
                seen = await ctx.probe(domain, ns)
                ctx.lights = {k: bool((seen or {}).get(k)) for k in NO_LIGHTS}
            else:
                ctx.lights = dict(NO_LIGHTS)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # one failed probe must not freeze the lights
            log.error("connection probe failed: %s", type(e).__name__)
            ctx.lights = dict(NO_LIGHTS)
        await asyncio.sleep(ctx.probe_every)


def default_convert_command(urdf: Path, sensors: Path, out: Path) -> list[str]:
    py = os.environ.get("WARDZERO_ISAAC_PYTHON") or str(Path.home() / "isaacsim" / "python.sh")
    return [py, str(SCRIPTS / "isaac_import_urdf.py"), "--urdf", str(urdf), "--sensors", str(sensors),
            "--out", str(out)]


# ---- the app ----------------------------------------------------------------------------------
def make_app(ctx: Ctx) -> web.Application:
    app = web.Application(middlewares=[_guard, _errors], client_max_size=MAX_BODY)
    app[CTX] = ctx
    app[SOCKETS] = set()
    r = web.RouteTableDef()

    # static page
    @r.get("/")
    async def index(request):
        page = STATIC / "index.html"
        if not page.is_file():
            return web.Response(text="WardZero Studio: the page is not installed yet.", content_type="text/plain")
        resp = web.Response(body=page.read_bytes(), content_type="text/html", charset="utf-8")
        resp.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; font-src 'self'; base-uri 'none'; form-action 'none'")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    # site and presets
    @r.get("/api/site")
    async def site(request):
        return web.Response(body=(ctx.config_root / "site" / "hospital.json").read_bytes(),
                            content_type="application/json")

    @r.get("/api/presets")
    async def presets(request):
        return j({"roles": [x.model_dump(mode="json") for x in ctx.roles.values()],
                  "moods": [x.model_dump(mode="json") for x in ctx.moods.values()],
                  "scenarios": [x.model_dump(mode="json") for x in ctx.presets.values()]})

    # projects
    @r.get("/api/projects")
    async def list_projects(request):
        return j([p.model_dump(mode="json") for p in ctx.projects.list()])

    @r.post("/api/projects")
    async def create_project(request):
        d = await _body(request)
        sid = d.get("scenario_id")
        if sid not in ctx.presets:
            raise BadRequest("'scenario_id' must be one of: " + ", ".join(ctx.presets) + ".")
        data = {**defaults_from_preset(ctx.presets[sid]), **d, "id": ctx.projects.new_id()}
        project = chars.check_all(Project.model_validate(data), ctx.site, ctx.roles, ctx.moods)
        check_robot_profile(ctx, project)
        ctx.projects.save(project)
        return j(project.model_dump(mode="json"), 201)

    @r.get("/api/projects/{id}")
    async def get_project(request):
        return j(_project(ctx, request).model_dump(mode="json"))

    @r.put("/api/projects/{id}")
    async def put_project(request):
        old = _project(ctx, request)
        d = await _body(request)
        d.pop("id", None)
        merged = {**old.model_dump(mode="json"), **d, "id": old.id}
        if d.get("scenario_id") not in (None, old.scenario_id):
            if d["scenario_id"] not in ctx.presets:
                raise BadRequest("'scenario_id' must be one of: " + ", ".join(ctx.presets) + ".")
            for k, v in defaults_from_preset(ctx.presets[d["scenario_id"]]).items():
                if k not in d:  # a new scenario brings its own cast, robot start and length
                    merged[k] = v
        project = chars.check_all(Project.model_validate(merged), ctx.site, ctx.roles, ctx.moods)
        check_robot_profile(ctx, project)
        ctx.projects.save(project)
        return j(project.model_dump(mode="json"))

    @r.delete("/api/projects/{id}")
    async def delete_project(request):
        ctx.projects.delete(request.match_info["id"])
        return j({"ok": True})

    # characters
    @r.post("/api/projects/{id}/characters")
    async def add_character(request):
        p = _project(ctx, request)
        d = await _body(request)
        d.setdefault("id", chars.new_character_id(p.characters))
        d.setdefault("room", "")
        d.setdefault("yaw", 0.0)
        c = chars.check_character(CharacterSpec.model_validate(d), ctx.site, ctx.roles, ctx.moods, p.characters)
        p = p.model_copy(update={"characters": [*p.characters, c]})
        ctx.projects.save(p)
        return j(c.model_dump(mode="json"), 201)

    @r.put("/api/projects/{id}/characters/{cid}")
    async def edit_character(request):
        p = _project(ctx, request)
        cid = request.match_info["cid"]
        old = next((c for c in p.characters if c.id == cid), None)
        if old is None:
            raise ApiError(404, f"no character '{cid}'")
        d = await _body(request)
        d.pop("id", None)
        if "x" in d or "y" in d:  # moved: a bed from before no longer applies unless it is sent again
            d.setdefault("bed", None)
        new = CharacterSpec.model_validate({**old.model_dump(mode="json"), **d, "id": cid})
        others = [c for c in p.characters if c.id != cid]
        new = chars.check_character(new, ctx.site, ctx.roles, ctx.moods, others)
        p = p.model_copy(update={"characters": [new if c.id == cid else c for c in p.characters]})
        ctx.projects.save(p)
        return j(new.model_dump(mode="json"))

    @r.delete("/api/projects/{id}/characters/{cid}")
    async def delete_character(request):
        p = _project(ctx, request)
        cid = request.match_info["cid"]
        if not any(c.id == cid for c in p.characters):
            raise ApiError(404, f"no character '{cid}'")
        ctx.projects.save(p.model_copy(update={"characters": [c for c in p.characters if c.id != cid]}))
        return j({"ok": True})

    # memory
    @r.get("/api/projects/{id}/memory")
    async def get_memory(request):
        from ..campaign.trust import TrustStore
        from ..memory import memory_path
        p = _project(ctx, request)
        store = TrustStore(memory_path(ctx.data))
        return j(chars.memory_rows(p, store, memory_path(ctx.data), ctx.roles))

    @r.post("/api/projects/{id}/memory/reset")
    async def reset_memory(request):
        from ..memory.runlock import RunActive
        p = _project(ctx, request)
        d = await _body(request)
        if "characters" not in d:
            raise BadRequest("'characters' is required: a list of character ids, or \"all\".")
        try:
            done = await asyncio.to_thread(chars.reset_memory, p, d["characters"], ctx.data,
                                           os.environ.get("WARDZERO_EMBEDDER", "hash"))
        except RunActive as e:
            raise ApiError(409, f"Refused: {e}") from None
        return j({"ok": True, "reset": done})

    # robots
    @r.get("/api/robots")
    async def list_robots(request):
        return j([profile_dict(p) for p in (*PROFILES.values(), *saved_profiles(ctx.data))])

    @r.post("/api/robots")
    async def save_robot(request):
        d = await _body(request)
        name = d.get("name")
        if not isinstance(name, str):
            raise BadRequest("'name' is required.")
        try:
            profile_path(name, ctx.data)
        except ValueError as e:
            raise BadRequest(str(e)) from None
        if name in PROFILES:
            raise BadRequest(f"'{name}' is the built-in robot; choose another name.")
        base = d.get("base", "mobile")
        if base not in ("static", "legged", "wheeled", "mobile"):
            raise BadRequest("'base' must be static, legged, wheeled or mobile.")
        controller = d.get("controller", "external")
        if controller not in ("builtin_claude", "external", "none"):
            raise BadRequest("'controller' must be builtin_claude, external or none.")
        reach = d.get("reach", [0.6, 1.3])
        if not (isinstance(reach, list) and len(reach) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                                    for v in reach) and 0 <= reach[0] < reach[1] <= 3):
            raise BadRequest("'reach' must be [lowest, highest] in metres, 0 to 3.")
        try:
            old = load_profile(name, ctx.data)
        except (KeyError, ValueError):
            old = None
        ns = d.get("ns") or name
        if not isinstance(ns, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,40}", ns):
            raise BadRequest("'ns' is a ROS namespace: letters, digits and underscores.")
        profile = RobotProfile(
            name=name, can_move=bool(d.get("can_move", base != "static")), base=base,
            radius=_number(d, "radius", 0.05, 2.0, 0.3), reach=(float(reach[0]), float(reach[1])),
            max_speed=_number(d, "max_speed", 0.05, 5.0, 1.0), ns=ns,
            sensors=old.sensors if old else (), urdf=old.urdf if old else None, controller=controller)
        save_profile(profile, ctx.data)
        return j(profile_dict(profile), 200 if old else 201)

    @r.delete("/api/robots/{name}")
    async def delete_robot(request):
        import shutil
        name = _robot_name(request)
        find_profile(ctx, name, editable=True)
        shutil.rmtree(profile_path(name, ctx.data).parent)
        return j({"ok": True})

    @r.post("/api/robots/{name}/urdf")
    async def robot_urdf(request):
        profile = find_profile(ctx, _robot_name(request), editable=True)
        d = await _body(request)
        folder = d.get("folder")
        if not isinstance(folder, str) or not folder.strip():
            raise BadRequest("'folder' is required: the folder holding the URDF.")
        info = await asyncio.to_thread(parse_urdf, Path(folder).expanduser())
        out = dataclasses.asdict(info)
        out["urdf_path"] = str(info.urdf_path) if info.urdf_path else None
        if info.urdf_path:
            save_profile(dataclasses.replace(profile, urdf=Path(info.urdf_path)), ctx.data)
        return j(out)

    @r.put("/api/robots/{name}/sensors")
    async def robot_sensors(request):
        profile = find_profile(ctx, _robot_name(request), editable=True)
        d = await _body(request)
        raw = d.get("sensors")
        if not isinstance(raw, list):
            raise BadRequest("'sensors' must be a list.")
        try:
            sensors = tuple(SensorSpec.from_dict(x) for x in raw)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            raise BadRequest(f"A sensor is not valid: {type(e).__name__} {e}") from None
        if len({s.id for s in sensors}) != len(sensors):
            raise BadRequest("Sensor ids must be unique.")
        profile = dataclasses.replace(profile, sensors=sensors)
        save_profile(profile, ctx.data)
        return j(profile_dict(profile))

    @r.post("/api/robots/{name}/convert")
    async def robot_convert(request):
        profile = find_profile(ctx, _robot_name(request), editable=True)
        if profile.urdf is None:
            raise BadRequest("Check the URDF first: this robot has no URDF yet.")
        folder = profile_path(profile.name, ctx.data).parent
        sensors = write_sensors_yaml(folder, profile.sensors)
        out = folder / "robot.usd"
        job = {"id": uuid.uuid4().hex[:12], "status": "running", "log": "", "result": None, "robot": profile.name}
        ctx.jobs[job["id"]] = job
        task = asyncio.create_task(_run_job(job, ctx.convert_command(profile.urdf, sensors, out)))
        ctx.tasks.add(task)
        task.add_done_callback(ctx.tasks.discard)
        return j({"job": job["id"]}, 202)

    @r.get("/api/jobs/{id}")
    async def get_job(request):
        job = ctx.jobs.get(request.match_info["id"])
        if job is None:
            raise ApiError(404, "no such job")
        return j(job)

    # connection
    @r.get("/api/connection")
    async def connection(request):
        from ..cli import ros_available
        conn = read_connection(ctx.data)
        try:
            profile = find_profile(ctx, request.query.get("robot", G1_NAME))
        except ApiError:
            profile = PROFILES[G1_NAME]
        ctx.probe_target = (conn["robot_domain"], profile.ns)  # the background probe looks here next
        return j({"ip": local_ip(), **conn, "ros": ros_available(), "topics": topic_table(profile),
                  "lights": dict(ctx.lights)})  # the last probe's result, never waited for

    @r.post("/api/connection")
    async def set_connection(request):
        d = await _body(request)
        dom = d.get("robot_domain")
        if isinstance(dom, bool) or not isinstance(dom, int) or not 0 <= dom <= 232:
            raise BadRequest("'robot_domain' must be a whole number from 0 to 232.")
        rmw = d.get("rmw")
        if rmw is not None and (not isinstance(rmw, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,60}", rmw)):
            raise BadRequest("'rmw' must be an RMW implementation name.")
        f = _connection_file(ctx.data)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(yaml.safe_dump({"robot_domain": dom, **({"rmw": rmw} if rmw else {})}))
        return j(read_connection(ctx.data))

    # run control
    @r.post("/api/run/start")
    async def run_start(request):
        d = await _body(request)
        pid = d.get("project_id")
        if not isinstance(pid, str):
            raise BadRequest("'project_id' is required.")
        project = ctx.projects.get(pid)
        conn = read_connection(ctx.data)
        try:
            run_id = await asyncio.to_thread(ctx.runner.start, project, conn["robot_domain"])
        except RunnerBusy as e:
            raise ApiError(409, str(e)) from None
        return j({"run_id": run_id}, 202)

    @r.post("/api/run/stop")
    async def run_stop(request):
        return j(await asyncio.to_thread(ctx.runner.stop))

    @r.get("/api/run/state")
    async def run_state(request):
        return j(await asyncio.to_thread(ctx.runner.state))

    @r.get("/api/runs")
    async def list_runs(request):
        return j(await asyncio.to_thread(list_project_runs, ctx.runs_dir, request.query.get("project")))

    @r.get("/runs/{run_id}/report.html")
    async def report(request):
        rid = request.match_info["run_id"]
        f = ctx.runs_dir / rid / "report.html"
        if not RUN_ID.match(rid) or not f.is_file():
            raise ApiError(404, "no report for that run")
        resp = web.Response(body=f.read_bytes(), content_type="text/html", charset="utf-8")
        resp.headers["Content-Security-Policy"] = ("default-src 'none'; style-src 'unsafe-inline'; "
                                                   "img-src data:; base-uri 'none'; form-action 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @r.get("/ws")
    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=15)
        await sock.prepare(request)
        request.app[SOCKETS].add(sock)
        try:
            while not sock.closed:
                try:
                    payload = _dumps(await asyncio.to_thread(ctx.runner.state))
                except Exception as e:
                    log.error("studio state failed: %s", type(e).__name__)
                    payload = _dumps({"error": "The run state is not available."})
                try:
                    await asyncio.wait_for(sock.send_str(payload), WS_SEND_TIMEOUT_S)
                except Exception:
                    break
                try:
                    msg = await asyncio.wait_for(sock.receive(), WS_PERIOD_S)
                except asyncio.TimeoutError:
                    continue
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        finally:
            request.app[SOCKETS].discard(sock)
            if not sock.closed:
                await sock.close()
        return sock

    async def on_shutdown(app):
        for s in list(app[SOCKETS]):
            await s.close()
        await asyncio.to_thread(ctx.runner.shutdown)

    async def start_probe(app):
        app[PROBE] = asyncio.create_task(refresh_lights(ctx))

    async def stop_probe(app):
        app[PROBE].cancel()
        try:
            await app[PROBE]
        except asyncio.CancelledError:
            pass

    app.on_startup.append(start_probe)
    app.on_cleanup.append(stop_probe)
    app.on_shutdown.append(on_shutdown)
    app.add_routes(r)
    if STATIC.is_dir():
        app.router.add_static("/static/", STATIC)
    return app


async def _run_job(job: dict, argv: list[str]) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
    except OSError as e:
        job["log"] = f"Could not start the converter: {e}. Set WARDZERO_ISAAC_PYTHON to Isaac Sim's python.sh."
        job["status"] = "failed"
        return
    assert proc.stdout is not None
    async for line in proc.stdout:
        job["log"] = (job["log"] + line.decode(errors="replace"))[-JOB_LOG_MAX:]
    code = await proc.wait()
    last = job["log"].strip().splitlines()[-1] if job["log"].strip() else ""
    try:
        job["result"] = json.loads(last)
    except ValueError:
        job["result"] = None
    ok = code == 0 and (job["result"] is None or job["result"].get("ok", True))
    job["status"] = "done" if ok else "failed"


def list_project_runs(runs_dir: Path, project: Optional[str]) -> list[dict]:
    out = []
    if not runs_dir.is_dir():
        return out
    for d in sorted(runs_dir.iterdir(), reverse=True):
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("project_id") is None:
            continue
        if project is not None and meta.get("project_id") != project:
            continue
        score = verdict = None
        try:
            rep = json.loads((d / "report.json").read_text())
            score, verdict = rep.get("score"), (rep.get("verdict") or {}).get("label")
        except (OSError, ValueError, AttributeError):
            pass
        out.append({"run_id": d.name, "project_id": meta.get("project_id"), "scenario_id": meta.get("scenario_id"),
                    "has_report": (d / "report.html").is_file(), "score": score, "verdict": verdict,
                    "mtime": d.stat().st_mtime})
    return out


def build_ctx(data: Optional[Path] = None, config_root: Optional[Path] = None,
              runs_dir: Optional[Path] = None, **kw) -> Ctx:
    from ..cli import runs_dir as default_runs
    from ..memory import data_dir
    data = Path(data) if data else data_dir()
    config_root = Path(config_root) if config_root else DEFAULT_CONFIG_ROOT
    runs_dir = Path(runs_dir) if runs_dir else default_runs()
    runner = kw.pop("runner", None) or Runner(data, config_root, runs_dir)
    return Ctx(data, config_root, runs_dir, runner, **kw)


def serve(port: int = 8770, open_browser: bool = True) -> int:
    async def main() -> None:
        runner = web.AppRunner(make_app(build_ctx()))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        url = f"http://127.0.0.1:{port}/"
        print(f"WardZero Studio: {url}  (Ctrl+C to quit)", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0
