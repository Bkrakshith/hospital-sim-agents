"""Scenario control panel server.

Binds to 127.0.0.1 only. The page and the JSON API are served by one aiohttp app.

DirectorAPI.state() must return a JSON-serialisable dict of exactly this shape
(Task 13 implements it; the page reads it):

    {
      "t": float,                  # scenario time, seconds
      "running": bool, "paused": bool,
      "run_id": str,
      "recording": bool,
      "cost_usd": float, "budget_usd": float,
      "chaos": {"enabled": bool, "rate_per_hour": float},
      "timeline": [ {"id", "card", "target", "state", "fired_t", "reason"?}, ... ],
                                   # StoryboardEngine.timeline()
      "cast": [ {"id", "display_name", "persona", "x", "y",
                 "posture",        # e.g. "standing", "sitting", "walking", "lying"
                 "mood", "last_decision", "reason", "last_said"}, ... ],
      "events": [ {"t": float, "text": str}, ... ],   # newest last; strings also accepted
      "deck": [ {"id", "label", "target_kind", "tests"}, ... ],
                                   # target_kind: "person" | "all" | "none"
      "map": {                     # static
        "bounds": [xmin, xmax, ymin, ymax],           # [-10, 12, -3, 8]
        "seats": [ {"id", "x", "y"} ],
        "landmarks": [ {"id", "label", "x", "y"} ],
        "signs": [ {"id", "text", "x", "y"} ],
        "robot": {"x", "y", "yaw"}
      }
    }

Seat occupancy is derived by the page from cast positions (a seat is taken when a
person is within 0.4 m of it).

Other methods: control(action) with action in start|pause|resume|stop|reset;
fire(card, target, params) -> {"ok": True, "firing_id"} or {"ok": False, "error"};
set_chaos(enabled, rate_per_hour); beat("skip"|"rerun", beat_id); snapshot(name) -> dict.
All are plain sync methods and must return quickly: handlers call them on the event loop.

Security: POST routes need the header X-WardZero-Panel: 1 (CSRF); the Host header must
be a loopback name (DNS rebinding); API exceptions give a generic 500 and only the
exception type is logged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Literal, Optional, Protocol

from aiohttp import WSMsgType, web

log = logging.getLogger("wardzero.panel")

CSRF_HEADER = "X-WardZero-Panel"
CONTROL_ACTIONS = ("start", "pause", "resume", "stop", "reset")
BEAT_ACTIONS = ("skip", "rerun")
SNAPSHOT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
WS_PERIOD_S = 0.5
WS_SEND_TIMEOUT_S = 2.0
MAX_BODY = 64 * 1024
MAX_RATE = 10000.0
STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


API_KEY = web.AppKey("api", object)
SOCKETS_KEY = web.AppKey("sockets", set)


class DirectorAPI(Protocol):
    def state(self) -> dict: ...
    def control(self, action: str) -> None: ...
    def fire(self, card: str, target: Optional[str], params: dict) -> dict: ...
    def set_chaos(self, enabled: bool, rate_per_hour: float) -> None: ...
    def beat(self, action: Literal["skip", "rerun"], beat_id: str) -> None: ...
    def snapshot(self, name: str) -> dict: ...


class BadRequest(Exception):
    pass


def _err(status: int, msg: str) -> web.Response:
    return web.json_response({"error": msg}, status=status)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


async def _body(request: web.Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        raise BadRequest("The request body must be valid JSON.")
    if not isinstance(data, dict):
        raise BadRequest("The request body must be a JSON object.")
    return data


def _str(data: dict, key: str, maxlen: int = 200, nullable: bool = False) -> Optional[str]:
    v = data.get(key)
    if v is None and nullable and key in data:
        return None
    if not isinstance(v, str) or not v.strip():
        raise BadRequest(f"'{key}' is required and must be a non-empty text value.")
    if len(v) > maxlen:
        raise BadRequest(f"'{key}' is too long.")
    return v


@web.middleware
async def _guard(request: web.Request, handler):
    try:
        host = urlsplit("//" + request.headers.get("Host", "")).hostname
    except ValueError:
        host = None
    if host not in ALLOWED_HOSTS:
        return _err(403, "This panel only answers on localhost.")
    if request.method == "POST" and request.headers.get(CSRF_HEADER) != "1":
        return _err(403, f"Missing the {CSRF_HEADER}: 1 header.")
    try:
        return await handler(request)
    except BadRequest as e:
        return _err(400, str(e))
    except web.HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:  # never leak details: they may carry secrets
        log.error("panel API call failed: %s", type(e).__name__)
        return _err(500, "The director could not do that. See the run log.")


def make_app(api: DirectorAPI) -> web.Application:
    app = web.Application(middlewares=[_guard], client_max_size=MAX_BODY)
    app[API_KEY] = api
    app[SOCKETS_KEY] = set()

    async def index(request):
        resp = web.Response(
            body=(STATIC / "index.html").read_bytes(), content_type="text/html", charset="utf-8")
        resp.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    async def get_state(request):
        return web.Response(text=_dumps(api.state()), content_type="application/json")

    async def control(request):
        d = await _body(request)
        a = d.get("action")
        if a not in CONTROL_ACTIONS:
            raise BadRequest("'action' must be one of: " + ", ".join(CONTROL_ACTIONS) + ".")
        api.control(a)
        return web.json_response({"ok": True})

    async def fire(request):
        d = await _body(request)
        card = _str(d, "card")
        if "target" not in d:
            raise BadRequest("'target' is required (use null for none).")
        target = _str(d, "target", nullable=True)
        params = d.get("params")
        if not isinstance(params, dict):
            raise BadRequest("'params' is required and must be an object.")
        return web.Response(text=_dumps(api.fire(card, target, params)), content_type="application/json")

    async def chaos(request):
        d = await _body(request)
        en, rate = d.get("enabled"), d.get("rate_per_hour")
        if not isinstance(en, bool):
            raise BadRequest("'enabled' must be true or false.")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) \
                or rate < 0 or rate > MAX_RATE:
            raise BadRequest(f"'rate_per_hour' must be a number from 0 to {int(MAX_RATE)}.")
        api.set_chaos(en, float(rate))
        return web.json_response({"ok": True})

    async def beat(request):
        d = await _body(request)
        a = d.get("action")
        if a not in BEAT_ACTIONS:
            raise BadRequest("'action' must be skip or rerun.")
        api.beat(a, _str(d, "beat_id"))
        return web.json_response({"ok": True})

    async def snapshot(request):
        d = await _body(request)
        n = d.get("name")
        if not isinstance(n, str) or not SNAPSHOT_RE.match(n):
            raise BadRequest("'name' must be 1 to 64 letters, digits, dot, dash or underscore.")
        return web.Response(text=_dumps(api.snapshot(n)), content_type="application/json")

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=15)
        await sock.prepare(request)
        request.app[SOCKETS_KEY].add(sock)
        try:
            while not sock.closed:
                try:
                    payload = _dumps(api.state())
                except Exception as e:
                    log.error("panel state() failed: %s", type(e).__name__)
                    payload = _dumps({"error": "The director state is not available."})
                try:
                    await asyncio.wait_for(sock.send_str(payload), WS_SEND_TIMEOUT_S)
                except Exception:
                    break  # dead or too slow
                # wait one tick but stay responsive to the client's close frame
                try:
                    msg = await asyncio.wait_for(sock.receive(), WS_PERIOD_S)
                except asyncio.TimeoutError:
                    continue
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            request.app[SOCKETS_KEY].discard(sock)
            if not sock.closed:
                try:
                    await sock.close()
                except Exception:
                    pass
        return sock

    async def on_shutdown(app):
        for s in list(app[SOCKETS_KEY]):
            try:
                await s.close()
            except Exception:
                pass

    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/api/state", get_state)
    app.router.add_post("/api/control", control)
    app.router.add_post("/api/cards/fire", fire)
    app.router.add_post("/api/chaos", chaos)
    app.router.add_post("/api/beat", beat)
    app.router.add_post("/api/snapshot", snapshot)
    app.router.add_get("/ws", ws)
    return app


async def run_panel(api: DirectorAPI, host: str = "127.0.0.1", port: int = 8765) -> web.AppRunner:
    if host != "127.0.0.1":
        raise ValueError("The panel binds to 127.0.0.1 only.")
    runner = web.AppRunner(make_app(api))
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
