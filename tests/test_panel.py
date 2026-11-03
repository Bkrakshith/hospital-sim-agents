import asyncio
import logging

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents.panel.server import make_app, run_panel

H = {"X-WardZero-Panel": "1"}


class FakeDirectorAPI:
    def __init__(self):
        self.fired = []
        self.controls = []
        self.chaos = []
        self.beats = []
        self.snaps = []
        self.reject = None
        self.boom = False

    def _maybe_boom(self):
        if self.boom:
            raise RuntimeError("sk-ant-SECRET traceback details")

    def state(self):
        self._maybe_boom()
        return {
            "t": 12.5, "running": True, "paused": False,
            "timeline": [{"id": "b1", "card": "coughing_fit", "target": "P2", "state": "fired", "fired_t": 3.0}],
            "cast": [{"id": "P2", "display_name": "Ann", "persona": "x", "x": -7.0, "y": 0.5,
                      "posture": "sitting", "mood": "calm", "last_decision": "wait",
                      "reason": "r", "last_said": ""}],
            "events": [], "chaos": {"enabled": False, "rate_per_hour": 6.0},
            "cost_usd": 0.1, "budget_usd": 20.0, "recording": True, "run_id": "r1",
            "deck": [{"id": "coughing_fit", "label": "Coughing fit", "target_kind": "person", "tests": "t"}],
            "map": {"bounds": [-10, 12, -3, 8], "seats": [{"id": "A1", "x": -8, "y": 0}],
                    "landmarks": [], "signs": [], "robot": {"x": 1, "y": 1, "yaw": 0}},
        }

    def control(self, action):
        self._maybe_boom()
        self.controls.append(action)

    def fire(self, card, target, params):
        self._maybe_boom()
        if self.reject:
            return {"ok": False, "error": self.reject}
        self.fired.append((card, target, params))
        return {"ok": True, "firing_id": "f1"}

    def set_chaos(self, enabled, rate_per_hour):
        self.chaos.append((enabled, rate_per_hour))

    def beat(self, action, beat_id):
        self.beats.append((action, beat_id))

    def snapshot(self, name):
        self.snaps.append(name)
        return {"ok": True, "name": name}


@pytest.fixture
def api():
    return FakeDirectorAPI()


@pytest.fixture
async def client(api):
    c = TestClient(TestServer(make_app(api)))
    await c.start_server()
    yield c
    await c.close()


async def test_state(client):
    r = await client.get("/api/state")
    assert r.status == 200 and "timeline" in await r.json()


async def test_fire_card(client, api):
    r = await client.post("/api/cards/fire", json={"card": "coughing_fit", "target": "P2", "params": {}}, headers=H)
    assert (await r.json())["ok"] and api.fired[-1][0] == "coughing_fit"


async def test_fire_rejected_shows_error(client, api):
    api.reject = "target P9 is not in the room"
    r = await client.post("/api/cards/fire", json={"card": "lost_visitor", "target": "P9", "params": {}}, headers=H)
    assert await r.json() == {"ok": False, "error": "target P9 is not in the room"}


async def test_fire_null_target_ok(client, api):
    r = await client.post("/api/cards/fire", json={"card": "fire_alarm", "target": None, "params": {}}, headers=H)
    assert r.status == 200 and api.fired[-1][1] is None


@pytest.mark.parametrize("body", [
    {}, {"card": ""}, {"card": 3, "target": None, "params": {}},
    {"card": "a", "target": 5, "params": {}}, {"card": "a", "target": None, "params": []},
    {"card": "a", "target": None},
])
async def test_fire_bad_body_400(client, body):
    r = await client.post("/api/cards/fire", json=body, headers=H)
    assert r.status == 400 and "error" in await r.json()


async def test_not_json_400(client):
    r = await client.post("/api/control", data="nope", headers=H)
    assert r.status == 400 and "error" in await r.json()
    r = await client.post("/api/control", json=[1], headers=H)
    assert r.status == 400


async def test_bad_control_400(client):
    assert (await client.post("/api/control", json={"action": "explode"}, headers=H)).status == 400


@pytest.mark.parametrize("a", ["start", "pause", "resume", "stop", "reset"])
async def test_control_ok(client, api, a):
    r = await client.post("/api/control", json={"action": a}, headers=H)
    assert r.status == 200 and api.controls[-1] == a


async def test_chaos(client, api):
    r = await client.post("/api/chaos", json={"enabled": True, "rate_per_hour": 4}, headers=H)
    assert r.status == 200 and api.chaos[-1] == (True, 4.0)
    for body in ({"enabled": "yes", "rate_per_hour": 4}, {"enabled": True, "rate_per_hour": -1},
                 {"enabled": True, "rate_per_hour": "x"}, {"enabled": True, "rate_per_hour": True},
                 {"enabled": True, "rate_per_hour": 1e9}, {"enabled": True}):
        assert (await client.post("/api/chaos", json=body, headers=H)).status == 400


async def test_beat(client, api):
    r = await client.post("/api/beat", json={"action": "skip", "beat_id": "b1"}, headers=H)
    assert r.status == 200 and api.beats[-1] == ("skip", "b1")
    assert (await client.post("/api/beat", json={"action": "delete", "beat_id": "b1"}, headers=H)).status == 400
    assert (await client.post("/api/beat", json={"action": "skip", "beat_id": ""}, headers=H)).status == 400


async def test_snapshot(client, api):
    r = await client.post("/api/snapshot", json={"name": "day_1.ok-2"}, headers=H)
    assert r.status == 200 and api.snaps == ["day_1.ok-2"]
    for n in ("../etc", "a b", "", "x" * 65, 5):
        assert (await client.post("/api/snapshot", json={"name": n}, headers=H)).status == 400
    assert api.snaps == ["day_1.ok-2"]


async def test_csrf_header_required(client, api):
    r = await client.post("/api/control", json={"action": "stop"})
    assert r.status == 403 and "error" in await r.json()
    assert api.controls == []
    r = await client.post("/api/control", json={"action": "stop"}, headers={"X-WardZero-Panel": "0"})
    assert r.status == 403


async def test_api_exception_is_500_without_leak(client, api, caplog):
    api.boom = True
    with caplog.at_level(logging.ERROR):
        r = await client.post("/api/control", json={"action": "stop"}, headers=H)
        s = await client.get("/api/state")
    for resp in (r, s):
        assert resp.status == 500
        text = await resp.text()
        assert "SECRET" not in text and "Traceback" not in text and "error" in text
    assert "SECRET" not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_index_has_no_external_scripts(client):
    r = await client.get("/")
    html = await r.text()
    assert r.status == 200 and "<script src=" not in html
    assert "http://" not in html and "https://" not in html
    assert "prefers-color-scheme" in html and "X-WardZero-Panel" in html and "<canvas" in html


async def test_ws_pushes_state(client):
    ws = await client.ws_connect("/ws")
    msg = await ws.receive_json(timeout=2)
    assert "cast" in msg and "map" in msg
    msg2 = await ws.receive_json(timeout=2)
    assert "cast" in msg2
    await ws.close()


async def test_ws_survives_state_error_and_disconnect(client, api):
    api.boom = True
    ws = await client.ws_connect("/ws")
    msg = await ws.receive_json(timeout=2)
    assert "error" in msg and "SECRET" not in str(msg)
    api.boom = False
    msg = await ws.receive_json(timeout=2)
    assert "cast" in msg
    await ws.close()
    await asyncio.sleep(0.1)
    ws2 = await client.ws_connect("/ws")
    assert "cast" in await ws2.receive_json(timeout=2)
    await ws2.close()


async def test_run_panel_refuses_non_local(api):
    with pytest.raises(ValueError):
        await run_panel(api, host="0.0.0.0")


async def test_run_panel_serves_local(api):
    import aiohttp
    runner = await run_panel(api, port=0)
    try:
        port = runner.addresses[0][1]
        assert runner.addresses[0][0] == "127.0.0.1"
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{port}/api/state") as r:
                assert r.status == 200
    finally:
        await runner.cleanup()


def test_index_is_packaged():
    from importlib import resources
    assert resources.files("wardzero_agents.panel").joinpath("static/index.html").is_file()
