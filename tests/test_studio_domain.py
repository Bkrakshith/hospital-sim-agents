"""The robot's ROS domain: a default of its own (never the private one), refused when set equal to it."""
import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents.studio import server as S
from wardzero_agents.studio.runner import Runner

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}


@pytest.mark.parametrize("private,want", [(0, 1), (44, 45), (100, 101), (101, 0)])
def test_the_default_is_private_plus_one_wrapping_within_0_101(private, want):
    assert S.default_robot_domain(private) == want


def test_the_default_skips_the_domains_to_avoid_and_never_the_private_one():
    assert S.default_robot_domain(5, avoid=(6, 7)) == 8
    assert S.default_robot_domain(101, avoid=(0,)) == 1
    for p in range(0, 102):
        assert S.default_robot_domain(p) != p


def test_read_connection_gives_a_default_domain_unless_the_user_set_one(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "44")
    c = S.read_connection(tmp_path)
    assert (c["robot_domain"], c["explicit"], c["private_domain"]) == (45, False, 44)
    monkeypatch.delenv("ROS_DOMAIN_ID")
    assert S.read_connection(tmp_path)["robot_domain"] == 1  # unset means private 0
    (tmp_path / "studio").mkdir()
    (tmp_path / "studio" / "connection.yaml").write_text("robot_domain: 12\n")
    c = S.read_connection(tmp_path)
    assert (c["robot_domain"], c["explicit"]) == (12, True)


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "44")
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    started = []
    runner = Runner(data, ROOT, runs, command=lambda p, s, b, d: started.append(d) or ["true"],
                    stop_publisher=lambda d: False, key_present=lambda: False)
    c = TestClient(TestServer(S.make_app(S.Ctx(data, ROOT, runs, runner))))
    await c.start_server()
    c.started, c.data = started, data
    yield c
    await asyncio.to_thread(runner.shutdown)
    await c.close()


async def test_the_api_shows_the_default_and_a_run_uses_it(client):
    d = await (await client.get("/api/connection")).json()
    assert d["robot_domain"] == 45 and d["explicit"] is False and d["private_domain"] == 44
    p = await (await client.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting"}, headers=H)).json()
    assert (await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202
    assert client.started == [45]  # the robot gets its own domain with no setting from the user


async def test_setting_the_robot_domain_to_the_private_one_is_refused(client):
    r = await client.post("/api/connection", json={"robot_domain": 44}, headers=H)
    assert r.status == 400
    msg = (await r.json())["error"]
    assert "44" in msg and "domain of its own" in msg
    assert (await client.post("/api/connection", json={"robot_domain": 45}, headers=H)).status == 200


async def test_a_hand_edited_connection_file_cannot_start_a_run_on_the_private_domain(client):
    (client.data / "studio").mkdir(parents=True)
    (client.data / "studio" / "connection.yaml").write_text("robot_domain: 44\n")
    p = await (await client.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting"}, headers=H)).json()
    r = await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)
    assert r.status == 400 and "domain of its own" in (await r.json())["error"] and client.started == []
