"""The Studio's live preview: a per-run token in the URL, an MJPEG stream of <run>/preview/latest.jpg, and a clean end
when the run stops. The run is a fake subprocess that writes the preview file, never the real CLI or Isaac."""
import asyncio
import re
import sys
import textwrap
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from wardzero_agents.studio import runner as runner_mod
from wardzero_agents.studio.runner import Runner
from wardzero_agents.studio.server import Ctx, make_app

ROOT = Path(__file__).resolve().parents[1] / "config"
H = {"X-WardZero-Panel": "1"}
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60 + b"\xff\xd9"

FAKE = textwrap.dedent('''
    import json, os, signal, sys, time
    from pathlib import Path
    runs = Path(os.environ["WARDZERO_RUNS_DIR"]); pid = sys.argv[sys.argv.index("--project-id") + 1]
    d = runs / ("run-" + str(os.getpid())); (d / "preview").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "project_id": pid, "scenario_id": "ae_waiting"}))
    print("Run folder: " + str(d), flush=True)
    jpg = bytes.fromhex(sys.argv[sys.argv.index("--jpeg") + 1])
    def done(*a):
        print("Run finished (exit code 0).", flush=True); sys.exit(0)
    signal.signal(signal.SIGINT, done)
    n = 0
    while True:
        if "--nopreview" not in sys.argv:
            (d / "preview" / "latest.tmp").write_bytes(jpg + bytes([n % 256])); os.replace(d / "preview" / "latest.tmp", d / "preview" / "latest.jpg")
        n += 1; time.sleep(0.05)
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(data))
    monkeypatch.setenv("WARDZERO_RUNS_DIR", str(runs))
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE)
    extra = []

    def command(project, staged, bot, domain):
        return [sys.executable, str(script), "--project-id", project.id, "--jpeg", JPEG.hex(), *extra]

    runner = Runner(data, ROOT, runs, command=command, stop_publisher=lambda d: False, key_present=lambda: False)
    return {"data": data, "runs": runs, "runner": runner, "extra": extra}


@pytest.fixture
async def client(env):
    c = TestClient(TestServer(make_app(Ctx(env["data"], ROOT, env["runs"], env["runner"]))))
    await c.start_server()
    yield c
    await asyncio.to_thread(env["runner"].shutdown)
    await c.close()


async def start_run(client):
    p = await (await client.post("/api/projects", json={"name": "T", "scenario_id": "ae_waiting"}, headers=H)).json()
    assert (await client.post("/api/run/start", json={"project_id": p["id"]}, headers=H)).status == 202


async def until(fn, timeout=15.0):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        v = await fn()
        if v:
            return v
        await asyncio.sleep(0.1)
    raise AssertionError("timed out")


async def state(client):
    return await (await client.get("/api/run/state")).json()


async def preview_url(client):
    return (await state(client)).get("preview_url")


async def test_state_carries_a_preview_url_with_a_random_token_while_running(client):
    await start_run(client)
    url = await until(lambda: preview_url(client))
    m = re.fullmatch(r"/preview/([A-Za-z0-9_-]{32,})\.mjpg", url)
    assert m, url  # at least 192 bits of randomness, URL-safe
    assert (await state(client))["preview_url"] == url  # stable within one run


async def test_no_preview_url_without_a_preview_file(client, env):
    env["extra"].append("--nopreview")
    await start_run(client)
    await until(lambda: _has_run(client))
    assert "preview_url" not in await state(client)


async def _has_run(client):
    s = await state(client)
    return s["run_id"] and not s["run_id"].startswith("pending-")


async def test_the_stream_is_multipart_jpeg_and_follows_new_frames(client):
    await start_run(client)
    url = await until(lambda: preview_url(client))
    r = await client.get(url)  # an <img> sends no custom header: none is needed for the stream
    assert r.status == 200
    assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace; boundary=")
    assert r.headers["Cache-Control"] == "no-store" and r.headers["X-Content-Type-Options"] == "nosniff"
    boundary = r.headers["Content-Type"].split("boundary=")[1]
    seen = []
    buf = b""
    while len(seen) < 3:
        buf += await asyncio.wait_for(r.content.readany(), 5)
        parts = buf.split(b"--" + boundary.encode())
        buf = parts[-1]
        for part in parts[:-1]:
            if b"image/jpeg" in part:
                head, _, body = part.partition(b"\r\n\r\n")
                n = int(re.search(rb"Content-Length: (\d+)", head).group(1))
                assert body[:n].startswith(b"\xff\xd8") and len(body.rstrip(b"\r\n")) >= n - 1
                seen.append(body[n - 1:n])
    assert len(set(seen)) >= 2  # newer frames arrive, the same frame is not sent twice
    r.close()


async def test_wrong_token_and_foreign_hosts_are_refused(client):
    await start_run(client)
    url = await until(lambda: preview_url(client))
    assert (await client.get("/preview/" + "x" * 32 + ".mjpg")).status == 404
    assert (await client.get("/preview/short.mjpg")).status == 404
    assert (await client.get("/preview/..%2f..%2fapi%2fsite.mjpg")).status == 404
    assert (await client.get(url, headers={"Host": "evil.example"})).status == 403  # the panel's loopback-only check
    assert (await client.get(url, headers={"Origin": "http://evil.example"})).status == 403


async def test_the_stream_ends_when_the_run_stops_and_the_token_dies(client):
    await start_run(client)
    url = await until(lambda: preview_url(client))
    r = await client.get(url)
    await asyncio.wait_for(r.content.readany(), 5)
    assert (await client.post("/api/run/stop", headers=H)).status == 200  # no control subscriber: SIGINT
    rest = await asyncio.wait_for(r.content.read(), 10)  # reads to the end of the body: the server closed the stream
    assert isinstance(rest, bytes) and r.content.at_eof()
    await until(lambda: _done(client))
    assert "preview_url" not in await state(client)
    assert (await client.get(url)).status == 404


async def _done(client):
    return (await state(client))["phase"] == "done"


async def test_a_new_run_gets_a_new_token(client):
    await start_run(client)
    first = await until(lambda: preview_url(client))
    await client.post("/api/run/stop", headers=H)
    await until(lambda: _done(client))
    await start_run(client)

    async def fresh():
        u = await preview_url(client)
        return u if u and u != first else None
    second = await until(fresh)
    assert second != first and (await client.get(first)).status == 404


def test_default_command_asks_isaac_for_a_headless_preview_only_when_isaac(monkeypatch, tmp_path):
    from wardzero_agents.studio.project import Project
    proj = Project(id="p1", name="n", scenario_id="ae_waiting")
    monkeypatch.delenv("WARDZERO_STUDIO_SIM", raising=False)
    argv = runner_mod.default_command(proj, tmp_path, "none", None)
    assert argv[argv.index("--sim") + 1] == "stub" and "--preview" not in argv
    monkeypatch.setenv("WARDZERO_STUDIO_SIM", "isaac")
    argv = runner_mod.default_command(proj, tmp_path, "none", None)
    assert argv[argv.index("--sim") + 1] == "isaac" and "--preview" in argv and "--headed" not in argv
