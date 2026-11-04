"""The Studio page is served as plain same-origin HTML, CSS and JS: no external scripts, a CSP header."""
import re

import pytest

from tests.test_studio_api import client, env  # noqa: F401  (fixtures)

PAGES = ["index.html", "studio.css", "studio.js", "api.js", "map.js", "pages/scene.js", "pages/robot.js", "pages/run.js", "pages/report.js"]


@pytest.mark.asyncio
async def test_index_served_with_csp_and_no_external_scripts(client):
    r = await client.get("/")
    assert r.status == 200
    html = await r.text()
    csp = r.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp
    assert not re.search(r"https?://", csp)
    assert not re.search(r"<script[^>]+src=[\"']?(https?:)?//", html, re.I)
    assert not re.search(r"<link[^>]+href=[\"']?(https?:)?//", html, re.I)
    assert "WardZero Studio" in html
    assert "frame-src 'self'" in csp
    for page in ("robot", "run", "report"):
        assert f"/static/pages/{page}.js" in html


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PAGES[1:])
async def test_assets_served_same_origin(client, path):
    r = await client.get("/static/" + path)
    assert r.status == 200
    body = await r.text()
    assert not re.search(r"""(src|href)\s*=\s*["']https?:""", body)
    assert not re.search(r"""\bimport\s.*["']https?:|fetch\(\s*["']https?:""", body)


@pytest.mark.asyncio
async def test_sensor_catalogue_and_batches(client):
    cat = await (await client.get("/api/sensors/catalogue")).json()
    assert cat and {"id", "kind", "source"} <= set(cat[0]) and "usd" not in cat[0]
    assert await (await client.get("/api/batches?scenario=none")).json() == []
    d = client.env["runs"] / "batch-ae_waiting-20261104-000000"
    d.mkdir(parents=True)
    (d / "batch.json").write_text('{"scenario_id": "ae_waiting", "title": "T", "n": 2, "arms": {"primary": {"pass_rate": {"k": 1, "n": 2}}}}')
    (d / "report.html").write_text("<p>x</p>")
    rows = await (await client.get("/api/batches?scenario=ae_waiting")).json()
    assert rows[0]["run_id"] == d.name and rows[0]["passed"] == 1 and rows[0]["has_report"]
    assert (await client.get(f"/runs/{d.name}/report.html")).status == 200
