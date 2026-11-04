"""The Studio page is served as plain same-origin HTML, CSS and JS: no external scripts, a CSP header."""
import re

import pytest

from tests.test_studio_api import client, env  # noqa: F401  (fixtures)

PAGES = ["index.html", "studio.css", "studio.js", "api.js", "map.js", "pages/scene.js"]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PAGES[1:])
async def test_assets_served_same_origin(client, path):
    r = await client.get("/static/" + path)
    assert r.status == 200
    body = await r.text()
    assert not re.search(r"""(src|href)\s*=\s*["']https?:""", body)
    assert not re.search(r"""\bimport\s.*["']https?:|fetch\(\s*["']https?:""", body)
