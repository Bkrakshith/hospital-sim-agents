"""The Studio in a real browser at 1440 x 900 (QA bug 28): the Simulator group says "Simulator" once, the cost meter
stays on one line, a speech bubble never covers a person or the robot, "Use this scenario" shows the character list
from its top, and a one-step branch of "At the same time" gets no extra "In order" row.

Needs Playwright and Chrome (/usr/bin/google-chrome or WARDZERO_TEST_CHROME); skipped without them."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
CHROME = os.environ.get("WARDZERO_TEST_CHROME", "/usr/bin/google-chrome")
pytestmark = pytest.mark.skipif(not Path(CHROME).exists(), reason="no Chrome for Playwright")
ROOT = Path(__file__).resolve().parents[1]

# the run state arrives over a WebSocket: a stand-in the test pushes states through
FAKE_WS = r"""
(() => {
  const subs = [];
  class FakeWS { constructor() { subs.push(this); setTimeout(() => this.onopen && this.onopen(), 0); } close() {} send() {} }
  window.WebSocket = FakeWS;
  window.__push = (s) => subs.forEach((w) => w.onmessage && w.onmessage({ data: JSON.stringify(s) }));
})();
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def studio_url(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("studio_ui")
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI"))}
    env.update(WARDZERO_DATA_DIR=str(tmp / "data"), WARDZERO_RUNS_DIR=str(tmp / "runs"), WARDZERO_TEST_FAKE_LLM="1",
               PYTHONPATH=os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]))
    proc = subprocess.Popen([sys.executable, "-m", "wardzero_agents.cli", "studio", "--port", str(port), "--no-browser"],
                            env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(url + "api/presets", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("the Studio did not start")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def page(studio_url):
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.add_init_script(FAKE_WS)
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(studio_url)
        pg.wait_for_timeout(1500)
        pg.errors = errors
        yield pg
        b.close()


def _use_reception(page, url):
    page.goto(url + "#/scene")
    page.wait_for_timeout(800)
    page.click(".seg-btn:has-text('Scenarios')")
    page.wait_for_timeout(800)
    page.locator(".card-btn:has-text('Reception and guide')").first.click()
    page.wait_for_timeout(1200)
    before = page.locator(".side-body").first.evaluate("(b) => b.scrollTop")
    page.locator(".card.open button:has-text('Use this scenario')").first.click()
    page.wait_for_timeout(500)
    if page.locator("dialog button:has-text('Replace characters')").count():
        page.locator("dialog button:has-text('Replace characters')").click()
    page.wait_for_timeout(1000)
    return before, page.locator(".side-body").first.evaluate("(b) => b.scrollTop")


def test_use_this_scenario_shows_the_characters_from_the_top(page, studio_url):
    before, after = _use_reception(page, studio_url)
    assert before > 0  # the card was scrolled into view, down the list
    assert after == 0
    assert not page.errors


def test_the_simulator_group_says_simulator_once(page, studio_url):
    page.goto(studio_url + "#/run")
    page.wait_for_timeout(1200)
    shown = page.evaluate("""() => [...document.querySelectorAll('.sim-group *')].filter((e) => e.children.length === 0
        && e.textContent.trim() === 'Simulator' && e.getBoundingClientRect().width > 2).length""")
    assert shown == 1
    assert page.evaluate("() => !!document.querySelector('label[for=s-sim]')")  # still labelled for screen readers


def _live_state(page):
    chars = page.evaluate("() => WZ.project.characters.map((c) => ({id: c.id, x: c.x, y: c.y}))")
    sp, other = chars[0], chars[1]
    other = {**other, "x": sp["x"] + 1.6, "y": sp["y"] + 1.2}  # right where the first bubble spot is
    people = [sp, other] + chars[2:]
    return {"running": True, "run_id": "r1", "project_id": None, "t": 30.0, "cost_usd": 12.34, "budget_usd": 25.0,
            "tokens": {"input": 1234567, "output": 123456, "cache_read": 9876543, "cache_write": 0},
            "people": people, "robot": {"x": sp["x"] - 1.0, "y": sp["y"] + 0.6, "yaw": 0.0}, "objects": [],
            "says_tail": [{"t": 29.0, "person": sp["id"], "text": "Excuse me, can you help me find the ward where my "
                                                                  "father is?"}],
            "flow": None, "phase": "running", "has_report": False}


def test_live_run_cost_on_one_line_and_bubbles_clear_of_markers(page, studio_url):
    page.goto(studio_url + "#/run")
    page.wait_for_timeout(1000)
    st = _live_state(page)
    for _ in range(2):
        page.evaluate("(s) => window.__push(s)", st)
        page.wait_for_timeout(600)
    page.wait_for_timeout(800)
    lines = page.evaluate("""() => { const e = document.getElementById('cost-text');
        return Math.round(e.getBoundingClientRect().height / parseFloat(getComputedStyle(e).lineHeight)); }""")
    assert lines == 1
    assert page.evaluate("() => { const t = document.querySelector('.top'); return t.scrollWidth - t.clientWidth; }") <= 0
    assert "in" in page.evaluate("() => document.getElementById('cost').title")  # the token counts, when hidden
    assert page.evaluate("() => document.querySelectorAll('.run-page svg .bub').length") == 1
    covered = page.evaluate("""() => {
      const out = [];
      const bubs = [...document.querySelectorAll('.run-page svg .bub')].map((b) => b.getBoundingClientRect());
      for (const g of document.querySelectorAll('.run-page svg g.person, .run-page svg g.robot')) {
        const c = g.querySelector('circle.dot') || g.querySelector('circle, path'); if (!c) continue;
        const r = c.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
        if (bubs.some((q) => x > q.left && x < q.right && y > q.top && y < q.bottom)) out.push(g.textContent.trim());
        const hit = document.elementFromPoint(x, y);
        if (hit && !g.contains(hit)) out.push('hidden: ' + g.textContent.trim());
      }
      return out; }""")
    assert covered == []


def test_bubbles_clear_a_lying_patients_marker(page, studio_url):
    """A lying patient (a person the project has no character for) sits where the first bubble spot is."""
    page.goto(studio_url + "#/run")
    page.wait_for_timeout(1000)
    st = _live_state(page)
    sp = st["people"][0]
    st["people"] = [sp, {"id": "bed_patient_x", "x": sp["x"] + 1.6, "y": sp["y"] + 1.2}] + st["people"][2:]
    for _ in range(2):
        page.evaluate("(s) => window.__push(s)", st)
        page.wait_for_timeout(600)
    page.wait_for_timeout(800)
    assert page.evaluate("() => document.querySelectorAll('.run-page svg .bub').length") == 1
    covered = page.evaluate("""() => {
      const bubs = [...document.querySelectorAll('.run-page svg .bub')].map((b) => b.getBoundingClientRect());
      return [...document.querySelectorAll('.run-page svg .static circle')].filter((c) => {
        const r = c.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
        return bubs.some((q) => x > q.left && x < q.right && y > q.top && y < q.bottom); }).length; }""")
    assert page.evaluate("() => document.querySelectorAll('.run-page svg .static circle').length") >= 1
    assert covered == 0


def test_one_step_branches_get_no_extra_in_order_row(page):
    res = page.evaluate("""async () => {
      const pr = await API.post('/api/projects', { name: 'flow rows', scenario_id: 'evacuation_guide' });
      const d = await API.get('/api/projects/' + pr.id + '/flow');
      await API.del('/api/projects/' + pr.id);
      let single = 0;
      const walk = (n) => { for (const c of n.children || []) { if (n.kind === 'parallel' && c.kind === 'sequence' &&
        (c.children || []).length === 1) single++; walk(c); } };
      walk(d.tree);
      const el = WZ.flowTree(d.tree);
      const extra = [...el.querySelectorAll('li.n-parallel > ul > li.n-sequence')]
        .filter((li) => li.querySelectorAll(':scope > ul > li').length === 1).length;
      return { single, extra }; }""")
    assert res["extra"] == 0
