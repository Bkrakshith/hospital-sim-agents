/* Page 3, Run: settings and Start; while a run is going only Stop works.
   The live map shows people and the robot moving, speech bubbles for four seconds and a transcript tail.
   The top bar (studio.js) shows cost and tokens. After Stop the phase chips run debrief, report, done. */
(function () {
  const { h } = window;
  const SVGNS = "http://www.w3.org/2000/svg";
  const PHASES = [["running", "Running"], ["debrief", "Debrief"], ["report", "Report"], ["done", "Done"]];
  const BUBBLE_S = 4;
  const clock = (t) => { t = Math.max(0, Math.round(t || 0)); return Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0"); };
  const svgEl = (tag, attrs, parent) => { const n = document.createElementNS(SVGNS, tag); for (const k in attrs || {}) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };

  function mount(root) {
    const S = { map: null, dead: false, pos: new Map(), raf: 0, framed: false, stopping: false, lastDraw: 0, starting: false, live: false, wasRunning: false };
    const P = () => WZ.project;

    const side = h("aside.side");
    const body = h("div.side-body");
    side.append(body);
    const mapBox = h("section.mapbox", { "aria-label": "Live map" });
    const svg = svgEl("svg", { role: "img", "aria-label": "Live top-down map of the run" });
    const chip = h("div.map-chip");
    const preview = h("img#preview", { alt: "Isaac Sim overhead view", hidden: true });
    const tools = h("div.map-tools",
      h("button.icon-btn", { type: "button", title: "Frame everyone", "aria-label": "Frame everyone", onclick: () => frame(true) }, icon("M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5")));
    mapBox.append(svg, chip, preview, tools, h("div.map-hint", "Scroll to zoom, drag to pan"));
    root.append(h("div.run-page", side, mapBox));

    function icon(path) {
      const s = svgEl("svg", { viewBox: "0 0 24 24", width: 18, height: 18, fill: "none", stroke: "currentColor", "stroke-width": 1.9, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" });
      svgEl("path", { d: path }, s); return s;
    }

    S.map = new HospitalMap(svg, WZ.site, {});
    S.gBubbles = svgEl("g", {}, S.map.gMarkers);

    // ---- settings (left) ----
    const set = {};
    const live = () => !!(WZ.state && WZ.state.running);
    const finished = () => WZ.state && WZ.state.run_id && !WZ.state.running && WZ.state.phase === "done";
    async function save(patch, input, back) {
      try { await WZ.saveProject(patch); }
      catch (e) { WZ.toast(e.message, "bad"); if (input) input.value = back; }
    }
    function numField(key, label, min, max, step, unit, opts) {
      const o = opts || {};
      const input = set[key] = h("input.text", { type: "number", id: "s-" + key, min, max, step, value: o.get ? o.get() : P()[key] });
      input.addEventListener("change", () => {
        let v = parseFloat(input.value);
        if (!Number.isFinite(v) || v < min || v > max) { WZ.toast(label + " must be between " + min + " and " + max + ".", "bad"); input.value = o.get ? o.get() : P()[key]; return; }
        if (o.int) v = Math.round(v);
        save({ [key]: v }, input, P()[key]);
      });
      return h("div.field.inline", h("label.flabel", { for: "s-" + key }, label), h("div.numwrap", input, unit ? h("em", unit) : null));
    }
    function switchRow(key, title, sub) {
      const input = set[key] = h("input", { type: "checkbox", id: "s-" + key, role: "switch", checked: !!P()[key], onchange: (e) => save({ [key]: e.target.checked }, null).then(() => {}) });
      return h("label.switch-row.flat", { for: "s-" + key }, input, h("span.track", h("i")), h("span.sw-text", h("b", title), h("small", sub)));
    }
    function wom() {
      const val = h("output.pct", { for: "s-wom" }, Math.round(P().word_of_mouth_p * 100) + "%");
      const input = set.word_of_mouth_p = h("input", { type: "range", id: "s-wom", min: 0, max: 100, step: 5, value: Math.round(P().word_of_mouth_p * 100),
        oninput: () => { val.textContent = input.value + "%"; }, onchange: () => save({ word_of_mouth_p: input.value / 100 }, null) });
      return h("div.field", h("div.flabel", h("label", { for: "s-wom" }, "Word of mouth"), val),
        input, h("small.muted", "How likely people are to tell each other what they thought of the robot."));
    }

    const startBtn = h("button.btn.primary.wide.big", { type: "button", onclick: start }, "Start run");
    const stopBtn = h("button.btn.danger-solid.wide.big", { type: "button", onclick: stop }, "Stop run");
    const stateBox = h("div.runstate");
    const phases = h("ol.phases", { "aria-label": "Run progress" });
    const tail = h("ul.tail", { "aria-label": "What was said" });
    const tailBox = h("section.rsec", h("h3.sub", "What was said"), tail, h("p.muted.small.empty-tail", "Nothing said yet. People start talking once the run begins."));
    const reportBtn = h("button.btn.primary.wide", { type: "button", hidden: true, onclick: () => { location.hash = "#/report"; } }, "Open the report");
    const errBox = h("p.error");

    function drawSettings() {
      const p = P();
      body.replaceChildren(
        h("section.rsec",
          h("div.sec-head", h("h2", "Run")),
          h("p.muted.lead", scenarioTitle() + ". " + p.characters.length + " " + (p.characters.length === 1 ? "person" : "people") + ", robot " + (p.robot_profile === "g1_claude" ? "built-in G1 (Claude)" : p.robot_profile) + ". Nobody steers the run: once it starts you can only stop it."),
          h("fieldset.settings", { id: "settings" },
            numField("duration_min", "Length", 1, 60, 1, "min"),
            numField("budget_usd", "Spending limit", 0.5, 1000, 0.5, "USD"),
            numField("repeat", "Repeats", 1, 20, 1, "runs", { int: true }),
            wom(),
            switchRow("carry_memory", "Carry memory between repeats", "People remember the earlier run. Applies when repeats is more than 1."),
            switchRow("baseline", "Also run the built-in robot", "Runs the same seeds with the built-in G1 to compare."))),
        h("section.rsec.actions", startBtn, stopBtn, errBox, stateBox, phases, reportBtn),
        tailBox);
      refresh();
    }
    const scenarioTitle = () => { const s = WZ.presets.scenarios.find((x) => x.id === P().scenario_id); return s ? s.title : P().scenario_id; };

    // ---- control ----
    async function start() {
      errBox.textContent = ""; S.starting = true; refresh();
      try {
        const r = await API.post("/api/run/start", { project_id: P().id });
        WZ.state = Object.assign({}, WZ.state, { running: true, phase: "running", run_id: r.run_id, has_report: false });
      } catch (e) { errBox.textContent = e.message; }
      S.starting = false; refresh();
    }
    async function stop() {
      S.stopping = true; refresh();
      try {
        const r = await API.post("/api/run/stop");
        if (r && r.stopped === false && r.note) { WZ.toast(r.note); S.stopping = false; }
      } catch (e) { errBox.textContent = e.message; S.stopping = false; }
      refresh();
    }

    function refresh() {
      const s = WZ.state || {}, running = !!s.running;
      startBtn.hidden = running; stopBtn.hidden = !running;
      startBtn.disabled = S.starting || running;
      const canStop = running && (s.phase === "running" || !s.phase) && !S.stopping;
      stopBtn.disabled = !canStop;
      stopBtn.textContent = S.stopping || (running && s.phase && s.phase !== "running") ? (s.phase === "debrief" ? "Debrief in progress" : s.phase === "report" ? "Writing the report" : "Stopping…") : "Stop run";
      const fs = document.getElementById("settings"); if (fs) fs.disabled = running;
      // phases
      const ph = s.phase, idx = PHASES.findIndex((x) => x[0] === ph);
      const show = !!s.run_id && (running || ph === "done" || S.stopping);
      phases.hidden = !show;
      phases.replaceChildren(...PHASES.map(([id, label], i) => h("li.phase" + (i < idx ? ".past" : i === idx ? ".now" : ""), { "aria-current": i === idx ? "step" : false }, h("i"), label)));
      stateBox.textContent = running ? (ph === "debrief" ? "The people are being asked how it went." : ph === "report" ? "Writing the report. This takes a moment." : "Time " + clock(s.t) + " of " + clock(P().duration_min * 60))
        : ph === "done" && s.run_id ? (s.error ? "The run ended with a problem." : "The run is finished.") : "";
      if (!running && ph === "done" && s.error) errBox.textContent = s.error;
      reportBtn.hidden = !(!running && s.has_report);
      chip.textContent = running ? (ph === "running" || !ph ? "Live · " + clock(s.t) : "Run ending") : (ph === "done" && s.run_id ? "Run finished" : "Start layout");
      // transcript
      if (s.says_tail && s.says_tail.length) { WZ.lastSays = { run: s.run_id, says: s.says_tail }; }
      else if (s.running && WZ.lastSays && WZ.lastSays.run !== s.run_id) WZ.lastSays = null;
      const says = ((WZ.lastSays && WZ.lastSays.says) || []).slice(-8);
      tail.replaceChildren(...says.map((e) => h("li", h("time", clock(e.t)), h("b", nameOf(e.person)), h("span", e.text || ""))));
      tailBox.querySelector(".empty-tail").hidden = says.length > 0;
      // overhead preview slot
      if (s.preview_url) { if (preview.getAttribute("src") !== s.preview_url) preview.setAttribute("src", s.preview_url); preview.hidden = false; }
      else { preview.hidden = true; preview.removeAttribute("src"); }
    }

    // ---- map ----
    const nameOf = (id) => {
      if (id === "robot") return "Robot";
      const c = P().characters.find((x) => x.id === id || x.name === id);
      return c ? c.name.split(" ")[0] : String(id || "?");
    };
    const charOf = (id) => P().characters.find((x) => x.id === id || x.name === id) || { id, name: String(id || "?"), role: "visitor" };
    const startRobot = () => P().robot_start || { x: 0, y: 0, yaw: 0 };
    function frame(animate) {
      const pts = P().characters.map((c) => ({ x: c.x, y: c.y })).concat([startRobot()]);
      const s = WZ.state || {};
      if (s.running && s.people && s.people.length) for (const q of s.people) if (Number.isFinite(q.x)) pts.push({ x: q.x, y: q.y });
      S.map.frame([], pts, 0.18, 16, !!animate);
    }
    function drawStatic() {
      const p = P();
      S.map.setStaticPatients(p.static_patients ? WZ.site.beds.filter((b) => /^B[1-9][0-9]*$/.test(b.id) && !p.characters.some((c) => c.bed === b.id)) : []);
    }
    function wrap(text, max) {
      const words = String(text).split(/\s+/), lines = [""];
      for (const w of words) {
        const cur = lines[lines.length - 1];
        if ((cur + " " + w).trim().length <= max) lines[lines.length - 1] = (cur + " " + w).trim(); else lines.push(w);
      }
      if (lines.length > 3) { lines.length = 3; lines[2] = lines[2].replace(/.{0,2}$/, "") + "…"; }
      return lines;
    }
    /* Place a bubble above-right, above-left or below the speaker, taking the first spot that covers
       no other bubble and no room label (screen pixels). `placed` collects the rectangles used. */
    function bubble(x, y, text, who, placed, labels) {
      const { g } = S.map._marker(S.gBubbles, x, y);
      const lines = wrap(text, 30), cw = 6.4, w = Math.max(...lines.map((l) => l.length), who.length + 2) * cw + 20, hgt = lines.length * 15 + 30;
      const r = svg.getBoundingClientRect(), v = S.map.vb;
      const sx = r.left + ((x - v.x) / v.w) * r.width, sy = r.top + ((-y - v.y) / v.h) * r.height;
      const cands = [{ dx: -16, dy: -24 - hgt, tail: "down" }, { dx: -w + 16, dy: -24 - hgt, tail: "down" }, { dx: -w / 2, dy: 24, tail: "up" },
        { dx: 50, dy: -hgt / 2 }, { dx: -w - 50, dy: -hgt / 2 }, { dx: -16, dy: -48 - hgt, tail: "down" }, { dx: -w + 16, dy: -48 - hgt, tail: "down" }];
      const rect = (c) => { const left = sx + c.dx, top = sy + c.dy; return { l: left, t: top, r: left + w, b: top + hgt }; };
      const hit = (a, bb) => a.l < bb.right && a.r > bb.left && a.t < bb.bottom && a.b > bb.top;
      let pick = cands.find((c) => { const q = rect(c); return q.l > r.left && q.r < r.right && q.t > r.top && q.b < r.bottom && !placed.some((p) => hit(q, { left: p.l, right: p.r, top: p.t, bottom: p.b })) && !labels.some((lb) => hit(q, lb)); }) || cands[0];
      placed.push(rect(pick));
      const bx = pick.dx, by = pick.dy;
      svgEl("rect", { x: bx, y: by, width: w, height: hgt, rx: 9, class: "bub" }, g);
      if (pick.tail) svgEl("path", { d: pick.tail === "down" ? "M-6," + (by + hgt) + " L0," + (by + hgt + 8) + " L6," + (by + hgt) + " Z" : "M-6," + by + " L0," + (by - 8) + " L6," + by + " Z", class: "bub-tail" }, g);
      const t0 = svgEl("text", { class: "bub-who", x: bx + 10, y: by + 17 }, g); t0.textContent = who;
      lines.forEach((l, i) => { const t = svgEl("text", { class: "bub-t", x: bx + 10, y: by + 17 + 15 * (i + 1) }, g); t.textContent = l; });
    }
    /* Everyone glides toward the latest reported position (state arrives every half second). */
    function glide(map, id, tx, ty) {
      const c = map.get(id);
      if (!c) { const n = { x: tx, y: ty }; map.set(id, n); return n; }
      const d = Math.hypot(tx - c.x, ty - c.y);
      if (d > 6) { c.x = tx; c.y = ty; } else { c.x += (tx - c.x) * 0.22; c.y += (ty - c.y) * 0.22; }
      return c;
    }
    function draw(now) {
      S.raf = requestAnimationFrame(draw);
      if (S.dead || now - S.lastDraw < 40) return;
      const s = WZ.state || {};
      if (!s.running || !S.live) return;
      S.lastDraw = now;
      const list = [], extras = [];
      for (const q of s.people || []) {
        if (!Number.isFinite(q.x)) continue;
        if (!P().characters.some((x) => x.id === q.id || x.name === q.id)) { extras.push({ id: q.id, x: q.x, y: q.y }); continue; }   // lying patients
        const c = glide(S.pos, q.id, q.x, q.y), ch = charOf(q.id);
        list.push({ id: q.id, name: ch.name, role: ch.role, x: c.x, y: c.y });
      }
      const ek = extras.map((e) => e.id).join(",");
      if (ek !== S.extraKey) { S.extraKey = ek; S.map.setStaticPatients(extras); }
      S.map.setCharacters(list, { names: true });
      let rp = null;
      if (s.robot && Number.isFinite(s.robot.x)) {
        const c = glide(S.pos, "__robot", s.robot.x, s.robot.y);
        rp = { x: c.x, y: c.y, yaw: s.robot.yaw || 0 };
        if (!S.map._robot) S.map.setRobot(rp, { label: "Robot" }); else S.map.moveRobot(rp.x, rp.y, rp.yaw);
      }
      S.map._clear(S.gBubbles);
      const seen = new Set(), placed = [];
      const labels = (s.says_tail || []).some((e) => e.text) ? S.map.roomLabels.filter((l) => !l.txt.hasAttribute("display")).map((l) => l.txt.getBoundingClientRect()) : [];
      for (const e of [...(s.says_tail || [])].reverse()) {
        if (!e.text || seen.has(e.person) || (s.t || 0) - (e.t || 0) > BUBBLE_S) continue;
        seen.add(e.person);
        const at = e.person === "robot" ? rp : list.find((p) => p.id === e.person || p.name === e.person);
        if (at) bubble(at.x, at.y, e.text, nameOf(e.person), placed, labels);
      }
      S.map._relayout();
    }
    function onState(s) {
      if (S.dead) return;
      if (s.running && !S.wasRunning) { S.framed = false; S.pos.clear(); S.extraKey = null; S.live = true; S.stopping = false; }
      if (s.running && !S.framed && s.people && s.people.length) { S.framed = true; frame(false); }
      if (!s.running && S.wasRunning) { S.live = false; S.stopping = false; S.map._clear(S.gBubbles); }
      S.wasRunning = !!s.running;
      refresh();
    }
    function showStart() {
      S.map._clear(S.gBubbles);
      S.map.setCharacters(P().characters, { names: true });
      S.map.setRobot(startRobot(), { label: "Robot" });
    }

    // ---- wiring ----
    WZ.on("project", S.listener = () => { if (S.dead || live()) return; drawSettings(); showStart(); drawStatic(); });
    WZ.on("state", S.stateListener = onState);
    S.wasRunning = live(); S.live = live();
    drawSettings(); drawStatic();
    if (live()) { S.map.setRobot(startRobot(), { label: "Robot" }); } else showStart();
    S.map.frame([], P().characters.concat([startRobot()]), 0.2, 16, false);
    S.raf = requestAnimationFrame(draw);
    root._runPage = S;
  }

  function unmount() {
    const S = document.getElementById("view")._runPage;
    if (S) { S.dead = true; cancelAnimationFrame(S.raf); if (S.map) S.map.destroy(); }
  }

  WZ.registerPage("run", { title: "Run", mount, unmount });
})();
