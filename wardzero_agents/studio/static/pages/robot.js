/* Page 2, Robot: pick the built-in G1 or connect your own robot, and place it on the map.
   Left: robot, URDF, sensors, USD conversion, connection. Right: the hospital with a draggable robot marker. */
(function () {
  const { h } = window;
  const G1 = "g1_claude";
  const KIND_LABEL = { rgbd: "RGB-D camera", rgb: "Camera", lidar: "Lidar", imu: "IMU", mic: "Microphone", speaker: "Speaker" };
  const FIXED = new Set(["mic", "speaker"]);
  const SVGNS = "http://www.w3.org/2000/svg";

  function svgIcon(path) {
    const s = document.createElementNS(SVGNS, "svg");
    for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", width: 16, height: 16, fill: "none", stroke: "currentColor", "stroke-width": 1.9, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) s.setAttribute(k, v);
    const p = document.createElementNS(SVGNS, "path"); p.setAttribute("d", path); s.appendChild(p);
    return s;
  }
  const COPY = "M9 9h10v11H9zM5 15V4h10";
  const CROSS = "M6 6l12 12M18 6L6 18";

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* not a secure context */ }
    const t = h("textarea", { style: "position:fixed;opacity:0" }); t.value = text;
    document.body.appendChild(t); t.select();
    let ok = false; try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    t.remove(); return ok;
  }
  const copyBtn = (text, label) => h("button.icon-btn.plain", { type: "button", title: "Copy " + label, "aria-label": "Copy " + label,
    onclick: async () => { const ok = await copyText(text); WZ.toast(ok ? "Copied " + label : "Copy failed, select the text instead", ok ? undefined : "bad"); } }, svgIcon(COPY));
  const NOISE = /absl|protobuf|\bW\d{4}\b|\bE\d{4}\b|Warning:|WARNING:|descriptor_database|message\.cc/i;
  /* The last meaningful lines of a conversion log, then its result when it has one. */
  function cleanLog(job) {
    const lines = String(job.log || "").split("\n").filter((l) => l.trim() && !NOISE.test(l)).slice(-14);
    if (job.result != null && job.result !== "") lines.push("Result: " + (typeof job.result === "string" ? job.result : JSON.stringify(job.result)));
    return lines.join("\n");
  }
  const num = (v, d) => { const n = parseFloat(v); return Number.isFinite(n) ? n : d; };
  const fmt = (n, d) => (Number.isFinite(n) ? n.toFixed(d == null ? 2 : d).replace(/\.?0+$/, "") : "0");

  function mount(root) {
    const S = { robots: [], mode: null, name: null, urdf: null, sensors: [], dirty: false, catalogue: [], conn: null, job: null, dead: false, timers: [], creating: false };
    const P = () => WZ.project;
    const side = h("aside.side");
    const body = h("div.side-body");
    side.append(h("div.seg", { role: "radiogroup", "aria-label": "Robot" },
      S.segBuilt = h("button.seg-btn", { type: "button", role: "radio", onclick: () => setMode("builtin") }, "Built-in G1 (Claude)"),
      S.segMine = h("button.seg-btn", { type: "button", role: "radio", onclick: () => setMode("mine") }, "My robot")), body);

    const mapBox = h("section.mapbox", { "aria-label": "Robot start position" });
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Top-down map of the hospital with the robot start marked");
    const chip = h("div.map-chip", "Drag the robot to place it, drag the round handle to turn it");
    const coord = h("div.coord-chip");
    const tools = h("div.map-tools",
      h("button.icon-btn", { type: "button", title: "Show the whole hospital", "aria-label": "Show the whole hospital", onclick: () => S.map.fit() }, svgIcon("M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5")),
      h("button.icon-btn", { type: "button", title: "Zoom in", "aria-label": "Zoom in", onclick: () => zoomBy(0.7) }, svgIcon("M12 5v14M5 12h14")),
      h("button.icon-btn", { type: "button", title: "Zoom out", "aria-label": "Zoom out", onclick: () => zoomBy(1.4) }, svgIcon("M5 12h14")));
    mapBox.append(svg, chip, tools, coord);
    root.append(h("div.robot-page", side, mapBox));

    function zoomBy(f) {
      const v = S.map.vb, cx = v.x + v.w / 2, cy = v.y + v.h / 2, w = Math.min(Math.max(v.w * f, 7), 150), r = w / v.w;
      S.map.setView({ x: cx - w / 2, y: cy - (v.h * r) / 2, w, h: v.h * r }, true);
    }
    S.map = new HospitalMap(svg, WZ.site, {});

    // ---- start marker ----
    const startOf = () => P().robot_start || { x: 0, y: 0, yaw: 0 };
    function drawRobot() {
      const rs = startOf();
      const R = S.map.setRobot(rs, { handle: true, label: "Start" });
      coordText(rs);
      if (!R) return;
      let drag = null, pose = null;
      const grab = (mode) => (ev) => {
        if (ev.button !== 0) return;
        ev.stopPropagation(); ev.preventDefault();
        drag = mode; pose = { x: R.m.x, y: R.m.y, yaw: R.yaw };
        svg.setPointerCapture(ev.pointerId);
      };
      R.g.addEventListener("pointerdown", grab("move"));
      R.handle.addEventListener("pointerdown", grab("turn"));
      if (S.move) { svg.removeEventListener("pointermove", S.move); svg.removeEventListener("pointerup", S.up); svg.removeEventListener("pointercancel", S.up); }
      svg.addEventListener("pointermove", S.move = (ev) => {
        if (!drag) return;
        const w = S.map._toWorld(ev);
        if (drag === "move") pose = { x: +w.x.toFixed(2), y: +w.y.toFixed(2), yaw: pose.yaw };
        else pose = { x: pose.x, y: pose.y, yaw: Math.atan2(w.y - pose.y, w.x - pose.x) };
        S.map.moveRobot(pose.x, pose.y, pose.yaw);
        coordText(pose);
      });
      const end = async (ev) => {
        if (!drag) return;
        drag = null;
        try { svg.releasePointerCapture(ev.pointerId); } catch (e) { /* already released */ }
        const out = { x: pose.x, y: pose.y, yaw: +pose.yaw.toFixed(3) };
        try { await WZ.saveProject({ robot_start: out }); }
        catch (e) { WZ.toast(e.message, "bad"); if (!S.dead) drawRobot(); }
      };
      svg.addEventListener("pointerup", S.up = end);
      svg.addEventListener("pointercancel", S.up);
    }
    function coordText(p) {
      const room = S.map.roomAt(p.x, p.y);
      const deg = Math.round((((p.yaw * 180) / Math.PI) % 360 + 360) % 360);
      coord.textContent = (room ? room.label + " · " : "") + "x " + p.x.toFixed(1) + " m, y " + p.y.toFixed(1) + " m, facing " + deg + "°";
    }
    function drawAll() {
      const p = P();
      S.map.setCharacters(p.characters, { names: false });
      S.map.setStaticPatients(p.static_patients ? WZ.site.beds.filter((b) => /^B[1-9][0-9]*$/.test(b.id) && !p.characters.some((c) => c.bed === b.id)) : []);
      drawRobot();
    }
    function frameOnce() { S.map.frame([], P().characters.concat([startOf()]), 0.2, 16, false); }

    // ---- data ----
    async function loadRobots() {
      S.robots = await API.get("/api/robots");
      S.mine = S.robots.filter((r) => !r.builtin);
    }
    const current = () => S.robots.find((r) => r.name === S.name) || null;

    async function adopt(name) {   // select a saved robot and read its URDF (the link list) when it has one
      S.name = name; S.urdf = null; S.job = null;
      const r = current();
      S.sensors = r ? r.sensors.map((x) => JSON.parse(JSON.stringify(x))) : [];
      S.dirty = false;
      if (r && r.urdf) {
        const folder = r.urdf.replace(/[\\/][^\\/]*$/, "");
        try { S.urdf = await API.post("/api/robots/" + encodeURIComponent(name) + "/urdf", { folder }); } catch (e) { /* shown by the check button */ }
      }
    }
    async function setMode(m) {
      try {
        if (m === "builtin") {
          S.mode = "builtin";
          if (P().robot_profile !== G1) await WZ.saveProject({ robot_profile: G1 });
        } else {
          S.mode = "mine"; S.creating = !S.mine.length;
          if (S.mine.length) {
            const keep = S.mine.find((r) => r.name === P().robot_profile) || S.mine[0];
            await adopt(keep.name);
            if (P().robot_profile !== keep.name) await WZ.saveProject({ robot_profile: keep.name });
          }
        }
      } catch (e) { WZ.toast(e.message, "bad"); }
      render();
    }

    // ---- left panel ----
    function render() {
      S.segBuilt.classList.toggle("on", S.mode === "builtin"); S.segMine.classList.toggle("on", S.mode === "mine");
      S.segBuilt.setAttribute("aria-checked", String(S.mode === "builtin")); S.segMine.setAttribute("aria-checked", String(S.mode === "mine"));
      const keep = body.scrollTop;
      body.replaceChildren(...(S.mode === "builtin" ? builtinPanel() : minePanel()), notePanel());
      body.scrollTop = keep;
    }

    function builtinPanel() {
      const g = S.robots.find((r) => r.builtin) || { sensors: [] };
      return [h("section.rsec",
        h("div.sec-head", h("h2", "Built-in G1 (Claude)")),
        h("p.muted.lead", "A Unitree G1 humanoid driven by Claude. It sees through a head camera, hears through a microphone and speaks through a speaker. Nothing to connect."),
        h("div.kv", kv("Body", "Humanoid, fixed post in this build"), kv("Reach", g.reach ? fmt(g.reach[0]) + " to " + fmt(g.reach[1]) + " m above the floor" : ""), kv("Controller", "Claude, inside WardZero")),
        h("h3.sub", "Sensors"),
        sensorList(g.sensors, false)),
        startSection()];
    }
    const kv = (k, v) => h("div.kvrow", h("span.k", k), h("span.v", v));

    function startSection() {
      const rs = startOf();
      return h("section.rsec", h("h3.sub", "Start position"),
        h("p.muted", "Drag the robot on the map, or turn it with the round handle. It starts here in every run."),
        h("div.kv", kv("Position", "x " + rs.x.toFixed(2) + " m, y " + rs.y.toFixed(2) + " m"),
          kv("Facing", Math.round((((rs.yaw || 0) * 180) / Math.PI % 360 + 360) % 360) + "°")));
    }

    function sensorList(list, editable) {
      if (!list.length) return h("p.muted", "No sensors yet. Add one from the catalogue below.");
      return h("ul.sensors", list.map((s, i) => {
        const approx = s.params && s.params.source === "generic" && s.model;
        const title = s.params && s.params.label ? s.params.label : (KIND_LABEL[s.kind] || s.kind) + (s.model ? " " + s.model : "");
        const fixed = FIXED.has(s.kind) && !s.model;
        if (!editable) {
          return h("li.sensor", h("div.s-top", h("span.s-name", title), h("span.s-kind", KIND_LABEL[s.kind] || s.kind)),
            h("div.s-meta", "on " + (s.link || "robot"), approx ? " · approx." : ""));
        }
        const linkField = S.urdf && S.urdf.links && S.urdf.links.length && !fixed
          ? h("select", { "aria-label": "Link for " + s.id, onchange: (e) => { s.link = e.target.value; dirty(); } },
            (S.urdf.links.includes(s.link) ? [] : [h("option", { value: s.link, selected: true }, s.link || "choose a link")]),
            S.urdf.links.map((l) => h("option", { value: l, selected: l === s.link }, l)))
          : h("input.text", { type: "text", value: s.link || "", "aria-label": "Link for " + s.id, placeholder: "link name", disabled: fixed && false, oninput: (e) => { s.link = e.target.value; dirty(); } });
        const triple = (key, names) => h("div.triple", names.map((nm, k) => h("label", h("span", nm),
          h("input.text", { type: "number", step: "0.01", value: s[key][k], "aria-label": s.id + " " + key + " " + nm, oninput: (e) => { s[key][k] = num(e.target.value, 0); dirty(); } }))));
        return h("li.sensor", h("div.s-top", h("span.s-name", title, approx ? h("span.tag.warn", "approx.") : null), h("span.s-kind", KIND_LABEL[s.kind] || s.kind),
          fixed ? h("span.s-fixed", "fixed") : h("button.icon-btn.danger", { type: "button", title: "Remove " + s.id, "aria-label": "Remove " + s.id, onclick: () => { S.sensors.splice(i, 1); dirty(); render(); } }, svgIcon(CROSS))),
          h("div.s-row", h("span.k", "Link"), linkField),
          h("div.s-row", h("span.k", "xyz m"), triple("xyz", ["x", "y", "z"])),
          h("div.s-row", h("span.k", "rpy rad"), triple("rpy", ["r", "p", "y"])));
      }));
    }

    function dirty() { S.dirty = true; if (S.saveSensors) S.saveSensors.disabled = false; }

    function minePanel() {
      const out = [];
      const r = current();
      const picker = h("div.field", h("div.flabel", h("label", { for: "robot-pick" }, "Your robot")),
        h("div.name-row",
          h("select#robot-pick", { onchange: async (e) => {
            if (e.target.value === "__new") { S.creating = true; render(); return; }
            S.creating = false;
            try { await adopt(e.target.value); await WZ.saveProject({ robot_profile: e.target.value }); } catch (err) { WZ.toast(err.message, "bad"); }
            render();
          } }, S.mine.map((x) => h("option", { value: x.name, selected: !S.creating && x.name === S.name }, x.name)),
            h("option", { value: "__new", selected: S.creating }, "New robot…")),
          r && !S.creating ? h("button.icon-btn.danger", { type: "button", title: "Delete this robot", "aria-label": "Delete this robot", onclick: deleteRobot }, svgIcon("M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12")) : null));
      out.push(h("section.rsec", h("div.sec-head", h("h2", "My robot")), S.mine.length ? picker : h("p.muted.lead", "Connect any robot: describe its body, point to its URDF and say where its sensors are."),
        S.creating || !r ? createForm() : bodyForm(r)));
      if (r && !S.creating) {
        out.push(urdfSection(r), sensorsSection(r), convertSection(r), connectSection(r), startSection());
      }
      return out;
    }

    function createForm() {
      const name = h("input.text", { type: "text", id: "new-name", maxlength: 40, placeholder: "my_robot", autocomplete: "off" });
      const base = h("select", { id: "new-base", "aria-label": "Base" }, ["legged", "wheeled", "mobile", "static"].map((b) => h("option", { value: b }, baseLabel(b))));
      const err = h("p.error");
      return h("form.rform", { onsubmit: async (e) => {
        e.preventDefault(); err.textContent = "";
        try {
          const saved = await API.post("/api/robots", { name: name.value.trim(), base: base.value, controller: "external" });
          await loadRobots(); await adopt(saved.name); S.creating = false;
          await WZ.saveProject({ robot_profile: saved.name });
          WZ.toast("Saved " + saved.name); render();
        } catch (ex) { err.textContent = ex.message; }
      } },
        h("div.field", h("div.flabel", h("label", { for: "new-name" }, "Name")), name, h("small.muted", "Letters, digits, - and _. This is also the ROS namespace unless you change it.")),
        h("div.field", h("div.flabel", h("label", { for: "new-base" }, "Base")), base),
        err, h("button.btn.primary", { type: "submit" }, "Create robot"));
    }
    const baseLabel = (b) => ({ legged: "Legged (walks)", wheeled: "Wheeled", mobile: "Mobile base", static: "Fixed post (cannot move)" }[b] || b);

    function bodyForm(r) {
      const f = {};
      const field = (key, label, val, step, unit) => h("label.mini", h("span", label),
        f[key] = h("input.text", { type: "number", step, min: 0, value: val, "aria-label": label + (unit ? " (" + unit + ")" : "") }), unit ? h("em", unit) : null);
      const err = h("p.error");
      const base = h("select", { id: "r-base", "aria-label": "Base" }, ["legged", "wheeled", "mobile", "static"].map((b) => h("option", { value: b, selected: b === r.base }, baseLabel(b))));
      return h("form.rform", { onsubmit: async (e) => {
        e.preventDefault(); err.textContent = "";
        try {
          await API.post("/api/robots", { name: r.name, base: base.value, radius: num(f.radius.value, r.radius), max_speed: num(f.max_speed.value, r.max_speed), controller: r.controller, ns: r.ns, reach: r.reach });
          await loadRobots(); WZ.toast("Saved " + r.name); render();
        } catch (ex) { err.textContent = ex.message; }
      } },
        h("div.rrow", h("div.field", h("div.flabel", h("label", { for: "r-base" }, "Base")), base)),
        h("div.rrow3", field("radius", "Radius", r.radius, "0.01", "m"), field("max_speed", "Top speed", r.max_speed, "0.05", "m/s"), h("label.mini", h("span", "ROS namespace"), h("code.ns", "/" + r.ns))),
        err, h("button.btn", { type: "submit" }, "Save body"));
    }

    async function deleteRobot() {
      const r = current(); if (!r) return;
      if (!(await WZ.ask({ title: "Delete " + r.name + "?", text: "Its URDF link, sensors and USD are removed from Studio. The files in your own folder are not touched.", ok: "Delete", danger: true }))) return;
      try {
        await API.del("/api/robots/" + encodeURIComponent(r.name));
        await WZ.saveProject({ robot_profile: G1 });
        await loadRobots(); S.name = null; S.urdf = null;
        await setMode(S.mine.length ? "mine" : "builtin");
      } catch (e) { WZ.toast(e.message, "bad"); }
    }

    function urdfSection(r) {
      const input = h("input.text", { type: "text", id: "urdf-folder", placeholder: "/home/you/robots/my_robot", "aria-label": "URDF folder",
        value: r.urdf ? r.urdf.replace(/[\\/][^\\/]*$/, "") : "", autocomplete: "off" });
      const out = h("div.urdf-out");
      const showInfo = () => {
        const u = S.urdf; out.replaceChildren();
        if (!u) return;
        const hints = (u.issues || []).filter((i) => /sensors/i.test(i)), problems = (u.issues || []).filter((i) => !/sensors/i.test(i));
        const movable = u.joints.filter((j) => ["revolute", "continuous", "prismatic"].includes(j.type)).length;
        out.append(...[h("div.stats",
          stat(String(u.dof), "DoF"), stat(fmt(u.height_m_est), "m tall"), stat(fmt(u.mass_kg, 1), "kg"), stat(String(u.links.length), "links"), stat(String(u.joints.length), "joints")),
          h("p.muted.small", (u.robot_name ? u.robot_name + ": " : "") + movable + " movable joints."),
          problems.length ? h("ul.issues", problems.map((i) => h("li", i))) : hints.length ? null : h("p.okline", svgIcon("M5 12.5l4.5 4.5L19 7.5"), "No problems found"),
          hints.length ? h("ul.hints", hints.map((i) => h("li", i))) : null].filter(Boolean));
      };
      showInfo();
      const btn = h("button.btn", { type: "button", onclick: async () => {
        btn.disabled = true; btn.textContent = "Checking…";
        try { S.urdf = await API.post("/api/robots/" + encodeURIComponent(r.name) + "/urdf", { folder: input.value.trim() }); await loadRobots(); }
        catch (e) { S.urdf = null; WZ.toast(e.message, "bad"); }
        render();
      } }, "Check URDF");
      return h("section.rsec", h("h3.sub", "URDF"),
        h("div.name-row", input, btn),
        h("small.muted", "A folder that holds your .urdf file and its meshes."), out);
    }
    const stat = (v, l) => h("div.stat", h("b", v), h("span", l));

    function sensorsSection(r) {
      const cat = h("select", { id: "cat-pick", "aria-label": "Sensor model" }, h("option", { value: "" }, "Add from catalogue…"),
        S.catalogue.map((c) => h("option", { value: c.id }, c.id.replace(/_/g, " ") + " · " + (KIND_LABEL[c.kind] || c.kind) + (c.source === "generic" ? " (approx.)" : ""))));
      const gaz = (S.urdf && S.urdf.gazebo_sensors) || [];
      const bits = [];
      cat.addEventListener("change", () => {
        const c = S.catalogue.find((x) => x.id === cat.value); if (!c) return;
        const links = (S.urdf && S.urdf.links) || [];
        let id = c.kind + "_1", n = 1; while (S.sensors.some((s) => s.id === id)) id = c.kind + "_" + (++n);
        S.sensors.push({ id, kind: c.kind, model: c.id, link: links[0] || "", xyz: [0, 0, 0], rpy: [0, 0, 0], params: { source: c.source } });
        dirty(); render();
      });
      const addFixed = (kind) => !S.sensors.some((s) => s.kind === kind) && h("button.btn.small", { type: "button", onclick: () => {
        S.sensors.push({ id: kind, kind, model: null, link: "base_link", xyz: [0, 0, 0], rpy: [0, 0, 0], params: { label: kind === "mic" ? "microphone" : "speaker" } }); dirty(); render();
      } }, "Add " + (kind === "mic" ? "microphone" : "speaker"));
      if (gaz.length) {
        bits.push(h("button.btn.small", { type: "button", onclick: () => {
          for (const g of gaz) {
            if (S.sensors.some((s) => s.id === g.name)) continue;
            S.sensors.push({ id: g.name || g.kind, kind: g.kind, model: null, link: g.link || "", xyz: (g.xyz || [0, 0, 0]).slice(), rpy: (g.rpy || [0, 0, 0]).slice(), params: { label: (KIND_LABEL[g.kind] || g.kind) + " from URDF" } });
          }
          dirty(); render();
        } }, "Use the " + gaz.length + " sensor" + (gaz.length > 1 ? "s" : "") + " in the URDF"));
      }
      S.saveSensors = h("button.btn.primary", { type: "button", disabled: !S.dirty, onclick: async () => {
        S.saveSensors.disabled = true;
        try { const saved = await API.put("/api/robots/" + encodeURIComponent(r.name) + "/sensors", { sensors: S.sensors }); S.dirty = false; await loadRobots(); S.sensors = saved.sensors; WZ.toast("Sensors saved"); render(); }
        catch (e) { WZ.toast(e.message, "bad"); S.saveSensors.disabled = false; }
      } }, "Save sensors");
      return h("section.rsec", h("h3.sub", "Sensors"),
        h("p.muted.small", "Take them from the URDF, add a model from the catalogue, and say which link each one sits on. Microphone and speaker are how people and the robot talk."),
        sensorList(S.sensors, true),
        h("div.addrow", cat, ...bits, addFixed("mic"), addFixed("speaker")),
        h("div.addrow", S.saveSensors));
    }

    function convertSection(r) {
      const log = h("pre.joblog", { "aria-label": "Conversion log", hidden: !S.job }, S.job ? cleanLog(S.job) : "");
      const status = h("span.tag" + jobTag(S.job), S.job ? S.job.status : "");
      status.hidden = !S.job;
      const btn = h("button.btn", { type: "button", disabled: !r.urdf || (S.job && S.job.status === "running"), onclick: async () => {
        try {
          if (S.dirty) { WZ.toast("Save the sensors first", "bad"); return; }
          const { job } = await API.post("/api/robots/" + encodeURIComponent(r.name) + "/convert");
          S.job = { id: job, status: "running", log: "Starting…\n" }; render(); pollJob();
        } catch (e) { WZ.toast(e.message, "bad"); }
      } }, "Convert to USD");
      S.logEl = log; S.statusEl = status; S.convBtn = btn;
      return h("section.rsec", h("div.sec-head.tight", h("h3.sub", "USD for Isaac Sim"), status),
        h("p.muted.small", r.urdf ? "Isaac Sim loads a USD. This converts your URDF and adds the sensors above." : "Check the URDF first."), btn, log);
    }
    const jobTag = (j) => (!j ? "" : j.status === "done" || j.status === "ok" ? ".pass" : j.status === "running" ? ".none" : ".fail");
    function pollJob() {
      clearTimeout(S.jobTimer);
      if (S.dead || !S.job || S.job.status !== "running") return;
      S.jobTimer = setTimeout(async () => {
        try {
          const j = await API.get("/api/jobs/" + S.job.id);
          S.job = { id: S.job.id, status: j.status, log: j.log || "", result: j.result };
          if (S.logEl && S.logEl.isConnected) {
            S.logEl.textContent = cleanLog(S.job); S.logEl.scrollTop = S.logEl.scrollHeight;
            S.statusEl.textContent = S.job.status; S.statusEl.className = "tag " + jobTag(S.job).slice(1);
            S.convBtn.disabled = S.job.status === "running";
          }
        } catch (e) { S.job = Object.assign({}, S.job, { status: "failed", log: S.job.log + "\n" + e.message }); }
        pollJob();
      }, 1000);
    }

    // ---- connection ----
    function connectSection(r) {
      const c = S.conn;
      const dom = h("input.text", { type: "number", min: 0, max: 232, id: "conn-domain", value: c ? c.robot_domain : 0, disabled: !!(c && !c.explicit && !S.editDomain), "aria-label": "ROS domain id", onchange: async (e) => {
        try { S.conn = Object.assign({}, S.conn, await API.post("/api/connection", Object.assign({ robot_domain: parseInt(e.target.value, 10) }, S.conn && S.conn.rmw && S.conn.rmw !== "default" ? { rmw: S.conn.rmw } : {}))); WZ.toast("Domain saved"); render(); }
        catch (err) { WZ.toast(err.message, "bad"); }
      } });
      const ip = c ? c.ip : "…";
      const env = c ? "export ROS_DOMAIN_ID=" + c.robot_domain + (c.rmw && c.rmw !== "default" ? "\nexport RMW_IMPLEMENTATION=" + c.rmw : "") : "";
      const lights = c ? c.lights || {} : {};
      const light = (key, label, hint) => h("li.light" + (lights[key] ? ".on" : ""), { title: hint }, h("i.lamp"), h("span", label), h("small", lights[key] ? "yes" : "waiting"));
      const topics = c ? c.topics : [];
      return h("section.rsec", h("h3.sub", "Connect your robot"),
        h("p.muted.small", "Your robot and this computer talk over ROS 2. Use the same domain on both, then publish and subscribe to the topics below."),
        h("div.kv", h("div.kvrow", h("span.k", "This computer"), h("span.v.mono", ip), copyBtn(ip, "the address")),
          h("div.kvrow", h("span.k", "Robot ROS domain"), dom,
            c && !c.explicit && !S.editDomain ? h("small.muted", "chosen for you (the people use " + c.private_domain + ")") : null,
            c && !c.explicit && !S.editDomain ? h("button.btn.small", { type: "button", onclick: () => { S.editDomain = true; render(); } }, "Change") : null),
          h("div.kvrow", h("span.k", "RMW"), h("span.v.mono", c ? c.rmw : "…"), env ? copyBtn(env, "the environment lines") : null),
          h("div.kvrow", h("span.k", "ROS 2"), h("span.v", c ? (c.ros ? "available on this computer" : "not installed on this computer; the topics below are for the robot side") : "…"))),
        h("ul.lights", { "aria-label": "Connection status" },
          light("controller_seen", "Robot controller seen", "The robot's controller is on the ROS graph"),
          light("sensors_subscribed", "Sensors subscribed", "Your robot is subscribed to the sensor topics"),
          light("commands_seen", "Commands seen", "The robot has sent commands")),
        h("h4.tlabel", "Topics"),
        topics.length ? h("ul.topics", topics.map((t) => h("li.topic", h("span.dir" + (t.dir === "robot_to_sim" ? ".out" : ""), { title: t.dir === "robot_to_sim" ? "Your robot publishes" : "Your robot subscribes" }, t.dir === "robot_to_sim" ? "robot → sim" : "sim → robot"),
          h("div.t-main", h("code", t.name), h("small", t.type)), copyBtn(t.name, "the topic name")))) : h("p.muted", "Loading…"));
    }
    async function pollConn() {
      if (S.dead) return;
      if (S.mode === "mine" && current() && !S.creating) {
        try {
          const prev = S.conn ? JSON.stringify(S.conn) : "";
          S.conn = await API.get("/api/connection?robot=" + encodeURIComponent(current().name));
          if (JSON.stringify(S.conn) !== prev && !(document.activeElement && body.contains(document.activeElement) && /input|select|textarea/i.test(document.activeElement.tagName))) render();
        } catch (e) { /* offline */ }
      }
      S.connTimer = setTimeout(pollConn, 3000);
    }

    function notePanel() {
      return h("p.rnote", "Joint-level control of your robot runs in Isaac Sim (coming in this build's Isaac host); the stand-in simulator drives speech, movement and task actions.");
    }

    // ---- start ----
    const onProject = () => { if (S.dead) return; drawAll(); frameOnce(); render(); };
    WZ.on("project", S.listener = onProject);
    (async () => {
      try {
        const [, cat] = await Promise.all([loadRobots(), API.get("/api/sensors/catalogue").catch(() => [])]);
        S.catalogue = cat;
        const inUse = S.mine.find((r) => r.name === P().robot_profile);
        S.mode = inUse ? "mine" : "builtin";
        if (inUse) await adopt(inUse.name);
      } catch (e) { WZ.toast(e.message, "bad"); S.mode = "builtin"; }
      if (S.dead) return;
      drawAll(); frameOnce(); render(); pollConn();
    })();
    root._robotPage = S;
  }

  function unmount() {
    const S = document.getElementById("view")._robotPage;
    if (S) { S.dead = true; clearTimeout(S.connTimer); clearTimeout(S.jobTimer); if (S.map) S.map.destroy(); }
  }

  WZ.registerPage("robot", { title: "Robot", mount, unmount });
})();
