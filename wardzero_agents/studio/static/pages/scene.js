/* Page 1, Scene: hospital map on the right; characters and scenarios on the left.
   Add or edit a character in the dialog (details and prompts on the left, room preview on the right). */
(function () {
  const { h } = window;
  const FEMALE = ["Amelia", "Olivia", "Isla", "Freya", "Priya", "Hannah", "Grace", "Aisha", "Megan", "Charlotte",
    "Sophie", "Eleanor", "Ruth", "Mei", "Fatima", "Rosie", "Joanna", "Maya", "Beth", "Nadia"];
  const MALE = ["Oliver", "Arthur", "George", "Harry", "Daniel", "Samuel", "Rohan", "Callum", "Ibrahim", "Thomas",
    "Mohammed", "Edward", "Jamal", "Peter", "Wei", "Owen", "Rhys", "Anil", "Marcus", "Henry"];
  const SURNAMES = ["Patel", "Evans", "Wood", "Khan", "Taylor", "Brown", "Wilson", "Davies", "Singh", "Clarke",
    "Hughes", "Walker", "Ahmed", "Murphy", "Lee", "Roberts", "Hall", "Okafor", "Green", "Baker", "Turner", "Cooper",
    "Reid", "Campbell", "Morgan", "Bennett", "Chowdhury", "Fraser", "Ward", "Lewis", "Nguyen", "Parker", "Shah",
    "Gray", "Mitchell", "Hussain", "Owen", "Stewart", "Kelly", "Adams"];
  const AGES = ["18-30", "31-50", "51-70", "71+"];
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const randomName = (sex) => pick(sex === "male" ? MALE : FEMALE) + " " + pick(SURNAMES);

  const ICONS = {
    edit: '<path d="M4 20h4L19 9l-4-4L4 16v4z" /><path d="M13.5 6.5l4 4" />',
    reset: '<path d="M4 12a8 8 0 1 0 2.6-5.9" /><path d="M4 4v4.5h4.5" />',
    trash: '<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" />',
    plus: '<path d="M12 5v14M5 12h14" />',
    shuffle: '<path d="M4 7h3c4 0 5 10 9 10h4M4 17h3c1.6 0 2.7-1.4 3.7-3M16 7h4m0 0l-2.5-2.5M20 7l-2.5 2.5M20 17l-2.5-2.5M20 17l-2.5 2.5" />',
    close: '<path d="M6 6l12 12M18 6L6 18" />',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />',
    minus: '<path d="M5 12h14" />',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5" />',
    area: '<rect x="4" y="5" width="16" height="14" rx="1.5" stroke-dasharray="3.2 2.4" /><path d="M9 9.5l6 5M15 9.5l-6 5" />',
  };
  function icon(name, size) {
    const s = h("span.ico");
    s.innerHTML = '<svg viewBox="0 0 24 24" width="' + (size || 18) + '" height="' + (size || 18) + '" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + ICONS[name] + "</svg>";
    return s;
  }
  const iconBtn = (name, label, onclick, cls) => h("button.icon-btn" + (cls ? "." + cls : ""), { type: "button", title: label, "aria-label": label, onclick }, icon(name));

  const roleLabel = (id) => { const r = WZ.presets.roles.find((x) => x.id === id); return r ? r.label : id; };
  const moodLabel = (id) => { const m = WZ.presets.moods.find((x) => x.id === id); return m ? m.label : id; };
  const roomLabel = (id) => { const r = WZ.site.rooms.find((x) => x.id === id); return r ? r.label : id; };
  const isPatientRole = (id) => id === "patient";
  const bedList = () => WZ.site.beds;
  const WARD_BED = /^B[1-9][0-9]*$/;

  /* Where a click lands: a free bed for patients, a free seat for others (the server snaps the same way). */
  function snapPoint(x, y, role, taken) {
    if (isPatientRole(role)) {
      let best = null, bd = 1.3;
      for (const b of bedList()) { if (taken.has(b.id)) continue; const d = Math.hypot(b.x - x, b.y - y); if (d < bd) { bd = d; best = b; } }
      if (best) return { x: best.x, y: best.y, bed: best.id, snap: "Bed " + best.id };
    } else {
      let best = null, bd = 0.3;
      for (const s of WZ.site.seats) { const d = Math.hypot(s.x - x, s.y - y); if (d <= bd) { bd = d; best = s; } }
      if (best) return { x: best.x, y: best.y, bed: null, snap: "Seat " + best.id };
    }
    return { x, y, bed: null, snap: "" };
  }
  /* A sensible first spot in a room: its centre if that is inside the room, else the nearest sample. */
  function roomPoint(roomId, role, taken) {
    const r = WZ.site.rooms.find((x) => x.id === roomId); if (!r) return null;
    if (isPatientRole(role)) {
      const bed = bedList().find((b) => b.room === roomId && !taken.has(b.id));
      if (bed) return { x: bed.x, y: bed.y, bed: bed.id, snap: "Bed " + bed.id };
    }
    const bb = WZMap.bbox(r.poly), cx = (bb.x0 + bb.x1) / 2, cy = (bb.y0 + bb.y1) / 2;
    let best = null, bd = Infinity;
    for (let i = 0; i <= 16; i++) for (let j = 0; j <= 16; j++) {
      const x = bb.x0 + ((bb.x1 - bb.x0) * i) / 16, y = bb.y0 + ((bb.y1 - bb.y0) * j) / 16;
      if (!WZMap.pointInPoly(x, y, r.poly)) continue;
      const d = Math.hypot(x - cx, y - cy); if (d < bd) { bd = d; best = { x, y }; }
    }
    best = best || { x: cx, y: cy };
    return { x: +best.x.toFixed(2), y: +best.y.toFixed(2), bed: null, snap: "" };
  }
  const takenBeds = (chars) => new Set(chars.map((c) => c.bed).filter(Boolean));
  function staticBeds(project) {
    if (!project.static_patients) return [];
    const taken = takenBeds(project.characters);
    return bedList().filter((b) => WARD_BED.test(b.id) && !taken.has(b.id));
  }

  // ---------------------------------------------------------------------------------------------
  function mount(root) {
    const S = { tab: "characters", sel: null, hover: null, scen: null, memory: {}, map: null, thumbs: [], dead: false };
    const P = () => WZ.project;

    const side = h("aside.side");
    const mapBox = h("section.mapbox", { "aria-label": "Hospital map" });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Top-down map of the hospital with the characters placed on it");
    const chip = h("div.map-chip");
    const legend = h("div.legend");
    const drawBtn = iconBtn("area", "Draw a forbidden area: drag a rectangle the robot must keep out of", () => setDrawing(!S.drawing));
    drawBtn.setAttribute("aria-pressed", "false");
    const tools = h("div.map-tools",
      iconBtn("fit", "Show the whole hospital", () => S.map.fit()),
      iconBtn("plus", "Zoom in", () => zoomBy(0.7)), iconBtn("minus", "Zoom out", () => zoomBy(1.4)), drawBtn);
    const hintEl = h("div.map-hint", "Scroll to zoom, drag to pan. Drag a person to move them.");
    mapBox.append(svg, chip, tools, legend, hintEl);
    root.append(h("div.scene", side, mapBox));

    function zoomBy(f) {
      const v = S.map.vb, cx = v.x + v.w / 2, cy = v.y + v.h / 2, w = Math.min(Math.max(v.w * f, 7), 150), r = w / v.w;
      S.map.setView({ x: cx - w / 2, y: cy - (v.h * r) / 2, w, h: v.h * r }, true);
    }

    S.map = new HospitalMap(svg, WZ.site, {});
    const busy = () => !!(WZ.state && WZ.state.running);

    // ---- forbidden areas: drag a rectangle on the map, then name it ----
    function setDrawing(on) {
      if (on && busy()) { WZ.toast("Not while a run is going", "bad"); return; }
      S.drawing = on; S.area = null; S.map.setDraft(null);
      mapBox.classList.toggle("drawing", on);
      drawBtn.classList.toggle("on", on); drawBtn.setAttribute("aria-pressed", String(on));
      hintEl.textContent = on ? "Drag a rectangle the robot must keep out of. Esc to cancel." : "Scroll to zoom, drag to pan. Drag a person to move them.";
    }
    mapBox.addEventListener("pointerdown", (ev) => {   // capture: before the map starts panning
      if (!S.drawing || ev.button !== 0 || !svg.contains(ev.target)) return;
      ev.stopPropagation(); ev.preventDefault();
      const w = S.map._toWorld(ev);
      S.area = { x0: w.x, y0: w.y, x1: w.x, y1: w.y };
      svg.setPointerCapture(ev.pointerId);
    }, true);
    document.addEventListener("keydown", S.onKey = (ev) => { if (ev.key === "Escape" && S.drawing && !document.querySelector("dialog[open]")) setDrawing(false); });
    async function finishArea() {
      const a = S.area; S.area = null;
      const r = { x0: +Math.min(a.x0, a.x1).toFixed(2), y0: +Math.min(a.y0, a.y1).toFixed(2), x1: +Math.max(a.x0, a.x1).toFixed(2), y1: +Math.max(a.y0, a.y1).toFixed(2) };
      if (r.x1 - r.x0 < 0.3 || r.y1 - r.y0 < 0.3) { S.map.setDraft(null); WZ.toast("Drag a bigger rectangle (at least 0.3 m on each side)."); return; }
      const room = S.map.roomAt((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2);
      const name = await WZ.ask({ title: "Name this forbidden area", text: "The robot must keep out of it. The report counts every time it goes in.", ok: "Add area", input: { value: room ? room.label.split(/\s*[,(]/)[0] : "Staff only", label: "Area name" } });
      S.map.setDraft(null);
      setDrawing(false);
      if (!name) return;
      try { await API.post(WZ.projectRoutes() + "/areas", Object.assign({ name }, r)); await reload(); WZ.toast("Added forbidden area " + name); }
      catch (e) { WZ.toast(e.message, "bad"); }
    }

    // ---- dragging characters (and turning the selected one), like the robot on the Robot page ----
    function grab(c, E, mode) {
      return (ev) => {
        if (ev.button !== 0 || S.drawing) return;
        ev.stopPropagation(); ev.preventDefault();
        S.drag = { c, E, mode, cx: ev.clientX, cy: ev.clientY, moved: false, pose: { x: c.x, y: c.y, yaw: c.yaw || 0 }, bed: c.bed || null };
        svg.setPointerCapture(ev.pointerId);
      };
    }
    svg.addEventListener("pointermove", (ev) => {
      if (S.area) { const w = S.map._toWorld(ev); S.area.x1 = w.x; S.area.y1 = w.y; S.map.setDraft(S.area); return; }
      const D = S.drag; if (!D) return;
      if (!D.moved && Math.hypot(ev.clientX - D.cx, ev.clientY - D.cy) < 4) return;
      if (!D.moved && busy()) { S.drag = null; WZ.toast("Not while a run is going", "bad"); return; }
      D.moved = true; D.E.g.classList.add("dragging");
      const w = S.map._toWorld(ev);
      if (D.mode === "turn") D.pose.yaw = Math.atan2(w.y - D.pose.y, w.x - D.pose.x);
      else {
        const others = P().characters.filter((o) => o.id !== D.c.id);
        const sp = snapPoint(w.x, w.y, D.c.role, takenBeds(others));
        D.pose.x = +sp.x.toFixed(2); D.pose.y = +sp.y.toFixed(2); D.bed = sp.bed; D.snap = sp.snap;
      }
      S.map.movePerson(D.E, D.pose.x, D.pose.y, D.pose.yaw);
      const room = S.map.roomAt(D.pose.x, D.pose.y);
      chip.textContent = (D.c.name || "").split(" ")[0] + ": " + (room ? room.label + " · " : "") + "x " + D.pose.x.toFixed(1) + ", y " + D.pose.y.toFixed(1) + (D.snap ? " · " + D.snap : "");
    });
    const endDrag = async (ev) => {
      if (S.area) { try { svg.releasePointerCapture(ev.pointerId); } catch (e) { /* released */ } if (ev.type === "pointerup") finishArea(); else { S.area = null; S.map.setDraft(null); } return; }
      const D = S.drag; if (!D) return;
      S.drag = null;
      try { svg.releasePointerCapture(ev.pointerId); } catch (e) { /* already released */ }
      if (!D.moved) { if (ev.type === "pointerup") pickChar(D.c); return; }
      const body = D.mode === "turn" ? { yaw: +D.pose.yaw.toFixed(3) } : { x: D.pose.x, y: D.pose.y, bed: D.bed };
      try {
        const saved = await API.put(WZ.projectRoutes() + "/characters/" + D.c.id, body);
        S.sel = saved.id;
        await reload();
        if (D.c.start_seated && !saved.start_seated) WZ.toast(saved.name + " is standing now (moved off the waiting seat).");
      } catch (e) { WZ.toast(e.message, "bad"); if (!S.dead) drawMap(); }
    };
    svg.addEventListener("pointerup", endDrag);
    svg.addEventListener("pointercancel", endDrag);

    function pickChar(c) {
      S.sel = c.id; setTab("characters"); drawMap();
      const r = body.querySelector('[data-cid="' + c.id + '"]'); if (r) r.scrollIntoView({ block: "nearest" });
    }

    async function setSeated(c, on, box) {
      try {
        const saved = await API.put(WZ.projectRoutes() + "/characters/" + c.id, { start_seated: on });
        S.sel = saved.id;
        await reload();
        const seat = WZ.site.seats.find((q) => Math.hypot(q.x - saved.x, q.y - saved.y) < 0.05);
        WZ.toast(on ? saved.name + " starts seated on seat " + (seat ? seat.id : "") : saved.name + " now starts standing");
      } catch (e) { WZ.toast(e.message, "bad"); if (box) box.checked = !on; }
    }

    // ---- left panel ----
    const segChars = h("button.seg-btn", { type: "button", role: "tab", onclick: () => setTab("characters") });
    const segScen = h("button.seg-btn", { type: "button", role: "tab", onclick: () => setTab("scenarios") }, "Scenarios");
    const body = h("div.side-body");
    const staticSwitch = h("input", { type: "checkbox", id: "static-switch", role: "switch", onchange: async (e) => {
      try { await WZ.saveProject({ static_patients: e.target.checked }); } catch (err) { WZ.toast(err.message, "bad"); e.target.checked = !e.target.checked; }
    } });
    side.append(
      h("div.seg", { role: "tablist", "aria-label": "Left panel" }, segChars, segScen),
      body,
      h("label.switch-row", { for: "static-switch" }, staticSwitch, h("span.track", h("i")),
        h("span.sw-text", h("b", "Static patients in ward beds"), h("small", "Lying patients fill the free ward beds. They do not speak."))));

    function setTab(t) {
      S.tab = t; renderSide();
      if (t === "characters") S.map.setFocus([]);
    }

    // ---- rendering ----
    function drawMap() {
      const p = P();
      if (S.drag) return;  // never rebuild a marker that is being dragged
      const marks = S.map.setCharacters(p.characters, { selected: S.sel, hover: S.hover, draggable: true, handle: S.sel });
      for (const c of p.characters) {
        const E = marks.get(c.id);
        E.g.addEventListener("pointerdown", grab(c, E, "move"));
        if (E.handle) E.handle.addEventListener("pointerdown", grab(c, E, "turn"));
      }
      S.map.setAreas(p.forbidden_areas || []);
      const st = staticBeds(p);
      S.map.setStaticPatients(st);
      S.map.setRobot(p.robot_start);
      S.map.setBedState(takenBeds(p.characters), null);
      const roles = [...new Set(p.characters.map((c) => c.role))];
      legend.replaceChildren(
        ...roles.map((r) => h("span.lg", h("i.dot", { style: "background:" + WZMap.roleColor(r) }), roleLabel(r))),
        st.length ? h("span.lg", h("i.dot.static")) : null, st.length ? h("span.lg-t", "In bed") : null,
        h("span.lg", h("i.tri")), h("span.lg-t", "Robot start"),
        ...((p.forbidden_areas || []).length ? [h("span.lg", h("i.farea-sw")), h("span.lg-t", "Forbidden area")] : []));
      legend.querySelectorAll(".lg-t").forEach((n) => n.previousSibling && n.previousSibling.classList.add("solo"));
      chip.textContent = p.characters.length + (p.characters.length === 1 ? " character" : " characters") + (st.length ? " and " + st.length + " in bed" : "");
      staticSwitch.checked = !!p.static_patients;
    }

    async function loadMemory() {
      try {
        const rows = await API.get(WZ.projectRoutes() + "/memory");
        S.memory = Object.fromEntries(rows.map((r) => [r.id, r]));
      } catch (e) { S.memory = {}; }
      if (!S.dead && S.tab === "characters") renderChars();
    }

    function renderSide() {
      const n = P().characters.length;
      segChars.replaceChildren("Characters", h("span.count", String(n)));
      segChars.setAttribute("aria-selected", S.tab === "characters"); segChars.classList.toggle("on", S.tab === "characters");
      segScen.setAttribute("aria-selected", S.tab === "scenarios"); segScen.classList.toggle("on", S.tab === "scenarios");
      S.thumbs.forEach((m) => m.destroy()); S.thumbs = [];
      if (S.tab === "characters") renderChars(); else renderScenarios();
    }

    function charRow(c) {
      const m = S.memory[c.id];
      const row = h("li.char" + (c.id === S.sel ? ".sel" : ""), { "data-cid": c.id,
        onmouseenter: () => { S.hover = c.id; drawMap(); }, onmouseleave: () => { S.hover = null; drawMap(); },
        onclick: () => { S.sel = c.id; S.map.zoomToPoint(c.x, c.y, 9); drawMap(); renderChars(); } },
        h("span.avatar", { style: "background:" + WZMap.roleColor(c.role) }, c.name.charAt(0).toUpperCase()),
        h("div.cbody",
          h("div.cname", c.name, c.auto ? h("span.chip.auto", { title: "Added automatically for this scenario. Remove it if you do not want it." }, "auto") : null),
          h("div.cmeta", roleLabel(c.role) + " · " + moodLabel(c.mood)),
          h("div.cmeta.dim", roomLabel(c.room) + (c.bed ? ", bed " + c.bed : "") + (seatAt(c) ? ", seat " + seatAt(c).id : "")),
          c.bed ? null : seatedBox(c),
          m ? h("span.trust", { title: "Trust " + m.trust.toFixed(2) + ", stage " + m.stage + ". " + m.runs + " runs, " + m.violations + " violations, " + m.repairs + " repairs." },
            h("i.tdot.s" + Math.min(4, m.stage)), "Stage " + m.stage + " · trust " + m.trust.toFixed(2)) : null),
        h("div.cacts",
          iconBtn("edit", "Edit " + c.name, (e) => { e.stopPropagation(); openDialog(c); }),
          iconBtn("reset", "Reset memory of " + c.name, async (e) => {
            e.stopPropagation();
            if (!(await WZ.ask({ title: "Reset memory?", text: c.name + " forgets earlier runs and goes back to the starting trust. Prompts stay.", ok: "Reset memory" }))) return;
            try { await API.post(WZ.projectRoutes() + "/memory/reset", { characters: [c.id] }); WZ.toast("Memory reset for " + c.name); loadMemory(); }
            catch (err) { WZ.toast(err.message, "bad"); }
          }),
          iconBtn("trash", "Remove " + c.name, async (e) => {
            e.stopPropagation();
            if (!(await WZ.ask({ title: "Remove " + c.name + "?", text: "They leave the scene.", ok: "Remove", danger: true }))) return;
            try { await API.del(WZ.projectRoutes() + "/characters/" + c.id); await reload(); } catch (err) { WZ.toast(err.message, "bad"); }
          }, "danger")));
      return row;
    }

    function seatAt(c) { return WZ.site.seats.find((q) => Math.hypot(q.x - c.x, q.y - c.y) < 0.05) || null; }
    function seatedBox(c) {
      const box = h("input", { type: "checkbox", checked: !!c.start_seated, "aria-label": "Start seated: " + c.name,
        onclick: (e) => e.stopPropagation(), onchange: (e) => setSeated(c, e.target.checked, e.target) });
      return h("label.seat-chk", { title: "Sit on a free A&E waiting seat at the start", onclick: (e) => e.stopPropagation() }, box, "Start seated");
    }

    function areaRow(a) {
      const room = S.map.roomAt((a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2);
      return h("li.area", { "data-aid": a.id, onclick: () => S.map.frame([], [{ x: a.x0, y: a.y0 }, { x: a.x1, y: a.y1 }], 0.6, 10) },
        h("span.sw", { "aria-hidden": "true" }),
        h("div.abody", h("div.aname", a.name),
          h("div.ameta", (a.x1 - a.x0).toFixed(1) + " × " + (a.y1 - a.y0).toFixed(1) + " m" + (room ? " · " + room.label : ""))),
        iconBtn("trash", "Remove forbidden area " + a.name, async (e) => {
          e.stopPropagation();
          try { await API.del(WZ.projectRoutes() + "/areas/" + encodeURIComponent(a.id)); await reload(); WZ.toast("Removed " + a.name); }
          catch (err) { WZ.toast(err.message, "bad"); }
        }, "danger"));
    }
    function areasSection() {
      const list = P().forbidden_areas || [];
      return h("section.areas-sec",
        h("div.sec-head", h("h2", "Forbidden areas"),
          h("button.btn", { type: "button", onclick: () => setDrawing(true) }, icon("area", 16), "Draw area")),
        list.length ? h("ul.areas", list.map(areaRow))
          : h("p.muted", "None yet. Draw a rectangle on the map that the robot must keep out of, such as Resus or a staff-only room."));
    }

    function renderChars() {
      const chars = P().characters;
      const head = h("div.sec-head", h("h2", "Characters"),
        h("button.btn.primary", { type: "button", onclick: () => openDialog(null) }, icon("plus", 16), "Add character"));
      if (!chars.length) {
        body.replaceChildren(head, h("div.empty",
          h("div.empty-art", { "aria-hidden": "true" }, h("i"), h("i"), h("i")),
          h("h3", "Nobody is on the map yet"),
          h("p.muted", "Add the people your robot will meet, or start from a ready-made scenario."),
          h("button.btn", { type: "button", onclick: () => setTab("scenarios") }, "Browse scenarios")), areasSection());
        return;
      }
      body.replaceChildren(head, h("ul.chars", chars.map(charRow)), areasSection());
    }

    // ---- scenarios ----
    function renderScenarios() {
      const list = WZ.presets.scenarios;
      const inUse = P().characters.length ? P().scenario_id : null;
      const cards = list.map((sc) => {
        const open = S.scen === sc.id;
        const tsvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        tsvg.setAttribute("aria-hidden", "true");
        const card = h("li.card" + (open ? ".open" : "") + (inUse === sc.id ? ".inuse" : ""),
          h("button.card-btn", { type: "button", "aria-expanded": open, onclick: () => selectScenario(open ? null : sc) },
            h("div.thumb", tsvg, inUse === sc.id ? h("span.badge", "In use") : null),
            h("div.card-title", sc.title),
            h("p.card-prompt", sc.prompt.replace(/\s+/g, " ").trim())),
          open ? scenarioDetails(sc) : null);
        const t = new HospitalMap(tsvg, WZ.site, { interactive: false, thumb: true, labels: false, zoomTo: sc.focus_rooms, pad: 3, focusRooms: sc.focus_rooms });
        t.setCharacters(sc.characters, { names: false }); t.setRobot(sc.robot_start);
        S.thumbs.push(t);
        return card;
      });
      body.replaceChildren(h("div.sec-head", h("h2", "Scenario generator")),
        h("p.muted.lead", "Pick a situation. It loads the characters, the robot's job and how it will be scored."),
        h("ul.cards", cards));
    }

    function scenarioDetails(sc) {
      return h("div.card-detail",
        h("button.btn.primary.wide", { type: "button", onclick: () => useScenario(sc) }, "Use this scenario"),
        h("h4", "Characters"),
        h("ul.chips", sc.characters.map((c) => h("li.chip", h("i.dot", { style: "background:" + WZMap.roleColor(c.role) }), c.name + " · " + roleLabel(c.role) + ", " + moodLabel(c.mood).toLowerCase()))),
        h("h4", "Robot job"), h("p", sc.robot_brief.job || sc.robot_brief.role),
        h("h4", "Success criteria"),
        h("ul.crit", sc.criteria.map((k) => h("li", icon("check", 14), k.label))),
        );
    }

    function selectScenario(sc) {
      S.scen = sc ? sc.id : null;
      renderScenarios();
      S.map.setFocus(sc ? sc.focus_rooms : []);
      if (sc) S.map.frame(sc.focus_rooms, sc.characters, 0.12, 16); else S.map.fit();
      if (sc) { const el = body.querySelector(".card.open"); if (el) el.scrollIntoView({ block: "start", behavior: "smooth" }); }
    }

    async function useScenario(sc) {
      if (P().characters.length && !(await WZ.ask({ title: "Replace the characters?", text: "This project has " + P().characters.length + " characters. Loading \"" + sc.title + "\" replaces them with the scenario's " + sc.characters.length + ".", ok: "Replace characters" }))) return;
      try {
        const patch = { scenario_id: sc.id, characters: sc.characters, robot_start: sc.robot_start, duration_min: sc.duration_min };
        if (/^Untitled/.test(P().name)) patch.name = sc.title;
        await WZ.saveProject(patch);
        S.sel = null; S.scen = null; S.tab = "characters";
        renderSide(); drawMap(); loadMemory();
        S.map.setFocus([]); S.map.frame(sc.focus_rooms, sc.characters.concat([sc.robot_start]), 0.12, 16);
        WZ.toast("Loaded " + sc.title + ": " + sc.characters.length + " characters");
      } catch (e) { WZ.toast(e.message, "bad"); }
    }

    async function reload() {
      WZ.setProject(await API.get(WZ.projectRoutes()));
    }

    // ---- character dialog ----
    function openDialog(existing) {
      const project = P(), presets = WZ.presets;
      const others = project.characters.filter((c) => !existing || c.id !== existing.id);
      const taken = takenBeds(others);
      const firstRole = presets.roles.find((r) => r.id === "ward_nurse") || presets.roles[0];
      const firstMood = presets.moods.find((m) => m.id === "anxious") || presets.moods[0];
      const D = existing ? { sex: existing.sex, name: existing.name, age: existing.age_band, role: existing.role, mood: existing.mood,
        rolePrompt: existing.role_prompt, moodPrompt: existing.mood_prompt, room: existing.room, x: existing.x, y: existing.y, bed: existing.bed || null, snap: existing.bed ? "Bed " + existing.bed : "", placed: true, seated: !!existing.start_seated }
        : { sex: "female", name: randomName("female"), age: "31-50", role: firstRole.id, mood: firstMood.id, rolePrompt: null, moodPrompt: null, room: "", x: 0, y: 0, bed: null, snap: "", placed: false, seated: false };
      const rolePre = () => presets.roles.find((r) => r.id === D.role), moodPre = () => presets.moods.find((m) => m.id === D.mood);
      const preText = (p) => (p && p.prompt ? p.prompt.trim() : "");

      // fields
      const sexBtns = ["female", "male"].map((s) => h("button.pill", { type: "button", role: "radio", "data-v": s, onclick: () => setSex(s) }, s === "female" ? "Female" : "Male"));
      const nameIn = h("input.text", { type: "text", id: "f-name", maxlength: 60, value: D.name, autocomplete: "off", oninput: () => { D.name = nameIn.value; drawPreview(); } });
      const ageSel = h("select", { id: "f-age", onchange: () => { D.age = ageSel.value; } }, AGES.map((a) => h("option", { value: a, selected: a === D.age }, a)));
      const roleSel = h("select", { id: "f-role", onchange: () => { D.role = roleSel.value; D.rolePrompt = null; syncPrompts(); fillRooms(); if (!D.placed) autoPlace(); drawPreview(); } },
        presets.roles.map((r) => h("option", { value: r.id, selected: r.id === D.role }, r.label)));
      const roleTa = h("textarea", { id: "f-roleprompt", rows: 5, spellcheck: "false", "aria-label": "Role prompt", oninput: () => { D.rolePrompt = roleTa.value; syncRestore(); autosize(roleTa, 10); } });
      const roleRestore = h("button.link", { type: "button", onclick: () => { D.rolePrompt = null; syncPrompts(); } }, "Restore preset");
      const moodSel = h("select", { id: "f-mood", onchange: () => { D.mood = moodSel.value; D.moodPrompt = null; syncPrompts(); } },
        presets.moods.map((m) => h("option", { value: m.id, selected: m.id === D.mood }, m.label)));
      const moodTa = h("textarea", { id: "f-moodprompt", rows: 3, spellcheck: "false", "aria-label": "Mood prompt", oninput: () => { D.moodPrompt = moodTa.value; syncRestore(); autosize(moodTa, 6); } });
      const moodRestore = h("button.link", { type: "button", onclick: () => { D.moodPrompt = null; syncPrompts(); } }, "Restore preset");
      const roomSel = h("select", { id: "f-room", onchange: () => { D.room = roomSel.value; D.placed = false; autoPlace(); preview.frame([D.room], [{ x: D.x, y: D.y }], 0.12, 20); drawPreview(); } });
      const err = h("p.error", { role: "alert" });
      const seatedIn = h("input", { type: "checkbox", id: "f-seated", checked: D.seated, onchange: () => { D.seated = seatedIn.checked; } });
      const seatedField = h("div.field.check", seatedIn, h("label", { for: "f-seated" }, "Start seated",
        h("small", "Sits on the free A&E waiting seat nearest to their spot. Untick to stand them 1 m in front of it.")));
      const coord = h("span.coord");
      const saveBtn = h("button.btn.primary", { type: "submit" }, existing ? "Save changes" : "Add to scene");

      function autosize(ta, lines) {
        ta.style.height = "auto";
        const max = lines * 20.3 + 20;
        ta.style.height = Math.min(ta.scrollHeight + 2, max) + "px";
        ta.style.overflowY = ta.scrollHeight + 2 > max ? "auto" : "hidden";
      }
      function sizeAreas() {
        autosize(roleTa, 10); autosize(moodTa, 6);
        const top = () => { roleTa.scrollTop = 0; moodTa.scrollTop = 0; };
        top(); requestAnimationFrame(top); setTimeout(top, 60);
      }
      function syncPrompts() {
        const rp = rolePre(), mp = moodPre();
        roleTa.value = D.rolePrompt != null ? D.rolePrompt : preText(rp);
        moodTa.value = D.moodPrompt != null ? D.moodPrompt : preText(mp);
        const rc = D.role === "custom", mc = D.mood === "custom";
        roleTa.placeholder = rc ? "Describe who this person is and how they behave. One behaviour per line works well." : "";
        moodTa.placeholder = mc ? "Describe how this person feels and how it shows. One behaviour per line works well." : "";
        syncRestore();
        sizeAreas();
      }
      function syncRestore() {
        const rc = D.role === "custom", mc = D.mood === "custom";
        roleRestore.hidden = rc || roleTa.value.trim() === preText(rolePre());
        moodRestore.hidden = mc || moodTa.value.trim() === preText(moodPre());
      }
      function setSex(s) {
        const old = D.sex; D.sex = s;
        if (!existing && (FEMALE.includes(nameIn.value.split(" ")[0]) || MALE.includes(nameIn.value.split(" ")[0])) && old !== s) { nameIn.value = randomName(s); D.name = nameIn.value; }
        sexBtns.forEach((b) => b.setAttribute("aria-checked", b.dataset.v === s));
        drawPreview();
      }
      function fillRooms() {
        const sug = new Set((rolePre() && rolePre().default_rooms) || []);
        const all = [...WZ.site.rooms].filter((r) => !/^(ensuite|wc|service)$/.test(r.use) || r.id === D.room).sort((a, b) => (a.label || a.id).localeCompare(b.label || b.id));
        const opt = (r) => h("option", { value: r.id }, r.label || r.id);
        const suggested = all.filter((r) => sug.has(r.id));
        const groups = [];
        if (suggested.length) groups.push(h("optgroup", { label: "Suggested for this role" }, suggested.map(opt)));
        groups.push(h("optgroup", { label: "All rooms" }, all.filter((r) => !sug.has(r.id)).map(opt)));
        roomSel.replaceChildren(...groups);
        if (!D.room || !all.some((r) => r.id === D.room)) D.room = (suggested[0] || all[0]).id;
        roomSel.value = D.room;
      }
      function autoPlace() {
        const p = roomPoint(D.room, D.role, taken);
        if (p) { Object.assign(D, p); D.placed = false; }
      }

      // preview map
      const psvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      psvg.setAttribute("role", "img"); psvg.setAttribute("aria-label", "Map preview of the chosen room. Click to place the character.");
      let preview = null;
      const previewWrap = h("div.preview", psvg, h("div.map-chip", "Click the map to place " ), coord);
      const hint = previewWrap.querySelector(".map-chip");
      function drawPreview() {
        if (!preview) return;
        const first = (D.name || "Character").split(" ")[0];
        hint.textContent = D.placed ? "Click to move " + first : "Click the map to place " + first;
        preview.setCharacters(others, { names: true });
        preview.setStaticPatients(staticBeds(Object.assign({}, project, { characters: others })));
        preview.setRobot(project.robot_start);
        preview.setGhost({ name: D.name, role: D.role, x: D.x, y: D.y });
        const free = new Set(bedList().filter((b) => !taken.has(b.id)).map((b) => b.id));
        preview.setBedState(taken, isPatientRole(D.role) ? free : null);
        const bits = ["x " + D.x.toFixed(1) + ", y " + D.y.toFixed(1)];
        if (D.snap) bits.push(D.snap);
        coord.textContent = bits.join(" · ");
      }
      function onPreviewClick(x, y) {
        const p = snapPoint(x, y, D.role, taken);
        Object.assign(D, { x: +p.x.toFixed(2), y: +p.y.toFixed(2), bed: p.bed, snap: p.snap, placed: true });
        const r = preview.roomAt(D.x, D.y);
        if (r && [...roomSel.options].some((o) => o.value === r.id)) { D.room = r.id; roomSel.value = r.id; }
        err.textContent = "";
        drawPreview();
      }

      // assemble
      const field = (label, forId, control, extra) => h("div.field", h("div.flabel", h("label", { for: forId }, label), extra || null), control);
      const form = h("form.charform", { novalidate: true, onsubmit: submit },
        h("div.dlg-head", h("h2#cd-title", existing ? "Edit " + existing.name : "Add character"),
          iconBtn("close", "Close", () => dlg.close(), "plain")),
        h("div.dlg-body",
          h("div.col-form",
            h("div.row2",
              h("div.field", h("div.flabel", h("span", "Sex")), h("div.pills", { role: "radiogroup", "aria-label": "Sex" }, sexBtns)),
              field("Age", "f-age", ageSel)),
            field("Name", "f-name", h("div.name-row", nameIn, iconBtn("shuffle", "Pick a random name", () => { nameIn.value = randomName(D.sex); D.name = nameIn.value; drawPreview(); nameIn.focus(); }))),
            field("Role", "f-role", roleSel),
            h("div.field.sub", h("div.flabel", h("label.small", { for: "f-roleprompt" }, "What this person is like"), roleRestore), roleTa),
            field("Mood", "f-mood", moodSel),
            h("div.field.sub", h("div.flabel", h("label.small", { for: "f-moodprompt" }, "How the mood shows"), moodRestore), moodTa)),
          h("div.col-map", field("Room", "f-room", roomSel), previewWrap, seatedField, err)),
        h("div.dlg-foot", h("button.btn", { type: "button", onclick: () => dlg.close() }, "Cancel"), saveBtn));
      const dlg = h("dialog.dlg.big", { "aria-labelledby": "cd-title" }, form);
      dlg.addEventListener("close", () => { if (preview) preview.destroy(); dlg.remove(); });
      document.body.appendChild(dlg);

      // initial values
      fillRooms(); syncPrompts();
      sexBtns.forEach((b) => b.setAttribute("aria-checked", b.dataset.v === D.sex));
      if (!existing) autoPlace();
      dlg.showModal();
      sizeAreas();
      preview = new HospitalMap(psvg, WZ.site, { zoomTo: [D.room], pad: 3, onClick: onPreviewClick });
      preview.frame([D.room], [{ x: D.x, y: D.y }], 0.12, 20, false);
      drawPreview();
      if (!existing) { nameIn.focus(); nameIn.select(); }

      async function submit(ev) {
        ev.preventDefault();
        err.textContent = "";
        const name = nameIn.value.trim();
        if (!name) { err.textContent = "Give the character a name."; nameIn.focus(); return; }
        const rp = roleTa.value.trim(), mp = moodTa.value.trim();
        if (D.role === "custom" && !rp) { err.textContent = "A custom role needs a description. Say who this person is."; roleTa.focus(); return; }
        const body = {
          name, sex: D.sex, age_band: D.age, role: D.role, mood: D.mood,
          role_prompt: D.role === "custom" ? rp : (rp === preText(rolePre()) ? null : rp),
          mood_prompt: D.mood === "custom" ? (mp || null) : (mp === preText(moodPre()) ? null : mp),
          room: D.room, x: D.x, y: D.y, bed: D.seated ? null : D.bed, start_seated: D.seated,
        };
        saveBtn.disabled = true;
        try {
          const saved = existing ? await API.put(WZ.projectRoutes() + "/characters/" + existing.id, body) : await API.post(WZ.projectRoutes() + "/characters", body);
          await reload();
          S.sel = saved.id; S.tab = "characters";
          dlg.close();
          renderSide(); drawMap(); loadMemory();
          S.map.frame([], P().characters.concat(P().robot_start ? [P().robot_start] : []), 0.2, 16);
          WZ.toast(existing ? "Saved " + saved.name : saved.name + " is on the map");
        } catch (e) { err.textContent = e.message; saveBtn.disabled = false; }
      }
    }

    // While a run is going (its debrief and report too) the characters, memories and project are the run's:
    // the server answers 409, and these controls are disabled so nobody meets that.
    function lockUi() {
      if (S.dead) return;
      const on = !!(WZ.state && WZ.state.running);
      for (const b of body.querySelectorAll(".cacts button, .sec-head .btn, .card-detail .btn.primary, .seat-chk input, .area .icon-btn")) {
        b.disabled = on;
        if (on) b.setAttribute("title", "Not while a run is going"); else b.removeAttribute("title");
      }
    }
    WZ.on("state", () => lockUi());
    new MutationObserver(lockUi).observe(body, { childList: true });

    // ---- wiring ----
    const onProject = () => { if (S.dead) return; renderSide(); drawMap(); loadMemory(); };
    S.onProject = onProject;
    WZ.on("project", S.listener = onProject);
    S.dead = false;
    renderSide(); drawMap(); loadMemory();
    root._scene = S;
  }

  function unmount() {
    const root = document.getElementById("view");
    const S = root._scene;
    if (S) { S.dead = true; if (S.map) S.map.destroy(); S.thumbs.forEach((m) => m.destroy()); if (S.onKey) document.removeEventListener("keydown", S.onKey); }
  }

  WZ.registerPage("scene", { title: "Scene", mount, unmount });
})();
