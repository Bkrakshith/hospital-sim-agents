/* HospitalMap: top-down SVG map of the hospital from config/site/hospital.json.
   World metres, +x east, +y north. SVG y points down, so every point is drawn at (x, -y).
   Shared by the Scene page, the character preview and the scenario card thumbnails. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  const USE_GROUP = {
    bedroom: "clinical", ae_treatment: "clinical",
    lobby: "public", waiting: "public", day_room: "public", ward_entrance: "public",
    corridor: "circ", lift_lobby: "circ",
    surgical: "surg", imaging: "surg",
    nurse_base: "staff", triage: "staff", ae_support: "staff",
    service: "svc", wc: "svc", ensuite: "svc",
  };

  const ROLE_COLORS = {
    ward_nurse: "#1f9d8f", receptionist: "#d98a1f", patient: "#d6567a", visitor: "#4a86d9",
    surgeon: "#7a5cc7", porter: "#8a9a2f", custom: "#6b7785",
  };
  const FALLBACK = ["#1f9d8f", "#d98a1f", "#d6567a", "#4a86d9", "#7a5cc7", "#8a9a2f"];
  function roleColor(role) {
    if (ROLE_COLORS[role]) return ROLE_COLORS[role];
    let s = 0; for (const c of String(role)) s = (s * 31 + c.charCodeAt(0)) >>> 0;
    return FALLBACK[s % FALLBACK.length];
  }

  const poly = (r) => r.poly.map((p) => p[0] + "," + (-p[1])).join(" ");
  function bbox(points) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of points) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
    return { x0, y0, x1, y1 };
  }
  function pointInPoly(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  class HospitalMap {
    /* opts: onClick(x, y, event), focusRooms [ids], zoomTo [ids], interactive (default true),
       labels (default true), thumb (small static card picture), pad (metres around zoomTo) */
    constructor(svg, site, opts) {
      this.svg = svg; this.site = site; this.opts = Object.assign({ interactive: true, labels: true, thumb: false, pad: 2.5 }, opts || {});
      this.rooms = new Map(site.rooms.map((r) => [r.id, r]));
      this.markers = [];       // {g, x, y, rot, s}
      this.roomLabels = [];
      this.focus = new Set(this.opts.focusRooms || []);
      this.tween = null;
      svg.classList.add("hmap");
      svg.replaceChildren();
      const b = site.bounds;
      this.bounds = { x0: b[0], x1: b[1], y0: b[2], y1: b[3] };
      this._initView();
      this._build();
      if (this.opts.interactive) this._wire();
      if (!this.opts.thumb && typeof ResizeObserver !== "undefined") {
        this.ro = new ResizeObserver(() => this._resize());
        this.ro.observe(svg);
      }
      this._relayout();
    }

    destroy() { if (this.ro) this.ro.disconnect(); cancelAnimationFrame(this.tween); }

    // ---- view ----
    _aspect() {
      if (this.opts.thumb) return 16 / 9;
      const r = this.svg.getBoundingClientRect();
      return r.width > 10 && r.height > 10 ? r.width / r.height : 1.5;
    }
    _boxFor(x0, y0, x1, y1) {   // world bbox -> view box (svg coords) with the container's aspect
      const a = this._aspect();
      let w = x1 - x0, h = y1 - y0;
      if (w / h < a) w = h * a; else h = w / a;
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      return { x: cx - w / 2, y: -cy - h / 2, w, h };
    }
    _initView() {
      const o = this.opts;
      if (o.zoomTo && o.zoomTo.length) this.vb = this._roomsBox(o.zoomTo, o.pad);
      else this.vb = this._fitBox();
      this._applyVb();
    }
    _fitBox() {
      const pts = [];
      for (const r of this.site.rooms) pts.push(...r.poly);
      for (const w of this.site.walls) pts.push([w[0], w[1]], [w[2], w[3]]);
      const b = pts.length ? bbox(pts) : this.bounds, p = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 0.04;
      return this._boxFor(b.x0 - p, b.y0 - p, b.x1 + p, b.y1 + p);
    }
    /* Frame rooms and points together with a fractional margin (and a minimum span in metres). */
    _frameBox(ids, points, frac, minSpan) {
      const pts = [...(points || []).map((p) => [p.x, p.y])];
      for (const id of ids || []) { const r = this.rooms.get(id); if (r) pts.push(...r.poly); }
      if (!pts.length) return this._fitBox();
      const b = bbox(pts);
      let w = b.x1 - b.x0, h = b.y1 - b.y0;
      const pad = Math.max(w, h) * (frac == null ? 0.08 : frac);
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, ms = (minSpan || 12) / 2;
      const hw = Math.max(w / 2 + pad, ms), hh = Math.max(h / 2 + pad, ms * 0.6);
      return this._boxFor(cx - hw, cy - hh, cx + hw, cy + hh);
    }
    frame(ids, points, frac, minSpan, animate) { this.setView(this._frameBox(ids, points, frac, minSpan), animate !== false); }
    _roomsBox(ids, pad) {
      const pts = [];
      for (const id of ids) { const r = this.rooms.get(id); if (r) pts.push(...r.poly); }
      if (!pts.length) return this._fitBox();
      const b = bbox(pts), p = Math.max(pad, 0.12 * Math.max(b.x1 - b.x0, b.y1 - b.y0));
      return this._boxFor(b.x0 - p, b.y0 - p, b.x1 + p, b.y1 + p);
    }
    _applyVb() { const v = this.vb; this.svg.setAttribute("viewBox", [v.x, v.y, v.w, v.h].join(" ")); }
    _resize() {
      const a = this._aspect(), v = this.vb, cx = v.x + v.w / 2, cy = v.y + v.h / 2;
      const w = v.w, h = w / a;
      this.vb = { x: cx - w / 2, y: cy - h / 2, w, h };
      this._applyVb(); this._relayout();
    }
    get k() {   // metres per screen pixel
      if (this.opts.thumb) return this.vb.w / 190;
      const r = this.svg.getBoundingClientRect();
      return r.width > 10 ? this.vb.w / r.width : this.vb.w / 800;
    }
    setView(box, animate) {
      cancelAnimationFrame(this.tween);
      if (!animate) { this.vb = box; this._applyVb(); this._relayout(); return; }
      const from = this.vb, t0 = performance.now(), dur = 260;
      const step = (now) => {
        const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
        const lerp = (a, b) => a + (b - a) * e;
        this.vb = { x: lerp(from.x, box.x), y: lerp(from.y, box.y), w: lerp(from.w, box.w), h: lerp(from.h, box.h) };
        this._applyVb(); this._relayout();
        if (t < 1) this.tween = requestAnimationFrame(step);
      };
      this.tween = requestAnimationFrame(step);
    }
    fit(animate) { this.setView(this._fitBox(), animate !== false); }
    zoomTo(ids, pad, animate) { this.setView(this._roomsBox(ids, pad == null ? this.opts.pad : pad), animate !== false); }
    zoomToPoint(x, y, halfWidth, animate) {
      const a = this._aspect();
      this.setView(this._boxFor(x - halfWidth, y - halfWidth / a, x + halfWidth, y + halfWidth / a), animate !== false);
    }
    setFocus(ids) {
      this.focus = new Set(ids || []);
      for (const [id, poly_] of this.polys) poly_.classList.toggle("focus", this.focus.has(id));
    }

    // ---- static drawing ----
    _visible(x0, y0, x1, y1) {
      if (!this.opts.thumb) return true;
      const v = this.vb, m = 3;
      return x1 >= v.x - m && x0 <= v.x + v.w + m && -y0 >= v.y - m && -y1 <= v.y + v.h + m;
    }
    _build() {
      const s = this.site, svg = this.svg, th = this.opts.thumb;
      const g = (cls) => el("g", { class: cls }, svg);
      const gRooms = g("l-rooms"), gFurn = g("l-furn"), gWalls = g("l-walls"), gBeds = g("l-beds"),
        gAreas = g("l-areas"), gLabels = g("l-labels"), gMarkers = g("l-markers");
      this.gMarkers = gMarkers;
      this.gAreas = gAreas;
      this.polys = new Map();
      for (const r of s.rooms) {
        const bb = bbox(r.poly);
        if (!this._visible(bb.x0, bb.y0, bb.x1, bb.y1)) continue;
        const p = el("polygon", { points: poly(r), class: "room u-" + (USE_GROUP[r.use] || "svc") + (this.focus.has(r.id) ? " focus" : "") }, gRooms);
        const t = el("title", {}, p); t.textContent = r.label || r.id;
        this.polys.set(r.id, p);
        if (this.opts.labels && !th) {
          const c = [(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2];
          const txt = el("text", { class: "room-label", "text-anchor": "middle", "dominant-baseline": "central" }, gLabels);
          const full = (r.label || "").trim();
          const cands = [full];
          const cut = full.split(/\s*(?:,|\+|\()/)[0].trim();
          if (cut && cut !== full) cands.push(cut);
          this.roomLabels.push({ txt, c, bb, cands: cands.filter(Boolean) });
        }
      }
      for (const st of s.stairs || []) el("polygon", { points: poly(st), class: "stair" }, gFurn);
      for (const l of s.lifts || []) {
        const c = l.car; el("rect", { x: c[0], y: -c[3], width: c[2] - c[0], height: c[3] - c[1], class: "lift" }, gFurn);
      }
      for (const f of s.furniture || []) {
        el("rect", { x: -f.w / 2, y: -f.d / 2, width: f.w, height: f.d, class: "furn", transform: `translate(${f.x} ${-f.y}) rotate(${-f.yaw * 180 / Math.PI})` }, gFurn);
      }
      if (!th) for (const q of s.seats || []) {
        el("rect", { x: -0.22, y: -0.22, width: 0.44, height: 0.44, rx: 0.08, class: "seat", transform: `translate(${q.x} ${-q.y}) rotate(${-q.yaw * 180 / Math.PI})` }, gFurn);
      }
      // walls, one path for all segments
      let d = "";
      for (const w of s.walls) if (this._visible(Math.min(w[0], w[2]), Math.min(w[1], w[3]), Math.max(w[0], w[2]), Math.max(w[1], w[3]))) d += `M${w[0]} ${-w[1]}L${w[2]} ${-w[3]}`;
      el("path", { d, class: "walls" }, gWalls);
      let dd = "";
      for (const o of s.doors) if (this._visible(Math.min(o.a[0], o.b[0]), Math.min(o.a[1], o.b[1]), Math.max(o.a[0], o.b[0]), Math.max(o.a[1], o.b[1]))) dd += `M${o.a[0]} ${-o.a[1]}L${o.b[0]} ${-o.b[1]}`;
      el("path", { d: dd, class: "doors" }, gWalls);
      this.bedEls = new Map();
      for (const b of s.beds) {
        if (!this._visible(b.x - 2, b.y - 2, b.x + 2, b.y + 2)) continue;
        const ang = Math.atan2(b.head[1] - b.y, b.head[0] - b.x);
        const bg = el("g", { class: "bed", transform: `translate(${b.x} ${-b.y}) rotate(${-ang * 180 / Math.PI})` }, gBeds);
        el("rect", { x: -1, y: -0.45, width: 2, height: 0.9, rx: 0.12, class: "bed-body" }, bg);
        el("rect", { x: 0.55, y: -0.36, width: 0.36, height: 0.72, rx: 0.1, class: "bed-pillow" }, bg);
        const t = el("title", {}, bg); t.textContent = "Bed " + b.id;
        this.bedEls.set(b.id, bg);
      }
      this.gAreaLabels = el("g", {}, gMarkers);
      this.gStatic = el("g", {}, gMarkers);
      this.gRobot = el("g", {}, gMarkers);
      this.gChars = el("g", {}, gMarkers);
      this.gGhost = el("g", {}, gMarkers);
    }

    _relayout() {
      const k = this.k;
      for (const m of this.markers) {
        m.g.setAttribute("transform", `translate(${m.x} ${-m.y}) scale(${k * (m.s || 1)})`);
      }
      if (!this.roomLabels.length) return;
      const fs = 11.5, v = this.vb, cw = fs * 0.58;
      for (const r of this.roomLabels) {
        const wpx = (r.bb.x1 - r.bb.x0) / k, hpx = (r.bb.y1 - r.bb.y0) / k;
        let text = null, at = null;
        if (hpx > 16) {
          // anchors inside the room, tried in order: centre, top, bottom, top-left, bottom-left
          const ix = 8 * k, iy = 12 * k;
          search:
          for (const cand of r.cands) {
            const tw = cand.length * cw;
            if (tw > wpx * 0.9) continue;
            const half = (tw / 2 + 8) * k, hh = 10 * k, mid = (r.bb.x0 + r.bb.x1) / 2;
            const anchors = [[mid, r.c[1]], [mid, r.bb.y1 - iy], [mid, r.bb.y0 + iy],
              [r.bb.x0 + half, r.bb.y1 - iy], [r.bb.x0 + half, r.bb.y0 + iy]];
            for (const a of anchors) {
              if (hpx < 16 && a !== anchors[0]) continue;
              if (a[0] - half < v.x || a[0] + half > v.x + v.w || -a[1] - hh < v.y || -a[1] + hh > v.y + v.h) continue;
              let clash = false;
              for (const mk of this.markers) {
                if (Math.abs(mk.x - a[0]) / k < tw / 2 + 22 && Math.abs(mk.y - a[1]) / k < (mk.r || 24)) { clash = true; break; }
              }
              if (!clash) { text = cand; at = a; break search; }
            }
          }
        }
        if (text === null) { r.txt.setAttribute("display", "none"); continue; }
        r.txt.removeAttribute("display");
        if (r.txt.textContent !== text) r.txt.textContent = text;
        r.txt.setAttribute("transform", `translate(${at[0]} ${-at[1]}) scale(${k})`);
        r.txt.setAttribute("font-size", fs);
      }
    }

    // ---- interaction ----
    _toWorld(ev) {
      const pt = this.svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const p = pt.matrixTransform(this.svg.getScreenCTM().inverse());
      return { x: p.x, y: -p.y };
    }
    _wire() {
      const svg = this.svg; let drag = null, raf = 0;
      svg.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        const p = this._toWorld(ev), v = this.vb;
        const f = Math.exp(ev.deltaY * 0.0016);
        const w = Math.min(Math.max(v.w * f, 7), 150), r = w / v.w;
        const px = p.x, py = -p.y;
        this.vb = { x: px - (px - v.x) * r, y: py - (py - v.y) * r, w, h: v.h * r };
        cancelAnimationFrame(this.tween);
        if (!raf) raf = requestAnimationFrame(() => { raf = 0; this._applyVb(); this._relayout(); });
      }, { passive: false });
      svg.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        drag = { x: ev.clientX, y: ev.clientY, vb: Object.assign({}, this.vb), moved: false };
        svg.setPointerCapture(ev.pointerId);
      });
      svg.addEventListener("pointermove", (ev) => {
        if (this.opts.onHover && !drag) { const p = this._toWorld(ev); this.opts.onHover(p.x, p.y, ev); }
        if (!drag) return;
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (!drag.moved && Math.hypot(dx, dy) < 4) return;
        drag.moved = true; svg.classList.add("panning");
        cancelAnimationFrame(this.tween);
        const k = drag.vb.w / svg.getBoundingClientRect().width;
        this.vb = { x: drag.vb.x - dx * k, y: drag.vb.y - dy * k, w: drag.vb.w, h: drag.vb.h };
        this._applyVb(); this._relayout();
      });
      const end = (ev) => {
        if (!drag) return;
        const moved = drag.moved; drag = null; svg.classList.remove("panning");
        if (!moved && ev.type === "pointerup" && this.opts.onClick) { const p = this._toWorld(ev); this.opts.onClick(p.x, p.y, ev); }
      };
      svg.addEventListener("pointerup", end);
      svg.addEventListener("pointercancel", end);
    }

    // ---- layers ----
    _marker(parent, x, y, rot) {
      const g = el("g", {}, parent);
      const m = { g, x, y, rot };
      this.markers.push(m);
      return { g, m };
    }
    _clear(layer) {
      const gone = new Set([...layer.children]);
      this.markers = this.markers.filter((m) => !gone.has(m.g));
      layer.replaceChildren();
    }
    _person(layer, c, o) {
      o = o || {};
      const { g, m } = this._marker(layer, c.x, c.y);
      m.s = o.scale || 1;
      g.setAttribute("class", "person" + (o.cls ? " " + o.cls : ""));
      const col = o.color || roleColor(c.role);
      if (o.ring) el("circle", { r: 17, class: "ring" }, g);
      el("circle", { r: o.r || 11, fill: col, class: "dot" }, g);
      const t = el("text", { class: "initial", "text-anchor": "middle", "dominant-baseline": "central" }, g);
      t.textContent = o.initial !== undefined ? o.initial : (c.name || "?").trim().charAt(0).toUpperCase();
      if (o.name) {
        const n = el("text", { class: "pname", y: 25, "text-anchor": "middle" }, g);
        n.textContent = o.name;
      }
      if (o.title) { const tt = el("title", {}, g); tt.textContent = o.title; }
      return g;
    }
    /* characters: [{id,name,role,x,y,yaw}], opts {selected, hover, onPick, names, draggable, handle (an id: that
       character gets a turn handle)}. Returns Map id -> {g, m, handle, line} for dragging (see movePerson). */
    setCharacters(list, o) {
      o = o || {};
      this._clear(this.gChars);
      const out = new Map();
      for (const c of list) {
        const g = this._person(this.gChars, c, {
          ring: c.id === o.selected, name: o.names === false ? null : (c.name || "").split(" ")[0],
          title: c.name + (o.draggable ? " (drag to move)" : ""), scale: this.opts.thumb ? 0.62 : 1, r: this.opts.thumb ? 11 : 11,
          cls: (c.id === o.hover ? "hover" : "") + (o.draggable ? " draggable" : ""),
        });
        const m = this.markers[this.markers.length - 1];
        const E = { g, m, yaw: c.yaw || 0 };
        if (o.handle === c.id) {
          E.line = el("line", { x1: 0, y1: 0, class: "person-arm" }, g);
          E.handle = el("circle", { r: 6, class: "person-handle" }, g);
          const ht = el("title", {}, E.handle); ht.textContent = "Drag to turn " + (c.name || "").split(" ")[0];
          g.insertBefore(E.line, g.firstChild);
          m.r = 40;
          this._posePerson(E, c.x, c.y, E.yaw);
        }
        out.set(c.id, E);
        if (o.onPick && !o.draggable) { g.style.cursor = "pointer"; g.addEventListener("pointerdown", (e) => e.stopPropagation()); g.addEventListener("click", (e) => { e.stopPropagation(); o.onPick(c); }); }
      }
      this._relayout();
      return out;
    }
    _posePerson(E, x, y, yaw) {
      E.m.x = x; E.m.y = y; E.yaw = yaw;
      if (E.handle) {
        const hx = 30 * Math.cos(yaw), hy = -30 * Math.sin(yaw);
        E.handle.setAttribute("cx", hx); E.handle.setAttribute("cy", hy);
        E.line.setAttribute("x2", hx); E.line.setAttribute("y2", hy);
      }
    }
    /* Move a character marker from setCharacters without rebuilding it (dragging). */
    movePerson(E, x, y, yaw) { this._posePerson(E, x, y, yaw == null ? E.yaw : yaw); this._relayout(); }
    /* Forbidden areas: [{id, name, x0, y0, x1, y1}] drawn as red rectangles with their names. */
    setAreas(list) {
      this.gAreas.replaceChildren();
      this._clear(this.gAreaLabels);
      for (const a of list || []) {
        const r = el("rect", { x: a.x0, y: -a.y1, width: a.x1 - a.x0, height: a.y1 - a.y0, class: "farea" }, this.gAreas);
        const t = el("title", {}, r); t.textContent = "Forbidden area: " + a.name;
        if (this.opts.thumb || this.opts.labels === false) continue;
        const { g, m } = this._marker(this.gAreaLabels, (a.x0 + a.x1) / 2, a.y1);
        m.r = 10;
        g.setAttribute("class", "farea-tag");
        const txt = el("text", { y: -5, "text-anchor": "middle" }, g);
        txt.textContent = a.name;
      }
      this._relayout();
    }
    /* The rectangle being drawn (or null). */
    setDraft(a) {
      if (!this._draft) this._draft = el("rect", { class: "farea draft" }, this.gAreas);
      if (!a) { this._draft.remove(); this._draft = null; return; }
      if (!this._draft.isConnected) this.gAreas.appendChild(this._draft);
      const x0 = Math.min(a.x0, a.x1), x1 = Math.max(a.x0, a.x1), y0 = Math.min(a.y0, a.y1), y1 = Math.max(a.y0, a.y1);
      for (const [k, v] of Object.entries({ x: x0, y: -y1, width: x1 - x0, height: y1 - y0 })) this._draft.setAttribute(k, v);
    }
    setStaticPatients(beds) {
      this._clear(this.gStatic);
      for (const b of beds) this._person(this.gStatic, b, { color: "var(--static)", r: 7, initial: "", cls: "static", scale: this.opts.thumb ? 0.62 : 1, title: "Patient in bed " + b.id });
      this._relayout();
    }
    /* r: {x, y, yaw}. opts: {handle: true} adds a turn handle (Robot page), {label: "text"} renames the label. */
    setRobot(r, opts) {
      opts = opts || {};
      this._clear(this.gRobot);
      this._robot = null;
      if (!r) return null;
      const { g, m } = this._marker(this.gRobot, r.x, r.y);
      m.s = this.opts.thumb ? 0.7 : 1;
      g.setAttribute("class", "robot");
      el("circle", { r: 15, class: "robot-halo" }, g);
      const body = el("path", { d: "M0,-11 L9,8 L0,4 L-9,8 Z", class: "robot-body" }, g);
      const R = { g, m, body, yaw: r.yaw || 0 };
      if (opts.handle) {
        R.line = el("line", { x1: 0, y1: 0, class: "robot-arm" }, g);
        R.handle = el("circle", { r: 7, class: "robot-handle" }, g);
        const ht = el("title", {}, R.handle); ht.textContent = "Drag to turn the robot";
      }
      if (opts.handle) m.r = 48;
      if (!this.opts.thumb && !opts.handle) { const t = el("text", { class: "pname robot-label", x: 20, y: 4, "text-anchor": "start" }, g); t.textContent = opts.label || "Robot start"; }
      const tt = el("title", {}, g); tt.textContent = opts.label || "Robot start";
      this._robot = R;
      this._poseRobot(R, r.x, r.y, r.yaw || 0);
      this._relayout();
      return R;
    }
    _poseRobot(R, x, y, yaw) {
      R.m.x = x; R.m.y = y; R.yaw = yaw;
      R.body.setAttribute("transform", `rotate(${-(yaw * 180) / Math.PI + 90})`);
      if (R.handle) {
        const hx = 36 * Math.cos(yaw), hy = -36 * Math.sin(yaw);
        R.handle.setAttribute("cx", hx); R.handle.setAttribute("cy", hy);
        R.line.setAttribute("x2", hx); R.line.setAttribute("y2", hy);
      }
    }
    /* Move the robot marker without rebuilding it (drag and live updates). */
    moveRobot(x, y, yaw) {
      if (!this._robot) return;
      this._poseRobot(this._robot, x, y, yaw);
      this._relayout();
    }
    setGhost(c) {
      this._clear(this.gGhost);
      if (!c) return;
      this._person(this.gGhost, c, { cls: "ghost", ring: true, name: c.name ? c.name.split(" ")[0] : null });
      this._relayout();
    }
    /* beds: {taken:Set, pickable:Set} */
    setBedState(taken, pickable) {
      for (const [id, g] of this.bedEls) {
        g.classList.toggle("taken", !!taken && taken.has(id));
        g.classList.toggle("pickable", !!pickable && pickable.has(id) && !(taken && taken.has(id)));
      }
    }
    roomAt(x, y) {
      for (const r of this.site.rooms) if (pointInPoly(x, y, r.poly)) return r;
      return null;
    }
  }

  window.HospitalMap = HospitalMap;
  window.WZMap = { roleColor, pointInPoly, bbox, USE_GROUP };
})();
