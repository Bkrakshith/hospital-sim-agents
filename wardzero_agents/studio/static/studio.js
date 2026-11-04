/* WardZero Studio shell: page router, project state, shared helpers.
   Pages (Scene, Robot, Run, Report) register themselves with WZ.registerPage(id, {title, mount, unmount}).
   mount(el) fills el and may return nothing; unmount() cleans up. Listen for project changes with
   WZ.on("project", fn). The shell loads the site map, presets and the current project once. */
(function () {
  /* h("div.class#id", {attr}, ...children): small DOM builder. Text goes in as text, never as HTML. */
  function h(spec, attrs, ...kids) {
    const m = /^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i.exec(spec);
    const node = document.createElement(m[1] || "div");
    for (const part of (m[2].match(/[.#][\w-]+/g) || [])) {
      if (part[0] === ".") node.classList.add(part.slice(1)); else node.id = part.slice(1);
    }
    if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) { kids.unshift(attrs); attrs = null; }
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v === false || v == null) continue;
      if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (k === "value") node.value = v;
      else if (k === "checked" || k === "disabled" || k === "selected" || k === "hidden") node[k] = !!v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    const add = (c) => {
      if (c == null || c === false) return;
      if (Array.isArray(c)) c.forEach(add);
      else node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    };
    kids.forEach(add);
    return node;
  }

  const bus = new EventTarget();
  const WZ = {
    h, pages: {}, current: null, project: null, projects: [], site: null, presets: null,
    state: { running: false },
    on(name, fn) { bus.addEventListener(name, (e) => fn(e.detail)); },
    emit(name, detail) { bus.dispatchEvent(new CustomEvent(name, { detail })); },
    registerPage(id, page) { WZ.pages[id] = page; },
  };
  const PAGES = [["scene", "Scene"], ["robot", "Robot"], ["run", "Run"], ["report", "Report"]];
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } },
  };

  // ---- toasts and dialogs ----
  WZ.toast = function (text, kind) {
    const t = h("div.toast" + (kind ? "." + kind : ""), text);
    document.getElementById("toasts").appendChild(t);
    setTimeout(() => t.classList.add("out"), 3600);
    setTimeout(() => t.remove(), 4000);
  };

  /* Promise-based dialog for confirmation and one-line prompts. */
  WZ.ask = function ({ title, text, ok = "OK", cancel = "Cancel", danger = false, input = null }) {
    return new Promise((resolve) => {
      const field = input ? h("input.text", { type: "text", value: input.value || "", maxlength: 80, "aria-label": input.label || title }) : null;
      const dlg = h("dialog.dlg.small", { "aria-labelledby": "ask-title" },
        h("form", { method: "dialog", onsubmit: (e) => { e.preventDefault(); done(field ? field.value.trim() || null : true); } },
          h("h2#ask-title", title), text ? h("p.muted", text) : null, field,
          h("div.dlg-foot", h("button.btn", { type: "button", onclick: () => done(field ? null : false) }, cancel),
            h("button.btn" + (danger ? ".danger-solid" : ".primary"), { type: "submit" }, ok))));
      const done = (v) => { dlg.close(); dlg.remove(); resolve(v); };
      dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(field ? null : false); });
      document.body.appendChild(dlg);
      dlg.showModal();
      if (field) { field.focus(); field.select(); }
    });
  };

  // ---- projects ----
  async function loadProjects() { WZ.projects = await API.get("/api/projects"); }

  WZ.setProject = function (p) {
    WZ.project = p;
    const i = WZ.projects.findIndex((q) => q.id === p.id);
    if (i >= 0) WZ.projects[i] = p; else WZ.projects.push(p);
    store.set("wz.project", p.id);
    renderProjectSelect();
    WZ.emit("project", p);
  };
  WZ.saveProject = async function (patch) {   // PUT and adopt the saved project
    const p = await API.put("/api/projects/" + WZ.project.id, patch);
    WZ.setProject(p);
    return p;
  };
  async function createProject(name, extra) {
    const p = await API.post("/api/projects", Object.assign({ name, scenario_id: "reception_guide", characters: [] }, extra || {}));
    WZ.setProject(p);
    return p;
  }
  WZ.projectRoutes = () => "/api/projects/" + WZ.project.id;

  function renderProjectSelect() {
    const sel = document.getElementById("project-select");
    sel.replaceChildren(...WZ.projects.map((p) => h("option", { value: p.id, selected: WZ.project && p.id === WZ.project.id }, p.name)));
  }
  function wireProjectMenu() {
    const sel = document.getElementById("project-select"), btn = document.getElementById("project-menu-btn"), menu = document.getElementById("project-menu");
    sel.addEventListener("change", async () => {
      try { WZ.setProject(await API.get("/api/projects/" + sel.value)); } catch (e) { WZ.toast(e.message, "bad"); }
    });
    const close = () => { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    btn.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); });
    document.addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    menu.addEventListener("click", async (e) => {
      const act = e.target.dataset && e.target.dataset.act; close();
      try {
        if (act === "new") {
          const name = await WZ.ask({ title: "New project", text: "Starts empty. You can add characters or load a scenario.", ok: "Create", input: { value: "Untitled project", label: "Project name" } });
          if (name) await createProject(name);
        } else if (act === "rename") {
          const name = await WZ.ask({ title: "Rename project", ok: "Rename", input: { value: WZ.project.name, label: "Project name" } });
          if (name) await WZ.saveProject({ name });
        } else if (act === "delete") {
          if (!(await WZ.ask({ title: "Delete this project?", text: "\"" + WZ.project.name + "\" and its characters will be removed.", ok: "Delete", danger: true }))) return;
          await API.del("/api/projects/" + WZ.project.id);
          WZ.projects = WZ.projects.filter((p) => p.id !== WZ.project.id);
          if (WZ.projects.length) WZ.setProject(WZ.projects[0]); else await createProject("Untitled project");
        }
      } catch (err) { WZ.toast(err.message, "bad"); }
    });
  }

  // ---- router ----
  function renderTabs(active) {
    document.getElementById("tabs").replaceChildren(...PAGES.map(([id, label], i) =>
      h("a.tab" + (id === active ? ".on" : ""), { href: "#/" + id, "aria-current": id === active ? "page" : false },
        h("span.num", String(i + 1)), label)));
  }
  function route() {
    const id = (location.hash.match(/^#\/(\w+)/) || [])[1];
    const page = PAGES.some((p) => p[0] === id) ? id : "scene";
    if (WZ.current && WZ.pages[WZ.current] && WZ.pages[WZ.current].unmount) WZ.pages[WZ.current].unmount();
    WZ.current = page;
    renderTabs(page);
    const view = document.getElementById("view");
    view.replaceChildren();
    view.dataset.page = page;
    if (WZ.pages[page]) WZ.pages[page].mount(view);
    else view.appendChild(h("div.placeholder", h("h1", PAGES.find((p) => p[0] === page)[1]), h("p.muted", "This step is not built yet. Page " + (PAGES.findIndex((p) => p[0] === page) + 1) + " of 4.")));
  }

  // ---- cost meter ----
  const kfmt = (n) => { n = Number(n || 0); return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n); };
  function renderCost(s) {
    WZ.state = s;
    const box = document.getElementById("cost");
    const show = !!s.running;
    box.hidden = !show;
    if (show) {
      const c = Number(s.cost_usd || 0), b = Number(s.budget_usd || 0), t = s.tokens || {};
      document.getElementById("cost-text").textContent = "$" + c.toFixed(2) + (b ? " of $" + b.toFixed(2) : "");
      document.getElementById("cost-tokens").textContent = kfmt(t.input) + " in · " + kfmt(t.output) + " out · " + kfmt(t.cache_read) + " cached";
      document.getElementById("cost-bar").style.width = (b ? Math.min(100, (c / b) * 100) : 0) + "%";
      box.classList.toggle("warn", b > 0 && c / b > 0.85);
    }
    WZ.emit("state", s);
  }

  WZ.start = async function () {
    wireProjectMenu();
    API.watchState(renderCost);
    window.addEventListener("hashchange", route);
    try {
      [WZ.site, WZ.presets] = await Promise.all([API.get("/api/site"), API.get("/api/presets")]);
      await loadProjects();
      const want = store.get("wz.project");
      let p = WZ.projects.find((q) => q.id === want) || WZ.projects[0];
      if (!p) p = await createProject("Untitled project");
      else WZ.setProject(p);
    } catch (e) {
      document.getElementById("view").replaceChildren(h("div.placeholder", h("h1", "Studio could not start"), h("p.error", e.message)));
      return;
    }
    route();
  };

  window.WZ = WZ;
  window.h = h;
})();
