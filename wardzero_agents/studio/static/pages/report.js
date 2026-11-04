/* Page 4, Report: the AI-written report of a run in a sandboxed frame, a list of runs, and repeat batches. */
(function () {
  const { h } = window;
  const when = (sec) => { try { return new Date(sec * 1000).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); } catch (e) { return ""; } };

  function mount(root) {
    const S = { runs: [], batches: [], sel: null, dead: false, timer: null };
    const P = () => WZ.project;
    const list = h("div.side-body");
    const side = h("aside.side", list);
    const frameBox = h("section.reportbox", { "aria-label": "Report" });
    root.append(h("div.report-page", side, frameBox));

    const scenarioTitle = (id) => { const s = WZ.presets.scenarios.find((x) => x.id === id); return s ? s.title : id; };

    async function load() {
      try {
        const [runs, batches] = await Promise.all([API.get("/api/runs?project=" + encodeURIComponent(P().id)), API.get("/api/batches?scenario=" + encodeURIComponent(P().scenario_id))]);
        S.runs = runs.filter((r) => r.has_report); S.batches = batches.filter((b) => b.has_report);
        S.pending = runs.some((r) => !r.has_report) && !!(WZ.state && WZ.state.running);
      } catch (e) { WZ.toast(e.message, "bad"); }
      if (S.dead) return;
      if (!S.sel || ![...S.runs, ...S.batches].some((r) => r.run_id === S.sel)) {
        const live = WZ.state && WZ.state.has_report && WZ.state.run_id;
        const pick = (live && S.runs.find((r) => r.run_id === live)) || S.runs[0] || S.batches[0];
        S.sel = pick ? pick.run_id : null;
      }
      renderList(); renderFrame();
    }

    function badge(r) {
      if (r.score == null && !r.verdict) return null;
      const v = String(r.verdict || "").toLowerCase();
      return h("span.tag" + (/pass/.test(v) ? ".pass" : /fail|invalid/.test(v) ? ".fail" : ".none"), (r.verdict || "") + (r.score != null ? " " + Math.round(r.score) : ""));
    }
    function item(r, batch) {
      const on = r.run_id === S.sel;
      return h("li", h("button.run-item" + (on ? ".on" : ""), { type: "button", "aria-pressed": String(on), onclick: () => { S.sel = r.run_id; renderList(); renderFrame(); } },
        h("div.ri-top", h("b", batch ? "Repeat batch, " + r.n + " runs" : scenarioTitle(r.scenario_id)), batch ? h("span.tag.none", r.passed + " of " + r.of + " passed") : badge(r)),
        h("div.ri-meta", when(r.mtime), h("span.ri-id", r.run_id))));
    }
    function renderList() {
      list.replaceChildren(
        h("div.sec-head", h("h2", "Reports"), h("button.link", { type: "button", onclick: load }, "Refresh")),
        h("p.muted.lead", "One report for each run of this project. The assessment is written by an AI from what was recorded."),
        S.runs.length ? h("ul.runlist", S.runs.map((r) => item(r, false))) : h("p.muted", "No finished runs yet."),
        ...(S.batches.length ? [h("h3.sub.gap", "Repeat batches"), h("ul.runlist", S.batches.map((b) => item(b, true)))] : []));
    }
    function renderFrame() {
      frameBox.replaceChildren();
      if (!S.sel) {
        const running = WZ.state && WZ.state.running;
        frameBox.append(h("div.empty.big", h("h3", running ? "The report appears here when the run ends" : "No report yet"),
          h("p.muted", running ? "Stop the run or let it finish. The report is written after the debrief." : "Run the scenario on the Run page and its report will show here."),
          running ? null : h("a.btn.primary", { href: "#/run" }, "Go to Run")));
        return;
      }
      const url = "/runs/" + encodeURIComponent(S.sel) + "/report.html";
      frameBox.append(
        h("div.rbar", h("span.muted", S.sel), h("a.link", { href: url, target: "_blank", rel: "noopener" }, "Open in a new tab")),
        h("iframe.rframe", { src: url, title: "Run report", sandbox: "", referrerpolicy: "no-referrer" }));
    }

    WZ.on("project", S.l1 = () => { if (!S.dead) { S.sel = null; load(); } });
    WZ.on("state", S.l2 = (s) => {
      if (S.dead) return;
      const key = (s.running ? "r" : "s") + (s.phase || "") + (s.has_report ? "1" : "0") + (s.run_id || "");
      if (key !== S.key) { const first = S.key === undefined; S.key = key; if (!first) load(); }
    });
    root._reportPage = S;
    load();
  }

  function unmount() {
    const S = document.getElementById("view")._reportPage;
    if (S) S.dead = true;
  }

  WZ.registerPage("report", { title: "Report", mount, unmount });
})();
