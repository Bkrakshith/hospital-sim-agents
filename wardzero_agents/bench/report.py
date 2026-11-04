"""The after-run report: report.json (every number computed by code) and a self-contained report.html.

An AI narrative is added on top. It may explain and suggest but may not bring numbers of its own:
check_numbers drops every sentence that holds a number the report data does not hold."""
from __future__ import annotations

import asyncio
import html
import json
import re
from pathlib import Path
from typing import Optional

from ..campaign.debrief import ROBOT_NAMES, clock, run_people
from ..campaign.experiences import detect
from ..llm.gateway import Gateway, LLMRequest
from .criteria import evaluate
from .metrics import RunData, compute_all

TEMPLATE = Path(__file__).with_name("templates") / "report.html"
MAX_EXCERPTS = 30
MAX_TIMELINE = 60
STAGE_NAMES = {0: "uses it as normal", 1: "double-checks", 2: "bypasses it", 3: "asks for a human",
               4: "refuses it"}

REPORT_SYSTEM = (
    "You write the assessment that follows a test of a robot against simulated people in a hospital. "
    "You are given the run's report data as JSON, in which every figure was computed by code, and up to "
    "30 numbered excerpts from the run, each with its time.\n"
    "Rules:\n"
    "1. Judge how the people behaved, and how the robot treated them, against the success criteria in the "
    "data. Say which criteria passed and which failed, and what the people did, rated and said in "
    "response to the robot.\n"
    "2. Cite every claim about an event with its time in square brackets as [mm:ss], using only times "
    "that appear in the excerpts or the data.\n"
    "3. Write no figure that is not in the data: no counts, durations, percentages, scores or dates of "
    "your own, and do not calculate new ones. Use a figure only as the data gives it, or describe it in "
    "words. Any sentence holding a figure that is not in the data is deleted.\n"
    "4. Give concrete suggestions for improving the robot's behaviour, each tied to a failed criterion, a "
    "low rating or a moment in the excerpts: say what to change and why. No general advice.\n"
    "5. Plain, direct English. Do not name any other scenario, product or robot.\n"
    "Return JSON with narrative (a few short paragraphs separated by blank lines) and suggestions (a list "
    "of strings).")
REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"narrative": {"type": "string"}, "suggestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["narrative", "suggestions"]}

# ---- check_numbers ------------------------------------------------------------------------------
_STAMP = re.compile(r"\[\d{1,3}:\d{2}\]")
_NUM = re.compile(r"(?<![A-Za-z_\d.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def _numbers(text: str) -> list[tuple[float, int]]:
    """(value, decimals shown) of each number in text, not counting [mm:ss] stamps or digits that are
    part of a name such as P1."""
    out = []
    for m in _NUM.finditer(_STAMP.sub(" ", text)):
        whole = m.group(1).replace(",", "")
        frac = (m.group(2) or "")[1:]
        out.append((float(whole + ("." + frac if frac else "")), len(frac)))
    return out


def _collect(obj, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(abs(float(obj)))
    elif isinstance(obj, str):
        acc.extend(v for v, _ in _numbers(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect(k, acc)
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def _known(known: list[float], x: float, decimals: int) -> bool:
    for v in known:
        if abs(round(v, decimals) - x) < 1e-9:
            return True
        if v <= 1.0 and abs(round(v * 100, decimals) - x) < 1e-9:  # a share written as a percentage
            return True
    return False


def check_numbers(narrative: str, report_json: dict) -> str:
    """The narrative without the sentences that hold a number not found in report_json (compared after
    rounding to the precision the sentence shows; a share may be written as a percentage). Numbers
    inside [mm:ss] stamps and digits that are part of a name (P1) are exempt."""
    known: list[float] = []
    _collect(report_json, known)
    paragraphs = []
    for para in narrative.split("\n"):
        parts = re.split(r"(?<=[.!?])(\s+)", para)
        kept = []
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if all(_known(known, v, d) for v, d in _numbers(sentence)):
                kept.append((sentence, parts[i + 1] if i + 1 < len(parts) else ""))
        text = "".join(s + sep for s, sep in kept).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


# ---- facts ---------------------------------------------------------------------------------------
def _r(v):
    return round(v, 2) if isinstance(v, float) else v


def _moments(d: RunData, exps, names: dict) -> list[dict]:
    def nm(p):
        return names.get(p, str(p))
    out = []

    def add(t, kind, text, weight):
        if isinstance(t, (int, float)):
            out.append({"t": round(float(t), 1), "clock": clock(t), "kind": kind, "text": text, "weight": weight})
    for c in d.cards:
        if c.get("ok", True):
            tgt = f" ({nm(c['target'])})" if c.get("target") else ""
            add(c.get("t"), "event", f"Scenario event: {c.get('card')}{tgt}", 3)
    for a in d.audio:
        if a.get("source_kind") == "alarm":
            add(a.get("t"), "event", "The fire alarm sounds", 3)
            break
    for w in d.world:
        if w.get("kind") == "contact":
            add(w.get("t"), "safety", f"The robot touched {nm(w.get('person'))}", 4)
    acts = {r.get("id"): r for r in d.robot_tasks if r.get("kind") == "action"}
    for r in d.robot_tasks:
        if r.get("kind") == "result":
            act = acts.get(r.get("id"), {})
            what = r.get("action") or act.get("action") or "an action"
            ok = bool(r.get("ok"))
            why = f" ({r['reason']})" if r.get("reason") and not ok else ""
            add(r.get("t"), "robot", f"Robot {what}: {'done' if ok else 'failed'}{why}", 2 if ok else 3)
    for s in d.says:
        if not s.get("text"):
            continue
        if s.get("person") in ROBOT_NAMES:
            add(s.get("t"), "robot says", f"Robot to {nm(s.get('addressee')) if s.get('addressee') else 'everyone'}: "
                f"\"{s['text']}\"", 1)
        elif s.get("addressee") in ROBOT_NAMES:
            add(s.get("t"), "asks robot", f"{nm(s.get('person'))}: \"{s['text']}\"", 1)
    for e in exps:
        text = {"failure": "went wrong for", "success": "went well for", "repair_immediate": "apologised to",
                "repair_at_request": "apologised, when asked again, to"}.get(e.kind)
        if text:
            add(e.t, "outcome", f"Robot {text} {nm(e.person)}", 3 if e.kind == "failure" else 2)
    out.sort(key=lambda m: (m["t"], -m["weight"]))
    return out


def _excerpts(moments: list[dict], limit: int = MAX_EXCERPTS) -> list[dict]:
    """The `limit` most telling moments (heavier first, earlier first), shown in time order."""
    best = sorted(range(len(moments)), key=lambda i: (-moments[i]["weight"], moments[i]["t"]))[:limit]
    return [moments[i] for i in sorted(best)]


def _tokens(d: RunData) -> dict:
    rows = d.tokens
    return {"input": sum(int(r.get("input_tokens") or 0) for r in rows),
            "output": sum(int(r.get("output_tokens") or 0) for r in rows),
            "calls": len(rows), "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 4)}


def build_facts(run_dir: Path, preset, narrative: Optional[dict] = None) -> dict:
    """report.json without the narrative: every number comes from the run folder."""
    d = RunData.load(run_dir)
    meta = d.meta
    res = evaluate(preset, d) if preset is not None else evaluate(_NoPreset, d)
    chars = {c.id: c.name for c in getattr(preset, "characters", [])}
    people = run_people(meta, d)
    names = {p: chars.get(p, p) for p in people}
    exps = detect(run_dir, preset)
    moments = _moments(d, exps, names)
    valid = not _fairness_rows(run_dir)
    if not valid:
        verdict = {"label": "INVALID", "passed": False}
    elif not res.rows:
        verdict = {"label": "NO CRITERIA", "passed": None}
    else:
        verdict = {"label": "PASS" if res.passed else "FAIL", "passed": res.passed}
    ratings = []
    for r in d.debrief:
        p = r.get("person")
        if isinstance(p, str):
            ratings.append({"person": p, "name": names.get(p, p), "role": d.role(p),
                            **{k: r.get(k) for k in ("likeability", "safety", "competence", "trust")},
                            "comment": str(r.get("comment") or "")})
    trust = [{"person": r.get("person"), "name": names.get(r.get("person"), r.get("person")),
              **{k: _r(r.get(k)) for k in ("before", "after", "stage_before", "stage_after")}} for r in d.trust]
    return {
        "run": {"run_id": meta.get("run_id") or Path(run_dir).name, "scenario_id": meta.get("scenario_id"),
                "title": getattr(preset, "title", None) or meta.get("scenario_id") or Path(run_dir).name,
                "seed": meta.get("seed"), "run_index": meta.get("run_index"),
                "word_of_mouth": bool(meta.get("word_of_mouth")), "duration_s": _r(d.end_t),
                "models": meta.get("models") or {}, "started_at": meta.get("started_at")},
        "valid": valid, "fairness_violations": len(_fairness_rows(run_dir)),
        "verdict": verdict, "score": _r(res.score),
        "criteria": [{**row, "value": _r(row["value"]), "threshold": _r(row["threshold"])} for row in res.rows],
        "metrics": {k: _r(v) for k, v in compute_all(d).items() if v is not None},
        "moments": moments, "ratings": ratings, "trust": trust,
        "tokens": _tokens(d), "narrative": "", "suggestions": [],
    }


class _NoPreset:
    criteria: list = []


def _fairness_rows(run_dir: Path) -> list:
    try:
        return [ln for ln in (Path(run_dir) / "fairness.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


# ---- narrative -----------------------------------------------------------------------------------
async def write_narrative(facts: dict, gateway: Optional[Gateway]) -> tuple[str, list[str]]:
    """One Sonnet call over the report data and up to 30 excerpts; the answer is number-checked."""
    if gateway is None:
        return "", []
    slim = {k: v for k, v in facts.items() if k not in ("narrative", "suggestions", "moments")}
    ex = _excerpts(facts["moments"])
    user = "EXCERPTS:\n" + ("\n".join(f"[{m['clock']}] ({m['kind']}) {m['text']}" for m in ex) or "(none)")
    res = await gateway.complete(LLMRequest(
        role="report", person="report", system_fixed=REPORT_SYSTEM,
        persona_card="REPORT DATA (JSON):\n" + json.dumps(slim, sort_keys=True), user=user, schema=REPORT_SCHEMA))
    if not res.ok or not isinstance(res.data, dict):
        return "", []
    narrative = check_numbers(str(res.data.get("narrative") or ""), facts)
    sugg = [check_numbers(str(s), facts) for s in (res.data.get("suggestions") or []) if isinstance(s, str)]
    return narrative, [s for s in sugg if s]


# ---- html ----------------------------------------------------------------------------------------
def _e(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _fmt(v) -> str:
    if v is None:
        return "not measured"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _table(head: list[str], rows: list[list[str]], num: tuple[int, ...] = ()) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    def cell(i, c):
        return ('<td class="n">' if i in num else "<td>") + c + "</td>"
    body = "".join("<tr>" + "".join(cell(i, c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(facts: dict) -> str:
    run, verdict = facts["run"], facts["verdict"]
    if not facts["valid"]:
        banner = ('<div class="banner invalid" role="alert">INVALID: fairness audit failed<small>The robot read '
                  "topics that only the simulation may read, so this run says nothing about the robot. "
                  f"{_e(facts['fairness_violations'])} violation(s) were recorded.</small></div>")
    elif verdict["passed"] is None:
        banner = '<div class="banner none">No verdict<small>This run has no success criteria.</small></div>'
    else:
        ok = verdict["passed"]
        banner = (f'<div class="banner {"pass" if ok else "fail"}">{verdict["label"]}: score {_fmt(facts["score"])} '
                  f'out of 100<small>{"Every must-pass criterion was met." if ok else "At least one must-pass criterion was missed."}'
                  "</small></div>")
    crit_rows = []
    for c in facts["criteria"]:
        tag = ('<span class="tag none">not measured</span>' if c["passed"] is None else
               '<span class="tag pass">pass</span>' if c["passed"] else '<span class="tag fail">fail</span>')
        crit_rows.append([_e(c["label"]), _e(_fmt(c["value"])), _e(f"{c['op']} {_fmt(c['threshold'])}"),
                          _e(_fmt(c["weight"])), tag])
    criteria = (_table(["Criterion", "Measured", "Needed", "Weight", "Result"], crit_rows, num=(1, 2, 3))
                if crit_rows else '<p class="note">This run has no success criteria.</p>')
    moments = facts["moments"][:MAX_TIMELINE]
    timeline = ('<ol class="tl">' + "".join(
        f'<li><time>{_e(m["clock"])}</time><div><span class="k">{_e(m["kind"])}</span>{_e(m["text"])}</div></li>'
        for m in moments) + "</ol>") if moments else '<p class="note">Nothing notable was recorded.</p>'
    if len(facts["moments"]) > MAX_TIMELINE:
        timeline += f'<p class="note">The first {MAX_TIMELINE} of {len(facts["moments"])} moments are shown.</p>'
    if facts["ratings"]:
        rrows = [[_e(r["name"]), _e(r["likeability"]), _e(r["safety"]), _e(r["competence"]), _e(r["trust"]),
                  f"<blockquote>{_e(r['comment'])}</blockquote>"] for r in facts["ratings"]]
        ratings = _table(["Person", "Liked", "Safe", "Competent", "Trust", "Comment"], rrows, num=(1, 2, 3, 4)) \
            + '<p class="note">Ratings are 1 (poor) to 5 (good), answered in character after the run.</p>'
    else:
        ratings = '<p class="note">No debrief was recorded for this run.</p>'
    if facts["trust"]:
        trows = [[_e(t["name"]), _e(_fmt(t["before"])), _e(_fmt(t["after"])),
                  _e(f"{t['stage_before']} ({STAGE_NAMES.get(t['stage_before'], '')}) to "
                     f"{t['stage_after']} ({STAGE_NAMES.get(t['stage_after'], '')})")] for t in facts["trust"]]
        trust = _table(["Person", "Trust before", "Trust after", "Stage"], trows, num=(1, 2))
    else:
        trust = '<p class="note">No trust update was recorded for this run.</p>'
    if facts["narrative"] or facts["suggestions"]:
        paras = "".join(f"<p>{_e(p)}</p>" for p in facts["narrative"].split("\n") if p.strip())
        sugg = ("<h3>Suggestions</h3><ul>" + "".join(f"<li>{_e(s)}</li>" for s in facts["suggestions"]) + "</ul>"
                if facts["suggestions"] else "")
        narrative = (f'<div class="narr">{paras}{sugg}</div><p class="note">Written by an AI from the data above. '
                     "Sentences with figures that are not in the data were removed.</p>")
    else:
        narrative = '<p class="note">No AI assessment was written for this run.</p>'
    models = run.get("models") or {}
    tok = facts["tokens"]
    facts_rows = [
        ("Run", run["run_id"]), ("Seed", run["seed"]), ("Run index", run["run_index"]),
        ("Word of mouth", "yes" if run["word_of_mouth"] else "no"),
        ("Duration", f"{clock(run['duration_s'] or 0)} (mm:ss)"),
        ("Models", ", ".join(f"{k}: {v}" for k, v in models.items() if k in ("decide", "summarise", "report", "robot"))),
        ("Tokens", f"{tok['input']} in, {tok['output']} out over {tok['calls']} calls"),
        ("Cost", f"${tok['cost_usd']:.2f}")]
    facts_html = "<dl>" + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in facts_rows) + "</dl>"
    sub = f"Scenario {_e(run['scenario_id'])}; run {_e(run['run_id'])}"
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, val in (("title", _e(run["title"])), ("subtitle", sub), ("banner", banner), ("criteria", criteria),
                     ("timeline", timeline), ("ratings", ratings), ("trust", trust), ("narrative", narrative),
                     ("facts", facts_html)):
        page = page.replace("{{" + key + "}}", val)
    return page


# ---- entry points --------------------------------------------------------------------------------
async def build_report_async(run_dir: "str | Path", preset, narrative_gateway: Optional[Gateway]) -> Path:
    """Write report.json and report.html into the run folder; returns the html path."""
    run_dir = Path(run_dir)
    facts = build_facts(run_dir, preset)
    facts["narrative"], facts["suggestions"] = await write_narrative(facts, narrative_gateway)
    facts["tokens"] = _tokens(RunData.load(run_dir))  # now with the narrative call
    (run_dir / "report.json").write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
    out = run_dir / "report.html"
    out.write_text(render_html(facts), encoding="utf-8")
    return out


def build_report(run_dir: "str | Path", preset, narrative_gateway: Optional[Gateway]) -> Path:
    return asyncio.run(build_report_async(run_dir, preset, narrative_gateway))


async def post_run(run_dir: "str | Path", *, report: bool = True, narrative: bool = True,
                   force_debrief: bool = False) -> Optional[Path]:
    """Debrief a finished run, then write its report (unless report is False). Returns the report path."""
    from ..campaign.debrief import debrief, make_post_gateway
    from ..config.presets import preset_for_run
    run_dir = Path(run_dir)
    gateway = make_post_gateway(run_dir)
    preset = preset_for_run(RunData.load(run_dir).meta)
    await debrief(run_dir, gateway, preset=preset, force=force_debrief)
    if not report:
        return None
    return await build_report_async(run_dir, preset, gateway if narrative else None)
