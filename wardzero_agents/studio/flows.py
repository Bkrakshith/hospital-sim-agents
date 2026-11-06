"""A project's scenario flow: the preset's own, or one adapted to the prompt the user wrote (projects/<id>/flow.yaml).

``state`` says which flow the next run uses and whether it must be adapted first; ``ensure`` adapts it (one model call,
a few at most) when the prompt changed; ``for_compile`` is what the staged scenario gets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from ..config.presets import ScenarioPreset
from ..flows.generate import adapt_flow, prompt_sha
from ..flows.spec import Flow, FlowError, dump_flow, load_flow, preset_flow, prune_flow
from .project import Project

log = logging.getLogger("wardzero.studio.flows")
NOTES = "flow_notes.txt"


def flow_file(data: Path, pid: str) -> Path:
    return Path(data) / "studio" / "projects" / pid / "flow.yaml"


def custom_prompt(project: Project, preset: ScenarioPreset) -> Optional[str]:
    """The user's own prompt, or None when it is the preset's (or blank)."""
    p = (project.scenario_prompt or "").strip()
    if not p or " ".join(p.split()) == " ".join(preset.prompt.split()):
        return None
    return p


def _stored(data: Path, project: Project) -> Optional[Flow]:
    f = flow_file(data, project.id)
    if not f.exists():
        return None
    try:
        return load_flow(f)
    except FlowError:
        return None


def state(config_root: Path, data: Path, project: Project, preset: ScenarioPreset) -> dict:
    """{source: preset | adapted | needs_adapting, flow, prompt, preset_prompt, notes}. ``flow`` is the one the next
    run uses (the preset's while an adaptation is pending)."""
    base = preset_flow(config_root, preset.id)
    prompt = custom_prompt(project, preset)
    out = {"prompt": prompt or preset.prompt, "preset_prompt": preset.prompt, "notes": ""}
    if prompt is None:
        out.update(source="preset", flow=base)
        return out
    stored = _stored(data, project)
    if stored is not None and stored.prompt_sha == prompt_sha(prompt):
        notes = flow_file(data, project.id).with_name(NOTES)
        out.update(source="adapted", flow=stored, notes=notes.read_text() if notes.exists() else "")
        return out
    out.update(source="needs_adapting", flow=base)
    return out


def for_compile(config_root: Path, data: Optional[Path], project: Project, preset: ScenarioPreset) -> Optional[Flow]:
    """The flow to stage for a run: an adapted one when it matches the prompt, else the preset's."""
    if data is None:
        return preset_flow(config_root, preset.id)
    st = state(config_root, data, project, preset)
    return st["flow"]


async def ensure(config_root: Path, data: Path, project: Project, preset: ScenarioPreset, *, deck: dict, site,
                 gateway=None) -> dict:
    """Adapt the flow to the project's prompt when it needs it (and save it); then the state. FlowError when the model
    could not give a usable flow."""
    st = state(config_root, data, project, preset)
    if st["source"] != "needs_adapting":
        return st
    if gateway is None:
        from ..config.loader import default_models
        from ..llm.gateway import Budget, make_gateway
        gateway = make_gateway("claude", default_models("claude"), Budget(0.5), lambda req, res: None)
    base = prune_flow(st["flow"], [c.id for c in project.characters]) if st["flow"] is not None else None
    if base is None:
        raise FlowError(f"scenario '{preset.id}' has no flow to adapt")
    people = [{"id": c.id, "name": c.name, "role": c.role, "room": c.room} for c in project.characters]
    flow, notes = await adapt_flow(gateway, scenario_id=preset.id, base=base, preset_prompt=preset.prompt,
                                   prompt=st["prompt"], people=people, deck=deck, rooms=site.rooms,
                                   duration_min=project.duration_min)
    f = flow_file(data, project.id)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(dump_flow(flow))
    f.with_name(NOTES).write_text(notes)
    log.info("project %s: flow adapted to the user's prompt (%d top-level steps)", project.id, len(flow.steps))
    return state(config_root, data, project, preset)


def view(st: dict) -> dict:
    """The state as JSON for the Studio: the flow as data and as a tree with every step pending."""
    from ..flows.engine import FlowEngine
    flow = st.get("flow")
    out = {k: st[k] for k in ("source", "prompt", "preset_prompt", "notes")}
    out["flow"] = yaml.safe_load(dump_flow(flow)) if flow is not None else None
    out["tree"] = FlowEngine(flow).tree() if flow is not None else None
    return out
