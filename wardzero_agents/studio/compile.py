"""Project -> staged scenario folder that `hhumans run` can load."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from ..config.loader import ConfigError
from ..config.presets import load_scenario_presets, preset_to_scenario
from ..robots.profile import G1_NAME, RobotProfile, get_profile, load_profile
from .characters import memory_key
from .project import Project


def robot_profile_for(name: str, data: Optional[Path] = None) -> RobotProfile:
    if name == G1_NAME or data is None:
        return get_profile(name)
    return load_profile(name, data)


def compile_project(project: Project, root: "str | Path", dest: "str | Path | None" = None,
                    data: Optional[Path] = None) -> Path:
    """Write the staged scenario for a project and return the path of its scenario.yaml.

    Built on preset_to_scenario: the user's characters replace the preset's, each with a memory key of its
    own (studio:<project>:<character>), edited role and mood prompts go into the persona text as they are,
    the timeline keeps the beats whose target is still in the cast, and the scenario is a campaign."""
    root = Path(root)
    presets = load_scenario_presets(root)
    preset = presets.get(project.scenario_id)
    if preset is None:
        raise ConfigError(f"unknown scenario preset '{project.scenario_id}'")
    cast_ids = {c.id for c in project.characters}
    preset = preset.model_copy(update={
        "timeline": [b for b in preset.timeline if b.target is None or b.target in cast_ids]})
    profile = robot_profile_for(project.robot_profile, data)
    robot = {"profile": profile.name, "ns": profile.ns, "reach": list(profile.reach), "radius": profile.radius,
             "max_speed": profile.max_speed, "enabled": profile.controller != "none"}
    if project.robot_start is not None:
        robot.update(project.robot_start.model_dump())
    dest = Path(dest) if dest is not None else Path(tempfile.mkdtemp(prefix=f"studio_{project.id}_"))
    if dest.exists():
        shutil.rmtree(dest)
    extra = {"seed": project.seed, "duration_min": project.duration_min, "budget_usd": project.budget_usd,
             "static_patients": project.static_patients, "campaign": True,
             "word_of_mouth_p": project.word_of_mouth_p}
    if project.forbidden_areas:
        extra["forbidden_areas"] = [a.model_dump(mode="json") for a in project.forbidden_areas]
    from .flows import for_compile
    flow = for_compile(root, data, project, presets[project.scenario_id])
    loaded = preset_to_scenario(
        preset, root, dest, characters=project.characters, scenario_extra=extra, robot_extra=robot,
        cast_extra={c.id: {"memory_key": memory_key(project.id, c.id)} for c in project.characters}, flow=flow)
    return Path(loaded.scenario_path)
