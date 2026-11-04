"""Studio projects: a scenario preset plus the characters the user placed, stored as project.yaml."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from ..config.presets import CharacterSpec, RobotStart, ScenarioPreset

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class ProjectError(Exception):
    """A bad project request (the API answers 400) or a missing one (404, see NotFound)."""


class NotFound(ProjectError):
    pass


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str = Field(min_length=1, max_length=80)
    seed: int = 7
    scenario_id: str
    characters: list[CharacterSpec] = Field(default_factory=list)
    static_patients: bool = True
    robot_profile: str = "g1_claude"
    robot_start: Optional[RobotStart] = None
    word_of_mouth_p: float = Field(default=0.5, ge=0, le=1)
    duration_min: float = Field(default=10, gt=0, le=60)
    budget_usd: float = Field(default=5.0, gt=0, le=1000)
    repeat: int = Field(default=1, ge=1, le=20)
    carry_memory: bool = False
    baseline: bool = False
    # the Run page's Simulator group: Isaac Sim or the quick 2D stub; Isaac with a window or headless (with the live
    # preview); recording (two MP4s and a rosbag) and its resolution
    sim: Literal["isaac", "stub"] = "stub"
    headed: StrictBool = False
    record: StrictBool = False
    record_res: Literal["1280x720", "1920x1080"] = "1280x720"


def valid_id(pid: str) -> bool:
    return bool(_ID.match(pid or ""))


def defaults_from_preset(preset: ScenarioPreset) -> dict:
    """Project fields a fresh project takes from its scenario preset (prompts stay None: use the preset)."""
    return {"characters": [c.model_dump(mode="json") for c in preset.characters],
            "robot_start": preset.robot_start.model_dump(mode="json"), "duration_min": preset.duration_min}


class ProjectStore:
    def __init__(self, data: Path):
        self.root = Path(data) / "studio" / "projects"

    def dir(self, pid: str) -> Path:
        if not valid_id(pid):
            raise NotFound(f"no project '{pid}'")
        return self.root / pid

    def new_id(self) -> str:
        return "p" + uuid.uuid4().hex[:8]

    def list(self) -> list[Project]:
        out = []
        for f in sorted(self.root.glob("*/project.yaml")):
            try:
                out.append(self.get(f.parent.name))
            except (ProjectError, OSError):
                continue
        return out

    def get(self, pid: str) -> Project:
        f = self.dir(pid) / "project.yaml"
        try:
            data = yaml.safe_load(f.read_text())
        except FileNotFoundError:
            raise NotFound(f"no project '{pid}'") from None
        except (OSError, yaml.YAMLError):
            raise ProjectError(f"project '{pid}' cannot be read") from None
        try:
            return Project.model_validate(data)
        except ValidationError as e:
            raise ProjectError(f"project '{pid}' is not valid: {e.errors()[0]['msg']}") from None

    def save(self, project: Project) -> None:
        d = self.dir(project.id)
        d.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(project.model_dump(mode="json"), sort_keys=False)
        with tempfile.NamedTemporaryFile("w", dir=d, prefix=".project.", suffix=".tmp", delete=False) as f:
            f.write(text)
            tmp = f.name
        os.replace(tmp, d / "project.yaml")

    def delete(self, pid: str) -> None:
        d = self.dir(pid)
        if not (d / "project.yaml").is_file():
            raise NotFound(f"no project '{pid}'")
        shutil.rmtree(d)


def validation_message(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'body'}: {err['msg']}" for err in e.errors())
