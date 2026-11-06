"""The robot process's own config: only what a robot needs, never the scenario with its people.

`hhumans run` writes ``robot_config.json`` into the run folder and starts the robot children with
``--robot-config`` pointing at it. It holds the robot's domain and namespace, its profile, its
spending budget, the places it may be asked about, the scenario's robot-facing brief, and the config
folder to read the site map from. It carries no persona, cast or storyboard data, and the robot
process loads only the site, the area and the signs from that folder."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROBOT_CONFIG = "robot_config.json"


@dataclass
class RobotView:
    root: Path
    site_name: str
    area: Optional[str]
    robot: Any  # config.models.RobotConfig
    known_places: list
    models: Any  # config.models.ModelRoles
    provider: str
    budget_usd: float
    brief: Optional[dict]
    domain: Optional[int]
    site: Any = None
    waiting_area: Any = None
    signs: list = field(default_factory=list)
    sim: str = "stub"  # "stub" (Quick 2D) or "isaac": the built-in G1 walks only where its body can (profile_for_sim)


def write_robot_config(run_dir: "str | Path", loaded, *, domain: Optional[int], budget_usd: float,
                       sim: str = "stub") -> Path:
    from ..site.areas import task_brief
    sc = loaded.scenario
    data = {"root": str(loaded.root), "site": sc.site, "area": sc.area, "sim": sim,
            "robot": sc.robot.model_dump(mode="json"), "known_places": list(sc.known_places),
            "provider": sc.provider, "models": sc.models.model_dump(mode="json") if sc.models else None,
            "budget_usd": float(budget_usd), "robot_brief": task_brief(sc.robot_brief, getattr(sc, "forbidden_areas", None)),
            "domain": domain}
    dest = Path(run_dir) / ROBOT_CONFIG
    with tempfile.NamedTemporaryFile("w", dir=dest.parent, prefix=".robot_config.", suffix=".tmp", delete=False,
                                     encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, dest)
    return dest


def load_robot_view(path: "str | Path") -> RobotView:
    from ..config.loader import default_models, load_robot_view as load_site_view
    from ..config.models import ModelRoles, RobotConfig
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    site, area, signs = load_site_view(data["root"], data.get("site") or "hospital", data.get("area"))
    models = ModelRoles(**data["models"]) if data.get("models") else default_models(data["provider"])
    return RobotView(root=Path(data["root"]), site_name=data.get("site") or "hospital", area=data.get("area"),
                     robot=RobotConfig(**data["robot"]), known_places=list(data.get("known_places") or []),
                     models=models, provider=data["provider"], budget_usd=float(data["budget_usd"]),
                     brief=data.get("robot_brief"), domain=data.get("domain"), site=site, waiting_area=area,
                     signs=signs, sim=data.get("sim") or "stub")
