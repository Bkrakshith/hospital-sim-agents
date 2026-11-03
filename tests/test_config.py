import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from wardzero_agents.config.loader import ConfigError, load_scenario
from wardzero_agents.config.models import ModelRoles

AGENTS_ROOT = Path(__file__).resolve().parents[1]
CFG = AGENTS_ROOT / "config"
HG1_ROOT = Path(os.environ.get("HG1_ROOT", AGENTS_ROOT.parent))
V1 = CFG / "scenarios" / "waiting_area_v1.yaml"


def write_patched(tmp_path, scenario_path, patch):
    """Copy the config tree under tmp_path and write a patched copy of the scenario.

    patch maps dotted paths such as "cast.2.persona" to new values. Relative paths in the
    scenario keep resolving because the whole config tree is copied.
    """
    cfg = tmp_path / "config"
    shutil.copytree(CFG, cfg)
    rel = Path(scenario_path).resolve().relative_to(CFG)
    target = cfg / rel
    data = yaml.safe_load(target.read_text())
    for dotted, value in patch.items():
        keys = dotted.split(".")
        node = data
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    return target


def inst(loaded, person_id):
    return next(i for i in loaded.instances if i.person_id == person_id)


def test_loads_v1():
    ls = load_scenario(V1)
    assert [i.person_id for i in ls.instances] == ["P1", "P2", "P3", "P4", "P5"]
    assert len(ls.area.seats) == 12
    assert len(ls.signs) == 6
    assert ls.root == CFG


def test_sampling_is_deterministic():
    assert load_scenario(V1).instances == load_scenario(V1).instances


def test_ranges_respected():
    p2 = inst(load_scenario(V1), "P2")
    assert 68 <= p2.age <= 82
    assert 30 <= p2.cough_per_hour <= 60
    assert p2.memory_key == "coughing_older_adult#7001"
    assert p2.seat == "C1"


def test_companion_link_and_names():
    ls = load_scenario(V1)
    assert inst(ls, "P3").companion_of == "P2"
    assert all(i.display_name.strip() for i in ls.instances)


def test_default_models_claude():
    assert load_scenario(V1).scenario.models == ModelRoles(
        decide="claude-opus-5-5", summarise="claude-haiku-5-5", vision="claude-opus-5-5")


@pytest.mark.parametrize("patch,msg", [
    ({"cast.2.persona": "nobody"}, "cast[2].persona: unknown persona 'nobody'"),
    ({"cast.1.seat": "B2"}, "cast[1].seat: seat 'B2' already taken by P1"),
    ({"cast.0.seat": "Z9"}, "cast[0].seat: unknown seat 'Z9'"),
    ({"cast.2.companion_of": "P9"}, "cast[2].companion_of: unknown person 'P9'"),
    ({"provider": "openai"}, "models: OpenAI needs model names for decide, summarise and vision"),
    ({"duration_min": 90}, "duration_min"),
])
def test_config_errors(tmp_path, patch, msg):
    with pytest.raises(ConfigError, match=re.escape(msg)):
        load_scenario(write_patched(tmp_path, V1, patch))


def test_persona_sources_exist():
    for p in load_scenario(V1).personas.values():
        assert p.sources
        assert all((HG1_ROOT / "research_notes" / s).exists() for s in p.sources)


def test_missing_storyboard_or_deck_is_an_error(tmp_path):
    for sub_dir, name, field in (("storyboards", "waiting_area_day.yaml", "storyboard"), ("cards", "starter.yaml", "deck")):
        p = write_patched(tmp_path / field, V1, {})
        (p.parent.parent / sub_dir / name).unlink()
        with pytest.raises(ConfigError, match=rf"^{field}: file not found: "):
            load_scenario(p)
