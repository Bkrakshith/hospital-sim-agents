"""Role, mood and scenario presets (Studio content layer)."""
import json
import re
from pathlib import Path

import pytest

from wardzero_agents.bench.metric_names import METRIC_NAMES
from wardzero_agents.config.loader import LoadedScenario
from wardzero_agents.config.presets import (BASIC_INTENTS, ROLE_INTENTS, CharacterSpec, Criterion, MoodPreset,
                                            RolePreset, load_moods, load_roles, load_scenario_presets,
                                            preset_to_scenario)
from wardzero_agents.config.loader import ConfigError
from wardzero_agents.site import load_site
from wardzero_agents.world.stub_world import StubWorld

ROOT = Path(__file__).resolve().parents[1] / "config"
IDS = {"reception_guide", "scrub_nurse", "meal_round", "ward_fire", "evacuation_guide", "specimen_run",
       "visiting_rush", "wandering_patient", "night_call_bell", "ae_waiting"}
PERSONA_ROLE = {"ward_nurse": "ward_nurse", "receptionist": "receptionist", "patient": "bed_patient", "ae_patient": "ae_patient",
                "visitor": "ward_visitor", "surgeon": "surgeon", "porter": "porter", "custom": "custom",
                "wheelchair_patient": "patient"}
TRUST = {"ward_nurse": 0.65, "receptionist": 0.65, "patient": 0.75, "ae_patient": 0.75, "visitor": 0.75, "surgeon": 0.62,
         "porter": 0.65, "custom": 0.70, "wheelchair_patient": 0.75}


@pytest.fixture(scope="module")
def presets():
    return load_scenario_presets(ROOT)


@pytest.fixture(scope="module")
def site():
    return load_site(ROOT)


def test_ten_scenarios_load(presets):
    assert set(presets) == IDS
    assert all(p.id == k for k, p in presets.items())


def test_every_preset_has_criteria_and_ladder(presets):
    for p in presets.values():
        assert len(p.criteria) >= 3, p.id
        assert sorted(p.ladder) == [0, 1, 2, 3, 4], p.id
        assert all(t.strip() for t in p.ladder.values())
        assert len(re.findall(r"[.!?](?:\s|$)", p.prompt)) in (2, 3), p.id
        assert p.sources and p.duration_min > 0


def test_criteria_metrics_known(presets):
    for p in presets.values():
        for c in p.criteria:
            assert c.metric in METRIC_NAMES, (p.id, c.metric)
            assert c.op in (">=", "<=", "==")
            assert c.weight > 0
    assert len(set(METRIC_NAMES)) == 30
    with pytest.raises(ValueError):
        Criterion(id="x", label="x", metric="no_such_metric", op=">=", threshold=1, weight=1)


def test_characters_placeable(presets, site):
    for p in presets.values():
        ids = [c.id for c in p.characters]
        assert len(ids) == len(set(ids)), p.id
        for c in p.characters:
            if c.bed:
                b = site.beds[c.bed]
                assert (c.x, c.y) == (b.x, b.y), (p.id, c.id)
            elif any(abs(q.x - c.x) < 0.05 and abs(q.y - c.y) < 0.05 for q in site.seats.values()):
                pass  # on a seat: the chair itself is blocked in the grid
            else:
                assert site.check_place(c.x, c.y) is None, (p.id, c.id, site.check_place(c.x, c.y))
            assert site.room_at(c.x, c.y) == c.room, (p.id, c.id, site.room_at(c.x, c.y))
        rs = p.robot_start
        assert site.check_place(rs.x, rs.y) is None, p.id


def test_presets_compile_and_start(presets, tmp_path):
    for p in presets.values():
        loaded = preset_to_scenario(p, ROOT, tmp_path / p.id)
        assert isinstance(loaded, LoadedScenario)
        assert loaded.scenario.robot_brief == p.robot_brief
        assert {i.person_id for i in loaded.instances} == {c.id for c in p.characters}
        w = StubWorld(loaded, 7)
        assert len(w.objects.items) == sum(1 for o in p.objects if o["kind"] in
                                           ("tray", "specimen_box", "instrument", "extinguisher"))
        for step in range(1, 21):
            w.step(step * 0.1)


def test_focus_rooms_match_navmesh_hints(presets, site):
    hints = json.loads((ROOT / "site" / "navmesh_hints.json").read_text())
    for p in presets.values():
        assert p.focus_rooms == hints[p.id]["focus_rooms"], p.id
        assert all(r in site.rooms for r in p.focus_rooms)
        assert all(site.resolve(k) is not None for k in p.known_places), p.id


def test_role_prompts_cite_sources():
    roles = load_roles(ROOT)
    assert set(roles) == set(TRUST)
    for r in roles.values():
        assert isinstance(r, RolePreset)
        if r.id == "custom":
            continue
        assert 4 <= len([ln for ln in r.prompt.splitlines() if ln.strip()]) <= 8, r.id
        assert all("[" in ln and ln.rstrip().endswith("]") for ln in r.prompt.splitlines() if ln.strip()), r.id
        assert r.sources, r.id


def test_role_ids_trust_and_intents():
    roles = load_roles(ROOT)
    for rid, r in roles.items():
        assert r.persona_role == PERSONA_ROLE[rid]
        assert r.trust_start == TRUST[rid]
        assert set(r.intents) == set(ROLE_INTENTS[r.persona_role])
        assert 0 <= r.self_confidence <= 1 and r.default_rooms or rid == "custom"


def test_mood_prompts_and_nudges():
    moods = load_moods(ROOT)
    assert set(moods) == {"sad", "happy", "anxious", "stressed", "excited", "relieved", "custom"}
    for m in moods.values():
        assert isinstance(m, MoodPreset)
        assert 0.8 <= m.walk_speed_x <= 1.2 and 0.8 <= m.patience_x <= 1.2
        assert abs(m.anxiety_shift) <= 0.3 and abs(m.sociability_shift) <= 0.3
        if m.id == "custom":
            continue
        assert 3 <= len([ln for ln in m.prompt.splitlines() if ln.strip()]) <= 8, m.id
        assert all("[" in ln for ln in m.prompt.splitlines() if ln.strip()), m.id
        assert m.sources, m.id
    assert moods["sad"].walk_speed_x < 1 < moods["happy"].walk_speed_x
    assert moods["anxious"].anxiety_shift > 0


def test_surgeon_role_intents_limited():
    roles = load_roles(ROOT)
    extra = {"request_instrument", "accept_item", "reject_item", "take_instrument_self"}
    allowed = (BASIC_INTENTS - {"walk_to", "sit_down", "stand_up", "leave", "approach", "sit_idle",
                                "use_phone", "read_sign"}) | extra
    assert set(roles["surgeon"].intents) <= allowed
    assert "walk_to" not in roles["surgeon"].intents


def test_cards_and_timeline_targets(presets):
    import yaml
    deck = {c["id"] for c in yaml.safe_load((ROOT / "cards" / "studio.yaml").read_text())["cards"]}
    for need in ("fire_start", "alarm", "call_bell_at", "patient_gets_up", "arrival", "instrument_request_batch"):
        assert need in deck
    for p in presets.values():
        ids = {c.id for c in p.characters}
        for b in p.timeline:
            assert b.card in deck, (p.id, b.card)
            assert b.target is None or b.target in ids, (p.id, b.id)


def test_character_roles_and_moods_exist(presets):
    roles, moods = load_roles(ROOT), load_moods(ROOT)
    for p in presets.values():
        for c in p.characters:
            assert isinstance(c, CharacterSpec)
            assert c.role in roles and c.mood in moods, (p.id, c.id)


def test_scrub_nurse_in_scenario_ids_only_here():
    # the surgeon scenario exists only as a preset file and in code lists
    assert (ROOT / "presets" / "scenarios" / "scrub_nurse.yaml").exists()


def test_bad_preset_rejected(tmp_path):
    import shutil
    cfg = tmp_path / "config"
    shutil.copytree(ROOT, cfg, ignore=shutil.ignore_patterns("hospital_occ.npz"))
    (cfg / "site" / "hospital_occ.npz").symlink_to(ROOT / "site" / "hospital_occ.npz")
    f = cfg / "presets" / "scenarios" / "reception_guide.yaml"
    f.write_text(f.read_text().replace("metric: contacts", "metric: nonsense_metric", 1))
    with pytest.raises((ConfigError, ValueError)):
        load_scenario_presets(cfg)


def _keys(text):
    """Source keys in square brackets, skipping the ones that start with 'inferred'."""
    out = []
    for grp in re.findall(r"\[([^\]]+)\]", text):
        for part in grp.split(";"):
            part = part.strip()
            if part.startswith("inferred"):
                continue
            part = re.sub(r"^(see|adapted from)\s+", "", part)
            part = part.split(",")[0]
            part = re.sub(r"\s+(section|queue)\b.*$", "", part).strip()
            if part:
                out.append(part)
    return out


def test_source_keys_appear_in_research_notes(presets):
    notes = ROOT.parents[1] / "research_notes"
    corpus = "\n".join(f.read_text() for d in ("persona_behaviour", "UK hospital visitor personas")
                       for f in (notes / d).glob("*.md"))
    texts = [r.prompt for r in load_roles(ROOT).values()] + [m.prompt for m in load_moods(ROOT).values()]
    for p in presets.values():
        texts += list(p.ladder.values())
    keys = {k for t in texts for k in _keys(t)}
    assert len(keys) > 20
    missing = sorted(k for k in keys if k not in corpus)
    assert not missing, missing


def test_ward_fire_ladder_is_its_own(presets):
    assert presets["ward_fire"].ladder != presets["evacuation_guide"].ladder
    assert "extinguisher" in presets["ward_fire"].ladder[2]


def test_specimen_run_has_beds_wheelchairs_and_a_crowded_lobby(presets):
    p = presets["specimen_run"]
    lobby = [c for c in p.characters if c.room == "Main_LiftLobby"]
    assert len(lobby) >= 3
    assert any("trolley bed" in (c.role_prompt or "") for c in lobby)
    assert sum("wheelchair" in (c.role_prompt or "") for c in p.characters) >= 2
    esc = [b for b in p.timeline if b.card == "escort_patient"]
    assert esc and esc[0].target == "TR1" and esc[0].params["patient"] == "W2"


def test_evacuation_blocks_a_route(presets, tmp_path):
    p = presets["evacuation_guide"]
    assert "blocked" in p.robot_brief["job"]
    beat = next(b for b in p.timeline if b.card == "route_blocked")
    loaded = preset_to_scenario(p, ROOT, tmp_path / "evac")
    from wardzero_agents.contracts import WorldEvent
    w = StubWorld(loaded, 7)
    x, y = beat.params["x"], beat.params["y"]
    before = w.router.route((x - 4, y), (x + 4, y))
    w.apply_world_event(WorldEvent(id="rb", t=0, kind="route_blocked", x=x, y=y))
    assert w.router._raw[w.site.grid.to_cell(x, y)[0]][w.site.grid.to_cell(x, y)[1]]
    after = w.router.route((x - 4, y), (x + 4, y))
    assert before is not None and after != before
    assert any(s.id == "route_blocked_1" for s in w.signs)


def test_ward_fire_scores_the_robots_own_extinguish(presets):
    p = presets["ward_fire"]
    out = next(c for c in p.criteria if c.id == "out")
    assert out.metric == "fire_out_by_robot_s" and out.weight == 4
    assert p.robot_brief.get("first_contact") is False
    # in its own field, so the staff duty line (built from the job) does not repeat it to the nurses
    assert "carry your own extinguisher" in p.robot_brief["equipment"] and "carry" not in p.robot_brief["job"]


def test_visiting_rush_scores_reaching_the_nurse_base(presets, site):
    p = presets["visiting_rush"]
    arrived = next(c for c in p.criteria if c.id == "arrived")
    assert (arrived.metric, arrived.op, arrived.threshold, arrived.weight) == \
        ("robot_reached_destination_s", "<=", 360, 4)
    b = p.robot_brief
    assert b["destination"] == "nurse_base_ward" and b.get("first_contact") is False
    assert "Nurse workstation 1" in b["job"] and "far end" not in b["job"]
    assert "comes first" in b["priority"]
    # the destination's room is kept in Isaac
    room = site.room_at(*site.resolve(b["destination"]))
    hints = json.loads((ROOT / "site" / "navmesh_hints.json").read_text())["visiting_rush"]
    assert room == "Ward_WS1" and room in hints["focus_rooms"] + hints["route_rooms"]
