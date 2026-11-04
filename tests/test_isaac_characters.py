"""The Isaac People allow-list and the deterministic character choice (no Isaac needed)."""
import os
from pathlib import Path

import pytest
import yaml

from wardzero_agents.isaac_host import characters as ch

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "config" / "isaac" / "characters.yaml"
ASSETS = Path(os.path.expanduser("~/isaacsim_assets/Assets/Isaac/6.0"))
STAFF = ("staff", "ward_nurse", "surgeon", "porter", "receptionist")
PUBLIC = ("patient", "companion", "visitor", "bed_patient", "ward_visitor", "custom")


@pytest.fixture(scope="module")
def allow():
    return ch.load_allowlist(YAML)


def test_no_banned_clothing_anywhere_in_the_list(allow):
    for c in allow.characters:
        text = f"{c.id} {c.usd}".lower()
        assert not any(k in text for k in ch.BANNED_WORDS), c.id


def test_every_entry_is_tagged(allow):
    for c in allow.characters:
        assert c.sex in ("female", "male") and c.age in ch.AGE_BANDS and c.outfit in ch.OUTFITS
        assert c.usd.endswith(".usd") and not c.usd.startswith("/")
    assert len({c.id for c in allow.characters}) == len(allow.characters)


def test_the_list_covers_both_sexes_and_both_outfits(allow):
    for sex in ("female", "male"):
        assert any(c.sex == sex and c.outfit in ch.STAFF_OUTFITS for c in allow.characters)
        assert sum(c.sex == sex and c.outfit == "civilian" for c in allow.characters) >= 4
    assert any(c.age == "older" for c in allow.characters)


def test_dh_characters_hide_their_safety_accessories(allow):
    dh = [c for c in allow.characters if "DH_Characters" in c.usd]
    assert dh and all("Accessories" in c.hide for c in dh)


def test_the_loader_refuses_a_banned_character(tmp_path):
    bad = {"version": 1, "characters": [
        {"id": "male_adult_construction_01_new", "usd": "Isaac/People/Characters/x/x.usd", "sex": "male",
         "age": "adult", "age_years": 40, "outfit": "civilian", "hide": []}]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="construction"):
        ch.load_allowlist(p)


def test_a_missing_field_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "characters": [{"id": "a", "usd": "a.usd"}]}))
    with pytest.raises(ValueError, match="sex"):
        ch.load_allowlist(p)


@pytest.mark.skipif(not ASSETS.exists(), reason="no Isaac assets")
def test_every_listed_usd_exists(allow):
    missing = [c.id for c in allow.characters if not (ASSETS / c.usd).exists()]
    assert not missing


@pytest.mark.parametrize("role", STAFF)
def test_staff_get_scrubs_or_a_uniform(allow, role):
    for sex in ("female", "male"):
        c = allow.choose("N1", role, sex, 40)
        assert c.outfit in ch.STAFF_OUTFITS and c.sex == sex


@pytest.mark.parametrize("role", PUBLIC)
def test_patients_and_visitors_get_civilian_clothes(allow, role):
    for sex in ("female", "male"):
        assert allow.choose("P1", role, sex, 40).outfit == "civilian"


def test_the_sex_and_age_band_are_matched_when_possible(allow):
    assert allow.choose("P2", "patient", "female", 70).sex == "female"
    assert allow.choose("P2", "patient", "male", 70).age == "older"
    assert allow.choose("P2", "patient", "female", 70).age == "older"
    assert allow.choose("P3", "patient", "male", 41).age == "adult"


def test_a_young_age_falls_back_to_the_nearest_band(allow):
    c = allow.choose("P3", "patient", "male", 19)
    assert c.sex == "male" and c.outfit == "civilian"


def test_unknown_sex_takes_any(allow):
    assert allow.choose("S1", "patient", "unknown", 70).outfit == "civilian"


def test_the_choice_is_deterministic_per_person(allow):
    a = [allow.choose(f"P{i}", "patient", "male", 45).id for i in range(1, 9)]
    b = [allow.choose(f"P{i}", "patient", "male", 45).id for i in range(1, 9)]
    assert a == b
    assert len(set(a)) > 1  # the person id spreads the choice


def test_people_in_one_scene_look_different_while_the_list_lasts(allow):
    taken: set[str] = set()
    ids = []
    for i in range(1, 6):
        c = allow.choose(f"P{i}", "patient", "male", 45, taken=taken)
        taken.add(c.id)
        ids.append(c.id)
    assert len(set(ids)) == 5


def test_when_everything_is_taken_a_character_repeats(allow):
    taken = {c.id for c in allow.characters}
    assert allow.choose("P1", "patient", "male", 45, taken=taken).id in taken


def test_cast_assignment_keeps_one_character_per_person(allow):
    people = [("P1", "patient", "male", 62), ("P2", "patient", "female", 70), ("N1", "ward_nurse", "female", 30),
              ("N2", "ward_nurse", "female", 31), ("P3", "visitor", "male", 33)]
    got = allow.assign(people)
    assert set(got) == {"P1", "P2", "N1", "N2", "P3"}
    assert got["N1"].id != got["N2"].id  # two female nurses, two different characters
    assert allow.assign(people) == got
