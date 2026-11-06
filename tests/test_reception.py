"""Reception and guide: the staff seat, the receptionist, tell_staff, follow, guide pacing and the arrival metric."""
from pathlib import Path

import pytest

from wardzero_agents.config.presets import CharacterSpec, load_moods, load_roles
from wardzero_agents.site import load_site
from wardzero_agents.studio import characters as chars

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG)


def _char(**kw):
    d = dict(id="C1", name="Test Person", sex="female", age_band="31-50", role="visitor", mood="happy", room="",
             x=0.0, y=0.0)
    d.update(kw)
    return CharacterSpec(**d)


def test_placing_near_the_staff_chair_never_snaps_to_it(site):
    roles, moods = load_roles(CONFIG), load_moods(CONFIG)
    r1 = site.seats["R1"]
    c = chars.check_character(_char(x=r1.x + 0.2, y=r1.y), site, roles, moods, [])
    assert (c.x, c.y) == (r1.x + 0.2, r1.y)  # stands where it was put: R1 is not a seat to snap to
    a1 = site.seats["A1"]
    c = chars.check_character(_char(x=a1.x + 0.2, y=a1.y + 0.1), site, roles, moods, [])
    assert (c.x, c.y) == (a1.x, a1.y)  # a waiting seat still snaps
