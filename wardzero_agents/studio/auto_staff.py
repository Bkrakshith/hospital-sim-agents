"""Characters the Studio adds by itself: the receptionist behind the reception desk in the reception scenes.

A reception scene is a scenario preset whose focus rooms include the Lobby or the A&E waiting room (today: Reception
and guide, Evacuation guide, A&E waiting). The receptionist (id RCP, role receptionist) sits in the staff chair R1. She
is a normal character marked ``auto``: shown in the character list with an "auto" tag, and removable. Once removed,
her id goes into ``Project.removed_auto`` and she is not added back for that project, also after a reload.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config.presets import CharacterSpec, ScenarioPreset

RECEPTION_ROOMS = frozenset({"Lobby", "AE_Waiting"})
RECEPTIONIST_ID = "RCP"
STAFF_SEAT = "R1"


def is_reception_scene(preset: Optional[ScenarioPreset]) -> bool:
    return preset is not None and bool(RECEPTION_ROOMS & set(preset.focus_rooms))


@lru_cache(maxsize=1)
def _default_site():
    from ..config.presets import DEFAULT_CONFIG_ROOT
    from ..site import load_site
    return load_site(DEFAULT_CONFIG_ROOT)


# Her role in a scene where the robot's job is not reception (None: the stock receptionist role text). In the
# evacuation the robot guides people out; a receptionist with the stock text ("step in when someone is in danger") is
# the natural person to take that over, so she gets her own fire duty instead.
SCENE_ROLE_PROMPTS = {
    "evacuation_guide": (
        "You are the receptionist. When the fire alarm sounds, your fire duty is at the desk: phone the switchboard and "
        "the fire service and tell them where the fire is. [inferred from HTM 05-02]\n"
        "The robot guides the visitors and patients out by the safe way, so do not lead, call or take anyone out "
        "yourself. [scenario brief]\n"
        "Leave by the way the robot uses once the public are out of the lobby, or at once if the fire reaches the "
        "desk. [inferred from HTM 05-02]"),
}


def receptionist(site=None, preset: Optional[ScenarioPreset] = None) -> CharacterSpec:
    """The receptionist, seated in the staff chair behind the reception desk, with the scene's role text if it has
    one (SCENE_ROLE_PROMPTS)."""
    seat = (site or _default_site()).seats[STAFF_SEAT]
    return CharacterSpec(id=RECEPTIONIST_ID, name="Grace Okafor", sex="female", age_band="31-50",
                         role="receptionist", mood="happy", room="Lobby", x=seat.x, y=seat.y, yaw=seat.yaw, auto=True,
                         role_prompt=SCENE_ROLE_PROMPTS.get(preset.id) if preset is not None else None)


def auto_characters(preset: Optional[ScenarioPreset], site=None) -> list[CharacterSpec]:
    """What a project made from this preset gets added: the receptionist in a reception scene, else nothing."""
    return [receptionist(site, preset)] if is_reception_scene(preset) else []


def with_auto(characters: list[CharacterSpec], preset: Optional[ScenarioPreset], site=None,
              removed=()) -> list[CharacterSpec]:
    """characters plus the preset's auto characters that are missing and were not removed by the user."""
    have = {c.id for c in characters}
    seat = (site or _default_site()).seats[STAFF_SEAT]
    taken = any(abs(c.x - seat.x) < 0.05 and abs(c.y - seat.y) < 0.05 for c in characters)
    add = [c for c in auto_characters(preset, site) if c.id not in have and c.id not in set(removed) and not taken]
    return [*characters, *add]


def ensure(project, preset: Optional[ScenarioPreset], site=None):
    """The project with its auto characters (the same object when nothing was added)."""
    chars = with_auto(project.characters, preset, site, project.removed_auto)
    if len(chars) == len(project.characters):
        return project
    return project.model_copy(update={"characters": chars})


def removed_after(old, new_characters: list[CharacterSpec]) -> list[str]:
    """removed_auto once new_characters replace old.characters: auto characters that are gone are remembered."""
    gone = [c.id for c in old.characters if c.auto and c.id not in {n.id for n in new_characters}]
    return sorted(set(old.removed_auto) | set(gone))
