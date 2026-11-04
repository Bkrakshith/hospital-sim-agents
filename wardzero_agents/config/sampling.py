"""Deterministic sampling of one person from a persona template."""
from __future__ import annotations

import numpy as np

from .models import CastEntry, Persona, PersonInstance

MALE_NAMES = [
    "Oliver", "George", "Harry", "Noah", "Jack", "Leo", "Arthur", "Oscar", "Charlie", "Henry",
    "Thomas", "William", "James", "Edward", "Alfie", "Freddie", "Joshua", "Daniel", "David", "Michael",
]
FEMALE_NAMES = [
    "Olivia", "Amelia", "Isla", "Ava", "Emily", "Sophie", "Grace", "Lily", "Charlotte", "Jessica",
    "Margaret", "Susan", "Patricia", "Sarah", "Hannah", "Rebecca", "Helen", "Karen", "Joan", "Ruth",
]
SURNAMES = [
    "Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson", "Davies", "Robinson", "Wright",
    "Thompson", "Evans", "Walker", "White", "Roberts", "Green", "Hall", "Wood", "Jackson", "Clarke",
    "Patel", "Khan", "Lewis", "James", "Phillips", "Mason", "Mitchell", "Rose", "Davis", "Rogers",
    "Cooper", "Harris", "Ward", "Turner", "Hill", "Moore", "Scott", "Morris", "Bailey", "Baker",
]
assert len(MALE_NAMES) == 20 and len(FEMALE_NAMES) == 20 and len(SURNAMES) == 40


def instance_seed(cast: CastEntry, scenario_seed: int, index: int) -> int:
    return cast.seed if cast.seed is not None else scenario_seed * 1000 + index


def sample_instance(persona: Persona, cast: CastEntry, scenario_seed: int, index: int) -> PersonInstance:
    seed = instance_seed(cast, scenario_seed, index)
    rng = np.random.default_rng(seed)

    def u(r) -> float:
        return float(rng.uniform(r[0], r[1]))

    # draw order is part of the contract: changing it changes every sampled person
    age = int(round(u(persona.age)))
    sex = persona.sex[int(rng.integers(len(persona.sex)))]
    cough = u(persona.symptoms.cough_per_hour)
    walk = u(persona.mobility.walk_speed)
    sit = u(persona.mobility.sit_to_stand_s)
    hearing = u(persona.hearing_loss_db)
    patience = u(persona.patience_min)
    anxiety = u(persona.anxiety)
    sociability = u(persona.sociability)
    phone = u(persona.phone_use)
    # the first name comes from the list that matches the sampled sex, so a name never contradicts it
    firsts = {"female": FEMALE_NAMES, "male": MALE_NAMES}.get(sex.lower(), FEMALE_NAMES + MALE_NAMES)
    name = f"{firsts[int(rng.integers(len(firsts)))]} {SURNAMES[int(rng.integers(40))]}"
    if cast.name:  # an authored name replaces the drawn one; the draws above stay the same
        name = cast.name

    return PersonInstance(
        person_id=cast.id,
        persona_id=persona.id,
        seed=seed,
        memory_key=f"{persona.id}#{seed}",
        display_name=name,
        age=age,
        sex=sex,
        cough_per_hour=cough,
        cough_severity=persona.symptoms.cough_severity,
        breathless=persona.symptoms.breathless,
        walk_speed=walk,
        sit_to_stand_s=sit,
        hearing_loss_db=hearing,
        patience_min=patience,
        anxiety=anxiety,
        sociability=sociability,
        phone_use=phone,
        attitude_to_robots=persona.attitude_to_robots,
        english_fluency=persona.english_fluency,
        mobility_aid=persona.mobility.aid,
        goals=list(persona.goals),
        backstory=persona.backstory,
        condition=persona.condition,
        role=persona.role,
        seat=cast.seat,
        companion_of=cast.companion_of,
        arrive_at_s=cast.arrive_at_s,
        entrance=cast.entrance,
        start=cast.start,
    )
