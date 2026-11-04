"""The Isaac People allow-list and the deterministic choice of a character for each person.

The characters under ``Isaac/People/Characters`` and ``DH_Characters`` include construction workers, police and
hi-vis workwear, which do not belong in a hospital. ``config/isaac/characters.yaml`` lists the ones that do
(built by ``scripts/curate_characters.py``, checked by eye on a contact sheet), tagged by sex, age band and
outfit. Staff get scrubs, everyone else civilian clothes. The choice depends only on the person id, so a run and
its rerun see the same people.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

AGE_BANDS = ("young_adult", "adult", "older")
OLDER_FROM = 60
YOUNG_BELOW = 30
OUTFITS = ("civilian", "scrubs", "uniform")
STAFF_OUTFITS = ("scrubs", "uniform")
STAFF_ROLES = frozenset({"staff", "ward_nurse", "surgeon", "porter", "receptionist"})
BANNED_WORDS = ("construction", "police", "hardhat", "hard_hat", "helmet", "hivis", "hi_vis", "hi-vis", "workwear",
                "safety", "vest")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "isaac" / "characters.yaml"


def age_band(age: int) -> str:
    return "older" if age >= OLDER_FROM else ("young_adult" if age < YOUNG_BELOW else "adult")


@dataclass(frozen=True)
class Character:
    id: str
    usd: str  # relative to the Isaac assets root
    sex: str
    age: str
    age_years: int
    outfit: str
    hide: tuple[str, ...] = ()  # prims under the SkelRoot to deactivate (safety accessories)


def _rank(pid: str, cid: str) -> str:
    return hashlib.sha1(f"{pid}|{cid}".encode()).hexdigest()


class AllowList:
    def __init__(self, characters: list[Character]):
        self.characters = characters

    def _pool(self, role: str, sex: str, age: int) -> list[Character]:
        staff = role in STAFF_ROLES
        pool = [c for c in self.characters if (c.outfit in STAFF_OUTFITS) == staff] or list(self.characters)
        if sex in ("female", "male") and any(c.sex == sex for c in pool):
            pool = [c for c in pool if c.sex == sex]
        want = AGE_BANDS.index(age_band(age))
        best = min(abs(AGE_BANDS.index(c.age) - want) for c in pool)
        return [c for c in pool if abs(AGE_BANDS.index(c.age) - want) == best]

    def choose(self, pid: str, role: str, sex: str, age: int, taken: Iterable[str] = ()) -> Character:
        taken = set(taken)
        pool = sorted(self._pool(role, (sex or "").lower(), int(age)), key=lambda c: _rank(pid, c.id))
        return next((c for c in pool if c.id not in taken), pool[0])

    def assign(self, people: Iterable[tuple[str, str, str, int]]) -> dict[str, Character]:
        """(person id, role, sex, age) for everyone -> one character each, as different as the list allows."""
        out: dict[str, Character] = {}
        taken: set[str] = set()
        for pid, role, sex, age in people:
            c = self.choose(pid, role, sex, age, taken)
            out[pid] = c
            taken.add(c.id)
        return out


def load_allowlist(path: Path = DEFAULT_PATH) -> AllowList:
    data = yaml.safe_load(Path(path).read_text()) or {}
    chars: list[Character] = []
    for raw in data.get("characters", []):
        for key in ("id", "usd", "sex", "age", "outfit"):
            if key not in raw:
                raise ValueError(f"character {raw.get('id', '?')}: missing {key}")
        text = f"{raw['id']} {raw['usd']}".lower()
        bad = next((w for w in BANNED_WORDS if w in text), None)
        if bad:
            raise ValueError(f"character {raw['id']} is not allowed in a hospital ({bad})")
        if raw["sex"] not in ("female", "male") or raw["age"] not in AGE_BANDS or raw["outfit"] not in OUTFITS:
            raise ValueError(f"character {raw['id']}: bad sex, age or outfit")
        chars.append(Character(raw["id"], raw["usd"], raw["sex"], raw["age"], int(raw.get("age_years", 0)),
                               raw["outfit"], tuple(raw.get("hide") or ())))
    if not chars:
        raise ValueError(f"{path}: no characters")
    return AllowList(chars)
