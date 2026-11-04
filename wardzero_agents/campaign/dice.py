"""Seeded dice. Every roll is reproducible from (project seed, run index, person, k)."""
from __future__ import annotations

import hashlib

import numpy as np


def _rng(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def roll_d20(project_seed: int, run_index: int, person: str, k: int) -> int:
    """The k-th d20 of this person in this run of this project."""
    return int(_rng("d20", project_seed, run_index, person, k).integers(1, 21))


def word_of_mouth(project_seed: int, run_index: int, p: float) -> bool:
    """Whether people talk about the robot to each other in this run."""
    return bool(_rng("wom", project_seed, run_index).random() < p)
