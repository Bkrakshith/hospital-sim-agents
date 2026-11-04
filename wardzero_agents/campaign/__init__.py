"""Campaign trust across runs: a per-person trust state, seeded dice and word of mouth."""
from __future__ import annotations

from types import SimpleNamespace

WORD_OF_MOUTH_P = 0.5  # chance that people talk about the robot in a given run
ROBOT_ID = "g1"


def role_defaults(roles: dict, role: str):
    """The role preset for a person's role (a preset id, else the first with that persona_role);
    a neutral one when there is none."""
    if role in roles:
        return roles[role]
    for r in roles.values():
        if r.persona_role == role:
            return r
    return SimpleNamespace(trust_start=0.7, self_confidence=0.5)


def start_campaign(loaded, ltm, db_path, run_index: int, roles: dict, robot_id: str = ROBOT_ID):
    """Load every person's trust once, at the start of a run, and decide word of mouth.

    The persona card is a cached prefix, so what it says about the robot is fixed for the run.
    Returns (stages: {memory_key: stage}, robot_memories: {memory_key: bool}, word_of_mouth: bool)."""
    from .dice import word_of_mouth
    from .trust import TrustStore
    store = TrustStore(db_path)
    stages, mems = {}, {}
    for inst in loaded.instances:
        s = store.get(inst.memory_key, robot_id, role_defaults(roles, inst.role))
        stages[inst.memory_key] = s.stage
        mems[inst.memory_key] = ltm.has_robot_memories(inst.memory_key, robot_id)
    return stages, mems, word_of_mouth(loaded.scenario.seed, run_index, WORD_OF_MOUTH_P)
