"""After-run debrief: questionnaire, hearsay, trust update."""
import asyncio
import json
from pathlib import Path

import pytest
from _rundir import make_ae_run

from wardzero_agents.bench.metrics import METRICS, RunData
from wardzero_agents.campaign.debrief import debrief, make_post_gateway
from wardzero_agents.campaign.trust import TrustStore
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_roles, load_scenario_presets, people_meta, preset_for_run, preset_to_scenario
from wardzero_agents.llm.fake import FakeGateway, default_policy
from wardzero_agents.memory import data_dir, memory_path

CONFIG = Path(__file__).resolve().parents[1] / "config"
ROLES = load_roles(CONFIG)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("WARDZERO_TEST_FAKE_LLM", raising=False)


def policy_with(ratings, hearsay=None, seen=None):
    """Questionnaire answers by person; hearsay items when asked."""
    def policy(req):
        if seen is not None:
            seen.append((req.role, req.person))
        props = req.schema.get("properties", {})
        if "likeability" in props:
            return {**ratings[req.person], "comment": f"{req.person} says hello"}
        if "items" in props:
            return {"items": hearsay or []}
        return default_policy(req)
    return policy


R = {"P1": dict(likeability=4, safety=5, competence=2, trust=3),
     "P2": dict(likeability=2, safety=4, competence=1, trust=2),
     "N1": dict(likeability=5, safety=5, competence=4, trust=4)}


def run(coro):
    return asyncio.run(coro)


def test_debrief_writes_ratings_and_updates_trust(tmp_path):
    rd = make_ae_run(tmp_path / "r1")
    seen = []
    res = run(debrief(rd, FakeGateway(policy_with(R, seen=seen)), preset=preset_for_run({"preset_id": "ae_waiting"})))
    rows = [json.loads(x) for x in (rd / "debrief.jsonl").read_text().splitlines()]
    assert {r["person"] for r in rows} == {"P1", "P2", "N1"}
    p1 = next(r for r in rows if r["person"] == "P1")
    assert p1 == {"person": "P1", "likeability": 4, "safety": 5, "competence": 2, "trust": 3, "comment": "P1 says hello"}
    # the metrics read exactly what was written
    d = RunData.load(rd)
    assert METRICS["rating_likeability"](d) == pytest.approx((4 + 2 + 5) / 3)
    assert METRICS["rating_safety"](d) == pytest.approx((5 + 4 + 5) / 3)
    assert METRICS["rating_competence"](d) == pytest.approx((2 + 1 + 4) / 3)
    assert METRICS["trust_item_mean"](d) == pytest.approx(3.0)
    # trust.jsonl has the documented fields and the db holds the new states
    trows = [json.loads(x) for x in (rd / "trust.jsonl").read_text().splitlines()]
    assert {r["person"] for r in trows} == {"P1", "P2", "N1"}
    for r in trows:
        assert set(r) == {"person", "robot_id", "before", "after", "stage_before", "stage_after"}
        assert r["robot_id"] == "g1" and 0 <= r["after"] <= 1
    assert METRICS["trust_delta_mean"](d) is not None
    store = TrustStore(memory_path(data_dir()))
    p2 = next(r for r in trows if r["person"] == "P2")
    assert store.get("P2", "g1", ROLES["patient"]).trust == pytest.approx(p2["after"], abs=1e-3)
    # one questionnaire call each, no hearsay call without word of mouth
    assert sorted(p for r, p in seen) == ["N1", "P1", "P2"]
    assert res.already is False and RunData.load(rd).meta["debriefed"] is True


def test_debrief_twice_does_not_apply_trust_twice(tmp_path):
    rd = make_ae_run(tmp_path / "r1")
    run(debrief(rd, FakeGateway(policy_with(R))))
    store = TrustStore(memory_path(data_dir()))
    first = store.get("P2", "g1", ROLES["patient"])
    again = run(debrief(rd, FakeGateway(policy_with(R))))
    assert again.already and store.get("P2", "g1", ROLES["patient"]) == first


def test_debrief_skips_static_patients(tmp_path):
    meta = {"people": {"P1": {"role": "patient", "preset_role": "patient", "static": False},
                       "SB1": {"role": "patient_static", "preset_role": None, "static": True}}}
    rd = make_ae_run(tmp_path / "r1", **meta)
    seen = []
    run(debrief(rd, FakeGateway(policy_with({"P1": R["P1"]}, seen=seen))))
    assert [p for _, p in seen] == ["P1"]
    assert all(json.loads(x)["person"] != "SB1" for x in (rd / "debrief.jsonl").read_text().splitlines())
    assert all(json.loads(x)["person"] != "SB1" for x in (rd / "trust.jsonl").read_text().splitlines())


def test_unusable_answers_are_skipped_and_scores_clamped(tmp_path):
    rd = make_ae_run(tmp_path / "r1")

    def policy(req):
        if "likeability" in req.schema.get("properties", {}):
            if req.person == "N1":
                return {"likeability": "great"}
            return {"likeability": 9, "safety": 0, "competence": 3.4, "trust": 5, "comment": " ok "}
        return default_policy(req)
    res = run(debrief(rd, FakeGateway(policy)))
    assert res.skipped == ["N1"]
    p1 = next(r for r in res.ratings if r["person"] == "P1")
    assert (p1["likeability"], p1["safety"], p1["competence"], p1["comment"]) == (5, 1, 3, "ok")


def test_hearsay_becomes_an_experience_and_moves_the_listener(tmp_path):
    item = {"line": 0, "speaker": "P1", "listener": "P2", "valence": -0.8}
    quiet = make_ae_run(tmp_path / "quiet")
    run(debrief(quiet, FakeGateway(policy_with(R, hearsay=[item]))))  # word_of_mouth False: not asked
    base = json.loads(next(x for x in (quiet / "trust.jsonl").read_text().splitlines() if '"P2"' in x))
    assert not run(debrief(quiet, FakeGateway(policy_with(R)))).hearsay

    TrustStore(memory_path(data_dir())).reset(["P1", "P2", "N1"])
    loud = make_ae_run(tmp_path / "loud", word_of_mouth=True)
    res = run(debrief(loud, FakeGateway(policy_with(R, hearsay=[item]))))
    assert [(e.kind, e.person, e.source_person, e.bad_outcome) for e in res.hearsay] == [("hearsay", "P2", "P1", True)]
    got = json.loads(next(x for x in (loud / "trust.jsonl").read_text().splitlines() if '"P2"' in x))
    assert got["after"] < base["after"]


def test_people_meta_has_roles_and_preset_id():
    preset = load_scenario_presets(CONFIG)["ae_waiting"]
    loaded = preset_to_scenario(preset, CONFIG)
    assert loaded.scenario.preset_id == "ae_waiting"
    people = people_meta(loaded)
    assert people["N1"] == {"role": "ward_nurse", "preset_role": "ward_nurse", "static": False}
    assert people["P1"] == {"role": "ae_patient", "preset_role": "ae_patient", "static": False}
    assert people["V1"]["preset_role"] == "visitor"
    assert preset_for_run({"preset_id": "ae_waiting"}).id == "ae_waiting"
    assert preset_for_run({"scenario_id": "waiting_area_v1"}) is None


def test_post_gateway_logs_costs_as_debrief_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("WARDZERO_TEST_FAKE_LLM", "1")
    rd = make_ae_run(tmp_path / "r1")
    run(debrief(rd, make_post_gateway(rd)))
    rows = [json.loads(x) for x in (rd / "tokens.jsonl").read_text().splitlines()]
    assert {r["person"] for r in rows[1:]} == {"debrief"} and len(rows) == 4


def test_failed_questionnaires_apply_nothing_and_do_not_mark(tmp_path):
    from wardzero_agents.llm.fake import FakeRefusal
    rd = make_ae_run(tmp_path / "r1")

    def policy(req):
        if "likeability" in req.schema.get("properties", {}):
            raise FakeRefusal()
        return default_policy(req)
    res = run(debrief(rd, FakeGateway(policy)))
    assert res.ok is False and not (rd / "trust.jsonl").exists() and not (rd / "debrief.jsonl").exists()
    store = TrustStore(memory_path(data_dir()))
    assert store.debriefed_before("r1") is None and store.snapshot_rows() == []
    assert not RunData.load(rd).meta.get("debriefed")
    # a later debrief that works is not blocked
    assert run(debrief(rd, FakeGateway(policy_with(R)))).ok


def test_force_resets_to_the_recorded_start_and_is_idempotent(tmp_path):
    rd = make_ae_run(tmp_path / "r1")
    store = TrustStore(memory_path(data_dir()))
    run(debrief(rd, FakeGateway(policy_with(R))))
    once = store.snapshot_rows()
    before = [json.loads(x) for x in (rd / "trust.jsonl").read_text().splitlines()]
    for _ in range(2):
        res = run(debrief(rd, FakeGateway(policy_with(R)), force=True))
        assert res.ok and not res.already
        assert store.snapshot_rows() == once          # the same as after the first debrief
    again = [json.loads(x) for x in (rd / "trust.jsonl").read_text().splitlines()]
    assert {r["person"]: r["before"] for r in again} == {r["person"]: r["before"] for r in before}
