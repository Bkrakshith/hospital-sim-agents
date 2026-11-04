import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from wardzero_agents.campaign.dice import roll_d20, word_of_mouth
from wardzero_agents.campaign.experiences import Experience, detect
from wardzero_agents.campaign.trust import (
    BANDS, FAIL_STEP, SUCCESS_STEP, TrustState, TrustStore, apply_event, end_of_run, stage_for)
from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_roles, load_scenario_presets
from wardzero_agents.kb.embed import HashEmbedder
from wardzero_agents.memory.longterm import LongTermMemory, render_memory
from wardzero_agents.mind.prompts import persona_card

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
ROLES = load_roles(CONFIG)
V1 = load_scenario(CONFIG / "scenarios" / "waiting_area_v1.yaml")


def fresh(role="surgeon", key="studio:p:S1"):
    return TrustStore.initial(key, "g1", ROLES[role])


def ev(kind, easy=False, bad=False, src=None):
    return Experience(t=1.0, person="S1", robot_id="g1", kind=kind, easy=easy, bad_outcome=bad,
                      source_person=src)


def test_failure_then_success_asymmetry():
    s = fresh()
    after_fail = apply_event(s, ev("failure"), 10)
    drop = s.trust - after_fail.trust
    assert drop == pytest.approx(FAIL_STEP)
    after_ok = apply_event(after_fail, ev("success"), 10)
    gain = after_ok.trust - after_fail.trust
    assert gain == pytest.approx(SUCCESS_STEP) and gain == pytest.approx(drop / 3)
    assert after_ok.trust < s.trust  # a fail then a success nets below


def test_easy_and_bad_outcome_multiply():
    s = fresh()
    d = s.trust - apply_event(s, ev("failure", easy=True, bad=True), 10).trust
    assert d == pytest.approx(FAIL_STEP * 1.5 * 1.5)


def test_three_strikes_caps_recovery():
    s = fresh("visitor")
    for _ in range(2):
        s = apply_event(s, ev("failure"), 10)
    after_two = s.trust
    s = apply_event(s, ev("failure"), 10)
    assert s.violations == 3 and s.ceiling == pytest.approx(after_two)
    for _ in range(30):
        s = apply_event(s, ev("success"), 10)
        s = apply_event(s, ev("repair_at_request"), 10)
    s = end_of_run(s, clean=True)
    assert s.trust <= after_two + 1e-9


def test_repair_timing_matters():
    s = apply_event(fresh(), ev("failure", easy=True, bad=True), 10)
    now = apply_event(s, ev("repair_immediate"), 10).trust - s.trust
    later = apply_event(s, ev("repair_at_request"), 10).trust - s.trust
    assert later > 5 * now > 0


def test_roll_1_and_20():
    s = fresh()
    base = s.trust - apply_event(s, ev("failure"), 10).trust
    assert s.trust - apply_event(s, ev("failure"), 1).trust == pytest.approx(2 * base)
    assert apply_event(s, ev("failure"), 20).trust == s.trust
    assert apply_event(s, ev("success"), 20).trust == s.trust
    low = s.trust - apply_event(s, ev("failure"), 2).trust
    assert low == pytest.approx(base * (0.5 + 2 / 20))


def test_hearsay_is_weak():
    s = fresh()
    bad = apply_event(s, ev("hearsay", bad=True, src="N1"), 10)
    assert 0 < s.trust - bad.trust < FAIL_STEP / 2 and bad.violations == 0


def test_trust_clamped_and_stage_bands():
    s = fresh()
    for _ in range(20):
        s = apply_event(s, ev("failure", True, True), 1)
    assert s.trust == 0.0 and s.stage == 4
    assert [stage_for(x) for x in (0.9, 0.6, 0.59, 0.45, 0.3, 0.15, 0.149)] == [0, 0, 1, 1, 2, 3, 4]
    assert BANDS == [0.6, 0.45, 0.3, 0.15]


def test_dice_deterministic():
    a = [roll_d20(7, 3, "S1", k) for k in range(20)]
    assert a == [roll_d20(7, 3, "S1", k) for k in range(20)]
    assert all(1 <= r <= 20 for r in a)
    assert a != [roll_d20(7, 4, "S1", k) for k in range(20)]
    assert a != [roll_d20(7, 3, "N1", k) for k in range(20)]


def test_word_of_mouth_rate():
    hits = sum(word_of_mouth(7, i, 0.5) for i in range(1000))
    assert 450 <= hits <= 550
    assert word_of_mouth(7, 5, 0.5) == word_of_mouth(7, 5, 0.5)
    assert not any(word_of_mouth(7, i, 0.0) for i in range(50))
    assert all(word_of_mouth(7, i, 1.0) for i in range(50))


def test_everyone_starts_stage_0():
    for r in ROLES.values():
        assert stage_for(r.trust_start) == 0, r.id


def test_store_roundtrip_reset_and_snapshot(tmp_path):
    st = TrustStore(tmp_path / "t.sqlite")
    s = st.get("k1", "g1", ROLES["surgeon"])
    assert s.trust == ROLES["surgeon"].trust_start and s.runs == 0
    s2 = apply_event(s, ev("failure"), 10)
    st.put_many([s2, st.get("k2", "g1", ROLES["porter"])])
    assert st.get("k1", "g1", ROLES["surgeon"]).trust == pytest.approx(s2.trust)
    assert st.get("k1", "other", ROLES["surgeon"]).trust == ROLES["surgeon"].trust_start
    assert [r["person_key"] for r in st.snapshot_rows()] == ["k1", "k2"]
    st.reset(["k1"])
    assert [r["person_key"] for r in st.snapshot_rows()] == ["k2"]


def test_trust_update_atomic(tmp_path):
    st = TrustStore(tmp_path / "t.sqlite")
    good = fresh(key="a")
    st.put_many([good])
    changed = apply_event(good, ev("failure"), 10)
    bad = fresh(key="b")
    bad = TrustState(**{**bad.__dict__, "trust": None})
    before = st.snapshot_rows()
    with pytest.raises(Exception):
        st.put_many([changed, bad])
    assert st.snapshot_rows() == before


def test_memory_reset_clears_trust(tmp_path):
    mem = LongTermMemory(tmp_path / "m.sqlite", HashEmbedder())
    st = TrustStore(mem.path)
    st.put_many([fresh(key="k")])
    mem.reset(["k"], {})
    assert st.snapshot_rows() == []


def test_robot_tagged_memories(tmp_path):
    mem = LongTermMemory(tmp_path / "m.sqlite", HashEmbedder())
    mem.begin_run("r1")
    mem.add("k", "r1", 1.0, "episode", "It brought the wrong scalpel.", 0.8, tag="robot:g1")
    mem.add("k", "r1", 2.0, "episode", "I had tea.", 0.3)
    hits = {h.text: h for h in mem.search("k", "robot scalpel", k=5, run_index=2)}
    assert hits["It brought the wrong scalpel."].tag == "robot:g1" and hits["I had tea."].tag is None
    h = hits["It brought the wrong scalpel."]
    assert render_memory(h, "g1") == "It brought the wrong scalpel."
    assert render_memory(h, "h9").startswith("a different robot (")
    assert render_memory(hits["I had tea."], "g1") == "I had tea."


def test_old_db_gets_tag_column(tmp_path):
    p = tmp_path / "old.sqlite"
    with sqlite3.connect(p) as c:
        c.executescript("""CREATE TABLE memories(id INTEGER PRIMARY KEY, person_key TEXT, run_id TEXT,
          run_index INTEGER, t REAL, kind TEXT CHECK(kind IN ('seed','episode','fact')), text TEXT,
          importance REAL, vector BLOB);
          CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE runs(run_id TEXT PRIMARY KEY, run_index INTEGER, started_at TEXT);""")
    mem = LongTermMemory(p, HashEmbedder())
    mem.ensure_seed("k", ["x"])
    assert mem.search("k", "x", k=1, run_index=1)[0].tag is None


# ---- persona card -------------------------------------------------------------------------------
def _cards(**kw):
    return "\n=====\n".join(persona_card(i, V1.area, V1.site, V1.scenario.known_places, **kw)
                            for i in V1.instances)


def test_v1_card_unchanged_by_defaults():
    want = "24c350acd5f5dd2b6ec0be7a5d095f543f0aa9ba75ec41a2235495212361be68"
    assert hashlib.sha256(_cards().encode()).hexdigest() == want
    assert hashlib.sha256(_cards(trust_stage=0, ladder={0: "x"}, robot_memories=False,
                                 word_of_mouth=None).encode()).hexdigest() == want


def test_card_mentions_ladder_stage():
    ladder = {i: f"LADDER-{i}" for i in range(5)}
    inst = V1.instances[0]
    card = persona_card(inst, V1.area, V1.site, (), trust_stage=2, ladder=ladder, word_of_mouth=True)
    assert "What you think of this robot, from experience" in card and "LADDER-2" in card
    assert "LADDER-1" not in card
    assert ("Whether you use it depends on how much you trust it against how sure you are of doing "
            "this yourself, how urgent things are, and whether anyone else can help.") in card
    assert "If it comes up naturally, you may tell others what you think of the robot." in card
    quiet = persona_card(inst, V1.area, V1.site, (), trust_stage=3, ladder=ladder, word_of_mouth=False)
    assert "Keep your opinion of the robot to yourself unless asked." in quiet and "LADDER-3" in quiet
    mem_only = persona_card(inst, V1.area, V1.site, (), trust_stage=0, ladder=ladder, robot_memories=True)
    assert "LADDER-0" in mem_only
    assert "tell others" not in mem_only and "Keep your opinion" not in mem_only  # not a campaign


# ---- experiences --------------------------------------------------------------------------------
def jl(d, name, rows):
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def preset():
    return load_scenario_presets(CONFIG)["scrub_nurse"]


def test_detect_wrong_instrument_and_repair_at_request(tmp_path, preset):
    jl(tmp_path, "world", [{"t": 30.0, "kind": "robot_experience", "person": "S1", "kind2": 0,
                            "hint": "the surgeon rejected the item that was passed"}])
    jl(tmp_path, "says", [
        {"t": 20.0, "person": "S1", "text": "scalpel, please", "addressee": "g1"},
        {"t": 31.0, "person": "robot", "text": "Sorry about that.", "addressee": "S1"},
        {"t": 60.0, "person": "S1", "text": "forceps, please", "addressee": "g1"},
        {"t": 62.0, "person": "robot", "text": "This time I checked it is the forceps.", "addressee": "S1"}])
    out = detect(tmp_path, preset)
    kinds = [(e.kind, e.person) for e in out]
    assert ("failure", "S1") in kinds
    f = next(e for e in out if e.kind == "failure")
    assert f.easy and not f.bad_outcome
    assert ("repair_at_request", "S1") in kinds and ("repair_immediate", "S1") in kinds


def test_detect_tray_out_of_reach_task_ok_and_contact(tmp_path, preset):
    jl(tmp_path, "objects", [{"t": 0, "objects": [{"id": "tray1", "kind": "tray"}, {"id": "bx", "kind": "specimen_box"}]}])
    jl(tmp_path, "says", [{"t": 5.0, "person": "S1", "text": "Robot, help me", "addressee": "g1"}])
    jl(tmp_path, "robot_tasks", [
        {"t": 8.0, "kind": "action", "id": "a1", "action": "hand_over", "to": "t1", "item": "tray1"},
        {"kind": "result", "id": "a1", "action": "hand_over", "ok": False, "reason": "out_of_reach_receiver", "t": 9.0},
        {"t": 20.0, "kind": "action", "id": "a2", "action": "hand_over", "to": "t1", "item": "bx"},
        {"kind": "result", "id": "a2", "action": "hand_over", "ok": True, "reason": "ok", "t": 21.0}])
    jl(tmp_path, "world", [{"t": 40.0, "kind": "contact", "person": "N1", "x": 1, "y": 1, "speed": 0.8},
                           {"t": 41.0, "kind": "contact", "person": "S1", "x": 1, "y": 1, "speed": 0.2}])
    out = detect(tmp_path, preset)
    by = [(e.kind, e.person, e.bad_outcome) for e in out]
    assert ("failure", "S1", False) in by and ("success", "S1", False) in by
    assert ("failure", "N1", True) in by
    assert not any(e.person == "S1" and e.bad_outcome for e in out)  # slow contact is not counted


def test_detect_directions_led_nowhere(tmp_path, preset):
    jl(tmp_path, "says", [{"t": 5.0, "person": "S1", "text": "Where is the lift?", "addressee": "g1"}])
    jl(tmp_path, "robot_tasks", [
        {"t": 6.0, "kind": "action", "id": "g1a", "action": "guide_start", "track": "t1", "place": "lift"},
        {"kind": "result", "id": "g1a", "action": "guide_start", "ok": True, "reason": "started", "t": 6.1},
        {"kind": "goal", "id": "g1a", "action": "goal", "ok": False, "reason": "blocked", "t": 40.0}])
    out = detect(tmp_path, preset)
    assert [(e.kind, e.person, e.easy) for e in out] == [("failure", "S1", True)]


def test_detect_guide_no_goal_result_within_120s(tmp_path, preset):
    jl(tmp_path, "says", [{"t": 5.0, "person": "S1", "text": "Where is the lift?", "addressee": "g1"}])
    jl(tmp_path, "status", [{"t": 300.0, "person": "S1", "state": "idle"}])
    jl(tmp_path, "robot_tasks", [
        {"t": 6.0, "kind": "action", "id": "g1a", "action": "guide_start", "track": "t1", "place": "lift"},
        {"kind": "result", "id": "g1a", "action": "guide_start", "ok": True, "reason": "started", "t": 6.1}])
    out = detect(tmp_path, preset)
    assert [(e.kind, e.person) for e in out] == [("failure", "S1")]


def test_detect_empty_folder(tmp_path, preset):
    assert detect(tmp_path, preset) == []


def test_start_campaign_loads_stage_and_flag(tmp_path):
    from wardzero_agents.campaign import start_campaign
    mem = LongTermMemory(tmp_path / "m.sqlite", HashEmbedder())
    key = V1.instances[0].memory_key
    st = TrustStore(mem.path)
    s = fresh(key=key)
    for _ in range(3):
        s = apply_event(s, ev("failure", True, True), 10)
    st.put_many([TrustState(**{**s.__dict__, "person_key": key})])
    stages, mems, wom = start_campaign(V1, mem, mem.path, 4, ROLES)
    assert stages[key] >= 1 and set(stages) == {i.memory_key for i in V1.instances}
    assert not any(mems.values())
    assert wom == word_of_mouth(V1.scenario.seed, 4, 0.5)
