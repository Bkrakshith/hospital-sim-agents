import shutil

import pytest
import yaml

from tests.test_config import CFG, V1, write_patched
from wardzero_agents.config.loader import ConfigError, load_scenario
from wardzero_agents.config.models import CardDef
from wardzero_agents.contracts import Nudge, ReflexOverride, Roster, RosterEntry, Status, WorldEvent
from wardzero_agents.director.cards import CardExecutor, CardRejected
from wardzero_agents.director.chaos import Chaos
from wardzero_agents.director.storyboard import CardFiring, StoryboardEngine, WorldView

LS = load_scenario(V1)
ALL = {"P1", "P2", "P3", "P4", "P5"}


def status(pid, x=0.0, y=0.0, posture="seated"):
    return Status(person=pid, t=0, x=x, y=y, yaw=0, posture=posture, progress=0, state="idle")


def view(t=0.0, seated_since=None, present=None, statuses=None, robot_xy=(0.0, 0.0)):
    present = set(ALL) if present is None else present
    st = statuses if statuses is not None else {p: status(p, 5, 5) for p in present}
    return WorldView(t=t, statuses=st, robot_xy=robot_xy, seated_since=seated_since or {}, present=present)


def roster():
    return Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name,
                                           seat=i.seat, companion_of=i.companion_of) for i in LS.instances])


def engine():
    return StoryboardEngine(LS.storyboard, LS.deck)


def after_engine():
    """Test-only storyboard that exercises `after` ordering (the v1 storyboard has none)."""
    from wardzero_agents.config.models import Beat, Storyboard
    sb = Storyboard(id="x", beats=[
        Beat(id="a", at="t+20:00", card="name_called", target="P1"),
        Beat(id="b", after="a", card="new_arrival"),
    ])
    return StoryboardEngine(sb, LS.deck)


def executor():
    return CardExecutor(LS.deck, {i.person_id: i for i in LS.instances})


def firing(card, target=None, params=None, source="panel"):
    return CardFiring(id=f"{source}-1", t=10.0, card=card, target=target, params=params or {}, source=source)


def test_v1_storyboard_loads():
    assert [b.id for b in LS.storyboard.beats] == ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
    assert len(LS.deck) == 12
    assert "fire_alarm" not in LS.deck
    assert all(c.tests for c in LS.deck.values())


def test_beat_at_time():
    e = engine()
    assert [f.card for f in e.tick(view(t=299))] == []
    fired = e.tick(view(t=300))
    assert [f.card for f in fired] == ["clinic_delay"] and fired[0].source == "storyboard"
    assert e.tick(view(t=301)) == []


def test_after_beat():
    e = after_engine()
    got = e.tick(view(t=1200))
    assert [f.card for f in got] == ["name_called"]
    assert [f.card for f in e.tick(view(t=1201))] == ["new_arrival"]
    assert e.tick(view(t=1202)) == []


def test_condition_beat():
    e = engine()
    v = view(t=3001, seated_since={"P4": 0})
    assert [f.card for f in e.tick(v) if f.card == "impatient_patient"] == ["impatient_patient"]
    assert [f for f in e.tick(view(t=3002, seated_since={"P4": 0})) if f.card == "impatient_patient"] == []


def test_condition_robot_within_and_posture():
    from wardzero_agents.config.models import Beat, Condition, Storyboard
    sb = Storyboard(id="x", beats=[
        Beat(id="a", when=Condition(kind="robot_within", person="P1", value=2.0), card="lost_visitor", target="P1"),
        Beat(id="b", when=Condition(kind="posture_is", person="P2", value="standing"), card="lost_visitor", target="P2"),
    ])
    e = StoryboardEngine(sb, LS.deck)
    assert e.tick(view(statuses={"P1": status("P1", 5, 0), "P2": status("P2")}, robot_xy=(0, 0))) == []
    got = e.tick(view(statuses={"P1": status("P1", 1.5, 0), "P2": status("P2", posture="standing")}))
    assert [f.target for f in got] == ["P1", "P2"]


def test_missing_target_skipped_and_rerun():
    e = engine()
    e.tick(view(t=720, present=ALL - {"P5"}))
    tl = {r["id"]: r for r in e.timeline()}
    assert tl["b3"]["state"] == "skipped" and "P5" in tl["b3"]["reason"]
    assert tl["b2"]["state"] == "fired"
    e.skip("b6")
    assert {r["id"]: r for r in e.timeline()}["b6"]["state"] == "skipped"
    e.rerun("b3")
    assert [f.card for f in e.tick(view(t=730))] == ["lost_visitor"]
    assert {r["id"]: r for r in e.timeline()}["b3"]["state"] == "fired"


def test_timeline_shape():
    tl = engine().timeline()
    assert tl[0] == {"id": "b1", "card": "clinic_delay", "target": None, "state": "pending", "fired_t": None}


def test_coughing_fit_outputs():
    outs = executor().execute(firing("coughing_fit", "P2"), view(), roster())
    assert any(isinstance(o, ReflexOverride) and o.person == "P2" and o.duration_s == 40 and o.severity == "severe" for o in outs)
    assert any(isinstance(o, Nudge) and o.person == "P3" and "struggling to breathe" in o.message for o in outs)
    assert isinstance(outs[0], ReflexOverride)


def test_name_called_and_templates():
    outs = executor().execute(firing("name_called", "P1"), view(), roster())
    name = next(i.display_name for i in LS.instances if i.person_id == "P1")
    n = [o for o in outs if isinstance(o, Nudge)][0]
    w = [o for o in outs if isinstance(o, WorldEvent)][0]
    assert "clinic_corridor" in n.message and w.kind == "pa_announcement" and name in w.text
    assert w.id == "panel-1"


def test_all_target_kind_and_new_arrival():
    everyone = CardDef(id="everyone", label="Everyone", channels=["nudge", "world"], target_kind="all",
                       nudge="Something happens.", world={"kind": "alarm", "text": "A test alarm."}, tests="test-only")
    ex = CardExecutor({**LS.deck, "everyone": everyone}, {i.person_id: i for i in LS.instances})
    outs = ex.execute(firing("everyone"), view(present={"P1", "P2"}), roster())
    assert sorted(o.person for o in outs if isinstance(o, Nudge)) == ["P1", "P2"]
    assert any(isinstance(o, WorldEvent) and o.kind == "alarm" for o in outs)
    outs = executor().execute(firing("new_arrival", params={"persona": "companion"}), view(), roster())
    assert [(o.kind, o.persona) for o in outs] == [("arrival", "companion")]


def test_ward_visitor_card_spawns_persona():
    outs = executor().execute(firing("ward_visitor"), view(), roster())
    assert [(o.kind, o.persona) for o in outs if isinstance(o, WorldEvent)] == [("arrival", "ward_visitor")]
    assert "ward_visitor" in LS.personas


def test_direction_nudges_never_name_the_robot():
    for c in LS.deck.values():
        text = " ".join(str(x) for x in (c.nudge, c.companion_nudge, c.world) if x).lower()
        assert "robot" not in text and "g1" not in text, c.id


def test_direction_cards_present():
    for cid in ("toilet_request", "way_out"):
        assert LS.deck[cid].target_kind == "person" and LS.deck[cid].nudge
        o = executor().execute(firing(cid, "P4"), view(), roster())
        assert [type(x) for x in o] == [Nudge] and o[0].person == "P4"


def test_wet_floor_and_params_override():
    o = executor().execute(firing("wet_floor", params={"x": 3.0}), view(), roster())[0]
    assert o.kind == "wet_floor" and o.x == 3.0 and o.y is not None
    o = executor().execute(firing("clinic_delay", params={"text": "Custom"}), view(), roster())[0]
    assert o.text == "Custom"


def test_card_for_missing_person_rejected():
    with pytest.raises(CardRejected, match="P9 is not in the room"):
        executor().execute(firing("lost_visitor", "P9"), view(), roster())
    with pytest.raises(CardRejected, match="P2 is not in the room"):
        executor().execute(firing("coughing_fit", "P2"), view(present={"P1"}), roster())


def test_missing_template_key_rejected():
    ex = CardExecutor({**LS.deck, "t": LS.deck["lost_visitor"].model_copy(update={"nudge": "Hi {params.where}"})}, {i.person_id: i for i in LS.instances})
    with pytest.raises(CardRejected, match="params.where"):
        ex.execute(firing("t", "P1"), view(), roster())
    assert [o.message for o in ex.execute(firing("t", "P1", {"where": "here"}), view(), roster())] == ["Hi here"]


def test_blocked_path_nudge():
    o = executor().execute(firing("blocked_path", "P4"), view(), roster())[0]
    assert "reception_front" in o.message and "bag" in o.message


def run_chaos(seed, hours=1, step=1.0, rate=6.0):
    c = Chaos(rate, LS.deck, seed)
    out, t = [], 0.0
    while t < hours * 3600:
        out += [(f.t, f.card, f.target, f.id) for f in c.tick(t, view(t=t))]
        t += step
    return out


def test_chaos_deterministic():
    a = run_chaos(7)
    assert a == run_chaos(7) and 2 <= len(a) <= 14
    assert all(x[3].startswith("chaos-") for x in a)
    assert a != run_chaos(8)


def test_chaos_independent_of_tick_spacing():
    assert [x[1:3] for x in run_chaos(7, step=1.0)] == [x[1:3] for x in run_chaos(7, step=5.0)]


def test_chaos_respects_targets():
    c = Chaos(3600.0, LS.deck, 1)
    got = []
    for t in range(1, 200):
        got += c.tick(float(t), view(t=t, present=set()))
    assert got and all(LS.deck[f.card].target_kind != "person" for f in got)
    assert all(f.source == "chaos" for f in got)


def test_bad_beat_rejected(tmp_path):
    def patched(fn):
        p = write_patched(tmp_path, V1, {})
        sb = p.parent.parent / "storyboards" / "waiting_area_day.yaml"
        data = yaml.safe_load(sb.read_text())
        fn(data)
        sb.write_text(yaml.safe_dump(data))
        return p
    p = patched(lambda d: d["beats"][0].update(after="b2"))
    with pytest.raises(ConfigError, match=r"storyboard\.beats\[0\]: exactly one of at, when, after"):
        load_scenario(p)


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d["beats"][1].update(card="nope"), r"storyboard\.beats\[1\]\.card: unknown card 'nope'"),
    (lambda d: d["beats"][1].update(target="P9"), r"storyboard\.beats\[1\]\.target: unknown person 'P9'"),
    (lambda d: d["beats"][3].update(at=None, after="b5"), r"storyboard\.beats\[3\]\.after: .*earlier beat"),
    (lambda d: d["beats"][0].update(at="5 min"), r"storyboard\.beats\[0\]\.at"),
])
def test_storyboard_cross_checks(tmp_path, edit, msg):
    p = write_patched(tmp_path, V1, {})
    sb = p.parent.parent / "storyboards" / "waiting_area_day.yaml"
    data = yaml.safe_load(sb.read_text())
    edit(data)
    sb.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=msg):
        load_scenario(p)


def test_missing_files_are_errors(tmp_path):
    p = write_patched(tmp_path, V1, {})
    (p.parent.parent / 'cards' / 'starter.yaml').unlink()
    with pytest.raises(ConfigError, match='^deck: file not found'):
        load_scenario(p)


def _deck_patch(tmp_path, fn):
    p = write_patched(tmp_path, V1, {})
    f = p.parent.parent / "cards" / "starter.yaml"
    data = yaml.safe_load(f.read_text())
    fn(data["cards"])
    f.write_text(yaml.safe_dump(data))
    return p


def _sb_patch(tmp_path, fn):
    p = write_patched(tmp_path, V1, {})
    f = p.parent.parent / "storyboards" / "waiting_area_day.yaml"
    d = yaml.safe_load(f.read_text())
    fn(d)
    f.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.parametrize("reflex", [{"severity": "severe"}, {"duration_s": 40}, {"duration_s": 0, "severity": "severe"},
                                    {"duration_s": 40, "severity": "extreme"}])
def test_bad_reflex_rejected_at_load(tmp_path, reflex):
    p = _deck_patch(tmp_path, lambda cs: next(c for c in cs if c["id"] == "coughing_fit").update(reflex=reflex))
    with pytest.raises(ConfigError, match=r"deck\.coughing_fit\.reflex: "):
        load_scenario(p)


def test_bad_reflex_params_rejected_at_fire():
    with pytest.raises(CardRejected, match="severity"):
        executor().execute(firing("coughing_fit", "P2", {"severity": "extreme"}), view(), roster())
    with pytest.raises(CardRejected, match="duration_s"):
        executor().execute(firing("coughing_fit", "P2", {"duration_s": -1}), view(), roster())


def test_unknown_template_key_rejected_at_load(tmp_path):
    p = _deck_patch(tmp_path, lambda cs: cs[0].update(nudge="Hello {nme}"))
    with pytest.raises(ConfigError, match=r"deck\.lost_visitor: unknown template key 'nme'"):
        load_scenario(p)


def test_condition_value_validated_at_load(tmp_path):
    p = _sb_patch(tmp_path / "a", lambda d: d["beats"][7]["when"].update(value="soon"))
    with pytest.raises(ConfigError, match=r"beats\[7\]\.when\.value"):
        load_scenario(p)
    p = _sb_patch(tmp_path / "b", lambda d: d["beats"][7].update(when={"kind": "posture_is", "person": "P4", "value": "flying"}))
    with pytest.raises(ConfigError, match=r"beats\[7\]\.when\.value"):
        load_scenario(p)


def test_skip_cascades_to_after_dependents():
    e = after_engine()
    e.skip("a")
    tl = {r["id"]: r for r in e.timeline()}
    assert tl["b"]["state"] == "skipped" and tl["b"]["reason"] == "predecessor a skipped"
    e.rerun("a")
    tl = {r["id"]: r for r in e.timeline()}
    assert tl["a"]["state"] == "pending" and tl["b"]["state"] == "pending"
    assert [f.card for f in e.tick(view(t=1200)) if f.card == "name_called"] == ["name_called"]
    assert [f.card for f in e.tick(view(t=1201)) if f.card == "new_arrival"] == ["new_arrival"]


def test_missing_target_skip_cascades():
    e = after_engine()
    e.tick(view(t=1200, present=ALL - {"P1"}))
    tl = {r["id"]: r for r in e.timeline()}
    assert tl["a"]["state"] == "skipped" and tl["b"]["reason"] == "predecessor a skipped"


def test_chaos_cap_of_three_per_tick():
    c = Chaos(3600.0 * 100, LS.deck, 3)
    assert len(c.tick(100.0, view(t=100.0))) == 3


def test_chaos_started_late_does_not_fire_a_backlog():
    late, base = Chaos(6.0, LS.deck, 7, t0=3600.0), Chaos(6.0, LS.deck, 7)
    assert late.tick(3600.0, view(t=3600.0)) == []
    a = [(f.t, f.card) for s in range(3600, 3 * 3600, 10) for f in late.tick(float(s), view(t=float(s)))]
    b = [(f.t + 3600.0, f.card) for s in range(0, 2 * 3600, 10) for f in base.tick(float(s), view(t=float(s)))]
    assert a and a == b
