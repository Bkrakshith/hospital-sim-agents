"""A walk counts as the robot's work only when the guide ended with the person at the place (``guided``): a guide that
started, or that ended held up by someone, is not credited in any preset flow that credits a walk."""
from pathlib import Path

import yaml

from wardzero_agents.config.loader import _load_deck
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Roster, RosterEntry, Status
from wardzero_agents.flows.spec import flows_dir

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = load_scenario_presets(CONFIG)
DECK = _load_deck(CONFIG / "cards" / "studio.yaml")


def _director(tmp_path, preset):
    from wardzero_agents.ros.director_node import DirectorService
    from wardzero_agents.runlog import RunLog

    class Out:
        def publish(self, kind, obj):
            pass

    loaded = preset_to_scenario(PRESETS[preset], CONFIG, tmp_path / "staged")
    d = DirectorService(loaded, RunLog(tmp_path / "run"), Out(), run_id="r1")
    d.on_roster(Roster(t=0, people=[RosterEntry(id=i.person_id, persona=i.persona_id, display_name=i.display_name)
                                    for i in loaded.instances]))
    return d, loaded


def _status(d, pid, x, y, t, posture="standing"):
    d.on_status(Status(person=pid, t=t, x=x, y=y, yaw=0.0, posture=posture, progress=0.0, state="idle"))


def _node(tree: dict, label: str) -> dict:
    if tree["label"] == label:
        return tree
    for c in tree["children"]:
        try:
            return _node(c, label)
        except KeyError:
            pass
    raise KeyError(label)


def _walk_actions(cond) -> list:
    """Every robot_did action named in a condition."""
    if isinstance(cond, dict):
        out = []
        for k, v in cond.items():
            if k == "robot_did" and isinstance(v, dict):
                out.append((v.get("action"), v.get("about")))
            else:
                out += _walk_actions(v)
        return out
    if isinstance(cond, list):
        return [x for c in cond for x in _walk_actions(c)]
    return []


def _expects(steps) -> list:
    out = []
    for s in steps:
        k, v = next(iter(s.items()))
        if k == "expect":
            out.append(v)
        elif k == "parallel":
            for b in v["branches"]:
                out += _expects(b)
        elif k == "unless":
            out += _expects(v.get("then") or [])
    return out


# ---- reception: H1's walk, in process through the director ----------------------------------------------------
REC_H1 = "The robot walks H1 to the clinic corridor"
REC_V1_PREFIX = "The robot walks V1"


def _reception_to_h1(tmp_path, guide_end: str):
    """Run the reception flow in the director: greeting, V1 walked to the ward entrance (arrived), H1's card fires,
    then a guide with H1 that starts and ends with `guide_end`. Returns the flow tree after the H1 expect's window."""
    d, loaded = _director(tmp_path, "reception_guide")
    where = {c.id: (c.x, c.y) for c in PRESETS["reception_guide"].characters}
    t = 0.0
    for pid, (x, y) in where.items():
        _status(d, pid, x, y, t)
    d.robot_xy = (where["V1"][0] + 1.0, where["V1"][1])
    d.ingest("audio", {"source_kind": "robot", "addressee": "V1", "text": "Hello", "heard_by": ["V1"], "t": 0.5})
    t = 0.5
    while t < 3.0:
        d.tick(t)
        t += 0.5
    d.ingest("robot_tasks", {"kind": "guide", "event": "start", "id": "a1", "person": "V1", "place": "ward_entrance", "x": 13.2, "y": 7.25,
                             "t": 3.0})
    d.ingest("robot_tasks", {"kind": "guide", "event": "end", "id": "a1", "person": "V1", "reason": "arrived",
                             "t": 20.0})
    _status(d, "V1", 13.2, 7.25, 20.0)  # at the ward entrance
    h1_fired = None
    while t < 400.0:
        d.tick(t)
        if h1_fired is None and any(ev["text"].startswith("hard_of_hearing fired") for ev in d.events):
            h1_fired = t
            d.ingest("robot_tasks", {"kind": "guide", "event": "start", "id": "a2", "person": "H1", "x": 5.6, "y": 5.4,
                                     "place": "clinic_corridor", "t": t + 2.0})
            d.ingest("robot_tasks", {"kind": "guide", "event": "end", "id": "a2", "person": "H1",
                                     "reason": guide_end, "t": t + 30.0})
            if guide_end == "arrived":
                _status(d, "H1", 5.6, 5.4, t + 30.0)  # in the clinic corridor with the robot
            if guide_end != "arrived":
                # H1 walks on to the clinics on his own after the robot gave up: not the robot's work
                _status(d, "H1", 30.0, 4.0, t + 40.0, "walking")
        t += 0.5
    assert h1_fired is not None, "the hard_of_hearing card never fired"
    return d.engine.tree()


def test_a_guide_that_ends_blocked_does_not_credit_walking_h1(tmp_path):
    tree = _reception_to_h1(tmp_path, "blocked_by_person")
    assert _node(tree, REC_H1)["state"] == "timed_out"


def test_a_guide_that_arrives_credits_walking_h1(tmp_path):
    tree = _reception_to_h1(tmp_path, "arrived")
    assert _node(tree, REC_H1)["state"] == "met"


def test_v1_is_credited_only_for_a_walk_that_arrived(tmp_path):
    for end, want in (("blocked_by_person", "timed_out"), ("arrived", "met")):
        d, loaded = _director(tmp_path / end, "reception_guide")
        where = {c.id: (c.x, c.y) for c in PRESETS["reception_guide"].characters}
        for pid, (x, y) in where.items():
            _status(d, pid, x, y, 0.0)
        d.robot_xy = (where["V1"][0] + 1.0, where["V1"][1])
        d.ingest("audio", {"source_kind": "robot", "addressee": "V1", "text": "Hello", "heard_by": ["V1"], "t": 0.5})
        t = 0.5
        while t < 150.0:
            d.tick(t)
            if t == 3.0:
                d.ingest("robot_tasks", {"kind": "guide", "event": "start", "id": "a1", "person": "V1",
                                         "place": "ward_entrance", "x": 13.2, "y": 7.25, "t": 3.0})
                d.ingest("robot_tasks", {"kind": "guide", "event": "end", "id": "a1", "person": "V1",
                                         "reason": end, "t": 12.0})
                _status(d, "V1", 13.2, 7.25, 12.0)  # the walk's end at the ward entrance
                d.ingest("audio", {"source_kind": "robot", "addressee": "V1", "text": "This way",
                                   "heard_by": ["V1"], "t": 3.0})
            if t == 30.0:  # V1 walks off to the wards on her own once the robot stopped
                _status(d, "V1", 30.0, 4.0, t, "walking")
            t += 0.5
        v1 = next(e for e in _expects(yaml.safe_load((flows_dir(CONFIG) / "reception_guide.yaml").read_text())["steps"])
                  if e["label"].startswith(REC_V1_PREFIX))
        assert _node(d.engine.tree(), v1["label"])["state"] == want, end


# ---- every preset flow: an expect that credits a walk credits only an arrived one ---------------------------------
WALK_CREDIT = {
    # flow id -> {expect label: people whose walk it credits}
    "reception_guide": {"V1", "H1"},
    "evacuation_guide": {"V1", "V2", "P1"},
    "ae_waiting": {"P1", "P2"},
    "wandering_patient": {"W1"},
}
# gates that only need the robot to have set off with someone may keep `guide` (contract item 10)
STARTED_GATES = {("evacuation_guide", "The robot helps V2")}


def test_flows_credit_a_walk_only_when_it_arrived():
    for fid in sorted(p.stem for p in flows_dir(CONFIG).glob("*.yaml")):
        doc = yaml.safe_load((flows_dir(CONFIG) / f"{fid}.yaml").read_text())
        for e in _expects(doc["steps"]):
            if (fid, e["label"]) in STARTED_GATES:
                continue
            started = [who for act, who in _walk_actions(e["when"]) if act == "guide"]
            assert not started, f"{fid}: '{e['label']}' credits a guide that only started ({started})"
    # and the flows that walk people do credit an arrived walk
    for fid, people in WALK_CREDIT.items():
        doc = yaml.safe_load((flows_dir(CONFIG) / f"{fid}.yaml").read_text())
        guided = {who for e in _expects(doc["steps"]) for act, who in _walk_actions(e["when"]) if act == "guided"}
        assert people <= guided, f"{fid}: no arrived-walk credit for {sorted(people - guided)}"


# ---- night_call_bell: a nurse at the bedside beside the robot has come ----------------------------------------------
def test_night_call_bell_nurse_beside_a_robot_at_the_bed_stops_the_second_ring(tmp_path):
    from wardzero_agents.flows.engine import FlowEngine, FlowView
    from wardzero_agents.flows.spec import load_flow
    from wardzero_agents.site import load_site
    site = load_site(CONFIG)
    p = PRESETS["night_call_bell"]
    start = {c.id: (c.x, c.y) for c in p.characters}
    px, py = start["P1"]

    def run(nurse_gap):
        e = FlowEngine(load_flow(flows_dir(CONFIG) / "night_call_bell.yaml"), DECK)
        fired, t = [], 0.0
        while t <= p.duration_min * 60 and not e.finished:
            where = dict(start)
            if t >= 45.0:  # the nurse the robot called stops beside it at the bed (re-QA: 1.68 to 1.71 m off P1)
                where["N1"] = (px + nurse_gap, py)
            sts = {k: Status(person=k, t=t, x=x, y=y, yaw=0.0, posture="standing", progress=0.0, state="idle")
                   for k, (x, y) in where.items()}
            fired += [f.card for f in e.tick(FlowView(t=t, statuses=sts, robot_xy=(px - 1.0, py), present=set(sts),
                                                      cast=set(sts), room_at=site.room_at,
                                                      robot_did=[(40.0, "press_call", None, None)] if t >= 40 else [],
                                                      robot_said_to={"P1"} if t >= 30 else set(),
                                                      robot_said_at={"P1": [30.0]} if t >= 30 else {}))]
            t += 0.5
        return fired
    assert run(1.75).count("ring_bell") == 1
    assert run(3.0).count("ring_bell") == 2  # she never came to the bed: P1 rings again


def test_a_guide_to_the_wrong_place_does_not_credit_walking_v1(tmp_path):
    """Third QA: V1 (needs the ward) walked to the toilets, or to the main entrance in the lobby; the walk arrived,
    and the flow credited it as 'where they need to go'. The director records the room of the guide's place."""
    for place, xy, want in (("toilets", (-11.5, 1.2), "timed_out"), ("entrance", (-1.05, -1.84), "timed_out"),
                            ("Ward entrance", (13.2, 7.25), "met")):
        d, loaded = _director(tmp_path / place.replace(" ", "_"), "reception_guide")
        where = {c.id: (c.x, c.y) for c in PRESETS["reception_guide"].characters}
        for pid, (x, y) in where.items():
            _status(d, pid, x, y, 0.0)
        d.robot_xy = (where["V1"][0] + 1.0, where["V1"][1])
        d.ingest("audio", {"source_kind": "robot", "addressee": "V1", "text": "Hello", "heard_by": ["V1"], "t": 0.5})
        t = 0.5
        while t < 150.0:
            d.tick(t)
            if t == 3.0:
                d.ingest("robot_tasks", {"kind": "guide", "event": "start", "id": "a1", "person": "V1",
                                         "place": place, "x": xy[0], "y": xy[1], "t": 3.0})
                d.ingest("robot_tasks", {"kind": "guide", "event": "end", "id": "a1", "person": "V1",
                                         "reason": "arrived", "t": 12.0})
            t += 0.5
        v1 = next(e for e in _expects(yaml.safe_load((flows_dir(CONFIG) / "reception_guide.yaml").read_text())["steps"])
                  if e["label"].startswith(REC_V1_PREFIX))
        assert _node(d.engine.tree(), v1["label"])["state"] == want, place
