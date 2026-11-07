"""Tests for the wz.v1 sim-boundary JSON contracts and ROS topic names."""
import json

import pytest
from pydantic import ValidationError

from wardzero_agents.contracts import (
    SCHEMA,
    AudioEvent,
    ContractError,
    Control,
    Heard,
    Intent,
    Nudge,
    Observation,
    ReflexOverride,
    Roster,
    RosterEntry,
    Say,
    Seen,
    Status,
    WorldEvent,
    from_json,
    to_json,
)
from wardzero_agents.ros import topics
from wardzero_agents.ros.topics import (
    AUDIO_EVENTS,
    CONTROL,
    FORBIDDEN_FOR_ROBOT,
    G1_HEARD,
    G1_SAY,
    PENDING,
    ROSTER,
    SCEN_EVENTS,
    TIMELINE,
    WORLD_EVENT,
    intent,
    nudge,
    observation,
    reflex,
    say,
    status,
)


def valid_intent_dict(**overrides):
    """A valid Intent payload as a dict, with keyword overrides applied."""
    data = {"schema": "wz.v1", "id": "a1", "person": "P1", "t": 0.0, "kind": "sit_idle"}
    data.update(overrides)
    return data


def to_json_dict_with(**overrides):
    """A valid Intent payload serialised to JSON text, with keyword overrides applied."""
    return json.dumps(valid_intent_dict(**overrides))


# --- Schema constant and envelope -------------------------------------------------------


def test_schema_constant():
    assert SCHEMA == "wz.v1"


def test_roundtrip_intent():
    i = Intent(id="a", person="P5", t=1.0, kind="walk_to", target="clinic_corridor")
    assert from_json(Intent, to_json(i)) == i


def test_to_json_uses_schema_alias_key():
    i = Intent(id="a", person="P5", t=1.0, kind="walk_to")
    payload = json.loads(to_json(i))
    assert payload["schema"] == "wz.v1"
    assert "schema_" not in payload


def test_models_build_in_python_without_schema_argument():
    i = Intent(id="a", person="P1", t=0.0, kind="sit_idle")
    assert i.schema_ == "wz.v1"


def test_missing_schema_rejected():
    with pytest.raises(ContractError, match="schema"):
        from_json(Intent, json.dumps({"id": "a", "person": "P1", "t": 0, "kind": "sit_idle"}))


def test_python_construction_without_schema_still_defaults():
    i = Intent(id="a", person="P1", t=0.0, kind="sit_idle")
    assert i.schema_ == "wz.v1"


def test_wrong_schema_message_names_the_value():
    with pytest.raises(ContractError, match="wz.v0"):
        from_json(Intent, '{"schema":"wz.v0","id":"a","person":"P1","t":0,"kind":"sit_idle"}')


def test_deeply_nested_json_raises_contract_error():
    deep = "[" * 200_000 + "]" * 200_000
    with pytest.raises(ContractError):
        from_json(Intent, deep)


def test_wrong_schema_rejected():
    with pytest.raises(ContractError):
        from_json(Intent, '{"schema":"wz.v0","id":"a","person":"P1","t":0,"kind":"sit_idle"}')


def test_unknown_intent_kind_rejected():
    with pytest.raises(ContractError):
        from_json(Intent, to_json_dict_with(kind="dance"))


def test_invalid_json_raises_contract_error():
    with pytest.raises(ContractError):
        from_json(Intent, "{not json")


def test_non_object_json_raises_contract_error():
    with pytest.raises(ContractError):
        from_json(Intent, "[1, 2, 3]")


def test_missing_required_field_raises_contract_error():
    with pytest.raises(ContractError):
        from_json(Intent, json.dumps({"schema": "wz.v1", "person": "P1", "t": 0, "kind": "sit_idle"}))


def test_contract_error_is_value_error():
    assert issubclass(ContractError, ValueError)


def test_extra_fields_rejected():
    with pytest.raises(ContractError):
        from_json(Intent, to_json_dict_with(unexpected=1))


def test_contract_error_chains_pydantic_error():
    with pytest.raises(ContractError) as info:
        from_json(Intent, to_json_dict_with(kind="dance"))
    assert isinstance(info.value.__cause__, ValidationError)


# --- Intent -----------------------------------------------------------------------------


def test_intent_optional_fields_default_to_none_or_false():
    i = Intent(id="a", person="P1", t=0.0, kind="sit_idle")
    assert i.target is None
    assert i.point is None
    assert i.speed is None
    assert i.style is None
    assert i.text is None
    assert i.addressee is None
    assert i.overlay is False


def test_intent_point_roundtrip_as_tuple():
    i = Intent(id="a", person="P1", t=0.0, kind="walk_to", point=(1.5, -2.0))
    back = from_json(Intent, to_json(i))
    assert back.point == (1.5, -2.0)


# --- Status -----------------------------------------------------------------------------


def test_roundtrip_status():
    s = Status(
        person="P2", t=3.5, x=1.0, y=2.0, yaw=0.5, posture="walking",
        intent_id="a1", intent_kind="walk_to", progress=0.4, state="running",
    )
    assert from_json(Status, to_json(s)) == s


def test_status_reason_optional():
    s = Status(
        person="P2", t=0.0, x=0.0, y=0.0, yaw=0.0, posture="seated",
        intent_id=None, intent_kind=None, progress=0.0, state="idle",
    )
    assert s.reason is None


def test_status_rejects_unknown_posture():
    payload = {
        "schema": "wz.v1", "person": "P2", "t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0,
        "posture": "crouching", "intent_id": None, "intent_kind": None,
        "progress": 0.0, "state": "idle",
    }
    with pytest.raises(ContractError):
        from_json(Status, json.dumps(payload))


# --- Observation, Seen, Heard -----------------------------------------------------------


def test_roundtrip_observation_with_nested_models():
    obs = Observation(
        person="P1",
        t=2.0,
        visible=[Seen(kind="sign", id="s1", distance=4.0, bearing_deg=-10.0, detail="Clinic")],
        heard=[
            Heard(
                source_kind="pa", source_id="pa1", text="Please wait",
                received_db=55.0, partial=False, addressed_to_me=False,
            )
        ],
    )
    back = from_json(Observation, to_json(obs))
    assert back == obs
    assert isinstance(back.visible[0], Seen)
    assert isinstance(back.heard[0], Heard)


def test_seen_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        Seen(kind="dog", id="d1", distance=1.0, bearing_deg=0.0)


def test_heard_rejects_unknown_source_kind():
    with pytest.raises(ValidationError):
        Heard(
            source_kind="radio", source_id="r", text=None,
            received_db=40.0, partial=False, addressed_to_me=False,
        )


def test_heard_round_trips_with_and_without_a_bearing():
    plain = Heard(source_kind="cough", source_id="unknown", received_db=50.0, partial=False, addressed_to_me=False)
    assert plain.bearing_deg is None and from_json(Heard, to_json(plain)) == plain
    aimed = plain.model_copy(update={"bearing_deg": 31.5})
    back = from_json(Heard, to_json(aimed))
    assert back == aimed and back.bearing_deg == 31.5
    with pytest.raises(ValidationError):
        Heard(source_kind="cough", source_id="unknown", received_db=50.0, partial=False, addressed_to_me=False,
              direction="left")


def test_observation_defaults_to_empty_lists():
    obs = Observation(person="P1", t=0.0)
    assert obs.visible == []
    assert obs.heard == []


# --- AudioEvent, Say, Nudge, ReflexOverride ---------------------------------------------


def test_roundtrip_audio_event():
    e = AudioEvent(
        id="e1", t=1.0, source_kind="cough", source_id="P3",
        x=1.0, y=2.0, z=1.1, loudness_db=62.0,
    )
    assert from_json(AudioEvent, to_json(e)) == e


def test_roundtrip_say():
    s = Say(person="P4", t=5.0, text="Excuse me", addressee="robot")
    assert from_json(Say, to_json(s)) == s


def test_roundtrip_nudge():
    n = Nudge(person="P4", t=5.0, card="waiting", message="Stay seated")
    assert from_json(Nudge, to_json(n)) == n


def test_roundtrip_reflex_override():
    r = ReflexOverride(person="P3", t=6.0, kind="cough_bout", duration_s=4.0, severity="moderate")
    assert from_json(ReflexOverride, to_json(r)) == r


def test_reflex_override_rejects_unknown_severity():
    with pytest.raises(ValidationError):
        ReflexOverride(person="P3", t=6.0, kind="cough_bout", duration_s=4.0, severity="extreme")


# --- WorldEvent, Roster, Control --------------------------------------------------------


def test_roundtrip_world_event():
    w = WorldEvent(id="w1", t=10.0, kind="pa_announcement", text="Clinic 3 is ready")
    assert from_json(WorldEvent, to_json(w)) == w


def test_world_event_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        WorldEvent(id="w1", t=10.0, kind="earthquake")


def test_roundtrip_roster():
    r = Roster(
        t=0.0,
        people=[
            RosterEntry(id="P1", persona="elderly_cough", display_name="Mary", seat="S1", companion_of=None),
            RosterEntry(id="P2", persona="companion", display_name="Tom", seat=None, companion_of="P1"),
        ],
    )
    assert from_json(Roster, to_json(r)) == r


def test_roster_entry_optional_fields_default_to_none():
    e = RosterEntry(id="P1", persona="visitor", display_name="Ann")
    assert e.seat is None
    assert e.companion_of is None


@pytest.mark.parametrize("action", ["start", "pause", "resume", "stop", "reset"])
def test_control_accepts_each_action(action):
    c = Control(action=action)
    assert from_json(Control, to_json(c)) == c


def test_control_rejects_unknown_action():
    with pytest.raises(ContractError):
        from_json(Control, '{"schema":"wz.v1","action":"explode"}')


# --- Topic names ------------------------------------------------------------------------


def test_topic_names():
    assert intent("P1") == "/humans/P1/intent"
    assert G1_HEARD == "/g1/audio/heard"


def test_per_person_topic_functions():
    assert status("P7") == "/humans/P7/status"
    assert observation("P7") == "/humans/P7/observation"
    assert say("P7") == "/humans/P7/say"
    assert nudge("P7") == "/humans/P7/nudge"
    assert reflex("P7") == "/humans/P7/reflex"


def test_fixed_topic_constants():
    assert ROSTER == "/humans/roster"
    assert AUDIO_EVENTS == "/sim/audio/events"
    assert G1_SAY == "/g1/speech/say"
    assert WORLD_EVENT == "/sim/world_event"
    assert TIMELINE == "/scenario/timeline"
    assert SCEN_EVENTS == "/scenario/events"
    assert CONTROL == "/scenario/control"
    assert PENDING == "/agents/pending"


def test_forbidden_for_robot_prefixes():
    assert FORBIDDEN_FOR_ROBOT == (
        "/humans/",
        "/scenario/",
        "/sim/audio/events",
        "/sim/world_event",
        "/agents/pending",
    )


def test_topics_module_has_no_rclpy_import():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(topics))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "rclpy" not in imported


def test_heard_takes_the_bell_kind_for_the_call_system():
    bell = Heard(source_kind="bell", source_id="unknown", text="call bell, bed B3", received_db=47.0, partial=False,
                 addressed_to_me=False, bearing_deg=12.0)
    assert from_json(Heard, to_json(bell)) == bell
