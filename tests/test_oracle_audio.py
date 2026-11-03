import numpy as np

from wardzero_agents.config.models import Sign
from wardzero_agents.contracts import AudioEvent
from wardzero_agents.world.audio import (
    ALARM_RECEIVED_DB,
    PA_RECEIVED_DB,
    ROBOT_SPEECH_DB,
    received_db,
    route,
)
from wardzero_agents.world.oracle import Entity, Pose, visible


def rng(seed=1):
    return np.random.default_rng(seed)


def ev(loudness, at=(0.0, 0.0), text="hello there", kind="robot", source_id="g1", addressee=None):
    return AudioEvent(
        id="e1", t=0.0, source_kind=kind, source_id=source_id, x=at[0], y=at[1], z=1.5,
        loudness_db=loudness, text=text, addressee=addressee,
    )


def test_behind_viewer_not_seen():
    assert visible(Pose(0, 0, 0), [Entity("robot", "g1", -3, 0, None)], []) == []


def test_in_front_seen_with_left_positive_bearing():
    seen = visible(Pose(0, 0, 0), [Entity("person", "P2", 3, 1, "seated")], [])
    assert len(seen) == 1 and seen[0].bearing_deg > 0 and seen[0].detail == "seated"
    seen = visible(Pose(0, 0, 0), [Entity("person", "P2", 3, -1, None)], [])
    assert seen[0].bearing_deg < 0


def test_out_of_range_not_seen():
    assert visible(Pose(0, 0, 0), [Entity("person", "P2", 13, 0, None)], []) == []


def test_sign_text_only_when_close():
    far = Sign(id="s", text="Toilets", x=8, y=0, z=2, facing_yaw=3.14)
    assert visible(Pose(0, 0, 0), [], [far]) == []
    near = far.model_copy(update={"x": 4})
    seen = visible(Pose(0, 0, 0), [], [near])
    assert seen[0].detail == "Toilets" and seen[0].kind == "sign" and seen[0].id == "s"


def test_sign_facing_away_not_seen():
    away = Sign(id="s", text="Toilets", x=4, y=0, z=2, facing_yaw=0.0)
    assert visible(Pose(0, 0, 0), [], [away]) == []


def test_los_blocks():
    e = Entity("person", "P2", 2, 0, None)
    assert visible(Pose(0, 0, 0), [e], [], los=lambda a, b: False) == []


def test_quiet_speech_partial_for_hearing_loss():
    h = route(ev(ROBOT_SPEECH_DB, at=(4, 0), text="clinic three is down the corridor"),
              Pose(0, 0, 0), "P4", 20, rng(1))
    assert h.partial and "…" in h.text
    assert any(w != "…" for w in h.text.split())


def test_loud_close_speech_complete_and_addressed():
    h = route(ev(ROBOT_SPEECH_DB, at=(2, 0), text="hello", addressee="P1"),
              Pose(0, 0, 0), "P1", 0, rng(1))
    assert not h.partial and h.text == "hello" and h.addressed_to_me
    assert h.source_kind == "robot" and h.source_id == "g1"


def test_far_speech_not_heard():
    assert route(ev(60, at=(40, 0)), Pose(0, 0, 0), "P1", 0, rng(1)) is None


def test_pa_heard_everywhere():
    h = route(ev(90, at=(30, 0), kind="pa", source_id="pa"), Pose(-8, 0, 0), "P1", 0, rng(1))
    assert h.received_db == PA_RECEIVED_DB == 62


def test_alarm_fixed_level():
    h = route(ev(10, at=(30, 0), kind="alarm", text=None), Pose(0, 0, 0), "P1", 0, rng(1))
    assert h.received_db == ALARM_RECEIVED_DB


def test_cough_has_no_text():
    h = route(ev(68, at=(1, 0), kind="cough", text="ahem", source_id="P2"), Pose(0, 0, 0), "P1", 0, rng(1))
    assert h.text is None and h.source_kind == "cough"


def test_received_db_distance():
    assert received_db(60, 10) == 40
    assert received_db(60, 0.2) == 60
