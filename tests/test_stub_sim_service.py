"""StubSimService: the stub sim node without ROS."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control, Intent, Say, WorldEvent
from wardzero_agents.ros.stub_sim_node import OBS_WINDOW_S, ROBOT_ADDRESS_M, StubSimService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def svc(tmp_path, **kw):
    kw.setdefault("wait_for", ())
    return StubSimService(load_scenario(V1), RunLog(tmp_path), **kw)


def lines(tmp_path, stream):
    p = tmp_path / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_clock_follows_time_scale(tmp_path):
    s = svc(tmp_path, time_scale=30.0)
    out = s.step(0.01)
    assert out.clock == pytest.approx(0.3) and s.t == pytest.approx(0.3)


def test_waits_for_ready_signals(tmp_path):
    s = svc(tmp_path, wait_for=("agents", "director"))
    assert s.step(0.1).clock == 0.0 and s.waiting == {"agents", "director"}
    s.ready("agents")
    assert s.step(0.1).clock == 0.0
    s.ready("director")
    assert s.step(0.1).clock == pytest.approx(0.1)


def test_pause_and_lockstep_hold_the_clock(tmp_path):
    s = svc(tmp_path, lockstep=True)
    s.step(0.1)
    s.control(Control(action="pause"))
    assert s.step(0.1).clock == pytest.approx(0.1)
    s.control(Control(action="resume"))
    s.set_pending(2)
    assert s.step(0.1).clock == pytest.approx(0.1)
    s.set_pending(0)
    assert s.step(0.1).clock == pytest.approx(0.2)


def test_pending_ignored_without_lockstep(tmp_path):
    s = svc(tmp_path)
    s.set_pending(3)
    assert s.step(0.1).clock == pytest.approx(0.1)


def test_roster_only_on_change_and_status_rate(tmp_path):
    s = svc(tmp_path)
    first = s.step(0.05)
    assert first.roster is not None and [p.id for p in first.roster.people] == ["P1", "P2", "P3", "P4", "P5"]
    assert len(first.statuses) == 5
    second = s.step(0.05)
    assert second.roster is None and second.statuses == []  # 10 Hz: next status at t >= 0.15
    third = s.step(0.06)
    assert len(third.statuses) == 5
    s.on_world_event(WorldEvent(id="w", t=s.t, kind="arrival", persona="companion"))
    out = s.step(0.01)
    assert out.roster is not None and out.roster.people[-1].id == "N1"


def test_status_recorded_at_2hz(tmp_path):
    s = svc(tmp_path)
    for _ in range(100):  # 1 s of sim time
        s.step(0.01)
    recs = lines(tmp_path, "status")
    times = sorted({r["t"] for r in recs})
    assert 2 <= len(times) <= 3 and all(set(r) >= {"person", "posture", "schema"} for r in recs)


def test_observations_at_2hz_and_immediately_on_audio(tmp_path):
    s = svc(tmp_path)
    assert len(s.step(0.01).observations) == 5
    assert s.step(0.1).observations == []
    s.on_say(Say(person="P2", t=0.0, text="Is anyone sitting here?", addressee="P3"))
    out = s.step(0.01)
    assert len(out.audio) == 1 and out.audio[0].source_kind == "person"
    p3 = {o.person: o for o in out.observations}["P3"]
    assert [h.text for h in p3.heard] == ["Is anyone sitting here?"] and p3.heard[0].addressed_to_me
    assert lines(tmp_path, "audio")[0]["text"] == "Is anyone sitting here?"


def test_observation_keeps_audio_for_2s_with_identical_heard(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="c", person="P2", t=0.0, kind="cough", overlay=True))
    a = {o.person: o for o in s.step(0.01).observations}["P3"].heard
    later = []
    for _ in range(int(OBS_WINDOW_S / 0.1) - 2):
        later += [o for o in s.step(0.1).observations if o.person == "P3"]
    assert later and all(o.heard == a for o in later)
    for _ in range(10):
        s.step(0.1)
    assert all(o.heard == [] for o in s.step(0.5).observations)


def test_robot_hears_and_speech_is_addressed_to_the_nearest_person(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="c", person="P1", t=0.0, kind="cough", overlay=True))
    out = s.step(0.01)
    assert [h.source_kind for h in out.robot_heard] == ["cough"]
    # nobody near the robot: speech is addressed to no one
    s.on_robot_say("Hello everyone")
    out = s.step(0.01)
    assert out.audio[0].source_kind == "robot" and out.audio[0].addressee is None
    # P5 walks up to the robot
    s.on_intent(Intent(id="a", person="P5", t=s.t, kind="approach", target="g1"))
    for _ in range(400):
        s.step(0.05)
    st = {x.person: x for x in s.world.statuses()}["P5"]
    assert st.state == "done"
    s.on_robot_say("The clinics are down the clinic_corridor.")
    out = s.step(0.01)
    assert out.audio[0].addressee == "P5" and ROBOT_ADDRESS_M >= 1.0
    p5 = {o.person: o for o in out.observations}["P5"]
    assert p5.heard[-1].source_kind == "robot" and p5.heard[-1].addressed_to_me
    assert out.robot_heard == []  # the robot does not hear itself


def test_talk_intent_and_say_with_the_same_text_speak_once(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="t", person="P3", t=5.0, kind="talk", text="Help!", addressee="g1"))
    s.on_say(Say(person="P3", t=5.0, text="Help!", addressee="g1"))
    assert len(s.step(0.01).audio) == 1
    s.on_say(Say(person="P3", t=9.0, text="Help!", addressee="g1"))  # a new utterance
    assert len(s.step(0.01).audio) == 1


def test_audio_is_stamped_with_sim_time(tmp_path):
    s = svc(tmp_path)
    for _ in range(10):
        s.step(0.1)
    s.on_say(Say(person="P1", t=0.0, text="late say"))  # a decision that took a while
    now = s.t
    assert now == pytest.approx(1.0) and s.step(0.01).audio[0].t == pytest.approx(now)


def test_bad_input_is_ignored(tmp_path, caplog):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="x", person="P99", t=0.0, kind="sit_idle"))
    s.on_say(Say(person="P99", t=0.0, text="boo"))
    assert s.step(0.01).audio == []
    assert "P99" in caplog.text


def test_world_event_audio(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_world_event(WorldEvent(id="pa1", t=0.0, kind="pa_announcement", text="Clinics are late"))
    out = s.step(0.01)
    assert out.audio[0].source_kind == "pa" and out.robot_heard[0].text == "Clinics are late"


def test_ends_at_duration_and_on_stop(tmp_path):
    s = svc(tmp_path, time_scale=600.0)
    outs = [s.step(0.5) for _ in range(12)]  # 60 min at 300 s per step
    assert outs[-1].stop and not outs[-2].stop and outs[-1].clock == pytest.approx(3600.0)
    s2 = svc(tmp_path / "b")
    s2.control(Control(action="stop"))
    assert s2.step(0.01).stop


def test_no_robot_means_no_microphone_and_no_voice(tmp_path):
    loaded = load_scenario(V1)
    loaded.scenario = loaded.scenario.model_copy(update={"robot": loaded.scenario.robot.model_copy(
        update={"enabled": False})})
    s = StubSimService(loaded, RunLog(tmp_path), wait_for=())
    s.step(0.01)
    s.on_intent(Intent(id="c", person="P1", t=0.0, kind="cough", overlay=True))
    s.on_robot_say("Hello")
    out = s.step(0.01)
    assert out.robot_heard == [] and [a.source_kind for a in out.audio] == ["cough"]


def test_silent_pending_heartbeat_releases_the_clock(tmp_path):
    """A dead agent node cannot hang a lockstep clock: after 5 s of silence pending counts as 0."""
    wall = [100.0]
    s = svc(tmp_path, lockstep=True, wall=lambda: wall[0])
    s.set_pending(2)
    assert s.step(0.1).clock == 0.0
    wall[0] += 4.9
    assert s.step(0.1).clock == 0.0
    wall[0] += 0.2
    assert s.step(0.1).clock == pytest.approx(0.1)
    s.set_pending(1)  # the heartbeat is back: pending holds again
    assert s.step(0.1).clock == pytest.approx(0.1)
