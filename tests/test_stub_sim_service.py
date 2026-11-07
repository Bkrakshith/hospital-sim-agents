"""StubSimService: the stub sim node without ROS."""
import json
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Control, Intent, Say, WorldEvent
from wardzero_agents.ros.stub_sim_node import OBS_WINDOW_S, StubSimService
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


def test_robot_hears_and_speech_is_addressed_only_by_name_or_reply(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="c", person="P1", t=0.0, kind="cough", overlay=True))
    out = s.step(0.01)
    assert [h.source_kind for h in out.robot_heard] == ["cough"]
    assert [h.source_id for h in out.robot_heard] == ["unknown"]  # not P1: the robot cannot tell who
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
    s.on_robot_say("Hello there.")  # P5 stands at the robot but never spoke to it, and it named nobody
    assert s.step(0.01).audio[0].addressee is None
    s.on_robot_toward(s.world.tracks.track("P5"))
    s.on_robot_say("The clinics are down the clinic_corridor.")
    out = s.step(0.01)
    assert out.audio[0].addressee == "P5"
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


def test_robot_reply_goes_to_the_person_it_heard_not_the_nearest(tmp_path):
    from wardzero_agents.ros.stub_sim_node import REPLY_WINDOW_S
    s = svc(tmp_path)
    s.step(0.01)
    s.on_intent(Intent(id="a", person="P5", t=s.t, kind="approach", target="g1"))
    for _ in range(400):
        s.step(0.05)
    # P2 speaks from 2.3 m away (the reply-to-last-listener rule reaches 3.0 m); P5 is the one standing at the robot
    _put(s.world, "P2", -1.2, 1.0)
    s.on_say(Say(person="P2", t=s.t, text="Where is Ward B please?", addressee="g1"))
    out = s.step(0.01)
    assert [h.source_id for h in out.robot_heard if h.text] == ["unknown"]  # the robot cannot tell who
    s.on_robot_say("Head for the wards.")
    assert s.step(0.01).audio[0].addressee == "P2"
    # one reply per utterance: the next line, with no name, is for nobody in particular (not the nearest person)
    s.on_robot_say("Anything else?")
    assert s.step(0.01).audio[0].addressee is None
    # a stale utterance is not answered by name
    s.on_say(Say(person="P2", t=s.t, text="Hello again", addressee="g1"))
    s.step(0.01)
    s.t += REPLY_WINDOW_S + 1
    s.on_robot_say("Hello")
    assert s.step(0.01).audio[0].addressee is None


def test_objects_stream_and_alarm_audio(tmp_path):
    import shutil
    import yaml
    cfg = tmp_path / "config"
    shutil.copytree(V1.parents[1], cfg)
    q = cfg / "scenarios" / "waiting_area_v1.yaml"
    data = yaml.safe_load(q.read_text())
    data["cast"][0].pop("seat")
    data["cast"][0]["start"] = {"bed": "B2"}
    q.write_text(yaml.safe_dump(data, sort_keys=False))
    run = tmp_path / "run"
    s = StubSimService(load_scenario(q), RunLog(run), wait_for=())
    s.on_world_event(WorldEvent(id="f", t=0, kind="fire_start", x=3.0, y=4.0))
    s.world.objects.fires["fire_1"].size = 0.995
    s.on_intent(Intent(id="b", person="P1", t=0, kind="call_bell"))
    heard_kinds, t = [], 0
    for _ in range(30):
        out = s.step(0.1)
        heard_kinds += [e.source_kind for e in out.audio]
    assert "alarm" in heard_kinds
    rows = lines(run, "objects")
    kinds = {o["kind"] for r in rows for o in r["objects"]}
    assert rows and {"fire", "call_bell"} <= kinds
    assert any(r["source_kind"] == "alarm" for r in lines(run, "audio"))


def test_robot_pose_recorded_at_status_rate_with_a_robot_only(tmp_path):
    s = svc(tmp_path)
    for _ in range(12):
        s.step(0.1)
    rows = lines(tmp_path, "robot_pose")
    if s.robot:
        assert rows and set(rows[0]) >= {"t", "x", "y", "yaw", "speed"}
        assert [r["t"] for r in rows] == sorted({r["t"] for r in rows})
        assert len(rows) == len({r["t"] for r in lines(tmp_path, "status")})
    else:
        assert rows == []


def test_no_robot_pose_without_a_robot(tmp_path):
    loaded = load_scenario(V1)
    loaded.scenario = loaded.scenario.model_copy(update={"robot": loaded.scenario.robot.model_copy(
        update={"enabled": False})})
    s = StubSimService(loaded, RunLog(tmp_path), wait_for=())
    for _ in range(12):
        s.step(0.1)
    assert lines(tmp_path, "robot_pose") == [] and lines(tmp_path, "status")


# ---- the robot names who it speaks to -----------------------------------------------------------------------------
def _put(w, pid, x, y):
    p = w.people[pid]
    p.x, p.y, p.posture, p.seat, p.plan = x, y, "standing", None, []


def _staged_people(tmp_path):
    s = svc(tmp_path)
    s.step(0.01)
    w = s.world
    _put(w, "P3", 1.5, 0.6)  # 1.1 m from the robot
    _put(w, "P2", 1.05, -1.38)  # 3 m straight ahead
    s.step(0.01)
    return s, w


def test_robot_speech_goes_to_the_named_track(tmp_path):
    s, w = _staged_people(tmp_path)
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    s.on_robot_toward(w.tracks.track("P2"))
    s.on_robot_say("Hello, can I help you?")
    assert s.step(0.01).audio[0].addressee == "P2"
    robot_rows = [r for r in lines(tmp_path, "audio") if r["source_kind"] == "robot"]
    assert robot_rows[-1]["addressee"] == "P2"
    # one name per line: the next line without a name goes by the usual rule (P3's question was answered by the
    # named line, so nobody in particular; never the nearest person)
    s.on_robot_say("Anything else?")
    assert s.step(0.01).audio[0].addressee is None


def test_a_name_that_arrives_after_its_words_still_addresses_them(tmp_path):
    """Over ROS the say callback often runs before the toward callback: the words wait for the step and take the
    name that arrived by then. speech/toward is a FIFO (each toward is the next say's): a name that comes only after
    its words were sent goes with the next line, if that comes within 2 s."""
    s, w = _staged_people(tmp_path)
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    s.on_robot_say("Hello, can I help you?")
    s.on_robot_toward(w.tracks.track("P2"))
    assert s.step(0.01).audio[0].addressee == "P2"
    s.on_robot_say("Over there.")  # no name, and P3's question was answered by the named line: nobody in particular
    assert s.step(0.01).audio[0].addressee is None
    s.on_robot_toward(w.tracks.track("P2"))  # after "Over there." went out: the next line's
    s.step(0.01)
    s.on_robot_say("Anything else?")
    assert s.step(0.01).audio[0].addressee == "P2"


def test_overheard_speech_does_not_capture_the_reply(tmp_path):
    s, w = _staged_people(tmp_path)
    s.on_say(Say(person="P2", t=s.t, text="Shall we sit down?", addressee="P3"))
    out = s.step(0.01)
    assert out.robot_heard and s._last_robot_listener is None
    s.on_say(Say(person="P2", t=s.t, text="Robot, where is the toilet?", addressee="g1"))
    s.step(0.01)
    assert s._last_robot_listener is not None and s._last_robot_listener[1] == "P2"
    s.on_say(Say(person="P2", t=s.t + 1, text="Did you hear me?", addressee=None))  # to nobody in particular
    s.step(0.01)
    assert s._last_robot_listener[1] == "P2"


def test_a_far_track_goes_to_nobody_not_the_nearest(tmp_path):
    s, w = _staged_people(tmp_path)
    _put(w, "P2", 1.05, -3.4)  # 5 m ahead: in view, but too far to be spoken to
    _put(w, "P5", 1.05, 6.0)  # 4.4 m behind the robot: too far as well
    s.step(0.01)
    for pid in ("P2", "P5"):
        s.on_robot_toward(w.tracks.track(pid))
        s.on_robot_say("Hello there.")
        assert s.step(0.01).audio[0].addressee is None, pid  # never P3, who stands nearest
    _put(w, "P2", 1.05, -1.38)
    s.step(0.01)
    s.on_robot_toward(w.tracks.track("P2"))
    s.t += 3.5  # a stale name
    s.on_robot_say("Hello there.")
    assert s.step(0.01).audio[0].addressee is None


def test_brief_carries_desk_staffed(tmp_path):
    from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
    from wardzero_agents.studio import auto_staff
    config = Path(__file__).resolve().parents[1] / "config"
    presets = load_scenario_presets(config)
    meal = StubSimService(preset_to_scenario(presets["meal_round"], config, tmp_path / "m"), RunLog(tmp_path / "rm"),
                          wait_for=())
    assert meal.robot_brief["desk_staffed"] is False and meal.robot_brief["meals"]
    ae = presets["ae_waiting"]
    loaded = preset_to_scenario(ae, config, tmp_path / "a", characters=auto_staff.with_auto(list(ae.characters), ae))
    s = StubSimService(loaded, RunLog(tmp_path / "ra"), wait_for=())
    assert s.robot_brief["desk_staffed"] is True
    s.world.people["RCP"].kind = "static"  # a receptionist who cannot move or read the desk computer
    assert s.robot_brief["desk_staffed"] is False
