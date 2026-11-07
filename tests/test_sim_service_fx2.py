"""The speech wire after the re-QA: a chime reaches the robot as source_kind "bell"; every person line's audio row says
whether the robot's microphone got it (robot_heard); a robot line goes to the track its speech/toward names and to
nobody else, and with no toward to the last person who spoke to it within 3 m, never to whoever stands nearest."""
import json
import math
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import Intent, Say, WorldEvent
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
V1 = ROOT / "config" / "scenarios" / "waiting_area_v1.yaml"


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    cache = {}
    presets = load_scenario_presets(CONFIG)

    def get(preset_id):
        if preset_id not in cache:
            p = presets[preset_id]
            cache[preset_id] = preset_to_scenario(p, CONFIG, tmp_path_factory.mktemp(preset_id),
                                                  characters=auto_staff.with_auto(list(p.characters), p))
        return cache[preset_id]
    return get


def rows(run, stream):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _put(w, pid, x, y):
    p = w.people[pid]
    p.x, p.y, p.posture, p.seat, p.plan = x, y, "standing", None, []


def _lobby(tmp_path):
    """The waiting-area lobby, the robot at (1.05, 1.62) facing -y: P3 1.1 m away to its side, P2 3 m ahead."""
    s = StubSimService(load_scenario(V1), RunLog(tmp_path), wait_for=())
    s.step(0.01)
    w = s.world
    _put(w, "P3", 1.5, 0.6)
    _put(w, "P2", 1.05, -1.38)
    s.step(0.01)
    return s, w


def _robot_line(s, text="Hello there.", toward=None):
    if toward is not None:
        s.on_robot_toward(toward)
    s.on_robot_say(text)
    return next(a for a in s.step(0.01).audio if a.source_kind == "robot")


# ---- chimes -------------------------------------------------------------------------------------------------------
def test_the_robot_hears_a_bell_and_its_own_staff_call_as_the_bell_kind(staged, tmp_path):
    s = StubSimService(staged("night_call_bell"), RunLog(tmp_path / "run"), wait_for=())
    w = s.world
    bed = w.people["P1"].bed
    chimes = w.apply_world_event(WorldEvent(id="rb", t=0.0, kind="ring_bell", person="P1"))
    for ev in chimes:  # the robot 3 m from each chime, so it hears it
        w.robot.teleport(ev.x + 3.0, ev.y, math.pi)
        if not w.robot.router.is_free(w.robot.x, w.robot.y):
            w.robot.teleport(ev.x, ev.y + 3.0, -math.pi / 2)
        h = w.robot_heard(ev)
        assert h is not None and h.source_kind == "bell" and h.text == f"call bell, bed {bed}"
        assert h.source_id == "unknown"
    own = w._event(1.0, "person", "bell_robot", w.robot.x + 1, w.robot.y, 2.0, 55.0)
    h = w.robot_heard(own)
    assert h.source_kind == "bell" and h.text == "staff call chime" and h.bearing_deg is None
    # people still hear a bell as the call system's sound, unchanged
    assert all(e.source_kind == "person" and e.text is None for e in chimes)


def test_a_person_speaking_is_still_a_person_to_the_robot(tmp_path):
    s, w = _lobby(tmp_path)
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    out = s.step(0.01)
    assert [h.source_kind for h in out.robot_heard] == ["person"]


# ---- robot_heard on person rows -----------------------------------------------------------------------------------
def test_every_person_line_says_whether_the_robot_heard_it(tmp_path):
    s, w = _lobby(tmp_path)
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    w.robot_heard = lambda ev: None  # the robot's microphone misses the next line
    s.on_say(Say(person="P2", t=s.t, text="Is anyone there?", addressee="g1"))
    s.step(0.01)
    s.on_intent(Intent(id="c", person="P2", t=s.t, kind="cough", overlay=True))
    s.step(0.01)
    s.on_robot_say("Hello.")
    s.step(0.01)
    person = [r for r in rows(tmp_path, "audio") if r["source_kind"] == "person"]
    assert [(r["text"], r["robot_heard"]) for r in person] == [("Excuse me, robot?", True),
                                                             ("Is anyone there?", False)]
    others = [r for r in rows(tmp_path, "audio") if r["source_kind"] != "person"]
    assert others and all("robot_heard" not in r for r in others)


def test_a_line_said_far_from_the_robot_is_not_heard_by_it(staged, tmp_path):
    s = StubSimService(staged("reception_guide"), RunLog(tmp_path / "run"), wait_for=())
    s.step(0.01)
    w = s.world
    pid = next(p for p in w.people if w.people[p].kind != "static")
    far = None
    for x in range(-60, 60, 2):  # a free spot behind enough walls that a voice does not reach the robot
        for y in range(-40, 60, 2):
            if w.robot.router.is_free(float(x), float(y)):
                ev = w._event(0.0, "person", pid, float(x), float(y), 1.6, 60.0, "Hello?", "g1")
                if w.robot_heard(ev) is None:
                    far = (float(x), float(y))
                    break
        if far:
            break
    assert far is not None
    _put(w, pid, *far)
    s.on_say(Say(person=pid, t=s.t, text="Hello?", addressee="g1"))
    s.step(0.01)
    row = [r for r in rows(tmp_path / "run", "audio") if r["source_kind"] == "person"][-1]
    assert row["text"] == "Hello?" and row["robot_heard"] is False


# ---- addressing -----------------------------------------------------------------------------------------------------
def test_with_no_toward_and_nobody_who_spoke_the_line_has_no_addressee(tmp_path):
    """The re-QA no-toward probe: the newcomer greeting went to bystander V2 at 1.71 m. The nearest-person fallback is
    gone: P3 stands 1.1 m away but never spoke to the robot."""
    s, w = _lobby(tmp_path)
    assert _robot_line(s, "Hello, welcome.").addressee is None
    row = [r for r in rows(tmp_path, "audio") if r["source_kind"] == "robot"][-1]
    assert row["addressee"] is None and "P3" in row["heard_by"]


def test_with_no_toward_the_line_goes_to_the_last_speaker_within_3_m(tmp_path):
    s, w = _lobby(tmp_path)
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    assert _robot_line(s, "Yes, how can I help?").addressee == "P3"
    s.on_say(Say(person="P2", t=s.t, text="Excuse me, robot?", addressee="g1"))  # 3 m away: just too far
    s.step(0.01)
    _put(w, "P2", 1.05, -1.6)
    assert _robot_line(s, "Yes?").addressee is None


def test_a_toward_naming_someone_behind_the_robot_within_reach_goes_to_them(tmp_path):
    """Evacuation o1 t=311.2: toward t3 (P1, 1.26 m away behind the robot, out of the camera's view) went to V1. The
    named person is the addressee when they are near enough to be spoken to, in view or not."""
    s, w = _lobby(tmp_path)
    _put(w, "P5", 1.05, 2.9)  # 1.3 m behind the robot
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    assert _robot_line(s, "You are safe here.", w.tracks.track("P5")).addressee == "P5"


def test_a_toward_naming_someone_out_of_reach_goes_to_nobody_else(tmp_path):
    """A named track that cannot be spoken to (too far, or gone) leaves the line with no addressee: never the last
    person who spoke to the robot, never whoever stands nearest."""
    s, w = _lobby(tmp_path)
    _put(w, "P2", 1.05, -3.4)  # 5 m ahead
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    assert _robot_line(s, "Hello there.", w.tracks.track("P2")).addressee is None
    # a name that came too long before its line is dropped: the line then goes by the no-toward rule
    _put(w, "P2", 1.05, -1.38)
    s.step(0.01)
    s.on_robot_toward(w.tracks.track("P2"))
    s.t += 3.5
    s.on_robot_say("Hello there.")
    assert next(a for a in s.step(0.01).audio if a.source_kind == "robot").addressee is None


def test_a_toward_naming_someone_who_cannot_hear_goes_to_nobody(tmp_path, monkeypatch):
    s, w = _lobby(tmp_path)
    from wardzero_agents.ros import sim_service
    s.on_say(Say(person="P3", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    real = sim_service.audio.route
    monkeypatch.setattr(sim_service.audio, "route",
                        lambda ev, pose, pid, *a, **k: None if pid == "P2" else real(ev, pose, pid, *a, **k))
    assert _robot_line(s, "Hello there.", w.tracks.track("P2")).addressee is None
