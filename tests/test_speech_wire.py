"""The robot's speech wire (contract item 4): <ns>/speech/toward is a FIFO paired with speech/say in order, a toward
older than 2 s with no say is dropped, the toward ``room`` addresses nobody and marks the audio row ``to_room``, the
last-listener fallback reaches only someone within 3.0 m, and every robot audio row carries ``heard_by``."""
import json
from pathlib import Path

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Say
from wardzero_agents.ros import topics
from wardzero_agents.ros.sim_service import LAST_LISTENER_M, TOWARD_WINDOW_S
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"


def rows(run, stream="audio"):
    p = run / f"{stream}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def robot_rows(run):
    return [r for r in rows(run) if r["source_kind"] == "robot"]


def _put(w, pid, x, y):
    p = w.people[pid]
    p.x, p.y, p.posture, p.seat, p.plan = x, y, "standing", None, []


def staged(tmp_path):
    """P3 1.1 m from the robot, P2 3 m straight ahead (both in view)."""
    run = tmp_path / "run"
    s = StubSimService(load_scenario(V1), RunLog(run), wait_for=())
    s.step(0.01)
    w = s.world
    _put(w, "P3", 1.5, 0.6)
    _put(w, "P2", 1.05, -1.38)
    s.step(0.01)
    return s, w, run


def test_contract_constants():
    assert TOWARD_WINDOW_S == 2.0 and LAST_LISTENER_M == 3.0 and topics.TOWARD_ROOM == "room"


def test_back_to_back_toward_say_pairs_keep_their_addressees(tmp_path):
    """QA ae_waiting run C: two toward/say pairs sent back to back swapped addressees."""
    s, w, run = staged(tmp_path)
    s.on_robot_toward(w.tracks.track("P2"))
    s.on_robot_say("Hello, you by the window.")
    s.on_robot_toward(w.tracks.track("P3"))
    s.on_robot_say("And hello to you.")
    out = s.step(0.01)
    assert [(a.text, a.addressee) for a in out.audio] == [("Hello, you by the window.", "P2"),
                                                          ("And hello to you.", "P3")]


def test_towards_sent_before_their_lines_pair_in_order(tmp_path):
    s, w, run = staged(tmp_path)
    s.on_robot_toward(w.tracks.track("P3"))
    s.on_robot_toward(w.tracks.track("P2"))
    s.on_robot_say("First line.")
    s.on_robot_say("Second line.")
    out = s.step(0.01)
    assert [a.addressee for a in out.audio] == ["P3", "P2"]


def test_a_toward_that_comes_after_its_line_pairs_with_the_next_line(tmp_path):
    """FIFO: a toward is the next say's, whenever that comes (within 2 s)."""
    s, w, run = staged(tmp_path)
    s.on_robot_say("Over there.")
    assert s.step(0.01).audio[0].addressee is None  # no name and nobody spoke to it: nobody in particular
    s.on_robot_toward(w.tracks.track("P2"))
    s.step(0.01)
    s.on_robot_say("Can I help you?")
    assert s.step(0.01).audio[0].addressee == "P2"


def test_a_toward_older_than_two_seconds_is_dropped(tmp_path):
    s, w, run = staged(tmp_path)
    s.on_robot_toward(w.tracks.track("P2"))
    for _ in range(25):
        s.step(0.1)  # 2.5 s with no say
    s.on_robot_say("Hello there.")
    assert s.step(0.01).audio[0].addressee is None  # by the no-toward rule (nobody spoke to it), not P2


def test_toward_room_addresses_nobody_and_marks_the_row(tmp_path):
    s, w, run = staged(tmp_path)
    _put(w, "P5", 1.05, -0.98)  # 2.6 m away; P3 is nearer
    s.step(0.01)
    s.on_say(Say(person="P5", t=s.t, text="Excuse me, robot?", addressee="g1"))
    s.step(0.01)
    s.on_robot_toward("room")
    s.on_robot_say("The doctor is running about 20 minutes late, everyone.")
    out = s.step(0.01)
    assert out.audio[0].addressee is None
    row = robot_rows(run)[-1]
    assert row["to_room"] is True and row["addressee"] is None
    assert {"P2", "P3"} <= set(row["heard_by"])
    # the room line is not a reply: P5's question still has its answer due, and the next unnamed line goes to her
    s.on_robot_say("Yes, how can I help?")
    assert s.step(0.01).audio[0].addressee == "P5"


def test_the_last_listener_fallback_reaches_only_someone_within_3_m(tmp_path):
    """QA ae_waiting run A: a line meant for someone out of view went to the receptionist 16 m away."""
    s, w, run = staged(tmp_path)
    _put(w, "P5", 1.05, -2.88)  # 4.5 m away
    s.step(0.01)
    s.on_say(Say(person="P5", t=s.t, text="Robot, where is the pharmacy?", addressee="g1"))
    s.step(0.01)
    s.on_robot_say("It is down the corridor on the left.")
    assert s.step(0.01).audio[0].addressee is None  # P5 is too far for the fallback: nobody, not the nearest (P3)
    _put(w, "P5", 1.05, -0.98)  # 2.6 m: close enough
    s.step(0.01)
    s.on_say(Say(person="P5", t=s.t, text="Robot, and the toilets?", addressee="g1"))
    s.step(0.01)
    s.on_robot_say("Next to the lifts.")
    assert s.step(0.01).audio[0].addressee == "P5"


def test_every_robot_audio_row_carries_heard_by(tmp_path):
    s, w, run = staged(tmp_path)
    s.on_robot_say("Hello.")
    s.step(0.01)
    s.on_robot_toward(w.tracks.track("P2"))
    s.on_robot_say("Hello to you.")
    s.step(0.01)
    s.on_robot_say("Test line", None)  # the test hook: nobody addressed
    s.step(0.01)
    rr = robot_rows(run)
    assert len(rr) == 3 and all(isinstance(r.get("heard_by"), list) for r in rr)
    assert all({"P2", "P3"} <= set(r["heard_by"]) for r in rr)
    assert not any(r.get("to_room") for r in rr)
    people = [r for r in rows(run) if r["source_kind"] != "robot"]
    assert all("heard_by" not in r for r in people)
