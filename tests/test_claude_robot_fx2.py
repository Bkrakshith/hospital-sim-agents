"""The built-in robot mind after the re-QA: a call bell (source_kind "bell") wakes it at once and shows as a call bell,
never as someone speaking; a track marked unwell shows as looking unwell; the places it knows are listed by name."""
import asyncio
from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.config.presets import load_scenario_presets, preset_to_scenario
from wardzero_agents.robots.claude_robot import RobotMind
from wardzero_agents.robots.profile import G1_CLAUDE

ROOT = Path(__file__).resolve().parents[1]
V1 = load_scenario(ROOT / "config" / "scenarios" / "waiting_area_v1.yaml")


@pytest.fixture(scope="module")
def reception(tmp_path_factory):
    p = load_scenario_presets(ROOT / "config")["reception_guide"]
    return preset_to_scenario(p, ROOT / "config", tmp_path_factory.mktemp("reception_guide"))


class _Site:  # the bare site the re-QA probe used (qa2/prompts/bell_wake.py)
    def resolve(self, p):
        return None


def _bell(text="call bell, bed B2", db=46.0, kind="bell", bearing=None):
    h = {"source_kind": kind, "source_id": "unknown", "text": text, "received_db": db, "partial": False,
         "addressed_to_me": False}
    if bearing is not None:
        h["bearing_deg"] = bearing
    return h


def _det(track, x, y, z=1.2, **extra):
    return {"id": track, "results": [{"class_id": "person", "score": 1.0}],
            "bbox": {"center": {"x": x, "y": y, "z": z / 2, "yaw": 0.0}, "size": {"x": 0.5, "y": 0.5, "z": z}},
            **extra}


def test_a_bell_with_its_words_wakes_the_mind_with_nobody_near():
    """The re-QA probe: "call bell, bed B2" with nobody within 3 m left _speech False (the mind waited for its 10 s
    heartbeat). As source_kind "bell" it wakes the mind at once."""
    m = RobotMind(_Site(), {"job": "x"}, None, lambda: 0.0)
    asyncio.run(m.on_heard(_bell(), 61.0))
    assert m._speech is True
    m._speech = False
    asyncio.run(m.on_heard(_bell("staff call chime", 55.0), 62.0))
    assert m._speech is True
    m._speech = False  # an old run's chime (a person sound with no words) still wakes it
    asyncio.run(m.on_heard({"source_kind": "person", "text": "", "received_db": 46.0}, 63.0))
    assert m._speech is True


def test_a_bell_shows_as_a_call_bell_in_the_situation_not_as_someone_speaking():
    m = RobotMind(V1.site, {"job": "x"}, None, lambda: 0.0, known_places=[], pose=(1.0, 1.0, 0.0))
    asyncio.run(m.on_heard(_bell("call bell, bed B3", 47.0, bearing=30.0), 92.0))
    asyncio.run(m.on_heard(_bell("staff call chime", 55.0), 93.0))
    asyncio.run(m.on_heard(_bell("call bell, bed B3", 44.0, kind="person"), 94.0))  # tolerated: the old wording
    s = m._situation(95.0)
    heard = s.split("Heard recently, newest last:")[1].split("People your camera sees")[0]
    assert "someone nearby" not in heard and '"call bell' not in heard, heard
    lines = [ln for ln in heard.splitlines() if ln.startswith("- ")]
    assert "a call bell (bed B3)" in lines[0] and "your staff call chime" in lines[1] and "a call bell (bed B3)" in lines[2]
    assert "degrees to your left" in lines[0]  # the bell at the bed still says where it came from


def test_a_track_marked_unwell_shows_as_looking_unwell():
    m = RobotMind(V1.site, {"job": "x"}, None, lambda: 0.0, known_places=[], pose=(0.0, 0.0, 0.0))
    m.on_people([_det("t1", 2.0, 0.0, unwell=True), _det("t2", 3.0, 0.0)], 1.0)
    tracks = m._situation(1.0).split("People your camera sees (anonymous tracks):")[1].split("Objects in view")[0]
    t1 = next(ln for ln in tracks.splitlines() if ln.startswith("- t1:"))
    t2 = next(ln for ln in tracks.splitlines() if ln.startswith("- t2:"))
    assert "looks unwell, slumped" in t1 and "unwell" not in t2


def test_the_places_it_knows_are_listed_by_name_not_by_id(reception):
    m = RobotMind(reception.site, {"role": "reception guide", "job": "Greet people."}, None, lambda: 0.0,
                  profile=G1_CLAUDE, known_places=reception.scenario.known_places, pose=(1.05, 1.62, -1.5708))
    places = m.card().split("Places you know:")[1].split("YOUR JOB")[0]
    assert "Main_LiftLobby" not in places and "_" not in places, places
    assert "- Protected bed-lift lobby: " in places and "(lift lobby)" in places


def test_a_person_who_mentions_a_call_bell_is_heard_speaking_not_as_a_chime():
    """Only an old run's chime row (a person sound whose whole text is the chime's words, not said to the robot) is
    read as a chime: a patient saying something about their call bell, to the robot or near it, keeps their words."""
    m = RobotMind(V1.site, {"job": "x"}, None, lambda: 0.0, known_places=[], pose=(1.0, 1.0, 0.0))
    said = "I pressed my call bell, bed B3, twenty minutes ago and nobody came"
    asyncio.run(m.on_heard({"source_kind": "person", "source_id": "unknown", "text": said, "received_db": 60.0,
                            "partial": False, "addressed_to_me": True}, 92.0))
    asyncio.run(m.on_heard({"source_kind": "person", "source_id": "unknown", "text": "staff call chime",
                            "received_db": 60.0, "partial": False, "addressed_to_me": True}, 93.0))
    asyncio.run(m.on_heard({"source_kind": "person", "source_id": "unknown", "text": "Is that a call bell, bed B4?",
                            "received_db": 60.0, "partial": False, "addressed_to_me": False}, 94.0))
    heard = m._situation(95.0).split("Heard recently, newest last:")[1].split("People your camera sees")[0]
    lines = [ln for ln in heard.splitlines() if ln.startswith("- ")]
    assert "someone speaking to you" in lines[0] and said in lines[0] and "a call bell" not in lines[0], lines
    assert "someone speaking to you" in lines[1] and '"staff call chime"' in lines[1], lines
    assert '"Is that a call bell, bed B4?"' in lines[2] and "a call bell (bed" not in lines[2], lines
