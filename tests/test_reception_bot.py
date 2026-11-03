from pathlib import Path

import pytest

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Heard
from wardzero_agents.examples.reception_bot import HELP_LINE, ReceptionBot

V1 = Path(__file__).resolve().parents[1] / "config" / "scenarios" / "waiting_area_v1.yaml"
LS = load_scenario(V1)


def bot():
    return ReceptionBot(LS.area, LS.signs)


def speech(text, db=60.0, partial=False):
    return Heard(source_kind="person", source_id="P5", text=text, received_db=db, partial=partial,
                 addressed_to_me=False)


def cough(db=57.0, who="P2"):
    return Heard(source_kind="cough", source_id=who, text=None, received_db=db, partial=False,
                 addressed_to_me=False)


def test_greets_hello_and_excuse_me():
    b = bot()
    assert "help" in b.on_heard(speech("Hello there"), 1.0).lower()
    assert b.on_heard(speech("Excuse me, can you help us?"), 2.0)
    assert b.on_heard(speech("Nice weather today"), 3.0) is None


@pytest.mark.parametrize("text,landmark,sign", [
    ("Where is clinic 3?", "clinic_corridor", "Outpatient clinics 1-6"),
    ("Excuse me, where are the wards?", "ward_doors", "Wards A-D and lifts"),
    ("Where's the toilet please", "toilets", "Toilets"),
    ("where is the exit", "entrance", "Way out"),
    ("Where's the way out?", "entrance", "Way out"),
])
def test_directions_name_the_landmark_id_and_the_sign(text, landmark, sign):
    reply = bot().on_heard(speech(text), 1.0)
    assert landmark in reply and sign in reply
    label = LS.area.landmarks[landmark].label.split(" (")[0].lower()
    assert label in reply.lower()


def test_directions_need_where_and_a_whole_word():
    b = bot()
    assert b.on_heard(speech("The clinic is running late"), 1.0) is None
    assert b.on_heard(speech("Where do I go towards"), 2.0) is None


def test_ignores_robots_and_pa():
    b = bot()
    pa = Heard(source_kind="pa", source_id="pa", text="hello where is the clinic", received_db=62,
               partial=False, addressed_to_me=False)
    assert b.on_heard(pa, 1.0) is None


def test_three_loud_coughs_within_20s_ask_for_help():
    b = bot()
    assert b.on_heard(cough(), 0.0) is None
    assert b.on_heard(cough(), 8.0) is None
    assert b.on_heard(cough(), 19.0) == HELP_LINE
    assert HELP_LINE == "Can I get some help for you? I'll let the nurse know."


def test_quiet_or_spread_out_coughs_do_not_count():
    b = bot()
    for t in (0.0, 5.0, 10.0):
        assert b.on_heard(cough(db=49.9), t) is None
    b = bot()
    for t in (0.0, 11.0, 22.0, 33.0):
        assert b.on_heard(cough(), t) is None


def test_help_has_a_60s_cool_down():
    b = bot()
    for t in (0.0, 1.0):
        b.on_heard(cough(), t)
    assert b.on_heard(cough(), 2.0) == HELP_LINE
    for t in (10.0, 11.0, 12.0, 30.0, 45.0, 55.0):
        assert b.on_heard(cough(), t) is None
    assert b.on_heard(cough(), 62.5) == HELP_LINE
