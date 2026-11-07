"""Sound and walls: each wall between a source and a listener costs WALL_LOSS_DB (an open doorway is a gap in the wall,
so it costs nothing). The PA and the alarm are heard everywhere as before. QA: in ae_waiting the triage nurse heard
George's coughs and the waiting room's talk 14 m away through the walls."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wardzero_agents.config.presets import DEFAULT_CONFIG_ROOT, load_scenario_presets, preset_to_scenario
from wardzero_agents.contracts import AudioEvent, Intent, Say
from wardzero_agents.ros.stub_sim_node import StubSimService
from wardzero_agents.runlog import RunLog
from wardzero_agents.studio import auto_staff
from wardzero_agents.world.audio import COUGH_DB, WALL_LOSS_DB, count_walls, received_db, route
from wardzero_agents.world.oracle import Pose


def ev(loudness, at, kind="person", text="hello there", source_id="P9"):
    return AudioEvent(id="e", t=0.0, source_kind=kind, source_id=source_id, x=at[0], y=at[1], z=1.5,
                      loudness_db=loudness, text=text)


def test_a_wall_costs_its_loss_and_none_costs_nothing():
    rng = np.random.default_rng(1)
    open_ = route(ev(60, (5, 0)), Pose(0, 0, 0), "P1", 0, rng, walls_between=lambda a, b: 0)
    walled = route(ev(60, (5, 0)), Pose(0, 0, 0), "P1", 0, rng, walls_between=lambda a, b: 1)
    assert open_.received_db == pytest.approx(received_db(60, 5))
    assert walled.received_db == pytest.approx(received_db(60, 5) - WALL_LOSS_DB)
    assert route(ev(60, (5, 0)), Pose(0, 0, 0), "P1", 0, rng) .received_db == pytest.approx(received_db(60, 5))


def test_speech_through_a_wall_is_not_heard_at_a_room_away():
    rng = np.random.default_rng(1)
    assert route(ev(60, (6, 0)), Pose(0, 0, 0), "P1", 0, rng, walls_between=lambda a, b: 1) is None
    assert route(ev(60, (6, 0)), Pose(0, 0, 0), "P1", 0, rng, walls_between=lambda a, b: 0) is not None


def test_pa_and_alarm_ignore_walls():
    rng = np.random.default_rng(1)
    for kind in ("pa", "alarm"):
        h = route(ev(90, (30, 0), kind=kind, source_id=kind), Pose(0, 0, 0), "P1", 0, rng,
                  walls_between=lambda a, b: 3)
        assert h is not None and not h.partial


def test_count_walls_counts_crossings_not_cells():
    class G:
        res, origin = 0.1, (0.0, 0.0)

        def to_cell(self, x, y):
            return int(math.floor(y / 0.1)), int(math.floor(x / 0.1))

        def in_grid(self, iy, ix):
            return 0 <= iy < 100 and 0 <= ix < 100
    sight = np.zeros((100, 100), dtype=bool)
    sight[:, 30:33] = True  # a thick wall at x 3.0-3.3
    sight[:, 60] = True  # a thin one at x 6.0
    sight[50:55, 60] = False  # a doorway in it, y 5.0-5.5
    g = G()
    assert count_walls(sight, g, (1.0, 2.0), (2.5, 2.0)) == 0
    assert count_walls(sight, g, (1.0, 2.0), (5.0, 2.0)) == 1
    assert count_walls(sight, g, (1.0, 2.0), (8.0, 2.0)) == 2
    assert count_walls(sight, g, (4.0, 5.25), (8.0, 5.25)) == 0  # through the doorway


@pytest.fixture(scope="module")
def ae():
    p = load_scenario_presets(DEFAULT_CONFIG_ROOT)["ae_waiting"]
    return preset_to_scenario(p, DEFAULT_CONFIG_ROOT, Path(tempfile.mkdtemp()),
                              characters=auto_staff.with_auto(list(p.characters), p))


def heard_by(s, pid):
    return [h for _, _, hh in s._recent for who, h in hh.items() if who == pid]


def test_triage_does_not_hear_the_waiting_room(ae, tmp_path):
    s = StubSimService(ae, RunLog(tmp_path), wait_for=())
    s.step(0.01)
    s.on_say(Say(person="P1", t=s.t, text="Excuse me, how long is the wait?", addressee=None))
    s.on_intent(Intent(id="c", person="P1", t=s.t, kind="cough", overlay=True))
    s.step(0.01)
    assert heard_by(s, "V1")  # beside her in the waiting room
    assert not heard_by(s, "N1")  # in Triage, 14 m away behind walls


def test_a_severe_cough_through_walls_is_quieter_than_in_the_open(ae, tmp_path):
    s = StubSimService(ae, RunLog(tmp_path), wait_for=())
    w = s.world
    n1, p1 = w.people["N1"], w.people["P1"]
    walls = count_walls(w._sight, w.site.grid, (p1.x, p1.y), (n1.x, n1.y))
    assert walls >= 1
    d = math.hypot(n1.x - p1.x, n1.y - p1.y)
    assert received_db(COUGH_DB["severe"], d) - WALL_LOSS_DB * min(walls, 3) < 30 + 6  # at most a faint cough
