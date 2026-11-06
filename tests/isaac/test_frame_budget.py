"""Regression probe for the Behavior System's robot obstacle cost (S1b, ``qR1_probe.py --edits late``), on the host's own start-up.

    WARDZERO_ISAAC=1 pytest -m isaac tests/isaac/test_frame_budget.py -q -s

``omni.anim.behavior.core`` registers every rigid body that exists when the timeline starts as a dynamic NavMesh
obstacle and re-carves for each one that moves: a G1 present at ``play()`` costs 230 ms a frame as soon as its arm
moves; added after ``play()`` (what ``IsaacHost.add_robots`` does) it costs about 35 ms. This test boots the host with
the G1 and five people, drives the G1's arm every frame and fails above 150 ms/frame. A failure means an Isaac or Kit
update changed the Behavior System, or robot placement moved before ``play()``: re-read spike_r1_report.md.
"""
import os

import pytest

from test_isaac_robot_smoke import OUT, isaac_free, run_demo  # noqa: F401  (the fixture and the runner)

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(os.environ.get("WARDZERO_ISAAC") != "1", reason="set WARDZERO_ISAAC=1 to boot Isaac Sim"),
]
LIMIT_MS = 150.0


def test_a_moving_g1_next_to_five_people_stays_under_the_frame_budget(tmp_path, isaac_free):
    r = run_demo(tmp_path, "budget", 10.0, OUT.parent / "isaac_robot_budget", cast=("P1", "P2", "P3", "P4", "P5"))
    st = r["stats"]
    assert st["people"] == 5
    assert st["arm_q"]["right_shoulder_pitch_joint"] == pytest.approx(-1.2, abs=0.1), st  # the arm is tracking its target: it was not blocked or NaN
    assert st["ms_median"] < LIMIT_MS, st
