"""Every scenario preset, end to end, with the scripted people and the scripted built-in robot (opt-in:
WARDZERO_E2E=1).

Each preset is staged as the Studio stages it (with the auto staff) and run on the stub world at time scale 10:

- with the built-in robot on the scripted model, the robot does the main task: the flow finishes, every robot expect
  is met, the score passes, nobody's decision fails, and the robot touches nobody;
- with no robot, nothing is credited: no robot expect is met and the score does not pass.

By default a run goes through the real launch path (``cli run --sim stub --bot claude|none`` in a clean ROS 2
environment, people on WARDZERO_E2E_DOMAIN, default 164, robot on the next domain). WARDZERO_E2E_MODE=inproc runs the
same services in one process instead (no ROS; quicker, for working on a policy). A failure lists, in words, every
step or criterion that went wrong. Run folders are kept under WARDZERO_E2E_DIR (default: pytest's temp dir).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import _e2e_harness as H

pytestmark = [pytest.mark.e2e,
              pytest.mark.skipif(os.environ.get("WARDZERO_E2E") != "1", reason="set WARDZERO_E2E=1 to run")]

MODE = os.environ.get("WARDZERO_E2E_MODE", "launch")
DOMAIN = int(os.environ.get("WARDZERO_E2E_DOMAIN", "164"))
PRESET_IDS = sorted(H.presets()) if os.environ.get("WARDZERO_E2E") == "1" else []


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = os.environ.get("WARDZERO_E2E_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        return Path(root)
    return tmp_path_factory.mktemp("e2e")


def run(preset_id: str, work: Path, bot: str) -> H.Verdict:
    out = work / bot / preset_id
    if MODE == "inproc":
        run_dir = H.run_inproc(preset_id, out, bot=bot)
    else:
        if not H.ros_ready():
            pytest.skip("ROS 2 Humble is not installed here (or set WARDZERO_E2E_MODE=inproc)")
        run_dir, _ = H.run_launch(preset_id, out, bot=bot, domain=DOMAIN)
    H.debrief(run_dir)
    return H.judge(preset_id, run_dir)


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_the_scripted_robot_does_the_main_task(preset_id, work):
    v = run(preset_id, work, "claude")
    problems = v.problems()
    assert v.expects, f"{preset_id}: the flow has no robot expects to judge ({v.run_dir})"
    assert not problems, f"{preset_id} with the scripted robot ({v.run_dir}):\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_nothing_is_credited_without_a_robot(preset_id, work):
    v = run(preset_id, work, "none")
    credited = v.credited()
    assert v.flow_done, f"{preset_id} with no robot: the flow did not finish ({v.run_dir})"
    assert not credited, f"{preset_id} with no robot ({v.run_dir}):\n  " + "\n  ".join(credited)
