from pathlib import Path

from wardzero_agents.config.loader import load_scenario
from wardzero_agents.contracts import Heard, Status
from wardzero_agents.reflex import ReflexEngine

SCENARIO = Path(__file__).resolve().parents[1] / "config/scenarios/waiting_area_v1.yaml"


def inst(**update):
    base = load_scenario(SCENARIO).instances[1]
    return base.model_copy(update=update)


def status(posture="seated"):
    return Status(person="P2", t=0.0, x=0, y=0, yaw=0, posture=posture, progress=0.0, state="idle")


def heard(source_id="g1"):
    return Heard(source_kind="robot", source_id=source_id, text="hi", received_db=55,
                 partial=False, addressed_to_me=False)


def frange(a, b, step):
    n = int(round((b - a) / step))
    return [a + i * step for i in range(n)]


def run(engine, t0, t1, step, posture="seated", idle=True):
    out = []
    for t in frange(t0, t1, step):
        out += [(t, a) for a in engine.tick(t, status(posture), [], idle)]
    return out


def count(acts, kind):
    return sum(1 for _, a in acts if a.kind == kind)


def test_cough_rate():
    e = ReflexEngine(inst(cough_per_hour=30), seed=3)
    n = count(run(e, 0, 36000, 1.0), "cough")
    assert 240 <= n <= 360


def test_zero_rate_never_coughs():
    e = ReflexEngine(inst(cough_per_hour=0), seed=3)
    assert count(run(e, 0, 36000, 1.0, idle=False), "cough") == 0


def test_cough_severity_from_persona():
    e = ReflexEngine(inst(cough_per_hour=60, cough_severity="moderate"), seed=3)
    coughs = [a for _, a in run(e, 0, 3600, 1.0) if a.kind == "cough"]
    assert coughs and all(a.severity == "moderate" for a in coughs)


def test_bout():
    e = ReflexEngine(inst(cough_per_hour=0), seed=3)
    e.start_bout(100.0, 40, "severe")
    acts = run(e, 100.0, 140.0, 0.1)
    coughs = [a for _, a in acts if a.kind == "cough"]
    assert 10 <= len(coughs) <= 20
    assert all(a.severity == "severe" for a in coughs)
    assert count(run(e, 141.0, 400.0, 0.1), "cough") == 0


def test_fidget_only_seated_and_idle():
    e = ReflexEngine(inst(cough_per_hour=0), seed=4)
    assert count(run(e, 0, 3600, 1.0, posture="standing"), "fidget") == 0
    assert count(run(e, 3600, 7200, 1.0, idle=False), "fidget") == 0
    n = count(run(e, 7200, 10800, 1.0), "fidget")
    assert 3600 / 120 - 3 <= n <= 3600 / 45 + 1


def test_gaze_on_voice():
    e = ReflexEngine(inst(), seed=3)
    acts = e.tick(5.0, status(), [heard(source_id="g1")], True)
    assert any(a.kind == "look_at" and a.target == "g1" for a in acts)


def test_gaze_rate_limited():
    e = ReflexEngine(inst(cough_per_hour=0), seed=3)
    first = e.tick(5.0, status(), [heard()], True)
    soon = e.tick(6.0, status(), [heard()], True)
    later = e.tick(8.5, status(), [heard()], True)
    assert any(a.kind == "look_at" for a in first)
    assert not any(a.kind == "look_at" for a in soon)
    assert any(a.kind == "look_at" for a in later)


def test_deterministic():
    a = run(ReflexEngine(inst(), seed=9), 0, 3600, 1.0)
    b = run(ReflexEngine(inst(), seed=9), 0, 3600, 1.0)
    assert a == b


def test_cough_count_independent_of_tick_spacing():
    end = 35990.0
    fine = ReflexEngine(inst(cough_per_hour=30), seed=3)
    n_fine = 0
    for i in range(int(end / 0.1) + 1):
        n_fine += sum(a.kind == "cough" for a in fine.tick(i * 0.1, status(), [], True))
    coarse = ReflexEngine(inst(cough_per_hour=30), seed=3)
    n_coarse = 0
    for i in range(int(end / 10.0) + 1):
        acts = coarse.tick(i * 10.0, status(), [], True)
        n_coarse += sum(a.kind == "cough" for a in acts)
        assert sum(a.kind == "cough" for a in acts) < 5  # cap not hit
    assert n_fine == n_coarse


def test_cough_cap_per_tick():
    e = ReflexEngine(inst(cough_per_hour=3600), seed=3)
    e.tick(0.0, status(), [], True)
    assert sum(a.kind == "cough" for a in e.tick(1000.0, status(), [], True)) == 5
