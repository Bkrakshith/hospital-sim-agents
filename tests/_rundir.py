"""A small synthetic finished run folder for the debrief and report tests."""
import json
from pathlib import Path


def write(d: Path, name: str, rows: list) -> None:
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_ae_run(d: Path, **meta_over) -> Path:
    """An A&E-like run: P1 deteriorates at 4:00, the robot alerts triage 30 s later, P2 asks about the
    wait and is ignored, plus a static patient SB1 that must be left out of everything."""
    d.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": d.name, "scenario_id": "ae_waiting", "preset_id": "ae_waiting", "seed": 7,
            "project_seed": 7, "run_index": 1, "word_of_mouth": False, "provider": "claude",
            "models": {"decide": "claude-sonnet-5-5", "summarise": "claude-haiku-5-5",
                       "vision": "claude-sonnet-5-5", "robot": "claude-sonnet-5-5", "report": "claude-sonnet-5-5"},
            "people": {"P1": {"role": "patient", "preset_role": "patient", "static": False},
                       "P2": {"role": "patient", "preset_role": "patient", "static": False},
                       "N1": {"role": "ward_nurse", "preset_role": "ward_nurse", "static": False}}}
    meta.update(meta_over)
    (d / "meta.json").write_text(json.dumps(meta))
    ts = [0.0, 60.0, 120.0, 240.0, 300.0]
    status = [{"t": t, "person": p, "x": x, "y": y, "yaw": 0.0, "posture": "standing", "state": "idle",
               "progress": 0.0} for t in ts for p, x, y in (("P1", -7.5, 0.2), ("P2", -8.3, 0.2), ("N1", -4.3, 14.0))]
    status += [{"t": t, "person": "SB1", "x": 0.0, "y": 0.0, "yaw": 0.0, "posture": "lying", "state": "idle",
                "progress": 0.0} for t in ts]
    write(d, "status", status)
    write(d, "robot_pose", [{"t": t, "x": -8.6, "y": -1.8, "yaw": 1.57, "speed": 0.0} for t in ts])
    write(d, "cards", [{"t": 240.0, "card": "deterioration", "target": "P1", "ok": True}])
    write(d, "says", [
        {"t": 100.0, "person": "P2", "addressee": "g1", "text": "How long is the wait?"},
        {"t": 250.0, "person": "g1", "addressee": "N1", "text": "A patient needs help, please come."},
        {"t": 300.0, "person": "P1", "addressee": "P2", "text": "That robot was useless."}])
    write(d, "audio", [{"t": 100.0, "source_kind": "person", "source": "P2", "addressee": "g1",
                        "text": "How long is the wait?"},
                       {"t": 250.0, "source_kind": "robot", "addressee": "N1", "text": "A patient needs help."}])
    write(d, "robot_tasks", [])
    write(d, "tokens", [{"t": 10.0, "person": "P1", "role": "decide", "model": "x", "ok": True,
                         "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.01}])
    return d
