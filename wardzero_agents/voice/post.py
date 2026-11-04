"""The voice post step: ``<run>/audio.jsonl`` to ``<run>/audio/voice.wav`` and an AAC track in the run's videos.

Runs on the CPU after the simulator has exited (it never touches Isaac), from ``--record`` (unless ``--no-voice``) and
from ``hhumans voice RUN``. Without voice models or Piper it warns and writes nothing; the run is unaffected.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import media, mix
from .assign import assign_voices, robot_voice
from .catalogue import installed_models, voices_dir
from .tts import PiperUnavailable, Synth

log = logging.getLogger("wardzero.voice")

SILENT_KINDS = ("cough", "alarm")  # sounds, not speech
TAIL_S = 1.0  # after the last line, when there is no video to set the length
VIDEOS = ("overhead.mp4", "follow.mp4")


def speech_events(run_dir: Path) -> list[dict]:
    """Lines to speak, in time order: audio.jsonl events with text, not coughs or alarms."""
    out = []
    try:
        lines = (Path(run_dir) / "audio.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        text = (ev.get("text") or "").strip()
        if text and ev.get("source_kind") not in SILENT_KINDS and isinstance(ev.get("t"), (int, float)):
            out.append(ev)
    out.sort(key=lambda e: e["t"])
    return out


def load_cast(run_dir: Path) -> tuple[list[tuple], Optional[str]]:
    """((person_id, sex, age) for the run's people, the robot profile's voice key) from the run's copied config.
    Both empty when the config cannot be loaded (people then get stable voices from their ids alone)."""
    try:
        from ..cli import _load_recorded, _recorded_meta
        from ..robots.profile import get_profile

        meta = _recorded_meta(Path(run_dir))
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _load_recorded(Path(run_dir), meta, Path(tmp))
        people = [(i.person_id, i.sex, i.age) for i in loaded.instances]
        try:
            vk = get_profile(loaded.scenario.robot.profile).voice
        except KeyError:
            vk = None
        return people, vk
    except Exception as exc:  # an old run, a missing config: not worth failing the voice for
        log.info("voice: no cast read from %s (%s)", run_dir, type(exc).__name__)
        return [], None


def video_timing(run_dir: Path) -> tuple[float, Optional[float], list[Path]]:
    """(run time of video frame 0, the videos' length, the videos). Runs recorded before timing.json started at 0."""
    vdir = Path(run_dir) / "video"
    vids = [vdir / n for n in VIDEOS if (vdir / n).is_file()]
    t0 = 0.0
    try:
        t0 = float(json.loads((vdir / "timing.json").read_text(encoding="utf-8")).get("t_start_s") or 0.0)
    except (OSError, ValueError):
        pass
    lengths = [d for d in (media.duration_s(v) for v in vids) if d]
    return t0, (max(lengths) if lengths else None), vids


def build_voice(run_dir: "str | Path", models_dir: Optional[Path] = None, say=print) -> dict:
    """Synthesise, mix and mux. Returns {"wav": path or None, "lines", "muxed": [videos], "synth_s", "total_s", "warning"}."""
    run_dir = Path(run_dir)
    t_begin = time.monotonic()
    res: dict = {"wav": None, "lines": 0, "muxed": [], "synth_s": 0.0, "total_s": 0.0, "warning": None}
    d = voices_dir() if models_dir is None else Path(models_dir)
    models = installed_models(d)
    if not models:
        res["warning"] = (f"No voice models in {d}: the run has no voices. Run scripts/fetch_voices.py to download them, "
                          f"then `hhumans voice {run_dir}`.")
        say(res["warning"])
        return res
    try:
        synth = Synth(d)
    except PiperUnavailable as exc:
        res["warning"] = f"No voices: {exc}."
        say(res["warning"])
        return res
    events = speech_events(run_dir)
    people, robot_key = load_cast(run_dir)
    rvoice = robot_voice(robot_key, models)
    ids = {str(e["source_id"]) for e in events if e.get("source_kind") != "robot"}
    known = {p[0] for p in people}
    cast = [p for p in people if p[0] in ids] + [(i, None, None) for i in sorted(ids - known)]
    seed = (json.loads((run_dir / "meta.json").read_text(encoding="utf-8")).get("run_id")
            if (run_dir / "meta.json").is_file() else None) or run_dir.name
    voices = assign_voices(cast, models, seed=seed, reserved=[rvoice.key] if rvoice else ())
    t0, vlen, vids = video_timing(run_dir)
    placed = []
    for ev in events:
        robot = ev.get("source_kind") == "robot"
        v = rvoice if robot else voices.get(str(ev["source_id"]))
        if v is None:
            continue
        a, sr = synth.say(v, ev["text"].strip(), robot=robot)
        placed.append(mix.Placed(float(ev["t"]), mix.resample(a, sr, mix.SAMPLE_RATE), float(ev.get("loudness_db") or mix.REF_DB),
                                 float(ev.get("x") or 0.0), float(ev.get("y") or 0.0)))
    if vlen is not None:
        duration = vlen
    else:
        duration = max([p.t - t0 + p.audio.size / mix.SAMPLE_RATE for p in placed], default=0.0) + TAIL_S
    rb = [e for e in events if e.get("source_kind") == "robot"]
    listener = (float(rb[0]["x"]), float(rb[0]["y"])) if rb else (
        (sum(p.x for p in placed) / len(placed), sum(p.y for p in placed) / len(placed)) if placed else (0.0, 0.0))
    data = mix.mix(placed, t0, duration, listener)
    wav = run_dir / "audio" / "voice.wav"
    mix.write_wav(wav, data)
    res.update(wav=wav, lines=len(placed), synth_s=round(synth.seconds, 2))
    for v in vids:
        if media.mux_audio(v, wav):
            res["muxed"].append(v)
        else:
            say(f"Could not add the voice track to {v}; it is left as it was.")
    res["total_s"] = round(time.monotonic() - t_begin, 2)
    res["per_line_s"] = round(synth.seconds / synth.lines, 2) if synth.lines else 0.0
    say(f"Voices: {len(placed)} lines, {res['synth_s']} s of synthesis, {wav}"
        + (f", added to {', '.join(v.name for v in res['muxed'])}" if res["muxed"] else ""))
    return res
