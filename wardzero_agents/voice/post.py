"""The voice post step: ``<run>/audio.jsonl`` to ``<run>/audio/voice.wav`` and an AAC track in the run's videos.

Runs on the CPU after the simulator has exited (it never touches Isaac), from ``--record`` (unless ``--no-voice``) and
from ``hhumans voice RUN``. Without voice models or Piper it warns and writes nothing; the run is unaffected.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import media, mix
from .assign import announcer_voice, assign_voices, robot_voice
from .catalogue import credits_text, installed_models, voices_dir
from .tts import PiperUnavailable, Synth

SILENT_KINDS = ("cough", "alarm")  # sounds, not speech
TAIL_S = 1.0  # after the last line, when there is no video to set the length
VIDEOS = ("overhead.mp4", "follow.mp4")
CREDITS = "CREDITS.txt"


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


def _run_meta(run_dir: Path) -> dict:
    try:
        d = json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _scenario_file(cfg: Path, meta: dict) -> Optional[Path]:
    name = Path(str(meta.get("scenario_path") or "")).name
    if name and (cfg / name).is_file():
        return cfg / name
    for f in sorted(cfg.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(raw, dict) and "cast" in raw:
            return f
    return None


def load_cast(run_dir: Path) -> tuple[list[tuple], Optional[str]]:
    """((person_id, sex, age) for the run's cast, the robot profile's voice key), read from the run's copied
    scenario and personas.yaml alone: it needs neither the site map nor the storyboard. Raises ValueError (with the
    reason) when the cast cannot be read."""
    from ..config.models import CastEntry, Persona
    from ..config.sampling import sample_instance
    from ..robots.profile import get_profile

    run_dir = Path(run_dir)
    cfg = run_dir / "config"
    scen = _scenario_file(cfg, _run_meta(run_dir))
    if scen is None or not (cfg / "personas.yaml").is_file():
        raise ValueError(f"{cfg} has no scenario and personas.yaml")
    try:
        raw = yaml.safe_load(scen.read_text(encoding="utf-8"))
        personas = {pid: Persona(**p) for pid, p in (yaml.safe_load((cfg / "personas.yaml").read_text(encoding="utf-8"))
                                                      or {}).get("personas", {}).items()}
        seed = int(raw.get("seed", 0))
        people = []
        for i, c in enumerate(raw["cast"]):
            entry = CastEntry(**c)
            inst = sample_instance(personas[entry.persona], entry, seed, i)
            people.append((inst.person_id, inst.sex, inst.age))
    except Exception as exc:
        raise ValueError(f"{scen.name}: {type(exc).__name__}: {str(exc)[:160]}") from exc
    vk = None
    try:
        vk = get_profile(str((raw.get("robot") or {}).get("profile") or "g1_claude")).voice
    except KeyError:
        pass
    return people, vk


def video_timing(run_dir: Path) -> tuple[float, Optional[float], list[Path]]:
    """(run time of video frame 0, the overhead video's length, the videos). Runs recorded before timing.json started at 0.
    The other videos are trimmed to that length when muxed."""
    vdir = Path(run_dir) / "video"
    vids = [vdir / n for n in VIDEOS if (vdir / n).is_file()]
    t0 = 0.0
    try:
        t0 = float(json.loads((vdir / "timing.json").read_text(encoding="utf-8")).get("t_start_s") or 0.0)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    lengths = [d for d in (media.duration_s(v) for v in vids) if d]
    return t0, (lengths[0] if lengths else None), vids  # overhead first: it is the reference


def build_voice(run_dir: "str | Path", models_dir: Optional[Path] = None, say: Callable = print,
                warn: Optional[Callable] = None) -> dict:
    """Synthesise, mix and mux. ``say`` takes progress lines and ``warn`` problems (default: the same). Returns
    {"wav": path or None, "lines", "muxed": [videos], "synth_s", "total_s", "warning"}."""
    warn = warn or say
    run_dir = Path(run_dir)
    t_begin = time.monotonic()
    res: dict = {"wav": None, "lines": 0, "muxed": [], "synth_s": 0.0, "total_s": 0.0, "warning": None}

    def fail(msg: str) -> dict:
        res["warning"] = msg
        warn(msg)
        return res

    d = voices_dir() if models_dir is None else Path(models_dir)
    models = installed_models(d)
    if not models:
        return fail(f"No voice models in {d}: the run has no voices. Run scripts/fetch_voices.py to download them, "
                    f"then `hhumans voice {run_dir}`.")
    try:
        synth = Synth(d)
    except PiperUnavailable as exc:
        return fail(f"No voices: {exc}.")
    events = speech_events(run_dir)
    try:
        people, robot_key = load_cast(run_dir)
    except ValueError as exc:
        people, robot_key = [], None
        warn(f"Voices: the cast could not be read ({exc}); people get voices from their ids alone, without sex or age.")
    rvoice = robot_voice(robot_key, models)
    avoice = announcer_voice(models, rvoice)
    ids = {str(e["source_id"]) for e in events if e.get("source_kind") == "person"}
    known = {p[0] for p in people}
    cast = [p for p in people if p[0] in ids] + [(i, None, None) for i in sorted(ids - known)]
    seed = _run_meta(run_dir).get("run_id") or run_dir.name
    voices = assign_voices(cast, models, seed=seed, reserved=[v.key for v in (rvoice, avoice) if v])
    t0, vlen, vids = video_timing(run_dir)
    items, who = [], []
    for ev in events:
        kind = ev.get("source_kind")
        robot = kind == "robot"
        v = rvoice if robot else (voices.get(str(ev["source_id"])) if kind == "person" else avoice)
        if v is not None:
            items.append((v, ev["text"].strip(), robot))
            who.append(ev)
    try:
        synthesised = synth.run(items)
    except PiperUnavailable as exc:
        return fail(f"No voices: {exc}.")
    placed = [mix.Placed(float(ev["t"]), mix.resample(a, sr, mix.SAMPLE_RATE), float(ev.get("loudness_db") or mix.REF_DB),
                         float(ev.get("x") or 0.0), float(ev.get("y") or 0.0))
              for ev, (a, sr) in zip(who, synthesised)]
    duration = vlen if vlen is not None else (
        max([p.t - t0 + p.audio.size / mix.SAMPLE_RATE for p in placed], default=0.0) + TAIL_S)
    rb = [e for e in who if e.get("source_kind") == "robot"]
    listener = (float(rb[0]["x"]), float(rb[0]["y"])) if rb else (
        (sum(p.x for p in placed) / len(placed), sum(p.y for p in placed) / len(placed)) if placed else (0.0, 0.0))
    data = mix.mix(placed, t0, duration, listener)
    wav = run_dir / "audio" / "voice.wav"
    mix.write_wav(wav, data)
    (wav.parent / CREDITS).write_text(credits_text({v.key for v, _, _ in items}), encoding="utf-8")
    res.update(wav=wav, lines=len(placed), synth_s=round(synth.seconds, 2))
    for v in vids:
        if media.mux_audio(v, wav, max_duration=vlen):
            res["muxed"].append(v)
        else:
            warn(f"Could not add the voice track to {v}; it is left as it was.")
    res["total_s"] = round(time.monotonic() - t_begin, 2)
    res["per_line_s"] = round(synth.seconds / synth.lines, 2) if synth.lines else 0.0
    say(f"Voices: {len(placed)} lines, {res['synth_s']} s of synthesis, {wav}"
        + (f", added to {', '.join(v.name for v in res['muxed'])}" if res["muxed"] else ""))
    return res
