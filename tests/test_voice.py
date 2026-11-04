"""Voices for recorded runs: assignment, timeline placement, mixing, the mux and the no-models fallback. The tests that
synthesise speech are marked `voice` and skip when Piper or the models are missing."""
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wardzero_agents.voice import media, mix
from wardzero_agents.voice.assign import assign_voices, robot_voice
from wardzero_agents.voice.catalogue import DEFAULT_ROBOT_VOICE, MODELS, VOICES, installed_models
from wardzero_agents.voice.post import build_voice, speech_events, video_timing

ALL = set(MODELS)
POOL = [v for v in VOICES.values() if v.key != DEFAULT_ROBOT_VOICE]


# ---- assignment --------------------------------------------------------------------------------
PEOPLE = [("P1", "female", 25), ("P2", "male", 71), ("P3", "female", 70), ("P4", "male", 33), ("P5", "female", 45)]


def test_assignment_is_deterministic_per_run_and_changes_with_the_run():
    a = assign_voices(PEOPLE, ALL, seed="run-a")
    assert a == assign_voices(list(reversed(PEOPLE)), ALL, seed="run-a")
    assert [a[p[0]].key for p in PEOPLE] != [v.key for v in assign_voices(PEOPLE, ALL, seed="run-b").values()]


def test_voices_match_sex_and_nobody_in_a_scene_shares_one():
    a = assign_voices(PEOPLE, ALL, seed=3)
    assert all(a[pid].sex == sex for pid, sex, _ in PEOPLE)
    assert len({v.key for v in a.values()}) == len(PEOPLE)


def test_older_people_get_voices_that_suit_an_older_age_first():
    a = assign_voices([("P1", "male", 75), ("P2", "female", 74)], ALL, seed=1)
    assert a["P1"].ages[0] == "older" and a["P2"].ages[0] == "older"


def test_the_robots_voice_is_kept_out_of_the_people_pool():
    models = {"en_GB-cori-medium", "en_GB-alba-medium"}
    a = assign_voices([("P1", "female", 40)], models, reserved=[DEFAULT_ROBOT_VOICE])
    assert a["P1"].key == "en_GB-alba-medium"


def test_a_crowd_larger_than_the_pool_reuses_voices_evenly_and_a_missing_sex_falls_back():
    models = {"en_GB-alba-medium", "en_GB-cori-medium"}  # two women, no man
    people = [(f"P{i}", "male", 40) for i in range(4)]
    a = assign_voices(people, models, seed=0)
    counts = sorted(sum(1 for v in a.values() if v.key == k) for k in ("en_GB-alba-medium", "en_GB-cori-medium"))
    assert counts == [2, 2]


def test_unknown_sex_and_age_still_get_a_stable_voice_and_no_models_gives_nothing():
    a = assign_voices([("panel-1", None, None)], ALL, seed=0)
    assert a == assign_voices([("panel-1", None, None)], ALL, seed=0) and a["panel-1"].key in VOICES
    assert assign_voices(PEOPLE, set()) == {}


def test_the_robot_voice_falls_back_to_the_default_then_to_anything_installed():
    assert robot_voice("en_GB-alba-medium", ALL).key == "en_GB-alba-medium"
    assert robot_voice("nonsense", ALL).key == DEFAULT_ROBOT_VOICE
    assert robot_voice(None, {"en_GB-alba-medium"}).key == "en_GB-alba-medium"
    assert robot_voice(None, set()) is None


def test_the_g1_profile_names_its_voice_and_a_saved_profile_keeps_one(tmp_path):
    from wardzero_agents.robots.profile import G1_CLAUDE, load_profile, save_profile
    import dataclasses
    assert G1_CLAUDE.voice == DEFAULT_ROBOT_VOICE and G1_CLAUDE.voice in VOICES
    save_profile(dataclasses.replace(G1_CLAUDE, name="mine", voice="en_GB-alba-medium", controller="external"), tmp_path)
    assert load_profile("mine", tmp_path).voice == "en_GB-alba-medium"


def test_every_voice_has_a_licence_and_pinned_hashes():
    for m in MODELS.values():
        assert m.licence and len(m.sha256_onnx) == 64 and len(m.sha256_json) == 64
    assert all(v.licence for v in VOICES.values())


# ---- placement and mixing ----------------------------------------------------------------------
def test_start_sample_is_the_video_time_times_the_rate():
    assert mix.start_sample(10.0, 2.0, 22050) == round(8.0 * 22050)
    assert mix.start_sample(12.3456, 0.0, 22050) == round(12.3456 * 22050)
    assert mix.start_sample(1.0, 1.5, 1000) == -500


def tone(seconds, f=220.0, amp=0.8):
    t = np.arange(int(seconds * mix.SAMPLE_RATE)) / mix.SAMPLE_RATE
    return (amp * np.sin(2 * math.pi * f * t)).astype(np.float32)


def test_a_line_lands_at_its_start_sample_and_the_mix_has_the_video_length():
    out = mix.mix([mix.Placed(3.0, tone(0.5))], t0=1.0, duration_s=5.0, pan=False)
    assert out.shape == (5 * mix.SAMPLE_RATE, 2)
    s = mix.start_sample(3.0, 1.0)
    assert np.abs(out[:s]).max() == 0 and np.abs(out[s + 100:s + 5000]).max() > 0.1
    assert np.abs(out[s + int(0.5 * mix.SAMPLE_RATE) + 10:]).max() == 0


def test_overlapping_lines_are_summed_not_cut_and_never_clip():
    items = [mix.Placed(1.0, tone(1.0, 220, 1.0)), mix.Placed(1.2, tone(1.0, 330, 1.0)),
             mix.Placed(1.4, tone(1.0, 440, 1.0)), mix.Placed(1.1, tone(1.0, 550, 1.0), loudness_db=70)]
    out = mix.mix(items, 0.0, 3.0, pan=False)
    assert np.abs(out).max() <= mix.PEAK + 1e-6
    s = mix.start_sample(1.5, 0.0)
    both = out[s:s + 2000, 0]
    solo = mix.mix([items[0]], 0.0, 3.0, pan=False)[s:s + 2000, 0]
    assert not np.allclose(both / np.abs(both).max(), solo / np.abs(solo).max(), atol=0.05)  # other lines are in there


def test_lines_before_the_video_start_are_trimmed_and_lines_past_its_end_are_cut():
    out = mix.mix([mix.Placed(-0.2, tone(0.5)), mix.Placed(2.8, tone(1.0))], 0.0, 3.0, pan=False)
    assert np.abs(out[:int(0.2 * mix.SAMPLE_RATE)]).max() > 0.1  # the tail of the first line plays from sample 0
    assert out.shape[0] == 3 * mix.SAMPLE_RATE and np.abs(out[-1000:]).max() > 0.1


def test_nearer_and_louder_lines_are_louder_and_far_ones_are_attenuated_gently():
    assert mix.gain_db(60, 1) == 0 and mix.gain_db(72, 1) == mix.MAX_BOOST_DB
    assert mix.gain_db(60, 10) == pytest.approx(-10.0) and mix.gain_db(40, 100) == -mix.MAX_CUT_DB
    near = mix.mix([mix.Placed(0, tone(0.3), 60, 1, 0)], 0, 1, listener=(0, 0), pan=False)
    far = mix.mix([mix.Placed(0, tone(0.3), 60, 8, 0), mix.Placed(0.5, tone(0.3), 60, 1, 0)], 0, 1, listener=(0, 0), pan=False)
    assert np.abs(far[:6000]).max() < np.abs(far[11025:17000]).max()  # the distant line sits below the near one
    assert near.max() > 0


def test_pan_puts_a_line_to_the_side_it_came_from():
    left = mix.mix([mix.Placed(0, tone(0.3), 60, -6, 0)], 0, 1, listener=(0, 0))
    right = mix.mix([mix.Placed(0, tone(0.3), 60, 6, 0)], 0, 1, listener=(0, 0))
    assert np.abs(left[:, 0]).max() > 2 * np.abs(left[:, 1]).max()
    assert np.abs(right[:, 1]).max() > 2 * np.abs(right[:, 0]).max()


def test_an_empty_mix_is_silence_of_the_right_length_and_wav_round_trips(tmp_path):
    out = mix.mix([], 0.0, 2.0)
    assert out.shape == (2 * mix.SAMPLE_RATE, 2) and not out.any()
    data = mix.mix([mix.Placed(0.1, tone(0.4))], 0.0, 1.0)
    mix.write_wav(tmp_path / "a" / "v.wav", data)
    back, sr = mix.read_wav(tmp_path / "a" / "v.wav")
    assert sr == mix.SAMPLE_RATE and back.shape == data.shape and np.abs(back - data).max() < 1e-3


# ---- events and the fallback -------------------------------------------------------------------
def make_run(tmp_path, events):
    run = tmp_path / "run"
    run.mkdir()
    (run / "audio.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run


EVENTS = [
    {"t": 2.0, "source_id": "P1", "source_kind": "person", "text": "Excuse me.", "loudness_db": 60, "x": -3, "y": 1, "z": 1.6},
    {"t": 1.0, "source_id": "P2", "source_kind": "cough", "text": None, "loudness_db": 68, "x": -3, "y": 1, "z": 1.6},
    {"t": 3.5, "source_id": "g1", "source_kind": "robot", "text": "Hello, how can I help?", "loudness_db": 72, "x": 1, "y": 1.6, "z": 1.2},
    {"t": 4.0, "source_id": "fire", "source_kind": "alarm", "text": "ALARM", "loudness_db": 90, "x": 0, "y": 0, "z": 0},
    {"t": 5.0, "source_id": "P1", "source_kind": "person", "text": "  ", "loudness_db": 60, "x": 0, "y": 0, "z": 0},
]


def test_only_spoken_lines_are_voiced_in_time_order(tmp_path):
    ev = speech_events(make_run(tmp_path, EVENTS))
    assert [(e["t"], e["source_id"]) for e in ev] == [(2.0, "P1"), (3.5, "g1")]
    assert speech_events(tmp_path / "nowhere") == []


def test_without_models_the_step_warns_and_writes_nothing(tmp_path):
    run = make_run(tmp_path, EVENTS)
    said = []
    res = build_voice(run, models_dir=tmp_path / "empty", say=said.append)
    assert res["wav"] is None and "No voice models" in res["warning"] and "fetch_voices" in said[0]
    assert not (run / "audio").exists()


def test_the_cli_step_never_fails_the_run_when_voices_are_missing(tmp_path, monkeypatch, capsys):
    from wardzero_agents import cli
    run = make_run(tmp_path, EVENTS)
    monkeypatch.setenv("WARDZERO_DATA_DIR", str(tmp_path / "data"))
    cli._post_voice(run)
    assert "No voice models" in capsys.readouterr().out and not (run / "audio").exists()


def test_a_run_without_timing_starts_at_zero_and_one_with_it_reads_the_stamp(tmp_path):
    run = tmp_path / "r"
    (run / "video").mkdir(parents=True)
    assert video_timing(run) == (0.0, None, [])
    (run / "video" / "timing.json").write_text(json.dumps({"t_start_s": 4.25}))
    assert video_timing(run)[0] == 4.25


# ---- the mux (ffmpeg only) ---------------------------------------------------------------------
def make_video(path, seconds=3):
    subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=30:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def streams(path):
    err = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return err


def test_the_mux_adds_an_aac_track_copies_the_video_and_keeps_the_length(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    mp4, wav = tmp_path / "overhead.mp4", tmp_path / "v.wav"
    make_video(mp4, 3)
    mix.write_wav(wav, mix.mix([mix.Placed(0.5, tone(0.8))], 0.0, media.duration_s(mp4)))
    assert media.mux_audio(mp4, wav)
    assert media.mux_audio(mp4, wav)  # again: the old track is replaced, not stacked
    err = streams(mp4)
    assert err.count("Audio: aac") == 1 and "Video: h264" in err
    assert abs(media.duration_s(mp4) - 3.0) < 0.1
    assert not list(tmp_path.glob("*.voice.mp4"))


def test_a_failed_mux_leaves_the_video_alone(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    mp4 = tmp_path / "overhead.mp4"
    make_video(mp4, 1)
    before = mp4.read_bytes()
    assert not media.mux_audio(mp4, tmp_path / "missing.wav")
    assert mp4.read_bytes() == before


# ---- with the models ---------------------------------------------------------------------------
needs_models = pytest.mark.skipif(
    not (pytest.importorskip("importlib").util.find_spec("piper") and {"en_GB-cori-medium", "en_GB-vctk-medium"} <= installed_models()),
    reason="Piper or the voice models are missing (scripts/fetch_voices.py)")


@pytest.mark.voice
@needs_models
def test_two_lines_are_synthesised_with_speech_in_them():
    from wardzero_agents.voice.tts import Synth
    s = Synth()
    a, sr = s.say(VOICES[DEFAULT_ROBOT_VOICE], "Hello, I am the reception robot.", robot=True)
    b, sr2 = s.say(VOICES["en_GB-vctk-medium#p225"], "Please take a seat.")
    for x, rate in ((a, sr), (b, sr2)):
        assert 0.8 < x.size / rate < 8.0 and np.abs(x).max() > 0.1 and np.sqrt((x ** 2).mean()) > 0.01


@pytest.mark.voice
@needs_models
def test_a_whole_run_is_voiced_into_a_wav_and_both_videos_in_sync(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    run = make_run(tmp_path, EVENTS)
    (run / "video").mkdir()
    for n in ("overhead.mp4", "follow.mp4"):
        make_video(run / "video" / n, 6)
    (run / "video" / "timing.json").write_text(json.dumps({"t_start_s": 1.0, "fps": 30}))
    res = build_voice(run, say=lambda m: None)
    wav = run / "audio" / "voice.wav"
    assert res["wav"] == wav and res["lines"] == 2 and len(res["muxed"]) == 2
    data, sr = mix.read_wav(wav)
    assert abs(len(data) / sr - media.duration_s(run / "video" / "overhead.mp4")) < 0.1
    first = mix.start_sample(2.0, 1.0)
    assert np.abs(data[:first - 100]).max() == 0 and np.abs(data[first:first + 8000]).max() > 0.05  # P1 speaks at video 1.0 s
    for n in ("overhead.mp4", "follow.mp4"):
        err = streams(run / "video" / n)
        assert "Audio: aac" in err and abs(media.duration_s(run / "video" / n) - 6.0) < 0.1
