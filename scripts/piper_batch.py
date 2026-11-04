#!/usr/bin/env python3
"""Piper batch runner: the only code of WardZero that imports Piper (GPL-3), run as its own process so the Apache-2.0
package never links it. Reads a JSON job file, writes one WAV per line, prints a one-line JSON summary.

    python scripts/piper_batch.py JOBS.json

JOBS.json: {"jobs": [{"model": "/path/en_GB-x-medium.onnx", "speaker": "p225" | null, "length_scale": 1.0,
                      "text": "...", "out": "/path/line.wav"}, ...]}
Each model is loaded once and all its lines are synthesised together. Exit 3: Piper is not installed.
"""
import json
import sys
import time
import wave


def main(path: str) -> int:
    try:
        from piper import PiperVoice, SynthesisConfig
    except ImportError:
        print("piper-tts is not installed", file=sys.stderr)
        return 3
    jobs = json.load(open(path, encoding="utf-8"))["jobs"]
    by_model: dict = {}
    for j in jobs:
        by_model.setdefault(j["model"], []).append(j)
    t0 = time.monotonic()
    for model, items in by_model.items():
        voice = PiperVoice.load(model)
        for j in items:
            sid = voice.config.speaker_id_map[j["speaker"]] if j.get("speaker") else None
            cfg = SynthesisConfig(speaker_id=sid, length_scale=float(j.get("length_scale") or 1.0))
            with wave.open(j["out"], "wb") as w:
                voice.synthesize_wav(j["text"], w, cfg)
    print(json.dumps({"lines": len(jobs), "seconds": round(time.monotonic() - t0, 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
