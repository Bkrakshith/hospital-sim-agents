#!/usr/bin/env python3
"""Download the Piper voice models WardZero speaks with into <data>/voices (default ~/.local/share/wardzero-agents/voices).

The models are never committed. Every file is checked against the SHA-256 pinned in wardzero_agents/voice/catalogue.py,
which also holds the licence table (printed with --licences). Piper itself: pip install "wardzero-agents[voice]".

  python scripts/fetch_voices.py                # all voices
  python scripts/fetch_voices.py --only en_GB-cori-medium en_GB-alba-medium
  python scripts/fetch_voices.py --licences
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wardzero_agents.voice.catalogue import HF_BASE, MODELS, voices_dir  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, want: str) -> None:
    if dest.is_file() and sha256(dest) == want:
        print(f"  have {dest.name}")
        return
    tmp = dest.with_name(dest.name + ".part")
    print(f"  fetching {dest.name} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
    except (OSError, ValueError) as exc:  # URLError, HTTPError, timeouts, a full disk
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {dest.name} ({type(exc).__name__}: {exc}). Check the network and run again.")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    got = sha256(tmp)
    if got != want:
        tmp.unlink()
        raise SystemExit(f"{dest.name}: SHA-256 {got} is not the pinned {want}; not installed")
    tmp.replace(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=None, help="folder for the models (default <data>/voices)")
    p.add_argument("--only", nargs="*", metavar="MODEL", help="model names (default all)")
    p.add_argument("--licences", action="store_true", help="print the licence table and exit")
    a = p.parse_args()
    if a.licences:
        for m in MODELS.values():
            print(f"{m.stem:42} {m.licence:38} {m.credit}")
        return 0
    dest = a.dest or voices_dir()
    dest.mkdir(parents=True, exist_ok=True)
    names = a.only or list(MODELS)
    for n in names:
        if n not in MODELS:
            raise SystemExit(f"unknown model {n!r}; choose from: {', '.join(MODELS)}")
    for n in names:
        m = MODELS[n]
        print(f"{n} ({m.licence})")
        fetch(f"{HF_BASE}/{m.path}.onnx", dest / f"{n}.onnx", m.sha256_onnx)
        fetch(f"{HF_BASE}/{m.path}.onnx.json", dest / f"{n}.onnx.json", m.sha256_json)
    print(f"voices are in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
