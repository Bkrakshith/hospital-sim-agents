#!/usr/bin/env python3
"""Fetch the NVIDIA GEAR-SONIC weights the walking G1 uses into a cache outside this repository.

Files: the sonic_v1_1 encoder and decoder, their observation config, and the V2 kinematic planner (about 930 MB)
from https://huggingface.co/nvidia/GEAR-SONIC. Every file is checked against the SHA-256 pinned in
wardzero_agents/isaac_host/sonic/weights.py. The weights are never committed.

Licensed by NVIDIA Corporation under the NVIDIA Open Model License (the agreement is on the model page above;
use must also follow https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/). Code: NVlabs
GR00T-WholeBodyControl, Apache-2.0. Cite: Luo et al., arXiv:2511.07820.

  python scripts/fetch_sonic.py                         # download into $WARDZERO_SONIC_DIR or ~/.cache/wardzero/sonic
  python scripts/fetch_sonic.py --use-existing ~/src/gear_sonic/weights
                                                        # verify a folder from download_from_hf.py --sonic-v1-1 and
                                                        # link its files into the cache (no copy, no download)
  python scripts/fetch_sonic.py --check                 # verify what is in the cache

Set HF_TOKEN if Hugging Face asks you to log in.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wardzero_agents.isaac_host.sonic.weights import (FILES, HF_BASE, LICENCE_NOTE, TOTAL_BYTES,  # noqa: E402
                                                      WeightFile, resolve_dir)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified(path: Path, wf: WeightFile) -> bool:
    try:
        return path.stat().st_size == wf.size and sha256(path) == wf.sha256
    except OSError:
        return False


def download(wf: WeightFile, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(f"{HF_BASE}/{wf.hf_path}")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    print(f"  fetching {wf.rel} ({wf.size / 1e6:.0f} MB) ...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except (OSError, ValueError) as exc:  # URLError, HTTPError, timeouts, a full disk
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {wf.hf_path} ({type(exc).__name__}: {exc}). "
                         "Check the network (and HF_TOKEN if the model asks for a login) and run again.")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not verified(tmp, wf):
        tmp.unlink()
        raise SystemExit(f"{wf.rel}: size or SHA-256 does not match the pinned {wf.sha256}; not installed")
    tmp.replace(dest)


def use_existing(src: Path, dest: Path) -> int:
    bad = [wf.rel for wf in FILES if not verified(src / wf.rel, wf)]
    if bad:
        print(f"{src} does not hold the pinned GEAR-SONIC files: {', '.join(bad)}")
        return 1
    if src.resolve() == dest.resolve():
        print(f"verified in place: {dest}")
        return 0
    for wf in FILES:
        link = dest / wf.rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((src / wf.rel).resolve())
    print(f"linked {len(FILES)} verified files from {src} into {dest}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=None, help="cache folder (default $WARDZERO_SONIC_DIR or "
                                                           "~/.cache/wardzero/sonic)")
    p.add_argument("--use-existing", type=Path, metavar="DIR", help="verify and link an existing download instead")
    p.add_argument("--check", action="store_true", help="verify the cache and exit")
    a = p.parse_args()
    dest = resolve_dir(a.dest)
    print(LICENCE_NOTE)
    if a.check:
        bad = [wf.rel for wf in FILES if not verified(dest / wf.rel, wf)]
        print(f"{dest}: " + ("all files verified" if not bad else "missing or wrong: " + ", ".join(bad)))
        return 1 if bad else 0
    dest.mkdir(parents=True, exist_ok=True)
    if a.use_existing:
        return use_existing(a.use_existing.expanduser(), dest)
    todo = [wf for wf in FILES if not verified(dest / wf.rel, wf)]
    need = sum(wf.size for wf in todo)
    free = shutil.disk_usage(dest).free
    if need and free < need + 200e6:
        raise SystemExit(f"Only {free / 1e9:.1f} GB free in {dest}; the weights need {need / 1e9:.1f} GB "
                         f"({TOTAL_BYTES / 1e9:.1f} GB in all). Free some space or set WARDZERO_SONIC_DIR elsewhere.")
    for wf in FILES:
        target = dest / wf.rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if wf in todo:
            download(wf, target)
        else:
            print(f"  have {wf.rel}")
    print(f"GEAR-SONIC weights are in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
