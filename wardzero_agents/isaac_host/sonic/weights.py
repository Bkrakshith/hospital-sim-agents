"""Where the GEAR-SONIC weights live and how they are loaded. The weights are never part of this repository.

They are under the NVIDIA Open Model License and are downloaded by ``scripts/fetch_sonic.py`` into a cache outside
the repo: ``$WARDZERO_SONIC_DIR``, else ``~/.cache/wardzero/sonic``. The layout is the one the upstream
``download_from_hf.py --sonic-v1-1`` writes, so an existing download folder works as it is.
onnxruntime is imported only when the models are loaded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HF_REPO = "nvidia/GEAR-SONIC"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
ENV_DIR = "WARDZERO_SONIC_DIR"
DEFAULT_DIR = Path("~/.cache/wardzero/sonic")
LICENCE_NOTE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License."


@dataclass(frozen=True)
class WeightFile:
    hf_path: str  # path in the Hugging Face repo
    rel: str  # path under the cache folder
    size: int
    sha256: str


ENCODER = WeightFile("sonic_v1_1/model_encoder.onnx", "policy/sonic_v1_1/model_encoder.onnx", 50_117_796,
                     "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae")
DECODER = WeightFile("sonic_v1_1/model_decoder.onnx", "policy/sonic_v1_1/model_decoder.onnx", 149_778_638,
                     "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509")
# The planner path keeps "V2": the deploy code picks the planner version from the file name.
PLANNER = WeightFile("planner_sonic.onnx", "planner/target_vel/V2/planner_sonic.onnx", 773_952_989,
                     "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea")
OBS_CONFIG = WeightFile("sonic_v1_1/observation_config.yaml", "policy/sonic_v1_1/observation_config.yaml", 2_475,
                        "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0")
FILES: tuple[WeightFile, ...] = (ENCODER, DECODER, PLANNER, OBS_CONFIG)
TOTAL_BYTES = sum(f.size for f in FILES)


def resolve_dir(weights_dir=None) -> Path:
    """The weights folder: the argument, else $WARDZERO_SONIC_DIR, else ~/.cache/wardzero/sonic."""
    d = weights_dir or os.environ.get(ENV_DIR) or DEFAULT_DIR
    return Path(d).expanduser()


def missing_files(weights_dir=None) -> list[str]:
    """The files that are absent or have the wrong size (a quick check; the fetch script verifies SHA-256)."""
    root = resolve_dir(weights_dir)
    out = []
    for f in FILES:
        p = root / f.rel
        try:
            if p.stat().st_size != f.size:
                out.append(f.rel)
        except OSError:
            out.append(f.rel)
    return out


def onnxruntime_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("onnxruntime") is not None


def load_sessions(weights_dir=None, *, threads: int = 4):
    """(encoder, decoder, planner) onnxruntime sessions on the CPU."""
    root = resolve_dir(weights_dir)
    gaps = missing_files(root)
    if gaps:
        raise FileNotFoundError(f"GEAR-SONIC weights missing in {root}: {', '.join(gaps)}. "
                                "Run `python scripts/fetch_sonic.py` in hospital-sim-agents.")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is needed to run GEAR-SONIC: `pip install onnxruntime` "
                          "(Isaac Sim's Python already has it)") from exc
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ["CPUExecutionProvider"]
    return tuple(ort.InferenceSession(str(root / f.rel), so, providers=prov) for f in (ENCODER, DECODER, PLANNER))


def first_present(candidates: Optional[list] = None) -> Optional[Path]:
    """The first folder among the candidates (default: the resolved one) that has every file."""
    for c in candidates or [resolve_dir()]:
        if c is not None and not missing_files(c):
            return Path(c).expanduser()
    return None
