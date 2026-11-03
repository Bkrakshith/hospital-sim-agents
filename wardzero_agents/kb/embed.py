"""Embedders: a deterministic local hash embedder (default) and an optional fastembed one."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Signed feature hashing of lower-case word unigrams and bigrams. Stable across processes."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash{dim}"

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out


class FastEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise RuntimeError(
                "The fastembed embedder needs the optional 'fastembed' package. "
                "Install it with: pip install -e .[embed]"
            ) from None
        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        np.divide(v, norms, out=v, where=norms > 0)
        return v


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    m = re.fullmatch(r"hash(\d+)", name)
    if m:
        return HashEmbedder(int(m.group(1)))
    if name == "fastembed":
        return FastEmbedder()
    raise ValueError(f"unknown embedder '{name}' (use 'hash' or 'fastembed')")
