"""Brute-force cosine vector index over chunks, and the default research-notes knowledge base."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from .chunker import Chunk, chunk_markdown
from .embed import Embedder

if TYPE_CHECKING:
    from ..config.loader import LoadedScenario
    from ..config.models import PersonInstance


def _embed_text(c: Chunk) -> str:
    return " ".join([*c.heading_path, c.text])


def _hash_sources(pairs: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted(pairs):
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return h.hexdigest()


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, sources_sha256: str = ""):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.sources_sha256 = sources_sha256

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, sources_sha256: str = "") -> "KnowledgeBase":
        chunks = list(chunks)
        if chunks:
            vectors = np.asarray(embedder.embed([_embed_text(c) for c in chunks]), dtype=np.float32)
        else:
            vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(chunks, vectors, embedder, sources_sha256 or _hash_sources([]))

    def save(self, directory: "str | Path") -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(d / "vectors.npy", self.vectors)
        meta = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sources_sha256": self.sources_sha256,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: "str | Path", embedder: Embedder) -> "KnowledgeBase":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta["embedder"] != embedder.name or meta["dim"] != embedder.dim:
            raise ValueError(
                f"kb index was built with embedder {meta['embedder']} (dim {meta['dim']}), "
                f"not {embedder.name} (dim {embedder.dim}); rebuild the index"
            )
        chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        vectors = np.load(d / "vectors.npy")
        return cls(chunks, vectors, embedder, meta.get("sources_sha256", ""))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def _persona_text(instance: "PersonInstance", loaded: "LoadedScenario") -> str:
    """Plain YAML dump of a person. Task 8 replaces this with persona_card()."""
    return yaml.safe_dump(instance.model_dump(), sort_keys=False, allow_unicode=True)


def _area_text(loaded: "LoadedScenario") -> str:
    lines = ["Waiting area landmarks (id, label, x, y):"]
    for lid, lm in loaded.area.landmarks.items():
        lines.append(f"- {lid}: {lm.label} at ({lm.x}, {lm.y})")
    lines.append("Signs:")
    for s in loaded.signs:
        lines.append(f"- {s.text}")
    return "\n".join(lines)


def build_default_kb(hg1_root: Path, loaded: "LoadedScenario", embedder: Embedder) -> KnowledgeBase:
    root = Path(hg1_root)
    chunks: list[Chunk] = []
    pairs: list[tuple[str, str]] = []
    for md in sorted((root / "research_notes").rglob("*.md")):
        chunks.extend(chunk_markdown(md, root))
        pairs.append((md.relative_to(root).as_posix(), hashlib.sha256(md.read_bytes()).hexdigest()))
    for i, inst in enumerate(loaded.instances):
        chunks.append(Chunk(f"<persona>#{i}", "<persona>", [inst.person_id], _persona_text(inst, loaded), []))
    chunks.append(Chunk("<area>#0", "<area>", ["waiting area"], _area_text(loaded), []))
    return KnowledgeBase.build(chunks, embedder, _hash_sources(pairs))
