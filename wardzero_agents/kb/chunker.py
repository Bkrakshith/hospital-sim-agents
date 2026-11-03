"""Split markdown files into heading-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*#*\s*$")
_URL = re.compile(r"https?://[^\s<>()\[\]\"']+")


@dataclass
class Chunk:
    id: str
    file: str
    heading_path: list[str]
    text: str
    links: list[str] = field(default_factory=list)


def _links(text: str) -> list[str]:
    seen: list[str] = []
    for m in _URL.findall(text):
        url = m.rstrip(".,;:!?*_`")
        if url not in seen:
            seen.append(url)
    return seen


def _split(body: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most max_chars; hard-split a paragraph that is too long."""
    pieces: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:
            cut = para.rfind("\n", 0, max_chars)
            if cut <= 0:
                cut = para.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if cur and len(cur) + 2 + len(para) > max_chars:
            pieces.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        pieces.append(cur)
    return pieces


def chunk_markdown(path: Path, root: Path, max_chars: int = 1200) -> list[Chunk]:
    path, root = Path(path), Path(root)
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            sections.append(([t for _, t in stack], []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for heading_path, lines in sections:
        for piece in _split("\n".join(lines), max_chars):
            chunks.append(Chunk(f"{rel}#{len(chunks)}", rel, list(heading_path), piece, _links(piece)))
    return chunks
