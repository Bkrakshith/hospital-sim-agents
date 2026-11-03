"""Shared image helpers."""
from __future__ import annotations


def media_type(b: bytes) -> str:
    if b.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if b[:4] == b"RIFF":
        return "image/webp"
    return "image/png"
