"""Patch the header duration of a streamed MP4.

Isaac's encoder (``omni.videoencoding``) writes a fragmented MP4 and takes the clip's length from the ``nframes`` it was
given at ``start_encoding``. A recording streams frames without knowing how many there will be (0 reads back as
143165576.5 s, and a wrong guess reads back as the guess), but the fragments themselves are right. This sets the three
header durations (``mvhd``, ``tkhd``, ``mdhd``) from the real count, in place.
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path

CONTAINERS = (b"moov", b"trak", b"mdia")


def _boxes(buf, start: int, end: int):
    """(kind, payload_start, box_end) for each box in [start, end)."""
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", buf[i:i + 8])
        head = 8
        if size == 1 and i + 16 <= end:
            size, head = struct.unpack(">Q", buf[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < head or i + size > end:
            return
        yield kind, i + head, i + size
        i += size


def _duration_fields(buf, kind: bytes, p: int) -> tuple[int, int, int]:
    """(duration offset, duration size, timescale offset or -1) of a full box payload at p."""
    v1 = buf[p] == 1
    if kind == b"tkhd":  # version, flags, ctime, mtime, track id, reserved, duration
        return (p + 4 + 16 + 4 + 4 if v1 else p + 4 + 8 + 4 + 4), (8 if v1 else 4), -1
    scale = p + 4 + (16 if v1 else 8)  # mvhd, mdhd: version, flags, ctime, mtime, timescale, duration
    return scale + 4, (8 if v1 else 4), scale


def _patch(buf, start: int, end: int, frames: int, fps: float, movie_scale: list) -> int:
    n = 0
    for kind, p, e in _boxes(buf, start, end):
        if kind in CONTAINERS:
            n += _patch(buf, p, e, frames, fps, movie_scale)
        elif kind in (b"mvhd", b"tkhd", b"mdhd"):
            off, size, scale_at = _duration_fields(buf, kind, p)
            if scale_at >= 0:
                scale = struct.unpack(">I", buf[scale_at:scale_at + 4])[0]
                if kind == b"mvhd":
                    movie_scale[0] = scale
            else:
                scale = movie_scale[0]  # a track's duration is in the movie's timescale (mvhd comes first)
            value = round(frames / fps * scale)
            buf[off:off + size] = value.to_bytes(size, "big")
            n += 1
    return n


def set_duration(path: Path, frames: int, fps: float) -> bool:
    """Write ``frames`` / ``fps`` seconds into the headers; False when there is nothing to patch or no MP4."""
    if frames <= 0 or fps <= 0:
        return False
    try:
        with open(path, "r+b") as f, mmap.mmap(f.fileno(), 0) as buf:
            return _patch(buf, 0, len(buf), frames, fps, [1000]) >= 3
    except (OSError, ValueError, struct.error):  # missing, empty, or not a box file
        return False
