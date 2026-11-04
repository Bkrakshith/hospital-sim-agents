"""The encoder writes a streamed MP4's header duration from the frame count it was told at the start. A recording does not
know it (0 reads as 143165576.5 s in ffprobe), so the host patches the header with the real count."""
import struct

import pytest

from wardzero_agents.isaac_host import mp4fix


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def mvhd(version, scale, duration):
    if version == 0:
        return box(b"mvhd", struct.pack(">B3xIIII", 0, 0, 0, scale, duration) + b"\0" * 80)
    return box(b"mvhd", struct.pack(">B3xQQIQ", 1, 0, 0, scale, duration) + b"\0" * 80)


def tkhd(version, duration):
    if version == 0:
        return box(b"tkhd", struct.pack(">B3xIIIII", 0, 0, 0, 1, 0, duration) + b"\0" * 60)
    return box(b"tkhd", struct.pack(">B3xQQIIQ", 1, 0, 0, 1, 0, duration) + b"\0" * 60)


def mdhd(version, scale, duration):
    if version == 0:
        return box(b"mdhd", struct.pack(">B3xIIII", 0, 0, 0, scale, duration) + b"\0" * 4)
    return box(b"mdhd", struct.pack(">B3xQQIQ", 1, 0, 0, scale, duration) + b"\0" * 4)


def mp4(version=0, movie_scale=1000, media_scale=15360, duration=0xFFFFFFFF):
    trak = box(b"trak", tkhd(version, duration) + box(b"mdia", mdhd(version, media_scale, duration) + box(b"hdlr", b"x" * 20)))
    return box(b"ftyp", b"isom" + b"\0" * 12) + box(b"moov", mvhd(version, movie_scale, duration) + trak
                                                    + box(b"mvex", b"\0" * 8)) + box(b"moof", b"\0" * 8) + box(b"mdat", b"\1" * 50)


def read(path):
    d = path.read_bytes()
    i = d.index(b"mvhd") + 4
    mv = struct.unpack(">IIII", d[i + 4:i + 20])  # v0: ctime, mtime, timescale, duration
    j = d.index(b"mdhd") + 4
    md = struct.unpack(">IIII", d[j + 4:j + 20])
    k = d.index(b"tkhd") + 4
    tk = struct.unpack(">I", d[k + 20:k + 24])[0]
    return mv[3], tk, md[3]


def test_header_durations_become_the_real_frame_count(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(mp4(movie_scale=1000, media_scale=15360))
    assert mp4fix.set_duration(f, frames=301, fps=30.0) is True
    mv, tk, md = read(f)
    assert mv == 10033 and tk == 10033  # movie timescale (1000/s): 301/30 s
    assert md == round(301 / 30 * 15360)  # media timescale
    d = f.read_bytes()
    assert d.endswith(b"\1" * 50) and len(d) == len(mp4())  # nothing else moved


def test_version_one_boxes_with_64_bit_durations(tmp_path):
    f = tmp_path / "b.mp4"
    f.write_bytes(mp4(version=1, duration=2**40))
    assert mp4fix.set_duration(f, frames=60, fps=30.0) is True
    d = f.read_bytes()
    i = d.index(b"mvhd") + 4
    assert struct.unpack(">Q", d[i + 4 + 8 + 8 + 4:i + 4 + 8 + 8 + 4 + 8])[0] == 2000


def test_a_file_that_is_not_an_mp4_is_left_alone(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"not a video")
    assert mp4fix.set_duration(f, 10, 30.0) is False and f.read_bytes() == b"not a video"
    assert mp4fix.set_duration(tmp_path / "missing.mp4", 10, 30.0) is False


def test_zero_frames_changes_nothing(tmp_path):
    f = tmp_path / "d.mp4"
    f.write_bytes(mp4())
    assert mp4fix.set_duration(f, 0, 30.0) is False and f.read_bytes() == mp4()


def test_mehd_is_patched_when_present(tmp_path):
    f = tmp_path / "e.mp4"
    data = mp4(movie_scale=1000)
    mehd = box(b"mehd", struct.pack(">B3xI", 0, 0xFFFFFFFF))
    data = data.replace(box(b"mvex", b"\0" * 8), box(b"mvex", mehd))
    # the moov box grew: rebuild its size
    i = data.index(b"moov") - 4
    moov_len = struct.unpack(">I", data[i:i + 4])[0] + (len(mehd) - 8)
    data = data[:i] + struct.pack(">I", moov_len) + data[i + 4:]
    f.write_bytes(data)
    assert mp4fix.set_duration(f, 301, 30.0) is True
    d = f.read_bytes()
    assert struct.unpack(">I", d[d.index(b"mehd") + 8:d.index(b"mehd") + 12])[0] == 10033
