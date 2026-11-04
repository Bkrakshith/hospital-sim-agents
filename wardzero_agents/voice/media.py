"""ffmpeg helpers: the binary (imageio-ffmpeg's, else the system one), durations and the audio mux."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def duration_s(path: Path) -> Optional[float]:
    """Container duration from ffmpeg's banner; None when it cannot be read."""
    try:
        err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                             timeout=60).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def mux_audio(video: Path, wav: Path, aac_kbps: int = 128) -> bool:
    """Replace ``video`` by a copy of it (the picture is not re-encoded) with the WAV as its AAC track. Any audio the
    video already had is dropped, so the step can be repeated. False when ffmpeg fails (the video is left as it was)."""
    video = Path(video)
    tmp = video.with_name(video.stem + ".voice.mp4")
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", f"{aac_kbps}k", "-shortest", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
        return False
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, video)
    return True
