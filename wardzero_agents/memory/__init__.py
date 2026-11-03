"""Long-term memory, snapshots and the run lock. Also the data-directory helpers used by the CLI."""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """$WARDZERO_DATA_DIR, default ~/.local/share/wardzero-agents."""
    env = os.environ.get("WARDZERO_DATA_DIR")
    return Path(env).expanduser() if env else Path.home() / ".local" / "share" / "wardzero-agents"


def memory_path(data: Path) -> Path:
    return Path(data) / "memory" / "longterm.sqlite"
