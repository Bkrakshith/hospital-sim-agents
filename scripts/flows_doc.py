#!/usr/bin/env python3
"""Write docs/scenario-flows.md from the preset flows (config/presets/flows/*.yaml).

    python scripts/flows_doc.py          # write it
    python scripts/flows_doc.py --check  # exit 1 when the file is out of date
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wardzero_agents.flows.render import markdown  # noqa: E402
from wardzero_agents.flows.spec import flows_dir, load_flow  # noqa: E402

DOC = ROOT / "docs" / "scenario-flows.md"


def text() -> str:
    return markdown([load_flow(p) for p in sorted(flows_dir(ROOT / "config").glob("*.yaml"))])


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if DOC.exists() and DOC.read_text() == text() else 1)
    DOC.write_text(text())
    print(f"wrote {DOC}")
