"""The README keeps the sections and the one sentence users must not miss."""
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

HEADINGS = [
    "## Accounts and API keys", "## Running", "## Control panel", "## Memory",
    "## Knowledge base", "## Re-running and replay", "## Perception modes",
    "## Motion data", "## Costs",
]


def test_readme_sections():
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    for h in HEADINGS:
        assert h in lines, f"missing heading {h}"
    assert "A Claude Pro or Max subscription cannot be used for API calls" in text
