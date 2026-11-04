"""keys.py imports without the provider SDKs or keyring (the Isaac host's Python has none of them)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCK = "import sys\nfor m in ('anthropic', 'openai', 'keyring'):\n    sys.modules[m] = None\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + code], cwd=ROOT, capture_output=True, text=True,
                          timeout=60)


def test_keys_and_runlog_import_without_sdks():
    r = _run("import wardzero_agents.keys as k, wardzero_agents.runlog\n"
             "k.install_log_redaction()\n"
             "print(k.redact('sk-' + 'ant-api03-abcdefghijk'))\n")
    assert r.returncode == 0, r.stderr
    assert "[redacted]" in r.stdout


def test_sim_service_imports_without_sdks():
    r = _run("import wardzero_agents.ros.sim_service, wardzero_agents.isaac_host.bodies, "
             "wardzero_agents.isaac_host.stage, wardzero_agents.isaac_host.service\n")
    assert r.returncode == 0, r.stderr


def test_sdk_needing_calls_still_import_lazily():
    r = _run("import wardzero_agents.keys as k\n"
             "try:\n    k.make_client('claude')\nexcept ImportError:\n    print('lazy')\n")
    assert r.returncode == 0, r.stderr
    assert "lazy" in r.stdout
