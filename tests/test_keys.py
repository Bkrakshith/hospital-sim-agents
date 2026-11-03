"""Tests for key storage, the start-up check, log redaction and the secret scan."""
import logging
import os
import subprocess
from pathlib import Path

import anthropic
import httpx2
import keyring
import keyring.backend
import openai
import pytest

from wardzero_agents.keys import (
    SERVICE,
    KeyMissing,
    KeyRejected,
    ModelUnavailable,
    check,
    clear_key,
    get_key,
    install_log_redaction,
    make_client,
    redact,
    set_key,
)

# Built by concatenation so this file never contains a literal key-shaped string
# (the pre-commit secret scan would otherwise block committing it).
ANT = "sk-" + "ant-api03-abcdefghijk"
ANT_SHORT = "sk-" + "ant-test-123456789"
PROJ = "sk-" + "proj-abcdefghijklmnopqrstu"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_secrets.sh"


class MemBackend(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        self.d = {}

    def set_password(self, service, username, password):
        self.d[(service, username)] = password

    def get_password(self, service, username):
        return self.d.get((service, username))

    def delete_password(self, service, username):
        self.d.pop((service, username), None)


@pytest.fixture
def memkeyring(monkeypatch):
    old = keyring.get_keyring()
    keyring.set_keyring(MemBackend())
    monkeypatch.delenv("WARDZERO_KEY_FILE", raising=False)
    yield
    keyring.set_keyring(old)


def _resp(code):
    return httpx2.Response(code, request=httpx2.Request("GET", "https://x"))


class FakeModels:
    def __init__(self, raises=None):
        self.raises = raises

    async def retrieve(self, model):
        if self.raises:
            raise self.raises
        return {"id": model}


class FakeClient:
    def __init__(self, raises=None):
        self.models = FakeModels(raises)


def test_service_name():
    assert SERVICE == "wardzero-agents"


def test_set_get_clear(memkeyring):
    set_key("claude", ANT_SHORT)
    assert get_key("claude") == ANT_SHORT
    clear_key("claude")
    assert get_key("claude") is None


def test_clear_missing_is_quiet(memkeyring):
    clear_key("openai")


def test_key_file_mode_refused(memkeyring, tmp_path, monkeypatch):
    f = tmp_path / "k.json"
    f.write_text('{"claude":"%s"}' % ANT_SHORT)
    f.chmod(0o644)
    monkeypatch.setenv("WARDZERO_KEY_FILE", str(f))
    with pytest.raises(KeyMissing, match="mode 600") as ei:
        get_key("claude")
    assert ANT_SHORT not in str(ei.value)


def test_key_file_600_ok(memkeyring, tmp_path, monkeypatch):
    f = tmp_path / "k.json"
    f.write_text('{"claude":"%s","openai":"oa-1"}' % ANT_SHORT)
    f.chmod(0o600)
    monkeypatch.setenv("WARDZERO_KEY_FILE", str(f))
    assert get_key("claude").startswith("sk-")
    assert get_key("openai") == "oa-1"


def test_keyring_wins_over_file(memkeyring, tmp_path, monkeypatch):
    f = tmp_path / "k.json"
    f.write_text('{"claude":"from-file"}')
    f.chmod(0o600)
    monkeypatch.setenv("WARDZERO_KEY_FILE", str(f))
    set_key("claude", "from-ring")
    assert get_key("claude") == "from-ring"


def test_no_key_anywhere(memkeyring):
    assert get_key("claude") is None


def test_make_client_claude_uses_key(memkeyring):
    set_key("claude", "k-123")
    assert isinstance(make_client("claude"), anthropic.AsyncAnthropic)


def test_make_client_claude_env(memkeyring, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert isinstance(make_client("claude"), anthropic.AsyncAnthropic)


def test_make_client_openai(memkeyring, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(KeyMissing, match="hhumans keys set openai"):
        make_client("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert isinstance(make_client("openai"), openai.AsyncOpenAI)
    set_key("openai", "ring-key")
    assert make_client("openai").api_key == "ring-key"


def test_redact():
    assert "sk-ant" not in redact("key " + ANT + " end")
    assert "[redacted]" in redact(PROJ)
    assert redact("plain text") == "plain text"


def test_log_redaction(caplog):
    install_log_redaction()
    install_log_redaction()  # idempotent
    logging.getLogger("x").warning("got %s", ANT)
    logging.getLogger("x").warning("direct " + ANT)
    assert "abcdefghijk" not in caplog.text
    assert "[redacted]" in caplog.text


@pytest.mark.parametrize("mod,name", [(anthropic, "claude"), (openai, "openai")])
async def test_check_maps_errors(mod, name):
    err = mod.AuthenticationError("bad", response=_resp(401), body=None)
    with pytest.raises(KeyRejected):
        await check(name, FakeClient(err), "m")
    err = mod.PermissionDeniedError("no", response=_resp(403), body=None)
    with pytest.raises(KeyRejected):
        await check(name, FakeClient(err), "m")
    err = mod.NotFoundError("nf", response=_resp(404), body=None)
    with pytest.raises(ModelUnavailable):
        await check(name, FakeClient(err), "m")
    await check(name, FakeClient(), "m")


async def test_check_connection_error_propagates():
    err = anthropic.APIConnectionError(request=httpx2.Request("GET", "https://x"))
    with pytest.raises(anthropic.APIConnectionError):
        await check("claude", FakeClient(err), "m")


async def test_check_no_credentials_is_key_missing():
    err = TypeError("Could not resolve authentication method. Expected one of api_key")
    with pytest.raises(KeyMissing, match="ant auth login"):
        await check("claude", FakeClient(err), "m")
    with pytest.raises(KeyMissing, match="hhumans keys set openai"):
        await check("openai", FakeClient(openai.OpenAIError("Missing credentials")), "m")


async def test_check_real_anthropic_without_credentials(memkeyring, monkeypatch):
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-wz")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/nonexistent-wz")
    try:
        client = make_client("claude")
    except KeyMissing:
        return
    with pytest.raises(KeyMissing):
        await check("claude", client, "m")


def _git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def test_check_secrets_script(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "ok.txt").write_text("nothing here\n")
    (tmp_path / "bad.txt").write_text("k=" + ANT + "\n")
    _git(tmp_path, "add", "ok.txt")
    assert subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path).returncode == 0
    _git(tmp_path, "add", "bad.txt")
    r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1
    assert "bad.txt" in r.stdout + r.stderr
    assert "abcdefghijk" not in r.stdout + r.stderr
