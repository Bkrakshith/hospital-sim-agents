"""API key storage, the start-up key check and log redaction.

Every user brings their own Claude or OpenAI account. Keys live in the OS keyring
(or a mode-600 JSON file) and never appear in logs, topics, run folders or git.
Exception messages never contain a key or key file contents.
"""
from __future__ import annotations

import json
import logging
import os
import re
import stat
from typing import Any, Literal

import anthropic
import keyring
import keyring.errors
import openai

SERVICE = "wardzero-agents"
Provider = Literal["claude", "openai"]

_FIX = {
    "claude": "No working Claude key. Run `hhumans keys set claude` or `ant auth login`.",
    "openai": "No working OpenAI key. Run `hhumans keys set openai`.",
}


class KeyMissing(Exception):
    """No credential found for the provider."""


class KeyRejected(Exception):
    """The provider refused the credential."""


class ModelUnavailable(Exception):
    """The provider does not offer the requested model to this account."""


def set_key(provider: Provider, secret: str) -> None:
    keyring.set_password(SERVICE, provider, secret)


def clear_key(provider: Provider) -> None:
    try:
        keyring.delete_password(SERVICE, provider)
    except keyring.errors.PasswordDeleteError:
        pass


def _file_key(provider: str) -> str | None:
    path = os.environ.get("WARDZERO_KEY_FILE")
    if not path:
        return None
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return None
    if mode & 0o077:
        raise KeyMissing(
            "WARDZERO_KEY_FILE key file must be mode 600 (run `chmod 600` on it). "
            + _FIX[provider]
        )
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        raise KeyMissing("WARDZERO_KEY_FILE could not be read as JSON. " + _FIX[provider]) from None
    value = data.get(provider) if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def get_key(provider: Provider) -> str | None:
    """Keyring first, then the mode-600 key file, else None."""
    try:
        value = keyring.get_password(SERVICE, provider)
    except keyring.errors.KeyringError:
        value = None
    if value:
        return value
    return _file_key(provider)


def make_client(provider: Provider):
    key = get_key(provider)
    if provider == "claude":
        try:
            return anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
        except (TypeError, anthropic.AnthropicError):
            raise KeyMissing(_FIX["claude"]) from None
    if provider == "openai":
        key = key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise KeyMissing(_FIX["openai"])
        return openai.AsyncOpenAI(api_key=key)
    raise ValueError(f"unknown provider {provider!r}")


def _no_credentials(exc: BaseException) -> bool:
    text = str(exc).lower()
    if isinstance(exc, TypeError):
        return "authentication" in text or "api_key" in text
    return isinstance(exc, (anthropic.AnthropicError, openai.OpenAIError)) and (
        "credentials" in text or "api_key" in text or "authentication method" in text
    )


async def check(provider: Provider, client: Any, model: str) -> None:
    """Prove the credential works and the model is reachable. Raises a fix-it error."""
    mod = anthropic if provider == "claude" else openai
    try:
        await client.models.retrieve(model)
    except (mod.AuthenticationError, mod.PermissionDeniedError):
        raise KeyRejected(_FIX[provider]) from None
    except mod.NotFoundError:
        raise ModelUnavailable(
            f"Model {model} is not available to this {provider} account."
        ) from None
    except mod.APIStatusError:
        raise
    except (TypeError, anthropic.AnthropicError, openai.OpenAIError) as exc:
        if _no_credentials(exc):
            raise KeyMissing(_FIX[provider]) from None
        raise


_ANT = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
_SK = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def redact(text: str) -> str:
    return _SK.sub("[redacted]", _ANT.sub("[redacted]", text))


def _redact_arg(arg: Any) -> Any:
    return redact(arg) if isinstance(arg, str) else arg


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(_redact_arg(a) for a in args)
        elif isinstance(args, dict):
            record.args = {k: _redact_arg(v) for k, v in args.items()}
        return True


_filter = _RedactFilter()
_factory_installed = False


def install_log_redaction() -> None:
    """Redact key-shaped strings from every log record (idempotent).

    A filter on the root logger only sees records logged directly on it, so the
    record factory is wrapped too; that covers records from every logger.
    """
    global _factory_installed
    root = logging.getLogger()
    if _filter not in root.filters:
        root.addFilter(_filter)
    if not _factory_installed:
        old = logging.getLogRecordFactory()

        def factory(*a, **kw):
            record = old(*a, **kw)
            _filter.filter(record)
            return record

        logging.setLogRecordFactory(factory)
        _factory_installed = True
