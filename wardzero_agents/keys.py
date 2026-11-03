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
        raise KeyMissing(
            "WARDZERO_KEY_FILE is set but the file is missing or unreadable. " + _FIX[provider]
        ) from None
    if mode & 0o077:
        raise KeyMissing(
            "WARDZERO_KEY_FILE key file must be mode 600 (run `chmod 600` on it). "
            + _FIX[provider]
        )
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        raise KeyMissing(
            "WARDZERO_KEY_FILE could not be read as JSON. " + _FIX[provider]
        ) from None
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


def _no_credentials(provider: str, exc: BaseException) -> bool:
    text = str(exc)
    if provider == "claude":
        return isinstance(exc, TypeError) and "authentication" in text.lower()
    return isinstance(exc, openai.OpenAIError) and (
        "api_key client option must be set" in text or "Missing credentials" in text
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
    except (TypeError, openai.OpenAIError) as exc:
        if _no_credentials(provider, exc):
            raise KeyMissing(_FIX[provider]) from None
        raise


_ANT = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
_SK = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def redact(text: str) -> str:
    return _SK.sub("[redacted]", _ANT.sub("[redacted]", text))


def _redact_arg(arg: Any) -> Any:
    if isinstance(arg, (int, float, bool)) or arg is None:
        return arg
    return redact(arg if isinstance(arg, str) else str(arg))


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg if isinstance(record.msg, str) else str(record.msg))
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(_redact_arg(a) for a in args)
        elif isinstance(args, dict):
            record.args = {k: _redact_arg(v) for k, v in args.items()}
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


_filter = _RedactFilter()
_installed = False


def install_log_redaction() -> None:
    """Redact key-shaped strings from all log output (idempotent).

    Records are scrubbed at creation (record factory, covers every logger) and the
    final formatted text, tracebacks included, is scrubbed in Formatter.format.
    """
    global _installed
    root = logging.getLogger()
    if _filter not in root.filters:
        root.addFilter(_filter)
    if _installed:
        return
    old_factory = logging.getLogRecordFactory()

    def factory(*a, **kw):
        record = old_factory(*a, **kw)
        _filter.filter(record)
        return record

    logging.setLogRecordFactory(factory)
    old_format = logging.Formatter.format

    def format(self, record):  # noqa: A001
        return redact(old_format(self, record))

    logging.Formatter.format = format
    _installed = True
