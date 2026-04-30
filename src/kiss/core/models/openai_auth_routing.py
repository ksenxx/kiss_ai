# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI/Codex auth routing helpers."""

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

CODEX_AUTH_DISABLED_FILE = Path("~/.kiss/codex_auth_disabled").expanduser()

_OPENAI_PREFIXES = (
    "chatgpt",
    "gpt",
    "text-embedding",
    "o1",
    "o3",
    "o4",
    "codex",
    "computer-use",
)
_CODEX_SUBSCRIPTION_MODEL_ORDER = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
)
_CODEX_PROVIDER_MODELS = frozenset(_CODEX_SUBSCRIPTION_MODEL_ORDER)
_CODEX_SUBSCRIPTION_MODELS = frozenset(_CODEX_SUBSCRIPTION_MODEL_ORDER)


def _is_openai_family_model(model_name: str) -> bool:
    return model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss")


def _is_codex_subscription_model(model_name: str) -> bool:
    """Return True when model_name is known to be available through Codex auth."""
    return model_name in _CODEX_SUBSCRIPTION_MODELS


def is_codex_provider_model(model_name: str) -> bool:
    """Return True when model_name belongs to the explicit Codex UI provider catalog."""
    return model_name in _CODEX_PROVIDER_MODELS


def codex_subscription_model_sort_order(model_name: str) -> int:
    """Return the official Codex Pro picker order for a subscription model."""
    try:
        return _CODEX_SUBSCRIPTION_MODEL_ORDER.index(model_name)
    except ValueError:
        return len(_CODEX_SUBSCRIPTION_MODEL_ORDER)


@lru_cache(maxsize=1)
def _is_codex_cli_auth_available() -> bool:
    """Return True when Codex CLI is installed and has an authenticated login."""
    if os.environ.get("KISS_DISABLE_CODEX_AUTH") == "1" or CODEX_AUTH_DISABLED_FILE.exists():
        return False
    codex_path = shutil.which("codex")
    if not codex_path:
        return False
    try:
        result = subprocess.run(
            [codex_path, "login", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    status_text = f"{result.stdout}\n{result.stderr}".lower()
    return "logged in" in status_text


@lru_cache(maxsize=1)
def _is_codex_native_auth_available() -> bool:
    """Return True when native Codex OAuth credentials are locally available."""
    if os.environ.get("KISS_DISABLE_CODEX_AUTH") == "1" or CODEX_AUTH_DISABLED_FILE.exists():
        return False
    try:
        from kiss.core.models.codex_oauth import OpenAICodexOAuthManager

        return OpenAICodexOAuthManager().has_credentials()
    except Exception:
        return False


def _is_codex_auth_available() -> bool:
    """Return True when either native OAuth or Codex CLI auth is available."""
    return _is_codex_native_auth_available() or _is_codex_cli_auth_available()


def _resolve_codex_transport() -> str:
    """Resolve Codex transport backend: native first, CLI fallback."""
    forced = os.environ.get("KISS_CODEX_TRANSPORT", "").strip().lower()
    if forced in {"native", "oauth", "direct"}:
        return "native" if _is_codex_native_auth_available() else "cli"
    if forced in {"cli", "exec", "subprocess"}:
        return "cli"
    if _is_codex_native_auth_available():
        return "native"
    return "cli"


def _resolve_openai_auth_mode(model_name: str, openai_api_key: str) -> str:
    """Resolve platform API key auth vs ChatGPT/Codex subscription auth."""
    forced = os.environ.get("KISS_OPENAI_AUTH", "").strip().lower()
    if forced in {"api", "api_key", "platform"}:
        return "api"
    if forced in {"codex", "subscription", "chatgpt"}:
        if _is_codex_auth_available() and _is_codex_subscription_model(model_name):
            return "codex"
        return "api"

    codex_auth = _is_codex_auth_available()
    supports_codex = _is_codex_subscription_model(model_name)
    if codex_auth and supports_codex:
        return "codex"
    if openai_api_key:
        return "api"
    if codex_auth and supports_codex:
        return "codex"
    return "api"
