"""Configuration management for the VS Code Sorcar extension.

Persists user preferences to ``~/.kiss/config.json`` and manages
API key injection into shell RC files and the running environment.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kiss.core.models.openai_auth_routing import CODEX_AUTH_DISABLED_FILE

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".kiss"
CONFIG_PATH = CONFIG_DIR / "config.json"
_oauth_lock = threading.Lock()
_oauth_pending: dict[str, Any] | None = None
_oauth_last_error = ""
_oauth_completed_states: set[str] = set()
_oauth_callback_server: ThreadingHTTPServer | None = None
_oauth_callback_thread: threading.Thread | None = None

DEFAULTS: dict[str, Any] = {
    "max_budget": 100,
    "custom_endpoint": "",
    "custom_api_key": "",
    "use_web_browser": True,
    "remote_password": "",
}

#: Map from UI label to environment variable name for API keys.
API_KEY_ENV_VARS: dict[str, str] = {
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY": "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    "MINIMAX_API_KEY": "MINIMAX_API_KEY",
}


def get_current_api_keys() -> dict[str, str]:
    """Return the current API key values from the environment.

    Reads each key listed in :data:`API_KEY_ENV_VARS` from ``os.environ``,
    returning an empty string for keys that are not set.

    Returns:
        A dict mapping each API key name to its current value (or ``""``).
    """
    return {k: os.environ.get(k, "") for k in API_KEY_ENV_VARS}


def load_config() -> dict[str, Any]:
    """Load configuration from ``~/.kiss/config.json``.

    Returns a dict with all keys from :data:`DEFAULTS`, falling back to
    default values for any missing keys.
    """
    result = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                result.update(stored)
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read config", exc_info=True)
    return result


def save_config(data: dict[str, Any]) -> None:
    """Save configuration to ``~/.kiss/config.json``.

    Merges incoming DEFAULTS keys with the existing file contents so
    that non-DEFAULTS keys already present (e.g. ``email``,
    ``tunnel_token``) are preserved.  API keys are never written to
    the config file.

    Args:
        data: Configuration dict.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                existing = stored
        except (json.JSONDecodeError, OSError):
            pass
    for k in DEFAULTS:
        if k in data:
            existing[k] = data[k]
    with open(CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def _get_user_shell() -> str:
    """Detect the user's default shell.

    Returns:
        One of ``'zsh'``, ``'bash'``, or ``'fish'``.
    """
    shell = os.environ.get("SHELL", "")
    if "fish" in shell:
        return "fish"
    if "zsh" in shell:
        return "zsh"
    return "bash"


def _shell_rc_path(shell: str) -> Path:
    """Return the RC file path for the given shell type.

    Args:
        shell: One of ``'zsh'``, ``'bash'``, ``'fish'``.

    Returns:
        Path to the shell's configuration file.
    """
    if shell == "fish":
        return Path.home() / ".config" / "fish" / "config.fish"
    if shell == "zsh":
        return Path.home() / ".zshrc"
    return Path.home() / ".bashrc"


def save_api_key_to_shell(key_name: str, key_value: str) -> None:
    """Write an ``export KEY=value`` line to the user's shell RC file.

    If the key already exists in the file, the existing line is replaced.
    Otherwise the new export is appended.

    Also sets the key in the current process environment and refreshes
    the :data:`kiss.core.config.DEFAULT_CONFIG` singleton so subsequent
    model queries see the new key immediately.

    Args:
        key_name: Environment variable name (e.g. ``"GEMINI_API_KEY"``).
        key_value: The API key string.
    """
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    rc.parent.mkdir(parents=True, exist_ok=True)

    if shell == "fish":
        export_line = f"set -gx {key_name} {key_value}"
        pattern = f"set -gx {key_name} "
    else:
        export_line = f'export {key_name}="{key_value}"'
        pattern = f"export {key_name}="

    lines: list[str] = []
    replaced = False
    if rc.exists():
        lines = rc.read_text().splitlines(keepends=True)
        new_lines: list[str] = []
        for line in lines:
            if line.strip().startswith(pattern):
                new_lines.append(export_line + "\n")
                replaced = True
            else:
                new_lines.append(line)
        lines = new_lines

    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(export_line + "\n")

    rc.write_text("".join(lines))

    # Set in current process
    os.environ[key_name] = key_value
    _refresh_config()


def _refresh_config() -> None:
    """Rebuild ``DEFAULT_CONFIG`` so it picks up new env vars."""
    from kiss.core import config as config_module

    config_module.DEFAULT_CONFIG = config_module.Config()


def apply_config_to_env(cfg: dict[str, Any]) -> None:
    """Apply loaded config values to the running process.

    Sets ``max_budget`` on the default config and registers a custom
    endpoint model if configured.

    Args:
        cfg: The configuration dict (from :func:`load_config`).
    """
    from kiss.core import config as config_module

    budget = cfg.get("max_budget", DEFAULTS["max_budget"])
    config_module.DEFAULT_CONFIG.max_budget = float(budget)


def get_custom_model_entry(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build a model-list entry for a custom endpoint if configured.

    Args:
        cfg: The configuration dict.

    Returns:
        A model dict suitable for the ``models`` broadcast list, or None.
    """
    endpoint = cfg.get("custom_endpoint", "")
    if not endpoint:
        return None
    return {
        "name": f"custom/{endpoint.rstrip('/').split('/')[-1]}",
        "inp": 0,
        "out": 0,
        "uses": 0,
        "vendor": "Custom",
        "endpoint": endpoint,
        "api_key": cfg.get("custom_api_key", ""),
    }


def source_shell_env() -> None:
    """Source the user's shell RC file and import exported variables.

    This picks up any API keys that were saved via
    :func:`save_api_key_to_shell` during previous sessions.
    """
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    if not rc.exists():
        return
    try:
        if shell == "fish":
            cmd = f"source {rc} 2>/dev/null; env"
        else:
            cmd = f"source {rc} 2>/dev/null && env"
        result = subprocess.run(
            [shell, "-c", cmd] if shell != "fish" else ["fish", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                if k in API_KEY_ENV_VARS:
                    os.environ[k] = v
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("Failed to source shell env", exc_info=True)
    _refresh_config()


def _clear_codex_auth_caches() -> None:
    try:
        from kiss.core.models import model_info

        model_info._is_codex_cli_auth_available.cache_clear()
        model_info._is_codex_native_auth_available.cache_clear()
    except Exception:
        logger.debug("Failed to clear Codex auth caches", exc_info=True)


def _set_codex_auth_disabled(disabled: bool) -> None:
    try:
        if disabled:
            CODEX_AUTH_DISABLED_FILE.parent.mkdir(parents=True, exist_ok=True)
            CODEX_AUTH_DISABLED_FILE.write_text(str(time.time()), encoding="utf-8")
        elif CODEX_AUTH_DISABLED_FILE.exists():
            CODEX_AUTH_DISABLED_FILE.unlink()
    except OSError:
        logger.debug("Failed to update Codex auth disabled marker", exc_info=True)


def _mask_auth_id(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return value
    return f"{value[:4]}...{value[-4:]}"


def get_codex_auth_status(model_name: str = "") -> dict[str, Any]:
    """Return OpenAI/Codex subscription auth status for the config UI."""
    from kiss.core import config as config_module
    from kiss.core.models import model_info

    requested_model = model_name or "gpt-5.4"
    is_openai_model = model_info._is_openai_family_model(requested_model)
    preferred_auth = model_info._resolve_openai_auth_mode(
        requested_model,
        config_module.DEFAULT_CONFIG.OPENAI_API_KEY,
    )
    codex_account_id = ""
    codex_cache_file = ""
    codex_source_file = ""
    try:
        from kiss.core.models.codex_oauth import (
            CODEX_OAUTH_CALLBACK_PORT,
            OpenAICodexOAuthManager,
        )

        manager = OpenAICodexOAuthManager()
        codex_account_id = manager.get_account_id() or ""
        codex_cache_file = str(manager.cache_file)
        codex_source_file = str(manager.source_file)
        callback_port = CODEX_OAUTH_CALLBACK_PORT
    except Exception:
        callback_port = 1455

    with _oauth_lock:
        login_pending = _oauth_pending is not None
        login_error = _oauth_last_error

    return {
        "model": requested_model,
        "is_openai_model": is_openai_model,
        "preferred_auth": preferred_auth,
        "codex_subscription_model": model_info._is_codex_subscription_model(requested_model),
        "openai_api_key_configured": bool(config_module.DEFAULT_CONFIG.OPENAI_API_KEY),
        "codex_auth_available": model_info._is_codex_auth_available(),
        "codex_native_available": model_info._is_codex_native_auth_available(),
        "codex_cli_available": model_info._is_codex_cli_auth_available(),
        "codex_transport": model_info._resolve_codex_transport(),
        "login_pending": login_pending,
        "login_error": login_error,
        "oauth_callback_port": callback_port,
        "forced_auth": os.environ.get("KISS_OPENAI_AUTH", "").strip().lower() or "auto",
        "forced_transport": os.environ.get("KISS_CODEX_TRANSPORT", "").strip().lower() or "auto",
        "codex_account_id": _mask_auth_id(codex_account_id),
        "codex_cache_file": codex_cache_file,
        "codex_source_file": codex_source_file,
    }


def _shutdown_oauth_callback_server() -> None:
    global _oauth_callback_server, _oauth_callback_thread
    server = _oauth_callback_server
    _oauth_callback_server = None
    _oauth_callback_thread = None
    if server is not None:
        try:
            server.shutdown()
            server.server_close()
        except OSError:
            logger.debug("Failed to stop Codex OAuth callback server", exc_info=True)


def _ensure_oauth_callback_server() -> tuple[bool, str]:
    global _oauth_callback_server, _oauth_callback_thread, _oauth_pending, _oauth_last_error
    if _oauth_callback_server is not None:
        return True, ""

    try:
        from kiss.core.models.codex_oauth import (
            CODEX_OAUTH_CALLBACK_PORT,
            OpenAICodexOAuthManager,
        )
    except ImportError as exc:
        return False, str(exc)

    class _OAuthCallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            global _oauth_pending, _oauth_last_error
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("code", [""])[0]
            state = params.get("state", [""])[0]
            error = params.get("error", [""])[0]

            status_code = 200
            message = "OpenAI Codex login completed. You can close this window."
            with _oauth_lock:
                pending = _oauth_pending
                if error:
                    _oauth_last_error = error
                    status_code = 400
                    message = f"OpenAI Codex login failed: {error}"
                elif not pending:
                    if state and state in _oauth_completed_states:
                        _oauth_last_error = ""
                    else:
                        _oauth_last_error = "OAuth callback did not match an active login."
                        status_code = 400
                        message = (
                            "OpenAI Codex login failed: callback did not match "
                            "an active login."
                        )
                elif state != pending.get("state"):
                    if OpenAICodexOAuthManager().has_credentials():
                        _oauth_last_error = ""
                        message = "OpenAI Codex login already completed. You can close this window."
                    else:
                        _oauth_last_error = "OAuth state mismatch."
                        status_code = 400
                        message = "OpenAI Codex login failed: OAuth state mismatch."
                elif time.time() > float(pending.get("expires_at", 0)):
                    _oauth_last_error = "OAuth login expired."
                    status_code = 400
                    message = "OpenAI Codex login failed: login expired."
                elif not code:
                    _oauth_last_error = "OAuth callback did not include a code."
                    status_code = 400
                    message = "OpenAI Codex login failed: missing code."
                else:
                    verifier = str(pending.get("code_verifier", ""))
                    pending_state = str(pending.get("state", ""))
                    token = OpenAICodexOAuthManager().exchange_authorization_code(code, verifier)
                    if token:
                        _oauth_pending = None
                        if pending_state:
                            _oauth_completed_states.add(pending_state)
                        _set_codex_auth_disabled(False)
                        _oauth_last_error = ""
                        _clear_codex_auth_caches()
                    else:
                        _oauth_last_error = "Token exchange failed."
                        status_code = 400
                        message = "OpenAI Codex login failed: token exchange failed."

            body = (
                "<!doctype html><html><body>"
                f"<p>{message}</p>"
                "</body></html>"
            ).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = ThreadingHTTPServer(
            ("127.0.0.1", CODEX_OAUTH_CALLBACK_PORT),
            _OAuthCallbackHandler,
        )
    except OSError as exc:
        return False, str(exc)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _oauth_callback_server = server
    _oauth_callback_thread = thread
    return True, ""


def start_codex_login(model_name: str = "") -> dict[str, Any]:
    """Start OpenAI Codex OAuth login and return a browser URL."""
    global _oauth_pending, _oauth_last_error
    try:
        from kiss.core.models.codex_oauth import (
            CODEX_OAUTH_REDIRECT_URI,
            build_authorization_url,
            generate_code_challenge,
            generate_code_verifier,
            generate_oauth_state,
        )
    except ImportError as exc:
        return {"status": "error", "error": str(exc), "auth": get_codex_auth_status(model_name)}

    verifier = generate_code_verifier()
    state = generate_oauth_state()
    login_url = build_authorization_url(
        generate_code_challenge(verifier),
        state,
        originator="kiss-ai",
        redirect_uri=CODEX_OAUTH_REDIRECT_URI,
    )
    with _oauth_lock:
        _oauth_pending = {
            "state": state,
            "code_verifier": verifier,
            "created_at": time.time(),
            "expires_at": time.time() + 300.0,
        }
        _oauth_last_error = ""

    started, error = _ensure_oauth_callback_server()
    if not started:
        with _oauth_lock:
            _oauth_pending = None
            _oauth_last_error = error
        return {"status": "error", "error": error, "auth": get_codex_auth_status(model_name)}

    try:
        webbrowser.open(login_url)
    except Exception:
        logger.debug("Failed to open Codex OAuth URL", exc_info=True)
    return {
        "status": "ok",
        "login_url": login_url,
        "auth": get_codex_auth_status(model_name),
    }


def cancel_codex_login(model_name: str = "") -> dict[str, Any]:
    """Cancel a pending OpenAI Codex OAuth login."""
    global _oauth_pending, _oauth_last_error
    with _oauth_lock:
        _oauth_pending = None
        _oauth_last_error = ""
    _shutdown_oauth_callback_server()
    return {"status": "ok", "auth": get_codex_auth_status(model_name)}


def logout_codex(model_name: str = "") -> dict[str, Any]:
    """Remove KISS-managed Codex OAuth cache and refresh auth status."""
    try:
        from kiss.core.models.codex_oauth import OpenAICodexOAuthManager

        manager = OpenAICodexOAuthManager()
        if manager.cache_file.exists():
            manager.cache_file.unlink()
        manager._credentials = None
    except OSError:
        logger.debug("Failed to remove Codex OAuth cache", exc_info=True)
    except Exception:
        logger.debug("Failed to initialize Codex OAuth manager", exc_info=True)
    _set_codex_auth_disabled(True)
    _clear_codex_auth_caches()
    return {"status": "ok", "auth": get_codex_auth_status(model_name)}
