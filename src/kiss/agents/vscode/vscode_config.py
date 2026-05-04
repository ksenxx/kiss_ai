"""Configuration management for the VS Code Sorcar extension.

Persists user preferences to ``~/.kiss/config.json`` and manages
API key injection into shell RC files and the running environment.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".kiss"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULTS: dict[str, Any] = {
    "max_budget": 100,
    "custom_endpoint": "",
    "custom_api_key": "",
    "use_web_browser": True,
    "remote_password": "",
}

API_KEY_ENV_VARS: frozenset[str] = frozenset({
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "MINIMAX_API_KEY",
})


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
