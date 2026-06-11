# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Configuration management for the VS Code Sorcar extension.

Persists user preferences to ``~/.kiss/config.json`` and manages
API key injection into shell RC files and the running environment.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Serializes the read-modify-write of ``config.json`` in ``save_config``.
# The on-disk write is already atomic (staged temp file + ``os.replace``),
# but atomicity alone does not prevent a lost update: two threads that
# each read the same old file, overlay only their own keys, and then
# replace it would drop one another's change.  Holding this lock across
# the entire load-merge-store sequence makes concurrent ``save_config``
# calls (e.g. an agent persisting ``last_model`` while the command
# handler persists a settings toggle) serialize so every update survives.
_config_lock = threading.Lock()

# ``KISS_HOME`` overrides the default ``~/.kiss`` location so that
# tests (see ``src/kiss/tests/conftest.py``) can redirect persistent
# state into a temporary directory without clobbering the user's real
# ``~/.kiss/config.json`` — which the running ``kiss-web`` daemon
# watches and would otherwise restart its cloudflared tunnel for every
# time a test calls ``save_config({"remote_password": ...})``.
CONFIG_DIR = Path(os.environ.get("KISS_HOME") or (Path.home() / ".kiss"))
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULTS: dict[str, Any] = {
    "max_budget": 100,
    "custom_endpoint": "",
    "custom_api_key": "",
    "custom_headers": "",
    "use_web_browser": True,
    "remote_password": "",
    # ``auto_commit_mode`` is the persistent "Auto commit" toggle in
    # the menu dropdown.  When True, post-task processing skips the
    # interactive merge/diff workflow and auto-commits the agent's
    # changes.  In worktree mode the worktree is additionally
    # auto-merged into the original branch.  Mirrored via
    # ``update_settings(auto_commit_mode=...)`` (distinct from the
    # one-shot ``auto_commit=True`` trigger used by the legacy icon
    # button).
    "auto_commit_mode": True,
    # Settings below were previously missing from DEFAULTS, causing
    # ``save_config`` to silently drop them.  Added so that
    # ``update_settings`` changes persist across tasks.
    "is_parallel": True,
    "is_worktree": True,
    "demo_mode": False,
    "work_dir": "",
    # ``last_model`` is the most recently selected model name.  It is a
    # persistent user preference (like the toggles above) and is stored
    # here in ``config.json`` rather than the SQLite ``model_usage``
    # table — the database now tracks only per-model usage *counts*, not
    # which model was last selected.
    "last_model": "",
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

    The write is **atomic** — content is staged in a sibling temp file
    and then ``os.replace``-d into position so that concurrent readers
    (e.g. the VS Code extension's ``readKissConfig``) never observe an
    empty or partially-written ``config.json``.

    Args:
        data: Configuration dict.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Serialize the whole read-modify-write so concurrent callers cannot
    # each read the same old file and clobber one another's keys.  The
    # threading lock covers callers inside this process; the ``flock``
    # on a sidecar lock file covers concurrent *processes* (e.g. the
    # kiss-web daemon persisting tunnel state while a VS Code window's
    # service daemon persists ``last_model``) — without it, two daemons
    # that each read the same old file and replace it would drop one
    # another's keys.
    with _config_lock, open(CONFIG_DIR / ".config.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
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
            serialized = json.dumps(existing, indent=2)
            fd, tmp = tempfile.mkstemp(
                prefix=".kiss-config-", dir=str(CONFIG_DIR),
            )
            try:
                os.write(fd, serialized.encode("utf-8"))
            finally:
                os.close(fd)
            os.replace(tmp, CONFIG_PATH)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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


# Fallback absolute locations searched when the shell binary cannot
# be found on ``PATH`` (e.g. when ``source_shell_env`` is invoked from
# a cron job started with a stripped environment).
_SHELL_FALLBACK_PATHS: dict[str, tuple[str, ...]] = {
    "zsh": ("/bin/zsh", "/usr/bin/zsh", "/usr/local/bin/zsh", "/opt/homebrew/bin/zsh"),
    "bash": ("/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash", "/opt/homebrew/bin/bash"),
    "fish": ("/usr/local/bin/fish", "/opt/homebrew/bin/fish", "/usr/bin/fish"),
}


def _resolve_shell_path(shell: str) -> str | None:
    """Return an absolute path to the requested shell binary.

    First consults ``PATH`` via :func:`shutil.which`; when the calling
    process has a minimal or empty ``PATH`` (typical for cron and
    launchd jobs), falls back to well-known absolute installation
    locations.

    Args:
        shell: Short shell name (``'zsh'``, ``'bash'``, or ``'fish'``).

    Returns:
        Absolute path to the shell binary, or ``None`` if no candidate
        exists on disk.
    """
    found = shutil.which(shell)
    if found:
        return found
    for candidate in _SHELL_FALLBACK_PATHS.get(shell, ()):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


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

    # H3 — shell-quote the value so embedded `"`, `$`, backticks, etc.
    # cannot break out of the export line and execute arbitrary commands
    # when the RC is sourced.
    quoted = shlex.quote(key_value)
    if shell == "fish":
        export_line = f"set -gx {key_name} {quoted}"
        pattern = f"set -gx {key_name} "
    else:
        export_line = f"export {key_name}={quoted}"
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

    # H3 — write atomically with mode 0600 so the RC (which now contains
    # API keys) is never world-readable, even momentarily.
    _atomic_write_text_secure(rc, "".join(lines))

    os.environ[key_name] = key_value
    _refresh_config()


def _atomic_write_text_secure(target: Path, content: str) -> None:
    """Write *content* to *target* atomically with mode 0600.

    Uses a temp file in the same directory + ``os.replace`` so that
    readers never observe a partially written RC, and forces the final
    file to be readable only by the owner.

    On Windows ``os.chmod`` honours only the read-only bit, but the
    atomic-replace pattern still applies.
    """
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".kiss-rc-", dir=str(parent))
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        logger.debug("chmod 0600 failed on temp RC", exc_info=True)
    os.replace(tmp, target)
    # Defensive: ensure the destination is 0600 even if it pre-existed
    # with looser permissions (replace preserves the source mode but
    # some filesystems don't honour that contract).
    try:
        os.chmod(target, 0o600)
    except OSError:
        logger.debug("chmod 0600 failed on RC", exc_info=True)


def _refresh_config() -> None:
    """Rebuild ``DEFAULT_CONFIG`` so it picks up new env vars."""
    from kiss.core import config as config_module

    config_module.DEFAULT_CONFIG = config_module.Config()


def apply_config_to_env(cfg: dict[str, Any]) -> None:
    """Apply loaded config values to the running process.

    Sets ``max_budget`` on the default config.

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
    headers: dict[str, str] = {}
    raw_headers = cfg.get("custom_headers", "")
    if raw_headers:
        for line in raw_headers.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()
    entry: dict[str, Any] = {
        "name": f"custom/{endpoint.rstrip('/').split('/')[-1]}",
        "inp": 0,
        "out": 0,
        "uses": 0,
        "vendor": "Custom",
        "endpoint": endpoint,
        "api_key": cfg.get("custom_api_key", ""),
        "extra_headers": headers,
    }
    return entry


def build_model_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build a model_config dict from the settings panel configuration.

    Constructs the ``model_config`` dictionary that can be passed to
    ``agent.run()`` so that the custom endpoint and any custom HTTP
    headers are forwarded to the underlying model client.

    Args:
        cfg: The configuration dict (from :func:`load_config`).

    Returns:
        A model_config dict with ``base_url`` and optionally
        ``extra_headers``, or ``None`` if no custom endpoint is set.
    """
    endpoint = cfg.get("custom_endpoint", "")
    if not endpoint:
        return None
    result: dict[str, Any] = {"base_url": endpoint}
    api_key = cfg.get("custom_api_key", "")
    if api_key:
        result["api_key"] = api_key
    raw_headers = cfg.get("custom_headers", "")
    if raw_headers:
        headers: dict[str, str] = {}
        for line in raw_headers.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()
        if headers:
            result["extra_headers"] = headers
    return result


def source_shell_env() -> None:
    """Source the user's shell RC file and import exported variables.

    This picks up any API keys that were saved via
    :func:`save_api_key_to_shell` during previous sessions.
    """
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    if not rc.exists():
        # On a fresh installation there may be no shell RC file yet, but
        # an API key can still be present in the inherited environment.
        # Refresh ``DEFAULT_CONFIG`` so it reflects ``os.environ`` even
        # when there is nothing to source (mirrors the other early-return
        # path below, which also refreshes).
        _refresh_config()
        return
    shell_path = _resolve_shell_path(shell)
    if shell_path is None:
        logger.warning(
            "Failed to source shell env: %s binary not found on PATH or fallback locations",
            shell,
        )
        _refresh_config()
        return
    # ``source_shell_env`` may run from a cron / launchd context with a
    # stripped ``PATH``.  Augment it with standard system locations so
    # the inner shell can locate ``env`` (and any RC-referenced
    # utilities such as ``brew shellenv``).
    augmented_path = os.pathsep.join(
        p for p in (
            os.environ.get("PATH", ""),
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ) if p
    )
    sub_env = {**os.environ, "PATH": augmented_path}
    try:
        # H1 — shell-quote ``rc`` so a HOME containing single-quotes,
        # spaces, or other metacharacters cannot inject extra shell
        # commands when the RC is sourced.
        rc_q = shlex.quote(str(rc))
        if shell == "fish":
            # fish does not honour shlex.quote's POSIX rules verbatim,
            # but the safe subset (no single-quote, no $) round-trips.
            cmd = f"source {rc_q} 2>/dev/null; env"
        else:
            cmd = f"source {rc_q} 2>/dev/null && env"
        result = subprocess.run(
            [shell_path, "-c", cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            env=sub_env,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                if k in API_KEY_ENV_VARS:
                    os.environ[k] = v
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("Failed to source shell env", exc_info=True)
    _refresh_config()
