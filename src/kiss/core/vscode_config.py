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
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.core.config import DEFAULT_MAX_BUDGET, kiss_home
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_config_lock = threading.Lock()

if TYPE_CHECKING:
    CONFIG_DIR: Path
    CONFIG_PATH: Path


def _config_dir() -> Path:
    """Return the config directory: test override or lazy $KISS_HOME."""
    override = globals().get("CONFIG_DIR")
    return override if override is not None else kiss_home()


def _config_path() -> Path:
    """Return the ``config.json`` path: test override or lazy resolution."""
    override = globals().get("CONFIG_PATH")
    return override if override is not None else _config_dir() / "config.json"


def __getattr__(name: str) -> Path:
    """Resolve ``CONFIG_DIR``/``CONFIG_PATH`` lazily (see comment above)."""
    if name == "CONFIG_DIR":
        return _config_dir()
    if name == "CONFIG_PATH":
        return _config_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DEFAULTS: dict[str, Any] = {
    "max_budget": DEFAULT_MAX_BUDGET,
    "custom_endpoint": "",
    "custom_api_key": "",
    "custom_headers": "",
    "use_web_browser": True,
    "remote_password": "",
    "auto_commit_mode": True,
    "is_worktree": True,
    "work_dir": "",
    "last_model": "",
}

RETIRED_KEYS: frozenset[str] = frozenset({"demo_mode", "is_parallel"})
"""Settings that used to exist and must be forgotten on sight.

``config.json`` is written by every previous release, so dropping a key
from :data:`DEFAULTS` is not enough: :func:`load_config` overlays whatever
the file holds, :func:`sanitize_config` passes unknown keys through (that
is how genuine extension-owned keys such as ``email`` and ``tunnel_token``
survive), and :func:`save_config` rewrites the file from its own previous
contents.  A retired key would therefore be read back, echoed to every
client in ``configData``, and re-persisted forever.  Listing it here
purges it from both the value read and the file written.

``is_parallel`` is retired rather than merely removed for that reason:
it never had a single reader — whether a run may spawn parallel
sub-agents comes from the run command's ``useParallel`` flag — yet it
was written to every user's ``config.json`` and broadcast in every
``configData``, which invites a future reader to wire it up and
silently disagree with the real source of truth.
"""

API_KEY_ENV_VARS: frozenset[str] = frozenset({
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
})


def get_current_api_keys() -> dict[str, str]:
    """Return the current API key values from the environment.

    Reads each key listed in :data:`API_KEY_ENV_VARS` from ``os.environ``,
    returning an empty string for keys that are not set.

    Returns:
        A dict mapping each API key name to its current value (or ``""``).
    """
    return {k: os.environ.get(k, "") for k in API_KEY_ENV_VARS}


def sanitize_config(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce every :data:`DEFAULTS`-keyed value to its expected type.

    Config values arrive from two untrusted sources — the ``saveConfig``
    payload of any connected client and the user-editable
    ``config.json``.  A junk-typed value used to escape deep into
    handlers: a non-string ``custom_endpoint`` raised
    ``AttributeError`` out of :func:`get_custom_model_entry` (killing
    the ``models`` reply in every window), ``custom_headers`` as a JSON
    list crashed ``splitlines``, a non-string ``work_dir`` corrupted
    the daemon-global working directory, and a truthy non-string
    ``remote_password`` was treated as a genuine password change —
    restarting the kiss-web daemon and killing every in-flight task.

    Coercion rules (per the type of the key's default): booleans accept
    any truthy/falsy value; numbers accept *finite* ints/floats and
    numeric strings — non-finite values (``NaN``/``Infinity`` literals
    survive ``json.load`` and would silently disable every
    ``cost > max_budget`` check) fall back to the default, as does
    anything non-numeric — including booleans, which are ``int``
    subclasses that ``float()`` would otherwise coerce to
    ``1.0``/``0.0``; strings keep only
    genuine ``str`` values (falling back to the default otherwise).
    Non-DEFAULTS keys (e.g. ``tunnel_token``, ``email``) pass through
    untouched, except the retired ones listed in :data:`RETIRED_KEYS`,
    which are dropped.

    Args:
        data: Raw configuration dict.

    Returns:
        A new dict with sanitized values; *data* is not modified.
    """
    result = {k: v for k, v in data.items() if k not in RETIRED_KEYS}
    for key, default in DEFAULTS.items():
        if key not in result:
            continue
        value = result[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                result[key] = bool(value)
        elif isinstance(default, int | float):
            if isinstance(value, bool):
                logger.debug(
                    "Ignoring boolean config value %s=%r", key, value,
                )
                result[key] = default
                continue
            if not isinstance(value, int | float):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    logger.debug(
                        "Ignoring non-numeric config value %s=%r", key, value,
                    )
                    result[key] = default
                    continue
            if not math.isfinite(value):
                logger.debug(
                    "Ignoring non-finite config value %s=%r", key, value,
                )
                result[key] = default
            else:
                result[key] = value
        elif not isinstance(value, str):
            logger.debug(
                "Ignoring non-string config value %s=%r", key, value,
            )
            result[key] = default
    return result


def load_config() -> dict[str, Any]:
    """Load configuration from ``~/.kiss/config.json``.

    Returns a dict with all keys from :data:`DEFAULTS`, falling back to
    default values for any missing keys.  Values of DEFAULTS keys are
    type-sanitized via :func:`sanitize_config` so a hand-edited junk
    value cannot break downstream consumers.
    """
    result = dict(DEFAULTS)
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                result.update(stored)
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read config", exc_info=True)
    return sanitize_config(result)


def save_config(data: dict[str, Any]) -> None:
    """Save configuration to ``~/.kiss/config.json``.

    Merges *data* into the existing file contents, so keys already
    present but absent from *data* are preserved.  Extension-owned keys
    outside :data:`DEFAULTS` (``tunnel_token``, ``skill_permissions``,
    ``mcp_permissions``, ``email``) are written like any other: they are
    read at runtime, so accepting them and then dropping them would be
    silent data loss that only surfaces after a daemon restart.  API
    keys are never written to the config file — they belong in the
    shell RC (see :func:`save_api_key_to_shell`) — and
    :data:`RETIRED_KEYS` are purged on every write.

    The write is **atomic** — content is staged in a sibling temp file
    and then ``os.replace``-d into position so that concurrent readers
    (e.g. the VS Code extension's ``readKissConfig``) never observe an
    empty or partially-written ``config.json``.

    Args:
        data: Configuration dict.
    """
    data = sanitize_config(data)
    cfg_dir = _config_dir()
    cfg_path = _config_path()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with (
        _config_lock,
        open(cfg_dir / ".config.lock", "w", encoding="utf-8") as lock_file,
    ):
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing: dict[str, Any] = {}
            if cfg_path.exists():
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        stored = json.load(f)
                    if isinstance(stored, dict):
                        existing = stored
                except (json.JSONDecodeError, OSError):
                    pass
            for k, v in data.items():
                if k not in API_KEY_ENV_VARS:
                    existing[k] = v
            for k in RETIRED_KEYS:
                existing.pop(k, None)
            # atomic_write_text stages the payload in a sibling temp file
            # through a buffered file object (a bare ``os.write`` may
            # legally write fewer bytes than asked and the truncated file
            # would be published) and ``os.replace``-s it into position.
            atomic_write_text(cfg_path, json.dumps(existing, indent=2))
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

    The read-modify-replace of the RC file runs under the in-process
    lock plus an ``fcntl`` flock keyed to the RC file itself — a
    sidecar ``<rc>.kiss.lock`` beside it: two concurrent savers (two
    daemon threads, or two processes) would otherwise both read the
    same RC snapshot and the second replace would silently drop the
    first saver's key.  The lock lives beside the RC rather than in
    ``$KISS_HOME`` because the RC is selected from ``$HOME``: two
    daemons sharing one HOME but running with different ``KISS_HOME``
    values edit the *same* RC file, so a KISS_HOME-based lock would
    give them two different locks and no exclusion.  The sidecar is
    flocked rather than the RC itself because the RC is atomically
    ``os.replace``-d: a lock taken on the old inode would not exclude
    a writer that opens the new one.

    Args:
        key_name: Environment variable name (e.g. ``"GEMINI_API_KEY"``).
            Must be a valid POSIX identifier — anything else (the name
            arrives from an untrusted client payload) is refused: the
            name is interpolated into the RC line verbatim, so a
            newline or shell metacharacter in it would write arbitrary
            commands into the RC, and ``os.environ`` raises
            ``ValueError`` on names containing ``=``.
        key_value: The API key string.
    """
    if not _ENV_VAR_NAME_RE.fullmatch(key_name):
        logger.warning(
            "Refusing to save API key with invalid name %r", key_name,
        )
        return
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    rc.parent.mkdir(parents=True, exist_ok=True)

    if shell == "fish":
        pattern = f"set -gx {key_name} "
    else:
        pattern = f"export {key_name}="

    # Empty value means delete: remove the export line and clear the env
    if not key_value:
        rc_lock = rc.with_name(rc.name + ".kiss.lock")
        with (
            _config_lock,
            open(rc_lock, "w", encoding="utf-8") as lock_file,
        ):
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if rc.exists():
                    lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
                    new_lines = [
                        line for line in lines if not line.strip().startswith(pattern)
                    ]
                    if len(new_lines) != len(lines):
                        _atomic_write_text_secure(rc, "".join(new_lines))
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        os.environ.pop(key_name, None)
        _refresh_config()
        return

    quoted = shlex.quote(key_value)
    if shell == "fish":
        export_line = f"set -gx {key_name} {quoted}"
    else:
        export_line = f"export {key_name}={quoted}"

    rc_lock = rc.with_name(rc.name + ".kiss.lock")
    with (
        _config_lock,
        open(rc_lock, "w", encoding="utf-8") as lock_file,
    ):
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            lines: list[str] = []
            replaced = False
            if rc.exists():
                lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
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

            _atomic_write_text_secure(rc, "".join(lines))
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    os.environ[key_name] = key_value
    _refresh_config()


def _atomic_write_text_secure(target: Path, content: str) -> None:
    """Write *content* to *target* atomically with mode 0600.

    The RC file holds API keys, so it must never be world-readable and
    must never be observed half-written by a shell that is sourcing it.

    On Windows ``os.chmod`` honours only the read-only bit, but the
    atomic-replace pattern still applies.
    """
    atomic_write_text(target, content, mode=0o600)


def _refresh_config() -> None:
    """Re-read the API keys from the environment into ``DEFAULT_CONFIG``.

    The singleton is updated **in place** rather than rebound to a fresh
    ``Config()``.  A rebuild re-reads only the environment-backed fields,
    so it also reset ``max_budget`` — which is not environment-backed —
    to its declared default, silently discarding whatever
    :func:`apply_config_to_env` had just applied.  That made a settings
    save that changed the budget *and* an API key in one payload lose the
    budget, because the handler applies the budget first and saves the
    keys second.
    """
    from kiss.core import config as config_module

    for key in API_KEY_ENV_VARS:
        setattr(config_module.DEFAULT_CONFIG, key, os.environ.get(key, ""))


def apply_config_to_env(cfg: dict[str, Any]) -> None:
    """Apply loaded config values to the running process.

    Sets ``max_budget`` on the default config.

    A junk value (boolean, non-numeric, or non-finite) falls back to
    ``DEFAULTS['max_budget']``: the value can come from the
    user-editable ``config.json`` (via :func:`load_config`) or from any
    client's ``saveConfig`` payload, and a bare ``float()`` raising out
    of the command handler would kill the whole client connection.  The
    coercion rules are :func:`sanitize_config`'s — it is applied to the
    ``max_budget`` entry so the two can never drift apart.

    Args:
        cfg: The configuration dict (from :func:`load_config`).
    """
    from kiss.core import config as config_module

    budget = cfg.get("max_budget", DEFAULTS["max_budget"])
    sanitized = sanitize_config({"max_budget": budget})
    config_module.DEFAULT_CONFIG.max_budget = float(sanitized["max_budget"])


def _parse_custom_headers(raw_headers: str) -> dict[str, str]:
    """Parse ``Key: Value`` lines from the custom-headers config string.

    Lines without a ``:`` are ignored; keys and values are trimmed.

    Args:
        raw_headers: The multi-line ``custom_headers`` config value.

    Returns:
        Dict of parsed header names to values (empty when none parse).
    """
    headers: dict[str, str] = {}
    for line in raw_headers.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return headers


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
    headers = _parse_custom_headers(cfg.get("custom_headers", ""))
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
    headers = _parse_custom_headers(cfg.get("custom_headers", ""))
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
        rc_q = shlex.quote(str(rc))
        if shell == "fish":
            cmd = f"source {rc_q} 2>/dev/null; env -0 2>/dev/null; or env"
        else:
            cmd = f"source {rc_q} 2>/dev/null; {{ env -0 2>/dev/null || env; }}"
        with subprocess.Popen(
            [shell_path, "-c", cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=sub_env,
        ) as proc:
            try:
                stdout, _stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
        records = (
            stdout.split("\0") if "\0" in stdout else stdout.splitlines()
        )
        for record in records:
            if "=" in record:
                k, _, v = record.partition("=")
                if k in API_KEY_ENV_VARS:
                    os.environ[k] = v
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("Failed to source shell env", exc_info=True)
    _refresh_config()
