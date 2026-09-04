# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Configuration management for the VS Code Sorcar extension.

Persists user preferences to ``~/.kiss/config.json`` and manages the
canonical API-key store ``$KISS_HOME/api_keys.env`` — the one file both
local and remote installs load keys from at daemon startup
(:func:`load_api_keys`) and the one file the settings panel writes them
to (:func:`save_api_key`).
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
import signal
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiss.core.config import DEFAULT_MAX_BUDGET, kiss_home
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Reentrant: save_api_key holds it across the whole file-plus-environment
# update while delegating the canonical-file edit to _edit_api_keys_env_file,
# which takes it again for callers (key migration) that edit only the file.
_config_lock = threading.RLock()

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
    "ANTHROPIC_WORKSPACE_ID",
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


def _read_stored_config(cfg_path: Path) -> dict[str, Any]:
    """Return the JSON object stored in *cfg_path*, or ``{}`` if unusable.

    The single reader behind :func:`load_config` and :func:`save_config`,
    so both agree on what "unreadable" means: a missing file, an
    ``OSError``, or any ``ValueError`` from decoding — invalid JSON
    (``json.JSONDecodeError``) and invalid UTF-8 (``UnicodeDecodeError``)
    alike.  The file is user-editable, so both kinds of junk happen; a
    top-level value that is not a JSON object is ignored too.
    """
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            stored = json.load(f)
    except (ValueError, OSError):
        logger.debug("Failed to read config %s", cfg_path, exc_info=True)
        return {}
    return stored if isinstance(stored, dict) else {}


def load_config() -> dict[str, Any]:
    """Load configuration from ``~/.kiss/config.json``.

    Returns a dict with all keys from :data:`DEFAULTS`, falling back to
    default values for any missing keys.  Values of DEFAULTS keys are
    type-sanitized via :func:`sanitize_config` so a hand-edited junk
    value cannot break downstream consumers.
    """
    result = dict(DEFAULTS)
    result.update(_read_stored_config(_config_path()))
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
    canonical key store (see :func:`save_api_key`) — and
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
            existing = _read_stored_config(cfg_path)
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


def _rc_line_sets_key(line: str, shell: str, key_name: str) -> bool:
    """Return True if the RC *line* assigns the variable *key_name*.

    Recognizes the canonical lines this module writes (``export KEY=…``
    for POSIX shells, ``set -gx KEY …`` for fish) plus valid
    horizontal-whitespace variants a user may have hand-written, such
    as ``export<TAB>KEY=…``.  A literal-prefix match would skip such a
    line, so a delete would leave it behind and a fresh shell would
    silently restore the key the settings panel just removed.  The
    trailing ``=`` (POSIX) / whitespace (fish) anchors the match so
    ``KEY`` never matches ``KEY_EXTRA``.

    Args:
        line: One physical line of the RC file.
        shell: One of ``'zsh'``, ``'bash'``, ``'fish'``.
        key_name: Environment variable name to look for.
    """
    if shell == "fish":
        pattern = rf"\s*set\s+-gx\s+{re.escape(key_name)}(\s|$)"
    else:
        pattern = rf"\s*export\s+{re.escape(key_name)}="
    return re.match(pattern, line) is not None


API_KEYS_ENV_FILE = "api_keys.env"
"""Basename of the canonical API-key store inside ``$KISS_HOME``.

One ``export KEY=value`` line per credential, bash syntax, mode 0600.
The settings panel writes it (:func:`save_api_key`), the daemon loads it
at startup (:func:`load_api_keys`), ``./rsorcar`` ships it to deploy
targets, and ``scripts/install-api-keys.sh`` makes ``~/.bashrc`` source
it — so every consumer, local or remote, reads the same file.
"""

SYSTEMD_ENV_FILE = "api_keys.systemd.env"
"""Basename of the legacy systemd ``EnvironmentFile=`` mirror.

Older ``./rsorcar`` deploys generated this second copy of the keys and
pointed the ``kiss-web`` unit's ``EnvironmentFile=`` at it.  Keys now
live in exactly one file — the canonical store, which the daemon parses
itself — so :func:`_edit_api_keys_env_file` *deletes* a mirror it finds:
the old unit's ``EnvironmentFile=-`` (dash: ignore if missing) tolerates
the removal, and a key deleted in the settings panel can never be
re-injected by systemd from a stale copy across a service restart.
"""


def api_keys_env_path() -> Path:
    """Return the path of the canonical API-key store.

    Resolved through :func:`_config_dir` so tests (and ``$KISS_HOME``
    overrides) redirect it together with ``config.json``.
    """
    return _config_dir() / API_KEYS_ENV_FILE


_ENV_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$",
)

_SHELL_ONLY_ENV_VAR_RE = re.compile(
    r"^(PATH|MANPATH|LD_LIBRARY_PATH|DYLD_LIBRARY_PATH|PS1|PROMPT"
    r"|LANG|LC_[A-Z]+|HOME|SHELL|TERM|EDITOR)$",
)
"""Variables never imported into the daemon environment.

``./rsorcar`` distils ``api_keys.env`` from a shell RC file, which may
legitimately contain lines like ``export PATH=/opt/bin:$PATH`` — shell
configuration, not credentials.  The same denylist previously guarded
the systemd mirror; it now guards :func:`load_api_keys` directly.
"""


def _env_line_sets_key(line: str, key_name: str) -> bool:
    """Return True if the env-file *line* assigns the variable *key_name*.

    Matches the canonical ``export KEY=…`` lines this module writes plus
    the bare ``KEY=…`` form, with any horizontal whitespace.  The
    trailing ``=`` anchors the match so ``KEY`` never matches
    ``KEY_EXTRA``.
    """
    return re.match(rf"\s*(?:export\s+)?{re.escape(key_name)}=", line) is not None


def _shell_would_expand(raw: str) -> bool:
    """Return True if a POSIX shell would expand ``$``/backtick in *raw*.

    Walks the string tracking quote state: ``$`` and `` ` `` expand only
    outside single quotes and only when not backslash-escaped.  This is
    what decides whether a stored value can be read literally — a
    ``$`` inside single quotes (including the ``'a'"'"'b'`` concatenated
    form ``shlex.quote`` emits for values holding an apostrophe) is a
    literal to the shell too, while an unquoted ``$PATH`` from an RC
    distillation was *meant* to expand and must not be imported verbatim.
    """
    state = ""
    i = 0
    while i < len(raw):
        c = raw[i]
        if state == "'":
            if c == "'":
                state = ""
        elif state == '"':
            if c == "\\":
                i += 1
            elif c == '"':
                state = ""
            elif c in "$`":
                return True
        else:
            if c == "\\":
                i += 1
            elif c in "'\"":
                state = c
            elif c in "$`":
                return True
        i += 1
    return False


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    """Parse one ``api_keys.env`` line into ``(name, value)``, or ``None``.

    Returns ``None`` for blank lines, comments, anything that is not a
    single ``[export] NAME=value`` assignment, shell-only variables
    (see :data:`_SHELL_ONLY_ENV_VAR_RE`), and values that would need
    shell expansion (see :func:`_shell_would_expand`) — those came from
    an RC distillation where the shell was *meant* to expand them, and
    importing them verbatim would corrupt the environment.

    Quoting is undone with ``shlex`` so a value stored as ``'sk-abc'``
    or ``"sk-abc"`` loads as ``sk-abc``.  When the remainder splits
    into several words — ``KEY=abc # note`` or ``KEY=two words`` — the
    first word is the value, exactly what a shell sourcing the same
    line assigns; a remainder ``shlex`` cannot parse at all (unbalanced
    quote) is kept as the raw text.
    """
    m = _ENV_ASSIGNMENT_RE.match(line)
    if m is None:
        return None
    name, raw = m.group(1), m.group(2).strip()
    if _SHELL_ONLY_ENV_VAR_RE.match(name):
        return None
    if _shell_would_expand(raw):
        return None
    try:
        parts = shlex.split(raw)
    except ValueError:
        return name, raw
    if not parts:
        return name, ""
    return name, parts[0]


def _remove_systemd_mirror() -> None:
    """Delete the legacy systemd mirror so keys live in exactly one file.

    See :data:`SYSTEMD_ENV_FILE` for why removal is safe on machines
    whose ``kiss-web`` unit still names the mirror.
    """
    mirror = _config_dir() / SYSTEMD_ENV_FILE
    try:
        mirror.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove %s", mirror, exc_info=True)


def _edit_api_keys_env_file(mutations: dict[str, str | None]) -> None:
    """Apply *mutations* to the canonical key store atomically.

    Each entry maps a variable name to its new value, or to ``None`` to
    delete every line assigning it.  Existing assignment lines are
    replaced in place; new keys are appended as ``export NAME=value``
    with the value ``shlex.quote``-d.  Unrelated lines — including ones
    this module would not import, such as RC-distilled ``PATH``
    entries — pass through byte-for-byte.

    Runs under :data:`_config_lock` plus an ``fcntl`` flock on a sidecar
    ``.api_keys.env.kiss.lock`` in the same directory, so two savers (two
    daemon threads, or two processes sharing one ``$KISS_HOME``) cannot
    both read the same snapshot and silently drop each other's key.  The
    sidecar is flocked rather than the store itself because the store is
    atomically ``os.replace``-d: a lock on the old inode would not
    exclude a writer that opens the new one.  A legacy systemd mirror is
    deleted inside the same critical section (see
    :func:`_remove_systemd_mirror`).
    """
    env_path = api_keys_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = env_path.with_name("." + env_path.name + ".kiss.lock")
    with (
        _config_lock,
        open(lock_path, "w", encoding="utf-8") as lock_file,
    ):
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            lines: list[str] = []
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines(
                    keepends=True,
                )
            replaced: set[str] = set()
            new_lines: list[str] = []
            for line in lines:
                hit = next(
                    (k for k in mutations if _env_line_sets_key(line, k)),
                    None,
                )
                if hit is None:
                    new_lines.append(line)
                    continue
                value = mutations[hit]
                if value is not None and hit not in replaced:
                    new_lines.append(f"export {hit}={shlex.quote(value)}\n")
                replaced.add(hit)
            for name, value in mutations.items():
                if value is not None and name not in replaced:
                    if new_lines and not new_lines[-1].endswith("\n"):
                        new_lines[-1] += "\n"
                    new_lines.append(f"export {name}={shlex.quote(value)}\n")
            _atomic_write_text_secure(env_path, "".join(new_lines))
            _remove_systemd_mirror()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def save_api_key(key_name: str, key_value: str) -> None:
    """Persist an API key in the canonical store — and nowhere else.

    The canonical ``$KISS_HOME/api_keys.env`` gets an ``export
    KEY=value`` line (replaced in place if the key is already there) —
    that is what :func:`load_api_keys` reads at every daemon start, on
    local and remote installs alike.  The key is stored in that one
    file only: any line assigning it in *any* supported shell RC
    (``~/.bashrc``, ``~/.zshrc``, fish's ``config.fish`` — where
    previous releases stored keys) is *removed*, and for a bash/zsh
    user a one-line hook that sources the canonical store is installed
    in their RC instead — the same block
    ``scripts/install-api-keys.sh`` writes on deploy targets — so
    interactive terminals and ``uv run`` sessions keep seeing the keys
    without holding a second copy.  fish cannot source a bash-syntax
    file, so fish gets no hook; tasks and tools spawned by the daemon
    still inherit the keys from its environment.

    An **empty** ``key_value`` means *delete*: the assignment is removed
    from the canonical store and from the shell RC, any legacy systemd
    mirror file is deleted, the variable is dropped from ``os.environ``
    and the config singleton is refreshed — so clearing an API-key field
    in the settings panel genuinely unsets the key everywhere, including
    across the next daemon or service restart.

    Values containing newlines are refused.  ``shlex.quote`` would
    write a *valid multiline* quoted assignment, but every edit here is
    line-oriented: a later replace or delete would rewrite only the
    first physical line and leave the file syntactically broken (an
    unterminated quote).  Real API keys are single-line, so the safe
    contract is to reject the value outright.

    Also sets the key in the current process environment and refreshes
    the :data:`kiss.core.config.DEFAULT_CONFIG` singleton so subsequent
    model queries see the new key immediately.

    The RC read-modify-replace runs under :data:`_config_lock` plus an
    ``fcntl`` flock on a sidecar ``<rc>.kiss.lock`` beside the RC.  The
    sidecar lives beside the RC rather than in ``$KISS_HOME`` because
    the RC is selected from ``$HOME``: two daemons sharing one HOME but
    running with different ``KISS_HOME`` values edit the *same* RC
    file, so a KISS_HOME-based lock would give them two different locks
    and no exclusion.  The runtime mutation (``os.environ`` +
    :func:`_refresh_config`) stays inside ``_config_lock`` so a
    concurrent saver of the same key cannot leave the files and the
    environment disagreeing.

    Args:
        key_name: Environment variable name (e.g. ``"GEMINI_API_KEY"``).
            Must be a valid POSIX identifier — anything else (the name
            arrives from an untrusted client payload) is refused: the
            name is interpolated into the stored line verbatim, so a
            newline or shell metacharacter in it would write arbitrary
            commands into a sourced file, and ``os.environ`` raises
            ``ValueError`` on names containing ``=``.
        key_value: The API key string; empty deletes the key.
    """
    if not _ENV_VAR_NAME_RE.fullmatch(key_name):
        logger.warning(
            "Refusing to save API key with invalid name %r", key_name,
        )
        return
    if "\n" in key_value or "\r" in key_value:
        logger.warning(
            "Refusing to save API key %s with an embedded newline", key_name,
        )
        return
    if "\x00" in key_value:
        # ``shlex.quote`` and the file writer would happily persist a NUL,
        # but ``os.environ[...] = value`` raises on it — both here and at
        # every later daemon start, permanently breaking startup from a
        # poisoned store.  The value arrives from an untrusted client
        # payload, so it is refused before anything is written.
        logger.warning(
            "Refusing to save API key %s with an embedded NUL", key_name,
        )
        return
    user_shell = _get_user_shell()

    with _config_lock:
        _edit_api_keys_env_file({key_name: key_value or None})
        # The key must disappear from EVERY RC a previous release may
        # have written it to, not only the current $SHELL's: a copy left
        # in another shell's RC would be re-imported by the legacy-key
        # migration (or re-shipped by a deploy) after the user deleted
        # the key — and users do switch default shells between releases.
        for shell in ("bash", "zsh", "fish"):
            rc = _shell_rc_path(shell)
            install_hook = bool(key_value) and shell == user_shell
            if not rc.exists() and not install_hook:
                continue
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc_lock = rc.with_name(rc.name + ".kiss.lock")
            with open(rc_lock, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    _update_rc_for_key(
                        rc, shell, key_name, install_hook=install_hook,
                    )
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        if key_value:
            os.environ[key_name] = key_value
        else:
            os.environ.pop(key_name, None)
        _refresh_config()


RC_HOOK_BEGIN = "# >>> sorcar-cloud API keys >>>"
RC_HOOK_END = "# <<< sorcar-cloud API keys <<<"
_RC_HOOK_LINE = '[ -f "$HOME/.kiss/api_keys.env" ] && . "$HOME/.kiss/api_keys.env"'
"""The RC block that sources the canonical key store.

Markers and hook line are byte-identical to what
``scripts/install-api-keys.sh`` writes on ``./rsorcar`` deploy targets,
so a machine that already has the deploy-installed block is recognized
and never gets a second one.
"""


def _update_rc_for_key(
    rc: Path, shell: str, key_name: str, install_hook: bool,
) -> None:
    """Scrub *key_name*'s RC assignment and (bash/zsh) ensure the hook.

    Keys live only in the canonical store, so any ``export KEY=…``
    (``set -gx KEY …`` for fish) line a previous release wrote into the
    RC is removed — on save as well as on delete, otherwise a stale RC
    line sourced *after* the hook would shadow the canonical value in
    interactive shells.  When *install_hook* is true and the shell can
    source a bash-syntax file (bash/zsh), the :data:`RC_HOOK_BEGIN`
    block is appended once so terminals read the store directly.  When
    the canonical store lives somewhere other than ``~/.kiss`` (a
    ``$KISS_HOME`` override), the ``$HOME``-based hook would read the
    wrong file, so it is skipped rather than written wrong.

    Callers hold the RC locks; this only performs the line-oriented
    edit and the atomic 0600 replace.  An RC that needs no change is
    left untouched (and an absent RC uncreated).  An RC that is a
    symlink into a dotfiles repository is edited where it really lives —
    atomically replacing the link path itself would disconnect the file
    somebody maintains (and leave the supposedly deleted key inside it),
    exactly what ``scripts/install-api-keys.sh`` guards against too.
    """
    if rc.is_symlink():
        try:
            target = rc.resolve(strict=True)
        except OSError:
            return
        if not target.is_file():
            return
        rc = target
    lines: list[str] = []
    if rc.exists():
        lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [
        line for line in lines if not _rc_line_sets_key(line, shell, key_name)
    ]
    changed = len(kept) != len(lines)
    if (
        install_hook
        and shell != "fish"
        and api_keys_env_path() == Path.home() / ".kiss" / API_KEYS_ENV_FILE
        and not any(line.rstrip("\n") == RC_HOOK_BEGIN for line in kept)
    ):
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += "\n"
        kept.append(RC_HOOK_BEGIN + "\n")
        kept.append(_RC_HOOK_LINE + "\n")
        kept.append(RC_HOOK_END + "\n")
        changed = True
    if changed:
        _atomic_write_text_secure(rc, "".join(kept))


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


def load_api_keys() -> None:
    """Load API keys from the canonical ``$KISS_HOME/api_keys.env``.

    The one key-loading mechanism shared by local and remote installs:
    the daemon calls this once at startup, and every assignment the
    file holds (API keys, channel tokens — anything ``./rsorcar``
    shipped or :func:`save_api_key` wrote) is parsed **in Python**, with
    no shell involved, and imported into ``os.environ``.  The result is
    a pure function of the file: no interactivity guard, RC syntax
    error, slow ``nvm`` init, or missing shell binary can change what
    gets loaded, which is why the remote systemd unit no longer needs an
    ``EnvironmentFile=`` and the local daemon no longer re-sources the
    user's RC.

    Before parsing, :func:`_migrate_legacy_rc_keys` imports keys that
    only exist as ``export`` lines in a shell RC — the store used
    before this file became canonical — so an install updated from the
    RC-based scheme keeps its keys without any manual step.  The stale
    ``api_keys.systemd.env`` mirror of an old deploy is retired here
    too: a code-only upgrade (``git pull`` + service restart, no fresh
    deploy and no settings change) must also converge on one key file.

    The snapshot (reading the store), the ``os.environ`` import, and
    ``_refresh_config()`` all run under :data:`_config_lock` — the same
    lock :func:`save_api_key` holds for its file → environment → config
    critical section.  Without it, a save completing between this
    function's file read and its environment import was silently
    reverted: the store kept the new value while ``os.environ`` and
    ``DEFAULT_CONFIG`` were overwritten with the stale snapshot (and a
    just-deleted key was resurrected).  The lock must span the snapshot
    itself, not just the import: a snapshot taken before the save but
    applied after it is exactly the lost update.
    """
    _migrate_legacy_rc_keys()
    _remove_systemd_mirror()
    with _config_lock:
        env_path = api_keys_env_path()
        try:
            text = env_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        except OSError:
            logger.warning("Failed to read %s", env_path, exc_info=True)
            text = ""
        for line in text.splitlines():
            parsed = _parse_env_assignment(line)
            if parsed is None:
                continue
            try:
                os.environ[parsed[0]] = parsed[1]
            except ValueError:
                # A hand-edited (or historically poisoned) line whose value
                # embeds a NUL: os.environ refuses it.  One junk line must
                # not abort daemon startup and drop every following key.
                logger.warning("Skipping unusable %s line for %s",
                               env_path.name, parsed[0])
        _refresh_config()


_MIGRATION_TIMEOUT_S = 5.0
"""How long the migration's RC-sourcing shell may run before it is killed."""

_FISH_ASSIGNMENT_RE = re.compile(
    r"^\s*set\s+-gx\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$",
)


def _scan_rc_text_for_keys(text: str, shell: str, wanted: set[str]) -> dict[str, str]:
    """Textually extract *wanted* API-key assignments from RC *text*.

    Reads the exact line formats previous releases wrote (``export
    KEY=value`` for POSIX shells, ``set -gx KEY value`` for fish)
    without running a shell, so an assignment sitting *after* a stock
    Debian/Ubuntu interactivity guard (``case $- in *i*) ;; *) return``)
    is still found — sourcing the RC non-interactively returns before
    reaching it.  Later assignments win, as they do in a shell.
    Expansion-dependent, multi-word, or NUL-carrying values are skipped:
    the canonical store cannot represent them faithfully.
    """
    found: dict[str, str] = {}
    pattern = _FISH_ASSIGNMENT_RE if shell == "fish" else _ENV_ASSIGNMENT_RE
    for line in text.splitlines():
        m = pattern.match(line)
        if m is None:
            continue
        name, raw = m.group(1), m.group(2).strip()
        if name not in wanted or _shell_would_expand(raw):
            continue
        try:
            parts = shlex.split(raw)
        except ValueError:
            continue
        if len(parts) == 1 and parts[0] and "\x00" not in parts[0]:
            found[name] = parts[0]
    return found


def _source_rc_for_keys(wanted: set[str]) -> dict[str, str]:
    """Source the user's shell RC once and return the *wanted* keys it sets.

    The shell runs with a **minimal clean environment** rather than the
    daemon's: only values the RC itself defines may migrate into the
    persistent store — a key injected into the daemon ad hoc (``KEY=x
    kiss-web``) must stay ephemeral.  The subprocess gets its own
    session so that on timeout the whole process group is killed: a
    background child started by the RC would otherwise keep the output
    pipe open and stall daemon startup long past the timeout.  Failures
    (missing shell binary, hanging or broken RC) are logged and yield
    nothing — a user-maintained RC must not block startup.
    """
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    if not rc.exists():
        return {}
    shell_path = _resolve_shell_path(shell)
    if shell_path is None:
        logger.warning(
            "Cannot migrate RC API keys: %s binary not found on PATH "
            "or fallback locations",
            shell,
        )
        return {}
    clean_env = {
        "HOME": str(Path.home()),
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "SHELL": os.environ.get("SHELL", shell_path),
        "TERM": "dumb",
        "USER": os.environ.get("USER", ""),
    }
    rc_q = shlex.quote(str(rc))
    if shell == "fish":
        cmd = f"source {rc_q} 2>/dev/null; env -0 2>/dev/null; or env"
    else:
        cmd = f"source {rc_q} 2>/dev/null; {{ env -0 2>/dev/null || env; }}"
    stdout = ""
    try:
        with subprocess.Popen(
            [shell_path, "-c", cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=clean_env,
            start_new_session=True,
        ) as proc:
            try:
                stdout, _stderr = proc.communicate(timeout=_MIGRATION_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Kill the whole session: proc.kill() alone leaves RC
                # descendants holding the pipes, and the follow-up
                # communicate() would block on them without a timeout.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                logger.warning(
                    "Shell RC took longer than %.0fs to source; "
                    "skipping key migration from it", _MIGRATION_TIMEOUT_S,
                )
                return {}
    except OSError:
        logger.warning("Failed to source shell RC for key migration", exc_info=True)
        return {}
    records = stdout.split("\0") if "\0" in stdout else stdout.splitlines()
    found: dict[str, str] = {}
    for record in records:
        if "=" in record:
            k, _, v = record.partition("=")
            if (
                k in wanted and v
                and "\n" not in v and "\r" not in v and "\x00" not in v
            ):
                found[k] = v
    return found


def _migrate_legacy_rc_keys() -> None:
    """Import RC-resident API keys into the canonical store, additively.

    Earlier releases persisted keys as ``export`` (``set -gx``) lines in
    a shell RC and re-sourced that RC at every daemon start.  For each
    :data:`API_KEY_ENV_VARS` variable *absent* from the canonical file,
    this looks in every supported RC — first textually (which sees past
    a stock ``.bashrc``'s non-interactive ``return`` guard), then by
    sourcing the user's shell RC once for values assembled indirectly —
    and copies what it finds into the file.  Keys already in the file
    are never touched, and when nothing is missing no RC is read at all.

    The whole read-source-write transaction holds :data:`_config_lock`:
    otherwise a concurrent settings-panel *deletion* could scrub the
    store between this function's RC read and its final write, and the
    stale RC observation would resurrect the key the user just deleted.
    """
    with _config_lock:
        env_path = api_keys_env_path()
        present: set[str] = set()
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    m = _ENV_ASSIGNMENT_RE.match(line)
                    if m is not None:
                        present.add(m.group(1))
            except OSError:
                logger.warning("Failed to read %s", env_path, exc_info=True)
                return
        missing = set(API_KEY_ENV_VARS) - present
        if not missing:
            return
        found: dict[str, str] = {}
        for shell in ("bash", "zsh", "fish"):
            rc = _shell_rc_path(shell)
            try:
                text = rc.read_text(encoding="utf-8")
            except OSError:
                continue
            found.update(_scan_rc_text_for_keys(text, shell, missing))
        still_missing = missing - set(found)
        if still_missing:
            found.update(_source_rc_for_keys(still_missing))
        if found:
            mutations: dict[str, str | None] = dict(found)
            _edit_api_keys_env_file(mutations)
