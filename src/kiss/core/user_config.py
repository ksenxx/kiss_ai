# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""User configuration of the daemon and its settings panel.

Persists user preferences to ``$KISS_HOME/config.json`` and manages the
canonical API-key store ``$KISS_HOME/api_keys.env`` — the one file both
local and remote installs load keys from at daemon startup
(:func:`load_api_keys`) and the one file the settings panel writes them
to (:func:`save_api_key`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from kiss.core import config as config_module
from kiss.core.brand import HOME_DIR
from kiss.core.config import DEFAULT_MAX_BUDGET, _env_flag, kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import locked_call
from kiss.core.processes import bash_version, run_process_group
from kiss.core.utils import (
    atomic_write_text,
    read_bytes_waiting_for_writer,
    read_text_waiting_for_writer,
)

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Serializes every writer and reader of config.json, the canonical key
# store, os.environ and DEFAULT_CONFIG within one process.  It also
# serializes access to _api_keys_store_flock(), so a single process never
# flocks the store sidecar through two descriptors at once (which would
# self-deadlock — see that function).  Holders call the ``_locked``
# helpers, which take neither this lock nor the flock again.  It is an
# RLock so that a nested acquisition of THIS lock (``record_recent_work_dir``
# once held it around ``save_config``) can never be the thing that
# deadlocks.  The
# writers a task thread reaches (``save_config`` behind the model
# picker, ``save_api_key`` behind a tool) take it through
# ``locked_call``: a Stop injection landing right after a plain
# ``with``'s ``__enter__`` would leak it for every later config access.
_config_lock = threading.RLock()

_LOCK_TIMEOUT_S = 30.0
"""Longest wait for another process's ``.config.lock``, store or RC lock.

The cross-process ``flock`` blocks in C, so a holder wedged in a
debugger or on a slow filesystem would otherwise stall the writer for
good — and, since the writer also holds ``_config_lock``, every other
config writer and the key-store importers with it.  A legitimate
holder is done in well under a second (the legacy-key migration's RC
sourcing, the slowest, in five),
so a wait past this bound is a wedged holder: the ``TimeoutError`` is
the writer's error.
"""

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
    """Resolve ``CONFIG_DIR``/``CONFIG_PATH`` on each read: ``$KISS_HOME`` may change."""
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
    # The remote-access password the first install.sh run set
    # (kiss.core.remote_password; later runs keep it), editable in the
    # Settings panel.  Empty: no password, so the daemon admits
    # loopback clients only.
    "remote_password": "",
    # Pre-run task classification (kiss.agents.sorcar.task_classifier):
    # one model call (a decisions model, or a single-call KISSAgent on
    # the run's model) decides is_simple (lite system prompt).
    "classify_tasks": True,
    # The one switch for OpenRouter's ~typesafe/jev-latest decisions
    # model (the settings panel's "Use Jev" checkbox; gate:
    # kiss.agents.sorcar.decide_tool.decisions_tool_available).  True
    # (default): the classifier asks Jev first (one typed question,
    # ~$0.00003, ~0.2 s) and falls back to the LLM classifier when no
    # OPENROUTER_API_KEY is configured or the call fails, and every
    # agent gets the ``decide`` tool.  False: the LLM classifier always
    # answers (one non-agentic call on the run's model, skipped for
    # cc/* and codex/* models) and no task calls Jev.
    "classify_with_decisions": True,
    # Persistent agent memory (kiss.core.memoryfield), on by default:
    # SorcarAgent runs get the memory_* tools and the MEMORY_PROTOCOL
    # system-prompt block.  Pages live in ``memory_dir`` when set, else
    # ``$KISS_HOME/memories`` (~/.kiss/memories).  The KISS_USE_MEMORY
    # environment variable overrides the stored toggle for one process.
    # Without an OPENAI_API_KEY the index falls back to the offline
    # hashed embedder (kiss.core.memoryfield.index.default_embedder).
    "use_memory": True,
    "memory_dir": "",
    # Context compaction (kiss.core.context_compaction), on by default:
    # every agent of the daemon replaces old, large tool outputs in its
    # conversation with short stubs once the context grows large.  The
    # settings panel's "Compaction" checkbox toggles it; mirrored into
    # ``DEFAULT_CONFIG.tool_output_compaction`` by :func:`apply_config_to_env`,
    # which every agent reads at each step, so a change applies to running
    # tasks too.  Off, no run compacts, whatever its own ``compaction`` run
    # option (a ``run()`` keyword, a SEA setting, a ``run_agent`` option)
    # says; on, a run's option turns it off for that run alone.  The same
    # KISS_TOOL_OUTPUT_COMPACTION flag as ``Config`` seeds the default, so
    # an environment opt-out holds until the panel's switch is saved.
    "compaction": _env_flag("KISS_TOOL_OUTPUT_COMPACTION", True),
    # Chrome bridge (kiss.agents.sorcar.chrome_bridge): the daemon starts
    # the loopback server for the "KISS Sorcar Bridge" extension at boot,
    # so the user's everyday Chrome is reachable (``use_my_chrome``) from
    # the first call.  Off: the server starts on the first call instead
    # and that call may have to wait for the extension to reconnect.
    "chrome_bridge": False,
    # Claude Code (``cc/`` models) bills the Claude subscription the
    # ``claude`` CLI is logged in to (or CLAUDE_CODE_OAUTH_TOKEN) rather
    # than an API key: ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN are kept
    # out of the CLI's environment.  Mirrored into
    # ``DEFAULT_CONFIG.claude_code_use_subscription`` by
    # :func:`apply_config_to_env`.  The same KISS_CLAUDE_CODE_USE_SUBSCRIPTION
    # flag as ``Config`` seeds the default, so an environment opt-out holds
    # until the panel's switch is saved.
    "claude_code_use_subscription": _env_flag("KISS_CLAUDE_CODE_USE_SUBSCRIPTION", True),
    # Codex (``codex/`` models) likewise bills the ChatGPT subscription
    # the ``codex`` CLI is logged in to (or CODEX_ACCESS_TOKEN):
    # OPENAI_API_KEY / CODEX_API_KEY are kept out of the CLI's environment.
    "codex_use_subscription": _env_flag("KISS_CODEX_USE_SUBSCRIPTION", True),
    "work_dir": "",
    "last_model": "",
    # The chat header's machine drop-down pick: the registered machine a
    # new chat's tasks run on ("" = this machine), one global setting
    # for every surface (``setMachine`` / ``machineChanged``).
    "last_machine": "",
}

RETIRED_KEYS: frozenset[str] = frozenset(
    {"demo_mode", "is_parallel", "remote_password_hash"},
)
"""Settings that used to exist and must be forgotten on sight.

``config.json`` is written by every previous release, so dropping a key
from :data:`DEFAULTS` is not enough: :func:`load_config` overlays whatever
the file holds, :func:`sanitize_config` passes unknown keys through (that
is how keys outside :data:`DEFAULTS` such as ``email`` and ``tunnel_token``
survive), and :func:`save_config` rewrites the file from its own previous
contents.  A retired key would therefore be read back, echoed to every
client in ``configData``, and re-persisted forever.  Listing it here
purges it from both the value read and the file written.

``is_parallel`` is retired rather than merely removed for that reason:
it never had a single reader — whether a run may spawn parallel
sub-agents comes from the run command's ``isParallel`` flag — yet it
was written to every user's ``config.json`` and broadcast in every
``configData``, which invites a future reader to wire it up and
silently disagree with the real source of truth.
"""

API_KEY_ENV_VARS: frozenset[str] = frozenset(
    {
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_WORKSPACE_ID",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
        "MOONSHOT_API_KEY",
        # Amazon Bedrock credentials (``bedrock/`` Claude models); AWS_REGION
        # is not a secret but is stored and edited alongside them.
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        # Claude subscription token from ``claude setup-token`` (``cc/``
        # models on a machine without a ``claude login``).
        "CLAUDE_CODE_OAUTH_TOKEN",
        # Codex access token (``codex/`` models on a machine without a
        # ``codex login``; ChatGPT Business/Enterprise workspaces).
        "CODEX_ACCESS_TOKEN",
    }
)


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
    the daemon-global working directory.

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
                    "Ignoring boolean config value %s=%r",
                    key,
                    value,
                )
                result[key] = default
                continue
            # An int too large for a float (json.loads turns a 400-digit
            # literal into one) raises OverflowError here, as
            # ``math.isfinite`` would on it below.
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                logger.debug(
                    "Ignoring non-numeric config value %s=%r",
                    key,
                    value,
                )
                result[key] = default
                continue
            if not math.isfinite(number):
                logger.debug(
                    "Ignoring non-finite config value %s=%r",
                    key,
                    value,
                )
                result[key] = default
            else:
                result[key] = value if isinstance(value, int | float) else number
        elif not isinstance(value, str):
            logger.debug(
                "Ignoring non-string config value %s=%r",
                key,
                value,
            )
            result[key] = default
    return result


def _read_stored_config(cfg_path: Path, strict: bool = False) -> dict[str, Any]:
    """Return the JSON object stored in *cfg_path*, or ``{}`` if unusable.

    The single reader behind :func:`load_config` and :func:`save_config`,
    so both agree on what "unreadable" means: a missing file, an
    ``OSError``, or any ``ValueError`` from decoding — invalid JSON
    (``json.JSONDecodeError``) and invalid UTF-8 (``UnicodeDecodeError``)
    alike.  The file is user-editable, so both kinds of junk happen; a
    top-level value that is not a JSON object is ignored too.

    With *strict*, an ``OSError`` other than a missing file (a
    ``PermissionError`` on a chmod-000 file, an I/O error) propagates:
    a writer that read ``{}`` for a file it could not read would
    ``os.replace`` a file holding only its own keys over the user's
    settings.  A missing file is still ``{}``; so is undecodable JSON,
    which the writer replaces by contract.  Missing is judged by the
    read's ``FileNotFoundError`` alone, never by an existence precheck:
    ``Path.exists()`` answers ``False`` for a link into a directory the
    process may not traverse, and that file is unreadable, not absent.
    """
    try:
        # Waits out a concurrent ``save_config`` replace on Windows, so
        # the reader does not fall back to an empty config mid-swap.
        stored = json.loads(read_bytes_waiting_for_writer(cfg_path).decode("utf-8"))
    except (ValueError, OSError) as exc:
        if strict and isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
            raise
        logger.debug("Failed to read config %s", cfg_path, exc_info=True)
        return {}
    return stored if isinstance(stored, dict) else {}


def load_config() -> dict[str, Any]:
    """Load configuration from ``$KISS_HOME/config.json``.

    Returns a dict with all keys from :data:`DEFAULTS`, falling back to
    default values for any missing keys.  Values of DEFAULTS keys are
    type-sanitized via :func:`sanitize_config` so a hand-edited junk
    value cannot break downstream consumers.
    """
    result = dict(DEFAULTS)
    result.update(_read_stored_config(_config_path()))
    return sanitize_config(result)


def save_config(data: dict[str, Any]) -> None:
    """Save configuration to ``$KISS_HOME/config.json``.

    Merges *data* into the existing file contents, so keys already
    present but absent from *data* are preserved.  Keys outside
    :data:`DEFAULTS` (``tunnel_token``, ``skill_permissions``,
    ``mcp_permissions``, ``email``) are written like any other: they are
    read at runtime, so accepting them and then dropping them would be
    silent data loss that only surfaces after a daemon restart.  API
    keys are never written to the config file — they belong in the
    canonical key store (see :func:`save_api_key`) — and
    :data:`RETIRED_KEYS` are purged on every write.

    The write is **atomic** — content is staged in a sibling temp file
    and then ``os.replace``-d into position so that concurrent readers
    never observe an empty or partially-written ``config.json``.

    Args:
        data: Configuration dict.

    Raises:
        TimeoutError: another process held ``.config.lock`` for longer
            than :data:`_LOCK_TIMEOUT_S`; nothing was written.
        OSError: ``config.json`` exists but could not be read (a
            ``PermissionError``, say); nothing was written, since the
            merge would have dropped every setting it holds.
    """
    locked_call(_config_lock, _save_config_locked, sanitize_config(data), False)


def setdefault_config(data: dict[str, Any]) -> dict[str, Any]:
    """Store the entries of *data* whose key is unset and return the stored config.

    Like :func:`save_config`, but an entry whose key already has a
    truthy value in ``config.json`` is left alone.  "Has a value" is
    judged on the stored value as :func:`load_config` reads it: a
    hand-edited junk-typed value that sanitizes to a falsy default (a
    number where a string belongs) counts as unset, so the entry stored
    here is always the one later reads return.  The read, the decision
    and the write happen in one ``.config.lock`` section, so two
    processes that both find a key unset (two ``install.sh`` runs
    storing a first ``remote_password``) cannot both write their own
    value: the second sees the first's and returns it.

    Args:
        data: Entries to store unless already set.

    Returns:
        The stored configuration after the write (raw file contents,
        without :data:`DEFAULTS`).

    Raises:
        TimeoutError: another process held ``.config.lock`` for longer
            than :data:`_LOCK_TIMEOUT_S`; nothing was written.
        OSError: ``config.json`` exists but could not be read (a
            ``PermissionError``, say); nothing was written, since the
            merge would have dropped every setting it holds.
    """
    return locked_call(_config_lock, _save_config_locked, sanitize_config(data), True)


def _save_config_locked(
    data: dict[str, Any], only_unset: bool, recent_work_dir: str | None = None,
) -> dict[str, Any]:
    """Merge *data* into ``config.json`` under its flock; the caller holds ``_config_lock``.

    With *only_unset*, an entry whose key already has a truthy stored
    value (as sanitized, i.e. as :func:`load_config` reads it) is
    skipped.  With *recent_work_dir*, that directory is moved to the
    front of the stored ``recent_work_dirs`` list in the same write;
    the list is read under the flock, so another process's rows cannot
    be replaced by a stale copy.  Returns the stored configuration
    after the write.

    Raises:
        OSError: ``config.json`` exists but could not be read; replacing
            it would silently drop every setting it holds.
    """
    cfg_path = _config_path()
    with exclusive_file_lock(_config_dir() / ".config.lock", timeout=_LOCK_TIMEOUT_S):
        existing = _read_stored_config(cfg_path, strict=True)
        if recent_work_dir is not None:
            same = os.path.normpath(recent_work_dir)
            entries = [
                e for e in _parse_recent_work_dirs(existing)
                if os.path.normpath(e["path"]) != same
            ]
            entries.insert(0, {"path": recent_work_dir, "ts": time.time()})
            data = {**data, "recent_work_dirs": entries[:MAX_RECENT_WORK_DIRS]}
        readable = sanitize_config(existing) if only_unset else {}
        for k, v in data.items():
            if k not in API_KEY_ENV_VARS and not (only_unset and readable.get(k)):
                existing[k] = v
        for k in RETIRED_KEYS:
            existing.pop(k, None)
        # atomic_write_text stages the payload in a sibling temp file
        # through a buffered file object (a bare ``os.write`` may
        # legally write fewer bytes than asked and the truncated file
        # would be published) and ``os.replace``-s it into position.
        # mode=0o600 is FORCED (not just the new-file default): the
        # config stores ``remote_password`` and ``tunnel_token``, so
        # it must never be group/world-readable — and forcing it also
        # repairs a config.json a prior release published as 0644.
        atomic_write_text(cfg_path, json.dumps(existing, indent=2), mode=0o600)
    return existing


# How many recently opened working directories ``config.json`` keeps.
MAX_RECENT_WORK_DIRS = 30


def recent_work_dirs() -> list[dict[str, Any]]:
    """Return the working directories opened so far, most recent first.

    Read from the ``recent_work_dirs`` key of ``config.json`` (written
    by :func:`record_recent_work_dir`).  Each entry is
    ``{"path": str, "ts": float}`` where ``ts`` is the epoch second the
    directory was last adopted as a working directory.  Entries whose
    directory no longer exists, and malformed entries from a hand-edited
    file, are skipped.
    """
    return _parse_recent_work_dirs(_read_stored_config(_config_path()))


def _parse_recent_work_dirs(stored: dict[str, Any]) -> list[dict[str, Any]]:
    """The usable ``recent_work_dirs`` rows of a stored config, most recent first."""
    raw = stored.get("recent_work_dirs")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        path, ts = entry.get("path"), entry.get("ts")
        if not isinstance(path, str) or not isinstance(ts, int | float):
            continue
        # A hand-edited ``1e309`` (inf) or ``10**1000`` (OverflowError on
        # float()) must not poison every configData reply.
        try:
            stamp = float(ts)
        except OverflowError:
            continue
        if isinstance(ts, bool) or not math.isfinite(stamp):
            continue
        if not os.path.isdir(path):
            continue
        result.append({"path": path, "ts": stamp})
    result.sort(key=lambda e: e["ts"], reverse=True)
    return result


def record_recent_work_dir(path: str, extra: dict[str, Any] | None = None) -> None:
    """Move *path* to the front of the recently opened working directories.

    A *path* that is not an existing directory gets no row; *extra*
    is stored all the same (the folder a first task will create, or
    one removed since the caller looked, is still the adopted
    ``work_dir``).  One row per
    directory: an earlier spelling of the same directory (a trailing
    separator, ``os.path.normpath`` equal) is replaced, the way the
    clients tell directories apart.  The list is capped at
    :data:`MAX_RECENT_WORK_DIRS` entries and persisted in
    ``config.json`` like :func:`save_config` does, so every browser
    window shares one history.  The read-merge-write runs under the
    cross-process ``.config.lock``, so two daemons on one ``KISS_HOME``
    cannot drop each other's rows.

    Args:
        path: The directory that was just adopted as a working directory.
        extra: Further config entries to store in the same write (the
            ``work_dir`` that adoption also persists), as
            :func:`save_config` would.

    Raises:
        TimeoutError: another process held ``.config.lock`` for longer
            than :data:`_LOCK_TIMEOUT_S`; nothing was written.
        OSError: ``config.json`` exists but could not be read; nothing
            was written.
    """
    recent = path if os.path.isdir(path) else None
    if recent is not None or extra:
        locked_call(
            _config_lock, _save_config_locked, sanitize_config(extra or {}), False, recent
        )


def _get_user_shell() -> str:
    """Detect the user's default shell from the binary ``$SHELL`` names.

    Returns:
        One of ``'zsh'``, ``'bash'``, or ``'fish'``.
    """
    shell = os.path.basename(os.environ.get("SHELL", ""))
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


def _bash_drops_nul_when_sourcing() -> bool:
    """Return whether the installed Bash drops NUL bytes while sourcing an RC file."""
    return bash_version(_resolve_shell_path("bash")) >= (4, 0)


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


_RC_REDIRECTION_OPERATOR_RE = re.compile(r"&>>?|>>|>\||>&|<<<|<<-|<<|<&|<>|>(?!\()|<(?!\()")
"""A redirection operator where an unquoted, unescaped ``<``, ``>`` or ``&`` stands in a line.

The longest operator first: ``>>`` before ``>``, ``<<<`` and ``<<-``
before ``<<``, so that a plain ``&`` or ``&&`` never matches.  A
``<`` or ``>`` opening a process substitution (``<(…)``, ``>(…)``) is
no operator: it is a character of the word around it, as for the
shell (``KEY=old<(:)`` is one word; ``< <(:)`` is ``<`` and its
target), while ``<<(`` is still ``<<``.
"""

_RC_IO_NUMBER_RE = re.compile(r"[0-9]+")
"""The descriptor number glued to a ``<``/``>`` operator (the ``2`` of ``2>&1``)."""


def _rc_tokens(line: str) -> list[tuple[str, int, int]]:
    """Split RC *line* into its top-level tokens as ``(kind, start, end)`` triples.

    *kind* is ``"word"`` for a shell word, ``"pattern"`` for one ended
    by a ``)`` that closes nothing (a ``case`` pattern: the command of
    its clause follows, ``x) export KEY=v``), ``"op"`` for an unquoted
    ``;``, ``&&``, ``||``, ``|``, ``&`` or zsh's ``&!`` separating two
    commands, and ``"comment"`` for a ``#`` opening a word, which runs
    to the end of the line.  Quotes (``$'…'`` with its escaped ``\\'``), backslash
    escapes and the nesting of ``(…)``, ``$(…)``, ``${…}`` (a ``(``
    inside it is text) and backticks are read as the shell reads them,
    so a ``;`` inside a value or a command substitution separates
    nothing and a word runs on through a quoted or nested stretch.
    A word a ``(`` began ends with its ``)``: a ``#`` after it opens a
    comment (``(cd x)# note``).
    An unquoted redirection operator (:data:`_RC_REDIRECTION_OPERATOR_RE`)
    ends the word before it, as it does for the shell (``KEY=x<<EOF``
    is the word ``KEY=x``, the operator ``<<`` and the delimiter
    ``EOF``), unless that word is its descriptor number (``2>&1``), and
    opens a word of its own that the target glued to it continues
    (``2>/dev/null``, ``&>file``): the ``&`` of ``>&``, ``<&`` and
    ``&>`` is the operator's, never a separator, while the one after an
    escaped ``\\>`` separates.  A process substitution (``<(…)``,
    ``>(…)``) is part of the word it stands in, nested like ``(…)``.
    """
    tokens: list[tuple[str, int, int]] = []
    stack: list[str] = []  # the open quotes and brackets, innermost last
    start = -1  # where the current word began; -1 between words
    i = 0
    while i < len(line):
        c = line[i]
        top = stack[-1] if stack else ""
        operator = _RC_REDIRECTION_OPERATOR_RE.match(line, i) if not stack and c in "<>&" else None
        if operator is not None:
            # The word before it ends there, as for the shell, unless it is the
            # descriptor number of a ``<``/``>`` operator (``2>&1``; not ``7&>f``).
            if start >= 0 and not (c in "<>" and _RC_IO_NUMBER_RE.fullmatch(line, start, i)):
                tokens.append(("word", start, i))
                start = -1
            if start < 0:
                start = i
            i = operator.end()
            continue
        if not stack and (c in " \t;&|" or (c == "#" and start < 0)):
            if start >= 0:
                tokens.append(("word", start, i))
                start = -1
            if c == "#":
                tokens.append(("comment", i, len(line)))
                break
            if c in ";&|":
                end = i + 2 if line[i : i + 2] in ("&&", "||", "&!") else i + 1
                tokens.append(("op", i, end))
                i = end
                continue
            i += 1
            continue
        if start < 0:
            start = i
        if top == "'":
            if c == "'":
                stack.pop()
        elif c == "\\":
            if start == i and i + 1 == len(line):
                start = -1  # alone at the line's end it continues the line: no word
            i += 1  # the escaped character is literal
        elif top in ("`", "A"):  # backticks; ``$'…'``
            if c == ("'" if top == "A" else "`"):
                stack.pop()
        elif line[i : i + 2] == "$$":
            i += 1  # the process id: its second ``$`` opens nothing (``$${``)
        elif top == '"':
            if c == '"':
                stack.pop()
            elif c == "`":
                stack.append(c)
            elif c == "$" and line[i + 1 : i + 2] in ("(", "{"):
                i += 1
                stack.append(line[i])
        elif c in "'\"`" or (c == "(" and top != "{"):
            stack.append(c)
        elif c == "$" and line[i + 1 : i + 2] in ("(", "{", "'"):
            i += 1
            stack.append("A" if line[i] == "'" else line[i])
        elif (c == ")" and top == "(") or (c == "}" and top == "{"):
            stack.pop()
            if not stack and line[start] == "(":
                tokens.append(("word", start, i + 1))
                start = -1
        elif c == ")" and not stack:
            tokens.append(("pattern", start, i + 1))
            start = -1
        i += 1
    if start >= 0:
        tokens.append(("word", start, len(line)))
    return tokens


def _rc_commands(line: str) -> list[str]:
    """Split RC *line* into its commands and the operators between them.

    Returns ``[command, operator, command, …]``: the commands keep their
    blanks (and a trailing comment) and the operators are the unquoted
    ``;``, ``&&``, ``||``, ``|`` and ``&`` of :func:`_rc_tokens`, so
    ``"".join`` gives the line back.  A ``;`` inside a value, a comment
    or a command substitution splits nothing.
    """
    pieces: list[str] = []
    start = 0
    for kind, begin, end in _rc_tokens(line):
        if kind == "op":
            pieces += [line[start:begin], line[begin:end]]
            start = end
    pieces.append(line[start:])
    return pieces


_RC_COMMAND_OPENERS = frozenset(
    {"if", "then", "elif", "else", "while", "until", "do", "{", "!", "begin", "and", "or", "not"}
)
"""Reserved words (POSIX and fish) that may precede a command in one piece of a line.

``then export KEY=x`` is the command ``export`` under ``then``:
:func:`_rc_command` skips them to find the command and the edit keeps
them, with a ``true`` no-op after them when the command itself goes.
"""

_RC_COMMAND_HEADS = frozenset({"for", "foreach", "repeat", "select"})
"""Words that begin a loop zsh takes the body of with no reserved word before it.

In the loop's own words (``for x (a b) command``) or on the next line
(``repeat 2`` and the command below it).
"""

_RC_BLOCK_WORDS: dict[bool, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    False: (
        frozenset({"if", "while", "until", "for", "select", "case", "{", "foreach"}),
        frozenset({"fi", "done", "esac", "}", "end"}),  # zsh: ``foreach … end``
        frozenset({"then", "elif", "else", "do"}),
    ),
    True: (
        frozenset({"if", "while", "for", "function", "begin", "switch", "{"}),
        frozenset({"end", "}"}),
        frozenset({"else", "case"}),
    ),
}
"""The words that open a compound command, close one and go on with one; ``True``: fish's."""

_RC_PRECOMMANDS = frozenset({"builtin", "command", "exec", "noglob", "nocorrect", "time", "-"})
"""Words a shell runs the command after them through: ``builtin export KEY=x`` exports."""

_RC_DECLARATIONS = frozenset(
    {"typeset", "declare", "local", "readonly", "integer", "float", "private", "export"}
)
"""Builtins of bash and zsh that assign the variable a ``NAME=value`` operand names."""

_RC_READERS = frozenset({"read", "mapfile", "readarray", "getopts", "vared", "sysread", "set"})
"""Builtins that assign the variable an operand names (``read KEY``, zsh's ``set -A KEY``)."""

_RC_ASSIGNERS = (
    _RC_DECLARATIONS
    | _RC_READERS
    | _RC_COMMAND_HEADS
    | {"printf", "print", "eval", "source", ".", "let", "unset"}
)
"""Every command word that may write a variable its words name (:func:`_rc_text_sets`)."""

_RC_SURE_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\+?=.*)?")
"""An operand that is one word whatever the shell expands: ``NAME`` or ``NAME=…``.

Any other may expand to none: ``$x``, a glob under ``nullglob``, bash's
``{,}``, zsh's ``x(N)``.
"""

_FISH_SET_READS_RE = re.compile(
    r"(-[A-Za-z]*e[A-Za-z]*|--erase)|-[A-Za-z]*[qS][A-Za-z]*|--query|--show"
)
"""An option that makes fish's ``set`` erase (group 1), query or show a variable: none is set."""

_FISH_OWN_RE = re.compile(r"[\\`{}]|[^\s;&|<>]&[^\s;&|<>]|\r(?!\n?\Z)")
"""What one line holds that fish reads differently from the POSIX reading of :func:`_rc_tokens`.

A backslash (an escape inside single quotes too), a backtick (an
ordinary character), a brace (they nest, around a ``;`` as well), an
``&`` inside a word (``a&b`` is one word) and a carriage return that
does not end the line (it ends a command).
"""


_RC_REDIRECTION_RE = re.compile(r"^(?:[0-9]*|&)(?:>>|>\||>&|<&|<>|<<<|<<-|<<|>|<)(.*)$")
"""A word that is a redirection: an optional descriptor, an operator and the (optional) target.

``2>/dev/null``, ``>>log``, ``2>&1``, ``&>file``, ``<<EOF``; the target
(a here-document's delimiter) is the next word when the operator ends
the word (``2> /dev/null``, ``<< EOF``).
"""

_RC_HERE_DOCUMENT_RE = re.compile(r"^[0-9]*<<-?(?!<)")
"""A word opening a here-document (``<<`` or ``<<-``, not ``<<<``); its lines follow the command."""

_RC_LINE_RE = re.compile(r"[^\n]*\n|[^\n]+")
"""A line of an RC as the shell counts them: ended by a line feed, by nothing else."""

_RC_ASSIGNMENT_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=")
"""The start of a word that assigns a variable: ``NAME=…``, ``NAME+=…``, ``NAME[1]=…``."""


def _rc_operands(words: list[str], start: int) -> list[int]:
    """Indices of the operands among ``words[start:]``: the words that are no redirection."""
    operands: list[int] = []
    i = start
    while i < len(words):
        m = _RC_REDIRECTION_RE.match(words[i])
        if m is None:
            operands.append(i)
        elif not m.group(1):
            i += 1  # the target is the next word
        i += 1
    return operands


class _RcCommand(NamedTuple):
    """One command of an RC line, read once (:func:`_rc_command`) for every use of it."""

    text: str
    words: list[str]  # its words, redirections among them, without a comment
    spans: list[tuple[int, int]]  # where each word stands in the text
    first: int  # its first word after ``case`` patterns, reserved words and a function's head
    last: int  # where its words end: with the piece, for zsh before a ``}``
    depth: int  # the compound commands open after it; -1: it closes one that none opened


def _rc_command(text: str, shell: str, depth: int = 0) -> _RcCommand:
    """Read *text*, one command of a line, with *depth* compound commands open before it.

    The command begins after the last ``case`` pattern of the piece
    (``x)``, or ``(x)`` before a word), the reserved words that open,
    go on with or close a compound command (``} else export KEY=x``),
    those of :data:`_RC_COMMAND_OPENERS` and the head of a function
    (``name()``, ``name ()``, ``function name``).  It ends with the
    piece, for zsh before a ``}``, which closes a group there (``name()
    { export KEY=x }``) where bash reads one more argument.

    A reserved word counts where a command begins; a ``{`` anywhere
    (nothing here tells zsh's ``if [[ … ]] {`` from an argument), as
    does a ``}`` for zsh.  The depth is ``-1`` when the command closes
    or goes on with a compound command none is open of: the reading
    has missed where one opened.
    """
    fish = shell == "fish"
    opens, closes, inner = _RC_BLOCK_WORDS[fish]
    words: list[str] = []
    spans: list[tuple[int, int]] = []
    first = 0
    leading = heading = True  # a reserved word counts here; the command has not begun
    named = target = lost = False  # a function's name, a redirection's target is due; no opener
    previous = ""
    for kind, begin, end in _rc_tokens(text):
        if kind == "comment":
            break
        word = text[begin:end]
        words.append(word)
        spans.append((begin, end))
        redirection = _RC_REDIRECTION_RE.match(word)
        if kind == "pattern":
            first, leading, heading, named, previous = len(words), True, True, False, ""
        elif target or redirection is not None:
            heading = False
            target = not target and redirection is not None and not redirection.group(1)
        elif named or (
            word.endswith("()") and "=" not in word and "$" not in word  # no ``KEY=v$()``
            and len(words) <= first + 2
        ):
            first, leading, heading, named = len(words), True, True, False
        elif leading or word in ("{", "}"):
            if word in opens and (leading or word == "{"):
                depth += not (fish and word == "if" and previous == "else")
            elif word in closes and (leading or shell == "zsh"):
                depth -= 1
            elif not leading:
                continue  # the ``}`` bash and fish take for an argument
            lost = lost or depth < (word in inner)
            named = word == "function" and not fish
            leading = (
                named
                or word in _RC_COMMAND_OPENERS
                or word in closes
                or (word[0] == "(" and word[-1] == ")")
            )
            heading = heading and leading
            if heading:
                first = len(words)
            previous = word
    if first == len(words) and words and words[-1][0] == "(" and words[-1] != "()":
        first -= 1  # a ``(…)`` no word follows is the command, no pattern
    last = len(words)
    if shell == "zsh" and "}" in words[first:]:
        last = words.index("}", first)
    return _RcCommand(text, words, spans, first, last, -1 if lost else depth)


def _rc_bare_key(key_name: str) -> re.Pattern[str]:
    """The pattern of *key_name* where a text names it other than to read the variable.

    Not ``$KEY`` nor ``${KEY…}``.  ``KEY`` never matches ``OTHER_KEY``.
    """
    return re.compile(rf"(?<![A-Za-z0-9_$])(?<!\$\{{){re.escape(key_name)}(?![A-Za-z0-9_])")


def _rc_text_sets(text: str, shell: str, key_name: str) -> bool:
    """Whether *text*, which is not read command by command, may assign *key_name*.

    For what a quote or a bracket holds together over several lines, a
    line fish reads in its own way (:data:`_FISH_OWN_RE`) and the
    lines not read for certain: where their commands begin is not
    known, so the text decides, on the safe side.  It may assign the
    key when it holds ``KEY=``, ``KEY+=`` or ``KEY[1]=`` (not for fish,
    which assigns with ``set``), an expansion that assigns
    (``${KEY:=…}``), or a line naming the key (:func:`_rc_bare_key`)
    after a word of :data:`_RC_ASSIGNERS`: ``read -r KEY``, and ``set
    -gx KEY …`` in a quoted help text too.  A text that only passes
    the name on (``docker run -e KEY``) assigns nothing.
    """
    name = re.escape(key_name)
    assigns = rf"(?<![A-Za-z0-9_]){name}(?:\[[^]]*\])?\+?=|\$\{{{name}:?:?="
    if shell != "fish" and re.search(assigns, text):
        return True
    bare = _rc_bare_key(key_name)
    return any(
        _RC_ASSIGNERS.intersection(re.findall(r"[\w.-]+", before))
        for line in text.split("\n")
        for before in bare.split(line)[:-1]
    )


def _rc_key_use(command: _RcCommand, shell: str, key_name: str) -> tuple[str | None, bool, bool]:
    """Read what *command* does with *key_name*: once, for the edit, its refusal and the scan.

    Returns ``(scrubbed, sets, alters)``:

    - *scrubbed* is the command without the assignments the edit
      removes, ``None`` when it has none: the ``KEY=…`` (``KEY+=…``)
      operands of an ``export`` (``export A=1 KEY=x B=2`` keeps ``A``
      and ``B``), those words of a command made of assignments alone
      (``KEY=x``, ``A=1 KEY=x``: what the migration imports) and fish's
      ``set -gx KEY …`` whole.  A command left without an operand goes
      with its blanks, options and redirections: a bare ``export``
      prints every exported variable.  The ``true`` no-op takes its
      place after a reserved word, so the compound command stays
      valid, and before a here-document (``<< EOF``), whose lines are
      then still read as text.  ``KEY`` never matches ``KEY_EXTRA``.
    - *sets* says the command may assign the key in the sourcing shell
      where the edit does not remove it (``typeset -x KEY=x``, ``read
      KEY``, ``KEY=v export KEY``, ``eval``: each case stands with the
      code below).
    - *alters* says it unsets the variable, decides what a later
      assignment makes of it (``unset KEY``, fish's ``set -e KEY``,
      ``readonly KEY``, ``declare -i KEY``) or assigns it only when
      its command word expands to nothing (``KEY=v "$@"``): the text
      cannot tell what the key holds after it.

    What only passes the name on does none of this: ``env KEY=x
    command``, ``docker run -e KEY``, ``[[ -v KEY ]]``, ``declare -p
    KEY``, a ``printf`` without ``-v``, the text of an ``alias`` or an
    ``echo``, a loop over names, an ``export`` of the name alone, the
    environment of one command (``KEY=x command``), also of a program
    an expansion gives the path of (``KEY=x "$HOME/bin/tool"``).
    """
    text, words, spans, first, last, _ = command
    own = _rc_operands(words[:last], first)  # the command's words but its redirections
    head = own[0] if own else first  # the command word: kept, or replaced by ``true``
    operands: list[int] = []  # the words the edit may remove
    if shell == "fish":
        if words[first : first + 3] == ["set", "-gx", key_name]:
            operands = list(range(first + 1, len(words)))
        hits = operands
    else:
        if words[head : head + 1] == ["export"]:
            operands = [i for i in own[1:] if not words[i].startswith("-")]
        elif all(_RC_ASSIGNMENT_WORD_RE.match(words[i]) for i in own):
            operands = own
        hits = [i for i in operands if words[i].startswith((key_name + "=", key_name + "+="))]
    kept = [words[i] for i in operands if i not in hits]
    scrubbed = None
    if hits and (kept or any(_RC_HERE_DOCUMENT_RE.match(word) for word in words[first:last])):
        scrubbed = text
        for i in reversed(hits):
            if i != head:
                scrubbed = scrubbed[: spans[i - 1][1]] + scrubbed[spans[i][1] :]
        if not kept:
            scrubbed = scrubbed[: spans[head][0]] + "true" + scrubbed[spans[head][1] :]
        elif head in hits:
            scrubbed = scrubbed[: spans[head][0]] + scrubbed[spans[head][1] :].lstrip(" \t")
    elif hits:
        scrubbed = text[: spans[first][0]]
        if first:  # a blank parts ``true`` from an opener glued to what went (``!<<<x KEY=v``)
            scrubbed += "true" if scrubbed[-1:] in (" ", "\t") else " true"
        scrubbed += text[spans[last - 1][1] :]
    left = " ".join(words[i] for i in range(len(words)) if i not in hits)  # the edit leaves it
    rest = [re.sub(r"[\\'\"]", "", words[i]) for i in own if i not in hits]
    n = 0
    while n < len(rest) and (
        _RC_ASSIGNMENT_WORD_RE.match(rest[n])
        or rest[n] in _RC_PRECOMMANDS
        or rest[n].startswith("-")
    ):
        n += 1  # the environment of the command, what runs it, an option of that
    lead, name, args = rest[:n], "".join(rest[n : n + 1]), rest[n + 1 :]
    key = re.escape(key_name)
    # A word assigning the key or an element of it (group 1), also after what may expand
    # to nothing (``"$x"KEY=``, ``{,}KEY=``); never the longer name of ``OTHER_KEY=``.
    assigns = re.compile(rf"(?:[^=]*[$`][^=]*|[^=]*[^=\w])?{key}(\[[^=]*\])?\+?=")
    names = re.compile(rf"(?:-[A-Za-z-]*)?{key}(?![A-Za-z0-9_])")  # also glued: ``-aKEY``
    assigned = any(assigns.match(word) for word in args)
    given = any(names.match(word) for word in args)
    prefixes = [m for m in map(assigns.match, lead) if m is not None]
    literal = re.sub(r"\$\{[^}]*\}|\$\([^)]*\)|`[^`]*`|\$[\w@*]+", "", name)  # but expansions
    plain = name in ("export", "typeset", "declare") and all(  # no attribute but ``-x``
        re.fullmatch("-[gpx]+|--", word) for word in args if word[:1] in ("-", "+")
    )
    # ``KEY=v "$@"`` keeps the assignment when the word is empty: a wrapper, never refused.
    alters = (given and (name == "unset" or (name in _RC_DECLARATIONS and not plain))) or (
        bool(prefixes) and bool(name) and not literal
    )
    if (
        (literal != name and "/" not in literal)  # only the shell knows its name; no path
        or name in ("eval", "source", ".", "let")  # a text run as commands
        or name.startswith("((")  # arithmetic assigns by name
        or (shell == "zsh" and name.startswith("{"))  # zsh opens a group at ``{read``
    ):  # they assign what they name other than to read it (``$x KEY=v``, ``"$x"read -aKEY``)
        sets = given or _rc_bare_key(key_name).search(" ".join([name, *args])) is not None
    elif name in _RC_DECLARATIONS or name == "[[":
        # An operand the edit does not reach: ``typeset -x KEY=x``, ``export "KEY=x"``,
        # ``\export``, ``builtin export``; zsh takes a command after a test (``[[ … ]] export``).
        sets = assigned
    elif name in ("printf", "print"):
        sets = given and any(re.match("-[A-Za-z]*v", word) for word in args)  # ``printf -vKEY``
    elif name in _RC_COMMAND_HEADS:  # the loop's variable; zsh's ``for x (a) export KEY=v``
        sets = assigned or args[:1] == [key_name]
    elif shell == "fish" and name == "set":  # but ``-gx``, an erasure, a query
        target = next((word for word in args if not word.startswith("-")), "")
        reads = [m for m in map(_FISH_SET_READS_RE.fullmatch, args) if m is not None]
        sets = target.partition("[")[0] == key_name and not reads
        alters = target == key_name and any(m.group(1) for m in reads)
    else:
        sets = name in _RC_READERS and given  # ``read KEY``, zsh's ``read 'KEY?prompt'``
    sets = (
        sets
        # any command expanding ``${KEY=…}``, ``${KEY:=…}``, ``${KEY::=…}``
        or (shell != "fish" and re.search(rf"\$\{{{key}:?:?=", left) is not None)
        # what follows the ``}`` zsh ends the command at
        or _rc_text_sets(" ".join(words[last + 1 :]), shell, key_name)
        # an ``export`` beside an operand that may expand to none: alone it prints every
        # exported variable, and zsh reports a glob that fails where it was quiet
        or (
            bool(hits)
            and words[head] == "export"
            and not all(_RC_SURE_WORD_RE.fullmatch(word) for word in kept)
        )
        # an assignment that stays: before no command word or one a glob may expand to
        # none (``KEY=v glob_*``), a declaration (bash: ``KEY=v export KEY``) or ``exec``
        # (zsh), and that of an element (zsh: ``KEY[1]=v true``)
        or (
            bool(prefixes)
            and (
                not name
                or (re.search(r"[*?\[{(]", literal) is not None and name not in ("[", "[["))
                or name in _RC_DECLARATIONS
                or "exec" in lead
                or any(m.group(1) for m in prefixes)
            )
        )
        # a command substitution of fish's, which runs in the shell itself
        or (
            shell == "fish"
            and any("(" in word and _rc_text_sets(word, shell, key_name) for word in words)
        )
    )
    return scrubbed, sets, alters


def _rc_awaits_command(text: str, awaited: bool) -> bool:
    """Whether the shell still awaits a command after the command *text*.

    It does after an ``&&``, ``||`` or ``|`` that ends *text*, after a
    reserved word of :data:`_RC_COMMAND_OPENERS` that does (``then``,
    ``do``, ``{``), after the head of a function (``name()``: zsh takes
    any command for its body) and when its last command begins with a
    word of :data:`_RC_COMMAND_HEADS`: the next line holds the command
    they need.  Text without a command (a blank or comment line)
    leaves *awaited* as it was.
    """
    tokens = [token for token in _rc_tokens(text.rstrip("\n")) if token[0] != "comment"]
    if not tokens:
        return awaited
    kind, begin, end = tokens[-1]
    if kind == "op":
        return text[:end].endswith(("&&", "||", "|", "|&"))
    operators = [i for i, token in enumerate(tokens) if token[0] == "op"]
    _, head, head_end = tokens[operators[-1] + 1 if operators else 0]
    word = text[begin:end]
    return kind == "word" and (
        word in _RC_COMMAND_OPENERS
        or (word.endswith("()") and "=" not in word)
        or text[head:head_end] in _RC_COMMAND_HEADS
    )


def _scrub_rc_line(
    line: str, shell: str, key_name: str, required: bool
) -> tuple[str | None, bool]:
    """Return *line* without its assignments of *key_name*, and whether it may still assign it.

    The line is ``None`` when nothing is left of it; the second result
    says one of its commands sets the key where no edit removes it
    (:func:`_rc_key_use`).  Only the key's own assignment goes
    (``export OPENAI_API_KEY=old; export ANTHROPIC_API_KEY=keep`` keeps
    the second command).  A command emptied between ``;`` (or line
    ends) goes with one ``;``; one emptied beside ``&&``, ``||``, ``|``
    or ``&`` becomes ``true`` so the list stays valid, also before a
    comment.  An export inside ``$(…)`` or ``(…)`` runs in a subshell
    and is not touched.  A line that assigns the key nowhere is
    returned as is; one that keeps a command keeps its indentation.

    *required* says the command before the line awaits one
    (:func:`_rc_awaits_command`): the line's first command, emptied,
    then becomes ``true`` as well.  Removed, it left ``then`` followed
    by ``fi``, a syntax error bash stops reading the RC at.
    """
    body, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
    pieces = _rc_commands(body)
    unsure = False
    for i in range(0, len(pieces), 2):
        scrubbed, sets, _ = _rc_key_use(_rc_command(pieces[i], shell), shell, key_name)
        unsure = unsure or sets
        if scrubbed is None:
            continue
        before = pieces[i - 1] if i else ""
        after = pieces[i + 1] if i + 1 < len(pieces) else ""
        bounded = before in ("", ";") and after in ("", ";") and not (required and i == 0)
        if before == ";" and i > 2 and not pieces[i - 2].strip():
            bounded = False  # zsh's ``cmd |; export …``: the ``;`` ends what the ``|`` awaits
        left = scrubbed.strip()  # of an emptied command: nothing, or the comment ending the line
        if any(kind != "comment" for kind, _, _ in _rc_tokens(scrubbed)) or (left and bounded):
            pieces[i] = scrubbed
        elif bounded:
            pieces[i] = ""
            if before:
                pieces[i - 1] = ""
            elif after:
                pieces[i + 1] = ""
        else:
            spaced = left or after not in ("", ";")
            pieces[i] = f"{' ' if before else ''}true{' ' if spaced else ''}{left}"
    rest = "".join(pieces)
    if rest == body:
        return line, unsure
    if not rest.strip():
        if not required:
            return None, unsure
        rest = "true"  # of a line that began with a ``;``: zsh reads an empty command there
    return body[: len(body) - len(body.lstrip(" \t"))] + rest.lstrip(" \t") + newline, unsure


def _rc_document_word(text: str, i: int) -> tuple[str, bool, int] | None:
    """Read the word that ends a here-document, opened at ``text[i]``, as the shell reads it.

    Returns ``(word, raw, end)``: the word after quote removal, whether
    any of it was quoted (the document's lines are then taken as they
    are) and the index after it; ``None`` for a word bash and zsh read
    differently or not at all (an expansion, ``$"…"``, an escape in
    ``$'…'``, no word).
    """
    while text.startswith((" ", "\t", "\\\n"), i):
        i += 2 if text[i] == "\\" else 1
    word, raw, begin = "", False, i
    while i < len(text) and text[i] not in " \t\n;&|()<>":
        c, after = text[i], text[i + 1 : i + 2]
        if c == "`" or (c == "#" and i == begin) or (c == "$" and after and after in '"({'):
            return None
        if c == "\\":
            word, raw, i = word + after.strip("\n"), raw or after != "\n", i + 2
        elif c == '"':
            i += 1
            while text[i : i + 1] != '"':
                c, after = text[i : i + 1], text[i + 1 : i + 2]
                if c in ("", "$", "`", "\n"):
                    return None
                if c == "\\" and after and after in '$`"\\\n':
                    c, i = after.strip("\n"), i + 1
                word, i = word + c, i + 1
            raw, i = True, i + 1
        elif c == "'" or (c == "$" and after == "'"):
            i += len(c + after) if c == "$" else 1
            close = text.find("'", i)
            if close < 0 or "\n" in text[i:close] or (c == "$" and "\\" in text[i:close]):
                return None
            word, raw, i = word + text[i:close], True, close + 1
        else:
            word, i = word + c, i + 1
    return (word, raw, i) if raw or word else None


_FISH_COMMENT_LINE_RE = re.compile(r"[ \t]*#")
"""A line that is a comment: after a backslash-newline fish goes on with the line after it."""


def _fish_rc_units(text: str) -> tuple[list[list[int]], int, int]:
    """:func:`_rc_units` for fish: the commands of *text*, by the lines each spans.

    A command runs on past the end of its line inside a quote (``'…'``,
    where ``\\'`` and ``\\\\`` are the only escapes, and ``"…"``, where
    ``$(`` opens a command substitution), inside ``(…)``, ``{…}`` and
    the ``[…]`` glued to a word, after a backslash, and after a ``|``,
    ``&&`` or ``||`` that ends the line (blank and comment lines are
    skipped then).  A ``#`` opens a comment where a word could begin.
    fish runs nothing of a file it cannot parse, and this reading does
    not follow fish everywhere: from the first line of a command that
    closes a bracket it did not open, or that a comment line follows
    its backslash-newline in, nothing is known.
    """
    lines = _RC_LINE_RE.findall(text)
    units: list[list[int]] = []
    unit = [0]  # the lines of the command being read
    stack: list[str] = []  # the open quotes and brackets, innermost last
    word = joined = False  # inside a word; a ``|``, ``&&`` or ``||`` awaits its command
    i = n = 0
    while i < len(text):
        c, after = text[i], text[i + 1 : i + 2]
        top = stack[-1] if stack else ""
        i += 1
        if c == "\n":
            n += 1
            word = False
            if stack or joined:
                unit.append(n)
            else:
                units.append(unit)
                unit = [n]
        elif c == "\\" and (
            top not in ("'", '"') or after in (top, "\\") or (top == '"' and after in ("$", "\n"))
        ):
            i += 1  # the escaped character is literal
            if after == "\n":
                n += 1
                unit.append(n)
                if _FISH_COMMENT_LINE_RE.match(text, i):
                    return units, unit[0], len(lines)  # fish reads on after that comment
            elif not top:
                word, joined = True, False
        elif top in ("'", '"'):
            if c == top:
                stack.pop()
            elif top == '"' and c == "$" and after == "(":
                i += 1
                stack.append("(")
                word = False
        elif c in "'\"":
            stack.append(c)
            word, joined = True, False
        elif c == "#" and not word:
            i = text.find("\n", i)  # a comment, to the end of its line
            i = len(text) if i < 0 else i
        elif c in "({" or (c == "[" and word and top != "["):
            stack.append(c)
            word = joined = False
        elif c in ")}":
            if top != "({"[")}".index(c)]:
                return units, unit[0], len(lines)  # it closes nothing: fish rejects the file
            stack.pop()
            word = True
        elif c == "]" and top == "[":
            stack.pop()
        elif c in " \t\r":
            word = False
        elif c in ";&|<>":
            word = False
            joined = not stack and (c == "|" or text[i - 2 : i] == "&&")
        else:
            word, joined = True, False
    if stack:
        return units, unit[0], unit[0]  # the text ends inside this command
    unit = [n for n in unit if n < len(lines)]
    if unit:
        units.append(unit)
    return units, len(lines), len(lines)


def _rc_units(text: str, shell: str) -> tuple[list[list[int]], int, int]:
    """Read *shell*'s RC *text* as the shell does, into its commands.

    Returns ``(units, read, end)``.  *units* lists the commands in order,
    each as the indices of the lines (:data:`_RC_LINE_RE`) it spans: one
    index for a command on a line of its own, several for one that runs
    on (a backslash, a quote or a bracket open at the line's end).  The
    lines of a here-document are text and belong to no unit.  *read* is
    the number of lines read for certain: from the first line of a
    command that bash and zsh read differently, or that the text ends
    in the middle of, nothing is known (``len(lines)`` when all is).
    *end* is the line a new command can be put before: ``len(lines)``,
    or the first line of the last command when the text ends inside it
    or inside its here-document.  From a line the two shells read
    differently nothing better is known than the end of the text.

    The lexical rules are those bash 5 and zsh 5 share: quotes, ``$'…'``,
    backticks, ``${…}``, ``$(…)``, ``((…))``, comments, backslash-newline
    and here-documents (``<<``, ``<<-``; any quoting of the word; several
    on a line; inside ``$(…)``; ended by the end of the text).  Where the
    two differ, each ``return`` below says how.  fish has no
    here-documents and rules of its own (:func:`_fish_rc_units`).
    """
    if shell == "fish":
        return _fish_rc_units(text)
    if shell == "bash" and _bash_drops_nul_when_sourcing():
        text = text.replace("\0", "")  # Bash 4+ drops them; zsh and Bash 3 read them
    lines = _RC_LINE_RE.findall(text)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    units: list[list[int]] = []
    unit = [0]  # the lines of the command being read
    stack: list[str] = []  # the open quotes and brackets, innermost last
    docs: list[tuple[int, str, bool, bool]] = []  # here-documents awaited: depth, word, tabs, raw
    word = args = False  # inside a word; a word of this command was read
    brackets = 0  # the ``[`` this command has not closed
    end = len(lines)
    i = n = 0
    while i < len(text):
        c, after = text[i], text[i + 1 : i + 2]
        top = stack[-1] if stack else ""
        i += 1
        if c == "\n":
            n += 1
            word = args = False
            if docs and (top not in ("", "(", "$") or any(doc[0] != len(stack) for doc in docs)):
                return units, unit[0], len(lines)  # a line feed in a quote: its text starts later
            for depth, mark, tabs, raw in docs:
                body = ""
                while n < len(lines):
                    # An unquoted document joins a line ending in a backslash to the next.
                    joins = not raw and (len(body) - len(body.rstrip("\\"))) % 2 == 1
                    if joins and tabs and body == "\\" and lines[n].startswith("\t"):
                        return units, unit[0], len(lines)  # bash strips these tabs, zsh does not
                    body = (body[:-1] if joins else "") + lines[n].removesuffix("\n")
                    body = body.lstrip("\t" if tabs else "")
                    n += 1
                    if body == mark:
                        break
                    if depth and body.startswith(mark):
                        return units, unit[0], len(lines)  # bash ends it at ``word)``, zsh does not
                else:
                    end = unit[0]
            docs, i = [], starts[n]
            if stack:
                unit.append(n)
            else:
                units.append(unit)
                unit, brackets = [n], 0
        elif top == "'":
            if c == "'":
                stack.pop()
        elif c == "\\":
            i += 1
            if after == "\n":
                n += 1
                unit.append(n)
            elif top in ("", "(", "$"):
                word = args = True
        elif top in ("A", "`"):
            if c == ("'" if top == "A" else "`"):
                stack.pop()
        elif c == "$" and after and after in "({'$" and (after != "'" or top != '"'):
            i += 1
            if (after == "'" and top == "{" and '"' in stack) or text[i - 1 : i + 1] == "$'":
                return units, unit[0], len(lines)  # ``"${x:-$'…'}"``, ``$$'…'``: they differ
            if after == "(" and text[i : i + 1] == "(":
                i += 1
                stack.append("M")
            elif after != "$":
                stack.append({"'": "A", "{": "{", "(": "$"}[after])
            word = args = after != "(" or stack[-1] == "M"
        elif c == "`":
            stack.append(c)
        elif c == '"':
            if top == '"':
                stack.pop()
            else:
                stack.append(c)
                word = args = True
        elif top == '"':
            pass
        elif c == "'":
            if top == "{" and '"' in stack:
                return units, unit[0], len(lines)  # bash opens a quote here, zsh does not
            stack.append(c)
            word = args = True
        elif top == "{":
            if c == "{":
                return units, unit[0], len(lines)  # zsh counts it, bash does not
            if c == "}":
                stack.pop()
        elif top in ("M", "m"):
            if c == "(":
                stack.append("m")
            elif c == ")":
                if top == "M" and after != ")":
                    return units, unit[0], len(lines)  # no arithmetic: a ``$(`` and a subshell
                i += top == "M"
                stack.pop()
        elif c == "#" and not word:
            i = text.find("\n", i)  # a comment, to the end of its line
            i = len(text) if i < 0 else i
        elif c == "(":
            literal = shell == "zsh" and (word or (args and after == "#"))  # glob text of a word
            if shell == "zsh" and after == "#" and (text[i - 2 : i - 1] == "=" or not literal):
                return units, unit[0], len(lines)  # a comment or glob flag: zsh's grammar decides
            if after == "(":
                i += 1
            stack.append("M" if after == "(" else "$" if literal else "(")
            word = args = literal
        elif c == ")":
            if top:
                stack.pop()
                if any(doc[0] > len(stack) for doc in docs):
                    return units, unit[0], len(lines)  # zsh ends the document here, bash does not
            word, args = top == "$", bool(top)
        elif c == "<" and after == "<" and text[i + 1 : i + 2] != "<":
            tabs = text[i + 1 : i + 2] == "-"
            opened = _rc_document_word(text, i + 1 + tabs)
            if opened is None or brackets:
                return units, unit[0], len(lines)  # ``a[1 <<x ]=y``: a shift for bash alone
            mark, raw, close = opened
            docs.append((len(stack), mark.lstrip("\t" if tabs else ""), tabs, raw))
            n += text.count("\n", i, close)
            unit.extend(range(unit[-1] + 1, n + 1))
            i, word, args = close, False, True
        elif top and not word and re.match(r"case[ \t\n]", text[i - 1 : i + 4]):
            return units, unit[0], len(lines)  # ``case`` in brackets: a ``)`` that closes nothing
        elif c in " \t;&|<>":
            word, args = False, args and c not in ";&|"
            i += 2 * (c == "<" and after == "<")  # a here-string
        else:
            word = args = True
            brackets = max(brackets + (c == "[") - (c == "]"), 0)
    if stack:
        return units, unit[0], unit[0]  # the text ends inside this command
    unit = [n for n in unit if n < len(lines)]
    if unit:
        units.append(unit)
        if docs:
            end = unit[0]
    return units, len(lines), end


def _rc_one_line(text: str, shell: str) -> bool:
    """Whether *text* is read command by command: one line, none fish reads in its own way."""
    return "\n" not in text and not (shell == "fish" and _FISH_OWN_RE.search(text))


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
itself — so :func:`_edit_api_keys_env_file_locked` *deletes* a mirror it finds:
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


def _expands_in_first_word(c: str, word: str, continued: bool) -> bool:
    """Whether some shell expands the unquoted *c* that follows *word*, the value so far.

    A parenthesis anywhere; a ``~`` that opens the value or follows
    the ``:`` of a POSIX assignment (*continued*).
    """
    if c in "()":
        return True
    return c == "~" and (not word or (continued and word[-1] == ":"))


def _rc_keeps_assignment(rest: str) -> bool:
    """Whether bash keeps an assignment that *rest*, the text after its words, follows.

    Not when a single ``&`` ends the and/or-list the assignment stands
    in (``KEY=v >/dev/null &``, ``KEY=v && true &``) nor when a ``|``
    follows it before any ``&&``, ``||`` or ``;``: bash runs the whole
    list in the background, and a member of a pipeline, in a subshell.
    (zsh puts only the last pipeline of such a list in the background.)
    Nor when its own command goes on with a word that is no redirection
    (``KEY=v >/dev/null command``: that command's environment; ``export
    KEY=v >/dev/null KEY=w``: a later operand).
    """
    pieces = _rc_commands(rest)
    words = [pieces[0][b:e] for kind, b, e in _rc_tokens(pieces[0]) if kind != "comment"]
    if _rc_operands(words, 0):
        return False
    piped = True  # no ``&&`` or ``||`` yet: a ``|`` makes the assignment a pipeline member
    for operator in pieces[1::2]:
        if operator == ";":
            break
        if operator == "&" or (operator == "|" and piped):
            return False
        piped = False
    return True


def _shell_words(raw: str, continued: bool) -> list[str] | None:
    """Split *raw*, the value of an assignment, into the words a POSIX shell reads.

    Quotes and backslash escapes are undone as the shell undoes them: in
    double quotes a backslash escapes only ``$``, a backtick, ``"`` and
    itself (``"sk-\\$x"`` is ``sk-$x``, ``"a\\nb"`` keeps its backslash);
    unquoted whitespace ends a word, an unquoted ``;``, ``&&`` or ``||``
    ends the assignment (what follows is another command: ``abc; x``
    is ``abc``), so does an unquoted redirection (``abc>/dev/null`` and
    ``abc 2>/dev/null`` are ``abc``: the shell redirects the command,
    and the descriptor number is no word) and a ``#`` that starts a
    word ends the line (``abc # note`` is ``abc``, ``abc#def`` is
    itself) -- but
    not the first word when *continued* says *raw* continues a word
    already begun, the ``NAME=`` of a POSIX assignment: ``KEY=#literal``
    assigns ``#literal``, as the shell does, whereas fish's ``set -gx
    KEY #literal`` opens a new word and that is a comment.  This
    is what decides whether a stored value can be read literally:
    ``None`` means the shell would expand it -- ``$`` or a backtick
    outside single quotes and not backslash-escaped, or a process
    substitution (``<(…)``, ``>(…)``) -- so an unquoted
    ``$PATH`` from an RC distillation, which was *meant* to expand, is
    never imported verbatim, while a ``$`` inside single quotes
    (including the ``'a'"'"'b'`` concatenated form ``shlex.quote`` emits
    for values holding an apostrophe) is a literal to the shell too.
    More spellings of the first word are not what some shell assigns,
    ``None`` as well: an unquoted ``~`` that opens it or, in a POSIX assignment,
    follows a ``:`` (``~/keys/token`` is a path in the user's home,
    ``a~b`` is itself; zsh takes the tilde of ``""~`` and ``\\:~``
    too), unquoted braces around a ``,`` or ``..`` (bash exports
    ``a{b,c}`` as ``ac``, zsh as itself), an unquoted parenthesis (an
    array: bash assigns ``(one)`` as ``one``) and a backslash that
    ends *raw* (the shell joins the next line).  ``None`` too when one
    ``&`` or ``|`` follows: the assignment runs in the background or in
    a pipeline, where the shell keeps none (an unquoted URL is cut at
    its ``&``), also when that operator stands further on, after a
    redirection or an ``&&`` (:func:`_rc_keeps_assignment`).  An ``=`` that opens
    the value is read as bash reads it, a literal, though zsh looks
    ``=ls`` up as a command: the store of a previous release may hold
    such a value unquoted (:func:`_store_quote`).

    Raises:
        ValueError: On an unbalanced quote, which no shell would accept.
    """
    words: list[str] = []
    word = ""
    begun = continued  # a word has begun: ``""`` is one, an empty one
    state = ""
    brace = -1  # where in the first word an unquoted ``{`` stands
    i = 0
    while i < len(raw):
        c = raw[i]
        if state == "'":
            if c == "'":
                state = ""
            else:
                word += c
        elif state == '"':
            if c == "\\" and raw[i + 1 : i + 2] in ('$', '`', '"', "\\"):
                i += 1
                word += raw[i]
            elif c == '"':
                state = ""
            elif c in "$`":
                return None
            else:
                word += c
        elif c in "'\"":
            state = c
            begun = True
        elif c == "\\":
            i += 1
            if i == len(raw):
                return None
            word += raw[i]
        elif c in "$`" or (not words and _expands_in_first_word(c, word, continued)):
            return None
        elif c in " \t":
            if word or begun:
                words.append(word)
                word, begun = "", False
        elif c == "#" and not word and not begun:
            break
        elif c in ";&|":
            if c != ";" and raw[i : i + 2] not in ("&&", "||", "&>"):
                return None  # in the background, in a pipeline: no assignment the shell keeps
            break
        elif c in "<>":
            if _RC_REDIRECTION_OPERATOR_RE.match(raw, i) is None:
                return None  # a process substitution
            if not begun and _RC_IO_NUMBER_RE.fullmatch(word):
                word = ""  # the redirection's descriptor number
            break
        else:
            if c == "}" and not words and brace >= 0 and re.search(r",|\.\.", word[brace:]):
                return None
            if c == "{":
                brace = len(word)
            word += c
        i += 1
    if state:
        raise ValueError("No closing quotation")
    if not _rc_keeps_assignment(raw[i:]):
        return None
    if word or begun:
        words.append(word)
    return words


def _split_assignment(
    line: str, m: re.Match[str]
) -> tuple[str, str, list[str] | None] | None:
    """Split the value of the ``NAME=value`` *line* matched by *m* into shell words.

    Returns ``None`` when the value would need shell expansion (see
    :func:`_shell_words`); otherwise ``(name, raw, words)`` with the
    stripped raw value and its words, ``None`` for a remainder no shell
    could parse (unbalanced quote).  A blank right after the ``=`` makes
    the value empty, whatever follows (``KEY= # removed``, ``KEY=
    other``): that is what the shell assigns.
    """
    name, value = m.group(1), m.group(2)
    raw = value.strip()
    if value[:1] in (" ", "\t"):
        return name, raw, [""]
    cut = value[value.index(raw) + len(raw) :]  # the blanks ``strip`` took off the end
    if cut and (len(raw) - len(raw.rstrip("\\"))) % 2:
        raw += cut[0]  # the first is one a backslash escapes: the value's
    # Right after ``NAME=`` the value continues the assignment's word;
    # after fish's ``set -gx NAME`` and its blanks it is a word of its own.
    try:
        words = _shell_words(raw, continued=line[m.end(1)] == "=")
    except ValueError:
        return name, raw, None
    return None if words is None else (name, raw, words)


def _assigns_for_a_command(line: str) -> bool:
    """Whether the assignments that open *line* are only the environment of a command.

    ``KEY=value command`` assigns nothing the shell keeps, whereas
    ``KEY=value OTHER=x`` assigns both.  A quoted word is no assignment.
    """
    command = _rc_commands(line)[0]
    words = [command[b:e] for kind, b, e in _rc_tokens(command) if kind != "comment"]
    return not all(_RC_ASSIGNMENT_WORD_RE.match(words[i]) for i in _rc_operands(words, 0))


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    """Parse one ``api_keys.env`` line into ``(name, value)``, or ``None``.

    Returns ``None`` for blank lines, comments, anything that is not a
    single ``[export] NAME=value`` assignment, shell-only variables
    (see :data:`_SHELL_ONLY_ENV_VAR_RE`), and values that would need
    shell expansion (see :func:`_shell_words`) — those came from
    an RC distillation where the shell was *meant* to expand them, and
    importing them verbatim would corrupt the environment.

    Quoting is undone so a value stored as ``'sk-abc'`` or ``"sk-abc"``
    loads as ``sk-abc``.  When the remainder splits into several words
    — ``export KEY=two words`` — the first word is the value, exactly
    what a shell sourcing the same line assigns.  ``None`` where the
    shell holds something else: a later word assigns the name again
    (``export KEY=a KEY=b``), or the assignment has no ``export`` and a
    command follows (``KEY=two words`` is the environment of the command
    ``words``, :func:`_assigns_for_a_command`).  A remainder no shell
    could parse (unbalanced quote) is kept as the raw text.
    """
    m = _ENV_ASSIGNMENT_RE.match(line)
    split = None if m is None else _split_assignment(line, m)
    if m is None or split is None:
        return None
    name, raw, parts = split
    if _SHELL_ONLY_ENV_VAR_RE.match(name):
        return None
    if parts is None:
        return name, raw
    if any(word.startswith((f"{name}=", f"{name}+=")) for word in parts[1:]):
        return None
    if not line[: m.start(1)].strip() and _assigns_for_a_command(line):
        return None
    return name, parts[0] if parts else ""


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


def _read_user_text(path: Path) -> str:
    """Read the key store or a shell RC, keeping bytes that are not UTF-8.

    Decoded with ``surrogateescape``: a stray Latin-1 byte (an old
    ``café`` comment) becomes a lone surrogate instead of raising
    ``UnicodeDecodeError``, which is no ``OSError`` and used to abort the
    daemon start, and :func:`atomic_write_text` with the same handler
    writes it back unchanged.  Read waiting out a concurrent
    :func:`atomic_write_text` publish of the same file (a Windows
    ``open`` fails during the replace): the store is read without its
    lock by :func:`_import_api_keys_store`, and a channel agent the
    daemon spawns must not start with no keys because the settings
    panel was saving one.
    """
    return read_text_waiting_for_writer(path, errors="surrogateescape")


def _read_rc_text(path: Path) -> str:
    """Read a shell RC as :func:`_read_user_text` does, its carriage returns kept.

    For :func:`_rc_units`: to the shell they are ordinary characters
    and end no line.
    """
    return read_bytes_waiting_for_writer(path).decode("utf-8", errors="surrogateescape")


def _has_line_break(value: str) -> bool:
    """Whether *value* holds a character ``str.splitlines`` breaks at.

    Every reader of the key store and the RCs splits with ``splitlines``,
    which also breaks at ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``,
    ``\\u2028`` and ``\\u2029``: a value holding one is stored as two
    lines and reloads cut short.
    """
    return "".join(value.splitlines()) != value


@contextlib.contextmanager
def _api_keys_store_flock() -> Iterator[None]:
    """Hold the cross-process flock guarding the canonical key store.

    A :mod:`kiss.core.file_lock` lock on a sidecar ``.api_keys.env.kiss.lock`` in the
    store's directory, so two writers (two daemon threads, or two
    processes sharing one ``$KISS_HOME``) cannot both read the same
    snapshot and silently drop each other's key.  The sidecar is
    flocked rather than the store itself because the store is
    atomically ``os.replace``-d: a lock on the old inode would not
    exclude a writer that opens the new one.

    Every holder must already hold :data:`_config_lock` — that is what
    serializes threads *within* one process, so a single process never
    flocks the sidecar through two file descriptors at once (which
    would self-deadlock: flock exclusion is per open file description,
    not per process).  Never call a function that takes this lock while
    already holding it.
    """
    env_path = api_keys_env_path()
    lock_path = env_path.with_name("." + env_path.name + ".kiss.lock")
    with exclusive_file_lock(lock_path, timeout=_LOCK_TIMEOUT_S):
        yield


def _store_quote(value: str) -> str:
    """Return *value* quoted for the store, which bash and zsh both source.

    ``shlex.quote``, and quotes around a value that opens with ``=``
    as well: zsh looks such an unquoted word up as a command (``=ls``)
    and stops sourcing the store when there is none.
    """
    quoted = shlex.quote(value)
    return f"'{quoted}'" if quoted.startswith("=") else quoted


def _edit_api_keys_env_file_locked(
    mutations: dict[str, str | None], trailer: str = ""
) -> None:
    """Apply *mutations* to the canonical key store atomically.

    Each entry maps a variable name to its new value, or to ``None`` to
    delete every line assigning it.  Existing assignment lines are
    replaced in place; new keys are appended as ``export NAME=value``
    with the value quoted (:func:`_store_quote`), followed by *trailer* (a comment
    line ending in a newline, or empty) unless the store holds that
    line already.  Unrelated lines — including ones
    this module would not import, such as RC-distilled ``PATH``
    entries — pass through byte-for-byte.

    The caller must hold :data:`_config_lock` **and**
    :func:`_api_keys_store_flock`; callers whose decision depends on
    prior reads of the store or of the shell RCs (:func:`save_api_key`'s
    delete-everywhere contract, :func:`_migrate_legacy_rc_keys`'s
    missing-key snapshot) hold the flock across the whole read-decide-
    write transaction so a concurrent process cannot invalidate the
    snapshot mid-flight.  A legacy systemd mirror is deleted inside the
    same critical section (see :func:`_remove_systemd_mirror`).
    """
    env_path = api_keys_env_path()
    lines: list[str] = []
    if env_path.exists():
        lines = _read_user_text(env_path).splitlines(keepends=True)
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
            new_lines.append(f"export {hit}={_store_quote(value)}\n")
        replaced.add(hit)
    tail = [
        f"export {name}={_store_quote(value)}\n"
        for name, value in mutations.items()
        if value is not None and name not in replaced
    ]
    # Compared without line ends: the store's last line may lack one.
    if trailer and trailer.rstrip("\n") not in (line.rstrip("\r\n") for line in new_lines):
        tail.append(trailer)
    if tail and new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    new_lines += tail
    # The store holds API keys: never world-readable, never observed
    # half-written by a shell that is sourcing it.
    atomic_write_text(env_path, "".join(new_lines), mode=0o600, errors="surrogateescape")
    _remove_systemd_mirror()


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

    Values containing a line break are refused: any character
    ``str.splitlines`` breaks at (see :func:`_has_line_break`), not only
    ``\\n`` and ``\\r``.  ``shlex.quote`` would write a *valid multiline*
    quoted assignment, but every read and edit here is line-oriented: the
    next start would load only the first line's fragment, and a later
    replace or delete would rewrite only that line and leave the file
    syntactically broken (an unterminated quote).  Real API keys are
    single-line, so the safe contract is to reject the value outright.

    Also sets the key in the current process environment and refreshes
    the :data:`kiss.core.config.DEFAULT_CONFIG` singleton so subsequent
    model queries see the new key immediately.

    The whole store-edit-plus-RC-scrub critical section runs under
    :data:`_config_lock` plus the canonical store's cross-process flock
    (:func:`_api_keys_store_flock`): a *deletion* is only complete once
    both the store and every RC are scrubbed, and holding the store
    flock across both makes that compound state transition atomic with
    respect to :func:`_migrate_legacy_rc_keys` running in another
    process — otherwise the migration could read a not-yet-scrubbed RC
    after this function's store edit and resurrect the deleted key.
    Lock order is store flock → RC flock, everywhere.  Each RC
    read-modify-replace additionally takes a :mod:`kiss.core.file_lock` lock on a
    sidecar ``<rc>.kiss.lock`` beside the RC, held until the store is
    written (beside the file itself where the RC is a symlink: two
    homes whose links name one dotfiles RC share that lock).  The
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
            "Refusing to save API key with invalid name %r",
            key_name,
        )
        return
    if _has_line_break(key_value):
        logger.warning(
            "Refusing to save API key %s with an embedded line break",
            key_name,
        )
        return
    if "\x00" in key_value:
        # ``shlex.quote`` and the file writer would happily persist a NUL,
        # but ``os.environ[...] = value`` raises on it — both here and at
        # every later daemon start, permanently breaking startup from a
        # poisoned store.  The value arrives from an untrusted client
        # payload, so it is refused before anything is written.
        logger.warning(
            "Refusing to save API key %s with an embedded NUL",
            key_name,
        )
        return
    locked_call(_config_lock, _save_api_key_locked, key_name, key_value, _get_user_shell())


def _save_api_key_locked(key_name: str, key_value: str, user_shell: str) -> None:
    """The store edit and RC scrub of :func:`save_api_key`; the caller holds ``_config_lock``."""
    with _api_keys_store_flock(), contextlib.ExitStack() as rc_locks:
        # The key must disappear from EVERY RC a previous release may
        # have written it to, not only the current $SHELL's: a copy left
        # in another shell's RC would be re-imported by the legacy-key
        # migration (or re-shipped by a deploy) after the user deleted
        # the key — and users do switch default shells between releases.
        # A save that fails must change nothing: the store edited before
        # a failed scrub would hold the new key while this daemon kept
        # using, and showing, the old one until a restart, and an RC
        # scrubbed before a later step failed would have lost the old
        # one.  So every RC is locked, read and edited in memory first
        # (the steps that fail: an unreadable RC, a lock another process
        # holds, an assignment no edit is safe for); then the RCs and,
        # last, the store are written, and a write that fails puts the
        # RCs written before it back.  The hook can only
        # go where the shell can source a bash-syntax file (not fish)
        # and where the ``$HOME``-based hook line names the canonical
        # store (no ``$KISS_HOME`` override); an absent RC that would
        # get no hook is left alone, lock file included.
        hook_wanted = (
            bool(key_value)
            and user_shell != "fish"
            and api_keys_env_path() == Path.home() / HOME_DIR / API_KEYS_ENV_FILE
        )
        readers: dict[Path, list[tuple[str, bool]]] = {}  # each RC file: its shells, their hook
        for shell in ("bash", "zsh", "fish"):
            rc = _shell_rc_path(shell)
            install_hook = hook_wanted and shell == user_shell
            if not rc.exists() and not install_hook:
                continue
            if rc.is_symlink():
                # An RC that is a symlink into a dotfiles repository is
                # edited where it really lives: replacing the link would
                # disconnect the file somebody maintains (and leave the
                # deleted key inside it), as ``scripts/install-api-keys.sh``
                # knows too.  Resolved once, here: the lock is that
                # file's, which the links of two homes both reach.
                try:
                    rc = rc.resolve(strict=True)
                except OSError:
                    continue
                if not rc.is_file():
                    continue
            readers.setdefault(rc, []).append((shell, install_hook))
        edits: list[tuple[Path, str | None, int, str]] = []  # RC, old text and mode, new text
        named = False  # an RC names the key after the edit: a spelling it does not reach
        for rc in sorted(readers):  # one order in every process: no two wait for each other
            rc_lock = rc.with_name(rc.name + ".kiss.lock")
            rc_locks.enter_context(exclusive_file_lock(rc_lock, timeout=_LOCK_TIMEOUT_S))
            old = _read_rc_text(rc) if rc.exists() else None
            new = old or ""
            for shell, install_hook in readers[rc]:
                new = _update_rc_for_key(rc, new, shell, key_name, install_hook)
            named = named or key_name in new
            if new != (old or ""):
                edits.append((rc, old, 0 if old is None else rc.stat().st_mode & 0o7777, new))
        # A deleted key an RC still names (given to a function of its own that exports
        # it) would be read back by the migration's one-time sourcing of that RC:
        # marked as done.
        trailer = _RC_SOURCED_MARKER + "\n" if named and not key_value else ""
        written = 0
        try:
            for rc, _, _, new in edits:
                written += 1
                # The RC may hold API keys: never world-readable, never observed
                # half-written by a shell that is sourcing it.  (On Windows
                # ``os.chmod`` honours only the read-only bit.)
                atomic_write_text(
                    rc, new, mode=0o600, errors="surrogateescape", follow_symlink=False
                )
            _edit_api_keys_env_file_locked({key_name: key_value or None}, trailer)
        except BaseException:
            for rc, old, mode, _ in reversed(edits[:written]):
                try:
                    if old is None:
                        rc.unlink(missing_ok=True)
                    else:
                        atomic_write_text(
                            rc, old, mode=mode, errors="surrogateescape", follow_symlink=False
                        )
                except OSError:
                    logger.warning("Failed to restore %s after a failed save", rc, exc_info=True)
            raise
        if key_value:
            os.environ[key_name] = key_value
        else:
            os.environ.pop(key_name, None)
        _refresh_config()


RC_HOOK_BEGIN = "# >>> sorcar-cloud API keys >>>"
RC_HOOK_END = "# <<< sorcar-cloud API keys <<<"
_RC_HOOK_LINE = (
    f'[ -f "$HOME/{HOME_DIR}/{API_KEYS_ENV_FILE}" ] && . "$HOME/{HOME_DIR}/{API_KEYS_ENV_FILE}"'
)
"""The RC block that sources the canonical key store.

Markers and hook line (for the stock ``~/.kiss``) are byte-identical to
what ``scripts/install-api-keys.sh`` writes on ``./rsorcar`` deploy
targets, so a machine that already has the deploy-installed block is
recognized and never gets a second one.  A white-label brand sources its
own home's key store.
"""


def _update_rc_for_key(rc: Path, text: str, shell: str, key_name: str, install_hook: bool) -> str:
    """Return *text*, what *shell*'s *rc* holds, without *key_name*'s assignment, with the hook.

    Keys live only in the canonical store, so any ``export KEY=…``
    (``set -gx KEY …`` for fish) line a previous release wrote into the
    RC is removed — on save as well as on delete, otherwise a stale RC
    line sourced *after* the hook would shadow the canonical value in
    interactive shells.  When *install_hook* is true (the caller's
    decision: a bash/zsh RC, a store at its ``~/.kiss`` home), the
    :data:`RC_HOOK_BEGIN` block is added once so terminals read the
    store directly.

    The text is read as the shell reads it (:func:`_rc_units`): only a
    command on a line of its own is edited, and only such a line is the
    hook or its end marker, so the text of a here-document is neither
    changed nor taken for one; where the reading is unsure every line
    counts, as nothing better is known.  All else the shell reads
    stays, and what it reads stays a script: a line emptied where the
    command before it awaits one (the only command between ``then``
    and ``fi``) keeps a ``true`` (:func:`_scrub_rc_line`).  The block
    goes where the shell reads a new command: to the end, or before a
    last command the text ends inside of.

    Raises:
        OSError: When the RC may assign the key where no edit is safe,
            naming the first such line: a command that sets it in a
            way the edit does not remove (:func:`_rc_key_use`), or
            that assigns it at all over several lines, which are not
            edited; a text not read command by command that may
            (:func:`_rc_text_sets`).  Nothing is to be written then:
            the key would outlive a deletion reported as done.
    """
    # Bash 4+ drops NUL characters while sourcing a file; Bash 3.2 keeps
    # reading the surrounding here-document, so editing the dropped view
    # there could remove text that is no command to the actual shell.
    drops_nul = shell == "bash" and _bash_drops_nul_when_sourcing()
    seen = text.replace("\0", "") if drops_nul else text
    lines = _RC_LINE_RE.findall(seen)
    units, read, end = _rc_units(seen, shell)
    heads = {unit[0]: unit for unit in units}
    tail = list(range(read, len(lines)))  # the lines not read for certain: one text
    kept: list[str] = []
    code: list[int] = []  # where in *kept* the lines that are, or may be, commands stand
    at = None  # where in *kept* line *end* stands
    awaited = False  # the command before this line needs one after it (``then``, ``&&``)
    for n, line in enumerate(lines):
        if n == end:
            at = len(kept)
        unit = tail if n == read else heads.get(n, [])
        single = len(unit) == 1 and n < read  # a command on a line of its own: the edit's
        if unit:
            # without its backslash-newlines, as for the shell
            whole = "".join(lines[m] for m in unit).replace("\\\n", "").removesuffix("\n")
            scrubbed: str | None = line
            if n == read or not _rc_one_line(whole, shell):
                unsure = _rc_text_sets(whole, shell, key_name)
            elif single:
                scrubbed, unsure = _scrub_rc_line(line, shell, key_name, awaited)
            else:  # the lines of a command of several are not edited
                edited, unsure = _scrub_rc_line(whole, shell, key_name, awaited)
                unsure = unsure or edited != whole
            if unsure:
                raise OSError(
                    f"{rc}: line {n + 1} may set {key_name} where it cannot be edited safely;"
                    " remove it there by hand"
                )
            awaited = _rc_awaits_command(whole if scrubbed is line else scrubbed or "", awaited)
            if scrubbed is None:
                continue
            line = scrubbed
        if single or read <= n < end:
            code.append(len(kept))
        kept.append(line)
    if install_hook and not any(kept[i].rstrip("\r\n") == _RC_HOOK_LINE for i in code):
        # A block from another brand's install (stock KISS sourcing
        # ~/.kiss/api_keys.env, say) shares the markers: this brand's
        # source line joins that block instead of opening a second one.
        ends = [i for i in code if kept[i].rstrip("\r\n") == RC_HOOK_END]
        if ends:
            kept.insert(ends[0], _RC_HOOK_LINE + "\n")
        else:
            if at is None and kept:
                # The last line must end before the block.  A backslash before its line
                # feed continues it: an empty line ends it.  One that ends the text is
                # itself for bash (as when doubled) and nothing for zsh.
                last = kept[-1].removesuffix("\n")
                continues = (len(last) - len(last.rstrip("\\"))) % 2 == 1
                literal = continues and shell == "bash" and last == kept[-1]
                kept[-1] = last + ("\\\n" if literal else "\n\n" if continues else "\n")
            at = len(kept) if at is None else at
            kept[at:at] = [RC_HOOK_BEGIN + "\n", _RC_HOOK_LINE + "\n", RC_HOOK_END + "\n"]
    edited = "".join(kept)
    return text if edited == seen else edited


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
    for key in API_KEY_ENV_VARS:
        setattr(config_module.DEFAULT_CONFIG, key, os.environ.get(key, ""))


def apply_config_to_env(cfg: dict[str, Any]) -> None:
    """Apply loaded config values to the running process.

    Sets ``max_budget``, ``claude_code_use_subscription``,
    ``codex_use_subscription`` and ``compaction`` (as
    ``tool_output_compaction``) on the default config.

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
    budget = cfg.get("max_budget", DEFAULTS["max_budget"])
    sanitized = sanitize_config({"max_budget": budget})
    config_module.DEFAULT_CONFIG.max_budget = float(sanitized["max_budget"])
    # The subscription switches are read by every ``cc/`` / ``codex/``
    # turn from ``DEFAULT_CONFIG`` (the model layer never opens config.json).
    for switch in ("claude_code_use_subscription", "codex_use_subscription"):
        setattr(
            config_module.DEFAULT_CONFIG,
            switch,
            bool(cfg.get(switch, DEFAULTS[switch])),
        )
    # The global Compaction switch: every agent's
    # ``_maybe_compact_conversation`` reads it at each step.
    config_module.DEFAULT_CONFIG.tool_output_compaction = bool(
        cfg.get("compaction", DEFAULTS["compaction"])
    )


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


def load_api_keys_readonly() -> None:
    """Import the canonical key store into ``os.environ`` without writing.

    Fallback for the direct CLI entry points when :func:`load_api_keys`
    cannot run (a read-only ``$KISS_HOME`` where its migration lock
    file cannot be created): the store is parsed with the same
    :func:`_parse_env_assignment` rules and the in-memory
    ``DEFAULT_CONFIG`` is refreshed, but the filesystem-writing steps
    (the legacy-RC migration and the systemd-mirror retirement) are
    skipped.  A store that cannot even be read is treated as empty,
    exactly as in :func:`load_api_keys` (same importer).
    """
    with _config_lock:
        _import_api_keys_store()


def _import_api_keys_store() -> None:
    """Import every assignment of the canonical key store into ``os.environ``.

    The one importer behind :func:`load_api_keys` and
    :func:`load_api_keys_readonly`: the store is parsed with
    :func:`_parse_env_assignment`, each value lands in ``os.environ``
    and :func:`_refresh_config` syncs ``DEFAULT_CONFIG``.  A missing
    store is empty; one that cannot be read is logged and treated as
    empty too, so a bad store never stops a daemon or CLI start.  The
    caller holds :data:`_config_lock`.
    """
    env_path = api_keys_env_path()
    try:
        text = _read_user_text(env_path)
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
            logger.warning("Skipping unusable %s line for %s", env_path.name, parsed[0])
    _refresh_config()


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

    The migration is best effort: a store lock another process holds past
    :data:`_LOCK_TIMEOUT_S` is logged and skipped, and the daemon starts
    with whatever the store already holds.
    """
    try:
        _migrate_legacy_rc_keys()
    except TimeoutError as exc:
        logger.warning("Skipping legacy RC key migration: %s", exc)
    _remove_systemd_mirror()
    with _config_lock:
        _import_api_keys_store()


_MIGRATION_TIMEOUT_S = 5.0
"""How long the migration's RC-sourcing shell may run before it is killed."""

_RC_SOURCED_MARKER = "# shell RC sourced once for legacy API keys (kiss)"
"""Comment line :func:`_migrate_legacy_rc_keys` appends to the store after sourcing.

:func:`save_api_key` appends it too when it deletes a key an RC still
names: sourcing that RC at the next start would bring the key back.

A comment, so the shell hook that sources the store and every
assignment parser ignore it, and :func:`_edit_api_keys_env_file_locked`
passes it through untouched.
"""

_FISH_ASSIGNMENT_RE = re.compile(
    r"^\s*set\s+-gx\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$",
)

_RC_ODD_BLANK_RE = re.compile(r"[^\S \t]")
"""Whitespace for ``re`` and ``str`` that is an ordinary character for bash and zsh.

A carriage return, a form feed, a no-break space: ``\\rexport KEY=x``
runs no ``export``, and the assignment patterns here would read one.
"""


def _scan_rc_text_for_keys(text: str, shell: str, wanted: set[str]) -> dict[str, str | None]:
    """Textually extract *wanted* API-key assignments from RC *text*.

    Reads the line formats previous releases wrote (``export
    KEY=value``; ``set -gx KEY value`` for fish) without running a
    shell, so an assignment *after* a stock Debian/Ubuntu interactivity
    guard (``case $- in *i*) ;; *) return``) is still found: sourcing
    the RC non-interactively returns before it.  The text is read as
    the shell reads it (:func:`_rc_units`: a here-document assigns
    nothing), every command of a line (``export A=…; export B=…``),
    and later assignments win.

    A name maps to ``None`` when the RC assigns it but the text cannot
    tell the value the store should hold: an earlier literal is
    forgotten with it and the sourcing fallback resolves it.  That is
    a last assignment that is empty, expansion-dependent, multi-word
    or NUL-carrying; one the shell may not run or runs in a subshell
    (a branch of ``&&``/``||``, a pipeline member, a background
    command, the line after one ended by ``&&``, ``||`` or ``|``); one
    that is not the assignment alone on a line of its own (under
    ``then``, in a ``case`` clause, after an option, a second
    operand); and, for the whole text whatever a later line assigns, a
    command that may assign or alter the key otherwise
    (:func:`_rc_key_use`: ``typeset -x KEY=…``, ``read KEY``, ``unset
    KEY``) or assigns it inside a compound command, followed over its
    lines (:func:`_rc_command`: the body of an ``if``, a loop, a
    function, which may run at any later time).  A command that only
    names the key changes nothing: an expansion, a message, an
    existence test, ``env -u KEY``, ``docker run -e KEY``, an
    ``export`` of the name alone.

    When a compound command is closed that the reading saw none open
    of, nothing of the text is trusted: every name it holds is the
    shell's.  What is not read command by command
    (:func:`_rc_one_line`; a line with whitespace that is none to the
    shell, :data:`_RC_ODD_BLANK_RE`; the lines not read for certain)
    leaves a key to the shell when it may assign it
    (:func:`_rc_text_sets`).  One carriage return before the line feed
    is the line's end, as before: a CRLF line's value is the shell's
    without it.
    """
    found: dict[str, str | None] = {}
    shell_only: set[str] = set()  # written where the text cannot tell what becomes of them
    pattern = _FISH_ASSIGNMENT_RE if shell == "fish" else _ENV_ASSIGNMENT_RE
    if shell == "bash" and _bash_drops_nul_when_sourcing():
        text = text.replace("\0", "")  # Bash 4+ drops them
    lines = _RC_LINE_RE.findall(text)
    units, read, _ = _rc_units(text, shell)
    depth = 0  # the compound commands open where a command begins
    awaited = False  # the command before this one needs one after it (``&&``, ``then``)
    for unit in units:
        line = "".join(lines[n] for n in unit).replace("\\\n", "")
        line = line.removesuffix("\n").removesuffix("\r")
        split = _rc_one_line(line, shell) and _RC_ODD_BLANK_RE.search(line) is None
        if not split:
            shell_only.update(name for name in wanted if _rc_text_sets(line, shell, name))
            line = line.replace("\n", " ")
        pieces = _rc_commands(line)
        for i in range(0, len(pieces), 2):
            # A command on a line of its own, of no compound command: the text may tell
            # what it assigns.
            top = split and len(unit) == 1 and depth == 0
            command = _rc_command(pieces[i], shell, depth)
            depth = command.depth
            if depth < 0:
                return dict.fromkeys(name for name in wanted if name in text)
            m = pattern.match(pieces[i])
            before = pieces[i - 1] if i else "&&" if awaited else ";"
            after = pieces[i + 1] if i + 1 < len(pieces) else ";"
            bounded = before in (";", "&") and after in (";", "&&", "||")
            if i >= 2 and before == ";" and not pieces[i - 2].strip() and (i > 2 or awaited):
                bounded = False  # zsh's ``cmd ||; export …``: a ``;`` after no command ends nothing
            for name in wanted if split else ():
                if name not in pieces[i]:
                    continue
                scrubbed, sets, alters = _rc_key_use(command, shell, name)
                if sets or alters or (scrubbed is not None and not (top and command.first == 0)):
                    shell_only.add(name)
                elif scrubbed is not None:
                    found[name] = None
                    if m is not None and m.group(1) == name and bounded:
                        parts = (_split_assignment(pieces[i], m) or (None, None, None))[2]
                        if parts is not None and len(parts) == 1 and "\x00" not in parts[0]:
                            found[name] = parts[0] or None
        awaited = _rc_awaits_command(line, awaited)
    tail = "".join(lines[read:]).replace("\\\n", "")
    shell_only.update(name for name in wanted if _rc_text_sets(tail, shell, name))
    found.update(dict.fromkeys(shell_only))
    return found


def _source_rc_for_keys(wanted: set[str]) -> dict[str, str]:
    """Source the user's shell RC once and return the *wanted* keys it sets.

    The shell runs with a **minimal clean environment** rather than the
    daemon's: only values the RC itself defines may migrate into the
    persistent store — a key injected into the daemon ad hoc (``KEY=x
    kiss-web``) must stay ephemeral.  The subprocess gets its own
    session so that on timeout, or on a Stop raised in this thread, the
    whole process group is killed: a background child started by the RC
    would otherwise keep the output pipe open and stall daemon startup
    long past the timeout.  Failures (missing shell binary, hanging or
    broken RC) are logged and yield nothing — a user-maintained RC must
    not block startup.
    """
    shell = _get_user_shell()
    rc = _shell_rc_path(shell)
    if not rc.exists():
        return {}
    shell_path = _resolve_shell_path(shell)
    if shell_path is None:
        logger.warning(
            "Cannot migrate RC API keys: %s binary not found on PATH or fallback locations",
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
    try:
        stdout = run_process_group(
            [shell_path, "-c", cmd], timeout=_MIGRATION_TIMEOUT_S, env=clean_env
        ).stdout
    except subprocess.TimeoutExpired:
        logger.warning(
            "Shell RC took longer than %.0fs to source; skipping key migration from it",
            _MIGRATION_TIMEOUT_S,
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
            if k in wanted and v and not _has_line_break(v) and "\x00" not in v:
                found[k] = v
    return found


def _migrate_legacy_rc_keys() -> None:
    """Import RC-resident API keys into the canonical store, additively.

    Earlier releases persisted keys as ``export`` (``set -gx``) lines in
    a shell RC and re-sourced that RC at every daemon start.  For each
    :data:`API_KEY_ENV_VARS` variable *absent* from the canonical file,
    this looks in every supported RC — first textually (which sees past
    a stock ``.bashrc``'s non-interactive ``return`` guard), then, only
    when some RC *names* a key the scan could not import (a value
    assembled indirectly), by sourcing the user's shell RC — and copies
    what it finds into the file.  The user's shell's RC is read first
    and the other shells' RCs only fill the keys it does not set: a key
    left behind in the RC of a shell the user no longer runs must not
    beat the current one -- not even when the current one is a value
    only the shell can resolve (sourcing does) or an explicit empty
    assignment (the key stays unset).  Keys already in the file are
    never touched, and when nothing is missing no RC is read at all.

    Sourcing runs arbitrary RC code for up to :data:`_MIGRATION_TIMEOUT_S`
    while holding both locks below, so it happens **at most once per
    store**: :data:`_RC_SOURCED_MARKER` is appended to the store
    afterwards and later starts never spawn the shell again, even when
    the RC keeps naming a key it does not set.  (Almost every store
    lacks some of the :data:`API_KEY_ENV_VARS`, so "some key is
    missing" alone would mean sourcing on every start.)

    The whole read-source-write transaction holds :data:`_config_lock`
    **and** the canonical store's cross-process flock
    (:func:`_api_keys_store_flock`): otherwise a concurrent
    settings-panel *deletion* — in this process or in another daemon
    sharing the same ``$KISS_HOME`` — could scrub the store and the RCs
    between this function's snapshot and its final write, and the stale
    RC observation would resurrect the key the user just deleted.
    :func:`save_api_key` holds the same flock across its own store edit
    plus RC scrub, so the two compound operations serialize in either
    order and a completed deletion stays deleted.
    """
    with _config_lock, _api_keys_store_flock():
        env_path = api_keys_env_path()
        store_text = ""
        if env_path.exists():
            try:
                store_text = _read_user_text(env_path)
            except OSError:
                logger.warning("Failed to read %s", env_path, exc_info=True)
                return
        present = {
            m.group(1)
            for line in store_text.splitlines()
            if (m := _ENV_ASSIGNMENT_RE.match(line)) is not None
        }
        missing = set(API_KEY_ENV_VARS) - present
        if not missing:
            return
        found: dict[str, str | None] = {}
        rc_texts: list[str] = []
        user_shell = _get_user_shell()
        others = [shell for shell in ("bash", "zsh", "fish") if shell != user_shell]
        for shell in (user_shell, *others):
            rc = _shell_rc_path(shell)
            try:
                text = _read_rc_text(rc)
            except OSError:
                continue
            rc_texts.append(text)
            # The first RC assigning a key decides it, also with a value
            # (``None``) the text cannot import: a leftover in a later
            # RC must not fill in for what only the shell can resolve.
            for name, value in _scan_rc_text_for_keys(text, shell, missing).items():
                found.setdefault(name, value)
        still_missing = {name for name in missing if found.get(name) is None}
        sourcing = _RC_SOURCED_MARKER not in store_text and any(
            name in text for text in rc_texts for name in still_missing
        )
        if sourcing:
            found.update(_source_rc_for_keys(still_missing))
        mutations: dict[str, str | None] = {
            name: value for name, value in found.items() if value is not None
        }
        if mutations or sourcing:
            _edit_api_keys_env_file_locked(
                mutations, trailer=_RC_SOURCED_MARKER + "\n" if sourcing else ""
            )
