# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slash-command registry for Sorcar Extension Agents (SEAs).

A SEA named ``xxx`` is a folder ``xxx/`` that contains the script
``xxx_sea.py`` plus whatever helper modules and data files the SEA
needs.  Every such folder visible to the daemon is exposed as a chat
command named after the folder: ``/xxx``.  When a user submits a
prompt that starts with ``/xxx`` — optionally followed by whitespace
and free-form text — the daemon runs the SEA directly on the trailing
text (:func:`slash_command_task`): the same run ``run_agent(agent="xxx",
task=text)`` makes.  Two special prompts do not run the SEA: ``/xxx
help`` answers with the return value of the script's mandatory
``description()`` function, and ``/xxx check`` executes the script
and reports its effective settings, model, tools and sample prompt —
or the first error (see :func:`help_text_if_command`,
:func:`sea_check`).

This module also evaluates a SEA for a run: :func:`sea_layers` executes
the script and the scripts it ``extends`` (each exactly once, with the
one loader :func:`kiss.agents.sorcar.sea_settings.execute_python_file`),
:func:`evaluate_sea` turns the layers into the run's configuration (a
:class:`SeaRun`), and :func:`sea_settings` gives the dispatcher the
merged settings.

The registry is built from four sources, in decreasing precedence:

1. ``src/kiss/agents/third_party_agents/`` (highest precedence,
   discovered through the ``kiss.agents.third_party_agents`` package).
2. The folders listed one per line in ``$KISS_HOME/SEAS.md`` (the
   Sorcar home's ``SEAS.md``).  Later lines in the file override earlier
   lines — i.e. the folder at the bottom of ``SEAS.md`` beats the one
   at the top when both contain the same command name.  Blank lines
   and lines starting with ``#`` are ignored; ``~`` and environment
   variables in a folder path are expanded.
3. ``src/kiss/agents/seas/`` (the bundled SEAs that extend Sorcar
   itself, e.g. ``/merge``; discovered through ``kiss.agents.seas``).
   Any ``SEAS.md`` folder that ships a file of the same name replaces
   the bundled one.
4. The built-in agent scripts of :data:`BUILTIN_COMMANDS` (``/cron``,
   lowest precedence).

The registry is refreshed lazily on every lookup and, in the daemon,
proactively by a background polling watcher (see
:func:`start_registry_watcher`) so edits to ``SEAS.md`` — or the
appearance/removal of SEA folders in any of its folders — take
effect while the daemon is running.

A registered SEA whose script defines ``register_as_model()`` returning
``True`` is also a *model-picker entry*: :func:`model_seas` lists such
SEAs under their command names and the daemon offers them in the model
picker next to the real models.  Picking one lays the SEA under every
task of the tab as the outermost layer (on the model its ``model``
setting names, else the default model), so a ``/xxx`` command or a
``run_agent`` child run on that tab keeps its routing protocol
(``add_to_system_prompt()``) and runs its own SEA on top (see
:func:`sea_layers` and :mod:`kiss.agents.sorcar.agent_file`).

Locking: two module locks, always acquired in the order
``_notify_lock`` -> ``_lock``.  ``_lock`` guards the registry and the
subscriber list; ``_notify_lock`` wraps the publish-and-notify section
of :func:`refresh_registry` so subscribers see snapshots in the order
they were published.  Nothing calls :func:`refresh_registry` while
holding ``_lock``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sea_settings import (
    SeaError,
    SettingsError,
    declares_hidden,
    execute_python_file,
    merge_settings,
    resolve_settings,
    safe_message,
    script_name,
)
from kiss.core.config import kiss_home

logger = logging.getLogger("kiss.sea_commands")

# The filename suffix that identifies a SEA agent script.
_SEA_SUFFIX = "_sea.py"

# Command names must be ASCII letters, digits, ``_`` or ``-``.  A file
# whose stem breaks this rule (``foo.bar_sea.py``, ``spaced name_sea.py``)
# is silently ignored: the frontend autocomplete recognises the same
# character class, and _split_slash_command / :func:`get_command` would
# refuse it downstream anyway.
_COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Poll interval for the background watcher, in seconds.
_WATCHER_POLL_SECONDS = 2.0

# In-memory registry: ``{command_name: absolute_path_to_sea_file}``.
_registry: dict[str, Path] = {}

# Guards ``_registry`` and ``_subscribers`` (RLock so a subscriber
# callback that itself calls :func:`list_commands` cannot deadlock).
_lock = threading.RLock()

# Serialises the publish-and-notify section of :func:`refresh_registry`
# so subscribers receive snapshots in the order they were published:
# without it two concurrent rescans could deliver snapshot B before A
# and leave every subscriber on the stale list A for good (the
# ``_last_broadcast`` dedupe then suppresses the correcting rescan).
#
# LOCK ORDER: ``_notify_lock`` -> ``_lock``.  ``_notify_lock`` is taken
# first and held through the callback loop; ``_lock`` is only ever
# taken inside it or on its own.  No function may call
# :func:`refresh_registry` while holding ``_lock``.  RLock so a
# subscriber that re-enters :func:`refresh_registry` (e.g. through
# :func:`get_command` on a miss) does not self-deadlock.
_notify_lock = threading.RLock()

# Callbacks invoked (under ``_notify_lock`` but outside ``_lock``)
# whenever the registry changes.
_subscribers: list[Callable[[list[str]], None]] = []

# Last snapshot broadcast to subscribers.  Deduplicates identical
# rescans so the daemon does not spam clients with unchanged lists.
_last_broadcast: tuple[str, ...] = ()

# ``register_as_model()`` verdict per SEA script, keyed by path and
# stamped with the file's (mtime_ns, size, inode) so an edited or
# replaced script is re-read.  Guarded by ``_lock``.
_model_sea_cache: dict[Path, tuple[tuple[int, int, int], bool]] = {}

# Watcher-thread coordination.
_watcher_thread: threading.Thread | None = None
# The running poller's own stop event (one per thread: a stopped poller
# still finishing a scan must not resume on a successor's cleared flag).
_watcher_stop: threading.Event | None = None


def seas_md_path() -> Path:
    """Return the path to ``SEAS.md`` in the current KISS home.

    Resolved on every call so tests that redirect ``$KISS_HOME`` see
    their own file.
    """
    return kiss_home() / "SEAS.md"


def _package_dir(package: str) -> Path | None:
    """Return the on-disk path of a bundled SEA package.

    Located through the import system so an editable install or a
    zipped install both work; returns ``None`` when the package is
    absent (test harnesses that trim optional packages).

    Args:
        package: Dotted package name, e.g. ``"kiss.agents.seas"``.
    """
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _third_party_dir() -> Path | None:
    """Return the on-disk path of the bundled third-party agents dir."""
    return _package_dir("kiss.agents.third_party_agents")


def _seas_dir() -> Path | None:
    """Return the on-disk path of the bundled ``kiss.agents.seas`` dir."""
    return _package_dir("kiss.agents.seas")


def sea_script_in(sea_dir: Path) -> Path:
    """Return the path of the SEA script that *sea_dir* must contain.

    Under the folder convention a SEA named ``xxx`` lives in a folder
    ``xxx/`` together with its helper modules and data files, and its
    entry script is ``xxx/xxx_sea.py``.

    Args:
        sea_dir: The SEA's folder; its name is the command name.

    Returns:
        ``sea_dir / "<folder name>_sea.py"`` (not checked for existence).
    """
    return sea_dir / f"{sea_dir.name}{_SEA_SUFFIX}"


def _scan_folder(folder: Path) -> dict[str, Path]:
    """Return ``{command_name: absolute_path}`` for every SEA in *folder*.

    A SEA is a sub-folder ``xxx/`` of *folder* that contains the script
    ``xxx_sea.py`` (see :func:`sea_script_in`); the sub-folder's name is
    the command name.  Loose ``*_sea.py`` files directly inside
    *folder* are NOT commands.  Silently skips folders that are
    missing, unreadable, or not a directory: a stale ``SEAS.md`` entry
    must not break the daemon.

    Args:
        folder: The directory to scan.

    Returns:
        Mapping from command name (sub-folder name) to the absolute,
        resolved SEA-script path.  An underscore-prefixed folder is a
        valid command (``_helper/_helper_sea.py`` becomes ``/_helper``);
        only names outside ``[A-Za-z0-9_-]`` and scripts whose
        ``settings()`` declare ``"hidden": True``
        (:func:`~kiss.agents.sorcar.sea_settings.declares_hidden`) are
        skipped.
    """
    out: dict[str, Path] = {}
    try:
        entries = list(folder.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return out
    for sea_dir in entries:
        command = sea_dir.name
        # Reject folder names that would produce a command name outside
        # the ``[A-Za-z0-9_-]`` alphabet the parser and the autocomplete
        # accept — ``foo.bar/`` and ``space name/`` are silently skipped
        # rather than surfacing as commands that cannot be typed.
        if not _COMMAND_NAME_RE.match(command):
            continue
        path = sea_script_in(sea_dir)
        try:
            if not path.is_file() or declares_hidden(path):
                continue
        except OSError:
            continue
        try:
            out[command] = path.resolve()
        except OSError:
            out[command] = path.absolute()
    return out


BUILTIN_COMMANDS: dict[str, str] = {"cron": "kiss.agents.sorcar.cron_agent"}
"""Agent scripts that are modules of the framework, by command name.

``/cron`` is the scheduled-automations agent
(:mod:`kiss.agents.sorcar.cron_agent`): the same script
``run_agent(agent="cron")`` dispatches, registered so one lookup —
:func:`list_commands` — knows every name a task may run.
"""


def _builtin_commands() -> dict[str, Path]:
    """Return ``{command_name: script_path}`` for :data:`BUILTIN_COMMANDS`.

    The modules are located, not imported.
    """
    out: dict[str, Path] = {}
    for command, module in BUILTIN_COMMANDS.items():
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin:
            out[command] = Path(spec.origin).resolve()
    return out


def bundled_commands() -> dict[str, Path]:
    """Return ``{command_name: script_path}`` for every command the package ships.

    The built-in scripts (:data:`BUILTIN_COMMANDS`), the bundled
    ``seas/`` folder and the bundled ``third_party_agents/`` folder, in
    the registry's precedence (a later source overwrites an earlier
    one), without the user's ``SEAS.md`` folders: the deterministic
    list ``sea docs`` renders.
    """
    merged = _builtin_commands()
    for folder in (_seas_dir(), _third_party_dir()):
        if folder is not None:
            merged.update(_scan_folder(folder))
    return merged


def _read_seas_md_folders() -> list[Path]:
    """Return the folder list from ``SEAS.md``, top to bottom.

    Blank lines, lines starting with ``#``, and comment tails after
    an unquoted ``#`` are ignored.  ``~`` and environment variables
    are expanded.  Relative paths are resolved against the user's
    home directory so a bare ``my-seas`` behaves the same across
    working directories.

    Returns:
        The folders in file order.  An unreadable / missing
        ``SEAS.md`` yields an empty list.
    """
    path = seas_md_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
        return []
    folders: list[Path] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # An inline ``  # comment`` tail is stripped only when there
        # is at least one whitespace character before the ``#`` — a
        # bare ``#`` inside a folder name (rare, but ``chan#01`` on
        # some filesystems) is preserved.  We deliberately do NOT
        # shlex-split: doing so mangles common unquoted paths such as
        # ``C:\Users\alice\seas`` on Windows.  A folder name that
        # contains whitespace works verbatim.
        idx = re.search(r"\s#", line)
        folder_text = line[: idx.start()] if idx else line
        folder_text = folder_text.strip()
        if not folder_text:
            continue
        expanded = os.path.expandvars(os.path.expanduser(folder_text))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = Path.home() / candidate
        folders.append(candidate)
    return folders


def refresh_registry() -> list[str]:
    """Rebuild the command registry from every source, applying precedence.

    Precedence (highest wins):

    1. ``src/kiss/agents/third_party_agents/`` (the bundled channel SEAs).
    2. Folders in ``$KISS_HOME/SEAS.md``, from bottom line to top line.
    3. ``src/kiss/agents/seas/`` (the bundled Sorcar-extending SEAs).

    Concretely: the merge walks the sources from LOWEST precedence to
    HIGHEST (bundled ``seas/`` first, then top-of-file SEAS.md entries,
    third-party last) and lets each source overwrite the previous one,
    so the highest source ends up in the registry.

    Returns:
        The sorted list of command names now installed.  Subscribers
        registered via :func:`subscribe` are invoked (under
        ``_notify_lock``, outside ``_lock``) when the list differs
        from the previous broadcast, in publish order.
    """
    global _last_broadcast

    sources: list[dict[str, Path]] = []
    # Built-in agent scripts and the bundled ``seas/`` first: any
    # SEAS.md folder may shadow them.
    sources.append(_builtin_commands())
    seas_dir = _seas_dir()
    if seas_dir is not None:
        sources.append(_scan_folder(seas_dir))
    # SEAS.md folders: top to bottom.  We want bottom to WIN, so
    # iterate top-first and let the bottom entries overwrite.
    for folder in _read_seas_md_folders():
        sources.append(_scan_folder(folder))
    # Third-party agents last: they override everything.
    third_party_dir = _third_party_dir()
    if third_party_dir is not None:
        sources.append(_scan_folder(third_party_dir))

    merged: dict[str, Path] = {}
    for src in sources:
        merged.update(src)

    # Compare-and-swap under ``_lock`` so two concurrent rescans can
    # never both observe the old ``_last_broadcast`` and fire the
    # same notification twice.  The whole publish-and-notify section
    # runs under ``_notify_lock`` (order: ``_notify_lock`` -> ``_lock``)
    # so snapshots reach subscribers in publish order; ``_lock`` itself
    # is released before the callbacks run so a subscriber that calls
    # back into the module — e.g. :func:`list_commands` — cannot
    # deadlock.
    with _notify_lock:
        with _lock:
            _registry.clear()
            _registry.update(merged)
            snapshot = sorted(_registry)
            snapshot_tuple = tuple(snapshot)
            if snapshot_tuple == _last_broadcast:
                return snapshot
            _last_broadcast = snapshot_tuple
            callbacks = list(_subscribers)

        for cb in callbacks:
            try:
                cb(list(snapshot))
            except Exception:  # pragma: no cover - subscribers own errors
                logger.debug("SEA registry subscriber failed", exc_info=True)
    return snapshot


def list_commands() -> list[str]:
    """Return the sorted list of currently-registered command names.

    Rebuilds the registry when it is empty so standalone callers
    (tests, CLI helpers) never see a stale-empty snapshot before the
    daemon watcher has run its first pass.
    """
    with _lock:
        if _registry:
            return sorted(_registry)
    return refresh_registry()


def get_command(name: str) -> Path | None:
    """Resolve *name* (e.g. ``"slack"``) to the SEA script path.

    Returns ``None`` when the name is not registered.  Rebuilds the
    registry on a miss so a SEA created after the daemon started (or
    a stale in-memory copy) does not surface a spurious "unknown
    command" the very first time it is used.
    """
    with _lock:
        hit = _registry.get(name)
    if hit is not None:
        return hit
    refresh_registry()
    with _lock:
        return _registry.get(name)


def subscribe(callback: Callable[[list[str]], None]) -> None:
    """Register *callback* to be invoked (outside the lock) on changes.

    The callback receives the new sorted command list.  It is invoked
    only when the list actually changes vs. the last broadcast.
    """
    with _lock:
        _subscribers.append(callback)


def unsubscribe(callback: Callable[[list[str]], None]) -> None:
    """Remove *callback* from the subscriber list (silent if absent).

    Used by :class:`RemoteAccessServer` on shutdown so a fresh server
    instance re-instantiated in the same process does not inherit a
    dead broadcaster bound to the old printer.
    """
    with _lock:
        try:
            _subscribers.remove(callback)
        except ValueError:
            pass


def _split_slash_command(prompt: str) -> tuple[str, str] | None:
    """Return ``(command, rest)`` when *prompt* starts with ``/xxx``.

    Recognises ``/xxx`` at the very first character of the prompt
    (no leading whitespace, so a stray newline before a slash is not
    treated as a command).  The command name accepts ASCII letters,
    digits, ``_`` and ``-`` — the same characters valid in a
    ``*_sea.py`` filename stem.  Everything after the first
    whitespace run is the sub-task text.

    Args:
        prompt: The raw user prompt.

    Returns:
        ``(command_name, remaining_text)`` when the prompt looks like
        a slash command, else ``None``.  ``remaining_text`` is left
        stripped so ``/xxx   hi`` yields ``("xxx", "hi")``, and is
        ``""`` when the prompt is just ``/xxx``.
    """
    if not prompt or not prompt.startswith("/"):
        return None
    end = 1
    while end < len(prompt):
        ch = prompt[end]
        if ch.isalnum() or ch in ("_", "-"):
            end += 1
            continue
        break
    if end == 1:
        return None
    command = prompt[1:end]
    rest = prompt[end:]
    # A slash command must be followed by whitespace or end of prompt,
    # otherwise ``/xxxfoo`` would be a command too.
    if rest and not rest[0].isspace():
        return None
    return command, rest.strip()


class SeaScriptError(SeaError, RuntimeError):
    """A SEA script failed to import, is misconfigured, or one of its getters raised.

    Raised by :func:`load_sea`, :func:`sea_layers`, :func:`evaluate_sea`
    and :func:`sea_getter_value` with the original raise as
    ``__cause__`` — ``BaseException`` included, so a SEA raising
    ``KeyboardInterrupt``/``SystemExit`` at import time is reported as
    a broken script, not as a cancelled task, while a genuinely
    requested stop that landed inside the import stays recognisable
    through the cause chain (``task_runner._stop_interrupt_wrapped``).
    """


def load_sea(sea_path: Path) -> dict[str, Any]:
    """Execute the SEA script at *sea_path* and return its namespace.

    :func:`kiss.agents.sorcar.sea_settings.execute_python_file` with
    this module's error class; see there for the loading rules.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Returns:
        The executed module's namespace dict.

    Raises:
        SeaScriptError: When the file cannot be read, compiled or
            executed (whatever it raises).
    """
    return execute_python_file(str(sea_path), SeaScriptError, "agent script")


def call_getter(namespace: Mapping[str, Any], label: str, name: str, *args: Any) -> Any:
    """Call the getter *name* of a loaded SEA, or return ``None`` when it defines none.

    Membership (not ``.get() is None``) decides absence: a DEFINED
    ``name = None`` is a broken getter, not a missing one.

    Args:
        namespace: The SEA's namespace (:func:`load_sea`).
        label: How to name the SEA in diagnostics (its path).
        name: The getter, e.g. ``"add_to_tools"``.
        args: Positional arguments of the getter (``prompt(task)``,
            ``on_picked_as_model(work_dir)``; the others take none).

    Returns:
        The getter's return value, or ``None`` when undefined.

    Raises:
        SeaScriptError: When the getter is not callable or raises
            (whatever it raises).
    """
    if name not in namespace:
        return None
    getter = namespace[name]
    if not callable(getter):
        raise SeaScriptError(
            f"{name} of agent script {label!r} must be a callable, got {type(getter).__name__}"
        )
    try:
        return getter(*args)
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        logger.warning("%s() of SEA %s raised", name, label, exc_info=True)
        raise SeaScriptError(
            f"{name}() of agent script {label!r} raised: {safe_message(exc)}"
        ) from exc


def sea_getter_value(sea_path: Path, getter: str, *args: Any) -> Any:
    """Return ``getter(*args)`` of the SEA at *sea_path*, or ``None`` when it defines none.

    Loads the script and calls :func:`call_getter`; for one getter of a
    script nothing else needs (``register_as_model()``).

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.
        getter: Name of the getter.
        args: Positional arguments of the getter.

    Returns:
        The getter's return value when the script defines a callable
        *getter*, else ``None``.

    Raises:
        SeaScriptError: When the script fails to import or *getter* is
            not callable or raises (whatever it raises).
    """
    return call_getter(load_sea(sea_path), str(sea_path), getter, *args)


@dataclass(frozen=True)
class SeaLayer:
    """One executed SEA of a run: its path, namespace and own resolved settings."""

    path: Path
    namespace: dict[str, Any]
    settings: dict[str, Any]


MAX_EXTENDS_DEPTH = 8
"""Longest ``extends`` chain a SEA may form; a longer one is reported as a cycle."""


def sea_layers(sea_path: Path, base: Path | None = None) -> list[SeaLayer]:
    """Execute the SEA at *sea_path* and every script it (transitively) extends.

    The daemon applies the layers in order, each on top of the previous
    (:func:`evaluate_sea`): settings of a later layer win, prompt and
    system-prompt additions concatenate, tool lists union.  A script
    names its base with ``settings()["extends"]``: a registered command
    name (``"bestrouter"``) or a path (``.py`` suffix or a separator;
    relative to the script's own directory).  *base* is an outermost
    layer the caller adds in front of the chain — the model-picker SEA
    of the tab a run was submitted from.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.
        base: Path of a SEA to lay under *sea_path*'s chain, or ``None``.

    Returns:
        The layers, outermost base first, *sea_path* last.  Every file
        is executed exactly once and appears once: a file both chains
        share (*base* itself, or a common ancestor) keeps its place in
        the base chain.

    Raises:
        SeaScriptError: When a script fails to import, has malformed
            settings, names an ``extends`` that is no registered
            command or existing ``.py`` file, forms a cycle (or a chain
            deeper than :data:`MAX_EXTENDS_DEPTH`), or when any layer
            but the last is of ``kind: "channel"``: a channel agent is
            a worker for an external service and cannot be extended.
    """
    loaded: dict[Path, SeaLayer] = {}
    layers = [] if base is None else _extends_chain(base, [], loaded)
    for layer in _extends_chain(sea_path, [], loaded):
        if not any(layer is seen for seen in layers):
            layers.append(layer)
    for layer in layers[:-1]:
        if layer.settings.get("kind") == "channel":
            raise SeaScriptError(
                f"agent script {str(layers[-1].path)!r}: cannot extend the channel agent "
                f"script {str(layer.path)!r}"
            )
    return layers


def _extends_chain(
    sea_path: Path, seen: list[Path], loaded: dict[Path, SeaLayer],
) -> list[SeaLayer]:
    """Return the layers of *sea_path*: its ``extends`` chain (recursively) then itself.

    *loaded* caches executed layers by resolved path across the chains
    of one :func:`sea_layers` call, so a shared file executes once.
    """
    try:
        key = sea_path.resolve()
    except (OSError, ValueError):
        key = sea_path  # e.g. an embedded NUL byte; load_sea reports it
    if key in seen or len(seen) >= MAX_EXTENDS_DEPTH:
        raise SeaScriptError(
            f"agent script {str(seen[0] if seen else sea_path)!r}: extends chain is a cycle: "
            + " -> ".join(str(p) for p in [*seen, sea_path])
        )
    layer = loaded.get(key)
    if layer is None:
        namespace = load_sea(sea_path)
        try:
            settings = resolve_settings(namespace)
        except SettingsError as exc:
            raise SeaScriptError(f"agent script {str(sea_path)!r}: {exc}") from exc
        _anchor_work_dir(settings, sea_path.parent)
        layer = loaded[key] = SeaLayer(sea_path, namespace, settings)
    extends = str(layer.settings.get("extends") or "")
    if not extends:
        return [layer]
    return [*_extends_chain(_resolve_extends(sea_path, extends), [*seen, key], loaded), layer]


def _anchor_work_dir(settings: dict[str, Any], script_dir: Path) -> None:
    """Make a ``work_dir`` setting absolute: ``~`` expanded, a relative path under the script.

    A script says ``"work_dir": "sandbox"`` to mean the ``sandbox``
    folder next to itself; resolved here, once, every way of running
    the script (``/command``, ``run_agent``, ``run_parallel``) works
    in that same folder instead of one relative to whatever the caller
    or the daemon happened to run in.
    """
    work_dir = settings.get("work_dir")
    if not isinstance(work_dir, str) or not work_dir:
        return
    path = Path(work_dir).expanduser()
    settings["work_dir"] = os.path.normpath(path if path.is_absolute() else script_dir / path)


def _resolve_extends(sea_path: Path, spec: str) -> Path:
    """Return the script a SEA's ``extends`` names (a command name or a path)."""
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        candidate = Path(spec).expanduser()
        if not candidate.is_absolute():
            candidate = sea_path.parent / candidate
        if candidate.suffix == ".py" and candidate.is_file():
            return candidate.resolve()
        raise SeaScriptError(
            f"agent script {str(sea_path)!r}: extends {spec!r} is not an existing Python (.py) file"
        )
    base = get_command(spec)
    if base is None:
        raise SeaScriptError(
            f"agent script {str(sea_path)!r}: extends {spec!r} is not a registered SEA command; "
            f"known commands: {', '.join(list_commands()) or 'none'}"
        )
    return base


def sea_settings(sea_path: Path) -> dict[str, Any]:
    """Return the effective ``settings()`` of the SEA at *sea_path*, bases included.

    :func:`sea_layers` merged with
    :func:`kiss.agents.sorcar.sea_settings.merge_settings`.  The
    dispatcher reads the ``kind``, ``timeout``, ``work_dir`` and
    ``model`` from it; the task runner the ``model`` and ``work_dir``.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Returns:
        The settings dict, at least ``{"kind": "session"}``.

    Raises:
        SeaScriptError: When a script of the chain fails to import or
            has malformed settings, so the caller fails with the
            diagnostic instead of running against a broken SEA.
    """
    return merge_settings([layer.settings for layer in sea_layers(sea_path)])


@dataclass
class SeaRun:
    """What a run takes from its SEA layers (:func:`evaluate_sea`).

    Attributes:
        settings: The merged settings (:func:`merge_settings`).
        prompt: The task text after every layer's ``prompt(task)``,
            ``{task_id}`` replaced by the calling task's id.
        system_prompt: The innermost ``system_prompt()`` text (the base
            system prompt), or ``None`` when no layer defines one.
        add_to_system_prompt: The layers' ``add_to_system_prompt()``
            texts, joined; empty when none.
        tools: The union of the layers' ``add_to_tools()`` lists; a
            later layer's tool replaces an earlier one of the same name.
        llm_call_hook: The innermost ``llm_call_hook()`` value, or ``None``.
        tool_call_hook: The innermost ``tool_call_hook()`` value, or ``None``.
    """

    settings: dict[str, Any]
    prompt: str
    system_prompt: str | None = None
    add_to_system_prompt: str = ""
    tools: list[Callable[..., Any]] = field(default_factory=list)
    llm_call_hook: Callable[..., Any] | None = None
    tool_call_hook: Callable[..., Any] | None = None


def evaluate_sea(layers: list[SeaLayer], task: str, task_id: str = "") -> SeaRun:
    """Evaluate the getters of *layers* for a run on *task*.

    Each layer is applied on top of the previous ones: ``prompt(task)``
    chains (a layer receives what the layer below returned),
    ``add_to_system_prompt()`` texts concatenate, ``add_to_tools()``
    lists union by tool name, and the innermost (last) layer that
    defines ``system_prompt()`` or a hook wins.

    Args:
        layers: The layers of :func:`sea_layers`.
        task: The task text the run was submitted with.
        task_id: The calling task's id, substituted for ``{task_id}``
            in what ``prompt(task)`` returns (empty when there is none).

    Returns:
        The run's configuration.

    Raises:
        SeaScriptError: When a getter is not callable, raises, or
            returns a value of the wrong type (``prompt()`` must return
            a non-empty string).
    """
    run = sea_configuration(layers)
    run.prompt = sea_prompt(layers, task, task_id)
    return run


def sea_configuration(layers: list[SeaLayer]) -> SeaRun:
    """Evaluate every getter of *layers* but ``prompt(task)``.

    :func:`evaluate_sea` without the task: the returned run's ``prompt``
    is ``""``.  For callers that configure a run once and shape many
    tasks with :func:`sea_prompt` afterwards (``skillopt`` rollouts),
    so a task-dependent ``prompt(task)`` never sees a placeholder.

    Args:
        layers: The layers of :func:`sea_layers`.

    Raises:
        SeaScriptError: As for :func:`evaluate_sea`.
    """
    run = SeaRun(merge_settings([layer.settings for layer in layers]), "")
    tools_by_name: dict[str, Callable[..., Any]] = {}
    for layer in layers:
        label = str(layer.path)
        namespace = layer.namespace
        if "system_prompt" in namespace:
            run.system_prompt = _check_text(
                label, "system_prompt", call_getter(namespace, label, "system_prompt"),
            )
        if "add_to_system_prompt" in namespace:
            run.add_to_system_prompt = join_text(
                run.add_to_system_prompt,
                _check_text(
                    label, "add_to_system_prompt",
                    call_getter(namespace, label, "add_to_system_prompt"),
                ),
            )
        if "add_to_tools" in namespace:
            for tool in _check_tools(label, call_getter(namespace, label, "add_to_tools")):
                tools_by_name[getattr(tool, "__name__", repr(tool))] = tool
        for name in ("llm_call_hook", "tool_call_hook"):
            if name in namespace:
                setattr(run, name, _check_hook(label, name, call_getter(namespace, label, name)))
    run.tools = list(tools_by_name.values())
    return run


def sea_prompt(layers: list[SeaLayer], task: str, task_id: str = "") -> str:
    """Return *task* after every layer's ``prompt(task)``, outermost first.

    The prompt half of :func:`evaluate_sea`, for callers that shape
    many tasks with one executed chain (``skillopt`` rollouts).  A
    layer that defines ``prompt`` has ``{task_id}`` in its result
    replaced by *task_id* (the calling task's id, or ``""`` when there
    is none); a task text shaped by no ``prompt`` getter is returned as
    submitted.

    Raises:
        SeaScriptError: When a ``prompt`` getter is not callable, raises,
            or returns anything but a non-empty string.
    """
    for layer in layers:
        if "prompt" not in layer.namespace:
            continue
        label = str(layer.path)
        task = _check_text(label, "prompt", call_getter(layer.namespace, label, "prompt", task))
        task = task.replace("{task_id}", task_id)
        if not task.strip():
            raise SeaScriptError(
                f"prompt() of agent script {label!r} must return a non-empty string"
            )
    return task


def join_text(base: str, addition: str) -> str:
    """Return *addition* appended to *base* with a blank line between; either may be empty."""
    if not base:
        return addition
    return f"{base}\n\n{addition}" if addition else base


def _check_text(label: str, name: str, value: Any) -> str:
    """Return *value* as a plain string, or raise :exc:`SeaScriptError`.

    A ``str`` subclass from an untrusted script is copied into an exact
    ``str`` (its own methods may raise), so later ``strip()``/joins
    cannot run script code.
    """
    if not isinstance(value, str):
        raise SeaScriptError(
            f"{name}() of agent script {label!r} must return a string, "
            f"got {type(value).__name__}"
        )
    try:
        return str(value)
    except BaseException as exc:  # noqa: BLE001 — an untrusted str subclass may raise
        raise SeaScriptError(
            f"{name}() of agent script {label!r} returned a broken value: {safe_message(exc)}"
        ) from exc


def _check_tools(label: str, value: Any) -> list[Callable[..., Any]]:
    """Return *value* as a list of tool callables, or raise :exc:`SeaScriptError`."""
    try:
        if isinstance(value, list | tuple) and all(callable(tool) for tool in value):
            return list(value)
    except BaseException as exc:  # noqa: BLE001 — an untrusted list may raise while iterated
        raise SeaScriptError(
            f"add_to_tools() of agent script {label!r} returned a broken list: {safe_message(exc)}"
        ) from exc
    raise SeaScriptError(
        f"add_to_tools() of agent script {label!r} must return a list of tool callables "
        f"(not a file path), got {type(value).__name__}"
    )


def _check_hook(label: str, name: str, value: Any) -> Any:
    """Return *value* when it is a callable or ``None``, or raise :exc:`SeaScriptError`."""
    if value is None or callable(value):
        return value
    raise SeaScriptError(
        f"{name}() of agent script {label!r} must return a callable or None, "
        f"got {type(value).__name__}"
    )


PICKED_HOOK_TIMEOUT_SECONDS = 15.0
"""How long :func:`run_picked_hook` waits for ``on_picked_as_model()`` before moving on."""


def _log_picked_hook(model: str, work_dir: str, namespace: Mapping[str, Any] | None) -> None:
    """Run ``on_picked_as_model(work_dir)`` of the SEA picked as *model*; log note or failure."""
    sea_path = model_sea(model)
    if sea_path is None:
        return
    try:
        if namespace is None:
            namespace = load_sea(sea_path)
        note = call_getter(namespace, str(sea_path), "on_picked_as_model", work_dir)
    except SeaScriptError:
        logger.warning("SEA %s: on_picked_as_model() failed", model, exc_info=True)
        return
    if note:
        logger.info("SEA %s picked as model: %s", model, note)


def run_picked_hook(
    model: str,
    work_dir: str,
    wait: bool = True,
    namespace: Mapping[str, Any] | None = None,
) -> None:
    """Run the picked SEA's ``on_picked_as_model(work_dir)`` hook, when it defines one, on a thread.

    The hook is a model-picker SEA's chance to act on being chosen as a
    tab's model — ``autorouter`` makes sure its weekly ``/rsi7d autorouter``
    cron job is scheduled.  The daemon calls it when the user picks the SEA
    (``selectModel``) and once per run whose model is the SEA.  Hooks
    from concurrent picks may overlap: a hook that must not race gets
    its atomicity from what it calls (``cron_job("ensure")`` for
    autorouter).  Whatever the hook returns is logged; a hook that
    raises is logged and never fails the caller, one that blocks is
    abandoned on its daemon thread after
    :data:`PICKED_HOOK_TIMEOUT_SECONDS`, and a thread that cannot start
    is logged too, because the task the user typed does not depend on it.

    Args:
        model: The picked model name, a SEA command name
            (:func:`model_sea` maps it to the file; a real model name runs nothing).
        work_dir: The tab's or run's work directory, passed to the hook.
        wait: Block up to :data:`PICKED_HOOK_TIMEOUT_SECONDS` for the hook
            (a run wants its side effects in place before it starts);
            ``False`` returns at once (the picker handler must not stall
            the daemon's command loop).
        namespace: The SEA's already-executed namespace (a run executes
            the script once and shares it); ``None`` loads the script.
    """
    thread = threading.Thread(
        target=_log_picked_hook, args=(model, work_dir, namespace),
        name="sea-picked-hook", daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:  # the process is out of threads
        logger.warning("SEA %s: on_picked_as_model() not run", model, exc_info=True)
        return
    if not wait:
        return
    thread.join(PICKED_HOOK_TIMEOUT_SECONDS)
    if thread.is_alive():
        logger.warning(
            "SEA %s: on_picked_as_model() still running after %.0f s; going on without it",
            model, PICKED_HOOK_TIMEOUT_SECONDS,
        )


def _registers_as_model(sea_path: Path) -> bool:
    """Return whether the SEA at *sea_path* defines ``register_as_model()`` returning ``True``.

    The verdict is cached per file stamp (mtime, size, inode), so both an
    edit and an atomic replacement are re-read.  Only a script whose
    source mentions ``register_as_model`` is imported: importing every
    registered SEA (channel agents with heavy dependencies among them)
    on each picker refresh would be slow for nothing.  A script that
    cannot be read, fails to import or whose getter raises is logged and
    treated as not registered, so one broken SEA cannot break the model
    picker.
    """
    try:
        st = sea_path.stat()
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        with _lock:
            cached = _model_sea_cache.get(sea_path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        source = sea_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("SEA %s: cannot be read", sea_path, exc_info=True)
        return False
    verdict = False
    if "register_as_model" in source:
        try:
            verdict = sea_getter_value(sea_path, "register_as_model") is True
        except SeaScriptError:
            logger.warning("SEA %s: register_as_model() failed", sea_path, exc_info=True)
    with _lock:
        _model_sea_cache[sea_path] = (stamp, verdict)
    return verdict


def model_seas() -> dict[str, Path]:
    """Return the registered SEAs that are model-picker entries, by command name.

    A SEA is a model-picker entry when its script defines
    ``register_as_model()`` returning ``True``.  The daemon offers each
    entry in the model picker under its command name; a task run with
    such a pick goes through the SEA (``task_runner._resolve_sea_model``).

    Returns:
        ``{command_name: absolute_sea_path}``, sorted by name.
    """
    list_commands()  # populate the registry on a cold start
    with _lock:
        entries = sorted(_registry.items())
    return {name: path for name, path in entries if _registers_as_model(path)}


def model_sea(name: str) -> Path | None:
    """Return the script of the model-picker SEA *name*, or ``None``.

    A *name* that is no registered command (every real model name)
    costs one :func:`get_command` lookup — a folder rescan on the miss,
    as for an unknown slash command — and touches no script, so the
    task runner can ask this for every run's model and a SEA installed
    after the registry was built is still found.

    Args:
        name: The model-picker value, e.g. ``"autorouter"`` or
            ``"gpt-6-astra"``.

    Returns:
        The absolute SEA path when *name* is a registered command whose
        ``register_as_model()`` returns ``True``, else ``None``.
    """
    path = get_command(name) if name else None
    if path is None or not _registers_as_model(path):
        return None
    return path


def sea_description(sea_path: Path) -> str:
    """Return the ``description()`` text of the SEA at *sea_path*.

    Every SEA must define a zero-argument ``description()`` returning
    one sentence that says what the SEA does and how to use it; this
    is the text ``/xxx help`` shows the user.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Returns:
        The stripped, non-empty string ``description()`` returned.

    Raises:
        SeaScriptError: When the script fails to import, does not define
            a callable ``description``, ``description()`` raises, or it
            returns anything but a non-empty string.
    """
    namespace = load_sea(sea_path)
    if "description" not in namespace:
        raise SeaScriptError(
            f"agent script {str(sea_path)!r}: description() must be a "
            "zero-argument function returning a string"
        )
    text = call_getter(namespace, str(sea_path), "description")
    if not isinstance(text, str) or not text.strip():
        raise SeaScriptError(
            f"agent script {str(sea_path)!r}: description() must return a non-empty "
            f"string, got {type(text).__name__}"
        )
    return text.strip()


CHECK_SAMPLE_TASK = "<the task text>"
"""The task :func:`sea_check` hands ``prompt(task)``; ``{task_id}`` becomes ``<task id>``."""

RESERVED_SUBCOMMANDS = ("help", "check")
"""The ``/xxx <word>`` prompts the daemon answers itself instead of running the SEA."""


def check_sea(
    sea_path: Path, require_description: bool = True,
) -> tuple[list[SeaLayer], dict[str, Any], str]:
    """Load the SEA at *sea_path* the way a run would and evaluate every getter.

    The daemon's own path (``load_layers`` + ``apply_agent_overrides``
    on a ``run`` command for :data:`CHECK_SAMPLE_TASK`), so a script
    that reads the run command from those frames works, plus the
    ``description()`` and model-picker checks a run does not make.

    Args:
        sea_path: The SEA's script.
        require_description: Whether a missing ``description()`` is an
            error (it is for a ``/command``; a script only loaded by
            path or as an ``extends`` base needs none).

    Returns:
        ``(layers, cmd, description)``: the loaded layers, the ``run``
        command with the script's overrides applied, and the
        description text (``""`` when not required and absent).

    Raises:
        SeaError: The script or one of its getters breaks the contract,
            in the words the daemon would use.
    """
    # Imported here: ``agent_file`` imports this module.
    from kiss.agents.sorcar.agent_file import apply_agent_overrides, load_layers

    cmd: dict[str, Any] = {
        "agentPath": str(sea_path), "prompt": CHECK_SAMPLE_TASK, "parentTaskId": "<task id>",
    }
    layers = load_layers(cmd)
    apply_agent_overrides(cmd, layers)
    innermost = layers[-1].namespace
    description = (
        sea_description(sea_path) if require_description or "description" in innermost else ""
    )
    label = str(layers[-1].path)
    registers = call_getter(innermost, label, "register_as_model")
    if "register_as_model" in innermost and not isinstance(registers, bool):
        raise SeaScriptError(
            f"register_as_model() of agent script {label!r} must return a bool, "
            f"got {type(registers).__name__}"
        )
    if "on_picked_as_model" in innermost:
        _check_picked_hook(innermost["on_picked_as_model"], label)
    return layers, cmd, description


def sea_check(name: str, sea_path: Path) -> str:
    """Execute the SEA *name* at *sea_path* and report what a run of it would use.

    What ``/xxx check`` shows, so a script author sees the effect of
    ``settings()`` and the getters without running a task: the layers
    (the script and what it ``extends``), the effective merged
    settings, the model a run takes, the names of the tools
    ``add_to_tools()`` adds, which optional getters and hooks are
    defined, and the prompt ``prompt(task)`` yields for
    :data:`CHECK_SAMPLE_TASK`.  A broken script yields the first
    error instead, in the words the daemon would use.

    Args:
        name: The command name.
        sea_path: The SEA's script.

    Returns:
        The report, one item per line.
    """
    try:
        layers, cmd, description = check_sea(sea_path)
    except SeaError as exc:
        return f"/{name} is broken: {exc}"
    merged = merge_settings([layer.settings for layer in layers])
    settings = {key: value for key, value in merged.items() if key != "kind"}
    model = settings.get("model") or "the calling task's model (else the default model)"
    tools = cmd.get("tools") or []
    defined = [
        getter for getter in (
            "system_prompt", "add_to_system_prompt", "prompt", "llm_call_hook",
            "tool_call_hook", "register_as_model", "on_picked_as_model",
        ) if any(getter in layer.namespace for layer in layers)
    ]
    lines = [
        f"/{name}: {description}",
        "layers: " + " > ".join(script_name(str(layer.path)) for layer in layers),
        f"kind: {merged['kind']}",
        "settings: " + (json.dumps(settings, sort_keys=True, default=str) or "{}"),
        f"model: {model}",
        "tools added: " + (", ".join(_tool_name(tool) for tool in tools) or "none"),
        "getters and hooks defined: " + (", ".join(defined) or "none"),
        f"prompt for {CHECK_SAMPLE_TASK}: {cmd['prompt']}",
    ]
    if "description" not in layers[-1].namespace:
        lines.append("note: description() comes from an extended script")
    return "\n".join(lines)


def _check_picked_hook(picked: Any, label: str) -> None:
    """Raise :exc:`SeaScriptError` unless *picked* is callable as ``on_picked_as_model(work_dir)``.

    The check :func:`run_picked_hook` would otherwise fail at pick time:
    the hook must be callable and must bind one positional argument.
    """
    if not callable(picked):
        raise SeaScriptError(
            f"on_picked_as_model of agent script {label!r} must be a callable, "
            f"got {type(picked).__name__}"
        )
    try:
        signature = inspect.signature(picked)
    except ValueError:
        return  # a builtin without introspectable signature: callable is all we can check
    try:
        signature.bind("<work_dir>")
    except TypeError as exc:
        raise SeaScriptError(
            f"on_picked_as_model of agent script {label!r} must take one positional "
            f"argument (work_dir): {exc}"
        ) from exc


def _tool_name(tool: Any) -> str:
    """Return the name a tool callable is registered under (its ``__name__`` or its type's)."""
    return str(getattr(tool, "__name__", None) or type(tool).__name__)


def help_text_if_command(prompt: str) -> str | None:
    """Return the daemon's own answer when *prompt* is ``/xxx help`` or ``/xxx check``.

    ``help`` and ``check`` (case-insensitive, nothing after them) are
    the sub-tasks every command reserves (:data:`RESERVED_SUBCOMMANDS`):
    instead of relaying them to the SEA, the daemon answers ``help``
    with the return value of the SEA's ``description()`` and ``check``
    with :func:`sea_check`'s report.

    Args:
        prompt: The raw user prompt (as submitted by the client).

    Returns:
        The description or the check report when *prompt* is
        ``/xxx help`` / ``/xxx check`` for a registered command
        ``xxx``, else ``None``.

    Raises:
        SeaScriptError: Propagated from :func:`sea_description` when the
            SEA is broken or lacks ``description()`` (``help`` only;
            ``check`` reports the error as its text).
    """
    if not isinstance(prompt, str):
        return None
    parsed = _split_slash_command(prompt)
    if parsed is None or parsed[1].lower() not in RESERVED_SUBCOMMANDS:
        return None
    sea_path = get_command(parsed[0])
    if sea_path is None:
        return None
    if parsed[1].lower() == "check":
        return sea_check(parsed[0], sea_path)
    return sea_description(sea_path)


def slash_command_task(prompt: str) -> tuple[str, Path] | None:
    """Split a ``/xxx text`` prompt into the SEA to run and its task.

    The daemon runs the SEA directly on the trailing text — the same
    run ``run_agent(agent="xxx", task=text)`` makes, with the SEA's
    ``settings()`` and getters applied by ``apply_agent_overrides``.

    Args:
        prompt: The raw user prompt (as submitted by the client).

    Returns:
        ``(task_text, sea_path)`` when *prompt* starts with a registered
        command followed by non-empty text other than ``help`` or
        ``check`` (which :func:`help_text_if_command` answers); else
        ``None`` — the prompt runs as an ordinary task.
    """
    if not isinstance(prompt, str):
        return None
    parsed = _split_slash_command(prompt)
    if parsed is None:
        return None
    command, task_text = parsed
    if not task_text or task_text.lower() in RESERVED_SUBCOMMANDS:
        return None
    sea_path = get_command(command)
    if sea_path is None:
        return None
    return task_text, sea_path

def start_registry_watcher(
    poll_interval: float = _WATCHER_POLL_SECONDS,
) -> None:
    """Start the background poller that keeps the registry fresh.

    Idempotent — a second call while the watcher is alive is a
    no-op — and safe to call from any thread.  The watcher runs as
    a daemon thread so it never blocks interpreter shutdown.

    Args:
        poll_interval: Seconds between rescans.  Bounded below at
            0.1 s to keep tests fast and above at 60 s to keep it
            useful.
    """
    global _watcher_thread, _watcher_stop
    interval = max(0.1, min(60.0, float(poll_interval)))
    # Always emit the first refresh synchronously so the very first
    # ``list_commands`` from a client cannot race the watcher's first
    # tick, then hand ongoing rescans to the poller.
    try:
        refresh_registry()
    except Exception:  # pragma: no cover - registry rescan is best-effort
        logger.debug("initial SEA registry refresh failed", exc_info=True)
    # Create, start and publish the thread in one critical section so
    # ``stop_registry_watcher`` can never join an unstarted thread and a
    # concurrent second start cannot spawn a second poller.
    with _lock:
        if _watcher_thread is not None and _watcher_thread.is_alive():
            return
        stop = threading.Event()
        thread = threading.Thread(
            target=_watcher_loop,
            args=(interval, stop),
            name="kiss-sea-registry-watcher",
            daemon=True,
        )
        thread.start()
        _watcher_thread, _watcher_stop = thread, stop


def stop_registry_watcher(timeout: float = 5.0) -> None:
    """Stop the background poller (used by the daemon on shutdown).

    Signals the watcher to exit and joins it within *timeout* seconds;
    a poller that outlives the timeout is left as a daemon thread
    (daemon threads die with the interpreter).  Idempotent.
    """
    global _watcher_thread, _watcher_stop
    with _lock:
        thread, stop = _watcher_thread, _watcher_stop
        _watcher_thread = _watcher_stop = None
        if thread is None or stop is None:
            return
        stop.set()
    thread.join(timeout=timeout)


def _watcher_loop(interval: float, stop: threading.Event) -> None:
    """Body of the background poller thread.

    Rebuilds the registry every *interval* seconds until *stop* (this
    poller's own event) is signalled.  Every rescan is wrapped in
    ``try/except`` so a transient filesystem error (e.g. a folder
    that appears mid-edit) cannot kill the watcher.
    """
    while not stop.wait(interval):
        try:
            refresh_registry()
        except Exception:  # pragma: no cover - watcher must never die
            logger.debug("SEA registry rescan failed", exc_info=True)


def _reset_for_tests() -> None:
    """Drop all in-memory state (registry, subscribers, watcher).

    Test-only helper: pytest fixtures use this to isolate SEA-command
    behaviour across tests without leaking a subscriber that fires on
    a later, unrelated rescan.
    """
    global _last_broadcast
    stop_registry_watcher()
    with _lock:
        _registry.clear()
        _subscribers.clear()
        _model_sea_cache.clear()
        _last_broadcast = ()
