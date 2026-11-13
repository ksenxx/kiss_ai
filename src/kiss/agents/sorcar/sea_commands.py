# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slash-command registry for Sorcar Extension Agents (SEAs), and their launcher.

A SEA named ``xxx`` is a folder ``xxx/`` that contains the file
``xxx_sea.py`` — which defines one subclass of
:class:`kiss.agents.seas.base.base_sea.BaseSea` — plus whatever helper
modules and data files the SEA needs.  Every such folder visible to
the daemon is exposed as a chat command named after the folder:
``/xxx``.  When a user submits a prompt that starts with ``/xxx`` —
optionally followed by whitespace and free-form text — the daemon runs
the SEA directly on the trailing text (:func:`slash_command_task`):
the same run ``run_agent(agent="xxx", task=text)`` makes.  Two special
prompts do not run the SEA: ``/xxx help`` answers with the SEA's
``description()``, and ``/xxx check`` loads the SEA and reports its
effective settings, model, tools and sample prompt — or the first
error (see :func:`help_text_if_command`, :func:`sea_check`).

This module is also the launcher: :func:`load_sea` executes a SEA file
(with the one loader
:func:`kiss.agents.sorcar.sea_settings.execute_python_file`) and
instantiates its class; the ``base_*`` functions (:func:`base_settings`,
:func:`base_prompt`, :func:`base_system_prompt`, :func:`base_tools`,
:func:`base_tool_call_hook`, :func:`base_llm_call_hook`,
:func:`base_llm_response_hook`) apply the
method of every class of the SEA's inheritance chain, base class first
— so a SEA extends another by deriving from its class, and never calls
``super()``; :func:`evaluate_sea` bundles them into a run's
configuration (a :class:`SeaRun`); :func:`sea_settings` gives the
dispatcher the effective settings.

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
4. The built-in SEAs of :data:`BUILTIN_COMMANDS` (``/cron``,
   lowest precedence).

The registry is refreshed lazily on every lookup and, in the daemon,
proactively by a background polling watcher (see
:func:`start_registry_watcher`) so edits to ``SEAS.md`` — or the
appearance/removal of SEA folders in any of its folders — take
effect while the daemon is running.

A registered SEA whose ``register_as_model()`` returns ``True`` is
also a *model-picker entry*: :func:`model_seas` lists such SEAs under
their command names and the daemon offers them in the model picker
next to the real models.  Picking one lays the SEA under every task of
the tab (on the model its ``model`` setting names, else the default
model), so a ``/xxx`` command or a ``run_agent`` child run on that tab
keeps its routing protocol (``system_prompt()``) and runs its own SEA
on top (see :func:`sea_layers` and :mod:`kiss.agents.sorcar.sea_apply`).

Locking: two module locks, always acquired in the order
``_notify_lock`` -> ``_lock``.  ``_lock`` guards the registry and the
subscriber list; ``_notify_lock`` wraps the delivery of a changed
snapshot to the subscribers (:func:`_publish`) so they see snapshots
in the order they were published.  Nothing calls
:func:`refresh_registry` while holding ``_lock``.  The folders are
read under no lock; a rescan that started before the last published
one is dropped instead of published.
"""

from __future__ import annotations

import dis
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import itertools
import json
import logging
import operator
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from kiss.agents.seas.base.base_sea import BaseSea, ChannelSea
from kiss.agents.sorcar.sea_settings import (
    _MID_IMPORT,
    SEA_BASES,
    SOURCE_REVISION,
    SeaError,
    declares_hidden,
    execute_python_file,
    expand_user,
    file_revision,
    file_stamp,
    imported_revision,
    is_stop,
    parsed_sea,
    refresh_imported_module,
    resolve_settings,
    safe_message,
    script_name,
)
from kiss.core.config import kiss_home
from kiss.core.llm_reply import LLMReply
from kiss.core.locked import locked, locked_call
from kiss.core.tool_verdict import ALLOW, Verdict

logger = logging.getLogger("kiss.sea_commands")

# The filename suffix that identifies a SEA file.
_SEA_SUFFIX = "_sea.py"
BASE_FOLDER = "base"
"""The folder of ``base_sea.py`` (:class:`BaseSea` itself, no SEA): never a command."""

# Command names must be ASCII letters, digits, ``_`` or ``-``.  A file
# whose stem breaks this rule (``foo.bar_sea.py``, ``spaced name_sea.py``)
# is silently ignored: the frontend autocomplete recognises the same
# character class, and _split_slash_command / :func:`get_command` would
# refuse it downstream anyway.
_COMMAND_NAME_CHARS = r"[A-Za-z0-9_-]+"
_COMMAND_NAME_RE = re.compile(rf"^{_COMMAND_NAME_CHARS}\Z")  # ``$`` would pass "name\n"
# A prompt that invokes a command: ``/<name>`` at the very first character,
# then whitespace (or the end) and the sub-task text.
_SLASH_RE = re.compile(rf"/({_COMMAND_NAME_CHARS})(?:\s+(.*))?\Z", re.S)
# A text that names a command: ``/<name>`` at its start or after whitespace,
# an opening bracket, a quote or Markdown emphasis, and not a path, a URL or
# a Markdown link target (``~/x``, ``a/x``, ``/x/``, ``/x.py``, ``/x?q=1``,
# ``/x#a``, ``[t](/x)``).
_MENTION_RE = re.compile(
    rf"(?<![^\s(\[\"'`*\u201c\u2018])(?<!\]\()/({_COMMAND_NAME_CHARS})(?![\w/-]|[.?#]\w)"
)

# Poll interval for the background watcher, in seconds.
_WATCHER_POLL_SECONDS = 2.0

# In-memory registry: ``{command_name: absolute_path_to_sea_file}``.
_registry: dict[str, Path] = {}

# Guards ``_registry`` and ``_subscribers`` (RLock so a subscriber
# callback that itself calls :func:`list_commands` cannot deadlock).
_lock = threading.RLock()

# Serialises the delivery of a changed snapshot to the subscribers
# (:func:`_notify`) so they receive snapshots in the order they were
# published: without it two concurrent rescans could deliver snapshot
# B before A and leave every subscriber on the stale list A for good
# (the ``_last_broadcast`` dedupe then suppresses the correcting
# rescan).  Taken only by a rescan with something to deliver: one whose
# snapshot is the last broadcast (a task thread's ``get_command`` miss,
# mostly) returns at once instead of waiting out a slow subscriber (the
# daemon's reads history.db, which a write-locked database stalls for
# SQLite's 30 s busy timeout).
#
# LOCK ORDER: ``_notify_lock`` -> ``_lock``.  ``_notify_lock`` is held
# through the callback loop; ``_lock`` is only ever taken inside it or
# on its own.  No function may call :func:`refresh_registry` while
# holding ``_lock``.  RLock so a subscriber that re-enters
# :func:`refresh_registry` (e.g. through :func:`get_command` on a miss)
# does not self-deadlock.
_notify_lock = threading.RLock()

# Callbacks invoked (under ``_notify_lock`` but outside ``_lock``)
# whenever the registry changes.
_subscribers: list[Callable[[list[str]], None]] = []

# Last snapshot broadcast to subscribers.  Deduplicates identical
# rescans so the daemon does not spam clients with unchanged lists.
# ``None`` once a Stop interrupted a broadcast before every subscriber
# had it (:func:`_publish`): the next rescan broadcasts whatever it
# finds, an empty registry included.
_last_broadcast: tuple[str, ...] | None = ()

# Numbers the rescans in the order they start (``next`` of the C counter
# is atomic), and the number of the last rescan published (guarded by
# ``_lock``): a rescan that started before the published one is stale
# and is dropped, so the registry never goes back to an older scan.
_scan_numbers = itertools.count(1)
_published_scan = 0

# What the ``register_as_model()`` verdict cache watches of a
# dependency of a SEA: the (mtime_ns, size, inode) of a file, the
# script a ``sea_class("name")`` command name resolves to, or whether
# a module binds a name (``(module, name)``) a failed load missed.
_Watched = Path | str | tuple[str, str]
_Reading = tuple[int, int, int] | Path | bool | None

# ``register_as_model()`` verdict per SEA script, keyed by path and
# stamped with the (mtime_ns, size, inode) of the script and of every
# file its class derives from (:func:`_base_stamps`), so an edit or a
# replacement of any of them is re-read, and with the script every
# command name its load resolved to (:func:`_reading`).  Guarded by
# ``_lock``.
_model_sea_cache: dict[Path, tuple[dict[_Watched, _Reading], bool]] = {}

# ``bases``: while :func:`_registers_as_model` loads a SEA on this
# thread, the dict every :func:`sea_class` call of that load records
# what it read in before running anything (``None`` outside such a
# load): the stamp of the file it was asked for, loaded or not, and
# the script its command name resolved to.  A load that fails in a
# base is stamped with that base too, so its repair (the child
# untouched) is re-read.  ``imported``: likewise the file of every
# module the load imports the ordinary way (:class:`_PickerImports`),
# stamped before it runs; read when the load fails, so a parent that
# breaks the child's first ``from parent import Parent`` is re-read
# once repaired (a load that succeeds is stamped from its class's
# bases instead, :func:`_base_stamps`).  A module no finder finds is
# recorded under its name: no file names it, so a load that fails
# after looking for one is not cached.
_picker_load = threading.local()


class _PickerImports(importlib.abc.MetaPathFinder):
    """Records, on ``sys.meta_path``, the files a model-picker load imports the ordinary way.

    Asked for every module not yet in ``sys.modules``.  Outside a
    picker load it finds nothing (the regular finders follow).  During
    the picker load running on this thread (``_picker_load``) it asks
    the finders that follow it, in their order, and answers with the
    first spec one of them gives: what the import would have found,
    each finder asked once (one that answers a name a single time
    still serves it).  The file of that spec is noted in the load's
    ``imported`` map, with its revision as it is before the module
    runs; a name none of them resolves is noted as such.

    Such a name is passed on (``None``), so the import asks those
    finders a second time before it gives up.  The search cannot be
    ended here instead: ``importlib.util.find_spec()`` answers
    ``None`` for a module that is not there, and code that probes for
    an optional module that way must load in the picker as it does
    anywhere else.
    """

    def find_spec(
        self, fullname: str, path: Any, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Give the spec the finders after this one find for *fullname* on *path*, noting
        its ``.py`` file, or note that none of them finds the module.

        Raises:
            ImportError: The one a later finder refuses the name with.
        """
        imported = getattr(_picker_load, "imported", None)
        if imported is None:
            return None
        # Every later finder, not the path finder alone: that one knows no built-in
        # module and none that a library's own finder serves, so the first import of
        # one in the process (``traceback`` imports ``_suggestions`` to format the first
        # ``ImportError``) was noted as a module not found, and the failed load was not
        # cached.
        finders = list(sys.meta_path)
        spec = None
        for finder in finders[finders.index(self) + 1 :] if self in finders else finders:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except ImportError:  # refused: as absent as a module nobody finds
                imported[fullname] = None
                raise
            if spec is not None:
                break
        if spec is None:
            imported[fullname] = None
        elif spec.origin is not None and spec.origin.endswith(".py"):
            # What the verdict cache watches of a file (:func:`_reading`:
            # an ``access`` and a ``stat``), not :func:`parsed_sea`: a
            # load's first ``import`` of a library would otherwise
            # parse and memoise every module of it.
            file = Path(spec.origin)
            imported[file] = _reading(file)
        return spec


sys.meta_path.insert(0, _PickerImports())

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
        only names outside ``[A-Za-z0-9_-]``, the :data:`BASE_FOLDER`
        and scripts whose ``settings`` declare ``"hidden": True``
        (:func:`~kiss.agents.sorcar.sea_settings.declares_hidden`) are
        skipped.
    """
    out: dict[str, Path] = {}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return out
    for sea_dir in entries:
        command = sea_dir.name
        # Reject folder names that would produce a command name outside
        # the ``[A-Za-z0-9_-]`` alphabet the parser and the autocomplete
        # accept — ``foo.bar/`` and ``space name/`` are silently skipped
        # rather than surfacing as commands that cannot be typed.
        if not _COMMAND_NAME_RE.match(command) or command == BASE_FOLDER:
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
"""SEAs that are modules of the framework, by command name.

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


def _scan_sources(user_folders: Iterable[Path]) -> dict[str, Path]:
    """Return ``{command_name: script_path}`` merged from every source, in registry precedence.

    The built-in scripts (:data:`BUILTIN_COMMANDS`), the bundled
    ``seas/`` folder, *user_folders* (the ``SEAS.md`` folders, top line
    first) and the bundled ``third_party_agents/`` folder, each source
    overwriting the ones before it (:func:`refresh_registry`).
    """
    merged = _builtin_commands()
    for folder in (_seas_dir(), *user_folders, _third_party_dir()):
        if folder is not None:
            merged.update(_scan_folder(folder))
    return merged


def bundled_commands() -> dict[str, Path]:
    """Return ``{command_name: script_path}`` for every command the package ships.

    The registry's sources and precedence (:func:`_scan_sources`)
    without the user's ``SEAS.md`` folders: the deterministic list
    ``sea docs`` renders.
    """
    return _scan_sources(())


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
    except (OSError, UnicodeDecodeError):
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
        if not folder_text or "\0" in folder_text:
            continue  # no path holds a NUL byte (what a file a crash cut short is filled with)
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
    HIGHEST (the built-in scripts, bundled ``seas/``, then top-of-file
    SEAS.md entries, third-party last) and lets each source overwrite
    the previous one (:func:`_scan_sources`), so the highest source
    ends up in the registry.

    Rescans run concurrently and hold no lock while they read the
    folders, so one stuck on a slow folder delays no other.  Each is
    numbered when it starts and published only if no later-started
    rescan was published first (:func:`_publish`): an older scan never
    replaces a newer one.

    Returns:
        The sorted list of command names now installed.  Subscribers
        registered via :func:`subscribe` are invoked (under
        ``_notify_lock``, outside ``_lock``) when the list differs
        from the previous broadcast, in publish order.
    """
    scan = next(_scan_numbers)
    return _publish(scan, _scan_sources(_read_seas_md_folders()))


def _publish(scan: int, merged: dict[str, Path]) -> list[str]:
    """Install rescan number *scan*'s *merged* as the registry; notify the subscribers of a change.

    Compare-and-swap under ``_lock`` (:func:`_swap_registry`) so two
    concurrent rescans can never both observe the old
    ``_last_broadcast`` and fire the same notification twice, and a
    rescan older than the one last published is dropped.  A changed
    snapshot is delivered under ``_notify_lock`` (order:
    ``_notify_lock`` -> ``_lock``), one delivery at a time, each
    ended as soon as a later snapshot is published
    (:func:`_notify`), so snapshots reach subscribers in publish
    order; a rescan with nothing to deliver never waits for a
    delivery.  ``_lock`` itself is released before the callbacks run
    so a subscriber that calls back into the module — e.g.
    :func:`list_commands` — cannot deadlock.  Both locks are taken
    through :func:`locked_call` because a task thread rescans too
    (:func:`get_command` on a miss) and the runner's Stop injection
    must not leak them.  That Stop can also land after
    ``_last_broadcast`` was advanced for a snapshot nobody delivered:
    inside a callback, waiting for ``_notify_lock``, or still in the
    swap, between the advance and its return.  The broadcast is then
    forgotten (:func:`_forget_broadcast`) before the Stop propagates,
    so the next rescan repeats it instead of deduplicating it away for
    good.
    """
    try:
        snapshot, callbacks = locked_call(_lock, _swap_registry, scan, merged)
        if callbacks:
            locked_call(_notify_lock, _notify, snapshot, callbacks)
    except BaseException:
        locked_call(_lock, _forget_broadcast)
        raise
    return snapshot


def _notify(snapshot: list[str], callbacks: list[Callable[[list[str]], None]]) -> None:
    """Deliver *snapshot* to *callbacks* unless a later one was published since; the caller
    holds ``_notify_lock``.  The later one's rescan delivers its own snapshot after this
    lock is released; this one, delivered after it, would leave every subscriber on the
    older list.  Checked before each callback: one that rescans (the lock is reentrant)
    has delivered the later snapshot to them all by the time it returns."""
    for cb in callbacks:
        if not locked_call(_lock, _is_last_broadcast, snapshot):
            return
        try:
            cb(list(snapshot))
        except Exception:  # pragma: no cover - subscribers own errors
            logger.debug("SEA registry subscriber failed", exc_info=True)


def _is_last_broadcast(snapshot: list[str]) -> bool:
    """Whether *snapshot* is the last one published for delivery; the caller holds ``_lock``."""
    return _last_broadcast == tuple(snapshot)


def _forget_broadcast() -> None:
    """Mark the last broadcast undelivered so the next rescan notifies whatever it finds."""
    global _last_broadcast
    _last_broadcast = None


def _swap_registry(
    scan: int, merged: dict[str, Path]
) -> tuple[list[str], list[Callable[[list[str]], None]]]:
    """Replace the registry with rescan *scan*'s *merged*; return the snapshot and the callbacks.

    A rescan that started before the last published one is stale: the
    registry is kept and nobody is notified.
    """
    global _last_broadcast, _published_scan
    if scan < _published_scan:
        return sorted(_registry), []
    _published_scan = scan
    _registry.clear()
    _registry.update(merged)
    snapshot = sorted(_registry)
    snapshot_tuple = tuple(snapshot)
    if snapshot_tuple == _last_broadcast:
        return snapshot, []
    _last_broadcast = snapshot_tuple
    return snapshot, list(_subscribers)


@locked(_lock)
def _registered_names() -> list[str]:
    """Return the sorted registry names (empty before the first rescan)."""
    return sorted(_registry)


@locked(_lock)
def _registered_path(name: str) -> Path | None:
    """Return the registered script of command *name*, or ``None``."""
    return _registry.get(name)


@locked(_lock)
def _registered_entries() -> list[tuple[str, Path]]:
    """Return the ``(name, script)`` registry entries sorted by name."""
    return sorted(_registry.items())


def list_commands() -> list[str]:
    """Return the sorted list of currently-registered command names.

    Rebuilds the registry when it is empty so standalone callers
    (tests, CLI helpers) never see a stale-empty snapshot before the
    daemon watcher has run its first pass.
    """
    return _registered_names() or refresh_registry()


def registered_entries() -> list[tuple[str, Path]]:
    """Return the sorted ``(name, script)`` entries of every registered command.

    One registry read; rebuilds an empty registry like :func:`list_commands`.
    """
    entries = _registered_entries()
    if not entries:
        refresh_registry()
        entries = _registered_entries()
    return entries


def get_command(name: str) -> Path | None:
    """Resolve *name* (e.g. ``"slack"``) to the SEA script path.

    Returns ``None`` when the name is not registered.  Rebuilds the
    registry on a miss so a SEA created after the daemon started (or
    a stale in-memory copy) does not surface a spurious "unknown
    command" the very first time it is used.
    """
    hit = _registered_path(name)
    if hit is not None:
        return hit
    refresh_registry()
    return _registered_path(name)


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
    match = _SLASH_RE.match(prompt or "")
    if match is None:
        return None
    return match.group(1), (match.group(2) or "").strip()


_LOADED_STAMP = "_loaded_stamp"
"""Name, in the namespace of a SEA :func:`load_sea` returned, of the revision of the text it
ran from (:func:`~kiss.agents.sorcar.sea_settings.source_revision`)."""


def load_sea(sea_path: Path) -> BaseSea:
    """Execute the SEA file at *sea_path* and return an instance of the SEA class it defines.

    The file is executed with
    :func:`kiss.agents.sorcar.sea_settings.execute_python_file` (so
    every call observes its current contents); the SEA class is the
    one subclass of :class:`BaseSea` the file itself defines (an
    imported base class does not count).  The instance's ``path`` is
    *sea_path*.  The revision of the text that ran is kept with the
    instance (``_LOADED_STAMP``): the loader reads the file once, when
    its turn to run it has come, and names the bytes it ran, so what
    :func:`run_picked_hook` runs under a revision is the text of that
    revision, whatever the file held while the load waited for
    another load of the same script and whatever it holds since.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Raises:
        SeaError: When the file cannot be read, compiled or
            executed (whatever it raises), defines no or several
            subclasses of :class:`BaseSea`, or its class cannot be
            instantiated without arguments.  A Stop landing in the
            import or the instantiation
            (:func:`~kiss.agents.sorcar.sea_settings.is_stop`) is
            re-raised unchanged instead.
    """
    namespace = execute_python_file(str(sea_path))
    # Keyed by identity: a second name for the class (``Main = Sea``) is not a second class.
    classes = list({
        id(value): value for value in namespace.values()
        if isinstance(value, type) and issubclass(value, BaseSea) and value is not BaseSea
        and value.__module__ == namespace["__name__"]
    }.values())
    if len(classes) != 1:
        names = ", ".join(cls.__name__ for cls in classes) or "none"
        raise SeaError(
            f"SEA {str(sea_path)!r} must define exactly one subclass of BaseSea "
            f"(kiss.agents.seas.base.base_sea); found {names}"
        )
    try:
        sea = classes[0]()
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        if is_stop(exc):
            raise
        raise SeaError(
            f"SEA {str(sea_path)!r}: {classes[0].__name__}() raised: {safe_message(exc)}"
        ) from exc
    sea.path = sea_path
    vars(sea)[_LOADED_STAMP] = namespace[SOURCE_REVISION]
    return sea


def sea_class(spec: str, relative_to: str | Path = "") -> type[BaseSea]:
    """Return the class of the SEA *spec* names, for a SEA that derives from it.

    ``class MySea(sea_class("bestrouter")): ...`` extends the registered
    command ``bestrouter`` whatever folder it lives in; a *spec* with a
    ``.py`` suffix or a path separator is a file (relative to
    *relative_to*, else the current directory).  A bundled SEA can
    instead be imported (``from kiss.agents.seas.bestrouter.bestrouter_sea
    import BestrouterSea``).

    Args:
        spec: A registered command name or a path.
        relative_to: The directory a relative path is resolved against
            (pass ``__file__``'s folder).

    Raises:
        SeaError: When *spec* is no registered command or existing
            ``.py`` file, or the file is broken (:func:`load_sea`).
    """
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        path = Path(expand_user(spec))
        if not path.is_absolute() and relative_to:
            path = Path(relative_to) / path
        _note_picker_base(path)
        if path.suffix != ".py" or not path.is_file():
            raise SeaError(f"sea_class({spec!r}): not an existing Python (.py) file")
    else:
        found = get_command(spec)
        _note_picker_base(spec, found)
        if found is None:
            raise SeaError(
                f"sea_class({spec!r}): not a registered SEA command; "
                f"known commands: {', '.join(list_commands()) or 'none'}"
            )
        path = found
        _note_picker_base(path)
    return type(load_sea(path.resolve()))


def _note_picker_base(base: Path | str, resolved: Path | None = None) -> None:
    """Record what the SEA the model picker is loading on this thread, if any, read of its
    base *base*: a file's stamp as it is now, before the file runs, or the script
    *resolved* a command name resolved to."""
    bases = getattr(_picker_load, "bases", None)
    if bases is None:
        return
    if isinstance(base, Path):
        base = base.resolve()
        bases[base] = _reading(base)
    else:
        bases[base] = resolved


def _reading(base: _Watched) -> _Reading:
    """Return what the verdict cache watches of *base* now: a file's revision stamp, the
    script the command name resolves to (``None`` when it is not registered), or whether
    the module ``sys.modules`` holds under ``base[0]`` binds the name ``base[1]``.

    A file that cannot be read is ``None`` whatever its stamp: a
    ``chmod`` that makes it readable changes no stamp, and the load
    that failed on it is to run again.  A file costs an ``access`` and
    a ``stat``, never a parse
    (:func:`~kiss.agents.sorcar.sea_settings.parsed_sea`, whose memo
    would also keep answering for a file that has become unreadable):
    a failed load watches every library module it imported.
    """
    if isinstance(base, Path):
        return file_stamp(base) if os.access(base, os.R_OK) else None
    if isinstance(base, tuple):
        return base[1] in getattr(sys.modules.get(base[0]), "__dict__", ())
    return _registered_path(base)


def sea_layers(
    sea_path: Path, base: Path | None = None, loaded: Iterable[BaseSea] = (),
) -> list[BaseSea]:
    """Load the SEA at *sea_path*, under the model-picker SEA *base* when there is one.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.
        base: Path of a SEA whose methods apply before *sea_path*'s
            (the model-picker SEA of the tab a run was submitted from),
            or ``None``.
        loaded: SEAs already loaded: one whose :attr:`BaseSea.path` is
            a layer's file is reused instead of executing it again.

    Returns:
        The loaded SEAs, *base* first; one entry when *base* is
        ``None`` or the same file.

    Raises:
        SeaError: When a file is broken (:func:`load_sea`).  Malformed
            settings fail in :func:`evaluate_sea`, which every caller
            runs next.
    """
    by_path = {sea.path: sea for sea in loaded}
    seas = [by_path.get(sea_path) or load_sea(sea_path)]
    if base is not None and base != sea_path:
        seas.insert(0, by_path.get(base) or load_sea(base))
    return seas


def _chain(
    seas: list[BaseSea], name: str, include_base: bool = True
) -> list[Callable[..., Any]]:
    """Return the bound *name* methods of *seas*, each SEA's classes base first.

    The launcher's chaining (``base_prompt`` etc.): every class of a
    SEA's inheritance chain that defines *name* itself contributes,
    base class first, so a SEA's method runs on top of its base's
    without calling ``super()``.  :class:`BaseSea` itself is the root
    of every chain, so its method runs first on every run and editing
    ``base_sea.py`` customizes every run; *include_base* ``False``
    leaves it out (:func:`defines`, which asks what a SEA adds).
    """
    methods = []
    seen: set[tuple[str, str]] = set()
    for sea in seas:
        for cls in reversed(type(sea).__mro__):
            # A class both SEAs share (the picker's, when the run's SEA
            # derives from it) contributes once, in the picker's place.
            # Keyed by source file, not module name: the same class is
            # one module when imported and another when its file is
            # executed by ``load_sea``.
            key = (str(_file_of(cls) or cls.__module__), cls.__qualname__)
            if cls is BaseSea and not include_base:
                continue
            if name in vars(cls) and key not in seen:
                seen.add(key)
                methods.append(_bound(sea, cls, name))
    return methods


def sea_name(seas: list[BaseSea]) -> str:
    """Return the display name of the SEA a run of *seas* is reported under.

    The outermost SEA's :func:`~kiss.agents.sorcar.sea_settings.script_name`;
    ``""`` for a run of the bare :class:`BaseSea` (a plain run names no
    SEA even though it goes through ``base_sea.py``).
    """
    if type(seas[-1]) is BaseSea:
        return ""
    return script_name(str(seas[-1].path))


def _file_of(obj: Any) -> Path | None:
    """Return the source file of the module a class or function was defined in, if known."""
    module = sys.modules.get(getattr(obj, "__module__", ""))
    file = getattr(module, "__file__", None)
    return Path(file) if file else None


def _bound(sea: BaseSea, cls: type, name: str) -> Callable[..., Any]:
    """Return the method *name* of *cls* bound to *sea*; a non-function attribute is an error.

    A plain Python function only: a class, a ``staticmethod``, a
    builtin (``tools = len``) or a ``functools.partial`` is callable
    but binds no instance or carries no ``__name__``, which the
    diagnostics (:func:`_label`) need.
    """
    attr = vars(cls)[name]
    if not inspect.isfunction(attr):
        raise SeaError(
            f"{name} of SEA {str(sea.path)!r} must be a method, "
            f"got {type(attr).__name__}"
        )
    return cast(Callable[..., Any], attr.__get__(sea, cls))


def _method(sea: BaseSea, name: str) -> Callable[..., Any]:
    """Return the bound method *name* of *sea* (its own or a base's), checked by :func:`_bound`."""
    for cls in type(sea).__mro__:
        if name in vars(cls):
            return _bound(sea, cls, name)
    raise AttributeError(name)  # unreachable: BaseSea defines every contract method


def defines(seas: list[BaseSea], name: str) -> bool:
    """Return whether a class of *seas* other than :class:`BaseSea` defines the method *name*.

    What ``/xxx check`` and the linter report as the SEA's own
    contribution; the launcher applies the chain whether or not a SEA
    adds to the base's method.
    """
    return bool(_chain(seas, name, include_base=False))


def _label(method: Callable[..., Any]) -> str:
    """Return ``name() of SEA '<path>'`` for a bound SEA method, for diagnostics."""
    sea = method.__self__  # type: ignore[attr-defined]
    return f"{method.__name__}() of SEA {str(sea.path)!r}"


def _call(method: Callable[..., Any], *args: Any, log: bool = True) -> Any:
    """Call a bound SEA method; whatever it raises becomes a :exc:`SeaError`.

    Except a Stop (:func:`~kiss.agents.sorcar.sea_settings.is_stop`),
    re-raised unchanged: the method did not fail, the task was stopped,
    so nothing (the model picker's verdict cache included) records it
    as broken.  The failure is logged with its traceback unless *log* is
    ``False`` (a probe whose failure is an ordinary outcome,
    :func:`own_settings` and :func:`inherited_settings`).
    """
    try:
        return method(*args)
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        if is_stop(exc):
            raise
        if log:
            logger.warning("%s raised", _label(method), exc_info=True)
        raise SeaError(f"{_label(method)} raised: {safe_message(exc)}") from exc


def _fold_methods[T](
    methods: list[Callable[..., Any]], value: T, check: Callable[[str, Any], T]
) -> T:
    """Thread *value* through *methods* in order, checking each result with *check*."""
    for method in methods:
        value = check(_label(method), _call(method, value))
    return value


def declared_settings(seas: list[BaseSea]) -> dict[str, Any]:
    """Return the settings *seas* declare: every ``settings`` method folded over ``{}``, base first.

    What a SEA writes before its base class's defaults are laid under it.  A
    ``work_dir`` is kept as written: the launcher anchors a relative one
    at the calling task's directory
    (:func:`kiss.agents.sorcar.sea_settings.anchored_work_dir`), as it
    does a call's ``work_dir`` option.  The record each layer returns is
    checked as :func:`~kiss.agents.sorcar.sea_settings.resolve_settings`
    checks the final one (known keys, every value of its key's type), so
    an ill-typed write is refused whatever a later layer writes over it:
    the verdict on a chain does not depend on the order of its layers.

    Raises:
        SeaError: When a ``settings`` method raises, returns no dict, or
            returns a record with an unknown key or an ill-typed value
            (the error names the SEA and the method that wrote it).
    """
    record: dict[str, Any] = {}
    for method in _chain(seas, "settings"):
        record = _check_dict(_label(method), _call(method, record))
        try:
            resolve_settings(record)
        except SeaError as exc:
            path = method.__self__.path  # type: ignore[attr-defined]
            raise SeaError(f"SEA {str(path)!r}: {exc} (in {method.__qualname__})") from exc
    return record


def own_settings(sea: BaseSea) -> dict[str, Any]:
    """Return what the SEA's own class (not its bases) writes in ``settings``, over ``{}``.

    A tooling probe (the linter's ``redundant-key`` comparison, the
    docs).  A ``settings`` that reads what its bases lay
    (``settings["x"]``) cannot run over ``{}``: it raises
    :exc:`SeaError` like any failing method, but nothing is logged,
    since a run folds the bases first and the script is not broken.
    """
    if "settings" not in vars(type(sea)):
        return {}
    method = _bound(sea, type(sea), "settings")
    return _check_dict(_label(method), _call(method, {}, log=False))


def inherited_settings(sea: BaseSea) -> dict[str, Any]:
    """Return what the SEA's base classes write in ``settings``, over ``{}``, base first.

    The values a key of :func:`own_settings` repeats when it equals one
    of these (the linter's ``redundant-key`` finding).  A tooling probe
    like it: a method that cannot run again on the instance raises
    :exc:`SeaError`, unlogged.
    """
    methods = _chain([sea], "settings")
    if "settings" in vars(type(sea)):
        methods.pop()  # the chain ends with the SEA's own class
    record: dict[str, Any] = {}
    for method in methods:
        record = _check_dict(_label(method), _call(method, record, log=False))
    return record


def is_channel(seas: list[BaseSea]) -> bool:
    """Return whether the run of *seas* is a channel (a SEA derives from ``ChannelSea``)."""
    return any(isinstance(sea, ChannelSea) for sea in seas)


def base_settings(seas: list[BaseSea]) -> dict[str, Any]:
    """Return the effective settings of *seas*: the declared ones resolved.

    :func:`declared_settings` through
    :func:`kiss.agents.sorcar.sea_settings.resolve_settings` (the
    channel locks for a :func:`is_channel` run; the type checks passed
    on every layer in :func:`declared_settings` already).

    Raises:
        SeaError: When a ``settings`` method raises or returns no
            dict, or the settings are malformed (naming the SEA and
            the method that wrote them).
    """
    return resolve_settings(declared_settings(seas), is_channel(seas))


def sea_settings(sea_path: Path) -> dict[str, Any]:
    """Return the effective settings of the SEA at *sea_path* (:func:`base_settings`).

    For a caller that holds only the path (a SEA such as ``rsi7d``
    reading another SEA's settings); the daemon's own run path loads the
    layers once (``sea_apply.load_layers``) and evaluates them with
    :func:`evaluate_sea`.

    Raises:
        SeaError: When the file is broken or its settings are
            malformed, so the caller fails with the diagnostic instead
            of running against a broken SEA.
    """
    return base_settings([load_sea(sea_path)])


def base_prompt(seas: list[BaseSea], task: str, task_id: str = "") -> str:
    """Return *task* after every ``prompt`` method of *seas*, base first, ``{task_id}`` filled in.

    Every ``{task_id}`` of the final text — written by a method or by
    the caller's own task text — is replaced by *task_id* (the calling
    task's id, or ``""`` when there is none).

    Raises:
        SeaError: When a ``prompt`` method raises or returns
            anything but a string, or changes the text to one that is
            empty once ``{task_id}`` is filled in (a method that returns
            its argument unchanged, the stock :class:`BaseSea`, may pass
            an empty task through).
    """
    for method in _chain(seas, "prompt"):
        result = _check_text(_label(method), _call(method, task))
        if result != task and not result.replace("{task_id}", task_id).strip():
            raise SeaError(f"{_label(method)} must return a non-empty string")
        task = result
    return task.replace("{task_id}", task_id)


def _fold_system_prompt(methods: list[Callable[..., Any]], system_prompt: str) -> str:
    """Thread *system_prompt* through the ``system_prompt`` *methods* in order."""
    return _fold_methods(methods, system_prompt, _check_text)


def _fold_tools(
    methods: list[Callable[..., Any]], tools: list[Callable[..., Any]]
) -> list[Callable[..., Any]]:
    """Thread a copy of *tools* through the ``tools`` *methods* in order."""
    return _fold_methods(methods, list(tools), _check_tools)


def _fold_llm_call_hook(methods: list[Callable[..., Any]], new_messages: list[Any]) -> list[Any]:
    """Thread *new_messages* through the ``llm_call_hook`` *methods* in order."""
    return _fold_methods(methods, new_messages, _check_list)


def _first_refusal(
    methods: list[Callable[..., Any]], name: str, args: dict[str, Any]
) -> Verdict:
    """Return the first refusing verdict of the ``tool_call_hook`` *methods*, else the allowance.

    The allowance carries the notes of the allowing verdicts
    (:func:`~kiss.core.tool_verdict.allow_with_note`), in chain order;
    it is ``ALLOW`` itself when none carries one.
    """
    notes = ""
    for method in methods:
        verdict = _check_verdict(_label(method), _call(method, name, args))
        if not verdict.allowed:
            return verdict
        notes += verdict.text
    return Verdict(True, notes) if notes else ALLOW


def _first_run_end(methods: list[Callable[..., Any]], reply: LLMReply) -> Verdict:
    """Return the first ``end_run`` verdict of the ``llm_response_hook`` *methods*, else ``ALLOW``.

    Raises:
        SeaError: When a method returns a verdict that neither allows
            nor ends the run (a ``refuse(text)``: a reply cannot be
            refused, only acted on or made the run's end).
    """
    for method in methods:
        verdict = _check_verdict(_label(method), _call(method, reply))
        if not verdict.allowed:
            if not verdict.ends_run:
                raise SeaError(
                    f"{_label(method)} must return ALLOW or end_run(result), "
                    f"got refuse({verdict.text!r})"
                )
            return verdict
    return ALLOW


def base_system_prompt(seas: list[BaseSea], system_prompt: str) -> str:
    """Return *system_prompt* after every ``system_prompt`` method of *seas*, base first.

    Like :func:`base_prompt`: each method receives the text so far and
    what it returns is the text, whether it appended to it or replaced
    it.  The run uses the last return verbatim.

    Raises:
        SeaError: When a ``system_prompt`` method raises or
            returns anything but a string.
    """
    return _fold_system_prompt(_chain(seas, "system_prompt"), system_prompt)


def base_tools(seas: list[BaseSea], tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """Return *tools* after every ``tools`` method of *seas*, base first (a copy is passed)."""
    return _fold_tools(_chain(seas, "tools"), tools)


def base_tool_call_hook(seas: list[BaseSea], name: str, args: dict[str, Any]) -> Verdict:
    """Return the first refusing verdict a ``tool_call_hook`` of *seas* gives, else ``ALLOW``.

    Every hook returns a :class:`~kiss.core.tool_verdict.Verdict`
    (``ALLOW`` or ``refuse(text)``); anything else — ``None``, a
    string — is a broken SEA (``uv run sea lint`` flags such returns as
    ``verdict``).
    """
    return _first_refusal(_chain(seas, "tool_call_hook"), name, args)


def base_llm_call_hook(seas: list[BaseSea], new_messages: list[Any]) -> list[Any]:
    """Return *new_messages* after every ``llm_call_hook`` method of *seas*, base first."""
    return _fold_llm_call_hook(_chain(seas, "llm_call_hook"), new_messages)


def base_llm_response_hook(seas: list[BaseSea], reply: LLMReply) -> Verdict:
    """Return the first ``end_run`` an ``llm_response_hook`` of *seas* returns, else ``ALLOW``."""
    return _first_run_end(_chain(seas, "llm_response_hook"), reply)


@dataclass
class SeaRun:
    """What a run takes from its SEAs (:func:`evaluate_sea`).

    Attributes:
        task: The task text the run was submitted with.
        settings: The effective settings (:func:`base_settings`).
        prompt: ``task`` after every ``prompt`` method,
            ``{task_id}`` replaced by the calling task's id.
        system_prompt_hook: ``system_prompt -> system_prompt``
            (:func:`base_system_prompt`).
        tools_hook: ``tools -> tools`` (:func:`base_tools`).
        llm_call_hook: ``new_messages -> messages``
            (:func:`base_llm_call_hook`).
        tool_call_hook: ``(name, args) -> verdict``
            (:func:`base_tool_call_hook`).
        llm_response_hook: ``reply -> verdict``
            (:func:`base_llm_response_hook`).
    """

    task: str
    settings: dict[str, Any]
    prompt: str
    system_prompt_hook: Callable[[str], str]
    tools_hook: Callable[[list[Any]], list[Any]]
    llm_call_hook: Callable[[list[Any]], list[Any]]
    tool_call_hook: Callable[[str, dict[str, Any]], Verdict]
    llm_response_hook: Callable[[LLMReply], Verdict]


def evaluate_sea(seas: list[BaseSea], task: str, task_id: str = "") -> SeaRun:
    """Return the configuration of a run of *seas* on *task*.

    The settings and the prompt are computed now; the system prompt,
    tools and the three hooks are returned as callables for the run to
    apply where it assembles its system prompt, builds its toolset, and
    makes its LLM and tool calls.  Every chain starts at
    :class:`BaseSea`, so the callables are always present (identities
    unless ``base_sea.py`` or a SEA changes something).

    Args:
        seas: The loaded SEAs (:func:`sea_layers`), or ``[BaseSea()]``
            for a run that names no SEA.
        task: The task text the run was submitted with.
        task_id: The calling task's id, substituted for ``{task_id}``
            in the prompt (empty when there is none).

    Raises:
        SeaError: When a ``settings`` or ``prompt`` method raises
            or returns a value of the wrong type, or a hook attribute is
            not a method.
    """
    # The chains are built once: a non-method attribute fails now, not at
    # the hook's first call, and the hooks run over the bound methods.
    chains = {
        name: _chain(seas, name)
        for name in (
            "system_prompt", "tools", "llm_call_hook", "tool_call_hook", "llm_response_hook",
        )
    }
    return SeaRun(
        task,
        base_settings(seas),
        base_prompt(seas, task, task_id),
        functools.partial(_fold_system_prompt, chains["system_prompt"]),
        functools.partial(_fold_tools, chains["tools"]),
        functools.partial(_fold_llm_call_hook, chains["llm_call_hook"]),
        functools.partial(_first_refusal, chains["tool_call_hook"]),
        functools.partial(_first_run_end, chains["llm_response_hook"]),
    )


def _reject(label: str, value: Any, expected: str) -> SeaError:
    """Return the :exc:`SeaError` for a method result *value* that is not *expected*.

    An ``async def`` method returns a coroutine: it is closed here, so
    the GC does not add ``RuntimeWarning: coroutine ... was never
    awaited`` on stderr to the error the caller gets.
    """
    if inspect.iscoroutine(value):
        value.close()
    return SeaError(f"{label} must return {expected}, got {type(value).__name__}")


def _check_verdict(label: str, value: Any) -> Verdict:
    """Return *value* as a :class:`Verdict`, or raise :exc:`SeaError`."""
    if not isinstance(value, Verdict):
        raise _reject(
            label, value,
            "a Verdict (ALLOW, or refuse(text), from kiss.agents.seas.base.base_sea)",
        )
    return value


def _check_text(label: str, value: Any) -> str:
    """Return *value* as a plain string, or raise :exc:`SeaError`.

    A ``str`` subclass from an untrusted script is copied into an exact
    ``str`` (its own methods may raise), so later ``strip()``/joins
    cannot run script code.  ``str()`` alone is no copy: it returns
    whatever the subclass's ``__str__`` does, the instance itself
    included, so ``str.__str__`` copies its result.
    """
    if not isinstance(value, str):
        raise _reject(label, value, "a string")
    try:
        return str.__str__(str(value))
    except BaseException as exc:  # noqa: BLE001 — an untrusted str subclass may raise
        if is_stop(exc):
            raise
        raise SeaError(f"{label} returned a broken value: {safe_message(exc)}") from exc


def _check_dict(label: str, value: Any) -> dict[str, Any]:
    """Return *value* as a dict, or raise :exc:`SeaError`."""
    if not isinstance(value, dict):
        raise _reject(label, value, "a dict")
    return value


def _check_list(label: str, value: Any) -> list[Any]:
    """Return *value* as a list, or raise :exc:`SeaError`."""
    if not isinstance(value, list):
        raise _reject(label, value, "a list")
    return value


def _check_tools(label: str, value: Any) -> list[Callable[..., Any]]:
    """Return *value* as a list of tool callables, or raise :exc:`SeaError`."""
    try:
        if isinstance(value, list) and all(callable(tool) for tool in value):
            return value
    except BaseException as exc:  # noqa: BLE001 — an untrusted list may raise while iterated
        if is_stop(exc):
            raise
        raise SeaError(f"{label} returned a broken list: {safe_message(exc)}") from exc
    raise _reject(label, value, "a list of tool callables (not a file path)")


PICKED_HOOK_TIMEOUT_SECONDS = 15.0
"""How long :func:`run_picked_hook` waits for ``on_picked_as_model()`` before moving on."""

_PickedHook = tuple[str, str, str | None]
"""A picked hook: the model name, the work directory and the revision of the text the SEA
ran from (``_LOADED_STAMP``; ``None`` for a SEA :func:`load_sea` did not load)."""

_picked_hooks: dict[_PickedHook, threading.Thread] = {}
"""The thread each picked hook is reserved for, until the hook is over (:func:`run_picked_hook`).

Guarded by ``_lock``.  A thread is entered before it is started and runs the hook only
while the entry is its own; an entry whose thread is not alive counts for nothing."""


def _log_picked_hook(hook: _PickedHook, sea: BaseSea | None) -> None:
    """Run ``on_picked_as_model(work_dir)`` of *sea*, else of the SEA the model names; log it.

    The body of the thread *hook* is reserved for, so a ``sea`` of
    ``None`` costs the caller no script load; a given ``sea`` runs as
    it was loaded, whether or not the registry still lists it when the
    thread gets to it.  A thread whose entry is no longer its own (a
    later pick replaced it before it got going) runs nothing.  The
    thread gives up its entry once the hook is over.  A script that
    loaded as another revision than the one its pick was keyed by
    (edited, or registered anew, in between) is picked again as it
    loaded: its hook runs under the revision that was executed, where
    the next pick finds it, and whoever waits for this thread waits
    for that run.
    """
    model, work_dir, revision = hook
    try:
        if locked_call(_lock, _picked_hooks.get, hook) is not threading.current_thread():
            return
        if sea is None:
            sea_path = model_sea(model)
            if sea_path is None:
                return
            sea = load_sea(sea_path)
        if vars(sea).get(_LOADED_STAMP) == revision:
            note = _call(_method(sea, "on_picked_as_model"), work_dir)
            if note:
                logger.info("SEA %s picked as model: %s", model, note)
            return
    except SeaError:
        logger.warning("SEA %s: on_picked_as_model() failed", model, exc_info=True)
        return
    finally:
        locked_call(_lock, _end_picked_hook, hook)
    run_picked_hook(model, work_dir, sea=sea)


def _end_picked_hook(hook: _PickedHook) -> None:
    """Remove the entry of *hook* if it is this thread's; the caller holds ``_lock``."""
    if _picked_hooks.get(hook) is threading.current_thread():
        del _picked_hooks[hook]


def _picked_hook_thread(hook: _PickedHook, sea: BaseSea | None) -> threading.Thread:
    """Return the thread *hook* is reserved for: the one running it, else a new one, started.

    The caller holds ``_lock``.  The thread is entered before it is
    started: a Stop landing in ``Thread.start()``, which waits for the
    new thread to get going, would otherwise leave a running hook that
    no pick finds, and every pick would start another.  The thread
    needs ``_lock`` to see that the entry is its own, so it runs
    nothing before its pick has left here.  A start that is refused
    takes its entry back.  One that a Stop interrupted leaves it,
    because nothing tells whether the thread exists: a thread that
    does not is not alive, so no pick shares it and the next one
    replaces it, as it replaces one that has yet to get going, which
    then finds the entry is not its own.

    Raises:
        RuntimeError: The process is out of threads.
    """
    thread = _picked_hooks.get(hook)
    if thread is None or not thread.is_alive():
        thread = threading.Thread(
            target=_log_picked_hook, args=(hook, sea), name="sea-picked-hook", daemon=True
        )
        _picked_hooks[hook] = thread
        try:
            thread.start()
        except RuntimeError:
            del _picked_hooks[hook]
            raise
    return thread


def run_picked_hook(
    model: str,
    work_dir: str,
    wait: bool = True,
    sea: BaseSea | None = None,
) -> None:
    """Run the picked SEA's ``on_picked_as_model(work_dir)`` hook on a thread.

    The hook is a model-picker SEA's chance to act on being chosen as a
    tab's model — ``autorouter`` makes sure its weekly ``/rsi7d autorouter``
    cron job is scheduled.  The daemon calls it when the user picks the SEA
    (``selectModel``) and once per run whose model is the SEA.  A pick
    whose hook is still running from an earlier pick of the same model,
    directory and script revision shares that run instead of starting
    another, so a hook that never returns holds one thread, not one per
    pick.  The revision is that of the text the SEA ran from (a digest
    of the bytes its load executed), not the one the file has when it
    is picked: the hook of a script fixed since runs beside the stuck
    hook of its old text, whichever was loaded or picked first, and a
    script saved again unchanged is the same hook.  The text is the
    picked script's own: a fix in a file its class derives from is no
    new revision of it.  Hooks of picks from different directories may
    overlap: a hook that must not race gets its atomicity from what it
    calls (``cron_job("ensure")`` for autorouter).  Whatever the hook
    returns is logged; a hook that raises is logged and never fails the
    caller, one that blocks is abandoned on its daemon thread after
    :data:`PICKED_HOOK_TIMEOUT_SECONDS`, and a thread that cannot start
    is logged too, because the task the user typed does not depend on
    it.

    Args:
        model: The picked model name, a SEA command name
            (:func:`model_sea` maps it to the file; a real model name
            runs nothing, and one that is no command starts no thread).
        work_dir: The tab's or run's work directory, passed to the hook.
        wait: Block up to :data:`PICKED_HOOK_TIMEOUT_SECONDS` for the hook,
            the shared run included (a run wants its side effects in
            place before it starts); ``False`` returns at once (the
            picker handler must not stall the daemon's command loop).
        sea: The already-loaded SEA (a run loads the file once and
            shares it; its hook runs even when the registry drops the
            entry meanwhile); ``None`` resolves *model* through
            :func:`model_sea` and loads the file, on the hook thread.
            Only the registry (filled first when it is still empty, as
            :func:`model_sea` fills it) and the bytes of the registered
            script are read here, for the revision its text has now.
    """
    if sea is None:
        list_commands()  # populate the registry on a cold start
        path = _registered_path(model)
        if path is None:
            return
        revision = file_revision(path)
    else:
        revision = vars(sea).get(_LOADED_STAMP)
    try:
        thread = locked_call(_lock, _picked_hook_thread, (model, work_dir, revision), sea)
    except RuntimeError:  # the process is out of threads
        logger.warning("SEA %s: on_picked_as_model() not run", model, exc_info=True)
        return
    if not wait:
        return
    # Sliced so the daemon's forced Stop (an async exception injected into
    # the task thread) lands between slices instead of after the whole wait.
    deadline = time.monotonic() + PICKED_HOOK_TIMEOUT_SECONDS
    while thread.is_alive() and time.monotonic() < deadline:
        thread.join(0.2)
    if thread.is_alive():
        logger.warning(
            "SEA %s: on_picked_as_model() still running after %.0f s; going on without it",
            model, PICKED_HOOK_TIMEOUT_SECONDS,
        )


def _registers_as_model(
    sea_path: Path, reads: list[dict[_Watched, _Reading]] | None = None
) -> bool:
    """Return whether the SEA at *sea_path* defines ``register_as_model()`` returning ``True``.

    *reads*, when given, receives the maps the call records what it
    reads in, as it fills them: what :meth:`_ModelCheck.outdated`
    compares while the call is still running.

    The verdict is cached per file stamp (the revision
    :func:`~kiss.agents.sorcar.sea_settings.parsed_sea` read, so both
    an edit and an atomic replacement are re-read) of the script and
    of every file its class derives from, since the method may be
    inherited from a custom base (``class X(sea_class(...))``) and an
    edit of that base alone changes the verdict; a base is stamped as
    it was before it ran, so one edited while the load was in flight
    is re-read too, a base named by command is also keyed on the
    script the name resolves to (:func:`_reading`), and a base the
    script imports the ordinary way is re-executed once it changed
    (:func:`~kiss.agents.sorcar.sea_settings.refresh_imported_module`),
    with every class deriving from it (:func:`_refresh_bases`), so the
    stamp describes the class the script derives from.  Only a script
    whose source mentions ``register_as_model`` or derives its class
    from something other than the plain bases (so it may inherit the
    registration) is imported: importing every registered SEA (channel
    agents with heavy dependencies among them) on each picker refresh
    would be slow for nothing.  A script that cannot be read, fails to
    import or whose method raises is logged and treated as not
    registered, so one broken SEA cannot break the model picker.  A
    load that missed a name of a module because the module, or its
    submodule of that name, was still being imported (``from module
    import name`` or ``module.name``, in the script or in a base
    :func:`sea_class` loads for it; ``import`` hands a module over
    half-initialised when its thread waits for a module this thread
    imports) is not cached, whether that import is still under way,
    has bound the name since or (``from module import name``, the
    statement's own failure, :func:`_import_raised`) has failed since,
    the module gone from ``sys.modules`` and its source no part of
    what this thread ran (:func:`_ran_source_of`; a module this thread
    was importing itself is the file's failure): the next lookup
    loads again.  Nor is a failed load that looked for a module
    no finder found (:class:`_PickerImports`): no file
    names such a module, so nothing would re-read the verdict once it
    is created or installed.  A name missed in a module imported
    before the load is watched itself: the verdict is re-read once
    that module binds it.
    """
    parsed = parsed_sea(sea_path)
    # ``parsed_sea`` memoises a revision it has read, so it still answers
    # for a script that has lost its read permission since; ``_reading`` does not.
    if parsed.stamp is None or _reading(sea_path) is None:
        logger.warning("SEA %s: cannot be read", sea_path)
        return False
    cached = locked_call(_lock, _model_sea_cache.get, sea_path)
    if cached is not None and _verdict_holds(cached):
        return cached[1]
    verdict = False
    stamps: dict[_Watched, _Reading] = {sea_path: parsed.stamp}
    if parsed.mentions_register_as_model or parsed.bases.difference(SEA_BASES):
        outer = getattr(_picker_load, "bases", None), getattr(_picker_load, "imported", None)
        asked: dict[_Watched, _Reading] = {}
        imported: dict[_Watched, _Reading] = {}
        if reads is not None:
            reads += [stamps, asked, imported]
        _picker_load.bases, _picker_load.imported = asked, imported
        sea: BaseSea | None = None
        try:
            # A base imported the ordinary way (``from parent import
            # Parent``) is re-executed before the child runs, else the
            # child would derive from its old class under its new stamp.
            # Only one that changed: a failed load also watches every
            # library module it imported, which must not run twice.
            for base, reading in cached[0].items() if cached is not None else ():
                if isinstance(base, Path) and _reading(base) != reading:
                    refresh_imported_module(base)
            sea = load_sea(sea_path)
            _refresh_bases(sea)
            if _superseded(type(sea)):
                # A base re-executed after the load (first met now, its
                # module's revision unknown before the load, or deriving
                # from a changed one): the class derives from the old module.
                sea = load_sea(sea_path)
            stamps.update(_base_stamps(sea))
            verdict = _call(_method(sea, "register_as_model")) is True
        except SeaError as exc:
            cause, errors = exc.__cause__, [exc]
            while isinstance(cause, SeaError) and all(cause is not error for error in errors):
                errors.append(cause)  # a base's failed load (a script's own chain may loop)
                cause = cause.__cause__
            left = False
            if isinstance(cause, ImportError) and isinstance(cause.name, str):
                module, name = sys.modules.get(cause.name), cause.name_from
                left = (
                    module is None
                    and _import_raised(cause)
                    and not _ran_source_of(cause.name, [*errors, cause])
                )
            elif isinstance(cause, AttributeError) and inspect.ismodule(cause.obj):
                module, name = cause.obj, cause.name
            else:
                module = name = None
            # ``from module import name`` or ``module.name`` missed the name while the module,
            # or its submodule of that name, was still being imported: it still is, or the
            # name is bound since, or the module has left ``sys.modules`` (*left*): the import,
            # another thread's, has failed.  Not the file's failure, so not cached.  (The
            # exception may be the script's own: only strings are looked up.)
            spec = getattr(module, "__spec__", None)
            if isinstance(name, str) and (
                left
                or getattr(spec, "_initializing", False)
                or name in getattr(spec, "_uninitialized_submodules", ())
                or name in getattr(module, "__dict__", ())
            ):
                return False
            logger.warning("SEA %s: register_as_model() failed", sea_path, exc_info=True)
            # The load looked for a module that was not found (recorded under its name):
            # no file names it, so its creation or install would go unread.  Not cached.
            if any(isinstance(base, str) for base in imported):
                return False
            # A base may be what broke (a parent mid-edit, or not yet
            # written): keep watching every base this load asked for
            # or imported, every one the last verdict depended on and,
            # once the script loaded, every one its class derives from
            # (what failed may be the re-execution of one imported
            # before this load), so its repair is re-read.
            for base in cached[0] if cached is not None else ():
                stamps.setdefault(base, _reading(base))
            stamps.update(imported)
            if sea is not None:
                stamps.update(_base_stamps(sea))
            # The name was missed in a module imported before this load, whose file
            # nothing above names: re-read once that module binds it (re-executed
            # for another SEA).
            owner = getattr(module, "__name__", None)
            if isinstance(owner, str) and isinstance(name, str):
                stamps[owner, name] = False
        finally:
            _picker_load.bases, _picker_load.imported = outer
        # Each base as it was read, not as it is now: one edited while
        # the load ran would otherwise lend its new revision to the
        # verdict of its old class.
        stamps.update(asked)
    locked_call(_lock, operator.setitem, _model_sea_cache, sea_path, (stamps, verdict))
    return verdict


def _verdict_holds(cached: tuple[dict[_Watched, _Reading], bool]) -> bool:
    """Whether the cached verdict *cached* still holds: all it was read from reads the same."""
    return all(_reading(base) == reading for base, reading in cached[0].items())


def _import_raised(error: ImportError) -> bool:
    """Whether a ``from module import name`` statement raised *error* itself: its traceback
    ends at that statement's instruction, not at a ``raise``.

    An ``ImportError`` a script raises of its own may name any module, one that was never
    in ``sys.modules`` as well as one that left it; that failure is the file's.
    """
    tb = error.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return tb is not None and tb.tb_frame.f_code.co_code.startswith(
        bytes([dis.opmap["IMPORT_FROM"]]), tb.tb_lasti
    )


def _ran_source_of(module: str, errors: Iterable[BaseException]) -> bool:
    """Whether the thread *errors* (a failure and its causes) were raised on was running the
    source of *module* then: one of their tracebacks holds a frame of that module.

    Tells an import of this thread's own that failed on a name the module had not bound yet
    (a circular import: the file's failure) from a module handed over half-initialised by
    another thread's import.  What ran after the failure was raised is not on a traceback:
    ``traceback`` imports the module an ``ImportError`` names to format it for the log.
    """
    for error in errors:
        tb = error.__traceback__
        while tb is not None:
            if tb.tb_frame.f_globals.get("__name__") == module:
                return True
            tb = tb.tb_next
    return False


def _base_stamps(sea: BaseSea) -> dict[_Watched, _Reading]:
    """Return the revision of every SEA file the class of *sea* derives from, as its class ran.

    The revision the base's module was re-executed from
    (:func:`~kiss.agents.sorcar.sea_settings.imported_revision`), not
    the file's revision now: the file may have been edited since the
    class ran, and the verdict is the old class's.  ``None`` (no
    revision matches it, so the next query re-reads) for a base whose
    module was re-executed again since the class derived from it; the
    file's revision now for one never re-executed (the bundled bases).
    The revision is read before the class is looked up: a re-execution
    publishes its module first and its revision last, so one landing
    between the two reads fails the lookup instead of lending the old
    class its revision.
    """
    stamps: dict[_Watched, _Reading] = {}
    for cls in type(sea).__mro__[1:]:
        file = _file_of(cls)
        if file is None or not issubclass(cls, BaseSea):
            continue
        revision = imported_revision(file) or _reading(file)
        if getattr(sys.modules.get(cls.__module__), cls.__name__, None) is not cls:
            stamps[file] = None
        else:
            stamps.setdefault(file, revision)
    return stamps


def _refresh_bases(sea: BaseSea) -> None:
    """Re-execute the changed imported bases of the class of *sea*, root first.

    Each base after every base of its own: one that derives from a
    class its module no longer holds (:func:`_superseded`, the base's
    own base re-executed before this walk or by it) is re-executed
    whatever its revision, so its class derives from the new one.
    Afterwards every base is current, and the script's own class is
    superseded exactly when the script must run again.
    """
    for cls in reversed(type(sea).__mro__[1:]):
        file = _file_of(cls)
        if file is not None and issubclass(cls, BaseSea):
            refresh_imported_module(file, _superseded(cls))


def _superseded(cls: type) -> bool:
    """Whether a base of *cls* is no longer the class its module holds under its name: the
    module was re-executed since *cls* derived from it, and *cls* must be re-executed too."""
    return any(
        getattr(sys.modules.get(base.__module__), base.__name__, None) is not base
        for base in cls.__bases__
    )


MODEL_CHECK_TIMEOUT_SECONDS = 15.0
"""How long :func:`model_seas` waits for one SEA's import and ``register_as_model()``."""


class _ModelCheck(threading.Thread):
    """Decides on a thread of its own whether a script is a model-picker entry.

    Deciding it runs the script (:func:`_registers_as_model`), and the
    daemon lists the picker on a thread no Stop reaches: run there, a
    script that does not return would hold that thread for good, and
    one more with every later listing.

    Attributes:
        sea_path: The script.
        deadline: The ``time.monotonic()`` until which a listing waits
            for the verdict.
        verdict: The verdict; ``False`` until the thread has ended.
        reads: The maps the check records what it reads in, as it fills
            them (:func:`_registers_as_model`).
    """

    def __init__(self, sea_path: Path) -> None:
        super().__init__(name="sea-model-check", daemon=True)
        self.sea_path = sea_path
        self.deadline = time.monotonic() + MODEL_CHECK_TIMEOUT_SECONDS
        self.verdict = False
        self.reads: list[dict[_Watched, _Reading]] = []

    def run(self) -> None:
        """Decide the verdict, then give up the script's entry in ``_model_checks``.

        A check that finds the entry is not its own (replaced before it
        got going, :func:`_model_check`) decides nothing.
        """
        try:
            if locked_call(_lock, _model_checks.get, self.sea_path) is self:
                self.verdict = _registers_as_model(self.sea_path, self.reads)
        except BaseException:  # noqa: BLE001 — a Stop the script raised itself: none lands here
            logger.warning("SEA %s: register_as_model() failed", self.sea_path, exc_info=True)
        finally:
            locked_call(_lock, _end_model_check, self)

    def outdated(self) -> bool:
        """Return whether a file the check has read so far, or a command name it resolved,
        reads differently now: the check decides on a script that is no longer the one on
        disk, its own or one its class derives from.

        A reading no file matches by design is not compared (``None``:
        a base that was missing, or re-executed since the class derived
        from it; the revision of a module still being imported): it
        differs at every listing, and each would start a thread.
        """
        return any(
            _reading(base) != reading
            for reads in list(self.reads)
            for base, reading in reads.copy().items()
            if reading is not None and reading != _MID_IMPORT
        )


_model_checks: dict[Path, _ModelCheck] = {}
"""The check of each script that is being decided for :func:`model_seas`; guarded by ``_lock``."""


def _end_model_check(check: _ModelCheck) -> None:
    """Remove the entry of *check* if it is still its own; the caller holds ``_lock``."""
    if _model_checks.get(check.sea_path) is check:
        del _model_checks[check.sea_path]


def _model_check(sea_path: Path, stale: _ModelCheck | None) -> _ModelCheck:
    """Return the check of *sea_path*: the one running unless it is *stale* (the one the
    caller found outdated), else a new one, started.

    The caller holds ``_lock``.  Entered before it is started and taken
    back when the start is refused, as a picked hook's thread is
    (:func:`_picked_hook_thread`): the thread needs ``_lock`` to see
    that the entry is its own, so one whose start a Stop interrupted
    runs nothing once the next listing has replaced it.

    Raises:
        RuntimeError: The process is out of threads.
    """
    check = _model_checks.get(sea_path)
    if check is None or check is stale or not check.is_alive():
        check = _ModelCheck(sea_path)
        _model_checks[sea_path] = check
        try:
            check.start()
        except RuntimeError:
            del _model_checks[sea_path]
            raise
    return check


def _listed_as_model(sea_path: Path) -> bool:
    """Return whether the script at *sea_path* is a model-picker entry, for :func:`model_seas`.

    The cached verdict of :func:`_registers_as_model` while it holds.
    Otherwise the script's :class:`_ModelCheck` decides, shared by every
    listing that asks meanwhile and waited for until its deadline: a
    script whose import or ``register_as_model()`` does not return is
    logged and not listed, holds one thread however many listings ask,
    and is listed once it has answered.  A check is given up for a new
    one only once something it read has changed
    (:meth:`_ModelCheck.outdated`): the repaired script is listed
    though the old check never returns.  A listing made by the script
    itself, from the code its check runs, does not wait for that check;
    a process out of threads lists nothing it has not decided yet.
    """
    cached = locked_call(_lock, _model_sea_cache.get, sea_path)
    if cached is not None and _verdict_holds(cached):
        return cached[1]
    check = locked_call(_lock, _model_checks.get, sea_path)
    if check is threading.current_thread():
        return False
    stale = check if check is not None and check.outdated() else None  # reads files: unlocked
    try:
        check = locked_call(_lock, _model_check, sea_path, stale)
    except RuntimeError:  # the process is out of threads
        logger.warning(
            "SEA %s: register_as_model() not run; not offered as a model", sea_path, exc_info=True
        )
        return False
    # Sliced so a Stop of the calling thread lands between slices instead of after the wait.
    while check.is_alive() and time.monotonic() < check.deadline:
        check.join(0.2)
    if check.is_alive():
        logger.warning(
            "SEA %s: register_as_model() still running after %.0f s; not offered as a model",
            sea_path, MODEL_CHECK_TIMEOUT_SECONDS,
        )
        return False
    return check.verdict


def model_seas() -> dict[str, Path]:
    """Return the registered SEAs that are model-picker entries, by command name.

    A SEA is a model-picker entry when its ``register_as_model()``
    returns ``True``.  The daemon offers each entry in the model picker
    under its command name; a task run with such a pick goes through
    the SEA (``task_runner._resolve_sea_model``).  A SEA that takes
    longer than :data:`MODEL_CHECK_TIMEOUT_SECONDS` to answer is left
    out until it has (:func:`_listed_as_model`), so one that never
    does cannot keep the picker from listing the others.

    Returns:
        ``{command_name: absolute_sea_path}``, sorted by name.
    """
    return {name: path for name, path in registered_entries() if _listed_as_model(path)}


def model_sea(name: str, bundled: bool = False) -> Path | None:
    """Return the script of the model-picker SEA *name*, or ``None``.

    Reads the registry snapshot only: a *name* that is no registered
    command (every real model name) costs one dict lookup and touches no
    script, so the task runner can ask this for every run's model.  The
    picker offers a SEA only once the registry holds it (the watcher
    refreshes every :data:`_WATCHER_POLL_SECONDS`), so unlike
    :func:`get_command` a miss here does not rescan the folders.

    Args:
        name: The model-picker value, e.g. ``"autorouter"`` or
            ``"gpt-6-astra"``.
        bundled: Look *name* up among the commands the package ships
            (:func:`bundled_commands`) instead of the registry: what
            ``sea lint`` asks about a bundled script, whose findings
            must not depend on what ``SEAS.md`` registers.

    Returns:
        The absolute SEA path when *name* is a registered (or, with
        *bundled*, a bundled) command whose ``register_as_model()``
        returns ``True``, else ``None``.
    """
    if not name:
        return None
    if bundled:
        path = bundled_commands().get(name)
    else:
        list_commands()  # populate the registry on a cold start
        path = _registered_path(name)
    if path is None or not _registers_as_model(path):
        return None
    return path


def sea_description(sea: BaseSea) -> str:
    """Return the ``description()`` text of *sea*: the text ``/xxx help`` shows the user.

    Raises:
        SeaError: When ``description()`` raises or returns
            anything but a non-empty string (a command must describe
            itself in one sentence).
    """
    method = _method(sea, "description")
    text = _check_text(_label(method), _call(method))
    if not text.strip():
        raise SeaError(
            f"SEA {str(sea.path)!r}: description() must return a non-empty string"
        )
    return text.strip()


CHECK_SAMPLE_TASK = "<the task text>"
"""The task :func:`sea_check` hands ``prompt(task)``; ``{task_id}`` becomes ``<task id>``."""

RESERVED_SUBCOMMANDS = ("help", "check")
"""The ``/xxx <word>`` prompts the daemon answers itself instead of running the SEA."""


@dataclass(frozen=True)
class SeaCheck:
    """What :func:`check_sea` found: the one evaluation ``sea_check`` and the linter read.

    Attributes:
        seas: The loaded SEAs (:func:`sea_layers`).
        cmd: The ``run`` command for :data:`CHECK_SAMPLE_TASK` with the
            SEA's overrides applied (``sea_apply.apply_run``).
        description: The ``description()`` text (``""`` when not
            required and absent).
        settings: The effective settings (:func:`base_settings`), the
            ones ``cmd`` carries.
        tools: What ``tools()`` adds to an empty toolset.
    """

    seas: list[BaseSea]
    cmd: dict[str, Any]
    description: str
    settings: dict[str, Any]
    tools: list[Any]


def check_sea(sea_path: Path, require_description: bool = True) -> SeaCheck:
    """Load the SEA at *sea_path* the way a run would and exercise every method.

    The daemon's own path (``load_layers``, :func:`evaluate_sea` and
    ``sea_apply.apply_run`` on a ``run`` command for
    :data:`CHECK_SAMPLE_TASK`), so a SEA that
    reads the run command from those frames works; then the hooks are
    applied to sample values and the ``description()`` and
    model-picker methods are checked.  Each method runs as often as a
    run makes it run; the result carries the merged settings and the
    tools (what a run gets), not the per-class ``settings`` dicts
    (``own_settings`` / ``inherited_settings`` run those again when the
    linter needs them).

    Args:
        sea_path: The SEA's file.
        require_description: Whether an empty ``description()`` is an
            error (it is for a ``/command``; a SEA only loaded by path
            or used as a base class needs none).

    Returns:
        The :class:`SeaCheck`.

    Raises:
        SeaError: The SEA breaks the contract, in the words the daemon
            would use.
    """
    # Imported here: ``sea_apply`` imports this module.
    from kiss.agents.sorcar.sea_apply import apply_run, load_layers

    cmd: dict[str, Any] = {
        "seaPath": str(sea_path), "prompt": CHECK_SAMPLE_TASK, "parentTaskId": "<task id>",
    }
    seas = load_layers(cmd)
    run = evaluate_sea(seas, CHECK_SAMPLE_TASK, "<task id>")
    apply_run(cmd, seas, run)
    run.system_prompt_hook("<the system prompt>")
    tools = run.tools_hook([])
    run.tool_call_hook("finish", {})
    run.llm_call_hook([])
    run.llm_response_hook(LLMReply("<the model's reply>"))
    description = ""
    if require_description or defines(seas, "description"):
        description = sea_description(seas[-1])
    registers = _call(_method(seas[-1], "register_as_model"))
    if not isinstance(registers, bool):
        raise SeaError(
            f"register_as_model() of SEA {str(sea_path)!r} must return a bool, "
            f"got {type(registers).__name__}"
        )
    # Not called: picking the SEA as a model has side effects (the
    # bundled autorouter schedules a cron job); only its shape is checked.
    picked = _method(seas[-1], "on_picked_as_model")
    try:
        inspect.signature(picked).bind("<work_dir>")
    except (TypeError, ValueError) as exc:
        raise SeaError(
            f"{_label(picked)} must accept the work directory as its one argument: {exc}"
        ) from exc
    return SeaCheck(seas, cmd, description, run.settings, tools)


def sea_check(name: str, sea_path: Path) -> str:
    """Load the SEA *name* at *sea_path* and report what a run of it would use.

    What ``/xxx check`` shows, so a SEA author sees the effect of the
    class without running a task: the classes of its inheritance
    chain, the effective settings, the model a run takes, the names of
    the tools ``tools()`` adds to an empty toolset, which methods are
    defined, and the prompt ``prompt(task)`` yields for
    :data:`CHECK_SAMPLE_TASK`.  A broken SEA yields the first error
    instead, in the words the daemon would use.

    Args:
        name: The command name.
        sea_path: The SEA's file.

    Returns:
        The report, one item per line.
    """
    try:
        check = check_sea(sea_path)
    except SeaError as exc:
        return f"/{name} is broken: {exc}"
    seas, settings = check.seas, check.settings
    model = settings.get("model") or "the calling task's model (else the default model)"
    classes = [
        cls.__name__ for cls in reversed(type(seas[-1]).__mro__) if issubclass(cls, BaseSea)
    ]
    defined = [
        method for method in (
            "system_prompt", "prompt", "tools", "llm_call_hook", "tool_call_hook",
            "llm_response_hook", "register_as_model", "on_picked_as_model",
        ) if defines(seas, method)
    ]
    lines = [
        f"/{name}: {check.description}",
        "classes: " + " > ".join(classes),
        f"settings: {_settings_text(settings)}",
        f"model: {model}",
        "tools added: " + (", ".join(_tool_name(tool) for tool in check.tools) or "none"),
        "methods defined: " + (", ".join(defined) or "none"),
        f"prompt for {CHECK_SAMPLE_TASK}: {check.cmd['prompt']}",
    ]
    if "description" not in vars(type(seas[-1])):
        lines.append("note: description() comes from a base class")
    return "\n".join(lines)


def _settings_text(settings: dict[str, Any]) -> str:
    """Return *settings* as JSON with sorted keys, for the ``check`` report.

    A value json cannot write (a ``model_config`` whose keys mix types
    and so cannot be sorted, an object whose ``__str__`` raises) is
    named in place of the text: the report is never an exception.
    """
    try:
        return json.dumps(settings, sort_keys=True, default=str)
    except BaseException as exc:  # noqa: BLE001 — an untrusted value may raise anything
        if is_stop(exc):
            raise
        return f"not JSON ({safe_message(exc)})"


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
        SeaError: Propagated from :func:`sea_description` when the
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
    return sea_description(load_sea(sea_path))


NAMED_COMMANDS_HEADING = "# Slash commands the user named"
"""The heading of the block :func:`named_commands_help` returns."""


def named_commands_help(task: str, earlier: Iterable[str] = ()) -> str:
    """Return the ``/xxx help`` text of every registered command *task* or *earlier* name.

    A task told to "use /discover" otherwise sees only the name.  A text
    names command ``xxx`` with ``/xxx`` at its start or after whitespace,
    an opening bracket, a quote or Markdown emphasis; ``~/xxx``,
    ``a/xxx``, ``/xxx/``, ``/xxx.py``, ``/xxx?q=1``, ``/xxx#a`` and the
    Markdown link target ``[t](/xxx)`` do not.

    Args:
        task: The task.  One that invokes a registered command itself (a
            bare ``/xxx`` that fell back to a plain prompt) gets no
            block, so that SEA file stays executed once.
        earlier: Earlier tasks of the chat.

    Returns:
        A prompt block: :data:`NAMED_COMMANDS_HEADING`, how to run a
        command, and one ``- `/xxx`: <description()>`` line per named
        command in order of first mention; ``""`` when none is named.
        A command whose SEA fails to load is left out.
    """
    texts = [task, *earlier]
    names = dict.fromkeys(name for text in texts for name in _MENTION_RE.findall(text))
    if not names:
        return ""
    registry = dict(registered_entries())
    invoked = _split_slash_command(task)
    if invoked is not None and invoked[0] in registry:
        return ""
    lines = []
    for name in names:
        if name in registry:
            try:
                lines.append(f"- `/{name}`: {sea_description(load_sea(registry[name]))}")
            except SeaError:
                continue
    if not lines:
        return ""
    return "\n".join([
        NAMED_COMMANDS_HEADING,
        'Each is an agent; run one with run_agent(agent="<name>", task="...") when the '
        "task calls for it.  What each does:",
        *lines,
    ])


def slash_command(prompt: str) -> tuple[str, Path] | None:
    """Split a ``/xxx text`` prompt into the SEA to run and its task.

    The daemon runs the SEA directly on the trailing text — the same
    run ``run_agent(agent="xxx", task=text)`` makes, with the SEA's
    ``settings()`` and getters applied by ``apply_sea``.  The SEA file
    is not executed here: a bare ``/xxx`` comes back as ``("", path)``
    and the caller, which loads the SEA anyway, decides whether its
    ``prompt()`` supplies a default task (see :func:`slash_command_task`).

    Args:
        prompt: The raw user prompt (as submitted by the client).

    Returns:
        ``(task_text, sea_path)`` when *prompt* starts with a registered
        command followed by text other than ``help`` or ``check`` (which
        :func:`help_text_if_command` answers), or by nothing; else
        ``None`` — the prompt runs as an ordinary task.
    """
    if not isinstance(prompt, str):
        return None
    parsed = _split_slash_command(prompt)
    if parsed is None:
        return None
    command, task_text = parsed
    if task_text.lower() in RESERVED_SUBCOMMANDS:
        return None
    sea_path = get_command(command)
    if sea_path is None:
        return None
    return task_text, sea_path


def slash_command_task(prompt: str) -> tuple[str, Path] | None:
    """:func:`slash_command`, with a bare ``/xxx`` resolved by executing its SEA.

    A bare ``/xxx`` runs the SEA on the empty task when its ``prompt()``
    turns that into text (the SEA has a default task, as ``/sync`` has);
    a SEA that passes the empty task through (the stock
    :class:`BaseSea`) has nothing to run, so the bare command stays an
    ordinary prompt.  A ``prompt()`` should append its default to the
    task rather than substitute it for an empty one, so a caller that
    prefixes text of its own (the cron preamble) keeps the default too.

    Args:
        prompt: The raw user prompt (as submitted by the client).

    Returns:
        As :func:`slash_command`, except ``None`` for a bare command
        whose SEA has no default task.

    Raises:
        SeaError: Propagated from :func:`load_sea` / :func:`base_prompt`
            when the SEA of a bare command is broken.
    """
    parsed = slash_command(prompt)
    if parsed is not None and not parsed[0] and not base_prompt([load_sea(parsed[1])], ""):
        return None
    return parsed


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
