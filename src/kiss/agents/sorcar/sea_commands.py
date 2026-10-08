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
:func:`base_tool_call_hook`, :func:`base_llm_call_hook`) apply the
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
subscriber list; ``_notify_lock`` wraps the publish-and-notify section
of :func:`refresh_registry` so subscribers see snapshots in the order
they were published.  Nothing calls :func:`refresh_registry` while
holding ``_lock``.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from kiss.agents.seas.base.base_sea import BaseSea, ChannelSea
from kiss.agents.sorcar.sea_settings import (
    SeaError,
    declared_bases,
    declares_hidden,
    execute_python_file,
    resolve_settings,
    safe_message,
    script_name,
)
from kiss.core.config import kiss_home
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
        only names outside ``[A-Za-z0-9_-]``, the :data:`BASE_FOLDER`
        and scripts whose ``settings`` declare ``"hidden": True``
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
    # Built-in SEAs and the bundled ``seas/`` first: any
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


def load_sea(sea_path: Path) -> BaseSea:
    """Execute the SEA file at *sea_path* and return an instance of the SEA class it defines.

    The file is executed with
    :func:`kiss.agents.sorcar.sea_settings.execute_python_file` (so
    every call observes its current contents); the SEA class is the
    one subclass of :class:`BaseSea` the file itself defines (an
    imported base class does not count).  The instance's ``path`` is
    *sea_path*.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Raises:
        SeaError: When the file cannot be read, compiled or
            executed (whatever it raises), defines no or several
            subclasses of :class:`BaseSea`, or its class cannot be
            instantiated without arguments.
    """
    namespace = execute_python_file(str(sea_path))
    classes = [
        value for value in namespace.values()
        if isinstance(value, type) and issubclass(value, BaseSea) and value is not BaseSea
        and value.__module__ == namespace["__name__"]
    ]
    if len(classes) != 1:
        names = ", ".join(cls.__name__ for cls in classes) or "none"
        raise SeaError(
            f"SEA {str(sea_path)!r} must define exactly one subclass of BaseSea "
            f"(kiss.agents.seas.base.base_sea); found {names}"
        )
    try:
        sea = classes[0]()
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        raise SeaError(
            f"SEA {str(sea_path)!r}: {classes[0].__name__}() raised: {safe_message(exc)}"
        ) from exc
    sea.path = sea_path
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
        path = Path(spec).expanduser()
        if not path.is_absolute() and relative_to:
            path = Path(relative_to) / path
        if path.suffix != ".py" or not path.is_file():
            raise SeaError(f"sea_class({spec!r}): not an existing Python (.py) file")
    else:
        found = get_command(spec)
        if found is None:
            raise SeaError(
                f"sea_class({spec!r}): not a registered SEA command; "
                f"known commands: {', '.join(list_commands()) or 'none'}"
            )
        path = found
    return type(load_sea(path.resolve()))


def sea_layers(sea_path: Path, base: Path | None = None) -> list[BaseSea]:
    """Load the SEA at *sea_path*, under the model-picker SEA *base* when there is one.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.
        base: Path of a SEA whose methods apply before *sea_path*'s
            (the model-picker SEA of the tab a run was submitted from),
            or ``None``.

    Returns:
        The loaded SEAs, *base* first; one entry when *base* is
        ``None`` or the same file.

    Raises:
        SeaError: When a file is broken (:func:`load_sea`) or has
            malformed settings.
    """
    seas = [load_sea(sea_path)]
    if base is not None and base != sea_path:
        seas.insert(0, load_sea(base))
    base_settings(seas)  # malformed settings fail at load time
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
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    file = getattr(module, "__file__", None)
    return Path(file) if file else None


def _bound(sea: BaseSea, cls: type, name: str) -> Callable[..., Any]:
    """Return the method *name* of *cls* bound to *sea*; a non-function attribute is an error."""
    attr = vars(cls)[name]
    if not callable(attr) or isinstance(attr, type):
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


def _call(method: Callable[..., Any], *args: Any) -> Any:
    """Call a bound SEA method; whatever it raises becomes a :exc:`SeaError`."""
    try:
        return method(*args)
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        logger.warning("%s raised", _label(method), exc_info=True)
        raise SeaError(f"{_label(method)} raised: {safe_message(exc)}") from exc


def _fold[T](seas: list[BaseSea], name: str, value: T, check: Callable[[str, Any], T]) -> T:
    """Thread *value* through every *name* method of *seas*, checking each result with *check*."""
    for method in _chain(seas, name):
        value = check(_label(method), _call(method, value))
    return value


def declared_settings(seas: list[BaseSea]) -> dict[str, Any]:
    """Return the settings *seas* declare: every ``settings`` method folded over ``{}``, base first.

    What a SEA writes before its base class's defaults are laid under it.  A
    ``work_dir`` is kept as written: the launcher anchors a relative one
    at the calling task's directory
    (:func:`kiss.agents.sorcar.sea_settings.anchored_work_dir`), as it
    does a call's ``work_dir`` option.

    Raises:
        SeaError: When a ``settings`` method raises or returns no dict.
    """
    settings: dict[str, Any] = {}
    for method in _chain(seas, "settings"):
        settings = _check_dict(_label(method), _call(method, settings))
    return settings


def own_settings(sea: BaseSea) -> dict[str, Any]:
    """Return what the SEA's own class (not its bases) writes in ``settings``, over ``{}``."""
    if "settings" not in vars(type(sea)):
        return {}
    method = _bound(sea, type(sea), "settings")
    return _check_dict(_label(method), _call(method, {}))


def inherited_settings(sea: BaseSea) -> dict[str, Any]:
    """Return what the SEA's base classes write in ``settings``, over ``{}``, base first.

    The values a key of :func:`own_settings` repeats when it equals one
    of these (the linter's ``redundant-key`` finding).
    """
    settings: dict[str, Any] = {}
    for cls in reversed(type(sea).__mro__[1:]):
        if "settings" in vars(cls):
            method = _bound(sea, cls, "settings")
            settings = _check_dict(_label(method), _call(method, settings))
    return settings


def is_channel(seas: list[BaseSea]) -> bool:
    """Return whether the run of *seas* is a channel (a SEA derives from ``ChannelSea``)."""
    return any(isinstance(sea, ChannelSea) for sea in seas)


def base_settings(seas: list[BaseSea]) -> dict[str, Any]:
    """Return the effective settings of *seas*: the declared ones resolved.

    :func:`declared_settings` through
    :func:`kiss.agents.sorcar.sea_settings.resolve_settings` (type
    checks, channel locks for a :func:`is_channel` run).

    Raises:
        SeaError: When a ``settings`` method raises or returns no
            dict, or the settings are malformed.
    """
    declared = declared_settings(seas)
    try:
        return resolve_settings(declared, is_channel(seas))
    except SeaError as exc:
        raise SeaError(f"SEA {str(seas[-1].path)!r}: {exc}") from exc


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


def base_system_prompt(seas: list[BaseSea], system_prompt: str) -> str:
    """Return *system_prompt* after every ``system_prompt`` method of *seas*, base first.

    Like :func:`base_prompt`: each method receives the text so far and
    what it returns is the text, whether it appended to it or replaced
    it.  The run uses the last return verbatim.

    Raises:
        SeaError: When a ``system_prompt`` method raises or
            returns anything but a string.
    """
    return _fold(seas, "system_prompt", system_prompt, _check_text)


def base_tools(seas: list[BaseSea], tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """Return *tools* after every ``tools`` method of *seas*, base first (a copy is passed)."""
    return _fold(seas, "tools", list(tools), _check_tools)


def base_tool_call_hook(seas: list[BaseSea], name: str, args: dict[str, Any]) -> Verdict:
    """Return the first refusing verdict a ``tool_call_hook`` of *seas* gives, else ``ALLOW``.

    Every hook returns a :class:`~kiss.core.tool_verdict.Verdict`
    (``ALLOW`` or ``refuse(text)``); anything else — ``None``, a
    string — is a broken SEA (``uv run sea lint`` flags such returns as
    ``verdict``).
    """
    for method in _chain(seas, "tool_call_hook"):
        verdict = _check_verdict(_label(method), _call(method, name, args))
        if not verdict.allowed:
            return verdict
    return ALLOW


def base_llm_call_hook(seas: list[BaseSea], new_messages: list[Any]) -> list[Any]:
    """Return *new_messages* after every ``llm_call_hook`` method of *seas*, base first."""
    return _fold(seas, "llm_call_hook", new_messages, _check_list)


@dataclass
class SeaRun:
    """What a run takes from its SEAs (:func:`evaluate_sea`).

    Attributes:
        settings: The effective settings (:func:`base_settings`).
        prompt: The task text after every ``prompt`` method,
            ``{task_id}`` replaced by the calling task's id.
        system_prompt_hook: ``system_prompt -> system_prompt``
            (:func:`base_system_prompt`).
        tools_hook: ``tools -> tools`` (:func:`base_tools`).
        llm_call_hook: ``new_messages -> messages``
            (:func:`base_llm_call_hook`).
        tool_call_hook: ``(name, args) -> verdict``
            (:func:`base_tool_call_hook`).
    """

    settings: dict[str, Any]
    prompt: str
    system_prompt_hook: Callable[[str], str]
    tools_hook: Callable[[list[Any]], list[Any]]
    llm_call_hook: Callable[[list[Any]], list[Any]]
    tool_call_hook: Callable[[str, dict[str, Any]], Verdict]


def evaluate_sea(seas: list[BaseSea], task: str, task_id: str = "") -> SeaRun:
    """Return the configuration of a run of *seas* on *task*.

    The settings and the prompt are computed now; the system prompt,
    tools and the two hooks are returned as callables for the run to
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
    for name in ("system_prompt", "tools", "llm_call_hook", "tool_call_hook"):
        _chain(seas, name)  # a non-method attribute fails now, not at the hook's first call
    return SeaRun(
        base_settings(seas),
        base_prompt(seas, task, task_id),
        functools.partial(base_system_prompt, seas),
        functools.partial(base_tools, seas),
        functools.partial(base_llm_call_hook, seas),
        functools.partial(base_tool_call_hook, seas),
    )


def _check_verdict(label: str, value: Any) -> Verdict:
    """Return *value* as a :class:`Verdict`, or raise :exc:`SeaError`."""
    if not isinstance(value, Verdict):
        raise SeaError(
            f"{label} must return a Verdict (ALLOW, or refuse(text), from "
            f"kiss.agents.seas.base.base_sea), got {type(value).__name__}"
        )
    return value


def _check_text(label: str, value: Any) -> str:
    """Return *value* as a plain string, or raise :exc:`SeaError`.

    A ``str`` subclass from an untrusted script is copied into an exact
    ``str`` (its own methods may raise), so later ``strip()``/joins
    cannot run script code.
    """
    if not isinstance(value, str):
        raise SeaError(f"{label} must return a string, got {type(value).__name__}")
    try:
        return str(value)
    except BaseException as exc:  # noqa: BLE001 — an untrusted str subclass may raise
        raise SeaError(f"{label} returned a broken value: {safe_message(exc)}") from exc


def _check_dict(label: str, value: Any) -> dict[str, Any]:
    """Return *value* as a dict, or raise :exc:`SeaError`."""
    if not isinstance(value, dict):
        raise SeaError(f"{label} must return a dict, got {type(value).__name__}")
    return value


def _check_list(label: str, value: Any) -> list[Any]:
    """Return *value* as a list, or raise :exc:`SeaError`."""
    if not isinstance(value, list):
        raise SeaError(f"{label} must return a list, got {type(value).__name__}")
    return value


def _check_tools(label: str, value: Any) -> list[Callable[..., Any]]:
    """Return *value* as a list of tool callables, or raise :exc:`SeaError`."""
    try:
        if isinstance(value, list) and all(callable(tool) for tool in value):
            return value
    except BaseException as exc:  # noqa: BLE001 — an untrusted list may raise while iterated
        raise SeaError(f"{label} returned a broken list: {safe_message(exc)}") from exc
    raise SeaError(
        f"{label} must return a list of tool callables (not a file path), "
        f"got {type(value).__name__}"
    )


PICKED_HOOK_TIMEOUT_SECONDS = 15.0
"""How long :func:`run_picked_hook` waits for ``on_picked_as_model()`` before moving on."""


def _log_picked_hook(model: str, work_dir: str, sea: BaseSea | None) -> None:
    """Run ``on_picked_as_model(work_dir)`` of the SEA picked as *model*; log note or failure."""
    sea_path = model_sea(model)
    if sea_path is None:
        return
    try:
        if sea is None:
            sea = load_sea(sea_path)
        note = _call(_method(sea, "on_picked_as_model"), work_dir)
    except SeaError:
        logger.warning("SEA %s: on_picked_as_model() failed", model, exc_info=True)
        return
    if note:
        logger.info("SEA %s picked as model: %s", model, note)


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
        sea: The already-loaded SEA (a run loads the file once and
            shares it); ``None`` loads the file.
    """
    thread = threading.Thread(
        target=_log_picked_hook, args=(model, work_dir, sea),
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


_PLAIN_BASES = frozenset({"BaseSea", "WorkerSea", "ChannelSea"})
"""The base classes that register as no model; a class deriving from anything else may."""


def _registers_as_model(sea_path: Path) -> bool:
    """Return whether the SEA at *sea_path* defines ``register_as_model()`` returning ``True``.

    The verdict is cached per file stamp (mtime, size, inode), so both an
    edit and an atomic replacement are re-read.  Only a script whose
    source mentions ``register_as_model`` or derives its class from
    something other than the plain bases (so it may inherit the
    registration; :func:`~kiss.agents.sorcar.sea_settings.declared_bases`)
    is imported: importing every registered SEA (channel agents with
    heavy dependencies among them) on each picker refresh would be slow
    for nothing.  A script that cannot be read, fails to import or whose
    method raises is logged and treated as not registered, so one
    broken SEA cannot break the model picker.
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
    if "register_as_model" in source or declared_bases(sea_path) - _PLAIN_BASES:
        try:
            verdict = _call(_method(load_sea(sea_path), "register_as_model")) is True
        except SeaError:
            logger.warning("SEA %s: register_as_model() failed", sea_path, exc_info=True)
    with _lock:
        _model_sea_cache[sea_path] = (stamp, verdict)
    return verdict


def model_seas() -> dict[str, Path]:
    """Return the registered SEAs that are model-picker entries, by command name.

    A SEA is a model-picker entry when its ``register_as_model()``
    returns ``True``.  The daemon offers each entry in the model picker
    under its command name; a task run with such a pick goes through
    the SEA (``task_runner._resolve_sea_model``).

    Returns:
        ``{command_name: absolute_sea_path}``, sorted by name.
    """
    list_commands()  # populate the registry on a cold start
    with _lock:
        entries = sorted(_registry.items())
    return {name: path for name, path in entries if _registers_as_model(path)}


def model_sea(name: str) -> Path | None:
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

    Returns:
        The absolute SEA path when *name* is a registered command whose
        ``register_as_model()`` returns ``True``, else ``None``.
    """
    if not name:
        return None
    list_commands()  # populate the registry on a cold start
    with _lock:
        path = _registry.get(name)
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
    run makes it run: the result carries the settings and tools so the
    callers need not run them again.

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
            "register_as_model", "on_picked_as_model",
        ) if defines(seas, method)
    ]
    lines = [
        f"/{name}: {check.description}",
        "classes: " + " > ".join(classes),
        "settings: " + (json.dumps(settings, sort_keys=True, default=str) or "{}"),
        f"model: {model}",
        "tools added: " + (", ".join(_tool_name(tool) for tool in check.tools) or "none"),
        "methods defined: " + (", ".join(defined) or "none"),
        f"prompt for {CHECK_SAMPLE_TASK}: {check.cmd['prompt']}",
    ]
    if "description" not in vars(type(seas[-1])):
        lines.append("note: description() comes from a base class")
    return "\n".join(lines)


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


def slash_command_task(prompt: str) -> tuple[str, Path] | None:
    """Split a ``/xxx text`` prompt into the SEA to run and its task.

    The daemon runs the SEA directly on the trailing text — the same
    run ``run_agent(agent="xxx", task=text)`` makes, with the SEA's
    ``settings()`` and getters applied by ``apply_sea``.

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
