# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slash-command registry for Sorcar Extension Agents (SEAs).

Every ``*_sea.py`` script visible to the daemon is exposed as a chat
command named ``/<stem>`` (stem = filename minus ``_sea.py``).  When a
user submits a prompt that starts with ``/xxx`` — optionally followed
by whitespace and free-form text — the daemon rewrites the prompt so
the agent immediately calls ``run_agent`` with the absolute path of
the resolved ``xxx_sea.py`` and the trailing text as the sub-task.

The registry is built from three sources, in decreasing precedence:

1. ``src/kiss/agents/third_party_agents/`` (highest precedence,
   discovered through the ``kiss.agents.third_party_agents`` package).
2. The folders listed one per line in ``~/.kiss/SEAS.md`` (or
   ``$KISS_HOME/SEAS.md``).  Later lines in the file override earlier
   lines — i.e. the folder at the bottom of ``SEAS.md`` beats the one
   at the top when both contain the same command name.  Blank lines
   and lines starting with ``#`` are ignored; ``~`` and environment
   variables in a folder path are expanded.
3. ``src/kiss/agents/seas/`` (lowest precedence: the bundled SEAs that
   extend Sorcar itself, e.g. ``/merge``; discovered through
   ``kiss.agents.seas``).  Any ``SEAS.md`` folder that ships a file of
   the same name replaces the bundled one.

The registry is refreshed lazily on every lookup and, in the daemon,
proactively by a background polling watcher (see
:func:`start_registry_watcher`) so edits to ``SEAS.md`` — or the
appearance/removal of ``*_sea.py`` files in any of its folders — take
effect while the daemon is running.

Locking: two module locks, always acquired in the order
``_notify_lock`` -> ``_lock``.  ``_lock`` guards the registry and the
subscriber list; ``_notify_lock`` wraps the publish-and-notify section
of :func:`refresh_registry` so subscribers see snapshots in the order
they were published.  Nothing calls :func:`refresh_registry` while
holding ``_lock``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import re
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

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

# Watcher-thread coordination.
_watcher_thread: threading.Thread | None = None
_watcher_stop = threading.Event()


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


def _scan_folder(folder: Path) -> dict[str, Path]:
    """Return ``{command_name: absolute_path}`` for every SEA in *folder*.

    Silently skips folders that are missing, unreadable, or not a
    directory: a stale ``SEAS.md`` entry must not break the daemon.

    Args:
        folder: The directory to scan.

    Returns:
        Mapping from command name (filename minus ``_sea.py``) to the
        absolute, resolved SEA-script path.  An underscore-prefixed
        stem is a valid command (``_helper_sea.py`` becomes
        ``/_helper``); only stems outside ``[A-Za-z0-9_-]`` are
        skipped.
    """
    out: dict[str, Path] = {}
    try:
        entries = list(folder.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return out
    for path in entries:
        name = path.name
        if not name.endswith(_SEA_SUFFIX):
            continue
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        stem = name[: -len(".py")]
        command = stem[: -len("_sea")]
        # Reject stems that would produce a command name outside the
        # ``[A-Za-z0-9_-]`` alphabet the parser and the autocomplete
        # accept — ``foo.bar_sea.py`` and ``space name_sea.py`` are
        # silently skipped rather than surfacing as commands that
        # cannot be typed.
        if not command or not _COMMAND_NAME_RE.match(command):
            continue
        try:
            out[command] = path.resolve()
        except OSError:
            out[command] = path.absolute()
    return out


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
    2. Folders in ``~/.kiss/SEAS.md``, from bottom line to top line.
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
    # Bundled ``seas/`` first: any SEAS.md folder may shadow it.
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


@contextlib.contextmanager
def _load_sea_module(sea_path: Path) -> Iterator[ModuleType]:
    """Import the SEA script at *sea_path* as a standalone module.

    Mirrors the daemon, which executes the agent file named by
    ``extension_agent_path`` and reads its ``X()`` getters from the
    resulting namespace.  Loading by path (rather than by dotted module
    name) keeps ``kiss.agents.sorcar`` free of any static or literal
    dependency on ``kiss.agents.third_party_agents``, which the
    layering invariants forbid.

    The module is registered in ``sys.modules`` before it executes,
    exactly as ``import`` does: ``@dataclass`` under ``from __future__
    import annotations`` (and ``typing.get_type_hints`` later on)
    resolve string annotations through
    ``sys.modules[cls.__module__].__dict__``, so an unregistered module
    makes every dataclass-bearing SEA fail with ``AttributeError:
    'NoneType' object has no attribute '__dict__'``.  The name is
    unique per load (``_kiss_sea_<stem>_<uuid>``): task threads load
    SEAs concurrently, and two same-stem files (or two loads of one
    file) sharing a name would overwrite each other's entry mid-use.
    The entry is removed when the ``with`` block ends, so a long-lived
    daemon does not accumulate one module per relay.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.

    Yields:
        The freshly executed module, registered for the block's duration.
    """
    name = f"_kiss_sea_{sea_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, sea_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


class SeaScriptError(RuntimeError):
    """An SEA script failed to import or one of its getters raised.

    Raised by :func:`sea_getter_is_false` with the original raise as
    ``__cause__`` — ``BaseException`` included, so an SEA raising
    ``KeyboardInterrupt``/``SystemExit`` at import time is reported as
    a broken script, not as a cancelled task, while a genuinely
    requested stop that landed inside the import stays recognisable
    through the cause chain (``task_runner._stop_interrupt_wrapped``).
    """


def sea_getter_is_false(sea_path: Path, getter: str) -> bool:
    """Return whether the SEA at *sea_path* defines ``getter()`` returning ``False``.

    Used by the task runner on the OUTER run of a ``/xxx`` command —
    the relay that calls ``run_agent`` with its own work directory —
    to honour an SEA's ``use_worktree()`` / ``auto_commit()`` verdicts
    on that relay as well: an SEA that declares it works on the real
    checkout (``/sh``, ``/merge``) must not be handed the relay's
    worktree, and the relay must not auto-commit what such an SEA left
    in the tree.

    Args:
        sea_path: Absolute path of the SEA ``.py`` file.
        getter: Name of the zero-argument getter, e.g. ``"use_worktree"``.

    Returns:
        ``True`` only when the script defines a callable *getter* and
        it returns exactly ``False``; a missing getter or any other
        value yields ``False``.

    Raises:
        SeaScriptError: When the script fails to import or *getter*
            raises (whatever it raises), so the relay fails with the
            diagnostic instead of running against a broken SEA.
    """
    try:
        with _load_sea_module(sea_path) as module:
            fn = getattr(module, getter, None)
            return callable(fn) and fn() is False
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        raise SeaScriptError(
            f"SEA {sea_path} failed while evaluating {getter}(): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def rewrite_prompt_if_command(prompt: str) -> tuple[str, Path] | None:
    """Rewrite a slash-command prompt into an explicit ``run_agent`` call.

    When *prompt* starts with a registered ``/xxx`` command, returns
    ``(rewritten_prompt, sea_path)`` where ``rewritten_prompt``
    instructs the calling agent to invoke the ``run_agent`` tool
    immediately with the SEA's absolute path and the user's trailing
    text as the sub-task.  Returns ``None`` when the prompt does not
    begin with a slash command, when the command is unknown, or when
    the trailing text is empty (an empty ``run_agent`` task would be
    rejected downstream).

    Args:
        prompt: The raw user prompt (as submitted by the client).

    Returns:
        ``(rewritten_prompt, absolute_sea_path)`` on a hit, else
        ``None``.
    """
    if not isinstance(prompt, str):
        return None
    parsed = _split_slash_command(prompt)
    if parsed is None:
        return None
    command, task_text = parsed
    if not task_text:
        return None
    sea_path = get_command(command)
    if sea_path is None:
        return None
    abs_path = str(sea_path)
    if command == "ask":
        # ``/ask <question>`` is a fixed side-channel Q&A over the
        # calling task's persisted events: the two ``append_to_*``
        # arguments must reach ``run_agent`` unchanged.  ``<task_id>``
        # is left as a literal placeholder here — the calling task's
        # id is not known until the daemon dispatch allocates one, so
        # ``_dispatch_reserved`` substitutes it into
        # ``append_to_prompt`` right before the daemon round trip.
        # The system-prompt suffix is owned by ``ask_sea.py`` (its
        # ``append_to_system_prompt()`` getter also overrides the
        # wire value daemon-side), so it is read from the resolved SEA
        # file itself.
        append_to_prompt = (
            "Read the events of the task <task_id> from "
            "~/.kiss/sorcar.db and answer the user question above."
        )
        with _load_sea_module(sea_path) as module:
            append_to_system_prompt = module.append_to_system_prompt()
        rewritten = (
            f"The user invoked the slash command /ask.  Call the "
            f"run_agent tool IMMEDIATELY, as your very first action, "
            f"with these arguments and no others:\n"
            f'  agent = "{abs_path}"\n'
            f"  task  = the text below, verbatim\n"
            f'  append_to_prompt = "{append_to_prompt}"\n'
            f'  append_to_system_prompt = "{append_to_system_prompt}"\n'
            f"Do not modify these arguments, do not explore any source "
            f"code, do not paraphrase the task, and do not call any "
            f"other tool first.  When run_agent returns, relay its "
            f"result to the user verbatim as your final answer.\n\n"
            f"TASK TEXT FOR run_agent:\n{task_text}"
        )
        return rewritten, sea_path
    # A directive, not a suggestion: the agent's routing rules already
    # tell it to prefer ``run_agent`` for channel-style work, and this
    # phrasing removes every reason to explore anything else first.
    rewritten = (
        f"The user invoked the slash command /{command}.  Call the "
        f"run_agent tool IMMEDIATELY, as your very first action, with "
        f"these arguments and no others:\n"
        f'  agent = "{abs_path}"\n'
        f"  task  = the text below, verbatim\n"
        f"Do not explore any source code, do not paraphrase the task, "
        f"and do not call any other tool first.  When run_agent "
        f"returns, relay its result to the user.\n\n"
        f"TASK TEXT FOR run_agent:\n{task_text}"
    )
    return rewritten, sea_path


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
    global _watcher_thread
    interval = max(0.1, min(60.0, float(poll_interval)))
    with _lock:
        if _watcher_thread is not None and _watcher_thread.is_alive():
            return
        _watcher_stop.clear()
        thread = threading.Thread(
            target=_watcher_loop,
            args=(interval,),
            name="kiss-sea-registry-watcher",
            daemon=True,
        )
        _watcher_thread = thread
    # Always emit the first refresh synchronously so the very first
    # ``list_commands`` from a client cannot race the watcher's first
    # tick, then hand ongoing rescans to the poller.
    try:
        refresh_registry()
    except Exception:  # pragma: no cover - registry rescan is best-effort
        logger.debug("initial SEA registry refresh failed", exc_info=True)
    thread.start()


def stop_registry_watcher(timeout: float = 5.0) -> None:
    """Stop the background poller (used by the daemon on shutdown).

    Signals the watcher to exit and joins it within *timeout* seconds;
    a poller that outlives the timeout is left as a daemon thread
    (daemon threads die with the interpreter).  Idempotent.
    """
    global _watcher_thread
    with _lock:
        thread = _watcher_thread
        _watcher_thread = None
    if thread is None:
        return
    _watcher_stop.set()
    thread.join(timeout=timeout)


def _watcher_loop(interval: float) -> None:
    """Body of the background poller thread.

    Rebuilds the registry every *interval* seconds until
    :attr:`_watcher_stop` is signalled.  Every rescan is wrapped in
    ``try/except`` so a transient filesystem error (e.g. a folder
    that appears mid-edit) cannot kill the watcher.
    """
    while not _watcher_stop.wait(interval):
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
        _last_broadcast = ()
