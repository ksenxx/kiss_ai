# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The settings of a SEA and the loader that executes one.

A SEA is a class deriving from
:class:`kiss.agents.seas.base.base_sea.BaseSea` — directly for an
ordinary session, through :class:`~kiss.agents.seas.base.base_sea.WorkerSea`
for a focused tool-bound run on the caller's tree, through
:class:`~kiss.agents.seas.base.base_sea.ChannelSea` for an agent that
serves an external service (its behaviours are
:data:`CHANNEL_BEHAVIOURS`).  Its run parameters come from its
``settings`` method, which lays the SEA's keys over what its base
classes declared::

    def settings(self, settings: dict) -> dict:
        return settings | {"tool_profile": "bash", "max_budget": 1.0}

The result is data: any of the per-run parameters of
:func:`kiss.server.sorcar.run` listed in :data:`SETTING_TYPES`, and
three keys read by the launcher: ``timeout`` (seconds a ``run_agent``
/ ``run_parallel`` call blocks for the run: the call's value, else
this setting, else 3600), ``locked`` (the keys an explicit caller
argument may not replace) and ``hidden`` (the file is no command).
Against the caller there is one precedence rule,
:data:`PRECEDENCE_RULE`, enforced by :func:`locked_conflicts` and
rendered by ``sea docs`` into every page that states it.  A relative
``work_dir`` — a SEA's setting or a call's option — is a path under
the calling task's directory (:func:`anchored_work_dir`).

The other methods of the class — ``prompt``, ``system_prompt``,
``tools``, ``tool_call_hook``, ``llm_call_hook`` — shape the run;
:mod:`kiss.agents.sorcar.sea_commands` loads the class and applies
them.

A SEA runs in one of three ways; ``run_parallel`` is N ``run_agent``
calls, so what differs is only the tab and the inheritance:

==================  ====================  ==========================================
                    ``/<name> task``      ``run_agent(agent=)`` / ``run_parallel``
==================  ====================  ==========================================
Where               the tab's own run     a daemon sub-task in its own tab
Settings honoured   all but ``timeout``   all
Parent inheritance  none (tab settings)   yes, unless the SEA is a channel or the
                                          ``inherit`` option is ``false``; budget =
                                          remaining / (N+1) for N sub-tasks
``timeout``         none                  argument > setting > 3600: the seconds the
                                          call blocks before returning the job ids
==================  ====================  ==========================================

The base class is the one thing with behaviour beyond a value passed
through — every behaviour of a channel is listed in
:data:`CHANNEL_BEHAVIOURS`.

:func:`execute_python_file` is the ONE loader every reader of a SEA
file uses: it compiles and executes the file into a throw-away module,
so every run observes the file's current contents.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import logging
import math
import sys
import threading
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS, BaseSea, ChannelSea, WorkerSea
from kiss.agents.sorcar._concurrency import locked_call
from kiss.core.stop_signal import stop_requested
from kiss.core.tool_interrupt import ToolCallInterrupted

logger = logging.getLogger(__name__)

PRECEDENCE_RULE = (
    "For every setting of a sub-task: what the call passes explicitly (a `run_agent` / "
    "`run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what "
    "the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), "
    "then the user's defaults. A SEA may list keys in `locked`: a call that passes a "
    "different value for a locked key is refused with an error, never silently overruled."
)
"""The one precedence rule of SEA settings, stated once.

:func:`locked_conflicts` enforces the ``locked`` clause; ``sea docs``
renders the sentence into the ``<!-- sea-docs: precedence -->`` block of
every documentation page, and the ``run_agent`` / ``run_parallel`` tool
docstrings quote it, so a change here changes every statement of it.
"""

SETTING_TYPES: dict[str, type | tuple[type, ...]] = {
    "work_dir": str,
    "model": str,
    "chat_id": str,
    "max_budget": (int, float),
    "model_config": dict,
    "use_web_tools": bool,
    "auto_classify": bool,
    "use_memory": bool,
    "tool_profile": str,
    "docker_image": str,
    "timeout": (int, float),
    "locked": list,
    "hidden": bool,
}
"""Every key ``settings()`` may return, with the type its value must have."""

SETTING_DOCS: dict[str, str] = {
    "work_dir": "The directory the run works in; default: the calling task's or the tab's. A "
                "relative path is a path under the calling task's directory (the tab's for "
                "`/<name>`), in a SEA's settings and in a call's `work_dir` option alike.",
    "model": "The LLM model, a catalogue name or a model-picker SEA; default: the caller's "
             "(`\"\"`, like `None`, is no override — true of every string key).",
    "chat_id": "The chat the run's events go to; default under `run_agent`: the calling "
               "task's chat, or a new chat when nothing is inherited (a channel, an "
               "`inherit: false` call); a `/<name>` run keeps the tab's chat.",
    "max_budget": "USD budget of the run, a finite number; default: the caller's share or the "
                  "daemon's default.",
    "model_config": "Model configuration dict passed to the LLM (temperature, base URL, ...).",
    "use_web_tools": "Give the run the browser tools (daemon default: on).",
    "auto_classify": "Let the pre-run classifier decide whether the task is simple (lite "
                     "prompt) (daemon default: the persisted setting).",
    "use_memory": "Give the run the `memory_*` tools (daemon default: the persisted setting).",
    "tool_profile": "The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full "
                    "toolset).",
    "docker_image": "Run inside this Docker image (default: the host).",
    "timeout": "Seconds a `run_agent` / `run_parallel` call blocks for the run: the call's "
               "`timeout` argument wins, then this setting, then 3600. When it expires the "
               "run is not stopped: it keeps going as an `agent_job` and the call returns "
               "its job id (a job still running when the calling task ends is killed). "
               "Ignored by `/<name>`.",
    "locked": "Keys an explicit `run_agent` / `run_parallel` argument or option may not change: "
              "a differing value is an error.",
    "hidden": "`True`: the script is no `/command` and no `run_agent` agent name (loadable by "
              "path and as a base class only). It must be the literal `True` in the class's "
              "`settings()` because the command registry reads it from the source without "
              "running the script (a computed value is ignored).",
}
"""One line of documentation per :data:`SETTING_TYPES` key (rendered by ``sea docs``)."""

META_SETTINGS = ("locked", "hidden")
"""Keys that shape the settings themselves rather than the run; never lockable."""

DISPATCHER_SETTINGS = ("timeout", "locked", "hidden")
"""``settings()`` keys with no ``run`` command wire field.

``timeout`` is read by the dispatcher
(:mod:`kiss.agents.sorcar.agent_dispatch`); ``locked`` names the keys
an explicit caller argument may not replace (:func:`locked_conflicts`);
``hidden`` keeps the script out of the command registry
(:func:`declares_hidden`).  Every other key is a parameter of
:func:`kiss.server.sorcar.run`, sent as :func:`wire_field` of the key.
"""

RENAMED_SETTINGS = {
    "classify_tasks": "auto_classify",
    "model_name": "model",
}
"""Former settings keys and their current names.

``auto_classify`` says whether the daemon's classifier decides whether
the task is simple (the wire field ``classifyTasks``); ``model_name``
is the ``SorcarAgent.run`` parameter callers keep reaching for in
place of the settings key ``model``.  The old names are refused — in
a script's ``settings()`` and in a ``run_agent`` / ``run_parallel``
``options`` object alike — with a message naming the new one; ``sea
lint --fix`` rewrites them in a script.  :func:`kiss.server.sorcar.run`
takes the current names as its keywords, so a key spelled the
settings way is right everywhere.
"""

RENAMED_OPTIONS = {
    **RENAMED_SETTINGS,
    "append_to_prompt": "add_to_prompt",
    "append_to_system_prompt": "add_to_system_prompt",
}
"""Former ``options`` keys of ``run_agent`` / ``run_parallel`` and their current names.

:data:`RENAMED_SETTINGS` plus the two prompt suffixes, which are
options only — ``append_to_prompt`` and ``append_to_system_prompt``
were the former ``run()`` keywords of the options ``add_to_prompt``
and ``add_to_system_prompt`` (the wire fields are still
``appendToPrompt`` and ``appendToSystemPrompt``).  In a script's
``settings()`` those keys are refused as removed
(:data:`REMOVED_SETTINGS`): a SEA shapes the prompts in its ``prompt``
and ``system_prompt`` methods instead.
"""

PROFILE_ALIASES = {"readonly": "review", "read_only": "review", "read-only": "review"}
"""Spellings accepted for a tool-profile key, and the key each means.

``review`` is the read-only profile and callers keep asking for it by
that property.  Every way a profile name arrives — a script's
``tool_profile`` setting (:func:`resolve_settings`), a ``run_agent`` /
``run_parallel`` argument, a ``run()`` parameter
(:func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`) —
replaces the alias with the key first, so locks, the ``pinned`` record
and the ``ran:`` line compare and show keys only.
"""


def alias_free_profile(name: str) -> str:
    """Return the profile *name* with every ``+``-joined part stripped and its alias replaced.

    Unknown parts pass through unchanged: this is spelling, not
    validation (:func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`
    validates against the profile table).
    """
    if not name.strip():
        return ""
    return "+".join(PROFILE_ALIASES.get(p.strip(), p.strip()) for p in name.split("+"))


_PROMPT_SUFFIX_REMOVED = (
    "a SEA shapes the task text in its `prompt(task)` method; `add_to_prompt` is a "
    "`run_agent` option, not a setting"
)
_SYSTEM_PROMPT_SUFFIX_REMOVED = (
    "a SEA shapes the system prompt in its `system_prompt(system_prompt)` method; "
    "`add_to_system_prompt` is a `run_agent` option, not a setting"
)

_WORKTREE_TOOLS_REMOVED = (
    "the agent isolates its own edits: it calls `create_worktree` before its first change "
    "to a tracked file and `commit_and_merge` when done (a `WorkerSea` has neither tool and "
    "edits the tree in place)"
)

_BASE_CLASS_REMOVED = (
    "what a SEA is became its base class: derive from `WorkerSea` (a tool-bound run on the "
    "caller's tree) or `ChannelSea` (an external-service agent) instead of `BaseSea` "
    "(`from kiss.agents.seas.base.base_sea import ...`); run `uv run sea lint --fix` to "
    "rewrite the script"
)

REMOVED_SETTINGS = {
    "kind": _BASE_CLASS_REMOVED,
    "preset": _BASE_CLASS_REMOVED,
    "channel": _BASE_CLASS_REMOVED,
    "allow_fan_out": "`run_parallel` is N `run_agent` calls, so there is nothing to allow or "
                     "forbid separately; `tool_profile` chooses the toolset",
    "is_parallel": "`run_parallel` is N `run_agent` calls, so there is nothing to allow or "
                   "forbid separately; `tool_profile` chooses the toolset",
    "inherit": "a channel never inherits from the calling task and every other SEA always "
               "does; the caller's `inherit` option opts out of inheriting",
    "append_to_prompt": _PROMPT_SUFFIX_REMOVED,
    "add_to_prompt": _PROMPT_SUFFIX_REMOVED,
    "append_to_system_prompt": _SYSTEM_PROMPT_SUFFIX_REMOVED,
    "add_to_system_prompt": _SYSTEM_PROMPT_SUFFIX_REMOVED,
    "append_basic_tools": "the built-in toolset is chosen by `tool_profile` (`none` gives a "
                          "run no built-in tools)",
    "extends": "a SEA extends another by Python inheritance: derive the SEA class from the "
               "base's class (import it, or `sea_class(\"name\")`)",
    "use_worktree": _WORKTREE_TOOLS_REMOVED,
    "auto_commit": _WORKTREE_TOOLS_REMOVED,
}
"""Former settings keys with no replacement, and why; refused with the explanation."""

BASE_CLASS_DOCS: dict[str, str] = {
    "BaseSea": "The default: an ordinary Sorcar session with the caller's or the user's "
               "settings (`/write`, `/write_paper`, `bestrouter`).",
    "WorkerSea": "A focused tool-bound run on the caller's tree as it is: lays the worker "
                 "defaults (no classifier, no browser, no memory) under the SEA's own keys "
                 "and drops the `create_worktree` / `commit_and_merge` tools (`/sh`, `/ask`, "
                 "`/merge`, `/remember`, `/forget`, `/task_update`).",
    "ChannelSea": "A worker that serves an external service (Slack, email, cron), not the "
                  "caller's project; every behaviour this adds is listed in the channel "
                  "table. The command registry reads the base name from the source, so a "
                  "channel derives from `ChannelSea` by that name.",
}
"""One line of documentation per base class (rendered by ``sea docs``)."""


def base_class_defaults() -> dict[str, dict[str, Any]]:
    """Return the settings each base class lays under a subclass's own keys.

    What ``sea docs`` renders next to :data:`BASE_CLASS_DOCS`, computed
    by running each class's own ``settings`` methods over ``{}`` (base
    first, as a run folds them), so the table always says what the
    classes do: :class:`~kiss.agents.seas.base.base_sea.BaseSea` lays
    nothing, :class:`~kiss.agents.seas.base.base_sea.WorkerSea` the
    :data:`WORKER_DEFAULTS`,
    :class:`~kiss.agents.seas.base.base_sea.ChannelSea` those plus the
    scratch ``work_dir``.
    """
    return {cls.__name__: _laid_by(cls) for cls in (BaseSea, WorkerSea, ChannelSea)}


def _laid_by(cls: type[BaseSea]) -> dict[str, Any]:
    """Return what *cls* and its bases write in ``settings`` over ``{}``, base first."""
    sea = cls()
    settings: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if "settings" in vars(base):
            settings = vars(base)["settings"](sea, settings)
    return settings


CHANNEL_BEHAVIOURS: tuple[tuple[str, str], ...] = (
    ("worker",
     "It is a `WorkerSea`, and the worker keys are locked, so no call may give it "
     "the classifier, the browser or memory (`resolve_settings`)."),
    ("scratch directory",
     "It runs in the shared `<home>/channel_work` scratch directory unless its own "
     "`work_dir` says otherwise (`/cron` runs in `<home>/cron/work`); `work_dir` is locked, "
     "so it never works in the caller's project (`resolve_settings`); a cron prompt job "
     "that runs a channel command (`/slack ...`) therefore passes no `work_dir` option "
     "unless the job names a directory of its own (`cron_agent`: the scratch directory is "
     "already the dispatch's default, and an explicit value would be refused by the "
     "lock)."),
    ("no inheritance",
     "Its `run_agent` dispatch takes nothing from the calling task: not the chat, model, "
     "budget share, container or exec bridge, or prompt suffixes; the `inherit: true` "
     "option is refused (`agent_dispatch`)."),
    ("workspace",
     "It holds its channel workspace — the `run_agent` option `workspace`, default "
     "`default`, the account its tools load credentials for — from before its `tools()` "
     "run until the run ends; `workspace` is refused for any other SEA "
     "(`sea_apply.channel_workspace`, `agent_dispatch`)."),
    ("preamble",
     "The channel preamble is appended to its system prompt before its own "
     "`system_prompt()` runs (`sea_apply.CHANNEL_PREAMBLE`)."),
    ("listed as a channel",
     "A third-party SEA folder whose class derives from `ChannelSea` by that name is a "
     "channel agent: `run_agent(agent=\"<folder>\")` finds it by name and the `run_agent` "
     "tool's description lists it among the installed channels "
     "(`agent_dispatch.available_channels`, `declares_channel`)."),
)
"""Every behaviour deriving from ``ChannelSea`` adds, as ``(name, what it does and where it
is enforced)``.

The one place the list exists: ``sea docs`` renders it, and the
modules named enforce exactly these.  A setting has no behaviour
beyond its value.
"""


class SeaError(Exception):
    """The one "this SEA is broken" error: a bad file, settings, method or dispatch.

    Raised by the loader, the settings resolver, the command registry,
    the dispatcher and the daemon's task runner alike, so a caller that
    wants to know "the SEA failed" catches one class and the message
    (prefixed with the SEA's file name where known) is the diagnostic.
    """


def wire_field(key: str) -> str:
    """Return the ``run`` command wire field of the ``run()`` keyword *key*.

    The wire vocabulary is the keyword vocabulary in camelCase
    (``use_web_tools`` -> ``useWebTools``), with one alias kept from
    the wire protocol's earlier vocabulary: ``auto_classify`` ->
    ``classifyTasks``.  This table is the only place the wire spelling
    of a settings key exists on the client side: settings keys,
    ``options`` keys and ``run()`` keywords all use the snake_case
    name.  (The prompt suffixes, which are no settings, are spelled
    ``appendToPrompt`` / ``appendToSystemPrompt`` by ``daemon_client``.)
    """
    if key in _WIRE_ALIASES:
        return _WIRE_ALIASES[key]
    first, *rest = key.split("_")
    return first + "".join(part.capitalize() for part in rest)


_WIRE_ALIASES = {"auto_classify": "classifyTasks"}
"""The settings keys whose wire field is not their camelCase spelling (:func:`wire_field`)."""


def script_name(path: str) -> str:
    """Return a SEA's display name: its file stem without a ``_sea`` suffix.

    The name ``run_agent`` reports the sub-task under (``"the write_paper
    agent task ..."``) and the ``{name}`` of the channel preamble.
    """
    return Path(path).stem.removesuffix("_sea")


def safe_message(exc: BaseException) -> str:
    """Format an untrusted exception without trusting its ``__str__``.

    ``str(exc)`` runs the exception's ``__str__``, which — for an
    exception minted by an untrusted SEA — may itself raise
    anything.  A diagnostic built here must never leak such a secondary
    raise, so the conversion is guarded and falls back to the type name.

    Args:
        exc: The exception raised by untrusted SEA code.

    Returns:
        ``"TypeName: message"`` when the message renders, else ``"TypeName"``.
    """
    name = type(exc).__name__
    try:
        return f"{name}: {exc}"
    except BaseException:  # noqa: BLE001 — untrusted __str__ may raise anything
        return name


_EXECUTING = threading.Condition()
"""Guards the two maps below and wakes the threads waiting for a file."""

_EXECUTING_OWNER: dict[str, tuple[threading.Thread, int]] = {}
"""Module name -> (the thread executing it, re-entry depth)."""

_EXECUTING_WAIT: dict[threading.Thread, str] = {}
"""Thread -> module name it waits to execute."""

_EXECUTING_RECHECK_SECONDS = 1.0
"""How often a thread waiting for a file re-checks that its record is still live
(:func:`_executing_owner`)."""


def _executing_owner(module_name: str) -> tuple[threading.Thread, int] | None:
    """Return the record of the thread executing *module_name*, or ``None``.

    Called under :data:`_EXECUTING`.  A record whose thread is not inside
    an :func:`execute_python_file` call of *module_name* (dead, or alive
    and doing something else) is dropped: the daemon's Stop
    (``KeyboardInterrupt``) and the per-tool Stop
    (``ToolCallInterrupted``) are injected at an arbitrary bytecode
    boundary (``PyThreadState_SetAsyncExc``), and one landing in
    :func:`_executing` after the record is written but before its
    ``try`` is entered (or inside its ``finally``) leaves the record
    behind, with no code left to remove it.  A thread survives a
    per-tool Stop and carries on with its task, so only its stack tells
    whether it still executes the file (:func:`_in_execution`).  The
    maps hold the thread, not its ident, which the next thread started
    may reuse; the ident is read only while the thread is alive.
    """
    owner = _EXECUTING_OWNER.get(module_name)
    if owner is not None and not (owner[0].is_alive() and _in_execution(owner[0], module_name)):
        del _EXECUTING_OWNER[module_name]
        return None
    return owner


def _in_execution(thread: threading.Thread, module_name: str) -> bool:
    """Whether *thread*'s stack holds an :func:`execute_python_file` call of *module_name*.

    :func:`_executing` makes and drops its claim inside that call, so
    the owner of a live record always has one on its stack.
    """
    frame = sys._current_frames().get(thread.ident or 0)
    while frame is not None:
        if (
            frame.f_code is execute_python_file.__code__
            and frame.f_locals.get("module_name") == module_name
        ):
            return True
        frame = frame.f_back
    return False


def _waits_on_thread(module_name: str, me: threading.Thread) -> bool:
    """Whether the thread executing *module_name* is (transitively) waiting on thread *me*.

    Called under :data:`_EXECUTING`: follows the owner-waits-for-module
    chain so a circular load across threads is reported, not deadlocked.
    A stale record on the chain is dropped (:func:`_executing_owner`),
    so it cannot fake a cycle.
    """
    seen: set[threading.Thread] = set()
    owner = _executing_owner(module_name)
    while owner is not None and owner[0] not in seen:
        if owner[0] is me:
            return True
        seen.add(owner[0])
        waited = _EXECUTING_WAIT.get(owner[0])
        owner = _executing_owner(waited) if waited is not None else None
    return False


@contextlib.contextmanager
def _executing(module_name: str, raw_path: str) -> Iterator[None]:
    """Serialise executions of the file named *module_name*.

    Per file, not global: a script's import-time code runs while it is
    held, and one task's slow SEA import (heavy dependencies, a script
    that blocks) must not stall every other loader in the daemon — the
    ``ready`` fan-out's model-picker probe, the slash-command registry,
    another task's SEA.  Re-entrant: a SEA that loads itself at import
    time executes on the same thread.  Two threads loading files that
    load each other at import time would wait for each other forever;
    that cycle raises :class:`SeaError` in the thread that closes it.
    A record left behind by a stopped thread (see
    :func:`_executing_owner`) is dropped by the next thread that waits
    for the file, which re-checks every
    :data:`_EXECUTING_RECHECK_SECONDS`.  :data:`_EXECUTING` itself is
    taken through :func:`locked_call`: this runs on task threads, where
    a Stop injected right after a plain ``with _EXECUTING:`` acquired
    it would skip the release and wedge every later SEA load in the
    daemon (gh-74174).
    """
    me = threading.current_thread()
    depth = locked_call(_EXECUTING, _claim, module_name, raw_path, me)
    try:
        yield
    finally:
        locked_call(_EXECUTING, _release, module_name, me, depth)


def _claim(module_name: str, raw_path: str, me: threading.Thread) -> int:
    """Record *me* as the thread executing *module_name*; the caller holds :data:`_EXECUTING`.

    Waits while another live thread executes the file.

    Returns:
        The re-entry depth before this claim (0 for an outermost one).

    Raises:
        SeaError: When the thread executing the file is waiting on *me*.
    """
    while (owner := _executing_owner(module_name)) is not None and owner[0] is not me:
        if _waits_on_thread(module_name, me):
            raise SeaError(
                f"SEA {raw_path!r} is being imported by another thread that is "
                "waiting for a SEA this thread is importing (circular SEA imports)"
            )
        _EXECUTING_WAIT[me] = module_name
        try:
            _EXECUTING.wait(_EXECUTING_RECHECK_SECONDS)
        finally:
            del _EXECUTING_WAIT[me]
    depth = owner[1] if owner is not None else 0
    _EXECUTING_OWNER[module_name] = (me, depth + 1)
    return depth


def _release(module_name: str, me: threading.Thread, depth: int) -> None:
    """Undo one :func:`_claim` of *module_name* by *me*; the caller holds :data:`_EXECUTING`.

    The outermost release drops the record and wakes the waiters.
    """
    if depth:
        _EXECUTING_OWNER[module_name] = (me, depth)
    else:
        del _EXECUTING_OWNER[module_name]
        _EXECUTING.notify_all()


def execute_python_file(path: str | Path) -> dict[str, Any]:
    """Execute a caller-supplied Python file and return its namespace.

    The one loader of SEAs (the daemon's ``seaPath``, the
    slash-command registry, the dispatcher, SEAs that load other
    scripts such as ``skillopt``).  The source is compiled and executed
    directly (no ``__pycache__`` read or write), so every call observes
    the file's CURRENT contents and the caller's directory is never
    littered with bytecode.  The module is registered in ``sys.modules``
    exactly as ``import`` does, under a name derived from the file's
    path, and stays registered: ``@dataclass`` under ``from __future__
    import annotations`` and ``typing.get_type_hints`` resolve string
    annotations through ``sys.modules[cls.__module__]`` whenever a
    getter or a tool of the script runs later.  A re-execution of the
    same file replaces the entry (a failed one restores the previous
    entry), so a long-lived daemon holds one module per script file,
    not one per run, and an earlier execution's classes resolve their
    annotations through the latest execution of the same source; two
    files with the same stem in different folders get different names.
    Executions of the SAME file are serialised (:func:`_executing`):
    task threads evaluate SEAs concurrently, and a failed execution in
    one thread must restore the entry that is current when it fails,
    not the one it saw before another thread replaced it.  Different
    files execute concurrently: one script's slow import-time code
    does not block the loaders of every other script, and a circular
    load across threads is reported as a :class:`SeaError`.

    Args:
        path: The path of the file, expected to be absolute.  The wire
            value a ``run`` command carries is screened by
            ``sea_apply.load_layers`` before it becomes a path.

    Returns:
        The executed module's namespace dict.

    Raises:
        SeaError: *path* is not the path of an existing ``.py`` file,
            or names a module that raises at import time
            (``BaseException`` included: a file raising
            ``KeyboardInterrupt`` or ``SystemExit`` at import time is a
            broken file, not a cancelled task; the original raise stays
            reachable as ``__cause__``).  A Stop that lands inside the
            import is re-raised unchanged instead: the per-tool Stop's
            ``ToolCallInterrupted``, and a ``KeyboardInterrupt`` while
            the calling thread's stop event is set.
    """
    path = Path(path)
    raw_path = str(path)
    try:
        is_py_file = path.suffix == ".py" and path.is_file()
    except (OSError, ValueError):
        # e.g. an embedded NUL byte makes ``is_file`` raise ValueError.
        is_py_file = False
    if not is_py_file:
        raise SeaError(f"SEA {raw_path!r} is not an existing Python (.py) file")
    module_name = f"_kiss_sea_{path.stem}_{hashlib.sha1(raw_path.encode()).hexdigest()[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = raw_path
    with _executing(module_name, raw_path):
        # A failed re-execution must not unregister the module of an
        # earlier, successful execution of the same file whose classes
        # still resolve their annotations through this name: the previous
        # entry is put back.
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            source = path.read_text(encoding="utf-8")
            code = compile(source, str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)  # noqa: S102 — the script is the user's own code
        except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            if is_stop(exc):
                raise  # a Stop (per-tool or the task's) landed inside the import
            logger.warning("Failed to import SEA %r", raw_path, exc_info=True)
            raise SeaError(f"SEA {raw_path!r} failed to import: {safe_message(exc)}") from exc
    return module.__dict__


def is_stop(exc: BaseException) -> bool:
    """Return whether *exc*, raised by untrusted SEA code, is a Stop and not the code's failure.

    The per-tool Stop's ``ToolCallInterrupted``, or a
    ``KeyboardInterrupt`` while the calling thread's stop event is set
    (the task's Stop).  A caller re-raises such an exception unchanged
    instead of reporting the SEA broken.
    """
    return isinstance(exc, ToolCallInterrupted) or (
        isinstance(exc, KeyboardInterrupt) and stop_requested()
    )


def anchored_work_dir(work_dir: str, base_dir: str) -> str:
    """Return *work_dir* as the absolute directory it names from the calling task's *base_dir*.

    The one rule for a relative ``work_dir``, wherever it is written: a
    SEA's ``work_dir`` setting, a ``run_agent`` / ``run_parallel``
    ``work_dir`` option and the daemon's staging of a SEA's setting
    all resolve it against the directory of the calling task (the tab's
    for ``/<name>``), never against the SEA file's folder.

    Args:
        work_dir: The directory as written; ``~`` is expanded.
        base_dir: The calling task's directory.

    Returns:
        *work_dir* itself when absolute, else ``base_dir/work_dir``.
    """
    path = Path(work_dir.strip()).expanduser()
    return str(path if path.is_absolute() else Path(base_dir).expanduser() / path)


def declares_hidden(path: Path) -> bool:
    """Return whether the SEA at *path* writes ``"hidden": True`` in its ``settings`` method.

    Read from the source (``ast``), never by executing the script: the
    command registry calls this for every scanned folder, and a hidden
    SEA (a test fixture, protocol plumbing, a base class other SEAs
    derive from) must be excluded without running it.  Hence the
    contract that ``hidden`` is a literal ``True`` in a dict inside the
    file's own ``settings`` method; a computed or inherited value is
    accepted by :func:`resolve_settings` but does not hide the SEA.
    Only a constant value under a string-literal key in a settings dict
    of that method counts (:func:`settings_dicts`: ``model_config``'s
    contents are not settings).
    """
    return parsed_sea(path).literals.get("hidden") is True


def declares_channel(path: Path) -> bool:
    """Return whether the SEA at *path* derives its class from ``ChannelSea`` by that name.

    Read from the source (``ast``), never by executing the script, like
    :func:`declares_hidden`: the channel listing scans every
    third-party folder without importing the channel modules.  Hence
    the contract that a channel names ``ChannelSea`` (bare, under an
    import alias, or as the last attribute of a dotted name) in the
    bases of a class the file defines; a channel reached through a
    base class of another file is still a channel when loaded
    (``isinstance``), but is not listed.
    """
    return "ChannelSea" in parsed_sea(path).bases


SEA_BASES = ("BaseSea", "WorkerSea", "ChannelSea")
"""The base classes a SEA derives from, least to most specific.

A class deriving from one of these alone registers as no model; one
deriving from anything else may (:attr:`ParsedSea.bases`).
"""


@dataclass(frozen=True)
class ParsedSea:
    """What :func:`parsed_sea` reads from a SEA's source without executing it.

    Attributes:
        stamp: The file revision ``(mtime_ns, size, inode)`` the rest
            was read from; ``None`` when the file cannot be read (the
            rest is then empty).
        bases: The last name of every base of every class the file
            defines.  A base imported under an alias (``from ... import
            ChannelSea as C``) counts under its imported name; a dotted
            base (``base_sea.BaseSea``) under its last attribute; a
            computed base (``sea_class('picker')``) under its source
            text.  Empty for an unreadable or unparsable file.
        literals: ``{settings key: literal value}`` of the settings
            dicts in the file's own ``settings`` method
            (:func:`settings_dicts`).
        mentions_register_as_model: Whether the source text contains
            ``register_as_model`` (a superset of defining the method:
            what decides whether the model picker imports the file).
    """

    stamp: tuple[int, int, int] | None
    bases: set[str]
    literals: dict[str, Any]
    mentions_register_as_model: bool


_UNREADABLE = ParsedSea(None, set(), {}, False)

PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError)
"""What reading a script as Python source raises when the source is not parsable: a syntax
error, undecodable bytes or a NUL byte (``ValueError``), or nesting too deep for the parser
(``RecursionError``, or ``MemoryError`` "too complex to parse")."""

_PARSE_CACHE: dict[Path, ParsedSea] = {}
"""The memo of :func:`parsed_sea`: one parse per file revision, so a registry refresh costs
one ``stat`` per SEA."""


def parsed_sea(path: Path) -> ParsedSea:
    """Return what the source of the SEA at *path* declares, parsed once per revision.

    The revision is the file's mtime, size and inode, so an atomic
    replacement that keeps the mtime and size is still re-read.  A
    revision that does not parse is memoised with empty ``bases`` and
    ``literals``, so the registry watcher does not re-read a broken SEA
    on every poll.
    """
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        return _UNREADABLE
    cached = _PARSE_CACHE.get(path)
    if cached is None or cached.stamp != stamp:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return _UNREADABLE
        mentions = "register_as_model" in source
        try:
            tree = ast.parse(source)
        except PARSE_ERRORS:
            cached = ParsedSea(stamp, set(), {}, mentions)
        else:
            cached = ParsedSea(stamp, _class_bases(tree), _settings_literals(tree), mentions)
        _PARSE_CACHE[path] = cached
    return cached


def _class_bases(tree: ast.Module) -> set[str]:
    """Return the last name of every base of every class definition in *tree*."""
    imported = {
        alias.asname: alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names if alias.asname
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.add(imported.get(base.id, base.id))
                elif isinstance(base, ast.Attribute):
                    names.add(base.attr)
                else:
                    names.add(ast.unparse(base))
    return names


def _settings_literals(tree: ast.Module) -> dict[str, Any]:
    """Return the constant ``"key": value`` entries of the settings dicts of *tree*."""
    literals: dict[str, Any] = {}
    for sub in settings_dicts(tree):
        for key, value in zip(sub.keys, sub.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                literals[str(key.value)] = value.value
    return literals


def settings_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Return every ``def settings`` of *tree* (the SEA class's method, wherever the class is)."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "settings"
    ]


def settings_dicts(tree: ast.Module) -> list[ast.Dict]:
    """Return the dict literals of the ``settings`` methods of *tree* that hold settings keys.

    Every dict literal in the function except dicts anywhere inside
    the *value* of another dict's key (``model_config``'s contents,
    however deep, are not settings).  The one rule for reading settings
    from the source: the registry's ``hidden`` check and the linter
    use it alike.
    """
    dicts: list[ast.Dict] = []
    for node in settings_functions(tree):
        nested = {
            id(inner)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Dict)
            for value in sub.values
            for inner in ast.walk(value)
            if isinstance(inner, ast.Dict)
        }
        dicts.extend(
            sub for sub in ast.walk(node) if isinstance(sub, ast.Dict) and id(sub) not in nested
        )
    return dicts


def resolve_settings(declared: Mapping[str, Any], channel: bool = False) -> dict[str, Any]:
    """Return the effective settings a SEA *declared*.

    *declared* is what the SEA's ``settings`` methods returned, base
    class first (:func:`kiss.agents.sorcar.sea_commands.base_settings`
    folds them, the worker and channel defaults of the base classes
    included).  Every value is type-checked against
    :data:`SETTING_TYPES`, a ``tool_profile`` alias (``readonly``)
    becomes its key (:func:`alias_free_profile`), and a *channel*
    gets the settings side of :data:`CHANNEL_BEHAVIOURS`: ``work_dir``
    plus the worker keys are locked.  ``work_dir`` is otherwise kept as
    declared; the launcher anchors a relative one at the calling task's
    directory (:func:`anchored_work_dir`).

    Args:
        declared: The declared settings.
        channel: Whether the SEA derives from ``ChannelSea``.

    Returns:
        A new dict of the effective settings.  A key whose value is
        ``None`` is dropped — as is a string key (``work_dir``,
        ``model``, ``chat_id``, ``tool_profile``, ``docker_image``)
        whose value is ``""`` — it means "no override", so the
        caller's or the persisted value stands.

    Raises:
        SeaError: When *declared* names an unknown, renamed or removed
            key, or a value has the wrong type.
    """
    sources = {key: f"settings()[{key!r}]" for key in declared}
    for key in declared:
        if key in RENAMED_SETTINGS:
            raise SeaError(
                f"settings() key {key!r} was renamed to {RENAMED_SETTINGS[key]!r}; "
                f"run `uv run sea lint --fix` to rewrite the script"
            )
        if key in REMOVED_SETTINGS:
            raise SeaError(f"settings() key {key!r} was removed: {REMOVED_SETTINGS[key]}")
        if key not in SETTING_TYPES:
            raise SeaError(
                f"settings() has an unknown key {key!r}; "
                f"known keys: {', '.join(SETTING_TYPES)}"
            )
    # ``None`` means "no override": the caller's or persisted value
    # stands.  So does ``""`` for a string key (the spelling of "none"
    # a script computing its model, profile or directory may produce),
    # as a blank ``run_agent`` option does (``agent_dispatch.parse_options``).
    settings = {
        key: value for key, value in declared.items()
        if value is not None and not (SETTING_TYPES[key] is str and value == "")
    }
    if isinstance(settings.get("tool_profile"), str):
        settings["tool_profile"] = alias_free_profile(settings["tool_profile"])
    for key, value in settings.items():
        expected = SETTING_TYPES[key]
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            names = (
                " or ".join(t.__name__ for t in expected)
                if isinstance(expected, tuple) else expected.__name__
            )
            raise SeaError(
                f"{sources[key]} must be {names}, got {type(value).__name__}"
            )
        settings[key] = _check_value(sources[key], key, value)
    if channel:
        # CHANNEL_BEHAVIOURS "worker" and "scratch directory": the base
        # classes laid the values; the locks are added here so a
        # subclass that writes its own ``locked`` cannot drop them.
        settings["locked"] = sorted({*settings.get("locked", ()), "work_dir", *WORKER_DEFAULTS})
    return settings


def locked_conflicts(
    settings: Mapping[str, Any], asked: Mapping[str, Any], base_dir: str = "",
) -> str:
    """Return why explicit values clash with a script's ``locked`` settings, or ``""``.

    The precedence of a sub-task's settings is: an explicit argument of
    the call wins over the script's ``settings()``, which win over
    what the calling task passes on, which win over the user's
    persisted settings — except for the keys the script lists in
    ``locked``: an explicit argument that differs from a locked value
    is an error, never silently replaced.

    Args:
        settings: The script's merged settings.
        asked: ``{setting key: value}`` as the call passed them
            explicitly (``None`` and ``""`` count as not passed).
        base_dir: The directory a relative ``work_dir`` (asked or
            locked) is resolved against before the two are compared;
            empty compares them as given.

    Returns:
        ``""`` when nothing clashes, else one sentence naming every
        clash, e.g. ``the script locks tool_profile='bash' (asked for
        'review')``.
    """
    clashes = [
        f"{key}={settings[key]!r} (asked for {asked[key]!r})"
        for key in settings.get("locked") or ()
        if key in settings and asked.get(key) not in (None, "")
        and not _same_setting(key, asked[key], settings[key], base_dir)
    ]
    if not clashes:
        return ""
    return "the script locks " + ", ".join(clashes)


def _same_setting(key: str, asked: Any, locked: Any, base_dir: str) -> bool:
    """Return whether *asked* equals *locked*; ``work_dir`` paths are compared resolved."""
    if key == "work_dir" and isinstance(asked, str) and isinstance(locked, str):
        base = Path(base_dir).expanduser() if base_dir else Path.cwd()
        return (base / Path(asked.strip()).expanduser()).resolve() == (
            base / Path(locked.strip()).expanduser()
        ).resolve()
    return bool(asked == locked)


def _check_value(source: str, key: str, value: Any) -> Any:
    """Validate one type-checked setting beyond its type.

    ``max_budget`` and ``timeout`` must be finite
    (``coerce_budget_override`` would otherwise SILENTLY discard a
    NaN/infinite budget downstream) and are returned as floats.  A
    value whose own methods raise (an untrusted number subclass) is
    reported as broken.

    Raises:
        SeaError: Naming *source* (``settings()['timeout']``).
    """
    if key == "locked":
        lockable = [k for k in SETTING_TYPES if k not in META_SETTINGS]
        bad = [item for item in value if not isinstance(item, str) or item not in lockable]
        if bad:
            raise SeaError(
                f"{source} may only name settings keys ({', '.join(lockable)}); got {bad!r}"
            )
        return sorted(set(value))
    if key in ("max_budget", "timeout"):
        try:
            value = float(value)
        except OverflowError:
            value = math.inf
        except BaseException as exc:  # noqa: BLE001 — an untrusted number subclass may raise
            if is_stop(exc):
                raise
            raise SeaError(
                f"{source} returned a broken value: {safe_message(exc)}"
            ) from exc
        if not math.isfinite(value):
            raise SeaError(f"{source} must return a finite number or None")
        if key == "timeout" and value <= 0:
            raise SeaError(f"{source} must be a positive number of seconds, got {value:g}")
    return value
