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
import importlib._bootstrap
import logging
import math
import os
import sys
import threading
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS, BaseSea, ChannelSea, WorkerSea
from kiss.core.locked import locked_call
from kiss.core.stop_signal import stop_requested
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.core.utils import read_bytes_waiting_for_writer

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
    "compaction": bool,
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
    "model_config": "Model configuration dict passed to the LLM (temperature, ...). "
                    "`base_url` (with `api_key`) routes every model call of the run, in a "
                    "tab and headless alike, to that OpenAI-compatible endpoint (a proxy "
                    "the SEA starts, say); `model` must still name a catalogued model so "
                    "the calls are priced. `plain_io: true` leaves the framework's own "
                    "additions out of what the model reads: the system prompt is the SEA's "
                    "`system_prompt` return alone (no `# Restricted tool profile` or "
                    "`# Web tools are off` note, no `# MOST IMPORTANT INSTRUCTIONS`, "
                    "`# Task Settings` or `AGENTS.md` trailer; the memory protocol stays "
                    "while the run has the memory tools, and a CLI-backed model's text tool "
                    "catalog stays) and tool results carry no usage line (`Steps: ..., "
                    "Context: ...`). `coding_gate: false` makes the run no coding run for "
                    "the finish gate: no verifier or review stage, whatever its edits (a sync "
                    "resolving conflicts), so its `finish` is not gated at all. Replaces the "
                    "caller's `model_config` whole, except that a dict "
                    "of such framework flags alone is added to the configuration that "
                    "applies anyway (a settings-panel endpoint stays).",
    "use_web_tools": "Give the run the browser tools (daemon default: on).",
    "auto_classify": "Let the pre-run classifier decide whether the task is simple (lite "
                     "prompt) (daemon default: the persisted setting).",
    "use_memory": "Give the run the `memory_*` tools (daemon default: the persisted setting).",
    "compaction": "Let the run compact its context: old, large tool outputs and arguments "
                  "become short stubs once the context grows large (default: on). `false` "
                  "turns it off for the run; the daemon's global Compaction setting (the "
                  "settings panel's checkbox) off turns it off for every run, whatever a "
                  "SEA or a call passes.",
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

_FAN_OUT_REMOVED = (
    "`run_parallel` is N `run_agent` calls, so there is nothing to allow or "
    "forbid separately; `tool_profile` chooses the toolset"
)

REMOVED_SETTINGS = {
    "kind": _BASE_CLASS_REMOVED,
    "preset": _BASE_CLASS_REMOVED,
    "channel": _BASE_CLASS_REMOVED,
    "allow_fan_out": _FAN_OUT_REMOVED,
    "is_parallel": _FAN_OUT_REMOVED,
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
"""Guards the map below and wakes the threads waiting for a file."""

_EXECUTING_OWNER: dict[str, tuple[threading.Thread, int]] = {}
"""Module name -> (the thread executing it, re-entry depth)."""

_IMPORT_WAITS: Any = getattr(importlib._bootstrap, "_blocking_on")
"""Thread ident -> the module locks the thread waits for in an ``import`` (each lock names
its owner's ident): the map ``importlib``'s own deadlock detection follows."""

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
    """Whether *thread*'s stack holds an :func:`execute_python_file` or
    :func:`refresh_imported_module` call of *module_name*.

    :func:`_executing` makes and drops its claim inside those calls, so
    the owner of a live record always has one on its stack.
    """
    return module_name in _stack_modules(
        thread.ident, execute_python_file, refresh_imported_module
    )


def _stack_modules(ident: int | None, *functions: Any) -> Iterator[str | None]:
    """Yield the ``module_name`` of every call of one of *functions* on the stack of the live
    thread *ident*, innermost first (``None`` for a call that has not named its module yet)."""
    codes = {function.__code__ for function in functions}
    frame = sys._current_frames().get(ident or 0)
    while frame is not None:
        if frame.f_code in codes:
            yield frame.f_locals.get("module_name")
        frame = frame.f_back


def _waited_module(ident: int | None) -> str | None:
    """Return the module name the thread *ident* waits to execute, or ``None``.

    Read from its stack, a :func:`_claim` call, so nothing is left to go
    stale when a Stop ends the wait.  :func:`_claim` runs under
    :data:`_EXECUTING` and gives it up only inside its wait: for a
    caller holding :data:`_EXECUTING`, another thread in :func:`_claim`
    is waiting.
    """
    return next(_stack_modules(ident, _claim), None)


def _waits_on_thread(module_name: str, me: threading.Thread) -> bool:
    """Whether the thread executing *module_name* is (transitively) waiting on thread *me*.

    Called under :data:`_EXECUTING`: follows the owner-waits-for-module
    chain so a circular load across threads is reported, not deadlocked.
    A stale record on the chain is dropped (:func:`_executing_owner`),
    so it cannot fake a cycle.  A thread's wait is of either kind: for
    the file another thread executes (:func:`_waited_module`), or for
    the lock Python's ``import`` holds on a module another thread is
    importing (``importlib``'s ``_blocking_on``, the map its own
    deadlock detection follows: thread ident -> the module locks it
    waits for, each naming its owner's ident).  A cycle through one
    wait of each kind — this thread importing a module that loads a
    SEA whose load imports that module — is invisible to Python, which
    cannot see the execution-slot wait, and to a check following the
    execution-slot waits alone.
    """
    owner = _executing_owner(module_name)
    pending = [owner[0].ident] if owner is not None else []
    seen: set[int | None] = set()
    while pending:
        ident = pending.pop()
        if ident == me.ident:
            return True
        if ident is None or ident in seen:
            continue
        seen.add(ident)
        waited = _waited_module(ident)
        owner = _executing_owner(waited) if waited is not None else None
        if owner is not None:
            pending.append(owner[0].ident)
        pending.extend(lock.owner for lock in _IMPORT_WAITS.get(ident) or ())
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
        _EXECUTING.wait(_EXECUTING_RECHECK_SECONDS)
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
        The executed module's namespace dict.  Its :data:`SOURCE_REVISION`
        entry, set once the source ran, is the revision of the text that
        ran in it.

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
    module_name = _module_name(path)
    module = types.ModuleType(module_name)
    module.__file__ = raw_path
    with _executing(module_name, raw_path):
        # A failed re-execution must not unregister the module of an
        # earlier, successful execution of the same file whose classes
        # still resolve their annotations through this name: the previous
        # entry is put back.  The new one is published inside the ``try``,
        # so a Stop landing right after it is undone like any failure.
        previous = sys.modules.get(module_name)
        try:
            sys.modules[module_name] = module
            module.__dict__[SOURCE_REVISION] = _run_source(path, module)
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


def _run_source(path: Path, module: types.ModuleType) -> str:
    """Compile the current source of *path*, run it in *module*'s namespace, return its revision.

    Directly, with no ``__pycache__`` read or write: the module runs
    the file's current contents, where the import machinery would run
    the bytecode it cached for an earlier revision of the same size
    and whole-second mtime.  The bytes are compiled, not decoded text,
    so the source is read as ``import`` reads it (PEP 263: a UTF-8
    byte-order mark is skipped, a ``coding`` cookie is honoured).  Read
    past a concurrent atomic publish of the script (``sea lint --fix``),
    which Windows otherwise reports to the reader as ``PermissionError``.
    The file is read once: the revision returned (:func:`source_revision`)
    is that of the bytes that ran, whatever the file held while the
    caller waited to run it and whatever it holds since.
    """
    source = read_bytes_waiting_for_writer(path)
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)  # noqa: S102 — the script is the user's own code
    return source_revision(source)


SOURCE_REVISION = "__source_revision__"
"""Name, in the namespace :func:`execute_python_file` returns, of the revision
(:func:`source_revision`) of the source that ran in it."""


def source_revision(source: bytes) -> str:
    """Return the revision of the script text *source*: the SHA-256 of its bytes.

    What names a text that ran.  A file's :func:`file_stamp` names what one
    ``stat`` saw: the file may change between that ``stat`` and the read
    of its text, and two writes of one size within one tick of the file
    system's clock share a stamp.
    """
    return hashlib.sha256(source).hexdigest()


def file_revision(path: Path) -> str | None:
    """Return the :func:`source_revision` of the text the file at *path* holds now.

    Returns:
        The revision, or ``None`` when the file cannot be read.
    """
    try:
        return source_revision(path.read_bytes())
    except OSError:
        return None


def _module_name(path: Path) -> str:
    """Return the ``sys.modules`` name :func:`execute_python_file` registers *path* under.

    The path is hashed as the bytes the file system holds (``surrogateescape``): a folder
    whose name is not UTF-8 is a path like any other.
    """
    digest = hashlib.sha1(str(path).encode("utf-8", "surrogateescape")).hexdigest()[:12]
    return f"_kiss_sea_{path.stem}_{digest}"


_IMPORTED_STAMPS: dict[str, tuple[int, int, int]] = {}
"""File path -> the revision :func:`refresh_imported_module` last re-executed its module from."""

_IMPORT_ATTRS = ("__file__", "__cached__", "__loader__", "__package__", "__spec__", "__path__")
"""What ``import`` sets on a module beside its name: carried over to its re-execution."""

_IMPORT_LOCK: Any = getattr(importlib._bootstrap, "_get_module_lock")
"""Module name -> the lock ``import`` holds while it runs that module's source."""

_IMPORT_CYCLE: Any = getattr(importlib._bootstrap, "_DeadlockError")
"""Raised for an import lock another thread holds: by ``import`` when that thread waits for a
module this thread is importing, by :func:`_lock_import` whatever that thread does."""

_MID_IMPORT = (-1, -1, -1)
"""The revision recorded for a module another thread was still importing: no file has it."""


def imported_revision(path: Path) -> tuple[int, int, int] | None:
    """Return the revision the module of *path* was last re-executed from, or ``None``.

    The :attr:`ParsedSea.stamp` :func:`refresh_imported_module` read
    before the run that published the module ``sys.modules`` holds for
    *path* now; ``None`` when it never re-executed the module (the
    revision ``import`` loaded it from is unknown), :data:`_MID_IMPORT`
    when it last found another thread still importing it.
    """
    return _IMPORTED_STAMPS.get(str(path))


def _lock_import(lock: Any) -> None:
    """Take a module's import *lock* unless another thread holds it; the caller holds its guard.

    Never waits for that thread: it is importing the module, for minutes or for good, and
    ``lock.acquire()`` would park this one in a C-level wait no Stop reaches.  Every change of
    the lock's state is made under the guard (``lock.lock``), so under it a lock that is free,
    or this thread's own, is acquired at once.

    Raises:
        _IMPORT_CYCLE: Another thread holds the lock.
    """
    if lock.count and lock.owner != threading.get_ident():
        raise _IMPORT_CYCLE(f"{lock.name!r} is being imported by another thread")
    lock.acquire()


def _run_imported(path: Path, old: types.ModuleType, module: types.ModuleType) -> None:
    """Run the current source of *path* in *module*, which replaces *old*, as ``import`` runs it.

    ``sys.modules`` names *module* while its source runs: ``@dataclass``
    under ``from __future__ import annotations`` reads a string
    annotation (``ClassVar``) through ``sys.modules[cls.__module__]``,
    and *old* lacks what the new source imports.  The module's import
    lock is held and its spec marked initializing meanwhile, so an
    ``import`` in another thread waits for the run to end instead of
    reading a half-run module; a run that raises puts *old* back and
    gives *module*, which that thread may hold by then, the namespace
    of *old*.  The lock is taken inside the ``try``: a Stop injected
    right after a plain ``with`` acquired it would leak it (gh-74174)
    and hang every later import of the module.  It is taken before
    anything is touched and only when no other thread holds it
    (:func:`_lock_import`): a module another thread is importing is
    left to that import.

    Raises:
        _IMPORT_CYCLE: Another thread holds the module's import lock.
    """
    lock = _IMPORT_LOCK(module.__name__)
    spec: Any = module.__spec__ or types.SimpleNamespace()  # a script's ``__main__`` has none
    try:
        locked_call(lock.lock, _lock_import, lock)
        try:
            spec._initializing = True
            sys.modules[module.__name__] = module
            _run_source(path, module)
        except BaseException:
            sys.modules[module.__name__] = old
            vars(module).clear()
            vars(module).update(vars(old))
            raise
    finally:
        with contextlib.suppress(RuntimeError):  # not ours: another thread's, or never taken
            lock.release()
            spec._initializing = False


def refresh_imported_module(path: Path, stale: bool = False) -> bool:
    """Re-execute the module a plain ``import`` loaded from *path* when the file may have changed.

    A SEA may derive from a base of its own through an ordinary import
    (``from parent_sea import Parent``) instead of ``sea_class``.
    Python executes an imported module once per process, so a SEA
    re-executed after an edit of such a base (the model picker
    re-reading a changed parent) would still derive from the base's old
    class.  The modules in ``sys.modules`` loaded from *path* are
    re-executed from the file's current source (:func:`_run_source`,
    not ``importlib.reload``: that runs the bytecode cached for an
    earlier revision of the same size and second), unless this
    function last did so from the file's current revision
    (:attr:`ParsedSea.stamp`) and *stale* is false; the revision is
    read before the run, so an edit during it is re-read next time,
    and the first call for a file re-executes once, the revision its
    module was imported from being unknown.  Each runs in a fresh
    module (``import``'s attributes carried over) that replaces the
    old one in ``sys.modules`` while it runs (:func:`_run_imported`)
    and, for a submodule, as the attribute of its package that
    ``from package import parent`` reads once all ran through, as
    :func:`execute_python_file` replaces a script's: a run that
    raises puts the old ones back, so the globals of the classes loaded before
    (another task's running SEA) are never half-rewritten under them,
    and a name the new source dropped is gone — an import of it fails
    as it would in a fresh process.  *stale* says the module is out of
    date whatever its revision: a class of its derives from one whose
    module was re-executed since, and only a re-execution rebinds it
    to the new base.  Modules of the kiss package itself (the bundled
    bases) are left alone: the daemon holds their classes everywhere,
    and a re-execution would make unrelated new ones.  Serialised per
    file like :func:`execute_python_file` (:func:`_executing`), so two
    threads refreshing one base run one re-execution and the second
    finds the finished module.  A module another thread is still
    importing is not waited for (:data:`_IMPORT_CYCLE`: that import
    may never end, and no Stop reaches a thread parked on an import
    lock): it stays as that import leaves it, under a revision no file
    has (:data:`_MID_IMPORT`), so the next call, and whoever stamped a
    result with that revision, tries again.

    Args:
        path: The file of the module (``Path(module.__file__)``).
        stale: Re-execute the module from an unchanged revision too.

    Returns:
        Whether a module was re-executed: a class loaded from it before
        derives from the old module and is to be loaded again.

    Raises:
        SeaError: The re-execution raised (a base mid-edit); a Stop
            landing in it is re-raised unchanged (:func:`is_stop`).
    """
    raw_path = str(path)
    module_name = _module_name(path)
    with _executing(module_name, raw_path):
        stamp = parsed_sea(path).stamp
        if stamp is None or (_IMPORTED_STAMPS.get(raw_path) == stamp and not stale):
            return False
        modules = [
            module for module in list(sys.modules.values())
            if _loaded_from(module, path)
            and not module.__name__.startswith(("kiss.", "_kiss_sea_"))
        ]
        if not modules:
            return False
        fresh = []
        for module in modules:
            fresh.append(types.ModuleType(module.__name__))
            for attr in _IMPORT_ATTRS:
                if hasattr(module, attr):
                    setattr(fresh[-1], attr, getattr(module, attr))
        try:
            for old, module in zip(modules, fresh, strict=True):
                _run_imported(path, old, module)
        except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
            for old, module in zip(modules, fresh, strict=True):
                if sys.modules.get(old.__name__) is module:  # ran through before the raise
                    sys.modules[old.__name__] = old
            if is_stop(exc):
                raise
            if isinstance(exc, _IMPORT_CYCLE):  # still being imported: not the file's failure
                _IMPORTED_STAMPS[raw_path] = _MID_IMPORT
                return False
            logger.warning("Failed to re-import SEA %r", raw_path, exc_info=True)
            raise SeaError(f"SEA {raw_path!r} failed to import: {safe_message(exc)}") from exc
        for old, module in zip(modules, fresh, strict=True):
            # ``import`` also binds a submodule as an attribute of its
            # package, and ``from package import parent`` reads that
            # attribute, not ``sys.modules``: rebind it when it still
            # names the replaced module (a reader between the two
            # publications sees the old module, which ``sys.modules``
            # then no longer holds, so its class is re-read).
            package, _, submodule = module.__name__.rpartition(".")
            if package and getattr(sys.modules.get(package), submodule, None) is old:
                setattr(sys.modules[package], submodule, module)
        _IMPORTED_STAMPS[raw_path] = stamp
    return True


def _loaded_from(module: Any, path: Path) -> bool:
    """Return whether *module* was loaded from the file *path*.

    ``__file__`` is the path as ``import`` joined it, so a ``sys.path``
    entry spelled ``/a//b`` or ``/a/./b`` is compared as the same file
    *path* (a ``Path``, which drops both) names.
    """
    file = getattr(module, "__file__", None)
    return isinstance(file, str) and file.endswith(path.name) and Path(file) == path


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
        work_dir: The directory as written; ``~`` is expanded (a
            ``~user`` that names no user is kept as written, as a
            shell keeps it).
        base_dir: The calling task's directory.

    Returns:
        *work_dir* itself when absolute, else ``base_dir/work_dir``.
    """
    path = Path(expand_user(work_dir.strip()))
    return str(path if path.is_absolute() else Path(expand_user(base_dir)) / path)


def expand_user(path: str) -> str:
    """Return *path* with a leading ``~`` or ``~user`` expanded; a ``~user`` that names no user
    is kept as written, as a shell keeps it (``Path.expanduser`` raises on one).

    Windows has no password database: :func:`os.path.expanduser` invents a
    sibling of the current home for every ``~user``.  Only the current user
    can be known there.  A name with a NUL byte is kept as written too.
    """
    if os.name == "nt" and path.startswith("~") and path[1:2] not in ("", "/", "\\"):
        user = path[1:].replace("\\", "/").split("/", 1)[0]
        if user.casefold() != Path(os.path.expanduser("~")).name.casefold():
            return path
    try:
        return os.path.expanduser(path)
    except ValueError:
        return path


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


def file_stamp(path: Path) -> tuple[int, int, int] | None:
    """Return the revision ``(mtime_ns, size, inode)`` of the file at *path*, from one ``stat``.

    The inode is part of it so that an atomic replacement that keeps
    the mtime and size is still another revision.  Permissions are not:
    a file that exists but cannot be read has a revision, and a
    ``chmod`` does not change it.

    Returns:
        The revision, or ``None`` when there is no file to ``stat``.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def parsed_sea(path: Path) -> ParsedSea:
    """Return what the source of the SEA at *path* declares, parsed once per revision.

    The revision is the file's :func:`file_stamp`.  The
    bytes are parsed, not decoded text, so the source is read as
    ``import`` and :func:`_run_source` read it (PEP 263: a UTF-8
    byte-order mark is skipped, a ``coding`` cookie is honoured).  A
    revision that does not parse is memoised with empty ``bases`` and
    ``literals``, so the registry watcher does not re-read a broken SEA
    on every poll.
    """
    stamp = file_stamp(path)
    if stamp is None:
        return _UNREADABLE
    cached = _PARSE_CACHE.get(path)
    if cached is None or cached.stamp != stamp:
        try:
            source = path.read_bytes()
        except OSError:
            return _UNREADABLE
        mentions = b"register_as_model" in source
        try:
            tree = ast.parse(source)
        except PARSE_ERRORS:
            cached = ParsedSea(stamp, set(), {}, mentions)
        else:
            literals = {key: node.value for key, node in settings_literals(tree).items()}
            cached = ParsedSea(stamp, _class_bases(tree), literals, mentions)
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


def settings_literals(tree: ast.Module) -> dict[str, ast.Constant]:
    """Return the value of each constant ``"key": value`` entry of the settings dicts of *tree*,
    laid in the order Python evaluates them (:func:`_settings_entries`), so the last one written
    wins: a computed value written after a literal drops the key, as it overwrites the literal."""
    literals: dict[str, ast.Constant] = {}
    for sub, index in _settings_entries(tree):
        key, value = sub.keys[index], sub.values[index]
        if not isinstance(key, ast.Constant):
            continue
        if isinstance(value, ast.Constant):
            literals[str(key.value)] = value
        else:
            literals.pop(str(key.value), None)
    return literals


def sea_methods(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    """Return every method *name* of *tree*: a ``def`` of that name in a class's own body.

    The SEA class's method, wherever the class is.  A mixin reaches the
    SEA class through an alias, a call, a decorator or a subscript
    (``Annotated[Mixin, ...]``) as well as by name, so the source does
    not say which classes of the file a run folds, and every class is
    read: leaving one out would miss the ``hidden`` a mixin declares.
    A helper class with a method of that name is read with them (the
    trial settings ``coding_sea.ContainerHarness.settings`` builds are
    linted that way).  Only a function of that name at module level or
    nested in another function is the script's helper, and what it
    returns is data.  The one rule for reading a SEA's methods from the
    source: the registry's ``hidden`` check, the linter and the
    settings tuner use it alike.
    """
    return [
        node
        for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
        for node in own_statements(cls)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def settings_dicts(tree: ast.Module) -> list[ast.Dict]:
    """Return the dict literals of the ``settings`` methods of *tree* that hold settings keys.

    The dict literals whose entries reach the method's return value:
    returned as they are, merged with ``|``, unpacked with ``**``,
    chosen by a conditional expression, or assigned to (or ``update``d
    into) a local name the return value is built from in those same
    ways.  The statements are read in order: a name holds what the
    last assignment at the method's top level gave it (an earlier dict
    it held is data then, unless the new value is built from the name:
    ``mine = mine | {...}``), plus what an assignment under an ``if``
    or a loop, which may or may not run, an ``update`` or a ``|=``
    added since; a ``return`` reads the names as they are where it
    stands.  A name assigned another at the top level (``mine =
    settings``, the method's parameter included) is the same dict from
    then on, so what ``update`` adds through either reaches both;
    assigned under an ``if`` or a loop it only gains what the other
    holds there.  A dict anywhere
    inside the *value* of another dict's key (``model_config``'s
    contents, however deep) and a dict the method only reads a value
    from (``info["model"]``) are data, not settings: rewriting their
    keys would break the script.  The one rule for reading settings
    from the source: the registry's ``hidden`` check and the linter
    use it alike.
    """
    dicts: list[ast.Dict] = []
    for sub, _ in _settings_entries(tree):
        if not any(sub is seen for seen in dicts):  # reached twice: listed once
            dicts.append(sub)
    return dicts


_Entry = tuple[ast.Dict, int]
"""An entry of a dict literal: the dict and the index of the entry in it."""


def _settings_entries(tree: ast.Module) -> list[_Entry]:
    """Return the explicit entries of the settings dicts of *tree* (:func:`settings_dicts`) in
    the order the returned value lays them: a ``**`` unpack's entries where it stands."""
    entries: list[_Entry] = []
    for node in sea_methods(tree, "settings"):
        held: dict[str, list[_Entry]] = {}
        for statement in own_statements(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                entries.extend(_result_entries(statement.value, held))
            elif (
                isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
                and statement.value is not None
            ):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                # Read once: the names of a chained assignment hold the one dict.
                found = _result_entries(statement.value, held)
                for target in targets:
                    if isinstance(target, ast.Name):
                        if isinstance(statement, ast.AugAssign):
                            held.setdefault(target.id, []).extend(found)
                        elif statement in node.body:
                            held[target.id] = found
                        else:
                            held[target.id] = held.get(target.id, []) + found
            elif (updated := _update_call(statement)) is not None:
                held.setdefault(updated[0], []).extend(_result_entries(updated[1], held))
    return entries


def _update_call(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    """Return ``(name, value)`` for a ``name.update(value)`` statement, else ``None``."""
    call = statement.value if isinstance(statement, ast.Expr) else None
    if (
        isinstance(call, ast.Call) and call.args and isinstance(call.func, ast.Attribute)
        and call.func.attr == "update" and isinstance(call.func.value, ast.Name)
    ):
        return call.func.value.id, call.args[0]
    return None


def own_statements(node: ast.AST) -> Iterator[ast.stmt]:
    """Yield the statements of *node*'s own body in source order: a nested function or class
    definition too, but not the statements inside it.

    Walked with a stack of the nodes that can hold a statement (no
    expression can), not by recursion: a script may nest deeper than
    the interpreter's recursion limit (a long ``elif`` chain, a long
    ``+`` chain) and still run.
    """
    pending = [node]
    while pending:
        current = pending.pop()
        if current is not node:
            if isinstance(current, ast.stmt):
                yield current
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
        pending.extend(reversed([
            child for child in ast.iter_child_nodes(current)
            if isinstance(child, ast.stmt | ast.ExceptHandler | ast.match_case)
        ]))


def _result_entries(expr: ast.expr, held: dict[str, list[_Entry]]) -> list[_Entry]:
    """Return the dict-literal entries that reach the value of *expr*, in the order it lays them.

    *held* maps the method's local names to the entries they hold where
    *expr* stands; a bare name yields that very list, so a name
    assigned another (``mine = base``) shares what ``update`` adds to
    either, as the one dict object the two names share would.  A name
    read before it holds anything (the method's parameter) gets its
    list here, so that it shares one from then on as well.  Read with
    a stack, not by recursion: a ``|`` chain may be longer than the
    interpreter's recursion limit (:func:`own_statements`).
    """
    if isinstance(expr, ast.Name):
        return held.setdefault(expr.id, [])
    found: list[_Entry] = []
    pending: list[ast.expr | _Entry] = [expr]
    while pending:
        item = pending.pop()
        if isinstance(item, tuple):
            found.append(item)
        elif isinstance(item, ast.Dict):
            # A ``**`` unpack (no key): the unpacked dict's entries are settings too.
            pending.extend(reversed([
                value if key is None else (item, index)
                for index, (key, value) in enumerate(zip(item.keys, item.values, strict=True))
            ]))
        elif isinstance(item, ast.BinOp) and isinstance(item.op, ast.BitOr):
            pending += [item.right, item.left]
        elif isinstance(item, ast.IfExp):
            pending += [item.orelse, item.body]
        elif isinstance(item, ast.Name):
            found.extend(held.setdefault(item.id, []))
    return found


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
        whose value is blank (``""`` or whitespace) — it means "no
        override", so the caller's or the persisted value stands.
        Every string is an exact ``str``, whatever subclass the script
        returned.

    Raises:
        SeaError: When *declared* names an unknown, renamed or removed
            key, or a value has the wrong type.
    """
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
    # stands.  So does a blank string for a string key (the spelling of
    # "none" a script computing its model, profile or directory may
    # produce; ``_check_value`` returns ``None`` for it), as a blank
    # ``run_agent`` option does (``agent_dispatch._parse_option``).
    settings: dict[str, Any] = {}
    for key, value in declared.items():
        if value is None:
            continue
        source = f"settings()[{key!r}]"
        expected = SETTING_TYPES[key]
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            names = (
                " or ".join(t.__name__ for t in expected)
                if isinstance(expected, tuple) else expected.__name__
            )
            raise SeaError(f"{source} must be {names}, got {type(value).__name__}")
        value = _check_value(source, key, value)
        if value is not None:
            settings[key] = value
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
        base = base_dir or os.getcwd()
        return _resolved(anchored_work_dir(asked, base)) == _resolved(
            anchored_work_dir(locked, base)
        )
    return bool(asked == locked)


def _resolved(path: str) -> Path:
    """Return *path* with its symlinks resolved; as it is when it has a NUL byte in it, which
    names no file and so resolves to none (``Path.resolve`` raises on it)."""
    try:
        return Path(path).resolve()
    except ValueError:
        return Path(path)


def _check_value(source: str, key: str, value: Any) -> Any:
    """Validate one type-checked setting beyond its type; ``None`` means "no override".

    A string (a ``locked`` name too) is copied into an exact ``str``
    with ``str.__str__``, which runs no method of an untrusted
    subclass, so neither this module nor the launcher can be made to
    raise by one; a ``tool_profile`` alias becomes its key
    (:func:`alias_free_profile`) and a blank string is ``None``.
    ``max_budget`` and ``timeout`` must be finite
    (``coerce_budget_override`` would otherwise SILENTLY discard a
    NaN/infinite budget downstream) and are returned as floats.  A
    number whose own methods raise (an untrusted number subclass) is
    reported as broken.

    Raises:
        SeaError: Naming *source* (``settings()['timeout']``).
    """
    if isinstance(value, str):
        text = str.__str__(value)
        if key == "tool_profile":
            text = alias_free_profile(text)
        return text if text.strip() else None
    if key == "locked":
        lockable = [k for k in SETTING_TYPES if k not in META_SETTINGS]
        names = [str.__str__(item) if isinstance(item, str) else item for item in value]
        bad = [item for item in names if not isinstance(item, str) or item not in lockable]
        if bad:
            raise SeaError(
                f"{source} may only name settings keys ({', '.join(lockable)}); got {bad!r}"
            )
        return sorted(set(names))
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
