# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Applying a SEA to a ``run`` command.

The caller of :func:`kiss.server.sorcar.run` may supply a Sorcar
Extension Agent — a Python file defining a subclass of
:class:`kiss.agents.seas.base.base_sea.BaseSea` that configures the
run — as a file path on the ``run`` command's ``seaPath`` field.  The
client validates and resolves the path
(:func:`kiss.agents.sorcar.daemon_client.resolve_sea_path`); the
daemon loads the SEA (:func:`kiss.agents.sorcar.sea_commands.sea_layers`),
evaluates it (:func:`kiss.agents.sorcar.sea_commands.evaluate_sea`)
and applies the result in place on the command dict
(:func:`apply_sea`):

* the effective settings: per-run parameters, each
  written over the command's corresponding wire field
  (:data:`SETTING_FIELDS`) unless the caller marked that field explicit
  (:data:`~kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`: an
  explicit value ranks above the SEA's, a persisted or inherited one
  below it); a relative ``work_dir`` is anchored at the calling task's
  directory (:func:`~kiss.agents.sorcar.sea_settings.anchored_work_dir`);
* ``prompt(task)``: the task text replaced by what the method returns
  (``{task_id}`` in it -> the calling task's id);
* the channel preamble, appended to ``appendToSystemPrompt`` of a
  channel run (a SEA deriving from ``ChannelSea``);
* the ``system_prompt``, ``tools``, ``llm_call_hook`` and
  ``tool_call_hook`` methods — callables no wire field can carry —
  staged on the daemon-side fields ``systemPromptHook``, ``toolsHook``,
  ``llmCallHook`` and ``toolCallHook``, which the run applies where it
  assembles its system prompt, builds its toolset and makes its calls;
* for a channel, the workspace the run holds for its lifetime
  (:func:`channel_workspace`), which the task runner enters BEFORE the
  tools are built — a channel's ``tools()`` binds the credentials of
  the workspace active at that moment — and releases when the run
  ends.

The functions execute in the daemon process on the task's worker
thread.  A broken SEA (malformed field, missing file, import failure,
no SEA class, a raising method, a wrong-typed value) raises
:exc:`~kiss.agents.sorcar.sea_settings.SeaError` so the task stops
with a diagnostic instead of silently running with the wrong
parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar.run_config import PROVENANCE_EXPLICIT, sea_pinned
from kiss.agents.sorcar.sea_commands import SeaRun, evaluate_sea, is_channel, sea_layers, sea_name
from kiss.agents.sorcar.sea_settings import (
    DISPATCHER_SETTINGS,
    SETTING_TYPES,
    SeaError,
    anchored_work_dir,
    locked_conflicts,
    script_name,
    wire_field,
)

logger = logging.getLogger("kiss-vscode")


SETTING_FIELDS: dict[str, str] = {
    key: wire_field(key) for key in SETTING_TYPES if key not in DISPATCHER_SETTINGS
}
"""``settings()`` key -> the ``run`` command wire field it overrides.

Every key is a parameter of :func:`kiss.server.sorcar.run`; the
:data:`~kiss.agents.sorcar.sea_settings.DISPATCHER_SETTINGS` have none.
"""

CHANNEL_PREAMBLE = (
    "You are the {name} agent: this session already has the {name} tools "
    "— use them directly and immediately, without exploring any source "
    "code.  Never call run_agent here: it would just recurse into another "
    "session like this one.  Act only through those tools: never edit "
    "source files or run test suites — when a tool or the {name} CLI is "
    "broken, report the failure in your result so it is fixed in a normal "
    "development task."
)
"""System-prompt preamble of every channel run (a SEA deriving from ``ChannelSea``);
``{name}`` is the SEA's name (``CHANNEL_BEHAVIOURS`` "preamble")."""


NO_TOOLS_PROFILE = "none"
"""The tool profile of a run whose only built-in tool is ``finish``."""

RUN_CONFIG_FIELD = "_runConfig"
"""The daemon-side ``run`` command field :func:`apply_sea` leaves its provenance record in.

``{"sea": <SEA name, "" for a plain run>, "channel": <bool>,
"pinned": {key: [before, pinned]}}``.
"""


def channel_workspace(cmd: dict[str, Any], seas: list[BaseSea]) -> str:
    """Return the workspace a run holds for its lifetime; ``""`` unless it is a channel.

    The command's ``workspace`` wire field (the ``run_agent`` option
    ``workspace``, a channel launcher's account), else ``"default"``
    (``CHANNEL_BEHAVIOURS`` "workspace").  Decided from the settings
    alone, so the task runner can enter the workspace before the
    channel's ``tools()`` runs.

    Args:
        cmd: The ``run`` command dict.
        seas: The run's loaded SEAs (:func:`load_layers`).
    """
    if not is_channel(seas):
        return ""
    workspace = cmd.get("workspace")
    return workspace.strip() if isinstance(workspace, str) and workspace.strip() else "default"


def calling_work_dir(cmd: dict[str, Any]) -> str:
    """Return the caller's directory of a ``run`` command (a relative ``work_dir`` is under it).

    The ``tabScopeWorkDir`` a ``run_agent`` dispatch records (the
    CALLING task's directory), else the command's own ``workDir`` (a
    ``/<name>`` run: the tab's), else ``""``.
    """
    for field in ("tabScopeWorkDir", "workDir"):
        value = cmd.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def load_layers(cmd: dict[str, Any], base: Path | None = None) -> list[BaseSea]:
    """Load the SEA a ``run`` command names, under the tab's model-picker SEA.

    Args:
        cmd: The ``run`` command dict.  An absent, ``None`` or empty
            ``seaPath`` means "no SEA": the run is one of the bare
            :class:`BaseSea` (``base_sea.py``), the root layer every
            run goes through, so editing that file customizes every
            run made from the chat.
        base: A SEA to lay under the file's (the tab's model-picker
            SEA), or ``None``.  With no ``seaPath`` the base alone is
            the run's SEA.

    Returns:
        The SEAs (see :func:`~kiss.agents.sorcar.sea_commands.sea_layers`).

    Raises:
        SeaError: When ``seaPath`` is not a string or names no existing
            ``.py`` file, or a file fails to import, defines no SEA
            class or has malformed settings.
    """
    raw_path = cmd.get("seaPath")
    if raw_path is None or (isinstance(raw_path, str) and raw_path == ""):
        if base is None:
            return [BaseSea()]
        raw_path = str(base)
    if not isinstance(raw_path, str):
        raise SeaError(f"SEA field must be a path string, got {type(raw_path).__name__}")
    return sea_layers(Path(raw_path), base)


def apply_sea(cmd: dict[str, Any], seas: list[BaseSea] | None = None) -> set[str]:
    """Apply a ``run`` command's SEA configuration, in place.

    Evaluates the SEAs
    (:func:`~kiss.agents.sorcar.sea_commands.evaluate_sea` on the
    command's ``prompt`` and ``parentTaskId``) and writes the result
    over the command (:func:`apply_run`).

    Args:
        cmd: The ``run`` command dict; mutated in place.
        seas: The already-loaded SEAs (:func:`load_layers`), so a run
            executes its files once; ``None`` loads them from the
            command's ``seaPath``.  An empty list leaves the command
            untouched.

    Returns:
        The set of setting and prompt wire fields that were overridden
        (:func:`apply_run`).

    Raises:
        SeaError: When the ``seaPath`` field is not a string, is not
            the path of an existing ``.py`` file, names a file that
            raises at import time or defines no SEA class, has
            malformed settings, or a method raises or returns the
            wrong type; or an explicit value clashes with a locked
            setting.
    """
    if seas is None:
        seas = load_layers(cmd)
    if not seas:
        return set()
    raw_prompt = cmd.get("prompt")
    task = raw_prompt if isinstance(raw_prompt, str) else ""
    parent_task_id = cmd.get("parentTaskId")
    try:
        run = evaluate_sea(
            seas, task, parent_task_id if isinstance(parent_task_id, str) else "",
        )
    except SeaError:
        logger.warning("SEA %s rejected", seas[-1].path, exc_info=True)
        raise
    return apply_run(cmd, seas, run)


def apply_run(cmd: dict[str, Any], seas: list[BaseSea], run: SeaRun) -> set[str]:
    """Write what the SEAs evaluated to (*run*) over the ``run`` command, in place.

    One wire field per effective setting (:data:`SETTING_FIELDS`; a
    relative ``work_dir`` anchored at :func:`calling_work_dir`);
    ``prompt`` when the evaluated prompt differs from the task (a
    ``prompt`` method rewrote it, or a ``{task_id}`` of the task text
    was filled in); ``appendToSystemPrompt`` extended with
    :data:`CHANNEL_PREAMBLE` for a channel; the daemon-side hook fields
    ``systemPromptHook``, ``toolsHook``, ``llmCallHook`` and
    ``toolCallHook``, always (every chain starts at :class:`BaseSea`,
    whose methods are identities unless ``base_sea.py`` is customized),
    so a client-sent value there is dropped.  The writes are atomic:
    they happen only after everything has succeeded, so a broken SEA
    leaves the command untouched.

    Args:
        cmd: The ``run`` command dict; mutated in place.
        seas: The loaded SEAs *run* came from.
        run: Their evaluation on the command's task
            (:func:`~kiss.agents.sorcar.sea_commands.evaluate_sea`).

    Returns:
        The set of setting and prompt wire fields that were overridden
        (empty when the SEAs pin nothing; the hook fields, written on
        every run, are not listed), so the caller can tell an actual
        SEA override apart from a client-sent value.

    Raises:
        SeaError: An explicit value clashes with a locked setting.
    """
    raw_prompt = cmd.get("prompt")
    task = raw_prompt if isinstance(raw_prompt, str) else ""
    base_dir = calling_work_dir(cmd)
    # Everything below is STAGED and applied to the command only after
    # every getter has succeeded: a broken getter must leave the command
    # completely untouched, or a direct ``_run_task`` caller (no
    # dispatch-created state) would seed its run state from a partially
    # overridden command.
    staged: dict[str, Any] = {}
    for key, field in SETTING_FIELDS.items():
        if key in run.settings:
            staged[field] = run.settings[key]
    if "work_dir" in run.settings:
        staged["workDir"] = anchored_work_dir(run.settings["work_dir"], base_dir)
    # Precedence (kiss.agents.sorcar.sea_settings.locked_conflicts): a
    # value the caller passed explicitly (``provenance`` wire field)
    # wins over the script's, unless the script locks the key — then
    # the clash is an error.  Inherited and persisted values lose to
    # the script's.
    provenance = cmd.get("provenance")
    marks = provenance if isinstance(provenance, dict) else {}
    explicit = {
        key: cmd.get(field)
        for key, field in SETTING_FIELDS.items()
        if marks.get(key) == PROVENANCE_EXPLICIT
    }
    # ``timeout``, the one dispatcher setting a caller can pass
    # explicitly, has no staged write but can clash with a lock.
    if marks.get("timeout") == PROVENANCE_EXPLICIT:
        explicit["timeout"] = cmd.get("timeout")
    conflict = locked_conflicts(run.settings, explicit, base_dir)
    if conflict:
        raise SeaError(f"{script_name(str(seas[-1].path))}: {conflict}")
    for key, asked in explicit.items():
        field = SETTING_FIELDS.get(key, "")
        # An explicit value that differs from the script's wins; one that
        # agrees keeps the script's write, so the field still counts as
        # script-pinned (the task runner's ``_worktreeDecided`` mark).
        if field in staged and asked is not None and asked != "" and asked != staged[field]:
            del staged[field]
    if run.prompt != task:
        staged["prompt"] = run.prompt
    channel = is_channel(seas)
    if channel:
        suffix = cmd.get("appendToSystemPrompt")
        preamble = CHANNEL_PREAMBLE.format(name=script_name(str(seas[-1].path)))
        staged["appendToSystemPrompt"] = (
            f"{suffix}\n\n{preamble}" if isinstance(suffix, str) and suffix else preamble
        )
    overridden = set(staged)
    staged["systemPromptHook"] = run.system_prompt_hook
    staged["toolsHook"] = run.tools_hook
    staged["llmCallHook"] = run.llm_call_hook
    staged["toolCallHook"] = run.tool_call_hook
    # The provenance record the task runner folds into the run's
    # ``task_settings`` event (see :mod:`kiss.agents.sorcar.run_config`):
    # computed from the command BEFORE the writes, so it names the
    # inherited or persisted values the SEA pinned to its own (explicit
    # values were removed from ``staged`` above and never appear here).
    cmd[RUN_CONFIG_FIELD] = {
        "sea": sea_name(seas),
        "channel": channel,
        "pinned": sea_pinned(cmd, staged, SETTING_FIELDS),
    }
    cmd.update(staged)
    return overridden
