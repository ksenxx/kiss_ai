# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Applying an agent script (SEA) to a ``run`` command.

The caller of :func:`kiss.server.sorcar.run` may supply an *agent
script* — a Sorcar Extension Agent, a Python file that configures the
run — as a file path on the ``run`` command's ``agentPath`` field.  The
client validates and resolves the path
(:func:`kiss.agents.sorcar.daemon_client.resolve_agent_path`); the
daemon executes the script and the scripts it extends
(:func:`kiss.agents.sorcar.sea_commands.sea_layers`), evaluates them
(:func:`kiss.agents.sorcar.sea_commands.evaluate_sea`) and applies the
result in place on the command dict (:func:`apply_agent_overrides`):

* the merged ``settings()``: a ``preset`` plus per-run parameters, each
  written over the command's corresponding wire field
  (:data:`SETTING_FIELDS`), so a script's choice wins over whatever
  the caller sent;
* ``prompt(task)``: the task text replaced by what the function
  returns (``{task_id}`` in it -> the calling task's id);
* ``system_prompt()``, written over ``systemPrompt`` (the run's base
  system prompt);
* the ``channel`` preset's preamble and ``add_to_system_prompt()``,
  appended to ``appendToSystemPrompt``;
* ``add_to_tools()`` callables and ``llm_call_hook()`` /
  ``tool_call_hook()`` hooks — values no wire field can carry — staged
  on the daemon-side fields ``tools`` / ``llmCallHook`` / ``toolCallHook``;
* for a ``channel`` preset, the workspace the run holds for its
  lifetime (:func:`channel_workspace`), which the task runner enters
  BEFORE the tools are built — a channel's ``add_to_tools()`` binds the
  credentials of the workspace active at that moment — and releases
  when the run ends.

The functions execute in the daemon process on the task's worker
thread.  A broken agent script (malformed field, missing file, import
failure, a raising getter, a wrong-typed value) raises
:exc:`AgentFileError` so the task stops with a diagnostic instead of
silently running with the wrong parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sea_commands import (
    SeaLayer,
    SeaScriptError,
    evaluate_sea,
    join_text,
    sea_layers,
)
from kiss.agents.sorcar.sea_settings import (
    SETTING_TYPES,
    SeaError,
    merge_settings,
    script_name,
)

logger = logging.getLogger("kiss-vscode")


class AgentFileError(SeaError):
    """A ``run`` command's agent script is broken and the task must stop.

    Raised by :func:`apply_agent_overrides` when the ``agentPath`` wire
    field is malformed, names a missing or non-``.py`` path, names a
    file that raises at import time, or names a module whose
    ``settings()`` or getters are non-callable, raise, or return a
    value of the wrong type.  The task runner turns the raise into a
    failed task result whose text carries this exception's diagnostic
    message, so a broken agent script stops the task loudly instead of
    silently running it with parameters the script did not compute.
    """


def wire_field(key: str) -> str:
    """Return the ``run`` command wire field of the ``run()`` keyword *key*.

    The wire vocabulary is the keyword vocabulary in camelCase
    (``use_web_tools`` -> ``useWebTools``), with two aliases kept from
    the wire protocol's earlier vocabulary: ``add_to_prompt`` ->
    ``appendToPrompt`` and ``add_to_system_prompt`` ->
    ``appendToSystemPrompt``.
    """
    aliases = {"add_to_prompt": "appendToPrompt", "add_to_system_prompt": "appendToSystemPrompt"}
    if key in aliases:
        return aliases[key]
    first, *rest = key.split("_")
    return first + "".join(part.capitalize() for part in rest)


DISPATCHER_SETTINGS = ("preset", "extends", "timeout", "inherit", "kind")
"""``settings()`` keys with no ``run`` command wire field.

``preset`` and ``extends`` are resolved by
:func:`~kiss.agents.sorcar.sea_commands.sea_layers`; ``timeout`` and
``inherit`` are read by the dispatcher
(:mod:`kiss.agents.sorcar.agent_dispatch`); ``kind`` is read from the
layers by :func:`channel_workspace` and :func:`apply_agent_overrides`.
"""

SETTING_FIELDS: dict[str, str] = {
    key: wire_field(key) for key in SETTING_TYPES if key not in DISPATCHER_SETTINGS
}
"""``settings()`` key -> the ``run`` command wire field it overrides.

Every key is a parameter of :func:`kiss.server.sorcar.run`.
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
"""System-prompt preamble of every ``kind: "channel"`` run; ``{name}`` is the script's name."""


def is_channel(layers: list[SeaLayer]) -> bool:
    """Return whether the run of *layers* is a channel agent (``kind: "channel"``)."""
    if not layers:
        return False
    return merge_settings([layer.settings for layer in layers]).get("kind") == "channel"

NO_TOOLS_PROFILE = "none"
"""The tool profile of a run whose only built-in tool is ``finish``."""

DAEMON_SIDE_FIELDS = ("tools", "llmCallHook", "toolCallHook")
"""Command fields only the daemon's SEA pipeline may set; a client-sent value is dropped."""


def channel_workspace(cmd: dict[str, Any], layers: list[SeaLayer]) -> str:
    """Return the workspace a run holds for its lifetime; ``""`` unless its kind is ``channel``.

    The command's ``workspace`` wire field (the ``run_agent`` option
    ``workspace``, a channel launcher's account), else ``"default"``.
    Decided from the layers' settings alone, so the task runner can
    enter the workspace before :func:`apply_agent_overrides` evaluates
    the channel's ``add_to_tools()``.

    Args:
        cmd: The ``run`` command dict.
        layers: The run's executed layers (:func:`load_layers`).
    """
    if not is_channel(layers):
        return ""
    workspace = cmd.get("workspace")
    return workspace.strip() if isinstance(workspace, str) and workspace.strip() else "default"


def load_layers(cmd: dict[str, Any], base: Path | None = None) -> list[SeaLayer]:
    """Execute the agent script a ``run`` command names, and its bases.

    Args:
        cmd: The ``run`` command dict.  An absent, ``None`` or empty
            ``agentPath`` means "no agent script": an empty list.
        base: An outermost layer to lay under the script (the tab's
            model-picker SEA), or ``None``.  With no ``agentPath`` the
            base alone is the run's script.

    Returns:
        The layers (see :func:`~kiss.agents.sorcar.sea_commands.sea_layers`).

    Raises:
        AgentFileError: When ``agentPath`` is not a string or names no
            existing ``.py`` file, or a script of the chain fails to
            import, has malformed settings or a bad ``extends``.
    """
    raw_path = cmd.get("agentPath")
    if raw_path is None or (isinstance(raw_path, str) and raw_path == ""):
        if base is None:
            return []
        raw_path = str(base)
    if not isinstance(raw_path, str):
        raise AgentFileError(
            f"agent script field must be a path string, got {type(raw_path).__name__}"
        )
    try:
        return sea_layers(Path(raw_path), base)
    except SeaScriptError as exc:
        raise AgentFileError(str(exc)) from exc


def apply_agent_overrides(
    cmd: dict[str, Any], layers: list[SeaLayer] | None = None,
) -> set[str]:
    """Apply a ``run`` command's agent-script configuration, in place.

    Evaluates the script's layers
    (:func:`~kiss.agents.sorcar.sea_commands.evaluate_sea` on the
    command's ``prompt`` and ``parentTaskId``) and writes the result
    over the command: one wire field per merged setting
    (:data:`SETTING_FIELDS`); ``prompt`` when a ``prompt(task)`` getter
    rewrote the task; ``systemPrompt`` from ``system_prompt()``;
    ``appendToSystemPrompt`` extended with :data:`CHANNEL_PREAMBLE`
    (``kind: "channel"``) and ``add_to_system_prompt()``; the
    daemon-side fields ``tools``, ``llmCallHook`` and ``toolCallHook``.
    The writes are atomic: they happen only after everything has
    succeeded, so a broken script leaves the command untouched.

    Args:
        cmd: The ``run`` command dict; mutated in place.
        layers: The already-executed layers (:func:`load_layers`), so
            a run executes its scripts once; ``None`` loads them from
            the command's ``agentPath``.  An empty list (no agent
            script) leaves the command untouched.

    Returns:
        The set of command-field names that were overridden (empty when
        the command carries no agent script), so the caller can tell an
        actual script override apart from a client-sent value.

    Raises:
        AgentFileError: When the ``agentPath`` field is not a string,
            is not the path of an existing ``.py`` file, names a module
            that raises at import time, has malformed settings, a
            non-callable or raising getter, or a getter returning the
            wrong type.
    """
    if layers is None:
        layers = load_layers(cmd)
    if not layers:
        return set()
    raw_prompt = cmd.get("prompt")
    parent_task_id = cmd.get("parentTaskId")
    try:
        run = evaluate_sea(
            layers,
            raw_prompt if isinstance(raw_prompt, str) else "",
            parent_task_id if isinstance(parent_task_id, str) else "",
        )
    except SeaScriptError as exc:
        logger.warning("agent script %s rejected: %s", layers[-1].path, exc)
        raise AgentFileError(str(exc)) from exc
    # Everything below is STAGED and applied to the command only after
    # every getter has succeeded: a broken getter must leave the command
    # completely untouched, or a direct ``_run_task`` caller (no
    # dispatch-created state) would seed its run state from a partially
    # overridden command.
    staged: dict[str, Any] = {}
    for key, field in SETTING_FIELDS.items():
        if key in run.settings:
            staged[field] = run.settings[key]
    if any("prompt" in layer.namespace for layer in layers):
        staged["prompt"] = run.prompt
    if run.system_prompt is not None:
        staged["systemPrompt"] = run.system_prompt
    system_suffix = cmd.get("appendToSystemPrompt")
    if run.settings.get("kind") == "channel":
        system_suffix = _add_text(
            system_suffix, CHANNEL_PREAMBLE.format(name=script_name(str(layers[-1].path))),
        )
        staged["appendToSystemPrompt"] = system_suffix
    if run.add_to_system_prompt:
        staged["appendToSystemPrompt"] = _add_text(system_suffix, run.add_to_system_prompt)
    if any("add_to_tools" in layer.namespace for layer in layers):
        staged["tools"] = run.tools
    if any("llm_call_hook" in layer.namespace for layer in layers):
        staged["llmCallHook"] = run.llm_call_hook
    if any("tool_call_hook" in layer.namespace for layer in layers):
        staged["toolCallHook"] = run.tool_call_hook
    cmd.update(staged)
    return set(staged)


def _add_text(base: Any, addition: str) -> str:
    """Return *addition* appended to *base* (a wire value; non-strings count as empty)."""
    return join_text(base if isinstance(base, str) else "", addition)
