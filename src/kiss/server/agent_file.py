# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-script loading for ``kiss.server.sorcar.run``'s ``extension_agent_path``.

The caller of :func:`kiss.server.sorcar.run` may supply an *agent
script* — a Sorcar Extension Agent (SEA), a Python file whose top-level
``X()`` functions compute the run's parameters — as a file path on
the ``run`` command's ``agentPath`` field.  The client validates and resolves the path
(:func:`kiss.agents.sorcar.daemon_client.resolve_agent_path`); the
daemon imports the file and, for
every ``run`` parameter ``X`` the script defines a top-level ``X()``
function for, calls that function and overrides the command's
corresponding wire field with its return value
(:func:`apply_agent_overrides`) — exactly like the daemon calls a
tools file's ``get_tools()``.  Parameters the script defines no getter
for keep the value the client sent (which is the parameter's default
when the caller did not pass one).  The functions therefore execute in
the daemon process, never serialized by the client.  A broken agent
script (malformed field, missing file, import failure, a raising
getter, or a wrong-typed return value) raises :exc:`AgentFileError` so
the task stops with a diagnostic error instead of silently running
with the wrong parameters.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from kiss.server.tools_file import _safe_message, execute_python_file

logger = logging.getLogger("kiss-vscode")


class AgentFileError(Exception):
    """A ``run`` command's agent script is broken and the task must stop.

    Raised by :func:`apply_agent_overrides` when the ``agentPath`` wire
    field is malformed, names a missing or non-``.py`` path, names a
    file that raises at import time, or names a module whose
    ``X()`` getter is non-callable, raises, or returns a value of
    the wrong type for parameter ``X``.  The task runner turns the
    raise into a failed task result whose text carries this exception's
    diagnostic message, so a broken agent script stops the task loudly
    instead of silently running it with parameters the script did not
    compute.
    """


PARAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("prompt", "prompt"),
    ("work_dir", "workDir"),
    ("model", "model"),
    ("chat_id", "chatId"),
    ("system_prompt", "systemPrompt"),
    ("use_worktree", "useWorktree"),
    ("auto_commit", "autoCommit"),
    ("max_budget", "maxBudget"),
    ("model_config", "modelConfig"),
    ("append_to_system_prompt", "appendToSystemPrompt"),
    ("append_to_prompt", "appendToPrompt"),
    ("use_web_tools", "webTools"),
    ("classify_tasks", "classifyTasks"),
    ("use_memory", "useMemory"),
    ("is_parallel", "useParallel"),
    ("tool_profile", "toolProfile"),
    ("docker_image", "dockerImage"),
)
"""The overridable ``run`` parameters, as ``(getter_name, wire_field)`` pairs.

Each entry maps the agent script's optional top-level ``X()`` getter
to the ``run`` command wire field it overrides.  The getter name is
the :func:`kiss.server.sorcar.run` parameter name.  ``tools`` and
``append_basic_tools`` are not here: a script's tool set comes from
the :data:`TOOL_FIELDS` getters ``tools()`` / ``add_to_tools()``,
which set both wire fields together.  ``scope_work_dir`` has no
getter: the calling workspace recorded on the run's registry tab is
the caller's identity, not the script's.  ``timeout``,
``stop_on_timeout``, and ``endpoint_file`` are absent by design: they are
client-transport parameters — the script only runs on the daemon that
``endpoint_file`` selects, ``timeout`` bounds the client's local wait, and
``stop_on_timeout`` picks the client's timeout behavior — so a
daemon-side getter could never take effect.
``use_web_tools()`` (wire field ``webTools``),
``classify_tasks()`` (wire field ``classifyTasks``), and
``use_memory()`` (wire field ``useMemory``) return a bool for a
per-run override or ``None`` to fall back to the daemon's default —
the persisted setting, except that ``use_memory``'s fallback also
honours a non-empty ``KISS_USE_MEMORY`` environment variable over the
stored value (``sorcar_agent._memory_settings``);
``is_parallel()`` (wire field ``useParallel``) returns a bool.
``tool_profile()`` (wire field ``toolProfile``) returns the name of the
tool profile the run's built-in toolset is cut down to — a key of
``sorcar_agent.TOOL_PROFILES`` (``"full"``, ``"review"``, ``"shell"``,
``"assistant"``, ``"bash"``) or ``""`` for the daemon's usual choice; the task runner
rejects an unknown name when the task starts.
``docker_image()`` (wire field ``dockerImage``) returns the Docker
image the run's shell and file tools execute in, or
``container:<name-or-id>`` to attach to a running container, or ``""``
for the host.
``parent_task_id`` / ``parent_tab_id`` (wire fields ``parentTaskId``
/ ``parentTabId``) are absent by design: they are the
CALLING task's identity — what marks the dispatched run as that
task's sub-agent — which the dispatched script must not be able to
forge or re-parent.
"""

TOOL_FIELDS: tuple[tuple[str, bool], ...] = (
    ("tools", False),
    ("add_to_tools", True),
)
"""The agent-script tool getters, as ``(getter_name, append_basic_tools)`` pairs.

Each getter returns the list of tool callables the script contributes;
the script then doubles as the run's tools file (its path is written to
``toolsFile`` and the task runner later imports it and calls the same
getter for the list, like a tools file's ``get_tools()``).  The second
element is the ``appendBasicTools`` wire value the getter implies:

- ``tools()`` — the run's tool set is EXACTLY these tools plus
  ``finish``; the built-in basic toolset is not built.
- ``add_to_tools()`` — these tools are ADDED to the built-in basic
  toolset (and ``finish``).

A script defines at most one of the two; returning a tools-file path
is not accepted.  A script with neither keeps the client-sent
``toolsFile`` / ``appendBasicTools`` values.
"""

HOOK_FIELDS: tuple[tuple[str, str], ...] = (
    ("llm_call_hook", "llmCallHook"),
    ("tool_call_hook", "toolCallHook"),
)
"""Agent-script-only hook getters, as ``(param, command_field)`` pairs.

Each entry maps an agent-script getter name (``llm_call_hook`` /
``tool_call_hook``) to the run-command field its returned callable
is staged into.  Unlike :data:`PARAM_FIELDS` these are NOT
:func:`kiss.server.sorcar.run` parameters and their fields are never
sent on the wire — a callable cannot be serialized, so the hooks exist
ONLY as agent-script getters, evaluated in the daemon process.  The
task runner forwards the staged callables to the underlying
:meth:`kiss.core.kiss_agent.KISSAgent.run` as its ``llm_call_hook`` /
``tool_call_hook`` arguments (see that docstring for the hooks'
semantics: ``llm_call_hook`` may rewrite the new messages before every
LLM call, ``tool_call_hook`` may veto every tool call).
"""

ADD_FIELDS: tuple[tuple[str, str], ...] = (
    ("add_to_system_prompt", "appendToSystemPrompt"),
)
"""Agent-script getters whose text is ADDED to a wire field, as ``(getter, field)`` pairs.

``add_to_system_prompt()`` returns text — a SEA's *model routing
protocol* — that is appended to the run's system prompt AFTER whatever
``appendToSystemPrompt`` already carries (the caller's text, or the
value an ``append_to_system_prompt()`` getter staged), separated by a
blank line.  Unlike the :data:`PARAM_FIELDS` getters it never replaces
the caller's value, so a SEA can carry its protocol into every run
while the caller's own additions survive.  A SEA that defines this
getter and whose ``register_as_model()`` returns ``True`` is also listed
in the model picker under its command name
(:func:`kiss.agents.sorcar.sea_commands.model_seas`); picking it runs
every task of the tab through the SEA.  ``register_as_model`` is a
registry flag, not a run parameter, so it is not evaluated here; nor is
``on_picked_as_model(work_dir)``, the hook
:func:`kiss.agents.sorcar.sea_commands.run_picked_hook` runs when the SEA
is picked and once per run whose model is the SEA (``autorouter``
schedules its weekly ``/rsi7d autorouter`` cron job there).
"""


def _check_override(raw_path: str, param: str, value: Any) -> Any:
    """Type-check one ``X()`` return value against parameter ``X``.

    Args:
        raw_path: The agent-script path, for diagnostic messages.
        param: The getter name (:data:`PARAM_FIELDS` /
            :data:`HOOK_FIELDS` / :data:`ADD_FIELDS` first element)
            whose ``{param}()`` produced *value*.
        value: The getter's return value.

    Returns:
        The value to use for the parameter — *value* itself, or its
        normalized form (a finite ``max_budget()`` number becomes a
        ``float``, a ``tools()`` / ``add_to_tools()`` tuple becomes a
        list).

    Raises:
        AgentFileError: When *value* has the wrong type for *param* —
            each parameter accepts exactly the types its
            :func:`kiss.server.sorcar.run` docstring documents
            (``prompt`` additionally must be non-empty, ``max_budget``
            finite); the :data:`TOOL_FIELDS` getters must return a list
            or tuple of callables (a tools-file path is rejected); the
            :data:`HOOK_FIELDS` getters must return a callable or
            ``None``.
    """
    ok = True
    expected = ""
    if param == "prompt":
        ok = isinstance(value, str) and bool(value.strip())
        expected = "a non-empty string"
    elif param in (
        "work_dir", "model", "chat_id", "system_prompt",
        "append_to_system_prompt", "add_to_system_prompt", "append_to_prompt",
        "tool_profile", "docker_image",
    ):
        ok = isinstance(value, str)
        expected = "a string"
    elif param in ("tools", "add_to_tools"):
        ok = isinstance(value, (list, tuple)) and all(
            callable(tool) for tool in value
        )
        expected = "a list of tool callables (not a tools-file path)"
        if ok:
            value = list(value)
    elif param in ("use_worktree", "auto_commit", "is_parallel"):
        ok = isinstance(value, bool)
        expected = "a bool"
    elif param in ("use_web_tools", "classify_tasks", "use_memory"):
        # ``None`` means "no per-run override": the run then falls back
        # to the daemon's default, exactly like an absent ``webTools``
        # / ``classifyTasks`` / ``useMemory`` wire field — the
        # persisted setting for the first two; for ``use_memory`` a
        # non-empty ``KISS_USE_MEMORY`` environment variable wins over
        # the stored value (``sorcar_agent._memory_settings``).
        ok = value is None or isinstance(value, bool)
        expected = "a bool or None"
    elif param == "max_budget":
        # Mirror ``coerce_budget_override``'s acceptance exactly: a
        # NaN/infinite/overflowing number would pass a bare isinstance
        # check here only to be SILENTLY discarded downstream — the
        # loader must reject it loudly instead.
        expected = "a finite number or None"
        if value is not None:
            ok = isinstance(value, (int, float)) and not isinstance(
                value, bool,
            )
            if ok:
                try:
                    value = float(value)
                except OverflowError:
                    ok = False
                else:
                    ok = math.isfinite(value)
    elif param == "model_config":
        ok = value is None or isinstance(value, dict)
        expected = "a dict or None"
    elif param in ("llm_call_hook", "tool_call_hook"):
        ok = value is None or callable(value)
        expected = "a callable or None"
    if not ok:
        raise AgentFileError(
            f"{param}() of agent script {raw_path!r} must return "
            f"{expected}, got {type(value).__name__}"
        )
    return value


def apply_agent_overrides(cmd: dict[str, Any]) -> set[str]:
    """Apply a ``run`` command's agent-script parameter overrides.

    Daemon-side counterpart of :func:`resolve_agent_path`: imports the
    Python file named by the command's ``agentPath`` field and, for
    every overridable ``run`` parameter ``X`` (:data:`PARAM_FIELDS`)
    whose top-level ``X()`` function the script defines, calls the
    function and writes its (type-checked) return value into the
    command's corresponding wire field, in place.  The writes are
    atomic: they happen only after EVERY defined getter has succeeded,
    so a broken script leaves the command untouched.  Parameters
    without a getter keep the field value the client sent.

    The script's tool set comes from the :data:`TOOL_FIELDS` getters.
    ``tools()`` returns a list of tool callables that, with ``finish``,
    become the run's ENTIRE tool set (``appendBasicTools`` is set to
    ``False``); ``add_to_tools()`` returns a list of tool callables
    ADDED to the built-in basic toolset (``appendBasicTools`` is set to
    ``True``).  Either way the script is its own tools file: its path is
    written to ``toolsFile`` and the task runner later imports it and
    calls the same getter for the list.  Defining both getters, or
    returning a tools-file path, is an error.

    The script may additionally define ``llm_call_hook()`` and
    ``tool_call_hook()`` (:data:`HOOK_FIELDS`), each returning a
    callable (or ``None``) that the task runner passes to the
    underlying :meth:`kiss.core.kiss_agent.KISSAgent.run` as its
    ``llm_call_hook`` / ``tool_call_hook`` argument.  Their staged
    fields (``llmCallHook`` / ``toolCallHook``) live only on the
    daemon-side command dict — callables are never wire-serialized.
    An ``add_to_system_prompt()`` getter (:data:`ADD_FIELDS`) is
    evaluated last: its text is appended to the ``appendToSystemPrompt``
    field — after the caller's text or the staged
    ``append_to_system_prompt()`` value — instead of replacing it.

    The getters run in the daemon process on the task's worker thread,
    like a tools file's ``get_tools()``, and the file is re-imported
    from source on every run (no ``__pycache__``).

    Args:
        cmd: The ``run`` command dict; mutated in place.  An absent,
            ``None``, or empty ``agentPath`` field means "no agent
            script" and leaves the command untouched.

    Returns:
        The set of wire-field names that were overridden (empty when
        the command carries no agent script), so the caller can tell an
        actual ``X()`` override apart from a client-sent value.

    Raises:
        AgentFileError: When the ``agentPath`` field is not a string,
            is not the path of an existing ``.py`` file, names a module
            that raises at import time, or names a module with a
            non-callable getter ``X``, an ``X()`` that raises, an
            ``X()`` return value of the wrong type, or both ``tools()``
            and ``add_to_tools()``.
    """
    raw_path = cmd.get("agentPath")
    if raw_path is None:
        return set()
    if isinstance(raw_path, str) and raw_path == "":
        return set()
    namespace = execute_python_file(raw_path, AgentFileError, "agent script")
    if "tools" in namespace and "add_to_tools" in namespace:
        raise AgentFileError(
            f"agent script {raw_path!r} defines both tools() and "
            f"add_to_tools(); define at most one"
        )
    tool_getters = dict(TOOL_FIELDS)
    # Overrides are STAGED and applied to the command only after every
    # getter has succeeded: a broken getter must leave the command
    # completely untouched, or a direct ``_run_task`` caller (no
    # dispatch-created state) would seed its run state from a partially
    # overridden command — e.g. an earlier successful ``chat_id()``
    # surviving a later getter's failure.
    staged: dict[str, Any] = {}
    # ``ADD_FIELDS`` last: an addition applies on top of the value an
    # ``append_to_system_prompt()`` getter may have staged.
    tool_params = tuple((param, "toolsFile") for param in tool_getters)
    for param, field in PARAM_FIELDS + tool_params + HOOK_FIELDS + ADD_FIELDS:
        # Membership (not ``.get() is None``) decides absence: a
        # DEFINED ``X = None`` is a broken getter, not a missing
        # one, and must stop the task like any other non-callable.
        if param not in namespace:
            continue
        getter = namespace[param]
        if not callable(getter):
            raise AgentFileError(
                f"{param} of agent script {raw_path!r} must be a "
                f"callable, got {type(getter).__name__}"
            )
        try:
            value = getter()
        except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
            logger.warning(
                "%s() of agentPath %r raised", param, raw_path,
                exc_info=True,
            )
            raise AgentFileError(
                f"{param}() of agent script {raw_path!r} raised: "
                f"{_safe_message(exc)}"
            ) from exc
        # Validate inside a BaseException guard: the returned value is
        # untrusted module data, so even validating it (e.g. a ``str``
        # subclass overriding ``strip``, an ``int`` subclass overriding
        # ``__float__``, or a raising ``__fspath__``) may raise
        # anything — such a raise must become an AgentFileError
        # diagnostic, not kill the task thread.
        try:
            value = _check_override(raw_path, param, value)
        except AgentFileError:
            raise
        except BaseException as exc:  # noqa: BLE001 — untrusted module data may raise anything
            logger.warning(
                "Validating %s() result of agentPath %r raised",
                param,
                raw_path,
                exc_info=True,
            )
            raise AgentFileError(
                f"{param}() of agent script {raw_path!r} returned a "
                f"broken value: {_safe_message(exc)}"
            ) from exc
        if param == "add_to_system_prompt":
            value = _add_text(staged.get(field, cmd.get(field)), value)
        elif param in tool_getters:
            # The script is its own tools file: the task runner
            # re-imports it and calls this getter for the list.
            staged["appendBasicTools"] = tool_getters[param]
            value = raw_path
        staged[field] = value
    cmd.update(staged)
    return set(staged)


def _add_text(base: Any, addition: str) -> str:
    """Return *addition* appended to *base* (a wire value; non-strings count as empty)."""
    if not isinstance(base, str) or not base:
        return addition
    return f"{base}\n\n{addition}" if addition else base
