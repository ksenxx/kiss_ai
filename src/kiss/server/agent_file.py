# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-script loading for ``kiss.server.sorcar.run``'s ``extension_agent_path``.

The caller of :func:`kiss.server.sorcar.run` may supply an *agent
script* — a Python file whose top-level ``get_X()`` functions compute
the run's parameters — as a file path on the ``run`` command's
``agentPath`` field.  The client validates and resolves the path
(:func:`resolve_agent_path`); the daemon imports the file and, for
every ``run`` parameter ``X`` the script defines a ``get_X()`` for,
calls that function and overrides the command's corresponding wire
field with its return value (:func:`apply_agent_overrides`) — exactly
like the daemon calls a tools file's ``get_tools()``.  Parameters the
script defines no getter for keep the value the client sent (which is
the parameter's default when the caller did not pass one).  The
functions therefore execute in the daemon process, never serialized by
the client.  A broken agent script (malformed field, missing file,
import failure, a raising getter, or a wrong-typed return value)
raises :exc:`AgentFileError` so the task stops with a diagnostic error
instead of silently running with the wrong parameters.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

# The client-side validator lives in the sorcar layer (the ``run_agent``
# dispatch tool uses it under the layering invariant); re-exported here
# unchanged as the public ``kiss.server.agent_file.resolve_agent_path``.
from kiss.agents.sorcar.daemon_client import (
    resolve_agent_path as resolve_agent_path,
)
from kiss.server.tools_file import _safe_message, execute_python_file

logger = logging.getLogger("kiss-vscode")


class AgentFileError(Exception):
    """A ``run`` command's agent script is broken and the task must stop.

    Raised by :func:`apply_agent_overrides` when the ``agentPath`` wire
    field is malformed, names a missing or non-``.py`` path, names a
    file that raises at import time, or names a module whose
    ``get_X()`` getter is non-callable, raises, or returns a value of
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
    ("tools", "toolsFile"),
    ("use_worktree", "useWorktree"),
    ("auto_commit", "autoCommit"),
    ("max_budget", "maxBudget"),
    ("model_config", "modelConfig"),
    ("web_tools", "webTools"),
    ("is_parallel", "useParallel"),
    ("append_basic_tools", "appendBasicTools"),
)
"""The overridable ``run`` parameters, as ``(param, wire_field)`` pairs.

Each entry maps a :func:`kiss.server.sorcar.run` parameter name (the
``X`` of the agent script's optional ``get_X()`` getter) to the ``run``
command wire field it overrides.  ``timeout`` and ``sock_path`` are
absent by design: they are client-transport parameters — the script
only runs on the daemon that ``sock_path`` selects, and ``timeout``
bounds the client's local wait — so a daemon-side getter could never
take effect.
"""


def _check_override(raw_path: str, param: str, value: Any) -> Any:
    """Type-check one ``get_X()`` return value against parameter ``X``.

    Args:
        raw_path: The agent-script path, for diagnostic messages.
        param: The ``run`` parameter name the getter overrides.
        value: The getter's return value.

    Returns:
        The value to use for the parameter — *value* itself, or its
        normalized form (a ``get_tools()`` :class:`os.PathLike` becomes
        its path string, a finite ``get_max_budget()`` number becomes a
        ``float``).

    Raises:
        AgentFileError: When *value* has the wrong type for *param* —
            each parameter accepts exactly the types its
            :func:`kiss.server.sorcar.run` docstring documents
            (``prompt`` additionally must be non-empty, ``max_budget``
            finite).
    """
    ok = True
    expected = ""
    if param == "prompt":
        ok = isinstance(value, str) and bool(value.strip())
        expected = "a non-empty string"
    elif param in ("work_dir", "model", "chat_id", "system_prompt"):
        ok = isinstance(value, str)
        expected = "a string"
    elif param == "tools":
        if isinstance(value, list):
            # The agent script doubles as its own tools file: a
            # ``get_tools()`` returning the tool callables themselves
            # (the tools-file contract — e.g. every channel agent
            # module) normalizes to the script's own path, which the
            # task runner later imports as the ``toolsFile`` and whose
            # ``get_tools()`` it calls for the actual list.
            value = raw_path
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        ok = value is None or isinstance(value, str)
        expected = (
            "a tools-file path (string or pathlib.Path), a list of "
            "tool callables, or None"
        )
    elif param in (
        "use_worktree", "auto_commit", "is_parallel", "append_basic_tools",
    ):
        ok = isinstance(value, bool)
        expected = "a bool"
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
    elif param == "web_tools":
        ok = value is None or isinstance(value, bool)
        expected = "a bool or None"
    if not ok:
        raise AgentFileError(
            f"get_{param}() of agent script {raw_path!r} must return "
            f"{expected}, got {type(value).__name__}"
        )
    return value


def apply_agent_overrides(cmd: dict[str, Any]) -> set[str]:
    """Apply a ``run`` command's agent-script parameter overrides.

    Daemon-side counterpart of :func:`resolve_agent_path`: imports the
    Python file named by the command's ``agentPath`` field and, for
    every overridable ``run`` parameter ``X`` (:data:`PARAM_FIELDS`)
    whose top-level ``get_X()`` function the script defines, calls the
    function and writes its (type-checked) return value into the
    command's corresponding wire field, in place.  The writes are
    atomic: they happen only after EVERY defined getter has succeeded,
    so a broken script leaves the command untouched.  Parameters
    without a getter keep the field value the client sent.  A ``get_tools()``
    return value is a tools-file *path* written to the ``toolsFile``
    field — the task runner later imports that file and calls its
    ``get_tools()`` exactly as for a client-passed ``tools`` path.  A
    ``get_tools()`` returning a *list* of tool callables instead (the
    tools-file contract) makes the agent script its own tools file:
    the script's path is written to ``toolsFile``.

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
        actual ``get_X()`` override apart from a client-sent value.

    Raises:
        AgentFileError: When the ``agentPath`` field is not a string,
            is not the path of an existing ``.py`` file, names a module
            that raises at import time, or names a module with a
            non-callable ``get_X``, a ``get_X()`` that raises, or a
            ``get_X()`` return value of the wrong type.
    """
    raw_path = cmd.get("agentPath")
    if raw_path is None:
        return set()
    if isinstance(raw_path, str) and raw_path == "":
        return set()
    namespace = execute_python_file(raw_path, AgentFileError, "agent script")
    # Overrides are STAGED and applied to the command only after every
    # getter has succeeded: a broken getter must leave the command
    # completely untouched, or a direct ``_run_task`` caller (no
    # dispatch-created state) would seed its run state from a partially
    # overridden command — e.g. an earlier successful ``get_chat_id()``
    # surviving a later getter's failure.
    staged: dict[str, Any] = {}
    for param, field in PARAM_FIELDS:
        getter_name = f"get_{param}"
        # Membership (not ``.get() is None``) decides absence: a
        # DEFINED ``get_X = None`` is a broken getter, not a missing
        # one, and must stop the task like any other non-callable.
        if getter_name not in namespace:
            continue
        getter = namespace[getter_name]
        if not callable(getter):
            raise AgentFileError(
                f"{getter_name} of agent script {raw_path!r} must be a "
                f"callable, got {type(getter).__name__}"
            )
        try:
            value = getter()
        except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
            logger.warning(
                "get_%s() of agentPath %r raised", param, raw_path,
                exc_info=True,
            )
            raise AgentFileError(
                f"get_{param}() of agent script {raw_path!r} raised: "
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
                "Validating get_%s() result of agentPath %r raised",
                param,
                raw_path,
                exc_info=True,
            )
            raise AgentFileError(
                f"get_{param}() of agent script {raw_path!r} returned a "
                f"broken value: {_safe_message(exc)}"
            ) from exc
        staged[field] = value
    cmd.update(staged)
    return set(staged)
