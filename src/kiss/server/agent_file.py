# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-script loading for ``kiss.server.sorcar.run``'s ``extension_agent_path``.

The caller of :func:`kiss.server.sorcar.run` may supply an *agent
script* — a Sorcar Extension Agent (SEA), a Python file that configures
the run — as a file path on the ``run`` command's ``agentPath`` field.
The client validates and resolves the path
(:func:`kiss.agents.sorcar.daemon_client.resolve_agent_path`); the
daemon imports the file (:func:`apply_agent_overrides`) and applies, in
place on the command dict:

* the script's ``settings()`` dict (:mod:`kiss.agents.sorcar.sea_settings`):
  a ``preset`` plus per-run parameters, each written over the
  command's corresponding wire field (:data:`SETTING_FIELDS`), so a
  script's choice wins over whatever the caller sent;
* the ``channel`` preset's preamble and the script's
  ``add_to_system_prompt()`` text, appended to ``appendToSystemPrompt``;
* the script's ``add_to_tools()`` callables and ``llm_call_hook()`` /
  ``tool_call_hook()`` hooks — values no wire field can carry — staged
  on the daemon-side fields ``tools`` / ``llmCallHook`` / ``toolCallHook``
  for the task runner.

The functions execute in the daemon process on the task's worker
thread.  A broken agent script (malformed field, missing file, import
failure, a raising getter, a wrong-typed value) raises
:exc:`AgentFileError` so the task stops with a diagnostic instead of
silently running with the wrong parameters.
"""

from __future__ import annotations

import logging
import sys
import types
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sea_settings import (
    SETTING_TYPES,
    SettingsError,
    resolve_settings,
    script_name,
)

logger = logging.getLogger("kiss-vscode")


def _safe_message(exc: BaseException) -> str:
    """Format an untrusted exception without running its raising code.

    ``str(exc)`` runs the exception's ``__str__``, which — for an
    exception minted by an untrusted agent script — may itself raise
    anything.  A diagnostic built here must never leak such a
    secondary raise, so the string conversion is guarded and falls
    back to the (trusted) type name alone.

    Args:
        exc: The exception raised by untrusted agent-script code.

    Returns:
        ``"TypeName: message"`` when the message renders, otherwise
        ``"TypeName"``.
    """
    name = type(exc).__name__
    try:
        return f"{name}: {exc}"
    except BaseException:  # noqa: BLE001 — untrusted __str__ may raise anything
        return name


class AgentFileError(Exception):
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
    (``use_web_tools`` -> ``useWebTools``), so no hand-kept table is
    needed.
    """
    first, *rest = key.split("_")
    return first + "".join(part.capitalize() for part in rest)


SETTING_FIELDS: dict[str, str] = {
    key: wire_field(key)
    for key in SETTING_TYPES
    if key not in ("preset", "timeout", "add_to_prompt")
}
"""``settings()`` key -> the ``run`` command wire field it overrides.

Every key is a parameter of :func:`kiss.server.sorcar.run`.  Three
keys have no field of their own: the ``preset`` is expanded by
:func:`~kiss.agents.sorcar.sea_settings.resolve_settings`, the
``timeout`` is read by the dispatcher
(:mod:`kiss.agents.sorcar.agent_dispatch`), and ``add_to_prompt`` is
appended to the caller's ``appendToPrompt`` text.
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
"""System-prompt preamble of every ``channel``-preset run; ``{name}`` is the script's name."""

NO_TOOLS_PROFILE = "none"
"""The tool profile of a run whose only built-in tool is ``finish``."""


def _getter_value(namespace: dict[str, Any], raw_path: str, name: str) -> Any:
    """Call the script's zero-argument getter *name*; ``None`` when undefined.

    Membership (not ``.get() is None``) decides absence: a DEFINED
    ``name = None`` is a broken getter, not a missing one.

    Raises:
        AgentFileError: When the getter is not callable or raises.
    """
    if name not in namespace:
        return None
    getter = namespace[name]
    if not callable(getter):
        raise AgentFileError(
            f"{name} of agent script {raw_path!r} must be a callable, "
            f"got {type(getter).__name__}"
        )
    try:
        return getter()
    except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
        logger.warning("%s() of agentPath %r raised", name, raw_path, exc_info=True)
        raise AgentFileError(
            f"{name}() of agent script {raw_path!r} raised: {_safe_message(exc)}"
        ) from exc


def _check_tools(raw_path: str, name: str, value: Any) -> list[Any]:
    """Return *value* as a list of tool callables, or raise :exc:`AgentFileError`."""
    try:
        if isinstance(value, list | tuple) and all(callable(tool) for tool in value):
            return list(value)
    except BaseException as exc:  # noqa: BLE001 — an untrusted list may raise while iterated
        raise AgentFileError(
            f"{name}() of agent script {raw_path!r} returned a broken list: "
            f"{_safe_message(exc)}"
        ) from exc
    raise AgentFileError(
        f"{name}() of agent script {raw_path!r} must return a list of tool "
        f"callables (not a file path), got {type(value).__name__}"
    )


def _check_text(raw_path: str, name: str, value: Any) -> str:
    """Return *value* as a string, or raise :exc:`AgentFileError`."""
    if isinstance(value, str):
        return value
    raise AgentFileError(
        f"{name}() of agent script {raw_path!r} must return a string, "
        f"got {type(value).__name__}"
    )


def _check_hook(raw_path: str, name: str, value: Any) -> Any:
    """Return *value* when it is a callable or ``None``, or raise :exc:`AgentFileError`."""
    if value is None or callable(value):
        return value
    raise AgentFileError(
        f"{name}() of agent script {raw_path!r} must return a callable or "
        f"None, got {type(value).__name__}"
    )


def execute_python_file(
    raw_path: Any,
    error_cls: type[Exception],
    label: str,
) -> dict[str, Any]:
    """Import a caller-supplied Python file and return its namespace.

    Daemon-side loader for the ``run`` command's ``agentPath`` agent
    script (also used by SEAs that load other scripts, e.g.
    ``skillopt``).  The source is compiled and executed directly (no
    ``__pycache__`` read or write), so every run observes the file's
    CURRENT contents and the caller's directory is never littered with
    bytecode.

    Args:
        raw_path: The wire field naming the file — expected to be an
            absolute path string, but treated as untrusted.
        error_cls: The exception class to raise on any failure (e.g.
            :exc:`AgentFileError`), so each caller keeps its own
            diagnostic type.
        label: Human-readable name of the file kind (e.g. ``"agent
            script"``), used in diagnostic messages.

    Returns:
        The executed module's namespace dict.

    Raises:
        Exception: An *error_cls* instance when *raw_path* is not a
            string, is not the path of an existing ``.py`` file, or
            names a module that raises at import time.
    """
    # Type-check FIRST: comparing or repr-ing an untrusted non-string
    # object could run arbitrary code (raising ``__eq__``/``__repr__``),
    # so nothing touches *raw_path* beyond isinstance until it is known
    # to be a plain string.
    if not isinstance(raw_path, str):
        raise error_cls(
            f"{label} field must be a path string, got "
            f"{type(raw_path).__name__}"
        )
    path = Path(raw_path)
    try:
        is_py_file = path.suffix == ".py" and path.is_file()
    except (OSError, ValueError):
        # e.g. an embedded NUL byte makes ``is_file`` raise ValueError.
        is_py_file = False
    if not is_py_file:
        raise error_cls(
            f"{label} {raw_path!r} is not an existing Python (.py) file"
        )
    module_name = f"_kiss_client_file_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        source = path.read_text(encoding="utf-8")
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
        # BaseException (not just Exception/SystemExit): a file raising
        # e.g. KeyboardInterrupt or SystemExit at import time is
        # converted into *error_cls* like any other bad module — the
        # task runner treats an escaping KeyboardInterrupt as a task
        # CANCELLATION, so letting it propagate unwrapped would report
        # a broken file as "task cancelled" instead of a task error
        # with a diagnostic.
        logger.warning("Failed to import %s %r", label, raw_path, exc_info=True)
        raise error_cls(
            f"{label} {raw_path!r} failed to import: "
            f"{_safe_message(exc)}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    return module.__dict__


def apply_agent_overrides(cmd: dict[str, Any]) -> set[str]:
    """Apply a ``run`` command's agent-script configuration, in place.

    Daemon-side counterpart of :func:`resolve_agent_path`: imports the
    Python file named by the command's ``agentPath`` field and applies
    its ``settings()`` (:func:`~kiss.agents.sorcar.sea_settings.resolve_settings`:
    preset defaults merged, deprecated per-field getters honoured), one
    wire field per key (:data:`SETTING_FIELDS`).  A ``channel`` preset
    appends :data:`CHANNEL_PREAMBLE` to the system prompt; the script's
    ``add_to_system_prompt()`` text (or the deprecated
    ``append_to_system_prompt()``) follows it; its ``add_to_prompt``
    text, ``{task_id}`` in it replaced by the command's
    ``parentTaskId``, is appended to the caller's ``appendToPrompt``.
    ``add_to_tools()`` callables are staged on the daemon-side ``tools``
    field (added to the run's built-in toolset); the deprecated
    ``tools()`` stages the same list with the ``none`` tool profile, so
    the list and ``finish`` are the run's whole tool set.
    ``llm_call_hook()`` / ``tool_call_hook()`` callables are staged on
    ``llmCallHook`` / ``toolCallHook``.  The writes are atomic: they
    happen only after everything has succeeded, so a broken script
    leaves the command untouched.

    Args:
        cmd: The ``run`` command dict; mutated in place.  An absent,
            ``None``, or empty ``agentPath`` field means "no agent
            script" and leaves the command untouched.

    Returns:
        The set of command-field names that were overridden (empty when
        the command carries no agent script), so the caller can tell an
        actual script override apart from a client-sent value.

    Raises:
        AgentFileError: When the ``agentPath`` field is not a string,
            is not the path of an existing ``.py`` file, names a module
            that raises at import time, has malformed settings, a
            non-callable or raising getter, a getter returning the
            wrong type, or both ``tools()`` and ``add_to_tools()``.
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
    # Everything below is STAGED and applied to the command only after
    # every getter has succeeded: a broken getter must leave the command
    # completely untouched, or a direct ``_run_task`` caller (no
    # dispatch-created state) would seed its run state from a partially
    # overridden command.
    staged: dict[str, Any] = {}
    try:
        settings = resolve_settings(namespace)
    except SettingsError as exc:
        logger.warning("settings of agentPath %r rejected: %s", raw_path, exc)
        raise AgentFileError(f"agent script {raw_path!r}: {exc}") from exc
    for key, field in SETTING_FIELDS.items():
        if key in settings:
            staged[field] = settings[key]
    if "add_to_prompt" in settings:
        parent_task_id = cmd.get("parentTaskId")
        try:
            addition = settings["add_to_prompt"].replace(
                "{task_id}", parent_task_id if isinstance(parent_task_id, str) else "",
            )
        except BaseException as exc:  # noqa: BLE001 — an untrusted str subclass may raise
            raise AgentFileError(
                f"agent script {raw_path!r}: add_to_prompt is a broken value: "
                f"{_safe_message(exc)}"
            ) from exc
        staged["appendToPrompt"] = _add_text(cmd.get("appendToPrompt"), addition)
    system_suffix = cmd.get("appendToSystemPrompt")
    if settings["preset"] == "channel":
        system_suffix = _add_text(
            system_suffix, CHANNEL_PREAMBLE.format(name=script_name(raw_path)),
        )
        staged["appendToSystemPrompt"] = system_suffix
    for name in ("append_to_system_prompt", "add_to_system_prompt"):
        if name in namespace:
            addition = _check_text(raw_path, name, _getter_value(namespace, raw_path, name))
            system_suffix = _add_text(system_suffix, addition)
            staged["appendToSystemPrompt"] = system_suffix
    if "add_to_tools" in namespace:
        staged["tools"] = _check_tools(
            raw_path, "add_to_tools", _getter_value(namespace, raw_path, "add_to_tools"),
        )
    elif "tools" in namespace:
        staged["tools"] = _check_tools(
            raw_path, "tools", _getter_value(namespace, raw_path, "tools"),
        )
        staged["toolProfile"] = NO_TOOLS_PROFILE
    for name, field in (("llm_call_hook", "llmCallHook"), ("tool_call_hook", "toolCallHook")):
        if name in namespace:
            staged[field] = _check_hook(raw_path, name, _getter_value(namespace, raw_path, name))
    cmd.update(staged)
    return set(staged)


def _add_text(base: Any, addition: str) -> str:
    """Return *addition* appended to *base* (a wire value; non-strings count as empty)."""
    if not isinstance(base, str) or not base:
        return addition
    return f"{base}\n\n{addition}" if addition else base
