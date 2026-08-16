# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tools-file loading for the synchronous ``kiss.server.sorcar.run`` API.

The caller of :func:`kiss.server.sorcar.run` supplies its extra agent
tools as a *file path* to a Python module rather than as live callables
— the client never serializes Python functions.  The client validates
and resolves the path (:func:`resolve_tools_file`) and sends it on the
``run`` command's ``toolsFile`` field; the daemon imports the file and
calls its required top-level ``get_tools()`` function, which returns
the callables the agent may invoke (:func:`load_tools_file`).  The
tools therefore execute in the daemon process, exactly like native
agent tools.  A broken tools file (malformed field, missing file,
import failure, missing or misbehaving ``get_tools()``) raises
:exc:`ToolsFileError` so the task stops with a diagnostic error
instead of silently running without the requested tools.
"""

from __future__ import annotations

import logging
import sys
import types
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("kiss-vscode")


def _safe_message(exc: BaseException) -> str:
    """Format an untrusted exception without running its raising code.

    ``str(exc)`` runs the exception's ``__str__``, which — for an
    exception minted by an untrusted tools file — may itself raise
    anything.  A diagnostic built here must never leak such a
    secondary raise, so the string conversion is guarded and falls
    back to the (trusted) type name alone.

    Args:
        exc: The exception raised by untrusted tools-file code.

    Returns:
        ``"TypeName: message"`` when the message renders, otherwise
        ``"TypeName"``.
    """
    name = type(exc).__name__
    try:
        return f"{name}: {exc}"
    except BaseException:  # noqa: BLE001 — untrusted __str__ may raise anything
        return name


class ToolsFileError(Exception):
    """A ``run`` command's tools file is broken and the task must stop.

    Raised by :func:`load_tools_file` when the ``toolsFile`` wire field
    is malformed, names a missing or non-``.py`` path, names a file
    that raises at import time, or names a module whose ``get_tools()``
    is missing, raises, or returns anything but callables.  The task
    runner's generic task-error handling turns the raise into a failed
    task result whose text carries this exception's diagnostic message,
    so a broken tools file stops the task loudly instead of silently
    running it without the tools the client asked for.
    """


def execute_python_file(
    raw_path: Any,
    error_cls: type[Exception],
    label: str,
) -> dict[str, Any]:
    """Import a caller-supplied Python file and return its namespace.

    Shared daemon-side loader for the ``run`` command's caller-supplied
    Python files (the ``toolsFile`` tools file and the ``agentPath``
    agent script).  The source is compiled and executed directly (no
    ``__pycache__`` read or write), so every run observes the file's
    CURRENT contents and the caller's directory is never littered with
    bytecode.

    Args:
        raw_path: The wire field naming the file — expected to be an
            absolute path string, but treated as untrusted.
        error_cls: The exception class to raise on any failure (e.g.
            :exc:`ToolsFileError`), so each caller keeps its own
            diagnostic type.
        label: Human-readable name of the file kind (e.g. ``"tools
            file"``), used in diagnostic messages.

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


def resolve_tools_file(tools: str | Path | None) -> str:
    """Validate a client-supplied tools path and resolve it absolutely.

    Client-side counterpart of :func:`load_tools_file`.  The path is
    resolved against the CLIENT's working directory (the daemon may run
    with a different one) and validated eagerly so a bad value fails
    fast, before any daemon connection is made.

    Args:
        tools: Path to a Python file whose ``get_tools()`` function
            supplies the agent tools, or ``None`` for no extra tools.

    Returns:
        The absolute path as a string, or ``""`` when *tools* is
        ``None``.

    Raises:
        ValueError: When *tools* is neither ``None`` nor a path, is not
            a ``.py`` file, or does not exist.
    """
    if tools is None:
        return ""
    if not isinstance(tools, (str, Path)):
        raise ValueError(
            f"tools must be a path to a Python file, got {type(tools).__name__}: {tools!r}"
        )
    path = Path(tools).expanduser().resolve()
    if path.suffix != ".py":
        raise ValueError(f"tools file {str(path)!r} is not a Python (.py) file")
    if not path.is_file():
        raise ValueError(f"tools file {str(path)!r} does not exist")
    return str(path)


def load_tools_file(raw_path: Any) -> list[Callable[..., Any]]:
    """Import a tools file and return the tools its ``get_tools()`` picks.

    Daemon-side counterpart of :func:`resolve_tools_file`: imports the
    Python file named by a ``run`` command's ``toolsFile`` field and
    calls the module's top-level ``get_tools()`` function, which must
    return the callables the agent may invoke.  The file's author —
    not the daemon — decides which of the module's functions become
    tools, so no scanning or suitability filtering happens here.

    The source is compiled and executed directly (no ``__pycache__``
    read or write), so every run observes the file's CURRENT contents
    and the caller's directory is never littered with bytecode.

    A broken tools file stops the task: a malformed field value, a
    missing file, a module that fails to import, a missing or raising
    ``get_tools()``, or a ``get_tools()`` return value that is not a
    list/tuple of callables raises :exc:`ToolsFileError` with a
    diagnostic message instead of silently running the task without
    the requested tools.

    Args:
        raw_path: The ``toolsFile`` field of a ``run`` command —
            expected to be an absolute path string produced by
            :func:`resolve_tools_file`, but treated as untrusted.

    Returns:
        The tool callables returned by the module's ``get_tools()``.

    Raises:
        ToolsFileError: When *raw_path* is not a string, is not the
            path of an existing ``.py`` file, names a module that
            raises at import time, or names a module whose
            ``get_tools()`` is missing, raises, or returns anything
            but a list/tuple of callables.
    """
    # ``None``/empty mean "no extra tools"; isinstance is checked FIRST
    # (before the == comparison) because comparing an untrusted
    # non-string object could run arbitrary code.
    if raw_path is None:
        return []
    if isinstance(raw_path, str) and raw_path == "":
        return []
    namespace = execute_python_file(raw_path, ToolsFileError, "tools file")
    get_tools = namespace.get("get_tools")
    if not callable(get_tools):
        raise ToolsFileError(
            f"tools file {raw_path!r} must define a top-level "
            f"get_tools() function returning the tool callables"
        )
    try:
        returned = get_tools()
    except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
        logger.warning(
            "get_tools() of toolsFile %r raised", raw_path, exc_info=True
        )
        raise ToolsFileError(
            f"get_tools() of tools file {raw_path!r} raised: "
            f"{_safe_message(exc)}"
        ) from exc
    if not isinstance(returned, (list, tuple)):
        raise ToolsFileError(
            f"get_tools() of tools file {raw_path!r} must return a list "
            f"or tuple of callables, got {type(returned).__name__}"
        )
    # Validate inside a BaseException guard: the returned value is
    # untrusted module data, so even iterating it (a list subclass
    # overriding ``__iter__``) may raise anything — such a raise must
    # become a ToolsFileError diagnostic, not a task cancellation.
    # Diagnostics use type names, never ``repr`` (which is user code).
    try:
        tools = list(returned)
        for index, tool in enumerate(tools):
            if not callable(tool):
                raise ToolsFileError(
                    f"get_tools() of tools file {raw_path!r} returned a "
                    f"non-callable entry at index {index} "
                    f"(type {type(tool).__name__})"
                )
    except ToolsFileError:
        raise
    except BaseException as exc:  # noqa: BLE001 — untrusted module data may raise anything
        logger.warning(
            "Validating get_tools() result of toolsFile %r raised",
            raw_path,
            exc_info=True,
        )
        raise ToolsFileError(
            f"get_tools() of tools file {raw_path!r} returned a broken "
            f"value: {_safe_message(exc)}"
        ) from exc
    return tools
