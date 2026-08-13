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
hands every top-level public function that is suitable as a tool
directly to the agent (:func:`load_tools_file`).  The tools therefore
execute in the daemon process, exactly like native agent tools.  A
broken tools file (malformed field, missing file, import failure)
raises :exc:`ToolsFileError` so the task stops with a diagnostic error
instead of silently running without the requested tools.
"""

from __future__ import annotations

import inspect
import logging
import sys
import types
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("kiss-vscode")

_SUPPORTED_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


class ToolsFileError(Exception):
    """A ``run`` command's tools file is broken and the task must stop.

    Raised by :func:`load_tools_file` when the ``toolsFile`` wire field
    is malformed, names a missing or non-``.py`` path, or names a file
    that raises at import time.  The task runner's generic task-error
    handling turns the raise into a failed task result whose text
    carries this exception's diagnostic message, so a broken tools file
    stops the task loudly instead of silently running it without the
    tools the client asked for.
    """


def resolve_tools_file(tools: str | Path | None) -> str:
    """Validate a client-supplied tools path and resolve it absolutely.

    Client-side counterpart of :func:`load_tools_file`.  The path is
    resolved against the CLIENT's working directory (the daemon may run
    with a different one) and validated eagerly so a bad value fails
    fast, before any daemon connection is made.

    Args:
        tools: Path to a Python file whose top-level public functions
            should become agent tools, or ``None`` for no extra tools.

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
    """Import a tools file and return its top-level public tool functions.

    Daemon-side counterpart of :func:`resolve_tools_file`: imports the
    Python file named by a ``run`` command's ``toolsFile`` field and
    collects every function that is

    * genuinely defined at the module's top level via ``def`` (merely
      *imported* functions, lambdas, aliases of other functions, and
      re-exported nested functions are excluded — the bound name must
      equal the function's own ``__name__``),
    * public (name does not start with ``_``), and
    * suitable as an agent tool (see :func:`_is_suitable_tool`).

    The source is compiled and executed directly (no ``__pycache__``
    read or write), so every run observes the file's CURRENT contents
    and the caller's directory is never littered with bytecode.

    A broken tools file stops the task: a malformed field value, a
    missing file, or a module that fails to import raises
    :exc:`ToolsFileError` with a diagnostic message instead of
    silently running the task without the requested tools.

    Args:
        raw_path: The ``toolsFile`` field of a ``run`` command —
            expected to be an absolute path string produced by
            :func:`resolve_tools_file`, but treated as untrusted.

    Returns:
        The tool callables, in module definition order.

    Raises:
        ToolsFileError: When *raw_path* is not a string, is not the
            path of an existing ``.py`` file, or names a module that
            raises at import time.
    """
    if raw_path is None or raw_path == "":
        return []
    if not isinstance(raw_path, str):
        raise ToolsFileError(
            f"tools file field must be a path string, got "
            f"{type(raw_path).__name__}: {raw_path!r}"
        )
    path = Path(raw_path)
    if path.suffix != ".py" or not path.is_file():
        raise ToolsFileError(
            f"tools file {raw_path!r} is not an existing Python (.py) file"
        )
    module_name = f"_kiss_tools_file_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        source = path.read_text(encoding="utf-8")
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
        # BaseException (not just Exception/SystemExit): a tools file
        # raising e.g. KeyboardInterrupt or SystemExit at import time
        # is converted into ToolsFileError like any other bad module —
        # the task runner treats an escaping KeyboardInterrupt as a
        # task CANCELLATION, so letting it propagate unwrapped would
        # report a broken tools file as "task cancelled" instead of a
        # task error with a diagnostic.
        logger.warning("Failed to import toolsFile %r", raw_path, exc_info=True)
        raise ToolsFileError(
            f"tools file {raw_path!r} failed to import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    tools: list[Callable[..., Any]] = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != module_name:
            continue
        if name != obj.__name__:
            logger.warning(
                "Skipping tools-file binding %r: not a top-level "
                "function definition (function __name__ is %r)",
                name,
                obj.__name__,
            )
            continue
        if _is_suitable_tool(obj):
            tools.append(obj)
    return tools


def _is_suitable_tool(func: Callable[..., Any]) -> bool:
    """Whether a top-level public function can be registered as a tool.

    The agent invokes tools synchronously by keyword
    (``func(**function_args)``), so a suitable tool must be a plain
    synchronous function whose every parameter is keyword-bindable.
    Coroutine / (async) generator functions and signatures with
    ``*args``, ``**kwargs``, or positional-only parameters are skipped
    with a warning.

    Args:
        func: A function defined at the tools module's top level.

    Returns:
        ``True`` when *func* should be handed to the agent as a tool.
    """
    if (
        inspect.iscoroutinefunction(func)
        or inspect.isgeneratorfunction(func)
        or inspect.isasyncgenfunction(func)
    ):
        logger.warning(
            "Skipping tools-file function %r: coroutine/generator "
            "functions are not supported as tools",
            func.__name__,
        )
        return False
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        logger.warning(
            "Skipping tools-file function %r: signature introspection "
            "failed",
            func.__name__,
            exc_info=True,
        )
        return False
    for param in signature.parameters.values():
        if param.kind not in _SUPPORTED_KINDS:
            logger.warning(
                "Skipping tools-file function %r: parameter %r has "
                "unsupported kind %r; only plain (keyword-bindable) "
                "parameters are supported",
                func.__name__,
                param.name,
                param.kind.description,
            )
            return False
    return True
