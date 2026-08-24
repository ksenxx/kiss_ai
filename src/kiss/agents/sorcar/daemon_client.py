# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Synchronous client for running tasks on the ``kiss-web`` daemon.

This module is the client half of the daemon's Python API: it speaks
the newline-delimited JSON protocol over the daemon's Unix-domain
socket and blocks until the submitted task finishes.  It lives in the
sorcar layer — not in ``kiss.server`` — because sorcar-layer code
(the ``run_agent`` dispatch tool in
:mod:`kiss.agents.sorcar.agent_dispatch` and the cron scheduler in
:mod:`kiss.agents.sorcar.cron_agent`) submits tasks back to the
daemon, and the layering invariant forbids sorcar code from importing
``kiss.server`` (see ``kiss.tests.agents.sorcar.test_layering_invariants``).
The public API surface is unchanged: :mod:`kiss.server.sorcar`
re-exports :func:`run` and :class:`TaskResult`, so
``kiss.server.sorcar.run(...)`` keeps working for external callers.

Depends only on the standard library and the sorcar/core layers.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.persistence import _default_kiss_dir

_MAX_LINE_BYTES = 64 * 1024 * 1024
"""Read buffer limit for a single daemon event line.

The daemon emits large single-line JSON events (e.g.
``system_prompt`` carrying the full SYSTEM.md), so this MUST match the
daemon-side transport frame limit (``web_server._MAX_LINE_BYTES``, 64
MiB).  A smaller client cap would split an oversized newline-delimited
frame; each fragment is then discarded as invalid JSON, and when the
oversized frame is the terminal ``result`` event the client would
return an empty unsuccessful :class:`TaskResult` for a task that
actually succeeded.
"""


@dataclass(frozen=True)
class TaskResult:
    """Final outcome of one synchronous daemon task run.

    Attributes:
        text: Human-readable result summary produced by the agent.
        success: Whether the agent reported the task as successful.
        cost: Budget consumed by the task in USD.
        tokens: Total LLM tokens consumed by the task.
        steps: Total agent steps taken by the task.
        chat_id: The daemon chat session id the task ran on.  Pass it
            back as the ``chat_id`` argument of :func:`run` to
            continue the chat, or use it to inspect the chat later;
            ``""`` when the run ended before the daemon assigned one.
        task_id: The daemon's persisted ``task_history`` row id of the
            run; ``""`` when the run ended before a row was allocated
            (e.g. the daemon had no model configured).
    """

    text: str
    success: bool
    cost: float
    tokens: int
    steps: int
    chat_id: str = ""
    task_id: str = ""


def _resolve_sock_path(sock_path: str | Path | None) -> Path:
    """Return the daemon UDS path to connect to.

    Precedence: explicit *sock_path* argument, then the
    ``KISS_SORCAR_SOCK`` environment variable, then the daemon's
    default ``$KISS_HOME/sorcar.sock``.

    Args:
        sock_path: Optional explicit socket path override.

    Returns:
        The resolved Unix-domain socket path.
    """
    if sock_path:
        return Path(sock_path)
    env = os.environ.get("KISS_SORCAR_SOCK")
    return Path(env) if env else _default_kiss_dir() / "sorcar.sock"


def _parse_cost(value: Any) -> float:
    """Parse a daemon cost field (``"$0.1234"``, ``"N/A"``, or a number).

    Args:
        value: The ``cost`` field of a daemon ``result`` event.

    Returns:
        The cost in USD; ``0.0`` when the field is absent or unparseable.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().lstrip("$"))
        except ValueError:
            return 0.0
    return 0.0


def _to_task_result(
    event: dict[str, Any] | None,
    chat_id: str = "",
    task_id: str = "",
) -> TaskResult:
    """Convert the final daemon ``result`` event into a :class:`TaskResult`.

    Args:
        event: The last ``result`` event received for the task's tab,
            or ``None`` when the task ended without one.
        chat_id: The daemon chat session id observed on the run's
            ``clear`` event (``""`` when none was seen).
        task_id: The persisted ``task_history`` row id observed on the
            run's event stream (``""`` when none was seen).

    Returns:
        The parsed :class:`TaskResult`.  The daemon enriches ``result``
        events with ``success`` / ``summary`` fields parsed from the
        agent's YAML result; ``summary`` is preferred over the raw
        ``text`` when present.
    """
    if event is None:
        return TaskResult(
            text="", success=False, cost=0.0, tokens=0, steps=0,
            chat_id=chat_id, task_id=task_id,
        )
    text = str(event.get("summary") or event.get("text") or "")
    return TaskResult(
        text=text,
        success=bool(event.get("success", False)),
        cost=_parse_cost(event.get("cost")),
        tokens=int(event.get("total_tokens", 0) or 0),
        steps=int(event.get("step_count", 0) or 0),
        chat_id=chat_id,
        task_id=task_id,
    )


def _resolve_py_file(
    value: str | Path, type_error: str, what: str, allow_path: bool = True,
) -> str:
    """Resolve and validate a client-supplied Python-file path.

    Shared tail of :func:`resolve_tools_file` and
    :func:`resolve_agent_path`: the path is resolved against the
    CLIENT's working directory (the daemon may run with a different
    one) and validated eagerly so a bad value fails fast, before any
    daemon connection is made.

    Args:
        value: The path to resolve; a wrong-typed value raises.
        type_error: Error message for a wrong-typed *value*;
            ``{type}`` and ``{value}`` placeholders are filled in.
        what: Noun used in the suffix/existence error messages, e.g.
            ``"tools file"`` or ``"agent script"``.
        allow_path: Whether a ``pathlib.Path`` *value* is accepted in
            addition to ``str``.

    Returns:
        The absolute path as a string.

    Raises:
        ValueError: When *value* has a wrong type, is not a ``.py``
            file, or does not exist.
    """
    if not isinstance(value, (str, Path) if allow_path else str):
        raise ValueError(
            type_error.format(type=type(value).__name__, value=repr(value))
        )
    path = Path(value).expanduser().resolve()
    if path.suffix != ".py":
        raise ValueError(f"{what} {str(path)!r} is not a Python (.py) file")
    if not path.is_file():
        raise ValueError(f"{what} {str(path)!r} does not exist")
    return str(path)


def resolve_tools_file(tools: str | Path | None) -> str:
    """Validate a client-supplied tools path and resolve it absolutely.

    Client-side counterpart of the daemon's
    ``kiss.server.tools_file.load_tools_file`` (see
    :func:`_resolve_py_file` for the resolution rules).

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
    return _resolve_py_file(
        tools,
        "tools must be a path to a Python file, got {type}: {value}",
        "tools file",
    )


def resolve_agent_path(agent_path: str | None) -> str:
    """Validate a client-supplied agent-script path and resolve it.

    Client-side counterpart of the daemon's
    ``kiss.server.agent_file.apply_agent_overrides`` (see
    :func:`_resolve_py_file` for the resolution rules).

    Args:
        agent_path: Path string of a Python file whose ``get_X()``
            functions compute the run's parameters, or ``None``/empty
            for no agent script.

    Returns:
        The absolute path as a string, or ``""`` when *agent_path* is
        ``None`` or empty.

    Raises:
        ValueError: When *agent_path* is neither ``None`` nor a string,
            is not a ``.py`` file, or does not exist.
    """
    if agent_path is None or agent_path == "":
        return ""
    return _resolve_py_file(
        agent_path,
        "agent_path must be a string path to a Python file, got {type}: {value}",
        "agent script",
        allow_path=False,
    )


def run(
    prompt: str,
    *,
    work_dir: str = "",
    scope_work_dir: str = "",
    model: str = "",
    chat_id: str = "",
    system_prompt: str = "",
    tools: str | Path | None = None,
    extension_agent_path: str = "",
    use_worktree: bool = True,
    auto_commit: bool = True,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = True,
    append_basic_tools: bool = True,
    append_to_system_prompt: str = "",
    append_to_prompt: str = "",
    timeout: float = 3600.0,
    sock_path: str | Path | None = None,
) -> TaskResult:
    """Run *prompt* as a task on the local Sorcar daemon and block until done.

    Connects to the ``kiss-web`` daemon's Unix-domain socket, sends
    the same ``run`` command a chat webview would, streams the task's
    events, and returns once the daemon reports the task finished.

    Args:
        prompt: The task instruction to run.
        work_dir: Working directory for the task; the daemon's current
            default is used when empty.
        scope_work_dir: The workspace-scope directory of the task's
            tab in the daemon's shared tab registry — the directory a
            client's tab bar matches against to decide whether to show
            the tab — kept separate from *work_dir* (the execution
            directory) so a ``run_agent`` sub-task that runs in a
            channel/cron scratch directory can still appear in the
            CALLING workspace's tab bar.  Empty (the default) leaves
            the tab's scope falling back to *work_dir*, unchanged from
            ordinary runs.  Like *timeout* and *sock_path* it is a
            client/UI-transport parameter with no agent-script getter.
        model: Model name; the daemon's selected default when empty.
        chat_id: Optional existing chat session id to continue.  Pass
            the ``chat_id`` of a previous :class:`TaskResult` to run
            this task in the same chat — the agent then sees the prior
            tasks and results of that chat as context.  A new chat is
            started when empty.
        system_prompt: Optional custom system prompt for the run.
            When non-empty it is used as the system prompt of the
            agent AND of every sub-agent it spawns (``run_parallel``),
            replacing the default system prompt shipped in
            ``src/kiss/SYSTEM.md``.  The daemon still appends its
            per-run operational instructions (work directory, process
            id, ``~/.kiss/SORCAR.md``) so the agent's tool contract
            keeps working.  Empty (default) runs with the default
            system prompt as usual.
        tools: Optional path to a Python file supplying extra tools
            for the agent.  The file must define a top-level
            ``get_tools()`` function returning the functions in the
            file the agent may call; the daemon imports the file,
            calls ``get_tools()``, and registers the returned
            callables as agent tools.  Each function's name, docstring
            (Google-style ``Args:`` section for parameter
            descriptions), and annotated keyword-bindable parameters
            define the tool schema the agent sees, exactly like a
            native tool.  The functions are never serialized by the
            client — they run **in the daemon process**.  The path is
            resolved against this process's working directory.  A
            broken tools file (deleted before the daemon reads it,
            raising at import time, or missing/misbehaving
            ``get_tools()``) stops the task: the daemon fails the run
            and the returned :class:`TaskResult` carries the
            diagnostic error in its ``text`` with ``success=False``.
        extension_agent_path: Optional path — a string — to a Python
            *agent script* that computes this run's parameters **on the
            daemon**.  When non-empty, the daemon imports the file and,
            for each parameter ``X`` of this function except
            ``extension_agent_path`` itself (and the getter-less
            parameters noted below), calls the script's top-level
            ``get_X()`` function — when the script defines one — and
            uses its return value for ``X``, replacing the value passed
            to this call.  A parameter whose ``get_X()`` the script
            does not define keeps the value passed here, which is the
            parameter's default when the caller did not pass one.

            Script format: a plain Python file defining any subset of
            these zero-argument top-level functions, each returning a
            value of the corresponding parameter's documented type::

                def get_prompt() -> str: ...          # non-empty
                def get_work_dir() -> str: ...
                def get_model() -> str: ...
                def get_chat_id() -> str: ...
                def get_system_prompt() -> str: ...
                def get_tools() -> str | Path | list | None: ...  # tools-file path or tool list
                def get_use_worktree() -> bool: ...
                def get_auto_commit() -> bool: ...
                def get_max_budget() -> float | None: ...   # finite
                def get_model_config() -> dict | None: ...
                def get_web_tools() -> bool | None: ...
                def get_is_parallel() -> bool: ...
                def get_append_basic_tools() -> bool: ...
                def get_append_to_system_prompt() -> str: ...
                def get_append_to_prompt() -> str: ...

            The ``get_X()`` functions are never serialized by the
            client — they run **in the daemon process**, exactly like a
            tools file's ``get_tools()``.  ``get_tools()`` here returns
            the *path* of a tools file (pass an absolute path — the
            daemon does not resolve it against this process's working
            directory), which the daemon then imports and whose
            ``get_tools()`` it calls as if the path had been passed as
            *tools*; a ``get_tools()`` that instead returns a *list*
            of tool callables (the tools-file contract, as in the
            channel agent modules) makes the script its own tools
            file.  ``timeout``, *sock_path*, and *scope_work_dir*
            have no getters by design: the first two are
            client-transport parameters — the script only runs on the
            daemon that *sock_path* selects, and *timeout* bounds this
            client's local wait — and *scope_work_dir* is the CALLING
            client's tab-bar scope, which the dispatched script must
            not be able to repoint at another workspace.  The
            *extension_agent_path* itself is resolved against this process's
            working directory and validated eagerly, like *tools*.  A
            broken agent script (deleted before the daemon reads it,
            raising at import time, a non-callable ``get_X``, a raising
            ``get_X()``, or a wrong-typed return value) stops the task:
            the daemon fails the run and the returned
            :class:`TaskResult` carries the diagnostic error in its
            ``text`` with ``success=False``.
        use_worktree: Run the task in an isolated git worktree.
            Defaults to True.
        auto_commit: Auto-commit the task's changes on success.
            Defaults to True.
        max_budget: Per-task budget override in USD; ``None`` uses the
            daemon's configured default.
        model_config: Per-task model configuration override (custom
            endpoint / headers); ``None`` uses the daemon's configured
            model endpoint.  Must be JSON-serializable.
        web_tools: Per-task browser-tool enablement override; ``None``
            uses the daemon's configured default.
        is_parallel: Whether the agent may spawn parallel sub-agents.
            Defaults to True.
        append_basic_tools: Whether the agent gets the built-in basic
            toolset (``Bash``, ``Read``, ``Edit``, ``Write``, browser
            tools, ``run_agent``, ``ask_user_question``, ``talk``,
            ``set_model``, ``summary``, ``run_parallel``, ...).
            Defaults to True.  When False the agent's ONLY tools are
            ``finish`` and the caller-supplied tools — the ones
            returned by the *tools* file's ``get_tools()`` — so the
            *web_tools* and *is_parallel* toggles have no tools left
            to act on.  The default system prompt (``SYSTEM.md``)
            assumes the basic toolset (e.g. it mandates a first
            ``Read("./SORCAR.md")`` call), so a restricted run should
            usually pass a *system_prompt* written for the tools it
            actually has.
        append_to_system_prompt: Extra text appended to the run's
            system prompt when the agent is executed — after the
            default ``SYSTEM.md`` prompt (or the *system_prompt*
            replacement) and before the daemon's per-run operational
            instructions.  ``run_parallel`` sub-agents inherit the
            suffix on their own system prompts, like a *system_prompt*
            replacement, so the extra instructions constrain the whole
            task tree.  Empty (default) appends nothing.
        append_to_prompt: Extra text appended to the executed task
            prompt.  A multi-``<task>`` *prompt* runs the agent once
            per subtask, and the text is appended to EACH subtask's
            prompt.  The appended text is part of the prompt the agent
            actually runs with, so it is also what the chat history
            records and what follow-up tasks of the same chat see as
            context.  Empty (default) appends nothing.
        timeout: Maximum seconds to wait for the task to finish.
        sock_path: Daemon UDS path override (defaults to
            ``$KISS_SORCAR_SOCK`` or ``$KISS_HOME/sorcar.sock``).

    Returns:
        A :class:`TaskResult` with the result text, success flag, cost
        (USD), total tokens, step count, chat id, and task id of the
        task.  ``chat_id`` is the daemon chat session id and
        ``task_id`` the persisted ``task_history`` row id — both
        usable later to look up or resume the run in the daemon's
        history.

    Raises:
        ValueError: When *prompt* is empty or blank, when *tools*
            is not the path of an existing Python (``.py``) file (see
            :func:`resolve_tools_file`), or when *extension_agent_path* is
            neither empty nor the path string of an existing Python
            (``.py``) file (see :func:`resolve_agent_path`).
        ConnectionError: When no daemon is listening on the socket, or
            the daemon drops the connection before the task finishes.
        TimeoutError: When the task does not finish within *timeout*
            seconds.  The client then sends the daemon an explicit
            ``closeTab`` for the task's tab and disconnects; the task
            keeps running and its state is disposed when it ends.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    tools_file = resolve_tools_file(tools)
    agent_file = resolve_agent_path(extension_agent_path)
    path = _resolve_sock_path(sock_path)
    tab_id = f"api-{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    reader: Any = None
    try:
        sock.settimeout(min(timeout, 10.0))
        try:
            sock.connect(str(path))
        except OSError as exc:
            raise ConnectionError(
                f"Cannot connect to the sorcar daemon at {path}: {exc} "
                f"— start it with `kiss-web`."
            ) from exc
        cmd = {
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": uuid.uuid4().hex,
            "chatId": chat_id,
            "workDir": work_dir,
            "tabScopeWorkDir": scope_work_dir,
            "model": model,
            "systemPrompt": system_prompt,
            "toolsFile": tools_file,
            "agentPath": agent_file,
            "useWorktree": use_worktree,
            "autoCommit": auto_commit,
            "maxBudget": max_budget,
            "modelConfig": model_config,
            "webTools": web_tools,
            "useParallel": is_parallel,
            "appendBasicTools": append_basic_tools,
            "appendToSystemPrompt": append_to_system_prompt,
            "appendToPrompt": append_to_prompt,
        }
        sock.sendall(json.dumps(cmd).encode("utf-8") + b"\n")
        reader = sock.makefile("rb", buffering=_MAX_LINE_BYTES)
        result_event: dict[str, Any] | None = None
        task_id = ""
        started = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Task did not finish within {timeout} seconds"
                )
            sock.settimeout(remaining)
            try:
                line = reader.readline(_MAX_LINE_BYTES)
            except TimeoutError:
                raise TimeoutError(
                    f"Task did not finish within {timeout} seconds"
                ) from None
            if not line:
                raise ConnectionError(
                    "The sorcar daemon closed the connection before the "
                    "task finished"
                )
            if len(line) >= _MAX_LINE_BYTES and not line.endswith(b"\n"):
                # ``readline(size)`` returned a full-size chunk with no
                # terminating newline: the daemon sent a frame larger
                # than the client cap.  Silently skipping the fragments
                # would discard a possibly terminal ``result`` event
                # and misreport the task as failed — fail loudly
                # instead.
                raise ConnectionError(
                    "The sorcar daemon sent an event frame larger than "
                    f"the {_MAX_LINE_BYTES}-byte client limit"
                )
            try:
                event = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("tabId") != tab_id:
                continue
            etype = event.get("type")
            if etype == "clear":
                chat_id = str(event.get("chat_id", "") or "") or chat_id
            elif etype != "status" and event.get("taskId"):
                task_id = str(event["taskId"])
            if etype == "result":
                result_event = event
            elif etype == "status":
                if event.get("running"):
                    started = True
                elif started:
                    return _to_task_result(result_event, chat_id, task_id)
    finally:
        # The synthetic tab is this client's alone, and a disconnect no
        # longer tears tabs down (tabs are global state shared by every
        # client), so explicitly ask the daemon to close it on every
        # exit path.  For a still-running task (timeout) this merely
        # flips ``frontend_closed`` and the state is disposed when the
        # task ends; for a finished task it is disposed immediately.
        # Best-effort: the daemon may be gone or the connect may have
        # failed.
        try:
            sock.settimeout(5.0)
            sock.sendall(
                json.dumps({"type": "closeTab", "tabId": tab_id})
                .encode("utf-8") + b"\n",
            )
        except OSError:
            pass
        # ``sock.makefile()`` holds an independent reference to the
        # socket descriptor, so closing only the socket object leaves
        # the buffered reader (and its multi-MiB buffer) alive whenever
        # a caller retains a raised exception whose traceback pins this
        # frame.  Close the reader first so the peer promptly sees EOF.
        try:
            if reader is not None:
                reader.close()
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
