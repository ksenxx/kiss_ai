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

_STOP_CONFIRM_GRACE_SECONDS = 20.0
"""Bounded wait for a stopped-on-timeout task's terminal status.

With ``stop_on_timeout`` a timeout sends the daemon a ``stop`` and
then KEEPS READING until the task's terminal ``status
running=false`` confirms it is dead, before the ``TimeoutError`` is
raised.  Without the confirmation the caller would resume — and, in
the ``run_agent`` channel dispatch, release the process-global
workspace reservation — while the child might still be starting up on
the daemon and could bind a LATER dispatch's workspace when its
channel tools load.  The daemon's stop path force-interrupts a
non-cooperating task after ~1 s, so confirmation normally arrives
quickly; this grace bounds the wait against a wedged daemon, after
which :class:`StopUnconfirmedTimeoutError` is raised (the stop stays
best-effort at that point).
"""


class StopUnconfirmedTimeoutError(TimeoutError):
    """Timeout whose ``stop_on_timeout`` stop was sent but never confirmed.

    Raised by :func:`run` instead of the plain :class:`TimeoutError`
    when the :data:`_STOP_CONFIRM_GRACE_SECONDS` wait for the stopped
    task's terminal ``status running=false`` expires without an
    answer: the ``stop`` was sent, but the daemon never confirmed the
    task is dead, so the stop stays best-effort and the task may still
    be running (and spending) on the daemon.  Raised even when a
    SUCCESSFUL ``result`` event was received — the result is emitted
    before the daemon's persistence/auto-commit/cleanup stages, so
    without the terminal status the task may still be touching the
    workspace.  Callers that report the timeout onward — the
    ``run_agent`` dispatch — must not claim the task was stopped.
    """

_NO_DEADLINE_WAKE_SECONDS = 10.0
"""Socket read wake-up interval for a ``timeout=None`` wait.

A thread's injected async exception — the ``KeyboardInterrupt`` the
daemon injects when the CALLING task is stopped while blocked in
:func:`run`'s event wait — is delivered between bytecode instructions
only, never inside a blocking C-level ``recv``.  An unbounded blocking
read on a silent daemon would therefore starve the stop cascade in
:func:`run`'s ``finally`` block forever.  With no deadline the socket
read instead times out and retries at this interval, giving Python a
chance to deliver the pending exception.
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


def _frame_limit_error() -> ConnectionError:
    """Return the error for a daemon frame exceeding the client cap.

    Shared by :func:`run`'s two detection sites — the no-newline
    accumulation check and the extracted-line length check — so the
    two cannot drift apart.  Reads :data:`_MAX_LINE_BYTES` at call
    time (tests shrink it).
    """
    return ConnectionError(
        "The sorcar daemon sent an event frame larger "
        f"than the {_MAX_LINE_BYTES}-byte client limit"
    )


def _send_stop(sock: socket.socket, tab_id: str, run_token: str) -> None:
    """Send the daemon a run-token-guarded ``stop`` for *tab_id*.

    Shared by :func:`run`'s stop-on-timeout path and the abort-cascade
    in its ``finally`` block: both stops MUST carry the run token (so
    the daemon's ``_stop_task`` guard rejects the stop when the tab
    was reused by a newer run), and a drifted duplicate would desync
    that guarantee.  The 5-second send bound keeps a wedged daemon
    from blocking the caller.

    Args:
        sock: The connected daemon socket.
        tab_id: The run's synthetic tab id.
        run_token: The client-minted per-submission run token.

    Raises:
        OSError: When the stop could not be written to the socket
            (including a send timeout).
    """
    sock.settimeout(5.0)
    sock.sendall(
        json.dumps({
            "type": "stop",
            "tabId": tab_id,
            "taskId": run_token,
        }).encode("utf-8") + b"\n",
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
    timeout: float | None = 3600.0,
    stop_on_timeout: bool = False,
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
            parameter's default when the caller did not pass one.  One
            getter is named differently from its parameter:
            ``get_if_append_basic_tools()`` overrides
            *append_basic_tools*.

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
                def get_if_append_basic_tools() -> bool: ...
                def get_append_to_system_prompt() -> str: ...
                def get_append_to_prompt() -> str: ...

            The script may also define two hook getters with no
            corresponding parameter on this function (a callable
            cannot travel the wire, so the hooks exist ONLY as
            agent-script getters)::

                def get_llm_call_hook() -> Callable | None: ...
                def get_tool_call_hook() -> Callable | None: ...

            Each returns a callable — ``llm_call_hook`` and
            ``tool_call_hook`` respectively — (or ``None`` for "no
            hook") that the daemon passes to the underlying
            :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
            task-executor sub-session of the task's agent (internal
            helper sessions, e.g. the failed-session trajectory
            summarizer, are not hooked).  Per that method's contract,
            ``llm_call_hook(new_messages)`` is called before every LLM
            call and its return value replaces the new messages about
            to be sent, and ``tool_call_hook(name, args)`` is called
            before every tool call — the tool executes only when the
            hook returns ``"OK"``; any other returned string is given
            to the model as the tool's result instead.  Like every
            getter, they execute **in the daemon process**; the hooks
            apply to the task's own agent, not to sub-agents it spawns
            via ``run_parallel``.

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
            file.  ``timeout``, *stop_on_timeout*, *sock_path*,
            *scope_work_dir*, *web_tools*, and *is_parallel* have no
            getters by design: the first three are client-transport
            parameters — the script only runs on the daemon that
            *sock_path* selects, *timeout* bounds this client's local
            wait, and *stop_on_timeout* picks this client's timeout
            behavior — *scope_work_dir* is the CALLING
            client's tab-bar scope, which the dispatched script must
            not be able to repoint at another workspace, and
            *web_tools* / *is_parallel* always keep the values passed
            to this call (their defaults when the caller passed
            none).  The
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
        timeout: Maximum seconds to wait for the task to finish;
            ``None`` waits indefinitely.
        stop_on_timeout: Whether a *timeout* expiry also STOPS the
            task.  ``False`` (the default) keeps the documented
            timeout contract — the caller stops waiting, the task
            keeps running.  With ``True`` the client sends the daemon
            a ``stop`` for the task and keeps reading until the task's
            terminal status confirms it is dead (waiting up to
            :data:`_STOP_CONFIRM_GRACE_SECONDS`; on a wedged daemon
            :class:`StopUnconfirmedTimeoutError` is raised instead,
            the stop then staying best-effort) before raising the
            ``TimeoutError``.
            ``True`` is for callers that must not let the task outlive
            the wait, e.g. the ``run_agent`` channel dispatch, whose
            process-global workspace reservation is released as soon
            as the call returns: a surviving sub-task could bind
            another account's credentials when its channel tools load.
            When the task finished ON ITS OWN — a SUCCESSFUL terminal
            ``result`` AND the terminal status raced the stop onto the
            wire — the completed :class:`TaskResult` is returned
            instead of a ``TimeoutError`` that would discard the
            finished work (only natural completions carry ``success:
            true``; every daemon stop/cancel/failure path broadcasts
            ``success: false``).  A successful result WITHOUT the
            terminal status never settles the wait: the daemon's
            persistence / auto-commit / worktree cleanup still run
            after the result is emitted, so only the terminal status
            proves the task is dead, and the grace expiring with just
            the result in hand raises
            :class:`StopUnconfirmedTimeoutError` all the same.
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
        ConnectionError: When no daemon is listening on the socket,
            the daemon drops the connection before the task finishes,
            or a *stop_on_timeout* stop cannot be sent on the broken
            connection (a plain ``TimeoutError`` would falsely imply
            the task was stopped).
        TimeoutError: When the task does not finish within *timeout*
            seconds (never raised when *timeout* is ``None``).  The
            client then sends the daemon an explicit
            ``closeTab`` for the task's tab and disconnects; the task
            keeps running and its state is disposed when it ends —
            unless *stop_on_timeout* is true, in which case the task
            is first stopped and its terminal status awaited (see the
            parameter's documentation).
        StopUnconfirmedTimeoutError: When *stop_on_timeout* is true
            and the stop was sent but the daemon never confirmed the
            task's death within :data:`_STOP_CONFIRM_GRACE_SECONDS` —
            the task may still be running.  A subclass of
            ``TimeoutError``, so a plain ``except TimeoutError`` still
            catches it.

    Every other abort of the wait — most importantly the
    ``KeyboardInterrupt`` injected when the CALLING task is stopped
    while blocked in a ``run_agent`` dispatch — additionally sends the
    daemon a ``stop`` for the dispatched task's tab before
    disconnecting.  Without the cascade, the orphaned sub-task kept
    running invisibly and, when it was a non-worktree task, kept its
    repository flagged busy: a manual Git Commit pressed after the
    parent showed "Task stopped by user" was refused with "A task is
    still running in this folder" with no visible task running.  A
    timeout intentionally does NOT stop the task (see above) unless
    *stop_on_timeout* is true: by default the caller chose to stop
    waiting, not to cancel the work.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    tools_file = resolve_tools_file(tools)
    agent_file = resolve_agent_path(extension_agent_path)
    path = _resolve_sock_path(sock_path)
    tab_id = f"api-{uuid.uuid4().hex}"
    # Client-minted per-submission run token.  Echoed on the run's
    # ``status`` events, and — critically — sent with the
    # abort-cascade ``stop`` below so the daemon only stops THIS run:
    # a late stop must never kill a newer run that reused the tab.
    run_token = uuid.uuid4().hex
    deadline = None if timeout is None else time.monotonic() + timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    aborted: BaseException | None = None
    try:
        sock.settimeout(10.0 if timeout is None else min(timeout, 10.0))
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
            "taskId": run_token,
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
        # Newline-framed events are assembled by hand from ``recv``
        # chunks instead of ``sock.makefile().readline()``: a buffered
        # reader DISCARDS the partial line it has accumulated when the
        # underlying read raises, so the periodic no-deadline wake-up
        # below (and a finite deadline expiring mid-line) would corrupt
        # the event stream.  ``recv_buf`` survives the raise unharmed.
        recv_buf = bytearray()
        scanned = 0  # recv_buf[:scanned] is known newline-free
        result_event: dict[str, Any] | None = None
        task_id = ""
        started = False
        stopping = False  # stop-on-timeout sent; awaiting confirmation
        timeout_msg = f"Task did not finish within {timeout} seconds"
        while True:
            newline_at = recv_buf.find(b"\n", scanned)
            if newline_at < 0:
                # Only bytes appended after this point need scanning
                # next round — a large frame arriving in many chunks
                # must not be rescanned from the start each time.
                scanned = len(recv_buf)
                if len(recv_buf) >= _MAX_LINE_BYTES:
                    # The daemon sent a frame larger than the client
                    # cap.  Silently skipping the fragments would
                    # discard a possibly terminal ``result`` event and
                    # misreport the task as failed — fail loudly
                    # instead.
                    raise _frame_limit_error()
                if deadline is None:
                    # No deadline: wake periodically so an injected
                    # abort (see _NO_DEADLINE_WAKE_SECONDS) can be
                    # delivered; the timeout is retried, not an error.
                    sock.settimeout(_NO_DEADLINE_WAKE_SECONDS)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if stop_on_timeout and not stopping:
                            # Stop the timed-out task, then KEEP
                            # READING (bounded by the confirmation
                            # grace) until its terminal status proves
                            # it is dead — the caller must not resume
                            # while the child could still act (see
                            # _STOP_CONFIRM_GRACE_SECONDS).
                            stopping = True
                            deadline = (
                                time.monotonic()
                                + _STOP_CONFIRM_GRACE_SECONDS
                            )
                            try:
                                _send_stop(sock, tab_id, run_token)
                            except OSError as send_exc:
                                # The stop could not even be sent, so
                                # the task was neither stopped nor
                                # confirmed dead — raising the plain
                                # TimeoutError here would let a caller
                                # (``_dispatch``) claim "was stopped".
                                # Surface the broken daemon connection
                                # instead, like every other mid-run
                                # socket failure.
                                raise ConnectionError(
                                    "The sorcar daemon connection "
                                    "failed while stopping the "
                                    f"timed-out task: {send_exc}"
                                ) from send_exc
                            continue
                        if stopping:
                            # The confirmation grace expired without a
                            # terminal status: the stop was sent but
                            # never answered, so the task may still be
                            # running — the caller must not be told it
                            # was stopped.  Even a stored SUCCESSFUL
                            # result is no proof the task is dead: the
                            # agent emits it BEFORE the daemon's
                            # persistence / auto-commit / worktree
                            # cleanup stages run, and a stop can still
                            # take effect during those stages, so
                            # returning the result here would let the
                            # caller (``run_agent``) release its
                            # workspace reservation while the task is
                            # still touching the workspace.  Only the
                            # terminal ``status running=false`` —
                            # broadcast by the outermost ``finally`` of
                            # ``task_runner._run_task`` — proves the
                            # task thread exited (see the
                            # terminal-status branch below, the one
                            # place a stored result may be returned).
                            raise StopUnconfirmedTimeoutError(timeout_msg)
                        raise TimeoutError(timeout_msg)
                    sock.settimeout(remaining)
                try:
                    chunk = sock.recv(65536)
                except TimeoutError:
                    if deadline is None:
                        continue  # pure wake-up; keep waiting
                    # Finite deadline: loop back — the remaining<=0
                    # branch above decides between the stop-on-timeout
                    # cascade and raising.
                    continue
                if not chunk:
                    raise ConnectionError(
                        "The sorcar daemon closed the connection before "
                        "the task finished"
                    )
                recv_buf += chunk
                continue
            if newline_at >= _MAX_LINE_BYTES:
                # The newline landed in the same chunk that pushed the
                # frame over the cap, so the no-newline check above
                # never saw the overflow — the frame (newline included,
                # length ``newline_at + 1``) is over the limit all the
                # same.
                raise _frame_limit_error()
            line = bytes(recv_buf[: newline_at + 1])
            del recv_buf[: newline_at + 1]
            scanned = 0
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
                elif stopping:
                    if result_event is not None and result_event.get("success"):
                        # The task finished ON ITS OWN while the client
                        # was declaring the timeout: its successful
                        # result and terminal status were already on
                        # the wire when the stop was sent (the daemon's
                        # run-token-guarded stop is a no-op for a
                        # finished run).  Every daemon-side stop /
                        # cancel / failure path broadcasts its terminal
                        # ``result`` with ``success: false``
                        # (``task_runner._broadcast_failure_result``),
                        # so a successful result can only be a natural
                        # completion — return it instead of discarding
                        # the completed work behind a ``TimeoutError``
                        # that falsely claims the task "was stopped".
                        return _to_task_result(result_event, chat_id, task_id)
                    # The terminal status confirms the
                    # stopped-on-timeout task is dead; the run still
                    # timed out.  ``started`` is deliberately not
                    # required here: a stop can interrupt the task
                    # during its setup, BEFORE the initial
                    # ``running=true`` was ever broadcast, while the
                    # daemon's ``finally`` still broadcasts the
                    # terminal ``running=false`` (see
                    # ``task_runner._run_task``) — that is a confirmed
                    # stop, not an unconfirmed one.
                    raise TimeoutError(timeout_msg)
                elif started:
                    return _to_task_result(result_event, chat_id, task_id)
    except BaseException as exc:
        aborted = exc
        raise
    finally:
        if aborted is not None and not isinstance(aborted, TimeoutError):
            # The wait was aborted — typically by the KeyboardInterrupt
            # injected when the CALLING task is stopped while blocked
            # here.  Cascade the stop to the dispatched task: without
            # it the orphan keeps running invisibly and, when it is a
            # non-worktree task, keeps its repository flagged busy, so
            # a manual Git Commit after the parent's "Task stopped by
            # user" is refused with "A task is still running in this
            # folder".  A TimeoutError is excluded on purpose: its
            # documented contract is "the task keeps running", and a
            # ``stop_on_timeout`` timeout already sent its stop (and
            # awaited confirmation) inside the read loop.
            # Best-effort, like the closeTab below.
            # ``taskId`` carries this run's token so the daemon
            # rejects the stop if the tab was already reused by a
            # newer run (see ``_stop_task``'s run_token guard).
            try:
                _send_stop(sock, tab_id, run_token)
            except OSError:
                pass
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
        try:
            sock.close()
        except OSError:
            pass
