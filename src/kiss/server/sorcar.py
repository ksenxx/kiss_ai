# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Minimal synchronous client API for the local ``sorcar web`` daemon.

Any process can launch a task on an already-running daemon and block
until it finishes::

    from kiss.server import sorcar

    result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
    print(result.text, result.success, result.cost, result.tokens, result.steps)
    print(result.chat_id, result.task_id)  # daemon chat session / task row ids

    # Continue the same chat (the agent sees the prior task as context):
    follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)

Caller-supplied tools become agent tools: pass the path of a Python
file via ``tools="/path/to/my_tools.py"`` and the daemon imports the
file and registers every top-level public function that is suitable as
a tool (plain synchronous functions with keyword-bindable,
type-annotated parameters and Google-style docstrings).  The client
never serializes Python functions — the daemon loads the file itself,
so the tools execute **in the daemon process** like native agent
tools::

    # my_tools.py
    def get_temperature(city: str) -> str:
        \"\"\"Return the current temperature of a city.

        Args:
            city: Name of the city to look up.
        \"\"\"
        return lookup_sensor(city)

    result = sorcar.run("What's the temperature in Paris?",
                        tools="my_tools.py")

The function speaks the daemon's newline-delimited JSON protocol over
its Unix-domain socket (``$KISS_SORCAR_SOCK``, defaulting to
``$KISS_HOME/sorcar.sock``) — the same transport the VS Code extension
and the CLI client use — so no HTTP server, password, or extra
dependency is involved.  POSIX file permissions (mode 0o600) on the
socket restrict access to the owning user.
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
from kiss.server.tools_file import resolve_tools_file

# Read buffer limit for a single daemon event line.  The daemon emits
# large single-line JSON events (e.g. ``system_prompt`` carrying the
# full SYSTEM.md), so mirror the CLI client's generous 16 MiB cap.
_MAX_LINE_BYTES = 16 * 1024 * 1024


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


def run(
    prompt: str,
    *,
    work_dir: str = "",
    model: str = "",
    chat_id: str = "",
    tools: str | Path | None = None,
    use_worktree: bool = False,
    auto_commit: bool = False,
    timeout: float = 3600.0,
    sock_path: str | Path | None = None,
) -> TaskResult:
    """Run *prompt* as a task on the local Sorcar daemon and block until done.

    Connects to the ``sorcar web`` daemon's Unix-domain socket, sends
    the same ``run`` command a chat webview would, streams the task's
    events, and returns once the daemon reports the task finished.

    Args:
        prompt: The task instruction to run.
        work_dir: Working directory for the task; the daemon's current
            default is used when empty.
        model: Model name; the daemon's selected default when empty.
        chat_id: Optional existing chat session id to continue.  Pass
            the ``chat_id`` of a previous :class:`TaskResult` to run
            this task in the same chat — the agent then sees the prior
            tasks and results of that chat as context.  A new chat is
            started when empty.
        tools: Optional path to a Python file supplying extra tools
            for the agent.  The daemon imports the file and registers
            every top-level public function that is suitable as a tool
            (plain synchronous functions whose parameters are all
            keyword-bindable; ``*args``/``**kwargs``/positional-only
            parameters and coroutine/generator functions are skipped).
            Each function's name, docstring (Google-style ``Args:``
            section for parameter descriptions), and annotated
            parameters define the tool schema the agent sees, exactly
            like a native tool.  The functions are never serialized by
            the client — they run **in the daemon process**.  The path
            is resolved against this process's working directory.
        use_worktree: Run the task in an isolated git worktree.
        auto_commit: Auto-commit the task's changes on success.
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
        ValueError: When *prompt* is empty or blank, or when *tools*
            is not the path of an existing Python (``.py``) file (see
            :func:`~kiss.server.tools_file.resolve_tools_file`).
        ConnectionError: When no daemon is listening on the socket, or
            the daemon drops the connection before the task finishes.
        TimeoutError: When the task does not finish within *timeout*
            seconds.  The client then disconnects, which asks the
            daemon to close the task's tab.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    tools_file = resolve_tools_file(tools)
    path = _resolve_sock_path(sock_path)
    tab_id = f"api-{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(min(timeout, 10.0))
        try:
            sock.connect(str(path))
        except OSError as exc:
            raise ConnectionError(
                f"Cannot connect to the sorcar daemon at {path}: {exc} "
                f"— start it with `sorcar web`."
            ) from exc
        cmd = {
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": uuid.uuid4().hex,
            "chatId": chat_id,
            "workDir": work_dir,
            "model": model,
            "toolsFile": tools_file,
            "useWorktree": use_worktree,
            "autoCommit": auto_commit,
        }
        sock.sendall(json.dumps(cmd).encode("utf-8") + b"\n")
        reader = sock.makefile("rb", buffering=_MAX_LINE_BYTES)
        result_event: dict[str, Any] | None = None
        # ``chat_id`` (the parameter) doubles as the accumulator: when
        # the caller passed an existing chat id the daemon continues
        # that chat, so it is already the correct fallback; the run's
        # ``clear`` event then confirms (or, for a new chat, supplies)
        # the daemon-assigned id.
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
            try:
                event = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("tabId") != tab_id:
                continue
            etype = event.get("type")
            if etype == "clear":
                # ``_cmd_run`` stamps the launcher tab's ``clear``
                # broadcast with the chat session id it minted (or
                # reused) for this run.
                chat_id = str(event.get("chat_id", "") or "") or chat_id
            elif etype != "status" and event.get("taskId"):
                # Task-stream events are fanned out with the persisted
                # ``task_history`` row id injected as ``taskId``.
                # ``status`` events are excluded: they echo the
                # client-supplied correlation id from the ``run``
                # command, not the daemon's row id.
                task_id = str(event["taskId"])
            if etype == "result":
                result_event = event
            elif etype == "status":
                if event.get("running"):
                    started = True
                elif started:
                    return _to_task_result(result_event, chat_id, task_id)
    finally:
        try:
            sock.close()
        except OSError:
            pass
