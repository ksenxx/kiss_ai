# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Synchronous client for running tasks on the ``kiss-web`` daemon.

This module is the client half of the daemon's Python API: it speaks
the daemon's JSON command protocol over its local WSS endpoint (found
through the endpoint file, see :mod:`kiss.agents.sorcar.local_endpoint`)
and blocks until the submitted task finishes.  It lives in the
sorcar layer — not in ``kiss.server`` — because sorcar-layer code
(the ``run_agent`` dispatch tool in
:mod:`kiss.agents.sorcar.agent_dispatch` and the cron scheduler in
:mod:`kiss.agents.sorcar.cron_agent`) submits tasks back to the
daemon, and the layering invariant forbids sorcar code from importing
``kiss.server`` (see ``kiss.tests.agents.sorcar.test_layering_invariants``).
The public API surface is unchanged: :mod:`kiss.server.sorcar`
re-exports :func:`run` and :class:`TaskResult`, so
``kiss.server.sorcar.run(...)`` keeps working for external callers.

Depends only on ``websockets`` and the sorcar/core layers.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import local_endpoint
from kiss.core import tool_interrupt

_MAX_LINE_BYTES = 64 * 1024 * 1024
"""Largest daemon event frame the client accepts.

The daemon emits large single JSON events (e.g. ``system_prompt``
carrying the full SYSTEM.md), so this MUST match the daemon-side frame
limit (``web_server._MAX_LINE_BYTES``, 64 MiB).  A frame over the cap
closes the connection with code 1009, which :func:`run` reports as a
:class:`ConnectionError` (see :func:`_frame_limit_error`) instead of
silently misreporting a possibly terminal ``result`` event.
"""

_STOP_CONFIRM_GRACE_SECONDS = 20.0
"""Bounded wait for a stopped task's terminal status.

A ``stop_on_timeout`` timeout or a *cancel* sends the daemon a
``stop`` and then KEEPS READING until the task's terminal ``status
running=false`` confirms it is dead, before the ``TimeoutError`` or
:class:`CancelledError` is raised.  Without the confirmation the
caller would report the task stopped while it might still be running
on the daemon (a channel sub-task still holding its workspace).  The
daemon's stop path force-interrupts a non-cooperating task after
~1 s, so confirmation normally arrives quickly; this grace bounds the
wait against a wedged daemon, after which
:class:`StopUnconfirmedTimeoutError` (or a ``CancelledError`` with
``confirmed=False``) is raised and the stop stays best-effort.
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

class CancelledError(Exception):
    """The caller's *cancel* event was set while :func:`run` waited.

    The task is stopped exactly like a ``stop_on_timeout`` timeout: the
    ``stop`` is sent and the read loop keeps going (bounded by
    :data:`_STOP_CONFIRM_GRACE_SECONDS`) until the task's terminal
    ``status running=false`` proves it is dead.  Not a
    :class:`TimeoutError`: the task did not outlive a deadline, the
    caller asked for it to go.

    Attributes:
        result: The task's :class:`TaskResult` with the spend reported
            so far (all-zero when no ``result`` event arrived).
        confirmed: ``True`` when the terminal status arrived; ``False``
            when the grace expired first, so the stop stays best-effort
            and the task may still be running on the daemon.
    """

    def __init__(self, message: str, result: TaskResult, confirmed: bool) -> None:
        super().__init__(message)
        self.result = result
        self.confirmed = confirmed


class StoppedOnTimeoutError(TimeoutError):
    """Timeout whose ``stop_on_timeout`` stop the daemon confirmed.

    Raised by :func:`run` when the timed-out task was stopped and its
    terminal ``status running=false`` arrived.  :attr:`result` is the
    stopped task's final ``result`` event parsed into a
    :class:`TaskResult` (the daemon's failure result still carries the
    ``cost`` / ``tokens`` / ``steps`` spent before the stop), so callers
    can charge that spend to whoever dispatched the task.  It is a
    plain :class:`TimeoutError` to every ``except TimeoutError``.

    Attributes:
        result: The stopped task's :class:`TaskResult`; all-zero spend
            when no ``result`` event arrived before the terminal status.
    """

    def __init__(self, message: str, result: TaskResult) -> None:
        """Build the error.

        Args:
            message: The timeout message.
            result: The stopped task's :class:`TaskResult`.
        """
        super().__init__(message)
        self.result = result


_TOOL_CALL_WAKE_SECONDS = 0.5
"""Socket read wake-up interval while :func:`run` serves a tool call or a *cancel* event.

The ``run_agent`` tool's panel has a Stop button; the wait polls the
tool call's interrupt on every wake (``tool_interrupt.raise_if_interrupted``),
so this bounds how long that button takes to act.  A ``run_agent`` job
thread is not a tool call but watches its *cancel* event on every wake,
so this also bounds how long an ``agent_job`` kill takes to act.
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
        settings: The run's ``task_settings`` event payload (its
            effective model, work directory, budget, SEA, kind,
            tool profile, timeout, inherited and overridden values; see
            :mod:`kiss.agents.sorcar.run_config`); empty when the run
            ended before the daemon emitted it.
    """

    text: str
    success: bool
    cost: float
    tokens: int
    steps: int
    chat_id: str = ""
    task_id: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


def _resolve_endpoint_file(endpoint_file: str | Path | None) -> Path:
    """Return the daemon endpoint file to read.

    Precedence: explicit *endpoint_file* argument, then the
    ``KISS_SORCAR_LOCAL`` environment variable, then the daemon's
    default ``$KISS_HOME/sorcar-local.json``.

    Args:
        endpoint_file: Optional explicit endpoint file override.

    Returns:
        The resolved endpoint file path.
    """
    if endpoint_file:
        return Path(endpoint_file)
    return local_endpoint.default_endpoint_path()


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


def _event_cost(event: dict[str, Any]) -> float:
    """Return the USD cost carried by a ``result`` / ``usage_info`` event.

    The daemon sends the exact figure as ``cost_usd`` next to the
    four-decimal ``"$x.xxxx"`` display string in ``cost``; the string is
    the fallback for events (older daemons, late-charge deltas) without
    it, so the spend a parent charges to its ledger is never truncated
    to the display precision.

    Args:
        event: A daemon event carrying ``cost`` and possibly ``cost_usd``.

    Returns:
        The cost in USD; ``0.0`` when neither field yields a number.
    """
    exact = event.get("cost_usd")
    if isinstance(exact, (int, float)):
        return float(exact)
    return _parse_cost(event.get("cost"))


def _net_totals(event: dict[str, Any], charged: dict[str, Any]) -> dict[str, Any]:
    """Return *event*'s task totals minus spend already charged elsewhere.

    A side channel's spend that lands after the task's row was saved is
    banked directly on the task's still-running ancestor — the task
    that is waiting on this run — and the task's new totals are
    broadcast with the banked delta in ``ancestor_charged`` (see
    :func:`kiss.server.task_update.charge_side_channel_usage`).  The
    waiting caller must not fold that delta a second time, in this
    event or in any later one (a merge agent's totals include it too),
    so each delta is added to *charged* and subtracted from every
    totals event from then on.

    Args:
        event: A ``result`` or ``usage_info`` event carrying ``cost``.
        charged: Running ``{"cost", "tokens", "steps"}`` sums of the
            deltas already charged elsewhere; updated in place.

    Returns:
        A totals dict with ``cost`` / ``total_tokens`` / ``step_count``.
    """
    delta = event.get("ancestor_charged")
    if isinstance(delta, dict):
        charged["cost"] += _event_cost(delta)
        charged["tokens"] += int(delta.get("tokens", 0) or 0)
        charged["steps"] += int(delta.get("steps", 0) or 0)
    steps = event.get("step_count", event.get("total_steps", 0))
    return {
        "cost": _event_cost(event) - charged["cost"],
        "total_tokens": int(event.get("total_tokens", 0) or 0) - charged["tokens"],
        "step_count": int(steps or 0) - charged["steps"],
    }


def _to_task_result(
    event: dict[str, Any] | None,
    chat_id: str = "",
    task_id: str = "",
    totals: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> TaskResult:
    """Convert the final daemon ``result`` event into a :class:`TaskResult`.

    Args:
        event: The last ``result`` event received for the task's tab,
            or ``None`` when the task ended without one.
        chat_id: The daemon chat session id observed on the run's
            ``clear`` event (``""`` when none was seen).
        task_id: The persisted ``task_history`` row id observed on the
            run's event stream (``""`` when none was seen).
        totals: The latest task totals (see :func:`_net_totals`)
            received for the task's tab; defaults to *event*'s.
            The spend comes from here because the task's totals can
            grow after its ``result``: the pre-run
            classifier's spend is folded in only after the agent
            emitted its result, and announced by a later
            ``usage_info``.
        settings: The ``settings`` payload of the run's
            ``task_settings`` event, or ``None`` when none was seen.

    Returns:
        The parsed :class:`TaskResult`.  The daemon enriches ``result``
        events with ``success`` / ``summary`` fields parsed from the
        agent's YAML result; ``summary`` is preferred over the raw
        ``text`` when present.
    """
    spend = totals if totals is not None else event or {}
    steps = spend.get("step_count", 0)
    return TaskResult(
        text=str((event or {}).get("summary") or (event or {}).get("text") or ""),
        success=bool((event or {}).get("success", False)),
        cost=_event_cost(spend),
        tokens=int(spend.get("total_tokens", 0) or 0),
        steps=int(steps or 0),
        chat_id=chat_id,
        task_id=task_id,
        settings=dict(settings or {}),
    )


def resolve_sea_path(sea_path: str | None) -> str:
    """Validate a client-supplied SEA path and resolve it.

    Client-side counterpart of the daemon's
    ``kiss.agents.sorcar.sea_apply.apply_sea``: the path is
    resolved against the CLIENT's working directory (the daemon may run
    with a different one) and validated eagerly so a bad value fails
    fast, before any daemon connection is made.

    Args:
        sea_path: Path string of a SEA file (a Python file defining a
            ``BaseSea`` subclass), or ``None``/empty for no SEA.

    Returns:
        The absolute path as a string, or ``""`` when *sea_path* is
        ``None`` or empty.

    Raises:
        ValueError: When *sea_path* is neither ``None`` nor a string,
            is not a ``.py`` file, or does not exist.
    """
    if sea_path is None or sea_path == "":
        return ""
    if not isinstance(sea_path, str):
        raise ValueError(
            "sea_path must be a string path to a Python file, got "
            f"{type(sea_path).__name__}: {sea_path!r}"
        )
    path = Path(sea_path).expanduser().resolve()
    # Quote the path literally rather than via repr(): repr doubles every
    # backslash of a Windows path, which misleads the reader.
    if path.suffix != ".py":
        raise ValueError(f"SEA '{path}' is not a Python (.py) file")
    if not path.is_file():
        raise ValueError(f"SEA '{path}' does not exist")
    return str(path)


def _frame_limit_error() -> ConnectionError:
    """Return the error for a daemon frame exceeding the client cap.

    Reads :data:`_MAX_LINE_BYTES` at call time (tests shrink it).
    """
    return ConnectionError(
        "The sorcar daemon sent an event frame larger "
        f"than the {_MAX_LINE_BYTES}-byte client limit"
    )


def _closed_error(exc: ConnectionClosed) -> ConnectionError:
    """Translate a closed daemon connection into the client's error.

    A close this client itself initiated with code 1009 (message too
    big) means the daemon sent a frame over :data:`_MAX_LINE_BYTES`;
    any other close means the daemon went away before the task
    finished.
    """
    if exc.sent is not None and exc.sent.code == 1009:
        return _frame_limit_error()
    return ConnectionError(
        "The sorcar daemon closed the connection before the task finished"
    )


def _send(ws: ClientConnection, cmd: dict[str, Any]) -> None:
    """Send one command frame, raising ``OSError`` when the connection is gone.

    Args:
        ws: The connected daemon connection.
        cmd: The JSON command to send.

    Raises:
        OSError: When the frame could not be written within
            :func:`local_endpoint.send`'s deadline or the connection is
            gone.
    """
    try:
        local_endpoint.send(ws, json.dumps(cmd))
    except ConnectionClosed as exc:
        raise OSError(str(exc)) from exc


def _send_stop(ws: ClientConnection, tab_id: str, run_token: str) -> None:
    """Send the daemon a run-token-guarded ``stop`` for *tab_id*.

    Shared by :func:`run`'s stop-on-timeout path and the abort-cascade
    in its ``finally`` block: both stops MUST carry the run token (so
    the daemon's ``_stop_task`` guard rejects the stop when the tab
    was reused by a newer run), and a drifted duplicate would desync
    that guarantee.

    Args:
        ws: The connected daemon connection.
        tab_id: The run's synthetic tab id.
        run_token: The client-minted per-submission run token.

    Raises:
        OSError: When the stop could not be written.
    """
    _send(ws, {"type": "stop", "tabId": tab_id, "taskId": run_token})


def _stop_and_await_confirmation(
    ws: ClientConnection, tab_id: str, run_token: str, what: str,
) -> float:
    """Send the stop for a *what* task and return the confirmation deadline.

    Args:
        ws: The connected daemon connection.
        tab_id: The run's synthetic tab id.
        run_token: The client-minted per-submission run token.
        what: ``"cancelled"`` or ``"timed-out"``, for the error text.

    Returns:
        The ``time.monotonic()`` deadline by which the daemon's terminal
        status must confirm the stop (:data:`_STOP_CONFIRM_GRACE_SECONDS`).

    Raises:
        ConnectionError: When the stop could not even be sent, so the
            task was neither stopped nor confirmed dead — raising a
            plain ``TimeoutError`` would let a caller claim "was
            stopped"; the broken daemon connection is surfaced instead,
            like every other mid-run connection failure.
    """
    # Clocked before the send: a slow but
    # successful send must not extend the caller's wait.
    deadline = time.monotonic() + _STOP_CONFIRM_GRACE_SECONDS
    try:
        _send_stop(ws, tab_id, run_token)
    except OSError as send_exc:
        raise ConnectionError(
            "The sorcar daemon connection failed while "
            f"stopping the {what} task: {send_exc}"
        ) from send_exc
    return deadline


def run(
    prompt: str,
    *,
    work_dir: str = "",
    scope_work_dir: str = "",
    parent_task_id: str = "",
    parent_tab_id: str = "",
    parent_reviewer: bool = False,
    side_channel: bool = False,
    model: str = "",
    chat_id: str = "",
    system_prompt: str = "",
    sea_path: str = "",
    use_worktree: bool = True,
    auto_commit: bool = True,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    use_web_tools: bool | None = None,
    auto_classify: bool | None = None,
    use_memory: bool | None = None,
    is_parallel: bool = True,
    add_to_system_prompt: str = "",
    add_to_prompt: str = "",
    tool_profile: str = "",
    docker_image: str = "",
    inherit_tools: bool = False,
    workspace: str = "",
    provenance: dict[str, str] | None = None,
    timeout: float | None = 3600.0,
    stop_on_timeout: bool = False,
    record_timeout: float | None = None,
    endpoint_file: str | Path | None = None,
    cancel: threading.Event | None = None,
    running: threading.Event | None = None,
) -> TaskResult:
    """Run *prompt* as a task on the local Sorcar daemon and block until done.

    Connects to the ``kiss-web`` daemon's local WSS endpoint, sends
    the same ``run`` command a chat webview would, streams the task's
    events, and returns once the daemon reports the task finished.

    Args:
        prompt: The task instruction to run.
        work_dir: Working directory for the task; the daemon's current
            default is used when empty.
        scope_work_dir: The CALLING workspace recorded on the task's
            tab in the daemon's shared tab registry (``scopeWorkDir``),
            kept separate from *work_dir* (the channel/cron scratch
            directory a standalone dispatch executes in).
            Informational only: every client shows every registry tab
            whatever folder it runs in.  Empty (the default) records
            nothing.  Irrelevant for a sub-agent run (non-empty
            *parent_task_id*), which gets no registry tab at all.
        parent_task_id: The persisted ``task_history`` row id of the
            CALLING task, when this run is dispatched on behalf of one
            (the ``run_agent`` tool).  Non-empty marks the run as a
            SUB-AGENT of that task, giving its tab the same frontend
            behavior as a ``run_parallel`` sub-task: no top-level tab
            of its own — instead every client viewing the parent gets
            a nested sub-agent tab (via the run's ``new_tab``
            broadcast), the run's history row nests under the parent
            task, and a ``subagentDone`` broadcast stops the tab's
            running indicator when the run ends.  Empty (the default)
            runs as an ordinary top-level task.  It is a
            client/UI-transport parameter with no SEA
            setting: a dispatched script must not be able to re-parent
            itself under an unrelated task.
        parent_tab_id: Frontend tab id of the calling task's tab,
            forwarded on the sub-agent's ``new_tab`` broadcast so the
            webview knows which tab spawned it (nested placement and
            cascade-close).  Only meaningful with *parent_task_id*;
            empty spawns a parentless sub-agent tab, exactly like a
            headless ``run_parallel`` fan-out.  No SEA setting.
        parent_reviewer: Whether the dispatched run belongs to a
            reviewer's sub-tree — the caller is a reviewer sub-agent,
            or *prompt* itself is a review task (see
            :mod:`kiss.agents.sorcar.fanout_guard`).  Stamped on the
            child's ``_subagent_info`` so it and its helpers keep the
            read-only ``review`` tool profile.  Only meaningful with
            *parent_task_id*; no SEA setting, for the same
            reason as *parent_task_id*.
        side_channel: Whether the run is a side channel of the parent
            — a sub-agent whose result is delivered into the PARENT's
            transcript (the ``/ask`` answer panel), so its own nested
            tab is scaffolding that is closed when the run ends and
            never re-opened by a replay.  Persisted on the child's
            history row; only meaningful with *parent_task_id*; no
            SEA setting.
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
            id, ``$KISS_HOME/AGENTS.md``) so the agent's tool contract
            keeps working.  Empty (default) runs with the default
            system prompt as usual.
        sea_path: Optional path — a string — to a Python
            file defining a Sorcar Extension Agent (SEA) that
            configures this run **on the daemon**.  When non-empty, the
            daemon loads the file's one subclass of
            :class:`kiss.agents.seas.base.base_sea.BaseSea` and applies
            its methods (:mod:`kiss.agents.sorcar.sea_commands`,
            :func:`kiss.agents.sorcar.sea_apply.apply_sea`) on top
            of the values passed to this call: a setting the SEA
            declares replaces the parameter of the same name; one it
            does not declare keeps the value passed here.

            File format — one class deriving from ``BaseSea`` that
            overrides any of its methods (every one has an identity
            default)::

                from kiss.agents.seas.base.base_sea import BaseSea

                class MySea(BaseSea):
                    def description(self) -> str: ...            # /xxx help text
                    def settings(self, settings: dict) -> dict:  # settings | {kind, run() keywords}
                        ...
                    def prompt(self, task: str) -> str: ...      # the task text -> the prompt
                    def system_prompt(self, system_prompt: str) -> str:  # assembled -> the run's
                        ...
                    def tools(self, tools: list) -> list: ...    # toolset -> the run's toolset
                    def llm_call_hook(self, new_messages: list) -> list: ...
                    def tool_call_hook(self, name: str, args: dict) -> Verdict: ...  # ALLOW/refuse
                    def register_as_model(self) -> bool: ...     # model-picker entry
                    def on_picked_as_model(self, work_dir: str) -> str: ...

            A SEA extends another by deriving from its class (the
            launcher runs every class of the chain, base first; do not
            call ``super()``).  ``settings`` returns its argument with
            any of the keyword parameters of this function except the
            transport, identity and prompt ones (``prompt`` and
            ``system_prompt`` are the methods above, not settings)
            laid over it: ``work_dir``, ``model``, ``chat_id``,
            ``use_worktree``, ``auto_commit``, ``max_budget`` (finite),
            ``model_config``, ``use_web_tools``, ``auto_classify``,
            ``use_memory``, ``tool_profile``, ``docker_image``; plus
            two dispatcher keys: ``timeout`` (seconds a ``run_agent``
            call waits for this SEA's sub-task) and ``locked`` (keys an
            explicit caller argument may not change).  ``prompt(task)``
            receives the task text and returns the prompt body; every
            ``{task_id}`` of the result is replaced by
            *parent_task_id*.  A ``None`` value, or ``""`` for a
            string key, means "no override".  The base class says what
            the SEA is: ``BaseSea`` changes nothing; ``WorkerSea``
            lays the worker defaults under the explicit keys
            (``use_worktree``, ``auto_commit``, ``auto_classify``,
            ``use_web_tools``, ``use_memory`` all off); ``ChannelSea``
            (a worker) marks a messaging channel: it defaults
            ``work_dir`` to ``$KISS_HOME/channel_work`` and locks it
            with the worker keys, gives the run the channel preamble
            and a workspace held for the run, makes a ``run_agent``
            sub-task of it inherit nothing from the caller, and lists
            it as a channel.

            ``system_prompt(system_prompt)`` receives the run's
            assembled system prompt (the base prompt plus
            *add_to_system_prompt*) and returns the run's: the same
            text with additions, or a replacement.  ``tools(tools)``
            receives the built-in toolset and returns the run's: a list
            of tool callables (never a file path); with
            ``"tool_profile": "none"`` it starts empty, so what it
            returns and ``finish`` are the run's whole tool set (the
            default ``SYSTEM.md`` prompt, whose workflow rules name
            ``Read``, ``Edit``, ``Bash`` and the browser tools, should
            then usually be replaced by a ``system_prompt`` written for
            the tools the run has).  Each tool's name, docstring
            (Google-style ``Args:`` section) and annotated
            keyword-bindable parameters define the tool schema the
            agent sees, exactly like a native tool.

            The two hook methods are passed to the underlying
            :meth:`kiss.core.kiss_agent.KISSAgent.run` of every
            task-executor sub-session of the task's agent (internal
            helper sessions, e.g. the failed-session trajectory
            summarizer, are not hooked).  Per that method's contract,
            ``llm_call_hook(new_messages)`` is called before every LLM
            call and its return value replaces the new messages about
            to be sent, and ``tool_call_hook(name, args)`` is called
            before every tool call and returns a ``Verdict``: the tool
            executes on ``ALLOW``; on ``refuse(text)`` the model is
            given *text* as the tool's result instead.  The hooks apply
            to the task's own agent, not to sub-agents it spawns via
            ``run_parallel``.

            A SEA whose ``register_as_model()`` returns ``True`` is
            listed in the model picker under its command name (the
            bundled ``autorouter`` and ``bestrouter``); picking it runs
            every task of the tab through the SEA on the model its
            ``settings()["model"]`` names (else the default model),
            with the protocol its ``system_prompt`` method adds to the
            system prompt.

            Everything the script defines runs **in the daemon
            process**; nothing is serialized by the client.
            ``timeout``, *stop_on_timeout*, *endpoint_file*,
            *scope_work_dir*, *parent_task_id*, and *parent_tab_id*
            have no settings key by design: the first three are
            client-transport parameters — the script only runs on the
            daemon that *endpoint_file* selects, *timeout* bounds this
            client's local wait, and *stop_on_timeout* picks this
            client's timeout behavior — and *scope_work_dir* /
            *parent_task_id* / *parent_tab_id* are the CALLING task's
            identity, which the script must not be able to forge.  The
            *sea_path* itself is resolved against this
            process's working directory and validated eagerly.  A
            broken SEA (deleted before the daemon reads it,
            raising at import time, a non-callable getter, a raising
            ``settings()`` or getter, an unknown settings key or
            kind, or a wrong-typed value) stops the task: the daemon
            fails the run and the returned :class:`TaskResult` carries
            the diagnostic error in its ``text`` with ``success=False``.
        use_worktree: Run the task in an isolated git worktree.
            Defaults to True.
        auto_commit: Auto-commit the task's changes on success.
            Defaults to True.
        max_budget: Per-task budget override in USD; ``None`` uses the
            daemon's configured default.
        model_config: Per-task model configuration override (custom
            endpoint / headers); ``None`` uses the daemon's configured
            model endpoint.  Must be JSON-serializable.
        use_web_tools: Per-task browser-tool enablement override,
            mapped to the agent's ``web_tools`` toggle
            (:meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`);
            ``None`` uses the daemon's configured default (the
            settings panel's "Use web tools" checkbox, persisted as
            ``use_web_browser``).
        auto_classify: Per-task override of pre-run task
            classification (``kiss.agents.sorcar.task_classifier``),
            which runs one lightweight model call before the task — a
            typed question to a decisions model when an OpenRouter key
            is configured and the settings panel's "Use Jev"
            checkbox (``classify_with_decisions``) is on, else a
            non-agentic call on the run's own LLM — to pick the system
            prompt (lite vs. full) and decide worktree isolation for
            the run (the verdict can only
            demote a run that asked for a worktree to direct
            execution; a *use_worktree* of ``False`` passed here is
            never overridden).  ``True`` forces
            classification on, ``False`` skips it — the run then keeps
            the *use_worktree* value passed here and the full system
            prompt — and ``None`` (the default) uses the daemon's
            configured default (the settings panel's "Classify tasks
            before running" checkbox, persisted as ``classify_tasks``).
        use_memory: Per-task persistent-memory override
            (``kiss.core.memoryfield``), mapped to the agent's
            ``use_memory`` toggle
            (:meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`).
            ``True`` gives the run (and its ``run_parallel``
            sub-agents) the ``memory_*`` tools plus the
            ``MEMORY_PROTOCOL`` prompt block, ``False`` withholds
            them, and ``None`` (the default) uses the daemon's
            configured default (the settings panel's "Use persistent
            memory" checkbox, persisted as ``use_memory``, or the
            daemon process's ``KISS_USE_MEMORY`` environment
            variable).  A boolean override never bypasses the memory
            safety gates: a run without the basic toolset (the
            ``none`` tool profile), a Docker run, a
            run-to-completion CLI model (``cc/*``, ``codex/*``), or a
            caller-supplied ``model_config["system_instruction"]``
            stays memory-free even with ``True``.
        is_parallel: Whether the agent gets the ``run_parallel`` tool
            (the user's parallel-mode toggle).  Defaults to True.
        add_to_system_prompt: Extra text appended to the run's
            system prompt when the agent is executed — after the
            default ``SYSTEM.md`` prompt (or the *system_prompt*
            replacement) and before the daemon's per-run operational
            instructions.  ``run_parallel`` sub-agents inherit the
            suffix on their own system prompts, like a *system_prompt*
            replacement, so the extra instructions constrain the whole
            task tree.  Empty (default) appends nothing.
        add_to_prompt: Extra text appended to the executed task
            prompt.  A multi-``<task>`` *prompt* runs the agent once
            per subtask, and the text is appended to EACH subtask's
            prompt.  The appended text is part of the prompt the agent
            actually runs with, so it is also what the chat history
            records and what follow-up tasks of the same chat see as
            context.  Empty (default) appends nothing.
        tool_profile: Name of the tool profile the task's built-in
            toolset is cut down to — a key of
            :data:`kiss.agents.sorcar.sorcar_agent.TOOL_PROFILES`
            (the composites ``"full"``, ``"review"``, ``"assistant"``,
            ``"bash"`` or the groups ``"shell"``, ``"edit"``,
            ``"browser"``, ``"memory"``, ``"agents"``, ``"mcp"``,
            ``"skills"``, ``"user"``, ``"decide"``, ``"control"``), or
            several keys joined with ``+`` for the union of their
            tools (``"shell+edit+memory"``); ``bash`` is the
            single-command runner of the bundled ``/sh`` agent:
            ``Bash`` and ``finish`` only; ``none`` keeps no built-in
            tool at all (``finish`` plus what the SEA's ``tools()``
            returns, the bundled ``/ask`` agent).  Empty
            (the default) keeps the daemon's usual choice (the full
            toolset).
            An unknown name stops the task with a diagnostic error.
        docker_image: Run the task's file and shell tools (``Bash``,
            ``run_commands_parallel``, ``Read``, ``Edit``, ``Write``)
            inside a Docker container instead of on the daemon's host.
            An image name (``"python:3.12"``) starts a fresh container
            that is removed when the task ends; ``container:<name-or-id>``
            attaches to a container the caller already runs and leaves
            it running.  ``run_parallel`` sub-agents share the task's
            container.  ``bash_job`` and persistent memory are
            unavailable in a Docker run.  Empty (default) runs the
            tools on the host.
        inherit_tools: Whether the task also gets the extra tools of
            the task *parent_task_id* names — the tool callables that
            parent's SEA added through ``tools()`` (and those the
            parent inherited itself), resolved on the daemon from the
            running parent (a callable cannot travel the wire).  They
            are added to the task's built-in toolset before the task's
            own SEA's ``tools()`` sees it, skipping names the task
            already has; a SEA on the ``none`` tool profile keeps
            exactly its own set.
            ``run_agent`` sets this for the
            sub-tasks it dispatches in path mode, so a sub-task that
            inherits the caller's system prompt also has the tools
            that prompt refers to.  ``False`` (default) adds nothing;
            ignored without *parent_task_id*.
        workspace: Workspace/account identifier for multi-account
            channels.  A channel SEA's run (``channel: True``) holds
            it (``KISS_CHANNEL_WORKSPACE``) for its whole lifetime, so
            its channel tools load that account's credentials; empty
            means ``"default"``.  Ignored by every other run.
        provenance: Where the command's values come from, ``{setting
            key: "explicit" | "inherited"}`` (see
            :mod:`kiss.agents.sorcar.run_config`): a field the caller
            was given explicitly, or one it filled from the agent
            calling it.  Recorded in the task's ``task_settings`` event
            (``inherited``), so the result's ``ran:`` line and the
            persisted history show which values the sub-task took over.
            ``None`` (default): every value is a persisted setting or
            the daemon's default.
        timeout: Maximum seconds to wait for the task to finish;
            ``None`` waits indefinitely.  With *stop_on_timeout* it is
            also sent to the daemon (wire field ``timeout``), which
            records it in the task's ``task_settings`` event.
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
            the wait (the ``/ask`` side channel, ``commands.py``).  The
            ``run_agent`` dispatch does not use it: its ``timeout``
            bounds the call only, and its sub-task is stopped through
            *cancel* (an ``agent_job`` kill, or the calling run's end).
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
        record_timeout: A bound to send as the wire field ``timeout``
            (recorded in the task's ``task_settings`` and checked
            against a SEA's locked ``timeout``) when this wait itself
            has none: ``run_agent`` waits here without a deadline on a
            job thread and bounds the call by joining that thread, so
            it passes the call's bound through this.  ``None`` sends
            *timeout* when *stop_on_timeout*, else nothing.
        endpoint_file: Daemon endpoint file override (defaults to
            ``$KISS_SORCAR_LOCAL`` or ``$KISS_HOME/sorcar-local.json``).
        cancel: An event the caller may set from another thread to
            stop the task (``agent_job(..., "kill")``): the daemon is
            sent a ``stop`` and the wait continues until the task's
            terminal status confirms it is dead (or the confirmation
            grace expires), then :class:`CancelledError` is raised with
            ``confirmed`` set accordingly.  ``None`` (the default)
            waits for the task or the timeout.
        running: An event set when the task's initial ``status
            running=true`` arrives — the moment its tab exists on every
            client.  Not set by a wait that ends without one: the
            ``run_agent`` job thread sets it after recording the
            dispatch's outcome, so a ``wait="false"`` call woken by it
            either has a tab to report or the finished result.

    Returns:
        A :class:`TaskResult` with the result text, success flag, cost
        (USD), total tokens, step count, chat id, and task id of the
        task.  ``chat_id`` is the daemon chat session id and
        ``task_id`` the persisted ``task_history`` row id — both
        usable later to look up or resume the run in the daemon's
        history.

    Raises:
        ValueError: When *prompt* is empty or blank, or when
            *sea_path* is neither empty nor the path string
            of an existing Python (``.py``) file (see
            :func:`resolve_sea_path`).
        ConnectionError: When no daemon is reachable at the endpoint,
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
        StoppedOnTimeoutError: When *stop_on_timeout* is true and the
            daemon confirmed the stop; its ``result`` carries the
            stopped task's spend.  A subclass of ``TimeoutError``.

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
    waiting, not to cancel the work.  Such an abort's exception gets a
    ``task_result`` attribute: a :class:`TaskResult` with the spend the
    dispatched task had reported so far, so the caller can still
    charge it.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    sea_file = resolve_sea_path(sea_path)
    path = _resolve_endpoint_file(endpoint_file)
    tab_id = f"api-{uuid.uuid4().hex}"
    # Client-minted per-submission run token.  Echoed on the run's
    # ``status`` events, and — critically — sent with the
    # abort-cascade ``stop`` below so the daemon only stops THIS run:
    # a late stop must never kill a newer run that reused the tab.
    run_token = uuid.uuid4().hex
    deadline = None if timeout is None else time.monotonic() + timeout
    ws: ClientConnection | None = None
    aborted: BaseException | None = None
    result_event: dict[str, Any] | None = None
    settings_event: dict[str, Any] | None = None  # the run's task_settings payload
    totals_event: dict[str, Any] | None = None  # latest spend totals
    charged = {"cost": 0.0, "tokens": 0, "steps": 0}  # see _net_totals
    task_id = ""
    try:
        ws = local_endpoint.connect(
            path,
            open_timeout=10.0 if timeout is None else min(timeout, 10.0),
            max_size=_MAX_LINE_BYTES,
        )
        cmd = {
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": run_token,
            "chatId": chat_id,
            "workDir": work_dir,
            "tabScopeWorkDir": scope_work_dir,
            "parentTaskId": parent_task_id,
            "parentTabId": parent_tab_id,
            "parentReviewer": parent_reviewer,
            "sideChannel": side_channel,
            "model": model,
            "systemPrompt": system_prompt,
            "seaPath": sea_file,
            "useWorktree": use_worktree,
            "autoCommit": auto_commit,
            "maxBudget": max_budget,
            "modelConfig": model_config,
            "useWebTools": use_web_tools,
            "classifyTasks": auto_classify,
            "useMemory": use_memory,
            "isParallel": is_parallel,
            "appendToSystemPrompt": add_to_system_prompt,
            "appendToPrompt": add_to_prompt,
            "toolProfile": tool_profile,
            "dockerImage": docker_image,
            "inheritTools": inherit_tools,
            "workspace": workspace,
            "provenance": provenance or {},
            "timeout": (timeout if stop_on_timeout else None) if record_timeout is None
            else record_timeout,
        }
        try:
            local_endpoint.send(ws, json.dumps(cmd))
        except ConnectionClosed as exc:
            raise _closed_error(exc) from exc
        started = False
        stopping = False  # stop sent (timeout or cancel); awaiting confirmation
        cancelled = False  # the stop was requested through *cancel*
        timeout_msg = f"Task did not finish within {timeout} seconds"
        # Inside a tool call, or with a *cancel* event to watch (the
        # run_agent job thread), the wait wakes often enough for the
        # call's Stop button or an ``agent_job`` kill to feel immediate.
        wake_seconds = (
            _TOOL_CALL_WAKE_SECONDS
            if tool_interrupt.current_tool_call() is not None or cancel is not None
            else _NO_DEADLINE_WAKE_SECONDS
        )
        while True:
            # The calling task's tool-call panel Stop is honored
            # cooperatively: every wake checks it (raising
            # ToolCallInterrupted, which the finally below turns
            # into a stop of the dispatched task).  A background job's
            # kill is the same abort, requested through *cancel*.
            tool_interrupt.raise_if_interrupted()
            if cancel is not None and cancel.is_set() and not cancelled:
                # Stop the task and keep reading until its terminal
                # status confirms it is dead — the same stop-and-confirm
                # as a stop-on-timeout, decided below in the
                # ``stopping`` branches with ``cancelled`` set.
                cancelled = stopping = True
                deadline = _stop_and_await_confirmation(ws, tab_id, run_token, "cancelled")
                continue
            if deadline is None:
                # No deadline: wake periodically so an injected
                # abort (see _NO_DEADLINE_WAKE_SECONDS) can be
                # delivered; the timeout is retried, not an error.
                wait = wake_seconds
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if stop_on_timeout and not stopping:
                        # Stop the timed-out task, then KEEP READING
                        # (bounded by the confirmation grace) until
                        # its terminal status proves it is dead — the
                        # caller must not resume while the child could
                        # still act (see _STOP_CONFIRM_GRACE_SECONDS).
                        stopping = True
                        deadline = _stop_and_await_confirmation(
                            ws, tab_id, run_token, "timed-out",
                        )
                        continue
                    if stopping:
                        # The confirmation grace expired without a
                        # terminal status: the stop was sent but never
                        # answered, so the task may still be running —
                        # the caller must not be told it was stopped.
                        # Even a stored SUCCESSFUL result is no proof
                        # the task is dead: the agent emits it BEFORE
                        # the daemon's persistence / auto-commit /
                        # worktree cleanup stages run, and a stop can
                        # still take effect during those stages, so
                        # returning the result here would let the
                        # caller (``run_agent``) release its workspace
                        # reservation while the task is still touching
                        # the workspace.  Only the terminal ``status
                        # running=false`` — broadcast by the outermost
                        # ``finally`` of ``task_runner._run_task`` —
                        # proves the task thread exited (see the
                        # terminal-status branch below, the one place
                        # a stored result may be returned).
                        if cancelled:
                            raise CancelledError(
                                "the stop was sent but not confirmed",
                                _to_task_result(
                            result_event, chat_id, task_id, totals_event, settings_event,
                        ),
                                confirmed=False,
                            )
                        raise StopUnconfirmedTimeoutError(timeout_msg)
                    raise TimeoutError(timeout_msg)
                # Capped like the no-deadline wait: an injected abort
                # (the calling task's Stop) cannot land inside ``recv``,
                # and a silent daemon would otherwise hold it back —
                # and the cooperative check above — for the whole
                # *remaining*.
                wait = min(remaining, wake_seconds)
            try:
                raw = ws.recv(timeout=wait)
            except TimeoutError:
                # Pure wake-up (or, with a finite deadline, loop back so
                # the remaining<=0 branch above decides between the
                # stop-on-timeout cascade and raising).
                continue
            except ConnectionClosed as exc:
                raise _closed_error(exc) from exc
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("tabId") != tab_id:
                continue
            etype = event.get("type")
            if etype == "clear":
                chat_id = str(event.get("chat_id", "") or "") or chat_id
            elif etype != "status" and event.get("taskId"):
                task_id = str(event["taskId"])
            if etype in ("result", "usage_info") and "cost" in event:
                totals_event = _net_totals(event, charged)
            if etype == "result":
                result_event = event
            elif etype == "task_settings" and isinstance(event.get("settings"), dict):
                settings_event = event["settings"]
            elif etype == "status":
                if event.get("running"):
                    started = True
                    if running is not None:
                        running.set()
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
                        return _to_task_result(
                            result_event, chat_id, task_id, totals_event, settings_event,
                        )
                    # The terminal status confirms the
                    # stopped-on-timeout task is dead; the run still
                    # timed out.  ``started`` is deliberately not
                    # required here: a stop can interrupt the task
                    # during its setup, BEFORE the initial
                    # ``running=true`` was ever broadcast, while the
                    # daemon's ``finally`` still broadcasts the
                    # terminal ``running=false`` (see
                    # ``task_runner._run_task``) — that is a confirmed
                    # stop, not an unconfirmed one.  The stopped task's
                    # failure result still reports its spend.
                    if cancelled:
                        raise CancelledError(
                            "the task was stopped by the caller",
                            _to_task_result(
                            result_event, chat_id, task_id, totals_event, settings_event,
                        ),
                            confirmed=True,
                        )
                    raise StoppedOnTimeoutError(
                        timeout_msg,
                        _to_task_result(
                            result_event, chat_id, task_id, totals_event, settings_event,
                        ),
                    )
                elif started or result_event is not None:
                    # A result before any ``running=true`` means the
                    # task failed during the daemon's setup (chat /
                    # work-dir resolution, a stop injected before the
                    # initial status): ``_run_task`` broadcasts its
                    # failure ``result`` and then the terminal status
                    # from its ``finally``.  Without ``result_event``
                    # in this condition that terminal status would be
                    # ignored and the loop would wait out the whole
                    # timeout (forever with ``timeout=None``).
                    return _to_task_result(
                            result_event, chat_id, task_id, totals_event, settings_event,
                        )
    except BaseException as exc:
        aborted = exc
        if ws is not None and not isinstance(exc, (TimeoutError, CancelledError)):
            # The dispatched task is stopped below, but whatever it
            # already spent stays spent: hand the caller the latest
            # totals so it can still charge them (``run_agent`` folds
            # them into the stopped calling task).
            exc.task_result = _to_task_result(  # type: ignore[attr-defined]
                result_event, chat_id, task_id, totals_event, settings_event,
            )
        raise
    finally:
        # Nothing to cascade or close when the connect itself failed:
        # there is no task and no tab on the daemon's side.
        if (
            ws is not None
            and aborted is not None
            and not isinstance(aborted, (TimeoutError, CancelledError))
        ):
            # The wait was aborted — typically by the KeyboardInterrupt
            # injected when the CALLING task is stopped while blocked
            # here.  Cascade the stop to the dispatched task: without
            # it the orphan keeps running invisibly and, when it is a
            # non-worktree task, keeps its repository flagged busy, so
            # a manual Git Commit after the parent's "Task stopped by
            # user" is refused with "A task is still running in this
            # folder".  A TimeoutError is excluded on purpose: its
            # documented contract is "the task keeps running", and a
            # ``stop_on_timeout`` timeout or a *cancel* already sent its
            # stop (and awaited confirmation) inside the read loop.
            # Best-effort, like the closeTab below.
            # ``taskId`` carries this run's token so the daemon
            # rejects the stop if the tab was already reused by a
            # newer run (see ``_stop_task``'s run_token guard).
            try:
                _send_stop(ws, tab_id, run_token)
            except OSError:
                pass
        # The synthetic tab is this client's alone, and a disconnect no
        # longer tears tabs down (tabs are global state shared by every
        # client), so explicitly ask the daemon to close it on every
        # exit path.  For a still-running task (timeout) this merely
        # flips ``frontend_closed`` and the state is disposed when the
        # task ends; for a finished task it is disposed immediately.
        # Best-effort: the daemon may be gone.
        if ws is not None:
            try:
                _send(ws, {"type": "closeTab", "tabId": tab_id})
            except OSError:
                pass
            try:
                ws.close()
            except Exception:
                pass
