# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Interrupt ONE running tool call without stopping its task.

The per-panel Stop button in the chat UI stops the tool call the user is
looking at; the task itself carries on and the model sees the string
:data:`USER_INTERRUPTED_MESSAGE` as that tool's result.

A tool runs synchronously on its agent's thread
(``KISSAgent._execute_tool``), so the interrupt has to reach that thread
while it is inside arbitrary tool code.  It is delivered in two stages:

1. **Cooperative.**  :func:`interrupt_tool_call` marks the call
   interrupted and sets its :class:`threading.Event`.  Tools that own a
   blocking wait watch that event (the Bash tool's output loop, the
   ``ask_user_question`` answer wait, the ``run_parallel`` fan-out wait,
   the ``run_agent`` daemon-client loop) and call
   :func:`raise_if_interrupted` at a point of their choosing, which
   raises :class:`ToolCallInterrupted` with no lock held and nothing
   half-done.
2. **Forced.**  A tool that has not returned :data:`_INJECT_AFTER_SECONDS`
   later is not watching the event, so :class:`ToolCallInterrupted` is
   raised asynchronously in its thread (``PyThreadState_SetAsyncExc``).
   CPython delivers it at the next bytecode boundary, which is immediate
   for tools running Python code but never inside a blocking C call.

Asynchronous delivery has one hazard that shapes this module: the
exception may land right after a ``lock.__enter__`` returned and before
the ``with`` body's exception range begins, so ``__exit__`` never runs
and the lock stays held.  Therefore the tool's own thread NEVER takes a
lock here — its registry operations are single dict stores and pops,
atomic under the GIL — and the interrupting side only ever holds
:data:`_INJECT_LOCK`, which the tool thread never touches.  The
``closing`` flag on the token orders the two sides: a tool that has
begun returning is never injected, and an injection that raced the
return is drained by :func:`end_tool_call` so it cannot surface later
in the agent loop.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import threading
import time

logger = logging.getLogger(__name__)

USER_INTERRUPTED_MESSAGE = "User interrupted the tool call."
"""The tool result the model sees for an interrupted tool call."""

_INJECT_AFTER_SECONDS = 1.0
"""Grace given to a tool to honor the cooperative signal before injection."""

_DRAIN_SECONDS = 1.0
"""How long :func:`drain_pending_interrupt` waits for an injected exception.

Delivery normally takes microseconds: the wait loop's backward jump is
an eval-breaker check.
"""


class ToolCallInterrupted(BaseException):
    """Raised inside a tool call the user interrupted.

    A ``BaseException`` (like ``KeyboardInterrupt``) so tool code that
    catches ``Exception`` cannot swallow it; ``KISSAgent._execute_tool``
    catches it and turns it into :data:`USER_INTERRUPTED_MESSAGE`.
    """


ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


def inject_async_exception(tid: int, exc_type: type[BaseException]) -> int:
    """Raise *exc_type* asynchronously in thread *tid*.

    Wraps ``PyThreadState_SetAsyncExc``.  When the call reports the
    exception was set in more than one thread state (``rc > 1``), the
    injection is undone as CPython's documentation requires.

    Args:
        tid: The target thread's ``ident``.
        exc_type: The exception class to raise in that thread.

    Returns:
        The number of thread states modified: ``0`` when *tid* no
        longer names a live thread, ``1`` on success (values above 1
        have already been rolled back here).
    """
    rc = int(
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(exc_type),
        )
    )
    if rc > 1:  # pragma: no cover — rare: exception set in multiple states
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
    return rc


_call_ids = itertools.count(1)


class ToolCallToken:
    """One tool call in progress on one thread.

    Attributes:
        name: The tool's name, as the model called it.
        call_id: Process-wide unique id of this call; the ``tool_call``
            event carries it so a Stop click names exactly this call.
        thread_ident: The ``ident`` of the thread running the tool.
        thread: That thread; injection is refused once it is no longer
            alive, because a dead thread's ident is reused by the next
            thread the process starts.
        event: Set when the call is interrupted; blocking tools poll it.
        interrupted: ``True`` once a Stop was accepted for this call.
        injected: ``True`` once :class:`ToolCallInterrupted` has been
            injected into ``thread_ident`` for this call.
        injecting: ``True`` while the interrupting side is between its
            ``closing`` check and its ``injected`` verdict; the tool
            thread waits it out (see :func:`unregister_tool_call`).
        closing: Set by the tool's thread the moment the tool returned
            (or raised an ordinary error); an interrupt arriving after
            that never injects.
    """

    __slots__ = (
        "call_id",
        "closing",
        "event",
        "injected",
        "injecting",
        "interrupted",
        "name",
        "thread",
        "thread_ident",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_id = next(_call_ids)
        self.thread = threading.current_thread()
        self.thread_ident = threading.get_ident()
        self.event = threading.Event()
        self.interrupted = False
        self.injected = False
        self.injecting = False
        self.closing = False


# Serializes the interrupting side only; the tool's own thread never
# acquires it (see the module docstring).
_INJECT_LOCK = threading.Lock()
# thread ident -> the tool call that thread is running right now.  Only
# the owning thread writes its own key.
_RUNNING: dict[int, ToolCallToken] = {}


def new_tool_call(name: str) -> ToolCallToken:
    """Create the token for a tool call about to start on this thread.

    Kept separate from :func:`register_tool_call` so the caller can hold
    the token BEFORE the call becomes interruptible: an exception
    injected after registration then always has a token to unregister.

    Args:
        name: The tool's name.

    Returns:
        The unregistered token.
    """
    return ToolCallToken(name)


def register_tool_call(token: ToolCallToken) -> None:
    """Make *token*'s call interruptible: publish it for its thread.

    A single dict store, so no lock is held on the tool thread.

    Args:
        token: The token :func:`new_tool_call` returned on this thread.
    """
    _RUNNING[token.thread_ident] = token


def begin_tool_call(name: str) -> ToolCallToken:
    """Create AND register a tool call on the calling thread.

    Convenience for callers whose registration point need not be inside
    a ``try`` (tests, tools that wrap a nested call).

    Args:
        name: The tool's name.

    Returns:
        The registered token.
    """
    token = new_tool_call(name)
    register_tool_call(token)
    return token


def drain_pending_interrupt() -> None:
    """Give an already-injected :class:`ToolCallInterrupted` a place to land.

    Called on the tool's own thread when an injection is known to have
    happened (``token.injected``) but the exception has not been raised
    yet: the tool returned in the window between the injection and
    :func:`end_tool_call`, or a blocking wait was woken by the event.
    Left alone, the exception would surface at some later bytecode
    boundary — inside the agent loop, the next model call, or an
    ``except`` handler an explicit raise entered — and be misattributed.
    A busy loop's backward jump is an eval-breaker check, so the
    exception lands here within microseconds; if it does not (a tool
    that swallowed it with ``except BaseException``), the same exception
    is raised explicitly, so the user's Stop is honored either way.  No
    lock is held while waiting.

    Raises:
        ToolCallInterrupted: Always — asynchronously or explicitly.
    """
    deadline = time.monotonic() + _DRAIN_SECONDS
    while time.monotonic() < deadline:
        pass
    raise ToolCallInterrupted(USER_INTERRUPTED_MESSAGE)


def unregister_tool_call(token: ToolCallToken) -> None:
    """Forget *token* without raising; safe to call more than once.

    Marks the call closing first, so an interrupt that arrives from now
    on never injects; then waits out an injection that is in progress
    (the interrupting side read ``closing`` before it was set), so
    ``token.injected`` is final when this returns; then drops the
    registry entry.  Plain stores, reads and a spin — no lock is taken
    on the tool thread.  Used from the ``except ToolCallInterrupted``
    handler and as the ``finally`` safety net on every other exit path
    of the tool call.

    Args:
        token: The token of the call on this thread.
    """
    token.closing = True
    deadline = time.monotonic() + _DRAIN_SECONDS
    while token.injecting and time.monotonic() < deadline:
        pass
    if _RUNNING.get(token.thread_ident) is token:
        _RUNNING.pop(token.thread_ident, None)


def end_tool_call(token: ToolCallToken) -> None:
    """Unregister *token* after its tool returned (or raised an ordinary error).

    Must be called on the tool's own thread, with no exception in
    flight.  When an interrupt was already injected for this call, the
    injected exception is drained here (see :func:`drain_pending_interrupt`)
    so it is raised inside the caller's ``except ToolCallInterrupted``
    rather than somewhere later.  A Stop that was accepted but only
    signalled cooperatively (the tool returned on its own before the
    grace period) leaves the tool's real result alone.

    Args:
        token: The token of the call on this thread.

    Raises:
        ToolCallInterrupted: When an injection raced the tool's return.
    """
    unregister_tool_call(token)
    if token.injected:
        drain_pending_interrupt()


def current_tool_call() -> ToolCallToken | None:
    """Return the tool call the calling thread is running, if any.

    Returns:
        The current thread's registered token, or ``None`` when no tool
        call is in progress on it.
    """
    return _RUNNING.get(threading.get_ident())


def current_tool_interrupt_event() -> threading.Event | None:
    """Return the interrupt event of the calling thread's running tool call.

    Blocking tools (a subprocess wait, an answer-queue wait) watch this
    event alongside the task's stop event so an interrupt unblocks
    them; it is set before any injection.

    Returns:
        The event, or ``None`` when this thread is not inside a tool
        call.
    """
    token = current_tool_call()
    return token.event if token is not None else None


def raise_if_interrupted() -> None:
    """Cooperative check for tools: raise when the current call was interrupted.

    Tools that watch :func:`current_tool_interrupt_event` call this once
    they are back at a safe point (child process killed, wait released).
    If the grace period already passed and the exception was injected,
    it is drained first so exactly one :class:`ToolCallInterrupted`
    surfaces.

    Raises:
        ToolCallInterrupted: When the calling thread's tool call was
            interrupted.
    """
    token = current_tool_call()
    if token is None or not token.interrupted:
        return
    # Closing first: the raise below unwinds through the agent's
    # handler, and the watchdog must not inject into that.
    unregister_tool_call(token)
    if token.injected:
        drain_pending_interrupt()
    raise ToolCallInterrupted(USER_INTERRUPTED_MESSAGE)


def running_tool_name(thread_ident: int) -> str | None:
    """Return the name of the tool thread *thread_ident* is running.

    Args:
        thread_ident: A thread's ``ident``.

    Returns:
        The tool name, or ``None`` when that thread is not inside a
        tool call.
    """
    token = _RUNNING.get(thread_ident)
    return token.name if token is not None else None


def _inject_unless_closing(token: ToolCallToken) -> bool:
    """Inject :class:`ToolCallInterrupted` into *token*'s thread if it still runs the call.

    Args:
        token: The interrupted call.

    Returns:
        ``True`` when the exception was set in the thread.
    """
    with _INJECT_LOCK:
        if token.injected:
            return True
        # ``injecting`` brackets the closing check and the verdict: a
        # tool thread that sets ``closing`` after the check waits for
        # the verdict, so it cannot return with an injection pending.
        token.injecting = True
        try:
            if token.closing or _RUNNING.get(token.thread_ident) is not token:
                return False
            if not token.thread.is_alive():
                # The ident may already name a NEW thread; never inject
                # into that.  A thread cannot die inside a tool call
                # without unregistering, so this is a stale entry.
                logger.info(
                    "Tool call %s (#%d) could not be interrupted: thread %s is gone",
                    token.name,
                    token.call_id,
                    token.thread_ident,
                )
                return False
            rc = inject_async_exception(token.thread_ident, ToolCallInterrupted)
            if rc == 0:  # pragma: no cover — alive thread always has a state
                return False
            token.injected = True
        finally:
            token.injecting = False
    logger.info(
        "Injected ToolCallInterrupted into tool call %s (#%d) on thread %s",
        token.name,
        token.call_id,
        token.thread_ident,
    )
    return True


def _inject_after_grace(token: ToolCallToken) -> None:
    """Watchdog body: force the interrupt once the cooperative grace expires.

    Polls the token's ``closing`` flag instead of waiting on an Event so
    the tool thread has nothing to signal (and no lock to leak) when it
    returns in time.

    Args:
        token: The interrupted call.
    """
    deadline = time.monotonic() + _INJECT_AFTER_SECONDS
    while time.monotonic() < deadline:
        if token.closing or _RUNNING.get(token.thread_ident) is not token:
            return
        time.sleep(0.02)
    _inject_unless_closing(token)


def interrupt_tool_call(
    thread_ident: int,
    tool_name: str = "",
    call_id: int | None = None,
) -> bool:
    """Interrupt the tool call thread *thread_ident* is running.

    Marks the call interrupted and sets its event at once (the
    cooperative signal), then arms a watchdog thread that injects
    :class:`ToolCallInterrupted` if the tool is still running after
    :data:`_INJECT_AFTER_SECONDS`.

    Args:
        thread_ident: The ``ident`` of the task thread.
        tool_name: When non-empty, the interrupt only applies if the
            thread is running a tool of that name.
        call_id: When given, the interrupt only applies to the call with
            that id (the one the clicked panel shows): a click that
            arrives after that call returned must not hit the next
            call, even one of the same name.

    Returns:
        ``True`` when a matching tool call was (or already is)
        interrupted; ``False`` when the thread is not inside a tool
        call, runs a different tool, or has already begun returning.
    """
    with _INJECT_LOCK:
        token = _RUNNING.get(thread_ident)
        if token is None or token.closing:
            return False
        if tool_name and token.name != tool_name:
            return False
        if call_id is not None and token.call_id != call_id:
            return False
        if token.interrupted:
            # Already signalled; a second click must not arm a second
            # watchdog.
            return True
        token.interrupted = True
        token.event.set()
    logger.info(
        "Interrupting tool call %s (#%d) on thread %s",
        token.name,
        token.call_id,
        thread_ident,
    )
    try:
        threading.Thread(
            target=_inject_after_grace,
            args=(token,),
            name=f"tool-interrupt-{token.call_id}",
            daemon=True,
        ).start()
    except RuntimeError:  # pragma: no cover — thread exhaustion
        # No forced fallback here: the caller may hold a lock the tool
        # thread is waiting for, and an injection landing right after
        # that acquire would leak it.  The cooperative signal stands.
        logger.warning(
            "Tool interrupt watchdog for %s could not start; the call is "
            "signalled cooperatively only",
            token.name,
            exc_info=True,
        )
    return True
