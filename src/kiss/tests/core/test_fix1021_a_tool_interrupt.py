# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tool-interrupt fixes of 2026-10-21, group A (audit-A F1 and F2).

F1 — ``inject_async_exception`` rolls an over-delivered injection back
with ``PyThreadState_SetAsyncExc(tid, None)``.  ``kiss.core.tool_interrupt``
used to pin the function's ``argtypes`` to ``(c_ulong, py_object)``, under
which Python ``None`` is marshalled as a pointer to ``Py_None`` instead of
NULL, so the "clear" SET a pending ``SystemError`` in the target thread.
Other modules may still pin ``argtypes`` on the shared
``ctypes.pythonapi.PyThreadState_SetAsyncExc`` object (two tests in this
repository do, at collection time), so the module now calls through a
private handle of its own.  The first test sets and then clears a pending
exception in a real worker thread through that handle and requires the
worker to finish untouched.

F2 — a per-panel Stop injects ``ToolCallInterrupted`` into the tool thread
once the grace period expires.  When the tool is already unwinding a task
Stop (``KeyboardInterrupt``) or a budget overrun at that moment, the
injection lands at the next check point: the first instruction of
``unregister_tool_call``, called from ``_execute_tool``'s ``finally``.
An exception raised in a ``finally`` is not seen by the ``except
ToolCallInterrupted`` of the same ``try``, so the bare interrupt escaped
into the agent loop and the Stop / budget error was reported as a stray
``ToolCallInterrupted``.

The deterministic reproduction below does not race the watchdog thread:
the tool itself leaves a ``ToolCallInterrupted`` pending on its own
thread and raises ``KeyboardInterrupt`` within ONE Python-level call
(``list(map(operator.call, (set_async, raise_kbi)))``, both C callables),
so no eval-breaker check runs between the two and the pending exception
can only land during the unwinding — exactly the window the watchdog
hits by luck.  The stochastic test drives the real watchdog with a zero
grace period many times over and accepts either legitimate outcome.
"""

from __future__ import annotations

import ctypes
import functools
import operator
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.core import tool_interrupt
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    interrupt_tool_call,
)

# The handle ``inject_async_exception`` rolls back with.  Deliberately NOT
# ``ctypes.pythonapi.PyThreadState_SetAsyncExc``: that shared object has its
# ``argtypes`` set by other tests at collection time, which is the very
# interference the private handle exists to be immune to.
_SET_ASYNC_EXC = tool_interrupt._SET_ASYNC_EXC


def _wait_then_count(gate: threading.Event, out: dict[str, object]) -> None:
    """Block on *gate*, then run through many check points and record the sum."""
    try:
        gate.wait()
        total = 0
        for i in range(1000):
            total += i
        out["total"] = total
    except BaseException as exc:  # noqa: BLE001 — the test inspects it
        out["error"] = exc


def test_pending_async_exception_is_cleared_by_none_after_importing_tool_interrupt() -> None:
    assert callable(tool_interrupt.inject_async_exception)  # the module is imported
    gate = threading.Event()
    out: dict[str, object] = {}
    worker = threading.Thread(target=_wait_then_count, args=(gate, out), daemon=True)
    worker.start()
    time.sleep(0.3)  # the worker is parked inside gate.wait() (GIL released)
    assert worker.ident is not None
    tid = ctypes.c_ulong(worker.ident)
    assert _SET_ASYNC_EXC(tid, ctypes.py_object(ToolCallInterrupted)) == 1
    assert _SET_ASYNC_EXC(tid, None) == 1  # the CPython-documented "clear"
    gate.set()
    worker.join(5)
    assert not worker.is_alive(), "the worker never finished"
    assert "error" not in out, f"the clear left an exception pending: {out['error']!r}"
    assert out["total"] == sum(range(1000))


def _make_agent(*tools: Callable[..., Any]) -> KISSAgent:
    """A KISSAgent with *tools* registered, ready for ``_execute_tool``."""
    agent = KISSAgent("fix1021-a")
    agent._reset(
        model_name="claude-haiku-4-5",
        is_agentic=True,
        max_steps=5,
        max_budget=0.01,
        model_config=None,
        verbose=False,
    )
    agent._add_functions(list(tools))
    return agent


def _leave_interrupt_pending_and_raise(exc_type: type[BaseException]) -> None:
    """Set a pending ``ToolCallInterrupted`` on this thread and raise *exc_type*.

    Both happen inside one ``list(map(...))`` call through C callables
    that run no Python code in between, so the pending exception cannot
    land before *exc_type* unwinds.  The raise is a ``throw`` into an
    exhausted generator, not a ctypes call: ctypes raises the
    ``ctypes.call_function`` audit event before every call, and a Python
    audit hook (``filelock`` adds one at import, for the life of the
    process) would be a check point for the pending exception.
    """
    exc = exc_type()  # built first: a Python ``__init__`` is a check point too
    finished = (None for _ in ())
    list(finished)
    set_async = functools.partial(
        _SET_ASYNC_EXC,
        ctypes.c_ulong(threading.get_ident()),
        ctypes.py_object(ToolCallInterrupted),
    )
    raise_now = functools.partial(finished.throw, exc)
    list(map(operator.call, (set_async, raise_now)))


def stop_while_interrupt_pending() -> str:
    """Tool hit by a task Stop while a panel Stop's injection is pending."""
    _leave_interrupt_pending_and_raise(KeyboardInterrupt)
    return "unreachable"  # pragma: no cover


def overrun_while_interrupt_pending() -> str:
    """Tool hit by a budget overrun while a panel Stop's injection is pending."""
    _leave_interrupt_pending_and_raise(BudgetExceededError)
    return "unreachable"  # pragma: no cover


def _run_tool(agent: KISSAgent, name: str, out: dict[str, object]) -> None:
    """Run tool *name* through ``_execute_tool``; record its result or exception."""
    try:
        out["result"] = agent._execute_tool({"name": name, "arguments": {}})
    except BaseException as exc:  # noqa: BLE001 — the test inspects it
        out["error"] = exc
    # Any exception still pending on this thread lands here, after the
    # agent gave it up, and shows as the worker never recording "done".
    for _ in range(10):
        time.sleep(0.001)
    out["done"] = True


def _execute_on_worker(agent: KISSAgent, name: str) -> dict[str, object]:
    """Run tool *name* on a fresh thread and return what ``_execute_tool`` did."""
    out: dict[str, object] = {}
    worker = threading.Thread(target=_run_tool, args=(agent, name, out), daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "the tool thread wedged"
    assert out.get("done") is True, f"an exception escaped after _execute_tool: {out}"
    out["leaked"] = worker.ident in tool_interrupt._RUNNING
    return out


def test_task_stop_unwinding_wins_over_a_late_panel_stop_injection() -> None:
    agent = _make_agent(stop_while_interrupt_pending)
    out = _execute_on_worker(agent, "stop_while_interrupt_pending")
    assert isinstance(out.get("error"), KeyboardInterrupt), out
    assert out["leaked"] is False, "the call's registry entry leaked"


def test_budget_overrun_unwinding_wins_over_a_late_panel_stop_injection() -> None:
    agent = _make_agent(overrun_while_interrupt_pending)
    out = _execute_on_worker(agent, "overrun_while_interrupt_pending")
    assert isinstance(out.get("error"), BudgetExceededError), out
    assert out["leaked"] is False, "the call's registry entry leaked"


def swallow_interrupt_then_stop() -> str:
    """Tool that eats the injected interrupt (``except BaseException``) and then raises a Stop.

    After this tool the token is ``injected`` but nothing is pending, so
    ``_execute_tool``'s ``finally`` has to drain explicitly and absorb the
    explicit ``ToolCallInterrupted`` the drain raises.
    """
    token = tool_interrupt.current_tool_call()
    assert token is not None
    try:
        # Armed inside the try: with a zero grace the injection may land
        # on the very next check point, which must already be protected.
        interrupt_tool_call(token.thread_ident)
        while not token.injected:
            time.sleep(0.001)
        for _ in range(1000):
            time.sleep(0.001)  # check points for the injection to land on
        raise AssertionError("the injection never landed")  # pragma: no cover
    except ToolCallInterrupted:
        pass
    raise KeyboardInterrupt


def test_finally_drains_a_swallowed_injection_and_keeps_the_stop() -> None:
    saved_grace = tool_interrupt._INJECT_AFTER_SECONDS
    tool_interrupt._INJECT_AFTER_SECONDS = 0.0
    try:
        agent = _make_agent(swallow_interrupt_then_stop)
        out = _execute_on_worker(agent, "swallow_interrupt_then_stop")
    finally:
        tool_interrupt._INJECT_AFTER_SECONDS = saved_grace
    assert isinstance(out.get("error"), KeyboardInterrupt), out


def stop_racing_the_watchdog() -> str:
    """Tool that arms a zero-grace panel Stop on itself and raises a task Stop at once."""
    interrupt_tool_call(threading.get_ident())
    raise KeyboardInterrupt


def test_watchdog_injection_racing_a_task_stop_never_escapes_bare() -> None:
    saved_grace = tool_interrupt._INJECT_AFTER_SECONDS
    tool_interrupt._INJECT_AFTER_SECONDS = 0.0
    outcomes = {"stop": 0, "interrupted": 0}
    try:
        agent = _make_agent(stop_racing_the_watchdog)
        for _ in range(150):
            out = _execute_on_worker(agent, "stop_racing_the_watchdog")
            error = out.get("error")
            if isinstance(error, KeyboardInterrupt):
                outcomes["stop"] += 1
            elif error is None and out.get("result") == (
                "stop_racing_the_watchdog",
                USER_INTERRUPTED_MESSAGE,
            ):
                outcomes["interrupted"] += 1  # the injection landed inside the tool
            else:
                raise AssertionError(f"unexpected outcome: {out}")
    finally:
        tool_interrupt._INJECT_AFTER_SECONDS = saved_grace
    assert sum(outcomes.values()) == 150, outcomes
