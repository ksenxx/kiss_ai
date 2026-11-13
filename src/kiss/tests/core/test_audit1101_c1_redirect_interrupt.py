# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop on a redirecting call's panel covers the calls it redirected to.

Audit 1101, C1 F4.  A call that raises
:class:`~kiss.core.tool_verdict.ToolRedirect` stays registered as the
thread's running call while the agent makes the redirected call and
its follow-up on the same thread.  Each nested call used to REPLACE the
thread's registry entry and unregistering it dropped the entry
altogether, so

* a Stop accepted on the outer panel before the redirect was lost: the
  nested calls watched their own events and the outer watchdog found its
  token gone;
* a Stop clicked on the outer panel while a nested call ran was rejected
  (its id was no longer registered).

Now a nested call is registered under its enclosing call and restores
it when it ends, a Stop on the enclosing call also signals the running
nested call, and the agent checks the enclosing call's interrupt before
and after each nested call.  Round 2: a nested call that starts
interrupted (the Stop landed between the agent's check and its
registration) is not run at all, whether or not its tool polls the
interrupt event, and a Stop landing while the nested call registers
itself is seen by one side or the other (the interrupting side stores
its mark before reading the registry; the registering side stores its
registry entry before reading the mark).  Round 3: the registering
side reads the marks of EVERY call it is nested in, not its parent's
alone, so a call nested two deep (``outer`` redirects to ``middle``,
which redirects to ``leaf``) inherits a Stop on ``outer`` that the
click's snapshot of the chain missed and has not yet put on ``middle``.
Driven through the real ``KISSAgent`` dispatch with a real printer (the
calls are issued by hand, as the tool loop does); every wait is bounded.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from functools import partial
from typing import Any

import pytest

from kiss.core.base import collecting_trajectories
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    _signal,
    current_tool_call,
    interrupt_tool_call,
    register_tool_call,
    wait_interruptible,
)
from kiss.core.tool_verdict import ALLOW, ToolRedirect, Verdict, followup_result


class _Recorder(Printer):
    """A real ``Printer`` keeping the call ids and the results' ``interrupted`` flags.

    Args:
        stop_outer_on_open: Accept a Stop on the outer call's panel the
            moment its ``tool_call`` event opens it (the click that lands
            before the redirect).
    """

    def __init__(self, stop_outer_on_open: bool = False) -> None:
        self.stop_outer_on_open = stop_outer_on_open
        self.call_ids: dict[str, int] = {}
        self.accepted: list[bool] = []
        self.results: list[tuple[str, bool]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Record the event; click the outer panel's Stop when asked to.

        Args:
            content: The tool's name for a ``tool_call``, its result for a ``tool_result``.
            type: The event type.
            **kwargs: ``call_id`` of a ``tool_call``; ``tool_name`` and ``interrupted``
                of a ``tool_result``.
        """
        if type == "tool_call":
            self.call_ids[str(content)] = kwargs["call_id"]
            if self.stop_outer_on_open and content == "outer":
                self.accepted.append(
                    interrupt_tool_call(threading.get_ident(), "outer", kwargs["call_id"])
                )
        elif type == "tool_result":
            self.results.append((kwargs["tool_name"], bool(kwargs.get("interrupted", False))))

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (unused).
        """


class _Chain:
    """``outer`` redirects to ``write`` with ``followup`` as the rest; one of them may click Stop.

    Args:
        printer: The recorder holding the panels' call ids.
        click_from: The nested tool that clicks Stops while it runs (``""`` for none).
        clicks: The panels it clicks, in order: ``"outer"`` or its own.
    """

    def __init__(
        self, printer: _Recorder, click_from: str = "", clicks: tuple[str, ...] = ()
    ) -> None:
        self.printer = printer
        self.click_from = click_from
        self.clicks = clicks
        self.effects: list[str] = []
        self.accepted: list[bool] = []

    def tools(self) -> dict[str, Any]:
        """The chain's tools by name, for the agent's ``function_map``."""
        return {"outer": self.outer, "write": self.write, "followup": self.followup}

    def _nested(self, name: str) -> str:
        """Run the nested tool *name*: click if asked, then wait a bounded, interruptible while."""
        if self.click_from == name:
            for panel in self.clicks:
                self.accepted.append(
                    interrupt_tool_call(threading.get_ident(), panel, self.printer.call_ids[panel])
                )
            # Interrupted at the first slice; the bound only keeps a failure finite.
            wait_interruptible(threading.Event(), 5.0)
        self.effects.append(name)
        return f"{name} done"

    def outer(self) -> str:
        """Stand for a shell rewrite with a rest after it."""
        raise ToolRedirect("write", {}, "Redirected:", ("followup", {}))

    def write(self) -> str:
        """The rewrite the redirect stands for."""
        return self._nested("write")

    def followup(self) -> str:
        """The rest of the compound command."""
        return self._nested("followup")


class _DeepChain:
    """``outer`` redirects to ``middle``, which redirects to ``leaf``, an ordinary tool."""

    def __init__(self) -> None:
        self.effects: list[str] = []

    def tools(self) -> dict[str, Any]:
        """The chain's tools by name, for the agent's ``function_map``."""
        return {"outer": self.outer, "middle": self.middle, "leaf": self.leaf}

    def outer(self) -> str:
        """Redirect to ``middle``."""
        raise ToolRedirect("middle", {}, "Redirected:")

    def middle(self) -> str:
        """Redirect once more, to ``leaf``."""
        raise ToolRedirect("leaf", {}, "Redirected again:")

    def leaf(self) -> str:
        """Commit the effect the moment the tool runs; never look at the interrupt event."""
        self.effects.append("leaf")
        return "leaf done"


def _dispatch(printer: _Recorder, chain: _Chain | _DeepChain, hook: Any = None) -> str:
    """Make the ``outer`` call through the agent's dispatch; return its result."""
    with collecting_trajectories(save=False):
        agent = KISSAgent("redirect interrupt")
        agent.printer = printer
        agent.tool_call_hook = hook
        agent.function_map = chain.tools()
        _, result, blocked = agent._guarded_execute_tool({"name": "outer", "arguments": {}})
    assert blocked is None
    assert current_tool_call() is None, "the registry keeps no entry once the chain is over"
    return str(result)


def test_a_stop_accepted_on_the_outer_panel_before_the_redirect_ends_the_chain() -> None:
    printer = _Recorder(stop_outer_on_open=True)
    chain = _Chain(printer)
    result = _dispatch(printer, chain)
    assert printer.accepted == [True]
    assert result == USER_INTERRUPTED_MESSAGE
    assert chain.effects == [], "neither the redirected write nor the follow-up ran"
    assert printer.results == [("outer", True)]
    assert "write" not in printer.call_ids, "the redirected call was never opened"


@pytest.mark.parametrize("nested", ["write", "followup"])
def test_a_stop_on_the_outer_panel_while_a_nested_call_runs_is_accepted_and_ends_it(
    nested: str,
) -> None:
    printer = _Recorder()
    chain = _Chain(printer, click_from=nested, clicks=("outer",))
    result = _dispatch(printer, chain)
    assert chain.accepted == [True], "the outer call is still running: its panel's Stop applies"
    assert result == USER_INTERRUPTED_MESSAGE
    assert chain.effects == ([] if nested == "write" else ["write"])
    expected = [("write", True)] if nested == "write" else [("write", False), ("followup", True)]
    assert printer.results == [*expected, ("outer", True)]


def test_a_stop_on_the_nested_panel_interrupts_that_call_only() -> None:
    printer = _Recorder()
    chain = _Chain(printer, click_from="write", clicks=("write",))
    result = _dispatch(printer, chain)
    assert chain.accepted == [True]
    assert chain.effects == []
    assert printer.results == [("write", True), ("outer", False)]
    assert result == "Redirected:\n" + USER_INTERRUPTED_MESSAGE + "\n" + followup_result(
        "followup", {}, None, "write"
    )


@pytest.mark.parametrize("nested", ["write", "followup"])
def test_a_stop_on_the_outer_panel_landing_before_a_nested_call_starts_stops_it_at_once(
    nested: str,
) -> None:
    """The click lands while the nested call's hook runs: after the agent's check and
    before the call is registered.  The call starts interrupted, so its blocking wait
    returns at once instead of running to its bound, and the outer ends the chain."""
    printer = _Recorder()
    chain = _Chain(printer, click_from=nested, clicks=())
    accepted: list[bool] = []

    def hook(name: str, args: dict[str, Any]) -> Verdict:
        if name == nested:
            accepted.append(
                interrupt_tool_call(threading.get_ident(), "outer", printer.call_ids["outer"])
            )
        return ALLOW

    started = time.monotonic()
    result = _dispatch(printer, chain, hook)
    assert time.monotonic() - started < 2.0, "the nested call's wait must not run to its bound"
    assert accepted == [True]
    assert result == USER_INTERRUPTED_MESSAGE
    assert chain.effects == ([] if nested == "write" else ["write"])
    expected = [("write", True)] if nested == "write" else [("write", False), ("followup", True)]
    assert printer.results == [*expected, ("outer", True)]


def test_a_stop_on_the_outer_panel_after_one_on_the_nested_panel_ends_the_chain() -> None:
    """The nested call is already signalled; the outer click still takes and ends the outer."""
    printer = _Recorder()
    chain = _Chain(printer, click_from="write", clicks=("write", "outer"))
    result = _dispatch(printer, chain)
    assert chain.accepted == [True, True]
    assert chain.effects == []
    assert printer.results == [("write", True), ("outer", True)]
    assert result == USER_INTERRUPTED_MESSAGE


@pytest.mark.parametrize("nested", ["write", "followup"])
def test_a_nested_call_that_starts_interrupted_does_not_run_even_without_polling(
    nested: str,
) -> None:
    """The click lands while the nested call's hook runs, as above, but the nested tool
    is an ordinary function that never looks at the interrupt event: it commits its
    effect the moment it runs.  The agent must not run it at all."""
    printer = _Recorder()
    chain = _Chain(printer)
    accepted: list[bool] = []

    def hook(name: str, args: dict[str, Any]) -> Verdict:
        if name == nested:
            accepted.append(
                interrupt_tool_call(threading.get_ident(), "outer", printer.call_ids["outer"])
            )
        return ALLOW

    result = _dispatch(printer, chain, hook)
    assert accepted == [True]
    assert result == USER_INTERRUPTED_MESSAGE
    assert chain.effects == ([] if nested == "write" else ["write"]), (
        "the nested call ran after the Stop was accepted"
    )
    expected = [("write", True)] if nested == "write" else [("write", False), ("followup", True)]
    assert printer.results == [*expected, ("outer", True)]


def _line_of(function: Any, statement: str) -> int:
    """Return the line number of the first line of *function*'s source containing *statement*."""
    lines, start = inspect.getsourcelines(function)
    return start + next(i for i, line in enumerate(lines) if statement in line)


def _run_traced(function: Any, trace: Any, outcome: list[Any]) -> None:
    """Call *function* on this thread under *trace* (``None`` for none); append its
    result or exception."""
    sys.settrace(trace)
    try:
        outcome.append(function())
    except BaseException as exc:  # noqa: BLE001 — reported by the test thread
        outcome.append(exc)
    finally:
        sys.settrace(None)


@pytest.mark.parametrize(
    "statement",
    ["_RUNNING[token.thread_ident] = token", "enclosing.interrupted"],
    ids=["before-the-registry-store", "before-the-parent-interrupt-check"],
)
def test_a_stop_landing_while_a_nested_call_registers_is_not_lost(statement: str) -> None:
    """The click lands inside ``register_tool_call`` of the nested call: right before
    the nested call takes over the registry, or right after it did and before it reads
    its parent's interrupt.  A trace barrier on the agent's thread holds the nested
    call's registration at that statement while the test thread clicks the outer
    panel's Stop; nothing in the code under test is replaced.  Either the click sees
    the nested call in the registry or the nested call sees the click's mark: the
    nested tool (an ordinary function that never polls) must not run."""
    printer = _Recorder()
    chain = _Chain(printer)
    at_barrier, resume = threading.Event(), threading.Event()
    barrier_line = _line_of(register_tool_call, statement)

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if (
            event == "line"
            and frame.f_code is register_tool_call.__code__
            and frame.f_lineno == barrier_line
            and frame.f_locals["token"].name == "write"
        ):
            at_barrier.set()
            resume.wait(5)
        return trace

    outcome: list[Any] = []
    worker = threading.Thread(
        target=_run_traced, args=(partial(_dispatch, printer, chain), trace, outcome), daemon=True
    )
    worker.start()
    try:
        assert at_barrier.wait(5), "the nested call never reached its registration"
        assert worker.ident is not None
        accepted = interrupt_tool_call(worker.ident, "outer", printer.call_ids["outer"])
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive(), "the chain did not end"
    assert accepted is True, "the outer call was running: its panel's Stop applies"
    assert outcome and not isinstance(outcome[0], BaseException), outcome
    assert outcome[0] == USER_INTERRUPTED_MESSAGE
    assert chain.effects == [], "the nested call ran after the Stop was accepted"
    assert printer.results == [("write", True), ("outer", True)]


def test_a_stop_landing_while_a_call_nested_two_deep_registers_is_not_lost() -> None:
    """``outer`` redirects to ``middle``, which redirects to ``leaf``.  The click on
    the outer panel marks ``outer``, snapshots the calls nested in it (``middle``,
    the registered call while ``leaf`` is held in its hook) and then marks ``middle``;
    a trace barrier on the clicking thread holds it right before that second mark
    while ``leaf`` is released and registers under ``middle``.  ``leaf`` is in neither
    the snapshot nor its parent's mark: it must inherit the Stop from ``outer``, the
    call it is nested in that carries it, and not run (its tool is an ordinary
    function that never polls).  The click resumes only once the chain has ended, so
    ``middle`` is never marked while it runs: the leaf alone can keep it from
    running on."""
    printer = _Recorder()
    chain = _DeepChain()
    leaf_in_hook, resume_hook = threading.Event(), threading.Event()
    at_barrier, resume_click = threading.Event(), threading.Event()
    barrier_line = _line_of(_signal, "call.interrupted = True")

    def hook(name: str, args: dict[str, Any]) -> Verdict:
        if name == "leaf":
            leaf_in_hook.set()
            resume_hook.wait(5)
        return ALLOW

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if (
            event == "line"
            and frame.f_code is _signal.__code__
            and frame.f_lineno == barrier_line
            and frame.f_locals["call"].name == "middle"
        ):
            at_barrier.set()
            resume_click.wait(5)
        return trace

    outcome: list[Any] = []
    accepted: list[Any] = []
    worker = threading.Thread(
        target=_run_traced,
        args=(partial(_dispatch, printer, chain, hook), None, outcome),
        daemon=True,
    )
    worker.start()
    clicker: threading.Thread | None = None
    try:
        assert leaf_in_hook.wait(5), "the leaf call never reached its hook"
        assert worker.ident is not None
        click = partial(interrupt_tool_call, worker.ident, "outer", printer.call_ids["outer"])
        clicker = threading.Thread(target=_run_traced, args=(click, trace, accepted), daemon=True)
        clicker.start()
        assert at_barrier.wait(5), "the click never reached the middle call's mark"
        resume_hook.set()
        worker.join(10)
    finally:
        resume_hook.set()
        resume_click.set()
        if clicker is not None:
            clicker.join(10)
        worker.join(10)
    assert not worker.is_alive(), "the chain did not end"
    assert clicker is not None and not clicker.is_alive(), "the click did not return"
    assert accepted == [True], "the outer call was running: its panel's Stop applies"
    assert outcome and not isinstance(outcome[0], BaseException), outcome
    assert outcome[0] == USER_INTERRUPTED_MESSAGE
    assert chain.effects == [], "the leaf ran after the Stop on the outer call was accepted"
    assert printer.results == [("leaf", True), ("middle", False), ("outer", True)]
