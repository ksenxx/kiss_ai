"""Verify the cooperative ``stop_event`` propagates from a parent task
to its parallel sub-agents and from those to their own (nested)
sub-sub-agents.

The VS Code task runner sets ``printer._thread_local.stop_event`` on the
parent's task thread.  ``BaseBrowserPrinter._check_stop`` reads that
thread-local and raises ``KeyboardInterrupt`` when the event is set —
that is the cooperative-stop signal.  Because every parallel sub-agent
runs in its own worker thread, the parent thread's
``threading.local`` is **not** visible to the sub-agent's printer
calls; the sub-agent's per-thread ``stop_event`` slot must be
populated explicitly inside the worker before any printer call (and
before ``SorcarAgent.run`` snapshots it into ``self._stop_event`` for
subprocess kills).

These tests use the real ``ChatSorcarAgent._run_tasks_parallel`` flow
but monkey-patch the sub-agent ``ChatSorcarAgent.run`` to a stub that
just inspects the per-thread ``stop_event`` — no LLM calls, no
network, no mocks of the production stop-propagation code path.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter


class _RecordingPrinter(BaseBrowserPrinter):
    """``BaseBrowserPrinter`` that records every broadcast event.

    Only broadcasts are recorded — no persistence path runs (we do
    not call into ``sorcar.db``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._ev_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._ev_lock:
            self.events.append(event)


def _install_capturing_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, dict[str, Any]],
    nested_tasks: list[str] | None = None,
    set_stop_inside: bool = False,
    call_check_stop: bool = False,
) -> None:
    """Replace ``ChatSorcarAgent.run`` with a stub that captures
    per-thread state instead of running the LLM.

    Each invocation records, keyed by the task prompt string:

      * ``stop_event``: the ``stop_event`` slot visible on the
        worker thread's ``printer._thread_local`` BEFORE any agent
        work happens.
      * ``stop_event_is_set``: same, but ``.is_set()`` checked
        immediately.
      * ``thread_ident``: the worker thread id.

    Optional behaviour:

      * ``nested_tasks`` — if provided, the stub assigns
        ``self.printer`` and recursively calls
        ``self._run_tasks_parallel(nested_tasks)`` so nested
        propagation can be observed.
      * ``set_stop_inside`` — when True the stub sets the captured
        ``stop_event`` (if any) before returning.  Useful for
        asserting that a stop initiated mid-flight reaches every
        descendant.
      * ``call_check_stop`` — when True the stub calls
        ``printer._check_stop()``; if a set stop_event is visible on
        the worker thread, this raises ``KeyboardInterrupt`` (the
        production cooperative-stop signal).
    """

    def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
        printer = kwargs.get("printer")
        assert printer is not None
        self.printer = printer
        tl = printer._thread_local
        task_key = kwargs.get("prompt_template", "") or "<no-task>"
        stop_event = getattr(tl, "stop_event", None)
        captured[task_key] = {
            "stop_event": stop_event,
            "stop_event_is_set": (
                stop_event.is_set() if stop_event is not None else None
            ),
            "thread_ident": threading.get_ident(),
        }
        if call_check_stop:
            printer._check_stop()
        if nested_tasks:
            self._run_tasks_parallel(nested_tasks)
        if set_stop_inside and stop_event is not None:
            stop_event.set()
        return "success: true\nsummary: stub\n"

    monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)


class TestStopEventPropagationToSubagents:
    """Cooperative stop_event must flow from parent → sub-agents → nested."""

    def test_stop_event_visible_in_direct_subagents(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each parallel sub-agent must see the parent's stop_event.

        Reproduces the bug: prior to the fix, the worker thread
        spawned by ``ThreadPoolExecutor`` started with a fresh
        ``threading.local`` whose ``stop_event`` slot was ``None``
        — so any ``_check_stop()`` from inside the sub-agent was a
        silent no-op and the sub-agent kept running after the user
        clicked Stop.
        """
        captured: dict[str, dict[str, Any]] = {}
        _install_capturing_run(monkeypatch, captured=captured)

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "parent-tab"
        parent_stop = threading.Event()
        printer._thread_local.stop_event = parent_stop

        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        parent._run_tasks_parallel(["task-a", "task-b"])

        sub_keys = sorted(captured.keys())
        assert sub_keys == ["task-a", "task-b"], sub_keys
        for sub_key in sub_keys:
            ev = captured[sub_key]["stop_event"]
            assert ev is parent_stop, (
                f"Sub-agent {sub_key} did not see the parent's stop_event "
                f"(saw {ev!r}); fix must copy printer._thread_local.stop_event "
                "into the worker thread before agent.run() is called."
            )
            assert captured[sub_key]["stop_event_is_set"] is False

    def test_set_stop_event_aborts_subagent_via_check_stop(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A set parent stop_event makes the sub-agent's
        ``printer._check_stop()`` raise ``KeyboardInterrupt``.

        ``KeyboardInterrupt`` is a ``BaseException`` (not a
        ``Exception``), so ``_run_single``'s ``except Exception``
        guard does not catch it; the interrupt bubbles up through
        ``ThreadPoolExecutor.map`` into the parent task thread.  In
        production this is the path ``_TaskRunnerMixin._run_task``'s
        ``except KeyboardInterrupt`` handler relies on to surface
        ``task_stopped`` to the UI.
        """
        captured: dict[str, dict[str, Any]] = {}
        _install_capturing_run(
            monkeypatch, captured=captured, call_check_stop=True,
        )

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "parent-stop-set"
        parent_stop = threading.Event()
        parent_stop.set()
        printer._thread_local.stop_event = parent_stop

        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        with pytest.raises(KeyboardInterrupt):
            parent._run_tasks_parallel(["t1"])

        # The sub-agent saw the set stop_event before raising.
        sub = captured["t1"]
        assert sub["stop_event"] is parent_stop
        assert sub["stop_event_is_set"] is True

    def test_stop_event_visible_in_nested_subagents(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stop_event must propagate through nested ``_run_tasks_parallel``.

        Parent → sub_0 → nested_0: every level's worker thread must
        observe the *same* ``stop_event`` instance on its own
        ``printer._thread_local``.
        """
        captured: dict[str, dict[str, Any]] = {}
        _install_capturing_run(
            monkeypatch, captured=captured, nested_tasks=["nested"],
        )

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "root"
        parent_stop = threading.Event()
        printer._thread_local.stop_event = parent_stop

        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        parent._run_tasks_parallel(["t-outer"])

        sub_key = "t-outer"
        nested_key = "nested"
        assert sub_key in captured, sorted(captured.keys())
        assert nested_key in captured, sorted(captured.keys())

        assert captured[sub_key]["stop_event"] is parent_stop, (
            "Direct sub-agent did not inherit parent stop_event."
        )
        assert captured[nested_key]["stop_event"] is parent_stop, (
            "Nested sub-sub-agent did not inherit parent stop_event; "
            "stop propagation must be transitive across thread-pool "
            "boundaries at every depth."
        )
        # Different worker threads at each level — propagation is
        # cross-thread, not just same-thread reuse.
        assert (
            captured[sub_key]["thread_ident"]
            != captured[nested_key]["thread_ident"]
        )
