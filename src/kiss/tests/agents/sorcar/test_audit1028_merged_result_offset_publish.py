"""The merged ``result`` event is never over-stated by a concurrent offset publish.

``RelentlessAgent._emit_merged_result_event`` subtracts the printer's
usage offsets from the agent's cumulative snapshot and the printer adds
its CURRENT offsets back inside ``print``.  A finished child's fold
(``sorcar_agent._attribute_sub_usage``) publishes new, larger offsets
from another thread; when that publish landed between the emit's read
and the printer's add, the result carried the old offsets subtracted
and the new ones added, over-stating the task's spend.  The emit now
reads and prints under ``_OFFSET_PUBLISH_LOCK``, the lock every
publisher holds, so the printed total can never exceed the ledger.

The ``result`` of a first session that fails terminally had the same
flaw by another route (audit 2026-11-11, S9): ``perform_task`` printed
it with the session's own totals after banking the session, so a fold
publishing in between made the printer add the session twice.  It goes
through the same emit now.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from types import FrameType
from typing import Any

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage
from kiss.core.printer import Printer
from kiss.core.utils import finish
from kiss.tests.agents.sorcar.test_multi_session_final_result_event import (
    OffsetAwarePrinter,
    _start_openai_server,
)

_WAIT = 30.0
_UNIT = (1.0, 100, 3)


class _OffsetAddingPrinter(Printer):
    """Printer that, like the server printer, adds its offsets inside ``print``.

    Each call records the total it would broadcast together with the
    agent's cumulative snapshot taken right after the add: the ledger
    only grows, so a correct total is never above that snapshot.
    """

    def __init__(self, agent: RelentlessAgent) -> None:
        self.agent = agent
        self.tokens_offset = 0
        self.budget_offset = 0.0
        self.steps_offset = 0
        self.records: list[tuple[tuple[float, int, int], tuple[float, int, int]]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Add the current offsets to the raw totals and record the outcome."""
        total = (
            float(kwargs["cost_usd"]) + self.budget_offset,
            int(kwargs["total_tokens"]) + self.tokens_offset,
            int(kwargs["step_count"]) + self.steps_offset,
        )
        self.records.append((total, self.agent.usage_snapshot()))

    def token_callback(self, token: str) -> None:
        """Unused by the merged-result emit path."""

    def reset(self) -> None:
        """Unused by the merged-result emit path."""


def test_merged_result_total_never_exceeds_ledger_under_concurrent_folds() -> None:
    """Concurrent child folds cannot over-state the merged result event."""
    agent = RelentlessAgent("merged-result-offsets")
    printer = _OffsetAddingPrinter(agent)
    agent.printer = printer  # type: ignore[assignment]
    stop = threading.Event()
    # The race window is a few microseconds wide; a short switch
    # interval makes the interpreter hand the GIL over inside it.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def fold_storm() -> None:
        while not stop.is_set():
            # Several banks, then ONE publish: the offsets lag the
            # ledger by a few units, so a publish landing inside the
            # emit shows up as a jump of several units.
            for _ in range(4):
                agent._attribute_usage(*_UNIT)
            _attribute_sub_usage(agent, *_UNIT)

    folders = [threading.Thread(target=fold_storm) for _ in range(3)]
    for folder in folders:
        folder.start()
    try:
        for _ in range(2000):
            agent._emit_merged_result_event(
                {"success": True, "is_continue": False, "summary": "done"}
            )
    finally:
        stop.set()
        for folder in folders:
            folder.join(timeout=_WAIT)
            assert not folder.is_alive()
        sys.setswitchinterval(old_interval)

    assert len(printer.records) == 2000
    for total, ceiling in printer.records:
        assert total[0] <= ceiling[0] + 1e-9, (total, ceiling)
        assert total[1] <= ceiling[1], (total, ceiling)
        assert total[2] <= ceiling[2], (total, ceiling)
    # The final offsets are the last published snapshot, untouched by
    # the emits themselves.
    assert (printer.budget_offset, printer.tokens_offset, printer.steps_offset) <= (
        agent.usage_snapshot()
    )


class _FoldAtErrorResult:
    """Profile hook: a finished child's fold lands as ``perform_task`` builds its error result.

    ``perform_task`` calls ``finish`` for the error text after it banked
    the failed session and before it prints the ``result`` event; the
    fold runs on its own thread, as a child's does, and is awaited so
    the interleaving is the same on every run.
    """

    def __init__(self, agent: RelentlessAgent) -> None:
        self.fold = threading.Thread(target=_attribute_sub_usage, args=(agent, *_UNIT))

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        """Run the fold once, at ``perform_task``'s call of ``finish``."""
        caller = frame.f_back
        if (
            event == "call"
            and frame.f_code is finish.__code__
            and caller is not None
            and caller.f_code is RelentlessAgent.perform_task.__code__
            and self.fold.ident is None
        ):
            self.fold.start()
            self.fold.join(_WAIT)


def test_error_result_of_a_failed_first_session_counts_the_session_once() -> None:
    """A fold between the failed session's bank and its ``result`` adds nothing twice."""
    no_choices = {"id": "chatcmpl-test", "object": "chat.completion", "choices": []}
    server, port = _start_openai_server([no_choices])
    agent = RelentlessAgent("error-result-offsets")
    printer = OffsetAwarePrinter()
    hook = _FoldAtErrorResult(agent)
    sys.setprofile(hook)
    try:
        with tempfile.TemporaryDirectory() as work_dir:
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Fail at the first model call.",
                max_budget=5.0,
                work_dir=work_dir,
                verbose=False,
                printer=printer,
                model_config={"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "sk-test"},
            )
    finally:
        sys.setprofile(None)
        server.shutdown()
    assert hook.fold.ident is not None and not hook.fold.is_alive()

    ((content, shown),) = printer.result_events()
    assert "returned no choices" in str(content)
    budget, tokens, steps = agent.usage_snapshot()
    assert steps > _UNIT[2], "the failed session's own step is banked"
    assert (shown["cost"], shown["total_tokens"], shown["step_count"]) == (
        f"${budget:.4f}",
        tokens,
        steps,
    )
