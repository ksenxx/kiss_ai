# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stop that lands inside the summarizer's bank must not lose its spend.

``RelentlessAgent._summarize_failed_session`` runs a helper
:class:`KISSAgent` after a failed session and banked its spend once, in
a ``finally``.  The runner's stop (an asynchronous ``KeyboardInterrupt``)
can land inside that ``_accumulate_usage`` call before its atomic
ledger append; unlike a session executor, the helper had no second bank
on ``BaseException``, so the task's final totals omitted the helper's
real cost, tokens and steps.  The helper is now banked like a session:
again, exactly once by its stable key, before the stop propagates.

Real scripted model calls (two executor calls, one summarizer call),
a real ``KeyboardInterrupt`` raised by a trace function at the first
call of the real bank of the summarizer; nothing is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import FrameType

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, tool_call_body

_BANK_CODE = RelentlessAgent._accumulate_usage.__code__


def note() -> str:
    """Do a harmless unit of work."""
    return "work completed"


def _run(
    tmp_path: Path, interrupt_summarizer_bank: bool
) -> tuple[tuple[float, int, int], tuple[float, int, int], str]:
    """Run one session that runs out of steps and gets summarized.

    Returns:
        ``(task usage snapshot, summarizer usage at its bank, what escaped the run)``.
    """
    agent = RelentlessAgent("summarizer-bank")
    fired = False
    summarizer_usage = (0.0, 0, 0)

    def stop_at_summarizer_bank(frame: FrameType, event: str, arg: object) -> None:
        nonlocal fired, summarizer_usage
        if event == "call" and frame.f_code is _BANK_CODE:
            banked = frame.f_locals["agent"]
            if banked.name.endswith(" Summarizer"):
                summarizer_usage = banked.usage_snapshot()
                if interrupt_summarizer_bank and not fired:
                    fired = True
                    raise KeyboardInterrupt("injected stop")

    script = [
        tool_call_body("note", {}, 1000),
        tool_call_body("note", {}, 1200),
        tool_call_body("finish", {"result": "<p>Two notes done; continue.</p>"}, 2000),
    ]
    escaped = ""
    with serve(script) as (url, requests):
        sys.settrace(stop_at_summarizer_bank)
        try:
            agent.run(
                model_name=MODEL,
                prompt_template="Take notes then continue.",
                tools=[note],
                work_dir=str(tmp_path),
                max_steps=2,
                max_sub_sessions=1,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local", "plain_io": True},
                verbose=False,
            )
        except (KeyboardInterrupt, KISSError) as exc:
            escaped = type(exc).__name__
        finally:
            sys.settrace(None)
        assert len(requests) == 3, "two executor calls and one summarizer call"
    assert fired == interrupt_summarizer_bank
    assert summarizer_usage[2] == 1 and summarizer_usage[0] > 0, summarizer_usage
    return agent.usage_snapshot(), summarizer_usage, escaped


def test_a_stop_inside_the_summarizer_bank_still_banks_the_summarizer(tmp_path: Path) -> None:
    """The totals of a run stopped inside the helper's bank equal an unstopped run's."""
    expected, helper, escaped = _run(tmp_path / "plain", interrupt_summarizer_bank=False)
    assert escaped == "KISSError", escaped
    assert expected[2] == 3 and expected[0] > helper[0], (expected, helper)
    banked, _helper, escaped = _run(tmp_path / "stopped", interrupt_summarizer_bank=True)
    assert escaped == "KeyboardInterrupt", "the injected stop must propagate out of the run"
    assert banked == expected, (banked, expected)
