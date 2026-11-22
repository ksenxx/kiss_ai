# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""2026-11-22 cost audit: a reopened chat shows the row's cost after a ``result`` too.

A task that is killed while its sub-agents run keeps the ``usage_info``
of its last step as its last cost event; the orphan sweep then raises
its row by what the sub-agents spent.  ``with_synthesized_events`` appends
the row's totals to the replay of such a task, so the reopened chat's
header shows the row's cost.  It did so only for a stream without any
``result`` event.  A task that ran on after ``finish(is_continue=True)``
has a ``result`` from every session it closed, and its replay ended on
the stale figure: on 2026-11-21 a task of $3,972.21 (its row, and what
the Spend panel sums) reopened as "Cost: $3,515.19".

The rule is now what the page does: its header takes the cost of each
``usage_info`` and ``result`` event that carries one, so the replay of
a finished task ends on the row's cost whenever its last such event
shows less.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from kiss.server.json_printer import with_synthesized_events

TASK = "a" * 32


def _session(cost: float, end_ts: int = 1795280341000) -> dict[str, Any]:
    """A loader session of a task whose row holds *cost* (1,300 tokens, 13 steps)."""
    extra = {
        "model": "claude-opus-5-5",
        "cost": cost,
        "tokens": 1300,
        "steps": 13,
        "startTs": 1795255807000,
        "endTs": end_ts,
    }
    return {"task": "t", "task_id": TASK, "chat_id": "c" * 32, "extra": json.dumps(extra)}


def _settings() -> dict[str, Any]:
    return {"type": "task_settings", "settings": {"task_id": TASK}, "taskId": TASK}


def _usage(cost: float) -> dict[str, Any]:
    return {
        "type": "usage_info",
        "text": "",
        "taskId": TASK,
        "total_tokens": 1000,
        "cost": f"${cost:.4f}",
        "cost_usd": cost,
        "total_steps": 10,
    }


def _result(cost: float | None) -> dict[str, Any]:
    event: dict[str, Any] = {"type": "result", "text": "done", "taskId": TASK, "step_count": 9}
    if cost is not None:
        event.update(cost=f"${cost:.4f}", cost_usd=cost)
    return event


def _header_cost(events: list[dict[str, Any]]) -> float | None:
    """The cost the page's header ends on: the last ``usage_info`` / ``result`` that has one."""
    for event in reversed(events):
        if event.get("type") in ("usage_info", "result") and "cost_usd" in event:
            return float(event["cost_usd"])
    return None


class ReplayCostAfterResult(unittest.TestCase):
    def test_a_killed_task_that_closed_a_session_reopens_on_the_row_cost(self) -> None:
        """The 2026-11-21 case: result of session 1, later steps, row raised by the sweep."""
        events = [_settings(), _usage(3253.49), _result(3253.49), _usage(3515.19)]

        replay = with_synthesized_events(list(events), _session(3972.21))

        self.assertEqual(replay[:-1], events)
        self.assertEqual(
            replay[-1],
            {
                "type": "usage_info",
                "text": "",
                "taskId": TASK,
                "total_tokens": 1300,
                "cost": "$3972.2100",
                "cost_usd": 3972.21,
                "total_steps": 13,
            },
        )
        self.assertEqual(_header_cost(replay), 3972.21)

    def test_a_stream_that_ends_on_a_result_below_the_row_gets_the_row_cost(self) -> None:
        """A late charge after the result that left no event of its own."""
        events = [_settings(), _usage(4.0), _result(4.0)]

        replay = with_synthesized_events(list(events), _session(5.0))

        self.assertEqual(len(replay), len(events) + 1)
        self.assertEqual(_header_cost(replay), 5.0)

    def test_a_result_without_usage_events_below_the_row_gets_the_row_cost(self) -> None:
        events = [_settings(), _result(4.0)]

        replay = with_synthesized_events(list(events), _session(5.0))

        self.assertEqual(_header_cost(replay), 5.0)

    def test_a_finished_task_whose_events_show_the_row_cost_is_left_alone(self) -> None:
        for events in (
            [_settings(), _usage(4.0), _result(4.0)],
            [_settings(), _usage(3.0), _result(3.0), _usage(4.0)],
            [_settings(), _result(4.0)],
            # A usage event after the result (a late charge that was published).
            [_settings(), _usage(3.0), _result(3.0), _usage(4.0), _result(4.0)],
        ):
            with self.subTest(types=[event["type"] for event in events]):
                self.assertEqual(with_synthesized_events(list(events), _session(4.0)), events)

    def test_the_last_cost_event_decides_not_the_largest(self) -> None:
        """An older, larger figure earlier in the stream does not hide a stale end."""
        events = [_settings(), _usage(9.0), _result(9.0), _usage(4.0)]

        replay = with_synthesized_events(list(events), _session(5.0))

        self.assertEqual(_header_cost(replay), 5.0)

    def test_a_running_task_is_left_to_its_live_events(self) -> None:
        events = [_settings(), _usage(3.0), _result(3.0), _usage(4.0)]

        replay = with_synthesized_events(list(events), _session(223.5, end_ts=0))

        self.assertEqual(replay, events)

    def test_a_row_below_its_events_never_lowers_the_header(self) -> None:
        events = [_settings(), _usage(3.0), _result(3.0), _usage(4.0)]

        self.assertEqual(with_synthesized_events(list(events), _session(3.5)), events)

    def test_events_without_a_cost_are_passed_over(self) -> None:
        """An event that sets no cost keeps the header: the one before it is compared."""
        events = [_settings(), _usage(4.0), _result(None)]

        self.assertEqual(with_synthesized_events(list(events), _session(4.0)), events)
        replay = with_synthesized_events(list(events), _session(223.5))
        self.assertEqual(_header_cost(replay), 223.5)
        # No event of the stream carries a cost: nothing is known to be stale.
        bare = [_settings(), _result(None)]
        self.assertEqual(with_synthesized_events(list(bare), _session(223.5)), bare)

    def test_an_older_usage_event_is_compared_by_its_text_after_a_result(self) -> None:
        text_only = {
            "type": "usage_info",
            "text": "Steps: 3/100, Total tokens: 1,000, Budget: $4.0000/$500.00, ",
            "taskId": TASK,
        }
        events = [_settings(), _result(3.0), text_only]

        self.assertEqual(with_synthesized_events(list(events), _session(4.0)), events)
        replay = with_synthesized_events(list(events), _session(223.5))
        self.assertEqual(replay[-1]["cost_usd"], 223.5)
        self.assertEqual(len(replay), len(events) + 1)

    def test_a_stream_without_cost_events_still_gets_the_row_totals(self) -> None:
        """Unchanged: neither a usage_info nor a result, finished or running."""
        for end_ts in (0, 1795280341000):
            with self.subTest(end_ts=end_ts):
                replay = with_synthesized_events([_settings()], _session(2.5, end_ts=end_ts))
                self.assertEqual(
                    [event["type"] for event in replay], ["task_settings", "usage_info"]
                )
                self.assertEqual(replay[-1]["cost_usd"], 2.5)


if __name__ == "__main__":
    unittest.main()
