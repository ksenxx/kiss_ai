# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the Running / Done label at the top of a chat
page must be computed from the agent's real start / end
timestamps — not from the client's wall clock at the moment the tab
or page is loaded.

Two bugs were patched to enforce this invariant:

1. ``startTimer()`` in ``media/main.js`` used to anchor ``t0`` to
   ``Date.now()`` when no value was set, which meant a chat that
   started an hour ago and is then loaded from history showed
   "Running 0s" instead of "Running 1h 0s".  The fix routes the
   agent's real start timestamp through the backend ``status``
   event (``ev.startTs`` in ms since epoch) and through the
   history-list payload (``s.startTs`` per row) so the frontend
   knows the true elapsed time.

2. The ``task_done`` handler in ``media/main.js`` computed the
   completion duration from ``Date.now() - doneT0``; for tabs that
   joined late this is wrong.  The fix uses ``ev.endTs - ev.startTs``
   when both are present.  The ``endTs`` value is persisted into the
   task_history ``extra`` JSON column by ``_save_task_extra`` and
   replayed on history list / replay so the frontend can flip the
   "Running …" label to the time the task took ("Xm Ys") as soon
   as ``Date.now() >= endTs``, even when no live ``task_done`` event
   arrives (e.g. the task finished while no client was connected).
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.tests.server.test_running_label_timestamp import _read

MAIN_JS = (
    Path(__file__).parent.parent.parent
    / "webapp"
    / "media"
    / "main.js"
)


def _extract_case_body(src: str, case_name: str) -> str:
    """Return the body of ``case '<case_name>':`` in main.js."""
    m = re.search(rf"case\s+'{re.escape(case_name)}'\s*:\s*\{{", src)
    assert m, f"case '{case_name}' not found"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


def _extract_fn_body(src: str, fn_name: str) -> str:
    """Return the body of ``function <fn_name>(...) { ... }`` in main.js."""
    m = re.search(rf"function\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{", src)
    assert m, f"function '{fn_name}' not found"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


def test_status_handler_anchors_t0_to_ev_start_ts():
    """The ``case 'status':`` handler must seed ``t0`` (and
    ``evTab.t0``) from ``ev.startTs`` before calling
    ``setRunningState(true)`` so the running label reflects the
    agent's actual elapsed time."""
    body = _extract_case_body(_read(MAIN_JS), "status")
    assert "ev.startTs" in body, (
        "case 'status' handler must consume ev.startTs"
    )
    # The start goes through a local (``const start = ... ev.startTs ...``)
    # because a running status without one starts the clock now.
    start = re.search(r"const\s+start\s*=[^;]*\bev\.startTs\b[^;]*;", body)
    assert start, "case 'status' must read the start from ev.startTs"
    assert re.search(r"\bt0\s*=\s*start\b", body[start.end():]), (
        "case 'status' must set t0 from ev.startTs"
    )


def test_timer_tick_flips_to_done_when_now_exceeds_end_ts():
    """``_renderTimerTick`` must check the agent's recorded end
    timestamp and switch the label to the time the task took
    (``Xm Ys``, computed from ``endTs - t0``) as soon as
    ``Date.now() >= endTs``."""
    src = _read(MAIN_JS)
    body = _extract_fn_body(src, "_renderTimerTick")
    assert "endTs" in body, (
        "_renderTimerTick must consult the agent's endTs"
    )
    assert "doneLabelFor" in body, (
        "_renderTimerTick must emit the time the task took when "
        "Date.now() >= endTs"
    )
    helper = _extract_fn_body(src, "doneLabelFor")
    assert "endMs - startMs" in helper, (
        "doneLabelFor must format the span endMs - startMs"
    )
    assert "Date.now()" in body and ">=" in body, (
        "_renderTimerTick must compare Date.now() >= endTs"
    )


def test_task_done_uses_event_start_and_end_ts():
    """The ``case 'task_done':`` handler must compute the duration
    from the agent-supplied ``ev.endTs - ev.startTs`` (not from the
    client wall clock at event arrival time).  The computation is
    ``endedTaskTime``, shared with the other terminal events."""
    src = _read(MAIN_JS)
    assert "endedTaskTime(ev)" in _extract_case_body(src, "task_done"), (
        "task_done must take its label from endedTaskTime(ev)"
    )
    body = _extract_fn_body(src, "endedTaskTime")
    assert "ev.endTs" in body and "ev.startTs" in body, (
        "endedTaskTime must compute elapsed from ev.endTs - ev.startTs"
    )
    assert re.search(
        r"ev\.endTs\s*-\s*ev\.startTs|ev\.startTs\s*&&\s*ev\.endTs",
        body,
    ), "endedTaskTime must subtract ev.startTs from ev.endTs"
