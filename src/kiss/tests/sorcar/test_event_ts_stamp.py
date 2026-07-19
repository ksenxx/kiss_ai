# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for event timestamp stamping (``ts`` field).

The chat webview (VS Code extension and remote web app) renders a
compact human-readable timestamp badge in every event panel's title,
to the left of the Copy button.  For that to work, every broadcast
event must carry a ``ts`` field (ms since the epoch) stamped at
emission time — on both the base :class:`JsonPrinter` path and the
:class:`WebPrinter` transport override (which does NOT call super) —
while replayed events must KEEP their original stamp.
"""

import time

from kiss.server.json_printer import JsonPrinter, stamp_event_ts
from kiss.server.server import _coalesced_replay_events
from kiss.server.web_server import WebPrinter


def _now_ms() -> int:
    return int(time.time() * 1000)


def test_stamp_event_ts_adds_ts_when_absent() -> None:
    """A fresh event gets an integer ms-epoch ``ts`` close to now."""
    before = _now_ms()
    event = {"type": "prompt", "text": "hi"}
    stamp_event_ts(event)
    after = _now_ms()
    assert isinstance(event["ts"], int)
    assert before <= event["ts"] <= after


def test_stamp_event_ts_preserves_existing_ts() -> None:
    """A pre-stamped (replayed) event keeps its original ``ts``."""
    original = 1614956820000  # 2021-03-05
    event = {"type": "tool_call", "name": "Bash", "ts": original}
    stamp_event_ts(event)
    assert event["ts"] == original


def test_json_printer_broadcast_stamps_recorded_events() -> None:
    """JsonPrinter.broadcast stamps ``ts`` on every recorded event."""
    printer = JsonPrinter()
    printer._thread_local.task_id = "101"
    printer.start_recording()
    before = _now_ms()
    printer.print("system prompt text", type="system_prompt")
    printer.print("user prompt text", type="prompt")
    printer.print("Bash", type="tool_call", tool_input={"command": "ls"})
    printer.print(
        "boom", type="tool_result", tool_name="Bash", is_error=True,
    )
    printer.print("done", type="result", total_tokens=1, step_count=1)
    after = _now_ms()
    events = printer.stop_recording()
    types = [e["type"] for e in events]
    assert "system_prompt" in types
    assert "prompt" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "result" in types
    for ev in events:
        assert isinstance(ev.get("ts"), int), f"missing ts on {ev['type']}"
        assert before <= ev["ts"] <= after


def test_json_printer_broadcast_preserves_replay_ts() -> None:
    """A replayed event broadcast through JsonPrinter keeps its ``ts``."""
    printer = JsonPrinter()
    printer._thread_local.task_id = "102"
    printer.start_recording()
    original = 1614956820000
    printer.broadcast({"type": "prompt", "text": "old", "ts": original})
    events = printer.stop_recording()
    assert len(events) == 1
    assert events[0]["ts"] == original


def test_web_printer_broadcast_stamps_task_events() -> None:
    """WebPrinter.broadcast (transport override) also stamps ``ts``."""
    printer = WebPrinter()
    printer._thread_local.task_id = "103"
    printer.start_recording()
    before = _now_ms()
    printer.broadcast({"type": "tool_call", "name": "Read"})
    after = _now_ms()
    events = printer.stop_recording()
    assert len(events) == 1
    assert isinstance(events[0].get("ts"), int)
    assert before <= events[0]["ts"] <= after


def test_web_printer_broadcast_preserves_replay_ts() -> None:
    """WebPrinter.broadcast keeps a pre-stamped ``ts`` (replay path)."""
    printer = WebPrinter()
    printer._thread_local.task_id = "104"
    printer.start_recording()
    original = 946684800000  # 2000-01-01
    printer.broadcast({"type": "result", "text": "ok", "ts": original})
    events = printer.stop_recording()
    assert len(events) == 1
    assert events[0]["ts"] == original


def test_replay_backfills_ts_from_legacy_timestamp() -> None:
    """Replay prep backfills ``ts`` from the persisted ``_timestamp``.

    Rows persisted before events carried a ``ts`` stamp still know
    when they happened via the ``events.timestamp`` DB column that the
    loaders inject as ``_timestamp`` (seconds float).
    """
    legacy_sec = 1593855000.5  # 2020-07-04
    events = _coalesced_replay_events([
        {"type": "prompt", "text": "old", "_timestamp": legacy_sec},
    ])
    assert events[0]["ts"] == int(legacy_sec * 1000)


def test_replay_keeps_explicit_ts_over_legacy_timestamp() -> None:
    """An event's own ``ts`` stamp wins over the DB row timestamp."""
    events = _coalesced_replay_events([
        {"type": "prompt", "text": "x", "ts": 111, "_timestamp": 222.0},
    ])
    assert events[0]["ts"] == 111


def test_replay_ignores_junk_legacy_timestamps() -> None:
    """Garbage ``_timestamp`` values never produce a ``ts``.

    Covers the data-forensics cases: TEXT/BLOB column junk, NaN, inf,
    non-positive values, and values beyond the ECMAScript Date range.
    """
    junk = ["garbage", b"\x00", float("nan"), float("inf"), 0, -1, 9e12]
    events = _coalesced_replay_events([
        {"type": "prompt", "text": str(i), "_timestamp": v}
        for i, v in enumerate(junk)
    ])
    assert all("ts" not in ev for ev in events)


def test_replay_non_list_events_returns_empty() -> None:
    """A malformed (non-list) events payload coalesces to []."""
    assert _coalesced_replay_events("not-a-list") == []


def test_coalesced_deltas_keep_first_ts() -> None:
    """Coalescing consecutive deltas preserves the ``ts`` of the first."""
    printer = JsonPrinter()
    printer._thread_local.task_id = "105"
    printer.start_recording()
    printer.broadcast({"type": "text_delta", "text": "a", "ts": 1000})
    printer.broadcast({"type": "text_delta", "text": "b", "ts": 2000})
    events = printer.stop_recording()
    assert len(events) == 1
    assert events[0]["text"] == "ab"
    assert events[0]["ts"] == 1000
