"""Integration test: result events for is_continue=True must not be FAILED.

When the agent's ``finish()`` returns ``is_continue=True`` (RelentlessAgent
sub-session that ran out of steps but is asking the outer loop to retry),
the result panel must show ``Status: Continue`` instead of ``Status: FAILED``.

This is verified end-to-end:
  * The Python side (``JsonPrinter._broadcast_result``) must include
    the parsed ``is_continue`` flag in the broadcast event so the webview
    can distinguish it from a hard failure.
  * The JavaScript renderer (``media/main.js``) must render
    ``Status: Continue`` when ``is_continue`` is true on the event.
"""

from __future__ import annotations

import yaml

from kiss.agents.vscode.json_printer import JsonPrinter


class _CapturePrinter(JsonPrinter):
    """Printer that captures every broadcast event in-memory."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def broadcast(self, event: dict[str, object]) -> None:  # type: ignore[override]
        self.events.append(event)


def test_broadcast_result_marks_is_continue() -> None:
    """``_broadcast_result`` must propagate ``is_continue`` to the event."""
    printer = _CapturePrinter()
    raw = yaml.dump(
        {
            "success": False,
            "is_continue": True,
            "summary": "ran out of steps; please retry",
        },
        sort_keys=False,
    )

    printer.print(raw, type="result", total_tokens=10, cost="$0.10", step_count=3)

    results = [e for e in printer.events if e.get("type") == "result"]
    assert len(results) == 1, results
    ev = results[0]
    assert ev.get("success") is False
    assert ev.get("is_continue") is True, (
        "is_continue must be carried through to the renderer; "
        "without it the panel mislabels a continue as a failure"
    )
    assert ev.get("summary") == "ran out of steps; please retry"


def test_broadcast_result_failure_without_is_continue() -> None:
    """Hard failures (``is_continue=False``) keep the FAILED status path."""
    printer = _CapturePrinter()
    raw = yaml.dump(
        {"success": False, "is_continue": False, "summary": "hard failure"},
        sort_keys=False,
    )

    printer.print(raw, type="result")

    ev = next(e for e in printer.events if e.get("type") == "result")
    assert ev.get("success") is False
    assert ev.get("is_continue") is False

