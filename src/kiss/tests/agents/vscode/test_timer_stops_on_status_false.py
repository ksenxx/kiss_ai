"""Regression test: timer must stop when ``setRunningState(false)`` runs.

Bug: in the web app, the header timer ("Running 1m 2s") sometimes kept
ticking after the task finished — visually contradicting the result panel
already rendered below.  The root cause is that the timer is only stopped
by the ``case 'task_done'`` handler, which requires ``ev.tabId`` to match
``activeTabId``.  If the backend emits a ``status: running: false`` event
without a matching ``task_done`` (for example, because ``task_done`` was
broadcast with an unexpected ``tabId`` or because the agent process
died before reaching the task-end broadcast), ``setRunningState(false)``
is called and the running flag flips, but ``stopTimer()`` is never
invoked — so the header keeps showing "Running …" forever.

This test pins the fix: ``setRunningState(false)`` must always stop the
timer, remove the spinner, and replace a "Running …" header with
"Done".
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _extract_function_body(src: str, name: str) -> str:
    """Return the body of the first ``function <name>(...) { ... }`` block."""
    m = re.search(rf"function {re.escape(name)}\(", src)
    assert m is not None, f"{name} not found in main.js"
    body_start = src.index("{", m.start())
    depth = 0
    i = body_start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[body_start : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in {name}")


class TestSetRunningStateStopsTimer(unittest.TestCase):
    """``setRunningState(false)`` always stops the header timer."""

    def test_set_running_state_calls_stop_timer_on_false_branch(self) -> None:
        body = _extract_function_body(MAIN_JS.read_text(), "setRunningState")
        # The "else" branch (running=false) must call stopTimer().
        else_match = re.search(r"\}\s*else\s*\{([^}]+)\}", body, re.DOTALL)
        assert else_match is not None, (
            "setRunningState must have an explicit else branch handling "
            "the running=false case; otherwise the timer never stops "
            "without a matching task_done event."
        )
        else_body = else_match.group(1)
        self.assertIn(
            "stopTimer()",
            else_body,
            "setRunningState(false) must call stopTimer() so the header "
            "never shows 'Running …' after the task ends.",
        )

    def test_set_running_state_clears_running_header_text(self) -> None:
        body = _extract_function_body(MAIN_JS.read_text(), "setRunningState")
        else_match = re.search(r"\}\s*else\s*\{([^}]+)\}", body, re.DOTALL)
        assert else_match is not None
        else_body = else_match.group(1)
        self.assertIn(
            "Running",
            else_body,
            "setRunningState(false) must inspect the header text and "
            "replace a 'Running …' label so the user does not see a "
            "stale running label.",
        )
        self.assertIn("Done", else_body)

    def test_set_running_state_removes_spinner_on_false(self) -> None:
        body = _extract_function_body(MAIN_JS.read_text(), "setRunningState")
        else_match = re.search(r"\}\s*else\s*\{([^}]+)\}", body, re.DOTALL)
        assert else_match is not None
        else_body = else_match.group(1)
        self.assertIn(
            "removeSpinner()",
            else_body,
            "setRunningState(false) must also remove the inline spinner "
            "so the UI matches the non-running state.",
        )


if __name__ == "__main__":
    unittest.main()
