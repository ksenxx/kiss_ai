"""Regression: ``merge_data`` events from the daemon must NOT be
forwarded to the chat webview by ``SorcarSidebarView``.

Context (post Phase 1–5 single-daemon refactor):

* ``WebPrinter.broadcast`` augments every ``merge_data`` event with
  ``base_text`` / ``current_text`` for each file so browser clients can
  render an inline diff in chat (the browser has no editor API to read
  buffers).  The same augmented event is fanned out over UDS to the
  VS Code extension.

* In the extension, ``merge_data`` is consumed exclusively by the
  native VS Code ``MergeManager`` (it opens a real merge editor tab).
  Forwarding the augmented event to the chat webview as well would
  cause ``media/main.js``'s ``renderMergeData`` to paint a *second*,
  inline diff in the chat output — duplicating the native merge editor
  UI.

* The fix is in ``SorcarSidebarView._installClientListener``: every
  message is forwarded to the webview via ``_sendToWebview(msg)``
  EXCEPT ``msg.type === 'merge_data'``.  ``merge_started`` /
  ``merge_ended`` / ``merge_nav`` are still forwarded so the in-input
  merge toolbar (Prev / Next / Accept / Reject) keeps working.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SIDEBAR_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "SorcarSidebarView.ts"
)


def _extract_install_listener(source: str) -> str:
    """Return the body of ``_installClientListener``."""
    m = re.search(r"_installClientListener\s*\([^)]*\)\s*:\s*void\s*\{", source)
    assert m is not None, "_installClientListener not found"
    start = m.end() - 1  # position at the opening brace
    depth = 0
    for i in range(start, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("Unterminated _installClientListener body")


class TestMergeDataNotForwarded(unittest.TestCase):
    """``merge_data`` is consumed by MergeManager only — never the webview."""

    def setUp(self) -> None:
        self.source = SIDEBAR_TS.read_text()
        self.body = _extract_install_listener(self.source)

    def test_send_to_webview_is_guarded_against_merge_data(self) -> None:
        """The ``_sendToWebview(msg)`` call in the listener must be guarded
        by a condition that excludes ``msg.type === 'merge_data'``.

        We look for the unconditional ``this._sendToWebview(msg);`` and
        assert it does NOT appear as a bare statement at the listener's
        top level — it must be wrapped in an ``if (msg.type !== 'merge_data')``
        (or equivalent).
        """
        # The fix wraps the forward in a guard.
        self.assertRegex(
            self.body,
            r"if\s*\(\s*msg\.type\s*!==\s*['\"]merge_data['\"]\s*\)\s*\{\s*this\._sendToWebview\(msg\)\s*;\s*\}",
            "Expected `_sendToWebview(msg)` to be guarded by "
            "`if (msg.type !== 'merge_data')` so the augmented merge "
            "event is not duplicated into the chat webview.",
        )

    def test_no_unguarded_forward_of_msg(self) -> None:
        """There must be no bare ``this._sendToWebview(msg);`` line that
        would forward ``merge_data`` to the webview unconditionally.

        We search for ``this._sendToWebview(msg)`` (the exact catch-all
        argument) and assert every occurrence inside the listener body
        is inside a guard that excludes ``merge_data``.
        """
        # Find every occurrence of `this._sendToWebview(msg)` in the body.
        for m in re.finditer(r"this\._sendToWebview\(msg\)\s*;", self.body):
            # Look backwards in the same line context for a guarding `if`.
            window = self.body[max(0, m.start() - 200) : m.start()]
            self.assertIn(
                "merge_data",
                window,
                "Found `this._sendToWebview(msg);` not preceded by a "
                "merge_data guard — this would forward the augmented "
                "merge_data event into the chat webview and double the "
                "merge UI.",
            )

    def test_merge_manager_still_consumes_merge_data(self) -> None:
        """The native merge path must still run for ``merge_data``."""
        self.assertIn("msg.type === 'merge_data'", self.body)
        self.assertIn("openMerge(msg.data)", self.body)


if __name__ == "__main__":
    unittest.main()
