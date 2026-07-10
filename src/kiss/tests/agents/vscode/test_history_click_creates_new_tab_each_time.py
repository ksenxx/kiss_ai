# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: clicking the same task in the task-history panel
multiple times must allocate a fresh chat tab on every click, instead
of focusing an existing tab keyed by the same id.

Why
---
Sorcar's backend is the multi-client source of truth: several browsers
or VS Code webviews may observe the same session concurrently, so a
frontend dedupe of the form ``if (tabs.find(t => t.id === presetId))
{ switchToTab(presetId); return; }`` is only correct for a single
client.  The history-row click handler in ``main.js`` was simplified
to always call ``createNewTab()`` and then post ``resumeSession`` with
the freshly minted tab id; this test pins that contract by replaying
the real ``createNewTab`` source through a Node-based harness and the
real ``main.js`` history-click branch.
"""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _extract_fn_body(src: str, header: str) -> str:
    """Return the source of a top-level ``function name(...) { ... }``
    block whose header matches ``header`` (e.g. ``function
    createNewTab(``).  Braces are matched by counting; string/comment
    handling is minimal and sufficient for ``main.js``.
    """
    start = src.index(header)
    brace = src.index("{", start)
    depth = 0
    i = brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated function body for {header}")


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestHistoryClickAlwaysCreatesNewTab(unittest.TestCase):
    """Behavioural test: load the real ``createNewTab`` from main.js,
    stub its DOM/VS Code dependencies, then invoke the history-click
    branch three times on the same persisted session row.  Assert that
    each click produces a *new* tab id, that none of the new tab ids
    collide with the session id (the multi-client routing key), and
    that ``resumeSession`` is posted once per click carrying the
    session id and the freshly minted tab id.
    """

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_three_clicks_on_same_history_row_create_three_tabs(self) -> None:
        create_new_tab_src = _extract_fn_body(self.js, "function createNewTab(")
        # Sanity: the source we're simulating must be the post-fix
        # version that allocates a fresh uuid every call and does NOT
        # short-circuit on a presetId.
        assert "function createNewTab()" in create_new_tab_src, (
            "createNewTab must take no parameters; the multi-client "
            "dedupe path was removed."
        )
        assert "switchToTab(" not in create_new_tab_src, (
            "createNewTab must not focus an existing tab; the dedupe "
            "branch was removed."
        )

        preamble = r"""
            var isRunning = false;
            var isMerging = false;
            var inp = { value: '', disabled: false, style: {}, focus: function() {} };
            var sendBtn = { disabled: false, style: {} };
            var stopBtn = { style: {} };
            var uploadBtn = { disabled: false };
            var worktreeToggleBtn = { disabled: false };
            var parallelToggleBtn = { disabled: false };
            var modelBtn = { disabled: false };
            var tabs = [];
            var activeTabId = '';
            var _idCounter = 0;
            var _postedMessages = [];
            var t0 = null;
            var vscode = { postMessage: function(m) { _postedMessages.push(m); } };
            var document = {
                body: { classList: { contains: function() { return false; } } },
            };

            // Mint a fresh-uuid-like id on every call so two calls
            // never collide.  Mirrors the real ``genTabId`` contract.
            function _mintId() {
                _idCounter += 1;
                return 'tab-' + _idCounter + '-' + Math.random().toString(36).slice(2);
            }

            function clearGhost() {}
            function hideAC() {}
            function closeModelDD() {}
            function startTimer() {}
            function stopTimer() {}
            function removeSpinner() {}
            function focusInputWithRetry() {}
            function renderTabBar() {}
            function persistTabState() {}
            function syncAskModalToActiveTab() {}
            function resetAdjacentState() {}
            function renderFileChips() {}
            function syncClearBtn() {}
            function updateChevronIcon() {}
            function applyChevronState() {}
            function clearDemoEndedUi() {}
            function setTaskText(text) { _lastTaskText = text; }
            var _lastTaskText = '';

            function updateInputDisabled() {
                var blocked = isRunning || isMerging;
                inp.disabled = blocked;
                sendBtn.disabled = blocked;
            }
            function setRunningState(running) {
                isRunning = running;
                sendBtn.style.display = running ? 'none' : 'flex';
                stopBtn.style.display = running ? 'flex' : 'none';
                uploadBtn.disabled = running;
                worktreeToggleBtn.disabled = running;
                parallelToggleBtn.disabled = running;
                modelBtn.disabled = running;
                updateInputDisabled();
            }
            function makeTab(title) {
                return {
                    id: _mintId(),
                    title: title || 'new chat',
                    isRunning: false,
                    isMerging: false,
                    inputValue: '',
                    attachments: [],
                    selectedModel: 'claude-opus-4-6',
                    panelsExpanded: false,
                };
            }
            function saveCurrentTab() {
                var tab = tabs.find(function(t) { return t.id === activeTabId; });
                if (!tab) return;
                tab.isRunning = isRunning;
                tab.isMerging = isMerging;
                tab.inputValue = inp.value;
            }
            function restoreTab(tab) {
                activeTabId = tab.id;
                inp.value = tab.inputValue || '';
                isMerging = tab.isMerging || false;
                updateInputDisabled();
            }

            // Seed: a baseline chat tab that the user is currently on.
            var baseline = makeTab('current chat');
            tabs.push(baseline);
            activeTabId = baseline.id;
            var seedTabId = baseline.id;
            """

        # The history-click body below mirrors the
        # ``if (s.has_events && s.id)`` branch in ``renderHistory``
        # exactly (see main.js, around the historyList click handler).
        # We replay it three times on the same ``s`` object — the
        # same row the user keeps clicking in the history panel.
        harness = r"""
            // The persisted session the user is repeatedly clicking.
            var s = {
                id: 'chat-abc-123',
                task_id: 42,
                preview: 'remember the magic word: banana',
                title: 'remember the magic word: banana',
                has_events: true,
            };

            function simulateHistoryRowClick() {
                // Mirrors the real renderHistory click branch.
                if (s.has_events && s.id) {
                    createNewTab();
                    var taskText = s.preview || s.title || '';
                    setTaskText(taskText);
                    inp.value = taskText;
                    syncClearBtn();
                    vscode.postMessage({
                        type: 'resumeSession',
                        id: s.id,
                        taskId: s.task_id,
                        tabId: activeTabId,
                    });
                }
            }

            simulateHistoryRowClick();
            simulateHistoryRowClick();
            simulateHistoryRowClick();

            var resumeMsgs = _postedMessages.filter(function(m) {
                return m.type === 'resumeSession';
            });
            var newChatMsgs = _postedMessages.filter(function(m) {
                return m.type === 'newChat';
            });
            process.stdout.write(JSON.stringify({
                tabIds: tabs.map(function(t) { return t.id; }),
                seedTabId: seedTabId,
                activeTabId: activeTabId,
                resumeMsgs: resumeMsgs,
                newChatMsgs: newChatMsgs,
            }));
            """

        script = preamble + "\n" + create_new_tab_src + "\n" + harness
        result = _run_node(script)
        assert result.returncode == 0, (
            f"node error: {result.stderr}\nstdout: {result.stdout}"
        )
        out = json.loads(result.stdout)

        tab_ids = out["tabIds"]
        # Seed tab + one per click → 4 tabs total.
        self.assertEqual(
            len(tab_ids), 4,
            f"expected 4 tabs (1 seed + 3 history clicks); got {tab_ids!r}",
        )
        # All tab ids must be unique — no dedupe by session id.
        self.assertEqual(
            len(set(tab_ids)), 4,
            f"tab ids must all be distinct; got {tab_ids!r}",
        )
        # None of the freshly minted tab ids must collide with the
        # session id (which used to be the "preset id" the dedupe path
        # would have keyed on).
        self.assertNotIn("chat-abc-123", tab_ids)
        # The active tab must be the most recent history-click tab
        # (and not the seed).
        self.assertNotEqual(out["activeTabId"], out["seedTabId"])
        self.assertEqual(out["activeTabId"], tab_ids[-1])

        # Exactly three resumeSession messages, one per click; each
        # carries the SAME chat id (the multi-client routing key) but
        # a DISTINCT tab id (the per-tab routing key).
        resume_msgs = out["resumeMsgs"]
        self.assertEqual(len(resume_msgs), 3)
        chat_ids = {m["id"] for m in resume_msgs}
        self.assertEqual(chat_ids, {"chat-abc-123"})
        task_ids = {m["taskId"] for m in resume_msgs}
        self.assertEqual(task_ids, {42})
        resume_tab_ids = [m["tabId"] for m in resume_msgs]
        self.assertEqual(len(set(resume_tab_ids)), 3,
                         f"resumeSession tabIds must be distinct; "
                         f"got {resume_tab_ids!r}")
        # Each resumeSession's tabId must be one of the freshly
        # created tab ids — never the chat id.
        for tid in resume_tab_ids:
            self.assertIn(tid, tab_ids)
            self.assertNotEqual(tid, "chat-abc-123")

        # Three newChat messages also fired — one per createNewTab().
        new_chat_msgs = out["newChatMsgs"]
        self.assertEqual(len(new_chat_msgs), 3)
        new_chat_tab_ids = [m["tabId"] for m in new_chat_msgs]
        self.assertEqual(new_chat_tab_ids, resume_tab_ids,
                         "each newChat must precede its matching "
                         "resumeSession with the same tabId.")


class TestHistoryClickHandlerSourceHasNoDedupe(unittest.TestCase):
    """Static guard: the history-row click branch in ``renderHistory``
    must NOT condition its side effects on a ``createNewTab`` return
    value, and must NOT look up an existing tab by session id before
    calling ``createNewTab``.  These were the two shapes the removed
    dedupe used to take.
    """

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_render_history_click_has_no_dedupe(self) -> None:
        body = _extract_fn_body(self.js, "function renderHistory(")

        # No ``const created = createNewTab();`` style gating.
        self.assertIsNone(
            re.search(r"=\s*createNewTab\s*\(", body),
            "renderHistory must not capture createNewTab()'s return "
            "value — createNewTab no longer signals dedupe.",
        )
        # No ``tabs.find(... === s.id ...)`` style dedupe lookup.
        self.assertIsNone(
            re.search(r"tabs\.find\([^)]*s\.id", body),
            "renderHistory must not search tabs by session id before "
            "creating a new tab; the multi-client backend routes by "
            "chat id, so frontend dedupe is incorrect.",
        )
        # And no ``switchToTab(s.id)`` shortcut.
        self.assertNotIn(
            "switchToTab(s.id", body,
            "renderHistory must not switch to a tab keyed by the "
            "session id; always allocate a fresh tab.",
        )


if __name__ == "__main__":
    unittest.main()
