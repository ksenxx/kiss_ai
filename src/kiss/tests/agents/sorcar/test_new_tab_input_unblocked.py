"""Regression test: creating a new tab while a task is running in the
current tab must NOT leave the new tab's input textbox and send button
disabled.

Root cause: ``createNewTab()`` delegates reset to ``restoreTab(tab)``
which ends with ``updateInputDisabled()``.  ``updateInputDisabled`` reads
the *module-global* ``isRunning`` (not ``tab.isRunning``), so when the
previous tab had ``isRunning=true`` the flag stays true across the tab
switch and the new tab's ``inp`` / ``sendBtn`` appear blocked.

``switchToTab`` and ``closeTab`` compensate by calling
``setRunningState(tab.isRunning)`` after ``restoreTab``; ``createNewTab``
must do the same.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _extract_fn_body(src: str, header: str) -> str:
    """Return the source of a top-level ``function name() { ... }`` block
    whose header matches ``header`` (e.g. ``function createNewTab()``).
    Braces are matched by counting; string/comment handling is minimal
    and sufficient for main.js."""
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


class TestCreateNewTabDoesNotBlockInput(unittest.TestCase):
    """End-to-end-ish behavioural test: load the real ``createNewTab``
    source from main.js, stub its DOM/VS Code dependencies, simulate a
    running task, invoke ``createNewTab``, and assert that the new tab's
    input controls are not disabled."""

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_new_tab_does_not_disable_input_when_prior_tab_running(
        self,
    ) -> None:
        create_new_tab_src = _extract_fn_body(self.js, "function createNewTab()")
        # Sanity: the extracted source must be the real function body.
        assert "saveCurrentTab()" in create_new_tab_src
        assert "makeTab(" in create_new_tab_src
        assert "restoreTab(tab)" in create_new_tab_src

        # The simulation below faithfully mirrors the *relevant* bits of
        # main.js: updateInputDisabled derives inp.disabled / sendBtn.disabled
        # from the module-global isRunning (and isMerging), and restoreTab
        # ends with updateInputDisabled() — matching the real file.  If
        # createNewTab fails to sync the module-global isRunning to the
        # new (non-running) tab, inp.disabled stays stuck at true.
        preamble = r"""
            var isRunning = false;
            var isMerging = false;
            var inp = { value: '', disabled: false, style: {} };
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
            var vscode = { postMessage: function(m) { _postedMessages.push(m); } };
            // createNewTab inspects document.body.classList for a
            // 'remote-chat' marker (used by the standalone web webview);
            // in this VS Code-extension simulation the marker is absent.
            var document = {
                body: { classList: { contains: function() { return false; } } },
            };

            function trimOldestTabs() {}
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

            // Mirror main.js exactly:
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
                if (running) startTimer();
            }

            function makeTab(title) {
                return {
                    id: 'tab-' + (++_idCounter),
                    title: title || 'new chat',
                    isRunning: false,
                    isMerging: false,
                    inputValue: '',
                    attachments: [],
                    selectedModel: 'claude-opus-4-6',
                    panelsExpanded: false,
                };
            }

            // saveCurrentTab's only relevant side effect for this test is
            // persisting the running flag onto the current tab — which
            // does NOT clear the module-global isRunning.
            function saveCurrentTab() {
                var tab = tabs.find(function(t) { return t.id === activeTabId; });
                if (!tab) return;
                tab.isRunning = isRunning;
                tab.isMerging = isMerging;
                tab.inputValue = inp.value;
            }

            // restoreTab mirrors main.js: restores isMerging from tab but
            // NOT isRunning, then calls updateInputDisabled() — which
            // reads the stale module-global isRunning.
            function restoreTab(tab) {
                activeTabId = tab.id;
                inp.value = tab.inputValue || '';
                isMerging = tab.isMerging || false;
                updateInputDisabled();
            }

            // Seed: tab 1 is running.
            var tab1 = makeTab('running task');
            tabs.push(tab1);
            activeTabId = tab1.id;
            setRunningState(true);

            // Invariant check on seed state:
            if (inp.disabled !== true || sendBtn.disabled !== true) {
                process.stdout.write('SEED_FAIL');
                process.exit(1);
            }

            // The REAL createNewTab from main.js is appended after this
            // preamble, followed by the harness invocation.
            """
        harness = r"""
            // Exercise the bug:
            createNewTab();

            // The freshly-created tab is not running.  Its input must be
            // enabled.
            if (inp.disabled === false && sendBtn.disabled === false) {
                process.stdout.write('PASS');
            } else {
                process.stdout.write(
                    'FAIL inp.disabled=' + inp.disabled
                    + ' sendBtn.disabled=' + sendBtn.disabled);
            }
            """
        script = preamble + "\n" + create_new_tab_src + "\n" + harness
        result = _run_node(script)
        assert result.returncode == 0, (
            f"node error: {result.stderr}\nstdout: {result.stdout}"
        )
        assert result.stdout == "PASS", (
            f"createNewTab left input blocked: {result.stdout}"
        )


class TestCreateNewTabStructural(unittest.TestCase):
    """Static guard: ``createNewTab`` must sync the module-global running
    state with the new tab's ``isRunning`` (which is false for fresh
    tabs).  Mirrors the explicit ``setRunningState(tab.isRunning)`` call
    that ``switchToTab`` and ``closeTab`` already make after
    ``restoreTab``."""

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_create_new_tab_syncs_running_state(self) -> None:
        body = _extract_fn_body(self.js, "function createNewTab()")
        # Must call setRunningState so that isRunning / sendBtn / upload /
        # worktree / parallel / model button states are synced with the
        # new (empty, non-running) tab.
        assert re.search(r"setRunningState\(\s*(false|tab\.isRunning)\s*\)", body), (
            "createNewTab() must call setRunningState(false) or "
            "setRunningState(tab.isRunning) after restoreTab(tab); otherwise "
            "the previous tab's running state persists and blocks input."
        )


if __name__ == "__main__":
    unittest.main()
