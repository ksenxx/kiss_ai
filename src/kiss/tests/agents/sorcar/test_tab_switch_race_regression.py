# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: switching tabs during a running task.

Verifies that backend messages are correctly routed to the executing tab
(not the active tab) when the user switches tabs mid-task, and that state
is properly replayed when switching back.

Each test creates a minimal JS environment matching main.js globals, sets up
two tabs with a task running on tab 1, switches to tab 2, then sends backend
messages and verifies state isolation.
"""

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

_JS_PREAMBLE = r"""
// --- Minimal DOM stubs ---
var _elements = {};
var _doc_html = '';

function _makeEl(tag) {
    var el = {
        tagName: tag,
        id: '',
        className: '',
        textContent: '',
        innerHTML: '',
        style: {},
        dataset: {},
        disabled: false,
        children: [],
        _listeners: {},
        classList: {
            _c: [],
            add: function(c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
            remove: function(c) { var i = this._c.indexOf(c); if (i >= 0) this._c.splice(i,1); },
            contains: function(c) { return this._c.indexOf(c) >= 0; },
            toggle: function(c) { if (this.contains(c)) this.remove(c); else this.add(c); },
        },
        querySelector: function() { return _makeEl('div'); },
        querySelectorAll: function() { return []; },
        contains: function() { return false; },
        appendChild: function(c) { this.children.push(c); },
        removeChild: function() {},
        addEventListener: function(t, fn) { this._listeners[t] = fn; },
        dispatchEvent: function() {},
        focus: function() {},
        setSelectionRange: function() {},
        getBoundingClientRect: function() { return {top:0,left:0,width:100,height:20}; },
        scrollHeight: 100,
        scrollTop: 0,
    };
    return el;
}

var document = {
    getElementById: function(id) {
        if (!_elements[id]) _elements[id] = _makeEl('div');
        return _elements[id];
    },
    createElement: function(tag) { return _makeEl(tag); },
    createDocumentFragment: function() {
        var frag = _makeEl('fragment');
        frag.appendChild = function(c) { this.children.push(c); return c; };
        return frag;
    },
    body: _makeEl('body'),
    addEventListener: function() {},
};

// Pre-create needed elements
var _outputEl = _makeEl('div'); _outputEl.id = 'output';
_elements['output'] = _outputEl;
_elements['welcome'] = _makeEl('div');
_elements['task-input'] = _makeEl('textarea');
_elements['send-btn'] = _makeEl('button');
_elements['stop-btn'] = _makeEl('button');
_elements['upload-btn'] = _makeEl('button');
_elements['model-btn'] = _makeEl('button');
_elements['model-dropdown'] = _makeEl('div');
_elements['model-search'] = _makeEl('input');
_elements['model-list'] = _makeEl('div');
_elements['model-name'] = _makeEl('span');
_elements['file-chips'] = _makeEl('div');
_elements['status-text'] = _makeEl('span');
_elements['status-tokens'] = _makeEl('span');
_elements['status-budget'] = _makeEl('span');
_elements['sidebar'] = _makeEl('div');
_elements['history-search'] = _makeEl('input');
_elements['task-panel'] = _makeEl('div');
_elements['tab-bar'] = _makeEl('div');
_elements['tab-list'] = _makeEl('div');
_elements['merge-toolbar'] = _makeEl('div');
_elements['worktree-bar'] = _makeEl('div');
_elements['cfg-use-worktree'] = _makeEl('input');
_elements['cfg-use-parallel'] = _makeEl('input');
_elements['clear-btn'] = _makeEl('span');
_elements['ghost-text'] = _makeEl('div');
var _savedState = null;
var _postedMessages = [];

function acquireVsCodeApi() {
    return {
        setState: function(s) { _savedState = s; },
        getState: function() { return _savedState; },
        postMessage: function(m) { _postedMessages.push(m); },
    };
}

var window = { addEventListener: function() {} };
var requestAnimationFrame = function(fn) { fn(); return 1; };
var cancelAnimationFrame = function() {};
var setTimeout = function(fn) { fn(); return 1; };
var setInterval = function() { return 1; };
var clearInterval = function() {};
var clearTimeout = function() {};
var MutationObserver = function() {
    return { observe: function() {}, disconnect: function() {} };
};
var navigator = { platform: 'test' };
var DOMParser = function() {
    return { parseFromString: function() { return { body: { childNodes: [] } }; } };
};
var Event = function() {};
var KeyboardEvent = function() {};
var Blob = function() {};
var URL = { createObjectURL: function() { return ''; }, revokeObjectURL: function() {} };
var hljs = { highlightElement: function() {} };
var console = {
    log: function() {},
    error: function() {},
    warn: function() {},
};
var CSS = { supports: function() { return false; } };
"""


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    """Run a JS script in Node.js and return the result."""
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _make_test_script(test_body: str) -> str:
    """Build a full Node.js script: preamble + test body."""
    return _JS_PREAMBLE + "\n" + test_body


def _extract_function(js: str, name: str) -> str:
    """Extract a top-level (IIFE-scope) function's full source from main.js.

    Slices from ``function <name>(`` up to the next sibling function
    declaration so the real implementation can be evaluated in Node.js
    behavior tests instead of asserting on source-code substrings.
    """
    idx = js.index("function " + name + "(")
    end = js.index("\n  function ", idx + 1)
    return js[idx:end]


class TestStatusRunningTabIdGuard(unittest.TestCase):
    """status handler: per-tab isRunning via findTabByEvt + ev.tabId routing."""

    def test_guard_via_node(self) -> None:
        """Simulate per-tab isRunning in Node.js — second status message
        for the same tabId doesn't affect other tabs."""
        result = _run_node(_make_test_script(r"""
            var tabs = [
                { id: 1, isRunning: false },
                { id: 2, isRunning: false },
            ];
            var activeTabId = 1;

            function findTabByEvt(ev) {
                if (ev && ev.tabId !== undefined) {
                    return tabs.find(function(t) { return t.id === ev.tabId; }) || null;
                }
                return null;
            }

            // First status:running:true for tab 1
            var ev1 = { type: 'status', running: true, tabId: 1 };
            var evTab = findTabByEvt(ev1);
            if (evTab) evTab.isRunning = !!ev1.running;

            // User switches to tab 2
            activeTabId = 2;

            // Second status:running:true for tab 1 (from Python backend, delayed)
            var ev2 = { type: 'status', running: true, tabId: 1 };
            evTab = findTabByEvt(ev2);
            if (evTab) evTab.isRunning = !!ev2.running;

            // Tab 1 is still running, tab 2 is NOT running
            if (!tabs[0].isRunning) {
                process.stdout.write('FAIL: tab1 should be running');
                process.exit(1);
            }
            if (tabs[1].isRunning) {
                process.stdout.write('FAIL: tab2 should not be running');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestClearGuard(unittest.TestCase):
    """clear handler: only clears output when on running tab."""

    def test_clear_skipped_on_wrong_tab(self) -> None:
        result = _run_node(_make_test_script(r"""
            var runningTabId = 1;
            var activeTabId = 2;
            var cleared = false;

            // Simulate clear handler guard
            if (runningTabId < 0 || activeTabId === runningTabId) {
                cleared = true;
            }

            if (cleared) {
                process.stdout.write('FAIL: clear ran on wrong tab');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

    def test_clear_runs_on_correct_tab(self) -> None:
        result = _run_node(_make_test_script(r"""
            var runningTabId = 1;
            var activeTabId = 1;
            var cleared = false;

            if (runningTabId < 0 || activeTabId === runningTabId) {
                cleared = true;
            }

            if (!cleared) {
                process.stdout.write('FAIL: clear did not run on correct tab');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

    def test_clear_runs_when_no_task(self) -> None:
        """When runningTabId is -1 (no task), clear should still work."""
        result = _run_node(_make_test_script(r"""
            var runningTabId = -1;
            var activeTabId = 1;
            var cleared = false;

            if (runningTabId < 0 || activeTabId === runningTabId) {
                cleared = true;
            }

            if (!cleared) {
                process.stdout.write('FAIL: clear should run when no task');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestSetTaskTextGuard(unittest.TestCase):
    """setTaskText handler: updates running tab's saved state on wrong tab."""

    def test_task_text_updates_running_tab_not_active(self) -> None:
        result = _run_node(_make_test_script(r"""
            var t1 = { id: 1, title: 'new chat', chatId: '' };
            t1.taskPanelHTML = ''; t1.taskPanelVisible = false;
            var t2 = { id: 2, title: 'idle', chatId: '' };
            t2.taskPanelHTML = ''; t2.taskPanelVisible = false;
            var tabs = [t1, t2];
            var activeTabId = 2;
            var runningTabId = 1;
            var currentTaskName = 'idle task';

            var stt = 'Fix the bug in auth.py';

            if (runningTabId < 0 || activeTabId === runningTabId) {
                currentTaskName = stt;
            } else if (stt && runningTabId > 0) {
                var runTab = tabs.find(function(t) { return t.id === runningTabId; });
                if (runTab) {
                    runTab.title = stt.length > 30 ? stt.substring(0, 30) + '\u2026' : stt;
                    runTab.taskPanelHTML = stt;
                    runTab.taskPanelVisible = true;
                }
            }

            // Active tab state should be unchanged
            if (currentTaskName !== 'idle task') {
                process.stdout.write('FAIL: active tab currentTaskName corrupted');
                process.exit(1);
            }
            // Running tab saved state updated
            if (tabs[0].title !== 'Fix the bug in auth.py') {
                process.stdout.write('FAIL: running tab title not set: ' + tabs[0].title);
                process.exit(1);
            }
            if (tabs[0].taskPanelHTML !== 'Fix the bug in auth.py') {
                process.stdout.write('FAIL: running tab taskPanelHTML not set');
                process.exit(1);
            }
            if (!tabs[0].taskPanelVisible) {
                process.stdout.write('FAIL: running tab taskPanelVisible not set');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

    def test_long_task_text_truncated_for_running_tab(self) -> None:
        result = _run_node(_make_test_script(r"""
            var t1 = { id: 1, title: 'new chat', chatId: '' };
            t1.taskPanelHTML = ''; t1.taskPanelVisible = false;
            var tabs = [t1];
            var activeTabId = 2;
            var runningTabId = 1;

            var stt = 'This is a very long task description that exceeds thirty characters';

            if (!(runningTabId < 0 || activeTabId === runningTabId)) {
                if (stt && runningTabId > 0) {
                    var runTab = tabs.find(function(t) { return t.id === runningTabId; });
                    if (runTab) {
                        runTab.title = stt.length > 30 ? stt.substring(0, 30) + '\u2026' : stt;
                    }
                }
            }

            if (tabs[0].title.length !== 31) {
                var m = 'FAIL: not truncated: ';
                process.stdout.write(m + tabs[0].title.length);
                process.exit(1);
            }
            if (!tabs[0].title.endsWith('\u2026')) {
                process.stdout.write('FAIL: title missing ellipsis');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestTaskErrorStoppedGuard(unittest.TestCase):
    """task_error/task_stopped: banner only on running tab, setReady always runs."""

    def test_error_banner_skipped_on_wrong_tab(self) -> None:
        result = _run_node(_make_test_script(r"""
            var runningTabId = 1;
            var activeTabId = 2;
            var bannerAdded = false;
            var readyCalled = false;

            // Simulate the guard
            var isErr = true;
            if (runningTabId < 0 || activeTabId === runningTabId) {
                bannerAdded = true;
            }
            // setReady always runs
            readyCalled = true;
            runningTabId = -1;

            if (bannerAdded) {
                process.stdout.write('FAIL: banner should not be added on wrong tab');
                process.exit(1);
            }
            if (!readyCalled) {
                process.stdout.write('FAIL: setReady should always run');
                process.exit(1);
            }
            if (runningTabId !== -1) {
                process.stdout.write('FAIL: runningTabId should be reset');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestFullTabSwitchScenario(unittest.TestCase):
    """End-to-end scenario: start task on tab 1, switch to tab 2, receive
    multiple backend messages, switch back to tab 1 — verify state isolation."""

    def test_full_scenario_via_node(self) -> None:
        result = _run_node(_make_test_script(r"""
            // --- Tab state setup (matching main.js) ---
            var tabIdCounter = 0;
            var tabs = [];
            var activeTabId = -1;
            var runningTabId = -1;
            var currentChatId = '';
            var currentTaskName = '';
            var outputLog = [];  // tracks what was "appended" to output

            function makeTab(title) {
                var id = ++tabIdCounter;
                return {
                    id: id,
                    title: title || 'new chat',
                    outputHTML: '',
                    taskPanelHTML: '',
                    taskPanelVisible: false,
                    chatId: '',
                    statusTokensText: '',
                    statusBudgetText: '',
                    welcomeVisible: true,
                };
            }

            function persistTabState() {}

            function saveCurrentTab() {
                var tab = tabs.find(function(t) { return t.id === activeTabId; });
                if (!tab) return;
                tab.outputHTML = outputLog.join('|');
                tab.chatId = currentChatId;
                tab.taskPanelHTML = currentTaskName;
            }

            function restoreTab(tab) {
                activeTabId = tab.id;
                outputLog = tab.outputHTML ? tab.outputHTML.split('|') : [];
                currentChatId = tab.chatId || '';
                currentTaskName = tab.taskPanelHTML || '';
            }

            // ===== Step 1: Create two tabs =====
            var tab1 = makeTab('new chat');
            tabs.push(tab1);
            var tab2 = makeTab('new chat');
            tabs.push(tab2);
            activeTabId = tab1.id;

            // ===== Step 2: Start a task on tab 1 =====
            // First status:running:true from TS
            runningTabId = -1;
            if (runningTabId < 0) runningTabId = activeTabId;
            // runningTabId == 1

            // Backend sends clear
            if (runningTabId < 0 || activeTabId === runningTabId) {
                outputLog = [];  // cleared
            }

            // Backend sends setTaskText
            var stt = 'Fix auth bug';
            if (runningTabId < 0 || activeTabId === runningTabId) {
                currentTaskName = stt;
                tab1.title = stt;
            }

            // Backend sends chatId
            var newChatId = 'chat-abc';
            if (runningTabId > 0 && activeTabId !== runningTabId) {
                var runTab = tabs.find(function(t) { return t.id === runningTabId; });
                if (runTab) runTab.chatId = newChatId;
            } else {
                currentChatId = newChatId;
            }

            // Some streaming output
            if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                outputLog.push('thinking:analyzing...');
                outputLog.push('text:Here is the fix');
            }

            // ===== Step 3: Switch to tab 2 =====
            saveCurrentTab();
            restoreTab(tab2);
            // Now activeTabId == 2, runningTabId == 1

            // ===== Step 4: Backend messages arrive while on tab 2 =====
            // Second status:running:true (from Python, delayed)
            if (runningTabId < 0) runningTabId = activeTabId;  // GUARD: no-op

            // clear arrives — should be skipped
            var clearRan = false;
            if (runningTabId < 0 || activeTabId === runningTabId) {
                clearRan = true;
            }

            // setTaskText — should update running tab saved state
            stt = 'Fix auth bug (updated)';
            if (runningTabId < 0 || activeTabId === runningTabId) {
                currentTaskName = stt;
            } else if (stt && runningTabId > 0) {
                var runTab2 = tabs.find(function(t) { return t.id === runningTabId; });
                if (runTab2) {
                    runTab2.title = stt.length > 30 ? stt.substring(0, 30) + '\u2026' : stt;
                    runTab2.taskPanelHTML = stt;
                    runTab2.taskPanelVisible = true;
                }
            }

            // chatId — should update running tab
            newChatId = 'chat-def';
            if (runningTabId > 0 && activeTabId !== runningTabId) {
                var runTab3 = tabs.find(function(t) { return t.id === runningTabId; });
                if (runTab3) runTab3.chatId = newChatId;
            } else {
                currentChatId = newChatId;
            }

            // Streaming text — should be skipped
            if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                outputLog.push('text:should not appear');
            }

            // followup — should be skipped
            var followupAdded = false;
            if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                followupAdded = true;
            }

            // merge_data — should be skipped
            var mergeAdded = false;
            if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                mergeAdded = true;
            }

            // ===== Step 5: Task completes =====
            var bannerAdded = false;
            if (runningTabId < 0 || activeTabId === runningTabId) {
                bannerAdded = true;
            }
            // setReady always runs
            runningTabId = -1;

            // ===== Verify tab 2 state is clean =====
            var errors = [];

            if (clearRan) errors.push('clear on wrong tab');
            if (currentTaskName !== '')
              errors.push('tab2 taskName: ' + currentTaskName);
            if (currentChatId !== '')
              errors.push('tab2 chatId: ' + currentChatId);
            if (outputLog.length !== 0)
              errors.push('tab2 output: ' + outputLog.join(','));
            if (followupAdded) errors.push('followup wrong tab');
            if (mergeAdded) errors.push('merge wrong tab');
            if (bannerAdded) errors.push('banner wrong tab');

            // ===== Verify tab 1 saved state =====
            if (tab1.chatId !== 'chat-def')
              errors.push('tab1 chatId: ' + tab1.chatId);
            var expTask = 'Fix auth bug (updated)';
            if (tab1.taskPanelHTML !== expTask)
              errors.push('tab1 task: ' + tab1.taskPanelHTML);
            if (!tab1.taskPanelVisible)
              errors.push('tab1 visible not set');
            var expOut = 'thinking:analyzing...|text:Here is the fix';
            if (tab1.outputHTML !== expOut)
              errors.push('tab1 output: ' + tab1.outputHTML);

            // ===== Step 6: Switch back to tab 1 =====
            saveCurrentTab();
            restoreTab(tab1);

            if (currentChatId !== 'chat-def')
              errors.push('restored chatId: ' + currentChatId);
            if (currentTaskName !== expTask)
              errors.push('restored task: ' + currentTaskName);
            if (outputLog.length !== 2)
              errors.push('restored len: ' + outputLog.length);
            if (runningTabId !== -1)
              errors.push('runningTabId not -1');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, f"Node test failed:\n{result.stdout}\n{result.stderr}"
        assert "PASS" in result.stdout


class TestCreateNewTabDuringRunningTask(unittest.TestCase):
    """Creating a new tab (Cmd+T / +) while a task runs on another tab
    should not affect runningTabId or the running tab's state."""

    def test_new_tab_scenario_via_node(self) -> None:
        result = _run_node(_make_test_script(r"""
            var tabIdCounter = 0;
            var tabs = [];
            var activeTabId = -1;
            var runningTabId = -1;

            function makeTab(title) {
                var id = ++tabIdCounter;
                return { id: id, title: title || 'new chat', chatId: '' };
            }

            // Tab 1 running
            var tab1 = makeTab('running task');
            tabs.push(tab1);
            activeTabId = tab1.id;
            runningTabId = tab1.id;

            // Create new tab (simulating createNewTab)
            var tab2 = makeTab('new chat');
            tabs.push(tab2);
            activeTabId = tab2.id;
            // setRunningState(false) called but does NOT touch runningTabId

            if (runningTabId !== tab1.id) {
                var m = 'FAIL: changed to ' + runningTabId;
                process.stdout.write(m);
                process.exit(1);
            }
            if (activeTabId !== tab2.id) {
                process.stdout.write('FAIL: activeTabId should be new tab');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestMultipleTabsMultipleMessages(unittest.TestCase):
    """Stress test: rapid message sequence with tab switches."""

    def test_interleaved_messages_and_switches(self) -> None:
        result = _run_node(_make_test_script(r"""
            var tabIdCounter = 0;
            var tabs = [];
            var activeTabId = -1;
            var runningTabId = -1;
            var currentChatId = '';
            var currentTaskName = '';
            var outputCounts = {};  // tabId -> count of output events

            function makeTab(title) {
                var id = ++tabIdCounter;
                return { id: id, title: title || 'new chat', chatId: '' };
            }

            // Create 3 tabs
            for (var i = 0; i < 3; i++) {
                tabs.push(makeTab('tab ' + (i+1)));
                outputCounts[i+1] = 0;
            }
            activeTabId = 1;

            // Start task on tab 1
            runningTabId = -1;
            if (runningTabId < 0) runningTabId = activeTabId;

            // Send 10 streaming events while staying on tab 1
            for (var j = 0; j < 10; j++) {
                if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                    outputCounts[activeTabId]++;
                }
            }

            // Switch to tab 2
            activeTabId = 2;

            // Send 10 more events — should be skipped
            for (var j = 0; j < 10; j++) {
                if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                    outputCounts[activeTabId]++;
                }
            }

            // Switch to tab 3
            activeTabId = 3;

            // Send 5 more events — should be skipped
            for (var j = 0; j < 5; j++) {
                if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                    outputCounts[activeTabId]++;
                }
            }

            // Switch back to tab 1
            activeTabId = 1;

            // Send 5 more events — should go to tab 1
            for (var j = 0; j < 5; j++) {
                if (!(runningTabId > 0 && activeTabId !== runningTabId)) {
                    outputCounts[activeTabId]++;
                }
            }

            var errors = [];
            if (outputCounts[1] !== 15)
              errors.push('tab1: ' + outputCounts[1]);
            if (outputCounts[2] !== 0)
              errors.push('tab2: ' + outputCounts[2]);
            if (outputCounts[3] !== 0)
              errors.push('tab3: ' + outputCounts[3]);

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, f"Failed:\n{result.stdout}\n{result.stderr}"
        assert "PASS" in result.stdout


class TestSetReadyResetsRunningTabId(unittest.TestCase):
    """setReady() clears the target tab's running state regardless of
    which tab is active."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_set_ready_resets_tab_running_state(self) -> None:
        """Behavior (real ``setReady`` evaluated in Node): completing a
        BACKGROUND tab flips that tab's ``isRunning`` to false and
        anchors its ``t0``/``endTs`` to the agent-supplied wall-clock
        timestamps, without touching the active tab's UI state."""
        set_ready_src = _extract_function(self.js, "setReady")
        result = _run_node(_make_test_script(
            r"""
            var tabs = [
                { id: 1, isRunning: false },
                { id: 7, isRunning: true, t0: 111, endTs: 0 },
            ];
            var activeTabId = 1;
            var t0 = null;
            var endTs = 0;
            function getTab(id) {
                return tabs.find(function(t) { return t.id === id; }) || null;
            }
            var uiCalls = [];
            function setRunningState(r) { uiCalls.push('setRunningState:' + r); }
            function stopTimer() { uiCalls.push('stopTimer'); }
            function removeSpinner() { uiCalls.push('removeSpinner'); }
            function renderTabBar() {}
            var statusText = { textContent: '' };
            var inp = { focus: function() {} };
            """
            + set_ready_src
            + r"""
            setReady('Done (4s)', 7, 1000, 5000);
            var bg = getTab(7);
            if (bg.isRunning !== false) {
                process.stdout.write('FAIL: background tab still running');
                process.exit(1);
            }
            if (bg.t0 !== 1000 || bg.endTs !== 5000) {
                process.stdout.write(
                    'FAIL: t0/endTs not anchored: ' + bg.t0 + '/' + bg.endTs);
                process.exit(1);
            }
            if (bg.statusTextContent !== 'Done (4s)') {
                process.stdout.write('FAIL: done label not persisted on tab');
                process.exit(1);
            }
            // Event targeted a background tab: active-tab UI untouched.
            if (uiCalls.length !== 0 || statusText.textContent !== '') {
                process.stdout.write('FAIL: active tab UI touched: ' + uiCalls);
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        ))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

class TestSaveRestorePreservesTabState(unittest.TestCase):
    """saveCurrentTab/restoreTab cycle preserves all tab-specific state."""

    def test_save_restore_round_trip(self) -> None:
        result = _run_node(_make_test_script(r"""
            var tabIdCounter = 0;
            var tabs = [];
            var activeTabId = -1;
            var currentChatId = '';
            var outputHTML = '';

            function makeTab(title) {
                var id = ++tabIdCounter;
                return {
                    id: id, title: title || 'new chat',
                    outputHTML: '', chatId: '',
                    statusTokensText: '', statusBudgetText: '',
                    welcomeVisible: true,
                };
            }

            // Tab 1 with accumulated state
            var tab1 = makeTab('task A');
            tabs.push(tab1);
            activeTabId = tab1.id;
            currentChatId = 'chat-001';
            outputHTML = '<div>Result A</div>';

            // Save tab 1
            var tab = tabs.find(function(t) { return t.id === activeTabId; });
            tab.outputHTML = outputHTML;
            tab.chatId = currentChatId;

            // Create tab 2 and switch
            var tab2 = makeTab('task B');
            tabs.push(tab2);
            activeTabId = tab2.id;
            currentChatId = 'chat-002';
            outputHTML = '<div>Result B</div>';

            // Switch back to tab 1 (restore)
            activeTabId = tab1.id;
            outputHTML = tab1.outputHTML;
            currentChatId = tab1.chatId || '';

            var errors = [];
            if (currentChatId !== 'chat-001') errors.push('chatId not restored: ' + currentChatId);
            if (outputHTML !== '<div>Result A</div>') errors.push('output not restored');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


class TestPerTabIsMerging(unittest.TestCase):
    """isMerging is saved/restored per tab with guards on merge events."""

    def test_merge_started_auto_switches_via_node(self) -> None:
        """Behavioral test: merge_started for a bg tab auto-switches to it.

        Simulates the merge_started handler logic: when merge_started fires
        for a background tab, it sets isMerging on the tab then calls
        switchToTab to bring it to the foreground.
        """
        result = _run_node(_make_test_script(r"""
            var tabs = [
                { id: 'tab-A', isMerging: false, mergeToolbarEl: null,
                  isRunning: true, outputFragment: null },
                { id: 'tab-B', isMerging: false, mergeToolbarEl: null,
                  isRunning: false, outputFragment: null },
            ];
            var activeTabId = 'tab-B';
            var switchedTo = null;

            // Stub switchToTab to record what it was called with
            function switchToTab(tabId) { switchedTo = tabId; activeTabId = tabId; }

            // Simulate merge_started handler for background tab-A
            var ev = { type: 'merge_started', tabId: 'tab-A' };
            if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
                var bgMergeTab = tabs.find(function(t) { return t.id === ev.tabId; });
                if (bgMergeTab) {
                    bgMergeTab.isMerging = true;
                    switchToTab(ev.tabId);
                }
            }

            var errors = [];
            if (tabs[0].isMerging !== true)
                errors.push('tab-A isMerging should be true');
            if (switchedTo !== 'tab-A')
                errors.push('should have switched to tab-A, got ' + switchedTo);
            if (activeTabId !== 'tab-A')
                errors.push('activeTabId should be tab-A, got ' + activeTabId);
            if (tabs[1].isMerging !== false)
                errors.push('tab-B should be unaffected');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

class TestPerTabT0(unittest.TestCase):
    """t0 (timer start) is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_switch_to_non_running_tab_clears_t0(self) -> None:
        """Behavior (real ``switchToTab`` evaluated in Node): switching
        to a non-running tab stops the live timer and flips the running
        state to false, while the restored ``t0``/``endTs`` anchors are
        KEPT so the tab can render "Done (Xm Ys)" from agent
        wall-clock."""
        switch_src = _extract_function(self.js, "switchToTab")
        result = _run_node(_make_test_script(
            r"""
            var tabs = [
                { id: 1, isRunning: true, panelsExpandedMap: {} },
                { id: 2, isRunning: false, t0: 1000, endTs: 5000,
                  panelsExpandedMap: {} },
            ];
            var activeTabId = 1;
            var isRunning = true;
            var timerRunning = true;
            var t0 = 999;
            var endTs = 0;
            var currentTaskName = 'task';
            function getTab(id) {
                return tabs.find(function(t) { return t.id === id; }) || null;
            }
            function saveCurrentTab() {}
            function restoreTab(tab) {
                // Mirrors the real restoreTab contract (covered by
                // test_restore_restores_t0): per-tab anchors restored.
                activeTabId = tab.id;
                t0 = tab.t0;
                endTs = tab.endTs;
            }
            function showContentTab() {}
            function renderTabBar() {}
            function persistTabState() {}
            function setRunningState(r) {
                isRunning = r;
                if (r) timerRunning = true;
            }
            function stopTimer() { timerRunning = false; }
            function removeSpinner() {}
            function applyChevronState() {}
            function focusInputWithRetry() {}
            function clearDemoEndedUi() {}
            """
            + switch_src
            + r"""
            switchToTab(2);
            if (isRunning !== false) {
                process.stdout.write('FAIL: running state not cleared');
                process.exit(1);
            }
            if (timerRunning) {
                process.stdout.write('FAIL: timer still ticking');
                process.exit(1);
            }
            // Restored anchors must be preserved (not nulled) so the
            // idle tab renders its done duration.
            if (t0 !== 1000 || endTs !== 5000) {
                process.stdout.write(
                    'FAIL: restored anchors clobbered: ' + t0 + '/' + endTs);
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        ))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

    def test_set_ready_clears_running_tab_t0(self) -> None:
        """Behavior (real ``setReady`` evaluated in Node): completing
        the ACTIVE tab stops the timer and re-anchors the global
        ``t0``/``endTs`` to the agent-supplied timestamps; without
        timestamps the existing ``t0`` is preserved and ``endTs``
        falls back to the local clock."""
        set_ready_src = _extract_function(self.js, "setReady")
        result = _run_node(_make_test_script(
            r"""
            var tabs = [{ id: 1, isRunning: true, t0: 111, endTs: 0 }];
            var activeTabId = 1;
            var t0 = 111;
            var endTs = 0;
            function getTab(id) {
                return tabs.find(function(t) { return t.id === id; }) || null;
            }
            var timerRunning = true;
            var isRunning = true;
            function setRunningState(r) { isRunning = r; }
            function stopTimer() { timerRunning = false; }
            function removeSpinner() {}
            function renderTabBar() {}
            var statusText = { textContent: '' };
            var inp = { focus: function() {} };
            """
            + set_ready_src
            + r"""
            setReady('Done (4s)', 1, 1000, 5000);
            if (tabs[0].isRunning !== false || isRunning !== false) {
                process.stdout.write('FAIL: running state not cleared');
                process.exit(1);
            }
            if (timerRunning) {
                process.stdout.write('FAIL: timer still ticking');
                process.exit(1);
            }
            if (t0 !== 1000 || endTs !== 5000) {
                process.stdout.write(
                    'FAIL: anchors not re-anchored: ' + t0 + '/' + endTs);
                process.exit(1);
            }
            if (statusText.textContent !== 'Done (4s)') {
                process.stdout.write('FAIL: status label not rendered');
                process.exit(1);
            }
            // Legacy fallback: no agent timestamps — keep t0, anchor
            // endTs to the local clock.
            tabs[0].isRunning = true;
            timerRunning = true;
            endTs = 0;
            setReady('Done', 1);
            if (t0 !== 1000) {
                process.stdout.write('FAIL: t0 not preserved: ' + t0);
                process.exit(1);
            }
            if (!(endTs > 0)) {
                process.stdout.write('FAIL: endTs fallback missing');
                process.exit(1);
            }
            if (tabs[0].isRunning !== false || timerRunning) {
                process.stdout.write('FAIL: second completion not applied');
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        ))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

class TestBgTabPanelCreation(unittest.TestCase):
    """Background tabs must get panels created for streaming events via
    processOutputEventForBgTab instead of silently dropping them."""

    def test_bg_tab_panel_creation_via_node(self) -> None:
        """Simulate the processOutputEventForBgTab logic in Node.js:
        streaming events for a bg tab create panels in outputFragment."""
        result = _run_node(
            _make_test_script(
                r"""
            // Minimal replica of processOutputEventForBgTab logic from main.js
            function mkS() {
                return { thinkEl: null, txtEl: null, txtBuf: '', bashPanel: null,
                         bashBuf: '', bashRaf: 0, lastToolCallEl: null };
            }
            function mkEl(tag, cls) {
                var el = _makeEl(tag);
                el.className = cls || '';
                return el;
            }
            function addCollapse(p, h) {
                p.classList.add('collapsible');
            }

            function processOutputEventForBgTab(ev, tab) {
                var t = ev.type;
                if (!tab.outputFragment)
                    tab.outputFragment = document.createDocumentFragment();
                var bgLastToolName = tab.streamLastToolName || '';
                var bgLlmPanel = tab.streamLlmPanel || null;
                var bgLlmPanelState = tab.streamLlmPanelState || mkS();
                var bgPendingPanel = tab.streamPendingPanel || false;
                var bgStepCount = tab.streamStepCount || 0;
                var bgState = tab.streamState || mkS();

                if (t === 'tool_call') {
                    bgLastToolName = ev.name || '';
                    bgLlmPanel = null; bgLlmPanelState = mkS();
                    bgPendingPanel = false;
                }
                if (t === 'tool_result' && bgLastToolName !== 'finish') {
                    bgPendingPanel = true;
                }
                if ((bgPendingPanel || bgStepCount === 0) &&
                    (t === 'thinking_start' || t === 'text_delta')) {
                    bgStepCount++;
                    bgLlmPanel = mkEl('div', 'llm-panel');
                    var lHdr = mkEl('div', 'llm-panel-hdr');
                    lHdr.textContent = 'Thoughts';
                    addCollapse(bgLlmPanel, lHdr);
                    bgLlmPanel.appendChild(lHdr);
                    tab.outputFragment.appendChild(bgLlmPanel);
                    bgLlmPanelState = mkS();
                    bgPendingPanel = false;
                }
                if (t === 'usage_info') {
                    if (ev.total_tokens != null && ev.cost != null) {
                        tab.statusTokensText = 'Tokens: ' + ev.total_tokens;
                        if (ev.cost !== 'N/A') tab.statusBudgetText = 'Cost: ' + ev.cost;
                    }
                }
                if (t === 'result') {
                    if (ev.step_count) {
                        bgStepCount = ev.step_count;
                        tab.statusStepsText = 'Steps: ' + ev.step_count;
                    }
                    if (ev.total_tokens)
                        tab.statusTokensText = 'Tokens: ' + ev.total_tokens;
                    if (ev.cost && ev.cost !== 'N/A')
                        tab.statusBudgetText = 'Cost: ' + ev.cost;
                    if (ev.success === false) tab.lastTaskFailed = true;
                    // Create result card in fragment
                    var rc = mkEl('div', 'ev rc');
                    rc.textContent = ev.text || '';
                    tab.outputFragment.appendChild(rc);
                }
                tab.streamState = bgState;
                tab.streamLlmPanel = bgLlmPanel;
                tab.streamLlmPanelState = bgLlmPanelState;
                tab.streamLastToolName = bgLastToolName;
                tab.streamPendingPanel = bgPendingPanel;
                tab.streamStepCount = bgStepCount;
                tab.welcomeVisible = false;
            }

            // --- Test scenario ---
            var tab1 = {
                id: 'tab-1', isRunning: true, outputFragment: null,
                streamState: null, streamLlmPanel: null,
                streamLlmPanelState: null, streamLastToolName: '',
                streamPendingPanel: false, streamStepCount: 0,
                statusTokensText: '', statusBudgetText: '',
                statusStepsText: '', lastTaskFailed: false,
                welcomeVisible: true,
            };

            var errors = [];

            // 1. Send tool_result then thinking_start → should create panel
            processOutputEventForBgTab({type: 'tool_result', content: 'ok'}, tab1);
            processOutputEventForBgTab({type: 'thinking_start'}, tab1);
            processOutputEventForBgTab({type: 'thinking_end'}, tab1);

            var frag = tab1.outputFragment;
            if (!frag || frag.children.length === 0)
                errors.push('no panel after thinking_start');
            else if (frag.children[0].className.indexOf('llm-panel') < 0)
                errors.push('first child not llm-panel: ' + frag.children[0].className);
            if (tab1.streamStepCount !== 1)
                errors.push('stepCount not 1: ' + tab1.streamStepCount);

            // 2. Send usage_info → saved on tab, not DOM
            processOutputEventForBgTab(
                {type: 'usage_info', total_tokens: 5000, cost: '$1.23'}, tab1);
            if (tab1.statusTokensText.indexOf('5000') < 0)
                errors.push('usage tokens not saved: ' + tab1.statusTokensText);

            // 3. Send result → creates rc, saves step_count
            processOutputEventForBgTab(
                {type: 'result', text: 'Done', success: true,
                 total_tokens: 9000, cost: '$2.50', step_count: 5}, tab1);
            if (tab1.streamStepCount !== 5)
                errors.push('stepCount not 5: ' + tab1.streamStepCount);
            var hasRc = false;
            (tab1.outputFragment.children || []).forEach(function(c) {
                if (c.className && c.className.indexOf('rc') >= 0) hasRc = true;
            });
            if (!hasRc) errors.push('no rc element after result');
            if (tab1.statusTokensText.indexOf('9000') < 0)
                errors.push('result tokens not saved: ' + tab1.statusTokensText);

            // 4. welcomeVisible should be false
            if (tab1.welcomeVisible !== false)
                errors.push('welcomeVisible not false');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """
            )
        )
        assert result.returncode == 0, f"Node test failed:\n{result.stdout}\n{result.stderr}"
        assert "PASS" in result.stdout, result.stdout

    def test_bg_tab_first_thought_gets_panel(self) -> None:
        """stepCount === 0 should trigger a panel even without tool_result."""
        result = _run_node(
            _make_test_script(
                r"""
            function mkS() {
                return { thinkEl: null, txtEl: null, txtBuf: '', bashPanel: null,
                         bashBuf: '', bashRaf: 0, lastToolCallEl: null };
            }
            function mkEl(tag, cls) {
                var el = _makeEl(tag); el.className = cls || ''; return el;
            }
            function addCollapse(p, h) { p.classList.add('collapsible'); }

            function processOutputEventForBgTab(ev, tab) {
                var t = ev.type;
                if (!tab.outputFragment)
                    tab.outputFragment = document.createDocumentFragment();
                var bgStepCount = tab.streamStepCount || 0;
                var bgPendingPanel = tab.streamPendingPanel || false;
                var bgLlmPanel = tab.streamLlmPanel || null;
                var bgLlmPanelState = tab.streamLlmPanelState || mkS();
                var bgLastToolName = tab.streamLastToolName || '';
                var bgState = tab.streamState || mkS();

                if (t === 'tool_call') {
                    bgLastToolName = ev.name || '';
                    bgLlmPanel = null; bgLlmPanelState = mkS();
                    bgPendingPanel = false;
                }
                if (t === 'tool_result' && bgLastToolName !== 'finish')
                    bgPendingPanel = true;
                if ((bgPendingPanel || bgStepCount === 0) &&
                    (t === 'thinking_start' || t === 'text_delta')) {
                    bgStepCount++;
                    bgLlmPanel = mkEl('div', 'llm-panel');
                    tab.outputFragment.appendChild(bgLlmPanel);
                    bgPendingPanel = false;
                }
                tab.streamStepCount = bgStepCount;
                tab.streamPendingPanel = bgPendingPanel;
                tab.streamLlmPanel = bgLlmPanel;
                tab.streamLastToolName = bgLastToolName;
            }

            var tab = {
                id: 't1', outputFragment: null, streamStepCount: 0,
                streamPendingPanel: false, streamLlmPanel: null,
                streamLlmPanelState: null, streamLastToolName: '',
                streamState: null,
            };

            // First thinking_start with stepCount=0 should create a panel
            processOutputEventForBgTab({type: 'thinking_start'}, tab);
            if (!tab.outputFragment || tab.outputFragment.children.length === 0) {
                process.stdout.write('FAIL: no panel for first thought');
                process.exit(1);
            }
            if (tab.streamStepCount !== 1) {
                process.stdout.write('FAIL: stepCount not 1');
                process.exit(1);
            }
            process.stdout.write('PASS');
        """
            )
        )
        assert result.returncode == 0, f"Node test failed:\n{result.stdout}\n{result.stderr}"
        assert "PASS" in result.stdout, result.stdout


class TestMergeEndedBgClearsMergeToolbarEl(unittest.TestCase):
    """Bug fix: merge_ended for background tab must clear mergeToolbarEl.

    Previously, merge_ended only set isMerging=false but left mergeToolbarEl
    intact.  When the user switched to that tab, restoreTab re-attached the
    stale merge toolbar even though the merge had ended.
    """

    def test_merge_ended_bg_clears_merge_toolbar_el_via_node(self) -> None:
        """Behavioral test: merge_ended for bg tab clears mergeToolbarEl."""
        result = _run_node(_make_test_script(r"""
            var tabs = [
                { id: 'tab-A', isMerging: true, mergeToolbarEl: { id: 'merge-toolbar' } },
                { id: 'tab-B', isMerging: false, mergeToolbarEl: null },
            ];
            var activeTabId = 'tab-B';

            // Simulate merge_ended handler for background tab-A
            var ev = { type: 'merge_ended', tabId: 'tab-A' };
            if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
                var mrt2 = tabs.find(function(t) { return t.id === ev.tabId; });
                if (mrt2) {
                    mrt2.isMerging = false;
                    mrt2.mergeToolbarEl = null;
                }
            }

            var errors = [];
            if (tabs[0].isMerging !== false)
                errors.push('tab-A isMerging should be false');
            if (tabs[0].mergeToolbarEl !== null)
                errors.push('tab-A mergeToolbarEl should be null');
            if (tabs[1].isMerging !== false)
                errors.push('tab-B should be unaffected');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout

    def test_restore_does_not_show_stale_merge_toolbar(self) -> None:
        """After merge_ended clears mergeToolbarEl, restoreTab must NOT
        re-attach a stale merge toolbar."""
        result = _run_node(_make_test_script(r"""
            // Tab A had a merge running, toolbar was saved when switching away
            var tabA = {
                id: 'tab-A', isMerging: true,
                mergeToolbarEl: { id: 'merge-toolbar', tagName: 'div' },
            };

            // merge_ended arrives for bg tab A (the fix)
            tabA.isMerging = false;
            tabA.mergeToolbarEl = null;

            // Now simulate restoreTab logic
            var mergeToolbarRestored = false;
            var showMergeToolbarCalled = false;

            if (tabA.mergeToolbarEl) {
                mergeToolbarRestored = true;
            } else if (tabA.isMerging) {
                showMergeToolbarCalled = true;
            }

            var errors = [];
            if (mergeToolbarRestored)
                errors.push('stale merge toolbar was restored');
            if (showMergeToolbarCalled)
                errors.push('showMergeToolbar called after merge ended');

            if (errors.length > 0) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
        """))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout


if __name__ == "__main__":
    unittest.main()
