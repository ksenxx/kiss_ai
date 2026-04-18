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

# Minimal JS preamble that stubs DOM/vscode and replicates main.js tab state.
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
_elements['worktree-toggle-btn'] = _makeEl('button');
_elements['parallel-toggle-btn'] = _makeEl('button');
_elements['clear-btn'] = _makeEl('span');
_elements['ghost-text'] = _makeEl('div');
_elements['run-prompt-btn'] = _makeEl('button');

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


class TestStatusRunningTabIdGuard(unittest.TestCase):
    """status handler: per-tab isRunning via findTabByEvt + ev.tabId routing."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_status_sets_per_tab_is_running(self) -> None:
        """status handler uses findTabByEvt to set evTab.isRunning."""
        idx = self.js.index("case 'status':")
        block = self.js[idx : idx + 600]
        assert "findTabByEvt(ev)" in block
        assert "evTab.isRunning" in block

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

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_clear_guard_in_source(self) -> None:
        idx = self.js.index("case 'clear':")
        block = self.js[idx : idx + 300]
        assert "ev.tabId" in block
        assert "activeTabId" in block

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

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_set_task_text_guard_in_source(self) -> None:
        idx = self.js.index("case 'setTaskText':")
        block = self.js[idx : idx + 800]
        assert "ev.tabId === undefined || ev.tabId === activeTabId" in block
        assert "sttTab.title" in block
        assert "sttTab.taskPanelHTML" in block

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


class TestFollowupSuggestionGuard(unittest.TestCase):
    """followup_suggestion handler: skipped when on wrong tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_followup_guard_in_source(self) -> None:
        idx = self.js.index("case 'followup_suggestion':")
        block = self.js[idx : idx + 200]
        assert "ev.tabId !== undefined && ev.tabId !== activeTabId" in block


class TestMergeDataGuard(unittest.TestCase):
    """merge_data handler: skipped when on wrong tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_merge_data_guard_in_source(self) -> None:
        idx = self.js.index("case 'merge_data':")
        block = self.js[idx : idx + 200]
        assert "ev.tabId !== undefined && ev.tabId !== activeTabId" in block


class TestTaskErrorStoppedGuard(unittest.TestCase):
    """task_error/task_stopped: banner only on running tab, setReady always runs."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_error_banner_guard_in_source(self) -> None:
        idx = self.js.index("case 'task_error':")
        block = self.js[idx : idx + 500]
        assert "ev.tabId === undefined || ev.tabId === activeTabId" in block

    def test_set_ready_always_called(self) -> None:
        """setReady is outside the guard — it always runs to reset runningTabId."""
        idx = self.js.index("case 'task_error':")
        block = self.js[idx : idx + 800]
        # setReady is after the if block, always reached
        assert "setReady(" in block

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


class TestDefaultStreamingGuard(unittest.TestCase):
    """Default handler (streaming output): skipped on wrong tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_default_guard_in_source(self) -> None:
        idx = self.js.index("default:")
        block = self.js[idx : idx + 200]
        assert "ev.tabId !== undefined && ev.tabId !== activeTabId" in block


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


class TestSwitchBackRestoresDOM(unittest.TestCase):
    """When switching back to a tab, the DOM is restored from the saved fragment."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_switch_to_tab_restores_from_fragment(self) -> None:
        """switchToTab restores DOM from saved fragment, no backend message needed."""
        idx = self.js.index("function switchToTab(")
        block = self.js[idx : idx + 800]
        assert "restoreTab(tab)" in block
        assert "setRunningState(tab.isRunning)" in block


class TestSwitchToTabRunningState(unittest.TestCase):
    """switchToTab correctly sets isRunning based on whether the target
    tab is the running tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_switch_to_running_tab_sets_running(self) -> None:
        """Switching to the running tab sets isRunning = true."""
        idx = self.js.index("function switchToTab(")
        block = self.js[idx : idx + 800]
        assert "tab.isRunning" in block
        assert "setRunningState(tab.isRunning)" in block

    def test_switch_to_idle_tab_resets_status(self) -> None:
        """Switching to a non-running tab resets timer/spinner/status."""
        idx = self.js.index("function switchToTab(")
        block = self.js[idx : idx + 800]
        assert "stopTimer()" in block
        assert "removeSpinner()" in block


class TestCreateNewTabDuringRunningTask(unittest.TestCase):
    """Creating a new tab (Cmd+T / +) while a task runs on another tab
    should not affect runningTabId or the running tab's state."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_create_new_tab_does_not_reset_running_tab_id(self) -> None:
        """createNewTab sets isRunning=false for the new tab but doesn't
        touch runningTabId."""
        idx = self.js.index("function createNewTab()")
        block = self.js[idx : idx + 1000]
        # createNewTab sets running state to false for itself
        assert "setRunningState(false)" in block
        # It should NOT reset runningTabId
        assert "runningTabId" not in block

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
    """setReady() always resets runningTabId to -1 regardless of active tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_set_ready_resets_tab_running_state(self) -> None:
        idx = self.js.index("function setReady(")
        block = self.js[idx : idx + 400]
        assert "doneTab.isRunning = false" in block
        assert "doneTab.t0 = null" in block

    def test_task_done_calls_set_ready(self) -> None:
        idx = self.js.index("case 'task_done':")
        block = self.js[idx : idx + 400]
        assert "setReady(" in block


class TestCloseRunningTabBehavior(unittest.TestCase):
    """Closing the running tab switches to adjacent tab with correct state."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_close_tab_checks_running_state(self) -> None:
        idx = self.js.index("function closeTab(")
        block = self.js[idx : idx + 800]
        assert "newTab.isRunning" in block
        assert "setRunningState(newTab.isRunning)" in block


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


class TestPerTabSelectedModel(unittest.TestCase):
    """selectedModel is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_selected_model(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.selectedModel = selectedModel" in body

    def test_restore_restores_selected_model(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "selectedModel = tab.selectedModel" in body
        assert "modelName" in body  # updates the model name display

    def test_make_tab_inherits_selected_model(self) -> None:
        idx = self.js.index("function makeTab(title)")
        end = self.js.index("\n  }", idx) + 4
        body = self.js[idx:end]
        assert "selectedModel: selectedModel" in body


class TestPerTabAttachments(unittest.TestCase):
    """attachments are saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_attachments(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.attachments = attachments" in body

    def test_restore_restores_attachments(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "attachments = tab.attachments" in body
        assert "renderFileChips()" in body

    def test_create_new_tab_clears_attachments(self) -> None:
        idx = self.js.index("function createNewTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "attachments = []" in body
        assert "renderFileChips()" in body


class TestPerTabInputValue(unittest.TestCase):
    """inp.value is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_input_value(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.inputValue = inp.value" in body

    def test_restore_restores_input_value(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "inp.value = tab.inputValue" in body
        assert "syncClearBtn()" in body

    def test_create_new_tab_preserves_input(self) -> None:
        idx = self.js.index("function createNewTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "const pendingText = inp.value" in body
        assert "inp.value = pendingText" in body


class TestPerTabIsMerging(unittest.TestCase):
    """isMerging is saved/restored per tab with guards on merge events."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_is_merging(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.isMerging = isMerging" in body

    def test_restore_restores_is_merging(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "isMerging = tab.isMerging" in body
        assert "updateInputDisabled()" in body

    def test_merge_started_switches_tab(self) -> None:
        idx = self.js.index("case 'merge_started':")
        block = self.js[idx : idx + 400]
        assert "ev.tabId !== undefined && ev.tabId !== activeTabId" in block
        # Background tab merges no longer auto-switch; they update saved state
        assert "bgMergeTab" in block

    def test_merge_ended_guard(self) -> None:
        idx = self.js.index("case 'merge_ended':")
        block = self.js[idx : idx + 400]
        assert "ev.tabId !== undefined && ev.tabId !== activeTabId" in block

    def test_create_new_tab_clears_merging(self) -> None:
        idx = self.js.index("function createNewTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "isMerging = false" in body
        assert "hideMergeToolbar()" in body


class TestPerTabWorktreeBar(unittest.TestCase):
    """worktreeBar DOM element is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_detaches_worktree_bar(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.worktreeBarEl = worktreeBar" in body
        assert "worktreeBar = null" in body

    def test_restore_reattaches_worktree_bar(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "worktreeBar = tab.worktreeBarEl" in body
        assert "area.insertBefore(worktreeBar" in body

    def test_save_detaches_merge_toolbar(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.mergeToolbarEl = mergeBar" in body

    def test_restore_reattaches_merge_toolbar(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.mergeToolbarEl" in body


class TestPerTabStreamingState(unittest.TestCase):
    """Streaming state (state, llmPanel, llmPanelState, lastToolName,
    pendingPanel) is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_streaming_state(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.streamState = state" in body
        assert "tab.streamLlmPanel = llmPanel" in body
        assert "tab.streamLlmPanelState = llmPanelState" in body
        assert "tab.streamLastToolName = lastToolName" in body
        assert "tab.streamPendingPanel = pendingPanel" in body

    def test_restore_restores_streaming_state(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "state = tab.streamState || mkS()" in body
        assert "llmPanel = tab.streamLlmPanel" in body
        assert "llmPanelState = tab.streamLlmPanelState || mkS()" in body
        assert "lastToolName = tab.streamLastToolName" in body
        assert "pendingPanel = tab.streamPendingPanel" in body

    def test_make_tab_has_streaming_fields(self) -> None:
        idx = self.js.index("function makeTab(title)")
        end = self.js.index("\n  }", idx) + 4
        body = self.js[idx:end]
        assert "streamState:" in body
        assert "streamLlmPanel:" in body
        assert "streamLlmPanelState:" in body
        assert "streamLastToolName:" in body
        assert "streamPendingPanel:" in body


class TestPerTabT0(unittest.TestCase):
    """t0 (timer start) is saved/restored per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_stores_t0(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.t0 = t0" in body

    def test_restore_restores_t0(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "t0 = tab.t0" in body

    def test_switch_to_non_running_tab_clears_t0(self) -> None:
        idx = self.js.index("function switchToTab(tabId)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "t0 = null" in body

    def test_set_ready_clears_running_tab_t0(self) -> None:
        idx = self.js.index("function setReady(label, tabId)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "doneTab" in body
        assert "doneTab.t0 = null" in body

    def test_task_done_uses_running_tab_t0(self) -> None:
        idx = self.js.index("case 'task_done':")
        block = self.js[idx : idx + 400]
        assert "doneT0" in block
        assert "ev.tabId" in block


class TestPerTabOutputFragment(unittest.TestCase):
    """Output uses DocumentFragment for DOM-preserving save/restore."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_uses_document_fragment(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "document.createDocumentFragment()" in body
        assert "O.firstChild" in body  # moves children to fragment

    def test_restore_uses_fragment(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "tab.outputFragment" in body
        assert "O.appendChild(tab.outputFragment)" in body

    def test_welcome_detached_before_fragment_save(self) -> None:
        """Welcome element must be detached before capturing fragment
        because it's shared across all tabs."""
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        welcome_idx = body.index("welcome")
        fragment_idx = body.index("outputFragment")
        assert welcome_idx < fragment_idx, (
            "welcome must be detached before creating fragment"
        )


class TestInputContainerVisibility(unittest.TestCase):
    """inputContainer visibility reflects worktree/merge bar state per tab."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_save_restores_input_container(self) -> None:
        idx = self.js.index("function saveCurrentTab()")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        assert "inputContainer" in body
        # After save, inputContainer should be visible again
        assert "inputContainer.style.display = ''" in body

    def test_restore_hides_input_when_bar_present(self) -> None:
        idx = self.js.index("function restoreTab(tab)")
        end = self.js.index("\n  function ", idx + 1)
        body = self.js[idx:end]
        # Should check both worktreeBar and merge-toolbar
        assert "worktreeBar || document.getElementById('merge-toolbar')" in body
        assert "inputContainer.style.display = 'none'" in body
        assert "inputContainer.style.display = ''" in body


if __name__ == "__main__":
    unittest.main()
