"""Integration tests for the Running sidebar tab.

The chat webview gained a fourth in-panel sidebar tab — *Running* —
which is the first tab in order and shows every currently running
regular (non-sub-agent) task as a coloured panel.  The frontend
polls the backend every 5s while the tab is visible to refresh the
list and the per-task usage metrics.  Tabs that have finished since
the last poll are implicitly removed because the list is fully
rebuilt on every response.

This module exercises:

* The backend ``getRunningTasks`` command handler / dispatcher.
* :meth:`VSCodeServer._get_running_tasks` enumerating live tabs and
  collecting ``tokens`` / ``cost`` / ``steps`` from each tab's
  ``agent`` object.
* Sub-agent tabs being excluded from the running list (sub-agent
  rows are filtered out of the History sidebar too).
* The new ``getRunningTasks`` -> ``runningTasks`` round-trip
  broadcasts the expected shape.
* The frontend ``renderRunningTasks`` function paints one
  ``.sidebar-item`` per task with a chat-id-derived background
  colour and a metrics span.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import unittest
from pathlib import Path

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

_MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents" / "vscode" / "media" / "main.js"
)


class _CapturingPrinter(BaseBrowserPrinter):
    """Printer that records every ``broadcast`` payload."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def broadcast(self, event: dict) -> None:  # type: ignore[override]
        self.events.append(dict(event))
        super().broadcast(event)


class _FakeAgent:
    """Stand-in for ``WorktreeSorcarAgent`` carrying just the
    attributes :meth:`VSCodeServer._get_running_tasks` reads."""

    def __init__(
        self,
        *,
        task_id: int,
        prompt: str,
        tokens: int,
        cost: float,
        steps: int,
    ) -> None:
        self._last_task_id = task_id
        self._last_user_prompt = prompt
        self.total_tokens_used = tokens
        self.budget_used = cost
        self.step_count = steps


def _spawn_blocking_thread() -> tuple[threading.Thread, threading.Event]:
    """Start a daemon thread that blocks on a fresh ``Event``.

    Returns the (thread, stop_event) so the test can release the
    thread at tear-down without leaking it.  The thread is alive
    while ``stop_event`` is unset, which is what :meth:`_get_running_tasks`
    checks via ``task_thread.is_alive()``.
    """
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    return thread, stop


class TestGetRunningTasks(unittest.TestCase):
    """Backend-only tests for :meth:`VSCodeServer._get_running_tasks`."""

    def setUp(self) -> None:
        self.printer = _CapturingPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self._threads: list[tuple[threading.Thread, threading.Event]] = []

    def tearDown(self) -> None:
        for thread, stop in self._threads:
            stop.set()
            thread.join(timeout=2.0)
        _RunningAgentState.running_agent_states.clear()

    def _register_running_tab(
        self,
        *,
        tab_id: str,
        chat_id: str,
        task_id: int,
        prompt: str,
        tokens: int,
        cost: float,
        steps: int,
        is_subagent: bool = False,
    ) -> _RunningAgentState:
        tab = _RunningAgentState(tab_id, "test-model")
        tab.chat_id = chat_id
        tab.agent = _FakeAgent(  # type: ignore[assignment]
            task_id=task_id,
            prompt=prompt,
            tokens=tokens,
            cost=cost,
            steps=steps,
        )
        tab.is_subagent = is_subagent
        thread, stop = _spawn_blocking_thread()
        self._threads.append((thread, stop))
        tab.task_thread = thread
        tab.user_answer_queue = queue.Queue()
        _RunningAgentState.running_agent_states[tab_id] = tab
        return tab

    def test_returns_empty_when_no_tabs(self) -> None:
        self.server._get_running_tasks()
        events = [e for e in self.printer.events if e.get("type") == "runningTasks"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tasks"], [])

    def test_returns_running_task_with_metrics(self) -> None:
        self._register_running_tab(
            tab_id="tab-A",
            chat_id="chat-A",
            task_id=42,
            prompt="Fix bug X in module Y",
            tokens=12345,
            cost=0.1234,
            steps=7,
        )
        self.server._get_running_tasks()
        events = [
            e for e in self.printer.events if e.get("type") == "runningTasks"
        ]
        self.assertEqual(len(events), 1)
        tasks = events[0]["tasks"]
        self.assertEqual(len(tasks), 1)
        t = tasks[0]
        self.assertEqual(t["tab_id"], "tab-A")
        self.assertEqual(t["chat_id"], "chat-A")
        self.assertEqual(t["task_id"], 42)
        self.assertEqual(t["title"], "Fix bug X in module Y")
        self.assertEqual(t["tokens"], 12345)
        self.assertAlmostEqual(t["cost"], 0.1234)
        self.assertEqual(t["steps"], 7)

    def test_excludes_subagent_tabs(self) -> None:
        self._register_running_tab(
            tab_id="parent", chat_id="chat-1", task_id=10, prompt="parent",
            tokens=10, cost=0.01, steps=1,
        )
        self._register_running_tab(
            tab_id="sub", chat_id="chat-1", task_id=11, prompt="sub",
            tokens=20, cost=0.02, steps=2, is_subagent=True,
        )
        self.server._get_running_tasks()
        events = [
            e for e in self.printer.events if e.get("type") == "runningTasks"
        ]
        tasks = events[-1]["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["tab_id"], "parent")

    def test_excludes_finished_tabs(self) -> None:
        # Spawn a tab whose thread has already exited.
        tab = _RunningAgentState("tab-done", "test-model")
        tab.chat_id = "chat-done"
        tab.agent = _FakeAgent(  # type: ignore[assignment]
            task_id=99, prompt="done", tokens=0, cost=0.0, steps=0,
        )
        dead_thread = threading.Thread(target=lambda: None, daemon=True)
        dead_thread.start()
        dead_thread.join(timeout=2.0)
        self.assertFalse(dead_thread.is_alive())
        tab.task_thread = dead_thread
        _RunningAgentState.running_agent_states["tab-done"] = tab

        self.server._get_running_tasks()
        events = [
            e for e in self.printer.events if e.get("type") == "runningTasks"
        ]
        self.assertEqual(events[-1]["tasks"], [])

    def test_dispatcher_routes_get_running_tasks(self) -> None:
        self._register_running_tab(
            tab_id="tab-X", chat_id="chat-X", task_id=1,
            prompt="hello", tokens=5, cost=0.5, steps=3,
        )
        self.server._handle_command({"type": "getRunningTasks"})
        events = [
            e for e in self.printer.events if e.get("type") == "runningTasks"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tasks"][0]["task_id"], 1)


# ── Frontend: renderRunningTasks under Node ─────────────────────────

_JS_PREAMBLE_DOM_STUB = r"""
var _elements = {};
function _makeEl(tag) {
    var _style = { display: '', backgroundColor: '', color: '' };
    var el = {
        tagName: tag, id: '', className: '', textContent: '', innerHTML: '',
        value: '', dataset: {}, disabled: false, children: [],
        _listeners: {},
        classList: {
            _c: [],
            add: function(c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
            remove: function(c) {
                var i = this._c.indexOf(c);
                if (i >= 0) this._c.splice(i,1);
            },
            contains: function(c) { return this._c.indexOf(c) >= 0; },
            toggle: function(c, force) {
                if (arguments.length >= 2) {
                    if (force) this.add(c); else this.remove(c);
                    return !!force;
                }
                if (this.contains(c)) { this.remove(c); return false; }
                this.add(c); return true;
            },
        },
        querySelector: function() { return _makeEl('div'); },
        querySelectorAll: function() { return []; },
        contains: function() { return false; },
        appendChild: function(c) { this.children.push(c); return c; },
        removeChild: function() {},
        addEventListener: function(t, fn) {
            if (!this._listeners[t]) this._listeners[t] = [];
            this._listeners[t].push(fn);
        },
        dispatchEvent: function() {},
        setAttribute: function(k, v) { this[k] = v; },
        getAttribute: function(k) { return this[k]; },
        removeAttribute: function(k) { delete this[k]; },
        focus: function() {}, setSelectionRange: function() {},
        scrollIntoView: function() {},
        getBoundingClientRect: function() {
            return {top:0,left:0,width:100,height:20};
        },
        insertBefore: function(n) { this.children.push(n); return n; },
        replaceChildren: function() { this.children = []; },
        remove: function() {},
        cloneNode: function() { return _makeEl(tag); },
        closest: function() { return null; },
        parentElement: null, parentNode: null, nextSibling: null,
        previousSibling: null, firstChild: null, lastChild: null,
        childNodes: [], nodeType: 1, ownerDocument: null,
        scrollHeight: 20, scrollTop: 0, clientHeight: 500,
    };
    Object.defineProperty(el, 'style', {
        get: function() { return _style; },
        set: function(v) { _style = v; },
    });
    return el;
}
function _fire(el, type, ev) {
    var ls = el._listeners[type] || [];
    for (var i = 0; i < ls.length; i++) ls[i](ev || {});
}
var document = {
    getElementById: function(id) {
        if (!_elements[id]) _elements[id] = _makeEl('div');
        return _elements[id];
    },
    createElement: function(tag) { return _makeEl(tag); },
    createDocumentFragment: function() { return _makeEl('fragment'); },
    body: _makeEl('body'),
    addEventListener: function() {},
    documentElement: _makeEl('html'),
};
var _postedMessages = [];
var _intervals = [];
var acquireVsCodeApi = function() {
    return {
        postMessage: function(msg) { _postedMessages.push(msg); },
        getState: function() { return null; },
        setState: function() {},
    };
};
var window = {
    addEventListener: function(type, fn) {
        if (type === 'message') window._messageHandler = fn;
    },
    matchMedia: function() {
        return { matches: false, addEventListener: function() {} };
    },
    innerHeight: 800,
    setTimeout: function(fn) { fn(); return 1; },
    clearTimeout: function() {},
    setInterval: function(fn, ms) {
        _intervals.push({fn: fn, ms: ms});
        return _intervals.length;
    },
    clearInterval: function(id) {
        if (id >= 1 && id <= _intervals.length) _intervals[id-1] = null;
    },
    requestAnimationFrame: function(fn) { fn(); },
    MutationObserver: function() {
        return { observe: function(){}, disconnect: function(){} };
    },
    _cancelDemoReplay: null,
};
var navigator = { userAgent: 'node-test' };
var MutationObserver = function() {
    return { observe: function(){}, disconnect: function(){} };
};
var ResizeObserver = function() {
    return { observe: function(){}, disconnect: function(){} };
};
var IntersectionObserver = function() {
    return { observe: function(){}, disconnect: function(){} };
};
var HTMLElement = function() {};
var CustomEvent = function(type, opts) {
    this.type = type;
    this.detail = (opts || {}).detail;
};
var MessageEvent = function(type, opts) {
    this.type = type;
    this.data = (opts || {}).data;
};
var setTimeout = window.setTimeout;
var clearTimeout = window.clearTimeout;
var setInterval = window.setInterval;
var clearInterval = window.clearInterval;
var requestAnimationFrame = window.requestAnimationFrame;
var hljs = {
    highlightElement: function() {},
    highlight: function() { return { value: '' }; },
    getLanguage: function() { return null; },
};
var marked = {
    parse: function(s) { return s; },
    setOptions: function() {},
    use: function() {},
};
var DOMPurify = { sanitize: function(s) { return s; } };
var console = { log: function(){}, warn: function(){}, error: function(){} };
"""

_JS_TEST_RENDER = r"""
// Open the sidebar on the Running tab; this should kick off a
// getRunningTasks request and start the 5s poll timer.
_postedMessages.length = 0;
_intervals.length = 0;
_fire(_elements['menu-btn'], 'click', {});

var openedPosted = _postedMessages.map(function(m) { return m.type; });
var pollScheduled = _intervals.filter(function(i) {
    return i && i.ms === 5000;
}).length;

// Simulate a runningTasks message coming back from the backend.
var handler = window._messageHandler;
handler({data: {
    type: 'runningTasks',
    tasks: [
        {
            tab_id: 'tab-1', chat_id: 'chat-aaa', task_id: 1,
            title: 'First running task',
            tokens: 12345, cost: 0.1234, steps: 5,
        },
        {
            tab_id: 'tab-2', chat_id: 'chat-bbb', task_id: 2,
            title: 'Second running task',
            tokens: 200, cost: 0.001, steps: 1,
        },
    ],
}});

var runningList = _elements['running-list'];
var items = runningList.children;

function _itemSummary(it) {
    var bg = it.style.backgroundColor;
    var metrics = '';
    for (var i = 0; i < it.children.length; i++) {
        if ((it.children[i].className || '').indexOf('running-item-metrics') >= 0) {
            metrics = it.children[i].textContent;
        }
    }
    var text = '';
    for (var i = 0; i < it.children.length; i++) {
        if ((it.children[i].className || '').indexOf('sidebar-item-text') >= 0) {
            text = it.children[i].textContent;
        }
    }
    return {
        className: it.className,
        bg: bg,
        text: text,
        metrics: metrics,
    };
}

var summaries = [];
for (var i = 0; i < items.length; i++) summaries.push(_itemSummary(items[i]));

// Switch to History and back to Running.  When leaving Running the
// poll timer must be cleared.
_postedMessages.length = 0;
_fire(_elements['sidebar-tab-history'], 'click', {});
var afterLeaveHistoryPosted = _postedMessages.map(function(m) { return m.type; });

_postedMessages.length = 0;
_fire(_elements['sidebar-tab-running'], 'click', {});
var afterReturnPosted = _postedMessages.map(function(m) { return m.type; });

// Render with empty list -> "No running tasks" placeholder.
handler({data: {type: 'runningTasks', tasks: []}});
var emptyHTML = runningList.innerHTML;

process.stdout.write(JSON.stringify({
    openedPosted: openedPosted,
    pollScheduled: pollScheduled,
    summaries: summaries,
    afterLeaveHistoryPosted: afterLeaveHistoryPosted,
    afterReturnPosted: afterReturnPosted,
    emptyHTML: emptyHTML,
}) + '\n');
"""


class TestRenderRunningTasksFrontend(unittest.TestCase):
    """Run ``main.js`` under Node and exercise the Running tab."""

    main_js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_js = _MAIN_JS.read_text(encoding="utf-8")

    def _run(self) -> dict:
        full = (
            _JS_PREAMBLE_DOM_STUB + "\n"
            + self.main_js + "\n"
            + _JS_TEST_RENDER
        )
        proc = subprocess.run(
            ["node", "-e", full],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            self.fail(f"node failed: {proc.stderr}")
        data: dict = json.loads(proc.stdout.strip().splitlines()[-1])
        return data

    def test_menu_btn_kicks_off_running_poll(self) -> None:
        out = self._run()
        self.assertIn("getRunningTasks", out["openedPosted"])
        self.assertGreaterEqual(
            out["pollScheduled"], 1,
            "Opening the Running tab must schedule a 5s setInterval",
        )

    def test_render_paints_tasks_with_chat_id_colour(self) -> None:
        out = self._run()
        summaries = out["summaries"]
        self.assertEqual(len(summaries), 2)
        # Both items must include the metrics line.
        for s in summaries:
            self.assertIn("steps", s["metrics"])
            self.assertIn("tok", s["metrics"])
            self.assertIn("$", s["metrics"])
            self.assertTrue(s["bg"].startswith("hsl("))
        # Different chat_ids must yield different background colours
        # (mirroring the History/Frequent colour scheme).
        self.assertNotEqual(summaries[0]["bg"], summaries[1]["bg"])
        self.assertIn("First running task", summaries[0]["text"])

    def test_returning_to_running_tab_reposts_get(self) -> None:
        out = self._run()
        self.assertIn("getRunningTasks", out["afterReturnPosted"])

    def test_empty_list_shows_placeholder(self) -> None:
        out = self._run()
        self.assertIn("No running tasks", out["emptyHTML"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
