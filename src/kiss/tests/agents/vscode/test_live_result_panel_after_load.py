# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Result panel must render live when a fresh task finishes in a tab
that previously had a different task loaded.

Reproduces the regression where the defensive misroute guard in
``media/main.js`` drops live ``result`` / ``usage_info`` events
whenever the active tab's ``currentTaskId`` differs from the event's
``taskId``.  The trap: ``currentTaskId`` is only ever assigned by the
``task_events`` replay handler, so once the user has loaded any task
from the sidebar into a tab and then submits a brand-new task in that
same tab, the tab's ``currentTaskId`` still points at the OLD task.
The new task's events flow through tagged with the NEW ``taskId`` and
the guard drops the terminal ``result`` event — leaving the chat
stopped at the last ``tool_call(finish)`` panel.  Reloading the task
later sends ``task_events`` containing the persisted ``result`` event,
which renders normally — exactly matching the user-reported asymmetry.

The test boots the real ``media/main.js`` under Node with a stub DOM,
fires a ``task_events`` message (task_id=100, tabId=tab-A) to seed
``currentTaskId=100``, then drives live broadcasts for a different
task (taskId=200, tabId=tab-A): ``tool_call(name='finish')`` followed
by ``result``.  It asserts that the active tab's output container
contains an ``.rc`` (result card) element.  Before the fix the guard
silently drops the result event and no ``.rc`` element is appended.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

_VSCODE = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_MAIN_JS = _VSCODE / "media" / "main.js"


_PREAMBLE = r"""
var _elements = {};

function _makeEl(tag) {
    var _realStyle = { height: '', display: '', color: '' };
    var el = {
        tagName: tag,
        id: '',
        className: '',
        textContent: '',
        value: '',
        checked: false,
        dataset: {},
        disabled: false,
        children: [],
        _listeners: {},
        classList: {
            _c: [],
            add: function(c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
            remove: function(c) {
                var i = this._c.indexOf(c);
                if (i >= 0) this._c.splice(i, 1);
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
        focus: function() {},
        setSelectionRange: function() {},
        scrollIntoView: function() {},
        scrollTo: function() {},
        scrollBy: function() {},
        getBoundingClientRect: function() {
            return { top: 0, left: 0, width: 100, height: 20 };
        },
        insertBefore: function(n) { this.children.push(n); return n; },
        replaceChildren: function() { this.children = []; },
        remove: function() {},
        cloneNode: function() { return _makeEl(tag); },
        closest: function() { return null; },
        parentElement: null,
        parentNode: null,
        nextSibling: null,
        previousSibling: null,
        firstChild: null,
        lastChild: null,
        childNodes: [],
        nodeType: 1,
        ownerDocument: null,
        scrollHeight: 20,
        scrollTop: 0,
        clientHeight: 500,
    };
    var _html = '';
    Object.defineProperty(el, 'innerHTML', {
        get: function() { return _html; },
        set: function(v) {
            _html = String(v);
            if (_html === '') el.children = [];
        },
    });
    Object.defineProperty(el, 'style', {
        get: function() { return _realStyle; },
        set: function(v) { _realStyle = v; },
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

var _ids = [
    'output', 'welcome', 'task-input', 'send-btn', 'stop-btn',
    'upload-btn', 'model-btn', 'model-dropdown', 'model-search',
    'model-list', 'model-name', 'file-chips', 'status-text',
    'status-tokens', 'status-budget', 'sidebar', 'sidebar-overlay',
    'sidebar-close', 'history-search', 'history-search-clear',
    'history-list', 'menu-btn', 'task-panel', 'tab-bar',
    'tab-list', 'clear-btn',
    'remote-url', 'autocomplete', 'ghost-text', 'input-row',
];
for (var i = 0; i < _ids.length; i++) _elements[_ids[i]] = _makeEl('div');

var _postedMessages = [];
var _savedState = null;
var acquireVsCodeApi = function() {
    return {
        postMessage: function(msg) { _postedMessages.push(msg); },
        getState: function() { return _savedState; },
        setState: function(s) { _savedState = s; },
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
    setInterval: function() { return 1; },
    clearInterval: function() {},
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

// Seed a single regular tab with a stable id so the test can drive
// task_events / live events with explicit tabId.
_savedState = {
    tabs: [
        { title: 'A', chatId: 'tab-A', isSubagentTab: false, isDone: false },
    ],
    activeTabIndex: 0,
    chatId: 'tab-A',
};
"""


_TEST = r"""
var handler = window._messageHandler;
if (!handler) {
    process.stderr.write('main.js never registered a message handler\n');
    process.exit(2);
}

function fireMessage(payload) {
    handler({ data: payload });
}

// 1. Simulate the user loading an OLD task (task_id=100) into tab-A
//    from the history sidebar.  This is what stamps ``currentTaskId``
//    on the tab.  Pass an empty events array — the loaded snapshot
//    content is irrelevant; only the side effect of binding
//    ``currentTaskId = 100`` matters for this test.
fireMessage({
    type: 'task_events',
    events: [],
    task: 'Previously loaded task',
    task_id: 100,
    chat_id: 'tab-A',
    extra: '',
    tabId: 'tab-A',
});

// 2. Reset the output children so we only observe what the LIVE
//    events render below.  ``task_events`` may seed the active tab
//    with replayed nodes; we only care about the live ``result``.
_elements['output'].children = [];

// 3. Now the user submits a fresh task in the same tab.  The agent
//    starts task_id=200 server-side and streams events back tagged
//    ``taskId: 200`` and ``tabId: 'tab-A'``.  Fire the terminal
//    ``tool_call(name='finish')`` and ``result`` events the way
//    ``WebPrinter`` would broadcast them after thread-local taskId
//    stamping + tab fanout.
fireMessage({
    type: 'tool_call',
    name: 'finish',
    args: { result: 'all done', success: true, is_continue: false },
    taskId: 200,
    tabId: 'tab-A',
});

// Omit ``summary`` so the renderer takes the plain-``text``
// branch instead of routing through marked.parse + kissSanitize
// (which needs a richer DOM than this stub provides).
fireMessage({
    type: 'result',
    text: 'all done',
    success: true,
    total_tokens: 1234,
    cost: '$0.01',
    step_count: 3,
    taskId: 200,
    tabId: 'tab-A',
});

// 4. Inspect the active tab's output container for a Result card.
//    The renderer creates a ``div.ev.rc`` and appends it to
//    ``_elements['output']`` for live result events.
var output = _elements['output'];
var hasResultCard = false;
var classNames = [];
for (var i = 0; i < output.children.length; i++) {
    var ch = output.children[i];
    classNames.push(ch.className || '');
    var cn = String(ch.className || '');
    if (cn.split(/\s+/).indexOf('rc') >= 0) {
        hasResultCard = true;
    }
}

process.stdout.write(JSON.stringify({
    hasResultCard: hasResultCard,
    childClassNames: classNames,
}) + '\n');
"""


class TestLiveResultPanelAfterLoad(unittest.TestCase):
    """Live result event on a tab with a stale currentTaskId still renders."""

    main_js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_js = _MAIN_JS.read_text(encoding="utf-8")

    def _run(self) -> dict:
        full = _PREAMBLE + "\n" + self.main_js + "\n" + _TEST
        result = subprocess.run(
            ["node", "-e", full],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            self.fail(f"node failed: {result.stderr}\nSTDOUT={result.stdout}")
        parsed: dict = json.loads(result.stdout.strip().splitlines()[-1])
        return parsed

    def test_live_result_renders_after_old_task_loaded(self) -> None:
        out = self._run()
        self.assertTrue(
            out["hasResultCard"],
            "live result event must render a .rc panel even when the "
            "tab still carries a stale currentTaskId from a previously "
            "loaded task; got children: "
            + repr(out["childClassNames"]),
        )


if __name__ == "__main__":
    unittest.main()
