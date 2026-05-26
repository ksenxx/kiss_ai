"""Backend-broadcast ``new_tab`` opens a fresh tab and posts ``resumeSession``.

The Python server can ask the frontend to open a fresh chat tab and
resume an existing task into it by broadcasting::

    {"type": "new_tab", "task_id": <int>}

This test boots the real ``media/main.js`` under Node with a minimal
DOM stub, fires that broadcast through the message bus, and asserts
that:

* the frontend allocated an additional tab (``tab-list`` gained a row),
* the frontend posted a single ``resumeSession`` command back to the
  backend carrying the same ``task_id`` as ``taskId`` and the freshly
  allocated tab's id as ``tabId``,
* the ``resumeSession`` payload does **not** carry an ``id`` /
  ``chatId`` field (``_cmd_resume_session`` supports a task-id-only
  resume — driving ``new_tab`` should not invent a chat id).

It also pins the Python-side helper ``VSCodeServer.broadcast_new_tab``
emits the exact dict shape the frontend handler expects.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer


class _CapturingPrinter(BaseBrowserPrinter):
    """Records every ``broadcast`` call verbatim."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

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

// Pre-populate persisted state so the IIFE restores exactly one tab.
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

function fireMessage(payload) { handler({ data: payload }); }

var tabList = _elements['tab-list'];
var tabsBefore = tabList.children.map(function(t) { return t.dataset.tabId; });

_postedMessages.length = 0;

// Backend broadcasts ``new_tab`` carrying only the task_id.
fireMessage({ type: 'new_tab', task_id: 42 });

var tabsAfter = tabList.children.map(function(t) { return t.dataset.tabId; });

var resumes = _postedMessages.filter(function(m) {
    return m && m.type === 'resumeSession';
});

process.stdout.write(JSON.stringify({
    tabsBefore: tabsBefore,
    tabsAfter: tabsAfter,
    posted: _postedMessages,
    resumes: resumes,
}) + '\n');
"""


class TestNewTabMessage(unittest.TestCase):
    """``new_tab`` broadcast opens a fresh tab and resumes the task."""

    main_js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_js = _MAIN_JS.read_text(encoding="utf-8")

    def _run_js(self) -> dict[str, Any]:
        full = _PREAMBLE + "\n" + self.main_js + "\n" + _TEST
        result = subprocess.run(
            ["node", "-e", full],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            self.fail(f"node failed: {result.stderr}\nSTDOUT={result.stdout}")
        parsed: dict[str, Any] = json.loads(
            result.stdout.strip().splitlines()[-1],
        )
        return parsed

    def test_new_tab_creates_tab_and_posts_resume_session(self) -> None:
        out = self._run_js()
        self.assertEqual(
            out["tabsBefore"], ["tab-A"],
            "only the restored tab should exist before new_tab fires",
        )
        # Frontend must have allocated exactly one additional tab.
        self.assertEqual(
            len(out["tabsAfter"]), len(out["tabsBefore"]) + 1,
            f"new_tab must allocate one additional tab; got "
            f"{out['tabsAfter']!r}",
        )
        self.assertIn("tab-A", out["tabsAfter"])
        new_tab_ids = [tid for tid in out["tabsAfter"] if tid != "tab-A"]
        self.assertEqual(
            len(new_tab_ids), 1,
            f"exactly one fresh tab should appear, got {new_tab_ids!r}",
        )
        new_tab_id = new_tab_ids[0]

        # Exactly one resumeSession was posted, carrying taskId=42 and
        # the freshly allocated tab id.  No chat id is invented.
        self.assertEqual(
            len(out["resumes"]), 1,
            f"exactly one resumeSession expected; got {out['resumes']!r}",
        )
        resume = out["resumes"][0]
        self.assertEqual(resume["taskId"], 42)
        self.assertEqual(resume["tabId"], new_tab_id)
        # ``id`` (chatId) must be absent — task-id-only resume.
        self.assertNotIn("id", resume)

    def test_new_tab_ignored_without_task_id(self) -> None:
        # A malformed event missing task_id must not open a tab nor
        # post anything back to the backend.
        broken = r"""
var handler = window._messageHandler;
_postedMessages.length = 0;
var tabList = _elements['tab-list'];
var before = tabList.children.length;
handler({ data: { type: 'new_tab' } });
var after = tabList.children.length;
process.stdout.write(JSON.stringify({
    before: before, after: after, posted: _postedMessages,
}) + '\n');
"""
        full = _PREAMBLE + "\n" + self.main_js + "\n" + broken
        result = subprocess.run(
            ["node", "-e", full],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            self.fail(f"node failed: {result.stderr}\nSTDOUT={result.stdout}")
        out = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(out["before"], out["after"])
        self.assertEqual(out["posted"], [])


class TestBroadcastNewTabHelper(unittest.TestCase):
    """``VSCodeServer.broadcast_new_tab`` emits the expected payload."""

    def test_helper_broadcasts_expected_shape(self) -> None:
        printer = _CapturingPrinter()
        server = VSCodeServer(printer=printer)
        server.broadcast_new_tab(7)
        # Explicit empty ``taskId`` keeps this a global system event
        # so the broadcast reaches every connected client (the
        # frontend needs the event to allocate the new tab; only
        # after allocation does it subscribe to the task's stream).
        self.assertEqual(
            printer.events,
            [{"type": "new_tab", "task_id": 7, "taskId": ""}],
        )

    def test_helper_coerces_task_id_to_int(self) -> None:
        printer = _CapturingPrinter()
        server = VSCodeServer(printer=printer)
        # Numeric-string task ids (e.g. read from JSON) should be
        # normalised to a plain int so the frontend doesn't have to
        # branch on the field's runtime type.
        server.broadcast_new_tab("11")  # type: ignore[arg-type]
        self.assertEqual(
            printer.events,
            [{"type": "new_tab", "task_id": 11, "taskId": ""}],
        )


if __name__ == "__main__":
    unittest.main()
