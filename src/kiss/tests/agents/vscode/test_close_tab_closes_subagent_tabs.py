# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Closing a parent tab also closes every (nested) sub-agent tab.

Boots the real ``media/main.js`` under Node with a stub DOM, lets the
IIFE create one default tab, then drives two ``openSubagentTab`` events
through the message bus to build the tree:

    tab-A (regular, default)
     ├── tab-B (sub-agent, parent_tab_id=tab-A)
     │    └── tab-C (sub-agent, parent_tab_id=tab-B)
     └── tab-D (regular sibling created via openSubagentTab fallback)

The test then synthesises a click on tab-A's close button and asserts
that the frontend posted ``closeTab`` to the backend for tab-A, tab-B,
*and* tab-C (in that descendant set) — proving the recursive close
covers grandchildren — while tab-D is untouched.

The test also exercises the fallback path where the backend omits the
explicit ``parent_tab_id`` field and only stamps the parent's tab id
into the broadcast as ``tabId`` (chat_sorcar_agent's fan-out path).
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
    // ``innerHTML = ''`` is used by renderTabBar() to clear the tab list
    // before re-rendering.  Mirror that by emptying ``children`` so test
    // assertions see only the freshest render.
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
    createDocumentFragment: function() {
        var frag = _makeEl('fragment');
        return frag;
    },
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

// Pre-populate the persisted state so the IIFE restores a single
// regular tab with a known id (``tab-A``) and an unrelated regular
// sibling (``tab-D``).  This lets the test reference the parent by id
// without first having to read it back out of the JS heap.
_savedState = {
    tabs: [
        { title: 'A', chatId: 'tab-A', isSubagentTab: false, isDone: false },
        { title: 'D', chatId: 'tab-D', isSubagentTab: false, isDone: false },
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

// 1. Spawn a direct sub-agent of tab-A.  Use the explicit
//    ``parent_tab_id`` field (sorcar_agent / server.py path).
fireMessage({
    type: 'openSubagentTab',
    tab_id: 'tab-B',
    parent_tab_id: 'tab-A',
    description: 'Sub of A',
    taskIndex: 0,
    isSubagentTab: true,
});

// 2. Spawn a nested sub-sub-agent (grandchild of tab-A) using ONLY the
//    daemon-stamped ``tabId`` field (chat_sorcar_agent broadcast
//    path).  This pins the fallback that links the new tab to its
//    parent when ``parent_tab_id`` is omitted.
fireMessage({
    type: 'openSubagentTab',
    tab_id: 'tab-C',
    tabId: 'tab-B',
    description: 'Sub of B',
    taskIndex: 0,
    isSubagentTab: true,
});

// Capture the order of tabs from the latest renderTabBar() so we can
// click the right close button.  Each rendered tab element has a
// ``dataset.tabId`` and a final ``children`` entry that is the close
// ``span`` registered with a 'click' listener.
function findTab(tabListEl, id) {
    for (var i = 0; i < tabListEl.children.length; i++) {
        var t = tabListEl.children[i];
        if (t.dataset && t.dataset.tabId === id) return t;
    }
    return null;
}

var tabList = _elements['tab-list'];
var snapshotBefore = tabList.children.map(function(t) {
    return t.dataset.tabId;
});

// Reset posted message log so we only count what closeTab emits.
_postedMessages.length = 0;

// Click tab-A's close button.  Last child of the tab element is the
// close span (per renderTabBar layout).
var tabA = findTab(tabList, 'tab-A');
if (!tabA) {
    process.stderr.write('tab-A not in rendered tab-list\n');
    process.exit(3);
}
var closeBtn = tabA.children[tabA.children.length - 1];
_fire(closeBtn, 'click', { stopPropagation: function(){} });

var snapshotAfter = tabList.children.map(function(t) {
    return t.dataset.tabId;
});

var closeMsgs = _postedMessages.filter(function(m) {
    return m && m.type === 'closeTab';
}).map(function(m) { return m.tabId; });

process.stdout.write(JSON.stringify({
    before: snapshotBefore,
    after: snapshotAfter,
    closeMessages: closeMsgs,
}) + '\n');
"""


class TestCloseTabClosesSubagentTabs(unittest.TestCase):
    """Closing tab-A removes tab-A, tab-B (child), tab-C (grandchild)."""

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

    def test_parent_close_propagates_to_every_descendant(self) -> None:
        out = self._run()
        # Pre-close the rendered tab list must contain all four tabs.
        self.assertEqual(
            sorted(out["before"]),
            ["tab-A", "tab-B", "tab-C", "tab-D"],
            "all four tabs should be rendered before close",
        )
        # Post-close only the unrelated sibling tab-D should remain.
        self.assertEqual(
            out["after"], ["tab-D"],
            "only the unrelated regular tab should survive",
        )
        # The frontend must have posted a closeTab message to the backend
        # for the parent AND for every (nested) sub-agent descendant.
        self.assertEqual(
            sorted(out["closeMessages"]),
            ["tab-A", "tab-B", "tab-C"],
            "closeTab must be posted for parent and every descendant",
        )


if __name__ == "__main__":
    unittest.main()
