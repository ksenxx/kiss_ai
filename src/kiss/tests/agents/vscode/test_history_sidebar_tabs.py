"""Integration tests for the History/Frequent-Tasks tabbed sidebar.

The chat webview previously had two separate tab-bar buttons — ``history-btn``
and ``frequent-btn`` — each of which opened a distinct sliding sidebar
(``#sidebar`` and ``#frequent-sidebar``).  This test pins down the new
behaviour:

* The tab-bar ships only one entry-point for both lists: ``history-btn``.
* Clicking ``history-btn`` opens ``#sidebar`` with the *History* sub-tab
  active.
* ``#sidebar`` contains two in-panel tab buttons
  (``sidebar-tab-history`` / ``sidebar-tab-frequent``) plus matching panel
  containers (``sidebar-tab-history-panel`` /
  ``sidebar-tab-frequent-panel``).
* Clicking the frequent in-panel tab posts ``getFrequentTasks`` to the
  backend and reveals the frequent panel while hiding the history panel.
* Clicking the history in-panel tab swaps the panels back and posts
  ``getHistory``.
* The standalone webapp (``_build_html``) mirrors the extension webview.

Tests exercise the real ``SorcarTab.ts`` source, the real
``_build_html`` output, and the real ``main.js`` running under Node.js
with a stubbed DOM.  No mocks of the implementation under test.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_VSCODE = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_SORCAR_TAB_TS = _VSCODE / "src" / "SorcarTab.ts"
_MAIN_JS = _VSCODE / "media" / "main.js"


def _ext_html() -> str:
    return _SORCAR_TAB_TS.read_text(encoding="utf-8")


class TestSidebarTabsMarkup(unittest.TestCase):
    """HTML in both surfaces wires the new tabbed sidebar correctly."""

    def test_frequent_btn_removed_from_extension_tab_bar(self) -> None:
        self.assertNotIn('id="frequent-btn"', _ext_html())

    def test_frequent_btn_removed_from_webapp_tab_bar(self) -> None:
        self.assertNotIn('id="frequent-btn"', _build_html())

    def test_separate_frequent_sidebar_removed_from_extension(self) -> None:
        self.assertNotIn('id="frequent-sidebar"', _ext_html())
        self.assertNotIn('id="frequent-sidebar-overlay"', _ext_html())
        self.assertNotIn('id="frequent-sidebar-close"', _ext_html())

    def test_separate_frequent_sidebar_removed_from_webapp(self) -> None:
        web = _build_html()
        self.assertNotIn('id="frequent-sidebar"', web)
        self.assertNotIn('id="frequent-sidebar-overlay"', web)
        self.assertNotIn('id="frequent-sidebar-close"', web)

    def test_extension_sidebar_has_tab_buttons_and_panels(self) -> None:
        html = _ext_html()
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-history-panel"',
            'id="sidebar-tab-frequent-panel"',
            'id="history-list"',
            'id="frequent-list"',
        ):
            self.assertIn(el, html, f"{el} missing from SorcarTab.ts")

    def test_webapp_sidebar_has_tab_buttons_and_panels(self) -> None:
        html = _build_html()
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-history-panel"',
            'id="sidebar-tab-frequent-panel"',
            'id="history-list"',
            'id="frequent-list"',
        ):
            self.assertIn(el, html, f"{el} missing from webapp HTML")


# ── Node.js-driven behaviour tests ───────────────────────────────────────

_JS_PREAMBLE = r"""
var _elements = {};

function _makeEl(tag) {
    var _realStyle = { height: '', display: '', color: '' };
    var el = {
        tagName: tag,
        id: '',
        className: '',
        textContent: '',
        innerHTML: '',
        value: '',
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
        frag.appendChild = function(c) { this.children.push(c); return c; };
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
    'tab-list', 'config-btn', 'config-sidebar', 'config-sidebar-overlay',
    'config-sidebar-close', 'config-panel', 'clear-btn',
    'remote-url', 'autocomplete', 'ghost-text', 'input-row',
    'merge-toolbar', 'merge-accept-all-btn', 'merge-reject-all-btn',
    'merge-accept-file-btn', 'merge-reject-file-btn',
    'merge-prev-btn', 'merge-next-btn', 'merge-file-label',
    'merge-counter', 'merge-accept-btn', 'merge-reject-btn',
    'sidebar-tab-history', 'sidebar-tab-frequent',
    'sidebar-tab-history-panel', 'sidebar-tab-frequent-panel',
    'frequent-list',
];
for (var i = 0; i < _ids.length; i++) _elements[_ids[i]] = _makeEl('div');

var _postedMessages = [];
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
"""

_JS_TEST = r"""
var menuBtn = _elements['menu-btn'];
var sidebar = _elements['sidebar'];
var tabHistoryBtn = _elements['sidebar-tab-history'];
var tabFrequentBtn = _elements['sidebar-tab-frequent'];
var historyPanel = _elements['sidebar-tab-history-panel'];
var frequentPanel = _elements['sidebar-tab-frequent-panel'];

function _snapshot(label) {
    return {
        label: label,
        posted: _postedMessages.map(function(m) { return m.type; }),
        sidebarOpen: sidebar.classList.contains('open'),
        historyDisplay: historyPanel.style.display,
        frequentDisplay: frequentPanel.style.display,
        historyActive: tabHistoryBtn.classList.contains('active'),
        frequentActive: tabFrequentBtn.classList.contains('active'),
    };
}

var snapshots = [];

// 1. Click menu-btn → open sidebar with History tab active.
_postedMessages.length = 0;
_fire(menuBtn, 'click', {});
snapshots.push(_snapshot('afterHistoryBtnOpen'));

// 2. Click frequent in-panel tab → show frequent panel.
_postedMessages.length = 0;
_fire(tabFrequentBtn, 'click', {});
snapshots.push(_snapshot('afterFrequentTab'));

// 3. Click history in-panel tab → back to history panel.
_postedMessages.length = 0;
_fire(tabHistoryBtn, 'click', {});
snapshots.push(_snapshot('afterHistoryTab'));

process.stdout.write(JSON.stringify(snapshots) + '\n');
"""


class TestSidebarTabsBehaviour(unittest.TestCase):
    """Run main.js under Node to verify the sidebar-tab interactions."""

    main_js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.main_js = _MAIN_JS.read_text(encoding="utf-8")

    def _run(self) -> list[dict]:
        full = _JS_PREAMBLE + "\n" + self.main_js + "\n" + _JS_TEST
        result = subprocess.run(
            ["node", "-e", full],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            self.fail(f"node failed: {result.stderr}")
        data: list[dict] = json.loads(result.stdout.strip().splitlines()[-1])
        return data

    def test_history_btn_opens_sidebar_on_history_tab(self) -> None:
        snaps = self._run()
        s = snaps[0]
        self.assertTrue(s["sidebarOpen"], "sidebar must open when menu-btn is clicked")
        self.assertIn("getHistory", s["posted"])
        self.assertTrue(s["historyActive"], "History tab must be active")
        self.assertFalse(s["frequentActive"], "Frequent tab must not be active")
        self.assertNotEqual(
            s["historyDisplay"], "none", "History panel must be visible"
        )
        self.assertEqual(
            s["frequentDisplay"], "none", "Frequent panel must be hidden"
        )

    def test_frequent_tab_switches_panel_and_fetches_tasks(self) -> None:
        snaps = self._run()
        s = snaps[1]
        self.assertTrue(s["sidebarOpen"])
        self.assertIn("getFrequentTasks", s["posted"])
        self.assertTrue(s["frequentActive"])
        self.assertFalse(s["historyActive"])
        self.assertEqual(s["historyDisplay"], "none")
        self.assertNotEqual(s["frequentDisplay"], "none")

    def test_history_tab_switches_back_and_fetches_history(self) -> None:
        snaps = self._run()
        s = snaps[2]
        self.assertTrue(s["sidebarOpen"])
        self.assertIn("getHistory", s["posted"])
        self.assertTrue(s["historyActive"])
        self.assertFalse(s["frequentActive"])
        self.assertNotEqual(s["historyDisplay"], "none")
        self.assertEqual(s["frequentDisplay"], "none")


if __name__ == "__main__":
    unittest.main()
