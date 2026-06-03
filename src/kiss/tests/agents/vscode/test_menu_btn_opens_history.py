# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ``#menu-btn`` ("Advanced options") placement and behaviour.

After the move, ``#menu-btn`` sits at the leftmost position inside
``#model-picker`` — *before* ``#model-btn`` — and clicking it toggles
the sidebar (opening it on the first in-panel tab, which is the
Running tab).  The tab-bar ``#history-btn`` has been removed, so
``#menu-btn`` is the only entry-point.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
_SORCAR_TS = _REPO_ROOT / "src" / "SorcarTab.ts"
_WEB_SERVER = _REPO_ROOT / "web_server.py"
_MAIN_JS = _REPO_ROOT / "media" / "main.js"


class TestMenuBtnPlacement(unittest.TestCase):
    """``#menu-btn`` must precede ``#model-btn`` in both templates."""

    def test_sorcar_tab_template_menu_btn_before_model_btn(self) -> None:
        html = _SORCAR_TS.read_text()
        menu_pos = html.index('id="menu-btn"')
        model_pos = html.index('id="model-btn"')
        upload_pos = html.index('id="upload-btn"')
        self.assertLess(menu_pos, model_pos)
        self.assertLess(model_pos, upload_pos)

    def test_web_server_template_menu_btn_before_model_btn(self) -> None:
        html = _WEB_SERVER.read_text()
        menu_pos = html.index('id="menu-btn"')
        model_pos = html.index('id="model-btn"')
        upload_pos = html.index('id="upload-btn"')
        self.assertLess(menu_pos, model_pos)
        self.assertLess(model_pos, upload_pos)


_MAIN_JS_TEXT = _MAIN_JS.read_text()


class TestMenuBtnHandlerWired(unittest.TestCase):
    """``main.js`` must wire ``#menu-btn`` to open the History sidebar."""

    def test_menu_btn_referenced(self) -> None:
        self.assertIn("getElementById('menu-btn')", _MAIN_JS_TEXT)
        self.assertIn("menuBtn", _MAIN_JS_TEXT)

    def test_menu_btn_click_opens_history_sidebar(self) -> None:
        """``menuBtn`` click handler must open the (history) sidebar."""
        # The handler is attached via ``menuBtn.addEventListener('click', ...)``.
        match = re.search(
            r"menuBtn\.addEventListener\(\s*'click'\s*,\s*([A-Za-z_$][\w$]*)",
            _MAIN_JS_TEXT,
        )
        assert match is not None, "menuBtn must register a click handler in main.js"
        handler_name = match.group(1)
        # The handler body must add 'open' to #sidebar and request the
        # latest history from the backend.
        body_match = re.search(
            r"function\s+"
            + re.escape(handler_name)
            + r"\s*\([^)]*\)\s*\{([\s\S]*?)\n\s{4}\}",
            _MAIN_JS_TEXT,
        )
        assert body_match is not None, (
            f"Could not locate body of {handler_name}()"
        )
        body = body_match.group(1)
        self.assertIn("sidebar.classList.add('open')", body)
        self.assertIn("getHistory", body)


_JS_PREAMBLE = r"""
var _elements = {};
var _calls = [];

function _makeEl(tag) {
    var el = {
        tagName: tag, id: '', className: '', textContent: '', innerHTML: '',
        value: '', style: {}, dataset: {}, disabled: false, children: [],
        _listeners: {},
        classList: {
            _c: [],
            add: function(c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
            remove: function(c) { var i = this._c.indexOf(c); if (i >= 0) this._c.splice(i,1); },
            contains: function(c) { return this._c.indexOf(c) >= 0; },
            toggle: function(c, force) {
                if (force === true) { this.add(c); return true; }
                if (force === false) { this.remove(c); return false; }
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
        focus: function() {}, setSelectionRange: function() {},
        scrollIntoView: function() {},
        getBoundingClientRect: function() { return {top:0,left:0,width:100,height:20}; },
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
        return frag;
    },
    body: _makeEl('body'),
    addEventListener: function() {},
    documentElement: _makeEl('html'),
};

var _postedMessages = [];
var acquireVsCodeApi = function() {
    return {
        postMessage: function(msg) { _postedMessages.push(msg); },
        getState: function() { return null; },
        setState: function() {},
    };
};

var window = {
    addEventListener: function() {},
    matchMedia: function() { return { matches: false, addEventListener: function() {} }; },
    innerHeight: 800,
    setTimeout: function(fn) { fn(); return 1; },
    clearTimeout: function() {},
    setInterval: function() { return 1; },
    clearInterval: function() {},
    requestAnimationFrame: function(fn) { fn(); },
    MutationObserver: function() { return { observe: function(){}, disconnect: function(){} }; },
    _cancelDemoReplay: null,
};
var navigator = { userAgent: 'node-test' };
var MutationObserver = function() { return { observe: function(){}, disconnect: function(){} }; };
var ResizeObserver = function() { return { observe: function(){}, disconnect: function(){} }; };
var IntersectionObserver = function() {
    return { observe: function(){}, disconnect: function(){} };
};
var HTMLElement = function() {};
var CustomEvent = function(type, opts) { this.type = type; this.detail = (opts||{}).detail; };
var MessageEvent = function(type, opts) { this.type = type; this.data = (opts||{}).data; };
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
var marked = { parse: function(s) { return s; }, setOptions: function() {}, use: function() {} };
var DOMPurify = { sanitize: function(s) { return s; } };
var console = { log: function(){}, warn: function(){}, error: function(){} };
"""

_JS_TEST = r"""
var menuBtn = _elements['menu-btn'];
var sidebar = _elements['sidebar'];
var sidebarOverlay = _elements['sidebar-overlay'];

// Pre-click state
var preOpen = sidebar.classList.contains('open');

// Fire the click listener registered by main.js
var listeners = menuBtn._listeners['click'] || [];
if (listeners.length === 0) {
    process.stdout.write(JSON.stringify({error: 'no menu-btn click listener'}) + '\n');
} else {
    listeners[0]();
    var postedGetHistory = false;
    for (var i = 0; i < _postedMessages.length; i++) {
        if (_postedMessages[i] && _postedMessages[i].type === 'getHistory') {
            postedGetHistory = true;
            break;
        }
    }
    var results = {
        preOpen: preOpen,
        sidebarOpen: sidebar.classList.contains('open'),
        overlayOpen: sidebarOverlay.classList.contains('open'),
        postedGetHistory: postedGetHistory,
    };
    process.stdout.write(JSON.stringify(results) + '\n');
}
"""


class TestMenuBtnClickOpensHistorySidebar(unittest.TestCase):
    """Integration: clicking ``#menu-btn`` must open the History sidebar."""

    def test_click_opens_sidebar_and_posts_get_history(self) -> None:
        import json

        full_js = _JS_PREAMBLE + "\n" + _MAIN_JS_TEXT + "\n" + _JS_TEST
        result = subprocess.run(
            ["node", "-e", full_js],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            self.fail(f"Node.js error:\n{result.stderr}")
        results = json.loads(result.stdout.strip())
        self.assertNotIn(
            "error", results, f"JS reported error: {results.get('error')}"
        )
        self.assertFalse(results["preOpen"])
        self.assertTrue(
            results["sidebarOpen"],
            "Clicking #menu-btn did not add 'open' to #sidebar",
        )
        self.assertTrue(results["overlayOpen"])
        self.assertTrue(
            results["postedGetHistory"],
            "Clicking #menu-btn did not post 'getHistory' to the backend",
        )


if __name__ == "__main__":
    unittest.main()
