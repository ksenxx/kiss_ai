"""Integration tests for the Settings tab inside the unified sidebar.

The chat webview previously had a separate ``config-btn`` in the tab-bar
that opened a dedicated ``#config-sidebar``.  This test pins the new
behaviour:

* The tab-bar no longer ships ``config-btn``.
* The History / Frequent / Settings panels all live inside a single
  ``#sidebar`` with a three-tab header.
* Clicking the in-panel Settings tab posts ``getConfig`` and reveals
  ``#sidebar-tab-settings-panel`` while hiding the other two panels.
* The ``cfg-*`` form controls live inside the new settings panel.
* Switching away from the settings tab after the form has been
  populated posts ``saveConfig``.
* Closing the sidebar while the settings tab is active and the form
  is populated also posts ``saveConfig``.
* The standalone webapp (``_build_html``) mirrors the extension.

Tests exercise the real ``SorcarTab.ts`` source, the real
``_build_html`` output, and the real ``main.js`` running under Node.js
with a stubbed DOM.  No mocks of the implementation under test.
"""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_VSCODE = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_SORCAR_TAB_TS = _VSCODE / "src" / "SorcarTab.ts"
_MAIN_JS = _VSCODE / "media" / "main.js"


def _ext_html() -> str:
    return _SORCAR_TAB_TS.read_text(encoding="utf-8")


def _section(html: str, container_id: str) -> str:
    """Return the markup of the ``<div>`` with the given id (balanced)."""
    pat = re.compile(r'<div\s+id="' + re.escape(container_id) + r'"')
    m = pat.search(html)
    if not m:
        return ""
    depth = 1
    i = m.end()
    while i < len(html) and depth:
        nxt = html.find("<", i)
        if nxt < 0:
            break
        if html.startswith("<div", nxt) and html[nxt + 4] in " \t\n>":
            depth += 1
            i = nxt + 4
        elif html.startswith("</div>", nxt):
            depth -= 1
            if depth == 0:
                return html[m.start() : nxt + len("</div>")]
            i = nxt + len("</div>")
        else:
            i = nxt + 1
    return html[m.start() :]


class TestSettingsTabMarkup(unittest.TestCase):
    """HTML in both surfaces wires the new Settings sub-tab correctly."""

    def test_config_btn_removed_from_extension_tab_bar(self) -> None:
        bar = _section(_ext_html(), "tab-bar")
        self.assertTrue(bar, "could not locate #tab-bar in SorcarTab.ts")
        self.assertNotIn('id="config-btn"', bar)

    def test_config_btn_removed_from_webapp_tab_bar(self) -> None:
        bar = _section(_build_html(), "tab-bar")
        self.assertTrue(bar, "could not locate #tab-bar in webapp HTML")
        self.assertNotIn('id="config-btn"', bar)

    def test_separate_config_sidebar_removed_from_extension(self) -> None:
        html = _ext_html()
        self.assertNotIn('id="config-sidebar"', html)
        self.assertNotIn('id="config-sidebar-overlay"', html)
        self.assertNotIn('id="config-sidebar-close"', html)

    def test_separate_config_sidebar_removed_from_webapp(self) -> None:
        html = _build_html()
        self.assertNotIn('id="config-sidebar"', html)
        self.assertNotIn('id="config-sidebar-overlay"', html)
        self.assertNotIn('id="config-sidebar-close"', html)

    def test_extension_sidebar_has_settings_tab_button_and_panel(self) -> None:
        sidebar = _section(_ext_html(), "sidebar")
        self.assertTrue(sidebar)
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-settings"',
            'id="sidebar-tab-history-panel"',
            'id="sidebar-tab-frequent-panel"',
            'id="sidebar-tab-settings-panel"',
        ):
            self.assertIn(el, sidebar, f"{el} missing from #sidebar in SorcarTab.ts")

    def test_webapp_sidebar_has_settings_tab_button_and_panel(self) -> None:
        sidebar = _section(_build_html(), "sidebar")
        self.assertTrue(sidebar)
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-settings"',
            'id="sidebar-tab-history-panel"',
            'id="sidebar-tab-frequent-panel"',
            'id="sidebar-tab-settings-panel"',
        ):
            self.assertIn(el, sidebar, f"{el} missing from #sidebar in webapp HTML")

    def test_extension_settings_panel_contains_config_form(self) -> None:
        panel = _section(_ext_html(), "sidebar-tab-settings-panel")
        self.assertTrue(panel, "#sidebar-tab-settings-panel missing in SorcarTab.ts")
        for inp in (
            'id="cfg-max-budget"',
            'id="cfg-custom-endpoint"',
            'id="cfg-custom-api-key"',
            'id="cfg-custom-headers"',
            'id="cfg-use-web-browser"',
            'id="cfg-remote-password"',
            'id="cfg-key-GEMINI_API_KEY"',
            'id="cfg-key-OPENAI_API_KEY"',
            'id="cfg-key-ANTHROPIC_API_KEY"',
        ):
            self.assertIn(
                inp,
                panel,
                f"{inp} must live inside #sidebar-tab-settings-panel in SorcarTab.ts",
            )

    def test_webapp_settings_panel_contains_config_form(self) -> None:
        panel = _section(_build_html(), "sidebar-tab-settings-panel")
        self.assertTrue(panel, "#sidebar-tab-settings-panel missing in webapp HTML")
        for inp in (
            'id="cfg-max-budget"',
            'id="cfg-custom-endpoint"',
            'id="cfg-custom-api-key"',
            'id="cfg-custom-headers"',
            'id="cfg-use-web-browser"',
            'id="cfg-remote-password"',
            'id="cfg-key-GEMINI_API_KEY"',
            'id="cfg-key-OPENAI_API_KEY"',
            'id="cfg-key-ANTHROPIC_API_KEY"',
        ):
            self.assertIn(
                inp,
                panel,
                f"{inp} must live inside #sidebar-tab-settings-panel in webapp HTML",
            )


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
    'history-list', 'history-btn', 'task-panel', 'tab-bar',
    'tab-list', 'clear-btn',
    'remote-url', 'autocomplete', 'ghost-text', 'input-row',
    'merge-toolbar', 'merge-accept-all-btn', 'merge-reject-all-btn',
    'merge-accept-file-btn', 'merge-reject-file-btn',
    'merge-prev-btn', 'merge-next-btn', 'merge-file-label',
    'merge-counter', 'merge-accept-btn', 'merge-reject-btn',
    'sidebar-tab-history', 'sidebar-tab-frequent', 'sidebar-tab-settings',
    'sidebar-tab-history-panel', 'sidebar-tab-frequent-panel',
    'sidebar-tab-settings-panel',
    'frequent-list',
    'cfg-max-budget', 'cfg-custom-endpoint', 'cfg-custom-api-key',
    'cfg-custom-headers', 'cfg-use-web-browser', 'cfg-remote-password',
    'cfg-key-GEMINI_API_KEY', 'cfg-key-OPENAI_API_KEY',
    'cfg-key-ANTHROPIC_API_KEY', 'cfg-key-TOGETHER_API_KEY',
    'cfg-key-OPENROUTER_API_KEY', 'cfg-key-MINIMAX_API_KEY',
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

# Simulate the backend replying to ``getConfig`` so that
# ``configFormPopulated`` flips to ``true`` and subsequent close/switch
# events trigger ``saveConfig``.
_JS_TEST = r"""
var historyBtn = _elements['history-btn'];
var sidebar = _elements['sidebar'];
var sidebarClose = _elements['sidebar-close'];
var sidebarOverlay = _elements['sidebar-overlay'];
var tabHistoryBtn = _elements['sidebar-tab-history'];
var tabFrequentBtn = _elements['sidebar-tab-frequent'];
var tabSettingsBtn = _elements['sidebar-tab-settings'];
var historyPanel = _elements['sidebar-tab-history-panel'];
var frequentPanel = _elements['sidebar-tab-frequent-panel'];
var settingsPanel = _elements['sidebar-tab-settings-panel'];

function _snap(label) {
    return {
        label: label,
        posted: _postedMessages.map(function(m) { return m.type; }),
        sidebarOpen: sidebar.classList.contains('open'),
        historyDisplay: historyPanel.style.display,
        frequentDisplay: frequentPanel.style.display,
        settingsDisplay: settingsPanel.style.display,
        historyActive: tabHistoryBtn.classList.contains('active'),
        frequentActive: tabFrequentBtn.classList.contains('active'),
        settingsActive: tabSettingsBtn.classList.contains('active'),
    };
}

function _populateConfigViaMessage() {
    // Replicate the backend's response to ``getConfig`` so that
    // configFormPopulated flips to true.
    window._messageHandler({ data: {
        type: 'configData',
        config: { max_budget: 100, use_web_browser: true },
        apiKeys: {},
    }});
}

var snapshots = [];

// 1. Open sidebar via history-btn → History tab active.
_postedMessages.length = 0;
_fire(historyBtn, 'click', {});
snapshots.push(_snap('afterHistoryOpen'));

// 2. Click Settings tab → posts getConfig, settings panel visible.
_postedMessages.length = 0;
_fire(tabSettingsBtn, 'click', {});
_populateConfigViaMessage();
snapshots.push(_snap('afterSettingsTab'));

// 3. Switch back to History from Settings → posts saveConfig + getHistory.
_postedMessages.length = 0;
_fire(tabHistoryBtn, 'click', {});
snapshots.push(_snap('afterSwitchBackToHistory'));

// 4. Switch to Settings again, populate, then close via sidebarClose →
//    posts saveConfig and sidebar closes.
_fire(tabSettingsBtn, 'click', {});
_populateConfigViaMessage();
_postedMessages.length = 0;
_fire(sidebarClose, 'click', {});
snapshots.push(_snap('afterCloseFromSettings'));

// 5. Re-open via history-btn → History tab active again (no stale settings).
_postedMessages.length = 0;
_fire(historyBtn, 'click', {});
snapshots.push(_snap('afterReopen'));

// 6. Switch Settings → Frequent (also a non-settings tab) saves config.
_fire(tabSettingsBtn, 'click', {});
_populateConfigViaMessage();
_postedMessages.length = 0;
_fire(tabFrequentBtn, 'click', {});
snapshots.push(_snap('afterSwitchSettingsToFrequent'));

process.stdout.write(JSON.stringify(snapshots) + '\n');
"""


class TestSettingsTabBehaviour(unittest.TestCase):
    """Run main.js under Node to verify the Settings sub-tab interactions."""

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

    def test_history_btn_opens_sidebar_history_active(self) -> None:
        snaps = self._run()
        s = snaps[0]
        self.assertTrue(s["sidebarOpen"])
        self.assertTrue(s["historyActive"])
        self.assertFalse(s["settingsActive"])
        self.assertEqual(s["settingsDisplay"], "none")

    def test_settings_tab_switches_panel_and_fetches_config(self) -> None:
        snaps = self._run()
        s = snaps[1]
        self.assertTrue(s["sidebarOpen"])
        self.assertIn("getConfig", s["posted"])
        self.assertTrue(s["settingsActive"])
        self.assertFalse(s["historyActive"])
        self.assertFalse(s["frequentActive"])
        self.assertEqual(s["historyDisplay"], "none")
        self.assertEqual(s["frequentDisplay"], "none")
        self.assertNotEqual(s["settingsDisplay"], "none")

    def test_switch_away_from_settings_saves_config(self) -> None:
        snaps = self._run()
        s = snaps[2]
        self.assertIn(
            "saveConfig",
            s["posted"],
            "switching from Settings → History must post saveConfig",
        )
        self.assertIn(
            "getHistory",
            s["posted"],
            "switching to History must still post getHistory",
        )
        self.assertTrue(s["historyActive"])
        self.assertFalse(s["settingsActive"])

    def test_close_sidebar_from_settings_saves_config(self) -> None:
        snaps = self._run()
        s = snaps[3]
        self.assertIn(
            "saveConfig",
            s["posted"],
            "closing the sidebar from Settings must post saveConfig",
        )
        self.assertFalse(s["sidebarOpen"], "sidebar must close")

    def test_reopen_after_close_resets_to_history(self) -> None:
        snaps = self._run()
        s = snaps[4]
        self.assertTrue(s["sidebarOpen"])
        self.assertTrue(s["historyActive"])
        self.assertFalse(s["settingsActive"])

    def test_switch_settings_to_frequent_saves_config(self) -> None:
        snaps = self._run()
        s = snaps[5]
        self.assertIn(
            "saveConfig",
            s["posted"],
            "switching from Settings → Frequent must post saveConfig",
        )
        self.assertIn(
            "getFrequentTasks",
            s["posted"],
            "switching to Frequent must still post getFrequentTasks",
        )
        self.assertTrue(s["frequentActive"])
        self.assertFalse(s["settingsActive"])


if __name__ == "__main__":
    unittest.main()
