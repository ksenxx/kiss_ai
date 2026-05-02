"""Integration test: remote chat must show exactly one tab.

In the remote chat webview (``<body class="remote-chat">``):

1. When a new chat is created, all other tabs must be closed so the
   tab bar contains exactly the freshly created chat.  The backend
   must receive ``closeTab`` messages for every removed tab.

2. On initial load with multiple tabs persisted in ``vscode.getState()``,
   only the active tab must survive (no resurrection of stale tabs).

The same code paths in the VS Code extension webview (no
``remote-chat`` body class) must NOT close other tabs — multi-tab is
the standard behavior there.

These tests load ``media/main.js`` in Node.js with a minimal DOM stub
and drive the same ``createNewTab`` code path the ``+`` button does,
using real frontend logic with no test doubles for the JS itself.
"""

import json
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

# Minimal DOM stub shared by all scenarios.  Mirrors the stub used by
# test_history_textarea_resize.py — adapted to expose tab-bar elements
# and the body classList that this test toggles.
_JS_PREAMBLE = r"""
var _elements = {};

function _makeEl(tag) {
    var el = {
        tagName: tag,
        id: '',
        className: '',
        textContent: '',
        innerHTML: '',
        title: '',
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
                if (i >= 0) this._c.splice(i,1);
            },
            contains: function(c) { return this._c.indexOf(c) >= 0; },
            toggle: function(c) {
                if (this.contains(c)) this.remove(c); else this.add(c);
            },
        },
        querySelector: function() { return null; },
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
            return {top:0,left:0,width:100,height:20};
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
    var _realStyle = { height: '', display: '', color: '' };
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
    createDocumentFragment: function() {
        var frag = _makeEl('fragment');
        frag.appendChild = function(c) { this.children.push(c); return c; };
        return frag;
    },
    body: _makeEl('body'),
    addEventListener: function() {},
    documentElement: _makeEl('html'),
};

var _bodyClasses = [];
document.body.classList = {
    _c: _bodyClasses,
    add: function(c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
    remove: function(c) {
        var i = this._c.indexOf(c);
        if (i >= 0) this._c.splice(i,1);
    },
    contains: function(c) { return this._c.indexOf(c) >= 0; },
    toggle: function(c) {
        if (this.contains(c)) this.remove(c); else this.add(c);
    },
};

// Pre-create elements main.js looks up by id during initialization.
var _ids = [
    'output','welcome','task-input','send-btn','stop-btn','upload-btn',
    'model-btn','model-dropdown','model-search','model-list','model-name',
    'file-chips','status-text','status-tokens','status-budget','status-steps',
    'sidebar','history-search','history-search-clear','history-list',
    'history-btn','task-panel','task-panel-text','task-panel-chevron',
    'tab-bar','tab-list','config-btn','config-panel','clear-btn',
    'remote-url','autocomplete','ghost-text','input-row','input-area',
    'app','merge-toolbar','merge-accept-all-btn','merge-reject-all-btn',
    'merge-accept-file-btn','merge-reject-file-btn','merge-prev-btn',
    'merge-next-btn','merge-file-label','merge-counter','merge-accept-btn',
    'merge-reject-btn','ask-user-slot',
];
for (var i = 0; i < _ids.length; i++) {
    _elements[_ids[i]] = _makeEl('div');
    _elements[_ids[i]].id = _ids[i];
}

// Capture every postMessage so tests can assert the WS message stream.
var _postedMessages = [];
// ``__INITIAL_STATE__`` is replaced per-scenario by the test harness.
var _INITIAL_STATE = __INITIAL_STATE__;
var _state = _INITIAL_STATE;
var acquireVsCodeApi = function() {
    return {
        postMessage: function(msg) { _postedMessages.push(msg); },
        getState: function() { return _state; },
        setState: function(s) { _state = s; },
    };
};

var window = {
    addEventListener: function() {},
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
    this.type = type; this.detail = (opts||{}).detail;
};
var MessageEvent = function(type, opts) {
    this.type = type; this.data = (opts||{}).data;
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

// ``__BODY_CLASS_INIT__`` is replaced by the harness with either ''
// (extension webview) or 'remote-chat'.  Set BEFORE main.js executes
// because the file's IIFE reads document.body.classList at top level.
var _BODY_CLASS_INIT = __BODY_CLASS_INIT__;
if (_BODY_CLASS_INIT) document.body.classList.add(_BODY_CLASS_INIT);
"""


def _run_main_js(
    initial_state: dict | None,
    body_class: str,
    test_code: str,
) -> dict:
    """Load main.js + scenario test code in Node.js and parse JSON stdout."""
    main_js = _MAIN_JS.read_text()
    preamble = _JS_PREAMBLE.replace(
        "__INITIAL_STATE__",
        json.dumps(initial_state) if initial_state is not None else "null",
    ).replace("__BODY_CLASS_INIT__", json.dumps(body_class))
    full_js = preamble + "\n" + main_js + "\n" + test_code
    result = subprocess.run(
        ["node", "-e", full_js],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise AssertionError(f"Node.js error:\n{result.stderr}")
    parsed: dict = json.loads(result.stdout.strip())
    return parsed


# Probe used by the "+" button scenarios: seeds tabs via the persisted
# state restore path (3 tabs), then dispatches a click on the
# ``.chat-tab-add`` element rendered by ``renderTabBar``.
_PROBE_PLUS_BUTTON = r"""
var tabBar = _elements['tab-bar'];
// renderTabBar appends children to #tab-list and the "+" to #tab-bar.
// Walk tab-bar's children for the .chat-tab-add element.
function _findAddBtn() {
    for (var i = 0; i < tabBar.children.length; i++) {
        var c = tabBar.children[i];
        if (c.className && c.className.indexOf('chat-tab-add') >= 0) return c;
    }
    return null;
}
var addBtn = _findAddBtn();
if (!addBtn) {
    process.stdout.write(JSON.stringify({error: 'no add button'}) + '\n');
    process.exit(0);
}
// Snapshot pre-click and clear posted messages to focus on the click effects.
var stateBefore = acquireVsCodeApi().getState();
_postedMessages.length = 0;
// Invoke the click handler that createNewTab is bound to.
var listeners = addBtn._listeners['click'] || [];
for (var i = 0; i < listeners.length; i++) listeners[i]();
var stateAfter = acquireVsCodeApi().getState();
var closeTabMsgs = _postedMessages.filter(function(m) {
    return m && m.type === 'closeTab';
});
var newChatMsgs = _postedMessages.filter(function(m) {
    return m && m.type === 'newChat';
});
process.stdout.write(JSON.stringify({
    tabsBeforeCount: stateBefore && stateBefore.tabs ? stateBefore.tabs.length : 0,
    tabsAfterCount: stateAfter && stateAfter.tabs ? stateAfter.tabs.length : 0,
    activeIdxAfter: stateAfter ? stateAfter.activeTabIndex : -1,
    closeTabMsgIds: closeTabMsgs.map(function(m) { return m.tabId; }),
    newChatMsgCount: newChatMsgs.length,
    bodyHasRemoteChat: document.body.classList.contains('remote-chat'),
}) + '\n');
"""


# Probe used to verify the initial restoration prune path:
# main.js has already run with 3 persisted tabs; just dump the
# restored state without firing any click.
_PROBE_RESTORATION = r"""
var st = acquireVsCodeApi().getState();
process.stdout.write(JSON.stringify({
    tabsCount: st && st.tabs ? st.tabs.length : 0,
    activeIdx: st ? st.activeTabIndex : -1,
    activeChatId: st ? st.chatId : '',
    persistedChatIds: st && st.tabs ? st.tabs.map(function(t) {
        return t.chatId;
    }) : [],
    bodyHasRemoteChat: document.body.classList.contains('remote-chat'),
}) + '\n');
"""


def _three_tab_state() -> dict:
    """Persisted state with three tabs, the middle one active."""
    return {
        "tabs": [
            {"title": "tab A", "chatId": "id-a", "backendChatId": "be-a"},
            {"title": "tab B", "chatId": "id-b", "backendChatId": "be-b"},
            {"title": "tab C", "chatId": "id-c", "backendChatId": "be-c"},
        ],
        "activeTabIndex": 1,
        "chatId": "id-b",
    }


class TestRemoteChatNewChatClosesOthers(unittest.TestCase):
    """Clicking ``+`` in remote chat must close every other tab."""

    def test_new_chat_in_remote_closes_all_other_tabs(self) -> None:
        """Three tabs restored, ``+`` clicked → exactly the new tab remains."""
        out = _run_main_js(
            _three_tab_state(), "remote-chat", _PROBE_PLUS_BUTTON,
        )
        self.assertTrue(out["bodyHasRemoteChat"])
        # Restoration in remote-chat already collapses to 1 tab, so
        # before the "+" click only the active tab persists.
        self.assertEqual(out["tabsBeforeCount"], 1)
        # After clicking "+" exactly one tab remains (the new one).
        self.assertEqual(out["tabsAfterCount"], 1)
        self.assertEqual(out["activeIdxAfter"], 0)
        # newChat is posted exactly once for the freshly created tab.
        self.assertEqual(out["newChatMsgCount"], 1)

    def test_new_chat_after_seeding_three_tabs_in_remote(self) -> None:
        """Seed 3 tabs at runtime in remote-chat, click "+" → all 3 closed."""
        # This bypasses the load-time prune by injecting tabs after
        # main.js has finished initializing — exercising createNewTab's
        # own close-others guard rather than the restoration guard.
        probe = r"""
// Inject extra tabs by reaching into the IIFE-private state via the
// persisted state mirror: we rebuild via createNewTab calls, but
// remote-chat collapses each call.  Instead, simulate via a direct
// state seed: replay the persistence by calling makeTab through the
// click sequence in extension mode then flip to remote-chat.

// Easier: temporarily remove remote-chat, click "+" twice to grow to
// three tabs, then add remote-chat back and click "+" once more.
document.body.classList.remove('remote-chat');
var tabBar = _elements['tab-bar'];
function _addBtn() {
    for (var i = 0; i < tabBar.children.length; i++) {
        var c = tabBar.children[i];
        if (c.className && c.className.indexOf('chat-tab-add') >= 0) return c;
    }
    return null;
}
function _click(el) {
    var ls = el._listeners['click'] || [];
    for (var i = 0; i < ls.length; i++) ls[i]();
}
_click(_addBtn());
_click(_addBtn());
var afterTwoClicks = acquireVsCodeApi().getState().tabs.length;
// Now switch to remote-chat and click "+" once more.
document.body.classList.add('remote-chat');
_postedMessages.length = 0;
_click(_addBtn());
var st = acquireVsCodeApi().getState();
var closeTabMsgs = _postedMessages.filter(function(m) {
    return m && m.type === 'closeTab';
});
process.stdout.write(JSON.stringify({
    tabsAfterTwoExtensionClicks: afterTwoClicks,
    tabsAfterRemoteClick: st.tabs.length,
    closeTabMsgCount: closeTabMsgs.length,
}) + '\n');
"""
        out = _run_main_js(None, "", probe)
        # In extension mode, two "+" clicks grow from 1 to 3 tabs.
        self.assertEqual(out["tabsAfterTwoExtensionClicks"], 3)
        # After flipping to remote-chat and clicking "+" once, only the
        # newly created tab survives — the 3 prior tabs are closed.
        self.assertEqual(out["tabsAfterRemoteClick"], 1)
        # And the backend was told to close exactly those 3 tabs.
        self.assertEqual(out["closeTabMsgCount"], 3)


class TestRemoteChatRestorationPrunesTabs(unittest.TestCase):
    """Restoring multiple persisted tabs in remote-chat keeps only active."""

    def test_remote_load_with_three_persisted_tabs_keeps_only_active(self):
        """3-tab session storage → tabs collapse to just the active one."""
        out = _run_main_js(
            _three_tab_state(), "remote-chat", _PROBE_RESTORATION,
        )
        self.assertTrue(out["bodyHasRemoteChat"])
        self.assertEqual(out["tabsCount"], 1)
        self.assertEqual(out["activeIdx"], 0)
        # The surviving tab is the one that was originally active
        # (index 1, chatId "id-b").
        self.assertEqual(out["activeChatId"], "id-b")
        self.assertEqual(out["persistedChatIds"], ["id-b"])

    def test_extension_load_with_three_persisted_tabs_keeps_all(self) -> None:
        """No remote-chat class → all 3 restored tabs survive load."""
        out = _run_main_js(_three_tab_state(), "", _PROBE_RESTORATION)
        self.assertFalse(out["bodyHasRemoteChat"])
        self.assertEqual(out["tabsCount"], 3)
        self.assertEqual(out["activeIdx"], 1)
        self.assertEqual(out["activeChatId"], "id-b")
        self.assertEqual(
            out["persistedChatIds"], ["id-a", "id-b", "id-c"],
        )


class TestExtensionNewChatPreservesOthers(unittest.TestCase):
    """In the VS Code extension webview, ``+`` adds a tab without closing."""

    def test_new_chat_in_extension_keeps_existing_tabs(self) -> None:
        """3 tabs restored, no remote-chat → ``+`` grows to 4 tabs."""
        out = _run_main_js(_three_tab_state(), "", _PROBE_PLUS_BUTTON)
        self.assertFalse(out["bodyHasRemoteChat"])
        self.assertEqual(out["tabsBeforeCount"], 3)
        # Extension mode appends — never closes.
        self.assertEqual(out["tabsAfterCount"], 4)
        self.assertEqual(out["closeTabMsgIds"], [])
        self.assertEqual(out["newChatMsgCount"], 1)


if __name__ == "__main__":
    unittest.main()
