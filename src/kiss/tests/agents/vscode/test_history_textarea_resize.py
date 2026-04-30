"""Integration test: textarea must resize when cycling history with ArrowUp/ArrowDown.

When a long task text is recalled from history via ArrowUp/ArrowDown, the textarea
height must be adjusted to fit the content (up to 200px max). Previously, the
history cycling code set inp.value but did not recalculate the textarea height,
leaving it at its previous (small) size even when the text overflowed.
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

# Minimal DOM stubs matching the pattern used by other main.js tests
_JS_PREAMBLE = r"""
var _elements = {};

function _makeEl(tag) {
    var el = {
        tagName: tag,
        id: '',
        className: '',
        textContent: '',
        innerHTML: '',
        value: '',
        style: {},
        dataset: {},
        disabled: false,
        children: [],
        _listeners: {},
        _heightLog: [],
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
        getBoundingClientRect: function() { return {top:0,left:0,width:100,height:20}; },
        insertBefore: function(n, ref) { this.children.push(n); return n; },
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
    // Intercept style.height setter to log assignments
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

// Pre-create needed elements
_elements['output'] = _makeEl('div');
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
_elements['history-search-clear'] = _makeEl('span');
_elements['history-list'] = _makeEl('div');
_elements['history-btn'] = _makeEl('button');
_elements['task-panel'] = _makeEl('div');
_elements['tab-bar'] = _makeEl('div');
_elements['tab-list'] = _makeEl('div');
_elements['config-btn'] = _makeEl('button');
_elements['config-panel'] = _makeEl('div');
_elements['clear-btn'] = _makeEl('button');
_elements['remote-url'] = _makeEl('div');
_elements['autocomplete'] = _makeEl('div');
_elements['ghost-text'] = _makeEl('span');
_elements['input-row'] = _makeEl('div');
_elements['merge-toolbar'] = _makeEl('div');
_elements['merge-accept-all-btn'] = _makeEl('button');
_elements['merge-reject-all-btn'] = _makeEl('button');
_elements['merge-accept-file-btn'] = _makeEl('button');
_elements['merge-reject-file-btn'] = _makeEl('button');
_elements['merge-prev-btn'] = _makeEl('button');
_elements['merge-next-btn'] = _makeEl('button');
_elements['merge-file-label'] = _makeEl('span');
_elements['merge-counter'] = _makeEl('span');
_elements['merge-accept-btn'] = _makeEl('button');
_elements['merge-reject-btn'] = _makeEl('button');

// VS Code API stub
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

// Global stubs Node.js doesn't have
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

var marked = {
    parse: function(s) { return s; },
    setOptions: function() {},
    use: function() {},
};

var DOMPurify = { sanitize: function(s) { return s; } };
var console = { log: function(){}, warn: function(){}, error: function(){} };
"""

_JS_TEST = r"""
// After main.js has been loaded and executed, the inp keydown listener is active.
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

// Simulate receiving history from backend
var msg = { data: {
    type: 'inputHistory',
    tasks: [
        'short task',
        'This is a very long task description that spans multiple lines'
        + ' and would\nrequire the textarea to grow taller to display'
        + ' all of its content\nproperly without scrolling',
    ],
}};
window._messageHandler(msg);

// --- Test ArrowUp ---
// Reset height tracking: simulate a tall scrollHeight when long text is set
var _origScrollHeight = 20;
Object.defineProperty(inp, 'scrollHeight', {
    get: function() {
        // When value has newlines, simulate a taller scrollHeight
        if (inp.value && inp.value.indexOf('\n') >= 0) return 120;
        if (inp.value && inp.value.length > 50) return 80;
        return _origScrollHeight;
    },
    configurable: true,
});

// Press ArrowUp once -> should get 'short task' (histCache[0])
inp.value = '';
inp.style.height = '20px';
var e1 = { key: 'ArrowUp', preventDefault: function(){}, shiftKey: false };
// Find the keydown listener on inp
var keydownListeners = inp._listeners['keydown'] || [];
for (var i = 0; i < keydownListeners.length; i++) {
    keydownListeners[i](e1);
}

var heightAfterFirst = inp.style.height;

// Press ArrowUp again -> should get the long task (histCache[1])
inp.style.height = '20px'; // reset to small
var e2 = { key: 'ArrowUp', preventDefault: function(){}, shiftKey: false };
for (var i = 0; i < keydownListeners.length; i++) {
    keydownListeners[i](e2);
}

var heightAfterLong = inp.style.height;

// Press ArrowDown -> should go back to 'short task'
inp.style.height = '200px'; // set to large (should shrink)
var e3 = { key: 'ArrowDown', preventDefault: function(){}, shiftKey: false };
for (var i = 0; i < keydownListeners.length; i++) {
    keydownListeners[i](e3);
}

var heightAfterDown = inp.style.height;

// Press ArrowDown again -> should clear input
var e4 = { key: 'ArrowDown', preventDefault: function(){}, shiftKey: false };
for (var i = 0; i < keydownListeners.length; i++) {
    keydownListeners[i](e4);
}

var heightAfterClear = inp.style.height;

// Output results
var results = {
    heightAfterFirst: heightAfterFirst,
    heightAfterLong: heightAfterLong,
    heightAfterDown: heightAfterDown,
    heightAfterClear: heightAfterClear,
    valueAfterFirst: inp.value,
};

// Print JSON results
var output = JSON.stringify(results);
// Use process.stdout to print
process.stdout.write(output + '\n');
"""


class TestHistoryTextareaResize(unittest.TestCase):
    """Textarea must auto-resize when cycling task history with arrow keys."""

    @classmethod
    def setUpClass(cls):
        cls.main_js = _MAIN_JS.read_text()

    def _run_js(self, test_code):
        """Run main.js + test code in Node.js and return stdout."""
        # We need to capture window.addEventListener('message', ...) handler
        # so we can invoke it in our test code.
        patched_preamble = _JS_PREAMBLE + r"""
// Capture the message handler registered via window.addEventListener
var _origAddEventListener = window.addEventListener;
window.addEventListener = function(type, fn) {
    if (type === 'message') window._messageHandler = fn;
};
"""
        full_js = patched_preamble + "\n" + self.main_js + "\n" + test_code
        result = subprocess.run(
            ["node", "-e", full_js],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            self.fail(f"Node.js error:\n{result.stderr}")
        return result.stdout.strip()

    def test_arrow_up_resizes_textarea_for_long_history(self):
        """ArrowUp into a long history item must resize the textarea height."""
        import json

        output = self._run_js(_JS_TEST)
        results = json.loads(output)

        # After ArrowUp to the long task, height must NOT stay at '20px'
        # It should be expanded (e.g., '120px' based on our scrollHeight mock)
        self.assertNotEqual(
            results["heightAfterLong"],
            "20px",
            "Textarea height was not adjusted after ArrowUp to a long history item",
        )

        # The height should be set to the scrollHeight value (120px in our mock)
        self.assertEqual(results["heightAfterLong"], "120px")

    def test_arrow_down_resizes_textarea(self):
        """ArrowDown back to shorter item must shrink the textarea."""
        import json

        output = self._run_js(_JS_TEST)
        results = json.loads(output)

        # After ArrowDown from long to short, height should adjust
        self.assertNotEqual(
            results["heightAfterDown"],
            "200px",
            "Textarea height was not adjusted after ArrowDown to shorter history item",
        )

    def test_arrow_down_to_empty_resizes_textarea(self):
        """ArrowDown past all history (empty input) must reset textarea height."""
        import json

        output = self._run_js(_JS_TEST)
        results = json.loads(output)

        # After ArrowDown to empty, height should be reset to scrollHeight (20px)
        self.assertEqual(results["heightAfterClear"], "20px")


if __name__ == "__main__":
    unittest.main()
