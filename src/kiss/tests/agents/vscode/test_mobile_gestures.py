"""Integration test: mobile touch gestures for history cycling and ghost text accept.

On mobile phones, virtual keyboards lack ArrowUp/ArrowDown and Tab keys.
main.js provides touch gesture equivalents on the input textarea:
  - Swipe right  → accept ghost text (replaces Tab)
  - Swipe up     → cycle to previous history item (replaces ArrowUp)
  - Swipe down   → cycle to next history item (replaces ArrowDown)
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

# Minimal DOM stubs (same pattern as test_history_textarea_resize)
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
        addEventListener: function(t, fn, opts) {
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
    addEventListener: function(type, fn) {
        if (type === 'message') window._messageHandler = fn;
    },
    matchMedia: function() { return { matches: false, addEventListener: function() {} }; },
    innerHeight: 800,
    innerWidth: 400,
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
var marked = {
    parse: function(s) { return s; },
    setOptions: function() {},
    use: function() {},
};
var DOMPurify = { sanitize: function(s) { return s; } };
var console = { log: function(){}, warn: function(){}, error: function(){} };
"""


class TestMobileGestures(unittest.TestCase):
    """Touch gestures on mobile replace ArrowUp/Down and Tab."""

    @classmethod
    def setUpClass(cls):
        cls.main_js = _MAIN_JS.read_text()

    def _run_js(self, test_code):
        full_js = _JS_PREAMBLE + "\n" + self.main_js + "\n" + test_code
        result = subprocess.run(
            ["node", "-e", full_js],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            self.fail(f"Node.js error:\n{result.stderr}")
        return result.stdout.strip()

    def _fire_touch(self, vp, sx, sy, ex, ey):
        """Generate JS to fire touchstart + touchend on inp."""
        ts = f"var {vp}_ts = {{touches: [{{clientX:{sx},clientY:{sy}}}]}};"
        te = (
            f"var {vp}_te = {{changedTouches: [{{clientX:{ex},"
            f"clientY:{ey}}}], preventDefault: function()"
            f" {{ {vp}_prevented = true; }} }};"
        )
        pv = f"var {vp}_prevented = false;"
        s = "for(var _i=0;_i<touchStartFns.length;_i++)"
        s += f" touchStartFns[_i]({vp}_ts);"
        e = "for(var _i=0;_i<touchEndFns.length;_i++)"
        e += f" touchEndFns[_i]({vp}_te);"
        return f"{ts}\n{te}\n{pv}\n{s}\n{e}\n"

    def test_swipe_up_cycles_history(self):
        """Swipe up on the textarea should recall previous history item."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

// Load history
window._messageHandler({ data: {
    type: 'inputHistory',
    tasks: ['task one', 'task two'],
}});

// Get touch listeners
var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

inp.value = '';
""" + self._fire_touch("s1", 100, 200, 100, 180) + r"""
// Small swipe (20px) - should NOT trigger (threshold is 30)
var afterSmall = inp.value;
""" + self._fire_touch("s2", 100, 200, 100, 100) + r"""
// Swipe up 100px - should cycle history
var afterSwipeUp = inp.value;
""" + self._fire_touch("s3", 100, 200, 100, 100) + r"""
// Swipe up again - should get 'task two'
var afterSwipeUp2 = inp.value;

process.stdout.write(JSON.stringify({
    afterSmall: afterSmall,
    afterSwipeUp: afterSwipeUp,
    afterSwipeUp2: afterSwipeUp2,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        # Small swipe shouldn't trigger
        self.assertEqual(results["afterSmall"], "")
        # First swipe up gets histCache[0] = 'task one'
        self.assertEqual(results["afterSwipeUp"], "task one")
        # Second swipe up gets histCache[1] = 'task two'
        self.assertEqual(results["afterSwipeUp2"], "task two")

    def test_swipe_down_cycles_history_forward(self):
        """Swipe down should cycle history forward (newer)."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

window._messageHandler({ data: {
    type: 'inputHistory',
    tasks: ['first', 'second'],
}});

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

inp.value = '';
// Swipe up twice to get to 'second'
""" + self._fire_touch("u1", 100, 200, 100, 100) + \
    self._fire_touch("u2", 100, 200, 100, 100) + r"""
var atSecond = inp.value;

// Swipe down to go back to 'first'
""" + self._fire_touch("d1", 100, 100, 100, 200) + r"""
var afterDown = inp.value;

// Swipe down again to clear
""" + self._fire_touch("d2", 100, 100, 100, 200) + r"""
var afterClear = inp.value;

// Swipe down when histIdx < 0 - should not crash
""" + self._fire_touch("d3", 100, 100, 100, 200) + r"""
var afterExtra = inp.value;

process.stdout.write(JSON.stringify({
    atSecond: atSecond,
    afterDown: afterDown,
    afterClear: afterClear,
    afterExtra: afterExtra,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        self.assertEqual(results["atSecond"], "second")
        self.assertEqual(results["afterDown"], "first")
        self.assertEqual(results["afterClear"], "")
        # Extra swipe down when already at bottom shouldn't change anything
        self.assertEqual(results["afterExtra"], "")

    def test_swipe_right_accepts_ghost(self):
        """Swipe right should accept the ghost text suggestion."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

inp.value = 'please ';

// Set ghost text via the ghost message (must match inp.value for query)
window._messageHandler({ data: {
    type: 'ghost',
    suggestion: 'complete this task',
    query: 'please ',
}});

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

// Swipe right 100px
""" + self._fire_touch("r1", 100, 200, 200, 200) + r"""
var afterSwipeRight = inp.value;

process.stdout.write(JSON.stringify({
    afterSwipeRight: afterSwipeRight,
    prevented: r1_prevented,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        # Ghost text should be appended to inp.value
        self.assertIn("complete this task", results["afterSwipeRight"])
        self.assertTrue(results["prevented"])

    def test_swipe_right_no_ghost_does_nothing(self):
        """Swipe right without ghost text should not change input."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

inp.value = 'hello';

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

""" + self._fire_touch("r1", 100, 200, 200, 200) + r"""
process.stdout.write(JSON.stringify({
    value: inp.value,
    prevented: r1_prevented,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        self.assertEqual(results["value"], "hello")
        self.assertFalse(results["prevented"])

    def test_swipe_up_resizes_textarea(self):
        """Swipe up to a long history item must resize the textarea."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

window._messageHandler({ data: {
    type: 'inputHistory',
    tasks: ['short', 'A very long task that spans multiple'
      + ' lines\nand needs more height\nto display properly'],
}});

Object.defineProperty(inp, 'scrollHeight', {
    get: function() {
        if (inp.value && inp.value.indexOf('\n') >= 0) return 120;
        if (inp.value && inp.value.length > 50) return 80;
        return 20;
    },
    configurable: true,
});

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

inp.value = '';
inp.style.height = '20px';

// Swipe up twice to get the long task
""" + self._fire_touch("u1", 100, 200, 100, 100) + \
    self._fire_touch("u2", 100, 200, 100, 100) + r"""
process.stdout.write(JSON.stringify({
    value: inp.value,
    height: inp.style.height,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        self.assertIn("very long task", results["value"])
        self.assertEqual(results["height"], "120px")

    def test_diagonal_swipe_ignored(self):
        """A diagonal swipe (equal dx and dy) should not trigger any action."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

window._messageHandler({ data: {
    type: 'inputHistory',
    tasks: ['task one'],
}});

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

inp.value = '';
// Diagonal swipe: dx=50, dy=-50 (equal magnitude)
""" + self._fire_touch("diag", 100, 200, 150, 150) + r"""
process.stdout.write(JSON.stringify({
    value: inp.value,
    prevented: diag_prevented,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        # Diagonal: absDx == absDy, so absDx > absDy is false and absDy > absDx is false
        self.assertEqual(results["value"], "")
        self.assertFalse(results["prevented"])

    def test_multi_touch_ignored(self):
        """Multi-finger touches should not trigger gestures."""
        js = r"""
var inp = _elements['task-input'];
var autocomplete = _elements['autocomplete'];
autocomplete.style.display = 'none';

window._messageHandler({ data: {
    type: 'inputHistory',
    tasks: ['task one'],
}});

var touchStartFns = inp._listeners['touchstart'] || [];
var touchEndFns = inp._listeners['touchend'] || [];

inp.value = '';
// Multi-touch start (2 fingers)
var mts = { touches: [{ clientX: 100, clientY: 200 }, { clientX: 150, clientY: 200 }] };
for (var _i = 0; _i < touchStartFns.length; _i++) touchStartFns[_i](mts);
// Multi-touch end (2 changed touches)
var mte_prevented = false;
var mte = {
    changedTouches: [{ clientX: 100, clientY: 100 }, { clientX: 150, clientY: 100 }],
    preventDefault: function() { mte_prevented = true; },
};
for (var _i = 0; _i < touchEndFns.length; _i++) touchEndFns[_i](mte);

process.stdout.write(JSON.stringify({
    value: inp.value,
    prevented: mte_prevented,
}) + '\n');
"""
        results = json.loads(self._run_js(js))
        self.assertEqual(results["value"], "")
        self.assertFalse(results["prevented"])


if __name__ == "__main__":
    unittest.main()
