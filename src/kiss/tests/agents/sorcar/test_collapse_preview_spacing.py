"""Tests for collapse preview text spacing.

Verifies that collapsePreview adds spaces between subsequent lines when
building the collapsed summary text.  The bug: innerText on hidden elements
(display:none due to .collapsed CSS) falls back to textContent behavior,
which concatenates text from adjacent block-level elements without any
separator.  The fix: use textContent and walk child elements to insert
separators between element nodes.
"""

import json
import re
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


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _extract_function(source: str, name: str) -> str:
    """Extract a function definition from the main.js source.

    Finds ``function <name>(`` and returns everything up to and including
    its closing brace by tracking brace depth.

    Args:
        source: Full JavaScript source text.
        name: Function name to extract.

    Returns:
        The complete function source text.
    """
    pattern = rf"function {name}\("
    match = re.search(pattern, source)
    assert match, f"Function {name} not found in source"
    start = match.start()
    brace_start = source.index("{", match.end())
    depth = 1
    i = brace_start + 1
    while depth > 0:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    return source[start:i]


_NODE_SHIM = r"""
function _MiniEl(tag) {
    this.tagName = tag || 'div';
    this.style = {};
    this.childNodes = [];
    this.children = [];
    this.parentElement = null;
    this.firstChild = null;
    this.nodeType = 1;
    this._ownClasses = [];
}

_MiniEl.prototype.appendChild = function(c) {
    c.parentElement = this;
    this.childNodes.push(c);
    if (c.nodeType === 1) this.children.push(c);
    if (!this.firstChild) this.firstChild = c;
    return c;
};

_MiniEl.prototype.insertBefore = function(newNode, refNode) {
    newNode.parentElement = this;
    var idx = this.childNodes.indexOf(refNode);
    if (idx === -1) { this.childNodes.push(newNode); }
    else { this.childNodes.splice(idx, 0, newNode); }
    this.children = this.childNodes.filter(function(c) { return c.nodeType === 1; });
    this.firstChild = this.childNodes[0] || null;
    return newNode;
};

_MiniEl.prototype.classList = Object.create(null);
Object.defineProperty(_MiniEl.prototype, 'classList', {
    get: function() {
        if (!this._cl) {
            var self = this;
            self._cl = {
                add: function(c) {
                    if (self._ownClasses.indexOf(c) === -1)
                        self._ownClasses.push(c);
                },
                remove: function(c) {
                    var i = self._ownClasses.indexOf(c);
                    if (i !== -1) self._ownClasses.splice(i, 1);
                },
                toggle: function(c) { if (this.contains(c)) this.remove(c); else this.add(c); },
                contains: function(c) { return self._ownClasses.indexOf(c) !== -1; },
            };
        }
        return this._cl;
    }
});

_MiniEl.prototype.querySelector = function(sel) {
    var cls = sel.replace(/^\./,'');
    for (var i = 0; i < this.children.length; i++) {
        if (this.children[i].classList.contains(cls)) return this.children[i];
        var found = this.children[i].querySelector(sel);
        if (found) return found;
    }
    return null;
};

_MiniEl.prototype.addEventListener = function() {};

// textContent: concatenate all descendant text WITHOUT separators (like real textContent)
function _computeTextContent(node) {
    if (node.nodeType === 3) return node.textContent || '';
    var t = '';
    for (var i = 0; i < node.childNodes.length; i++) {
        t += _computeTextContent(node.childNodes[i]);
    }
    return t;
}

function _mkTextNode(text) {
    return { nodeType: 3, textContent: text, parentElement: null, childNodes: [] };
}

Object.defineProperty(_MiniEl.prototype, 'textContent', {
    get: function() { return _computeTextContent(this); },
    set: function(v) {
        this.childNodes = [];
        this.children = [];
        this.firstChild = null;
        if (v) {
            var tn = _mkTextNode(v);
            tn.parentElement = this;
            this.childNodes.push(tn);
            this.firstChild = tn;
        }
    },
});

// KEY: innerText on hidden elements falls back to textContent (no block separators).
// Our shim always returns textContent to simulate the collapsed (display:none) case.
Object.defineProperty(_MiniEl.prototype, 'innerText', {
    get: function() { return _computeTextContent(this); },
    set: function(v) { this.textContent = v; },
});

function mkTestEl(tag) {
    return new _MiniEl(tag);
}

// Build a panel DOM structure with body content from child specs
function buildPanel(bodyChildren) {
    // Panel element (like .tc)
    var panel = mkTestEl('div');
    panel.classList.add('tc');

    // Header (like .tc-h)
    var hdr = mkTestEl('div');
    hdr.classList.add('tc-h');
    hdr.textContent = 'Bash';
    panel.appendChild(hdr);

    // Add collapse infrastructure: chevron and preview on header
    var chv = mkTestEl('span');
    chv.classList.add('collapse-chv');
    chv.textContent = '\u25BE';
    hdr.insertBefore(chv, hdr.firstChild);

    var prev = mkTestEl('span');
    prev.classList.add('collapse-preview');
    hdr.appendChild(prev);

    // Body element (like .tc-b)
    var body = mkTestEl('div');
    body.classList.add('tc-b');
    panel.appendChild(body);

    // Add child elements to body
    for (var i = 0; i < bodyChildren.length; i++) {
        var spec = bodyChildren[i];
        var child = mkTestEl(spec.tag || 'div');
        if (spec.cls) child.classList.add(spec.cls);
        child.textContent = spec.text;
        body.appendChild(child);
    }

    return panel;
}
"""


def _build_test_script(body_children_json: str, collapse: bool = True) -> str:
    """Build a Node.js script that creates a panel, collapses it, and prints the preview text.

    Args:
        body_children_json: JSON array of {tag, cls, text} specs for body children.
        collapse: Whether to collapse the panel.

    Returns:
        Node.js script string.
    """
    source = _MAIN_JS.read_text()
    collect_fn = _extract_function(source, "collectText")
    collapse_fn = _extract_function(source, "collapsePreview")
    mkel_fn = _extract_function(source, "mkEl")

    script = _NODE_SHIM + "\n"
    script += "var document = { createElement: mkTestEl };\n"
    script += mkel_fn + "\n"
    script += collect_fn + "\n"
    script += collapse_fn + "\n"
    script += f"var bodyChildren = {body_children_json};\n"
    script += "var panel = buildPanel(bodyChildren);\n"
    if collapse:
        script += "panel.classList.add('collapsed');\n"
    script += "collapsePreview(panel);\n"
    script += "var prev = panel.querySelector('.collapse-preview');\n"
    script += "console.log(JSON.stringify(_computeTextContent(prev)));\n"
    return script


class TestCollapsePreviewSpacingStructural(unittest.TestCase):
    """Structural: verify collapsePreview uses collectText instead of innerText."""

    source: str
    fn_source: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _MAIN_JS.read_text()
        cls.fn_source = _extract_function(cls.source, "collapsePreview")

    def test_uses_collect_text_not_inner_text(self) -> None:
        """collapsePreview must use collectText (or equivalent) instead of ch.innerText."""
        assert "ch.innerText" not in self.fn_source, (
            "collapsePreview still uses ch.innerText which fails on hidden elements"
        )

    def test_collect_text_function_exists(self) -> None:
        """A collectText helper function must exist in main.js."""
        assert "function collectText(" in self.source, (
            "Expected collectText helper function in main.js"
        )


class TestCollapsePreviewSpacingBehavioral(unittest.TestCase):
    """Behavioral: verify collapsed preview text has spaces between lines."""

    def _get_preview(self, body_children: list[dict[str, str]], collapse: bool = True) -> str:
        """Build a panel, collapse it, and return the preview text."""
        script = _build_test_script(json.dumps(body_children), collapse)
        r = _run_node(script)
        assert r.returncode == 0, f"Node.js error:\nstderr: {r.stderr}\nstdout: {r.stdout}"
        result: str = json.loads(r.stdout.strip())
        return result

    def test_adjacent_divs_have_space(self) -> None:
        """Two adjacent <div> elements must have a space between them in the preview."""
        children = [
            {"tag": "div", "cls": "extra", "text": "key1: value1"},
            {"tag": "div", "cls": "extra", "text": "key2: value2"},
        ]
        preview = self._get_preview(children)
        assert "value1 key2" in preview, (
            f"Expected space between adjacent div texts, got: {preview!r}"
        )

    def test_multiple_divs_all_separated(self) -> None:
        """Three adjacent divs must each be separated by spaces."""
        children = [
            {"tag": "div", "cls": "extra", "text": "line1"},
            {"tag": "div", "cls": "extra", "text": "line2"},
            {"tag": "div", "cls": "extra", "text": "line3"},
        ]
        preview = self._get_preview(children)
        assert "line1 line2 line3" in preview, (
            f"Expected all lines separated by spaces, got: {preview!r}"
        )

    def test_pre_blocks_separated(self) -> None:
        """Adjacent <pre> blocks must have spaces between them."""
        children = [
            {"tag": "pre", "text": "command1"},
            {"tag": "pre", "text": "command2"},
        ]
        preview = self._get_preview(children)
        assert "command1 command2" in preview, (
            f"Expected space between pre blocks, got: {preview!r}"
        )

    def test_div_then_pre_separated(self) -> None:
        """A div followed by a pre must have a space between them."""
        children = [
            {"tag": "div", "cls": "extra", "text": "description: hello"},
            {"tag": "pre", "text": "echo hello"},
        ]
        preview = self._get_preview(children)
        assert "hello echo" in preview, (
            f"Expected space between div and pre, got: {preview!r}"
        )

    def test_empty_panel_collapsed(self) -> None:
        """A collapsed panel with no body content should produce empty or header-only preview."""
        preview = self._get_preview([])
        assert isinstance(preview, str)

    def test_not_collapsed_produces_empty_preview(self) -> None:
        """A non-collapsed panel should have empty preview text."""
        children = [{"tag": "div", "cls": "extra", "text": "content"}]
        preview = self._get_preview(children, collapse=False)
        assert preview == "", f"Expected empty preview for non-collapsed panel, got: {preview!r}"

    def test_single_div_no_extra_spaces(self) -> None:
        """A single div should produce clean text without extra spaces."""
        children = [{"tag": "div", "cls": "extra", "text": "key: value"}]
        preview = self._get_preview(children)
        assert "key: value" in preview, (
            f"Expected clean single-div preview, got: {preview!r}"
        )


if __name__ == "__main__":
    unittest.main()
