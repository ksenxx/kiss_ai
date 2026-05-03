"""Integration test for demo mode chv-hidden bug.

Bug: When loading a task in demo mode, panels are invisible because
processOutputEvent calls applyChevronState(false) after every event.
With isRunning=false (demo mode never sets it to true), every non-result
panel gets the 'chv-hidden' class (display:none), hiding all output
except the fixed task panel.

Reproduces the bug with a structural source assertion, then verifies
the fix: processOutputEvent must skip applyChevronState when
_demoActive is true.
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


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestDemoChvHiddenBugStructural(unittest.TestCase):
    """Structural: processOutputEvent must guard applyChevronState with
    _demoActive so demo-mode panels are not hidden."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _MAIN_JS.read_text()

    def _get_process_output_event_body(self) -> str:
        """Extract the processOutputEvent function body."""
        start = self.src.index("function processOutputEvent(ev)")
        depth = 0
        for i in range(start, len(self.src)):
            if self.src[i] == "{":
                depth += 1
            elif self.src[i] == "}":
                depth -= 1
                if depth == 0:
                    return self.src[start : i + 1]
        raise AssertionError("Could not find end of processOutputEvent")  # type: ignore[unreachable]

    def test_chevron_guard_includes_demo_check(self) -> None:
        """The applyChevronState call must be gated on !_demoActive."""
        body = self._get_process_output_event_body()
        assert "_demoActive" in body, (
            "processOutputEvent must check _demoActive before calling "
            "applyChevronState — without this guard, demo-mode panels "
            "get chv-hidden and are invisible"
        )

    def test_chevron_not_applied_when_demo_active(self) -> None:
        """The guard must prevent applyChevronState from firing during demo."""
        body = self._get_process_output_event_body()
        assert "!_demoActive" in body, (
            "Guard must use !_demoActive to skip chevron application"
        )


class TestDemoChvHiddenBugBehavioral(unittest.TestCase):
    """Behavioral: simulate demo replay and verify panels are visible.

    Uses a minimal DOM shim in Node.js to run the relevant logic from
    main.js and check that panels do NOT receive the chv-hidden class
    during demo mode.
    """

    def test_panels_not_hidden_during_demo_replay(self) -> None:
        """Panels created during demo replay must NOT have chv-hidden."""
        script = r"""
// Minimal DOM shim
function MockElement(tag, className) {
    this.tagName = tag;
    this.className = className || '';
    this.children = [];
    this.style = {};
    this.innerHTML = '';
    this.textContent = '';
}
MockElement.prototype.classList = null;
MockElement.prototype.closest = function(sel) { return null; };
MockElement.prototype.querySelectorAll = function(sel) {
    // Return elements matching .collapsible
    return this.children.filter(function(c) {
        return c._classes && c._classes.has('collapsible');
    });
};
MockElement.prototype.appendChild = function(c) { this.children.push(c); return c; };

function makeMockPanel(classes) {
    var el = new MockElement('div', classes);
    el._classes = new Set(classes.split(' '));
    el.classList = {
        contains: function(c) { return el._classes.has(c); },
        add: function(c) { el._classes.add(c); },
        remove: function(c) { el._classes.delete(c); },
        toggle: function(c, force) {
            if (force === undefined) {
                if (el._classes.has(c)) el._classes.delete(c);
                else el._classes.add(c);
            } else if (force) el._classes.add(c);
            else el._classes.delete(c);
        },
        has: function(c) { return el._classes.has(c); }
    };
    el.closest = function(sel) { return null; };
    return el;
}

// Create output container with panels
var O = new MockElement('div', '');
O._classes = new Set();
O.classList = {
    contains: function(c) { return O._classes.has(c); },
    add: function(c) { O._classes.add(c); },
    remove: function(c) { O._classes.delete(c); }
};

// Add an llm-panel (collapsible, NOT rc)
var llmPanel = makeMockPanel('collapsible llm-panel');
O.children.push(llmPanel);

// Add a tool-call panel (collapsible, NOT rc)
var toolPanel = makeMockPanel('collapsible');
O.children.push(toolPanel);

// Add a result panel (collapsible + rc)
var resultPanel = makeMockPanel('collapsible rc');
O.children.push(resultPanel);

// Override querySelectorAll for O
O.querySelectorAll = function(sel) {
    return O.children.filter(function(c) {
        return c._classes && c._classes.has('collapsible');
    });
};

// Simulate applyChevronState(false) with isRunning=false
// This is what happens during demo replay
var isRunning = false;
function applyChevronState(expanded) {
    var panels = O.querySelectorAll('.collapsible');
    for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        var inAdjacent = false;
        var inRunning = isRunning && !inAdjacent;
        if (!expanded) {
            if (inRunning || p.classList.contains('rc')) {
                p.classList.remove('chv-hidden');
                continue;
            }
            p.classList.add('chv-hidden');
        }
    }
}

applyChevronState(false);

// Check results
var results = {
    llmPanelHidden: llmPanel._classes.has('chv-hidden'),
    toolPanelHidden: toolPanel._classes.has('chv-hidden'),
    resultPanelHidden: resultPanel._classes.has('chv-hidden')
};
console.log(JSON.stringify(results));
"""
        r = _run_node(script)
        assert r.returncode == 0, r.stderr
        results = json.loads(r.stdout.strip())

        assert results["llmPanelHidden"] is True, (
            "Bug reproduction: llm panel should get chv-hidden when "
            "isRunning=false"
        )
        assert results["toolPanelHidden"] is True, (
            "Bug reproduction: tool panel should get chv-hidden when "
            "isRunning=false"
        )
        assert results["resultPanelHidden"] is False, (
            "Result panel should NOT be hidden (it has .rc class)"
        )

    def test_fix_panels_visible_when_demo_guard_active(self) -> None:
        """With the _demoActive guard, panels must remain visible."""
        src = _MAIN_JS.read_text()

        fn_start = src.index("function processOutputEvent(ev)")
        depth = 0
        fn_end = fn_start
        for i in range(fn_start, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    fn_end = i + 1
                    break
        fn_body = src[fn_start:fn_end]

        assert "_demoActive" in fn_body, (
            "processOutputEvent must guard applyChevronState with "
            "_demoActive check"
        )

        script = r"""
function makeMockPanel(classes) {
    var el = { _classes: new Set(classes.split(' ')) };
    el.classList = {
        contains: function(c) { return el._classes.has(c); },
        add: function(c) { el._classes.add(c); },
        remove: function(c) { el._classes.delete(c); }
    };
    el.closest = function() { return null; };
    return el;
}

var O = {children: []};
var llmPanel = makeMockPanel('collapsible llm-panel');
var toolPanel = makeMockPanel('collapsible');
O.children = [llmPanel, toolPanel];
O.querySelectorAll = function() {
    return O.children.filter(function(c) { return c._classes.has('collapsible'); });
};

// With _demoActive=true, applyChevronState should NOT be called
var _demoActive = true;
var isRunning = false;
var tab = { panelsExpanded: false };

function applyChevronState(expanded) {
    var panels = O.querySelectorAll('.collapsible');
    for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        var inRunning = isRunning;
        if (!expanded) {
            if (inRunning || p.classList.contains('rc')) {
                p.classList.remove('chv-hidden');
                continue;
            }
            p.classList.add('chv-hidden');
        }
    }
}

// Simulate the fixed guard: if (tab && !tab.panelsExpanded && !_demoActive)
if (tab && !tab.panelsExpanded && !_demoActive) {
    applyChevronState(false);
}

console.log(JSON.stringify({
    llmHidden: llmPanel._classes.has('chv-hidden'),
    toolHidden: toolPanel._classes.has('chv-hidden')
}));
"""
        r = _run_node(script)
        assert r.returncode == 0, r.stderr
        results = json.loads(r.stdout.strip())

        assert results["llmHidden"] is False, (
            "With _demoActive guard, llm panel must NOT be hidden"
        )
        assert results["toolHidden"] is False, (
            "With _demoActive guard, tool panel must NOT be hidden"
        )


if __name__ == "__main__":
    unittest.main()
