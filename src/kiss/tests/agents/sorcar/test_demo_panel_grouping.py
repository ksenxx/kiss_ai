"""Tests for demo mode panel-by-panel replay with collapse.

Verifies that:
  - Events are grouped into logical panels (llm, tool_call, result)
  - Each panel is loaded in 0.5s and collapsed before moving on
  - The result panel is streamed word-by-word (not collapsed)
  - main.js exposes collapsePanels in the _demoApi bridge
  - groupEventsIntoPanels is exposed on window for testing
"""

import subprocess
import unittest
from pathlib import Path

_DEMO_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "demo.js"
)

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    """Run a JS script in Node.js and return the result."""
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


# Minimal browser shim for running demo.js in Node.js
_NODE_SHIM = r"""
var window = {};
var document = {
    getElementById: function() {
        return {
            tagName: 'div', className: '', textContent: '', innerHTML: '',
            style: {}, children: [],
            appendChild: function(c) { this.children.push(c); return c; },
            querySelectorAll: function() { return []; },
            querySelector: function() { return null; },
        };
    },
    createElement: function(tag) {
        return {
            tagName: tag, className: '', textContent: '', innerHTML: '',
            style: { cssText: '' }, children: [],
            appendChild: function(c) { this.children.push(c); return c; },
            querySelectorAll: function() { return { forEach: function() {} }; },
            querySelector: function() { return null; },
        };
    },
};
var setTimeout = function(fn, ms) { return 1; };
var marked = undefined;
var hljs = undefined;
"""


class TestGroupEventsIntoPanelsStructure(unittest.TestCase):
    """Structural: groupEventsIntoPanels exists and is exposed."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _DEMO_JS.read_text()

    def test_function_exists(self) -> None:
        assert "function groupEventsIntoPanels(events)" in self.src

    def test_exposed_on_window(self) -> None:
        assert "window._groupEventsIntoPanels = groupEventsIntoPanels" in self.src

    def test_skip_types_defined(self) -> None:
        assert "SKIP_TYPES" in self.src
        assert "task_done" in self.src
        assert "followup_suggestion" in self.src


class TestGroupEventsIntoPanelsBehavioral(unittest.TestCase):
    """Behavioral: groupEventsIntoPanels groups events correctly."""

    def _group(self, events_json: str) -> subprocess.CompletedProcess[str]:
        script = (
            _NODE_SHIM
            + "\n"
            + _DEMO_JS.read_text()
            + "\n"
            + f"var events = {events_json};\n"
            + "var groups = window._groupEventsIntoPanels(events);\n"
            + "console.log(JSON.stringify(groups));\n"
        )
        return _run_node(script)

    def test_single_llm_panel(self) -> None:
        """thinking_start + thinking_delta + text_delta = one group."""
        events = (
            '[{"type":"thinking_start"},'
            '{"type":"thinking_delta","text":"hi"},'
            '{"type":"thinking_end"},'
            '{"type":"text_delta","text":"hello"},'
            '{"type":"text_end"}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        assert len(groups) == 1
        assert groups[0][0]["type"] == "thinking_start"
        assert len(groups[0]) == 5

    def test_llm_then_tool_then_llm(self) -> None:
        """LLM panel → tool_call panel → LLM panel = 3 groups."""
        events = (
            '[{"type":"thinking_start"},{"type":"text_delta","text":"a"},{"type":"text_end"},'
            '{"type":"tool_call","name":"Bash"},{"type":"system_output","text":"out"},'
            '{"type":"tool_result","content":"ok"},'
            '{"type":"thinking_start"},{"type":"text_delta","text":"b"}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        assert len(groups) == 3
        # Group 1: llm panel
        assert groups[0][0]["type"] == "thinking_start"
        # Group 2: tool_call panel
        assert groups[1][0]["type"] == "tool_call"
        assert groups[1][-1]["type"] == "tool_result"
        # Group 3: second llm panel
        assert groups[2][0]["type"] == "thinking_start"

    def test_result_is_own_group(self) -> None:
        """result event always gets its own single-element group."""
        events = (
            '[{"type":"thinking_start"},{"type":"text_delta","text":"x"},'
            '{"type":"result","summary":"done","total_tokens":100}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        assert len(groups) == 2
        assert len(groups[1]) == 1
        assert groups[1][0]["type"] == "result"

    def test_skips_lifecycle_events(self) -> None:
        """task_done, task_error, task_stopped, followup_suggestion are skipped."""
        events = (
            '[{"type":"thinking_start"},{"type":"task_done"},'
            '{"type":"task_error"},{"type":"task_stopped"},'
            '{"type":"followup_suggestion"},{"type":"text_delta","text":"x"}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        # All lifecycle events skipped, only thinking_start + text_delta in one group
        assert len(groups) == 1
        types = [e["type"] for e in groups[0]]
        assert "task_done" not in types
        assert "task_error" not in types

    def test_multiple_tool_calls(self) -> None:
        """Each tool_call starts a new group."""
        events = (
            '[{"type":"thinking_start"},{"type":"text_end"},'
            '{"type":"tool_call","name":"Read"},{"type":"tool_result","content":"a"},'
            '{"type":"tool_call","name":"Write"},{"type":"tool_result","content":"b"},'
            '{"type":"result","summary":"done"}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        # LLM panel, tool_call 1, tool_call 2, result
        assert len(groups) == 4
        assert groups[1][0]["name"] == "Read"
        assert groups[2][0]["name"] == "Write"

    def test_usage_info_stays_in_group(self) -> None:
        """usage_info events stay in whatever group is current."""
        events = (
            '[{"type":"thinking_start"},{"type":"usage_info","total_tokens":50},'
            '{"type":"text_delta","text":"x"}]'
        )
        r = self._group(events)
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        assert len(groups) == 1
        types = [e["type"] for e in groups[0]]
        assert "usage_info" in types

    def test_empty_events(self) -> None:
        """Empty event list → empty groups."""
        r = self._group("[]")
        assert r.returncode == 0, r.stderr
        import json

        groups = json.loads(r.stdout.strip())
        assert groups == []


class TestDemoReplayUsesPanel(unittest.TestCase):
    """Structural: _startDemoReplay uses groupEventsIntoPanels and 500ms delay."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _DEMO_JS.read_text()

    def test_calls_groupEventsIntoPanels(self) -> None:
        assert "groupEventsIntoPanels(events)" in self.src

    def test_uses_500ms_delay(self) -> None:
        assert "await sleep(500)" in self.src

    def test_calls_collapsePanels(self) -> None:
        assert "api.collapsePanels()" in self.src

    def test_no_1s_per_event_delay(self) -> None:
        """The old 1-second per-event delay should be gone."""
        # The old pattern was: api.processEvent(ev); ... await sleep(1000);
        # in the event loop. Now there should be no sleep(1000) inside the
        # panel group processing loop.
        replay_fn_start = self.src.index("window._startDemoReplay")
        replay_fn = self.src[replay_fn_start:]
        # The only sleep(1000) should be the pause between tasks
        occurrences = replay_fn.count("await sleep(1000)")
        assert occurrences == 1, (
            f"Expected 1 sleep(1000) (inter-task pause), found {occurrences}"
        )

    def test_result_streamed_not_collapsed(self) -> None:
        """Result panels go through streamResultEvent, not collapsePanels."""
        replay_fn_start = self.src.index("window._startDemoReplay")
        replay_fn = self.src[replay_fn_start:]
        assert "streamResultEvent(api, group[0])" in replay_fn


class TestMainJsCollapsePanelsBridge(unittest.TestCase):
    """Structural: main.js exposes collapsePanels in _demoApi."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _MAIN_JS.read_text()

    def test_collapsePanels_in_demoApi(self) -> None:
        assert "collapsePanels:" in self.src
        assert "collapseAllExceptResult(O)" in self.src

    def test_collapsePanels_function_reference(self) -> None:
        idx = self.src.index("window._demoApi = {")
        bridge = self.src[idx : idx + 1000]
        assert "collapsePanels" in bridge


if __name__ == "__main__":
    unittest.main()
