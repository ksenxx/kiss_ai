# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for demo mode panel-by-panel replay with collapse.

Verifies (behaviorally, by running the real demo.js in Node.js) that
events are grouped into logical panels (llm, tool_call, result).
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

def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    """Run a JS script in Node.js and return the result."""
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


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
        assert groups[0][0]["type"] == "thinking_start"
        assert groups[1][0]["type"] == "tool_call"
        assert groups[1][-1]["type"] == "tool_result"
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


if __name__ == "__main__":
    unittest.main()
