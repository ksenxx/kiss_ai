"""Integration tests for demo mode stop button and spinner.

Bug: During demo replay, setRunningState(true) is never called, so the
stop button and spinner never appear — the send button stays visible.
Also, clicking the stop button during a demo sends a backend 'stop'
message instead of cancelling the replay.

Fix requirements:
  1. Demo replay must call setRunningState(true) and showSpinner() at start.
  2. Demo replay must call setRunningState(false) and removeSpinner() at end.
  3. _cancelDemoReplay must also restore UI (setRunningState(false) +
     removeSpinner()) for immediate feedback when stopped mid-replay.
  4. The stop button click handler must detect _demoActive and cancel the
     demo instead of posting a 'stop' message to the backend.
  5. _demoApi in main.js must expose setRunningState, showSpinner, and
     removeSpinner.
"""

import json
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


def _extract_fn(src: str, name: str) -> str:
    """Extract a named function/assignment body from JS source."""
    start = src.index(name)
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError(f"Could not find end of {name}")


class TestDemoStopButtonSpinnerStructural(unittest.TestCase):
    """Structural tests: verify the source code contains the required
    calls and wiring."""

    demo_src: str
    main_src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo_src = _DEMO_JS.read_text()
        cls.main_src = _MAIN_JS.read_text()

    def test_replay_calls_set_running_state_true_at_start(self) -> None:
        """_startDemoReplay must call api.setRunningState(true) before
        the main loop."""
        fn = _extract_fn(self.demo_src, "window._startDemoReplay")
        # setRunningState(true) must appear before the for loop
        rs_idx = fn.index("setRunningState(true)")
        loop_idx = fn.index("for (let i =")
        assert rs_idx < loop_idx, (
            "setRunningState(true) must be called before the replay loop"
        )

    def test_replay_calls_show_spinner_at_start(self) -> None:
        """_startDemoReplay must call api.showSpinner() before the loop."""
        fn = _extract_fn(self.demo_src, "window._startDemoReplay")
        sp_idx = fn.index("showSpinner()")
        loop_idx = fn.index("for (let i =")
        assert sp_idx < loop_idx, (
            "showSpinner() must be called before the replay loop"
        )

    def test_replay_calls_set_running_state_false_at_end(self) -> None:
        """_startDemoReplay must call setRunningState(false) after the loop."""
        fn = _extract_fn(self.demo_src, "window._startDemoReplay")
        # Find the last occurrence of setRunningState(false)
        last_rs = fn.rfind("setRunningState(false)")
        loop_end = fn.rfind("}")
        assert last_rs != -1, (
            "setRunningState(false) must be called after the replay loop"
        )

    def test_replay_calls_remove_spinner_at_end(self) -> None:
        """_startDemoReplay must call removeSpinner() after the loop."""
        fn = _extract_fn(self.demo_src, "window._startDemoReplay")
        assert "removeSpinner()" in fn, (
            "removeSpinner() must be called at the end of the replay"
        )

    def test_cancel_replay_restores_ui(self) -> None:
        """_cancelDemoReplay must call setRunningState(false) and
        removeSpinner() for immediate UI feedback."""
        fn = _extract_fn(self.demo_src, "window._cancelDemoReplay")
        assert "setRunningState(false)" in fn, (
            "_cancelDemoReplay must call setRunningState(false)"
        )
        assert "removeSpinner()" in fn, (
            "_cancelDemoReplay must call removeSpinner()"
        )

    def test_demo_api_exposes_running_state_and_spinner(self) -> None:
        """main.js _demoApi must expose setRunningState, showSpinner,
        and removeSpinner."""
        idx = self.main_src.index("window._demoApi = {")
        bridge = self.main_src[idx : idx + 2000]
        assert "setRunningState" in bridge, (
            "_demoApi must expose setRunningState"
        )
        assert "showSpinner" in bridge, "_demoApi must expose showSpinner"
        assert "removeSpinner" in bridge, "_demoApi must expose removeSpinner"

    def test_stop_button_handles_demo_mode(self) -> None:
        """The stop button click handler must check _demoActive and
        cancel the demo instead of sending a stop message to the backend."""
        # Find the stopBtn click handler
        stop_handler_start = self.main_src.index(
            "stopBtn.addEventListener('click'"
        )
        # Extract the handler body
        depth = 0
        handler = ""
        for i in range(stop_handler_start, len(self.main_src)):
            if self.main_src[i] == "{":
                depth += 1
            elif self.main_src[i] == "}":
                depth -= 1
                if depth == 0:
                    handler = self.main_src[stop_handler_start : i + 1]
                    break
        assert "_demoActive" in handler, (
            "Stop button handler must check _demoActive"
        )
        assert "_cancelDemoReplay" in handler, (
            "Stop button handler must call _cancelDemoReplay when "
            "demo is active"
        )


class TestDemoStopButtonSpinnerBehavioral(unittest.TestCase):
    """Behavioral: run demo.js in Node.js and verify the API call
    sequence includes running state and spinner calls."""

    def test_replay_shows_stop_button_and_spinner(self) -> None:
        """Track API calls and verify setRunningState(true) + showSpinner
        at start and setRunningState(false) + removeSpinner at end."""
        demo_src = _DEMO_JS.read_text()

        # Patch sleep to be instant and log calls
        patched = demo_src.replace(
            "function sleep(ms) {\n"
            "    return new Promise(resolve => {\n"
            "      setTimeout(resolve, ms);\n"
            "    });\n"
            "  }",
            "function sleep(ms) {\n"
            "    return new Promise(resolve => {\n"
            "      if (typeof _demoCalls !== 'undefined') "
            "_demoCalls.push('sleep:' + ms);\n"
            "      resolve();\n"
            "    });\n"
            "  }",
        )

        script = (
            _NODE_SHIM
            + "\n"
            + r"""
var _demoCalls = [];
var _eventsToReturn = [
    [{type:'thinking_start'},{type:'text_delta',text:'x'},
     {type:'result',summary:'done',total_tokens:10,cost:'$0.01'}],
];
var _taskIdx = 0;

window._demoApi = {
    active: false,
    resolveEvents: null,
    createNewTab: function() { _demoCalls.push('createNewTab'); },
    setInput: function(t) { _demoCalls.push('setInput:' + t); },
    clearInput: function() { _demoCalls.push('clearInput'); },
    clearForReplay: function() { _demoCalls.push('clearForReplay'); },
    resetOutputState: function() { _demoCalls.push('resetOutputState'); },
    setRunningState: function(v) { _demoCalls.push('setRunningState:' + v); },
    showSpinner: function() { _demoCalls.push('showSpinner'); },
    removeSpinner: function() { _demoCalls.push('removeSpinner'); },
    setTaskText: function(t) { _demoCalls.push('setTaskText:' + t); },
    updateTabTitle: function(t) { _demoCalls.push('updateTabTitle:' + t); },
    hideWelcome: function() { _demoCalls.push('hideWelcome'); },
    scrollToBottom: function() {},
    getActiveTabId: function() { return 'tab1'; },
    sendMessage: function(msg) {
        var self = window._demoApi;
        if (self.resolveEvents) {
            var events = _eventsToReturn[_taskIdx++] || [];
            var resolve = self.resolveEvents;
            self.resolveEvents = null;
            resolve(events);
        }
    },
    collapsePanels: function() { _demoCalls.push('collapsePanels'); },
    processEvent: function(ev) { _demoCalls.push('processEvent:' + ev.type); },
};
"""
            + "\n"
            + patched
            + r"""

var sessions = [
    {id: 's1', has_events: true, preview: 'Task A'},
];

window._startDemoReplay(sessions).then(function() {
    console.log(JSON.stringify(_demoCalls));
}).catch(function(e) {
    console.error(e.stack || e.message);
    process.exit(1);
});
"""
        )
        r = _run_node(script)
        assert r.returncode == 0, f"Node failed: {r.stderr}"
        calls = json.loads(r.stdout.strip())

        # setRunningState:true must be the first meaningful call
        assert "setRunningState:true" in calls, (
            f"setRunningState(true) was not called: {calls}"
        )
        rs_true_idx = calls.index("setRunningState:true")
        # It must come before hideWelcome (first per-task call)
        hw_idx = calls.index("hideWelcome")
        assert rs_true_idx < hw_idx, (
            f"setRunningState(true) (idx {rs_true_idx}) must come "
            f"before hideWelcome (idx {hw_idx}): {calls}"
        )

        # showSpinner must be called at the start
        assert "showSpinner" in calls, (
            f"showSpinner was not called: {calls}"
        )
        sp_idx = calls.index("showSpinner")
        assert sp_idx < hw_idx, (
            f"showSpinner (idx {sp_idx}) must come before "
            f"hideWelcome (idx {hw_idx}): {calls}"
        )

        # setRunningState:false must be called at the end
        assert "setRunningState:false" in calls, (
            f"setRunningState(false) was not called: {calls}"
        )
        rs_false_idx = calls.index("setRunningState:false")
        # It must come after the last task event
        last_process = max(
            i for i, c in enumerate(calls) if c.startswith("processEvent:")
            or c.startswith("sleep:")
        )
        assert rs_false_idx > last_process, (
            f"setRunningState(false) (idx {rs_false_idx}) must come "
            f"after last event (idx {last_process}): {calls}"
        )

        # removeSpinner must be called at the end
        assert "removeSpinner" in calls, (
            f"removeSpinner was not called: {calls}"
        )
        rm_sp_idx = calls.index("removeSpinner")
        assert rm_sp_idx > last_process, (
            f"removeSpinner (idx {rm_sp_idx}) must come after "
            f"last event (idx {last_process}): {calls}"
        )

    def test_cancel_replay_restores_ui_immediately(self) -> None:
        """When _cancelDemoReplay is called, setRunningState(false) and
        removeSpinner must be called immediately (not deferred to the
        loop exit)."""
        demo_src = _DEMO_JS.read_text()

        script = (
            _NODE_SHIM
            + "\n"
            + r"""
var _demoCalls = [];
window._demoApi = {
    active: false,
    resolveEvents: null,
    setRunningState: function(v) { _demoCalls.push('setRunningState:' + v); },
    showSpinner: function() { _demoCalls.push('showSpinner'); },
    removeSpinner: function() { _demoCalls.push('removeSpinner'); },
};
"""
            + "\n"
            + demo_src
            + r"""

// Simulate active state
window._demoApi.active = true;

// Cancel
window._cancelDemoReplay();

console.log(JSON.stringify(_demoCalls));
"""
        )
        r = _run_node(script)
        assert r.returncode == 0, f"Node failed: {r.stderr}"
        calls = json.loads(r.stdout.strip())

        assert "setRunningState:false" in calls, (
            f"_cancelDemoReplay must call setRunningState(false): {calls}"
        )
        assert "removeSpinner" in calls, (
            f"_cancelDemoReplay must call removeSpinner(): {calls}"
        )


if __name__ == "__main__":
    unittest.main()
