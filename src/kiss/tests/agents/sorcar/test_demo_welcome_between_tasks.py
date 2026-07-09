# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test for demo mode welcome page bug between tasks.

Bug: When the demo replay switches from one task to another, the welcome
page is shown during the transition because:
  1. createNewTab() is called which sets welcomeVisible=true and restoreTab
     shows the welcome page.
  2. hideWelcome() is only called AFTER the 2-second input display pause.

The fix: demo replay must be a continuation of the same chat:
  - No createNewTab() calls for subsequent tasks.
  - The task header must be set (setTaskText) before any replay pause —
    main.js hides the welcome page when the task text is set, so
    demo.js no longer calls hideWelcome (the old input-typing +
    2-second display pause design was removed).
  - Output must NOT be cleared for subsequent tasks (only reset state).
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


class TestDemoWelcomeBetweenTasksStructural(unittest.TestCase):
    """Structural: the replay loop must not create new tabs and must
    render the task header before pausing for the event fetch."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _DEMO_JS.read_text()

    def _extract_fn(self, name: str) -> str:
        """Extract a named function/assignment body from demo.js."""
        start = self.src.index(name)
        depth = 0
        for i in range(start, len(self.src)):
            if self.src[i] == "{":
                depth += 1
            elif self.src[i] == "}":
                depth -= 1
                if depth == 0:
                    return self.src[start : i + 1]
        raise AssertionError(f"Could not find end of {name}")

    def _get_replay_fn(self) -> str:
        """Extract the demo replay implementation: the
        ``_startDemoReplay`` entry point plus the ``replaySessions``
        helper it awaits (the per-session loop was extracted there so
        the entry point can guard it with try/catch)."""
        return self._extract_fn(
            "window._startDemoReplay"
        ) + self._extract_fn("async function replaySessions")

    def test_no_create_new_tab_in_replay_loop(self) -> None:
        """The replay loop must NOT call createNewTab — tasks continue
        in the same chat."""
        fn = self._get_replay_fn()
        assert "createNewTab" not in fn, (
            "Demo replay must not call createNewTab; subsequent tasks "
            "should be a continuation of the same chat"
        )

    def test_task_text_set_before_event_fetch(self) -> None:
        """The task header must be set BEFORE the event fetch of each
        task so the welcome page is never visible during transitions.

        demo.js no longer types the task into the input box (the old
        hideWelcome + 2s display pause design): welcome hiding is
        delegated to main.js, which hides it when the task text is set
        (setTaskText / task_events path).  The replay loop therefore
        has to render the task header before any await."""
        fn = self._get_replay_fn()
        text_idx = fn.index("setTaskText(")
        fetch_idx = fn.index("await requestEvents(")
        assert text_idx < fetch_idx, (
            "setTaskText() must appear before await requestEvents() in "
            "the replay loop so the task header replaces the welcome "
            "page before the replay pauses for the event fetch"
        )

    def test_output_not_cleared_for_continuation(self) -> None:
        """For tasks after the first, the output DOM must NOT be cleared
        (clearForReplay resets DOM). Only resetOutputState should be used."""
        fn = self._get_replay_fn()
        assert "resetOutputState" in fn, (
            "Replay must call resetOutputState for subsequent tasks "
            "so that new panels are created without clearing existing output"
        )


class TestDemoWelcomeBetweenTasksBehavioral(unittest.TestCase):
    """Behavioral: simulate a multi-task demo replay and verify that:
    - createNewTab is never called
    - the task header is set for every task before its events render
    - clearForReplay is only called for the first task
    - resetOutputState is called for subsequent tasks

    Uses a patched copy of demo.js where the IIFE's internal sleep()
    is replaced with an instant-resolve version that logs calls.
    """

    def test_replay_call_sequence(self) -> None:
        """Track the exact sequence of API calls during a 2-task replay."""
        demo_src = _DEMO_JS.read_text()

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
// Global call tracker
var _demoCalls = [];
var _eventsToReturn = [
    [{type:'thinking_start'},{type:'text_delta',text:'x'},
     {type:'result',summary:'done',total_tokens:10,cost:'$0.01'}],
    [{type:'thinking_start'},{type:'text_delta',text:'y'},
     {type:'result',summary:'done2',total_tokens:20,cost:'$0.02'}]
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
    setRunningState: function(v) { _demoCalls.push('setRunningState:' + v); },
    showSpinner: function() { _demoCalls.push('showSpinner'); },
    removeSpinner: function() { _demoCalls.push('removeSpinner'); },
};
"""
            + "\n"
            + patched
            + r"""

var sessions = [
    {id: 's1', has_events: true, preview: 'Task A'},
    {id: 's2', has_events: true, preview: 'Task B'}
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
        import json

        calls = json.loads(r.stdout.strip())

        assert "createNewTab" not in calls, (
            f"createNewTab was called during replay: {calls}"
        )

        # The task header must be rendered for EVERY replayed task
        # before that task's events are processed — the header is what
        # replaces the welcome page (main.js hides it on setTaskText /
        # task_events; demo.js no longer calls hideWelcome).
        text_indices = [
            i for i, c in enumerate(calls) if c.startswith("setTaskText:")
        ]
        assert len(text_indices) >= 2, (
            f"setTaskText should be called for each task, "
            f"found {len(text_indices)}: {calls}"
        )
        first_process = next(
            i for i, c in enumerate(calls) if c.startswith("processEvent:")
        )
        assert text_indices[0] < first_process, (
            f"setTaskText (idx {text_indices[0]}) must come before the "
            f"first processEvent (idx {first_process}): {calls}"
        )

        clear_count = calls.count("clearForReplay")
        assert clear_count == 1, (
            f"clearForReplay should be called once (first task only), "
            f"but was called {clear_count} times: {calls}"
        )

        reset_count = calls.count("resetOutputState")
        assert reset_count >= 1, (
            f"resetOutputState should be called for tasks after the first, "
            f"but was called {reset_count} times: {calls}"
        )


class TestMainJsResetOutputStateInDemoApi(unittest.TestCase):
    """Structural: main.js must expose resetOutputState in _demoApi."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _MAIN_JS.read_text()

    def test_reset_output_state_in_demo_api(self) -> None:
        idx = self.src.index("window._demoApi = {")
        bridge = self.src[idx : idx + 1500]
        assert "resetOutputState" in bridge, (
            "main.js _demoApi must expose resetOutputState so demo.js "
            "can reset panel state without clearing the output DOM"
        )


if __name__ == "__main__":
    unittest.main()
