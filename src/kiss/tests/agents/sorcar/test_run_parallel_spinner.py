# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Spinner-during-run_parallel regression test.

Bug: When the parent agent calls ``run_parallel``, the wait-spinner
(to the left of the send button in the chat webview) disappears for
the entire duration of the parallel fan-out.

Event flow that triggers the bug:

1. Parent agent emits a ``tool_call`` event for ``run_parallel`` →
   ``processOutputEvent`` runs the default branch which calls
   ``showSpinner()`` (a 250 ms timer is set to add the ``active``
   class).
2. The backend broadcasts one ``new_tab`` event per sub-agent.  The
   frontend handler calls ``createNewTab()`` which calls
   ``setRunningState(tab.isRunning=false)`` and an explicit
   ``removeSpinner()`` — this CANCELS the pending 250 ms timer.
3. The handler then calls ``switchToTab(parentTabBeforeNew)`` to
   restore focus to the still-running parent tab.  Before the fix,
   ``switchToTab`` called ``setRunningState(true)`` which did NOT
   re-arm the spinner.  Result: the parent tab is left with no
   spinner timer and the ``#wait-spinner`` element stays inactive
   until run_parallel finishes (potentially minutes later).

Fix: ``setRunningState(true)`` MUST call ``showSpinner()`` so the
spinner is consistently shown whenever the UI flips to running —
including every code path that switches back to a running tab.

This module runs the actual ``setRunningState`` source from
``main.js`` in a Node.js shim that records spinner-related call
sequences.  No mocks of the function under test; the spies are
applied to its collaborators (showSpinner / removeSpinner /
startTimer / stopTimer) so we can observe the order and presence of
calls.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _extract_set_running_state(src: str) -> str:
    """Extract the ``function setRunningState(running) { ... }`` source.

    Searches for the declaration and walks balanced braces.
    """
    marker = "function setRunningState(running)"
    start = src.index(marker)
    open_brace = src.index("{", start)
    depth = 0
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError("Could not find end of setRunningState")


def _node_harness(set_running_state_src: str) -> str:
    """Return a Node script that loads ``setRunningState`` with spies."""
    return (
        r"""
'use strict';

const calls = [];

const sendBtn = { style: { display: 'none' } };
const stopBtn = { style: { display: 'none' } };
const statusText = { textContent: 'Ready' };

let isRunning = false;
let t0 = null;
let endTs = 0;

function updateInputDisabled() { calls.push('updateInputDisabled'); }
function startTimer() { calls.push('startTimer'); }
function stopTimer() { calls.push('stopTimer'); }
function showSpinner() { calls.push('showSpinner'); }
function removeSpinner() { calls.push('removeSpinner'); }
function doneLabelFor(start, end) { return 'Done'; }

"""
        + set_running_state_src
        + r"""

const phase = process.argv[1] || 'true';
if (phase === 'true') {
    setRunningState(true);
} else if (phase === 'false') {
    isRunning = true;
    setRunningState(false);
} else if (phase === 'switch_back') {
    // Simulates the run_parallel sequence on the parent tab:
    //   (1) parent is already running and a tool_call just showed the
    //       spinner (timer pending) -- emulated by an initial
    //       showSpinner() marker.
    //   (2) createNewTab on the sub-agent tab calls
    //       setRunningState(false) (which clears the spinner) +
    //       explicit removeSpinner().
    //   (3) switchToTab(parent) calls setRunningState(true).
    //   Expectation: spinner is re-shown after step (3).
    calls.push('initial_show');
    showSpinner();
    setRunningState(false);
    removeSpinner();
    setRunningState(true);
}

console.log(JSON.stringify(calls));
"""
    )


def test_set_running_state_true_shows_spinner() -> None:
    """``setRunningState(true)`` MUST call ``showSpinner``.

    Before the fix the function only called ``startTimer`` on the
    running branch — leaving any cancelled spinner timer un-armed.
    """
    src = _MAIN_JS.read_text()
    body = _extract_set_running_state(src)
    r = subprocess.run(
        ["node", "-e", _node_harness(body), "true"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"Node failed: {r.stderr}"
    calls = json.loads(r.stdout.strip())
    assert "showSpinner" in calls, (
        f"setRunningState(true) must call showSpinner; got {calls}"
    )
    # And the spinner must be (re)shown AFTER startTimer so it's the
    # most-recent spinner signal on the running branch.
    assert calls.index("showSpinner") > calls.index("startTimer"), (
        f"showSpinner must follow startTimer on the running branch; "
        f"got {calls}"
    )


def test_set_running_state_false_still_removes_spinner() -> None:
    """``setRunningState(false)`` must still remove the spinner.

    The fix only adds ``showSpinner`` to the running branch — it must
    NOT regress the existing ``removeSpinner`` on the not-running
    branch.
    """
    src = _MAIN_JS.read_text()
    body = _extract_set_running_state(src)
    r = subprocess.run(
        ["node", "-e", _node_harness(body), "false"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"Node failed: {r.stderr}"
    calls = json.loads(r.stdout.strip())
    assert "removeSpinner" in calls, (
        f"setRunningState(false) must call removeSpinner; got {calls}"
    )
    assert "showSpinner" not in calls, (
        f"setRunningState(false) must NOT call showSpinner; got {calls}"
    )


def test_run_parallel_switch_back_restores_spinner() -> None:
    """End-to-end: the parent's spinner is restored after the
    ``run_parallel`` createNewTab → switchToTab(parent) sequence.

    Models the exact event ordering in main.js's ``case 'new_tab'``
    handler.  The LAST spinner-related call on the parent tab must be
    ``showSpinner`` (not ``removeSpinner``); otherwise the parent tab
    would be left without a visible wait-spinner while sub-agents run.
    """
    src = _MAIN_JS.read_text()
    body = _extract_set_running_state(src)
    r = subprocess.run(
        ["node", "-e", _node_harness(body), "switch_back"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"Node failed: {r.stderr}"
    calls = json.loads(r.stdout.strip())
    spinner_calls = [c for c in calls if c in ("showSpinner", "removeSpinner")]
    assert spinner_calls, f"Expected spinner activity; got {calls}"
    assert spinner_calls[-1] == "showSpinner", (
        "After the createNewTab + switchToTab(parent) sequence, the "
        "LAST spinner action on the parent tab must be showSpinner; "
        f"got spinner sequence {spinner_calls} (full: {calls})"
    )
