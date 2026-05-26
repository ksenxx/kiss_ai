"""Regression: when a still-running task is loaded into a freshly-
created tab from the history sidebar, the buttons and labels below
the input textbox MUST be in the same state they are during a fresh
run of the same task — specifically the stop button must be visible
and the "Running …" timer label must appear immediately (not after a
1-second delay).

Bug background
--------------

The frontend's ``case 'status':`` handler calls ``setRunningState(true)``
which shows ``#stop-btn`` and starts a 1-second ``setInterval`` timer.
The interval callback is what writes ``"Running 0s"`` / ``"Running
1m 5s"`` into ``#status-text``.  Because ``setInterval`` does not fire
its callback synchronously, the status label remained at ``"Ready"``
(set by ``restoreTab`` on the freshly-created tab) with a red colour
for up to a full second after the running state turned on.

For a freshly-launched task this delay was largely invisible because
the task's own first event (e.g. an LLM step) fired within milliseconds
of the ``status:true`` broadcast and updated the panel.  For a
history-resumed running task the same delay was very visible — the
user clicked the history row, watched the panels replay, then stared
at a stale ``"Ready"`` label for a beat before the timer finally
flipped to ``"Running 0s"``.

Fix
---

``startTimer()`` in ``media/main.js`` now renders the first timer
tick immediately (synchronous textContent write) before arming the
``setInterval``.  Both the immediate write and the interval call the
shared ``_renderTimerTick`` helper so the formatting stays in one
place.

This test asserts the source-level structure of that fix.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_function_body(src: str, name: str) -> str:
    """Return the body (between matching braces) of ``function <name>``."""
    m = re.search(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", src)
    assert m, f"Could not find function {name} in main.js"
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start:i]


class TestStartTimerRendersImmediately:
    """``startTimer`` must paint ``"Running …"`` into ``#status-text``
    synchronously (i.e. before the first ``setInterval`` tick) so the
    label is visible the instant the running state turns on — which
    is what happens when a running task is loaded into a new tab
    from history."""

    def test_start_timer_renders_first_tick_before_interval(self) -> None:
        body = _extract_function_body(_read_main_js(), "startTimer")
        # Both the synchronous first-paint and the setInterval body
        # must update statusText.textContent.  We assert the
        # synchronous one happens BEFORE setInterval is armed.
        interval_idx = body.find("setInterval(")
        assert interval_idx >= 0, (
            "startTimer must arm a setInterval to keep the timer "
            "updating"
        )
        head = body[:interval_idx]
        # The head (everything before setInterval) must contain a
        # call that writes the "Running …" label — either an inline
        # statusText.textContent assignment or a call to a shared
        # tick helper.
        has_inline = bool(
            re.search(
                r"statusText\.textContent\s*=\s*['\"]Running",
                head,
            ),
        )
        has_tick_call = bool(re.search(r"_renderTimerTick\s*\(", head))
        assert has_inline or has_tick_call, (
            "startTimer must render the first timer tick BEFORE "
            "calling setInterval so the 'Running …' label appears "
            "immediately when a running task is loaded into a new "
            "tab from history.  Without this the label stays at "
            "the restoreTab-default ('Ready') for up to one second."
        )

    def test_start_timer_writes_running_label(self) -> None:
        """``startTimer`` (or its tick helper) must produce a
        ``"Running …"`` label string so users see the task is
        live."""
        src = _read_main_js()
        body = _extract_function_body(src, "startTimer")
        # If the tick helper is used, inline its body too.
        if "_renderTimerTick" in body:
            helper = _extract_function_body(src, "_renderTimerTick")
            body = body + "\n" + helper
        assert re.search(
            r"statusText\.textContent\s*=\s*['\"]Running", body,
        ), (
            "startTimer's render path must assign a 'Running' label "
            "to statusText.textContent"
        )

    def test_start_timer_sets_red_color(self) -> None:
        """Red colour is the canonical 'running' indicator on the
        status-text label."""
        body = _extract_function_body(_read_main_js(), "startTimer")
        assert "var(--red)" in body, (
            "startTimer must colour statusText red to signal the "
            "running state visually"
        )


class TestStatusHandlerShowsStopButton:
    """``case 'status':`` (the event-bus handler that fires when a
    history-resumed running task's backend broadcasts ``status
    running=true``) must call ``setRunningState(true)`` on the active
    tab — that's what flips ``#stop-btn`` to ``display:flex`` and
    starts the timer (which now renders synchronously)."""

    def test_status_case_calls_set_running_state(self) -> None:
        src = _read_main_js()
        m = re.search(r"case\s+'status'\s*:\s*\{", src)
        assert m, "Could not find case 'status' in main.js"
        start = m.end()
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        body = src[start:i]
        assert "setRunningState(ev.running)" in body, (
            "case 'status' must call setRunningState(ev.running) so "
            "the stop button below the input textbox flips to "
            "display:flex when a running task is loaded in a new tab"
        )


class TestSetRunningStateShowsStopButton:
    """``setRunningState(true)`` must show the stop button (the only
    "button below the input textbox" whose visibility differs between
    fresh-run and idle states)."""

    def test_set_running_state_toggles_stop_btn(self) -> None:
        body = _extract_function_body(_read_main_js(), "setRunningState")
        assert re.search(
            r"stopBtn\.style\.display\s*=\s*running\s*\?\s*['\"]flex",
            body,
        ), (
            "setRunningState must set stopBtn.style.display to 'flex' "
            "when running so the stop button below the input textbox "
            "is visible (matching the fresh-run state of the same tab)"
        )

    def test_set_running_state_keeps_send_btn_visible(self) -> None:
        body = _extract_function_body(_read_main_js(), "setRunningState")
        assert re.search(
            r"sendBtn\.style\.display\s*=\s*['\"]flex",
            body,
        ), (
            "setRunningState must keep sendBtn.style.display='flex' so "
            "users can still queue follow-up tasks while the agent is "
            "running"
        )

    def test_set_running_state_starts_timer_when_running(self) -> None:
        body = _extract_function_body(_read_main_js(), "setRunningState")
        # The startTimer call must be inside the `if (running)` branch.
        m = re.search(r"if\s*\(\s*running\s*\)\s*\{", body)
        assert m, "setRunningState must branch on `if (running)`"
        branch_start = m.end()
        # Find the matching close-brace of this if-block by tracking depth.
        depth = 1
        i = branch_start
        while i < len(body) and depth > 0:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            i += 1
        running_branch = body[branch_start:i]
        assert "startTimer()" in running_branch, (
            "setRunningState's running branch must call startTimer() so "
            "the 'Running …' label below (now) appears immediately "
            "when the running state is restored from a history resume"
        )
