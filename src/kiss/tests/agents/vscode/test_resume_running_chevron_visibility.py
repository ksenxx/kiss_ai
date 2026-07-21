# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: resuming a running task from history must show live
streaming events in the chat webview even when the fixed-panel chevron
is in the default *collapsed* state.

Bug: ``_replay_session`` (server.py) broadcasts the ``task_events``
event before the ``status`` event with ``running: true``.  On the
frontend, ``replayTaskEvents`` (main.js) calls
``applyChevronState(false, currentTaskName)`` at the end of replay.
Because the ``status`` event hasn't arrived yet, the JS module-global
``isRunning`` is still ``false``, so ``applyChevronState``'s
``inRunning`` branch does not fire and every replayed ``.collapsible``
panel of the resumed task gets ``chv-hidden`` (display:none).  When
the live agent then emits more events, they too are hidden until some
later code path happens to re-run ``applyChevronState`` with
``isRunning=true``.

Fix: the ``status`` handler in main.js must, after
``setRunningState(true)`` flips ``isRunning`` to true, re-apply the
chevron state for ``currentTaskName`` so that
``applyChevronState``'s ``inRunning`` branch removes ``chv-hidden``
from every running-task panel.

This test asserts the source-level structure of that fix.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_status_case_body(src: str) -> str:
    """Return the body of the ``case 'status':`` block in main.js's
    handleEvent switch."""
    m = re.search(r"case\s+'status'\s*:\s*\{", src)
    assert m, "Could not find case 'status' block in main.js"
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


class TestStatusHandlerRefreshesChevron:
    """The ``case 'status':`` block must re-apply chevron state after
    flipping ``isRunning`` to true so panels of a resumed running task
    become visible immediately."""

    def test_status_handler_calls_apply_chevron_state(self) -> None:
        body = _extract_status_case_body(_read_main_js())
        assert "applyChevronState" in body, (
            "case 'status' must call applyChevronState() after toggling "
            "isRunning so the running task's panels become visible "
            "after resuming a running chat from history"
        )

    def test_status_handler_apply_chevron_is_guarded_by_running(
        self,
    ) -> None:
        """The refresh call must only fire when ``ev.running`` is true.
        Calling applyChevronState on running=false would re-hide every
        panel of a freshly-finished task and break the existing
        post-completion view."""
        body = _extract_status_case_body(_read_main_js())
        m = re.search(r"if\s*\(\s*ev\.running\s*\)", body)
        assert m, (
            "case 'status' must guard its applyChevronState refresh "
            "with `if (ev.running)` so the call only fires on the "
            "running=true transition"
        )
        guarded = body[m.end():]
        # The next applyChevronState mention must be inside this guard.
        next_call = guarded.find("applyChevronState")
        assert next_call >= 0, (
            "case 'status' must call applyChevronState inside the "
            "if (ev.running) guard"
        )

    def test_status_handler_uses_current_task_name(self) -> None:
        """The refresh must scope to ``currentTaskName`` (per-task
        chevron map)."""
        body = _extract_status_case_body(_read_main_js())
        # Slice from the applyChevronState CALL (not a comment mention)
        # to the end of the case block so the assertions only inspect
        # the refresh call's surrounding context.
        acs = body.rfind("applyChevronState(")
        assert acs >= 0, "Could not find applyChevronState call in status case"
        refresh_region = body[acs : acs + 400]
        assert "currentTaskName" in refresh_region, (
            "applyChevronState refresh in case 'status' must reference "
            "currentTaskName so it scopes to the resumed task's panels"
        )

    def test_status_handler_refresh_is_after_set_running_state(
        self,
    ) -> None:
        """``isRunning`` is set inside ``setRunningState``; the
        applyChevronState refresh must run AFTER setRunningState so the
        ``inRunning`` branch sees ``isRunning=true``."""
        body = _extract_status_case_body(_read_main_js())
        srs = body.find("setRunningState(")
        acs = body.rfind("applyChevronState(")
        assert srs >= 0 and acs >= 0
        assert srs < acs, (
            "setRunningState must run before applyChevronState in the "
            "status case so isRunning is true when applyChevronState "
            "evaluates its `inRunning` branch"
        )


class TestApplyChevronStateInRunningBranchUnhidesPanels:
    """Sanity check: applyChevronState's collapsed-chevron branch keeps
    running-task panels visible (the precondition that makes the fix
    in ``case 'status'`` work)."""

    def test_apply_chevron_state_unhides_running_panels(self) -> None:
        src = _read_main_js()
        m = re.search(
            r"function\s+applyChevronState\s*\([^)]*\)\s*\{",
            src,
        )
        assert m, "Could not find applyChevronState"
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
        assert "inRunning" in body, (
            "applyChevronState must compute an `inRunning` flag so "
            "running-task panels are kept visible even when the "
            "collapse pass runs"
        )
        # The collapse pass must remove chv-hidden when inRunning.
        assert "remove('chv-hidden')" in body, (
            "applyChevronState must remove chv-hidden from "
            "running-task panels (its `inRunning` arm)"
        )
        in_running_idx = body.find("inRunning || p.classList.contains('rc')")
        assert in_running_idx >= 0, (
            "applyChevronState must check inRunning (or .rc) before "
            "adding chv-hidden"
        )
