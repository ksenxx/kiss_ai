# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: ``task_events`` for a sub-agent tab must NOT fall
through to the parent (active) tab's replay path in the frontend.

Reproduces the bug where ``_open_persisted_subagent_tabs`` broadcasts
``task_events`` for each sub-agent tab, and if the targeted sub-agent
tab has not yet been created in the frontend (race between
``openSubagentTab`` and ``task_events``), the ``task_events`` handler's
guard ``if (teTabId !== activeTabId && teTab)`` evaluates to false
(because ``teTab`` is ``null``), causing the sub-agent's events to
fall through to the "Active tab" replay branch — which renders them
in the parent's output area, after the parent's result and followup
suggestion.

The fix adds an early ``break`` when ``teTabId !== activeTabId`` and
``teTab`` is null, so the sub-agent events are silently dropped
instead of leaking into the parent tab.

This structural test verifies the fix is present in the frontend code.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_JS = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / "main.js"


def _read_main_js() -> str:
    return _MAIN_JS.read_text(encoding="utf-8")


class TestSubagentTaskEventsNoParentFallthrough:
    """The task_events handler must not fall through to the active tab
    when the targeted tab doesn't exist.
    """

    def test_task_events_handler_guards_missing_tab(self) -> None:
        """When teTabId !== activeTabId and teTab is null, the handler
        must break instead of falling through to the active tab branch.

        The guard pattern is:
            if (teTabId !== activeTabId) {
                if (!teTab) break;
                ...
            }

        This ensures sub-agent task_events never render in the parent tab.
        """
        src = _read_main_js()

        # Find the task_events case body
        # Look for `case 'task_events':` and extract the body up to the next top-level case
        match = re.search(
            r"case\s+'task_events':\s*\{(.*?)(?=\n\s*case\s+')",
            src,
            re.DOTALL,
        )
        assert match, "Could not find case 'task_events' in main.js"
        body = match.group(1)

        # Verify the guard: when teTabId !== activeTabId, there must be
        # an early break for missing teTab BEFORE any replay logic.
        # The fix pattern is: if (teTabId !== activeTabId) { if (!teTab) break; ... }
        guard_pattern = re.search(
            r"if\s*\(\s*teTabId\s*!==\s*activeTabId\s*\)\s*\{[^}]*"
            r"if\s*\(\s*!teTab\s*\)\s*break",
            body,
            re.DOTALL,
        )
        assert guard_pattern, (
            "The task_events handler must guard against missing teTab "
            "when teTabId !== activeTabId. Expected pattern: "
            "if (teTabId !== activeTabId) { if (!teTab) break; ... }\n"
            "This prevents sub-agent events from falling through to the "
            "parent (active) tab's replay path."
        )

    def test_active_tab_replay_path_still_works(self) -> None:
        """The active-tab replay path (replayTaskEvents) must still exist
        for events targeted at the active tab.
        """
        src = _read_main_js()
        match = re.search(
            r"case\s+'task_events':\s*\{(.*?)(?=\n\s*case\s+')",
            src,
            re.DOTALL,
        )
        assert match, "Could not find case 'task_events' in main.js"
        body = match.group(1)

        # The active tab path should still call replayTaskEvents
        assert "replayTaskEvents" in body, (
            "The task_events handler must still call replayTaskEvents "
            "for events targeted at the active tab."
        )
