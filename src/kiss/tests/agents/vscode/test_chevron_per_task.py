"""Integration tests: chevron expand/collapse must be per-task, not global.

Bug: clicking the chevron in the fixed panel collapses/expands panels
for ALL tasks in the chat session (current + adjacent tasks loaded via
scrolling).  The chevron state must be specific to each task so that
toggling one task's panels does not affect another task's panels.

These tests verify the JavaScript source structure in ``main.js``:

1. ``panelsExpanded`` must be stored as a per-task map (not a single
   boolean) so that each task in a chat session has its own state.

2. ``applyChevronState`` must scope its DOM mutations to panels
   belonging to the specified task, not all ``.collapsible`` panels.

3. The chevron click handler must identify the currently visible task
   and toggle only that task's state in the map.

4. ``updateVisibleTask`` must sync the chevron icon to match the
   visible task's expanded state from the map.
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


class TestChevronStateIsPerTask:
    """The chevron expanded/collapsed state must be per-task, not a single
    boolean per tab."""

    def test_tab_uses_panels_expanded_map_not_boolean(self) -> None:
        """makeTab() must initialise a panelsExpandedMap (object/dict),
        not a scalar panelsExpanded boolean."""
        src = _read_main_js()
        # Must have panelsExpandedMap initialised as an object literal
        assert re.search(r"panelsExpandedMap\s*:\s*\{", src), (
            "makeTab() must use panelsExpandedMap: {} instead of "
            "panelsExpanded: false"
        )
        # Must NOT have the old scalar panelsExpanded property
        # (check in makeTab context — between 'function makeTab' and the next 'function')
        make_tab_match = re.search(
            r"function\s+makeTab\b.*?return\s*\{(.*?)\};",
            src,
            re.DOTALL,
        )
        assert make_tab_match, "Could not find makeTab function"
        make_tab_body = make_tab_match.group(1)
        assert "panelsExpanded:" not in make_tab_body or "panelsExpandedMap:" in make_tab_body, (
            "makeTab must use panelsExpandedMap, not panelsExpanded"
        )

    def test_apply_chevron_state_accepts_task_name(self) -> None:
        """applyChevronState must accept a taskName parameter so it can
        scope its mutations to a single task's panels."""
        src = _read_main_js()
        # The function signature must include a taskName parameter
        m = re.search(r"function\s+applyChevronState\s*\(([^)]*)\)", src)
        assert m, "Could not find applyChevronState function"
        params = m.group(1)
        assert "taskName" in params, (
            "applyChevronState must accept a taskName parameter to scope "
            "panel mutations to a single task"
        )

    def test_apply_chevron_state_scopes_to_task(self) -> None:
        """applyChevronState must filter panels by task membership,
        not blindly iterate all .collapsible panels."""
        src = _read_main_js()
        # Extract the function body
        m = re.search(
            r"function\s+applyChevronState\s*\([^)]*\)\s*\{",
            src,
        )
        assert m, "Could not find applyChevronState"
        start = m.end()
        # Find matching closing brace
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        body = src[start:i]
        # Must check task membership for each panel
        assert "taskName" in body, (
            "applyChevronState body must use taskName to filter panels "
            "by task membership"
        )

    def test_chevron_click_uses_visible_task(self) -> None:
        """The chevron click handler must determine the currently visible
        task and toggle only that task's state in panelsExpandedMap."""
        src = _read_main_js()
        # Find the chevron click handler — search for the addEventListener
        # call specifically (not the getElementById declaration)
        m = re.search(
            r"taskPanelChevron\)\s*\{[^}]*addEventListener\s*\(\s*'click'",
            src,
            re.DOTALL,
        )
        assert m, "Could not find taskPanelChevron click handler"
        # Extract handler body (from the addEventListener match onward)
        handler_start = m.end()
        handler_region = src[handler_start : handler_start + 1000]
        # Must reference panelsExpandedMap and a task identifier
        assert "panelsExpandedMap" in handler_region, (
            "Chevron click handler must use panelsExpandedMap "
            "(per-task state), not panelsExpanded (global boolean)"
        )

    def test_update_visible_task_syncs_chevron_icon(self) -> None:
        """updateVisibleTask must call updateChevronIcon to reflect the
        visible task's expanded state from panelsExpandedMap."""
        src = _read_main_js()
        # Find updateVisibleTask function body
        m = re.search(r"function\s+updateVisibleTask\s*\(\s*\)\s*\{", src)
        assert m, "Could not find updateVisibleTask function"
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
        assert "updateChevronIcon" in body, (
            "updateVisibleTask must call updateChevronIcon to sync the "
            "chevron icon with the visible task's expanded state"
        )
        assert "panelsExpandedMap" in body, (
            "updateVisibleTask must read from panelsExpandedMap to "
            "determine the visible task's chevron state"
        )

    def test_no_stale_panels_expanded_boolean_usage(self) -> None:
        """There must be no remaining references to tab.panelsExpanded
        (the old scalar boolean) outside of panelsExpandedMap."""
        src = _read_main_js()
        # Find all occurrences of .panelsExpanded that are NOT .panelsExpandedMap
        # Use negative lookahead to exclude panelsExpandedMap
        stale_refs = re.findall(r"\.panelsExpanded(?!Map)\b", src)
        assert len(stale_refs) == 0, (
            f"Found {len(stale_refs)} stale references to .panelsExpanded "
            f"(should be .panelsExpandedMap): {stale_refs}"
        )

    def test_get_visible_task_name_helper_exists(self) -> None:
        """A getVisibleTaskName helper function must exist to determine
        which task the chevron should affect."""
        src = _read_main_js()
        assert re.search(r"function\s+getVisibleTaskName\s*\(", src), (
            "main.js must define a getVisibleTaskName() helper to determine "
            "which task the chevron toggle should affect"
        )
