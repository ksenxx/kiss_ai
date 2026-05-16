"""Regression test: clicking a sub-agent task in the history sidebar
must hide the input textbox and the buttons below it (model picker,
upload, menu, send/stop) in BOTH the VS Code extension and the web app.

Background
----------
Both surfaces share ``src/kiss/agents/vscode/media/main.js``.  The
input textbox and all buttons below it live inside
``<div id="input-container">``.  Hiding that single container hides
everything the user sees as "the input bar".

``restoreTab(tab)`` already toggles ``inputContainer.style.display``
based on ``tab.isSubagentTab`` when switching tabs.  However the
history-click flow goes:

  1. user clicks a sub-agent row in the history sidebar
  2. ``createNewTab()`` runs (regular chat tab, isSubagentTab=false)
  3. ``switchToTab(new_tab_id)`` → ``restoreTab`` → input bar SHOWN
  4. backend ``_replay_session`` broadcasts ``openSubagentTab`` for
     the same ``tab_id`` that was just switched to
  5. the ``openSubagentTab`` handler in ``main.js`` sets
     ``subTab.isSubagentTab = true`` and calls ``renderTabBar()`` /
     ``persistTabState()`` — neither of those touch
     ``#input-container``, so the input bar stays visible

The fix: in the ``openSubagentTab`` handler, if the converted tab is
the currently-active tab, hide ``#input-container``.
"""

from __future__ import annotations

from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _open_subagent_case_block() -> str:
    js = MAIN_JS.read_text()
    case_start = js.find("case 'openSubagentTab':")
    assert case_start > 0, "case 'openSubagentTab' not found in main.js"
    case_end = js.find("case 'subagentDone':", case_start)
    assert case_end > case_start
    return js[case_start:case_end]


class TestOpenSubagentTabHidesInputForActiveTab:
    """The ``openSubagentTab`` handler must hide ``#input-container``
    when the converted tab is the active tab — otherwise loading a
    sub-agent task from the history panel leaves the input bar
    visible in both the extension and the web app."""

    def test_handler_checks_active_tab(self) -> None:
        block = _open_subagent_case_block()
        # Must compare the converted sub-tab's id against activeTabId.
        assert "subTab.id === activeTabId" in block, (
            "openSubagentTab handler must guard the visibility update "
            "with `subTab.id === activeTabId` so it only hides the "
            "input bar when the converted tab is currently visible. "
            "Block was:\n" + block
        )

    def test_handler_hides_input_container(self) -> None:
        block = _open_subagent_case_block()
        # Must set ``inputContainer.style.display = 'none'`` inside
        # the case block.  Look for the exact assignment.
        assert "inputContainer.style.display = 'none'" in block, (
            "openSubagentTab handler must set "
            "`inputContainer.style.display = 'none'` to hide the input "
            "textbox and the buttons below it. Block was:\n" + block
        )

    def test_visibility_update_appears_after_flag_is_set(self) -> None:
        """The hide must execute AFTER ``isSubagentTab = true``; doing
        it before would be a no-op semantically and easy to misread.
        """
        block = _open_subagent_case_block()
        flag_idx = block.find("subTab.isSubagentTab = true")
        hide_idx = block.find("inputContainer.style.display = 'none'")
        assert flag_idx > 0
        assert hide_idx > flag_idx, (
            "The input-container hide must appear after "
            "`subTab.isSubagentTab = true` so the order of effects is "
            "obvious to readers."
        )


class TestRestoreTabAlreadyHidesForSubagentTab:
    """Sanity: ``restoreTab`` continues to enforce the rule on every
    tab switch — the openSubagentTab fix is needed only because the
    backend converts the tab AFTER the tab switch already ran."""

    def test_restore_tab_hides_input_when_subagent_tab(self) -> None:
        js = MAIN_JS.read_text()
        # restoreTab uses tab.isSubagentTab in the hide condition.
        # Locate the function and verify the condition is present.
        rt_start = js.find("function restoreTab")
        assert rt_start > 0
        # End of function is approximated by the next `function `.
        rt_end = js.find("function renderTabBar", rt_start)
        rt_block = js[rt_start:rt_end]
        assert "tab.isSubagentTab" in rt_block
        assert "inputContainer.style.display = 'none'" in rt_block
