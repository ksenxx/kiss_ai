"""Regression test: ``run_parallel`` must NOT cause phantom sub-agent
tabs to appear in webviews bound to a different chat.

Background
----------
When the user invokes ``run_parallel`` in a chat session "A" with 3
sub-tasks, the user reported seeing 6 sub-agent tabs in their tab
bar: the 3 expected children of the current run + 3 phantoms from a
PREVIOUS ``run_parallel`` invocation that ran under a different
``chat_id``.

Root cause
----------
1. Each sub-agent (in ``ChatSorcarAgent.run``) emits a ``new_tab``
   broadcast with ``taskId=""`` so the ``WebPrinter.broadcast`` treats
   it as a "global system event" and forwards it verbatim to every
   connected WS / UDS client (including webviews open against a
   different chat).
2. The ``openSubagentTab`` broadcasts emitted by
   ``VSCodeServer._replay_session`` and ``_open_persisted_subagent_tabs``
   likewise carry no routing ``tabId`` key and broadcast globally.
3. The frontend handlers (``case 'new_tab':`` and
   ``case 'openSubagentTab':`` in ``media/main.js``) unconditionally
   materialise the tab, regardless of whether the receiving webview
   actually owns the parent tab id.

Fix
----
- The backend's sub-agent ``new_tab`` broadcast must include
  ``parent_tab_id`` so the frontend can route correctly.
- Both the ``case 'new_tab':`` and ``case 'openSubagentTab':``
  handlers must short-circuit when ``ev.parent_tab_id`` is set AND no
  local tab carries that id — i.e. this webview does not own the
  parent tab and must not materialise the child.
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

CHAT_AGENT_PY = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "sorcar"
    / "chat_sorcar_agent.py"
)


def _case_block(case_label: str, next_case_label: str) -> str:
    js = MAIN_JS.read_text()
    case_start = js.find(case_label)
    assert case_start > 0, f"{case_label} not found in main.js"
    case_end = js.find(next_case_label, case_start)
    assert case_end > case_start, f"{next_case_label} not found after {case_label}"
    return js[case_start:case_end]


class TestOpenSubagentTabGuardsOnParentTabId:
    """The ``openSubagentTab`` handler must skip the event when its
    ``parent_tab_id`` does not correspond to a tab in the receiving
    webview's local ``tabs[]`` — that webview does not own the parent
    and must not spawn a phantom sub-agent tab."""

    def test_handler_short_circuits_when_parent_tab_id_unknown(
        self,
    ) -> None:
        block = _case_block(
            "case 'openSubagentTab':", "case 'subagentDone':",
        )
        # Must check ev.parent_tab_id and bail when the receiving
        # webview's tabs[] does not contain a tab with that id.
        assert "ev.parent_tab_id" in block
        assert "tabs.find" in block, (
            "openSubagentTab handler must consult local tabs[] to "
            "filter phantom sub-tab broadcasts from other webviews / "
            "chats.  Block was:\n" + block
        )
        # The guard must short-circuit; "break" or "return" must
        # appear in the early-guard expression that references
        # parent_tab_id and tabs.find.
        guard_idx = block.find("ev.parent_tab_id")
        # Look only at the early part of the block before makeTab so
        # the guard is provably an EARLY return.
        make_tab_idx = block.find("makeTab")
        assert 0 < guard_idx < make_tab_idx, (
            "The parent_tab_id guard must appear BEFORE makeTab so "
            "no phantom sub-tab is created."
        )
        early = block[:make_tab_idx]
        assert "tabs.find" in early and (
            "break" in early or "return" in early
        ), (
            "Early-guard must short-circuit with break/return when "
            "parent_tab_id is unknown locally.  Early block was:\n" + early
        )


class TestNewTabHandlerGuardsOnParentTabId:
    """The ``case 'new_tab':`` handler must apply the same guard so
    sub-agent ``new_tab`` broadcasts (taskId='') from chat A do not
    cause webviews bound to chat B to allocate phantom tabs and post
    ``resumeSession`` for the cross-chat sub-agent task."""

    def test_new_tab_handler_short_circuits_when_parent_unknown(
        self,
    ) -> None:
        block = _case_block("case 'new_tab':", "case 'openSubagentTab':")
        assert "ev.parent_tab_id" in block, (
            "case 'new_tab' must inspect ev.parent_tab_id to filter "
            "out cross-chat sub-agent new_tab broadcasts.  Block was:\n"
            + block
        )
        assert "tabs.find" in block
        guard_idx = block.find("ev.parent_tab_id")
        # Use the call form ``createNewTab()`` rather than the bare
        # identifier so we don't match a doc-comment occurrence.
        create_idx = block.find("createNewTab()")
        assert 0 < guard_idx < create_idx, (
            "The parent_tab_id guard must appear BEFORE createNewTab()."
        )


class TestSubagentNewTabBroadcastIncludesParentTabId:
    """The sub-agent's ``new_tab`` broadcast (emitted in
    ``ChatSorcarAgent.run`` when ``_subagent_info`` is set) must
    include ``parent_tab_id`` so the frontend guard above can decide
    whether this webview owns the parent."""

    def test_broadcast_payload_includes_parent_tab_id_field(self) -> None:
        src = CHAT_AGENT_PY.read_text()
        # Locate the ``"type": "new_tab"`` literal — there is only
        # one such broadcast in the file (the sub-agent spawn one).
        idx = src.find('"type": "new_tab"')
        assert idx > 0, "could not locate sub-agent new_tab broadcast"
        # Look at the surrounding ~600 chars (the dict literal).
        block = src[idx : idx + 600]
        assert '"parent_tab_id"' in block, (
            "Sub-agent new_tab broadcast must include parent_tab_id so "
            "the frontend can route the new tab + resumeSession to the "
            "owning webview only.  Block was:\n" + block
        )

    def test_run_tasks_parallel_stores_parent_tab_id_in_subagent_info(
        self,
    ) -> None:
        """``_run_tasks_parallel`` must capture the parent's frontend
        tab id from the running-agent registry and pass it to the
        sub-agent via ``_subagent_info`` so ``ChatSorcarAgent.run``
        can stamp the ``new_tab`` broadcast with it."""
        src = CHAT_AGENT_PY.read_text()
        # Locate _run_tasks_parallel body
        marker = "def _run_tasks_parallel"
        start = src.find(marker)
        assert start > 0
        # Bound by next def
        next_def = src.find("\n    def ", start + len(marker))
        block = src[start:next_def] if next_def > 0 else src[start:]
        assert "parent_tab_id" in block, (
            "_run_tasks_parallel must resolve the parent's tab_id and "
            "thread it through _subagent_info so the sub-agent's "
            "new_tab broadcast carries parent_tab_id.  Block was:\n"
            + block
        )
