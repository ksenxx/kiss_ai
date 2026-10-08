# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

CHAT_AGENT_PY = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "sorcar"
    / "chat_sorcar_agent.py"
)


class TestSubagentNewTabBroadcastIncludesParentTabId:
    """The sub-agent's ``new_tab`` broadcast (emitted in
    ``ChatSorcarAgent.run`` when ``_subagent_info`` is set) must
    include ``parent_tab_id`` so the frontend guard above can decide
    whether this webview owns the parent."""

    def test_broadcast_payload_includes_parent_tab_id_field(self) -> None:
        src = CHAT_AGENT_PY.read_text()
        idx = src.find('"type": "new_tab"')
        assert idx > 0, "could not locate sub-agent new_tab broadcast"
        block = src[idx : idx + 600]
        assert '"parent_tab_id"' in block, (
            "Sub-agent new_tab broadcast must include parent_tab_id so "
            "the frontend can route the new tab + resumeSession to the "
            "owning webview only.  Block was:\n" + block
        )
