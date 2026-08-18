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

    def test_run_tasks_parallel_stores_parent_tab_id_in_subagent_info(
        self,
    ) -> None:
        """The fan-out must thread the parent's tab id to each child.

        ``ChatSorcarAgent.run`` stamps its ``new_tab`` broadcast with
        ``_subagent_info["parent_tab_id"]``, so the child has to be
        given the parent's real frontend tab id when it is spawned.
        """
        import threading
        from typing import Any

        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        class _Printer:
            """Thread-local-only printer: the engine needs nothing else."""

            def __init__(self) -> None:
                self._thread_local = threading.local()

        seen: list[Any] = []
        original_run = ChatSorcarAgent.run

        def _record(
            self: ChatSorcarAgent,
            prompt_template: str = "",
            **kwargs: Any,
        ) -> str:
            seen.append(self._subagent_info)
            return "success: true\nsummary: done"

        parent = ChatSorcarAgent("cross-chat-parent")
        parent._tab_id = "tab-owner-1"
        parent.printer = _Printer()  # type: ignore[assignment]
        try:
            ChatSorcarAgent.run = _record  # type: ignore[method-assign]
            parent._run_tasks_parallel(["a task"], max_workers=1)
        finally:
            ChatSorcarAgent.run = original_run  # type: ignore[method-assign]

        assert seen and seen[0] is not None
        assert seen[0].get("parent_tab_id") == "tab-owner-1", (
            "the sub-agent was not told which tab spawned it, so its "
            "new_tab broadcast cannot be routed to the owning webview"
        )
