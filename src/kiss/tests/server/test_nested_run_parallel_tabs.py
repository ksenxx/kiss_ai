"""Regression test: a nested ``run_parallel`` parent must name the tab the
frontend knows.

Bug
---
When a sub-agent spawned by ``run_parallel`` itself calls
``run_parallel``, the nested (grand-child) sub-agents never opened any
tabs in the VS Code webview.

Root cause
----------
The child's ``new_tab`` broadcast carried the parent's ``_tab_id`` as
``parent_tab_id``.  For a TOP-LEVEL parent that is the real frontend
tab id (set by the VS Code server), so the broadcast passes the
frontend guard::

    if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
      break;

But a NESTED parent (a sub-agent that calls ``run_parallel``) carries
the BACKEND synthetic tab id ``task-{grandparent_task_id}__sub_{idx}``
— while its frontend viewer tab was created by
``createBackgroundSubagentTab`` under a different id.  The nested
children's ``new_tab`` broadcasts would therefore carry a
``parent_tab_id`` that no frontend tab has, every webview would drop
them, and no tabs would open.

Fix
---
When the parent is itself a sub-agent (``self._subagent_info is not
None``), ``SorcarAgent._subagent_parent_tab_id`` derives the FRONTEND
tab id every webview gave this sub-agent —
``{parent_tab_id}__sub_{task_id}``, the ``subagentTabIdFor`` formula of
``media/main.js`` — and ``run_agent`` / ``run_parallel`` dispatches use
it as ``parent_tab_id``.  (An earlier fix looked the viewer up in the
printer's subscriber map instead, which is empty until a client's
``resumeSession`` arrives: a sub-agent fanning out right away still
lost its children's tabs.)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter

ROOT_TAB_ID = "frontend-root-tab"


class TestNestedRunParallelOpensTabs:
    """A nested ``run_parallel`` parent must name its frontend tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self.saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        self.saved_states = dict(agent_state.agent_states)
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        agent_state.agent_states.update(self.saved_states)
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parent_tab_id_is_derived_until_viewers_say_otherwise(self) -> None:
        """``_subagent_parent_tab_id`` of a sub-agent: the derived
        ``{parent_tab_id}__sub_{task_id}`` id before any client has
        subscribed and while the subscribed viewers agree with it; a
        viewer under another id (the parent chat was reopened in a new
        tab, so every webview shows this sub-agent as
        ``{new_tab}__sub_{task_id}``) wins over the derived id; the
        sub-agent's own synthetic id is never a candidate."""
        printer = JsonPrinter()
        sub = ChatSorcarAgent("nested-tabs-child")
        sub.printer = printer  # type: ignore[assignment]
        sub._tab_id = "task-grandparent__sub_0"  # type: ignore[attr-defined]
        sub._subagent_info = {  # type: ignore[attr-defined]
            "parent_task_id": "grandparent",
            "parent_tab_id": ROOT_TAB_ID,
            "reviewer": False,
        }
        task_id, _chat_id = th._add_task("child prompt", chat_id="")
        with sub._task_id_lock:
            sub._last_task_id = task_id
        derived = f"{ROOT_TAB_ID}__sub_{task_id}"

        # No viewer yet (the child fans out immediately).
        assert sub._subagent_parent_tab_id() == derived
        # Its own synthetic id is registered too (run_agent dispatches
        # do that) and a webview subscribed under the derived id.
        printer.subscribe_tab(task_id, sub._tab_id)
        printer.subscribe_tab(task_id, derived)
        assert sub._subagent_parent_tab_id() == derived
        # The parent chat was reopened under another tab: the webviews
        # now show this sub-agent as ``reopened-tab__sub_…``.
        printer.cleanup_tab(derived)
        printer.subscribe_tab(task_id, f"reopened-tab__sub_{task_id}")
        assert sub._subagent_parent_tab_id() == f"reopened-tab__sub_{task_id}"
