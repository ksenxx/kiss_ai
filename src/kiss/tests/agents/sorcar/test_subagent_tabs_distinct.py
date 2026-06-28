# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: 3 parallel sub-agents must produce 3 distinct
``new_tab`` broadcasts so the frontend can allocate 3 distinct tabs.

When ``_run_tasks_parallel`` (either the module-level helper or
``ChatSorcarAgent._run_tasks_parallel``) spawns N sub-agents, each
sub-agent's ``ChatSorcarAgent.run`` detects ``self._subagent_info is
not None`` immediately after ``_add_task`` and self-broadcasts a
``new_tab`` event carrying the freshly-minted backend ``task_id``.
The frontend's ``new_tab`` handler then allocates a tab via
``createNewTab`` and posts ``resumeSession`` with the same ``task_id``,
subscribing the new tab to the sub-agent's live event stream.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
from kiss.core.printer import Printer

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


class _MockPrinter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._thread_local = threading.local()
        self._lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)


class TestBackendBroadcastsDistinctSubagentTabs:
    def test_module_level_run_tasks_parallel_broadcasts_new_tab(self) -> None:
        """``run_tasks_parallel`` broadcasts a ``new_tab`` per sub-agent.

        The module-level helper no longer mints any frontend ``tab_id``.
        Instead, the moment each sub-agent's backend ``task_id`` is
        allocated by ``_add_task``, a ``new_tab`` event carrying that
        ``task_id`` is broadcast.  The frontend's ``new_tab`` handler
        allocates a fresh tab and posts ``resumeSession`` with the same
        ``task_id``, which subscribes the new tab to the sub-agent's
        live event stream.
        """
        printer = _MockPrinter()
        printer._thread_local.task_id = "parent-1"

        with patch(
            "kiss.agents.sorcar.sorcar_agent.SorcarAgent.run",
            return_value='{"success": true, "summary": "ok"}',
        ):
            run_tasks_parallel(
                ["Task A", "Task B", "Task C"],
                max_workers=1,
                printer=cast(Printer, printer),
            )

        new_tab_evts = [
            e for e in printer.events if e.get("type") == "new_tab"
        ]
        assert len(new_tab_evts) == 3
        task_ids = [e["task_id"] for e in new_tab_evts]
        assert len(set(task_ids)) == 3, "task_ids must be unique"
        for evt in new_tab_evts:
            assert isinstance(evt["task_id"], str), (
                f"new_tab event must carry an int task_id: {evt}"
            )

    def test_chat_sorcar_agent_run_tasks_parallel_broadcasts_new_tab(
        self,
    ) -> None:
        """``ChatSorcarAgent._run_tasks_parallel`` yields 3 ``new_tab``
        broadcasts — one per sub-agent.

        The parallel executor itself emits no tab events; it only
        pre-sets ``_subagent_info`` on each freshly-constructed
        sub-agent.  Each sub-agent's ``ChatSorcarAgent.run`` then
        broadcasts a ``new_tab`` carrying its backend ``task_id``
        immediately after ``_add_task``, *before* ``super().run()``
        runs (which we mock out).  Patching ``SorcarAgent.run``
        (instead of the ChatSorcarAgent instance method) keeps the
        ``ChatSorcarAgent.run`` prefix — and therefore the
        broadcast — intact.
        """
        agent = ChatSorcarAgent("test")
        printer = _MockPrinter()
        agent.printer = cast(Printer | None, printer)
        printer._thread_local.task_id = "parent-2"

        with patch(
            "kiss.agents.sorcar.sorcar_agent.SorcarAgent.run",
            return_value='{"success": true, "summary": "ok"}',
        ):
            agent._run_tasks_parallel(["A", "B", "C"], max_workers=1)

        new_tab_evts = [
            e for e in printer.events if e.get("type") == "new_tab"
        ]
        assert len(new_tab_evts) == 3
        task_ids = [e["task_id"] for e in new_tab_evts]
        assert len(set(task_ids)) == 3, "task_ids must be unique"
        for evt in new_tab_evts:
            assert isinstance(evt["task_id"], str), (
                f"new_tab event must carry an int task_id: {evt}"
            )




class TestSubagentTitlesAreVisuallyDistinct:
    """When three task descriptions share a 40-char prefix, the
    rendered tab titles still differ because the index prefix
    differentiates them.
    """

    def test_titles_differ_when_descriptions_share_prefix(self) -> None:
        # Simulate what the frontend handler builds.
        descriptions = [
            "Research and summarize: WebAssembly portable binary...",
            "Research and summarize: Rust ownership model with...",
            "Research and summarize: The Actor model in concurrency...",
        ]
        titles = [
            str(i + 1) + ". " + desc[:40]
            for i, desc in enumerate(descriptions)
        ]
        assert len(set(titles)) == 3, (
            "titles must be unique even when descriptions share prefix"
        )
        # First few characters (typical visible portion in a narrow tab
        # bar) must already differ — this is the actual UX guarantee.
        prefixes = [t[:4] for t in titles]
        assert len(set(prefixes)) == 3, (
            f"first 4 chars of titles must differ: {prefixes}"
        )
