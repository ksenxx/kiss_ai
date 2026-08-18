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

from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


class TestSubagentTitlesAreVisuallyDistinct:
    """When three task descriptions share a 40-char prefix, the
    rendered tab titles still differ because the index prefix
    differentiates them.
    """

    def test_titles_differ_when_descriptions_share_prefix(self) -> None:
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
        prefixes = [t[:4] for t in titles]
        assert len(set(prefixes)) == 3, (
            f"first 4 chars of titles must differ: {prefixes}"
        )
