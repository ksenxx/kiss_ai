# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Work-loss audit: git-ignored task output must survive worktree teardown.

A worktree task that creates files matched by ``.gitignore`` (a
downloaded dataset in an ignored ``data/`` directory, a generated
``*.csv`` report, ...) used to lose them silently on EVERY teardown
path: ``git add -A`` skips ignored files, so the auto-commit cannot
capture them, and ``git worktree remove --force`` then deletes the
directory.  Had the same task run without a worktree, those files
would still be on disk.

These tests drive the real ``WorktreeSorcarAgent`` against a real git
repository (the parent class' ``run`` is replaced with a deterministic
stub that writes files — the same no-mock pattern the rest of the
worktree suite uses) and assert the rescue behavior:

* merge / release rescues ignored files into the main repository;
* an existing main-tree file is NEVER overwritten by a rescue;
* regenerable cache directories (``__pycache__``, ``.venv``, ...) are
  not rescued;
* the automatic discard paths rescue too (``rescue_ignored=True``),
  while a user-explicit ``discard()`` still throws everything away;
* the orphan-worktree reclaim pass rescues before it removes;
* the post-task auto path in the server rescues a task whose ONLY
  output was ignored files (the changed-files probe reports the
  worktree as empty and picks "discard").
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.test_worktree_ignored_file_rescue import (  # noqa: F401
    _IGNORED_CONTENT,
    _IGNORED_REL,
    _git,
    _make_repo,
    _redirect_db,
    _restore_db,
    _stub_parent_run,
)


class TestIgnoredFileRescueServerPostTask:
    """The server's post-task auto path rescues ignored-only output.

    A task whose ONLY output is ignored files makes the changed-files
    probe report an empty worktree, so the post-task auto-commit path
    picks the internal "discard" action — which must rescue before
    removing.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-rescue-server-")
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self._original_run: Any = None

    def teardown_method(self) -> None:
        if self._original_run is not None:
            cast(Any, SorcarAgent.__mro__[1]).run = self._original_run
        from kiss.server import agent_state
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup
                    pass
        agent_state.agent_states.clear()
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_post_task_auto_discard_rescues_ignored_only_output(
        self,
    ) -> None:
        """End-to-end through VSCodeServer._run_task_inner."""
        from kiss.server.server import VSCodeServer

        self._original_run = _stub_parent_run(
            {_IGNORED_REL: _IGNORED_CONTENT},
        )
        server = VSCodeServer()
        server.work_dir = str(self.repo)
        events: list[dict] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        server._run_task_inner({
            "prompt": "task with ignored-only output",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })
        rescued = self.repo / _IGNORED_REL
        assert rescued.is_file(), (
            "the post-task auto-discard destroyed ignored-only task "
            f"output; events: {[e.get('type') for e in events]}"
        )
        assert rescued.read_text() == _IGNORED_CONTENT
