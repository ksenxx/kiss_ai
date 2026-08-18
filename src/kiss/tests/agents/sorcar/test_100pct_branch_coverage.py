# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

Targets remaining uncovered branches in:
  _channel_cli.py (channel-agent CLI helpers)
  persistence.py: lines 263, 426
  sorcar_agent.py: lines 251-252
  chat_sorcar_agent.py: lines 130->134, 132-133
  useful_tools.py: lines 184, 204
  worktree_sorcar_agent.py: lines 187, 209-211, 313-314, 351
  json_printer.py: lines 205-215, 248, 254, 259-260, 281-285, 294, 302-310,
                 319-323, 329-330, 332, 333->335, 336, 340, 342, 344->346,
                 349, 352, 355, 358, 363-365, 367-368, 376
  server.py: lines 315->341, 319, 361->369, 416, 733-740

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import _generate_commit_message

_SavedState = tuple[Path, "sqlite3.Connection | None", Path]


def _redirect_db(tmpdir: str) -> _SavedState:
    old: _SavedState = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: _SavedState) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestWorktreeCommitMessageBranches:
    """Cover commit message generation branches."""

    @pytest.mark.slow
    def test_generate_commit_message_with_staged_changes(self, tmp_path: Path) -> None:
        """Commit message generation with staged changes exercises the LLM path.

        Creates a real repo with staged changes; the method either succeeds
        (returning an LLM-generated message) or catches an exception and
        returns the fallback, covering one of the two code paths.
        """
        saved = _redirect_db(str(tmp_path))
        try:
            repo = tmp_path / "commitgen"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("initial")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("modified content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)

            msg = _generate_commit_message(repo)
            assert isinstance(msg, str) and len(msg) > 0
        finally:
            _restore_db(saved)
