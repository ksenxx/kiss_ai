# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shutdown must not return while a merge is still rewriting the repo.

The interactive "Auto-commit and merge" / "Discard" action arrives as a
forwarded command and runs in the event loop's default executor, not on
a task worker thread.  By the time the user clicks it the task is over,
so ``AgentState.task_thread`` is already ``None`` and the shutdown
sweep of in-flight *tasks* does not see it.  If the server's lifecycle
ends there, the merge keeps committing, checking out and merging after
the promised shutdown boundary — and on ``SIGTERM`` the failsafe kills
the process in the middle of it.

Everything here is real: a real ``RemoteAccessServer`` on a temporary
Unix socket and an ephemeral port with an isolated ``KISS_HOME``, a
real git repository with a real ``git worktree``, a real blocking
``pre-commit`` hook, the real command dispatcher running in the real
default executor, and the real ``stop_async`` lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core import vscode_config
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.sorcar.parallel_agent_harness import OfflineFastModel

_TAB = "merge-tab"
_TASK_ID = "task-merge-shutdown"
_MERGED_FILE = "merge-me.txt"

#: Blocks the commit the merge makes until the test releases it, so the
#: merge is provably still running when shutdown starts.  The timeout
#: keeps a failing test from wedging the suite.
_HOOK = """#!/bin/sh
touch "$1"
i=0
while [ ! -f "$2" ] && [ $i -lt 200 ]; do
  sleep 0.05
  i=$((i+1))
done
exit 0
"""


def _git(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd* and return the completed process."""
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        check=False,
    )


def _free_port() -> int:
    """Return a free localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestShutdownWaitsForInteractiveMerge(IsolatedAsyncioTestCase):
    """``stop_async`` waits for an interactive merge to finish."""

    async def asyncSetUp(self) -> None:
        """Stand up an isolated home, a repo with a dirty worktree, a server."""
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-merge-shutdown-"))
        self.home = self.tmp / ".kiss"
        self.home.mkdir(parents=True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self._enter_isolated_home()
        self._init_repo()
        self.hook_started = self.tmp / "hook-started"
        self.hook_release = self.tmp / "hook-release"
        self._install_blocking_hook()
        self.offline = OfflineFastModel()
        self.offline.__enter__()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_free_port(),
            work_dir=str(self.repo),
            use_tunnel=False,
            uds_path=self.home / "test-sorcar.sock",
            url_file=self.home / "remote_url.txt",
        )
        await self.server.start_async()
        self.state = agent_state.AgentState(
            _TASK_ID, tab_id=_TAB, server_owned=True,
        )
        self.state.agent = self._pending_worktree_agent()
        self.state.use_worktree = True
        agent_state.register(self.state)

    async def asyncTearDown(self) -> None:
        """Release the hook, stop the server and drop all global state."""
        self._release_hook()
        await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        self.offline.__exit__(None, None, None)
        self._leave_isolated_home()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _enter_isolated_home(self) -> None:
        """Redirect KISS_HOME, the history DB and the config file."""
        self._saved_env = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(self.home)
        self._saved_db = (
            persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR,
        )
        persistence._KISS_DIR = self.home
        persistence._DB_PATH = self.home / "sorcar.db"
        persistence._db_conn = None
        self._saved_cfg = (
            getattr(vscode_config, "CONFIG_DIR", None),
            getattr(vscode_config, "CONFIG_PATH", None),
        )
        vscode_config.CONFIG_DIR = self.home  # type: ignore[attr-defined]
        vscode_config.CONFIG_PATH = self.home / "config.json"  # type: ignore[attr-defined]
        (self.home / "config.json").write_text(
            json.dumps({"auto_commit_mode": True, "is_worktree": True}),
            encoding="utf-8",
        )

    def _leave_isolated_home(self) -> None:
        """Undo :meth:`_enter_isolated_home`."""
        if persistence._db_conn is not None:
            try:
                persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR,
        ) = self._saved_db
        for name, value in zip(
            ("CONFIG_DIR", "CONFIG_PATH"), self._saved_cfg, strict=True,
        ):
            if value is not None:
                setattr(vscode_config, name, value)
        if self._saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_env

    def _init_repo(self) -> None:
        """Create a git repo with one commit on ``main``."""
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        (self.repo / "README.md").write_text("hello\n", encoding="utf-8")
        _git(self.repo, "add", "README.md")
        _git(self.repo, "commit", "-q", "-m", "initial")

    def _install_blocking_hook(self) -> None:
        """Install a real ``pre-commit`` hook that blocks the merge's commit."""
        hooks = self.repo / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        script = hooks / "pre-commit"
        script.write_text(
            f'#!/bin/sh\nexec "{hooks / "block.sh"}" '
            f'"{self.hook_started}" "{self.hook_release}"\n',
            encoding="utf-8",
        )
        block = hooks / "block.sh"
        block.write_text(_HOOK, encoding="utf-8")
        script.chmod(0o755)
        block.chmod(0o755)

    def _pending_worktree_agent(self) -> WorktreeSorcarAgent:
        """Return an agent holding a real worktree with uncommitted work."""
        branch = "kiss/wt-merge-shutdown"
        wt_dir = self.repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        (wt_dir / _MERGED_FILE).write_text("agent work\n", encoding="utf-8")
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=self.repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        return agent

    def _release_hook(self) -> None:
        """Let the blocking ``pre-commit`` hook proceed."""
        try:
            self.hook_release.write_text("go", encoding="utf-8")
        except OSError:  # pragma: no cover — temp tree already removed
            pass

    def _release_after_shutdown_starts(self) -> threading.Thread:
        """Release the hook once shutdown has begun, on its own thread."""

        def _worker() -> None:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.server._shutdown_initiated:
                    break
                time.sleep(0.01)
            # Give a shutdown that does NOT wait a real chance to run to
            # completion while the merge is still inside the hook.
            time.sleep(0.5)
            self._release_hook()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread

    async def test_stop_async_waits_for_the_merge_it_started(self) -> None:
        """A merge in the executor finishes before shutdown returns."""
        merge_returned = threading.Event()
        command = {
            "type": "worktreeAction",
            "action": "merge",
            "tabId": _TAB,
            "workDir": str(self.repo),
        }

        def _merge() -> None:
            try:
                self.server._vscode_server._handle_command(command)
            finally:
                merge_returned.set()

        loop = asyncio.get_running_loop()
        merge_future = loop.run_in_executor(None, _merge)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not self.hook_started.exists():
            await asyncio.sleep(0.01)
        self.assertTrue(
            self.hook_started.exists(), "the merge never reached its commit",
        )

        releaser = self._release_after_shutdown_starts()
        await self.server.stop_async()

        self.assertTrue(
            merge_returned.is_set(),
            "stop_async returned while the interactive merge was still "
            "committing and merging the repository",
        )
        with agent_state.STATE_LOCK:
            self.assertFalse(self.state.is_merging)

        await merge_future
        releaser.join(timeout=30)
        merged = _git(self.repo, "cat-file", "-e", f"HEAD:{_MERGED_FILE}")
        self.assertEqual(
            merged.returncode, 0,
            "the merge did not finish, so the user's work never landed",
        )


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
