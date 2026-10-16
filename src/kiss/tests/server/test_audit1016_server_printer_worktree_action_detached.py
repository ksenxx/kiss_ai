# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``worktreeAction`` must not stall the connection's other commands.

Each WebSocket connection's commands are dispatched one after another
(``RemoteAccessServer._ws_handler`` awaits every dispatch before it
reads the next frame; ``ServerApi.ping`` relies on that order).  A
worktree merge generates its commit message with an LLM call and may
run the merge-conflict SEA for minutes, so when the ``worktreeAction``
ran inline every later command of the same browser window — a ``stop``
of another tab's task, a ``submit``, the transport's ``ping`` — waited
behind it.  ``ServerApi.worktree_action`` now runs the backend command
as a background task, like ``getAppsStatus`` / ``getMachines``.

Everything here is real: a real ``RemoteAccessServer`` on an ephemeral
port with an isolated ``KISS_HOME``, a real git repository with a real
``git worktree``, a real blocking ``pre-commit`` hook that holds the
merge's commit until the test releases it, and a real WebSocket
client.  The hook is how the test knows the merge is still running when
the ``pong`` arrives.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.core import user_config
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl, run_git
from kiss.tests.server.parallel_agent_harness import OfflineFastModel

_TAB = "detached-merge-tab"
_TASK_ID = "task-detached-merge"
_MERGED_FILE = "merge-me.txt"

#: Blocks the commit the merge makes until the release file appears.
#: The 20 s cap keeps a failing test from wedging the suite.
_HOOK = """#!/bin/sh
touch "$1"
i=0
while [ ! -f "$2" ] && [ $i -lt 400 ]; do
  sleep 0.05
  i=$((i+1))
done
exit 0
"""


class TestWorktreeActionDetached(IsolatedAsyncioTestCase):
    """A ``ping`` sent after a merge is answered while the merge still runs."""

    async def asyncSetUp(self) -> None:
        """Stand up an isolated home, a repo with a dirty worktree, a server."""
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-merge-detached-"))
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
        self.addCleanup(self.offline.__exit__, None, None, None)

        self.port = find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=str(self.repo),
            use_tunnel=False,
            local_endpoint_file=self.home / "test-sorcar-local.json",
            url_file=self.home / "remote_url.txt",
        )
        await self.server.start_async()
        self.state = agent_state.AgentState(
            _TASK_ID, tab_id=_TAB, server_owned=True,
        )
        self.state.agent = self._pending_worktree_agent()
        agent_state.register(self.state)

    async def asyncTearDown(self) -> None:
        """Release the hook, stop the server and drop all global state."""
        self._release_hook()
        await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
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
        persistence._DB_PATH = self.home / "history.db"
        persistence._db_conn = None
        self._saved_cfg = (user_config.CONFIG_DIR, user_config.CONFIG_PATH)
        user_config.CONFIG_DIR = self.home
        user_config.CONFIG_PATH = self.home / "config.json"
        (self.home / "config.json").write_text("{}", encoding="utf-8")

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
        user_config.CONFIG_DIR, user_config.CONFIG_PATH = self._saved_cfg
        if self._saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_env

    def _init_repo(self) -> None:
        """Create a git repo with one commit on ``main``."""
        run_git(self.repo, "init", "-q", "-b", "main")
        run_git(self.repo, "config", "user.email", "test@example.com")
        run_git(self.repo, "config", "user.name", "Test User")
        run_git(self.repo, "config", "commit.gpgsign", "false")
        (self.repo / "README.md").write_text("hello\n", encoding="utf-8")
        run_git(self.repo, "add", "README.md")
        run_git(self.repo, "commit", "-q", "-m", "initial")

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

    def _pending_worktree_agent(self) -> ChatSorcarAgent:
        """Return an agent holding a real worktree with uncommitted work."""
        branch = "kiss/wt-merge-detached"
        wt_dir = self.repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        (wt_dir / _MERGED_FILE).write_text("agent work\n", encoding="utf-8")
        agent = ChatSorcarAgent("Sorcar")
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

    @staticmethod
    async def _recv_type(ws: Any, wanted: str, timeout: float) -> dict[str, Any] | None:
        """Read frames until one of type *wanted* arrives, or *timeout* passes."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                return None
            event: dict[str, Any] = json.loads(raw)
            if event.get("type") == wanted:
                return event

    async def test_ping_is_answered_while_the_merge_runs(self) -> None:
        """A ``ping`` queued behind a merge gets its ``pong`` before the merge ends."""
        async with connect(f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl()) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)

            await ws.send(json.dumps({
                "type": "worktreeAction",
                "action": "merge",
                "tabId": _TAB,
                "workDir": str(self.repo),
            }))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not self.hook_started.exists():
                await asyncio.sleep(0.01)
            self.assertTrue(
                self.hook_started.exists(), "the merge never reached its commit",
            )

            await ws.send(json.dumps({"type": "ping"}))
            pong = await self._recv_type(ws, "pong", timeout=5)
            with agent_state.STATE_LOCK:
                still_merging = self.state.is_merging
            self.assertIsNotNone(
                pong,
                "the ping was held up behind the running merge: the "
                "connection's command channel is stalled by worktreeAction",
            )
            self.assertFalse(self.hook_release.exists())
            self.assertTrue(still_merging, "the merge finished before the pong")

            self._release_hook()
            result = await self._recv_type(ws, "worktree_result", timeout=60)
            self.assertIsNotNone(result, "the merge never reported its result")
            assert result is not None
            self.assertTrue(result.get("success"), result)
            self.assertEqual(result.get("tabId"), _TAB)

        merged = run_git(self.repo, "cat-file", "-e", f"HEAD:{_MERGED_FILE}")
        self.assertEqual(merged.returncode, 0, "the merged file never landed on main")
        with agent_state.STATE_LOCK:
            self.assertFalse(self.state.is_merging)

    async def test_shutdown_returns_as_soon_as_the_merge_completes(self) -> None:
        """``stop_async`` waits for the merge itself, not for the executor worker.

        The detached merge runs on a thread of the event loop's default
        executor, which is pooled and stays alive after the merge.
        Shutdown used to ``join`` that thread and so waited the whole
        30 s grace period although the repository had been merged
        within a second.  It must return promptly once the merge's
        claim is released.
        """
        async with connect(f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl()) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            await ws.send(json.dumps({
                "type": "worktreeAction",
                "action": "merge",
                "tabId": _TAB,
                "workDir": str(self.repo),
            }))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not self.hook_started.exists():
                await asyncio.sleep(0.01)
            self.assertTrue(
                self.hook_started.exists(), "the merge never reached its commit",
            )
            with agent_state.STATE_LOCK:
                self.assertTrue(self.state.is_merging)

            started = time.monotonic()
            stopping = asyncio.ensure_future(self.server.stop_async())
            await asyncio.sleep(0.5)
            self.assertFalse(stopping.done(), "shutdown returned mid-merge")
            self._release_hook()
            await asyncio.wait_for(stopping, timeout=20)
            elapsed = time.monotonic() - started

        self.assertLess(
            elapsed, 5.0,
            f"stop_async took {elapsed:.1f}s: it waited for the pooled executor "
            "worker instead of the merge operation",
        )
        with agent_state.STATE_LOCK:
            self.assertFalse(self.state.is_merging)
        merged = run_git(self.repo, "cat-file", "-e", f"HEAD:{_MERGED_FILE}")
        self.assertEqual(merged.returncode, 0, "shutdown cut the merge short")


if __name__ == "__main__":
    unittest.main()
