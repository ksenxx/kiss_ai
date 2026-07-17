# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :func:`kiss.server.sorcar.run`.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
temporary Unix-domain socket and drive the new synchronous
``kiss.server.sorcar.run`` API against it.  The only replaced boundary
is the LLM itself: like the other task-runner suites in this
directory, ``SorcarAgent``'s parent ``run`` is swapped for a stub so
the daemon's full run pipeline (``run`` command dispatch → worker
thread → agent wiring → event broadcast → status end) executes for
real without any model API calls.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer


def _init_repo(repo: str) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=False,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    git("config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    git("add", "seed.txt")
    git("commit", "-q", "-m", "seed")


class SorcarRunApiTest(unittest.TestCase):
    """Drive ``kiss.server.sorcar.run`` against a real daemon over UDS."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sorcar_run_api_")
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.repo,
        )
        self.server._printer._loop = self.loop
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
        _RunningAgentState.running_agent_states.clear()

        async def _shutdown() -> None:
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            for writer in writers:
                try:
                    writer.close()
                except Exception:
                    pass
            self.uds_server.close()
            await self.uds_server.wait_closed()
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence
        saved_dir, saved_path = self._saved_config_override
        if saved_dir is None:
            if "CONFIG_DIR" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_DIR")
        else:
            vscode_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            if "CONFIG_PATH" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_PATH")
        else:
            vscode_config.CONFIG_PATH = saved_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_success_returns_summary_cost_tokens_steps(self) -> None:
        """A successful task returns the parsed summary and metrics."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 1234
            self_agent.budget_used = 0.4567
            self_agent.total_steps = 7
            raw = (
                "success: true\n"
                "is_continue: false\n"
                "summary: API test done\n"
            )
            # Emit the terminal result event exactly like
            # ``RelentlessAgent.run`` does on a real completion.
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(
                    raw,
                    type="result",
                    step_count=7,
                    total_tokens=1234,
                    cost="$0.4567",
                )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "say hi",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert result.text == "API test done"
        assert result.tokens == 1234
        assert result.steps == 7
        assert abs(result.cost - 0.4567) < 1e-9
        # The returned ids must identify the run in the daemon's
        # persistence: the task row exists and belongs to the chat.
        assert result.task_id
        assert result.chat_id
        assert _persistence._get_task_chat_id(result.task_id) == result.chat_id

    def test_failure_returns_not_success_with_metrics(self) -> None:
        """A failing agent yields ``success=False`` plus its usage.

        Mirrors :meth:`RelentlessAgent.run`'s error contract: on a
        non-recoverable failure it broadcasts a terminal ``result``
        event carrying the error YAML and its usage counters, then
        returns that YAML to the task runner.
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 55
            self_agent.budget_used = 0.0123
            self_agent.total_steps = 3
            raw = "success: false\nis_continue: false\nsummary: boom\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(
                    raw,
                    type="result",
                    step_count=3,
                    total_tokens=55,
                    cost="$0.0123",
                )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "explode please",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False
        assert result.text == "boom"
        assert result.tokens == 55
        assert result.steps == 3
        assert abs(result.cost - 0.0123) < 1e-9
        assert result.task_id
        assert result.chat_id
        assert _persistence._get_task_chat_id(result.task_id) == result.chat_id

    def test_chat_id_continues_existing_chat(self) -> None:
        """Passing ``chat_id`` runs the task on that chat with context.

        The second run must (a) report the SAME ``chat_id`` it was
        given, (b) persist its task row under that chat, and (c) build
        its agent prompt from the first task's recorded task/result
        pair — proving the daemon truly continued the chat rather than
        minting a fresh session.
        """
        prompts_seen: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompts_seen.append(str(kwargs.get("prompt_template", "")))
            self_agent.total_tokens_used = 10
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = (
                "success: true\n"
                "is_continue: false\n"
                "summary: first answer marker\n"
            )
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(
                    raw,
                    type="result",
                    step_count=1,
                    total_tokens=10,
                    cost="$0.0010",
                )
            return raw

        self._parent_class.run = stub_run
        first = sorcar.run(
            "remember the magic word xyzzy",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert first.success is True
        assert first.chat_id
        second = sorcar.run(
            "what was the magic word?",
            work_dir=self.repo,
            chat_id=first.chat_id,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert second.success is True
        assert second.chat_id == first.chat_id
        assert second.task_id and second.task_id != first.task_id
        assert (
            _persistence._get_task_chat_id(second.task_id) == first.chat_id
        )
        # The second agent's prompt embeds the first task and its
        # result as prior chat context.
        assert len(prompts_seen) == 2
        assert "remember the magic word xyzzy" in prompts_seen[1]
        assert "first answer marker" in prompts_seen[1]

    def test_no_daemon_raises_connection_error(self) -> None:
        """A missing daemon socket raises a helpful ConnectionError."""
        missing = str(Path(self.tmpdir) / "nowhere.sock")
        with self.assertRaises(ConnectionError):
            sorcar.run("hello", sock_path=missing, timeout=5)

    def test_blank_prompt_raises_value_error(self) -> None:
        """Blank prompts are rejected before any connection is made."""
        with self.assertRaises(ValueError):
            sorcar.run("   ", sock_path=self.sock_path, timeout=5)
