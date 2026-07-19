# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: third-party agents launch via ``kiss.server.sorcar.run``.

Feature under test
------------------
Every agent in ``kiss/agents/third_party_agents/`` must launch through
``run_agent_via_kiss_web``, which is implemented ON TOP OF the public
synchronous client API :func:`kiss.server.sorcar.run`: the launcher
connects to a daemon's Unix-domain socket, sends the documented ``run``
command, and supplies the agent's live channel tools through the
API's ``tools=`` *file path* contract (a generated tools file whose
top-level functions bridge back to the live callables — see
``kiss.agents.third_party_agents._api_tools_bridge``).  The task is
executed by a daemon-built chat agent, NOT by the passed instance.

Test strategy (no mocks)
------------------------
A real :class:`kiss.server.web_server.RemoteAccessServer` is served on
a temporary Unix-domain socket (the production daemon transport) with
isolated persistence/config.  The only replaced boundary is the LLM
itself: ``RelentlessAgent.run`` (``SorcarAgent.__mro__[1].run``) is
swapped for a stub returning canned YAML (precedent:
``test_server_sorcar_run.py``), so the daemon's full pipeline — UDS
dispatch → ``_cmd_run`` → worker thread → tools-file loading → event
broadcast → status end — executes for real without model API calls.
"""

from __future__ import annotations

import ast
import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebChatSorcarAgent,
    KissWebWorktreeSorcarAgent,
    run_agent_via_kiss_web,
)
from kiss.core import vscode_config
from kiss.server.web_server import RemoteAccessServer

STUB_SUMMARY = "stub summary done"


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


class _ApiLaunchBase(unittest.TestCase):
    """Real daemon over a temp UDS; only the LLM boundary is stubbed."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tp-api-launch-")
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

        # Route launches that do not pass ``sock_path`` explicitly
        # (ChannelRunner, channel_main, the pollers) at this test's
        # daemon instead of the process-global in-process one.
        self._saved_sock_override = launcher._SOCK_PATH_OVERRIDE
        launcher._SOCK_PATH_OVERRIDE = self.sock_path

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.stub_calls: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        launcher._SOCK_PATH_OVERRIDE = self._saved_sock_override
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

    def _install_stub(
        self,
        summary: str = STUB_SUMMARY,
        success: bool = True,
        tokens: int = 42,
        cost: float = 0.0420,
        steps: int = 4,
        block: threading.Event | None = None,
        raise_exc: BaseException | None = None,
        on_run: Any = None,
    ) -> None:
        """Install the LLM-boundary stub on ``RelentlessAgent.run``.

        The stub records its ``self``/kwargs, optionally blocks, raises
        or invokes *on_run* (whose return value, if a string, replaces
        the summary), then emits the terminal ``result`` event exactly
        like ``RelentlessAgent.run`` does on a real completion.
        """
        calls = self.stub_calls

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            calls.append({
                "agent": self_agent,
                "kwargs": kwargs,
                "thread": threading.current_thread(),
            })
            if block is not None:
                block.wait(timeout=30)
            if raise_exc is not None:
                raise raise_exc
            text = summary
            if on_run is not None:
                out = on_run(self_agent, kwargs)
                if isinstance(out, str):
                    text = out
            self_agent.total_tokens_used = tokens
            self_agent.budget_used = cost
            self_agent.total_steps = steps
            raw: str = yaml.safe_dump(
                {"success": success, "is_continue": False, "summary": text},
                sort_keys=False,
            )
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw,
                    type="result",
                    step_count=steps,
                    total_tokens=tokens,
                    cost=f"${cost:.4f}",
                )
            return raw

        self._parent_class.run = stub_run


class TestLaunchViaApi(_ApiLaunchBase):
    """The launcher must run tasks through ``kiss.server.sorcar.run``."""

    def test_task_runs_on_daemon_agent_not_passed_instance(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        result = run_agent_via_kiss_web(
            agent,
            "hello slack task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert self.stub_calls, "the daemon never ran the task"
        call = self.stub_calls[0]
        # The API contract: the daemon builds its own chat agent — the
        # passed third-party instance must NOT be executed.
        assert call["agent"] is not agent, (
            "the task must run on a daemon-built agent, not the passed "
            "third-party agent instance"
        )
        assert isinstance(call["agent"], ChatSorcarAgent)
        assert call["thread"] is not threading.main_thread(), (
            "the task must run on the daemon's worker thread"
        )
        prompt = str(call["kwargs"].get("prompt_template", ""))
        assert "hello slack task" in prompt
        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        assert parsed["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result

    def test_channel_prompt_appended_to_task_prompt(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "auth prompt task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        prompt = str(self.stub_calls[0]["kwargs"].get("prompt_template", ""))
        # The channel guidance the old flow injected as a system
        # prompt must now travel inside the task prompt (the API has
        # no system-prompt field).
        assert "Slack Authentication" in prompt
        assert "start_slack_browser_auth" in prompt

    def test_auth_and_extra_tools_bridged_via_tools_file(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        live_calls: list[str] = []

        def mytool(text: str, repeat: int = 1) -> str:
            """Echo *text* repeated *repeat* times.

            Args:
                text: The text to echo.
                repeat: How many times to repeat it.
            """
            live_calls.append(text)
            return text * repeat

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            # Slack auth tools (live closures over the agent instance)
            # must be present as tools-file wrappers.
            for expected in (
                "check_slack_auth",
                "authenticate_slack",
                "clear_slack_auth",
                "start_slack_browser_auth",
                "mytool",
            ):
                assert expected in tools, f"missing bridged tool {expected}"
            wrapper = tools["mytool"]
            # Proof of the tools-FILE path (not serialized callables):
            # the daemon imported a generated module.
            assert wrapper is not mytool
            assert wrapper.__module__.startswith("_kiss_tools_file_")
            assert "Echo *text* repeated" in (wrapper.__doc__ or "")
            # Invoking the wrapper reaches the live closure, with the
            # original default applied for the omitted parameter.
            assert wrapper(text="hi") == "hi"
            assert wrapper("bye", repeat=2) == "byebye"
            # The auth tool executes the real closure over the real
            # SlackAgent instance (unauthenticated → guidance text).
            auth_out = tools["check_slack_auth"]()
            assert "Not authenticated with Slack" in auth_out
            return "tools bridged ok"

        self._install_stub(on_run=on_run)
        result = run_agent_via_kiss_web(
            SlackAgent(),
            "use the tools",
            work_dir=self.repo,
            tools=[mytool],
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == "tools bridged ok"
        assert live_calls == ["hi", "bye"], (
            "bridged tool invocations must execute the live closure in "
            "the launching code"
        )

    def test_backend_tools_included_when_authenticated(self) -> None:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            BaseChannelAgent,
            ToolMethodBackend,
        )

        class _Backend(ToolMethodBackend):
            def __init__(self) -> None:
                self.notes: list[str] = []

            def add_note(self, note: str) -> str:
                """Record a note on the backend.

                Args:
                    note: The note text to record.
                """
                self.notes.append(note)
                return f"recorded:{note}"

        class _Agent(BaseChannelAgent, SorcarAgent):
            def __init__(self) -> None:
                super().__init__("Backend Test Agent")
                self._backend = _Backend()

            def _is_authenticated(self) -> bool:
                return True

            def _get_auth_tools(self) -> list:
                return []

        agent = _Agent()

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            assert "add_note" in tools, "backend tool must be bridged"
            return str(tools["add_note"](note="from daemon"))

        self._install_stub(on_run=on_run)
        result = run_agent_via_kiss_web(
            agent,
            "note task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == "recorded:from daemon"
        assert agent._backend.notes == ["from daemon"], (
            "the bridged bound method must mutate the live backend "
            "instance in the launching code"
        )

    def test_unauthenticated_backend_tools_excluded(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            names = {t.__name__ for t in (kwargs.get("tools") or [])}
            assert "post_message" not in names, (
                "backend tools must not be exposed when unauthenticated"
            )
            return "ok"

        self._install_stub(on_run=on_run)
        run_agent_via_kiss_web(
            SlackAgent(),
            "no backend",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert self.stub_calls

    def test_overrides_forwarded_through_run_command(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
            max_budget=1.25,
            model_config={"base_url": "http://localhost:9999/v1"},
            web_tools=False,
            is_parallel=True,
        )
        call = self.stub_calls[0]
        assert call["kwargs"].get("max_budget") == 1.25
        assert call["kwargs"].get("model_config") == {
            "base_url": "http://localhost:9999/v1",
        }
        assert getattr(call["agent"], "_use_web_tools", None) is False
        assert getattr(call["agent"], "_is_parallel", None) is True

    def test_zero_budget_override_is_honored(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
            max_budget=0.0,
        )
        assert self.stub_calls[0]["kwargs"].get("max_budget") == 0.0

    def test_defaults_use_daemon_config(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        call = self.stub_calls[0]
        budget = call["kwargs"].get("max_budget")
        assert budget is not None and budget > 0, (
            "without an override the daemon config budget applies"
        )
        assert call["kwargs"].get("model_name"), (
            "the daemon default model applies when none is passed"
        )

    def test_model_name_forwarded(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            model_name="gpt-5.5",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert self.stub_calls[0]["kwargs"].get("model_name") == "gpt-5.5"

    def test_stats_recorded_on_agent_for_cli_stats(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub(tokens=1234, cost=0.4567, steps=7)
        agent = SlackAgent()
        run_agent_via_kiss_web(
            agent,
            "stats task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert agent.total_tokens_used == 1234
        assert abs(agent.budget_used - 0.4567) < 1e-9
        assert agent.total_steps == 7

    def test_agent_failure_returns_failure_yaml(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        # ``RelentlessAgent.run``'s failure contract: broadcast a
        # terminal ``result`` event and return the failure YAML.
        self._install_stub(summary="boom-fail happened", success=False)
        agent = SlackAgent()
        result = run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert "boom-fail" in str(parsed["summary"])
        assert agent.last_run_result == result

    def test_abrupt_agent_crash_maps_to_failure_yaml(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        # An abrupt crash (no terminal result event broadcast at all)
        # must still yield a non-empty failure summary.
        self._install_stub(raise_exc=RuntimeError("boom-crash"))
        agent = SlackAgent()
        result = run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert str(parsed["summary"]).strip(), (
            "an abrupt crash must not produce an empty summary"
        )
        assert agent.last_run_result == result

    def test_blank_prompt_returns_failure_yaml(self) -> None:
        agent = KissWebChatSorcarAgent("Blank Prompt Agent")
        result = run_agent_via_kiss_web(
            agent,
            "   ",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert "empty" in str(parsed["summary"]).lower()

    def test_timeout_returns_empty_result(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        release = threading.Event()
        self._install_stub(block=release)
        try:
            result = run_agent_via_kiss_web(
                SlackAgent(),
                "task",
                work_dir=self.repo,
                sock_path=self.sock_path,
                timeout=0.5,
            )
            assert result == "", "timed-out launch must return empty result"
        finally:
            release.set()
            deadline = 30.0
            for state in list(
                _RunningAgentState.running_agent_states.values()
            ):
                if state.task_thread is not None:
                    state.task_thread.join(timeout=deadline)

    def test_invalid_tool_raises_before_connecting(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        with self.assertRaises(ValueError):
            run_agent_via_kiss_web(
                SlackAgent(),
                "task",
                work_dir=self.repo,
                tools=[lambda x: x],
                sock_path=self.sock_path,
            )
        assert not self.stub_calls, "no task may start for invalid tools"


class TestInProcessDaemonBootstrap(_ApiLaunchBase):
    """Launches without a socket start the process-global daemon."""

    def test_global_daemon_started_once_and_reused(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub(summary="global daemon ok")
        saved_override = launcher._SOCK_PATH_OVERRIDE
        launcher._SOCK_PATH_OVERRIDE = None
        try:
            result = run_agent_via_kiss_web(
                SlackAgent(),
                "first global task",
                work_dir=self.repo,
            )
            assert yaml.safe_load(result)["summary"] == "global daemon ok"
            assert launcher._API_SERVER is not None
            first_sock = launcher._API_SERVER_SOCK
            assert Path(first_sock).exists(), (
                "the in-process daemon must serve a real UDS"
            )
            server_before = launcher._API_SERVER
            result2 = run_agent_via_kiss_web(
                SlackAgent(),
                "second global task",
                work_dir=self.repo,
            )
            assert yaml.safe_load(result2)["summary"] == "global daemon ok"
            assert launcher._API_SERVER is server_before, (
                "the process-global daemon must be created exactly once"
            )
            assert launcher._API_SERVER_SOCK == first_sock
        finally:
            launcher._SOCK_PATH_OVERRIDE = saved_override


class TestCarrierAgentDirectRuns(_ApiLaunchBase):
    """Direct (non-launcher) runs of the carrier agents still record
    their result, preserving the pre-API contract for direct callers."""

    def test_chat_agent_direct_run_records_result(self) -> None:
        self._install_stub(summary="direct chat ok")
        agent = KissWebChatSorcarAgent("Direct Chat")
        result = agent.run(
            prompt_template="direct task", work_dir=self.repo,
        )
        assert yaml.safe_load(result)["summary"] == "direct chat ok"
        assert agent.last_run_result == result

    def test_chat_agent_direct_run_records_failure(self) -> None:
        self._install_stub(raise_exc=RuntimeError("direct-boom"))
        agent = KissWebChatSorcarAgent("Direct Chat")
        with self.assertRaises(RuntimeError):
            agent.run(prompt_template="direct task", work_dir=self.repo)
        parsed = yaml.safe_load(agent.last_run_result)
        assert parsed["success"] is False
        assert "direct-boom" in parsed["summary"]

    def test_chat_agent_direct_run_records_interrupt(self) -> None:
        self._install_stub(raise_exc=KeyboardInterrupt())
        agent = KissWebChatSorcarAgent("Direct Chat")
        with self.assertRaises(KeyboardInterrupt):
            agent.run(prompt_template="direct task", work_dir=self.repo)
        parsed = yaml.safe_load(agent.last_run_result)
        assert parsed["success"] is False
        assert "Task interrupted" in parsed["summary"]

    def test_worktree_agent_direct_run_records_result(self) -> None:
        self._install_stub(summary="direct wt ok")
        agent = KissWebWorktreeSorcarAgent("Direct WT")
        result = agent.run(
            prompt_template="direct task",
            work_dir=self.repo,
            use_worktree=False,
        )
        assert yaml.safe_load(result)["summary"] == "direct wt ok"
        assert agent.last_run_result == result

    def test_worktree_agent_direct_run_records_interrupt(self) -> None:
        # KeyboardInterrupt is a BaseException, so WorktreeSorcarAgent's
        # own ``except Exception`` fallback does NOT swallow it — the
        # carrier's except-branch must record the failure YAML before
        # re-raising.
        self._install_stub(raise_exc=KeyboardInterrupt())
        agent = KissWebWorktreeSorcarAgent("Direct WT")
        with self.assertRaises(KeyboardInterrupt):
            agent.run(
                prompt_template="direct task",
                work_dir=self.repo,
                use_worktree=False,
            )
        parsed = yaml.safe_load(agent.last_run_result)
        assert parsed["success"] is False
        assert "Task interrupted" in parsed["summary"]


class TestBaseChannelAgentDirectRuns(_ApiLaunchBase):
    """Direct (non-launcher) channel agent runs keep their contract."""

    def _plain_agent(self) -> Any:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            BaseChannelAgent,
        )

        class _Plain(BaseChannelAgent, SorcarAgent):
            def _is_authenticated(self) -> bool:
                return False

            def _get_auth_tools(self) -> list:
                return []

        return _Plain("Plain Direct Agent")

    def test_direct_run_appends_channel_prompt_to_system_prompt(
        self,
    ) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        result = agent.run(
            prompt_template="direct slack",
            work_dir=self.repo,
            use_worktree=True,  # chat-session-only kwarg must be popped
            _skip_persistence=True,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result
        system_prompt = str(
            self.stub_calls[0]["kwargs"].get("system_prompt", ""),
        )
        assert "Slack Authentication" in system_prompt

    def test_direct_run_without_channel_prompt(self) -> None:
        self._install_stub()
        agent = self._plain_agent()
        result = agent.run(
            prompt_template="direct plain", work_dir=self.repo,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        system_prompt = str(
            self.stub_calls[0]["kwargs"].get("system_prompt", ""),
        )
        assert "## Slack Authentication" not in system_prompt

    def test_direct_run_failure_recorded_and_reraised(self) -> None:
        self._install_stub(raise_exc=RuntimeError("plain-boom"))
        agent = self._plain_agent()
        with self.assertRaises(RuntimeError):
            agent.run(prompt_template="direct plain", work_dir=self.repo)
        parsed = yaml.safe_load(agent.last_run_result)
        assert parsed["success"] is False
        assert "plain-boom" in parsed["summary"]

    def test_direct_run_interrupt_recorded_and_reraised(self) -> None:
        self._install_stub(raise_exc=KeyboardInterrupt())
        agent = self._plain_agent()
        with self.assertRaises(KeyboardInterrupt):
            agent.run(prompt_template="direct plain", work_dir=self.repo)
        parsed = yaml.safe_load(agent.last_run_result)
        assert parsed["success"] is False
        assert "Task interrupted" in parsed["summary"]


class TestKissWebPollerAgents(_ApiLaunchBase):
    """The chat-id carrier agents used by the Slack pollers."""

    def test_chat_agent_gets_daemon_chat_id(self) -> None:
        self._install_stub()
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "poller task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result
        assert agent.chat_id, (
            "the daemon-minted chat id must be propagated onto the "
            "carrier agent so the poller can resume the thread later"
        )

    def test_chat_agent_resume_chat_by_id(self) -> None:
        self._install_stub()
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.new_chat()
        run_agent_via_kiss_web(
            agent,
            "first task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        first_chat = agent.chat_id
        assert first_chat

        prompts: list[str] = []

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            prompts.append(str(kwargs.get("prompt_template", "")))
            return "resumed fine"

        self._install_stub(on_run=on_run)
        resumed = KissWebChatSorcarAgent("Test Poller")
        resumed.resume_chat_by_id(first_chat)
        run_agent_via_kiss_web(
            resumed,
            "second task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert resumed.chat_id == first_chat, "existing chat id must be kept"
        # True continuation: the daemon persisted the second task under
        # the same chat and fed the first task as context.
        assert "first task" in prompts[0], (
            "the resumed run must see the prior task as chat context"
        )

    def test_worktree_agent_records_result(self) -> None:
        self._install_stub()
        agent = KissWebWorktreeSorcarAgent("Test WT Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result

    def test_slack_sorcar_poller_run_sorcar(self) -> None:
        from kiss.agents.third_party_agents import slack_sorcar_poller

        self._install_stub(summary="poller done")
        text, chat_id = slack_sorcar_poller._run_sorcar("do stuff", "")
        assert "poller done" in text
        assert chat_id, "a fresh chat id must be returned"
        assert isinstance(self.stub_calls[0]["agent"], ChatSorcarAgent)
        assert self.stub_calls[0]["agent"] is not None

    def test_slack_sorcar_poller_resumes_chat(self) -> None:
        from kiss.agents.third_party_agents import slack_sorcar_poller

        self._install_stub(summary="first")
        _text, chat_id = slack_sorcar_poller._run_sorcar("start", "")
        assert chat_id
        self._install_stub(summary="second")
        _text2, chat_id2 = slack_sorcar_poller._run_sorcar(
            "continue", chat_id,
        )
        assert chat_id2 == chat_id, "existing chat id must be reused"

    def test_slack_channel_poller_run_sorcar(self) -> None:
        from kiss.agents.third_party_agents import slack_channel_sorcar_poller

        self._install_stub(summary="wt poller done")
        text, chat_id = slack_channel_sorcar_poller._run_sorcar(
            "wt stuff", "",
        )
        assert "wt poller done" in text
        assert chat_id


class TestChannelRunnerViaApi(_ApiLaunchBase):
    """ChannelRunner._handle_message must launch through the API."""

    def _make_runner(self) -> Any:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            ChannelRunner,
        )

        outbox: list[tuple[str, str, str]] = []

        class _FakeBackend:
            def strip_bot_mention(self, text: str) -> str:
                return text

            def send_message(
                self,
                channel_id: str,
                text: str,
                thread_ts: str = "",
            ) -> None:
                outbox.append((channel_id, text, thread_ts))

            def disconnect(self) -> None:
                pass

        runner = ChannelRunner(
            backend=_FakeBackend(),
            channel_name="chan",
            agent_name="Test Channel Agent",
            work_dir=str(Path(self.tmpdir) / "chanwork"),
        )
        return runner, outbox

    def test_handle_message_bridges_reply_tool(self) -> None:
        runner, outbox = self._make_runner()

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            assert "reply" in tools, (
                "the per-message reply closure must be bridged as a tool"
            )
            out = tools["reply"](message="hello from the daemon agent")
            assert '"ok": true' in out
            return "replied myself"

        self._install_stub(on_run=on_run)
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        # The agent called reply() itself, so the runner must NOT also
        # send the summary.
        assert outbox == [("C123", "hello from the daemon agent", "1.0")]

    def test_handle_message_sends_summary_when_no_reply(self) -> None:
        runner, outbox = self._make_runner()
        self._install_stub(summary="channel summary")
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert outbox == [("C123", "channel summary", "1.0")]

    def test_handle_message_agent_error_sends_error_reply(self) -> None:
        runner, outbox = self._make_runner()
        self._install_stub(summary="chan-blast happened", success=False)
        runner._handle_message("C9", {"text": "x", "ts": "2.0"})
        # The daemon's failure summary is relayed as the reply.
        assert outbox, "an error reply must still be sent"
        channel, text, ts = outbox[0]
        assert channel == "C9" and ts == "2.0"
        assert "chan-blast" in text


class TestChannelMainInteractiveViaApi(_ApiLaunchBase):
    """channel_main's interactive (-t) mode must launch through the API."""

    def test_interactive_mode_uses_api(self) -> None:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            channel_main,
        )
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        orig_argv = sys.argv
        sys.argv = [
            "kiss-slack",
            "-t",
            "do the interactive thing",
            "-w",
            self.repo,
            "-b",
            "2.5",
            "-e",
            "http://localhost:7777/v1",
            "--header",
            "X-Test: yes",
            "--no-web",
            "--no-parallel",
        ]
        try:
            channel_main(SlackAgent, "kiss-slack", channel_name="Slack")
        finally:
            sys.argv = orig_argv

        assert self.stub_calls, "interactive channel_main never ran a task"
        call = self.stub_calls[0]
        assert call["agent"] is not None
        assert isinstance(call["agent"], ChatSorcarAgent)
        prompt = str(call["kwargs"].get("prompt_template", ""))
        assert "do the interactive thing" in prompt
        assert call["kwargs"].get("max_budget") == 2.5
        assert call["kwargs"].get("model_config") == {
            "base_url": "http://localhost:7777/v1",
            "extra_headers": {"X-Test": "yes"},
        }
        assert getattr(call["agent"], "_use_web_tools", None) is False
        assert getattr(call["agent"], "_is_parallel", None) is False


class TestNoDirectRunCallSites(unittest.TestCase):
    """Every third-party module must route runs through the launcher."""

    def test_no_direct_agent_run_calls_remain(self) -> None:
        tp_dir = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "third_party_agents"
        )
        offenders: list[str] = []
        for py in sorted(tp_dir.glob("*.py")):
            if py.name == "_kiss_web_launcher.py":
                continue
            source = py.read_text()
            # Use the AST so only executable code is scanned —
            # docstrings ("Example:: agent.run(...)") are usage
            # documentation, not launch call sites.
            tree = ast.parse(source, filename=str(py))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "run"):
                    continue
                # ``super().run(...)`` — the agent class delegating up
                # its own MRO — is fine.
                base = func.value
                if (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "super"
                ):
                    continue
                # Only flag ``<something named *agent*>.run(...)`` —
                # e.g. ``agent.run(...)`` / ``self._agent.run(...)``.
                name = ""
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute):
                    name = base.attr
                if "agent" in name.lower():
                    offenders.append(
                        f"{py.name}:{node.lineno}: "
                        + source.splitlines()[node.lineno - 1].strip()
                    )
        assert not offenders, (
            "third-party agents must launch via run_agent_via_kiss_web "
            "(kiss.server.sorcar.run), not agent.run() directly:\n"
            + "\n".join(offenders)
        )


if __name__ == "__main__":
    unittest.main()
