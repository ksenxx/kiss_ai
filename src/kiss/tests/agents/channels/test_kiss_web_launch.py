# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: third-party agents launch via ``_CommandsMixin._cmd_run``.

Feature under test
------------------
Every agent in ``kiss/agents/third_party_agents/`` must launch through
``run_agent_via_kiss_web`` (which drives ``VSCodeServer._cmd_run``)
instead of calling ``SorcarAgent.run`` directly.  Launching through
``_cmd_run`` registers the agent as a *kiss-web registered agent* — a
live entry in ``_RunningAgentState.running_agent_states`` — so remote
webviews can discover, open and interact with the running task.

Test strategy (no mocks)
------------------------
The innermost LLM-driving method — ``RelentlessAgent.run``
(``SorcarAgent.__mro__[1].run``) — is replaced with a stub that returns
a canned YAML result (precedent: ``test_bughunt_server_runner.py``).
Everything above it (BaseChannelAgent.run shim, ChatSorcarAgent.run,
the launcher, ``_cmd_run``, ``_run_task`` and the running-agent
registry) is the REAL production code path.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebChatSorcarAgent,
    KissWebWorktreeSorcarAgent,
    default_server,
    run_agent_via_kiss_web,
)
from kiss.agents.vscode.server import VSCodeServer

STUB_RESULT = "success: true\nsummary: stub summary done\n"


class _LaunchBase(unittest.TestCase):
    """Shared scaffolding: stub RelentlessAgent.run, capture broadcasts."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tp-launch-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.stub_calls: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _install_stub(
        self,
        result: str = STUB_RESULT,
        block: threading.Event | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append(
                {"agent": self_agent, "kwargs": kwargs, "thread": threading.current_thread()}
            )
            if block is not None:
                block.wait(timeout=30)
            if raise_exc is not None:
                raise raise_exc
            return result

        self._parent_class.run = stub_run


class TestLauncherRegistersAgent(_LaunchBase):
    """The launcher must register the agent while the task runs."""

    def test_agent_registered_during_run_and_disposed_after(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        release = threading.Event()
        started = threading.Event()
        seen: dict[str, Any] = {}

        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append({"agent": self_agent, "kwargs": kwargs})
            started.set()
            release.wait(timeout=30)
            return STUB_RESULT

        self._parent_class.run = stub_run

        agent = SlackAgent()
        out: dict[str, Any] = {}

        def launch() -> None:
            out["result"] = run_agent_via_kiss_web(
                agent,
                "hello slack task",
                work_dir=self.tmpdir,
                server=self.server,
            )

        t = threading.Thread(target=launch, daemon=True)
        t.start()
        assert started.wait(timeout=30), "stubbed agent run never started"

        # While the task runs, the agent must be a kiss-web registered
        # agent: a registry entry whose ``agent`` IS our instance.
        with _RunningAgentState._registry_lock:
            entries = {
                tid: st
                for tid, st in _RunningAgentState.running_agent_states.items()
                if st.agent is agent
            }
        assert len(entries) == 1, (
            "launched third-party agent must appear exactly once in "
            "_RunningAgentState.running_agent_states while running"
        )
        tab_id, state = next(iter(entries.items()))
        seen["tab_id"] = tab_id
        assert state.is_task_active, "registered state must be task-active"
        assert state.task_thread is not None and state.task_thread.is_alive()
        assert isinstance(self.server._get_running_task_ids(), set), (
            "kiss-web running-task scans must tolerate plain "
            "SorcarAgent-based third-party agents that have not "
            "allocated a chat task id"
        )

        release.set()
        t.join(timeout=30)
        assert not t.is_alive(), "launcher never returned"

        # The run went through _cmd_run's worker thread, not the
        # caller's thread.
        assert self.stub_calls, "stub was never invoked"
        assert out["result"] == STUB_RESULT

        # After the launcher returns the registry entry is disposed.
        with _RunningAgentState._registry_lock:
            assert seen["tab_id"] not in (_RunningAgentState.running_agent_states), (
                "registry entry must be disposed after the task completes"
            )

    def test_run_executes_on_worker_thread(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert self.stub_calls
        assert self.stub_calls[0]["thread"] is not threading.main_thread(), (
            "_cmd_run must run the agent on a background worker thread"
        )

    def test_clear_event_broadcast_on_launch(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        clear_events = [e for e in self.events if e.get("type") == "clear"]
        assert clear_events, "_cmd_run must broadcast the initial clear event"


class TestLauncherParameters(_LaunchBase):
    """Branch coverage for the launcher's parameters."""

    def test_model_name_forwarded(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            model_name="gpt-5.5",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert self.stub_calls[0]["kwargs"].get("model_name") == "gpt-5.5"

    def test_empty_model_uses_tab_default(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        # tab.selected_model (the server default) is used when the
        # caller passes no model.
        assert self.stub_calls[0]["kwargs"].get("model_name")

    def test_max_budget_override_reaches_agent(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            max_budget=1.25,
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert self.stub_calls[0]["kwargs"].get("max_budget") == 1.25

    def test_no_budget_uses_config_default(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        budget = self.stub_calls[0]["kwargs"].get("max_budget")
        assert budget is not None and budget > 0

    def test_zero_budget_override_is_honored(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            max_budget=0.0,
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert self.stub_calls[0]["kwargs"].get("max_budget") == 0.0

    def test_budget_override_does_not_leak_between_reuses(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        run_agent_via_kiss_web(
            agent,
            "first",
            max_budget=1.25,
            work_dir=self.tmpdir,
            server=self.server,
        )
        run_agent_via_kiss_web(
            agent,
            "second",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert self.stub_calls[0]["kwargs"].get("max_budget") == 1.25
        assert self.stub_calls[1]["kwargs"].get("max_budget") != 1.25

    def test_model_config_web_and_parallel_overrides_forward(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.tmpdir,
            server=self.server,
            model_config={"base_url": "http://localhost:9999/v1"},
            web_tools=False,
            is_parallel=True,
        )
        kwargs = self.stub_calls[0]["kwargs"]
        assert kwargs.get("model_config") == {
            "base_url": "http://localhost:9999/v1",
        }
        assert getattr(agent, "_use_web_tools", None) is False
        assert getattr(agent, "_is_parallel", None) is True

    def test_extra_tools_merged_into_run(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()

        def mytool(x: str) -> str:
            """Test tool."""
            return x

        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            tools=[mytool],
            work_dir=self.tmpdir,
            server=self.server,
        )
        tools = self.stub_calls[0]["kwargs"].get("tools") or []
        assert any(getattr(t, "__name__", "") == "mytool" for t in tools), (
            "extra tools passed to the launcher must reach the agent run"
        )

    def test_no_tools_ok(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        result = run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert result == STUB_RESULT

    def test_base_channel_agent_preserves_positional_direct_calls(self) -> None:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            BaseChannelAgent,
        )

        class _PositionalAgent(BaseChannelAgent, SorcarAgent):
            def _is_authenticated(self) -> bool:
                return False

            def _get_auth_tools(self) -> list:
                return []

        self._install_stub()
        result = _PositionalAgent("positional test").run(
            "model-pos",
            "prompt-pos",
        )
        assert result == STUB_RESULT
        assert self.stub_calls[0]["kwargs"].get("model_name") == "model-pos"
        assert "prompt-pos" in str(
            self.stub_calls[0]["kwargs"].get("prompt_template", ""),
        )

    def test_prompt_reaches_agent(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "my unique prompt text",
            work_dir=self.tmpdir,
            server=self.server,
        )
        prompt = str(self.stub_calls[0]["kwargs"].get("prompt_template", ""))
        assert "my unique prompt text" in prompt

    def test_default_server_lazily_created_and_cached(self) -> None:
        s1 = default_server()
        s2 = default_server()
        assert s1 is s2, "default_server must cache one process-global server"
        assert isinstance(s1, VSCodeServer)

    def test_agent_failure_returns_failure_yaml(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub(raise_exc=RuntimeError("boom-fail"))
        result = run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        # The agent raised: the launcher must not hang or raise, and
        # it must return a failure YAML instead of an empty string so
        # channel runners / pollers can surface the error to users.
        assert "success: false" in result.lower()
        assert "boom-fail" in result, f"unexpected failure result {result!r}"
        # And the registry entry must still be disposed.
        with _RunningAgentState._registry_lock:
            assert not any(tid.startswith("tp-") for tid in _RunningAgentState.running_agent_states)

    def test_cmd_run_failure_disposes_registered_state(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        def boom(_event: dict[str, Any]) -> None:
            raise RuntimeError("clear-broadcast-boom")

        self.server.printer.broadcast = boom  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            run_agent_via_kiss_web(
                SlackAgent(),
                "task",
                work_dir=self.tmpdir,
                server=self.server,
            )
        with _RunningAgentState._registry_lock:
            assert not any(
                tid.startswith("tp-") for tid in _RunningAgentState.running_agent_states
            ), "_cmd_run startup failures must not leak launcher states"

    def test_timeout_returns_and_state_survives(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        release = threading.Event()
        self._install_stub(block=release)
        try:
            result = run_agent_via_kiss_web(
                SlackAgent(),
                "task",
                work_dir=self.tmpdir,
                server=self.server,
                timeout=0.3,
            )
            assert result == "", "timed-out launch must return empty result"
        finally:
            release.set()
            # Let the worker finish so teardown can clean the registry.
            deadline = time.time() + 30
            while time.time() < deadline:
                with _RunningAgentState._registry_lock:
                    threads = [
                        st.task_thread
                        for st in _RunningAgentState.running_agent_states.values()
                        if st.task_thread is not None
                    ]
                if not threads:
                    break
                for th in threads:
                    th.join(timeout=1)


class TestKissWebPollerAgents(_LaunchBase):
    """The chat-capable launcher agents used by the Slack pollers."""

    def test_chat_agent_records_result_and_mints_chat_id(self) -> None:
        self._install_stub()
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "poller task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert result == STUB_RESULT
        assert agent.last_run_result == STUB_RESULT
        assert agent.chat_id, (
            "a chat id must be minted by the run so the poller can resume the thread later"
        )

    def test_chat_agent_resume_chat_by_id(self) -> None:
        self._install_stub()
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.resume_chat_by_id("existing-chat-123")
        run_agent_via_kiss_web(
            agent,
            "resume task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert agent.chat_id == "existing-chat-123"

    def test_worktree_agent_records_result(self) -> None:
        self._install_stub()
        agent = KissWebWorktreeSorcarAgent("Test WT Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert result == STUB_RESULT
        assert agent.last_run_result == STUB_RESULT

    def test_chat_agent_failure_records_failure_result(self) -> None:
        self._install_stub(raise_exc=RuntimeError("chat-boom"))
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "poller task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert "success: false" in result.lower()
        assert "chat-boom" in result
        assert agent.last_run_result == result

    def test_chat_agent_keyboard_interrupt_records_interrupted_result(self) -> None:
        self._install_stub(raise_exc=KeyboardInterrupt())
        agent = KissWebChatSorcarAgent("Test Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "poller task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert "Task interrupted" in result
        assert agent.last_run_result == result

    def test_worktree_agent_keyboard_interrupt_records_interrupted_result(self) -> None:
        # KeyboardInterrupt is a BaseException, so WorktreeSorcarAgent's
        # own ``except Exception`` fallback does NOT swallow it — it
        # propagates to KissWebWorktreeSorcarAgent.run's except-branch,
        # which must record the failure YAML before re-raising.
        self._install_stub(raise_exc=KeyboardInterrupt())
        agent = KissWebWorktreeSorcarAgent("Test WT Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert "Task interrupted" in result
        assert agent.last_run_result == result

    def test_worktree_agent_failure_records_failure_result(self) -> None:
        self._install_stub(raise_exc=RuntimeError("wt-boom"))
        agent = KissWebWorktreeSorcarAgent("Test WT Poller")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.tmpdir,
            server=self.server,
        )
        assert "success" in result.lower() and "false" in result.lower()
        assert "wt-boom" in result
        assert agent.last_run_result == result

    def test_worktree_agent_is_chat_sorcar_subclass(self) -> None:
        assert issubclass(KissWebWorktreeSorcarAgent, ChatSorcarAgent)
        assert issubclass(KissWebChatSorcarAgent, ChatSorcarAgent)


class TestChannelRunnerViaCmdRun(_LaunchBase):
    """ChannelRunner._handle_message must launch through _cmd_run."""

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

    def test_handle_message_registers_and_replies(self) -> None:
        runner, outbox = self._make_runner()

        registry_during_run: list[bool] = []
        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append({"agent": self_agent, "kwargs": kwargs})
            with _RunningAgentState._registry_lock:
                registry_during_run.append(
                    any(
                        st.agent is self_agent
                        for st in _RunningAgentState.running_agent_states.values()
                    )
                )
            return STUB_RESULT

        self._parent_class.run = stub_run

        import kiss.agents.third_party_agents._kiss_web_launcher as launcher

        orig = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        try:
            runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        finally:
            launcher._DEFAULT_SERVER = orig

        assert calls, "agent run was never invoked"
        assert registry_during_run == [True], (
            "channel-runner agent must be kiss-web registered during run"
        )
        # The reply tool must be present in the tools the agent saw.
        tools = calls[0]["kwargs"].get("tools") or []
        assert any(getattr(t, "__name__", "") == "reply" for t in tools), (
            "the per-message reply tool must be merged into the run tools"
        )
        # The stub never called reply, so the summary is sent.
        assert outbox == [("C123", "stub summary done", "1.0")]

    def test_handle_message_agent_error_sends_error_reply(self) -> None:
        runner, outbox = self._make_runner()
        self._install_stub(raise_exc=RuntimeError("chan-blast"))

        import kiss.agents.third_party_agents._kiss_web_launcher as launcher

        orig = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        try:
            runner._handle_message("C9", {"text": "x", "ts": "2.0"})
        finally:
            launcher._DEFAULT_SERVER = orig
        assert outbox, "an error reply must still be sent"
        channel, text, ts = outbox[0]
        assert channel == "C9" and ts == "2.0"
        assert "chan-blast" in text


class TestChannelMainInteractiveViaCmdRun(_LaunchBase):
    """channel_main's interactive (-t) mode must launch through _cmd_run."""

    def test_interactive_mode_uses_kiss_web(self) -> None:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            channel_main,
        )
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        registry_during_run: list[bool] = []
        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append({"agent": self_agent, "kwargs": kwargs})
            with _RunningAgentState._registry_lock:
                registry_during_run.append(
                    any(
                        st.agent is self_agent
                        for st in _RunningAgentState.running_agent_states.values()
                    )
                )
            return STUB_RESULT

        self._parent_class.run = stub_run

        import kiss.agents.third_party_agents._kiss_web_launcher as launcher

        orig_argv = sys.argv
        orig_server = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        sys.argv = [
            "kiss-slack",
            "-t",
            "do the interactive thing",
            "-w",
            self.tmpdir,
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
            launcher._DEFAULT_SERVER = orig_server

        assert calls, "interactive channel_main never ran the agent"
        assert isinstance(calls[0]["agent"], SlackAgent)
        assert registry_during_run == [True], (
            "interactive agent must be kiss-web registered during run"
        )
        prompt = str(calls[0]["kwargs"].get("prompt_template", ""))
        assert "do the interactive thing" in prompt
        assert calls[0]["kwargs"].get("max_budget") == 2.5
        assert calls[0]["kwargs"].get("model_config") == {
            "base_url": "http://localhost:7777/v1",
            "extra_headers": {"X-Test": "yes"},
        }
        assert getattr(calls[0]["agent"], "_use_web_tools", None) is False
        assert getattr(calls[0]["agent"], "_is_parallel", None) is False


class TestSlackPollersViaCmdRun(_LaunchBase):
    """Both slack pollers' _run_sorcar must launch through _cmd_run."""

    def test_slack_sorcar_poller_run_sorcar(self) -> None:
        import kiss.agents.third_party_agents._kiss_web_launcher as launcher
        from kiss.agents.third_party_agents import slack_sorcar_poller

        registry_during_run: list[bool] = []
        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append({"agent": self_agent, "kwargs": kwargs})
            with _RunningAgentState._registry_lock:
                registry_during_run.append(
                    any(
                        st.agent is self_agent
                        for st in _RunningAgentState.running_agent_states.values()
                    )
                )
            return "success: true\nsummary: poller done\n"

        self._parent_class.run = stub_run

        orig = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        try:
            text, chat_id = slack_sorcar_poller._run_sorcar("do stuff", "")
        finally:
            launcher._DEFAULT_SERVER = orig

        assert registry_during_run == [True], "poller agent must be kiss-web registered during run"
        assert "poller done" in text
        assert chat_id, "a fresh chat id must be returned"
        assert isinstance(calls[0]["agent"], ChatSorcarAgent)

    def test_slack_sorcar_poller_resumes_chat(self) -> None:
        import kiss.agents.third_party_agents._kiss_web_launcher as launcher
        from kiss.agents.third_party_agents import slack_sorcar_poller

        self._install_stub()
        orig = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        try:
            _text, chat_id = slack_sorcar_poller._run_sorcar(
                "continue",
                "chat-xyz",
            )
        finally:
            launcher._DEFAULT_SERVER = orig
        assert chat_id == "chat-xyz", "existing chat id must be reused"

    def test_slack_channel_poller_run_sorcar(self) -> None:
        import kiss.agents.third_party_agents._kiss_web_launcher as launcher
        from kiss.agents.third_party_agents import slack_channel_sorcar_poller

        registry_during_run: list[bool] = []
        calls = self.stub_calls

        def stub_run(self_agent: object, **kwargs: object) -> str:
            calls.append({"agent": self_agent, "kwargs": kwargs})
            with _RunningAgentState._registry_lock:
                registry_during_run.append(
                    any(
                        st.agent is self_agent
                        for st in _RunningAgentState.running_agent_states.values()
                    )
                )
            return "success: true\nsummary: wt poller done\n"

        self._parent_class.run = stub_run

        orig = launcher._DEFAULT_SERVER
        launcher._DEFAULT_SERVER = self.server
        try:
            text, chat_id = slack_channel_sorcar_poller._run_sorcar(
                "wt stuff",
                "",
            )
        finally:
            launcher._DEFAULT_SERVER = orig

        assert registry_during_run == [True]
        assert "wt poller done" in text
        assert chat_id


class TestNoDirectRunCallSites(unittest.TestCase):
    """Every third-party module must route runs through the launcher."""

    def test_no_direct_agent_run_calls_remain(self) -> None:
        tp_dir = Path(__file__).resolve().parents[3] / "agents" / "third_party_agents"
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
                        f"{py.name}:{node.lineno}: " + source.splitlines()[node.lineno - 1].strip()
                    )
        assert not offenders, (
            "third-party agents must launch via run_agent_via_kiss_web "
            "(_cmd_run), not agent.run() directly:\n" + "\n".join(offenders)
        )


if __name__ == "__main__":
    unittest.main()
