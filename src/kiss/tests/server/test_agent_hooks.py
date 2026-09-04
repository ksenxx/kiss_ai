# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for agent-script ``get_llm_call_hook``/``get_tool_call_hook``.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
temporary Unix-domain socket (the :class:`DaemonRunApiHarness` from
``test_append_basic_tools``) and drive ``kiss.server.sorcar.run``
against it with an ``extension_agent_path`` agent script.  The only
replaced boundary is the LLM itself: the per-session executor's
:meth:`kiss.core.kiss_agent.KISSAgent.run` is swapped for a stub that
records the ``llm_call_hook`` / ``tool_call_hook`` it was handed, so
the daemon's full run pipeline — ``run`` command dispatch → worker
thread → ``apply_agent_overrides`` hook staging →
``WorktreeSorcarAgent.run`` → ``SorcarAgent.run`` →
``RelentlessAgent.perform_task`` → ``KISSAgent.run`` — executes for
real without any model API calls.

Contract under test: an agent script's ``get_llm_call_hook()`` /
``get_tool_call_hook()`` return the ``llm_call_hook`` /
``tool_call_hook`` functions the underlying :class:`KISSAgent` receives;
without them (or with a getter returning ``None``) the executor
receives ``None``; a wrong-typed getter result stops the task loudly;
and a non-callable hook field arriving over the wire is ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.server import sorcar
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness


class AgentScriptHooksApiTest(DaemonRunApiHarness):
    """Agent-script hook getters must reach the underlying ``KISSAgent``."""

    def _executor_call(
        self, calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the single task-executor ``KISSAgent.run`` record.

        Args:
            calls: The record list filled by the executor stub.

        Returns:
            The one recorded call whose arguments carry
            ``task_description`` — the executor session
            ``RelentlessAgent.perform_task`` spawned for the task.
        """
        executor_calls = [
            c for c in calls if "task_description" in c["arguments"]
        ]
        assert len(executor_calls) == 1, calls
        return executor_calls[0]

    def _write_hooks_agent(self) -> str:
        """Write an agent script defining both hook getters.

        The hooks stamp marker files under the test tmpdir when
        invoked, so the test can prove the recorded callables are the
        script's own functions executing in the daemon process.

        Returns:
            The absolute path of the written agent script.
        """
        return self._write_py(
            "hooks_agent.py",
            f'''
            """Agent script installing LLM-call and tool-call hooks."""

            from pathlib import Path

            MARKER_DIR = Path(r"{self.tmpdir}")


            def llm_call_hook(new_messages):
                """Stamp a marker and append a message to the batch.

                Args:
                    new_messages: Messages about to be sent to the LLM.
                """
                marker = MARKER_DIR / "llm_hook_called.txt"
                marker.write_text(str(len(new_messages)))
                return [*new_messages, dict(role="user", content="hooked")]


            def tool_call_hook(name, args):
                """Stamp a marker; allow finish, veto everything else.

                Args:
                    name: Tool name about to be called.
                    args: The tool call's arguments dict.
                """
                marker = MARKER_DIR / "tool_hook_called.txt"
                marker.write_text(name)
                return "OK" if name == "finish" else "blocked by hook"


            def get_llm_call_hook():
                """Return the LLM-call hook."""
                return llm_call_hook


            def get_tool_call_hook():
                """Return the tool-call hook."""
                return tool_call_hook
            ''',
        )

    def test_agent_script_hooks_reach_executor(self) -> None:
        """Both hook getters' functions reach ``KISSAgent.run`` unchanged.

        The recorded callables must be the agent script's own
        ``llm_call_hook`` / ``tool_call_hook``: invoking them performs
        the script-defined behavior (message rewrite, tool veto) and
        stamps the script's marker files.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with agent-script hooks",
            work_dir=self.repo,
            extension_agent_path=self._write_hooks_agent(),
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        call = self._executor_call(calls)
        llm_hook = call["llm_call_hook"]
        tool_hook = call["tool_call_hook"]
        assert callable(llm_hook), call
        assert callable(tool_hook), call

        rewritten = llm_hook([{"role": "user", "content": "hi"}])
        assert isinstance(rewritten, list), rewritten
        assert rewritten[-1] == {"role": "user", "content": "hooked"}
        assert (
            Path(self.tmpdir) / "llm_hook_called.txt"
        ).read_text() == "1"

        assert tool_hook("finish", {}) == "OK"
        assert tool_hook("Bash", {"command": "ls"}) == "blocked by hook"
        assert (
            Path(self.tmpdir) / "tool_hook_called.txt"
        ).read_text() == "Bash"

    def test_no_agent_script_passes_none_hooks(self) -> None:
        """Without an agent script the executor receives no hooks."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task without hooks",
            work_dir=self.repo,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        call = self._executor_call(calls)
        assert call["llm_call_hook"] is None
        assert call["tool_call_hook"] is None

    def test_hook_getter_returning_none_passes_none(self) -> None:
        """A hook getter may return ``None``, meaning "no hook".

        The script defines both getters; only ``get_llm_call_hook()``
        returns a callable — the executor must get that callable and a
        ``None`` tool-call hook.
        """
        agent_path = self._write_py(
            "half_hooks_agent.py",
            '''
            """Agent script with one real hook and one None hook."""


            def _reverse_messages(new_messages):
                """Reverse the batch of new messages.

                Args:
                    new_messages: Messages about to be sent to the LLM.
                """
                return list(reversed(new_messages))


            def get_llm_call_hook():
                """Return the LLM-call hook."""
                return _reverse_messages


            def get_tool_call_hook():
                """Install no tool-call hook."""
                return None
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a None tool hook",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        call = self._executor_call(calls)
        assert callable(call["llm_call_hook"]), call
        assert call["llm_call_hook"]([1, 2, 3]) == [3, 2, 1]
        assert call["tool_call_hook"] is None

    def test_wrong_typed_hook_getter_fails_task(self) -> None:
        """A non-callable ``get_tool_call_hook()`` result stops the task."""
        agent_path = self._write_py(
            "bad_hook_agent.py",
            '''
            """Agent script with a wrong-typed hook getter."""


            def get_tool_call_hook() -> int:
                """Return the wrong type."""
                return 42
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a broken hook getter",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False
        assert "get_tool_call_hook" in result.text
        assert "a callable or None" in result.text
        assert calls == [], "no executor session may start for a broken script"

    def test_wire_non_callable_hook_fields_ignored(self) -> None:
        """Hook fields sent as JSON values over the wire mean "no hook".

        The hook command fields are daemon-internal (a callable cannot
        be JSON-serialized), but ``validate_command`` lets unknown
        extra fields through — a buggy or malicious client can send
        them as plain JSON values.  The task must run normally with no
        hooks instead of crashing the executor.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run(
            {"llmCallHook": "evil-string", "toolCallHook": 123},
            events_out=events,
        )
        call = self._executor_call(calls)
        assert call["llm_call_hook"] is None
        assert call["tool_call_hook"] is None
        results = [e for e in events if e.get("type") == "result"]
        assert results and results[-1].get("success") is True, events
