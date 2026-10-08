# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for SEA ``llm_call_hook``/``tool_call_hook``.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` with a
temporary local endpoint file (the :class:`DaemonRunApiHarness` from
``test_append_basic_tools``) and drive ``kiss.server.sorcar.run``
against it with an ``sea_path`` SEA.  The only
replaced boundary is the LLM itself: the per-session executor's
:meth:`kiss.core.kiss_agent.KISSAgent.run` is swapped for a stub that
records the ``llm_call_hook`` / ``tool_call_hook`` it was handed, so
the daemon's full run pipeline — ``run`` command dispatch → worker
thread → ``apply_sea`` hook staging →
``WorktreeSorcarAgent.run`` → ``SorcarAgent.run`` →
``RelentlessAgent.perform_task`` → ``KISSAgent.run`` — executes for
real without any model API calls.

Contract under test: the ``llm_call_hook`` / ``tool_call_hook`` methods
of a SEA's ``BaseSea`` subclass become the ``llm_call_hook`` /
``tool_call_hook`` callables the underlying :class:`KISSAgent` receives
(``evaluate_sea`` wraps the ``base_*`` folds over the SEA chain, which
starts at ``BaseSea``, in a ``functools.partial``); the executor ALWAYS
gets callables — a hook no SEA overrides is the identity (same
messages, ``ALLOW``); a ``tool_call_hook`` that is not a method stops
the task loudly before any executor session starts; and a non-callable hook
field arriving over the wire is overwritten by the staged callable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.core.tool_verdict import ALLOW, refuse
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
        """Write a SEA whose SEA overrides both hook methods.

        The hooks stamp marker files under the test tmpdir when
        invoked, so the test can prove the recorded callables run the
        script's own methods in the daemon process.

        Returns:
            The absolute path of the written SEA.
        """
        return self._write_py(
            "hooks_agent.py",
            f'''
            """SEA installing LLM-call and tool-call hooks."""

            from pathlib import Path

            from kiss.agents.seas.base.base_sea import ALLOW, BaseSea, refuse

            MARKER_DIR = Path(r"{self.tmpdir}")


            class Sea(BaseSea):
                def llm_call_hook(self, new_messages):
                    """Stamp a marker and append a message to the batch.

                    Args:
                        new_messages: Messages about to be sent to the LLM.
                    """
                    marker = MARKER_DIR / "llm_hook_called.txt"
                    marker.write_text(str(len(new_messages)))
                    return [*new_messages, dict(role="user", content="hooked")]

                def tool_call_hook(self, name, args):
                    """Stamp a marker; allow finish, veto everything else.

                    Args:
                        name: Tool name about to be called.
                        args: The tool call's arguments dict.
                    """
                    marker = MARKER_DIR / "tool_hook_called.txt"
                    marker.write_text(name)
                    return ALLOW if name == "finish" else refuse("blocked by hook")
            ''',
        )

    def test_agent_script_hooks_reach_executor(self) -> None:
        """Both hook methods reach ``KISSAgent.run`` as callables.

        The recorded callables must run the SEA's own
        ``llm_call_hook`` / ``tool_call_hook`` methods: invoking them
        performs the script-defined behavior (message rewrite, tool
        veto) and stamps the script's marker files.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with SEA hooks",
            work_dir=self.repo,
            sea_path=self._write_hooks_agent(),
            use_worktree=False,
            endpoint_file=self.endpoint_file,
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

        assert tool_hook("finish", {}) == ALLOW
        assert tool_hook("Bash", {"command": "ls"}) == refuse("blocked by hook")
        assert (
            Path(self.tmpdir) / "tool_hook_called.txt"
        ).read_text() == "Bash"

    def _assert_identity_hooks(self, call: dict[str, Any]) -> None:
        """Assert the executor call carries the bare ``BaseSea`` identity hooks.

        Args:
            call: The recorded executor ``KISSAgent.run`` call.
        """
        assert callable(call["llm_call_hook"]), call
        assert callable(call["tool_call_hook"]), call
        messages = [{"role": "user", "content": "hi"}]
        assert call["llm_call_hook"](messages) == messages
        assert call["tool_call_hook"]("Bash", {"command": "ls"}) == ALLOW
        assert call["tool_call_hook"]("finish", {}) == ALLOW

    def test_no_agent_script_passes_identity_hooks(self) -> None:
        """Without a SEA the executor receives ``BaseSea``'s identity hooks.

        Every run goes through ``base_sea.py``: the hooks are always
        callables, and with nothing overriding them they leave the
        messages alone and allow every tool call.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task without hooks",
            work_dir=self.repo,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        self._assert_identity_hooks(self._executor_call(calls))

    def test_hook_not_overridden_is_the_identity(self) -> None:
        """A SEA that overrides only one hook gets the identity for the other.

        The script overrides ``llm_call_hook`` and leaves
        ``tool_call_hook`` to the do-nothing default of ``BaseSea`` —
        the executor must get a callable LLM hook running the script's
        method and a tool-call hook answering ``None`` to everything.
        """
        sea_path = self._write_py(
            "half_hooks_agent.py",
            '''
            """SEA with one real hook and the default for the other."""

            from kiss.agents.seas.base.base_sea import BaseSea


            class Sea(BaseSea):
                def llm_call_hook(self, new_messages):
                    """Reverse the batch of new messages.

                    Args:
                        new_messages: Messages about to be sent to the LLM.
                    """
                    return list(reversed(new_messages))
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a default tool hook",
            work_dir=self.repo,
            sea_path=sea_path,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        call = self._executor_call(calls)
        assert callable(call["llm_call_hook"]), call
        assert call["llm_call_hook"]([1, 2, 3]) == [3, 2, 1]
        assert callable(call["tool_call_hook"]), call
        assert call["tool_call_hook"]("Bash", {"command": "ls"}) == ALLOW

    def test_wrong_typed_hook_getter_fails_task(self) -> None:
        """A ``tool_call_hook`` that is not a method stops the task at staging.

        ``evaluate_sea`` walks every hook's chain before the run
        starts, so the ``must be a method`` diagnostic (naming the
        method and the script) fails the task before any executor
        session exists.
        """
        sea_path = self._write_py(
            "bad_hook_agent.py",
            '''
            """SEA whose tool_call_hook is a plain attribute."""

            from kiss.agents.seas.base.base_sea import BaseSea


            class Sea(BaseSea):
                tool_call_hook = 42
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with a broken hook getter",
            work_dir=self.repo,
            sea_path=sea_path,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is False
        assert "tool_call_hook of SEA '" in result.text, result.text
        assert "bad_hook_agent.py' must be a method, got int" in result.text, result.text
        assert calls == [], "no executor session may start for a broken script"

    def test_wire_non_callable_hook_fields_overwritten(self) -> None:
        """Hook fields sent as JSON values over the wire are replaced by the staged hooks.

        The hook command fields are daemon-internal (a callable cannot
        be JSON-serialized), but ``validate_command`` lets unknown
        extra fields through — a buggy or malicious client can send
        them as plain JSON values.  ``apply_sea`` writes
        the ``BaseSea`` identity hooks over them, so the task runs
        normally with callable hooks instead of crashing the executor.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run(
            {"llmCallHook": "evil-string", "toolCallHook": 123},
            events_out=events,
        )
        call = self._executor_call(calls)
        assert call["llm_call_hook"] != "evil-string"
        assert call["tool_call_hook"] != 123
        self._assert_identity_hooks(call)
        results = [e for e in events if e.get("type") == "result"]
        assert results and results[-1].get("success") is True, events
