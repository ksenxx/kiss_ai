# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.server.sorcar.run``'s ``append_basic_tools``.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
loopback local endpoint and drive ``kiss.server.sorcar.run``
against it.  The only replaced boundary is the LLM itself: the
per-session executor's :meth:`kiss.core.kiss_agent.KISSAgent.run` is
swapped for a stub that records the ``tools`` it was handed, so the
daemon's full run pipeline — ``run`` command dispatch → worker thread →
agent-script overrides → ``WorktreeSorcarAgent.run`` →
``SorcarAgent.perform_task`` tool assembly →
``RelentlessAgent.perform_task``'s ``finish`` prepend — executes for
real without any model API calls.

Contract under test: ``append_basic_tools`` defaults to ``True`` (the
agent gets the built-in basic toolset on top of the caller's tools);
when ``False`` the agent's ONLY tools are ``finish`` and the tools the
client's tools file's ``get_tools()`` returned.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import textwrap
import threading
import unittest
import uuid
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar import persistence as _persistence
from kiss.core import vscode_config
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.test_agent_path import _init_repo


class DaemonRunApiHarness(unittest.TestCase):
    """Shared real-daemon harness for ``sorcar.run`` end-to-end suites.

    Owns the temporary repo + local daemon lifecycle, the recording
    executor-LLM stub, and the raw local-connection ``run`` driver.  Test suites
    (this module's ``append_basic_tools`` tests, the
    ``append_to_system_prompt``/``append_to_prompt`` tests in
    ``test_append_to_prompts``) subclass it and add only test methods.
    """

    def setUp(self) -> None:
        # Resolved: macOS mkdtemp returns a symlinked /var/... path while
        # the worktree machinery canonicalizes the repo, so un-resolved
        # paths break startswith checks.
        self.tmpdir = str(
            Path(tempfile.mkdtemp(prefix="sorcar_append_basic_")).resolve()
        )
        self.endpoint_file = str(Path(self.tmpdir) / "sorcar-local.json")
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
        _persistence._DB_PATH = kiss_dir / "history.db"
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
            local_endpoint_file=self.endpoint_file, work_dir=self.repo,
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)

        self._original_executor_run = KISSAgent.run

    def tearDown(self) -> None:
        cast(Any, KISSAgent).run = self._original_executor_run
        from kiss.server import agent_state

        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
        agent_state.agent_states.clear()

        async def _shutdown() -> None:
            ws_server = self.server._ws_server
            if ws_server is not None:
                ws_server.close()
                await ws_server.wait_closed()
            local_endpoint.remove_endpoint_if_owned(
                self.server._local_endpoint_file, self.server._local_token,
            )
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

    def _write_py(self, name: str, content: str) -> str:
        """Write a Python file under the test tmpdir and return its path.

        Args:
            name: File name (e.g. ``"my_tools.py"``).
            content: Python source for the file.

        Returns:
            The absolute path of the written file.
        """
        path = Path(self.tmpdir) / name
        path.write_text(textwrap.dedent(content))
        return str(path)

    def _install_executor_stub(
        self,
        calls: list[dict[str, Any]],
        fail_first_executor: bool = False,
    ) -> None:
        """Swap the executor LLM loop for a stub recording its tool names.

        ``RelentlessAgent.perform_task`` creates a fresh
        :class:`KISSAgent` per sub-session and hands it the fully
        assembled ``tools`` list — the exact list this suite must
        assert on — so the stub replaces that executor's ``run`` and
        everything above it (``SorcarAgent.perform_task`` included)
        executes for real.

        Args:
            calls: List receiving, per agentic ``KISSAgent.run``
                invocation, a dict with the ``tool_names`` handed to
                the session, the call's ``arguments`` (the
                task-executor calls carry ``task_description``, the
                failure-path trajectory summarizer carries
                ``trajectory_path``), the call's ``system_prompt``,
                and the ``llm_call_hook`` / ``tool_call_hook``
                callables handed to the session (``None`` when the
                run carries no agent-script hook).
            fail_first_executor: When True, the FIRST task-executor
                session raises a retryable :class:`KISSError` after
                two steps, driving ``RelentlessAgent.perform_task``
                into its failed-session summarizer branch before the
                second session succeeds.
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                # The daemon runs one-shot non-agentic ``KISSAgent``
                # sessions off the task thread — e.g. the follow-up
                # proposer AFTER the task's status ended.  Answer them
                # with the empty-fallback response WITHOUT touching the
                # recording list: a non-empty answer would make the
                # proposer's daemon thread persist chat events into the
                # test's SQLite database while ``tearDown`` is closing
                # it and deleting the temp tree (a reproducible native
                # crash), and recording would let a straggler thread
                # contaminate a later test's ``calls`` list.
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            tools = kwargs.get("tools") or []
            calls.append({
                "tool_names": [
                    getattr(t, "__name__", "?") for t in tools
                ],
                "arguments": arguments,
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "llm_call_hook": kwargs.get("llm_call_hook"),
                "tool_call_hook": kwargs.get("tool_call_hook"),
            })
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            if "task_description" not in arguments:
                # The failed-session trajectory summarizer.
                self_agent.step_count = 1
                return "result: prior progress\n"
            executor_index = sum(
                1 for c in calls if "task_description" in c["arguments"]
            )
            if fail_first_executor and executor_index == 1:
                # Retryable failure: a plain KISSError (no cause) after
                # more than one step continues into the next
                # sub-session instead of aborting the task.
                self_agent.step_count = 2
                raise KISSError("injected retryable failure")
            self_agent.step_count = 1
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1,
                    cost="$0.0001",
                )
            return raw

        cast(Any, KISSAgent).run = stub_run

    def _executor_tool_names(
        self, calls: list[dict[str, Any]],
    ) -> list[str]:
        """Return the tool names of the task-executor ``KISSAgent`` call.

        Args:
            calls: The record list filled by the executor stub.

        Returns:
            The ``tool_names`` of the (single) call whose arguments
            carry ``task_description`` — the executor session
            ``RelentlessAgent.perform_task`` spawned for the task.
        """
        executor_calls = [
            c for c in calls if "task_description" in c["arguments"]
        ]
        assert len(executor_calls) == 1, calls
        return list(executor_calls[0]["tool_names"])

    def _write_client_tools(self) -> str:
        """Write a tools file exporting one ``client_tool`` and return its path."""
        return self._write_py(
            "client_tools.py",
            '''
            """Client-supplied tools."""


            def client_tool(x: int) -> int:
                """Double a value.

                Args:
                    x: Value to double.
                """
                return 2 * x


            def get_tools():
                """Return the tools the agent may call."""
                return [client_tool]
            ''',
        )

    def _raw_daemon_run(
        self,
        extra_cmd: dict[str, Any],
        events_out: list[dict[str, Any]] | None = None,
    ) -> None:
        """Drive one raw ``run`` command over the local endpoint and wait for the end.

        Bypasses :func:`kiss.server.sorcar.run` so an absent or
        malformed wire field (``appendBasicTools``,
        ``appendToSystemPrompt``, ``appendToPrompt``) can be sent
        exactly as an arbitrary/buggy client would (the Python client
        always sends the fields).

        Args:
            extra_cmd: Raw fields merged into the ``run`` command.
            events_out: Optional list receiving every event the daemon
                emitted for this run's tab, in order.
        """
        tab_id = f"raw-{uuid.uuid4().hex}"
        ws = local_endpoint.connect(Path(self.endpoint_file), open_timeout=60)
        try:
            cmd = {
                "type": "run",
                "prompt": "raw client task",
                "tabId": tab_id,
                "taskId": uuid.uuid4().hex,
                "workDir": self.repo,
                "model": "",
                "useWorktree": False,
                "webTools": False,
                **extra_cmd,
            }
            ws.send(json.dumps(cmd))
            started = False
            while True:
                event = json.loads(ws.recv(timeout=60))
                if event.get("tabId") != tab_id:
                    continue
                if events_out is not None:
                    events_out.append(event)
                if event.get("type") != "status":
                    continue
                if event.get("running"):
                    started = True
                elif started:
                    return
        finally:
            ws.close()


class AppendBasicToolsApiTest(DaemonRunApiHarness):
    """Drive ``sorcar.run(append_basic_tools=...)`` against a real daemon."""

    def test_default_appends_basic_tools(self) -> None:
        """Without the parameter, the built-in basic toolset is added.

        The executor must see the basic tools (Bash/Read/Edit/Write,
        summary, run_agent, ask_user_question, talk, set_model,
        run_parallel, number_of_cores), the caller's client tool, AND
        ``finish`` — the historical default behavior, now behind
        ``append_basic_tools=True``.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with basic tools",
            work_dir=self.repo,
            tools=self._write_client_tools(),
            use_worktree=False,
            use_web_tools=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        names = self._executor_tool_names(calls)
        for basic in (
            "Bash", "Read", "Edit", "Write", "summary", "run_agent",
            "ask_user_question", "talk", "set_model", "run_parallel",
            "number_of_cores",
        ):
            assert basic in names, f"{basic} missing from {names}"
        assert "finish" in names
        assert "client_tool" in names

    def test_false_only_finish_and_client_tools(self) -> None:
        """``append_basic_tools=False`` leaves only finish + client tools.

        The executor's tool list must be EXACTLY ``finish`` (prepended
        by ``RelentlessAgent.perform_task``) followed by the tools the
        client's tools file's ``get_tools()`` returned — no Bash, no
        summary, no run_agent, nothing else.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with only client tools",
            work_dir=self.repo,
            tools=self._write_client_tools(),
            append_basic_tools=False,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert self._executor_tool_names(calls) == ["finish", "client_tool"]

    def test_false_without_client_tools_only_finish(self) -> None:
        """``append_basic_tools=False`` with no tools file: finish only."""
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "task with no tools at all",
            work_dir=self.repo,
            append_basic_tools=False,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert self._executor_tool_names(calls) == ["finish"]

    def test_agent_script_tools_is_the_whole_tool_set(self) -> None:
        """A script ``tools()`` strips the run down to finish + its own tools.

        The client sends the default (``append_basic_tools=True``) and a
        tools file; the script's ``tools()`` wins on both counts: the
        executor sees exactly ``finish`` and the script's tool — no
        basic tool and no client tool.
        """
        agent_path = self._write_py(
            "own_tools_agent.py",
            '''
            """Agent script whose tools() is the whole tool set."""


            def script_tool() -> str:
                """Return a marker."""
                return "script"


            def tools() -> list:
                """Run with only finish and script_tool."""
                return [script_tool]
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "script strips basic tools",
            work_dir=self.repo,
            tools=self._write_client_tools(),
            extension_agent_path=agent_path,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert self._executor_tool_names(calls) == ["finish", "script_tool"]

    def test_agent_script_add_to_tools_extends_the_basic_toolset(self) -> None:
        """A script ``add_to_tools()`` adds its tools to the basic toolset.

        The client sends ``append_basic_tools=False``; the script's
        ``add_to_tools()`` switches the basic toolset back on and the
        executor sees Bash/Read/... plus ``finish`` and the script's tool.
        """
        agent_path = self._write_py(
            "add_tools_agent.py",
            '''
            """Agent script adding a tool to the basic toolset."""


            def script_tool() -> str:
                """Return a marker."""
                return "script"


            def add_to_tools() -> list:
                """Add script_tool to the basic toolset."""
                return [script_tool]
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "script adds to basic tools",
            work_dir=self.repo,
            append_basic_tools=False,
            extension_agent_path=agent_path,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        names = self._executor_tool_names(calls)
        for expected in ("Bash", "Read", "Edit", "Write", "finish", "script_tool"):
            assert expected in names, f"{expected} missing from {names}"

    def test_agent_script_tools_path_fails_task(self) -> None:
        """A ``tools()`` returning a tools-file path stops the task loudly."""
        client_tools = self._write_client_tools()
        agent_path = self._write_py(
            "path_tools_agent.py",
            f'''
            """Agent script with a path-returning tools()."""


            def tools() -> str:
                """Return a path (no longer accepted)."""
                return {client_tools!r}
            ''',
        )
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        result = sorcar.run(
            "script with broken getter",
            work_dir=self.repo,
            extension_agent_path=agent_path,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is False
        assert "tools" in result.text
        assert "list of tool callables" in result.text
        assert calls == [], "no executor session may start for a broken script"

    def test_restricted_failure_skips_summarizer(self) -> None:
        """A restricted run's failure path launches no Read/Bash summarizer.

        ``RelentlessAgent.perform_task`` normally summarizes a failed
        sub-session's trajectory with a helper ``KISSAgent`` equipped
        with Read and Bash — tools an ``append_basic_tools=False`` run
        promised NO LLM session would get.  The restricted run must
        skip that summarizer and continue with the plain failure text.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls, fail_first_executor=True)
        result = sorcar.run(
            "restricted task whose first session fails",
            work_dir=self.repo,
            append_basic_tools=False,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        executor_calls = [
            c for c in calls if "task_description" in c["arguments"]
        ]
        assert len(executor_calls) == 2, calls
        for call in executor_calls:
            assert call["tool_names"] == ["finish"]
        assert [
            c for c in calls if "trajectory_path" in c["arguments"]
        ] == [], "restricted run must not launch the Read/Bash summarizer"

    def test_default_failure_uses_summarizer(self) -> None:
        """The default (basic-tools) failure path keeps its summarizer.

        Counterpart of the restricted test above: with
        ``append_basic_tools`` left True, the failed first sub-session
        is followed by the Read/Bash-equipped trajectory summarizer
        before the second sub-session continues.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls, fail_first_executor=True)
        result = sorcar.run(
            "default task whose first session fails",
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        summarizer_calls = [
            c for c in calls if "trajectory_path" in c["arguments"]
        ]
        assert len(summarizer_calls) == 1, calls
        assert summarizer_calls[0]["tool_names"] == ["Read", "Bash", "bash_job"]

    def test_absent_wire_field_defaults_true(self) -> None:
        """A raw command without ``appendBasicTools`` keeps the basics.

        Every pre-existing client (the webview ``submit`` path, old
        Python clients) omits the field — their agents must keep the
        full toolset.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        self._raw_daemon_run({})
        names = self._executor_tool_names(calls)
        assert "Bash" in names
        assert "summary" in names
        assert "finish" in names

    def test_malformed_wire_field_defaults_true(self) -> None:
        """A non-boolean ``appendBasicTools`` is ignored, not applied.

        The daemon treats the ``run`` command as untrusted input: a
        string ``"false"`` (or any non-bool) falls back to the default
        ``True`` instead of stripping tools or crashing the task
        thread.
        """
        calls: list[dict[str, Any]] = []
        self._install_executor_stub(calls)
        self._raw_daemon_run({"appendBasicTools": "false"})
        names = self._executor_tool_names(calls)
        assert "Bash" in names
        assert "summary" in names
        assert "finish" in names


if __name__ == "__main__":
    unittest.main()
