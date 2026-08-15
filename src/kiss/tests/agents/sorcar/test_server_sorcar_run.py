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
import inspect
import json
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
import unittest
import uuid
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer


def _task_chat_id(task_id: str) -> str:
    """Return the persisted chat_id of *task_id* via ``_load_history``."""
    for row in _persistence._load_history():
        if row["id"] == task_id:
            return str(row["chat_id"] or "")
    return ""


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
        from kiss.server import agent_state

        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
        agent_state.agent_states.clear()

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
        assert result.task_id
        assert result.chat_id
        assert _task_chat_id(result.task_id) == result.chat_id

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
        assert _task_chat_id(result.task_id) == result.chat_id

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
        assert _task_chat_id(second.task_id) == first.chat_id
        assert len(prompts_seen) == 2
        assert "remember the magic word xyzzy" in prompts_seen[1]
        assert "first answer marker" in prompts_seen[1]

    def _raw_daemon_run(
        self,
        tools_file: Any,
        extra_cmd: dict[str, Any] | None = None,
        events_out: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Drive one raw ``run`` command over the UDS and wait for the end.

        Bypasses :func:`kiss.server.sorcar.run` so malformed
        ``toolsFile`` payloads (or other malformed command fields via
        *extra_cmd*) can be sent exactly as an arbitrary/buggy client
        would.

        Args:
            tools_file: Raw value for the ``run`` command's
                ``toolsFile`` field.
            extra_cmd: Additional raw fields merged into the ``run``
                command.
            events_out: Optional list that receives every event the
                daemon broadcast for this run's tab, in order.

        Returns:
            The task's last ``result`` event, or ``None`` when the
            task produced none.
        """
        tab_id = f"raw-{uuid.uuid4().hex}"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        try:
            sock.connect(self.sock_path)
            cmd = {
                "type": "run",
                "prompt": "raw client task",
                "tabId": tab_id,
                "taskId": uuid.uuid4().hex,
                "workDir": self.repo,
                "model": "",
                "toolsFile": tools_file,
                **(extra_cmd or {}),
            }
            sock.sendall(json.dumps(cmd).encode() + b"\n")
            reader = sock.makefile("rb")
            started = False
            result_event: dict[str, Any] | None = None
            while True:
                event = json.loads(reader.readline())
                if events_out is not None and event.get("tabId") == tab_id:
                    events_out.append(event)
                if event.get("type") == "result":
                    # Each test runs its task alone on a private
                    # daemon, so any result event seen here belongs to
                    # this run (a failure before the agent publishes a
                    # task-history row is keyed by tabId, a success by
                    # taskId).
                    result_event = event
                    continue
                if event.get("tabId") != tab_id or event.get("type") != "status":
                    continue
                if event.get("running"):
                    started = True
                elif started:
                    return result_event
        finally:
            sock.close()

    def _write_tools_file(self, name: str, content: str) -> str:
        """Write a tools module under the test tmpdir and return its path.

        Args:
            name: File name (e.g. ``"my_tools.py"``).
            content: Python source for the file.

        Returns:
            The absolute path of the written file.
        """
        path = Path(self.tmpdir) / name
        path.write_text(textwrap.dedent(content))
        return str(path)

    def test_tools_file_functions_become_agent_tools(self) -> None:
        """The tools returned by ``get_tools()`` become agent tools.

        The daemon must import the client-supplied Python file itself
        (no serialization by the client), call its ``get_tools()``,
        and hand every returned function to the agent AS-IS: original
        object identity semantics (docstring, exact signature
        including keyword-only markers and the return annotation),
        native return values (an ``int`` stays an ``int`` — no string
        round trip), and execution in the daemon's task thread.
        """
        tools_path = self._write_tools_file(
            "my_tools.py",
            '''
            """Example tools module."""

            import threading


            def get_temperature(city: str, unit: str = "C", *, note: str = "") -> str:
                """Return the current temperature of a city.

                Args:
                    city: Name of the city to look up.
                    unit: Temperature unit to report.
                    note: Optional note echoed back.
                """
                return f"21{unit} in {city}{note}"


            def magic_number(seed: int, factor: int = 2) -> int:
                """Multiply a seed.

                Args:
                    seed: The seed.
                    factor: The factor.
                """
                return seed * factor


            def which_thread() -> str:
                """Report the executing thread's name."""
                return threading.current_thread().name


            def get_tools():
                """Return the tools the agent may call."""
                return [get_temperature, magic_number, which_thread]
            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            tools = {t.__name__: t for t in kwargs.get("tools") or []}
            seen["names"] = sorted(tools)
            temp = tools["get_temperature"]
            seen["doc"] = inspect.getdoc(temp)
            seen["signature"] = str(inspect.signature(temp))
            seen["r1"] = temp("Paris")
            seen["r2"] = temp(city="Berlin", unit="F", note="!")
            seen["r3"] = tools["magic_number"](seed=20)
            seen["thread"] = tools["which_thread"]()
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: tools ok\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "use my tools",
            work_dir=self.repo,
            tools=tools_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert result.text == "tools ok"
        assert seen["names"] == ["get_temperature", "magic_number", "which_thread"]
        assert seen["doc"] == (
            "Return the current temperature of a city.\n"
            "\n"
            "Args:\n"
            "    city: Name of the city to look up.\n"
            "    unit: Temperature unit to report.\n"
            "    note: Optional note echoed back."
        )
        assert seen["signature"] == (
            "(city: str, unit: str = 'C', *, note: str = '') -> str"
        )
        assert seen["r1"] == "21C in Paris"
        assert seen["r2"] == "21F in Berlin!"
        assert seen["r3"] == 40
        assert seen["thread"] != threading.current_thread().name

    def test_get_tools_selects_exactly_the_returned_functions(self) -> None:
        """``get_tools()`` alone decides which functions become tools.

        The daemon must not scan the module: functions the file
        defines but ``get_tools()`` does not return (helpers, private
        functions) never become tools, and the returned list's order
        is preserved.
        """
        tools_path = self._write_tools_file(
            "selected_tools.py",
            '''
            """Selection tools module."""


            def good(x: str = "a") -> str:
                """Echo.

                Args:
                    x: Value to echo.
                """
                return x


            def also_good(y: int) -> int:
                """Identity.

                Args:
                    y: Value to return.
                """
                return y


            def helper_not_a_tool(x: str) -> str:
                """Defined at top level but NOT returned by get_tools."""
                return x


            def get_tools():
                """Return only the selected tools, in this order."""
                return [also_good, good]
            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["names"] = [t.__name__ for t in kwargs.get("tools") or []]
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "use the selected tools",
            work_dir=self.repo,
            tools=tools_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert seen["names"] == ["also_good", "good"]

    def test_tools_file_relative_path_and_pathlib(self) -> None:
        """A relative ``Path`` is resolved by the CLIENT before sending.

        The daemon may run with a different working directory than the
        caller, so the client must resolve the path against ITS cwd.
        """
        self._write_tools_file(
            "rel_tools.py",
            '''
            """Relative-path tools module."""


            def greet(name: str) -> str:
                """Greet.

                Args:
                    name: Who to greet.
                """
                return f"hi {name}"


            def get_tools():
                """Return the tools the agent may call."""
                return [greet]
            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            (tool,) = cast(list[Any], kwargs.get("tools"))
            seen["result"] = tool(name="bob")
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            result = sorcar.run(
                "greet bob",
                work_dir=self.repo,
                tools=Path("rel_tools.py"),
                sock_path=self.sock_path,
                timeout=60,
            )
        finally:
            os.chdir(old_cwd)
        assert result.success is True
        assert seen["result"] == "hi bob"

    def _run_with_tools_file(self, tools_path: str, seen: dict[str, Any]) -> None:
        """Run one stubbed task with *tools_path* and record its tools.

        Installs a stub agent that appends the received tool names to
        ``seen["tool_lists"]`` and stores the tools themselves in
        ``seen["tools"]``, then drives one successful
        :func:`kiss.server.sorcar.run` with ``tools=tools_path``.

        Args:
            tools_path: Path of the tools file to pass to ``run``.
            seen: Cross-thread recording dict, mutated in place.
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            tools = list(kwargs.get("tools") or [])
            seen.setdefault("tool_lists", []).append([t.__name__ for t in tools])
            seen["tools"] = tools
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "use the tools file",
            work_dir=self.repo,
            tools=tools_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True

    def test_edited_tools_file_reloads_fresh_code(self) -> None:
        """A run always sees the tools file's CURRENT code.

        Regression: loading through ``importlib``'s ``SourceFileLoader``
        cached bytecode in ``__pycache__`` keyed on (mtime, size) — two
        same-length edits within one mtime granule made the second run
        silently execute the FIRST version's code.  The daemon must
        compile the source directly, and must not litter the caller's
        directory with ``__pycache__``.
        """
        tools_path = self._write_tools_file(
            "editable_tools.py",
            '''
            def version() -> str:
                """Report the tools file version."""
                return "ONE"


            def get_tools():
                """Return the tools the agent may call."""
                return [version]
            ''',
        )
        seen: dict[str, Any] = {}
        self._run_with_tools_file(tools_path, seen)
        (v1,) = seen["tools"]
        assert v1() == "ONE"
        self._write_tools_file(
            "editable_tools.py",
            '''
            def version() -> str:
                """Report the tools file version."""
                return "TWO"


            def get_tools():
                """Return the tools the agent may call."""
                return [version]
            ''',
        )
        self._run_with_tools_file(tools_path, seen)
        (v2,) = seen["tools"]
        assert v2() == "TWO"
        assert not (Path(self.tmpdir) / "__pycache__").exists()

    def test_missing_or_misbehaving_get_tools_fails_task(self) -> None:
        """A tools file with a bad ``get_tools()`` fails the task loudly.

        The contract requires a top-level callable ``get_tools()``
        returning a list/tuple of callables.  A module that lacks it,
        binds it to a non-callable, raises inside it, or returns a
        non-sequence or non-callable entries must stop the task with a
        ``ToolsFileError`` diagnostic — never invoke the agent.
        """
        no_get_tools = self._write_tools_file(
            "no_get_tools.py",
            '''
            def orphan(x: str) -> str:
                """Never exposed.

                Args:
                    x: Value to echo.
                """
                return x
            ''',
        )
        not_callable = self._write_tools_file(
            "not_callable_get_tools.py",
            "get_tools = 42\n",
        )
        raising_get_tools = self._write_tools_file(
            "raising_get_tools.py",
            '''
            def get_tools():
                """Raise instead of returning tools."""
                raise RuntimeError("boom in get_tools")
            ''',
        )
        bad_return = self._write_tools_file(
            "bad_return_get_tools.py",
            '''
            def get_tools():
                """Return a non-sequence."""
                return "not a list"
            ''',
        )
        non_callable_entry = self._write_tools_file(
            "non_callable_entry_get_tools.py",
            '''
            def get_tools():
                """Return a list with a non-callable entry."""
                return [42]
            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.setdefault("tool_lists", []).append(
                [t.__name__ for t in kwargs.get("tools") or []],
            )
            raise AssertionError("agent must not run with a broken tools file")

        self._parent_class.run = stub_run
        for tools_file, diagnostic in (
            (no_get_tools, "must define a top-level get_tools()"),
            (not_callable, "must define a top-level get_tools()"),
            (raising_get_tools, "RuntimeError: boom in get_tools"),
            (bad_return, "must return a list or tuple"),
            (non_callable_entry, "non-callable entry"),
        ):
            result_event = self._raw_daemon_run(tools_file)
            assert result_event is not None, f"no result for {tools_file!r}"
            assert result_event["success"] is False, f"for {tools_file!r}"
            assert "ToolsFileError" in result_event["text"], f"for {tools_file!r}"
            assert diagnostic in result_event["text"], f"for {tools_file!r}"
        assert "tool_lists" not in seen

    def test_sys_exit_in_tools_file_fails_task_with_diagnostic(self) -> None:
        """A tools file calling ``sys.exit()`` fails the task loudly.

        ``SystemExit`` is not an ``Exception`` subclass; the loader
        must convert it into ``ToolsFileError`` (letting it escape
        unwrapped would kill the task thread) so the task stops with a
        diagnostic result instead of silently running without the
        requested tools — and without ever invoking the agent.
        """
        tools_path = self._write_tools_file(
            "exiting_tools.py",
            '''
            import sys

            sys.exit(7)


            def never_loaded() -> str:
                """Unreachable."""
                return ""
            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.setdefault("tool_lists", []).append(
                [t.__name__ for t in kwargs.get("tools") or []],
            )
            raise AssertionError("agent must not run with a broken tools file")

        self._parent_class.run = stub_run
        result = sorcar.run(
            "use the broken tools file",
            work_dir=self.repo,
            tools=tools_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is False
        assert "ToolsFileError" in result.text
        assert "SystemExit" in result.text
        assert tools_path in result.text
        assert "tool_lists" not in seen

    def test_broken_tools_file_stops_task_with_diagnostic(self) -> None:
        """A broken ``toolsFile`` fails the task with a diagnostic error.

        A hand-crafted client can send anything: a non-string value, a
        missing path, a directory, a non-``.py`` file, a module that
        raises at import time, or one with a syntax error.  The daemon
        must stop the task with a failed result whose text carries the
        loader's diagnostic — never invoke the agent — and stay alive
        for later tasks.  An absent tools file (``None``) still runs
        the task normally with no extra tools.
        """
        raising = self._write_tools_file(
            "raising_tools.py",
            'raise RuntimeError("boom at import")\n',
        )
        broken = self._write_tools_file("broken_tools.py", "def broken(:\n")
        not_py = str(Path(self.tmpdir) / "tools.txt")
        Path(not_py).write_text("not python\n")
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.setdefault("tool_lists", []).append(
                [t.__name__ for t in kwargs.get("tools") or []],
            )
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run
        for tools_file, diagnostic in (
            (42, "path string"),
            (str(Path(self.tmpdir) / "nowhere.py"), "not an existing"),
            (self.tmpdir, "not an existing"),
            (not_py, "not an existing"),
            (raising, "RuntimeError: boom at import"),
            (broken, "SyntaxError"),
        ):
            result_event = self._raw_daemon_run(tools_file)
            assert result_event is not None, f"no result for {tools_file!r}"
            assert result_event["success"] is False, f"for {tools_file!r}"
            assert "ToolsFileError" in result_event["text"], f"for {tools_file!r}"
            assert diagnostic in result_event["text"], f"for {tools_file!r}"
        assert "tool_lists" not in seen
        result_event = self._raw_daemon_run(None)
        assert result_event is not None
        assert result_event["success"] is True
        assert seen["tool_lists"] == [[]]
        result = sorcar.run(
            "still alive?",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True

    def test_invalid_tools_file_raises_value_error(self) -> None:
        """Invalid ``tools`` values are rejected before connecting.

        ``sock_path`` points at a nonexistent socket, so reaching the
        connect stage would raise ``ConnectionError`` instead of the
        expected ``ValueError`` — proving validation is pre-connect.
        """
        missing_sock = str(Path(self.tmpdir) / "nowhere.sock")

        def a_tool(x: str) -> str:
            """Echo.

            Args:
                x: Value to echo.
            """
            return x

        cases: list[Any] = [
            42,
            [a_tool],
            str(Path(self.tmpdir) / "nowhere.py"),
            self.tmpdir,
            str(Path(self.tmpdir) / "tools.txt"),
        ]
        Path(self.tmpdir, "tools.txt").write_text("not python\n")
        for tools in cases:
            with self.assertRaises(ValueError):
                sorcar.run(
                    "hello",
                    tools=tools,
                    sock_path=missing_sock,
                    timeout=5,
                )

    def test_per_task_overrides_forwarded(self) -> None:
        """``max_budget`` / ``model_config`` / ``web_tools`` /
        ``is_parallel`` reach the daemon-built agent."""
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["max_budget"] = kwargs.get("max_budget")
            seen["model_config"] = kwargs.get("model_config")
            seen["web_tools"] = getattr(self_agent, "_use_web_tools", None)
            seen["is_parallel"] = getattr(self_agent, "_is_parallel", None)
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer")
            if printer is not None:
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=1, cost="$0.0001",
                )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "apply overrides",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
            max_budget=2.5,
            model_config={"base_url": "http://localhost:9999/v1"},
            web_tools=False,
            is_parallel=True,
        )
        assert result.success is True
        assert seen["max_budget"] == 2.5
        assert seen["model_config"] == {
            "base_url": "http://localhost:9999/v1",
        }
        assert seen["web_tools"] is False
        assert seen["is_parallel"] is True

    def test_malformed_override_fields_ignored(self) -> None:
        """Malformed override fields fall back to the daemon config.

        The daemon treats the ``run`` command as untrusted input: a
        boolean ``maxBudget``, a non-dict ``modelConfig``, and a
        non-boolean ``webTools`` are ignored rather than applied or
        crashing the task thread.
        """
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["max_budget"] = kwargs.get("max_budget")
            seen["model_config"] = kwargs.get("model_config")
            seen["web_tools"] = getattr(self_agent, "_use_web_tools", None)
            return "success: true\nis_continue: false\nsummary: ok\n"

        self._parent_class.run = stub_run
        self._raw_daemon_run(
            "",
            extra_cmd={
                "maxBudget": True,
                "modelConfig": "junk",
                "webTools": "yes",
            },
        )
        assert isinstance(seen["max_budget"], float)
        assert seen["max_budget"] > 0, "config default budget must apply"
        assert seen["model_config"] is None or isinstance(
            seen["model_config"], dict,
        ), "malformed modelConfig must not reach the agent"
        assert seen["model_config"] != "junk"
        assert seen["web_tools"] is True, "config default web tools apply"

    def test_custom_system_prompt_replaces_default(self) -> None:
        """A non-empty ``system_prompt`` replaces the SYSTEM.md prompt.

        The custom prompt must become the BASE of the composed system
        instructions the agent runs with (the default ``SYSTEM_PROMPT``
        must not appear anywhere in them), and it must be stored on the
        agent so the ``run_parallel`` fan-out forwards it to
        sub-agents.
        """
        from kiss.core.base import SYSTEM_PROMPT

        custom = (
            "You are a terse haiku-only assistant.\n"
            "Answer every request with a single haiku."
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["system_prompt"] = kwargs.get("system_prompt")
            seen["base_system_prompt"] = getattr(
                self_agent, "_base_system_prompt", None,
            )
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer")
            if printer is not None:
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=1, cost="$0.0001",
                )
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "say hi",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
            system_prompt=custom,
        )
        assert result.success is True
        composed = seen["system_prompt"]
        assert isinstance(composed, str)
        assert composed.startswith(custom), (
            "custom system prompt must be the base of the composed "
            "system instructions"
        )
        assert SYSTEM_PROMPT not in composed, (
            "the default SYSTEM.md prompt must be replaced, not kept"
        )
        assert seen["base_system_prompt"] == custom, (
            "the override must be stored on the agent for sub-agent "
            "fan-out"
        )

    def test_empty_system_prompt_runs_as_usual(self) -> None:
        """An empty ``system_prompt`` keeps the default SYSTEM.md prompt."""
        from kiss.core.base import SYSTEM_PROMPT

        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["system_prompt"] = kwargs.get("system_prompt")
            seen["base_system_prompt"] = getattr(
                self_agent, "_base_system_prompt", None,
            )
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer")
            if printer is not None:
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=1, cost="$0.0001",
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
        composed = seen["system_prompt"]
        assert isinstance(composed, str)
        assert composed.startswith(SYSTEM_PROMPT)
        assert seen["base_system_prompt"] == ""

    def test_malformed_or_blank_system_prompt_uses_default(self) -> None:
        """Non-string or whitespace-only ``systemPrompt`` wire fields
        fall back to the default system prompt instead of crashing."""
        from kiss.core.base import SYSTEM_PROMPT

        for bad in (42, ["x"], {"a": 1}, None, "   \n\t"):
            seen: dict[str, Any] = {}

            def stub_run(self_agent: Any, **kwargs: Any) -> str:
                seen["system_prompt"] = kwargs.get("system_prompt")
                return "success: true\nis_continue: false\nsummary: ok\n"

            self._parent_class.run = stub_run
            self._raw_daemon_run(None, extra_cmd={"systemPrompt": bad})
            composed = seen.get("system_prompt")
            assert isinstance(composed, str), (
                f"task must still run for systemPrompt={bad!r}"
            )
            assert composed.startswith(SYSTEM_PROMPT), (
                f"systemPrompt={bad!r} must fall back to the default"
            )

    def test_custom_system_prompt_shown_in_early_panel(self) -> None:
        """The early ``system_prompt`` UI event shows the override text."""
        custom = "Custom base prompt for the early panel."
        events: list[dict[str, Any]] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            return "success: true\nis_continue: false\nsummary: ok\n"

        self._parent_class.run = stub_run
        self._raw_daemon_run(
            None,
            extra_cmd={"systemPrompt": custom},
            events_out=events,
        )
        early = [
            e for e in events
            if e.get("type") == "system_prompt" and e.get("early")
        ]
        assert early, "an early system_prompt event must be broadcast"
        assert early[0].get("text", "").startswith(custom), (
            "the early panel must show the caller-supplied system prompt"
        )

    def test_custom_system_prompt_reaches_subagents(self) -> None:
        """The fan-out engine passes the override to every sub-agent.

        Covers both halves of the sub-agent wiring: the engine's
        ``base_system_prompt`` parameter (called directly) and the
        parent-agent forwarding of its stored ``_base_system_prompt``
        (``SorcarAgent._run_tasks_parallel``).
        """
        import threading as _threading

        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
        from kiss.core.base import SYSTEM_PROMPT

        custom = "You are a security-review sub-agent. Be paranoid."
        lock = _threading.Lock()
        composed_prompts: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            with lock:
                composed_prompts.append(str(kwargs.get("system_prompt")))
            return "success: true\nis_continue: false\nsummary: ok\n"

        self._parent_class.run = stub_run

        # Half 1: the engine parameter, as forwarded by a parent.
        results = run_tasks_parallel(
            ["child task one", "child task two"],
            work_dir=self.repo,
            base_system_prompt=custom,
        )
        assert len(results) == 2
        assert len(composed_prompts) == 2
        for composed in composed_prompts:
            assert composed.startswith(custom)
            assert SYSTEM_PROMPT not in composed

        # Half 2: a parent agent that ran with the override stores it
        # and forwards it through its own fan-out.
        composed_prompts.clear()
        parent = ChatSorcarAgent("system-prompt-parent")
        parent._base_system_prompt = custom
        results = parent._run_tasks_parallel(["nested child task"])
        assert len(results) == 1
        assert len(composed_prompts) == 1
        assert composed_prompts[0].startswith(custom)
        assert SYSTEM_PROMPT not in composed_prompts[0]

        # A parent WITHOUT an override spawns default-prompt children.
        composed_prompts.clear()
        plain_parent = ChatSorcarAgent("default-prompt-parent")
        plain_parent._run_tasks_parallel(["plain child task"])
        assert len(composed_prompts) == 1
        assert composed_prompts[0].startswith(SYSTEM_PROMPT)

    def test_api_tab_state_disposed_after_run(self) -> None:
        """``run()`` explicitly closes its synthetic tab; no state leaks.

        A client disconnect no longer tears tabs down (tabs are global
        state shared by every client), so the API client itself sends
        the daemon a ``closeTab`` for its ``api-…`` tab on exit.
        Without it the ``server_owned`` ``AgentState`` and per-tab chat
        view of every ``run()`` call would accumulate in the daemon
        forever, one leaked entry per fresh ``api-{uuid}`` tab.
        """
        import time as _time

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(raw, type="result", step_count=1)
            return raw

        self._parent_class.run = stub_run
        result = sorcar.run(
            "say hi",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True

        from kiss.server import agent_state

        # The closeTab is dispatched asynchronously after run() returns.
        api_states: list[Any] = []
        api_views: list[str] = []
        deadline = _time.monotonic() + 10.0
        while _time.monotonic() < deadline:
            api_states = [
                state for state in agent_state.snapshot()
                if state.tab_id.startswith("api-")
            ]
            with self.server._vscode_server._state_lock:
                api_views = [
                    tab for tab in self.server._vscode_server._tab_chat_views
                    if tab.startswith("api-")
                ]
            if not api_states and not api_views:
                break
            _time.sleep(0.05)
        assert api_states == [], (
            "api tab AgentState leaked after run(): the client must "
            "send an explicit closeTab on exit"
        )
        assert api_views == [], "api tab chat view leaked after run()"

    def test_no_daemon_raises_connection_error(self) -> None:
        """A missing daemon socket raises a helpful ConnectionError."""
        missing = str(Path(self.tmpdir) / "nowhere.sock")
        with self.assertRaises(ConnectionError):
            sorcar.run("hello", sock_path=missing, timeout=5)

    def test_blank_prompt_raises_value_error(self) -> None:
        """Blank prompts are rejected before any connection is made."""
        with self.assertRaises(ValueError):
            sorcar.run("   ", sock_path=self.sock_path, timeout=5)
