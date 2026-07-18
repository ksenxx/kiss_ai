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

    def _raw_daemon_run(self, tools_file: Any) -> None:
        """Drive one raw ``run`` command over the UDS and wait for the end.

        Bypasses :func:`kiss.server.sorcar.run` so malformed
        ``toolsFile`` payloads can be sent exactly as an
        arbitrary/buggy client would.

        Args:
            tools_file: Raw value for the ``run`` command's
                ``toolsFile`` field.
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
            }
            sock.sendall(json.dumps(cmd).encode() + b"\n")
            reader = sock.makefile("rb")
            started = False
            while True:
                event = json.loads(reader.readline())
                if event.get("tabId") != tab_id or event.get("type") != "status":
                    continue
                if event.get("running"):
                    started = True
                elif started:
                    return
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
        """Top-level public functions of the tools file become agent tools.

        The daemon must import the client-supplied Python file itself
        (no serialization by the client) and hand every top-level
        public function to the agent AS-IS: original object identity
        semantics (docstring, exact signature including keyword-only
        markers and the return annotation), native return values (an
        ``int`` stays an ``int`` — no string round trip), and
        execution in the daemon's task thread.
        """
        tools_path = self._write_tools_file(
            "my_tools.py",
            '''
            """Example tools module."""

            import threading
            from os.path import join  # imported: must NOT become a tool

            GREETING = "hello"  # not a function


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


            def _private_helper(x: str) -> str:
                return x


            class NotATool:
                """Classes are not tools."""
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
        # Private helpers, imported functions, classes, and constants
        # are excluded; every top-level public function is included.
        assert seen["names"] == ["get_temperature", "magic_number", "which_thread"]
        # The daemon loaded the REAL function: full docstring and the
        # exact signature (keyword-only marker, defaults, and return
        # annotation) survive because nothing was serialized.
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
        # Native return value — an ``int``, not a stringified proxy
        # round trip.
        assert seen["r3"] == 40
        # The tools ran in the DAEMON's task thread (the stub agent's
        # thread), not in the client thread blocked in ``sorcar.run``.
        assert seen["thread"] != threading.current_thread().name

    def test_tools_file_skips_unsuitable_functions(self) -> None:
        """Functions unsuitable as tools are skipped, the rest are kept."""
        tools_path = self._write_tools_file(
            "mixed_tools.py",
            '''
            """Mixed suitability tools module."""


            def good(x: str = "a") -> str:
                """Echo.

                Args:
                    x: Value to echo.
                """
                return x


            def star_args(*args: str) -> str:
                """Unsupported: *args."""
                return ""


            def kw_args(**kwargs: str) -> str:
                """Unsupported: **kwargs."""
                return ""


            def pos_only(x: str, /) -> str:
                """Unsupported: positional-only parameter."""
                return x


            async def async_tool(x: str) -> str:
                """Unsupported: coroutine function."""
                return x


            def gen_tool(x: str):
                """Unsupported: generator function."""
                yield x
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
            "use the suitable tools",
            work_dir=self.repo,
            tools=tools_path,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert seen["names"] == ["good"]

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
        # Same byte length, written back-to-back (same mtime granule).
        tools_path = self._write_tools_file(
            "editable_tools.py",
            '''
            def version() -> str:
                """Report the tools file version."""
                return "ONE"
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
            ''',
        )
        self._run_with_tools_file(tools_path, seen)
        (v2,) = seen["tools"]
        assert v2() == "TWO"
        assert not (Path(self.tmpdir) / "__pycache__").exists()

    def test_lambdas_aliases_and_broken_functions_are_skipped(self) -> None:
        """Only genuine ``def`` bindings with sane metadata become tools.

        A lambda binding (``__name__ == "<lambda>"`` would break the
        tool schema), an alias of another top-level function (would
        register the same tool twice and crash ``_add_functions``), a
        re-exported nested function, and functions whose signature
        introspection raises (corrupted ``__signature__``,
        self-referential ``__wrapped__``) must all be skipped — while
        the well-formed functions are kept and callable.
        """
        tools_path = self._write_tools_file(
            "tricky_tools.py",
            '''
            """Tricky bindings tools module."""

            shout = lambda x: x.upper()  # noqa: E731


            def good(x: str) -> str:
                """Echo.

                Args:
                    x: Value to echo.
                """
                return x


            alias = good


            def _outer():
                def nested(x: str) -> str:
                    return x
                return nested


            exported = _outer()


            def bad_signature(x: str) -> str:
                """Corrupted ``__signature__``."""
                return x


            bad_signature.__signature__ = "not a signature"


            def wrapper_loop(x: str) -> str:
                """Self-referential ``__wrapped__``."""
                return x


            wrapper_loop.__wrapped__ = wrapper_loop
            ''',
        )
        seen: dict[str, Any] = {}
        self._run_with_tools_file(tools_path, seen)
        assert seen["tool_lists"][-1] == ["good"]
        (good,) = seen["tools"]
        assert good(x="hi") == "hi"

    def test_sys_exit_in_tools_file_is_contained(self) -> None:
        """A tools file calling ``sys.exit()`` cannot kill the task.

        ``SystemExit`` is not an ``Exception`` subclass; the loader
        must contain it (mirroring ``KISSAgent._execute_tool``) so the
        task still runs — with no extra tools.
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
        self._run_with_tools_file(tools_path, seen)
        assert seen["tool_lists"] == [[]]

    def test_malformed_tools_file_ignored_by_daemon(self) -> None:
        """A malformed ``toolsFile`` field never kills the task thread.

        A hand-crafted client can send anything: a non-string value, a
        missing path, a directory, a non-``.py`` file, a module that
        raises at import time, or one with a syntax error.  The daemon
        must log, run the task with NO extra tools, and stay alive.
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
        for tools_file in (
            42,  # not a string
            str(Path(self.tmpdir) / "nowhere.py"),  # missing file
            self.tmpdir,  # a directory, not a .py file
            not_py,  # wrong suffix
            raising,  # import-time exception
            broken,  # syntax error
            None,  # absent field (plain webview submits)
        ):
            self._raw_daemon_run(tools_file)
        assert seen["tool_lists"] == [[]] * 7
        # The daemon survived it all: a normal API run still works.
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
            42,  # not a path
            [a_tool],  # the old list-of-callables API shape
            str(Path(self.tmpdir) / "nowhere.py"),  # missing file
            self.tmpdir,  # a directory
            str(Path(self.tmpdir) / "tools.txt"),  # wrong suffix
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

    def test_no_daemon_raises_connection_error(self) -> None:
        """A missing daemon socket raises a helpful ConnectionError."""
        missing = str(Path(self.tmpdir) / "nowhere.sock")
        with self.assertRaises(ConnectionError):
            sorcar.run("hello", sock_path=missing, timeout=5)

    def test_blank_prompt_raises_value_error(self) -> None:
        """Blank prompts are rejected before any connection is made."""
        with self.assertRaises(ValueError):
            sorcar.run("   ", sock_path=self.sock_path, timeout=5)
