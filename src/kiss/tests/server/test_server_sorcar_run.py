# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :func:`kiss.server.sorcar.run`.

Spin up a real :class:`kiss.server.web_server.RemoteAccessServer` on a
loopback local endpoint and drive the new synchronous
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
import subprocess
import tempfile
import textwrap
import threading
import unittest
import uuid
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.server import sorcar
from kiss.server.web_server import RemoteAccessServer


def _sea_tools(agent: Any) -> list[Any]:
    """Return the tools a stubbed run gets from its SEA's ``tools()``.

    The daemon hands ``SorcarAgent.run`` the SEA's ``tools`` method as
    its ``tools_hook`` (kept on the agent, applied by ``perform_task``
    to the built-in toolset plus the inherited tools); the stub that
    replaces the parent ``run`` applies it the way ``perform_task``
    does, to the inherited tools alone (a stub never builds the
    built-ins).

    Args:
        agent: The :class:`SorcarAgent` whose parent ``run`` is stubbed.
    """
    tools = list(agent._inherited_tools)
    hook = agent._tools_hook
    return hook(tools) if hook is not None else tools


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
    """Drive ``kiss.server.sorcar.run`` against a real daemon over its local endpoint."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sorcar_run_api_")
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

        _persistence._close_db()
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
            endpoint_file=self.endpoint_file,
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
            endpoint_file=self.endpoint_file,
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
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert first.success is True
        assert first.chat_id
        second = sorcar.run(
            "what was the magic word?",
            work_dir=self.repo,
            chat_id=first.chat_id,
            endpoint_file=self.endpoint_file,
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
        sea_path: Any,
        extra_cmd: dict[str, Any] | None = None,
        events_out: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Drive one raw ``run`` command over the local endpoint and wait for the end.

        Bypasses :func:`kiss.server.sorcar.run` so malformed
        ``seaPath`` payloads (or other malformed command fields via
        *extra_cmd*) can be sent exactly as an arbitrary/buggy client
        would.

        Args:
            sea_path: Raw value for the ``run`` command's
                ``seaPath`` field.
            extra_cmd: Additional raw fields merged into the ``run``
                command.
            events_out: Optional list that receives every event the
                daemon broadcast for this run's tab, in order.

        Returns:
            The task's last ``result`` event, or ``None`` when the
            task produced none.
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
                "seaPath": sea_path,
                **(extra_cmd or {}),
            }
            ws.send(json.dumps(cmd))
            started = False
            result_event: dict[str, Any] | None = None
            while True:
                event = json.loads(ws.recv(timeout=60))
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
            ws.close()

    def _write_agent_script(self, name: str, content: str) -> str:
        """Write a SEA under the test tmpdir and return its path.

        Args:
            name: File name (e.g. ``"my_agent.py"``).
            content: Python source for the file.

        Returns:
            The absolute path of the written file.
        """
        path = Path(self.tmpdir) / name
        path.write_text(textwrap.dedent(content))
        return str(path)

    def test_agent_script_tools_become_agent_tools(self) -> None:
        """The tools returned by the SEA's ``tools()`` become agent tools.

        The daemon must import the client-supplied SEA itself
        (no serialization by the client), hand the run its ``tools()``
        as the tools hook, and that hook must return every function
        AS-IS: original
        object identity semantics (docstring, exact signature
        including keyword-only markers and the return annotation),
        native return values (an ``int`` stays an ``int`` — no string
        round trip), and execution in the daemon's task thread.
        """
        tools_path = self._write_agent_script(
            "my_tools.py",
            '''
            """Example tools module."""

            import threading
            from kiss.agents.seas.base.base_sea import BaseSea


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


            class Sea(BaseSea):
                def tools(self, tools):
                    """Return the tools the agent may call."""
                    return tools + [get_temperature, magic_number, which_thread]


            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            tools = {t.__name__: t for t in _sea_tools(self_agent)}
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
            sea_path=tools_path,
            endpoint_file=self.endpoint_file,
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

    def test_add_to_tools_selects_exactly_the_returned_functions(self) -> None:
        """The SEA's ``tools()`` alone decides which functions become tools.

        The daemon must not scan the module: functions the file
        defines but ``tools()`` does not return (helpers, private
        functions) never become tools, and the returned list's order
        is preserved.  With ``tool_profile: "none"`` the returned list
        is the whole tool set: the basic toolset is switched off.
        """
        tools_path = self._write_agent_script(
            "selected_tools.py",
            '''
            """Selection tools module."""

            from kiss.agents.seas.base.base_sea import BaseSea


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
                """Defined at top level but NOT returned by tools()."""
                return x


            class Sea(BaseSea):
                def settings(self, settings):
                    """Switch the basic toolset off."""
                    return settings | {"tool_profile": "none"}

                def tools(self, tools):
                    """Return only the selected tools, in this order."""
                    return tools + [also_good, good]


            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["names"] = [t.__name__ for t in _sea_tools(self_agent)]
            seen["append_basic_tools"] = self_agent._append_basic_tools
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
            sea_path=tools_path,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert seen["names"] == ["also_good", "good"]
        assert seen["append_basic_tools"] is False

    def test_agent_script_relative_path_resolved_by_client(self) -> None:
        """A relative path is resolved by the CLIENT before sending.

        The daemon may run with a different working directory than the
        caller, so the client must resolve the path against ITS cwd.
        """
        self._write_agent_script(
            "rel_tools.py",
            '''
            """Relative-path tools module."""

            from kiss.agents.seas.base.base_sea import BaseSea


            def greet(name: str) -> str:
                """Greet.

                Args:
                    name: Who to greet.
                """
                return f"hi {name}"


            class Sea(BaseSea):
                def tools(self, tools):
                    """Return the tools the agent may call."""
                    return tools + [greet]


            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            (tool,) = _sea_tools(self_agent)
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
                sea_path="rel_tools.py",
                endpoint_file=self.endpoint_file,
                timeout=60,
            )
        finally:
            os.chdir(old_cwd)
        assert result.success is True
        assert seen["result"] == "hi bob"

    def _run_with_agent_script(self, tools_path: str, seen: dict[str, Any]) -> None:
        """Run one stubbed task with *tools_path* and record its tools.

        Installs a stub agent that appends the received tool names to
        ``seen["tool_lists"]`` and stores the tools themselves in
        ``seen["tools"]``, then drives one successful
        :func:`kiss.server.sorcar.run` with
        ``sea_path=tools_path``.

        Args:
            tools_path: Path of the SEA to pass to ``run``.
            seen: Cross-thread recording dict, mutated in place.
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            tools = _sea_tools(self_agent)
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
            "use the SEA's tools",
            work_dir=self.repo,
            sea_path=tools_path,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True

    def test_edited_agent_script_reloads_fresh_code(self) -> None:
        """A run always sees the SEA's CURRENT code.

        Regression: loading through ``importlib``'s ``SourceFileLoader``
        cached bytecode in ``__pycache__`` keyed on (mtime, size) — two
        same-length edits within one mtime granule made the second run
        silently execute the FIRST version's code.  The daemon must
        compile the source directly, and must not litter the caller's
        directory with ``__pycache__``.
        """
        tools_path = self._write_agent_script(
            "editable_tools.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            def version() -> str:
                """Report the SEA version."""
                return "ONE"


            class Sea(BaseSea):
                def tools(self, tools):
                    """Return the tools the agent may call."""
                    return tools + [version]


            ''',
        )
        seen: dict[str, Any] = {}
        self._run_with_agent_script(tools_path, seen)
        (v1,) = seen["tools"]
        assert v1() == "ONE"
        self._write_agent_script(
            "editable_tools.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            def version() -> str:
                """Report the SEA version."""
                return "TWO"


            class Sea(BaseSea):
                def tools(self, tools):
                    """Return the tools the agent may call."""
                    return tools + [version]


            ''',
        )
        self._run_with_agent_script(tools_path, seen)
        (v2,) = seen["tools"]
        assert v2() == "TWO"
        assert not (Path(self.tmpdir) / "__pycache__").exists()

    def test_misbehaving_tool_getter_fails_task(self) -> None:
        """A SEA with a bad ``tools()`` fails the task loudly.

        The contract requires ``tools()``, when defined, to return a
        list of callables, and ``settings()['tool_profile']`` to be a
        string.  A file that defines no SEA class or names a
        wrong-typed tool profile is rejected when the daemon loads it:
        the task stops with an ``SeaError`` diagnostic and the
        agent never runs.  A ``tools()`` that raises, returns a
        non-list (e.g. a file path) or non-callable entries is caught
        when the run builds its toolset (the hook the daemon hands the
        run raises the diagnostic), so the stubbed run applies the
        hook the way ``perform_task`` does and must never get past it.
        """
        no_sea_class = self._write_agent_script(
            "not_callable_add_to_tools.py",
            "add_to_tools = 42\n",
        )
        raising_getter = self._write_agent_script(
            "raising_add_to_tools.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            class Sea(BaseSea):
                def tools(self, tools):
                    """Raise instead of returning tools."""
                    raise RuntimeError("boom in add_to_tools")


            ''',
        )
        bad_return = self._write_agent_script(
            "bad_return_tools.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            class Sea(BaseSea):
                def tools(self, tools):
                    """Return a path instead of a list."""
                    return "/some/tools_file.py"


            ''',
        )
        non_callable_entry = self._write_agent_script(
            "non_callable_entry_add_to_tools.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            class Sea(BaseSea):
                def tools(self, tools):
                    """Return a list with a non-callable entry."""
                    return tools + [42]


            ''',
        )
        bad_profile = self._write_agent_script(
            "bad_tool_profile.py",
            '''
            from kiss.agents.seas.base.base_sea import BaseSea

            class Sea(BaseSea):
                def settings(self, settings):
                    """Name a tool profile of the wrong type."""
                    return settings | {"tool_profile": 7}

                def tools(self, tools):
                    """Additions to the basic toolset."""
                    return tools + []


            ''',
        )
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.setdefault("hooks", []).append(self_agent._tools_hook is not None)
            names = [t.__name__ for t in _sea_tools(self_agent)]
            seen.setdefault("tool_lists", []).append(names)
            raise AssertionError("agent must not run with a broken SEA")

        self._parent_class.run = stub_run
        for sea_path, diagnostic in (
            (no_sea_class, "must define exactly one subclass of BaseSea"),
            (bad_profile, "settings()['tool_profile'] must be str, got int"),
        ):
            result_event = self._raw_daemon_run(sea_path)
            assert result_event is not None, f"no result for {sea_path!r}"
            assert result_event["success"] is False, f"for {sea_path!r}"
            assert "SeaError" in result_event["text"], f"for {sea_path!r}"
            assert diagnostic in result_event["text"], f"for {sea_path!r}"
        assert "hooks" not in seen, "the agent must not run with a broken SEA"
        for sea_path, diagnostic in (
            (raising_getter, "tools() of SEA"),
            (raising_getter, "raised: RuntimeError: boom in add_to_tools"),
            (bad_return, "must return a list of tool callables (not a file path), got str"),
            (non_callable_entry, "must return a list of tool callables"),
        ):
            seen.clear()
            events: list[dict[str, Any]] = []
            self._raw_daemon_run(sea_path, events_out=events)
            assert seen["hooks"] == [True], f"for {sea_path!r}"
            assert "tool_lists" not in seen, f"the hook must raise for {sea_path!r}"
            # The run ends as a failed task (a ``task_error`` event
            # carrying the diagnostic) whose persisted result carries
            # the diagnostic too.
            (end_event,) = [e for e in events if e.get("type") == "task_error"]
            assert diagnostic in end_event["text"], f"for {sea_path!r}: {end_event!r}"
            (task_id,) = {e["taskId"] for e in events if e.get("type") == "task_settings"}
            (row,) = [r for r in _persistence._load_history() if r["id"] == task_id]
            assert "Task failed" in str(row["result"]), f"for {sea_path!r}: {row!r}"
            assert diagnostic in str(row["result"]), f"for {sea_path!r}: {row!r}"

    def test_sys_exit_in_agent_script_fails_task_with_diagnostic(self) -> None:
        """A SEA calling ``sys.exit()`` fails the task loudly.

        ``SystemExit`` is not an ``Exception`` subclass; the loader
        must convert it into ``SeaError`` (letting it escape
        unwrapped would kill the task thread) so the task stops with a
        diagnostic result instead of silently running without the
        requested tools — and without ever invoking the agent.
        """
        tools_path = self._write_agent_script(
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
            raise AssertionError("agent must not run with a broken SEA")

        self._parent_class.run = stub_run
        result = sorcar.run(
            "use the broken SEA",
            work_dir=self.repo,
            sea_path=tools_path,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is False
        assert "SeaError" in result.text
        assert "SystemExit" in result.text
        # The diagnostic quotes the resolved path with repr() (doubled
        # backslashes on Windows, /private/var on macOS).
        assert repr(str(Path(tools_path).resolve())) in result.text
        assert "tool_lists" not in seen

    def test_broken_agent_script_stops_task_with_diagnostic(self) -> None:
        """A broken ``seaPath`` fails the task with a diagnostic error.

        A hand-crafted client can send anything: a non-string value, a
        missing path, a directory, a non-``.py`` file, a module that
        raises at import time, or one with a syntax error.  The daemon
        must stop the task with a failed result whose text carries the
        loader's diagnostic — never invoke the agent — and stay alive
        for later tasks.  An absent SEA (``None``) still runs
        the task normally with no extra tools.
        """
        raising = self._write_agent_script(
            "raising_tools.py",
            'raise RuntimeError("boom at import")\n',
        )
        broken = self._write_agent_script("broken_tools.py", "def broken(:\n")
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
        for sea_path, diagnostic in (
            (42, "path string"),
            (str(Path(self.tmpdir) / "nowhere.py"), "not an existing"),
            (self.tmpdir, "not an existing"),
            (not_py, "not an existing"),
            (raising, "RuntimeError: boom at import"),
            (broken, "SyntaxError"),
        ):
            result_event = self._raw_daemon_run(sea_path)
            assert result_event is not None, f"no result for {sea_path!r}"
            assert result_event["success"] is False, f"for {sea_path!r}"
            assert "SeaError" in result_event["text"], f"for {sea_path!r}"
            assert diagnostic in result_event["text"], f"for {sea_path!r}"
        assert "tool_lists" not in seen
        result_event = self._raw_daemon_run(None)
        assert result_event is not None
        assert result_event["success"] is True
        assert seen["tool_lists"] == [[]]
        result = sorcar.run(
            "still alive?",
            work_dir=self.repo,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True

    def test_invalid_sea_path_raises_value_error(self) -> None:
        """Invalid ``sea_path`` values are rejected before connecting.

        ``endpoint_file`` names a nonexistent file, so reaching the
        connect stage would raise ``ConnectionError`` instead of the
        expected ``ValueError`` — proving validation is pre-connect.
        """
        missing_endpoint = str(Path(self.tmpdir) / "nowhere.json")

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
        for sea_path in cases:
            with self.assertRaises(ValueError):
                sorcar.run(
                    "hello",
                    sea_path=sea_path,
                    endpoint_file=missing_endpoint,
                    timeout=5,
                )

    def test_per_task_overrides_forwarded(self) -> None:
        """``max_budget`` / ``model_config`` / ``use_web_tools`` /
        ``use_memory`` / ``is_parallel`` reach the daemon-built agent."""
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["max_budget"] = kwargs.get("max_budget")
            seen["model_config"] = kwargs.get("model_config")
            seen["web_tools"] = getattr(self_agent, "_use_web_tools", None)
            seen["is_parallel"] = getattr(self_agent, "_is_parallel", None)
            seen["use_memory"] = getattr(
                self_agent, "_use_memory_override", "MISSING",
            )
            # Recorded DURING the run, before SorcarAgent.run's finally
            # clears it: with the explicit False below the memory
            # decision must have built NO MemoryTools.
            seen["memory_tools"] = getattr(
                self_agent, "_memory_tools", "MISSING",
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
            "apply overrides",
            work_dir=self.repo,
            endpoint_file=self.endpoint_file,
            timeout=60,
            max_budget=2.5,
            model_config={"base_url": "http://localhost:9999/v1"},
            use_web_tools=False,
            use_memory=False,
            allow_fan_out=True,
        )
        assert result.success is True
        assert seen["max_budget"] == 2.5
        assert seen["model_config"] == {
            "base_url": "http://localhost:9999/v1",
        }
        assert seen["web_tools"] is False
        assert seen["is_parallel"] is True
        assert seen["use_memory"] is False
        assert seen["memory_tools"] is None

    def test_malformed_override_fields_ignored(self) -> None:
        """Malformed override fields fall back to the daemon config.

        The daemon treats the ``run`` command as untrusted input: a
        boolean ``maxBudget``, a non-dict ``modelConfig``, a
        non-boolean ``useWebTools``, and a non-boolean ``useMemory`` are
        ignored rather than applied or crashing the task thread.
        """
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["max_budget"] = kwargs.get("max_budget")
            seen["model_config"] = kwargs.get("model_config")
            seen["web_tools"] = getattr(self_agent, "_use_web_tools", None)
            seen["use_memory"] = getattr(
                self_agent, "_use_memory_override", "MISSING",
            )
            return "success: true\nis_continue: false\nsummary: ok\n"

        self._parent_class.run = stub_run
        self._raw_daemon_run(
            "",
            extra_cmd={
                "maxBudget": True,
                "modelConfig": "junk",
                "useWebTools": "yes",
                "useMemory": "yes",
            },
        )
        assert isinstance(seen["max_budget"], float)
        assert seen["max_budget"] > 0, "config default budget must apply"
        assert seen["model_config"] is None or isinstance(
            seen["model_config"], dict,
        ), "malformed modelConfig must not reach the agent"
        assert seen["model_config"] != "junk"
        assert seen["web_tools"] is True, "config default web tools apply"
        assert seen["use_memory"] is None, (
            "a malformed useMemory means no per-run override — the agent "
            "falls back to the persisted setting"
        )

    def test_classify_tasks_override_forwarded(self) -> None:
        """The per-run ``classify_tasks`` toggle reaches the classifier.

        ``kiss.server.sorcar.run(auto_classify=...)`` rides the wire
        as ``classifyTasks`` and the task runner must hand it to
        ``classify_task_for_run(enabled=...)`` — the run-side gate of
        the settings panel's "Classify tasks before running" option.
        Absent and malformed values mean "no override" (``None``), the
        same untrusted-input contract as ``useWebTools``.  The suite-wide
        ``KISS_DISABLE_TASK_CLASSIFIER=1`` kill switch keeps the real
        ``classify_task_for_run`` from ever calling a model here, so
        the full daemon pipeline runs with only the ``run`` stub.
        """
        seen_enabled: list[Any] = []
        original_classify = SorcarAgent.classify_task_for_run

        def recording_classify(
            self_agent: Any, *args: Any, **kwargs: Any,
        ) -> Any:
            seen_enabled.append(kwargs.get("enabled"))
            return original_classify(self_agent, *args, **kwargs)

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer")
            if printer is not None:
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=1, cost="$0.0001",
                )
            return raw

        self._parent_class.run = stub_run
        SorcarAgent.classify_task_for_run = recording_classify  # type: ignore[assignment,method-assign]
        try:
            for override in (False, True):
                result = sorcar.run(
                    "task with a classification override",
                    work_dir=self.repo,
                    use_worktree=False,
                    auto_classify=override,
                    endpoint_file=self.endpoint_file,
                    timeout=60,
                )
                assert result.success is True
            # No parameter passed: the wire carries null → no override.
            result = sorcar.run(
                "task without a classification override",
                work_dir=self.repo,
                use_worktree=False,
                endpoint_file=self.endpoint_file,
                timeout=60,
            )
            assert result.success is True
            # A malformed (non-boolean) classifyTasks is ignored.
            self._raw_daemon_run("", extra_cmd={"classifyTasks": "yes"})
        finally:
            SorcarAgent.classify_task_for_run = original_classify  # type: ignore[method-assign]
        assert seen_enabled == [False, True, None, None], (
            "classify_tasks was not forwarded verbatim on the "
            "run → classify_task_for_run path"
        )

    def test_persisted_use_web_tools_setting_applies(self) -> None:
        """The settings panel's "Use web tools" checkbox binds the run.

        The checkbox persists as config key ``use_web_browser``: a run
        WITHOUT a per-run ``use_web_tools`` override must fall back to
        the persisted ``False``, and an explicit ``use_web_tools=True``
        on the same daemon must beat the persisted setting.
        """
        vscode_config.save_config({"use_web_browser": False})
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["web_tools"] = getattr(self_agent, "_use_web_tools", None)
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
            "run under the persisted web-tools setting",
            work_dir=self.repo,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert seen["web_tools"] is False, (
            "the persisted use_web_browser=False setting never reached "
            "the agent"
        )

        result = sorcar.run(
            "run with an explicit per-run override",
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=True,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert seen["web_tools"] is True, (
            "the per-run use_web_tools=True override must beat the "
            "persisted setting"
        )

    def test_use_memory_true_beats_persisted_off(self) -> None:
        """An explicit ``use_memory=True`` beats the persisted ``False``.

        With the settings panel's "Use persistent memory" checkbox off
        (config key ``use_memory``), a run WITHOUT an override must stay
        memory-free, and an explicit ``use_memory=True`` on the same
        daemon must build the memory tools anyway (the run hits none of
        the hard gates: basic tools on, no Docker, an API model, no
        caller ``system_instruction``).
        """
        vscode_config.save_config({"use_memory": False})
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen["use_memory"] = getattr(
                self_agent, "_use_memory_override", "MISSING",
            )
            seen["memory_tools"] = getattr(
                self_agent, "_memory_tools", "MISSING",
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
            "run under the persisted memory-off setting",
            work_dir=self.repo,
            use_worktree=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert seen["use_memory"] is None
        assert seen["memory_tools"] is None, (
            "the persisted use_memory=False setting never reached the "
            "agent"
        )

        result = sorcar.run(
            "run with an explicit per-run memory override",
            work_dir=self.repo,
            use_worktree=False,
            use_memory=True,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        assert seen["use_memory"] is True
        assert seen["memory_tools"] is not None, (
            "the per-run use_memory=True override must beat the "
            "persisted setting and build the memory tools"
        )

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
            endpoint_file=self.endpoint_file,
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
            endpoint_file=self.endpoint_file,
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
            endpoint_file=self.endpoint_file,
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

    def test_scope_work_dir_pins_tab_visibility_scope(self) -> None:
        """``run(scope_work_dir=…)`` pins the tab's workspace scope.

        A ``run_agent`` sub-task executes in a channel/cron scratch
        directory (``work_dir``) but must appear in the CALLING
        workspace's tab bar, so ``run()`` forwards ``scope_work_dir``
        as the tab's ``scopeWorkDir`` registry field — distinct from
        the execution ``workDir`` — and it is broadcast to every
        client in the ``tabs_state`` snapshot.
        """
        captured: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0
            self_agent.total_steps = 1
            # Snapshot the registry WHILE the api- tab is still open
            # (run() closes it on exit).
            captured["tabs"] = [
                dict(entry)
                for entry in self.server._vscode_server.tab_registry.snapshot()
                if entry["tabId"].startswith("api-")
            ]
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(raw, type="result", step_count=1)
            return raw

        self._parent_class.run = stub_run
        scratch = str(Path(self.tmpdir) / "channel_work")
        workspace = str(Path(self.tmpdir) / "caller_workspace")
        result = sorcar.run(
            "say hi",
            work_dir=scratch,
            scope_work_dir=workspace,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        api_tabs = captured.get("tabs") or []
        assert len(api_tabs) == 1, f"expected one api tab, got {api_tabs!r}"
        entry = api_tabs[0]
        # The execution dir stays the scratch dir; the visibility
        # scope is pinned to the caller's workspace.
        assert entry["workDir"] == scratch
        assert entry["scopeWorkDir"] == workspace

    def test_scope_survives_agent_script_work_dir_override(self) -> None:
        """A ``settings()['work_dir']`` override re-pins ``workDir``, not the scope.

        The SEA overrides the execution directory via
        ``settings()['work_dir']``, and ``_run_task`` re-pins the
        registry tab's ``workDir`` to the overridden value.  The tab's
        ``scopeWorkDir`` must survive that re-pin — it is what keeps a
        ``run_agent``-dispatched cron tab visible in the CALLING
        workspace — because a script must not disturb the client-sent
        scope.
        """
        captured: dict[str, Any] = {}
        override_dir = str(Path(self.tmpdir) / "script_work")
        agent_script = Path(self.tmpdir) / "scoped_agent.py"
        agent_script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n"
            "\n"
            "class Sea(BaseSea):\n"
            "    def settings(self, settings):\n"
            f"        return settings | {{'work_dir': {override_dir!r}}}\n"
        )

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0
            self_agent.total_steps = 1
            # Snapshot AFTER apply_sea re-pinned workDir
            # (both run on this worker thread before the agent runs).
            captured["tabs"] = [
                dict(entry)
                for entry in self.server._vscode_server.tab_registry.snapshot()
                if entry["tabId"].startswith("api-")
            ]
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(raw, type="result", step_count=1)
            return raw

        self._parent_class.run = stub_run
        workspace = str(Path(self.tmpdir) / "caller_workspace")
        result = sorcar.run(
            "say hi",
            work_dir=str(Path(self.tmpdir) / "initial_work"),
            scope_work_dir=workspace,
            sea_path=str(agent_script),
            use_worktree=False,
            auto_commit=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        api_tabs = captured.get("tabs") or []
        assert len(api_tabs) == 1, f"expected one api tab, got {api_tabs!r}"
        entry = api_tabs[0]
        assert entry["workDir"] == override_dir, (
            "work_dir() must re-pin the registry workDir"
        )
        assert entry["scopeWorkDir"] == workspace, (
            "the workDir re-pin must not clobber the visibility scope"
        )

    def test_scope_work_dir_is_not_an_agent_script_getter(self) -> None:
        """A SEA's ``scope_work_dir()`` is a plain method: the client scope stays.

        The calling workspace recorded on the tab is the caller's
        identity, not the script's, so the dispatch handler's pin from
        the client-sent ``tabScopeWorkDir`` must survive a SEA class
        that happens to define ``scope_work_dir()``.
        """
        captured: dict[str, Any] = {}
        agent_script = Path(self.tmpdir) / "scope_agent.py"
        agent_script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n"
            "\n"
            "class Sea(BaseSea):\n"
            "    def scope_work_dir(self) -> str:\n"
            f"        return {str(Path(self.tmpdir) / 'script_workspace')!r}\n"
        )

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0
            self_agent.total_steps = 1
            captured["tabs"] = [
                dict(entry)
                for entry in self.server._vscode_server.tab_registry.snapshot()
                if entry["tabId"].startswith("api-")
            ]
            raw = "success: true\nis_continue: false\nsummary: ok\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(raw, type="result", step_count=1)
            return raw

        self._parent_class.run = stub_run
        client_scope = str(Path(self.tmpdir) / "client_workspace")
        result = sorcar.run(
            "say hi",
            work_dir=str(Path(self.tmpdir) / "scope_exec_work"),
            scope_work_dir=client_scope,
            sea_path=str(agent_script),
            use_worktree=False,
            auto_commit=False,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True
        api_tabs = captured.get("tabs") or []
        assert len(api_tabs) == 1, f"expected one api tab, got {api_tabs!r}"
        assert api_tabs[0]["scopeWorkDir"] == client_scope, (
            "the client-sent scope must survive a script-level scope_work_dir()"
        )

    def test_no_daemon_raises_connection_error(self) -> None:
        """A missing daemon endpoint file raises a helpful ConnectionError."""
        missing = str(Path(self.tmpdir) / "nowhere.json")
        with self.assertRaises(ConnectionError):
            sorcar.run("hello", endpoint_file=missing, timeout=5)

    def test_blank_prompt_raises_value_error(self) -> None:
        """Blank prompts are rejected before any connection is made."""
        with self.assertRaises(ValueError):
            sorcar.run("   ", endpoint_file=self.endpoint_file, timeout=5)

    def _stub_dispatch_run(self) -> None:
        """Install a parent-run stub that spends a fixed, known amount."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 500
            self_agent.budget_used = 0.25
            self_agent.total_steps = 4
            raw = (
                "success: true\n"
                "is_continue: false\n"
                "summary: dispatched done\n"
            )
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:
                printer.print(
                    raw,
                    type="result",
                    step_count=4,
                    total_tokens=500,
                    cost="$0.2500",
                )
            return raw

        self._parent_class.run = stub_run

    def _dispatch_through_daemon(self, parent: Any) -> str:
        """Dispatch a path-mode ``run_agent`` sub-task through the daemon.

        Args:
            parent: The calling agent passed to ``make_run_agent_tool``
                (``None`` exercises the no-attribution path).

        Returns:
            The tool's YAML result string.
        """
        from kiss.agents.sorcar import cron_agent
        from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool

        # No model(): the daemon's default model applies (the run
        # is stubbed, so no model API call ever happens, but the model
        # name must pass the runner's availability guard).
        script = Path(self.tmpdir) / "noop_dispatch_agent.py"
        script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n"
            "\n"
            "class Sea(BaseSea):\n"
            '    """A SEA that changes nothing."""\n'
        )
        saved_endpoint = cron_agent._daemon_endpoint_file
        cron_agent._daemon_endpoint_file = self.endpoint_file
        try:
            tool = make_run_agent_tool(self.repo, parent)
            return tool("do nothing", str(script))
        finally:
            cron_agent._daemon_endpoint_file = saved_endpoint

    def test_run_agent_tool_attributes_subtask_cost_to_parent(self) -> None:
        """A dispatched sub-task's spend folds into the calling agent.

        Reproduces the cost-accounting bug where ``run_agent`` sub-task
        cost/tokens/steps (returned by the daemon's terminal ``result``
        event) were discarded, so the calling task's end-of-task cost
        under-reported.  The parent's counters must grow by exactly the
        sub-task's reported spend.
        """
        self._stub_dispatch_run()
        parent = SorcarAgent("dispatch-parent")
        parent.budget_used = 0.1
        parent.total_tokens_used = 10
        parent.total_steps = 1

        out = self._dispatch_through_daemon(parent)

        assert "dispatched done" in out
        assert abs(parent.budget_used - 0.35) < 1e-9
        assert parent.total_tokens_used == 510
        assert parent.total_steps == 5

    def test_run_agent_tool_without_parent_attributes_nothing(self) -> None:
        """Standalone use (no calling agent) still works.

        ``make_run_agent_tool`` builds the tool with no parent agent;
        the dispatch must succeed without any attribution attempt.
        """
        self._stub_dispatch_run()
        out = self._dispatch_through_daemon(None)
        assert "dispatched done" in out
