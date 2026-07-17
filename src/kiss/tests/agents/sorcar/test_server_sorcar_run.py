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
import functools
import inspect
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
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

    def _raw_daemon_run(
        self,
        tools: Any,
        pre_commands: list[dict[str, Any]] | None = None,
    ) -> None:
        """Drive one raw ``run`` command over the UDS and wait for the end.

        Bypasses :func:`kiss.server.sorcar.run` so malformed ``tools``
        payloads (and stray ``toolResponse`` commands) can be sent
        exactly as an arbitrary/buggy client would.

        Args:
            tools: Raw value for the ``run`` command's ``tools`` field.
            pre_commands: Extra commands to send before the ``run``.
        """
        tab_id = f"raw-{uuid.uuid4().hex}"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        try:
            sock.connect(self.sock_path)
            for pre in pre_commands or []:
                sock.sendall(json.dumps(pre).encode() + b"\n")
            cmd = {
                "type": "run",
                "prompt": "raw client task",
                "tabId": tab_id,
                "taskId": uuid.uuid4().hex,
                "workDir": self.repo,
                "model": "",
                "tools": tools,
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

    def test_tools_round_trip_end_to_end(self) -> None:
        """Client tools are rebuilt as agent tools and proxied back.

        The agent (daemon task thread) must see one proxy per client
        tool with the original name, docstring, and signature; calling
        a proxy must execute the caller's function in the client
        process and return its ``str()``-ified result to the agent.
        """
        client_thread = threading.current_thread()
        calls: list[tuple[str, str, threading.Thread]] = []

        def get_temperature(city: str, unit: str = "C", *, note: str = "") -> str:
            """Return the current temperature of a city.

            Args:
                city: Name of the city to look up.
                unit: Temperature unit to report.
                note: Optional note echoed back.
            """
            calls.append((city, unit, threading.current_thread()))
            # Exceed the daemon proxy's 0.25 s poll interval so its
            # ``queue.Empty`` → re-poll path is exercised for real.
            time.sleep(0.4)
            return f"21{unit} in {city}{note}"

        def magic_number(seed: int, factor=2, opts=()) -> int:  # type: ignore[no-untyped-def]
            return int(seed * factor + len(opts))

        def tag(prefix: str = ">", *, label: str) -> str:
            """Tag a label.

            Args:
                prefix: Prefix to prepend.
                label: Label to tag.
            """
            return prefix + label

        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            proxies = {t.__name__: t for t in kwargs.get("tools") or []}
            seen["names"] = sorted(proxies)
            temp = proxies["get_temperature"]
            seen["doc"] = inspect.getdoc(temp)
            seen["signature"] = str(inspect.signature(temp))
            seen["magic_doc"] = inspect.getdoc(proxies["magic_number"])
            seen["magic_sig"] = str(inspect.signature(proxies["magic_number"]))
            seen["tag_sig"] = str(inspect.signature(proxies["tag"]))
            # Positional call, keyword call with the optional param,
            # and an int-returning tool (round-trips as a string).
            seen["r1"] = temp("Paris")
            seen["r2"] = temp(city="Berlin", unit="F")
            seen["r3"] = proxies["magic_number"](seed=20)
            # Required keyword-only after an optional parameter — a
            # legal Python signature the proxy must reproduce exactly.
            seen["r4"] = proxies["tag"](label="urgent")
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
            tools=[get_temperature, magic_number, tag],
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert result.text == "tools ok"
        assert seen["names"] == ["get_temperature", "magic_number", "tag"]
        # Name / docstring / signature fidelity (docstring parameter
        # descriptions feed the model's tool schema).
        assert seen["doc"] == inspect.getdoc(get_temperature)
        # (The proxy preserves the keyword-only kind of ``note`` and
        # carries no return annotation.)
        assert seen["signature"] == "(city: str, unit: str = 'C', *, note: str = '')"
        # A required keyword-only parameter after an optional one is a
        # valid signature and must survive the round trip unchanged.
        assert seen["tag_sig"] == "(prefix: str = '>', *, label: str)"
        assert seen["r4"] == ">urgent"
        # Undocumented tool gets a fallback docstring; an unannotated
        # parameter degrades to ``str`` and a non-JSON-primitive
        # default (``()``) is transmitted as ``None`` — the client
        # function's real default still applies because omitted
        # arguments are never forwarded.
        assert seen["magic_doc"] == "Client tool magic_number."
        assert seen["magic_sig"] == "(seed: int, factor: str = 2, opts: str = None)"
        assert seen["r1"] == "21C in Paris"
        assert seen["r2"] == "21F in Berlin"
        assert seen["r3"] == "40"
        # The tools genuinely ran in the CLIENT thread (inside
        # ``sorcar.run``'s event loop), not on the daemon task thread.
        assert [c[:2] for c in calls] == [("Paris", "C"), ("Berlin", "F")]
        assert all(c[2] is client_thread for c in calls)

    def test_tool_exception_returns_error_string(self) -> None:
        """A raising client tool yields an ``Error: ...`` result string."""

        def explode(reason: str) -> str:
            """Blow up.

            Args:
                reason: Why to blow up.
            """
            raise RuntimeError(f"boom {reason}")

        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            (proxy,) = cast(list[Any], kwargs.get("tools"))
            seen["result"] = proxy(reason="now")
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
            "use the exploding tool",
            work_dir=self.repo,
            tools=[explode],
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        assert seen["result"] == "Error: tool 'explode' failed: RuntimeError: boom now"

    def test_stopped_task_cancels_pending_tool_call(self) -> None:
        """A stopped task unblocks a pending proxy call with an error.

        Also exercises the stale-``toolResponse`` drop: the client
        still answers the broadcast ``toolRequest``, but by then the
        pending entry is gone and the reply must be ignored.
        """

        def slow_tool(x: str) -> str:
            """Echo.

            Args:
                x: Value to echo.
            """
            return x

        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            tab = _RunningAgentState.running_agent_states[self_agent._tab_id]
            assert tab.stop_event is not None
            tab.stop_event.set()
            (proxy,) = cast(list[Any], kwargs.get("tools"))
            seen["result"] = proxy(x="hi")
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
            "stop mid tool call",
            work_dir=self.repo,
            tools=[slow_tool],
            sock_path=self.sock_path,
            timeout=60,
        )
        # The proxy unblocked with the cancellation error instead of
        # hanging, and the daemon reported the run as user-stopped.
        assert seen["result"] == (
            "Error: tool call 'slow_tool' was cancelled "
            "because the task was stopped."
        )
        assert result.success is False
        assert result.text == "Task stopped by user"

    def test_malformed_tools_and_tool_responses_are_ignored(self) -> None:
        """Malformed specs are skipped; bad ``toolResponse`` is harmless.

        A hand-crafted client sends a ``run`` whose ``tools`` list
        mixes malformed entries with two well-formed specs, preceded by
        stray/malformed ``toolResponse`` commands.  The daemon must
        build proxies only for the valid specs and stay alive.
        """
        seen: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.setdefault("tool_names", []).append(
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
        self._raw_daemon_run(
            tools=[
                42,  # not a dict
                {"name": "not an identifier"},  # invalid tool name
                {"name": "dup_params", "params": [{"name": "x"}, {"name": "x"}]},
                {"name": "bad_param", "params": [{"name": "1bad"}]},
                # ``class`` passes ``isidentifier`` but is rejected by
                # ``inspect.Parameter`` (reserved word).
                {"name": "kw_param", "params": [{"name": "class"}]},
                # Required parameter after an optional one with no
                # keyword-only marker: invalid rebuilt signature.
                {"name": "bad_order", "params": [{"name": "a", "default": 1},
                                                 {"name": "b"}]},
                {"name": "junk_params", "params": "junk", "description": ""},
                {
                    "name": "good",
                    "description": "A good tool.",
                    "params": [{"name": "a", "type": "int", "default": 1}],
                },
                # An unrecognized ``kind`` value degrades to a plain
                # keyword-bindable parameter.
                {
                    "name": "odd_kind",
                    "params": [{"name": "a", "kind": 42, "default": 0}],
                },
            ],
            pre_commands=[
                {"type": "toolResponse"},  # no callId
                {"type": "toolResponse", "callId": 7, "result": "x"},
                {"type": "toolResponse", "callId": "stale", "result": "x"},
            ],
        )
        # Non-list ``tools`` field: every spec ignored, task still runs.
        self._raw_daemon_run(tools="junk")
        # Absent/null ``tools`` field (plain webview submits): no
        # proxies, no warning.
        self._raw_daemon_run(tools=None)
        assert seen["tool_names"] == [["junk_params", "good", "odd_kind"], [], []]
        # The daemon survived it all: a normal API run still works.
        result = sorcar.run(
            "still alive?",
            work_dir=self.repo,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True

    def test_invalid_tools_raise_value_error(self) -> None:
        """Unsupported tool shapes are rejected before connecting."""

        def dup(a: str) -> str:
            """Dup.

            Args:
                a: A value.
            """
            return a

        def star_args(*args: str) -> str:
            """Star args."""
            return ""

        cases: list[Any] = [
            [42],  # not callable
            [lambda x: x],  # no valid __name__
            [functools.partial(dup, a="x")],  # no __name__ at all
            [dup, dup],  # duplicate tool name
            [star_args],  # unsupported parameter kind
        ]
        for tools in cases:
            with self.assertRaises(ValueError):
                sorcar.run(
                    "hello",
                    tools=tools,
                    sock_path=self.sock_path,
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
