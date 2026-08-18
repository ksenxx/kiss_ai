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
command, and supplies the agent's channel tools through the API's
``tools=`` *file path* contract: the agent's OWN module is the tools
file, and the daemon imports it and calls its top-level ``get_tools()``
to build a fresh agent from the credentials persisted under the
active kiss home.  No bridge, registry, wrapper, or generated file is
involved.  The task is executed by a daemon-built chat agent, NOT by
the passed instance.

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
import os
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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebChatAgent,
    run_agent_via_kiss_web,
)
from kiss.core import vscode_config
from kiss.server import agent_state
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

        self._saved_sock_override = launcher._SOCK_PATH_OVERRIDE
        launcher._SOCK_PATH_OVERRIDE = self.sock_path

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.stub_calls: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        launcher._SOCK_PATH_OVERRIDE = self._saved_sock_override
        self._parent_class.run = self._original_run
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
        assert "Slack Authentication" in prompt
        assert "start_slack_browser_auth" in prompt

    def test_agent_module_is_the_tools_file(self) -> None:
        from kiss.agents.third_party_agents import slack_agent
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        agent = SlackAgent()
        assert agent.tools_file == str(slack_agent.__file__), (
            "the agent's own module must be its tools file"
        )

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            for expected in (
                "check_slack_auth",
                "authenticate_slack",
                "clear_slack_auth",
                "start_slack_browser_auth",
            ):
                assert expected in tools, f"missing channel tool {expected}"
            auth_out = tools["check_slack_auth"]()
            assert "Not authenticated with Slack" in auth_out
            return "module tools loaded ok"

        self._install_stub(on_run=on_run)
        # The channel agents persist credentials under ``~/.kiss`` (a
        # ``Path.home()``-based path, re-evaluated when the daemon
        # re-executes the module as the task's tools file), so point
        # HOME at this test's empty tmpdir for the run: the tools must
        # observe the deterministic "not authenticated" state, not the
        # developer machine's real Slack credentials.
        saved_home = os.environ.get("HOME")
        os.environ["HOME"] = self.tmpdir
        try:
            result = run_agent_via_kiss_web(
                agent,
                "use the tools",
                work_dir=self.repo,
                sock_path=self.sock_path,
            )
        finally:
            if saved_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = saved_home
        assert yaml.safe_load(result)["summary"] == "module tools loaded ok"

    def test_explicit_tools_file_overrides_agent_module(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        tools_py = Path(self.tmpdir) / "extra_tools.py"
        tools_py.write_text(
            "from pathlib import Path\n"
            f"_LOG = Path({str(Path(self.tmpdir) / 'tool_calls.log')!r})\n"
            "\n"
            "def mytool(text: str, repeat: int = 1) -> str:\n"
            '    """Echo *text* repeated *repeat* times.\n'
            "\n"
            "    Args:\n"
            "        text: The text to echo.\n"
            "        repeat: How many times to repeat it.\n"
            '    """\n'
            "    with _LOG.open('a') as f:\n"
            "        f.write(text + '\\n')\n"
            "    return text * repeat\n"
            "\n"
            "def get_tools():\n"
            '    """Return the tools the agent may call."""\n'
            "    return [mytool]\n",
            encoding="utf-8",
        )

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            assert set(tools) >= {"mytool"}, "explicit tools file must win"
            assert "check_slack_auth" not in tools, (
                "an explicit tools= path must replace the agent module"
            )
            assert "Echo *text* repeated" in (tools["mytool"].__doc__ or "")
            assert tools["mytool"](text="hi") == "hi"
            assert tools["mytool"]("bye", repeat=2) == "byebye"
            return "explicit tools ok"

        self._install_stub(on_run=on_run)
        result = run_agent_via_kiss_web(
            SlackAgent(),
            "use the tools",
            work_dir=self.repo,
            tools=str(tools_py),
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == "explicit tools ok"
        log = Path(self.tmpdir) / "tool_calls.log"
        assert log.read_text().splitlines() == ["hi", "bye"], (
            "tools-file tools must really execute in the daemon process"
        )

    def test_backend_tools_included_when_authenticated(self) -> None:
        notes = Path(self.tmpdir) / "notes.log"
        agent_py = Path(self.tmpdir) / "note_agent.py"
        agent_py.write_text(
            "from pathlib import Path\n"
            "\n"
            "from kiss.agents.third_party_agents._channel_agent_utils import (\n"
            "    BaseChannelAgent,\n"
            "    ToolMethodBackend,\n"
            ")\n"
            "\n"
            f"_NOTES = Path({str(notes)!r})\n"
            "\n"
            "\n"
            "class NoteBackend(ToolMethodBackend):\n"
            "    def add_note(self, note: str) -> str:\n"
            '        """Record a note in the persistent notes file.\n'
            "\n"
            "        Args:\n"
            "            note: The note text to record.\n"
            '        """\n'
            "        with _NOTES.open('a') as f:\n"
            "            f.write(note + '\\n')\n"
            "        return f'recorded:{note}'\n"
            "\n"
            "\n"
            "class NoteAgent(BaseChannelAgent):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__('Backend Test Agent')\n"
            "        self._backend = NoteBackend()\n"
            "\n"
            "    def _is_authenticated(self) -> bool:\n"
            "        return True\n"
            "\n"
            "    def _get_auth_tools(self) -> list:\n"
            "        return []\n"
            "\n"
            "\n"
            "def get_tools() -> list:\n"
            '    """Return the note-channel tools."""\n'
            "    return NoteAgent()._get_tools()\n",
            encoding="utf-8",
        )

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            assert "add_note" in tools, (
                "the authenticated backend's tool must come from the "
                "module's get_tools()"
            )
            return str(tools["add_note"](note="from daemon"))

        self._install_stub(on_run=on_run)
        result = run_agent_via_kiss_web(
            KissWebChatAgent("Note Launch"),
            "note task",
            work_dir=self.repo,
            tools=str(agent_py),
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == "recorded:from daemon"
        assert notes.read_text().splitlines() == ["from daemon"], (
            "backend tools must act on state shared through persistence, "
            "not on the launcher-side instance"
        )

    def test_workspace_env_var_set_while_task_runs(self) -> None:
        import os

        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        seen: list[str | None] = []

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            seen.append(os.environ.get("KISS_CHANNEL_WORKSPACE"))
            return "ws ok"

        self._install_stub(on_run=on_run)
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None
        run_agent_via_kiss_web(
            SlackAgent(workspace="teamspace"),
            "ws task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert seen == ["teamspace"], (
            "the daemon-side get_tools() must see the launch workspace"
        )
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None, (
            "the launcher must restore the workspace env var"
        )

    def test_workspace_env_var_survives_overlapping_launches(self) -> None:
        import os

        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        release = {"A": threading.Event(), "B": threading.Event()}
        started = {"A": threading.Event(), "B": threading.Event()}

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            key = "A" if "task-A" in str(kwargs.get("prompt_template")) else "B"
            started[key].set()
            release[key].wait(timeout=30)
            return f"overlap-{key}"

        self._install_stub(on_run=on_run)
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None

        def launch(key: str, workspace: str) -> None:
            run_agent_via_kiss_web(
                SlackAgent(workspace=workspace),
                f"task-{key}",
                work_dir=self.repo,
                sock_path=self.sock_path,
            )

        thread_a = threading.Thread(target=launch, args=("A", "wsA"), daemon=True)
        thread_b = threading.Thread(target=launch, args=("B", "wsB"), daemon=True)
        try:
            thread_a.start()
            assert started["A"].wait(timeout=30)
            thread_b.start()
            assert started["B"].wait(timeout=30)
            # A finishes first while B is still running: the env var
            # must keep naming an ACTIVE workspace (B's), not be popped
            # or reset to A's out-of-order snapshot.
            release["A"].set()
            thread_a.join(timeout=30)
            assert not thread_a.is_alive()
            assert os.environ.get("KISS_CHANNEL_WORKSPACE") == "wsB", (
                "finishing one launch must not clobber the env var of a "
                "still-running launch"
            )
        finally:
            release["A"].set()
            release["B"].set()
            thread_a.join(timeout=30)
            thread_b.join(timeout=30)
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None, (
            "the env var must be removed once every launch has finished"
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

    def test_append_basic_tools_false_forwarded(self) -> None:
        """``append_basic_tools=False`` reaches the daemon-built agent.

        The launcher (and the ``LAUNCH_KWARG_NAMES`` filter the channel
        agents' ``run()`` shims pass their kwargs through) must forward
        the restriction to ``sorcar.run`` — a dropped kwarg would
        silently hand the channel task the full basic toolset.
        """
        from kiss.agents.third_party_agents._channel_agent_utils import (
            filter_launch_kwargs,
        )
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        assert filter_launch_kwargs(
            {"append_basic_tools": False, "system_prompt": "dropped"}
        ) == {"append_basic_tools": False}
        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
            append_basic_tools=False,
        )
        agent = self.stub_calls[0]["agent"]
        assert getattr(agent, "_append_basic_tools", None) is False

    def test_append_to_prompts_forwarded(self) -> None:
        """Both append suffixes reach the daemon-built agent's run.

        The launcher (and the ``LAUNCH_KWARG_NAMES`` filter) must
        forward ``append_to_system_prompt`` / ``append_to_prompt`` to
        ``sorcar.run`` — a dropped kwarg would silently run the channel
        task without the caller's extra instructions.  The stub sits on
        ``RelentlessAgent.run``, so its ``system_prompt`` kwarg is the
        fully assembled system instructions (base + suffix) and its
        ``prompt_template`` the executed prompt (task + channel
        guidance + suffix).
        """
        from kiss.agents.third_party_agents._channel_agent_utils import (
            filter_launch_kwargs,
        )
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        assert filter_launch_kwargs(
            {
                "append_to_system_prompt": "S",
                "append_to_prompt": "P",
                "_skip_persistence": True,
            }
        ) == {"append_to_system_prompt": "S", "append_to_prompt": "P"}
        self._install_stub()
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            sock_path=self.sock_path,
            append_to_system_prompt="\nLAUNCHER-SYS-SUFFIX-2210",
            append_to_prompt="\nLAUNCHER-PROMPT-SUFFIX-2210",
        )
        kwargs = self.stub_calls[0]["kwargs"]
        assert "LAUNCHER-SYS-SUFFIX-2210" in str(
            kwargs.get("system_prompt", "")
        )
        assert "LAUNCHER-PROMPT-SUFFIX-2210" in str(
            kwargs.get("prompt_template", "")
        )

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
        agent = KissWebChatAgent("Blank Prompt Agent")
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
            for state in agent_state.snapshot():
                if state.task_thread is not None:
                    state.task_thread.join(timeout=deadline)

    def test_invalid_tools_file_raises_before_connecting(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        with self.assertRaises(ValueError):
            run_agent_via_kiss_web(
                SlackAgent(),
                "task",
                work_dir=self.repo,
                tools=str(Path(self.tmpdir) / "missing_tools.py"),
                sock_path=self.sock_path,
            )
        assert not self.stub_calls, "no task may start for a bad tools file"


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
    """``run()`` on the carrier agents routes through the daemon API.

    The carriers are not executable agents: ``run()`` submits the task
    to the kiss-web daemon via ``kiss.server.sorcar.run`` and records
    the returned YAML, so a crashing daemon-built agent surfaces as a
    failure envelope, never as a re-raised exception.
    """

    def test_chat_agent_direct_run_records_result(self) -> None:
        self._install_stub(summary="direct chat ok")
        agent = KissWebChatAgent("Direct Chat")
        result = agent.run(
            prompt_template="direct task", work_dir=self.repo,
        )
        assert yaml.safe_load(result)["summary"] == "direct chat ok"
        assert agent.last_run_result == result
        call = self.stub_calls[0]
        assert call["agent"] is not agent, (
            "carrier run() must execute on a daemon-built agent"
        )

    def test_chat_agent_direct_run_records_failure(self) -> None:
        self._install_stub(raise_exc=RuntimeError("direct-boom"))
        agent = KissWebChatAgent("Direct Chat")
        result = agent.run(
            prompt_template="direct task", work_dir=self.repo,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert str(parsed["summary"]).strip(), (
            "a crashed task must not produce an empty summary"
        )
        assert agent.last_run_result == result

    def test_direct_run_with_use_worktree_false_records_result(self) -> None:
        self._install_stub(summary="direct wt ok")
        agent = KissWebChatAgent("Direct No-WT")
        result = agent.run(
            prompt_template="direct task",
            work_dir=self.repo,
            use_worktree=False,
        )
        assert yaml.safe_load(result)["summary"] == "direct wt ok"
        assert agent.last_run_result == result

    def test_direct_run_with_use_worktree_false_records_failure(self) -> None:
        self._install_stub(raise_exc=RuntimeError("wt-boom"))
        agent = KissWebChatAgent("Direct No-WT")
        result = agent.run(
            prompt_template="direct task",
            work_dir=self.repo,
            use_worktree=False,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert str(parsed["summary"]).strip(), (
            "a crashed task must not produce an empty summary"
        )
        assert agent.last_run_result == result


class TestBaseChannelAgentDirectRuns(_ApiLaunchBase):
    """Channel agent ``run()`` calls route through the daemon API."""

    def _plain_agent(self) -> Any:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            BaseChannelAgent,
        )

        class _Plain(BaseChannelAgent):
            def _is_authenticated(self) -> bool:
                return False

            def _get_auth_tools(self) -> list:
                return []

        return _Plain("Plain Direct Agent")

    def test_direct_run_appends_channel_prompt_to_prompt(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        self._install_stub()
        agent = SlackAgent()
        result = agent.run(
            prompt_template="direct slack",
            work_dir=self.repo,
            use_worktree=True,
            _skip_persistence=True,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result
        call = self.stub_calls[0]
        assert call["agent"] is not agent, (
            "channel agent run() must execute on a daemon-built agent"
        )
        prompt = str(call["kwargs"].get("prompt_template", ""))
        assert "direct slack" in prompt
        assert "Slack Authentication" in prompt

    def test_direct_run_without_channel_prompt(self) -> None:
        self._install_stub()
        agent = self._plain_agent()
        result = agent.run(
            prompt_template="direct plain", work_dir=self.repo,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        prompt = str(
            self.stub_calls[0]["kwargs"].get("prompt_template", ""),
        )
        assert "direct plain" in prompt
        assert "## Slack Authentication" not in prompt

    def test_direct_run_failure_returns_failure_yaml(self) -> None:
        self._install_stub(raise_exc=RuntimeError("plain-boom"))
        agent = self._plain_agent()
        result = agent.run(
            prompt_template="direct plain", work_dir=self.repo,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert str(parsed["summary"]).strip(), (
            "a crashed task must not produce an empty summary"
        )
        assert agent.last_run_result == result

    def test_direct_run_bridges_channel_auth_tools(self) -> None:
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        def on_run(self_agent, kwargs):
            names = {t.__name__ for t in (kwargs.get("tools") or [])}
            assert "check_slack_auth" in names
            return "auth tools bridged"

        self._install_stub(on_run=on_run)
        agent = SlackAgent()
        result = agent.run(
            prompt_template="direct slack tools", work_dir=self.repo,
        )
        assert yaml.safe_load(result)["summary"] == "auth tools bridged"


class TestKissWebChatCarrierAgents(_ApiLaunchBase):
    """The chat-id carrier agents used by the channel runner."""

    def test_chat_agent_gets_daemon_chat_id(self) -> None:
        self._install_stub()
        agent = KissWebChatAgent("Test Carrier")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "carrier task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result
        assert agent.chat_id, (
            "the daemon-minted chat id must be propagated onto the "
            "carrier agent so the channel runner can resume the thread later"
        )

    def test_chat_agent_resume_chat_by_id(self) -> None:
        self._install_stub()
        agent = KissWebChatAgent("Test Carrier")
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
        resumed = KissWebChatAgent("Test Carrier")
        resumed.resume_chat_by_id(first_chat)
        run_agent_via_kiss_web(
            resumed,
            "second task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert resumed.chat_id == first_chat, "existing chat id must be kept"
        assert "first task" in prompts[0], (
            "the resumed run must see the prior task as chat context"
        )

    def test_carrier_records_result_with_new_chat(self) -> None:
        self._install_stub()
        agent = KissWebChatAgent("Test Carrier 2")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.repo,
            sock_path=self.sock_path,
        )
        assert yaml.safe_load(result)["summary"] == STUB_SUMMARY
        assert agent.last_run_result == result


class TestChannelRunnerViaApi(_ApiLaunchBase):
    """ChannelRunner._handle_message must launch through the API."""

    def _make_runner(
        self,
        tools_file: str = "",
        thread_replies: list[dict[str, Any]] | None = None,
    ) -> Any:
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

            def is_from_bot(self, msg: dict[str, Any]) -> bool:
                return bool(msg.get("bot_id"))

            def disconnect(self) -> None:
                pass

        backend = _FakeBackend()
        if thread_replies is not None:
            def poll_thread_messages(
                channel_id: str,
                thread_ts: str,
                oldest: str,
                limit: int = 100,
            ) -> tuple[list[dict[str, Any]], str]:
                return list(thread_replies), "0"

            backend.poll_thread_messages = (  # type: ignore[attr-defined]
                poll_thread_messages
            )
        runner = ChannelRunner(
            backend=backend,
            channel_name="chan",
            agent_name="Test Channel Agent",
            tools_file=tools_file,
            work_dir=str(Path(self.tmpdir) / "chanwork"),
        )
        return runner, outbox

    def test_handle_message_passes_tools_file_and_context(self) -> None:
        tools_py = Path(self.tmpdir) / "chan_tools.py"
        tools_py.write_text(
            "def shout(text: str) -> str:\n"
            '    """Return *text* uppercased.\n'
            "\n"
            "    Args:\n"
            "        text: The text to uppercase.\n"
            '    """\n'
            "    return text.upper()\n"
            "\n"
            "def get_tools():\n"
            '    """Return the channel tools."""\n'
            "    return [shout]\n",
            encoding="utf-8",
        )
        runner, outbox = self._make_runner(tools_file=str(tools_py))

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            tools = {t.__name__: t for t in (kwargs.get("tools") or [])}
            assert "shout" in tools, (
                "the runner's tools file must supply the task's tools"
            )
            assert tools["shout"](text="hi") == "HI"
            prompt = str(kwargs.get("prompt_template", ""))
            assert "'C123'" in prompt and "'1.0'" in prompt, (
                "the prompt must carry the channel/thread context"
            )
            assert "posted to the thread automatically" in prompt, (
                "a backend without thread polling cannot suppress the "
                "summary, so the agent must be told not to self-post"
            )
            assert "no automatic summary" not in prompt
            return "channel tools ok"

        self._install_stub(on_run=on_run)
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert outbox == [("C123", "channel tools ok", "1.0")], (
            "the summary must be posted as the thread reply"
        )

    def test_handle_message_skips_summary_when_bot_replied(self) -> None:
        runner, outbox = self._make_runner(
            thread_replies=[
                {"user": "U_BOT", "bot_id": "B1", "ts": "1.5", "text": "done"},
            ],
        )
        self._install_stub(summary="already answered in-thread")
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert outbox == [], (
            "no summary reply may be posted when the agent already "
            "replied in the thread with its channel tools"
        )

    def test_context_promises_suppression_only_with_thread_polling(self) -> None:
        tools_py = Path(self.tmpdir) / "noop_tools.py"
        tools_py.write_text(
            "def ping() -> str:\n"
            '    """Return pong."""\n'
            "    return 'pong'\n"
            "\n"
            "def get_tools():\n"
            '    """Return the channel tools."""\n'
            "    return [ping]\n",
            encoding="utf-8",
        )
        runner, outbox = self._make_runner(
            tools_file=str(tools_py), thread_replies=[],
        )

        def on_run(self_agent: Any, kwargs: dict[str, Any]) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            assert "no automatic summary reply is sent" in prompt, (
                "a thread-polling backend suppresses duplicate summaries, "
                "so the agent may be promised suppression"
            )
            return "suppression promised"

        self._install_stub(on_run=on_run)
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert outbox == [("C123", "suppression promised", "1.0")], (
            "with no bot reply in the thread the summary is still posted"
        )

    def test_handle_message_sends_summary_when_no_reply(self) -> None:
        runner, outbox = self._make_runner()
        self._install_stub(summary="channel summary")
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert outbox == [("C123", "channel summary", "1.0")]

    def test_handle_message_agent_error_sends_error_reply(self) -> None:
        runner, outbox = self._make_runner()
        self._install_stub(summary="chan-blast happened", success=False)
        runner._handle_message("C9", {"text": "x", "ts": "2.0"})
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
            tree = ast.parse(source, filename=str(py))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "run"):
                    continue
                base = func.value
                if (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "super"
                ):
                    continue
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
