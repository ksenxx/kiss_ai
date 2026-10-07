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
connects to a daemon's local endpoint, sends the documented ``run``
command, and supplies the agent's channel tools through the API's
``sea_path`` SEA contract: the agent's OWN
module is the SEA, and the daemon imports it and calls its
SEA class's ``tools()`` to build a fresh agent from the
credentials persisted under the active kiss home.  No bridge,
registry, wrapper, or generated file is involved.  The task is executed
by a daemon-built chat agent, NOT by the passed instance.

Test strategy (no mocks)
------------------------
A real :class:`kiss.server.web_server.RemoteAccessServer` is served on
a temporary loopback WSS endpoint (the production local transport) with
isolated persistence/config, and every launched task runs the REAL
agent loop.  The only thing outside the process is the LLM provider,
which is replaced by a scripted OpenAI-compatible HTTP endpoint
(:class:`LaunchModelServer`, built on the shared
``parallel_agent_harness`` stand-in): it records every chat-completions
request the daemon-built agent sends and answers the tool calls the test
scripted.  Everything a test asserts is therefore something the real
run produced: the ``tools`` schema list and the system / task messages
of the model request, the tool-result messages of the following
request, the ``finish`` summary the launcher returned, the usage the
agent accounted, and the daemon's registered task state.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
import unittest
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import yaml

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebChatAgent,
    run_agent_via_kiss_web,
)
from kiss.core.models import model_info
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    _PROVIDER_KEYS,
    STANDIN_MODEL,
    IsolatedKissHome,
    _StandInHandler,
    finish_response,
    request_text,
    tool_call_response,
)

SUMMARY = "scripted summary done"
# Deliberate per-call usage so token accounting is checkable exactly.
USAGE = {"prompt_tokens": 1000, "completion_tokens": 234, "total_tokens": 1234}


def _failed_finish(summary: str) -> dict[str, Any]:
    """Build a completion whose ``finish`` call reports ``success=False``."""
    return tool_call_response(
        "finish", {"success": "false", "summary_in_html": summary},
    )


def _tool_names(request: dict[str, Any]) -> list[str]:
    """Return the function names of a chat-completions *request*'s ``tools``."""
    return [t["function"]["name"] for t in request.get("tools", [])]


def _system_text(request: dict[str, Any]) -> str:
    """Return the system message of *request* (the run's system prompt)."""
    first = request["messages"][0]
    assert first["role"] == "system", first["role"]
    return str(first["content"])


def _task_text(request: dict[str, Any]) -> str:
    """Return the user messages of *request* (the executed prompt)."""
    return "\n".join(
        str(m.get("content"))
        for m in request["messages"]
        if m["role"] == "user"
    )


def _tool_results(request: dict[str, Any]) -> list[str]:
    """Return the tool-result messages of *request* (what the tools returned).

    The agent appends its step/budget footer to every tool result, so
    callers match the tool's own output with ``startswith``.
    """
    return [
        str(m.get("content")) for m in request["messages"] if m["role"] == "tool"
    ]


def _only_tool_result(request: dict[str, Any]) -> str:
    """Return the single tool-result message of *request*."""
    results = _tool_results(request)
    assert len(results) == 1, results
    return results[0]


def _restore_env(name: str, value: str | None) -> None:
    """Put environment variable *name* back to *value* (``None`` = unset)."""
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _call_and_store(fn: Callable[[], str], out: list[str]) -> None:
    """Thread target: append ``fn()``'s return value to *out*."""
    out.append(fn())


def _budget_line(request: dict[str, Any]) -> float:
    """Return the ``Max budget (USD)`` the run's system prompt declares."""
    match = re.search(r"Max budget \(USD\): \$([0-9.]+)", _system_text(request))
    assert match is not None, "the system prompt must declare the budget"
    return float(match.group(1))


class _LaunchHandler(_StandInHandler):
    """The harness handler plus request-header capture and scripted HTTP errors."""

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Record the headers, then fail with ``fail_status`` or answer normally."""
        model_server = cast(Any, self.server).launch
        model_server.headers.append(
            {k.lower(): v for k, v in self.headers.items()}
        )
        if model_server.fail_status:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_error(model_server.fail_status, "scripted model failure")
            return
        super().do_POST()


class LaunchModelServer:
    """Scripted OpenAI-compatible endpoint the launched tasks talk to.

    ``script`` holds the completions answered in order; once it is
    exhausted every call gets ``finish(SUMMARY)``.  ``requests`` and
    ``headers`` record each call.  ``hold()`` makes calls wait until
    ``release()``; ``fail_status`` makes the endpoint answer that HTTP
    status instead of a completion.  ``answer`` may be replaced by a
    test that needs to route on the request's content.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.script: list[dict[str, Any]] = []
        self.fail_status = 0
        self.seen = threading.Event()
        self.answer: Callable[[dict[str, Any]], dict[str, Any]] = self._scripted
        self._holding = False
        self._gate = threading.Event()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _LaunchHandler)
        cast(Any, server).launch = self
        cast(Any, server).responder = self._record_and_answer
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """Return the ``/v1`` base URL of this endpoint."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the ``model_config`` that points a run at this endpoint."""
        return {"base_url": self.url, "api_key": "kiss-test-key"}

    def hold(self) -> None:
        """Make every following call wait until :meth:`release`."""
        self._holding = True

    def release(self) -> None:
        """Let held calls proceed."""
        self._gate.set()

    def stop(self) -> None:
        """Release held calls, stop serving and join the server thread."""
        self.release()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _record_and_answer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        self.seen.set()
        if self._holding:
            assert self._gate.wait(timeout=60), "held model call never released"
        return self.answer(request)

    def _scripted(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.script.pop(0) if self.script else finish_response(SUMMARY)


class _ApiLaunchBase(unittest.TestCase):
    """Real daemon over a temp local endpoint; the LLM is the scripted endpoint."""

    def setUp(self) -> None:
        # Every global mutation registers its restoration with
        # ``addCleanup`` immediately: cleanups run (in LIFO order) even
        # when ``setUp`` itself fails partway, unlike ``tearDown``.
        self.home = IsolatedKissHome(prefix="kiss-tp-api-launch-")
        self.addCleanup(self.home.cleanup)
        # Runs before ``home.cleanup``: stops the event writer and
        # invalidates every thread's cached connection, so the harness
        # finds no live connection to close raw under another thread.
        self.addCleanup(_persistence._close_db)
        self.tmpdir = str(self.home.tmpdir)
        self.repo = str(self.home.repo)
        self.endpoint_file = str(self.home.tmpdir / "sorcar-local.json")
        # The daemon imports each channel agent's module as the task's
        # SEA, so credential paths under ``~`` are re-evaluated
        # there: point HOME at the empty tmpdir so every test observes
        # the deterministic "not authenticated" state and never the
        # developer machine's real credentials.
        self.addCleanup(_restore_env, "HOME", os.environ.get("HOME"))
        os.environ["HOME"] = self.tmpdir

        self.model_server = LaunchModelServer()
        self.addCleanup(self.model_server.stop)
        # Launches without an explicit model/model_config use the
        # daemon's default model (``last_model``) and the endpoint
        # MY_MODELS.json registers for it: both point at the stand-in.
        self._saved_my_models = model_info.USER_MY_MODELS_PATH
        model_info.USER_MY_MODELS_PATH = self.home.kiss_home / "MY_MODELS.json"
        self.addCleanup(self._restore_my_models)
        error = model_info.save_custom_model(
            STANDIN_MODEL, endpoint=self.model_server.url, api_key="kiss-test-key",
        )
        assert error is None, error
        # No pre-run classifier and no auto-commit: the scripted
        # endpoint then sees exactly the agentic calls.
        self.home.write_config(
            last_model=STANDIN_MODEL, classify_tasks=False, auto_commit_mode=False,
        )

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.addCleanup(self._stop_loop)
        self.server = RemoteAccessServer(
            local_endpoint_file=self.endpoint_file, work_dir=self.repo,
        )
        # Registered before the start is awaited so a start that times
        # out is still shut down.
        self.addCleanup(self._shutdown_server)
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)

        self._saved_endpoint_override = launcher._ENDPOINT_FILE_OVERRIDE
        launcher._ENDPOINT_FILE_OVERRIDE = self.endpoint_file
        self.addCleanup(self._restore_endpoint_override)
        # Registered last so they run first: release any held model
        # call, then join the task workers while the daemon, the model
        # server and the history DB are all still up.
        self.addCleanup(self._join_tasks_and_discard)
        self.addCleanup(self.model_server.release)

    def _restore_my_models(self) -> None:
        model_info.USER_MY_MODELS_PATH = self._saved_my_models

    def _restore_endpoint_override(self) -> None:
        launcher._ENDPOINT_FILE_OVERRIDE = self._saved_endpoint_override

    def _join_tasks_and_discard(self) -> None:
        # The daemon answers the launcher before its task thread has
        # finished its bookkeeping (``_record_frequent_task`` and the
        # event writer still use the test's history.db); join those
        # threads before the states are dropped and, later, the DB is
        # closed and the tmpdir removed.
        for state in agent_state.snapshot():
            # Read once: the worker clears ``task_thread`` as it ends.
            task_thread = state.task_thread
            if task_thread is not None:
                task_thread.join(timeout=30)
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
        agent_state.agent_states.clear()

    def _shutdown_server(self) -> None:
        async def _shutdown() -> None:
            # Closes the listener (and with it every established local
            # connection) and joins the handlers before the loop stops.
            await self.server.stop_async()
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
            ).result(timeout=30)
        except Exception:
            pass

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def _run_while_held(
        self,
        launch: Callable[[], str],
        model_server: LaunchModelServer | None = None,
    ) -> tuple[Any, threading.Thread, str]:
        """Run *launch* and capture the daemon's task state while it is in flight.

        The registry detaches a task's agent when the task ends, so the
        model call (to *model_server*, default the registered endpoint)
        is held while the agent and its worker thread are read off the
        single registered :class:`~kiss.server.agent_state.AgentState`.
        Returns ``(daemon_agent, task_thread, launch_result)``.
        """
        model_server = model_server or self.model_server
        model_server.hold()
        results: list[str] = []
        thread = threading.Thread(
            target=_call_and_store, args=(launch, results), daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while (
                not model_server.seen.is_set()
                and thread.is_alive()
                and time.monotonic() < deadline
            ):
                model_server.seen.wait(timeout=0.1)
            assert model_server.seen.is_set(), "the task never ran"
            states = agent_state.snapshot()
            assert len(states) == 1, states
            daemon_agent, task_thread = states[0].agent, states[0].task_thread
        finally:
            model_server.release()
            thread.join(timeout=60)
        assert not thread.is_alive(), "the launch never returned"
        assert len(results) == 1, "the launch raised"
        assert task_thread is not None
        return daemon_agent, task_thread, results[0]

    def _only_request(self) -> dict[str, Any]:
        """Return the single model request the run made."""
        assert len(self.model_server.requests) == 1, len(self.model_server.requests)
        return self.model_server.requests[0]


class TestLaunchViaApi(_ApiLaunchBase):
    """The launcher must run tasks through ``kiss.server.sorcar.run``."""

    def test_task_runs_on_daemon_agent_not_passed_instance(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        agent = SlackAgent()

        def launch() -> str:
            return run_agent_via_kiss_web(
                agent,
                "hello slack task",
                work_dir=self.repo,
            )

        daemon_agent, task_thread, result = self._run_while_held(launch)
        request = self._only_request()
        assert "hello slack task" in _task_text(request)
        assert daemon_agent is not agent, (
            "the task must run on a daemon-built agent, not the passed "
            "third-party agent instance"
        )
        assert isinstance(daemon_agent, ChatSorcarAgent)
        assert task_thread is not threading.main_thread(), (
            "the task must run on the daemon's worker thread"
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        assert SUMMARY in parsed["summary"]
        assert agent.last_run_result == result

    def test_channel_prompt_appended_to_system_prompt(self) -> None:
        """The channel guidance reaches the run as system-prompt text, not task text.

        The launcher sends the task verbatim; the daemon applies the
        channel module's ``system_prompt()`` hook (the agent class's
        ``channel_system_prompt``) to the run's system prompt.
        """
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        run_agent_via_kiss_web(
            SlackAgent(),
            "auth prompt task",
            work_dir=self.repo,
        )
        request = self._only_request()
        task = _task_text(request)
        assert "auth prompt task" in task
        assert "Slack Authentication" not in task
        system_prompt = _system_text(request)
        assert "Slack Authentication" in system_prompt
        assert "finish_slack_auth()" in system_prompt

    def test_agent_module_is_the_agent_script(self) -> None:
        from kiss.agents.third_party_agents.slack import slack_sea
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        agent = SlackAgent()
        assert agent.sea_path == str(slack_sea.__file__), (
            "the agent's own module must be its SEA"
        )
        assert not hasattr(agent, "tools_file")

        self.model_server.script = [
            tool_call_response("check_slack_auth", {}),
            finish_response("module tools loaded ok"),
        ]
        result = run_agent_via_kiss_web(
            agent,
            "use the tools",
            work_dir=self.repo,
        )
        first, second = self.model_server.requests
        names = _tool_names(first)
        for expected in (
            "check_slack_auth",
            "authenticate_slack",
            "finish_slack_auth",
            "clear_slack_auth",
        ):
            assert expected in names, f"missing channel tool {expected}"
        assert any(
            "Not authenticated with Slack" in out for out in _tool_results(second)
        ), "check_slack_auth must have run against the empty kiss home"
        assert "module tools loaded ok" in yaml.safe_load(result)["summary"]

    def test_launcher_has_no_tools_path_parameters(self) -> None:
        """Extra tools come only from the agent's module (``sea_path``).

        ``run_agent_via_kiss_web`` exposes neither ``tools`` nor
        ``append_basic_tools``: both went with the tools-file wire
        contract, and the ``LAUNCH_KWARG_NAMES`` filter the channel
        agents' ``run()`` shims pass their kwargs through drops them.
        """
        import inspect

        from kiss.agents.third_party_agents._channel_agent_utils import (
            filter_launch_kwargs,
        )

        params = inspect.signature(run_agent_via_kiss_web).parameters
        assert "tools" not in params
        assert "append_basic_tools" not in params
        assert "sea_path" not in params
        assert filter_launch_kwargs(
            {"tools": "/x.py", "append_basic_tools": False, "max_budget": 2.0}
        ) == {"max_budget": 2.0}

    def test_backend_tools_included_when_authenticated(self) -> None:
        notes = Path(self.tmpdir) / "notes.log"
        agent_py = Path(self.tmpdir) / "note_agent.py"
        agent_py.write_text(
            "from pathlib import Path\n"
            "\n"
            "from kiss.agents.seas.base.base_sea import BaseSea\n"
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
            "class Sea(BaseSea):\n"
            "    def tools(self, tools: list) -> list:\n"
            '        """Add the note-channel tools."""\n'
            "        return tools + NoteAgent()._get_tools()\n",
            encoding="utf-8",
        )
        self.model_server.script = [
            tool_call_response("add_note", {"note": "from daemon"}),
            finish_response("note recorded"),
        ]
        result = run_agent_via_kiss_web(
            KissWebChatAgent("Note Launch", sea_path=str(agent_py)),
            "note task",
            work_dir=self.repo,
        )
        first, second = self.model_server.requests
        assert "add_note" in _tool_names(first), (
            "the authenticated backend's tool must come from the SEA's tools()"
        )
        assert _only_tool_result(second).startswith("recorded:from daemon")
        assert "note recorded" in yaml.safe_load(result)["summary"]
        assert notes.read_text().splitlines() == ["from daemon"], (
            "backend tools must act on state shared through persistence, "
            "not on the launcher-side instance"
        )

    def test_workspace_env_var_set_while_task_runs(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None
        self.model_server.hold()
        thread = threading.Thread(
            target=run_agent_via_kiss_web,
            args=(SlackAgent(workspace="teamspace"), "ws task"),
            kwargs={"work_dir": self.repo},
            daemon=True,
        )
        thread.start()
        try:
            assert self.model_server.seen.wait(timeout=30), "the task never ran"
            # The daemon-side ``tools()`` (already built by now) read
            # this process-global variable; it stays exported for the
            # whole run.
            assert os.environ.get("KISS_CHANNEL_WORKSPACE") == "teamspace", (
                "the daemon-side tools() must see the launch workspace"
            )
        finally:
            self.model_server.release()
            thread.join(timeout=30)
        assert not thread.is_alive()
        self._join_tasks_and_discard()
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None, (
            "the workspace env var must be removed once the run has finished"
        )

    def test_workspace_env_var_blocks_conflicting_overlap(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        release = {"A": threading.Event(), "B": threading.Event()}
        started = {"A": threading.Event(), "B": threading.Event()}

        def answer(request: dict[str, Any]) -> dict[str, Any]:
            key = "A" if "task-A" in request_text(request) else "B"
            started[key].set()
            assert release[key].wait(timeout=30), f"launch {key} never released"
            return finish_response(f"overlap-{key}")

        self.model_server.answer = answer
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None

        def launch(key: str, workspace: str) -> None:
            run_agent_via_kiss_web(
                SlackAgent(workspace=workspace),
                f"task-{key}",
                work_dir=self.repo,
            )

        thread_a = threading.Thread(target=launch, args=("A", "wsA"), daemon=True)
        thread_b = threading.Thread(target=launch, args=("B", "wsB"), daemon=True)
        try:
            thread_a.start()
            assert started["A"].wait(timeout=30)
            thread_b.start()
            # The daemon registers B's task before its worker reaches
            # the workspace guard: wait for that, so the absence check
            # below is about a launch that IS in progress.
            deadline = time.monotonic() + 30
            while len(agent_state.snapshot()) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert len(agent_state.snapshot()) == 2, "B's task was never admitted"
            # B uses a DIFFERENT workspace: it must wait for A instead
            # of overwriting the env var A's daemon-side tools()
            # reads — that would load the wrong account's credentials.
            assert not started["B"].wait(timeout=1.0), (
                "a launch must not reach the model while a different "
                "workspace is still exported"
            )
            assert os.environ.get("KISS_CHANNEL_WORKSPACE") == "wsA", (
                "a blocked launch must not clobber the env var of a "
                "still-running launch"
            )
            release["A"].set()
            thread_a.join(timeout=30)
            assert not thread_a.is_alive()
            # A finished: B unblocks and publishes its own workspace.
            assert started["B"].wait(timeout=30)
            assert os.environ.get("KISS_CHANNEL_WORKSPACE") == "wsB"
        finally:
            release["A"].set()
            release["B"].set()
            thread_a.join(timeout=30)
            if thread_b.ident is not None:
                thread_b.join(timeout=30)
        assert not thread_b.is_alive()
        self._join_tasks_and_discard()
        assert os.environ.get("KISS_CHANNEL_WORKSPACE") is None, (
            "the env var must be removed once every launch has finished"
        )

    def test_unauthenticated_backend_tools_excluded(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        run_agent_via_kiss_web(
            SlackAgent(),
            "no backend",
            work_dir=self.repo,
        )
        names = _tool_names(self._only_request())
        assert "check_slack_auth" in names
        assert "post_message" not in names, (
            "backend tools must not be exposed when unauthenticated"
        )

    def test_overrides_forwarded_through_run_command(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        # A second endpoint: only an explicitly forwarded ``model_config``
        # can make the run call it instead of the registered default.
        explicit = LaunchModelServer()
        self.addCleanup(explicit.stop)
        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            max_budget=1.25,
            model_config=explicit.model_config,
            web_tools=False,
            is_parallel=True,
        )
        assert self.model_server.requests == []
        assert len(explicit.requests) == 1
        request = explicit.requests[0]
        assert _budget_line(request) == 1.25
        names = _tool_names(request)
        assert "go_to_url" not in names, "web_tools=False must drop the browser tools"
        # The channel module's ``settings()`` (the ``channel`` preset:
        # no fan-out) win over the launcher's ``is_parallel=True`` on
        # the daemon, like every SEA's settings do.
        assert "run_parallel" not in names
        assert "Parallel mode: sequential" in _system_text(request)

    def test_carrier_tools_getter_restricts_the_daemon_built_agent(self) -> None:
        """A carrier's ``sea_path`` script restricts the run to its ``tools()``.

        The channel runner hands its channel module to the
        :class:`KissWebChatAgent` carrier as ``sea_path``; the launcher
        must pass it on as ``sea_path`` so the daemon-built
        agent gets exactly what that script decides: the ``none`` tool
        profile drops the built-in tools and ``tools()`` adds
        the script's own.
        """
        agent_py = Path(self.tmpdir) / "restricting_agent.py"
        agent_py.write_text(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

def only_tool() -> str:
    """Return a marker."""
    return 'only'

class Sea(BaseSea):
    def settings(self, settings):
        """Run with only finish and only_tool."""
        return settings | {'tool_profile': 'none'}

    def tools(self, tools):
        return tools + [only_tool]
''',
            encoding="utf-8",
        )
        self.model_server.script = [
            tool_call_response("only_tool", {}),
            finish_response("restricted ok"),
        ]
        result = run_agent_via_kiss_web(
            KissWebChatAgent("Restricted", sea_path=str(agent_py)),
            "task",
            work_dir=self.repo,
        )
        first, second = self.model_server.requests
        assert _tool_names(first) == ["finish", "only_tool"]
        assert _only_tool_result(second).startswith("only")
        assert "restricted ok" in yaml.safe_load(result)["summary"]

    def test_retired_launch_kwargs_are_dropped(self) -> None:
        """``run()`` ignores the kwargs the launcher no longer takes.

        ``append_to_system_prompt`` / ``append_to_prompt``,
        ``use_worktree``, ``timeout`` and ``endpoint_file`` left the
        launcher's surface (extra text comes from the SEA's
        ``system_prompt()`` / ``prompt()`` methods, the worktree mode
        from its ``channel`` kind).  A caller still passing them gets a
        normal run with none of their effects, not a ``TypeError``.
        The model request shows that neither suffix landed: the system
        message is the assembled system instructions and the user
        message the executed prompt.
        """
        import inspect

        from kiss.agents.third_party_agents._channel_agent_utils import (
            LAUNCH_KWARG_NAMES,
            filter_launch_kwargs,
        )
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        retired = {
            "append_to_system_prompt": "\nLAUNCHER-SYS-SUFFIX-2210",
            "append_to_prompt": "\nLAUNCHER-PROMPT-SUFFIX-2210",
            "use_worktree": False,
            "timeout": 0.5,
            "endpoint_file": "/nonexistent/sorcar-local.json",
        }
        keyword_only = {
            name for name, p in inspect.signature(run_agent_via_kiss_web).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert keyword_only == LAUNCH_KWARG_NAMES, (
            "the filter and the launcher signature must name the same kwargs"
        )
        assert not retired.keys() & keyword_only
        assert filter_launch_kwargs({**retired, "max_budget": 2.0}) == {
            "max_budget": 2.0
        }
        agent = SlackAgent()
        result = agent.run(prompt_template="task", work_dir=self.repo, **retired)
        assert SUMMARY in yaml.safe_load(result)["summary"]
        request = self._only_request()
        assert "LAUNCHER-SYS-SUFFIX-2210" not in _system_text(request)
        assert "LAUNCHER-PROMPT-SUFFIX-2210" not in _task_text(request)

    def test_zero_budget_override_is_honored(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        result = run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
            max_budget=0.0,
        )
        # A zero budget is exhausted before the first model call; the
        # daemon default would have let the run reach the model.
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert "budget" in str(parsed["summary"]).lower()
        assert "$0.00" in str(parsed["summary"])
        assert self.model_server.requests == []

    def test_defaults_use_daemon_config(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            work_dir=self.repo,
        )
        request = self._only_request()
        assert _budget_line(request) > 0, (
            "without an override the daemon config budget applies"
        )
        assert request["model"] == STANDIN_MODEL
        assert f"Model name: {STANDIN_MODEL}" in _system_text(request), (
            "the daemon default model applies when none is passed"
        )

    def test_model_name_forwarded(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        run_agent_via_kiss_web(
            SlackAgent(),
            "task",
            model_name="gpt-5.5",
            model_config=self.model_server.model_config,
            work_dir=self.repo,
        )
        request = self._only_request()
        assert request["model"] == "gpt-5.5"
        assert "Model name: gpt-5.5" in _system_text(request)

    def test_stats_recorded_on_agent_for_cli_stats(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        self.model_server.script = [
            tool_call_response("check_slack_auth", {}) | {"usage": dict(USAGE)},
            tool_call_response("check_slack_auth", {}) | {"usage": dict(USAGE)},
            finish_response("stats done") | {"usage": dict(USAGE)},
        ]
        agent = SlackAgent()
        run_agent_via_kiss_web(
            agent,
            "stats task",
            work_dir=self.repo,
        )
        assert len(self.model_server.requests) == 3
        assert agent.total_tokens_used == 3 * USAGE["total_tokens"]
        assert agent.total_steps == 3
        assert 0 < agent.budget_used < 0.01, agent.budget_used

    def test_agent_failure_returns_failure_yaml(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        self.model_server.script = [_failed_finish("boom-fail happened")]
        agent = SlackAgent()
        result = run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.repo,
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert "boom-fail" in str(parsed["summary"])
        assert agent.last_run_result == result

    def test_abrupt_agent_crash_maps_to_failure_yaml(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        # Every model call fails with HTTP 500: the real loop gives up
        # after its consecutive-error limit and the exception leaves
        # the agent's ``run``.
        self.model_server.fail_status = 500
        agent = SlackAgent()
        result = run_agent_via_kiss_web(
            agent,
            "task",
            work_dir=self.repo,
        )
        assert self.model_server.headers, "the run never reached the model"
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
        )
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert "empty" in str(parsed["summary"]).lower()
        assert self.model_server.requests == []

    def test_invalid_sea_path_raises_before_connecting(self) -> None:
        with self.assertRaises(ValueError):
            run_agent_via_kiss_web(
                KissWebChatAgent(
                    "Bad Script",
                    sea_path=str(Path(self.tmpdir) / "missing_agent.py"),
                ),
                "task",
                work_dir=self.repo,
            )
        assert self.model_server.requests == [], (
            "no task may start for a bad SEA"
        )
        assert agent_state.snapshot() == []


class TestInProcessDaemonBootstrap(_ApiLaunchBase):
    """Launches without an endpoint override start the process-global daemon."""

    def test_global_daemon_started_once_and_reused(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        self.model_server.script = [
            finish_response("global daemon ok"),
            finish_response("global daemon ok"),
        ]
        saved_override = launcher._ENDPOINT_FILE_OVERRIDE
        launcher._ENDPOINT_FILE_OVERRIDE = None
        try:
            result = run_agent_via_kiss_web(
                SlackAgent(),
                "first global task",
                work_dir=self.repo,
            )
            assert "global daemon ok" in yaml.safe_load(result)["summary"]
            assert launcher._API_SERVER is not None
            first_endpoint = launcher._API_SERVER_ENDPOINT
            assert Path(first_endpoint).exists(), (
                "the in-process daemon must publish a real endpoint file"
            )
            server_before = launcher._API_SERVER
            result2 = run_agent_via_kiss_web(
                SlackAgent(),
                "second global task",
                work_dir=self.repo,
            )
            assert "global daemon ok" in yaml.safe_load(result2)["summary"]
            assert launcher._API_SERVER is server_before, (
                "the process-global daemon must be created exactly once"
            )
            assert launcher._API_SERVER_ENDPOINT == first_endpoint
            assert len(self.model_server.requests) == 2
        finally:
            launcher._ENDPOINT_FILE_OVERRIDE = saved_override
            # The process-global daemon was created against this test's
            # temporary persistence/config environment; stop it (listeners,
            # endpoint file, loop thread, private dir) so later tests build
            # their own instead of reusing a daemon wired to a deleted tmpdir.
            self._join_tasks_and_discard()
            launcher._stop_api_server()
        assert launcher._API_SERVER is None
        assert not Path(first_endpoint).exists(), (
            "stopping the daemon must remove its endpoint file"
        )
        assert not any(t.name == "kiss-tp-api-server" for t in threading.enumerate())


class TestCarrierAgentDirectRuns(_ApiLaunchBase):
    """``run()`` on the carrier agents routes through the daemon API.

    The carriers are not executable agents: ``run()`` submits the task
    to the kiss-web daemon via ``kiss.server.sorcar.run`` and records
    the returned YAML, so a crashing daemon-built agent surfaces as a
    failure envelope, never as a re-raised exception.
    """

    def test_chat_agent_direct_run_records_result(self) -> None:
        self.model_server.script = [finish_response("direct chat ok")]
        agent = KissWebChatAgent("Direct Chat")

        def launch() -> str:
            return agent.run(prompt_template="direct task", work_dir=self.repo)

        daemon_agent, _, result = self._run_while_held(launch)
        assert "direct chat ok" in yaml.safe_load(result)["summary"]
        assert agent.last_run_result == result
        assert "direct task" in _task_text(self._only_request())
        assert daemon_agent is not agent and isinstance(daemon_agent, ChatSorcarAgent), (
            "carrier run() must execute on a daemon-built agent"
        )

    def test_chat_agent_direct_run_records_failure(self) -> None:
        self.model_server.fail_status = 500
        agent = KissWebChatAgent("Direct Chat")
        result = agent.run(
            prompt_template="direct task", work_dir=self.repo,
        )
        assert self.model_server.headers, "the run never reached the model"
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

    def test_direct_run_appends_channel_prompt_to_system_prompt(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        agent = SlackAgent()

        def launch() -> str:
            return agent.run(
                prompt_template="direct slack",
                work_dir=self.repo,
                _skip_persistence=True,
            )

        daemon_agent, _, result = self._run_while_held(launch)
        assert SUMMARY in yaml.safe_load(result)["summary"]
        assert agent.last_run_result == result
        assert daemon_agent is not agent and isinstance(daemon_agent, ChatSorcarAgent), (
            "channel agent run() must execute on a daemon-built agent"
        )
        request = self._only_request()
        task = _task_text(request)
        assert "direct slack" in task
        assert "Slack Authentication" not in task
        # The daemon applies the module's ``system_prompt()`` hook.
        assert "Slack Authentication" in _system_text(request)

    def test_direct_run_without_channel_prompt(self) -> None:
        agent = self._plain_agent()
        result = agent.run(
            prompt_template="direct plain", work_dir=self.repo,
        )
        assert SUMMARY in yaml.safe_load(result)["summary"]
        request = self._only_request()
        assert "direct plain" in _task_text(request)
        assert "## Slack Authentication" not in _task_text(request)
        assert "## Slack Authentication" not in _system_text(request)

    def test_direct_run_failure_returns_failure_yaml(self) -> None:
        self.model_server.fail_status = 500
        agent = self._plain_agent()
        result = agent.run(
            prompt_template="direct plain", work_dir=self.repo,
        )
        assert self.model_server.headers, "the run never reached the model"
        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert str(parsed["summary"]).strip(), (
            "a crashed task must not produce an empty summary"
        )
        assert agent.last_run_result == result

    def test_direct_run_bridges_channel_auth_tools(self) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        self.model_server.script = [finish_response("auth tools bridged")]
        agent = SlackAgent()
        result = agent.run(
            prompt_template="direct slack tools", work_dir=self.repo,
        )
        assert "check_slack_auth" in _tool_names(self._only_request())
        assert "auth tools bridged" in yaml.safe_load(result)["summary"]


class TestKissWebChatCarrierAgents(_ApiLaunchBase):
    """The chat-id carrier agents used by the channel runner."""

    def test_chat_agent_gets_daemon_chat_id(self) -> None:
        agent = KissWebChatAgent("Test Carrier")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "carrier task",
            work_dir=self.repo,
        )
        assert SUMMARY in yaml.safe_load(result)["summary"]
        assert agent.last_run_result == result
        assert agent.chat_id, (
            "the daemon-minted chat id must be propagated onto the "
            "carrier agent so the channel runner can resume the thread later"
        )
        assert f"Chat id: {agent.chat_id}" in _system_text(self._only_request())

    def test_chat_agent_resume_chat_by_id(self) -> None:
        agent = KissWebChatAgent("Test Carrier")
        agent.new_chat()
        run_agent_via_kiss_web(
            agent,
            "first task",
            work_dir=self.repo,
        )
        first_chat = agent.chat_id
        assert first_chat

        self.model_server.script = [finish_response("resumed fine")]
        resumed = KissWebChatAgent("Test Carrier")
        resumed.resume_chat_by_id(first_chat)
        run_agent_via_kiss_web(
            resumed,
            "second task",
            work_dir=self.repo,
        )
        assert resumed.chat_id == first_chat, "existing chat id must be kept"
        assert len(self.model_server.requests) == 2
        second = self.model_server.requests[1]
        assert "second task" in _task_text(second)
        assert "first task" in request_text(second), (
            "the resumed run must see the prior task as chat context"
        )

    def test_carrier_records_result_with_new_chat(self) -> None:
        agent = KissWebChatAgent("Test Carrier 2")
        agent.new_chat()
        result = run_agent_via_kiss_web(
            agent,
            "wt task",
            work_dir=self.repo,
        )
        assert SUMMARY in yaml.safe_load(result)["summary"]
        assert agent.last_run_result == result


class _RecordingBackend:
    """A channel backend that records what the runner posts."""

    def __init__(self) -> None:
        self.outbox: list[tuple[str, str, str]] = []

    def strip_bot_mention(self, text: str) -> str:
        """Return *text* unchanged (no bot mention syntax in this backend)."""
        return text

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Record the posted message."""
        self.outbox.append((channel_id, text, thread_ts))

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return whether *msg* carries a ``bot_id``."""
        return bool(msg.get("bot_id"))

    def disconnect(self) -> None:
        """Nothing to close."""


class _ThreadPollingBackend(_RecordingBackend):
    """A recording backend that can also list a thread's replies."""

    def __init__(self, thread_replies: list[dict[str, Any]]) -> None:
        super().__init__()
        self.thread_replies = thread_replies

    def poll_thread_messages(
        self, channel_id: str, thread_ts: str, oldest: str, limit: int = 100,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the fixed thread replies and a zero cursor."""
        return list(self.thread_replies), "0"


class TestChannelRunnerViaApi(_ApiLaunchBase):
    """ChannelRunner._handle_message must launch through the API."""

    def _make_runner(
        self,
        sea_path: str = "",
        thread_replies: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, list[tuple[str, str, str]]]:
        from kiss.agents.third_party_agents._channel_agent_utils import (
            ChannelRunner,
        )

        backend: _RecordingBackend = (
            _RecordingBackend()
            if thread_replies is None
            else _ThreadPollingBackend(thread_replies)
        )
        runner = ChannelRunner(
            backend=backend,
            channel_name="chan",
            agent_name="Test Channel Agent",
            sea_path=sea_path,
            work_dir=str(Path(self.tmpdir) / "chanwork"),
        )
        return runner, backend.outbox

    def test_handle_message_passes_sea_path_and_context(self) -> None:
        tools_py = Path(self.tmpdir) / "chan_tools.py"
        tools_py.write_text(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

def shout(text: str) -> str:
    """Return *text* uppercased.

    Args:
        text: The text to uppercase.
    """
    return text.upper()

class Sea(BaseSea):
    def tools(self, tools):
        """Return the channel tools."""
        return tools + [shout]
''',
            encoding="utf-8",
        )
        runner, outbox = self._make_runner(sea_path=str(tools_py))
        self.model_server.script = [
            tool_call_response("shout", {"text": "hi"}),
            finish_response("channel tools ok"),
        ]
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        first, second = self.model_server.requests
        assert "shout" in _tool_names(first), (
            "the runner's SEA must supply the task's tools"
        )
        assert _only_tool_result(second).startswith("HI")
        prompt = _task_text(first)
        assert "'C123'" in prompt and "'1.0'" in prompt, (
            "the prompt must carry the channel/thread context"
        )
        assert "posted to the thread automatically" in prompt, (
            "a backend without thread polling cannot suppress the "
            "summary, so the agent must be told not to self-post"
        )
        assert "no automatic summary" not in prompt
        assert len(outbox) == 1, "the summary must be posted as the thread reply"
        channel, text, ts = outbox[0]
        assert (channel, ts) == ("C123", "1.0")
        assert "channel tools ok" in text

    def test_handle_message_skips_summary_when_bot_replied(self) -> None:
        runner, outbox = self._make_runner(
            thread_replies=[
                {"user": "U_BOT", "bot_id": "B1", "ts": "1.5", "text": "done"},
            ],
        )
        self.model_server.script = [finish_response("already answered in-thread")]
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert len(self.model_server.requests) == 1
        assert outbox == [], (
            "no summary reply may be posted when the agent already "
            "replied in the thread with its channel tools"
        )

    def test_context_promises_suppression_only_with_thread_polling(self) -> None:
        tools_py = Path(self.tmpdir) / "noop_tools.py"
        tools_py.write_text(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

def ping() -> str:
    """Return pong."""
    return 'pong'

class Sea(BaseSea):
    def tools(self, tools):
        """Return the channel tools."""
        return tools + [ping]
''',
            encoding="utf-8",
        )
        runner, outbox = self._make_runner(
            sea_path=str(tools_py), thread_replies=[],
        )
        self.model_server.script = [finish_response("suppression promised")]
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        prompt = _task_text(self._only_request())
        assert "no automatic summary reply is sent" in prompt, (
            "a thread-polling backend suppresses duplicate summaries, "
            "so the agent may be promised suppression"
        )
        assert len(outbox) == 1, (
            "with no bot reply in the thread the summary is still posted"
        )
        channel, text, ts = outbox[0]
        assert (channel, ts) == ("C123", "1.0")
        assert "suppression promised" in text

    def test_handle_message_sends_summary_when_no_reply(self) -> None:
        runner, outbox = self._make_runner()
        self.model_server.script = [finish_response("channel summary")]
        runner._handle_message("C123", {"text": "hi", "ts": "1.0"})
        assert len(outbox) == 1
        channel, text, ts = outbox[0]
        assert (channel, ts) == ("C123", "1.0")
        assert "channel summary" in text

    def test_handle_message_agent_error_sends_error_reply(self) -> None:
        runner, outbox = self._make_runner()
        self.model_server.script = [_failed_finish("chan-blast happened")]
        runner._handle_message("C9", {"text": "x", "ts": "2.0"})
        assert outbox, "an error reply must still be sent"
        channel, text, ts = outbox[0]
        assert channel == "C9" and ts == "2.0"
        assert "chan-blast" in text


class TestChannelMainInteractiveViaApi(_ApiLaunchBase):
    """channel_main's interactive (-t) mode must launch through the API.

    The provider keys are removed from the environment for every run
    here: a CLI launch that bypassed the stand-in would otherwise call
    the real provider.
    """

    def _run_cli(self, argv: list[str]) -> Callable[[], str]:
        """Return a launch callable running ``channel_main`` with *argv*."""
        from kiss.agents.third_party_agents._channel_agent_utils import (
            channel_main,
        )
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent
        from kiss.core.vscode_config import _refresh_config

        # ``channel_main`` imports the isolated key store into
        # ``DEFAULT_CONFIG``; re-sync it from the restored environment.
        self.addCleanup(_refresh_config)
        for name in _PROVIDER_KEYS:
            self.addCleanup(_restore_env, name, os.environ.pop(name, None))
        orig_argv = sys.argv
        sys.argv = ["kiss-slack", *argv]

        def launch() -> str:
            try:
                channel_main(SlackAgent, "kiss-slack", channel_name="Slack")
            finally:
                sys.argv = orig_argv
            return "returned"

        return launch

    def test_interactive_mode_uses_api(self) -> None:
        # ``-e``/``--header`` must reach the model client: the run calls
        # this second endpoint (not the registered default) with the
        # extra header.  ``-e`` names no key and the endpoint is not a
        # registered vendor's host, so the store's OpenAI key must NOT
        # be sent there; the placeholder token stands in.
        explicit = LaunchModelServer()
        self.addCleanup(explicit.stop)
        (self.home.kiss_home / "api_keys.env").write_text(
            "OPENAI_API_KEY=kiss-cli-key\n", encoding="utf-8",
        )
        launch = self._run_cli([
            "-t", "do the interactive thing",
            "-w", self.repo,
            "-m", STANDIN_MODEL,
            "-b", "2.5",
            "-e", explicit.url,
            "--header", "X-Test: yes",
            "--no-web",
            "--no-parallel",
        ])
        daemon_agent, _, _ = self._run_while_held(launch, explicit)
        assert isinstance(daemon_agent, ChatSorcarAgent)
        assert self.model_server.requests == []
        assert len(explicit.requests) == 1, "interactive channel_main never ran a task"
        request = explicit.requests[0]
        assert explicit.headers[0].get("x-test") == "yes"
        assert explicit.headers[0].get("authorization") == (
            f"Bearer {model_info.KEYLESS_ENDPOINT_TOKEN}"
        )
        assert request["model"] == STANDIN_MODEL
        assert "do the interactive thing" in _task_text(request)
        assert _budget_line(request) == 2.5
        names = _tool_names(request)
        assert "go_to_url" not in names
        assert "run_parallel" not in names
        assert "Parallel mode: sequential" in _system_text(request)

    def test_keyless_endpoint_gets_placeholder_token(self) -> None:
        # A Claude model served by a local OpenAI-compatible proxy: the
        # daemon admits the model (its vendor key is configured) but no
        # OpenAI-compatible key exists for the bare ``-e`` endpoint, so
        # the request carries the fixed placeholder bearer token a
        # key-less local server ignores — instead of no request at all.
        explicit = LaunchModelServer()
        self.addCleanup(explicit.stop)
        (self.home.kiss_home / "api_keys.env").write_text(
            "ANTHROPIC_API_KEY=kiss-anthropic-key\n", encoding="utf-8",
        )
        launch = self._run_cli([
            "-t", "keyless", "-w", self.repo, "-m", "claude-haiku-4-5", "-e", explicit.url,
        ])
        self._run_while_held(launch, explicit)
        assert len(explicit.requests) == 1
        assert explicit.requests[0]["model"] == "claude-haiku-4-5"
        assert explicit.headers[0].get("authorization") == (
            f"Bearer {model_info.KEYLESS_ENDPOINT_TOKEN}"
        )

    def test_without_endpoint_uses_the_models_registered_endpoint(self) -> None:
        # No ``-e``/``--header``: the CLI sends no model_config at all,
        # so the daemon routes the run to the endpoint (and key)
        # MY_MODELS.json registers for the model instead of treating an
        # empty override as explicit and calling the real provider with
        # the store's key (the daemon admits the model only because
        # that vendor key is configured).
        (self.home.kiss_home / "api_keys.env").write_text(
            "OPENAI_API_KEY=kiss-cli-key\n", encoding="utf-8",
        )
        launch = self._run_cli([
            "-t", "registered endpoint", "-w", self.repo, "-m", STANDIN_MODEL, "--no-web",
        ])
        self._run_while_held(launch)
        request = self._only_request()
        assert self.model_server.headers[0].get("authorization") == "Bearer kiss-test-key"
        assert request["model"] == STANDIN_MODEL
        assert "registered endpoint" in _task_text(request)


if __name__ == "__main__":
    unittest.main()
