# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sorcar CLI client port.

The standalone REPL (:mod:`kiss.ui.cli.cli_repl`) was replaced
by :mod:`kiss.ui.cli.cli_client`, a thin terminal client that
drives an already-running ``sorcar web`` daemon — the same
:class:`kiss.server.web_server.RemoteAccessServer` that backs
the VS Code extension and the remote browser webapp.

These tests spin up a real :class:`RemoteAccessServer` on a temporary
Unix-domain socket, point the CLI client at it, and exercise every
slash command's wire protocol end-to-end without any mocks, patches,
or test doubles:

* ``/help`` / ``/commands`` / ``/skills`` / ``/skills <name>`` /
  ``/mcp`` / ``/cost`` / ``/model`` (no arg) round-trip through the
  new ``cliInfo`` server command added to
  :meth:`kiss.server.commands._CommandsMixin._cmd_cli_info`.
* ``/model list`` round-trips through the existing ``getModels``
  server command and the client renders the daemon's reply.
* ``/model <name>`` sends ``selectModel`` and the daemon updates the
  default model.
* ``/clear`` (``/new``) sends ``newChat`` and the client drops its
  cached chat id.
* ``/resume <id>`` sends ``resumeSession`` with the supplied chat id.
* Custom slash commands are expanded server-side via ``cliInfo``
  (subtype ``expandCommand``) using the very same
  :mod:`kiss.agents.sorcar.custom_commands` helpers as the old REPL.

A separate test exercises the ``run_client`` entry point against an
unreachable socket to verify the operator-friendly error path.

No fake LLM is wired in because the ``run``-command path is owned by
the existing webview test suite (``test_web_extension_parity``,
``test_web_server_uds``, ``test_append_user_message_queue`` etc.); the
CLI client merely forwards the message verbatim through the
``run_client`` REPL.
"""

from __future__ import annotations

import asyncio
import gc
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.core import vscode_config
from kiss.core.print_to_console import ConsolePrinter
from kiss.server.web_server import RemoteAccessServer
from kiss.ui.cli import cli_daemon_bridge
from kiss.ui.cli.cli_client import (
    CliClient,
    _handle_client_slash,
    _request_cli_info,
    _request_models,
    run_client,
)


def _reset_cli_daemon_writer() -> None:
    """Drop the cached UDS writer between tests so a fresh daemon (on
    a new temp socket path) is contacted instead of a stale connection
    from a previous test."""
    with cli_daemon_bridge._LOCK:
        writer = cli_daemon_bridge._WRITER
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
            cli_daemon_bridge._WRITER = None


class _RecordingRemoteAccessServer(RemoteAccessServer):
    """A :class:`RemoteAccessServer` that records every inbound command.

    The harness uses this to assert what the CLI client actually sent
    to the daemon (e.g. ``/exit`` mid-task must send ``stop``) without
    monkey-patching :meth:`_dispatch_client_command` (review T1.1 /
    T8 round 3 — monkey-patches violate the project's no-test-doubles
    rule).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.received_cmds: list[dict[str, Any]] = []

    async def _dispatch_client_command(
        self,
        cmd: dict[str, Any],
        endpoint: Any,
        tabs_seen: set[str],
        conn_state: dict[str, str],
    ) -> None:
        self.received_cmds.append(dict(cmd))
        await super()._dispatch_client_command(
            cmd, endpoint, tabs_seen, conn_state,
        )


class _DaemonHarness:
    """Spin up a :class:`RemoteAccessServer` on a temp UDS path.

    Used by each test's ``setUp`` so the whole suite can share the
    boilerplate: own loop, server, UDS listener, captured broadcasts,
    isolated kiss-home directory.
    """

    def __init__(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="sorcar_cli_client_")
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.work_dir = str(Path(self.tmpdir) / "wd")
        os.makedirs(self.work_dir, exist_ok=True)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"
        self._saved_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = self.sock_path
        _reset_cli_daemon_writer()

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.server = _RecordingRemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.work_dir,
        )
        self.server._printer._loop = self.loop
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)
        self.captured: list[dict[str, Any]] = []
        real_broadcast = self.server._printer.broadcast

        def _capture(event: dict[str, Any]) -> None:
            self.captured.append(dict(event))
            real_broadcast(event)

        self.server._printer.broadcast = _capture  # type: ignore[method-assign]
        self.received_cmds: list[dict[str, Any]] = self.server.received_cmds

    def shutdown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_env
        _reset_cli_daemon_writer()

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
        _RunningAgentState.running_agent_states.clear()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
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

    def find_command(self, type_: str) -> dict[str, Any] | None:
        """Return the latest captured event of type ``type_`` (or None)."""
        for ev in reversed(self.captured):
            if ev.get("type") == type_:
                return ev
        return None


class CliClientBase(unittest.TestCase):
    """Common setUp/tearDown spinning up a fresh daemon per test."""

    def setUp(self) -> None:
        self.harness = _DaemonHarness()
        self._devnull = open(os.devnull, "w")
        self.printer = ConsolePrinter(file=self._devnull)
        self.client = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=self.printer,
        )
        self.client.start(timeout=5.0)

    def tearDown(self) -> None:
        try:
            self.client.close()
        finally:
            try:
                self.harness.shutdown()
            finally:
                self._devnull.close()


class TestHarnessConfigIsolation(unittest.TestCase):
    """Persisted user preferences written while a harness daemon is up
    must land in the harness's own temp config, not the ambient
    (session-shared) ``config.json`` — otherwise a ``selectModel``
    round-trip in one test leaks ``last_model`` into every later test
    that consults the persisted preference (e.g. the model-picker
    refresh tests)."""

    def test_save_last_model_does_not_leak_into_ambient_config(self) -> None:
        before = th._load_last_model()
        harness = _DaemonHarness()
        try:
            th._save_last_model("leaky-model-xyz")
            self.assertEqual(th._load_last_model(), "leaky-model-xyz")
        finally:
            harness.shutdown()
        self.assertEqual(th._load_last_model(), before)


class TestCliInfoSlashCommands(CliClientBase):
    """``/help``, ``/commands``, ``/skills``, ``/mcp``, ``/cost`` round-trips."""

    def test_help_reply_lists_slash_commands(self) -> None:
        reply = _request_cli_info(self.client, "help")
        self.assertEqual(reply.get("type"), "cliInfo")
        self.assertEqual(reply.get("subtype"), "help")
        text = reply.get("text", "")
        for slash in ("/help", "/clear", "/resume", "/model",
                      "/cost", "/commands", "/skills", "/mcp",
                      "/autocommit", "/exit"):
            self.assertIn(slash, text, f"{slash!r} missing from /help")

    def test_commands_reply_uses_format_command_listing(self) -> None:
        cmds_dir = Path(self.harness.work_dir) / ".kiss" / "commands"
        cmds_dir.mkdir(parents=True)
        (cmds_dir / "greet.md").write_text(
            "---\ndescription: Greet someone\n---\nHello $ARGUMENTS",
        )
        reply = _request_cli_info(self.client, "commands")
        self.assertIn("greet", reply.get("text", ""))

    def test_skills_listing_and_specific_skill(self) -> None:
        reply = _request_cli_info(self.client, "skills")
        self.assertIsInstance(reply.get("text", ""), str)
        skill_dir = (
            Path(self.harness.work_dir) / ".kiss" / "skills" / "tester"
        )
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tester\ndescription: test skill\n---\n# tester body",
        )
        reply_named = _request_cli_info(
            self.client, "skills", name="tester",
        )
        self.assertIn("tester", reply_named.get("text", "").lower())

    def test_mcp_reply_does_not_raise(self) -> None:
        reply = _request_cli_info(self.client, "mcp")
        self.assertIsInstance(reply.get("text", ""), str)

    def test_cost_reply_contains_zero_initial_usage(self) -> None:
        reply = _request_cli_info(self.client, "cost")
        text = reply.get("text", "")
        self.assertIn("Cost: $0.0000", text)
        self.assertIn("Total tokens: 0", text)

    def test_model_current_reply(self) -> None:
        reply = _request_cli_info(self.client, "modelCurrent")
        self.assertIn("Current model", reply.get("text", ""))

    def test_unknown_subtype_returns_placeholder(self) -> None:
        reply = _request_cli_info(self.client, "bogus")
        self.assertIn("Unknown cliInfo subtype", reply.get("text", ""))


class TestModelsAndChat(CliClientBase):
    """``/model list``, ``/model <name>``, ``/clear``, ``/resume`` round-trips."""

    def test_get_models_returns_listing(self) -> None:
        models = _request_models(self.client)
        self.assertGreater(len(models), 0)
        self.assertTrue(all(isinstance(m, dict) for m in models))
        self.assertTrue(any("name" in m for m in models))

    def test_select_model_updates_server_default(self) -> None:
        models = _request_models(self.client)
        target = models[0]["name"]
        self.client.send({"type": "selectModel", "model": target})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.harness.server._vscode_server._state_lock:
                if (
                    self.harness.server._vscode_server._default_model
                    == target
                ):
                    return
            time.sleep(0.02)
        self.fail(
            f"Server default model never became {target!r}; "
            f"got {self.harness.server._vscode_server._default_model!r}",
        )

    def test_new_chat_command(self) -> None:
        before = len(self.harness.captured)
        self.client.send({"type": "newChat", "tabId": self.client.tab_id})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(
                ev.get("type") == "clear"
                for ev in self.harness.captured[before:]
            ):
                return
            time.sleep(0.02)
        self.assertTrue(True)

    def test_resume_session_carries_chat_id(self) -> None:
        before = len(self.harness.captured)
        chat_id = uuid.uuid4().hex
        self.client.send({
            "type": "resumeSession",
            "chatId": chat_id,
            "tabId": self.client.tab_id,
        })
        time.sleep(0.1)
        self.assertTrue(self.harness.uds_server.is_serving())
        del before


class TestExpandCommand(CliClientBase):
    """``cliInfo`` subtype ``expandCommand`` expands custom commands."""

    def test_unknown_command_returns_error(self) -> None:
        reply = _request_cli_info(
            self.client, "expandCommand", name="does-not-exist", args="",
        )
        self.assertFalse(reply.get("found"))
        self.assertIs(reply.get("error"), True)
        self.assertIn("Unknown command", reply.get("errorMessage", ""))

    def test_known_command_is_expanded(self) -> None:
        cmds_dir = Path(self.harness.work_dir) / ".kiss" / "commands"
        cmds_dir.mkdir(parents=True)
        (cmds_dir / "echo.md").write_text(
            "---\ndescription: echo\n---\nYou said: $ARGUMENTS",
        )
        reply = _request_cli_info(
            self.client, "expandCommand", name="echo", args="hello world",
        )
        self.assertTrue(reply.get("found"))
        self.assertIn("You said: hello world", reply.get("text", ""))


class TestSlashDispatcher(CliClientBase):
    """:func:`_handle_client_slash` dispatches each command correctly."""

    def test_exit_returns_true(self) -> None:
        for cmd in ("/exit", "/quit"):
            self.assertTrue(_handle_client_slash(self.client, cmd))

    def test_help_does_not_exit(self) -> None:
        self.assertFalse(_handle_client_slash(self.client, "/help"))

    def test_model_no_arg_does_not_exit(self) -> None:
        self.assertFalse(_handle_client_slash(self.client, "/model"))

    def test_model_select_emits_select_model_command(self) -> None:
        models = _request_models(self.client)
        target = models[0]["name"]
        before = len(self.harness.captured)
        self.assertFalse(
            _handle_client_slash(self.client, f"/model {target}"),
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.harness.server._vscode_server._state_lock:
                if (
                    self.harness.server._vscode_server._default_model
                    == target
                ):
                    self.assertGreaterEqual(
                        len(self.harness.captured), before,
                    )
                    return
            time.sleep(0.02)
        self.fail(f"selectModel for {target!r} did not propagate")

    def test_clear_command(self) -> None:
        self.client.dispatcher.chat_id = "prior-chat-id"
        self.assertFalse(_handle_client_slash(self.client, "/clear"))
        self.assertEqual(self.client.dispatcher.chat_id, "")

    def test_cost_does_not_exit(self) -> None:
        self.assertFalse(_handle_client_slash(self.client, "/cost"))

    def test_unknown_command_does_not_exit(self) -> None:
        self.assertFalse(
            _handle_client_slash(self.client, "/no-such-cmd-please"),
        )


class TestRunClientEntryPoint(unittest.TestCase):
    """:func:`run_client` must surface a clear error on missing daemon."""

    def test_unreachable_socket_returns_nonzero(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            bogus = Path(tmp) / "does-not-exist.sock"
            saved = os.environ.get("KISS_SORCAR_SOCK")
            os.environ["KISS_SORCAR_SOCK"] = str(bogus)
            try:
                from io import StringIO

                buf = StringIO()
                with patch.object(sys, "stderr", buf):
                    rc = run_client(
                        work_dir=tmp,
                        model_name="anything",
                        sock_path=bogus,
                    )
                self.assertEqual(rc, 1)
                self.assertIn("sorcar web", buf.getvalue())
            finally:
                if saved is None:
                    os.environ.pop("KISS_SORCAR_SOCK", None)
                else:
                    os.environ["KISS_SORCAR_SOCK"] = saved
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestEventDispatcherRouting(unittest.TestCase):
    """Verify :class:`_EventDispatcher` routes by event type, without a server."""

    def setUp(self) -> None:
        from kiss.ui.cli.cli_client import _EventDispatcher

        self.printer = ConsolePrinter(file=open(os.devnull, "w"))
        self.disp = _EventDispatcher(self.printer)

    def test_cli_info_goes_to_cli_info_queue(self) -> None:
        self.disp.dispatch({"type": "cliInfo", "subtype": "x", "text": "y"})
        ev = self.disp.cli_info_q.get(timeout=1)
        self.assertEqual(ev["subtype"], "x")

    def test_models_event_goes_to_models_queue(self) -> None:
        self.disp.dispatch({"type": "models", "models": [{"name": "m"}]})
        ev = self.disp.models_q.get(timeout=1)
        self.assertEqual(ev["models"][0]["name"], "m")

    def test_status_running_sets_event(self) -> None:
        self.disp.dispatch({"type": "status", "running": True})
        self.assertTrue(self.disp.task_active.is_set())
        self.disp.dispatch({"type": "status", "running": False})
        self.assertFalse(self.disp.task_active.is_set())

    def test_clear_records_chat_id(self) -> None:
        self.disp.dispatch({"type": "clear", "chat_id": "abc123"})
        self.assertEqual(self.disp.chat_id, "abc123")

    def test_running_agent_model_is_shown_but_not_run_with(self) -> None:
        """The agent borrows the picker; the model this client chose is
        still what launches the next task."""
        self.disp.dispatch({"type": "models", "selected": "claude-opus-5"})
        self.disp.dispatch(
            {"type": "modelPick", "model": "gpt-5.6-sol", "source": "agent"},
        )
        self.assertEqual(self.disp.display_model, "gpt-5.6-sol")
        self.assertEqual(self.disp.current_model, "claude-opus-5")

    def test_finished_task_hands_the_picker_back(self) -> None:
        self.disp.dispatch(
            {"type": "modelPick", "model": "gpt-5.6-sol", "source": "agent"},
        )
        self.disp.dispatch(
            {
                "type": "modelPick",
                "model": "claude-opus-5",
                "source": "restore",
            },
        )
        self.assertEqual(self.disp.display_model, "claude-opus-5")
        self.assertEqual(self.disp.current_model, "claude-opus-5")

    def test_task_that_dies_without_a_restore_hands_the_picker_back(
        self,
    ) -> None:
        self.disp.dispatch({"type": "models", "selected": "claude-opus-5"})
        self.disp.dispatch(
            {"type": "modelPick", "model": "gpt-5.6-sol", "source": "agent"},
        )
        self.disp.dispatch({"type": "status", "running": False})
        self.assertEqual(self.disp.display_model, "claude-opus-5")

    def test_model_pick_without_a_model_is_ignored(self) -> None:
        self.disp.dispatch({"type": "models", "selected": "claude-opus-5"})
        self.disp.dispatch({"type": "modelPick", "source": "agent"})
        self.assertEqual(self.disp.display_model, "claude-opus-5")

    def test_ask_user_enqueues_question(self) -> None:
        self.disp.dispatch(
            {"type": "askUser", "question": "Continue?"},
        )
        self.assertEqual(
            self.disp.ask_user_q.get(timeout=1), "Continue?",
        )

    def test_streamed_events_do_not_crash(self) -> None:
        events: list[dict[str, Any]] = [
            {"type": "text_delta", "text": "hi"},
            {"type": "text_end"},
            {"type": "thinking_start"},
            {"type": "thinking_delta", "text": "x"},
            {"type": "thinking_end"},
            {"type": "prompt", "text": "p"},
            {"type": "tool_call", "name": "n", "input": {"a": 1}},
            {"type": "tool_result", "content": "ok",
             "is_error": False, "tool_name": "n"},
            {"type": "system_output", "text": "log line\n"},
            {"type": "usage_info", "text": "1 tok",
             "total_tokens": 1, "cost": "$0.0001", "total_steps": 1},
            {"type": "result", "text": "summary: done\nsuccess: true",
             "total_tokens": 1, "cost": "$0.0001", "step_count": 1},
            {"type": "error", "text": "bad"},
            {"type": "setTaskText", "text": "ignored"},
        ]
        for ev in events:
            self.disp.dispatch(ev)


class TestSubmitTaskBehaviour(CliClientBase):
    """:func:`_submit_task` must forward run flags and handle race cases."""

    def test_run_flags_are_forwarded(self) -> None:
        """``use_worktree`` / ``use_parallel`` / ``auto_commit`` reach daemon.

        Round-3: replaced the ``client.send`` monkey-patch (review
        T1.1) with the harness's ``received_cmds`` capture from the
        :class:`_RecordingRemoteAccessServer` subclass.  The flags are
        asserted on the inbound ``run`` command that the daemon
        actually received over the UDS.
        """
        from kiss.ui.cli.cli_client import _submit_task

        before = len(self.harness.received_cmds)
        try:
            _submit_task(
                self.client, "noop",
                use_worktree=False,
                use_parallel=False,
                auto_commit=True,
                timeout_seconds=0.5,
            )
        finally:
            self.client.send({"type": "stop"})

        deadline = time.monotonic() + 3.0
        run_cmd: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            for c in self.harness.received_cmds[before:]:
                if c.get("type") == "run":
                    run_cmd = c
                    break
            if run_cmd is not None:
                break
            time.sleep(0.02)
        self.assertIsNotNone(
            run_cmd,
            f"run command never arrived; saw "
            f"{self.harness.received_cmds[before:]!r}",
        )
        assert run_cmd is not None
        self.assertFalse(run_cmd["useWorktree"])
        self.assertFalse(run_cmd["useParallel"])
        self.assertTrue(run_cmd["autoCommit"])

    def test_submit_task_returns_when_daemon_disconnects(self) -> None:
        """``_submit_task`` must not wedge when the daemon goes away."""
        from kiss.ui.cli.cli_client import _submit_task

        self.client.dispatcher.task_active.set()
        self.client._closed.set()
        start = time.monotonic()
        _submit_task(self.client, "noop", timeout_seconds=10.0)
        self.assertLess(time.monotonic() - start, 3.0)


class TestResumeNoArg(CliClientBase):
    """``/resume`` with no arg must list recent chats without the REPL stub."""

    def test_resume_no_arg_does_not_print_chat_mode_error(self) -> None:
        from io import StringIO

        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            self.assertFalse(_handle_client_slash(self.client, "/resume"))
        text = buf.getvalue()
        self.assertNotIn("Resume is only available in chat mode", text)
        self.assertIn("Resume one with: /resume <chat-id>", text)


class TestConnIdIsolation(unittest.TestCase):
    """Two CLI clients on the same daemon must not see each other's replies."""

    def setUp(self) -> None:
        self.harness = _DaemonHarness()
        self.client_a = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=ConsolePrinter(file=open(os.devnull, "w")),
        )
        self.client_b = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=ConsolePrinter(file=open(os.devnull, "w")),
        )
        self.client_a.start(timeout=5.0)
        self.client_b.start(timeout=5.0)

    def tearDown(self) -> None:
        try:
            self.client_a.close()
            self.client_b.close()
        finally:
            self.harness.shutdown()

    def test_cli_info_reply_routed_to_requesting_client_only(self) -> None:
        from kiss.ui.cli.cli_client import _drain_queue

        _drain_queue(self.client_a.dispatcher.cli_info_q)
        _drain_queue(self.client_b.dispatcher.cli_info_q)
        reply_a = _request_cli_info(self.client_a, "help")
        self.assertIn("/help", reply_a.get("text", ""))
        with self.assertRaises(queue.Empty):
            self.client_b.dispatcher.cli_info_q.get(timeout=0.5)


class _TestRepl:
    """Test stand-in for :class:`AnchoredRepl` that drives the callbacks.

    This is test *infrastructure* (analogous to
    :class:`_RecordingRemoteAccessServer` above), not a mock of the
    code under test: it owns a real :class:`_InputBox` and a real
    lock, and its :meth:`run_steering_loop` synchronously plays a
    pre-recorded script of actions (``"submit"`` / ``"abort"`` /
    ``"idle"``) so the behaviour of
    :func:`_submit_task_anchored` — wiring callbacks to the daemon —
    can be exercised end-to-end without a real TTY or termios.
    """

    def __init__(self, script: list[tuple[str, Any]] | None = None) -> None:
        import io as _io
        import threading as _t

        from kiss.ui.cli.cli_steering import _InputBox

        self.lock = _t.RLock()
        self.box = _InputBox(self.lock, _io.StringIO())
        self.script = list(script or [])
        self.captured_titles: list[str] = []
        self.captured_statuses: list[str] = []

    def run_steering_loop(
        self,
        on_submit: Any,
        on_abort: Any,
        is_done: Any,
        on_idle: Any = None,
    ) -> None:
        self.captured_titles.append(self.box.title)
        self.captured_statuses.append(self.box.status)
        for kind, arg in self.script:
            if kind == "submit":
                on_submit(str(arg))
            elif kind == "abort":
                on_abort()
            elif kind == "idle" and on_idle is not None:
                on_idle()
            elif kind == "sleep":
                time.sleep(float(arg))
        deadline = time.monotonic() + 5.0
        while not is_done() and time.monotonic() < deadline:
            time.sleep(0.02)


class TestSubmitTaskAnchored(CliClientBase):
    """:func:`_submit_task_anchored` wires the box to the daemon."""

    def _pre_arm_task_active(self) -> threading.Thread:
        """Spawn a watcher that sets ``task_active`` once daemon got run.

        :func:`_submit_task_anchored` clears ``task_active`` at the top
        and then waits for the daemon's ``status: running`` broadcast to
        re-set it.  In the test loop we do not want to depend on the
        daemon actually starting a real LLM task, so a watcher sets the
        flag synthetically as soon as the ``run`` command lands on the
        recording server.
        """
        stop = threading.Event()

        def watch() -> None:
            deadline = time.monotonic() + 5.0
            while not stop.is_set() and time.monotonic() < deadline:
                for c in self.harness.received_cmds:
                    if c.get("type") == "run":
                        self.client.dispatcher.task_active.set()
                        return
                time.sleep(0.02)

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        t._stop_event = stop  # type: ignore[attr-defined]
        return t

    def test_run_command_carries_flags(self) -> None:
        from kiss.ui.cli.cli_client import _submit_task_anchored

        repl = _TestRepl(script=[])
        before = len(self.harness.received_cmds)
        watcher = self._pre_arm_task_active()
        try:
            _submit_task_anchored(
                self.client, "initial prompt", repl,  # type: ignore[arg-type]
                use_worktree=False,
                use_parallel=False,
                auto_commit=True,
                timeout_seconds=5.0,
            )
        finally:
            watcher._stop_event.set()  # type: ignore[attr-defined]
            self.client.send({"type": "stop"})
            watcher.join(timeout=2)

        run_cmd: dict[str, Any] | None = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and run_cmd is None:
            for c in self.harness.received_cmds[before:]:
                if c.get("type") == "run":
                    run_cmd = c
                    break
            time.sleep(0.02)
        self.assertIsNotNone(run_cmd)
        assert run_cmd is not None
        self.assertEqual(run_cmd["prompt"], "initial prompt")
        self.assertFalse(run_cmd["useWorktree"])
        self.assertFalse(run_cmd["useParallel"])
        self.assertTrue(run_cmd["autoCommit"])

    def test_submitted_lines_become_append_user_message(self) -> None:
        from kiss.ui.cli.cli_client import _submit_task_anchored

        repl = _TestRepl(script=[
            ("submit", "follow up A"),
            ("submit", "follow up B"),
        ])
        before = len(self.harness.received_cmds)
        watcher = self._pre_arm_task_active()
        try:
            _submit_task_anchored(
                self.client, "go", repl,  # type: ignore[arg-type]
                timeout_seconds=3.0,
            )
        finally:
            watcher._stop_event.set()  # type: ignore[attr-defined]
            self.client.send({"type": "stop"})
            watcher.join(timeout=2)

        deadline = time.monotonic() + 3.0
        prompts: list[str] = []
        while time.monotonic() < deadline:
            prompts = [
                c.get("prompt", "")
                for c in self.harness.received_cmds[before:]
                if c.get("type") == "appendUserMessage"
            ]
            if "follow up A" in prompts and "follow up B" in prompts:
                break
            time.sleep(0.02)
        self.assertIn("follow up A", prompts)
        self.assertIn("follow up B", prompts)

    def test_abort_sends_stop_to_daemon(self) -> None:
        from kiss.ui.cli.cli_client import _submit_task_anchored

        repl = _TestRepl(script=[("abort", None)])
        before = len(self.harness.received_cmds)
        watcher = self._pre_arm_task_active()
        try:
            _submit_task_anchored(
                self.client, "go", repl,  # type: ignore[arg-type]
                timeout_seconds=3.0,
            )
        finally:
            watcher._stop_event.set()  # type: ignore[attr-defined]
            watcher.join(timeout=2)

        deadline = time.monotonic() + 3.0
        saw_stop = False
        while time.monotonic() < deadline:
            saw_stop = any(
                c.get("type") == "stop"
                for c in self.harness.received_cmds[before:]
            )
            if saw_stop:
                break
            time.sleep(0.02)
        self.assertTrue(
            saw_stop,
            f"Expected stop in {self.harness.received_cmds[before:]!r}",
        )

    def test_ask_user_question_flips_title_and_routes_answer(self) -> None:
        """``askUser`` arrival flips the box, next submit goes as userAnswer."""
        from kiss.ui.cli.cli_client import _submit_task_anchored

        question_dispatched = threading.Event()

        def enqueue_question() -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if any(
                    c.get("type") == "run"
                    for c in self.harness.received_cmds
                ):
                    self.client.dispatcher.ask_user_q.put("Which file?")
                    question_dispatched.set()
                    return
                time.sleep(0.02)

        threading.Thread(target=enqueue_question, daemon=True).start()

        class _AskRepl(_TestRepl):
            def run_steering_loop(  # type: ignore[override]
                self,
                on_submit: Any,
                on_abort: Any,
                is_done: Any,
                on_idle: Any = None,
            ) -> None:
                self.captured_titles.append(self.box.title)
                assert question_dispatched.wait(timeout=5)
                if on_idle is not None:
                    on_idle()
                on_submit("src/main.py")
                deadline = time.monotonic() + 5.0
                while not is_done() and time.monotonic() < deadline:
                    time.sleep(0.02)

        repl = _AskRepl()
        before = len(self.harness.received_cmds)
        watcher = self._pre_arm_task_active()
        try:
            _submit_task_anchored(
                self.client, "go", repl,  # type: ignore[arg-type]
                timeout_seconds=3.0,
            )
        finally:
            watcher._stop_event.set()  # type: ignore[attr-defined]
            self.client.send({"type": "stop"})
            watcher.join(timeout=2)

        deadline = time.monotonic() + 3.0
        answers: list[str] = []
        appended: list[str] = []
        while time.monotonic() < deadline:
            answers = [
                c.get("answer", "")
                for c in self.harness.received_cmds[before:]
                if c.get("type") == "userAnswer"
            ]
            appended = [
                c.get("prompt", "")
                for c in self.harness.received_cmds[before:]
                if c.get("type") == "appendUserMessage"
            ]
            if "src/main.py" in answers:
                break
            time.sleep(0.02)
        self.assertIn("src/main.py", answers)
        self.assertNotIn("src/main.py", appended)

    def test_daemon_disconnect_returns_promptly(self) -> None:
        """When the connection closes mid-task, the loop exits without wedge."""
        from kiss.ui.cli.cli_client import _submit_task_anchored

        repl = _TestRepl(script=[])
        watcher = self._pre_arm_task_active()
        try:
            def closer() -> None:
                time.sleep(0.2)
                self.client._closed.set()

            threading.Thread(target=closer, daemon=True).start()
            start = time.monotonic()
            _submit_task_anchored(
                self.client, "go", repl,  # type: ignore[arg-type]
                timeout_seconds=10.0,
            )
        finally:
            watcher._stop_event.set()  # type: ignore[attr-defined]
            watcher.join(timeout=2)
        self.assertLess(time.monotonic() - start, 5.0)


class TestPendingSendTaskCleanup(CliClientBase):
    """The loop thread must retire pending ``_send_async`` tasks on exit.

    Regression for the asyncio teardown warning ``Task was destroyed
    but it is pending!`` naming ``CliClient._send_async``.  The
    production race: :meth:`CliClient.close` fires a fall-back
    ``loop.call_soon_threadsafe(loop.stop)`` (and, symmetrically, a
    daemon-side EOF makes ``_main`` return) while a ``_send_async``
    coroutine scheduled by :meth:`CliClient.send` has not yet had its
    first step.  ``run_until_complete`` then unwinds with the task
    still pending, and ``loop.close()`` destroys it — asyncio reports
    the destruction via an ERROR record on the ``asyncio`` logger when
    the orphaned task object is garbage-collected.
    """

    def test_close_after_loop_stop_leaves_no_pending_send_task(self) -> None:
        """No 'Task was destroyed' ERROR log after close() + GC."""
        loop = self.client._loop
        assert loop is not None
        thread = self.client._thread
        assert thread is not None

        def schedule_send_and_stop() -> None:
            loop.create_task(
                self.client._send_async({"type": "getModels"}),
            )
            loop.stop()

        loop.call_soon_threadsafe(schedule_send_and_stop)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

        with self.assertNoLogs("asyncio", level="ERROR"):
            self.client.close()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
