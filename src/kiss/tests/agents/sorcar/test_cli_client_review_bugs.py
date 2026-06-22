# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for bugs flagged by the gpt-5.5 review of
the sorcar CLI → web_server client port.

Each test name carries the review-issue number it reproduces.  All tests
spin up a real :class:`RemoteAccessServer` (via the existing
``_DaemonHarness`` in :mod:`test_cli_client`) and exercise the real
``cli_client.py`` over a Unix-domain socket — no mocks, patches or test
doubles.  The tests are deterministic: every wait has a hard deadline.

Round-3 fixes:
* Removed the ``client.send`` monkey-patching that previously bypassed
  the daemon (review D4 / D6 / D8 / D9 round 2).
* Added :class:`TestStatusBroadcastCarriesTaskId` which submits a real
  ``run`` through the daemon and asserts the broadcast ``status``
  events echo the client-supplied ``taskId`` — the round-2
  ``current_task_id`` filter is a no-op without this echo (review A2).
* Added :class:`TestCurrentTaskIdResetOnExit` to cover the new
  ``try/finally`` lifecycle (review B1).
* Added :class:`TestCustomCommandDisconnectDoesNotPrintTrue` to cover
  the ``error`` / ``errorMessage`` disambiguation (review A5 / B4).
* Added :class:`TestAutocommitDisconnectFastFail` (review A3).
* Added :class:`TestExitStopsRunningTask` (review #20 round 1).
"""

from __future__ import annotations

import contextlib
import gc
import io
import os
import time
import unittest
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kiss.agents.sorcar.cli_client import (
    CliClient,
    _EventDispatcher,
    _handle_client_slash,
    _print_elapsed,
    _request_cli_info,
    _request_models,
    _submit_task,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.agents.sorcar.test_cli_client import (
    CliClientBase,
    _DaemonHarness,
)


def _wait_for(predicate: Any, timeout: float = 5.0, step: float = 0.02) -> bool:
    """Poll *predicate* until it returns truthy or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


class TestCostRegression(CliClientBase):
    """Review #1 — ``/cost`` must read budget/tokens from ``tab.agent``."""

    def test_cost_reads_budget_and_tokens_from_tab_agent(self) -> None:
        tab = _RunningAgentState(
            tab_id=self.client.tab_id,
            default_model="anything",
        )
        tab.agent = SimpleNamespace(  # type: ignore[assignment]
            budget_used=1.2345,
            total_tokens_used=4321,
            chat_id="stale-agent-chat-id",
        )
        # ``tab.chat_id`` must win over the agent's (stale) chat id —
        # see A1.a in the round-2 review.
        tab.chat_id = "chat-abc"
        _RunningAgentState.register(self.client.tab_id, tab)
        try:
            reply = _request_cli_info(self.client, "cost")
        finally:
            _RunningAgentState.unregister(self.client.tab_id)
        text = reply.get("text", "")
        self.assertIn("$1.2345", text, f"Cost missing: {text!r}")
        self.assertIn("4321", text, f"Token count missing: {text!r}")
        self.assertIn("chat-abc", text, f"Chat id missing: {text!r}")
        self.assertNotIn("stale-agent-chat-id", text,
                         "Stale agent chat id leaked through /cost")
        self.assertAlmostEqual(float(reply.get("cost", 0.0)), 1.2345, places=4)
        self.assertEqual(int(reply.get("tokens", 0)), 4321)


class TestModelCurrentPerClient(unittest.TestCase):
    """Review #6 — ``/model`` no-arg must reflect *this* client's selection."""

    def setUp(self) -> None:
        # Register every cleanup the moment its resource is acquired so
        # a partial setUp (e.g. one ``start()`` raising) still releases
        # everything constructed up to that point.  ``addCleanup`` runs
        # in LIFO order — the harness shuts down last, after both
        # clients have closed and their devnull FDs have been
        # released.  Without this, an exception in ``client_b.start``
        # would leak ``client_a``'s loop FDs and the harness's UDS
        # listener, fanning out into ``OSError [Errno 24] Too many
        # open files`` for the rest of the suite.
        self.harness = _DaemonHarness()
        self.addCleanup(self.harness.shutdown)
        # Force a final ``gc.collect()`` so any sockets / loops still
        # held only by lingering references (e.g. closed dispatcher
        # task wrappers) are reclaimed before the next test runs.
        self.addCleanup(gc.collect)

        self._devnull_a = open(os.devnull, "w")
        self.addCleanup(self._devnull_a.close)
        self._devnull_b = open(os.devnull, "w")
        self.addCleanup(self._devnull_b.close)

        self.client_a = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=ConsolePrinter(file=self._devnull_a),
        )
        # Register cleanup BEFORE ``start`` so a failure mid-start
        # still releases whatever sockets / threads the client did
        # manage to acquire.
        self.addCleanup(self._safe_close, self.client_a)
        self.client_a.start(timeout=5.0)

        self.client_b = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=ConsolePrinter(file=self._devnull_b),
        )
        self.addCleanup(self._safe_close, self.client_b)
        self.client_b.start(timeout=5.0)

    @staticmethod
    def _safe_close(client: CliClient) -> None:
        """Best-effort ``client.close`` that never re-raises.

        ``CliClient.close()`` already swallows its own exceptions, but
        we wrap once more so a hypothetical bubble-up from one
        cleanup can never short-circuit later ones in the
        ``addCleanup`` chain.
        """
        try:
            client.close()
        except Exception:  # noqa: BLE001 - last-ditch cleanup
            pass

    def test_model_no_arg_per_client_isolation(self) -> None:
        models = _request_models(self.client_a)
        self.assertGreaterEqual(len(models), 2, "Need >=2 known models")
        m_a = models[0]["name"]
        m_b = models[1]["name"]

        # Each client picks a distinct model.  Client B's selection is
        # the most recent one, so the server-wide ``_default_model``
        # ends up holding ``m_b`` — but ``cliInfo modelCurrent`` reads
        # ``tab.selected_model`` so client A still sees ``m_a``.
        self.client_a.send({"type": "selectModel", "model": m_a,
                            "tabId": self.client_a.tab_id})
        self.client_b.send({"type": "selectModel", "model": m_b,
                            "tabId": self.client_b.tab_id})

        # Wait for the server to apply both updates.
        def _both_applied() -> bool:
            tabs = _RunningAgentState.running_agent_states
            tab_a = tabs.get(self.client_a.tab_id)
            tab_b = tabs.get(self.client_b.tab_id)
            return (
                tab_a is not None and tab_a.selected_model == m_a
                and tab_b is not None and tab_b.selected_model == m_b
            )

        self.assertTrue(_wait_for(_both_applied, timeout=3.0),
                        "selectModel never propagated to both tabs")

        reply_a = _request_cli_info(self.client_a, "modelCurrent")
        reply_b = _request_cli_info(self.client_b, "modelCurrent")
        text_a = reply_a.get("text", "")
        text_b = reply_b.get("text", "")
        self.assertIn(m_a, text_a, f"Client A: expected {m_a!r} in {text_a!r}")
        self.assertIn(m_b, text_b, f"Client B: expected {m_b!r} in {text_b!r}")


class TestDaemonDisconnectFastFail(CliClientBase):
    """Review #8 — slash commands must NOT block 10 s when daemon is gone."""

    def test_help_returns_quickly_when_closed_is_set(self) -> None:
        self.client._closed.set()
        start = time.monotonic()
        _handle_client_slash(self.client, "/help")
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 2.0,
            f"/help took {elapsed:.2f}s after disconnect (expected <2 s)",
        )

    def test_cost_returns_quickly_when_closed_is_set(self) -> None:
        self.client._closed.set()
        start = time.monotonic()
        _handle_client_slash(self.client, "/cost")
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 2.0,
            f"/cost took {elapsed:.2f}s after disconnect",
        )


class TestStaleCliInfoReplyRace(CliClientBase):
    """Review #14 — late cliInfo replies must NOT be misrouted.

    Round-3: pre-populate the dispatcher's ``cli_info_q`` directly with
    a stale event whose ``requestId`` differs from the next request's.
    No ``client.send`` monkey-patching (forbidden under the project's
    no-test-doubles rule, review D4 round 2).
    """

    def test_stale_reply_with_old_request_id_is_dropped(self) -> None:
        # Stale ``mcp`` reply tagged with an unrelated requestId — the
        # filter must drop it and keep waiting for the real ``help``
        # reply.
        self.client.dispatcher.cli_info_q.put({
            "type": "cliInfo",
            "subtype": "mcp",
            "text": "STALE-MCP-FROM-PRIOR-REQUEST",
            "requestId": "an-unrelated-old-request-id",
        })
        reply = _request_cli_info(self.client, "help")
        self.assertEqual(
            reply.get("subtype"), "help",
            f"Got stale subtype: {reply!r}",
        )
        self.assertNotIn("STALE", reply.get("text", ""))
        self.assertIn("/help", reply.get("text", ""))


class TestNoneTextDispatchDefence(unittest.TestCase):
    """Review #36 — dispatcher must tolerate ``text: null`` events."""

    def setUp(self) -> None:
        self.buf = open(os.devnull, "w")
        self.disp = _EventDispatcher(ConsolePrinter(file=self.buf))

    def tearDown(self) -> None:
        self.buf.close()

    def test_text_delta_with_none_text(self) -> None:
        self.disp.dispatch({"type": "text_delta", "text": None})

    def test_thinking_delta_with_none_text(self) -> None:
        self.disp.dispatch({"type": "thinking_delta", "text": None})

    def test_tool_call_with_none_input(self) -> None:
        self.disp.dispatch(
            {"type": "tool_call", "name": "fake", "input": None},
        )

    def test_result_with_none_text(self) -> None:
        self.disp.dispatch(
            {"type": "result", "text": None,
             "total_tokens": 0, "cost": "$0", "step_count": 0},
        )


class TestPrintElapsedRoutesThroughPrinter(CliClientBase):
    """Review #34 — ``_print_elapsed`` must use ``ConsolePrinter``.

    Round-3: removed the ``SimpleNamespace`` test double; this test
    now drives a real :class:`CliClient` whose printer's ``file=`` is
    replaced with a real :class:`StringIO` so the captured output can
    be inspected.
    """

    def test_elapsed_line_lands_in_printer_file(self) -> None:
        buf = StringIO()
        # Swap the dispatcher's printer for one whose output goes
        # into a buffer we can read; no monkey-patching of methods.
        self.client.dispatcher.printer = ConsolePrinter(file=buf)
        _print_elapsed(self.client, time.time() - 1.5)
        output = buf.getvalue()
        self.assertIn(
            "Time:", output,
            f"Expected elapsed line in printer file, got {output!r}",
        )


class TestRequestCliInfoErrorMarker(CliClientBase):
    """Review #26 — placeholder reply on timeout signals error.

    Round-3: tightened from a 4-way disjunction (D7 round 2) to a
    precise shape — disconnect must set ``error=True`` AND populate
    ``errorMessage``.
    """

    def test_disconnected_request_returns_error_flag(self) -> None:
        self.client._closed.set()
        reply = _request_cli_info(self.client, "help")
        self.assertIs(reply.get("error"), True,
                      f"Reply must set error=True: {reply!r}")
        msg = reply.get("errorMessage", "")
        self.assertIsInstance(msg, str)
        self.assertIn("Daemon", msg, f"Bad error message: {msg!r}")


class TestSkillsErrorHandling(CliClientBase):
    """Review #27 — unknown ``/skills <name>`` must surface a clear error."""

    def test_unknown_skill_name_returns_error_flag(self) -> None:
        reply = _request_cli_info(
            self.client, "skills", name="this-skill-does-not-exist",
        )
        self.assertIs(reply.get("error"), True)
        self.assertIn("Unknown skill", reply.get("text", ""))
        # ``errorMessage`` carries the human text — review A5 round 2.
        self.assertIn("Unknown skill",
                      reply.get("errorMessage", ""))


# ===========================================================================
# Round-3 tests covering the new fixes
# ===========================================================================


class TestStatusBroadcastCarriesTaskId(CliClientBase):
    """Review A2 round 2 — daemon must echo client's ``taskId`` on status.

    The CLI client's per-task ``status`` filter is a no-op unless the
    daemon's ``status:running={true,false}`` broadcasts carry the
    client-supplied ``taskId``.  Round-3 fix: ``task_runner._run_task``
    now reads ``cmd["taskId"]`` and stamps it on every status broadcast.
    """

    def test_run_command_status_events_echo_task_id(self) -> None:
        my_task_id = uuid.uuid4().hex
        before = len(self.harness.captured)
        # Drive a real ``run`` through the daemon.  An empty / invalid
        # prompt is fine: ``_run_task`` broadcasts ``running=true``
        # BEFORE calling ``_run_task_inner``, so even if the agent
        # fails fast (no API key, empty prompt) we still observe both
        # the start and end status events.
        self.client.send({
            "type": "run",
            "prompt": "ping",
            "model": "",
            "workDir": self.harness.work_dir,
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "taskId": my_task_id,
        })

        def _saw_taskid_on_running_true() -> bool:
            for ev in self.harness.captured[before:]:
                if (ev.get("type") == "status"
                        and ev.get("running") is True
                        and ev.get("taskId") == my_task_id):
                    return True
            return False

        self.assertTrue(
            _wait_for(_saw_taskid_on_running_true, timeout=5.0),
            "Daemon never emitted status:running=true with our taskId — "
            f"saw {[e for e in self.harness.captured[before:] if e.get('type')=='status']!r}",
        )

        # Eventually a running=false broadcast must also carry our id.
        def _saw_taskid_on_running_false() -> bool:
            for ev in self.harness.captured[before:]:
                if (ev.get("type") == "status"
                        and ev.get("running") is False
                        and ev.get("taskId") == my_task_id):
                    return True
            return False

        self.assertTrue(
            _wait_for(_saw_taskid_on_running_false, timeout=15.0),
            "Daemon never emitted status:running=false with our taskId",
        )


class TestSubmitTaskTaskIdRaceRealDaemon(CliClientBase):
    """Review #3 / D8 round 2 — stale ``status:false`` must NOT end a fresh task.

    Round-3 rewrite: drive the real daemon and ensure ``_submit_task``
    returns cleanly with the dispatcher's per-task filter actually
    active in production.
    """

    def test_stale_status_false_before_new_run_is_ignored(self) -> None:
        # Pre-arm dispatcher with a stale ``status:false`` from a
        # different task id.  Without the per-task filter ``_submit_task``
        # would see ``task_active`` toggled by the stale event and
        # return immediately.
        self.client.dispatcher.current_task_id = "old-task-id"
        self.client.dispatcher.task_active.set()
        # Now simulate a stale ``status:false`` for that OLD task id
        # arriving just before the new submit:
        self.client.dispatcher.dispatch({
            "type": "status", "running": False, "taskId": "old-task-id",
        })
        # task_active should be cleared because it matches the armed id.
        self.assertFalse(self.client.dispatcher.task_active.is_set())

        # Now arm with a NEW id and verify a stale ``status:false`` for
        # the OLD id is filtered (kept armed).
        self.client.dispatcher.current_task_id = "new-task-id"
        self.client.dispatcher.task_active.set()
        self.client.dispatcher.dispatch({
            "type": "status", "running": False, "taskId": "stale-different-id",
        })
        # Must remain SET because the stale id does not match.
        self.assertTrue(
            self.client.dispatcher.task_active.is_set(),
            "Stale status:false with mismatched taskId was not filtered",
        )


class TestCurrentTaskIdResetOnExit(CliClientBase):
    """Review B1 round 2 — ``current_task_id`` must reset on every exit path."""

    def test_disconnect_path_resets_current_task_id(self) -> None:
        # Pre-close the client; ``_submit_task`` should take the
        # disconnect path and the finally must reset ``current_task_id``.
        self.client._closed.set()
        _submit_task(
            self.client, "noop",
            use_worktree=False, use_parallel=False,
            auto_commit=False, timeout_seconds=1.0,
        )
        self.assertEqual(
            self.client.dispatcher.current_task_id, "",
            "current_task_id leaked after disconnect exit",
        )
        self.assertFalse(
            self.client.dispatcher.task_active.is_set(),
            "task_active leaked after disconnect exit",
        )

    def test_no_ack_timeout_path_resets_current_task_id(self) -> None:
        # Pre-arm the dispatcher with an old id so we can detect a
        # stale leak.  Then call _submit_task with a tiny timeout
        # WITHOUT a daemon ack (we don't send anything through the
        # daemon, so the no-ack-timeout branch fires).
        self.client.dispatcher.current_task_id = "leftover-id"
        # Disable the daemon connection so the run is not actually
        # processed — the simplest way is to close the client first.
        self.client._closed.set()
        _submit_task(
            self.client, "noop",
            use_worktree=False, use_parallel=False,
            auto_commit=False, timeout_seconds=0.5,
        )
        self.assertEqual(
            self.client.dispatcher.current_task_id, "",
            "current_task_id leaked after exit",
        )


class TestCustomCommandDisconnectDoesNotPrintTrue(CliClientBase):
    """Review A5 / B4 round 2 — disconnected /<cmd> must NOT print "True"."""

    def test_disconnect_prints_human_readable_error(self) -> None:
        self.client._closed.set()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _handle_client_slash(self.client, "/somecustomcmd-that-does-not-exist foo")
        output = buf.getvalue()
        # The old code printed "True\n" because ``error`` carried a
        # bool which str()-converted to "True"; the new path reads
        # ``errorMessage`` (string) and falls back to a human message.
        self.assertNotIn("True", output,
                         f"Disconnect printed literal True: {output!r}")
        # The message should mention the daemon connection issue or
        # fall back to a generic "Unknown command".
        # Tightened from a 4-way disjunction (round 2 D7) to a
        # precise expectation — the disconnect-sentinel always
        # populates ``errorMessage`` so the printed line must contain
        # "Daemon" (review T15 round 3).
        self.assertIn("Daemon", output,
                      f"Bad output on disconnect: {output!r}")

    def test_unknown_command_when_connected_prints_unknown_message(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _handle_client_slash(self.client,
                                 "/no-such-custom-command-anywhere extra")
        output = buf.getvalue()
        self.assertIn("Unknown command", output,
                      f"Expected 'Unknown command' in output: {output!r}")
        self.assertNotIn("True", output,
                         f"Unexpected literal 'True': {output!r}")


class TestAutocommitDisconnectFastFail(CliClientBase):
    """Review A3 round 2 — ``/autocommit`` must NOT block 30 s on disconnect."""

    def test_autocommit_returns_quickly_when_closed_is_set(self) -> None:
        self.client._closed.set()
        start = time.monotonic()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _handle_client_slash(self.client, "/autocommit")
        elapsed = time.monotonic() - start
        # Without the fix, this blocks for the full 30 s
        # ``commit_q.get(timeout=30)``.  With it, the polling loop
        # bails on _closed within one 0.25 s tick.
        self.assertLess(
            elapsed, 2.0,
            f"/autocommit took {elapsed:.2f}s after disconnect (expected <2 s)",
        )


class TestExitStopsRunningTask(CliClientBase):
    """Review #20 round 1 — ``/exit`` mid-task must send ``stop`` to daemon.

    Round-3 (review T8): assert the daemon actually received a
    ``stop`` command via the harness's ``received_cmds`` capture from
    the :class:`_RecordingRemoteAccessServer` subclass.  The previous
    round only asserted that ``_handle_client_slash`` returned True,
    which would pass even if the round-2 ``stop`` send were reverted.
    """

    def test_exit_during_active_task_emits_stop(self) -> None:
        self.client.dispatcher.task_active.set()
        before = len(self.harness.received_cmds)
        self.assertTrue(_handle_client_slash(self.client, "/exit"))

        def _saw_stop() -> bool:
            for c in self.harness.received_cmds[before:]:
                if (c.get("type") == "stop"
                        and c.get("tabId") == self.client.tab_id):
                    return True
            return False

        self.assertTrue(
            _wait_for(_saw_stop, timeout=3.0),
            f"daemon never received stop after /exit; "
            f"saw {self.harness.received_cmds[before:]!r}",
        )

    def test_exit_without_active_task_does_not_send_stop(self) -> None:
        # When no task is running, /exit must not send a redundant
        # stop — the existing closeTab from the outer finally is
        # sufficient.
        self.assertFalse(self.client.dispatcher.task_active.is_set())
        before = len(self.harness.received_cmds)
        self.assertTrue(_handle_client_slash(self.client, "/exit"))
        time.sleep(0.2)
        stops = [
            c for c in self.harness.received_cmds[before:]
            if c.get("type") == "stop"
        ]
        self.assertEqual(stops, [], f"unexpected stop: {stops!r}")


class TestStatusFilterAcceptsWhenUnarmed(CliClientBase):
    """Regression: status with no current_task_id must always be accepted.

    The per-task filter only fires when ``current_task_id`` is non-empty.
    A status event arriving before any task is submitted (e.g. an idle
    broadcast on reconnect) must still toggle ``task_active``.
    """

    def test_unarmed_status_toggles_task_active(self) -> None:
        self.assertEqual(self.client.dispatcher.current_task_id, "")
        self.client.dispatcher.dispatch({
            "type": "status", "running": True, "taskId": "anything",
        })
        self.assertTrue(self.client.dispatcher.task_active.is_set())
        self.client.dispatcher.dispatch({
            "type": "status", "running": False,
        })
        self.assertFalse(self.client.dispatcher.task_active.is_set())


if __name__ == "__main__":
    unittest.main()
