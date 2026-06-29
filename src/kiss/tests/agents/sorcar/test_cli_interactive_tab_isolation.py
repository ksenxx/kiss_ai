# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for cross-tab event isolation in the
``sorcar`` CLI interactive client.

The daemon (:class:`~kiss.agents.vscode.web_server.RemoteAccessServer`)
fans every task event out to ALL connected WebSocket clients AND all
local Unix-domain-socket writers via
:meth:`~kiss.agents.vscode.web_server.WebPrinter._send_to_ws_clients`.
The frontend (the chat webview in ``media/main.js``) filters incoming
events client-side by ``tabId`` so that one VS Code window /
browser-tab/ extension only renders panels for its own tab.

Pre-fix the sorcar CLI interactive client (:mod:`cli_client`)
did NOT filter events by ``tabId``: as soon as ANOTHER client (the
VS Code extension webview, a remote browser tab, or a second sorcar
CLI instance) started a task, the CLI started rendering that other
client's ``text_delta`` / ``tool_call`` / ``tool_result`` / ``result``
panels too — polluting the terminal with output from a task the CLI
operator did not start.

These tests reproduce the bug by driving the dispatcher directly
(it is the routing layer that decides whether an event is rendered
or dropped) using a real :class:`~kiss.core.print_to_console.ConsolePrinter`
writing into :class:`io.StringIO`.  No mocks, no fakes, no test
doubles — each assertion observes actual terminal output bytes
emitted by the printer.
"""

from __future__ import annotations

import io
import unittest

from kiss.agents.sorcar.cli_client import _EventDispatcher
from kiss.core.print_to_console import ConsolePrinter


class TestEventDispatcherFiltersByTabId(unittest.TestCase):
    """``_EventDispatcher.dispatch`` of events whose ``tabId`` does
    not match the CLI client's own tab must be dropped silently.

    Streamed display events (``text_delta`` / ``tool_call`` /
    ``tool_result`` / ``result`` / ``system_prompt`` / ``prompt``)
    belonging to another client's task must NOT reach the terminal.
    Targeted control events (``status`` / ``askUser`` / ``clear``)
    must NOT mutate the CLI client's per-task / per-chat state when
    they belong to another tab.  Untargeted global events
    (``configData`` with no ``tabId``, plain ``result`` with no
    ``tabId``) must still be processed as before.
    """

    def setUp(self) -> None:
        self.buf = io.StringIO()
        self.printer = ConsolePrinter(file=self.buf)
        self.disp = _EventDispatcher(self.printer, tab_id="CLI_TAB")

    # ----- streamed display events -----

    def test_text_delta_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "text_delta",
                "tabId": "OTHER_TAB",
                "text": "hello from another window",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_text_delta_from_own_tab_is_rendered(self) -> None:
        self.disp.dispatch(
            {
                "type": "text_delta",
                "tabId": "CLI_TAB",
                "text": "hello from me",
            },
        )
        self.assertIn("hello from me", self.buf.getvalue())

    def test_text_delta_with_no_tab_id_is_rendered(self) -> None:
        # Pre-tab-fanout daemon builds emit some events with no
        # ``tabId`` (system-prompt + initial banners).  We must not
        # accidentally swallow these.
        self.disp.dispatch(
            {"type": "text_delta", "text": "global text"},
        )
        self.assertIn("global text", self.buf.getvalue())

    def test_tool_call_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "tool_call",
                "tabId": "OTHER_TAB",
                "name": "Read",
                "path": "/etc/passwd",
                "description": "leaked",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_tool_result_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "tool_result",
                "tabId": "OTHER_TAB",
                "content": "secret tool output",
                "tool_name": "Bash",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_result_panel_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "result",
                "tabId": "OTHER_TAB",
                "text": "another task's final answer",
                "total_tokens": 100,
                "cost": "$0.01",
                "step_count": 3,
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_system_prompt_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "system_prompt",
                "tabId": "OTHER_TAB",
                "text": "OTHER WINDOW SYSTEM PROMPT",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_usage_info_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "usage_info",
                "tabId": "OTHER_TAB",
                "text": "1234 tokens",
                "total_tokens": 1234,
                "cost": "$0.05",
                "total_steps": 7,
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    # ----- targeted control events -----

    def test_status_from_other_tab_does_not_toggle_task_active(self) -> None:
        # Arm dispatcher with the current task id so it would
        # accept matching status events.
        self.disp.current_task_id = "TASK_X"
        self.disp.dispatch(
            {
                "type": "status",
                "tabId": "OTHER_TAB",
                "taskId": "TASK_Y",
                "running": True,
            },
        )
        self.assertFalse(self.disp.task_active.is_set())

    def test_status_from_own_tab_toggles_task_active(self) -> None:
        # Unarmed (no current_task_id) → every matching status applies.
        self.disp.dispatch(
            {
                "type": "status",
                "tabId": "CLI_TAB",
                "running": True,
            },
        )
        self.assertTrue(self.disp.task_active.is_set())

    def test_ask_user_from_other_tab_is_not_queued(self) -> None:
        self.disp.dispatch(
            {
                "type": "askUser",
                "tabId": "OTHER_TAB",
                "question": "Should I rm -rf /?",
            },
        )
        self.assertTrue(self.disp.ask_user_q.empty())

    def test_ask_user_from_own_tab_is_queued(self) -> None:
        self.disp.dispatch(
            {
                "type": "askUser",
                "tabId": "CLI_TAB",
                "question": "Mine?",
            },
        )
        self.assertEqual(self.disp.ask_user_q.get_nowait(), "Mine?")

    def test_clear_from_other_tab_does_not_set_chat_id(self) -> None:
        # Pre-fix bug: a webview's ``newChat`` broadcast a
        # ``clear`` carrying its tabId + chat_id to ALL clients,
        # and the CLI dispatcher silently overwrote its own
        # cached chat id with the OTHER tab's chat id.
        self.disp.chat_id = "MY_CHAT"
        self.disp.dispatch(
            {
                "type": "clear",
                "tabId": "OTHER_TAB",
                "chat_id": "ANOTHER_CHAT",
            },
        )
        self.assertEqual(self.disp.chat_id, "MY_CHAT")

    def test_clear_from_own_tab_updates_chat_id(self) -> None:
        self.disp.dispatch(
            {
                "type": "clear",
                "tabId": "CLI_TAB",
                "chat_id": "MY_CHAT_2",
            },
        )
        self.assertEqual(self.disp.chat_id, "MY_CHAT_2")

    def test_notification_from_other_tab_is_dropped(self) -> None:
        # Auto-commit notifications carry the originating tab id
        # (see :class:`WorktreeSorcarAgent._broadcast_autocommit_notification`).
        # A webview-driven autocommit must not surface as a toast on
        # the CLI operator's terminal.
        self.disp.dispatch(
            {
                "type": "notification",
                "tabId": "OTHER_TAB",
                "id": "autocommit-1",
                "severity": "info",
                "message": "Generating commit message",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_notification_from_own_tab_is_rendered(self) -> None:
        self.disp.dispatch(
            {
                "type": "notification",
                "tabId": "CLI_TAB",
                "id": "autocommit-1",
                "severity": "info",
                "message": "Generating commit message",
            },
        )
        self.assertIn("Generating commit message", self.buf.getvalue())

    def test_notification_with_no_tab_id_is_rendered(self) -> None:
        # Server-reset notifications and other global toasts have no
        # ``tabId``; the webview shows them, so the CLI must too.
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "server-reset-1",
                "severity": "info",
                "message": "Restarting the KISS Sorcar web server",
            },
        )
        self.assertIn(
            "Restarting the KISS Sorcar web server",
            self.buf.getvalue(),
        )

    # ----- request/reply events still routed properly -----

    def test_cli_info_with_other_tab_is_dropped(self) -> None:
        # ``cliInfo`` replies the CLI client requested itself carry
        # the CLI's own tabId.  A reply targeted at another tab
        # (theoretical — the server uses ``connId`` to route, but
        # nothing prevents a broadcast leak) must not be enqueued
        # to the CLI's synchronous waiter.
        self.disp.dispatch(
            {
                "type": "cliInfo",
                "tabId": "OTHER_TAB",
                "subtype": "help",
                "text": "wrong tab help",
            },
        )
        self.assertTrue(self.disp.cli_info_q.empty())

    def test_cli_info_with_own_tab_is_enqueued(self) -> None:
        self.disp.dispatch(
            {
                "type": "cliInfo",
                "tabId": "CLI_TAB",
                "subtype": "help",
                "text": "my help",
            },
        )
        ev = self.disp.cli_info_q.get_nowait()
        self.assertEqual(ev.get("text"), "my help")

    def test_models_with_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "models",
                "tabId": "OTHER_TAB",
                "models": [{"name": "wrong-tab-model"}],
            },
        )
        self.assertTrue(self.disp.models_q.empty())

    def test_models_with_no_tab_id_is_enqueued(self) -> None:
        # ``_cmd_get_models`` routes via ``connId`` and emits no
        # ``tabId`` — must still reach the waiter.
        self.disp.dispatch(
            {
                "type": "models",
                "models": [{"name": "global-model"}],
            },
        )
        ev = self.disp.models_q.get_nowait()
        self.assertEqual(ev.get("models"), [{"name": "global-model"}])

    def test_configdata_with_no_tab_id_updates_current_model(self) -> None:
        # ``configData`` from the daemon's ``ready`` fanout carries no
        # ``tabId``; the CLI must still pick up the canonical model.
        self.disp.dispatch(
            {
                "type": "configData",
                "config": {"model": "global-model"},
            },
        )
        self.assertEqual(self.disp.current_model, "global-model")

    def test_configdata_from_other_tab_does_not_overwrite_model(self) -> None:
        self.disp.current_model = "my-model"
        self.disp.dispatch(
            {
                "type": "configData",
                "tabId": "OTHER_TAB",
                "config": {"model": "other-model"},
            },
        )
        self.assertEqual(self.disp.current_model, "my-model")

    # ----- error events -----

    def test_error_event_from_other_tab_is_dropped(self) -> None:
        self.disp.dispatch(
            {
                "type": "error",
                "tabId": "OTHER_TAB",
                "text": "other window error",
            },
        )
        self.assertEqual(self.buf.getvalue(), "")

    def test_error_event_with_no_tab_id_is_rendered(self) -> None:
        self.disp.dispatch(
            {"type": "error", "text": "global error"},
        )
        self.assertIn("global error", self.buf.getvalue())


class TestEventDispatcherBackwardsCompatibility(unittest.TestCase):
    """``_EventDispatcher`` constructed without a ``tab_id`` (the
    pre-fix signature) must keep working — events are accepted
    regardless of their ``tabId`` so existing tests and any external
    callers continue to function.
    """

    def setUp(self) -> None:
        self.buf = io.StringIO()
        self.printer = ConsolePrinter(file=self.buf)
        # Construct WITHOUT tab_id (backwards-compat path).
        self.disp = _EventDispatcher(self.printer)

    def test_no_tab_id_accepts_all_tabs(self) -> None:
        self.disp.dispatch(
            {
                "type": "text_delta",
                "tabId": "ANY_TAB",
                "text": "hi",
            },
        )
        self.assertIn("hi", self.buf.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
