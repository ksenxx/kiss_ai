# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for notification rendering in the sorcar CLI.

The chat webview renders ``{"type": "notification", ...}`` broadcasts
as toast messages in
:func:`kiss/agents/vscode/media/main.js::showNotification`.  These
notifications include life-cycle messages such as
"Generating commit message" / "Committed <subject>" emitted by the
worktree auto-commit pipeline as well as server-reset messages such
as "Restarting the KISS Sorcar web server…".

Pre-fix the sorcar CLI (both the legacy in-process path that uses
:class:`~kiss.ui.cli.cli_printer.RecordingConsolePrinter` and
the newer interactive client
:class:`~kiss.ui.cli.cli_client.CliClient` whose dispatcher
routes streamed display events through
:meth:`_EventDispatcher._render`) silently dropped every notification
event — they were treated as "frontend-only / merge / setTaskText /
focus events that have no useful CLI rendering" and never reached the
terminal.  As a result, an operator running ``sorcar`` in a terminal
saw none of the same toasts that a webview user saw, including the
auto-commit life-cycle progress.

These tests reproduce the gap end-to-end (no mocks, no fakes) by
capturing the printer's terminal output via a real
:class:`~rich.console.Console` writing into a
:class:`io.StringIO` and asserting the notification body — including
severity, message, and any progress sub-message — appears in the
captured output.
"""

from __future__ import annotations

import io
import unittest

from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli.cli_client import _EventDispatcher
from kiss.ui.cli.cli_printer import RecordingConsolePrinter


class TestEventDispatcherRendersNotifications(unittest.TestCase):
    """``_EventDispatcher.dispatch`` of a notification event must
    render the notification to the terminal via
    :class:`ConsolePrinter`.

    This is the path that drives the new interactive CLI
    (``sorcar`` connecting to a running ``sorcar web`` daemon over
    UDS): the daemon broadcasts ``notification`` events the chat
    webview already handles, and the CLI client must surface them
    too instead of silently dropping them.
    """

    def setUp(self) -> None:
        self.buf = io.StringIO()
        self.printer = ConsolePrinter(file=self.buf)
        self.disp = _EventDispatcher(self.printer)

    def test_info_notification_message_appears_on_terminal(self) -> None:
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "n-1",
                "severity": "info",
                "message": "Generating commit message",
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Generating commit message", out)

    def test_warning_notification_shows_severity_label(self) -> None:
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "n-2",
                "severity": "warning",
                "message": "Disk space low",
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Disk space low", out)
        # Severity label must surface so the operator can tell info
        # from warning from error at a glance.
        self.assertIn("WARNING", out.upper())

    def test_error_notification_shows_severity_label(self) -> None:
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "n-3",
                "severity": "error",
                "message": "Daemon connection lost",
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Daemon connection lost", out)
        self.assertIn("ERROR", out.upper())

    def test_default_severity_is_info_when_missing(self) -> None:
        """Missing ``severity`` must default to ``info`` — the
        webview's ``showNotification`` does the same."""
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "n-4",
                "message": "Hello",
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Hello", out)
        self.assertIn("INFO", out.upper())

    def test_progress_message_is_rendered(self) -> None:
        """``progressMessage`` (subtitle below the main message in
        the webview toast) must surface on the terminal too."""
        self.disp.dispatch(
            {
                "type": "notification",
                "id": "n-5",
                "severity": "info",
                "message": "Indexing repository",
                "progressMessage": "42% — scanning src/",
                "progress": 0.42,
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Indexing repository", out)
        self.assertIn("42% — scanning src/", out)

    def test_notification_without_id_does_not_crash(self) -> None:
        """A daemon version-drift event that omits ``id`` must not
        crash the dispatcher — render best-effort."""
        self.disp.dispatch(
            {
                "type": "notification",
                "severity": "info",
                "message": "no id present",
            },
        )
        self.assertIn("no id present", self.buf.getvalue())

    def test_notification_with_empty_message_does_not_crash(self) -> None:
        """A defensive zero-content event must be silently rendered
        without raising; the chat webview tolerates it the same way."""
        # No assertion on output content — just that dispatch returns
        # cleanly with no exception.
        self.disp.dispatch({"type": "notification", "id": "n-x"})

    def test_unrelated_events_still_silently_ignored(self) -> None:
        """The notification handler must not accidentally take over
        other unhandled event types (e.g. ``setTaskText`` / ``focus``)
        that are still expected to be silently dropped."""
        self.disp.dispatch({"type": "setTaskText", "text": "x"})
        self.disp.dispatch({"type": "focus"})
        self.assertEqual(self.buf.getvalue(), "")


class TestRecordingConsolePrinterRendersNotificationsLocally(
    unittest.TestCase,
):
    """``RecordingConsolePrinter.broadcast`` of a notification event
    must also render to the terminal, not only persist & forward to
    the daemon.

    This is the path taken by the legacy in-process CLI
    (:mod:`kiss.ui.cli.cli_repl` and the worktree auto-commit
    pipeline calling ``self.printer.broadcast({...})`` directly from
    :class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`).
    Pre-fix the ``broadcast`` override only recorded and forwarded the
    event, so the operator at the terminal saw nothing.
    """

    def setUp(self) -> None:
        # Build a RecordingConsolePrinter then swap its inner
        # ConsolePrinter for one that writes into a StringIO so we
        # can assert on the captured terminal output.  We cannot use
        # the printer's own constructor knobs because it defaults to
        # the real ``sys.stdout``.
        self.buf = io.StringIO()
        self.printer = RecordingConsolePrinter()
        self.printer._console = ConsolePrinter(file=self.buf)

    def test_notification_broadcast_renders_to_terminal(self) -> None:
        self.printer.broadcast(
            {
                "type": "notification",
                "id": "autocommit-1",
                "severity": "info",
                "message": "Generating commit message",
                "sticky": True,
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Generating commit message", out)

    def test_committed_notification_renders_subject(self) -> None:
        self.printer.broadcast(
            {
                "type": "notification",
                "id": "autocommit-1",
                "severity": "info",
                "message": "Committed feat: add notifications",
            },
        )
        out = self.buf.getvalue()
        self.assertIn("Committed feat: add notifications", out)

    def test_non_notification_broadcast_does_not_render(self) -> None:
        """Persistence-only broadcasts (``status``, ``clear``,
        ``configData``…) must NOT print anything on the terminal —
        they are control-plane events the webview consumes silently.
        """
        self.printer.broadcast({"type": "status", "running": True})
        self.printer.broadcast({"type": "clear", "chat_id": "abc"})
        self.printer.broadcast(
            {"type": "configData", "config": {"model": "x"}},
        )
        self.assertEqual(self.buf.getvalue(), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
