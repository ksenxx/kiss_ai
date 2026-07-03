# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``/resume --task <task-id>`` in the CLI REPL.

The VS Code webview and the remote browser interface can open the chat
of a specific task by its task id (the History panel sends
``resumeSession`` with a ``taskId``).  These tests verify that the CLI
client gained the same capability via ``/resume --task <task-id>``:

* :func:`kiss.agents.sorcar.cli_helpers._parse_resume_arg` parses the
  new ``--task <id>`` / ``--task=<id>`` forms (and rejects invalid
  combinations).
* ``/resume --task <id>`` resolves the task's owning chat from the
  real SQLite persistence DB, sends ``resumeSession`` carrying BOTH
  ``chatId`` and ``taskId`` to the daemon, and updates the client's
  cached chat id so the next ``run`` continues that chat.
* An unknown task id prints a clear error and sends nothing.

The slash-command tests reuse the real-daemon harness from
:mod:`kiss.tests.agents.sorcar.test_cli_client` (a genuine
:class:`RemoteAccessServer` on a temp Unix socket with an isolated
persistence DB) — no mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import sys
import time
import unittest
from io import StringIO
from typing import Any
from unittest.mock import patch

from kiss.agents.sorcar.cli_client import _handle_client_slash
from kiss.agents.sorcar.cli_helpers import _parse_resume_arg
from kiss.agents.sorcar.persistence import _add_task, _save_task_result
from kiss.tests.agents.sorcar.test_cli_client import CliClientBase


class TestParseResumeArgTaskFlag(unittest.TestCase):
    """`_parse_resume_arg` parses the ``--task`` flag forms."""

    def test_empty_arg_lists_with_defaults(self) -> None:
        self.assertEqual(_parse_resume_arg(""), ("", "", 20))

    def test_bare_chat_id(self) -> None:
        self.assertEqual(_parse_resume_arg("abc123"), ("abc123", "", 20))

    def test_task_flag_with_space(self) -> None:
        self.assertEqual(
            _parse_resume_arg("--task deadbeef"), ("", "deadbeef", 20),
        )

    def test_task_flag_with_equals(self) -> None:
        self.assertEqual(
            _parse_resume_arg("--task=deadbeef"), ("", "deadbeef", 20),
        )

    def test_task_flag_with_limit(self) -> None:
        self.assertEqual(
            _parse_resume_arg("--task deadbeef --limit 5"),
            ("", "deadbeef", 5),
        )

    def test_limit_only(self) -> None:
        self.assertEqual(_parse_resume_arg("--limit 5"), ("", "", 5))

    def test_task_flag_without_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_resume_arg("--task")

    def test_task_flag_equals_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_resume_arg("--task=")

    def test_chat_id_and_task_flag_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_resume_arg("abc123 --task deadbeef")

    def test_extra_positional_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_resume_arg("abc123 def456")


class TestResumeByTaskId(CliClientBase):
    """``/resume --task <task-id>`` round-trips through a real daemon."""

    def _wait_for_cmd(
        self, type_: str, start: int, timeout: float = 3.0,
    ) -> dict[str, Any] | None:
        """Poll the harness for the first *type_* command after *start*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for cmd in self.harness.received_cmds[start:]:
                if cmd.get("type") == type_:
                    return cmd
            time.sleep(0.02)
        return None

    def test_resume_by_task_id_sends_task_and_chat_ids(self) -> None:
        # Seed the (isolated) persistence DB with a real chat: two
        # tasks in one chat so opening by task id is distinguishable
        # from opening the latest task of the chat.
        first_task_id, chat_id = _add_task("first task")
        _save_task_result("first result", task_id=first_task_id)
        second_task_id, _ = _add_task("second task", chat_id=chat_id)
        _save_task_result("second result", task_id=second_task_id)

        before = len(self.harness.received_cmds)
        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            done = _handle_client_slash(
                self.client, f"/resume --task {first_task_id}",
            )
        self.assertFalse(done)
        text = buf.getvalue()
        self.assertIn(f"Resumed task {first_task_id}", text)
        self.assertIn(chat_id, text)
        # The client must now target the resolved chat for follow-ups.
        self.assertEqual(self.client.dispatcher.chat_id, chat_id)
        # The daemon must receive resumeSession with BOTH ids so its
        # _replay_session opens the specific task, not the latest one.
        cmd = self._wait_for_cmd("resumeSession", before)
        self.assertIsNotNone(
            cmd,
            f"resumeSession never arrived; saw "
            f"{self.harness.received_cmds[before:]!r}",
        )
        assert cmd is not None  # for the type-checker
        self.assertEqual(cmd.get("taskId"), first_task_id)
        self.assertEqual(cmd.get("chatId"), chat_id)

    def test_unknown_task_id_prints_error_and_sends_nothing(self) -> None:
        before = len(self.harness.received_cmds)
        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            done = _handle_client_slash(
                self.client, "/resume --task 0123456789abcdef",
            )
        self.assertFalse(done)
        self.assertIn(
            "No task found with id 0123456789abcdef", buf.getvalue(),
        )
        self.assertEqual(self.client.dispatcher.chat_id, "")
        # Give any (erroneous) send a moment to arrive, then assert
        # no resumeSession was dispatched.
        time.sleep(0.2)
        self.assertIsNone(self._wait_for_cmd("resumeSession", before, 0.1))

    def test_chat_id_plus_task_flag_prints_invalid(self) -> None:
        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            done = _handle_client_slash(
                self.client, "/resume abc123 --task deadbeef",
            )
        self.assertFalse(done)
        self.assertIn("Invalid /resume argument", buf.getvalue())

    def test_no_arg_listing_hint_mentions_task_flag(self) -> None:
        buf = StringIO()
        with patch.object(sys, "stdout", buf):
            self.assertFalse(_handle_client_slash(self.client, "/resume"))
        self.assertIn("/resume --task <task-id>", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
