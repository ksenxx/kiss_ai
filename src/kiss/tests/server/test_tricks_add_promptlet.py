# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the Inject promptlet panel's Add button backend.

Covers :func:`kiss.server.tricks.append_my_injection_trick` against a
real ``MY_INJECTION.md`` (``KISS_HOME`` pinned to a temp dir) and the
daemon's ``addTrick`` command handler in ``kiss.server.commands``,
including its catalog entry in ``kiss.server.sorcar.API`` and the
``tricksData`` / ``error`` events it answers with.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.server import tricks
from kiss.server.commands import _CommandsMixin
from kiss.server.sorcar import API, validate_command
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import is_root, posix_only, requires_unix_sockets

_BUNDLED = "## Trick\n\nBundled promptlet one.\n\n## Trick\n\nBundled two.\n"


class _TricksHome(unittest.TestCase):
    """Pin ``KISS_HOME`` and ``KISS_INJECTIONS_PATH`` to temp files."""

    def setUp(self) -> None:
        self._saved = {
            k: os.environ.get(k) for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")
        }
        self.kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_add_trick_")) / ".kiss"
        self.kiss_dir.mkdir(parents=True)
        bundled = self.kiss_dir / "bundled_INJECTIONS.md"
        bundled.write_text(_BUNDLED, encoding="utf-8")
        os.environ["KISS_HOME"] = str(self.kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(bundled)
        self.user_file = self.kiss_dir / "MY_INJECTION.md"

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def user_text(self) -> str:
        return self.user_file.read_text(encoding="utf-8")


class TestAppendMyInjectionTrick(_TricksHome):
    """The helper appends one ``## Trick`` section per call."""

    def test_seeds_missing_file_then_appends(self) -> None:
        self.assertFalse(self.user_file.exists())
        self.assertIsNone(tricks.append_my_injection_trick("  Be terse.  "))
        self.assertEqual(
            self.user_text(),
            tricks.DEFAULT_MY_INJECTION + "\n## Trick\n\nBe terse.\n",
        )
        self.assertEqual(
            tricks.read_tricks(),
            [
                tricks.MY_INJECTION_DEFAULT_BODY,
                "Be terse.",
                "Bundled promptlet one.",
                "Bundled two.",
            ],
        )

    def test_appends_after_file_without_trailing_newline(self) -> None:
        self.user_file.write_text("## Trick\n\nFirst.", encoding="utf-8")
        self.assertIsNone(tricks.append_my_injection_trick("Second."))
        self.assertEqual(
            self.user_text(), "## Trick\n\nFirst.\n\n## Trick\n\nSecond.\n"
        )
        self.assertEqual(tricks.read_tricks()[:2], ["First.", "Second."])

    def test_appends_to_empty_file(self) -> None:
        self.user_file.write_text("", encoding="utf-8")
        self.assertIsNone(tricks.append_my_injection_trick("Only one."))
        self.assertEqual(self.user_text(), "\n## Trick\n\nOnly one.\n")
        self.assertEqual(tricks.read_tricks()[0], "Only one.")

    def test_multiline_body_survives_the_round_trip(self) -> None:
        body = "Line one.\nLine two with `code`."
        self.assertIsNone(tricks.append_my_injection_trick(body))
        self.assertIn(body, tricks.read_tricks())

    def test_rejects_empty_body_without_touching_file(self) -> None:
        self.assertEqual(
            tricks.append_my_injection_trick("   \n"),
            "Promptlet must not be empty",
        )
        self.assertFalse(self.user_file.exists())

    def test_rejects_body_that_starts_a_section(self) -> None:
        error = tricks.append_my_injection_trick("Do this.\n## Trick\nSneaky.")
        self.assertEqual(
            error, "Promptlet must not contain a line starting with '## '"
        )
        self.assertFalse(self.user_file.exists())

    def test_rejects_duplicate_of_existing_user_trick(self) -> None:
        self.assertIsNone(tricks.append_my_injection_trick("Twice."))
        before = self.user_text()
        self.assertEqual(
            tricks.append_my_injection_trick("Twice."),
            "That promptlet is already in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), before)

    def test_markdown_escapes_round_trip_verbatim(self) -> None:
        # The reader strips CommonMark backslash escapes, so the writer
        # doubles the backslashes that would be stripped: the panel
        # shows exactly what was typed, and a retry is a duplicate.
        body = "Keep the escape \\* and the path C:\\dir\\.hidden literal"
        self.assertIsNone(tricks.append_my_injection_trick(body))
        self.assertIn(body, tricks.read_tricks())
        self.assertIn(
            "Keep the escape \\\\* and the path C:\\dir\\\\.hidden literal\n",
            self.user_text(),
        )
        self.assertEqual(
            tricks.append_my_injection_trick(body),
            "That promptlet is already in ~/.kiss/MY_INJECTION.md",
        )
        # A backslash before a non-escapable character is stored as is.
        self.assertIsNone(tricks.append_my_injection_trick("Say \\n here"))
        self.assertIn("\n## Trick\n\nSay \\n here\n", self.user_text())
        self.assertIn("Say \\n here", tricks.read_tricks())

    def test_concurrent_identical_adds_write_one_section(self) -> None:
        body = "Only once, please."
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(tricks.append_my_injection_trick, [body] * 8)
            )
        self.assertEqual(results.count(None), 1, results)
        self.assertEqual(
            [r for r in results if r is not None],
            ["That promptlet is already in ~/.kiss/MY_INJECTION.md"] * 7,
        )
        self.assertEqual(self.user_text().count(body), 1)

    def test_non_utf8_file_reports_error_without_writing(self) -> None:
        self.user_file.write_bytes(b"## Trick\n\n\xff\xfe broken\n")
        before = self.user_file.read_bytes()
        self.assertEqual(
            tricks.append_my_injection_trick("New."),
            "~/.kiss/MY_INJECTION.md is not UTF-8 text",
        )
        self.assertEqual(self.user_file.read_bytes(), before)

    def test_bundled_duplicate_is_still_added_to_user_file(self) -> None:
        # Only the user's own file is deduplicated: a copy of a bundled
        # trick is how a user pins it above the bundled list.
        self.assertIsNone(tricks.append_my_injection_trick("Bundled two."))
        self.assertEqual(tricks.read_tricks().count("Bundled two."), 2)

    @posix_only("directory and file permission bits")
    @unittest.skipIf(is_root(), "root ignores directory permissions")
    def test_unwritable_home_reports_error(self) -> None:
        mode = self.kiss_dir.stat().st_mode
        self.kiss_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            self.assertEqual(
                tricks.append_my_injection_trick("Nope."),
                "Could not write ~/.kiss/MY_INJECTION.md",
            )
        finally:
            self.kiss_dir.chmod(mode)


class _FakePrinter:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def broadcast(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)


class _FakeServer(_CommandsMixin):
    def __init__(self) -> None:
        self.printer: Any = _FakePrinter()
        self.work_dir = "/tmp"
        self._state_lock = threading.RLock()

    def last(self, event_type: str) -> dict[str, Any] | None:
        for msg in reversed(self.printer.messages):
            if msg.get("type") == event_type:
                return dict(msg)
        return None


class TestAddTrickCommand(_TricksHome):
    """The daemon's ``addTrick`` writes the file and answers the panel."""

    def setUp(self) -> None:
        super().setUp()
        self.server = _FakeServer()

    def test_command_is_in_the_catalog_and_handler_table(self) -> None:
        self.assertEqual(API["addTrick"].required, ("text",))
        self.assertEqual(API["addTrick"].handler, "forward")
        self.assertIn("addTrick", _CommandsMixin._HANDLERS)
        self.assertIsNone(validate_command({"type": "addTrick", "text": "x"}))
        self.assertIsNotNone(validate_command({"type": "addTrick"}))

    def test_add_writes_file_and_broadcasts_unstamped_list(self) -> None:
        self.server._cmd_add_trick({"text": "Ship it.", "connId": "c1"})
        msg = self.server.last("tricksData")
        assert msg is not None
        self.assertNotIn(
            "connId", msg, "adds repaint every window, not only the sender"
        )
        self.assertEqual(
            msg["tricks"],
            [
                tricks.MY_INJECTION_DEFAULT_BODY,
                "Ship it.",
                "Bundled promptlet one.",
                "Bundled two.",
            ],
        )
        self.assertIn("\n## Trick\n\nShip it.\n", self.user_text())
        self.assertIsNone(self.server.last("error"))

    def test_rejected_body_answers_sender_with_error(self) -> None:
        self.server._cmd_add_trick({"text": "   ", "connId": "c9"})
        self.assertIsNone(self.server.last("tricksData"))
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c9")
        self.assertEqual(err["text"], "Promptlet must not be empty")
        self.assertFalse(self.user_file.exists())

    def test_non_string_text_is_treated_as_empty(self) -> None:
        self.server._cmd_add_trick({"text": 42})
        err = self.server.last("error")
        assert err is not None
        self.assertNotIn("connId", err)
        self.assertEqual(err["text"], "Promptlet must not be empty")

    @posix_only("directory and file permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_os_error_on_write_answers_error_not_crash(self) -> None:
        # The seeded file exists but is read-only: the seeding step
        # succeeds (it only checks existence), the append raises.
        self.user_file.write_text(tricks.DEFAULT_MY_INJECTION, encoding="utf-8")
        self.user_file.chmod(stat.S_IRUSR)
        try:
            self.server._cmd_add_trick({"text": "Blocked.", "connId": "c2"})
        finally:
            self.user_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c2")
        self.assertTrue(
            err["text"].startswith("Could not write ~/.kiss/MY_INJECTION.md: ")
        )
        self.assertIsNone(self.server.last("tricksData"))


@requires_unix_sockets
class TestAddTrickOverUds(_TricksHome):
    """``addTrick`` travels the real transport: UDS → catalog → handler."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        # ``forward`` runs the backend handler on the loop's executor.
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path
            ),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)],
            timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()
        super().tearDown()

    def _roundtrip(
        self, cmd: dict[str, Any], want_type: str
    ) -> dict[str, Any]:
        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(
                        reader.readline(), timeout=10
                    )
                    if not line:
                        raise AssertionError(
                            f"connection closed before a {want_type!r} event"
                        )
                    event: dict[str, Any] = json.loads(line)
                    if event.get("type") == want_type:
                        return event
            finally:
                writer.close()
                await writer.wait_closed()

        return asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )

    def test_add_over_the_wire_writes_file_and_returns_list(self) -> None:
        event = self._roundtrip(
            {"type": "addTrick", "text": "Over the wire."}, "tricksData"
        )
        self.assertEqual(
            event["tricks"],
            [
                tricks.MY_INJECTION_DEFAULT_BODY,
                "Over the wire.",
                "Bundled promptlet one.",
                "Bundled two.",
            ],
        )
        self.assertIn("\n## Trick\n\nOver the wire.\n", self.user_text())

    def test_missing_text_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip({"type": "addTrick"}, "error")
        self.assertEqual(
            event["text"], "Invalid addTrick command: missing text"
        )
        self.assertFalse(self.user_file.exists())

    def test_rejected_body_error_reaches_the_sender(self) -> None:
        event = self._roundtrip({"type": "addTrick", "text": " "}, "error")
        self.assertEqual(event["text"], "Promptlet must not be empty")


if __name__ == "__main__":
    unittest.main()
