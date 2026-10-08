# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the Inject promptlet panel's edit (pencil) button backend.

Covers :func:`kiss.server.tricks.edit_my_injection_trick` against a real
``MY_INJECTION.md`` (``KISS_HOME`` pinned to a temp dir), the daemon's
``editTrick`` command handler in ``kiss.server.commands`` (catalog
entry, ``tricksData`` / ``error`` events and the stamped resync after
a rejection), and the command over the real local WSS transport.
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
from kiss.server.server import broadcast_to_conn
from kiss.server.sorcar import API, validate_command
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.local_ws import open_local_connection

_BUNDLED = "## Trick\n\nBundled promptlet one.\n\n## Trick\n\nBundled two.\n"
_USER = (
    "Notes the user typed above the first heading.\n\n"
    "## Trick\n\nFirst mine.\n\n"
    "## Other\n\nNot a trick, stays put.\n\n"
    "## Trick\n\nSecond mine.\n\n"
    "## Trick\n\nThird mine.\n"
)


class _TricksHome(unittest.TestCase):
    """Pin ``KISS_HOME`` and ``KISS_INJECTIONS_PATH`` to temp files."""

    def setUp(self) -> None:
        self._saved = {
            k: os.environ.get(k) for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")
        }
        self.kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_edit_trick_")) / ".kiss"
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


class TestEditMyInjectionTrick(_TricksHome):
    """The helper rewrites exactly the matching ``## Trick`` section, in place."""

    def setUp(self) -> None:
        super().setUp()
        self.user_file.write_text(_USER, encoding="utf-8")

    def test_edits_middle_section_in_place_and_keeps_everything_else(self) -> None:
        self.assertIsNone(
            tricks.edit_my_injection_trick("  Second mine.  ", "  Second, edited.  ")
        )
        self.assertEqual(
            self.user_text(),
            "Notes the user typed above the first heading.\n\n"
            "## Trick\n\nFirst mine.\n\n"
            "## Other\n\nNot a trick, stays put.\n\n"
            "## Trick\n\nSecond, edited.\n\n"
            "## Trick\n\nThird mine.\n",
        )
        self.assertEqual(
            tricks.read_tricks_data(),
            {
                "tricks": [
                    "First mine.",
                    "Second, edited.",
                    "Third mine.",
                    "Bundled promptlet one.",
                    "Bundled two.",
                ],
                "userCount": 3,
            },
        )

    def test_edits_first_and_last_sections_keeping_their_spacing(self) -> None:
        self.assertIsNone(tricks.edit_my_injection_trick("First mine.", "One"))
        self.assertIsNone(tricks.edit_my_injection_trick("Third mine.", "Three"))
        self.assertEqual(
            self.user_text(),
            "Notes the user typed above the first heading.\n\n"
            "## Trick\n\nOne\n\n"
            "## Other\n\nNot a trick, stays put.\n\n"
            "## Trick\n\nSecond mine.\n\n"
            "## Trick\n\nThree\n",
        )

    def test_section_without_trailing_newline_gets_one(self) -> None:
        self.user_file.write_text("## Trick\n\nOnly one.", encoding="utf-8")
        self.assertIsNone(tricks.edit_my_injection_trick("Only one.", "Edited."))
        self.assertEqual(self.user_text(), "## Trick\n\nEdited.\n")
        self.assertEqual(tricks.read_tricks_data()["tricks"][0], "Edited.")

    def test_only_the_first_duplicate_section_is_edited(self) -> None:
        self.user_file.write_text(
            "## Trick\n\nSame.\n\n## Trick\n\nSame.\n", encoding="utf-8"
        )
        self.assertIsNone(tricks.edit_my_injection_trick("Same.", "Changed."))
        self.assertEqual(
            self.user_text(), "## Trick\n\nChanged.\n\n## Trick\n\nSame.\n"
        )

    def test_new_body_is_stored_escaped_and_listed_as_typed(self) -> None:
        self.assertIsNone(
            tricks.edit_my_injection_trick("First mine.", r"Match a\*b then \n")
        )
        # ``\*`` would be read as an escape, so its backslash is doubled;
        # ``\n`` (a letter) is not escapable and is stored as typed.
        self.assertIn("## Trick\n\nMatch a\\\\*b then \\n\n", self.user_text())
        self.assertEqual(
            tricks.read_tricks_data()["tricks"][0], r"Match a\*b then \n"
        )
        # ...and the panel's listed text is what the next edit matches on.
        self.assertIsNone(
            tricks.edit_my_injection_trick(r"Match a\*b then \n", "Plain again")
        )
        self.assertEqual(tricks.read_tricks_data()["tricks"][0], "Plain again")

    def test_matches_the_unescaped_body_the_panel_shows(self) -> None:
        self.user_file.write_text("## Trick\n\nUse a\\\\*b.\n", encoding="utf-8")
        self.assertEqual(tricks.read_tricks_data()["tricks"][0], "Use a\\*b.")
        self.assertIsNone(tricks.edit_my_injection_trick("Use a\\*b.", "Plain."))
        self.assertEqual(self.user_text(), "## Trick\n\nPlain.\n")

    def test_multiline_bodies_are_matched_and_written_whole(self) -> None:
        self.user_file.write_text(
            "## Trick\n\nLine one.\nLine two.\n\n## Trick\n\nOther.\n",
            encoding="utf-8",
        )
        self.assertIsNone(
            tricks.edit_my_injection_trick("Line one.\nLine two.", "A\nB\nC")
        )
        self.assertEqual(
            self.user_text(), "## Trick\n\nA\nB\nC\n\n## Trick\n\nOther.\n"
        )
        self.assertEqual(tricks.read_tricks_data()["tricks"][:2], ["A\nB\nC", "Other."])

    def test_crlf_text_from_the_vscode_page_matches_and_is_normalised(self) -> None:
        self.user_file.write_text(
            "## Trick\r\n\r\nLine one.\r\nLine two.\r\n\r\n## Trick\r\n\r\nOther.\r\n",
            encoding="utf-8",
            newline="",
        )
        self.assertIsNone(
            tricks.edit_my_injection_trick("Line one.\r\nLine two.", "New\r\nbody")
        )
        self.assertEqual(
            tricks.read_tricks_data()["tricks"][:2], ["New\nbody", "Other."]
        )
        # The rewrite writes LF on every platform (like a delete does).
        self.assertEqual(
            self.user_file.read_bytes(),
            b"## Trick\n\nNew\nbody\n\n## Trick\n\nOther.\n",
        )

    def test_unchanged_text_is_a_no_op_that_does_not_touch_the_file(self) -> None:
        before = self.user_file.stat().st_mtime_ns
        self.user_file.chmod(stat.S_IRUSR)
        try:
            self.assertIsNone(
                tricks.edit_my_injection_trick("First mine.", "  First mine. ")
            )
        finally:
            self.user_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.assertEqual(self.user_text(), _USER)
        self.assertEqual(self.user_file.stat().st_mtime_ns, before)

    def test_new_body_duplicating_another_promptlet_is_rejected(self) -> None:
        self.assertEqual(
            tricks.edit_my_injection_trick("First mine.", "Third mine."),
            "That promptlet is already in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), _USER)

    def test_new_body_may_equal_a_bundled_promptlet(self) -> None:
        # Only the user file is checked for duplicates, like the Add button.
        self.assertIsNone(
            tricks.edit_my_injection_trick("First mine.", "Bundled two.")
        )
        self.assertEqual(tricks.read_tricks_data()["tricks"][0], "Bundled two.")

    def test_empty_new_body_is_rejected_before_reading_the_file(self) -> None:
        self.user_file.unlink()
        self.assertEqual(
            tricks.edit_my_injection_trick("First mine.", "  \n "),
            "Promptlet must not be empty",
        )
        self.assertFalse(self.user_file.exists(), "nothing was seeded or written")

    def test_new_body_starting_a_heading_is_rejected(self) -> None:
        self.assertEqual(
            tricks.edit_my_injection_trick("First mine.", "Fine line\n## Trick"),
            "Promptlet must not contain a line starting with '## '",
        )
        self.assertEqual(self.user_text(), _USER)

    def test_unknown_body_reports_error_without_touching_file(self) -> None:
        for missing in ("Bundled two.", "Not a trick, stays put.", "nope"):
            self.assertEqual(
                tricks.edit_my_injection_trick(missing, "Whatever"),
                "That promptlet is not in ~/.kiss/MY_INJECTION.md",
                missing,
            )
        self.assertEqual(self.user_text(), _USER)

    def test_missing_file_is_seeded_then_default_can_be_edited(self) -> None:
        self.user_file.unlink()
        self.assertEqual(
            tricks.edit_my_injection_trick("nope", "x"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), tricks.DEFAULT_MY_INJECTION)
        self.assertIsNone(
            tricks.edit_my_injection_trick(tricks.MY_INJECTION_DEFAULT_BODY, "Mine.")
        )
        self.assertEqual(self.user_text(), "## Trick\n\nMine.\n")

    def test_non_utf8_file_reports_error_without_writing(self) -> None:
        self.user_file.write_bytes(b"## Trick\n\n\xff\xfe\n")
        self.assertEqual(
            tricks.edit_my_injection_trick("x", "y"),
            "~/.kiss/MY_INJECTION.md is not UTF-8 text",
        )
        self.assertEqual(self.user_file.read_bytes(), b"## Trick\n\n\xff\xfe\n")

    @posix_only("directory permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_unwritable_home_reports_error(self) -> None:
        self.user_file.unlink()
        self.kiss_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            self.assertEqual(
                tricks.edit_my_injection_trick("x", "y"),
                "Could not read ~/.kiss/MY_INJECTION.md",
            )
        finally:
            self.kiss_dir.chmod(stat.S_IRWXU)
        self.assertFalse(self.user_file.exists())

    @posix_only("directory permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_unwritable_directory_raises_os_error(self) -> None:
        # The rewrite is atomic (staged sibling + ``os.replace``), so a
        # read-only FILE is replaced fine; an unwritable DIRECTORY is
        # what makes the write fail.
        self.kiss_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            with self.assertRaises(OSError):
                tricks.edit_my_injection_trick("First mine.", "Changed.")
        finally:
            self.kiss_dir.chmod(stat.S_IRWXU)
        self.assertEqual(self.user_text(), _USER)

    def test_concurrent_edits_of_different_tricks_keep_both(self) -> None:
        results: list[str | None] = []
        barrier = threading.Barrier(2)

        def run(body: str, new: str) -> None:
            barrier.wait(timeout=5)
            results.append(tricks.edit_my_injection_trick(body, new))

        threads = [
            threading.Thread(target=run, args=("First mine.", "One")),
            threading.Thread(target=run, args=("Third mine.", "Three")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(results, [None, None])
        self.assertEqual(
            tricks.read_tricks_data(),
            {
                "tricks": [
                    "One",
                    "Second mine.",
                    "Three",
                    "Bundled promptlet one.",
                    "Bundled two.",
                ],
                "userCount": 3,
            },
        )

    def test_add_and_delete_still_work_on_the_shared_helpers(self) -> None:
        # ``_split_sections`` / ``_reject_new_body`` are shared with the
        # add and delete helpers; their behaviour must not have moved.
        self.assertEqual(
            tricks.append_my_injection_trick(" "), "Promptlet must not be empty"
        )
        self.assertEqual(
            tricks.append_my_injection_trick("## x"),
            "Promptlet must not contain a line starting with '## '",
        )
        self.assertIsNone(tricks.append_my_injection_trick("Fourth mine."))
        self.assertIsNone(tricks.delete_my_injection_trick("Second mine."))
        self.assertEqual(
            self.user_text(),
            "Notes the user typed above the first heading.\n\n"
            "## Trick\n\nFirst mine.\n\n"
            "## Other\n\nNot a trick, stays put.\n\n"
            "## Trick\n\nThird mine.\n\n"
            "## Trick\n\nFourth mine.\n",
        )
        self.assertEqual(
            tricks.delete_my_injection_trick("gone"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.user_file.write_text("no headings here\n", encoding="utf-8")
        self.assertEqual(
            tricks.delete_my_injection_trick("gone"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(
            tricks.edit_my_injection_trick("gone", "x"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )


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

    def _broadcast_to_conn(self, event: dict[str, Any], conn_id: str) -> None:
        broadcast_to_conn(self.printer, event, conn_id)

    def last(self, event_type: str) -> dict[str, Any] | None:
        for msg in reversed(self.printer.messages):
            if msg.get("type") == event_type:
                return dict(msg)
        return None


class TestEditTrickCommand(_TricksHome):
    """The daemon's ``editTrick`` rewrites the file and answers the panel."""

    def setUp(self) -> None:
        super().setUp()
        self.server = _FakeServer()
        self.user_file.write_text(_USER, encoding="utf-8")

    def test_command_is_in_the_catalog_and_handler_table(self) -> None:
        self.assertEqual(API["editTrick"].required, ("text", "newText"))
        self.assertEqual(API["editTrick"].handler, "forward")
        self.assertIn("editTrick", _CommandsMixin._HANDLERS)
        self.assertIsNone(
            validate_command({"type": "editTrick", "text": "x", "newText": "y"})
        )
        self.assertEqual(
            validate_command({"type": "editTrick", "text": "x"}),
            "Invalid editTrick command: missing newText",
        )
        self.assertEqual(
            validate_command({"type": "editTrick", "newText": "y"}),
            "Invalid editTrick command: missing text",
        )

    def test_edit_rewrites_file_and_broadcasts_unstamped_list(self) -> None:
        self.server._cmd_edit_trick(
            {"text": "Second mine.", "newText": "Second, edited.", "connId": "c1"}
        )
        msg = self.server.last("tricksData")
        assert msg is not None
        self.assertNotIn("connId", msg, "edits repaint every window, not only the sender")
        self.assertEqual(
            msg["tricks"],
            [
                "First mine.",
                "Second, edited.",
                "Third mine.",
                "Bundled promptlet one.",
                "Bundled two.",
            ],
        )
        self.assertEqual(msg["userCount"], 3)
        self.assertIn("## Trick\n\nSecond, edited.\n", self.user_text())
        self.assertIsNone(self.server.last("error"))

    def test_rejection_answers_sender_with_error_then_stamped_resync(self) -> None:
        self.server._cmd_edit_trick(
            {"text": "First mine.", "newText": "Third mine.", "connId": "c9"}
        )
        self.assertEqual(
            [m["type"] for m in self.server.printer.messages],
            ["error", "tricksData"],
        )
        err, resync = self.server.printer.messages
        self.assertEqual(err["connId"], "c9")
        self.assertEqual(
            err["text"], "That promptlet is already in ~/.kiss/MY_INJECTION.md"
        )
        # The panel showed the new text before asking; the list on disk,
        # stamped for the sender, puts the old text back.
        self.assertEqual(resync["connId"], "c9")
        self.assertEqual(resync["userCount"], 3)
        self.assertEqual(
            resync["tricks"][:3], ["First mine.", "Second mine.", "Third mine."]
        )
        self.assertEqual(self.user_text(), _USER)

    def test_unknown_body_answers_sender_with_error(self) -> None:
        self.server._cmd_edit_trick(
            {"text": "Bundled two.", "newText": "x", "connId": "c2"}
        )
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c2")
        self.assertEqual(err["text"], "That promptlet is not in ~/.kiss/MY_INJECTION.md")
        self.assertEqual(self.user_text(), _USER)

    def test_non_string_fields_are_treated_as_empty(self) -> None:
        self.server._cmd_edit_trick({"text": "First mine.", "newText": ["x"]})
        err = self.server.last("error")
        assert err is not None
        self.assertNotIn("connId", err)
        self.assertEqual(err["text"], "Promptlet must not be empty")
        self.server.printer.messages.clear()
        self.server._cmd_edit_trick({"text": 7, "newText": "x"})
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["text"], "That promptlet is not in ~/.kiss/MY_INJECTION.md")
        self.assertEqual(self.user_text(), _USER)
        # No connId to stamp: the resync reaches every window, unstamped.
        resync = self.server.last("tricksData")
        assert resync is not None
        self.assertNotIn("connId", resync)

    @posix_only("directory and file permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_os_error_on_write_answers_error_not_crash(self) -> None:
        # Unwritable directory: the atomic rewrite cannot stage its temp file.
        self.kiss_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            self.server._cmd_edit_trick(
                {"text": "First mine.", "newText": "Changed.", "connId": "c3"}
            )
        finally:
            self.kiss_dir.chmod(stat.S_IRWXU)
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c3")
        self.assertTrue(
            err["text"].startswith("Could not write ~/.kiss/MY_INJECTION.md: ")
        )
        self.assertEqual(self.user_text(), _USER)
        resync = self.server.last("tricksData")
        assert resync is not None
        self.assertEqual(resync["connId"], "c3")
        self.assertEqual(resync["tricks"][0], "First mine.")


class TestEditTrickOverLocalChannel(_TricksHome):
    """``editTrick`` travels the real transport: local WSS → catalog → handler."""

    def setUp(self) -> None:
        super().setUp()
        self.user_file.write_text(_USER, encoding="utf-8")
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            await self.server.stop_async()

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
            reader, writer = await open_local_connection(self.server)
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

    def test_edit_over_the_wire_rewrites_file_and_returns_list(self) -> None:
        event = self._roundtrip(
            {"type": "editTrick", "text": "First mine.", "newText": "First, edited."},
            "tricksData",
        )
        self.assertEqual(
            event["tricks"],
            [
                "First, edited.",
                "Second mine.",
                "Third mine.",
                "Bundled promptlet one.",
                "Bundled two.",
            ],
        )
        self.assertEqual(event["userCount"], 3)
        self.assertIn("## Trick\n\nFirst, edited.\n", self.user_text())

    def test_missing_new_text_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip({"type": "editTrick", "text": "First mine."}, "error")
        self.assertEqual(event["text"], "Invalid editTrick command: missing newText")
        self.assertEqual(self.user_text(), _USER)

    def test_rejection_error_reaches_the_sender(self) -> None:
        event = self._roundtrip(
            {"type": "editTrick", "text": "First mine.", "newText": "Second mine."},
            "error",
        )
        self.assertEqual(
            event["text"], "That promptlet is already in ~/.kiss/MY_INJECTION.md"
        )
        self.assertEqual(self.user_text(), _USER)


if __name__ == "__main__":
    unittest.main()
