# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the Inject promptlet panel's delete button backend.

Covers :func:`kiss.server.tricks.delete_my_injection_trick` and
:func:`kiss.server.tricks.read_tricks_data` against a real
``MY_INJECTION.md`` (``KISS_HOME`` pinned to a temp dir), the daemon's
``deleteTrick`` command handler in ``kiss.server.commands`` (catalog
entry, ``tricksData`` / ``error`` events, ``userCount`` on both the
add and the delete reply), the remote page's ``MY_TRICKS_COUNT``
substitution, and the command over the real local WSS transport.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.server import tricks
from kiss.server.commands import _CommandsMixin
from kiss.server.sorcar import API, validate_command
from kiss.server.web_server import RemoteAccessServer, _build_html
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
        self.kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_del_trick_")) / ".kiss"
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


class TestReadTricksData(_TricksHome):
    """``read_tricks_data`` counts the leading user-owned entries."""

    def test_counts_user_tricks_before_bundled(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        self.assertEqual(
            tricks.read_tricks_data(),
            {
                "tricks": [
                    "First mine.",
                    "Second mine.",
                    "Third mine.",
                    "Bundled promptlet one.",
                    "Bundled two.",
                ],
                "userCount": 3,
            },
        )
        self.assertEqual(
            tricks.read_tricks(), tricks.read_tricks_data()["tricks"]
        )

    def test_seeded_file_counts_one(self) -> None:
        data = tricks.read_tricks_data()
        self.assertEqual(data["userCount"], 1)
        self.assertEqual(data["tricks"][0], tricks.MY_INJECTION_DEFAULT_BODY)

    def test_empty_user_file_counts_zero(self) -> None:
        self.user_file.write_text("", encoding="utf-8")
        self.assertEqual(
            tricks.read_tricks_data(),
            {"tricks": ["Bundled promptlet one.", "Bundled two."], "userCount": 0},
        )


class TestDeleteMyInjectionTrick(_TricksHome):
    """The helper drops exactly the matching ``## Trick`` section(s)."""

    def test_deletes_middle_section_and_keeps_everything_else(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        self.assertIsNone(tricks.delete_my_injection_trick("  Second mine.  "))
        self.assertEqual(
            self.user_text(),
            "Notes the user typed above the first heading.\n\n"
            "## Trick\n\nFirst mine.\n\n"
            "## Other\n\nNot a trick, stays put.\n\n"
            "## Trick\n\nThird mine.\n",
        )
        self.assertEqual(
            tricks.read_tricks_data(),
            {
                "tricks": [
                    "First mine.",
                    "Third mine.",
                    "Bundled promptlet one.",
                    "Bundled two.",
                ],
                "userCount": 2,
            },
        )

    def test_deletes_first_and_last_sections(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        self.assertIsNone(tricks.delete_my_injection_trick("First mine."))
        self.assertIsNone(tricks.delete_my_injection_trick("Third mine."))
        self.assertEqual(
            self.user_text(),
            "Notes the user typed above the first heading.\n\n"
            "## Other\n\nNot a trick, stays put.\n\n"
            "## Trick\n\nSecond mine.\n",
        )

    def test_deleting_the_only_trick_leaves_an_empty_file_unseeded(self) -> None:
        # "Notes" that are only blank lines vanish with the section.
        # First read seeds the default; deleting it must not re-seed it,
        # or the user could never get rid of the starter promptlet.
        self.assertEqual(tricks.read_tricks_data()["userCount"], 1)
        self.assertIsNone(
            tricks.delete_my_injection_trick(tricks.MY_INJECTION_DEFAULT_BODY)
        )
        self.assertEqual(self.user_text(), "")
        self.assertEqual(
            tricks.read_tricks_data(),
            {"tricks": ["Bundled promptlet one.", "Bundled two."], "userCount": 0},
        )

    def test_deletes_every_duplicate_section(self) -> None:
        self.user_file.write_text(
            "## Trick\n\nTwice.\n\n## Trick\n\nKeep.\n\n## Trick\n\nTwice.\n",
            encoding="utf-8",
        )
        self.assertIsNone(tricks.delete_my_injection_trick("Twice."))
        self.assertEqual(self.user_text(), "## Trick\n\nKeep.\n")

    def test_matches_the_unescaped_body_the_panel_shows(self) -> None:
        # Added through the panel: the stored body is backslash-escaped,
        # the panel lists (and sends back) the unescaped text.
        self.assertIsNone(tricks.append_my_injection_trick(r"Use C:\*.txt globs"))
        self.assertIn(r"Use C:\\*.txt globs", self.user_text())
        self.assertIn(r"Use C:\*.txt globs", tricks.read_tricks())
        self.assertIsNone(tricks.delete_my_injection_trick(r"Use C:\*.txt globs"))
        self.assertEqual(self.user_text(), tricks.DEFAULT_MY_INJECTION)

    def test_crlf_body_from_the_vscode_page_matches_the_lf_section(self) -> None:
        # The VS Code page lists a CRLF file's bodies with their CRLFs
        # (SorcarTab.ts keeps them); this module's parser reads them as
        # LF, so the delete must accept either spelling.
        self.user_file.write_bytes(
            b"## Trick\r\n\r\nFirst line\r\nSecond line\r\n\r\n## Trick\r\n\r\nKeep.\r\n"
        )
        self.assertEqual(
            tricks.read_tricks()[:2], ["First line\nSecond line", "Keep."]
        )
        self.assertIsNone(
            tricks.delete_my_injection_trick("First line\r\nSecond line\r\n")
        )
        # The rewrite writes LF on every platform (the CRLF input is normalized).
        self.assertEqual(self.user_file.read_bytes(), b"## Trick\n\nKeep.\n")
        self.assertEqual(tricks.read_tricks_data()["userCount"], 1)

    def test_multiline_body_is_matched_whole(self) -> None:
        self.assertIsNone(tricks.append_my_injection_trick("Line one.\nLine two."))
        self.assertEqual(
            tricks.delete_my_injection_trick("Line one."),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertIsNone(tricks.delete_my_injection_trick("Line one.\nLine two."))
        self.assertEqual(self.user_text(), tricks.DEFAULT_MY_INJECTION)

    def test_unknown_body_reports_error_without_touching_file(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        self.assertEqual(
            tricks.delete_my_injection_trick("Bundled promptlet one."),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(
            tricks.delete_my_injection_trick(""),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), _USER)

    def test_section_with_another_heading_is_never_deleted(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        self.assertEqual(
            tricks.delete_my_injection_trick("Not a trick, stays put."),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), _USER)

    def test_file_without_headings_reports_error(self) -> None:
        self.user_file.write_text("just prose\n", encoding="utf-8")
        self.assertEqual(
            tricks.delete_my_injection_trick("just prose"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), "just prose\n")

    def test_missing_file_is_seeded_then_default_can_be_deleted(self) -> None:
        self.assertFalse(self.user_file.exists())
        self.assertEqual(
            tricks.delete_my_injection_trick("nope"),
            "That promptlet is not in ~/.kiss/MY_INJECTION.md",
        )
        self.assertEqual(self.user_text(), tricks.DEFAULT_MY_INJECTION)

    def test_non_utf8_file_reports_error_without_writing(self) -> None:
        self.user_file.write_bytes(b"## Trick\n\n\xff\xfe\n")
        self.assertEqual(
            tricks.delete_my_injection_trick("x"),
            "~/.kiss/MY_INJECTION.md is not UTF-8 text",
        )
        self.assertEqual(self.user_file.read_bytes(), b"## Trick\n\n\xff\xfe\n")

    @posix_only("directory permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_unwritable_home_reports_error(self) -> None:
        self.kiss_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            self.assertEqual(
                tricks.delete_my_injection_trick("x"),
                "Could not read ~/.kiss/MY_INJECTION.md",
            )
        finally:
            self.kiss_dir.chmod(stat.S_IRWXU)
        self.assertFalse(self.user_file.exists())

    def test_concurrent_deletes_of_different_tricks_keep_the_rest(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        results: list[str | None] = []
        barrier = threading.Barrier(2)

        def run(body: str) -> None:
            barrier.wait(timeout=5)
            results.append(tricks.delete_my_injection_trick(body))

        threads = [
            threading.Thread(target=run, args=("First mine.",)),
            threading.Thread(target=run, args=("Third mine.",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(results, [None, None])
        self.assertEqual(
            tricks.read_tricks_data(),
            {
                "tricks": ["Second mine.", "Bundled promptlet one.", "Bundled two."],
                "userCount": 1,
            },
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

    def last(self, event_type: str) -> dict[str, Any] | None:
        for msg in reversed(self.printer.messages):
            if msg.get("type") == event_type:
                return dict(msg)
        return None


class TestDeleteTrickCommand(_TricksHome):
    """The daemon's ``deleteTrick`` rewrites the file and answers the panel."""

    def setUp(self) -> None:
        super().setUp()
        self.server = _FakeServer()
        self.user_file.write_text(_USER, encoding="utf-8")

    def test_command_is_in_the_catalog_and_handler_table(self) -> None:
        self.assertEqual(API["deleteTrick"].required, ("text",))
        self.assertEqual(API["deleteTrick"].handler, "forward")
        self.assertIn("deleteTrick", _CommandsMixin._HANDLERS)
        self.assertIsNone(validate_command({"type": "deleteTrick", "text": "x"}))
        self.assertIsNotNone(validate_command({"type": "deleteTrick"}))

    def test_delete_rewrites_file_and_broadcasts_unstamped_list(self) -> None:
        self.server._cmd_delete_trick({"text": "Second mine.", "connId": "c1"})
        msg = self.server.last("tricksData")
        assert msg is not None
        self.assertNotIn(
            "connId", msg, "deletes repaint every window, not only the sender"
        )
        self.assertEqual(
            msg["tricks"],
            ["First mine.", "Third mine.", "Bundled promptlet one.", "Bundled two."],
        )
        self.assertEqual(msg["userCount"], 2)
        self.assertNotIn("## Trick\n\nSecond mine.", self.user_text())
        self.assertIsNone(self.server.last("error"))

    def test_add_reply_also_carries_user_count(self) -> None:
        self.server._cmd_add_trick({"text": "Fourth mine."})
        msg = self.server.last("tricksData")
        assert msg is not None
        self.assertEqual(msg["userCount"], 4)
        self.assertEqual(msg["tricks"][3], "Fourth mine.")

    def test_unknown_body_answers_sender_with_error(self) -> None:
        self.server._cmd_delete_trick(
            {"text": "Bundled promptlet one.", "connId": "c9"}
        )
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c9")
        self.assertEqual(
            err["text"], "That promptlet is not in ~/.kiss/MY_INJECTION.md"
        )
        self.assertEqual(self.user_text(), _USER)
        # The panel dropped the row before asking; the error is followed
        # by the list on disk, stamped for the sender, to put it back.
        self.assertEqual(
            [m["type"] for m in self.server.printer.messages],
            ["error", "tricksData"],
        )
        resync = self.server.printer.messages[-1]
        self.assertEqual(resync["connId"], "c9")
        self.assertEqual(resync["userCount"], 3)
        self.assertEqual(resync["tricks"][:3], ["First mine.", "Second mine.", "Third mine."])

    def test_non_string_text_is_treated_as_empty(self) -> None:
        self.server._cmd_delete_trick({"text": ["First mine."]})
        err = self.server.last("error")
        assert err is not None
        self.assertNotIn("connId", err)
        self.assertEqual(
            err["text"], "That promptlet is not in ~/.kiss/MY_INJECTION.md"
        )
        self.assertEqual(self.user_text(), _USER)
        # No connId to stamp: the resync reaches every window, unstamped.
        resync = self.server.last("tricksData")
        assert resync is not None
        self.assertNotIn("connId", resync)

    @posix_only("directory and file permission bits")
    @unittest.skipIf(is_root(), "root ignores file permissions")
    def test_os_error_on_write_answers_error_not_crash(self) -> None:
        self.user_file.chmod(stat.S_IRUSR)
        try:
            self.server._cmd_delete_trick({"text": "First mine.", "connId": "c2"})
        finally:
            self.user_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c2")
        self.assertTrue(
            err["text"].startswith("Could not write ~/.kiss/MY_INJECTION.md: ")
        )
        self.assertEqual(self.user_text(), _USER)
        resync = self.server.last("tricksData")
        assert resync is not None
        self.assertEqual(resync["connId"], "c2")
        self.assertEqual(resync["userCount"], 3)
        self.assertIn("First mine.", resync["tricks"])


class TestRemotePageUserCount(_TricksHome):
    """The remote page freezes the user count next to the trick list."""

    def test_build_html_sets_my_tricks_count(self) -> None:
        self.user_file.write_text(_USER, encoding="utf-8")
        html = _build_html()
        self.assertNotIn("{{MY_TRICKS_COUNT}}", html)
        m = re.search(r"window\.__MY_TRICKS_COUNT__\s*=\s*(\d+);", html)
        assert m is not None, "MY_TRICKS_COUNT placeholder was not substituted"
        self.assertEqual(int(m.group(1)), 3)
        t = re.search(r"window\.__TRICKS__\s*=\s*(\[.*?\]);", html, re.DOTALL)
        assert t is not None
        self.assertEqual(json.loads(t.group(1))[:3], [
            "First mine.", "Second mine.", "Third mine.",
        ])


class TestDeleteTrickOverLocalConnection(_TricksHome):
    """``deleteTrick`` travels the real transport: local WSS → catalog → handler."""

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
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=30,
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

    def test_delete_over_the_wire_rewrites_file_and_returns_list(self) -> None:
        event = self._roundtrip(
            {"type": "deleteTrick", "text": "First mine."}, "tricksData"
        )
        self.assertEqual(
            event["tricks"],
            ["Second mine.", "Third mine.", "Bundled promptlet one.", "Bundled two."],
        )
        self.assertEqual(event["userCount"], 2)
        self.assertNotIn("First mine.", self.user_text())

    def test_missing_text_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip({"type": "deleteTrick"}, "error")
        self.assertEqual(
            event["text"], "Invalid deleteTrick command: missing text"
        )
        self.assertEqual(self.user_text(), _USER)

    def test_unknown_body_error_reaches_the_sender(self) -> None:
        event = self._roundtrip(
            {"type": "deleteTrick", "text": "Bundled two."}, "error"
        )
        self.assertEqual(
            event["text"], "That promptlet is not in ~/.kiss/MY_INJECTION.md"
        )


if __name__ == "__main__":
    unittest.main()
