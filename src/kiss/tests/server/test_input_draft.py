# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the daemon's side of the composer's Save button.

``saveInputDraft {text, was, seq}`` and ``getInputDraft`` keep one text
in ``$KISS_HOME/saved_input.txt`` (:mod:`kiss.server.input_draft`).
Every test talks to a live :class:`RemoteAccessServer` over real local
connections, the ``_ws_handler`` path every client uses, with
``KISS_HOME`` in a folder of its own, and checks both the ``inputDraft``
replies and the file.  The commands are the ones the chat page posts in
the scenarios of ``webapp/test/composerSaveInput.test.js``, where the
replies asserted here are played back to the page.

A failing write needs no test double: a directory standing where the
file belongs makes the write and the read fail, and a read-only home
(on Windows, where a folder has no such bit, a read-only file) makes the
delete fail.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.server.input_draft import MAX_INPUT_DRAFT_BYTES
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.conftest import environ


class TestInputDraftCommands(unittest.TestCase):
    """The live dispatcher saves, returns and deletes the composer's saved text."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.enterContext(environ(KISS_HOME=str(self.home)))
        self.saved = self.home / "saved_input.txt"
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.addCleanup(self._stop_loop)
        self.server = self._start_daemon()

    def _stop_loop(self) -> None:
        self._stop_daemon()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def _start_daemon(self) -> RemoteAccessServer:
        """Start a daemon on the test's home and return it."""
        server = RemoteAccessServer(
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        asyncio.run_coroutine_threadsafe(server.start_private_async(), self.loop).result(timeout=30)
        return server

    def _stop_daemon(self) -> None:
        """Stop the running daemon and wait until it is down."""
        stopping = asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)
        concurrent.futures.wait([stopping], timeout=30)

    async def _exchange(self, peers: int, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send each command on its own turn and return its reply.

        Args:
            peers: How many connections to open; command *i* travels on
                connection ``i % peers``.
            commands: The commands, sent one after the other.

        Returns:
            The first event each command's connection received after it.
        """
        conns = [await open_local_connection(self.server) for _ in range(peers)]
        replies: list[dict[str, Any]] = []
        try:
            for i, cmd in enumerate(commands):
                reader, writer = conns[i % peers]
                writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                replies.append(json.loads(await asyncio.wait_for(reader.readline(), timeout=10)))
        finally:
            for _reader, writer in conns:
                writer.close()
                await writer.wait_closed()
        return replies

    def _ask(self, *commands: dict[str, Any], peers: int = 1) -> list[dict[str, Any]]:
        """Run :meth:`_exchange` on the server's loop and return the replies."""
        return asyncio.run_coroutine_threadsafe(
            self._exchange(peers, list(commands)), self.loop,
        ).result(timeout=60)

    def _event(self, text: str, **extra: Any) -> dict[str, Any]:
        """The ``inputDraft`` event expected when the file holds *text*."""
        return {"type": "inputDraft", "text": text, "path": str(self.saved), **extra}

    @staticmethod
    def _save(text: Any, seq: int, *was: str) -> dict[str, Any]:
        """The ``saveInputDraft`` command number *seq* of a page that names *was*."""
        return {"type": "saveInputDraft", "text": text, "was": list(was), "seq": seq}

    def test_save_read_back_overwrite_and_removal(self) -> None:
        read = {"type": "getInputDraft"}
        self.assertEqual(self._ask(read), [self._event("")])
        # A lone surrogate is a string JavaScript can hold: it is kept as it came.
        first = "first\n  draft \u00e9 \U0001f600 \ud800"
        self.assertEqual(self._ask(self._save(first, 1, first)), [self._event(first, seq=1)])
        self.assertEqual(self.saved.read_bytes(), first.encode("utf-8", "surrogatepass"))
        self.assertEqual(self._ask(read), [self._event(first)])
        # The file is this daemon's: a machine stamp sends the text nowhere else.
        second = {**self._save("second", 2, first, "second"), "machine": "elsewhere"}
        self.assertEqual(self._ask(second), [self._event("second", seq=2)])
        self.assertEqual(self.saved.read_text(), "second")
        # Scenario 6: the answer to "second" never reached the page, so its
        # removal names both texts; whichever the file holds, it goes.
        removal = self._save("", 3, first, "second")
        for held in ("second", first):
            self.saved.write_bytes(held.encode("utf-8", "surrogatepass"))
            self.assertEqual(self._ask(removal, read), [self._event("", seq=3), self._event("")])
            self.assertFalse(self.saved.exists())
        # Sent again (the page posts what is unanswered at every connection).
        self.assertEqual(self._ask(removal), [self._event("", seq=3)])

    def test_removal_keeps_a_text_its_browser_cannot_account_for(self) -> None:
        # Scenario 10: browser A holds saved "T", browser B saves "T2", A sends its box.
        replies = self._ask(
            self._save("T", 1, "T"), self._save("T2", 1, "T2"), self._save("", 2, "T"), peers=2,
        )
        self.assertEqual(replies[2], self._event("T2", seq=2))
        self.assertEqual(self.saved.read_text(), "T2")
        # A removal that names nothing, or no list, deletes nothing either.
        nameless = {"type": "saveInputDraft", "text": ""}
        replies = self._ask(self._save("", 3), {**nameless, "was": "T2", "seq": 4}, nameless)
        self.assertEqual(
            replies,
            [self._event("T2", seq=3), self._event("T2", seq=4), self._event("T2", seq=None)],
        )
        self.assertEqual(self.saved.read_text(), "T2")

    def test_every_command_answers_its_own_number_and_may_be_repeated(self) -> None:
        # Scenario 11: A, B, A clicked fast; each answer says what the file held then.
        texts = ("A", "B")
        saves = [self._save("A", 1, "A"), self._save("B", 2, *texts), self._save("A", 3, *texts)]
        self.assertEqual(
            self._ask(*saves),
            [self._event("A", seq=1), self._event("B", seq=2), self._event("A", seq=3)],
        )
        # Scenarios 3 and 4: the transport replays a save the page posts again.
        self.assertEqual(
            self._ask(saves[2], {**saves[2], "seq": 4}),
            [self._event("A", seq=3), self._event("A", seq=4)],
        )
        self.assertEqual(self.saved.read_text(), "A")

    def test_text_over_1_mb_or_no_string_is_refused_and_the_file_kept(self) -> None:
        at_cap = "a" * MAX_INPUT_DRAFT_BYTES
        self.assertEqual(self._ask(self._save(at_cap, 1)), [self._event(at_cap, seq=1)])
        too_long = self._event(at_cap, error="Not saved: the text is longer than 1 MB.", seq=2)
        # Bytes count, not characters: the second text is half as many letters.
        letters = "a" * (MAX_INPUT_DRAFT_BYTES + 1)
        accents = "\u00e9" * (MAX_INPUT_DRAFT_BYTES // 2 + 1)
        replies = self._ask(
            self._save(letters, 2), self._save(accents, 2), self._save(["a"], 2),
            {"type": "saveInputDraft", "seq": 2},
        )
        self.assertEqual(replies, [
            too_long,
            too_long,
            self._event(at_cap, error="Not saved: the text is no string.", seq=2),
            {"type": "error", "text": "Invalid saveInputDraft command: missing text"},
        ])
        self.assertEqual(self.saved.read_text(), at_cap)

    def test_reply_goes_to_the_sender_only(self) -> None:
        saved, other = self._ask(self._save("mine", 1), {"type": "activeTasksQuery"}, peers=2)
        self.assertEqual(saved, self._event("mine", seq=1))
        self.assertEqual(other["type"], "activeTasksResponse")

    def test_the_file_survives_a_daemon_restart_and_is_all_that_is_kept(self) -> None:
        self.assertEqual(self._ask(self._save("kept", 1)), [self._event("kept", seq=1)])
        self._stop_daemon()
        self.server = self._start_daemon()
        self.assertEqual(self._ask({"type": "getInputDraft"}), [self._event("kept")])
        # A removal a page owed across the restart still finds its text.
        self.assertEqual(self._ask(self._save("again", 1), self._save("", 2, "kept", "again")), [
            self._event("again", seq=1),
            self._event("", seq=2),
        ])
        # No record beside the file, and nothing left of the writes.
        self._ask(self._save("last", 3))
        names = [p.name for p in self.home.iterdir() if "saved_input" in p.name]
        self.assertEqual(names, ["saved_input.txt"])

    def test_a_file_that_cannot_be_changed_or_read_is_reported(self) -> None:
        # A directory stands where the file belongs: nothing is saved, nothing reads.
        self.saved.mkdir()
        reply = self._ask(self._save("words", 1))[0]
        self.assertEqual({**reply, "error": ""}, self._event("", error="", seq=1))
        self.assertTrue(
            reply["error"].startswith(f"The saved text in {self.saved} was not changed: "), reply,
        )
        self.assertEqual(
            self._ask(self._save("", 2, ""), {"type": "getInputDraft"}),
            [self._event("", seq=2), self._event("")],
        )
        self.assertEqual(os.listdir(self.saved), [])
        # Bytes that are no text read as nothing saved, and are not a removal's to delete.
        self.saved.rmdir()
        self.saved.write_bytes(b"\xff\xfe no text")
        self.assertEqual(self._ask(self._save("", 3, "")), [self._event("", seq=3)])
        self.assertTrue(self.saved.exists())
        # The page posts the save that failed again: now it is written.
        self.assertEqual(self._ask(self._save("words", 4)), [self._event("words", seq=4)])
        # A home that takes no change: the removal says so and the text stays.  Windows
        # keeps no such bit on a folder, and deletes no file that takes no change.
        self.home.chmod(0o500)
        self.addCleanup(self.home.chmod, 0o700)
        self.saved.chmod(0o400)
        self.addCleanup(self.saved.chmod, 0o600)
        reply = self._ask(self._save("", 5, "words"))[0]
        self.assertEqual({**reply, "error": ""}, self._event("words", error="", seq=5))
        self.assertIn("was not changed", reply["error"])


if __name__ == "__main__":
    unittest.main()
