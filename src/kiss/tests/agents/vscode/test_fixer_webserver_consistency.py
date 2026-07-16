# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2e regression tests for the web_server consistency fixer wave.

Covers three behavior changes:

1. ``_fanout_cli_status`` now stamps ``taskId`` into the ``status``
   event it fans out to subscribed tabs (previously the hand-rolled
   fanout omitted it, unlike every other task-scoped status
   broadcast).  Driven end-to-end over a real UDS transport: a CLI
   connection announces ``cliTaskStart``, a viewer tab subscribes via
   the production history-click path (``_replay_session``), and the
   CLI's ``cliTaskEnd`` must deliver ``status:running=false`` carrying
   BOTH the viewer's ``tabId`` and the task's ``taskId``.

2. ``_MAX_PROMPT_BYTES`` is now enforced in UTF-8 **bytes** (matching
   its name), not characters.  A 400k-character ``€`` prompt is
   1.2 MB on the wire — under the old character-based check it passed
   untruncated; now it must be truncated to <= 1 MB without ever
   splitting inside a multibyte character.

3. ``_version_tuple`` is now strict digits-only per component
   (``/^\\d+$/`` like the extension's ``UpdateChecker.js`` twin), so
   ``"+1"`` / ``"1_0"`` / unicode digits no longer parse, and
   ``_parse_version_py`` uses the exact JS regex (accepting indented
   assignments, rejecting unquoted values).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.persistence import _add_task, _append_chat_event
from kiss.server.web_server import (
    _MAX_LINE_BYTES,
    _MAX_PROMPT_BYTES,
    RemoteAccessServer,
    _compare_versions,
    _parse_version_py,
    _version_tuple,
)


class _UdsHarness(unittest.TestCase):
    """Shared UDS harness: RemoteAccessServer on a temp unix socket.

    Mirrors the production wiring used by
    ``test_cli_history_click_resumes_live_stream.py``: a background
    event loop runs the server's ``_uds_handler`` on a temp socket and
    persistence is pointed at a temp sqlite DB so nothing pollutes the
    user's real ``~/.kiss``.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-fixer-webserver-")
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")

        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        self.server = RemoteAccessServer(uds_path=self.sock_path)
        # ``_send_to_ws_clients`` / ``_run_cmd`` need the loops set the
        # way ``start_async`` would set them.
        self.server._printer._loop = self.loop
        self.server._loop = self.loop

        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler,
                path=self.sock_path,
                limit=_MAX_LINE_BYTES,
            ),
            self.loop,
        ).result(timeout=5)

        self._writers: list[asyncio.StreamWriter] = []

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            for writer in self._writers:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            self.uds_server.close()
            await self.uds_server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_conn(
        self,
    ) -> tuple[asyncio.StreamWriter, list[str], threading.Event]:
        """Open a UDS client; return (writer, received-lines, got-event)."""

        async def _open() -> tuple[
            asyncio.StreamReader, asyncio.StreamWriter,
        ]:
            return await asyncio.open_unix_connection(
                self.sock_path, limit=_MAX_LINE_BYTES,
            )

        reader, writer = asyncio.run_coroutine_threadsafe(
            _open(), self.loop,
        ).result(timeout=5)
        self._writers.append(writer)

        received: list[str] = []
        got = threading.Event()

        async def _drain() -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                received.append(line.decode("utf-8"))
                got.set()

        asyncio.run_coroutine_threadsafe(_drain(), self.loop)
        return writer, received, got

    def _send_line(self, writer: asyncio.StreamWriter,
                   cmd: dict[str, Any]) -> None:
        """Write one JSON command line on *writer* from the test thread."""

        async def _send() -> None:
            writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
            await writer.drain()

        asyncio.run_coroutine_threadsafe(_send(), self.loop).result(timeout=10)

    @staticmethod
    def _decoded(received: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in list(received):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _wait_for(self, predicate: Any, message: str,
                  timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(message)


class TestFanoutCliStatusCarriesTaskId(_UdsHarness):
    """``cliTaskEnd`` fanout must stamp ``taskId`` like every other
    task-scoped status broadcast."""

    def test_cli_task_end_status_carries_task_id(self) -> None:
        """Viewer subscribed to a CLI task gets status with taskId."""
        task_id, chat_id = _add_task(task="fanout taskId task", chat_id="")
        _append_chat_event(
            {"type": "prompt", "text": "fanout taskId task"},
            task_id=task_id,
        )
        task_id_str = str(task_id)

        # Phase 1: the CLI process announces the running task over UDS
        # (exercises the merged cliTaskStart/cliTaskEnd dispatch branch
        # and ``_validated_cli_task_id``).
        cli_writer, _cli_received, _cli_got = self._open_conn()
        self._send_line(
            cli_writer, {"type": "cliTaskStart", "taskId": task_id_str},
        )
        self._wait_for(
            lambda: self.server._is_cli_task_running(task_id_str),
            "cliTaskStart never registered the task as running",
        )

        # Phase 2: a viewer tab clicks the running task in the History
        # panel — the production subscription path.
        _viewer_writer, received, got = self._open_conn()
        viewer_tab = "fanout-taskid-tab"
        self.server._vscode_server._replay_session(
            chat_id=chat_id, tab_id=viewer_tab, task_id=task_id,
        )
        self._wait_for(
            lambda: viewer_tab in self.server._printer._subscribers.get(
                task_id_str, set(),
            ),
            "viewer tab never subscribed to the CLI task",
        )
        # Drain the replay burst before watching for the end event.
        time.sleep(0.2)
        received.clear()
        got.clear()

        # Phase 3: the CLI ends the task; the fanned-out status event
        # must carry running=False, the viewer's tabId AND the taskId.
        self._send_line(
            cli_writer, {"type": "cliTaskEnd", "taskId": task_id_str},
        )

        def _got_status_end() -> bool:
            return any(
                d.get("type") == "status" and d.get("running") is False
                and d.get("tabId") == viewer_tab
                for d in self._decoded(received)
            )

        self._wait_for(
            _got_status_end,
            f"viewer never received status:running=false; got {received}",
        )
        status_evs = [
            d for d in self._decoded(received)
            if d.get("type") == "status" and d.get("running") is False
            and d.get("tabId") == viewer_tab
        ]
        for ev in status_evs:
            self.assertEqual(
                ev.get("taskId"), task_id_str,
                "status:running=false fanned out by cliTaskEnd must "
                f"carry the taskId like every other status broadcast; "
                f"got {ev}",
            )


class TestPromptTruncationIsByteBased(_UdsHarness):
    """``submit`` prompts are capped at ``_MAX_PROMPT_BYTES`` UTF-8
    bytes, never splitting inside a character."""

    def test_multibyte_prompt_truncated_to_byte_cap(self) -> None:
        """A 400k-char (1.2 MB) '€' prompt is truncated to <= 1 MB."""
        n_chars = 400_000
        prompt = "\u20ac" * n_chars  # 3 bytes each → 1.2 MB > 1 MB cap
        self.assertGreater(
            len(prompt.encode("utf-8")), _MAX_PROMPT_BYTES,
            "test prompt must exceed the byte cap",
        )
        self.assertLess(
            len(prompt), _MAX_PROMPT_BYTES,
            "test prompt must be under the cap in CHARACTERS so the "
            "old char-based check would not have truncated it",
        )

        writer, received, got = self._open_conn()
        # Empty tabId: ``_handle_submit`` broadcasts the setTaskText
        # echo first, then the translated ``run`` is dropped by
        # ``_cmd_run``'s empty-tabId guard — no real agent starts.
        self._send_line(writer, {
            "type": "submit",
            "prompt": prompt,
            "tabId": "",
            "model": "",
            "attachments": [],
        })

        def _task_text() -> dict[str, Any] | None:
            for d in self._decoded(received):
                if d.get("type") == "setTaskText":
                    return d
            return None

        self._wait_for(
            lambda: _task_text() is not None,
            "never received the setTaskText echo",
            timeout=10.0,
        )
        text = _task_text()["text"]  # type: ignore[index]
        encoded = text.encode("utf-8")
        self.assertLessEqual(
            len(encoded), _MAX_PROMPT_BYTES,
            f"setTaskText echo is {len(encoded)} bytes — the prompt cap "
            "must be enforced in BYTES, not characters",
        )
        # The cut must never split inside a character: the result is
        # exactly the longest whole-character prefix under the cap.
        self.assertEqual(text, "\u20ac" * (_MAX_PROMPT_BYTES // 3))

    def test_lone_surrogate_prompt_does_not_kill_transport(self) -> None:
        """JSON permits escaped lone surrogates; prompt sizing must too.

        A strict ``prompt.encode('utf-8')`` raises UnicodeEncodeError on
        this value and used to escape from ``_dispatch_client_command``;
        the command produced no ``setTaskText`` reply (and less-isolated
        transports may close their receive loop).
        """
        prompt = "before-\ud800-after"
        writer, received, _got = self._open_conn()
        self._send_line(writer, {
            "type": "submit",
            "prompt": prompt,
            "tabId": "",
            "model": "",
            "attachments": [],
        })

        def _echo() -> dict[str, Any] | None:
            return next(
                (d for d in self._decoded(received)
                 if d.get("type") == "setTaskText"),
                None,
            )

        self._wait_for(
            lambda: _echo() is not None,
            f"lone-surrogate prompt killed the transport; got {received}",
        )
        self.assertEqual(_echo().get("text"), prompt)  # type: ignore[union-attr]


class TestVersionHelpersStrict(unittest.TestCase):
    """Python version helpers must match the JS UpdateChecker twins."""

    def test_version_tuple_rejects_non_digit_components(self) -> None:
        """int()-isms like '+1', '1_0' and unicode digits are rejected."""
        self.assertIsNone(_version_tuple("+1"))
        self.assertIsNone(_version_tuple("2026.+1"))
        self.assertIsNone(_version_tuple("1_0"))
        self.assertIsNone(_version_tuple("2026.1_0"))
        self.assertIsNone(_version_tuple("\u0661.\u0662"))  # ١.٢
        self.assertIsNone(_version_tuple("1.-2"))
        self.assertIsNone(_version_tuple(" "))

    def test_version_tuple_keeps_lenient_whitespace_and_dots(self) -> None:
        """Trim + empty-component filtering behavior is preserved."""
        self.assertEqual(_version_tuple("2026.6.1"), (2026, 6, 1))
        self.assertEqual(_version_tuple(" 2026.6 "), (2026, 6))
        self.assertEqual(_version_tuple("2026..6"), (2026, 6))

    def test_compare_versions_treats_malformed_as_equal(self) -> None:
        """Unparseable inputs compare equal — never a false update."""
        self.assertEqual(_compare_versions("2026.+1", "2026.0"), 0)
        self.assertEqual(_compare_versions("1_0", "0.1"), 0)
        self.assertEqual(_compare_versions("2026.7", "2026.6.9"), 1)
        self.assertEqual(_compare_versions("2026.6", "2026.6.0"), 0)

    def test_parse_version_py_matches_js_regex(self) -> None:
        """Indented assignments parse; unquoted values do not."""
        tmpdir = Path(tempfile.mkdtemp(prefix="kiss-fixer-version-"))
        try:
            indented = tmpdir / "indented_version.py"
            indented.write_text(
                'if True:\n    __version__ = "1.2.3"\n', encoding="utf-8",
            )
            self.assertEqual(_parse_version_py(indented), "1.2.3")

            unquoted = tmpdir / "unquoted_version.py"
            unquoted.write_text("__version__ = 1.2\n", encoding="utf-8")
            self.assertEqual(_parse_version_py(unquoted), "")

            self.assertEqual(
                _parse_version_py(tmpdir / "missing.py"), "",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
