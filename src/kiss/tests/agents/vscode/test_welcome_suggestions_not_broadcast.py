"""Regression test: a webapp client requesting welcome suggestions must
NOT cause a ``welcome_suggestions`` event to reach other already-connected
clients (notably the VS Code extension over its UDS connection).

Bug
---
When the user opens a new chat in the webapp, the webapp sends a
``ready`` (or ``getWelcomeSuggestions``) command to the daemon.  The
old implementation responded by **broadcasting**
``{"type": "welcome_suggestions", "suggestions": []}`` to every
connected client via ``WebPrinter.broadcast``.  Because the VS Code
extension is also a client (over UDS), it received the empty list and
forwarded it to its webview, where ``renderWelcomeSuggestions`` cleared
the ``#suggestions`` container — wiping out the SAMPLE_TASKS chips that
the extension populates locally from ``SAMPLE_TASKS.json``.

Fix
---
``_send_welcome_info`` no longer broadcasts ``welcome_suggestions`` (the
remote-chat webapp hides ``#suggestions`` via CSS so it never needed
the event anyway).  The remote URL is still broadcast since multiple
webapp tabs may all be interested in the same URL.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestWelcomeSuggestionsNotBroadcast(IsolatedAsyncioTestCase):
    """Welcome suggestions must not leak from webapp -> extension."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _read_event(
        self, reader: asyncio.StreamReader, timeout: float = 1.0,
    ) -> dict[str, object]:
        """Read one newline-delimited JSON message from the UDS."""
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "UDS closed unexpectedly"
        msg = json.loads(line.decode("utf-8"))
        assert isinstance(msg, dict)
        return msg

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        wanted_type: str,
        max_events: int = 50,
        timeout: float = 1.0,
    ) -> dict[str, object]:
        for _ in range(max_events):
            msg = await self._read_event(reader, timeout=timeout)
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(f"never observed {wanted_type!r}")

    async def test_webapp_new_chat_does_not_clobber_extension_suggestions(
        self,
    ) -> None:
        """A second client sending ``getWelcomeSuggestions`` (as the
        webapp does whenever a new chat tab is opened) must NOT cause
        a ``welcome_suggestions`` event to be delivered to a
        previously connected client (the extension).
        """
        # 1) Extension-side connection: registers with the broadcaster.
        ext_reader, ext_writer = await asyncio.open_unix_connection(
            str(self.uds_path),
        )
        try:
            ext_writer.write(
                json.dumps(
                    {
                        "type": "ready",
                        "tabId": "ext-tab",
                        "restoredTabs": [],
                    },
                ).encode("utf-8") + b"\n",
            )
            await ext_writer.drain()
            # Drain the extension's own ready handshake events.
            await self._drain_until(ext_reader, "focusInput", timeout=2.0)
            # Pull out any welcome_suggestions emitted for THIS client's
            # own ready handshake — those are expected.  We only care
            # about events that arrive AFTER the second client connects.
            try:
                while True:
                    await asyncio.wait_for(ext_reader.readline(), timeout=0.2)
            except TimeoutError:
                pass

            # 2) Second client (the webapp opening a new chat tab) sends
            #    a getWelcomeSuggestions command.
            web_reader, web_writer = await asyncio.open_unix_connection(
                str(self.uds_path),
            )
            try:
                web_writer.write(
                    json.dumps(
                        {"type": "getWelcomeSuggestions"},
                    ).encode("utf-8") + b"\n",
                )
                await web_writer.drain()

                # 3) The first (extension) client must NOT receive a
                #    welcome_suggestions event triggered by the second
                #    client's request.
                for _ in range(50):
                    try:
                        msg = await self._read_event(
                            ext_reader, timeout=0.3,
                        )
                    except TimeoutError:
                        break
                    self.assertNotEqual(
                        msg.get("type"),
                        "welcome_suggestions",
                        "Other client received welcome_suggestions from "
                        f"another client's request: {msg}",
                    )
            finally:
                web_writer.close()
                try:
                    await web_writer.wait_closed()
                except Exception:
                    pass
        finally:
            ext_writer.close()
            try:
                await ext_writer.wait_closed()
            except Exception:
                pass


if __name__ == "__main__":
    import unittest
    unittest.main()
