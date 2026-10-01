# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: autocomplete must not echo-strip suffix suggestions.

The ghost-text pipeline is purely local now: the call site of
:func:`kiss.server.helpers.clip_autocomplete_suggestion`
(``_AutocompleteMixin._complete``) passes a suggestion that is
ALREADY the continuation suffix (never an LLM echo of the query) —
``match[len(query):]`` or an identifier suffix from
``_active_file_identifier_matches``.

``clip_autocomplete_suggestion`` nevertheless kept a vestigial
"strip the query prefix if the LLM echoed it" step::

    if s.lower().startswith(query.lower()):
        s = s[len(query):]

which DOUBLE-strips any legitimate suffix that happens to begin with
the query text.  Example: the active file contains the identifier
``quxqux_token`` and the user types ``qux``.  The correct continuation
is ``qux_token`` (accepting the ghost yields ``quxqux_token``), but the
echo-strip clipped it to ``_token`` so accepting produced the
non-existent identifier ``qux_token``.

These are end-to-end reproductions through the real pipeline:
the daemon's local-channel ``complete`` command -> ``ghost`` event.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection


def _redirect_persistence(tmpdir: str) -> tuple[Any, Any, Any]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple[Any, Any, Any]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestGhostSuffixNotEchoStripped(IsolatedAsyncioTestCase):
    """Local ``complete`` -> ``ghost`` round trip through the real daemon."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self._writers: list[LocalWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
        reader, writer = await open_local_connection(self.server)
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: LocalWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: LocalReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "local connection closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    @staticmethod
    def _ghost_for(query: str) -> Callable[[dict[str, Any]], bool]:
        def _pred(msg: dict[str, Any]) -> bool:
            return msg.get("type") == "ghost" and msg.get("query") == query
        return _pred

    async def test_suffix_starting_with_query_survives(self) -> None:
        """Identifier ``quxqux_token`` typed as ``qux`` must suggest
        ``qux_token`` so the accepted text is the real identifier."""
        reader, writer = await self._connect()
        await self._send(writer, {
            "type": "complete",
            "query": "qux",
            "activeFile": "/tmp/bughunt3_file.py",
            "activeFileContent": "x = quxqux_token\n",
        })
        ghost = await self._drain_until(reader, self._ghost_for("qux"))
        self.assertEqual(
            ghost.get("suggestion"), "qux_token",
            "suffix beginning with the query was echo-stripped: accepting "
            f"the ghost would type 'qux{ghost.get('suggestion')}' instead "
            "of the real identifier 'quxqux_token'",
        )

    async def test_doubled_word_continuation_survives(self) -> None:
        """Dot-chain ``ab.ab.cd`` typed as ``ab.`` must suggest the full
        remaining chain ``ab.cd``, not ``cd``."""
        reader, writer = await self._connect()
        await self._send(writer, {
            "type": "complete",
            "query": "ab.",
            "activeFile": "/tmp/bughunt3_chain.py",
            "activeFileContent": "y = ab.ab.cd\n",
        })
        ghost = await self._drain_until(reader, self._ghost_for("ab."))
        self.assertEqual(
            ghost.get("suggestion"), "ab.cd",
            "chain continuation beginning with the query was echo-stripped",
        )
