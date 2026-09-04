# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: Python trick parsing must unescape CommonMark backslashes like the TS twin.

``SorcarTab.ts``'s ``readMarkdownSections`` (the VS Code webview's
source of ``window.__TRICKS__``) strips CommonMark backslash escapes
from every ``## Trick`` body — ``mdformat`` rewrites ``<<x>>`` as
``\\<<x>>`` and ``snake_case`` as ``snake\\_case`` — so the webview
shows the trick the author actually wrote.  The Python parser in
``kiss.server.tricks`` documents itself as that function's mirror and
feeds two OTHER surfaces of the same trick list: the remote webapp's
``window.__TRICKS__`` (``web_server._build_html``) and the daemon's
ghost-text / picker ``trick`` completions
(``autocomplete._AutocompleteMixin._complete_many``).  Without the same
unescaping the three surfaces disagree: the VS Code panel shows
``<<x>>`` while the remote panel shows ``\\<<x>>`` and accepting the
ghost suggestion types literal backslashes into the prompt.

Exercised end to end through the real ``INJECTIONS.md`` /
``MY_INJECTION.md`` file reads (env overrides pin temp files), the real
autocomplete mixin, and a real ``RemoteAccessServer`` serving ``/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import tricks
from kiss.server.server import VSCodeServer
from kiss.server.web_server import RemoteAccessServer

# Exactly what mdformat writes for ``Replace <<x>> with snake_case.``
_ESCAPED_BODY = "Replace \\<<x>> with snake\\_case in a\\*b. Keep \\q as is."
# Exactly what SorcarTab.ts's ``unescapeMarkdown`` yields for it: every
# CommonMark ASCII-punctuation escape is removed, ``\q`` is preserved.
_UNESCAPED_BODY = "Replace <<x>> with snake_case in a*b. Keep \\q as is."

_FAKE_INJECTIONS = "## Trick\n\n" + _ESCAPED_BODY + "\n"


class _RecordingPrinter:
    """Minimal printer capturing broadcast events (no transport)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class _TricksEnv(TestCase):
    """Pin ``KISS_HOME`` and ``KISS_INJECTIONS_PATH`` to temp files."""

    def setUp(self) -> None:
        self._saved = {
            k: os.environ.get(k)
            for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")
        }
        kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_tricks_unescape_")) / ".kiss"
        kiss_dir.mkdir(parents=True)
        self.kiss_dir = kiss_dir
        fake_path = kiss_dir / "fake_INJECTIONS.md"
        fake_path.write_text(_FAKE_INJECTIONS, encoding="utf-8")
        # An empty user file so only the bundled override contributes.
        (kiss_dir / "MY_INJECTION.md").write_text("", encoding="utf-8")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(fake_path)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestReadTricksUnescapes(_TricksEnv):
    """``read_tricks`` must return the trick as the VS Code panel shows it."""

    def test_bundled_trick_is_unescaped(self) -> None:
        self.assertEqual(tricks.read_tricks(), [_UNESCAPED_BODY])

    def test_user_trick_is_unescaped(self) -> None:
        (self.kiss_dir / "MY_INJECTION.md").write_text(
            "## Trick\n\nUse `a\\_b` and \\[x\\](y).\n", encoding="utf-8",
        )
        self.assertEqual(
            tricks.read_tricks(),
            ["Use `a_b` and [x](y).", _UNESCAPED_BODY],
        )

    def test_prefix_match_returns_unescaped_trick(self) -> None:
        # The user typed the start of the trick as the panel displays it.
        self.assertEqual(
            tricks.prefix_match_tricks("Replace <<x>>"), [_UNESCAPED_BODY],
        )
        # ``\q`` is not a CommonMark escape and must survive unchanged.
        self.assertIn("\\q", tricks.prefix_match_tricks("Replace")[0])


class TestAutocompleteTrickCompletionUnescaped(_TricksEnv):
    """The daemon's ``trick`` completions carry the unescaped text."""

    def test_completions_and_ghost_are_unescaped(self) -> None:
        printer = _RecordingPrinter()
        server = VSCodeServer(printer=printer)  # type: ignore[arg-type]
        server._complete("Replace <<x")
        by_type = {e["type"]: e for e in printer.events}
        completions = by_type["completions"]["completions"]
        trick_items = [c for c in completions if c["type"] == "trick"]
        self.assertEqual(
            trick_items, [{"type": "trick", "text": _UNESCAPED_BODY}],
        )
        self.assertEqual(
            by_type["ghost"]["suggestion"],
            _UNESCAPED_BODY[len("Replace <<x"):],
        )


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestRemoteWebappTricksJsonUnescaped(IsolatedAsyncioTestCase):
    """``window.__TRICKS__`` served by the remote webapp is unescaped."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self._env = _TricksEnv()
        self._env.setUp()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._env.tearDown()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)

    async def test_served_tricks_json_is_unescaped(self) -> None:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=_no_verify_ssl(),
            server_hostname="localhost",
        )
        writer.write(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=15)
        writer.close()
        body = raw.split(b"\r\n\r\n", 1)[1].decode("utf-8")
        m = re.search(r"window\.__TRICKS__\s*=\s*(\[.*?\]);", body, re.DOTALL)
        assert m is not None, "TRICKS_JSON placeholder was not substituted"
        served = json.loads(m.group(1).replace("<\\/", "</"))
        self.assertEqual(served, [_UNESCAPED_BODY])
