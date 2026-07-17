# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: TRICKS_JSON must be ``</``-escaped in the remote HTML page.

A trick body (user-editable via INJECTIONS.md) containing the literal
string ``</script>`` must not appear raw inside the inline ``<script>``
block that assigns ``window.__TRICKS__`` — a raw ``</script>`` would
terminate the script element early, leaving ``window.__TRICKS__``
undefined and rendering the rest of the JSON as page text.  The
adjacent ``TIPS_JSON`` substitution already applies the
``.replace("</", "<\\/")`` escaping; this test pins the same protection
for ``TRICKS_JSON``.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer

_TRICK_WITH_SCRIPT = 'Add </script><b>pwned</b> to the page. Then fix it.'

_FAKE_INJECTIONS = "## Trick\n\n" + _TRICK_WITH_SCRIPT + "\n"


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


class TestTricksJsonEscaped(IsolatedAsyncioTestCase):
    """Serve ``/`` with a ``</script>`` trick and inspect the inline JSON."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        # Pin the tricks sources to a private temp dir BEFORE the server
        # builds its HTML page (RemoteAccessServer caches the page at init).
        kiss_dir = Path(tempfile.mkdtemp(prefix="kiss_tricks_test_")) / ".kiss"
        kiss_dir.mkdir(parents=True)
        fake_path = kiss_dir / "fake_INJECTIONS.md"
        fake_path.write_text(_FAKE_INJECTIONS)
        (kiss_dir / "MY_INJECTION.md").write_text("")
        self._saved_kiss_home = os.environ.get("KISS_HOME")
        self._saved_injections = os.environ.get("KISS_INJECTIONS_PATH")
        os.environ["KISS_HOME"] = str(kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(fake_path)

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._saved_kiss_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_kiss_home
        if self._saved_injections is None:
            os.environ.pop("KISS_INJECTIONS_PATH", None)
        else:
            os.environ["KISS_INJECTIONS_PATH"] = self._saved_injections
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _http_get(self, path: str) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        url = f"https://127.0.0.1:{self.port}{path}"
        ctx = _no_verify_ssl()

        def _fetch() -> tuple[int, str]:
            try:
                resp = urllib.request.urlopen(url, timeout=5, context=ctx)
                return resp.status, resp.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode() if e.fp else ""

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    async def test_tricks_json_has_no_raw_script_close(self) -> None:
        """The __TRICKS__ JSON blob must contain no raw ``</`` sequence."""
        status, body = await self._http_get("/")
        self.assertEqual(status, 200)
        marker = "window.__TRICKS__ = "
        self.assertIn(marker, body)
        line = next(ln for ln in body.splitlines() if marker in ln)
        blob = line.split(marker, 1)[1]
        end = blob.rindex(";</script>")
        payload = blob[:end]
        # The raw sequence ``</`` inside the inline script terminates it
        # in a browser — it must be escaped as ``<\/``.
        self.assertNotIn("</", payload)
        # Round-trip: unescaping yields valid JSON containing the trick.
        tricks = json.loads(payload.replace("<\\/", "</"))
        self.assertIn(_TRICK_WITH_SCRIPT, tricks)
