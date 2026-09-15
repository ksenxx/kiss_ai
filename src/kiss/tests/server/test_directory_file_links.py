# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: directory paths in transcripts are clickable links.

The chat webview linkifies path-looking strings and asks the host which
ones exist (``checkPaths``); clicking a confirmed link sends
``openFile``.  Directories used to be excluded — every path naming a
directory stayed inert plain text.  These tests pin the new behaviour
end-to-end over a real ``wss://`` connection: ``checkPaths`` reports
directories as existing (covered in ``test_web_server.py``) and
``openFile`` on a directory replies with a plain-text listing
(:func:`kiss.server.web_server._directory_listing_text`) instead of an
error, with directories first (trailing ``/``), sorted names, an
``(empty directory)`` marker, and truncation beyond
``_DIR_LISTING_MAX_ENTRIES`` entries.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, skipIf

from websockets.asyncio.client import connect

import kiss.core.vscode_config as vc
from kiss.server.web_server import (
    _DIR_LISTING_MAX_CHARS,
    _DIR_LISTING_MAX_ENTRIES,
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestDirectoryFileLinks(IsolatedAsyncioTestCase):
    """openFile on a directory serves a listing the client can render."""

    async def asyncSetUp(self) -> None:
        # resolve(): the server replies with resolved paths, and on macOS
        # mkdtemp returns /var/... which is a symlink to /private/var/...
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-dirlink-")).resolve()
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.work_dir = self.tmpdir / "repo"
        self.work_dir.mkdir()
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
        # Restore permissions dropped by the unreadable-dir tests so the
        # temp tree can be deleted.
        for root, dirs, _files in os.walk(self.tmpdir):
            for d in dirs:
                os.chmod(Path(root) / d, 0o755)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _open(self, path: str) -> dict[str, Any]:
        """Authenticate, send ``openFile`` for *path*, return the reply."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg.get("type") == "auth_ok":
                    break
            await ws.send(
                json.dumps(
                    {
                        "type": "openFile",
                        "path": path,
                        "workDir": str(self.work_dir),
                        "tabId": "t1",
                    }
                )
            )
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg.get("type") == "fileContent":
                    return dict(msg)

    async def test_directory_click_returns_sorted_listing(self) -> None:
        """A clicked directory link serves dirs-first sorted listing text."""
        pkg = self.work_dir / "pkg"
        (pkg / "zsub").mkdir(parents=True)
        (pkg / "asub").mkdir()
        (pkg / "b.txt").write_text("b\n")
        (pkg / ".hidden").write_text("h\n")
        reply = await self._open("pkg")
        self.assertNotIn("error", reply)
        self.assertEqual(reply["path"], str(pkg))
        self.assertEqual(reply["name"], "pkg")
        self.assertIs(reply["isDirectory"], True)
        self.assertEqual(
            reply["content"],
            f"{pkg}:\n"
            "\n"
            f"{pkg / 'asub'}/\n"
            f"{pkg / 'zsub'}/\n"
            f"{pkg / '.hidden'}\n"
            f"{pkg / 'b.txt'}\n",
        )

    async def test_absolute_directory_path_resolves(self) -> None:
        """Absolute directory paths (as printed in results) resolve too."""
        sub = self.work_dir / "abs"
        sub.mkdir()
        reply = await self._open(str(sub))
        self.assertNotIn("error", reply)
        self.assertEqual(reply["name"], "abs")
        self.assertTrue(reply["content"].startswith(f"{sub}:\n"))

    async def test_file_reply_has_no_directory_flag(self) -> None:
        """Regular-file replies carry no isDirectory marker."""
        (self.work_dir / "plain.txt").write_text("x\n")
        reply = await self._open("plain.txt")
        self.assertNotIn("error", reply)
        self.assertEqual(reply["content"], "x\n")
        self.assertNotIn("isDirectory", reply)

    async def test_markdown_named_directory_is_flagged(self) -> None:
        """A directory named like a markdown/HTML file still lists as text.

        The client routes fileContent replies by name extension, so a
        directory named ``notes.md`` (or ``site.html``) must carry the
        ``isDirectory`` flag telling the client to render the listing
        as plain text instead of markdown/HTML.
        """
        for dirname in ("notes.md", "site.html"):
            mddir = self.work_dir / dirname
            mddir.mkdir()
            (mddir / "inner.txt").write_text("")
            reply = await self._open(dirname)
            self.assertNotIn("error", reply)
            self.assertIs(reply["isDirectory"], True)
            self.assertEqual(reply["name"], dirname)
            self.assertIn(f"{mddir / 'inner.txt'}", reply["content"])

    async def test_empty_directory_shows_marker(self) -> None:
        """An empty directory lists an explicit marker, not blank text."""
        (self.work_dir / "empty").mkdir()
        reply = await self._open("empty")
        self.assertNotIn("error", reply)
        self.assertIn("(empty directory)", reply["content"])

    async def test_huge_directory_listing_is_truncated(self) -> None:
        """Listings stop at _DIR_LISTING_MAX_ENTRIES with a trailing note."""
        big = self.work_dir / "big"
        big.mkdir()
        for i in range(_DIR_LISTING_MAX_ENTRIES + 3):
            (big / f"f{i:05d}.txt").write_text("")
        reply = await self._open("big")
        self.assertNotIn("error", reply)
        lines = reply["content"].splitlines()
        self.assertEqual(lines[-1], "... 3 more entries not shown")
        # header + blank + capped entries + note
        self.assertEqual(len(lines), 2 + _DIR_LISTING_MAX_ENTRIES + 1)

    async def test_long_entry_names_hit_character_cap(self) -> None:
        """The character cap truncates before the entry cap when lines
        are long, keeping one click's reply bounded in bytes."""
        deep = self.work_dir / "deep"
        deep.mkdir()
        total = _DIR_LISTING_MAX_ENTRIES
        for i in range(total):
            (deep / f"{'x' * 245}{i:05d}").write_text("")
        reply = await self._open("deep")
        self.assertNotIn("error", reply)
        lines = reply["content"].splitlines()
        shown = len(lines) - 3  # header + blank + entries + note
        self.assertLess(shown, total, "char cap must truncate the listing")
        self.assertEqual(
            lines[-1],
            f"... {total - shown} more entries not shown",
        )
        # The cap is strict for entry lines: only the header, the blank
        # separator, and the truncation note come on top of it.
        entry_chars = sum(len(ln) + 1 for ln in lines[2:-1])
        self.assertLessEqual(entry_chars, _DIR_LISTING_MAX_CHARS)

    async def test_filesystem_root_uses_full_path_as_name(self) -> None:
        """Path("/").name is empty; the reply falls back to the path."""
        reply = await self._open("/")
        self.assertNotIn("error", reply)
        self.assertEqual(reply["name"], "/")
        self.assertTrue(reply["content"].startswith("/:\n"))

    @skipIf(os.geteuid() == 0, "root ignores directory permissions")
    async def test_unreadable_directory_replies_error(self) -> None:
        """A directory that cannot be listed produces an error reply."""
        locked = self.work_dir / "locked"
        locked.mkdir()
        os.chmod(locked, 0o000)
        reply = await self._open("locked")
        self.assertIn("error", reply)
        self.assertIn("Failed to read", reply["error"])

    @skipIf(os.geteuid() == 0, "root ignores directory permissions")
    async def test_unstatable_entry_is_listed_as_file(self) -> None:
        """An entry whose type cannot be stat'ed still shows up (no /).

        A directory with read permission but no execute permission lets
        ``iterdir()`` enumerate names while ``entry.is_dir()`` raises
        ``PermissionError``: the entry must degrade to the files group
        instead of breaking the whole listing.
        """
        noexec = self.work_dir / "noexec"
        child = noexec / "childdir"
        child.mkdir(parents=True)
        os.chmod(noexec, 0o644)
        reply = await self._open("noexec")
        self.assertNotIn("error", reply)
        self.assertIn(f"\n{child}\n", reply["content"])
        self.assertNotIn(f"{child}/", reply["content"])
