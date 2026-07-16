# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regression tests pinning the merge-review engine before extraction.

Second simplification pass (simplify2): the merge-review cluster of
``web_server.py`` (``_WebMergeState``, ``_apply_exec_bit``,
``_restore_base_bytes``, ``_reject_hunk_in_file``,
``_record_hunk_rejected``, ``_hunk_unresolved``,
``_reject_all_hunks_in_file``) moves to a new ``web_merge`` module with
the names re-exported from ``web_server``.  These tests pin the exact
on-disk behavior of every function in the cluster plus the HTTP wire
protocol of ``RemoteAccessServer`` (GET / and HEAD / over a real TLS
connection), so the extraction is provably behavior-preserving.

All tests drive real objects and real files on disk — no mocks,
patches, or fakes.
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _apply_exec_bit,
    _hunk_unresolved,
    _record_hunk_rejected,
    _reject_all_hunks_in_file,
    _reject_hunk_in_file,
    _restore_base_bytes,
    _WebMergeState,
)


class TestRejectHunkInFile(unittest.TestCase):
    """Pin _reject_hunk_in_file splice semantics and CRLF preservation."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp2-rh-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_splice_restores_base_lines_only_for_hunk(self) -> None:
        base = self.tmpdir / "f.base"
        cur = self.tmpdir / "f.txt"
        base.write_text("one\ntwo\nthree\nfour\n")
        cur.write_text("ONE\ntwo\nTHREE\nfour\n")
        _reject_hunk_in_file(
            str(cur), str(base), {"bs": 2, "bc": 1, "cs": 2, "cc": 1},
        )
        self.assertEqual(cur.read_text(), "ONE\ntwo\nthree\nfour\n")

    def test_crlf_lines_outside_hunk_are_preserved(self) -> None:
        base = self.tmpdir / "c.base"
        cur = self.tmpdir / "c.txt"
        base.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        cur.write_bytes(b"one\r\nTWO!\r\nthree\r\n")
        _reject_hunk_in_file(
            str(cur), str(base), {"bs": 1, "bc": 1, "cs": 1, "cc": 1},
        )
        # Every CRLF ending must survive byte-for-byte.
        self.assertEqual(cur.read_bytes(), b"one\r\ntwo\r\nthree\r\n")

    def test_deleted_file_placeholder_writes_to_target(self) -> None:
        base = self.tmpdir / "d.base"
        placeholder = self.tmpdir / "d.deleted"
        target = self.tmpdir / "real" / "d.txt"
        base.write_text("alpha\nbeta\n")
        placeholder.write_text("")
        _reject_hunk_in_file(
            str(placeholder), str(base),
            {"bs": 0, "bc": 2, "cs": 0, "cc": 0},
            str(target),
        )
        self.assertEqual(target.read_text(), "alpha\nbeta\n")
        # Placeholder itself is untouched.
        self.assertEqual(placeholder.read_text(), "")

    def test_binary_flag_restores_base_bytes_wholesale(self) -> None:
        base = self.tmpdir / "b.base"
        cur = self.tmpdir / "b.bin"
        base.write_bytes(b"\x00\x01\x02")
        cur.write_bytes(b"\xff\xfe")
        _reject_hunk_in_file(
            str(cur), str(base), {"bs": 0, "bc": 1, "cs": 0, "cc": 1},
            binary=True,
        )
        self.assertEqual(cur.read_bytes(), b"\x00\x01\x02")

    def test_undecodable_content_falls_back_to_base_bytes(self) -> None:
        base = self.tmpdir / "u.base"
        cur = self.tmpdir / "u.txt"
        base.write_bytes(b"plain\n")
        # UTF-16 content without NUL in first bytes decodes as neither
        # utf-8 nor does the splice apply — wholesale restore expected.
        cur.write_bytes("héllo\n".encode("utf-16"))
        _reject_hunk_in_file(
            str(cur), str(base), {"bs": 0, "bc": 1, "cs": 0, "cc": 1},
        )
        self.assertEqual(cur.read_bytes(), b"plain\n")

    def test_exec_bit_reapplied_after_splice(self) -> None:
        base = self.tmpdir / "s.base"
        cur = self.tmpdir / "s.sh"
        base.write_text("#!/bin/sh\necho ok\n")
        cur.write_text("#!/bin/sh\necho HACKED\n")
        _reject_hunk_in_file(
            str(cur), str(base), {"bs": 1, "bc": 1, "cs": 1, "cc": 1},
            make_executable=True,
        )
        self.assertEqual(cur.read_text(), "#!/bin/sh\necho ok\n")
        self.assertTrue(os.access(cur, os.X_OK))

    def test_symlink_current_is_replaced_not_written_through(self) -> None:
        precious = self.tmpdir / "precious.txt"
        precious.write_text("do not clobber\n")
        base = self.tmpdir / "l.base"
        base.write_text("base\n")
        link = self.tmpdir / "l.txt"
        os.symlink(precious, link)
        _reject_hunk_in_file(
            str(link), str(base), {"bs": 0, "bc": 1, "cs": 0, "cc": 1},
        )
        self.assertFalse(link.is_symlink())
        self.assertEqual(link.read_text(), "base\n")
        self.assertEqual(precious.read_text(), "do not clobber\n")


class TestRestoreBaseBytes(unittest.TestCase):
    """Pin _restore_base_bytes symlink-safety and edge cases."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp2-rb-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_symlink_dest_replaced_target_untouched(self) -> None:
        precious = self.tmpdir / "precious.bin"
        precious.write_bytes(b"KEEP")
        base = self.tmpdir / "x.base"
        base.write_bytes(b"BASE")
        dest = self.tmpdir / "x.bin"
        os.symlink(precious, dest)
        _restore_base_bytes(str(base), str(dest))
        self.assertFalse(dest.is_symlink())
        self.assertEqual(dest.read_bytes(), b"BASE")
        self.assertEqual(precious.read_bytes(), b"KEEP")

    def test_link_target_recreates_symlink(self) -> None:
        dest = self.tmpdir / "lnk"
        dest.write_text("regular file now")
        _restore_base_bytes(str(self.tmpdir / "nope.base"), str(dest),
                            "some/target")
        self.assertTrue(dest.is_symlink())
        self.assertEqual(os.readlink(dest), "some/target")

    def test_missing_base_restores_empty_file(self) -> None:
        dest = self.tmpdir / "sub" / "empty.txt"
        _restore_base_bytes(str(self.tmpdir / "missing.base"), str(dest))
        self.assertEqual(dest.read_bytes(), b"")

    def test_make_executable_applies_exec_bit(self) -> None:
        base = self.tmpdir / "e.base"
        base.write_text("#!/bin/sh\n")
        dest = self.tmpdir / "e.sh"
        _restore_base_bytes(str(base), str(dest), make_executable=True)
        self.assertTrue(os.access(dest, os.X_OK))

    def test_apply_exec_bit_mirrors_read_perms(self) -> None:
        f = self.tmpdir / "m.sh"
        f.write_text("x")
        os.chmod(f, 0o644)
        _apply_exec_bit(str(f))
        self.assertEqual(os.stat(f).st_mode & 0o777, 0o755)


class TestRecordHunkRejected(unittest.TestCase):
    """Pin _record_hunk_rejected offset bookkeeping."""

    def test_cc_updated_and_pending_later_hunks_shifted(self) -> None:
        hunks: list[dict[str, Any]] = [
            {"bs": 0, "bc": 3, "cs": 0, "cc": 1},   # delta +2
            {"bs": 5, "bc": 1, "cs": 4, "cc": 1},   # pending -> shifted
            {"bs": 8, "bc": 1, "cs": 7, "cc": 2},   # resolved -> NOT shifted
        ]
        pending = {1}
        _record_hunk_rejected(hunks, 0, pending.__contains__)
        self.assertEqual(hunks[0]["cc"], 3)  # idempotent retry bookkeeping
        self.assertEqual(hunks[1]["cs"], 6)  # 4 + 2
        self.assertEqual(hunks[2]["cs"], 7)  # untouched

    def test_hunk_unresolved_predicate_reads_state(self) -> None:
        state = _WebMergeState({"files": [
            {"name": "a", "hunks": [{}, {}]},
        ]})
        self.assertTrue(_hunk_unresolved(state, 0, 1))
        state.mark_resolved(0, 1, "accepted")
        self.assertFalse(_hunk_unresolved(state, 0, 1))
        still = functools.partial(_hunk_unresolved, state, 0)
        self.assertFalse(still(1))
        self.assertTrue(still(0))


class TestRejectAllHunksInFile(unittest.TestCase):
    """Pin _reject_all_hunks_in_file partial + binary + symlink paths."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-simp2-ra-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_partial_indices_keep_accepted_hunk_content(self) -> None:
        base = self.tmpdir / "p.base"
        cur = self.tmpdir / "p.txt"
        base.write_text("a\nb\nc\nd\ne\n")
        # Hunk 0 replaces line0 with 2 lines; hunk 1 edits line c; hunk 2
        # edits line e.  User accepted hunk 1 (keeps "C!").
        cur.write_text("A1\nA2\nb\nC!\nd\nE!\n")
        file_data: dict[str, Any] = {
            "name": "p.txt",
            "base": str(base),
            "current": str(cur),
            "hunks": [
                {"bs": 0, "bc": 1, "cs": 0, "cc": 2},
                {"bs": 2, "bc": 1, "cs": 3, "cc": 1},
                {"bs": 4, "bc": 1, "cs": 5, "cc": 1},
            ],
        }
        _reject_all_hunks_in_file(file_data, [0, 2])
        self.assertEqual(cur.read_text(), "a\nb\nC!\nd\ne\n")
        # Offsets were fixed up: hunk 2's cs shifted by hunk 0's delta.
        self.assertEqual(file_data["hunks"][0]["cc"], 1)
        self.assertEqual(file_data["hunks"][2]["cs"], 4)

    def test_all_hunks_when_indices_none(self) -> None:
        base = self.tmpdir / "n.base"
        cur = self.tmpdir / "n.txt"
        base.write_text("x\ny\n")
        cur.write_text("X\nY\n")
        file_data = {
            "name": "n.txt", "base": str(base), "current": str(cur),
            "hunks": [
                {"bs": 0, "bc": 1, "cs": 0, "cc": 1},
                {"bs": 1, "bc": 1, "cs": 1, "cc": 1},
            ],
        }
        _reject_all_hunks_in_file(file_data)
        self.assertEqual(cur.read_text(), "x\ny\n")

    def test_binary_entry_restores_wholesale_with_exec(self) -> None:
        base = self.tmpdir / "bin.base"
        cur = self.tmpdir / "bin"
        base.write_bytes(b"\x7fELF-base")
        cur.write_bytes(b"\x7fELF-hacked")
        file_data = {
            "name": "bin", "base": str(base), "current": str(cur),
            "binary": True, "exec": True,
            "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
        }
        _reject_all_hunks_in_file(file_data, [0])
        self.assertEqual(cur.read_bytes(), b"\x7fELF-base")
        self.assertTrue(os.access(cur, os.X_OK))

    def test_binary_entry_with_empty_indices_is_noop(self) -> None:
        base = self.tmpdir / "bn.base"
        cur = self.tmpdir / "bn"
        base.write_bytes(b"BASE")
        cur.write_bytes(b"CURRENT")
        file_data = {
            "name": "bn", "base": str(base), "current": str(cur),
            "binary": True,
            "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
        }
        _reject_all_hunks_in_file(file_data, [])
        self.assertEqual(cur.read_bytes(), b"CURRENT")

    def test_symlink_base_entry_recreates_link(self) -> None:
        target_file = self.tmpdir / "pointee.txt"
        target_file.write_text("pointee")
        cur = self.tmpdir / "lnk"
        cur.write_text("agent replaced the link with a file")
        file_data = {
            "name": "lnk", "base": str(self.tmpdir / "lnk.base"),
            "current": str(cur), "binary": True,
            "link_target": "pointee.txt",
            "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
        }
        _reject_all_hunks_in_file(file_data, [0])
        self.assertTrue(cur.is_symlink())
        self.assertEqual(os.readlink(cur), "pointee.txt")


def _free_port() -> int:
    """Grab an ephemeral localhost port for the live-server test."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestHttpWireProtocol(unittest.IsolatedAsyncioTestCase):
    """Real TLS GET / and HEAD / round-trips against RemoteAccessServer."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-simp2-http-")
        self.port = _free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _raw_request(self, payload: bytes) -> bytes:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", self.port, ssl=ctx,
        )
        try:
            writer.write(payload)
            await writer.drain()
            return await asyncio.wait_for(reader.read(), timeout=5.0)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_get_root_serves_html(self) -> None:
        raw = await self._raw_request(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        head, _, body = raw.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 200"), head[:60])
        self.assertIn(b"text/html", head.lower())
        self.assertIn(b"<html", body[:4096].lower())

    async def test_head_root_returns_200(self) -> None:
        raw = await self._raw_request(b"HEAD / HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertTrue(raw.startswith(b"HTTP/1.1 200"), raw[:60])


if __name__ == "__main__":
    unittest.main()
