# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon resolves a VS Code window's file links over the Unix socket.

The extension host used to resolve its webview's ``checkPaths`` and
``openFile`` itself (workspace root, then the tab's pending worktree)
while the daemon dropped the same commands from UDS clients as no-ops.
Now the host forwards both, and the daemon's ``_resolve_tab_file`` is
the one resolution every surface uses:

* ``checkPaths`` is answered with ``pathsExist`` on the requesting UDS
  connection, exactly as for a browser;
* ``openFile`` is answered with ``openResolvedFile`` — the resolved
  path (plus the request's ``line``) the host opens in a real editor
  tab — instead of the browser's ``fileContent``; a path that resolves
  to nothing carries ``error``.

Both honour the tab's pending worktree, so a report a worktree task
committed on its un-merged branch is clickable and opens from the
worktree copy.  The tests drive a real ``RemoteAccessServer`` over a
real UDS connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_server_liveness_and_run_refusals import (
    _ServerHarness,
)


class TestUdsFileLinkResolution(_ServerHarness):
    """``checkPaths`` / ``openFile`` from a local client are served."""

    async def _reply(
        self,
        reader: LocalReader,
        writer: LocalWriter,
        cmd: dict[str, Any],
        reply_type: str,
    ) -> dict[str, Any]:
        """Send *cmd* and return the first frame of *reply_type*."""
        await self._send(writer, cmd)
        return await self._drain_until(
            reader, lambda m: m.get("type") == reply_type, timeout=10.0,
        )

    def _pending_worktree(self, tab_id: str) -> Path:
        """Give *tab_id* a pending worktree holding ``reports/analysis.html``."""
        wt_dir = self.work_dir.parent / "wt"
        (wt_dir / "reports").mkdir(parents=True)
        (wt_dir / "reports" / "analysis.html").write_text(
            "<h1>report</h1>\n", encoding="utf-8",
        )
        self.server._printer.broadcast(
            {
                "type": "worktree_done",
                "branch": "kiss/wt-1",
                "worktreeDir": str(wt_dir),
                "tabId": tab_id,
            }
        )
        return wt_dir

    async def test_check_paths_is_answered_over_uds(self) -> None:
        """``pathsExist`` reports real files and directories, per path."""
        (self.work_dir / "src").mkdir()
        (self.work_dir / "src" / "main.py").write_text("x\n", encoding="utf-8")
        reader, writer = await self._connect()

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "checkPaths",
                "paths": ["src/main.py", "src", "missing.py", "", 7],
                "workDir": str(self.work_dir),
                "tabId": "tab-links",
            },
            "pathsExist",
        )

        self.assertEqual(
            reply["results"],
            {"src/main.py": True, "src": True, "missing.py": False},
        )
        self.assertEqual(reply["workDir"], str(self.work_dir))
        self.assertEqual(reply["tabId"], "tab-links")

    async def test_check_paths_echoes_the_clients_work_dir_key(self) -> None:
        """A tab that sent ``workDir: ""`` gets ``""`` back, resolved by the pin.

        ``main.js`` stamps every candidate with the workDir it sent and
        applies only a reply whose ``workDir`` matches; the extension
        host pins the connection to the window's folder (``setWorkDir``)
        and forwards the webview's ``checkPaths`` as sent.  The paths
        must resolve against the pin while the echo stays the client's
        key — the pin echoed instead would never promote the links.
        """
        (self.work_dir / "pinned.py").write_text("x\n", encoding="utf-8")
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "setWorkDir", "workDir": str(self.work_dir)},
        )

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "checkPaths",
                "paths": ["pinned.py"],
                "workDir": "",
                "tabId": "tab-unpinned",
            },
            "pathsExist",
        )

        self.assertEqual(reply["results"], {"pinned.py": True})
        self.assertEqual(reply["workDir"], "")

    async def test_check_paths_sees_the_tabs_pending_worktree(self) -> None:
        """A worktree-only artifact is clickable for its tab alone."""
        wt_tab = "tab-wt"
        self._pending_worktree(wt_tab)
        reader, writer = await self._connect()
        query: dict[str, Any] = {
            "type": "checkPaths",
            "paths": ["reports/analysis.html"],
            "workDir": str(self.work_dir),
        }

        own = await self._reply(
            reader, writer, {**query, "tabId": wt_tab}, "pathsExist",
        )
        other = await self._reply(
            reader, writer, {**query, "tabId": "tab-other"}, "pathsExist",
        )

        self.assertEqual(own["results"], {"reports/analysis.html": True})
        self.assertEqual(other["results"], {"reports/analysis.html": False})

    async def test_open_file_replies_with_the_resolved_path(self) -> None:
        """A UDS ``openFile`` gets the path to open natively, not content."""
        target = self.work_dir / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("print(1)\nprint(2)\n", encoding="utf-8")
        reader, writer = await self._connect()

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "openFile",
                "path": "src/main.py",
                "line": 2,
                "workDir": str(self.work_dir),
                "tabId": "tab-open",
            },
            "openResolvedFile",
        )

        self.assertEqual(reply["path"], str(target.resolve()))
        self.assertEqual(reply["line"], 2)
        self.assertEqual(reply["tabId"], "tab-open")
        self.assertNotIn("error", reply)
        self.assertNotIn("content", reply)

    async def test_open_file_falls_back_to_the_pending_worktree(self) -> None:
        """The worktree copy opens when the workspace has no such file."""
        wt_tab = "tab-wt-open"
        wt_dir = self._pending_worktree(wt_tab)
        reader, writer = await self._connect()

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "openFile",
                "path": "reports/analysis.html",
                "workDir": str(self.work_dir),
                "tabId": wt_tab,
            },
            "openResolvedFile",
        )

        self.assertEqual(
            reply["path"], str((wt_dir / "reports" / "analysis.html").resolve()),
        )
        self.assertNotIn("line", reply, "no line was requested")

    async def test_open_file_of_a_missing_path_reports_an_error(self) -> None:
        """Nothing to open: the raw path comes back with ``error``."""
        reader, writer = await self._connect()

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "openFile",
                "path": "no/such/file.txt",
                "line": 0,
                "workDir": str(self.work_dir),
                "tabId": "tab-missing",
            },
            "openResolvedFile",
        )

        self.assertEqual(reply["path"], "no/such/file.txt")
        self.assertEqual(reply["error"], "File not found: no/such/file.txt")
        self.assertNotIn("line", reply, "a non-positive line is not echoed")

    async def test_open_file_of_a_directory_resolves(self) -> None:
        """A directory link resolves too (the host reveals it in Explorer)."""
        (self.work_dir / "docs").mkdir()
        reader, writer = await self._connect()

        reply = await self._reply(
            reader,
            writer,
            {
                "type": "openFile",
                "path": "docs",
                "workDir": str(self.work_dir),
                "tabId": "tab-dir",
            },
            "openResolvedFile",
        )

        self.assertEqual(reply["path"], str((self.work_dir / "docs").resolve()))
        self.assertNotIn("error", reply)
