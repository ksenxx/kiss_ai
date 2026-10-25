# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon resolves a client's file links over the local endpoint.

``_resolve_tab_file`` is the one resolution every client uses:

* ``checkPaths`` is answered with ``pathsExist`` on the requesting
  connection, echoing the client's ``workDir`` key and ``tabId``;
* ``openFile`` is answered with ``fileContent`` (the resolved path, the
  text and a version stamp, plus the request's ``line``); a path that
  resolves to nothing carries ``error``;
* a path-only ``submit`` (a single line naming an existing regular
  file) opens that file -- ``promptOpened`` then ``fileContent`` --
  instead of starting a task; a directory or an ordinary prompt still
  starts one.

All of them honour the tab's pending worktree, so a report a worktree
task committed on its un-merged branch is clickable and opens from the
worktree copy, and a work-dir DIRECTORY never shadows a worktree FILE
at the same relative path.  The tests drive a real
``RemoteAccessServer`` over a real connection to its
token-authenticated local WSS endpoint.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_server_liveness_and_run_refusals import (
    _UNAVAILABLE_MODEL,
    _ServerHarness,
)


class TestFileLinkResolution(_ServerHarness):
    """``checkPaths`` / ``openFile`` / path-only ``submit`` are served."""

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
        # newline="\n" everywhere a fixture is written: ``openFile`` replies
        # with the file's bytes as they are (an editor must not rewrite
        # line endings), and Windows would otherwise store "\r\n".
        (wt_dir / "reports" / "analysis.html").write_text(
            "<h1>report</h1>\n", encoding="utf-8", newline="\n",
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

    async def test_check_paths_is_answered(self) -> None:
        """``pathsExist`` reports real files and directories, per path."""
        (self.work_dir / "src").mkdir()
        (self.work_dir / "src" / "main.py").write_text("x\n", encoding="utf-8", newline="\n")
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
        applies only a reply whose ``workDir`` matches.  A connection
        pinned to a folder (``setWorkDir``) resolves the paths against
        the pin while the echo stays the client's key -- the pin echoed
        instead would never promote the links.
        """
        (self.work_dir / "pinned.py").write_text("x\n", encoding="utf-8", newline="\n")
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

    async def test_open_file_replies_with_the_file(self) -> None:
        """``openFile`` gets the resolved path, the text and the line."""
        target = self.work_dir / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("print(1)\nprint(2)\n", encoding="utf-8", newline="\n")
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
            "fileContent",
        )

        self.assertEqual(reply["path"], str(target.resolve()))
        self.assertEqual(reply["name"], "main.py")
        self.assertEqual(reply["line"], 2)
        self.assertEqual(reply["tabId"], "tab-open")
        self.assertEqual(reply["content"], "print(1)\nprint(2)\n")
        self.assertRegex(reply["version"], r"^\d+:\d+$")
        self.assertNotIn("error", reply)

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
            "fileContent",
        )

        self.assertEqual(
            reply["path"], str((wt_dir / "reports" / "analysis.html").resolve()),
        )
        self.assertEqual(reply["content"], "<h1>report</h1>\n")
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
            "fileContent",
        )

        self.assertEqual(reply["path"], "no/such/file.txt")
        self.assertEqual(reply["error"], "File not found: no/such/file.txt")
        self.assertNotIn("content", reply)
        self.assertNotIn("line", reply, "a non-positive line is not echoed")
        self.assertNotIn("token", reply, "no token sent, none echoed")

    async def test_open_file_echoes_the_token_on_every_reply(self) -> None:
        """A tokened ``openFile`` gets its token back: picture, text, error.

        ``main.js`` fetches the pictures of a result or preview
        (``<img src="tmp/plot.png">``) with a tokened ``openFile`` and
        routes the reply by the echoed token to the ``<img>`` instead of
        a content tab, whatever the reply turned out to be.
        """
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
            "60e6kgAAAABJRU5ErkJggg=="
        )
        (self.work_dir / "tmp").mkdir()
        (self.work_dir / "tmp" / "plot.png").write_bytes(png)
        # Bytes: the daemon serves the file verbatim, text mode would write CRLF on Windows.
        (self.work_dir / "tmp" / "d.svg").write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>\n")
        reader, writer = await self._connect()

        async def tokened(path: str, token: str) -> dict[str, Any]:
            return await self._reply(
                reader,
                writer,
                {
                    "type": "openFile",
                    "path": path,
                    "workDir": str(self.work_dir),
                    "tabId": "tab-img",
                    "token": token,
                },
                "fileContent",
            )

        picture = await tokened("tmp/plot.png", "img-1")
        self.assertEqual(picture["token"], "img-1")
        self.assertTrue(picture["binary"])
        self.assertEqual(picture["mime"], "image/png")
        self.assertEqual(base64.b64decode(picture["base64"]), png)

        svg = await tokened("tmp/d.svg", "img-2")
        self.assertEqual(svg["token"], "img-2")
        self.assertEqual(svg["content"], "<svg xmlns='http://www.w3.org/2000/svg'/>\n")
        self.assertNotIn("binary", svg)

        missing = await tokened("tmp/none.png", "img-3")
        self.assertEqual(missing["token"], "img-3")
        self.assertEqual(missing["error"], "File not found: tmp/none.png")

        listing = await tokened("tmp", "img-4")
        self.assertEqual(listing["token"], "img-4")
        self.assertIn("plot.png", listing["content"])

    async def test_open_file_of_a_directory_lists_it(self) -> None:
        """A directory link resolves to its listing."""
        (self.work_dir / "docs").mkdir()
        (self.work_dir / "docs" / "a.md").write_text("a\n", encoding="utf-8", newline="\n")
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
            "fileContent",
        )

        self.assertEqual(reply["path"], str((self.work_dir / "docs").resolve()))
        self.assertTrue(reply.get("isDirectory"))
        self.assertIn("a.md", reply["content"])
        self.assertNotIn("error", reply)

    # -- path-only submit ------------------------------------------------

    async def _submit(
        self, writer: LocalWriter, prompt: str, tab_id: str,
    ) -> None:
        await self._send(writer, {
            "type": "submit",
            "prompt": prompt,
            "model": _UNAVAILABLE_MODEL,
            "attachments": [],
            "workDir": str(self.work_dir),
            "tabId": tab_id,
        })

    async def _frames_until_idle(
        self, reader: LocalReader, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Collect frames until the tab reports ``running: false``."""
        frames: list[dict[str, Any]] = []

        def done(m: dict[str, Any]) -> bool:
            frames.append(m)
            return (
                m.get("type") == "status"
                and m.get("running") is False
                and m.get("tabId") == tab_id
            )

        await self._drain_until(reader, done, timeout=30.0)
        return frames

    async def test_path_only_submit_opens_the_worktree_copy(self) -> None:
        """A path-only submit opens the pending worktree copy -- no run."""
        wt_tab = "tab-wt-submit"
        wt_dir = self._pending_worktree(wt_tab)
        reader, writer = await self._connect()

        await self._submit(writer, "reports/analysis.html", wt_tab)
        opened = await self._drain_until(
            reader, lambda m: m.get("type") == "promptOpened", timeout=10.0,
        )
        self.assertEqual(opened["tabId"], wt_tab)
        content = await self._drain_until(
            reader, lambda m: m.get("type") == "fileContent", timeout=10.0,
        )
        self.assertEqual(
            content["path"],
            str((wt_dir / "reports" / "analysis.html").resolve()),
        )
        self.assertEqual(content["tabId"], wt_tab)
        # No task started: nothing reports the tab as running.
        await self._await_dispatch(reader, writer)
        frames = await self._collect_frames(
            reader, lambda m: False, timeout=0.5,
        )
        self.assertFalse(
            [
                f for f in frames
                if f.get("type") == "status" and f.get("running") is True
                and f.get("tabId") == wt_tab
            ],
            f"a path-only submit must not start a task: {frames}",
        )

    async def test_work_dir_directory_does_not_shadow_worktree_file(
        self,
    ) -> None:
        """A work-dir DIRECTORY at the path does not hide the worktree FILE."""
        wt_tab = "tab-wt-shadow"
        wt_dir = self._pending_worktree(wt_tab)
        (self.work_dir / "shadow.html").mkdir()
        (wt_dir / "shadow.html").write_text("<h1>wt copy</h1>\n", encoding="utf-8", newline="\n")
        reader, writer = await self._connect()

        await self._submit(writer, "shadow.html", wt_tab)
        await self._drain_until(
            reader, lambda m: m.get("type") == "promptOpened", timeout=10.0,
        )
        content = await self._drain_until(
            reader, lambda m: m.get("type") == "fileContent", timeout=10.0,
        )
        self.assertEqual(
            content["path"], str((wt_dir / "shadow.html").resolve()),
        )
        self.assertEqual(content["content"], "<h1>wt copy</h1>\n")

    async def test_non_path_prompt_still_starts_a_task(self) -> None:
        """An ordinary prompt runs (and ends at the no-model guard)."""
        tab_id = "tab-run"
        reader, writer = await self._connect()

        await self._submit(writer, "summarize the repo", tab_id)
        frames = await self._frames_until_idle(reader, tab_id)

        self.assertTrue(
            any(
                f.get("type") == "status" and f.get("running") is True
                and f.get("tabId") == tab_id
                for f in frames
            ),
            f"a non-path prompt must start a task: {frames}",
        )
        self.assertFalse([f for f in frames if f.get("type") == "promptOpened"])
        self.assertFalse([f for f in frames if f.get("type") == "fileContent"])

    async def test_directory_prompt_still_starts_a_task(self) -> None:
        """The shortcut is for regular files: a directory name is a task."""
        tab_id = "tab-dir-run"
        (self.work_dir / "reports").mkdir()
        reader, writer = await self._connect()

        await self._submit(writer, "reports", tab_id)
        frames = await self._frames_until_idle(reader, tab_id)

        self.assertTrue(
            any(
                f.get("type") == "status" and f.get("running") is True
                and f.get("tabId") == tab_id
                for f in frames
            ),
            f"a directory-path prompt must start a task: {frames}",
        )
        self.assertFalse([f for f in frames if f.get("type") == "promptOpened"])
