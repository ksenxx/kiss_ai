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
at the same relative path.  An ``openFile`` with ``report`` asks for
the report a finished task wrote (``_resolve_report_file``): the
checkout's file once the daemon has announced the merge of the worktree
the task wrote it in, no file before, the file itself for a run that
made no worktree, and no file for a run the daemon kept no record of.
The tests drive a real ``RemoteAccessServer`` over a real connection to
its token-authenticated local WSS endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.server import agent_state
from kiss.server.web_server import _TAB_RUNS_MAX, RemoteAccessServer, _file_version
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.conftest import assert_repo_root, run_git_checked
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials
from kiss.tests.server.test_server_liveness_and_run_refusals import (
    _UNAVAILABLE_MODEL,
    _ServerHarness,
)

#: A ``pre-commit`` hook that lets a task worktree's own commit pass and
#: holds the checkout's merge commit until the release file appears (20 s
#: at most), then rejects it.  ``{started}`` appears once it is waiting.
_BLOCK_THEN_REJECT_MERGE = """#!/bin/sh
case "$PWD" in */.kiss-worktrees/*) exit 0 ;; esac
touch "{started}"
i=0
while [ ! -f "{release}" ] && [ $i -lt 400 ]; do
  sleep 0.05
  i=$((i+1))
done
exit 1
"""


def _finish_at_once(_request: dict[str, Any]) -> dict[str, Any]:
    """The stand-in model's answer to every call: the task is done."""
    return finish_response("done")


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

    async def test_check_paths_looks_past_a_removed_worktree(self) -> None:
        """A folder naming a merged, removed worktree stands for its repository.

        Only for a path inside it: one that leaves the folder resolves as
        written, and a worktree that exists is not looked past.  An
        absolute path into the removed worktree (a file or a directory
        the transcript printed in full) stands for the repository's
        copy the same way.
        """
        (self.work_dir / "docs").mkdir()
        (self.work_dir / "docs" / "a.txt").write_text("merged\n", encoding="utf-8")
        worktree = self.work_dir / ".kiss-worktrees" / "kiss_wt-1-abc"
        worktree.parent.mkdir()
        inside = [str(worktree / "docs" / "a.txt"), str(worktree / "docs")]
        leaving = str(worktree / ".." / ".." / "docs" / "a.txt")
        reader, writer = await self._connect()
        query: dict[str, Any] = {
            "type": "checkPaths",
            "paths": ["docs/a.txt", "../../docs/a.txt", *inside, leaving],
            "workDir": str(worktree),
        }

        removed = await self._reply(reader, writer, query, "pathsExist")
        worktree.mkdir()
        present = await self._reply(reader, writer, query, "pathsExist")

        escapes = {"../../docs/a.txt": True, leaving: True}
        self.assertEqual(
            removed["results"], {"docs/a.txt": True, **dict.fromkeys(inside, True), **escapes},
        )
        self.assertEqual(
            present["results"], {"docs/a.txt": False, **dict.fromkeys(inside, False), **escapes},
        )

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
        self.assertEqual(reply["version"], _file_version(target.read_bytes()))
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

    def _worktree_event(self, tab_id: str, kind: str, **fields: Any) -> None:
        """Have the daemon announce the worktree event *kind* to *tab_id*."""
        self.server._printer.broadcast({"type": kind, "tabId": tab_id, **fields})

    def _task_worktree(self, tab_id: str) -> Path:
        """Start a run of *tab_id* whose task leaves a worktree of the checkout ``work_dir``."""
        wt_dir = self.work_dir / ".kiss-worktrees" / "kiss_wt-demo"
        wt_dir.mkdir(parents=True)
        self._worktree_event(tab_id, "clear")
        self._worktree_event(
            tab_id, "worktree_done", branch="kiss/wt-demo", worktreeDir=str(wt_dir),
        )
        return wt_dir

    async def _report(
        self,
        reader: LocalReader,
        writer: LocalWriter,
        path: str,
        work_dir: Path,
        tab_id: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Ask for the report a finished task of *tab_id* wrote at *path*, as the page does."""
        query = {"type": "openFile", "path": path, "workDir": str(work_dir), "tabId": tab_id}
        return await self._reply(
            reader, writer, {**query, "token": "report:1", "report": True, **extra}, "fileContent",
        )

    async def test_a_report_of_a_worktree_that_is_not_merged_has_no_file_to_show(self) -> None:
        """Until its worktree is merged a report is not shown, however the task named it.

        A task's ``Write`` of an absolute checkout path lands in its
        worktree, and the checkout keeps its older copy (or none).  The
        page asks with ``report``: no file, for the absolute path, the
        relative one and the worktree's own, although the checkout's
        copy is seconds old (the tab was told of the worktree, which is
        all the daemon knows here).  A path that cannot name a file (a
        NUL byte) is no report either.  A plain open of the same path
        resolves as written.  Discarding the worktree changes nothing:
        the checkout's copy is still not the report.  Nor do copies of
        another task's events that reach the tab after it stopped
        watching that task: its merge merges nothing here, and its
        worktree says nothing about the report the tab's next run
        writes straight into the checkout.
        """
        tab = "tab-report"
        wt_dir = self._task_worktree(tab)
        for folder in (self.work_dir, wt_dir):
            (folder / "reports").mkdir()
            (folder / "reports" / "existing.md").write_text(
                "OLD\n", encoding="utf-8", newline="\n",
            )
        existing = self.work_dir / "reports" / "existing.md"
        fresh = self.work_dir / "reports" / "fresh.md"
        since = int(time.time() * 1000)
        tools = UsefulTools(work_dir=str(wt_dir))
        tools.Read(str(existing))
        for target, text in ((existing, "NEW\n"), (fresh, "FRESH\n")):
            self.assertIn("Successfully wrote", tools.Write(str(target), text))
        self.assertEqual(existing.read_text(encoding="utf-8"), "OLD\n")
        self.assertFalse(fresh.exists())
        reader, writer = await self._connect()

        plain = await self._reply(
            reader,
            writer,
            {"type": "openFile", "path": str(existing), "workDir": str(wt_dir), "tabId": tab},
            "fileContent",
        )
        asked = [
            str(existing),
            "reports/existing.md",
            str(fresh),
            str(wt_dir / "reports" / "existing.md"),
            "reports/a\x00b.md",
        ]
        track_copy = self.server._printer._track_worktree_event
        track_copy({"type": "worktree_result", "action": "merge", "success": True}, tab, "left")
        pending = [
            await self._report(reader, writer, path, self.work_dir, tab, since=since)
            for path in asked
        ]
        shutil.rmtree(wt_dir)
        self._worktree_event(tab, "worktree_result", action="discard", success=True)
        discarded = await self._report(
            reader, writer, str(existing), self.work_dir, tab, since=since,
        )
        self._worktree_event(tab, "clear")
        other = str(wt_dir.with_name("kiss_wt-other"))
        track_copy({"type": "worktree_created", "worktreeDir": other}, tab, "left")
        later = int(time.time() * 1000)
        existing.write_text("NEXT\n", encoding="utf-8", newline="\n")
        straight = await self._report(
            reader, writer, str(existing), self.work_dir, tab, since=later,
        )

        self.assertEqual(plain["content"], "OLD\n")
        for path, reply in zip([*asked, str(existing)], [*pending, discarded], strict=True):
            self.assertEqual(reply["error"], f"File not found: {path}")
            self.assertEqual([reply["token"], "content" in reply], ["report:1", False])
        self.assertEqual(straight.get("content"), "NEXT\n")

    async def test_a_report_of_a_run_that_made_no_worktree_is_the_file_it_wrote(self) -> None:
        """A run without a worktree wrote its report into the checkout: that file is shown.

        The tab's earlier run left a worktree behind; its next run
        (``clear``) made none (an announcement that names no directory
        tells of none) and wrote the report straight into the checkout.
        The file is shown, by its absolute and its relative path,
        although it is an hour older than the ``Write`` is said to be:
        the daemon logged the run and goes by nothing else.  No file
        for a request the log does not hold: one that names no time,
        one of before the run, and one of a tab that was not sent the
        run's start (whatever else it was sent).  Nor for a path into a
        worktree the run knows nothing of, whether its directory is
        gone (nobody says that the checkout's file is what was written
        there) or still holds the file, nor for a report that is not
        there.  Nor, in the end, for the report itself, once as many
        other tabs have started a run as the daemon keeps logs for.
        """
        tab = "tab-report-plain"
        report = self.work_dir / "reports" / "plain.md"
        gone = self.work_dir / ".kiss-worktrees" / "kiss_wt-gone"
        kept = self.work_dir / ".kiss-worktrees" / "kiss_wt-kept" / "reports" / "plain.md"
        for path in (report, kept):
            path.parent.mkdir(parents=True)
            path.write_text("PLAIN\n", encoding="utf-8", newline="\n")
        hour_ago = time.time() - 3600
        os.utime(report, (hour_ago, hour_ago))
        self._worktree_event(tab, "clear")
        self._worktree_event(tab, "worktree_created", worktreeDir=str(gone))
        before = int(time.time() * 1000) - 1
        self._worktree_event(tab, "clear")
        self._worktree_event(tab, "worktree_done", worktrees=["junk", {"worktreeDir": 7}])
        self._worktree_event("tab-report-elsewhere", "worktree_result", action="discard")
        since = int(time.time() * 1000)
        reader, writer = await self._connect()

        shown = [
            await self._report(reader, writer, path, self.work_dir, tab, since=since)
            for path in (str(report), "reports/plain.md")
        ]
        not_logged = [
            await self._report(reader, writer, str(report), self.work_dir, asker, **when)
            for asker, when in (
                (tab, {}),
                (tab, {"since": True}),
                (tab, {"since": "soon"}),
                (tab, {"since": before}),
                ("tab-report-elsewhere", {"since": since}),
            )
        ]
        absent = [gone / "reports" / "plain.md", kept, report.with_name("none.md")]
        of_absent = [
            await self._report(reader, writer, str(path), self.work_dir, tab, since=since)
            for path in absent
        ]
        for later in range(_TAB_RUNS_MAX):
            self._worktree_event(f"tab-report-later-{later}", "clear")
        evicted = await self._report(reader, writer, str(report), self.work_dir, tab, since=since)

        for reply in shown:
            self.assertEqual([reply["path"], reply["content"]], [str(report.resolve()), "PLAIN\n"])
        for reply in (*not_logged, evicted):
            self.assertEqual(reply.get("error"), f"File not found: {report}")
        for path, reply in zip(absent, of_absent, strict=True):
            self.assertEqual(reply.get("error"), f"File not found: {path}")

    def _agent_in_a_worktree(self, tab_id: str) -> ChatSorcarAgent:
        """Start a task of *tab_id* in ``work_dir``, as far as its ``create_worktree`` call.

        ``work_dir`` becomes a repository with its files committed on
        ``main``.  The task's agent, registered as the tab's, makes its
        worktree with the real tool and works in it from then on.  It
        has the daemon's printer, as every task the daemon runs has, so
        the tab is told of the worktree.  Merges need no model.
        """
        self.enterContext(OfflineFastModel())
        repo = self.work_dir
        run_git_checked(repo, "init", "-q", "-b", "main")
        assert_repo_root(repo)
        for key, value in (("user.email", "t@example.com"), ("user.name", "T")):
            run_git_checked(repo, "config", key, value)
        run_git_checked(repo, "config", "commit.gpgsign", "false")
        run_git_checked(repo, "add", "-A")
        run_git_checked(repo, "commit", "-q", "--allow-empty", "-m", "initial")
        agent = ChatSorcarAgent("Sorcar")
        agent.work_dir = str(repo)
        agent._tab_id = tab_id
        agent.printer = self.server._printer
        self.assertIn("Created worktree", agent.create_worktree())
        state = agent_state.AgentState(f"task-{tab_id}", tab_id=tab_id, server_owned=True)
        state.agent = agent
        agent_state.register(state)
        return agent

    async def _until(self, reached: Callable[[], bool]) -> None:
        """Wait, half a minute at most, until *reached* returns true."""
        deadline = time.monotonic() + 30
        while not reached():
            self.assertLess(time.monotonic(), deadline, "the awaited state was never reached")
            await asyncio.sleep(0.02)

    @staticmethod
    def _merging(tab_id: str) -> bool:
        """Whether a Merge / Discard of the worktree of *tab_id* is running."""
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        with agent_state.STATE_LOCK:
            return state.is_merging

    async def _restart_daemon(self) -> None:
        """Stop the daemon and start a new one on the same endpoint and repository."""
        await self.server.stop_async()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(Path(self.tmpdir) / "cert.pem"),
            keyfile=str(Path(self.tmpdir) / "key.pem"),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()

    async def test_a_report_of_a_kept_worktree_is_not_shown_by_a_restarted_daemon(self) -> None:
        """A daemon that did not log the run has no file to show, whatever the files say.

        The run's task made its worktree and rewrote a report the
        checkout already held, written half a second before.  The user
        kept the worktree ("Do nothing"): no file, the worktree is not
        merged.  The daemon is then restarted and the page, which keeps
        the report across reloads, asks again: still no file, although
        no agent holds the worktree any more and the checkout's copy is
        about as old as the ``Write``.
        """
        tab = "tab-report-restart"
        report = self.work_dir / "reports" / "kept.md"
        report.parent.mkdir()
        report.write_text("OLD\n", encoding="utf-8", newline="\n")
        self._worktree_event(tab, "clear")
        agent = self._agent_in_a_worktree(tab)
        draft = agent.worktrees[0].wt_dir / "reports" / "kept.md"
        since = int(time.time() * 1000)
        os.utime(report, ((since - 500) / 1000, (since - 500) / 1000))
        tools = UsefulTools(work_dir=agent.work_dir)
        tools.Read(str(report))
        self.assertIn("Successfully wrote", tools.Write(str(report), "NEW\n"))
        reader, writer = await self._connect()

        await self._send(writer, {"type": "worktreeAction", "action": "nothing", "tabId": tab})
        result = await self._drain_until(reader, lambda m: m.get("type") == "worktree_result")
        kept = await self._report(reader, writer, str(report), self.work_dir, tab, since=since)
        await self._restart_daemon()
        reader, writer = await self._connect()
        restarted = await self._report(reader, writer, str(report), self.work_dir, tab, since=since)

        self.assertEqual([result.get("kept"), agent.worktrees], [True, []])
        self.assertEqual(
            [path.read_text(encoding="utf-8") for path in (report, draft)], ["OLD\n", "NEW\n"],
        )
        for reply in (kept, restarted):
            self.assertEqual(reply.get("error"), f"File not found: {report}")

    async def test_a_report_is_not_shown_while_its_merge_runs_or_after_the_merge_failed(
        self,
    ) -> None:
        """A merge that removed the worktree and has not committed yet has merged nothing.

        The task rewrote a report with an absolute checkout path, which
        landed in its worktree, and ended with the worktree pending.
        The user's Merge commits the branch, removes the worktree
        directory and applies the changes to the checkout; its commit
        is still held by a hook.  A page that asks now (a reload, a
        reconnect) gets no file, although the directory is gone and the
        checkout's file is new: no merge was announced.  The hook then
        rejects the commit and the checkout's older copy is put back,
        new date and all: still no file.  The user's second Merge
        succeeds, and the report is shown.  A second tab that views the
        task was not sent the run's start: no file for it at any time.
        """
        tab, viewer = "tab-report-merging", "tab-report-viewing"
        report = self.work_dir / "reports" / "merging.md"
        report.parent.mkdir()
        report.write_text("OLD\n", encoding="utf-8", newline="\n")
        self._worktree_event(tab, "clear")
        agent = self._agent_in_a_worktree(tab)
        wt_dir = agent.worktrees[0].wt_dir
        self.server._printer.subscribe_tab(f"task-{tab}", viewer)
        since = int(time.time() * 1000)
        tools = UsefulTools(work_dir=agent.work_dir)
        tools.Read(str(report))
        self.assertIn("Successfully wrote", tools.Write(str(report), "NEW\n"))
        present = self.server._agent_server._present_pending_worktree
        self.assertTrue(await asyncio.to_thread(present, tab, discard_if_empty=False))
        started, release = (Path(self.tmpdir) / name for name in ("started", "release"))
        hook = self.work_dir / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text(
            _BLOCK_THEN_REJECT_MERGE.format(started=started, release=release),
            encoding="utf-8",
            newline="\n",
        )
        hook.chmod(0o755)
        reader, writer = await self._connect()
        merge = {"type": "worktreeAction", "action": "merge", "tabId": tab, "workDir": str(wt_dir)}

        await self._send(writer, merge)
        try:
            await self._until(started.exists)
            applied = [wt_dir.exists(), report.read_text(encoding="utf-8")]
            merging = [
                await self._report(reader, writer, str(report), self.work_dir, asker, since=since)
                for asker in (tab, viewer)
            ]
        finally:
            release.touch()
        await self._until(lambda: not self._merging(tab))
        failed = await self._report(reader, writer, str(report), self.work_dir, tab, since=since)
        put_back = [report.read_text(encoding="utf-8"), len(agent.worktrees)]
        put_back_at = report.stat().st_mtime * 1000
        hook.unlink()
        await self._send(writer, merge)
        await self._drain_until(
            reader, lambda m: m.get("type") == "worktree_result" and m.get("success") is True,
        )
        merged, not_logged = [
            await self._report(reader, writer, str(report), self.work_dir, asker, since=since)
            for asker in (tab, viewer)
        ]

        self.assertEqual(applied, [False, "NEW\n"])
        self.assertEqual(put_back, ["OLD\n", 1])
        self.assertGreaterEqual(put_back_at, since, "the copy put back is new")
        for reply in (*merging, failed, not_logged):
            self.assertEqual(reply.get("error"), f"File not found: {report}")
        self.assertEqual(merged.get("content"), "NEW\n")

    async def test_a_merged_report_is_shown_whatever_the_date_of_its_file(self) -> None:
        """A merge leaves a file whose bytes did not change untouched, old date and all.

        The task made its worktree (the tab was told), rewrote an
        existing report with the same bytes, wrote a new one and a
        git-ignored one, and merged with its own ``commit_and_merge``.
        The unchanged report is shown although its file is an hour older
        than the ``Write``: the worktree it was written in was merged.
        It still is after the page reloaded (a replay drops the tab's
        subscriptions and its worktree directory, not what the daemon
        logged of its run).  A report the task named by its path in the
        worktree is shown from the checkout as well; were the merged
        worktree's directory still there, its file would not be.  A
        merge carries no ignored file, so the checkout's older file at
        that path is not the report.  Nor is the file the report of an
        earlier run that named the same path.
        """
        tab = "tab-report-same"
        reports = self.work_dir / "reports"
        same, changed, ignored = (reports / name for name in ("same.md", "new.md", "ignored.md"))
        reports.mkdir()
        same.write_text("SAME\n", encoding="utf-8", newline="\n")
        ignored.write_text("OLD\n", encoding="utf-8", newline="\n")
        (self.work_dir / ".gitignore").write_text(
            "reports/ignored.md\n", encoding="utf-8", newline="\n",
        )
        hour_ago = time.time() - 3600
        for old in (same, ignored):
            os.utime(old, (hour_ago, hour_ago))
        before = int(time.time() * 1000) - 1
        self._worktree_event(tab, "clear")
        agent = self._agent_in_a_worktree(tab)
        draft = agent.worktrees[0].wt_dir / "reports" / "same.md"
        since = int(time.time() * 1000)
        tools = UsefulTools(work_dir=agent.work_dir)
        tools.Read(str(same))
        for target, text in ((same, "SAME\n"), (changed, "NEW\n"), (ignored, "IGNORED\n")):
            self.assertIn("Successfully wrote", tools.Write(str(target), text))
        reader, writer = await self._connect()

        outcome = await asyncio.to_thread(agent.commit_and_merge)
        asks = [(same, since), (draft, since), (changed, since), (ignored, since), (same, before)]
        shown, by_draft, new, not_carried, earlier_run = [
            await self._report(reader, writer, str(path), self.work_dir, tab, since=at)
            for path, at in asks
        ]
        self.server._printer.cleanup_tab(tab)
        reloaded = await self._report(reader, writer, str(same), self.work_dir, tab, since=since)
        run_git_checked(self.work_dir, "worktree", "add", "-q", "--detach", str(draft.parents[1]))
        left = await self._report(reader, writer, str(draft), self.work_dir, tab, since=since)

        self.assertEqual(agent.worktrees, [], outcome)
        self.assertLess(same.stat().st_mtime, hour_ago + 60, "git rewrote the unchanged file")
        for reply in (shown, by_draft, reloaded):
            self.assertEqual(
                [reply.get("path"), reply.get("content")], [str(same.resolve()), "SAME\n"],
            )
        self.assertEqual(left.get("error"), f"File not found: {draft}")
        self.assertEqual(new.get("content"), "NEW\n")
        self.assertEqual(ignored.read_text(encoding="utf-8"), "OLD\n")
        self.assertEqual(not_carried.get("error"), f"File not found: {ignored}")
        self.assertEqual(earlier_run.get("error"), f"File not found: {same}")

    async def test_the_users_merge_merges_the_worktrees_its_tab_was_offered(self) -> None:
        """The user's Merge names no worktree: it merges the ones the tab's task ended with.

        A run of two tasks.  The first left a worktree pending, which
        the second put aside (no event says so) for one of its own.  The
        run ended with that one pending, and the user merged it.  A
        report the second task named by a path into its worktree is
        shown, from the checkout.  One named by a path into the first
        task's worktree is not: nothing merged that worktree.  Nor is
        one named by its checkout path, which either task may have
        written.

        In a second tab the user's Do nothing failed, after it may have
        detached the worktree the tab was offered, and the Merge that
        followed succeeded: nobody says that worktree is merged.
        """
        tab, second = "tab-report-aside", "tab-report-detached"
        report = self.work_dir / "reports" / "twice.md"
        report.parent.mkdir()
        report.write_text("REPORT\n", encoding="utf-8", newline="\n")
        aside, offered, detached = (
            self.work_dir / ".kiss-worktrees" / f"kiss_wt-{n}" for n in "abc"
        )
        for asker in (tab, second):
            self._worktree_event(asker, "clear")
        since = int(time.time() * 1000)
        for wt_dir in (aside, offered):
            self._worktree_event(tab, "worktree_created", worktreeDir=str(wt_dir))
        for asker, wt_dir in ((tab, offered), (second, detached)):
            pending = [{"worktreeDir": str(wt_dir)}]
            self._worktree_event(asker, "worktree_done", worktreeDir=str(wt_dir), worktrees=pending)
        self._worktree_event(second, "worktree_result", action="nothing", success=False)
        for asker in (tab, second):
            self._worktree_event(asker, "worktree_result", action="merge", success=True)
        reader, writer = await self._connect()

        in_offered, in_aside, in_detached = (
            wt_dir / "reports" / "twice.md" for wt_dir in (offered, aside, detached)
        )
        asks = [(tab, in_offered), (tab, in_aside), (tab, report), (second, in_detached)]
        of_offered, of_aside, of_either, of_detached = [
            await self._report(reader, writer, str(path), self.work_dir, asker, since=since)
            for asker, path in asks
        ]

        merged = [of_offered.get("path"), of_offered.get("content")]
        self.assertEqual(merged, [str(report.resolve()), "REPORT\n"])
        for path, reply in ((in_aside, of_aside), (report, of_either), (in_detached, of_detached)):
            self.assertEqual(reply.get("error"), f"File not found: {path}")

    async def test_a_report_goes_with_the_worktrees_of_its_own_repository(self) -> None:
        """A task with worktrees in two repositories: each report waits for its own repository's.

        The task worked in the checkout and in a repository nested in
        it, and rewrote a report in each.  Its ``commit_and_merge``
        merged one worktree and failed on the other, which the user
        then discarded.  In the first tab the nested repository's
        worktree is the merged one: its report is merged, although the
        worktree of a repository that contains it is not, and the
        checkout's is not, although its file is new.  In the second tab
        the checkout's is the merged one: its report is merged, and the
        nested repository's is not, although the worktree of a
        repository that contains it was merged.
        """
        nested = self.work_dir / "nested"
        repos = (nested, self.work_dir)
        inner, outer = (repo / "reports" / "r.md" for repo in repos)
        for report in (inner, outer):
            report.parent.mkdir(parents=True)
            report.write_text("OLD\n", encoding="utf-8", newline="\n")
        merged_repo = {"tab-report-inner-merged": nested, "tab-report-outer-merged": self.work_dir}
        for tab, merged in merged_repo.items():
            self._worktree_event(tab, "clear")
            wt_dirs = [str(repo / ".kiss-worktrees" / f"kiss_wt-{tab}") for repo in repos]
            for wt_dir in wt_dirs:
                self._worktree_event(tab, "worktree_created", worktreeDir=wt_dir)
            for repo, wt_dir in zip(repos, wt_dirs, strict=True):
                self._worktree_event(
                    tab, "worktree_result", action="merge", success=repo == merged,
                    worktreeDir=wt_dir,
                )
            self._worktree_event(tab, "worktree_result", action="discard", success=True)
        since = int(time.time() * 1000)
        reader, writer = await self._connect()

        replies = [
            await self._report(reader, writer, str(report), self.work_dir, tab, since=since)
            for tab in merged_repo
            for report in (inner, outer)
        ]

        self.assertEqual(
            [(reply.get("content"), reply.get("error")) for reply in replies],
            [
                ("OLD\n", None), (None, f"File not found: {outer}"),
                (None, f"File not found: {inner}"), ("OLD\n", None),
            ],
        )

    async def _run(
        self, reader: LocalReader, writer: LocalWriter, tab_id: str, work_dir: Path, **fields: Any,
    ) -> None:
        """Have the daemon run a task of *tab_id* in *work_dir*, to its end.

        The run ends for want of a model unless *fields* name one.
        """
        run = {"type": "run", "prompt": "write the report", "model": _UNAVAILABLE_MODEL}
        await self._send(writer, {**run, "workDir": str(work_dir), "tabId": tab_id, **fields})
        await self._frames_until_idle(reader, tab_id)

    @staticmethod
    def _told_folder(work_dir: object) -> Path:
        """The folder the page of a run in *work_dir* is told (``task_settings``) and asks in."""
        settings = ChatSorcarAgent("Sorcar")._build_extra_payload("m", str(work_dir), False)
        return Path(str(settings["work_dir"]))

    async def test_reports_written_in_a_worktree_by_other_runs_wait_for_its_merge(self) -> None:
        """Work in a task's worktree by its sub-agent, and by runs that start inside it.

        The task made its worktree: its tab was told.  A sub-agent of it
        works in the task's folder and writes with the real tool: by a
        relative path, by a path in the worktree and by a checkout path,
        which the tool redirects.  The page asks for those under the
        task's tab.  Three other tabs start inside the worktree and are
        sent no ``worktree_created``; the ``clear`` that began each run
        named its folder.  The user picked a link to the worktree as
        the first one's work dir (its task runs to the end, on a
        stand-in model), the second was viewing the first one's chat,
        and a script moved the third one's run into a sub-folder (and
        pinned its chat).  Each asks as its page does, in the folder its
        ``task_settings`` name: no file for any request while the
        checkout keeps its older copy, the checkout's file once the task
        has merged.  A tab whose run began in the checkout itself is
        shown the checkout's file, as ever.
        """
        owner_tab, picked, moved, viewing, plain = (
            f"tab-report-{start}" for start in ("owner", "picked", "moved", "viewing", "checkout")
        )
        report = self.work_dir / "reports" / "inside.md"
        report.parent.mkdir()
        report.write_text("OLD\n", encoding="utf-8", newline="\n")
        self._worktree_event(owner_tab, "clear")
        owner = self._agent_in_a_worktree(owner_tab)
        wt_dir = owner.worktrees[0].wt_dir
        link = Path(self.tmpdir) / "picked"
        link.symlink_to(wt_dir, target_is_directory=True)
        inside = wt_dir / "reports"
        script = Path(self.tmpdir) / "moved.py"
        moved_settings = {"chat_id": "chat-report-moved", "work_dir": str(inside)}
        script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def settings(self, settings):\n"
            f"        return settings | {moved_settings!r}\n",
            encoding="utf-8",
        )
        credentials = _FakeCredentials()
        self.addCleanup(credentials.restore)
        model = StandInModelServer(_finish_at_once)
        self.addCleanup(model.stop)
        real_run = {
            "model": STANDIN_MODEL, "modelConfig": model.model_config, "maxBudget": 5.0,
            "chatId": "chat-report-picked", "isParallel": False, "useWebTools": False,
        }
        agent_server = self.server._agent_server
        with agent_server._state_lock:
            agent_server._tab_chat_views[viewing] = "chat-report-picked"
        reader, writer = await self._connect()

        await self._run(reader, writer, picked, link, **real_run)
        await self._run(reader, writer, moved, self.work_dir, seaPath=str(script))
        await self._run(reader, writer, plain, self.work_dir)
        since = int(time.time() * 1000)
        sub_agent = UsefulTools(work_dir=owner.work_dir)
        sub_agent.Read(str(report))
        texts = {"relative.md": "REL\n", "within.md": "IN\n", "inside.md": "NEW\n"}
        written = ["reports/relative.md", str(inside / "within.md"), str(report)]
        for path in written:
            self.assertIn("Successfully wrote", sub_agent.Write(path, texts[Path(path).name]))
        in_checkout = sorted(path.name for path in report.parent.iterdir())
        in_link, in_repo = (self._told_folder(folder) for folder in (link, inside))
        asks = [
            *((owner_tab, self._told_folder(owner.work_dir), path) for path in written),
            (picked, in_link, str(report)), (picked, in_link, "reports/inside.md"),
            (viewing, in_link, "reports/inside.md"), (moved, in_repo, str(report)),
        ]
        unmerged = [
            await self._report(reader, writer, path, folder, tab, since=since)
            for tab, folder, path in asks
        ]
        unannounced = await self._report(reader, writer, str(report), in_repo, plain, since=since)
        outcome = await asyncio.to_thread(owner.commit_and_merge)
        merged = [
            await self._report(reader, writer, path, folder, tab, since=since)
            for tab, folder, path in asks
        ]

        self.assertEqual(in_checkout, ["inside.md"], "every Write landed in the worktree")
        # The worktree is under the checkout as git names it, with links resolved
        # (macOS's temp directory is under /var, a link to /private/var).
        self.assertEqual(
            [in_link, in_repo], [link, self.work_dir.resolve()], "what the pages are told",
        )
        self.assertEqual(
            [reply.get("error") for reply in unmerged],
            [f"File not found: {path}" for _tab, _folder, path in asks],
        )
        self.assertEqual(unannounced.get("content"), "OLD\n")
        self.assertEqual(owner.worktrees, [], outcome)
        self.assertEqual(
            [(reply.get("path"), reply.get("content")) for reply in merged],
            [
                (str((report.parent / Path(path).name).resolve()), texts[Path(path).name])
                for _tab, _folder, path in asks
            ],
        )

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
