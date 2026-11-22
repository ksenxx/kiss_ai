# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The Task Info panel's Tags row on the daemon: ``getTaskTags`` / ``setTaskTags``.

A task's tags are derived when it ends and stored comma-joined in
``task_history.tags``.  ``getTaskTags`` reads them for the page that
asked.  ``setTaskTags`` replaces them with the set the user edited and
tells every page (``taskTags``, then ``tasks_updated`` for the Chats
lists); a set that is empty, names a tag outside ``ALL_TAGS`` or
targets a row without tags or of a task still running changes nothing
and is answered to the sender alone, with the reason.  Edited tags are
the user's: nothing derives them again.

The commands of a tab bound to a registered machine are forwarded to
that machine: see ``test_machine_history.py``.  The last test follows
an edit through two real browsers, from the click to the database.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest
from playwright.sync_api import Page, expect, sync_playwright

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.task_metadata import ALL_TAGS
from kiss.scripts import backfill_task_metadata as backfill
from kiss.server import commands
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.server.conftest import (
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _open_page

NO_TAGS = "This task has no tags to edit: it is still running or was deleted."


def _finished(task: str, chat_id: str = "", extra: dict[str, object] | None = None) -> str:
    """Add *task* and finish it the way a run does, which derives its tags."""
    task_id, _ = th._add_task(task, chat_id, extra)
    th._save_task_result(result="done", task_id=task_id)
    th._save_task_extra({"endTs": 1_000}, task_id=task_id)
    return task_id


def _stored(task_id: str) -> list[str]:
    """The tags on row *task_id*, as a daemon running no task reads them."""
    return th._load_task_tags(task_id, set())[0]


def _tags_event(task_id: str, tags: list[str], **extra: Any) -> dict[str, Any]:
    """The ``taskTags`` the daemon sends for *task_id* carrying *tags*, those of a task over."""
    return {
        "type": "taskTags", "taskId": task_id, "tags": tags,
        "allTags": list(ALL_TAGS), "editable": bool(tags), **extra,
    }


class TestTaskTagsOfARow:
    """The row's functions and the handlers' refusals, on a scratch database."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self.paper = _finished("Write the paper introduction")
        self.running, _ = th._add_task("Still running, no tags yet")

    def teardown_method(self) -> None:
        th._close_thread_db()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tags_are_read_and_replaced_on_a_finished_row_only(self) -> None:
        assert th._load_task_tags(self.paper, set()) == (["work", "paper"], True)
        assert th._load_task_tags(self.running, set()) == ([], False)
        assert th._load_task_tags("no-such-task", set()) == ([], False)

        assert th._set_task_tags(self.paper, ["work", "coding"], set()) is True
        assert _stored(self.paper) == ["work", "coding"]
        # The Chats list's Tag filter reads the same column.
        assert [str(row["id"]) for row in th._load_history(tag="coding")] == [self.paper]
        assert th._load_history(tag="paper") == []
        # An empty column marks a row not finalized: its task's end derives the tags.
        assert th._set_task_tags(self.running, ["work"], set()) is False
        assert th._set_task_tags("no-such-task", ["work"], set()) is False
        assert _stored(self.running) == []

    def test_a_refused_set_changes_nothing_and_tells_the_sender_why(self) -> None:
        server, events = make_capturing_server()
        refusals: list[tuple[str, Any, str]] = [
            (self.paper, [], "A task keeps at least one tag."),
            (self.paper, "work", "A task keeps at least one tag."),
            (self.paper, ["work", "urgent"], "There is no tag named urgent."),
            (self.running, ["work"], NO_TAGS),
            ("no-such-task", ["work"], NO_TAGS),
        ]
        for task_id, tags, _why in refusals:
            server._handle_command(
                {"type": "setTaskTags", "taskId": task_id, "tags": tags, "connId": "c1"},
            )
        stored = {self.paper: ["work", "paper"]}
        assert events == [
            _tags_event(task_id, stored.get(task_id, []), error=why, connId="c1")
            for task_id, _tags, why in refusals
        ]
        assert _stored(self.paper) == ["work", "paper"]

    def test_a_history_that_cannot_be_opened_is_answered_with_the_failure(self) -> None:
        """A directory stands where ``history.db`` was: both commands are still answered.

        A machine link (the ``generation``) would otherwise wait out its timeout.
        """
        server, events = make_capturing_server()
        db_file = Path(th._DB_PATH)
        blocked = db_file.with_name("history.db.blocked")
        blocked.mkdir()
        th._close_thread_db()
        th._DB_PATH = blocked
        try:
            server._handle_command({"type": "getTaskTags", "taskId": self.paper, "connId": "c1"})
            server._handle_command({
                "type": "setTaskTags", "taskId": self.paper, "tags": ["work"],
                "connId": "c1", "generation": 7,
            })
        finally:
            th._close_thread_db()
            th._DB_PATH = db_file
            blocked.rmdir()
        failure = "unable to open database file."
        assert events == [
            _tags_event(
                self.paper, [], error=f"Could not read the task's tags: {failure}", connId="c1",
            ),
            _tags_event(
                self.paper, [], error=f"Could not save the task's tags: {failure}",
                generation=7, connId="c1",
            ),
        ]
        assert _stored(self.paper) == ["work", "paper"]

    def test_an_edit_outlives_the_events_that_reach_the_history_late(self) -> None:
        """A replay of journalled events classifies finished rows again, except an edited one."""
        server, events = make_capturing_server()
        other = _finished("Write the paper introduction")
        # The tags of a release whose classifier knew fewer of them.
        th._get_db().execute("UPDATE task_history SET tags = 'paper' WHERE id = ?", (other,))
        server._handle_command({"type": "setTaskTags", "taskId": self.paper, "tags": ["review"]})
        late = json.dumps({"type": "text_delta", "text": "late"})
        th._journal_failed_events(
            [(row, late, time.time(), th._current_db_path()) for row in (self.paper, other)], 0,
        )
        th._flush_chat_events()
        # Nor does a final save repeated for the edited row, or a backfill, derive its tags again.
        th._save_task_extra({"endTs": 2_000}, task_id=self.paper)
        assert backfill.main(["--db", str(th._DB_PATH)]) == 0
        events.clear()
        for row in (self.paper, other):
            server._handle_command({"type": "getTaskTags", "taskId": row, "connId": "c1"})
        assert events == [
            _tags_event(self.paper, ["review"], connId="c1"),
            _tags_event(other, ["work", "paper"], connId="c1"),
        ]

    def test_a_history_from_before_edits_were_marked_keeps_the_ones_made_since(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Rows without an end, their owner gone or none: tasks that are over, so editable.

        What a killed daemon and an old release left, in a database from
        before ``tags_edited``.  Both rows are dead ones the next daemon
        start's sweep finalizes, and a ``--refresh`` derives every tag
        again: all but the edited row's.
        """
        th._close_thread_db()
        conn = sqlite3.connect(th._DB_PATH, isolation_level=None)
        # The table's own text without the column and the comment above it: SQLite's
        # ``DROP COLUMN`` of the last column trips over the comma in that comment.
        (ddl,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'task_history'"
        ).fetchone()
        legacy_ddl = re.sub(r",\s*(?:--[^\n]*\n\s*)*tags_edited INTEGER DEFAULT 0", "", ddl)
        assert "tags_edited" not in legacy_ddl
        conn.execute("DROP TABLE task_history")
        conn.execute(legacy_ddl)
        edited, derived = uuid.uuid4().hex, uuid.uuid4().hex
        conn.executemany(
            "INSERT INTO task_history (id, timestamp, task, result, tags, owner) "
            "VALUES (?, 1.0, 'Write the paper introduction', 'Agent Failed Abruptly', ?, ?)",
            [(edited, "work,paper,failed", "a-daemon-long-gone"), (derived, "paper", "")],
        )
        conn.close()
        marks = th._get_db().execute("SELECT tags_edited FROM task_history").fetchall()
        assert [tuple(mark) for mark in marks] == [(0,), (0,)]

        tags = ["work", "paper", "failed"]
        assert th._load_task_tags(edited, set()) == (tags, True)
        # Unless this daemon is the one running the task.
        assert th._load_task_tags(edited, {edited}) == (tags, False)
        assert th._set_task_tags(edited, ["review"], {edited}) is False
        assert th._set_task_tags(edited, ["review"], set()) is True

        assert th._recover_orphaned_tasks(set()) == 2
        assert (_stored(edited), _stored(derived)) == (["review"], tags)
        assert backfill.main(["--db", str(th._DB_PATH), "--refresh"]) == 0
        assert "tagged 1 tasks" in capsys.readouterr().out
        assert _stored(edited) == ["review"]
        assert th._load_task_tags(derived, set()) == (tags, True)

    @pytest.mark.parametrize("edit", [False, True], ids=["read", "edit"])
    @pytest.mark.parametrize("replaces", [False, True], ids=["no row", "a dead daemon's row"])
    def test_a_task_starting_past_the_liveness_probe_is_not_the_row_probed(
        self, edit: bool, replaces: bool,
    ) -> None:
        """The probe holds no lock: a task can start on the id before the statement it guards.

        The task's row was not there to probe, or replaces one a dead daemon
        left, and a backfill tags it at once.  The probe's "not alive" is
        not about this row: its task runs, so its tags are read-only.
        """
        server, events = make_capturing_server()
        task_id, pending = uuid.uuid4().hex, [True]
        if replaces:
            th._add_task("An old task", task_id=task_id)
            th._get_db().execute(
                "UPDATE task_history SET tags = 'work', owner = 'a-daemon-long-gone' WHERE id = ?",
                (task_id,),
            )

        def start_the_task(_entered: dict[str, Any]) -> bool:
            """Run at the ``_get_db`` a tags read or edit calls once its probe is done."""
            callers = [frame.function for frame in inspect.stack(0)]
            caller = callers[callers.index("_get_db") + 1]
            if pending and caller in ("_load_task_tags", "_set_task_tags"):
                pending.clear()
                th._delete_task(task_id)
                th._add_task("Write the paper introduction", task_id=task_id)
                assert backfill.main(["--db", str(th._DB_PATH)]) == 0
            return False

        command = {"type": "setTaskTags", "tags": ["review"]} if edit else {"type": "getTaskTags"}
        with Pause(th._get_db.__code__, start_the_task):
            server._handle_command({**command, "taskId": task_id, "connId": "c1"})
        assert not pending, "the command made no liveness probe"
        tags, refusal = ["work", "paper", "failed"], {"error": NO_TAGS} if edit else {}
        assert events == [_tags_event(task_id, tags, editable=False, connId="c1", **refusal)]
        assert _stored(task_id) == tags


class TestTaskTagsOnTheWire(IsolatedAsyncioTestCase):
    """Two pages connected to one daemon: which of them is sent what."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.saved = redirect_persistence(self.tmpdir)
        self.addCleanup(restore_persistence, self.saved)
        self.paper = _finished("Write the paper introduction")
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_async()
        self.pages = []
        for _ in range(2):
            reader, writer = await open_local_connection(self.server)
            self.addCleanup(writer.close)
            self.pages.append((reader, writer))
            await self._sent_to(len(self.pages) - 1)

    async def _write(self, page: int, *cmds: dict[str, Any] | None) -> None:
        """Send each of *cmds* that is one from page *page*."""
        writer = self.pages[page][1]
        for frame in cmds:
            if frame is not None:
                writer.write(json.dumps(frame).encode("utf-8") + b"\n")
        await writer.drain()

    async def _sent_to(self, page: int, cmd: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Send *cmd* from page *page*, if any; what the page got before the pongs of two pings.

        The first pong says the daemon took *cmd*, not that its answer
        is out.  The command's thread hands the answer to the event
        loop; when the command ends before the loop has chained its
        future (``run_in_executor``), Python 3.14 sets that future at
        once: the dispatcher goes on to the ping without yielding to
        the loop, and the pong leaves first.  The debug mode a test's
        loop runs in makes that the rule on a warm thread.  What the
        command handed over is sent before the pong of a ping the
        daemon gets after that pong.
        """
        await self._write(page, cmd, {"type": "ping"})
        frames = await self._until_pong(page)
        await self._write(page, {"type": "ping"})
        return frames + await self._until_pong(page)

    async def _until_pong(self, page: int) -> list[dict[str, Any]]:
        """What page *page* is sent before the pong of the ping it sent.

        Returned are the ``taskTags``, ``tasks_updated`` and ``error``
        frames, without the printer's ``ts`` stamp: the daemon pushes
        lists of its own (the SEA commands) to a fresh connection.
        """
        reader = self.pages[page][0]
        frames: list[dict[str, Any]] = []
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=20.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == "pong":
                return frames
            if msg.get("type") in ("taskTags", "tasks_updated", "error"):
                msg.pop("ts", None)
                frames.append(msg)

    async def test_a_read_answers_its_page_and_an_edit_tells_every_page(self) -> None:
        get = {"type": "getTaskTags", "taskId": self.paper}
        self.assertEqual(await self._sent_to(0, get), [_tags_event(self.paper, ["work", "paper"])])
        self.assertEqual(await self._sent_to(1), [])

        # Stored without repeats, in the vocabulary's order; every page repaints and refetches.
        edit = {"type": "setTaskTags", "taskId": self.paper, "tags": ["paper", "coding", "work"]}
        told = [
            _tags_event(self.paper, ["work", "coding", "paper"], tabId=""),
            {"type": "tasks_updated"},
        ]
        self.assertEqual(await self._sent_to(0, {**edit, "tags": [*edit["tags"], "coding"]}), told)
        self.assertEqual(await self._sent_to(1), told)
        self.assertEqual(_stored(self.paper), ["work", "coding", "paper"])

        # A machine link's command names a generation: its answer is a copy echoing it.
        pushed = _tags_event(self.paper, ["work"], tabId="")
        self.assertEqual(
            await self._sent_to(0, {**edit, "tags": ["work"], "generation": 5}),
            [pushed, _tags_event(self.paper, ["work"], generation=5), {"type": "tasks_updated"}],
        )
        self.assertEqual(await self._sent_to(1), [pushed, {"type": "tasks_updated"}])
        self.assertEqual(
            await self._sent_to(1, {**get, "generation": 9}),
            [_tags_event(self.paper, ["work"], generation=9)],
        )

        # A refusal is the sender's alone, and `tags` is a required field.
        self.assertEqual(
            await self._sent_to(1, {**edit, "tags": []}),
            [_tags_event(self.paper, ["work"], error="A task keeps at least one tag.")],
        )
        (invalid,) = await self._sent_to(1, {"type": "setTaskTags", "taskId": self.paper})
        self.assertEqual(invalid["type"], "error")
        self.assertIn("tags", invalid["text"])
        self.assertEqual(await self._sent_to(0), [])

    async def test_tags_held_before_they_are_sent_do_not_overtake_a_later_edit(self) -> None:
        """A page shows the tags of the last ``taskTags`` it is sent: those are the stored ones.

        A read's answer, or an edit's own event, built and not sent yet
        was overtaken by another page's edit and its event and sent
        after them: the row's chips went back to tags no longer stored.
        The first page's command is held on the daemon's handler thread
        with its answer built, and what a page shows is what the
        printer sent its connection, in that order.
        """
        edit = {"type": "setTaskTags", "taskId": self.paper, "tags": ["review"]}
        for first in ({"type": "getTaskTags"}, {"type": "setTaskTags", "tags": ["coding"]}):
            with self.subTest(first=first["type"]):
                with Pause(commands._task_tags_event.__code__, bool) as held:
                    await self._write(0, {**first, "taskId": self.paper})
                    self.assertTrue(await asyncio.to_thread(held.paused.wait, 20.0))
                    # Half a second to overtake the held answer: an edit
                    # that has to wait for it is still waiting then.
                    await self._write(1, edit, {"type": "ping"})
                    await asyncio.sleep(0.5)
                # The other page's pong follows its edit's events to every page.
                await self._until_pong(1)
                sent = [f["tags"] for f in await self._sent_to(0) if f["type"] == "taskTags"]
                self.assertEqual(sent[-1], _stored(self.paper))
                self.assertEqual(_stored(self.paper), ["review"])

    async def test_a_task_tagged_while_it_runs_is_read_only_until_it_ends(self) -> None:
        """Tags do not say a task is over: a metadata backfill tags the row of a running one too."""
        running, _ = th._add_task("Still running, no tags yet")
        self.assertEqual(backfill.main(["--db", str(th._DB_PATH)]), 0)
        shown = _tags_event(running, ["work", "failed"], editable=False)
        get = {"type": "getTaskTags", "taskId": running}
        self.assertEqual(await self._sent_to(0, get), [shown])
        edit = {"type": "setTaskTags", "taskId": running, "tags": ["review"]}
        self.assertEqual(await self._sent_to(0, edit), [{**shown, "error": NO_TAGS}])
        self.assertEqual(await self._sent_to(1), [])
        # The task's end derives its tags again, and they are the user's to edit from then on.
        th._save_task_result(result="done", task_id=running)
        th._save_task_extra({"endTs": 2_000}, task_id=running)
        self.assertEqual(await self._sent_to(0, get), [_tags_event(running, ["work"])])


# Whether the page's Tags row shows exactly the given tags, in their order.
_SHOWS_TAGS_JS = """tags => tags.join() ===
  [...document.querySelectorAll('#meta-tags .meta-tag')].map(chip => chip.firstChild.data).join()"""


# Whether the page's Connectors subpanel has its answer (its status reads "Checking…" until then).
_APPS_ANSWERED_JS = """() =>
  !document.getElementById('meta-apps-status').textContent.startsWith('Checking')"""


def _wait_for_tags(pages: list[Page], tags: list[str]) -> None:
    """Wait until the Tags row of each of *pages* shows *tags*."""
    for page in pages:
        page.wait_for_function(_SHOWS_TAGS_JS, arg=tags, timeout=30000)


def test_an_edit_in_one_browser_shows_in_the_other_and_in_the_chats_list() -> None:
    """Two real pages on one real daemon, both showing the same finished task."""
    harness = _ServerHarness()
    try:
        chat, prompt = "chat-task-tags", "Write the paper introduction"
        task_id = _finished(prompt, chat, {"work_dir": str(harness.work_dir)})
        assert th._set_task_tags(task_id, ["work", "testing", "paper"], set())
        # The transcript a run leaves: its settings name the task the panel shows.
        events: list[dict[str, object]] = [
            {"type": "task_settings", "settings": {"task_id": task_id}},
            {"type": "prompt", "text": prompt},
            {"type": "result", "summary": "<p>done</p>", "success": True},
        ]
        for event in events:
            th._queue_chat_event(event, task_id=task_id)
        th._flush_chat_events()
        group = f'#history-list .history-chat-group[data-chat-id="{chat}"]'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--ignore-certificate-errors"])
            pages: list[Page] = []
            errors: list[str] = []
            for _ in range(2):
                _context, page, _sent = _open_page(browser, harness)
                page.set_viewport_size({"width": 1400, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                # The chat's task sits folded under its header, the Task Info list under its own.
                page.locator(f"{group} .history-chat-header > .history-chat-collapse").click()
                page.locator(f"{group} .sidebar-item").click()
                page.locator('.meta-section-toggle[aria-controls="meta-list"]').click()
                pages.append(page)
            one, two = pages
            _wait_for_tags(pages, ["work", "testing", "paper"])

            one.locator("#meta-tags .meta-tag-remove").first.click()
            _wait_for_tags(pages, ["testing", "paper"])
            assert _stored(task_id) == ["testing", "paper"]

            two.select_option("#meta-tags .meta-tag-add", "review")
            _wait_for_tags(pages, ["testing", "review", "paper"])
            assert _stored(task_id) == ["testing", "review", "paper"]
            # The task's Chats row, which the Tag filter reads, follows on the other page too.
            expect(one.locator(f"{group} .sidebar-item")).to_have_attribute(
                "data-tags", "testing,review,paper", timeout=30000,
            )
            assert errors == []
            # Each page asked for its Connectors subpanel on connecting, and the daemon
            # answers once its probe of the apps, a subprocess, has ended: a daemon
            # stopped before that leaves the probe running.
            for page in pages:
                page.wait_for_function(_APPS_ANSWERED_JS, timeout=60000)
            browser.close()
    finally:
        harness.stop()
