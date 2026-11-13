# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``shareChat`` / ``shareChatTasks`` API.

The chat page's share button first sends ``shareChatTasks``: the
daemon answers with a ``share_tasks`` reply carrying the persisted
transcripts of ALL of the chat's tasks (oldest first) — the page's
own DOM holds only the one task the session replay repainted.  The
page assembles those transcripts (plus the live screen) into one
page body and sends it back via ``shareChat``; the daemon wraps it
into a standalone page
(``kiss.server.web_server._build_share_page``) and writes it to
``<workDir>/reports/chat-<title-slug>-<chatId>.html``.  These tests drive the REAL
production path — a live :class:`RemoteAccessServer` dispatcher over a
real local WSS connection, and over real remote WSS exactly like
the chat page — against a real history database, and assert on the
``share_tasks`` / ``share_done`` replies and on the page written to
disk.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss import __version__
from kiss.agents.sorcar import machines as mach
from kiss.core.brand import PRODUCT_NAME
from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _MAX_LINE_BYTES,
    _SHARE_TASKS_MAX_REPLY_BYTES,
    RemoteAccessServer,
    _share_subagent_entries,
)
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.persistence_helpers import close_db
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.pause import Pause

_PASSWORD = "share-chat-test-password"


class _LocalServerTestCase(unittest.TestCase):
    """Harness: a live local-channel dispatcher over the daemon's local endpoint."""

    def setUp(self) -> None:
        # ignore_cleanup_errors: the server's worker threads keep their
        # per-thread history.db connections open, and Windows refuses to
        # delete an open file (the OS reclaims the handle at exit).
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.work_dir = os.path.join(self.tmp.name, "workspace")
        os.makedirs(self.work_dir)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=self.work_dir,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=15)

    def tearDown(self) -> None:
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=10,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def _roundtrip(
        self, cmds: list[dict[str, Any]], want_type: str
    ) -> dict[str, Any]:
        """Send *cmds* over one fresh local connection; return the first
        received event of type *want_type*."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await open_local_connection(
                self.server,
                # A share_tasks reply can approach the transport's own
                # frame cap; the reader must accept what the daemon is
                # allowed to send.
                limit=_MAX_LINE_BYTES,
            )
            try:
                for cmd in cmds:
                    writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(
                        reader.readline(), timeout=10
                    )
                    if not line:
                        raise AssertionError(
                            f"connection closed before a {want_type!r} event"
                        )
                    event: dict[str, Any] = json.loads(line)
                    if event.get("type") == want_type:
                        return event
            finally:
                writer.close()
                await writer.wait_closed()

        return asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )


class TestShareChatOverLocal(_LocalServerTestCase):
    """``shareChat`` writes the shared page and answers ``share_done``."""

    def _share(self, **fields: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"type": "shareChat", "workDir": self.work_dir}
        cmd.update(fields)
        return self._roundtrip([cmd], "share_done")

    BODY = (
        '<div id="output"><div class="ev task-panel collapsible">'
        '<div class="task-panel-h collapse-header">Task</div>'
        '<div class="task-panel-text">list files</div></div>'
        '<div class="ev tc collapsible">'
        '<div class="tc-h collapse-header">Bash</div>'
        "<pre>ls -la</pre></div></div>"
    )

    def test_share_writes_standalone_page_and_replies_ok(self) -> None:
        event = self._share(
            chatId="chat-42",
            html=self.BODY,
            title="My chat",
            tabId="tab-1",
        )
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "tab-1")
        out = Path(self.work_dir) / "reports" / "chat-my-chat-chat-42.html"
        self.assertEqual(event["path"], str(out))
        page = out.read_text(encoding="utf-8")
        # The transcript body travels verbatim.
        self.assertIn(self.BODY, page)
        self.assertIn("<title>My chat</title>", page)
        # Self-contained: the chat page stylesheet, the highlight.js
        # theme, the VS Code palette and the collapse script are all
        # inlined, so the page needs no server and no other file.
        self.assertIn(".ev.task-panel, .ev.user-msg {", page)  # main.css
        self.assertIn(".collapse-preview", page)  # main.css
        self.assertIn(".hljs", page)  # highlight theme
        self.assertIn("--theme-editor-background: #1f1f1f", page)
        self.assertIn("renderShareTabBar", page)  # share.js
        # share.js's click delegation — a marker that appears in the
        # script only, never in this test's transcript body.
        self.assertIn(".closest('.collapse-header')", page)
        self.assertNotIn("{{", page.split("</title>")[0])
        # The page records the code that wrote it, after the chat: the
        # version and the commit of this checkout, as git names it.
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertIn(
            '\n</div>\n<footer id="share-version">'
            f"Saved by {PRODUCT_NAME} version {__version__}, commit {commit}"
            "</footer>\n",
            page,
        )
        self.assertIn("#share-version {", page)  # _SHARE_PAGE_CSS

    def test_share_page_carries_a_working_light_dark_toggle(self) -> None:
        """The written page ships the light/dark mode toggle: the
        floating button, both inlined highlight.js themes (light one
        parked behind ``media="not all"``), the ``html.light-theme``
        palette overrides and share.js's theme wiring."""
        event = self._share(chatId="chat-theme", html=self.BODY)
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn('id="share-theme-btn"', page)
        self.assertIn('aria-label="Switch to light mode"', page)
        self.assertIn('<style id="hljs-style-dark">', page)
        self.assertIn('<style id="hljs-style-light" media="not all">', page)
        # Light Modern variable overrides keyed off the toggled class.
        self.assertIn("html.light-theme,\n    body.remote-chat.light-theme {", page)
        self.assertIn("--theme-editor-background: #ffffff", page)
        # share.js's toggle logic (markers that appear in the script
        # only, never in this test's transcript body).
        self.assertIn("kissShareTheme", page)
        self.assertIn("Switch to dark mode", page)
        self.assertIn("#share-theme-btn {", page)  # _SHARE_PAGE_CSS

    def test_share_page_carries_the_subagent_tab_strip(self) -> None:
        # The page must ship everything the sub-agent tabs need with
        # it: share.js's tab-strip code and the page CSS that pins the
        # strip to the top and keeps unselected sections hidden.
        event = self._share(chatId="tabs", html=self.BODY)
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text()
        self.assertIn("data-rp-subagents", page)
        self.assertIn("#tab-bar", page)
        self.assertIn(".share-task[hidden]", page)

    def test_chat_id_is_sanitized_into_the_filename(self) -> None:
        event = self._share(chatId="a/b c!*", html=self.BODY)
        self.assertTrue(event["ok"], event)
        out = Path(self.work_dir) / "reports" / "chat-a-b-c.html"
        self.assertEqual(event["path"], str(out))
        self.assertTrue(out.is_file())

    def test_missing_html_field_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip(
            [{"type": "shareChat", "chatId": "c", "tabId": "tab-9"}],
            "error",
        )
        self.assertEqual(
            event["text"], "Invalid shareChat command: missing html"
        )
        self.assertEqual(event["tabId"], "tab-9")

    def test_blank_html_reports_an_empty_chat(self) -> None:
        event = self._share(chatId="c1", html="   ")
        self.assertFalse(event["ok"])
        self.assertEqual(event["error"], "Nothing to share: the chat is empty")
        self.assertFalse((Path(self.work_dir) / "reports").exists())

    def test_non_string_chat_id_reports_missing_chat_id(self) -> None:
        event = self._share(chatId=7, html=self.BODY)
        self.assertFalse(event["ok"])
        self.assertEqual(event["error"], "Missing chat id")

    def test_all_punctuation_chat_id_falls_back_to_chat(self) -> None:
        event = self._share(chatId="///", html=self.BODY)
        self.assertTrue(event["ok"], event)
        self.assertEqual(
            event["path"],
            str(Path(self.work_dir) / "reports" / "chat-chat.html"),
        )

    def test_title_is_escaped_and_non_string_title_is_dropped(self) -> None:
        event = self._share(
            chatId="esc", html=self.BODY, title="<script>alert(1)</script>"
        )
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn(
            "<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>", page
        )
        event = self._share(chatId="esc2", html=self.BODY, title=17)
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn(f"<title>{PRODUCT_NAME} chat</title>", page)

    def test_lone_surrogate_in_title_still_shares(self) -> None:
        # The page clips a title by UTF-16 units, so one ending mid-emoji
        # carries a lone high surrogate; JSON transports it, UTF-8 has
        # no form for it.  The share is still written and answered
        # (A22-4, audit 2026-11-07: the encode error escaped the handler
        # and no share_done ever came).
        event = self._share(chatId="emoji", html=self.BODY, title="x" * 29 + "\ud83d…")
        self.assertTrue(event["ok"], event)
        page = Path(event["path"]).read_text(encoding="utf-8")
        self.assertIn("<title>" + "x" * 29 + "?…</title>", page)

    def test_non_string_tab_id_is_normalized(self) -> None:
        event = self._share(chatId="tabless", html=self.BODY, tabId=99)
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "")

    def test_unwritable_reports_dir_reports_the_os_error(self) -> None:
        # A FILE named "reports" makes mkdir(parents=True) raise.
        (Path(self.work_dir) / "reports").write_text("in the way")
        event = self._share(chatId="c2", html=self.BODY)
        self.assertFalse(event["ok"])
        self.assertIn("Failed to write the chat page", event["error"])
        # A failure that is not an OSError is answered too: a NUL byte
        # in the folder raises ValueError.
        event = self._share(chatId="c2", html=self.BODY, workDir="nul\x00byte")
        self.assertFalse(event["ok"])
        self.assertIn("Failed to write the chat page", event["error"])

    def test_work_dir_comes_from_the_connection_when_not_sent(self) -> None:
        other = os.path.join(self.tmp.name, "other-workspace")
        os.makedirs(other)
        event = self._roundtrip(
            [
                {"type": "setWorkDir", "workDir": other},
                {"type": "shareChat", "chatId": "pinned", "html": self.BODY},
            ],
            "share_done",
        )
        self.assertTrue(event["ok"], event)
        self.assertEqual(
            event["path"],
            str(Path(other) / "reports" / "chat-pinned.html"),
        )


class TestShareChatTasksOverLocal(_LocalServerTestCase):
    """``shareChatTasks`` lists every task of a chat, oldest first."""

    def setUp(self) -> None:
        super().setUp()
        self.saved_db = redirect_persistence(self.tmp.name)

    def tearDown(self) -> None:
        restore_persistence(self.saved_db)
        super().tearDown()

    def _tasks(self, **fields: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"type": "shareChatTasks"}
        cmd.update(fields)
        return self._roundtrip([cmd], "share_tasks")

    def _seed(
        self, chat_id: str, task: str, content: str, events: int = 1,
    ) -> str:
        """Persist one task with *events* tool_call events; return id."""
        task_id, _ = th._add_task(task, chat_id=chat_id)
        for _ in range(events):
            th._append_chat_event(
                {"type": "tool_call", "name": "Bash", "command": content},
                task_id=task_id,
            )
        # timestamp is the ordering key; keep the rows distinct.
        time.sleep(0.01)
        return task_id

    def _seed_half_budget(self, chat_id: str, task: str) -> str:
        """Persist a task whose events total just over half the reply budget.

        Spread over rows each well under the persistence loader's
        ``MAX_EVENT_ROW_CHARS`` (4 MiB): a single oversized row would be
        clipped to a short note at load time and never overflow anything.
        """
        chunk = 1024 * 1024
        return self._seed(
            chat_id, task, "x" * chunk,
            events=_SHARE_TASKS_MAX_REPLY_BYTES // 2 // chunk + 1,
        )

    def test_every_task_of_the_chat_is_returned_oldest_first(self) -> None:
        first = self._seed("chat-a", "first task", "ls -la")
        second = self._seed("chat-a", "second task", "pwd")
        self._seed("chat-OTHER", "foreign task", "whoami")
        event = self._tasks(chatId="chat-a", tabId="tab-7")
        self.assertEqual(event["tabId"], "tab-7")
        self.assertEqual(event["chatId"], "chat-a")
        self.assertFalse(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [first, second]
        )
        self.assertEqual(
            [t["task"] for t in event["tasks"]],
            ["first task", "second task"],
        )
        # Each task's events start with the ensured task_settings event
        # (synthesized from the row when the stream carries none).
        self.assertEqual(
            [t["events"][0]["type"] for t in event["tasks"]],
            ["task_settings", "task_settings"],
        )
        self.assertEqual(
            event["tasks"][0]["events"][1]["command"], "ls -la"
        )
        self.assertEqual(event["tasks"][1]["events"][1]["command"], "pwd")

    def test_every_entry_carries_the_rows_run_metrics(self) -> None:
        # The saved page prints the Task Info rows under every task;
        # its Tokens / Cost / Steps / Time rows come from the row's
        # typed columns (the end time is in no transcript event).
        done = self._seed("chat-m", "finished task", "ls")
        th._save_task_extra(
            {
                "tokens": 4321,
                "cost": 0.4242,
                "steps": 7,
                "startTs": 1_700_000_000_000,
                "endTs": 1_700_000_125_000,
            },
            task_id=done,
        )
        running = self._seed("chat-m", "running task", "pwd")
        sub = self._seed("chat-m", "sub task", "echo sub")
        th._save_task_extra(
            {"parent_task_id": running, "tokens": 10, "steps": 1},
            task_id=sub,
        )
        event = self._tasks(chatId="chat-m")
        self.assertEqual(
            event["tasks"][0]["metrics"],
            {
                "tokens": 4321,
                "cost": 0.4242,
                "steps": 7,
                "start_ts": 1_700_000_000_000,
                "end_ts": 1_700_000_125_000,
            },
        )
        live = event["tasks"][1]["metrics"]
        self.assertEqual(
            (live["tokens"], live["cost"], live["steps"], live["end_ts"]),
            (0, 0.0, 0, 0),
            "a row without totals or an end reads as zero, not as missing",
        )
        self.assertGreater(
            live["start_ts"], 0, "the start falls back to the row's stamp"
        )
        sub_metrics = event["tasks"][1]["subagents"][0]["metrics"]
        self.assertEqual((sub_metrics["tokens"], sub_metrics["steps"]), (10, 1))

    def test_subagent_rows_are_not_chat_tasks(self) -> None:
        parent = self._seed("chat-s", "parent task", "ls")
        sub = self._seed("chat-s", "sub-agent task", "internal")
        th._save_task_extra(
            {"subagent": {"parent_task_id": parent}}, task_id=sub
        )
        event = self._tasks(chatId="chat-s")
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]],
            [parent],
            "a sub-agent transcript replays inside its parent's panels,"
            " never as a chat task of its own",
        )

    def test_subagent_transcripts_ride_inside_their_task(self) -> None:
        # A task that fanned out two sub-agents, the first of which ran
        # a fan-out of its own: the reply's task entry must carry all
        # three transcripts (parents before children), so the shared
        # page can open them as tabs like the live chat page does.
        parent = self._seed("chat-sub", "parent task", "ls")
        sub_one = self._seed("chat-sub", "first sub task", "echo one")
        sub_two = self._seed("chat-sub", "second sub task", "echo two")
        grand = self._seed("chat-sub", "grand sub task", "echo grand")
        plain = self._seed("chat-sub", "plain task", "pwd")
        th._save_task_extra({"parent_task_id": parent}, task_id=sub_one)
        th._save_task_extra({"parent_task_id": parent}, task_id=sub_two)
        th._save_task_extra({"parent_task_id": sub_one}, task_id=grand)
        event = self._tasks(chatId="chat-sub")
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [parent, plain]
        )
        subs = event["tasks"][0]["subagents"]
        self.assertEqual(
            [s["task_id"] for s in subs],
            [sub_one, sub_two, grand],
            "direct children first (enqueue order), then grandchildren",
        )
        self.assertEqual(
            [s["parent_task_id"] for s in subs],
            [parent, parent, sub_one],
        )
        self.assertEqual(
            [s["task"] for s in subs],
            ["first sub task", "second sub task", "grand sub task"],
        )
        # Each sub-agent's events are led by the ensured task_settings
        # event, exactly like the chat tasks' own event streams.
        self.assertEqual(
            [s["events"][0]["type"] for s in subs],
            ["task_settings"] * 3,
        )
        self.assertEqual(subs[0]["events"][1]["command"], "echo one")
        self.assertEqual(subs[2]["events"][1]["command"], "echo grand")
        self.assertEqual(
            event["tasks"][1]["subagents"],
            [],
            "a task that fanned out nothing carries an empty list",
        )

    def test_task_id_narrows_the_reply_to_one_task(self) -> None:
        # A sub-agent tab's share: its row is not a chat task, so the
        # page names it by taskId and gets back exactly that task
        # and its own descendants — never the chat's other tasks.
        parent = self._seed("chat-one", "parent task", "ls")
        sub = self._seed("chat-one", "sub task", "echo sub")
        grand = self._seed("chat-one", "grand task", "echo grand")
        self._seed("chat-one", "other task", "pwd")
        th._save_task_extra({"parent_task_id": parent}, task_id=sub)
        th._save_task_extra({"parent_task_id": sub}, task_id=grand)
        event = self._tasks(chatId="tab-sub-9", taskId=sub)
        self.assertEqual(event["chatId"], "tab-sub-9")
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [sub]
        )
        self.assertEqual(
            [s["task_id"] for s in event["tasks"][0]["subagents"]],
            [grand],
            "only the sub-agent's OWN descendants ride along",
        )
        self.assertFalse(event["truncated"])

    def test_unknown_task_id_yields_no_tasks(self) -> None:
        self._seed("chat-u", "some task", "ls")
        event = self._tasks(chatId="chat-u", taskId="no-such-task")
        self.assertEqual(event["tasks"], [])

    def test_non_string_task_id_is_ignored(self) -> None:
        task = self._seed("chat-n", "the task", "ls")
        event = self._tasks(chatId="chat-n", taskId=7)
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]],
            [task],
            "a malformed taskId falls back to the whole-chat listing",
        )

    def test_running_subagent_events_come_from_the_live_recording(
        self,
    ) -> None:
        # A still-running sub-agent's persisted events lag behind the
        # asynchronous writer; the share must use the printer's live
        # recording instead — the same safeguard the session replay
        # applies.
        parent = self._seed("chat-live", "parent task", "ls")
        sub = self._seed("chat-live", "sub task", "stale-cmd")
        th._save_task_extra({"parent_task_id": parent}, task_id=sub)
        printer = self.server._printer
        printer.ensure_recording_for_task(sub)
        printer.broadcast(
            {
                "type": "tool_call",
                "name": "Bash",
                "command": "live-cmd",
                "taskId": sub,
            }
        )
        event = self._tasks(chatId="chat-live")
        subs = event["tasks"][0]["subagents"]
        self.assertEqual([s["task_id"] for s in subs], [sub])
        commands = [
            e.get("command")
            for e in subs[0]["events"]
            if e.get("type") == "tool_call"
        ]
        self.assertEqual(
            commands,
            ["live-cmd"],
            "the live recording replaces the lagging persisted events",
        )

    def test_subagent_bytes_count_against_the_reply_budget(self) -> None:
        # The oldest task's sub-agent is what makes the pair overflow:
        # the old task must be dropped WITH its sub-agent, and the
        # newest kept — the same rule the plain task loader applies.
        old = self._seed("chat-sb", "old task", "ls")
        old_sub = self._seed_half_budget("chat-sb", "old sub")
        th._save_task_extra({"parent_task_id": old}, task_id=old_sub)
        newest = self._seed_half_budget("chat-sb", "new task")
        event = self._tasks(chatId="chat-sb")
        self.assertTrue(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [newest]
        )

    def test_subagent_transcripts_are_loaded_one_child_at_a_time(self) -> None:
        # The budget check must run BEFORE the next sibling's transcript
        # is read: a parent with four 256 KiB sub-agents and a 1 KiB
        # budget overflows at the first child, so the other three must
        # never be SELECTed (the same rule the task loader applies).
        # A child with a live recording is not read from disk at all.
        parent = self._seed("chat-lazy", "parent task", "ls")
        children = [
            self._seed("chat-lazy", f"child {i}", "x" * (256 * 1024))
            for i in range(4)
        ]
        for child in children:
            th._save_task_extra({"parent_task_id": parent}, task_id=child)
        selects: list[str] = []
        db = th._get_db()
        db.set_trace_callback(
            lambda sql: selects.append(sql) if "FROM events WHERE task_id" in sql else None,
        )
        try:
            entries, used, overflowed = _share_subagent_entries(
                parent, 1024, lambda task_id: [],
            )
            self.assertEqual((entries, overflowed), ([], True))
            self.assertGreater(used, 1024)
            self.assertEqual(len(selects), 1, "only the rejected first child was read")
            selects.clear()
            live = {children[0]: [{"type": "tool_call", "name": "Bash", "command": "live"}]}
            entries, _used, overflowed = _share_subagent_entries(
                parent, 600 * 1024, lambda task_id: live.get(task_id, []),
            )
        finally:
            db.set_trace_callback(None)
        self.assertEqual([e["task_id"] for e in entries], children[:3])
        self.assertTrue(overflowed)
        self.assertEqual(len(selects), 3, "the live child is not read from disk")

    def test_unknown_chat_id_yields_no_tasks(self) -> None:
        event = self._tasks(chatId="never-seen", tabId="tab-1")
        self.assertEqual(event["tasks"], [])
        self.assertFalse(event["truncated"])

    def test_blank_chat_id_yields_no_tasks(self) -> None:
        self._seed("chat-b", "some task", "ls")
        event = self._tasks(chatId="   ")
        self.assertEqual(event["tasks"], [])

    def test_non_string_chat_id_yields_no_tasks(self) -> None:
        event = self._tasks(chatId=7, tabId=42)
        self.assertEqual(event["tasks"], [])
        self.assertEqual(event["chatId"], "")
        self.assertEqual(event["tabId"], "")

    def test_missing_chat_id_is_rejected_by_the_catalog(self) -> None:
        event = self._roundtrip(
            [{"type": "shareChatTasks", "tabId": "tab-9"}], "error"
        )
        self.assertEqual(
            event["text"], "Invalid shareChatTasks command: missing chatId"
        )
        self.assertEqual(event["tabId"], "tab-9")

    def test_overlong_echo_ids_are_capped(self) -> None:
        # The echoed identifiers ride in every reply; a client must not
        # be able to push the reply past the frame cap by inflating
        # them.
        event = self._tasks(chatId="x" * 10_000, tabId="y" * 10_000)
        self.assertEqual(len(event["chatId"]), 256)
        self.assertEqual(len(event["tabId"]), 256)
        self.assertEqual(event["tasks"], [])

    def test_database_failure_reports_an_error(self) -> None:
        # A DIRECTORY at the database path makes sqlite3 unable to open
        # it — a real filesystem failure mode, not a simulated one.
        close_db()
        th._DB_PATH = Path(self.tmp.name) / "db-in-the-way"
        th._DB_PATH.mkdir()
        event = self._tasks(chatId="chat-x", tabId="tab-x")
        self.assertIn("Failed to load the chat history", event["error"])
        self.assertEqual(event["tasks"], [])
        self.assertEqual(event["tabId"], "tab-x")

    def test_database_failure_during_the_subagent_walk_reports_an_error(self) -> None:
        # The chat's rows are read, then the database goes away (the
        # same directory in its place) before the task's sub-agents are
        # looked up: the share click must still get its reply.
        self._seed("chat-w", "parent task", "ls")
        with (
            Pause(_share_subagent_entries.__code__, lambda frame_locals: True) as pause,
            concurrent.futures.ThreadPoolExecutor(1) as pool,
        ):
            pending = pool.submit(self._tasks, chatId="chat-w", tabId="tab-w")
            self.assertTrue(pause.paused.wait(10))
            close_db()
            th._DB_PATH = Path(self.tmp.name) / "db-in-the-way"
            th._DB_PATH.mkdir()
            pause.release()
            event = pending.result(timeout=15)
        self.assertIn("Failed to load the chat history", event["error"])
        self.assertEqual(event["tasks"], [])
        self.assertEqual(event["tabId"], "tab-w")

    def test_oldest_tasks_are_dropped_when_the_reply_would_overflow(
        self,
    ) -> None:
        # Two tasks whose payloads together exceed the reply budget but
        # fit it one at a time: the OLDER one must be dropped — the
        # newest transcripts are the ones the page cannot redraw
        # from its own DOM — and the reply must both fit the transport
        # frame and say it was truncated.
        self._seed_half_budget("chat-t", "old task")
        newest = self._seed_half_budget("chat-t", "new task")
        event = self._tasks(chatId="chat-t")
        self.assertTrue(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [newest]
        )


class TestShareChatOverWss(IsolatedAsyncioTestCase):
    """The remote webapp's share button path, over real WSS.

    Drives the exact frame sequence the browser produces — the
    ``auth`` handshake, then a ``shareChat`` catalog command on the
    authenticated socket — against a live :class:`RemoteAccessServer`,
    and asserts the page lands on disk and the ``share_done`` reply
    reaches the sending client.
    """

    async def asyncSetUp(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = int(sock.getsockname()[1])
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._work_dir = tempfile.mkdtemp()
        self._saved_db = redirect_persistence(self._work_dir)
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=self._work_dir,
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        restore_persistence(self._saved_db)
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def test_share_over_wss_writes_page_and_replies(self) -> None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=ctx,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
            while True:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if msg.get("type") == "auth_ok":
                    break
            await ws.send(json.dumps({
                "type": "shareChat",
                "chatId": "wss-chat",
                "title": "Remote chat",
                "html": TestShareChatOverLocal.BODY,
                "tabId": "tab-wss",
                "password": _PASSWORD,
            }))
            while True:
                event = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if event.get("type") == "share_done":
                    break
        self.assertTrue(event["ok"], event)
        self.assertEqual(event["tabId"], "tab-wss")
        out = Path(self._work_dir) / "reports" / "chat-remote-chat-wss-chat.html"
        self.assertEqual(event["path"], str(out))
        page = out.read_text(encoding="utf-8")
        self.assertIn(TestShareChatOverLocal.BODY, page)
        self.assertIn("<title>Remote chat</title>", page)

    async def test_share_chat_tasks_over_wss_lists_every_task(self) -> None:
        first, _ = th._add_task("first task", chat_id="wss-chat-2")
        th._append_chat_event(
            {"type": "tool_call", "name": "Bash", "command": "ls -la"},
            task_id=first,
        )
        time.sleep(0.01)
        second, _ = th._add_task("second task", chat_id="wss-chat-2")
        th._append_chat_event(
            {"type": "tool_call", "name": "Bash", "command": "pwd"},
            task_id=second,
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=ctx,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
            while True:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if msg.get("type") == "auth_ok":
                    break
            await ws.send(json.dumps({
                "type": "shareChatTasks",
                "chatId": "wss-chat-2",
                "tabId": "tab-wss-2",
                "password": _PASSWORD,
            }))
            while True:
                event = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                if event.get("type") == "share_tasks":
                    break
        self.assertEqual(event["tabId"], "tab-wss-2")
        self.assertEqual(event["chatId"], "wss-chat-2")
        self.assertFalse(event["truncated"])
        self.assertEqual(
            [t["task_id"] for t in event["tasks"]], [first, second]
        )
        self.assertEqual(
            event["tasks"][1]["events"][0]["type"], "task_settings"
        )
        self.assertEqual(
            event["tasks"][1]["events"][1]["command"], "pwd"
        )


class TestShareChatTasksOfAMachine(IsolatedAsyncioTestCase):
    """``shareChatTasks`` stamped ``machine`` is answered from that machine's history.

    A tab bound to a registered machine shows the machine's chat, whose
    tasks are in the machine's history database: the page stamps the
    share export's ``shareChatTasks`` with the binding and the daemon
    forwards it (:func:`kiss.server.sidebar_panels.machine_share_tasks_reply`)
    in the background on the ``panels`` link lane, the reply stamped
    ``machine``.  A real local daemon, a real remote daemon registered
    as ``box`` and seeded with the chat, a client on the local
    connection; this daemon holds a copy of the chat of its own, which
    the forwarded reply must not carry.
    """

    async def asyncSetUp(self) -> None:
        # Cleanups, each registered as its resource starts, not asyncTearDown:
        # unittest skips the tear-down when the set-up fails.
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.saved_db = redirect_persistence(self.tmpdir)
        self.addCleanup(restore_persistence, self.saved_db)
        mach.save_machines([])
        self.addCleanup(mach.save_machines, [])
        th._add_task("this daemon's task", chat_id="chat-shared")
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            seed=[
                {"chat_id": "chat-shared", "task": "older remote task",
                 "timestamp": 1_700_000_000.0},
                {"chat_id": "chat-shared", "task": "newer remote task",
                 "timestamp": 1_700_000_001.0},
            ],
        )
        self.addCleanup(self.remote.stop)
        await asyncio.to_thread(mach.add_machine, "box", self.remote.url, "pw-remote")
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        # Registered before the start: one that fails part-way is stopped too.
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(
            self.server, limit=_MAX_LINE_BYTES,
        )
        self.addCleanup(self.writer.close)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _next(self, reply_type: str) -> dict[str, Any]:
        """Return the next *reply_type* message of the local connection."""
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _share_tasks(self, **fields: Any) -> dict[str, Any]:
        """Ask ``shareChatTasks`` of ``chat-shared`` with *fields*; return the reply."""
        await self._send({"type": "shareChatTasks", "chatId": "chat-shared", **fields})
        return await self._next("share_tasks")

    async def test_the_bound_machines_history_answers_the_stamped_share(self) -> None:
        """The reply is the machine's tasks, stamped ``machine``; this daemon's copy stays out."""
        older = self.remote.task_ids["chat-shared/older remote task"]
        newer = self.remote.task_ids["chat-shared/newer remote task"]
        reply = await self._share_tasks(tabId="tab-1", machine="box")
        self.assertEqual(reply.get("machine"), "box", reply)
        self.assertNotIn("error", reply)
        self.assertEqual(
            (reply["tabId"], reply["chatId"], reply["truncated"]), ("tab-1", "chat-shared", False),
        )
        self.assertEqual([t["task_id"] for t in reply["tasks"]], [older, newer])
        self.assertEqual(
            [t["task"] for t in reply["tasks"]], ["older remote task", "newer remote task"],
        )
        self.assertEqual(reply["tasks"][0]["events"][0]["type"], "task_settings")
        # A sub-agent tab's share narrows the reply to its task: the
        # command reaches the machine whole (minus the stamp).
        narrowed = await self._share_tasks(tabId="tab-1", taskId=newer, machine="box")
        self.assertEqual(narrowed["machine"], "box")
        self.assertEqual([t["task_id"] for t in narrowed["tasks"]], [newer])
        # Unstamped, the command is this daemon's own, answered from its
        # history and unstamped.
        own = await self._share_tasks(tabId="tab-1")
        self.assertNotIn("machine", own)
        self.assertEqual([t["task"] for t in own["tasks"]], ["this daemon's task"])

    async def test_an_unknown_machine_is_an_empty_reply_carrying_the_error(self) -> None:
        reply = await self._share_tasks(tabId="tab-2", machine="nowhere")
        self.assertEqual(reply, {
            "type": "share_tasks", "tabId": "tab-2", "chatId": "chat-shared",
            "tasks": [], "truncated": False, "error": "no machine named 'nowhere'",
            "machine": "nowhere",
        })

    async def test_the_forwarded_share_holds_up_no_later_command(self) -> None:
        """The reply is produced in the background, on the machine's ``panels`` lane.

        While that lane's link is held (as a request in flight holds
        it), a ``ping`` sent after the share is answered; the share's
        reply, the machine's, follows once the link is free.
        """
        await self._share_tasks(machine="box")  # the lane's link exists now
        (box,) = [m for m in mach.load_machines() if m.name == "box"]
        panels = mach.link_for(box, "panels")
        with panels._lock:
            await self._send({
                "type": "shareChatTasks", "chatId": "chat-shared", "tabId": "tab-3",
                "machine": "box",
            })
            await self._send({"type": "ping"})
            while True:
                line = await asyncio.wait_for(self.reader.readline(), timeout=10.0)
                msg = json.loads(line.decode("utf-8"))
                self.assertNotEqual(
                    msg.get("type"), "share_tasks",
                    "answered while the machine's link is held: not by the machine",
                )
                if msg.get("type") == "pong":
                    break
        reply = await self._next("share_tasks")
        self.assertEqual((reply["tabId"], reply["machine"]), ("tab-3", "box"))
        self.assertEqual(
            [t["task"] for t in reply["tasks"]], ["older remote task", "newer remote task"],
        )


if __name__ == "__main__":
    unittest.main()
