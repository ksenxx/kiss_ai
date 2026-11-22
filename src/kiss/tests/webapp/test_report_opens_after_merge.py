# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: a finished task's report opens in the editor after its merge.

When a task writes a report (a ``.md`` / ``.html`` file under a
``reports`` folder), the page asks the daemon for that file once the
task has ended and shows it in the editor pane as the file it is: the
text on disk, editable, not the text of the ``Write`` call.  A report
written while the tab's run had a worktree is not merged yet: the
daemon has no file for it until it has announced that worktree's merge,
then it is shown from the repository's copy, whatever that file's date,
and never when the worktree is discarded.

The tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` — no mocks.  The
task's events are delivered through the window ``message`` dispatch the
WebSocket shim uses for real frames; the daemon answers the page's
``openFile`` requests from real files, and the run's start and the
worktree's events (created, merged, discarded) are the daemon's own
broadcasts.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _uncover_pane
from kiss.tests.webapp.test_content_tab_file_links import (
    _wait_ready,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_preview_editing import (
    _CONTENT_TAB,
    _FRAME,
    _MODE_BTN,
)
from kiss.tests.webapp.test_lazy_chat_loading_e2e import _open_task, _seed
from kiss.tests.webapp.test_result_ls_listing_file_links import _open_page


def _deliver(page, *events: dict) -> None:
    """Deliver *events* to the page the way the WebSocket shim does."""
    page.evaluate(
        """(events) => {
             for (const data of events) {
               window.dispatchEvent(new MessageEvent('message', {data}));
             }
           }""",
        list(events),
    )


def _deliver_at(page, start_ms: int, events: list[dict]) -> None:
    """Have the page deliver *events* in one go once its clock reads *start_ms*.

    ``window._delivered`` resolves when it has.
    """
    page.evaluate(
        """([start, events]) => {
             window._delivered = new Promise(resolve => {
               setTimeout(() => {
                 for (const data of events) {
                   window.dispatchEvent(new MessageEvent('message', {data}));
                 }
                 resolve(true);
               }, Math.max(0, start - Date.now()));
             });
           }""",
        [start_ms, events],
    )


def _active_tab(page) -> str:
    """The id of the chat tab on screen."""
    tab_id: str = page.evaluate("() => window._testApi.getActiveTabId()")
    return tab_id


def _start_run(harness, page) -> str:
    """Have the daemon start a run in the chat tab on screen; return the tab's id.

    The run's ``clear`` is the daemon's own broadcast.  Returns once the
    page has handled it: it empties the tab's transcript.
    """
    tab_id = _active_tab(page)
    page.evaluate(
        """() => {
             window._runStarted = new Promise(resolve => {
               window.addEventListener('message', function started(e) {
                 if (!e.data || e.data.type !== 'clear') return;
                 window.removeEventListener('message', started);
                 resolve(true);
               });
             });
           }""",
    )
    harness.server._printer.broadcast({"type": "clear", "tabId": tab_id})
    page.evaluate("() => window._runStarted")
    return tab_id


def _write_events(tab_id: str, path: str, written: str) -> list[dict]:
    """The events of a ``Write`` of *written* to *path* by the task shown in *tab_id*.

    The call is stamped with the time, as the daemon stamps every event.
    """
    return [
        {
            "type": "tool_call", "name": "Write", "path": path, "content": written,
            "tabId": tab_id, "ts": int(time.time() * 1000),
        },
        {
            "type": "tool_result", "tool_name": "Write", "path": path, "is_error": False,
            "content": f"Successfully wrote {len(written)} characters to {path}",
            "tabId": tab_id,
        },
    ]


def _finish_task_writing(page, path: str, written: str) -> None:
    """Deliver the task of the tab on screen: it writes *written* to *path* and ends."""
    tab_id = _active_tab(page)
    _deliver(
        page,
        *_write_events(tab_id, path, written),
        {"type": "task_done", "success": True, "tabId": tab_id},
    )


def _sub_agent_writes(page, tab_id: str, path: str, written: str) -> None:
    """Deliver a sub-agent of the running task of *tab_id*: it writes *written* to *path* and ends.

    The page shows the sub-agent in a tab of its own, which the
    daemon's ``subagentDone`` closes.
    """
    sub_tab = f"{tab_id}__sub_e2e"
    _deliver(
        page,
        {"type": "status", "running": True, "tabId": tab_id},
        {
            "type": "openSubagentTab", "tab_id": sub_tab, "parent_tab_id": tab_id,
            "description": "sub", "task_id": "e2e", "taskIndex": 0,
        },
        *_write_events(sub_tab, path, written),
        {"type": "subagentDone", "tab_id": sub_tab},
    )


def _report_requests(page, sent: list[dict], count: int) -> list[dict]:
    """Wait until the page has asked the daemon for *count* reports; return the requests."""
    requests: list[dict] = []
    for _ in range(100):
        requests = [
            frame for frame in sent
            if frame.get("type") == "openFile"
            and str(frame.get("token", "")).startswith("report:")
        ]
        if len(requests) >= count:
            return requests
        page.wait_for_timeout(100)
    raise AssertionError(f"the page asked for {len(requests)} report(s), expected {count}")


def _first_report_asked_after_reload(page, sent: list[dict]) -> str:
    """Reload *page*; return the path of the first report its new connection asks for."""
    mark = len(sent)
    page.reload()
    for _ in range(100):
        frames = sent[mark:]
        starts = [i for i, frame in enumerate(frames) if frame.get("type") == "auth"]
        for frame in frames[starts[-1]:] if starts else []:
            if str(frame.get("token", "")).startswith("report:"):
                path: str = frame["path"]
                return path
        page.wait_for_timeout(100)
    raise AssertionError("the reloaded page asked for no report")


# JavaScript: the names of the localStorage items a page keeps its pending reports in.
_STORED_REPORTS = (
    "Object.keys(localStorage).filter(name => name.startsWith('kiss-pending-report:'))"
)


def _stored_reports(page) -> list[str]:
    """The names of the localStorage items the page keeps its pending reports in."""
    names: list[str] = page.evaluate(f"() => {_STORED_REPORTS}")
    return names


def _fill_storage(page) -> None:
    """Use up the browser's quota of the page's localStorage: no further item fits."""
    page.evaluate(
        """() => {
             let size = 0;
             for (let more = 1 << 23; more; more >>= 1) {
               try {
                 localStorage.setItem('filler', 'x'.repeat(size + more));
                 size += more;
               } catch (_full) {}
             }
           }""",
    )


# JavaScript, the condition of a breakpoint on a method of ``Storage``: the call names the
# item of a single report, whose name it notes.
_NAMES_REPORT_ITEM = (
    "/^kiss-(pending|settled)-report:/.test(arguments[0]) && (window._stalledAt = arguments[0])"
)


def _next_pause(page, pauses: list[dict]) -> dict:
    """Wait until the debugger has stopped a window; return that pause.

    *page* is another window, which keeps running meanwhile.
    """
    for _ in range(300):
        if pauses:
            return pauses.pop(0)
        page.wait_for_timeout(100)
    raise AssertionError("the debugger did not stop the window")


def _stall_at_report_item(context, page, url: str):
    """Open *url* in a new window; stop it at its first access to a single report's item.

    The debugger (a breakpoint on ``Storage.getItem`` and
    ``Storage.setItem``) stops the window while it loads, as a slow
    machine might anywhere: no function of the page is replaced.
    *page* is another window, which keeps running.  Returns the
    stopped window, its debugger session (``Debugger.disable`` lets
    the window get on), the name of the item it is about to access and
    the live list of the frames it sends to the daemon.
    """
    window = context.new_page()
    sent: list[dict] = []
    window.on("websocket", lambda socket: socket.on(
        "framesent", lambda payload: sent.append(json.loads(payload)),
    ))
    debugger = context.new_cdp_session(window)
    pauses: list[dict] = []
    debugger.on("Debugger.paused", lambda pause: pauses.append(pause))
    debugger.send("Page.enable")
    debugger.send("Debugger.enable")
    # The document of *url* stops before its first script: its Storage methods get the
    # breakpoint.  No other document stops: a navigation that is retried after a network
    # change leaves Chromium's error page behind, whose stop would come first and name a
    # frame that is gone.
    first = debugger.send("Page.addScriptToEvaluateOnNewDocument", {
        "source": f"if (location.href === {json.dumps(url)}) debugger;",
    })
    goto_retrying_network_change(window, url, wait_until="commit")
    frame = _next_pause(page, pauses)["callFrames"][0]["callFrameId"]
    for method in ("getItem", "setItem"):
        function = debugger.send("Debugger.evaluateOnCallFrame", {
            "callFrameId": frame, "expression": f"Storage.prototype.{method}",
        })
        debugger.send("Debugger.setBreakpointOnFunctionCall", {
            "objectId": function["result"]["objectId"], "condition": _NAMES_REPORT_ITEM,
        })
    debugger.send("Page.removeScriptToEvaluateOnNewDocument", first)
    debugger.send("Debugger.resume")
    stop = _next_pause(page, pauses)["callFrames"][0]["callFrameId"]
    name: str = debugger.send("Debugger.evaluateOnCallFrame", {
        "callFrameId": stop, "expression": "window._stalledAt",
    })["result"]["value"]
    return window, debugger, name, sent


def _drop_reports_while_another_window_stores(context, page, other, tab_id: str) -> int:
    """Have *page* drop the pending reports of *tab_id* while *other* stores reports of its own.

    The chat's next run starts in *page* (its ``clear``), which drops
    the reports.  The debugger (a breakpoint on ``Storage.setItem``)
    stops that window at each write of a single report's item, as a
    slow machine might anywhere; before the window gets on, the task
    *other* shows ends, having written one more report, which that
    window stores if there is room.  No function of the page is
    replaced.  Returns how often *page* was stopped.
    """
    debugger = context.new_cdp_session(page)
    pauses: list[dict] = []
    debugger.on("Debugger.paused", lambda pause: pauses.append(pause))
    debugger.send("Debugger.enable")
    set_item = debugger.send("Runtime.evaluate", {"expression": "Storage.prototype.setItem"})
    debugger.send("Debugger.setBreakpointOnFunctionCall", {
        "objectId": set_item["result"]["objectId"], "condition": _NAMES_REPORT_ITEM,
    })
    # From a timer: the call that starts it returns, though the window is stopped on the way.
    page.evaluate(
        """(data) => {
             setTimeout(() => {
               window.dispatchEvent(new MessageEvent('message', {data}));
               window._dropped = true;
             }, 0);
           }""",
        {"type": "clear", "tabId": tab_id},
    )
    stops = 0
    for _ in range(600):
        if pauses:
            pauses.pop(0)
            _deliver(
                other,
                *_write_events(tab_id, f"reports/meanwhile-{stops}.md", "# Draft\n"),
                {"type": "task_done", "success": True, "tabId": tab_id},
            )
            stops += 1
            debugger.send("Debugger.resume")
        elif debugger.send("Runtime.evaluate", {"expression": "window._dropped === true"})[
            "result"
        ]["value"]:
            debugger.send("Debugger.disable")
            return stops
        else:
            other.wait_for_timeout(100)
    raise AssertionError("the window did not finish dropping its reports")


def _content_paths(page) -> list[str]:
    """The paths of the content tabs open in the page."""
    paths: list[str] = page.evaluate(
        "() => window._testApi.openTabs().filter(t => t.isContentTab).map(t => t.contentPath)",
    )
    return paths


def _older_checkout_copy(harness, name: str) -> Path:
    """Leave an hour-old report *name* in the harness repository; return its path."""
    older = Path(harness.work_dir) / "reports" / name
    older.parent.mkdir(exist_ok=True)
    older.write_text("# Older Copy\n", encoding="utf-8", newline="\n")
    hour_ago = time.time() - 3600
    os.utime(older, (hour_ago, hour_ago))
    return older


def _worktree_report(harness, tab_id: str, name: str, text: str) -> Path:
    """Write the report *name* into a worktree the daemon announces for *tab_id*."""
    report = Path(harness.work_dir) / ".kiss-worktrees" / f"kiss_wt-{name}" / "reports" / name
    report.parent.mkdir(parents=True)
    report.write_text(text, encoding="utf-8", newline="\n")
    harness.server._printer.broadcast({
        "type": "worktree_created", "branch": "kiss/wt-e2e",
        "worktreeDir": str(report.parents[1]), "tabId": tab_id,
    })
    return report


def _worktree_result(harness, tab_id: str, action: str, worktree: Path) -> None:
    """Remove *worktree* and have the daemon announce its merge or discard to *tab_id*."""
    shutil.rmtree(worktree)
    harness.server._printer.broadcast({
        "type": "worktree_result", "tabId": tab_id, "action": action, "success": True,
        "message": f"{action}: branch 'kiss/wt-e2e'.", "worktreeDir": str(worktree),
    })


class TestReportOpensAfterMerge:
    """Browser E2E: when a finished task's report reaches the editor."""

    def test_report_opens_as_the_file_on_disk(self, browser, harness) -> None:
        """A report of a run without a worktree opens when the task ends:
        the file on disk (the ``Write`` call held only its first chunk),
        with the Edit source toggle of an ordinary file."""
        context, page, sent = _open_page(browser, harness)
        try:
            _start_run(harness, page)
            report = Path(harness.work_dir) / "reports" / "summary.md"
            report.parent.mkdir(exist_ok=True)
            report.write_text("# Final Report\n\nall chunks\n", encoding="utf-8", newline="\n")
            _finish_task_writing(page, str(report), "# Fin")
            page.wait_for_selector(_FRAME, timeout=30000)
            page.frame_locator(_FRAME).locator("h1", has_text="Final Report").wait_for(
                timeout=30000,
            )
            _uncover_pane(page)
            assert page.locator(_MODE_BTN).inner_text() == "Edit source"
            (request,) = _report_requests(page, sent, 1)
            assert request["path"] == str(report)
            assert request["report"] is True
        finally:
            context.close()

    def test_report_in_a_worktree_opens_only_after_the_merge(self, browser, harness) -> None:
        """A report in an unmerged worktree stays closed when the task
        ends; the merge's result opens the repository's copy of it.
        The task named the checkout's path, which its ``Write``
        redirects into the worktree: the checkout's older copy is never
        shown.  What the merge brings is shown whatever its date: a
        merge leaves a file whose bytes did not change as old as it
        was.  A report a sub-agent of the task wrote in that worktree
        is the task's as well: nothing opens when the sub-agent ends,
        it is asked for under the task's tab when the task ends, and
        the merge opens it too."""
        context, page, sent = _open_page(browser, harness)
        try:
            tab_id = _start_run(harness, page)
            merged = _older_checkout_copy(harness, "merged.md")
            sub = _older_checkout_copy(harness, "sub.md")
            report = _worktree_report(harness, tab_id, "merged.md", "# Unmerged Draft\n")
            report.with_name("sub.md").write_text("# Sub Draft\n", encoding="utf-8")
            _sub_agent_writes(page, tab_id, "reports/sub.md", "# Sub Draft\n")
            # The page keeps what it has asked for; so far that is nothing.
            assert _stored_reports(page) == []
            _finish_task_writing(page, str(merged), "# Unmerged Draft\n")
            asked = _report_requests(page, sent, 2)
            assert [(request["tabId"], request["path"]) for request in asked] == [
                (tab_id, "reports/sub.md"), (tab_id, str(merged)),
            ]
            assert len(_stored_reports(page)) == 2
            # Time for the daemon's replies: the reports must stay closed.
            page.wait_for_timeout(1000)
            assert page.locator(_CONTENT_TAB).count() == 0

            # The merge: the repository gets the files, the worktree goes.
            merged.write_text("# Merged Report\n", encoding="utf-8", newline="\n")
            sub.write_text("# Sub Report\n", encoding="utf-8", newline="\n")
            hour_ago = time.time() - 3600
            os.utime(merged, (hour_ago, hour_ago))
            _worktree_result(harness, tab_id, "merge", report.parents[1])
            page.wait_for_function(
                "() => window._testApi.openTabs().filter(t => t.isContentTab).length === 2",
                timeout=30000,
            )
            # The one on screen (the last the daemon answered) shows what the merge brought.
            page.frame_locator(_FRAME).locator("h1", has_text="Report").wait_for(timeout=30000)
            assert sorted(_content_paths(page)) == sorted([str(merged), str(sub)])
            assert len(_report_requests(page, sent, 4)) == 4
            assert _stored_reports(page) == []
        finally:
            context.close()

    def test_report_waiting_for_its_merge_survives_a_reload(self, browser, harness) -> None:
        """A page reloaded while a report waits for its merge shows the
        report after the merge all the same.  The reloaded page gets
        the chat's tab back from the daemon and asks for the report
        again: nothing opens before the merge, not even the checkout's
        older copy.  The page that loads after one of an earlier
        version does the same: that version kept all pending reports in
        one localStorage item, ``kiss-pending-reports``, which the page
        reads them from, also when the browser's storage quota is used
        up.  A report of that item that has been shown is not asked for
        again."""
        _seed(harness)
        context, page, sent = _open_page(browser, harness)
        try:
            _open_task(page)
            tab_id = _start_run(harness, page)
            merged = _older_checkout_copy(harness, "reloaded.md")
            report = _worktree_report(harness, tab_id, "reloaded.md", "# Unmerged Draft\n")
            _finish_task_writing(page, str(merged), "# Unmerged Draft\n")
            _report_requests(page, sent, 1)
            page.wait_for_timeout(1000)

            page.reload()
            again = _report_requests(page, sent, 2)[1]
            assert (again["tabId"], again["path"]) == (tab_id, str(merged))
            page.wait_for_timeout(1000)
            assert page.locator(_CONTENT_TAB).count() == 0

            # The storage as the page of an earlier version left it, with a
            # second report, which the merge will not bring, and full.
            waiting = dict(again, path=str(merged.with_name("not-merged.md")))
            page.evaluate(
                f"""(reports) => {{
                     {_STORED_REPORTS}.forEach(name => localStorage.removeItem(name));
                     localStorage.setItem('kiss-pending-reports', JSON.stringify(reports));
                   }}""",
                {tab_id: [
                    {name: rep[name] for name in ("path", "workDir", "since")}
                    for rep in (again, waiting)
                ]},
            )
            _fill_storage(page)
            page.reload()
            upgraded = _report_requests(page, sent, 4)[2:]
            assert [(request["path"], request["since"]) for request in upgraded] == [
                (again["path"], again["since"]), (waiting["path"], again["since"]),
            ]

            merged.write_text("# Merged Report\n", encoding="utf-8", newline="\n")
            _worktree_result(harness, tab_id, "merge", report.parents[1])
            page.wait_for_selector(_FRAME, timeout=30000)
            page.frame_locator(_FRAME).locator("h1", has_text="Merged Report").wait_for(
                timeout=30000,
            )
            assert _content_paths(page) == [str(merged)]

            # The next page asks for the one that is not shown; the shown
            # one would come before it.
            assert _first_report_asked_after_reload(page, sent) == waiting["path"]
        finally:
            context.close()

    def test_window_stalled_while_loading_brings_no_dropped_report_back(
        self, browser, harness,
    ) -> None:
        """A window that is slow to load brings back no report another
        window dropped meanwhile.  The report waits in the item of an
        earlier version, ``kiss-pending-reports``; another one has an
        item of its own.  The debugger stops a second window while it
        loads, at its first access to the item of a single report: it
        has read the old item by then.  The first window reloads, asks
        for both reports and drops them when the chat's next run
        starts; then the stopped window gets on with what it read, and
        is closed before it hears of that run.  The run writes a report
        of its own, which stays pending: the page that loads next asks
        for it, and not for a dropped one, which would come before
        it."""
        _seed(harness, "write a report again", "written again")
        context, page, sent = _open_page(browser, harness)
        try:
            _open_task(page, "write a report again", "written again")
            page.evaluate(
                """([tab, workDir, since]) => {
                     localStorage.setItem('kiss-pending-reports', JSON.stringify(
                       {[tab]: [{path: 'reports/dropped.md', workDir, since}]},
                     ));
                     localStorage.setItem(
                       'kiss-pending-report:'
                         + JSON.stringify([tab, workDir, 'reports/kept.md', since])
                         + crypto.randomUUID(),
                       String(Date.now()),
                     );
                   }""",
                [_active_tab(page), str(harness.work_dir), int(time.time() * 1000)],
            )
            stalled, debugger, _, _ = _stall_at_report_item(
                context, page, harness.base_url + "/",
            )
            page.reload()
            assert [request["path"] for request in _report_requests(page, sent, 2)] == [
                "reports/dropped.md", "reports/kept.md",
            ]
            _start_run(harness, page)
            # The script that was stopped runs to its end before the window closes.
            debugger.send("Debugger.resume")
            stalled.close()

            _finish_task_writing(page, "reports/waiting.md", "# Draft\n")
            _report_requests(page, sent, 3)
            assert _first_report_asked_after_reload(page, sent) == "reports/waiting.md"
        finally:
            context.close()

    def test_window_stalled_while_loading_holds_the_reports_that_wait(
        self, browser, harness,
    ) -> None:
        """A window that is slow to load holds every report that still
        waits.  Three reports of the chat have items of their own.  The
        debugger stops a second window at the first of them it reads;
        meanwhile another window removes that item, as it does for a
        report it shows, and every item that is no report's.  The
        stopped window gets on and asks the daemon for the other two."""
        _seed(harness, "write three reports", "all three written")
        context, page, sent = _open_page(browser, harness)
        try:
            _open_task(page, "write three reports", "all three written")
            paths = [f"reports/still-{i}.md" for i in range(3)]
            page.evaluate(
                """([tab, workDir, since, paths]) => paths.forEach(path => localStorage.setItem(
                     'kiss-pending-report:' + JSON.stringify([tab, workDir, path, since])
                       + crypto.randomUUID(),
                     String(Date.now()),
                   ))""",
                [_active_tab(page), str(harness.work_dir), int(time.time() * 1000), paths],
            )
            stalled, debugger, shown, asked = _stall_at_report_item(
                context, page, harness.base_url + "/",
            )
            page.evaluate(
                """(shown) => Object.keys(localStorage).forEach(name => {
                     if (name === shown || !name.startsWith('kiss-pending-report:'))
                       localStorage.removeItem(name);
                   })""",
                shown,
            )
            debugger.send("Debugger.disable")
            assert sorted(
                request["path"] for request in _report_requests(stalled, asked, 2)
            ) == [path for path in paths if json.dumps(path) not in shown]
        finally:
            context.close()

    def test_report_dropped_in_a_full_storage_stays_dropped(self, browser, harness) -> None:
        """A report that is dropped while storage is full stays dropped,
        whatever the other windows store meanwhile.  Three windows show
        the chat.  Its task writes a report; the first window hears of
        the task's end and stores the report, the third one does not
        yet.  Then storage fills up to the browser's quota, and the
        chat's next run starts in the first window, which drops the
        report: the debugger stops that window at each write of an
        item of the report, and each time a task ends in the second
        window that wrote a report of its own, which takes the room
        there is.  Afterwards there is room again; the third window
        hears of the first task's end and stores the dropped report
        again, and then the report of a later task.  A page that loads
        now asks for that last one, and before it for none that was
        dropped."""
        _seed(harness, "write a report at the quota", "written at the quota")
        context, page, sent = _open_page(browser, harness)
        try:
            _open_task(page, "write a report at the quota", "written at the quota")
            tab_id = _start_run(harness, page)
            other, late = context.new_page(), context.new_page()
            for window in (other, late):
                goto_retrying_network_change(window, harness.base_url + "/")
                _wait_ready(window)
                window.wait_for_function(
                    "(tabId) => window._testApi.openTabs().some(t => t.id === tabId)",
                    arg=tab_id, timeout=30000,
                )
            dropped = "reports/dropped-at-the-quota.md"
            written = _write_events(tab_id, dropped, "# Draft\n")
            ended = {"type": "task_done", "success": True, "tabId": tab_id}
            _deliver(page, *written, ended)
            _deliver(late, *written)
            _report_requests(page, sent, 1)
            assert [dropped in name for name in _stored_reports(page)] == [True]
            _fill_storage(page)
            assert page.evaluate(
                """() => {
                     try {
                       localStorage.setItem('one-more', 'x');
                     } catch (full) {
                       return full.name;
                     }
                     return 'it fits';
                   }""",
            ) == "QuotaExceededError"

            assert _drop_reports_while_another_window_stores(context, page, other, tab_id) > 0
            other.evaluate("() => localStorage.removeItem('filler')")
            last = "reports/of-a-later-task.md"
            _deliver(late, ended, *_write_events(tab_id, last, "# Draft\n"), ended)

            loaded = context.new_page()
            asked: list[dict] = []
            loaded.on("websocket", lambda socket: socket.on(
                "framesent", lambda payload: asked.append(json.loads(payload)),
            ))
            goto_retrying_network_change(loaded, harness.base_url + "/")
            paths: list[str] = []
            for _ in range(300):
                paths = [request["path"] for request in _report_requests(loaded, asked, 1)]
                if last in paths:
                    break
                loaded.wait_for_timeout(100)
            assert last in paths
            assert dropped not in paths, "the dropped report is asked for again"
        finally:
            context.close()

    def test_two_windows_keep_the_reports_each_other_left_waiting(self, browser, harness) -> None:
        """Two windows of one browser keep every report that waits for
        its merge.  Both show the chat and end tasks of it at the same
        moment, each writing reports of its own (the daemon has no file
        for any); a page reloaded afterwards asks the daemon for the
        reports of both windows."""
        count = 30
        # A task of its own: the daemon is the module's, and keeps the other tests' tasks.
        _seed(harness, "write many reports", "all written")
        context, page, sent = _open_page(browser, harness)
        try:
            _open_task(page, "write many reports", "all written")
            tab_id = _active_tab(page)
            other = context.new_page()
            goto_retrying_network_change(other, harness.base_url + "/")
            _wait_ready(other)
            other.wait_for_function(
                "(tabId) => window._testApi.openTabs().some(t => t.id === tabId)",
                arg=tab_id, timeout=30000,
            )
            start_ms = int(time.time() * 1000) + 1000
            paths: set[str] = set()
            for name, window in (("first", page), ("second", other)):
                events: list[dict] = []
                for i in range(count):
                    paths.add(f"reports/{name}-{i}.md")
                    events += _write_events(tab_id, f"reports/{name}-{i}.md", "# Draft\n")
                    events.append({"type": "task_done", "success": True, "tabId": tab_id})
                _deliver_at(window, start_ms, events)
            for window in (page, other):
                window.evaluate("() => window._delivered")
            # What the other window stored reaches this one.
            page.wait_for_function(
                f"(count) => {_STORED_REPORTS}.length === count", arg=2 * count, timeout=30000,
            )
            before = len(_report_requests(page, sent, count))

            page.reload()
            asked = _report_requests(page, sent, before + 2 * count)[before:]
            assert sorted(request["path"] for request in asked) == sorted(paths)
        finally:
            context.close()

    def test_report_of_a_discarded_worktree_never_opens(self, browser, harness) -> None:
        """Discarding the worktree drops its report: nothing opens, not
        even the older copy of that report the repository already held."""
        context, page, sent = _open_page(browser, harness)
        try:
            tab_id = _start_run(harness, page)
            older = _older_checkout_copy(harness, "dropped.md")
            report = _worktree_report(harness, tab_id, "dropped.md", "# Dropped Draft\n")
            _finish_task_writing(page, str(older), "# Dropped Draft\n")
            _report_requests(page, sent, 1)
            page.wait_for_timeout(1000)
            assert len(_stored_reports(page)) == 1

            _worktree_result(harness, tab_id, "discard", report.parents[1])
            # The discard is the daemon's broadcast; the merge result below is handed to the
            # page itself and could get there first, and the page would ask again.  A page
            # that has heard of the discard keeps the report no longer.
            page.wait_for_function(f"() => {_STORED_REPORTS}.length === 0", timeout=30000)
            # A later merge result of the tab finds nothing left to show.
            _deliver(page, {
                "type": "worktree_result", "tabId": tab_id, "action": "merge",
                "success": True, "message": "Merged.",
            })
            page.wait_for_timeout(1000)
            assert page.locator(_CONTENT_TAB).count() == 0
            assert len(_report_requests(page, sent, 1)) == 1
        finally:
            context.close()
