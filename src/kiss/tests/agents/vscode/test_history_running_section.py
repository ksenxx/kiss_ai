# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Running history sections through real SQLite, daemon tasks, WSS and Chromium.

The live task uses a real model and Bash tool, waiting for a release file. No
worker, transport, clock, model or browser API is replaced. Completed rows and
historical dates are fixture data in the isolated test database.

The daemon writes finite timestamps and chat IDs. Missing/invalid timestamp and
ID compatibility branches require legacy or malformed protocol payloads; the
existing renderer regressions cover them without replacing the real daemon here.
Set HISTORY_JS_COVERAGE to a directory under tmp/ to collect Chromium coverage.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import expect, sync_playwright

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.daemon_client import run
from kiss.tests.agents.vscode.test_content_tab_file_links import _open_page
from kiss.tests.server.test_content_tab_file_links import _ServerHarness


def _seed(harness: _ServerHarness, title: str, chat: str, timestamp: float) -> str:
    """Persist a completed task with a historical launch time."""
    task_id, _ = persistence._add_task(
        title,
        chat,
        {"work_dir": str(harness.work_dir)},
    )
    persistence._save_task_result("done", task_id=task_id)
    _date(task_id, timestamp)
    return task_id


def _date(task_id: str, timestamp: float) -> None:
    """Set fixture chronology without replacing the real clock."""
    with persistence._rw_lock.write_lock():
        persistence._get_db().execute(
            "UPDATE task_history SET timestamp = ? WHERE id = ?",
            (timestamp, task_id),
        )


def _record_error(errors: list[str], error: Exception) -> None:
    """Keep browser errors without replacing any application behavior."""
    errors.append(str(error))


def _shape(page) -> list[str]:
    """Describe visible sections and top-level rows/chats in DOM order."""
    return cast(
        list[str],
        page.locator("#history-list > :visible").evaluate_all(
            """els => els.map(el => {
            if (el.classList.contains('history-day-sep')) return el.textContent;
            if (el.classList.contains('history-chat-group')) return el.dataset.chatId;
            return el.querySelector('.sidebar-item-text')?.textContent || el.textContent;
        })""",
        ),
    )


def _load_more(page, count: int) -> None:
    """Keep scrolling through concurrent real status refreshes until the page arrives."""
    page.wait_for_function(
        """count => {
            const list = document.getElementById('history-list');
            if (list.querySelectorAll('.sidebar-item').length === count) return true;
            list.scrollTop = list.scrollHeight;
            list.dispatchEvent(new Event('scroll'));
            return false;
        }""",
        arg=count,
        timeout=10000,
    )


@pytest.mark.parametrize("flat", [False, True], ids=["grouped", "flat"])
def test_running_history_section_lifecycle(flat: bool) -> None:
    """An old live task is pinned, survives paging, and returns on completion."""
    harness = _ServerHarness()
    release = harness.work_dir / "release-history-task"
    started = threading.Event()
    cancel = threading.Event()
    today = dt.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    yesterday = (today - dt.timedelta(days=1)).timestamp()
    prompt = (
        "section-live: Use Bash to run this command once, then finish successfully: "
        f"`while [ ! -f '{release}' ]; do sleep 0.2; done`. "
        "Do not create the release file. Do not do any other work."
    )
    pool = ThreadPoolExecutor(max_workers=1)
    task = pool.submit(
        run,
        prompt,
        chat_id="running-section-chat",
        model="gpt-5.4-mini",
        work_dir=str(harness.work_dir),
        use_worktree=False,
        auto_commit=False,
        auto_classify=False,
        use_memory=False,
        use_web_tools=False,
        is_parallel=False,
        tool_profile="bash",
        max_budget=2,
        endpoint_file=Path(harness.tmpdir) / "sorcar-local.json",
        running=started,
        cancel=cancel,
        # The task must outlive every UI step below: a parallel full-suite
        # run stretches them well past 180 s, and a task stopped early
        # drops out of the Running section before the test releases it.
        timeout=600,
        stop_on_timeout=True,
    )
    try:
        assert started.wait(60), "the real daemon task must start"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, args=["--ignore-certificate-errors"]
            )
            context, page, _ = _open_page(browser, harness)
            coverage_dir = os.environ.get("HISTORY_JS_COVERAGE")
            profiler = context.new_cdp_session(page) if coverage_dir else None
            if profiler:
                profiler.send("Profiler.enable")
                profiler.send(
                    "Profiler.startPreciseCoverage", {"callCount": True, "detailed": True},
                )
            errors: list[str] = []
            page.on("pageerror", partial(_record_error, errors))
            try:
                # Wait for the task runner's actual persisted history row.
                # This test covers all history, not the current workspace filter.
                page.click("#history-filters-toggle")
                page.locator("label:has(#hf-workspace)").click()
                expect(page.locator("#hf-workspace")).not_to_be_checked()
                page.click("#history-filters-toggle")
                live = page.locator("#history-list .sidebar-item").filter(has_text=prompt)
                expect(live).to_have_count(1, timeout=60000)
                live_id = next(
                    str(row["id"]) for row in persistence._load_history() if row["task"] == prompt
                )
                _date(live_id, yesterday)
                _seed(
                    harness,
                    "section-live companion",
                    "running-section-chat",
                    today.timestamp() - 60,
                )
                _seed(harness, "section-live newest", "newest-chat", today.timestamp())
                for i in range(52):
                    _seed(
                        harness, f"other completed {i}", "other-chat", today.timestamp() - 120 - i
                    )
                _seed(harness, "older completed", "older-chat", yesterday - 60)
                # This is the production refresh signal for changed persisted history.
                harness.server._printer.broadcast({"type": "tasks_updated"})
                expect(page.locator("#history-list .sidebar-item")).to_have_count(50)
                if flat:
                    page.click("#history-view-toggle")
                expect(page.locator("#history-list > .history-day-sep").first).to_have_text(
                    "Running"
                )
                expect(live.locator(".status-spinner")).to_have_count(1)
                assert _shape(page)[1] == (prompt if flat else "running-section-chat")
                assert "Today" in _shape(page)

                # Expand the completed chat so grouped mode has enough visible
                # rows to scroll and request its next real page.
                if not flat:
                    page.locator('[data-chat-id="other-chat"] .history-chat-header').click()
                _load_more(page, 56)
                ids = page.locator("#history-list .ids-copy-task").evaluate_all(
                    "els => els.map(el => el.parentElement.textContent)",
                )
                assert len(ids) == len(set(ids)) == 56
                assert _shape(page)[0] == "Running"
                # Visibility/midnight relabeling must not turn Running into a date.
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                assert _shape(page)[0] == "Running"

                # Every layout switch keeps the pinned section and date boundary.
                page.click("#history-view-toggle")
                assert _shape(page)[0] == "Running"
                page.click("#history-view-toggle")
                assert _shape(page)[0] == "Running"

                # Search/tag predicates must apply before running-first paging.
                # More than 50 matches means client-only sorting cannot pass.
                with persistence._rw_lock.write_lock():
                    persistence._get_db().execute(
                        "UPDATE task_history SET tags = 'testing' WHERE task != 'older completed'",
                    )
                page.fill("#history-search", "e")
                expect(page.locator("#history-list .sidebar-item")).to_have_count(50)
                expect(live).to_have_count(1)
                page.click("#history-filters-toggle")
                page.select_option("#hf-tag", "testing")
                expect(page.locator("#history-list > .history-day-sep").first).to_have_text(
                    "Running"
                )
                expect(live).to_be_visible()
                _load_more(page, 55)
                expect(
                    page.locator("#history-list .sidebar-item").filter(has_text="older completed")
                ).to_have_count(0)
                ids = page.locator("#history-list .ids-copy-task").evaluate_all(
                    "els => els.map(el => el.parentElement.textContent)",
                )
                assert len(ids) == len(set(ids)) == 55
                page.select_option("#hf-tag", "")
                page.click("#history-filters-toggle")

                # The newer companion shares the running chat: its date and row
                # order must remain correct when the prioritized row arrived first.
                page.fill("#history-search", "section-live")
                expect(page.locator("#history-list .sidebar-item")).to_have_count(3)
                assert _shape(page) == (
                    ["Running", prompt, "Today", "section-live newest", "section-live companion"]
                    if flat
                    else ["Running", "running-section-chat", "Today", "newest-chat"]
                )
                if not flat:
                    titles = page.locator(
                        '[data-chat-id="running-section-chat"] .sidebar-item-text',
                    ).all_text_contents()
                    assert titles == ["section-live companion", prompt]

                # Empty filters hide the Running label as well as dated sections.
                page.locator("#history-filters-toggle").click()
                for selector in ("#hf-running", "#hf-completed", "#hf-errors"):
                    page.locator(f"label:has({selector})").click()
                    expect(page.locator(selector)).not_to_be_checked()
                expect(page.locator("#history-list .sidebar-empty-filter")).to_be_visible()
                expect(page.locator("#history-list > .history-day-sep:visible")).to_have_count(0)
                for selector in ("#hf-running", "#hf-completed", "#hf-errors"):
                    page.locator(f"label:has({selector})").click()
                    expect(page.locator(selector)).to_be_checked()
                assert _shape(page)[0] == "Running"

                # Natural task-runner completion must update the already-open
                # history, without reload, manual refresh or injected UI messages.
                release.touch()
                result = task.result(timeout=120)
                assert result.success, result.text
                expect(
                    page.locator("#history-list > .history-day-sep").filter(has_text="Running")
                ).to_have_count(0)
                expect(live).to_be_visible()
                expect(page.locator("#history-list .sidebar-item")).to_have_count(3)
                assert _shape(page) == (
                    ["Today", "section-live newest", "section-live companion", "Yesterday", prompt]
                    if flat
                    else ["Today", "newest-chat", "running-section-chat"]
                )
                assert not errors, json.dumps([str(error) for error in errors])
            finally:
                if profiler and coverage_dir:
                    output = Path(coverage_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    (output / ("flat.json" if flat else "grouped.json")).write_text(
                        json.dumps(profiler.send("Profiler.takePreciseCoverage")),
                    )
                context.close()
                browser.close()
    finally:
        release.touch()
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)
        harness.stop()
