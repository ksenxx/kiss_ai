# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The CHATS panel loads the rows that fill it without a "Loading..." row.

``webapp/test/historyFill.test.js`` covers the requests the list sends as
its panel fills; JSDOM has no layout, so the heights there are stand-ins.
Here the shipped page runs in a headless Chromium with the real
stylesheets: a first page whose tasks do not fill the docked panel pulls
in more rows with no "Loading..." row in sight, a refresh asks once
for every row the list holds, the rows of a refresh that come in two
pages replace the list's when both are in, and a list the user scrolls
asks for its next rows while less than a panel's height of tasks is left
under the view.
"""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Page, sync_playwright

from kiss.tests.webapp.test_history_failed_red_cross import _open_remote_page, _sample_sessions

# What the list shows and holds, every getHistory the page has sent, and
# how many panels' heights of rows lie under the view.  The "Loading..."
# row counts when it is drawn: it has a box.
_STATE_JS = """() => {
  const list = document.getElementById('history-list');
  const loader = document.getElementById('history-loader');
  return {
    asks: window.__postedMessages.filter(m => m.type === 'getHistory')
      .map(m => [m.offset, m.limit]),
    generation: window.__postedMessages.filter(m => m.type === 'getHistory')
      .at(-1).generation,
    rows: list.querySelectorAll('.sidebar-item').length,
    top: list.querySelector('.sidebar-item')?.dataset.tooltip,
    loader: !!loader && loader.getClientRects().length > 0,
    filled: list.scrollHeight > list.clientHeight,
    under: (list.scrollHeight - list.scrollTop - list.clientHeight) / list.clientHeight,
  };
}"""

# Scroll the list until half a panel's height of rows is left under the
# view, and wait until the page has heard of it.
_SCROLL_JS = """async () => {
  const list = document.getElementById('history-list');
  const heard = new Promise(resolve => list.addEventListener('scroll', resolve, {once: true}));
  list.scrollTop = list.scrollHeight - 1.5 * list.clientHeight;
  await heard;
}"""


def _rows(first: int, count: int, cron: bool) -> list[dict[str, Any]]:
    """Return *count* finished tasks, each in a chat of its own, the newest first."""
    finished = _sample_sessions()[2]
    return [
        dict(finished, id=f"chat-{index}", task_id=index, title=f"task {index}",
             timestamp=1_700_000_000 - index, is_cron=cron)
        for index in range(first, first + count)
    ]


def _deliver(page: Page, event: dict[str, Any]) -> dict[str, Any]:
    """Hand the page *event* as its daemon would and return the list's state."""
    page.evaluate("event => window.__post(event)", event)
    state: dict[str, Any] = page.evaluate(_STATE_JS)
    return state


def test_a_docked_panel_fills_without_a_loading_row_and_refreshes_in_one_request() -> None:
    """Four tasks among 46 hidden cron tasks leave the panel short: the rest comes unannounced."""
    first = _rows(0, 4, cron=False) + _rows(4, 46, cron=True)
    second = _rows(50, 50, cron=False)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            _context, page = _open_remote_page(browser, 1200, 700)
            state: dict[str, Any] = page.evaluate(_STATE_JS)
            assert state["asks"] == [[0, 50]]
            page_of = {"type": "history", "generation": state["generation"]}

            state = _deliver(page, {**page_of, "sessions": first, "offset": 0, "limit": 50})
            assert (state["rows"], state["filled"], state["loader"]) == (50, False, False)
            assert state["asks"] == [[0, 50], [50, 50]]
            state = _deliver(page, {**page_of, "sessions": second, "offset": 50, "limit": 50})
            assert (state["rows"], state["filled"], state["loader"]) == (100, True, False)
            assert state["asks"] == [[0, 50], [50, 50]]

            # Task news: one request for the rows asked for so far, and nothing after its answer.
            state = _deliver(page, {"type": "tasks_updated"})
            assert state["asks"][2:] == [[0, 100]]
            state = _deliver(page, {
                "type": "history", "generation": state["generation"],
                "sessions": first + second, "offset": 0, "limit": 100,
            })
            assert (state["rows"], state["filled"], state["loader"]) == (100, True, False)
            assert len(state["asks"]) == 3

            # A new task, and a daemon that grants 60 rows a page: the list stays as it is,
            # full, until the rows of the second page are in too.
            now = [dict(first[0], id="chat-new", task_id=1000, title="new task")] + first + second
            state = _deliver(page, {"type": "tasks_updated"})
            page_of = {"type": "history", "generation": state["generation"]}
            state = _deliver(page, {**page_of, "sessions": now[:60], "offset": 0, "limit": 60})
            assert state["asks"][3:] == [[0, 100], [60, 40]]
            assert (state["rows"], state["filled"], state["loader"]) == (100, True, False)
            assert state["top"] == "task 0"
            state = _deliver(page, {**page_of, "sessions": now[60:100], "offset": 60, "limit": 40})
            assert (state["rows"], state["filled"], state["loader"]) == (100, True, False)
            assert state["top"] == "new task"
            assert len(state["asks"]) == 5

            # The user scrolls.  With half a panel's height of tasks left under the view,
            # well before the list's end, the next rows are asked for under a "Loading..."
            # row: a page at least.  A page that shows no task (cron tasks only) is followed
            # by the next, and the asking stops with a panel's height of tasks under the view.
            assert state["under"] > 1
            page.evaluate(_SCROLL_JS)
            state = page.evaluate(_STATE_JS)
            offset, limit = state["asks"][-1]
            assert (len(state["asks"]), offset, state["loader"]) == (6, 100, True)
            assert limit >= 50 and 0.4 < state["under"] < 0.7, state
            page_of = {"type": "history", "generation": state["generation"]}
            state = _deliver(page, {
                **page_of, "sessions": _rows(100, limit, cron=True), "offset": 100, "limit": limit,
            })
            offset, more = state["asks"][-1]
            assert (len(state["asks"]), offset, state["loader"]) == (7, 100 + limit, True)
            state = _deliver(page, {
                **page_of, "sessions": _rows(offset, more, cron=False), "offset": offset,
                "limit": more,
            })
            assert (state["rows"], state["loader"]) == (100 + limit + more, False)
            assert len(state["asks"]) == 7 and state["under"] >= 1, state
        finally:
            browser.close()
