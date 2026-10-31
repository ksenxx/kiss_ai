# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The page fetches the composer history once and refreshes it by delta.

The ArrowUp history (``histCache`` in ``media/main.js``) used to be
fetched whole at every ``ready`` and every ``tasks_updated``: the 500
newest distinct texts, hundreds of kilobytes on a slow network, in
front of the CHATS page and the transcript the user is waiting for on
the same socket.  Now the page remembers the newest use the daemon
reported and asks for the texts used since (``getInputHistory``
``since``); a reconnect's ``ready`` names it too
(``inputHistorySince``), so only the first reply of a page is the
whole list.

Real daemon (``RemoteAccessServer``) with a real history database,
real Chromium (Playwright); the frames both ways are recorded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest
from playwright.sync_api import sync_playwright

from kiss.agents.sorcar import persistence
from kiss.tests.agents.sorcar.test_steer_inputs import _set_task_timestamp
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _wait_ready


@pytest.fixture(scope="module")
def harness():
    """One real daemon for the module."""
    h = _ServerHarness()
    yield h
    h.stop()


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        yield b
        b.close()


def _open_page(browser, harness):
    """Open the webapp and record every WebSocket frame sent and received."""
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    sent: list[dict] = []
    received: list[dict] = []

    def _record(frames: list[dict]) -> Callable[[object], None]:
        def _on_frame(payload) -> None:
            try:
                frames.append(json.loads(payload))
            except Exception:
                pass

        return _on_frame

    def _on_ws(ws) -> None:
        ws.on("framesent", _record(sent))
        ws.on("framereceived", _record(received))

    page.on("websocket", _on_ws)
    goto_retrying_network_change(page, harness.base_url + "/")
    _wait_ready(page)
    return context, page, sent, received


def _seed(harness: _ServerHarness, text: str, ts: float) -> str:
    task_id, _chat = persistence._add_task(text, "chat-hist", {"work_dir": str(harness.work_dir)})
    _set_task_timestamp(task_id, ts)
    return task_id


def _wait(page, frames: list[dict], pick: Callable[[dict], bool], what: str) -> dict:
    """Return the first recorded frame *pick* accepts, polling for up to 30 s."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for frame in frames:
            if pick(frame):
                return frame
        page.wait_for_timeout(100)
    raise AssertionError(f"no {what} within 30 s; got {[f.get('type') for f in frames]}")


def _is_input_history(frame: dict) -> bool:
    return frame.get("type") == "inputHistory"


def _deliver(page, data: dict) -> None:
    """Deliver a daemon message to main.js the way the shim does."""
    page.evaluate("data => window.dispatchEvent(new MessageEvent('message', {data}))", data)


class TestIncrementalInputHistory:
    """Browser E2E: one whole list per page, deltas after that."""

    def test_whole_list_once_then_deltas(self, browser, harness) -> None:
        now = time.time()
        _seed(harness, "older prompt", now - 200)
        _seed(harness, "newer prompt", now - 100)
        context, page, sent, received = _open_page(browser, harness)
        try:
            # The ready's fan-out: the whole list, newest first, with
            # the time of its newest use; the page had nothing yet.
            whole = _wait(page, received, _is_input_history, "inputHistory")
            assert whole["tasks"] == ["newer prompt", "older prompt"]
            assert whole["newest"] == pytest.approx(now - 100)
            assert "since" not in whole
            readies = [f for f in sent if f.get("type") == "ready"]
            assert readies and readies[0]["inputHistorySince"] == 0

            # The ready's tasks_updated nudge: the page asks for what
            # came since and gets an empty delta.
            empty = _wait(
                page, received,
                lambda f: _is_input_history(f) and "since" in f,
                "delta inputHistory",
            )
            assert empty["tasks"] == []
            assert empty["since"] == pytest.approx(now - 100)
            asked = [f for f in sent if f.get("type") == "getInputHistory"]
            assert asked and asked[-1]["since"] == pytest.approx(now - 100)

            # A task starts: the delta is that one text.
            _seed(harness, "latest prompt", now)
            harness.server._printer.broadcast({"type": "tasks_updated"})
            delta = _wait(
                page, received,
                lambda f: _is_input_history(f) and "since" in f and bool(f.get("tasks")),
                "non-empty delta",
            )
            assert delta["tasks"] == ["latest prompt"]
            assert delta["since"] == pytest.approx(now - 100)
            assert delta["newest"] == pytest.approx(now)

            # ArrowUp walks the merged list, newest first.
            page.click("#task-input")
            seen = []
            for _ in range(3):
                page.keyboard.press("ArrowUp")
                seen.append(page.input_value("#task-input"))
            assert seen == ["latest prompt", "newer prompt", "older prompt"]

            # A reconnect's ready names the newest text on hand, so the
            # fan-out answers with a delta, not the whole list again.
            _deliver(page, {"type": "daemonStatus", "connected": False, "reconnecting": True})
            _deliver(page, {"type": "daemonStatus", "connected": True})
            _wait(
                page, sent,
                lambda f: f.get("type") == "ready" and f.get("inputHistorySince", 0) > 0,
                "second ready",
            )
            readies = [f for f in sent if f.get("type") == "ready"]
            assert readies[-1]["inputHistorySince"] == pytest.approx(now)
            after = _wait(
                page, received,
                lambda f: _is_input_history(f) and f.get("since", 0) >= now - 1,
                "reconnect delta",
            )
            assert after["tasks"] == [] and after["since"] == pytest.approx(now)
            page.wait_for_timeout(500)
            wholes = [f for f in received if _is_input_history(f) and "since" not in f]
            assert len(wholes) == 1, "the whole list travelled more than once"
        finally:
            context.close()

    def test_a_deleted_prompt_leaves_the_list(self, browser, harness) -> None:
        """Deleting a task takes its prompt out of ArrowUp on every page."""
        now = time.time()
        _seed(harness, "kept prompt", now - 60)
        doomed = _seed(harness, "doomed prompt", now - 50)
        context, page, sent, received = _open_page(browser, harness)
        try:
            whole = _wait(page, received, _is_input_history, "inputHistory")
            assert "doomed prompt" in whole["tasks"] and "kept prompt" in whole["tasks"]
            _wait(
                page, received, lambda f: _is_input_history(f) and "since" in f,
                "the ready's delta",
            )
            received.clear()

            # The delete nudges every page; this one asks for a delta
            # (its cursor) and gets the whole list instead, without
            # the prompt, which replaces the list on hand.
            harness.server._agent_server._handle_command({"type": "deleteTask", "taskId": doomed})
            replaced = _wait(page, received, _is_input_history, "inputHistory after delete")
            assert "since" not in replaced
            assert "doomed prompt" not in replaced["tasks"]
            assert "kept prompt" in replaced["tasks"]
            asked = [f for f in sent if f.get("type") == "getInputHistory"]
            assert asked[-1]["since"] == pytest.approx(whole["newest"])

            page.click("#task-input")
            seen = []
            for _ in range(len(replaced["tasks"])):
                page.keyboard.press("ArrowUp")
                seen.append(page.input_value("#task-input"))
            assert seen == replaced["tasks"]

            # The cursor moved past the deletion: the next refresh is
            # a delta again.
            received.clear()
            harness.server._printer.broadcast({"type": "tasks_updated"})
            delta = _wait(page, received, _is_input_history, "delta after the replace")
            assert delta["tasks"] == []
            assert delta["since"] == pytest.approx(replaced["newest"])
            assert delta["since"] >= now
        finally:
            context.close()
