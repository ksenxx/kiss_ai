# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A replayed transcript's pictures arrive only when they come into view.

A click on a CHATS row replays the task's transcript with each
picture of a tool result as a ``lazy`` handle instead of its base64
bytes (``server._lazy_replay_images``); ``media/main.js`` watches the
picture's ``<img>`` and asks the daemon for the bytes
(``getReplayImage``) once it is on screen.  A transcript with
screenshots is thus a fraction of its size on a slow network, and a
picture nobody scrolls to is never sent.

Real daemon (``RemoteAccessServer``) with a real history database,
real Chromium (Playwright):

* a picture at the top of a long transcript (the page lands at the
  bottom) is not asked for until the page scrolls up to it; then the
  real reply makes the browser decode it (``naturalWidth > 0``);
* a handle the daemon cannot serve marks the picture missing.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from playwright.sync_api import expect, sync_playwright

from kiss.agents.sorcar import persistence
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _open_page

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)
_PROMPT = "take a screenshot then list many files"
_FILLER_RESULTS = 80


def _seed_task_with_picture(harness: _ServerHarness) -> str:
    """Persist a finished task whose first tool result carries a picture."""
    task_id, _ = persistence._add_task(_PROMPT, "chat-lazy", {"work_dir": str(harness.work_dir)})
    stream: list[dict[str, Any]] = [
        {"type": "tool_call", "tool": "screenshot", "input": {"file_path": "shot.png"}},
        {
            "type": "tool_result", "content": "saved shot.png", "is_error": False,
            "tool_name": "screenshot",
            "images": [{"b64": _PNG_B64, "mime": "image/png", "path": "shot.png"}],
        },
    ]
    for i in range(_FILLER_RESULTS):
        stream.append({"type": "tool_call", "tool": "Bash", "input": {"command": f"ls {i}"}})
        stream.append({
            "type": "tool_result", "content": "\n".join(f"file-{i}-{j}" for j in range(6)),
            "is_error": False, "tool_name": "Bash",
        })
    stream.append({"type": "result", "summary": "done", "success": True})
    for ev in stream:
        persistence._queue_chat_event(dict(ev), task_id=task_id)
    persistence._flush_chat_events()
    persistence._save_task_result("done", task_id=task_id)
    assert base64.b64decode(_PNG_B64)[:4] == b"\x89PNG"
    return task_id


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


def _replay_image_requests(sent: list[dict]) -> list[dict]:
    return [f for f in sent if f.get("type") == "getReplayImage"]


class TestReplayLazyImages:
    """Browser E2E: the picture is fetched on view, through the real daemon."""

    def test_picture_is_fetched_once_scrolled_into_view(self, browser, harness) -> None:
        task_id = _seed_task_with_picture(harness)
        harness.server._printer.broadcast({"type": "tasks_updated"})
        context, page, sent = _open_page(browser, harness)
        page.set_viewport_size({"width": 1200, "height": 700})
        try:
            group = page.locator('#history-list .history-chat-group[data-chat-id="chat-lazy"]')
            expect(group).to_have_count(1, timeout=30000)
            # The chat's rows sit folded under its header: unfold, then click the task.
            group.locator(".history-chat-collapse").click()
            row = group.locator(".sidebar-item").filter(has_text=_PROMPT)
            expect(row).to_have_count(1, timeout=30000)
            row.first.click()
            img = page.locator("#output img.tr-img-lazy")
            expect(img).to_have_count(1, timeout=30000)
            assert page.evaluate(
                "() => { const i = document.querySelector('#output img.tr-img-lazy');"
                " return [i.dataset.replayTask, i.dataset.replayIndex]; }"
            ) == [task_id, "0"]
            # A replayed panel is folded, so the picture is not on
            # screen: it has no bytes and nothing was asked for.
            page.wait_for_timeout(500)
            assert page.evaluate(
                "() => { const i = document.querySelector('#output img.tr-img-lazy');"
                " return [i.getAttribute('src'), !!i.closest('.collapsible.collapsed'),"
                " i.getClientRects().length]; }"
            ) == [None, True, 0]
            assert _replay_image_requests(sent) == []

            # Unfolding the panel and scrolling to the picture is what
            # makes the page ask for it.
            page.evaluate(
                "() => document.querySelector('#output img.tr-img-lazy')"
                ".closest('.collapsible.collapsed').querySelector('.collapse-header').click()"
            )
            page.locator("#output .tr-img[data-replay-index]").first.scroll_into_view_if_needed()
            page.wait_for_function(
                "() => { const i = document.querySelector('#output .tr-img[data-replay-index]');"
                " return i && i.complete && i.naturalWidth > 0; }",
                timeout=30000,
            )
            requests = _replay_image_requests(sent)
            assert len(requests) == 1
            assert requests[0]["taskId"] == task_id
            assert requests[0]["index"] == 0
            assert requests[0]["token"].startswith("rimg-")
            assert page.evaluate(
                "() => document.querySelector('#output .tr-img[data-replay-index]')"
                ".getAttribute('src').slice(0, 22)"
            ) == "data:image/png;base64,"
            assert page.evaluate(
                "() => document.querySelectorAll('#output img.tr-img-lazy').length"
            ) == 0
        finally:
            context.close()

    def test_unknown_picture_is_marked_missing(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            # A page shows the tab the daemon's registry lists (the task
            # the test above opened); the daemon offers its transcript,
            # the page asks for it, and the replay REPLACES the tab's
            # transcript when it lands, late on a loaded machine.  A
            # picture added before that vanishes with it: wait for the
            # restored transcript (its own lazy picture) first.
            if harness.server._agent_server.tab_registry.snapshot():
                page.wait_for_selector(
                    "#output img[data-replay-task]", state="attached", timeout=30000
                )
            page.evaluate(
                """() => {
                     window.dispatchEvent(new MessageEvent('message', {data: {
                       type: 'tool_result', content: 'saved x.png', is_error: false,
                       tool_name: 'screenshot',
                       images: [{mime: 'image/png', path: 'x.png',
                                 lazy: {taskId: 'no-such-task', index: 5}}],
                     }}));
                   }"""
            )
            # The daemon restores the tabs of earlier pages (a replayed
            # task's own lazy picture among them), so pick this picture
            # by its handle; the panel may have folded and the picture
            # may sit below a long transcript: unfold and scroll to it,
            # as a reader would, so it is on screen.
            sel = '#output img[data-replay-task="no-such-task"]'
            page.wait_for_selector(sel, state="attached", timeout=30000)
            page.evaluate(
                "(sel) => { const i = document.querySelector(sel);"
                " const p = i.closest('.collapsible.collapsed');"
                " if (p) p.querySelector('.collapse-header').click();"
                " i.scrollIntoView(); }",
                sel,
            )
            page.wait_for_selector(sel + ".inline-img-missing", state="attached", timeout=30000)
            assert page.evaluate(
                "(sel) => { const i = document.querySelector(sel);"
                " return [i.title, i.className]; }",
                sel,
            ) == ["Cannot show x.png", "tr-img inline-img-missing"]
            requests = [r for r in _replay_image_requests(sent) if r["taskId"] == "no-such-task"]
            assert [r["index"] for r in requests] == [5]
        finally:
            context.close()
