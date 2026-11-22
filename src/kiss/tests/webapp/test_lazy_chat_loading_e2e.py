# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A chat opens as stubs; a panel's events arrive when the reader opens it.

The chat page asks for transcripts in pieces (``lazy: true`` on
``resumeSession``, ``getAdjacentTask`` and ``ready``; see
``kiss/server/lazy_replay.py``).  A finished task then arrives as its
text, one folded and empty "Trajectory N events" panel and its result.
The panels are fetched (``getTaskEvents``) when the reader opens the
Trajectory panel, and those a ``summary`` call folded when the reader
opens that summary.

Real daemon (``RemoteAccessServer``) with a real history database,
real Chromium (Playwright); what the page asked for is read off its
WebSocket:

* a click on a CHATS row shows the stub and asks for no events;
* opening the Trajectory panel asks once, for the outline: the
  summaries, folded and empty, and the panels no summary folded;
* opening a summary asks for the panels it folded;
* a reloaded page, and a page in another browser, get the stub again;
* so does a page that lost its connection, from its next ``ready``;
* the chat's previous task, pulled in by scrolling past the top edge,
  arrives as a stub too and opens the same way.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
from typing import Any

import pytest
from playwright.sync_api import Locator, Page, expect, sync_playwright

from kiss.agents.sorcar import persistence
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _open_page

_CHAT = "chat-lazy-e2e"
_PROMPT = "count the files"
_RESULT = "Two files."
_TIMEOUT = 30000
# The panels of _transcript: in the Trajectory panel, and folded by each summary.
_OUTLINE_PANELS = 5
_FOLDED_PANELS = [4, 1]


def _call(name: str, **fields: Any) -> dict[str, Any]:
    """A ``tool_call`` event of tool *name*."""
    return {"type": "tool_call", "name": name, **fields}


def _done(name: str, content: str = "ok") -> dict[str, Any]:
    """The ``tool_result`` event of tool *name*."""
    return {"type": "tool_result", "content": content, "is_error": False, "tool_name": name}


def _thought(text: str) -> list[dict[str, Any]]:
    """A streamed thought."""
    return [
        {"type": "thinking_start"},
        {"type": "thinking_delta", "text": text},
        {"type": "thinking_end"},
    ]


def _transcript(work_dir: str, result: str) -> list[dict[str, Any]]:
    """A finished task whose two ``summary`` calls folded its first steps.

    Its Trajectory panel holds five panels: the two summaries, the
    message typed into the task, the last thought and the ``finish``
    call.  The first summary folded four (the system prompt, the
    prompt, a thought and a command), the second one (a command).
    """
    return [
        {"type": "task_settings", "settings": {"model": "m", "work_dir": work_dir}},
        {"type": "system_prompt", "text": "You are the agent."},
        {"type": "prompt", "text": "Count them."},
        *_thought("Let me look."),
        {"type": "usage_info", "text": "Steps: 1/100", "total_tokens": 10, "cost": "$0.01",
         "total_steps": 1},
        _call("Bash", command="ls", description="look into the folder"),
        {"type": "system_output", "text": "a.txt\nb.txt\n"},
        _done("Bash", ""),
        {"type": "prompt", "text": "skip the hidden ones", "steer": True,
         "promptIds": ["prompt-1"]},
        _call("summary", description="- listed the folder"),
        _done("summary"),
        _call("Bash", command="wc -l a.txt", description="measure a.txt"),
        _done("Bash", "3 a.txt"),
        _call("summary", description="- measured a file"),
        _done("summary"),
        *_thought("Wrapping up."),
        _call("finish", extras={"success": "True"}),
        _done("finish", "finished"),
        {"type": "result", "text": "success: true", "summary": f"<p>{result}</p>",
         "success": True, "step_count": 3, "total_tokens": 100, "cost": "$0.05"},
        {"type": "usage_info", "text": "Steps: 3/100", "total_tokens": 100, "cost": "$0.05"},
        {"type": "task_done", "success": True},
        {"type": "followup_suggestion", "text": "Measure b.txt."},
    ]


def _seed(harness: _ServerHarness, prompt: str = _PROMPT, result: str = _RESULT) -> str:
    """Persist a finished task of the chat; returns its id."""
    work_dir = str(harness.work_dir)
    task_id, _ = persistence._add_task(prompt, _CHAT, {"work_dir": work_dir})
    for ev in _transcript(work_dir, result):
        persistence._queue_chat_event(ev, task_id=task_id)
    persistence._flush_chat_events()
    persistence._save_task_result(result, task_id=task_id)
    return task_id


@pytest.fixture
def harness():
    """A real daemon with its own history database and tab registry."""
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


def _open(browser, harness: _ServerHarness) -> tuple[Any, Page, list[dict], list[str]]:
    """Open the webapp; returns its context, the page, the frames it sent and its errors."""
    context, page, sent = _open_page(browser, harness)
    page.set_viewport_size({"width": 1200, "height": 700})
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    return context, page, sent, errors


def _open_task(page: Page, prompt: str = _PROMPT, result: str = _RESULT) -> None:
    """Click the CHATS row of the chat's task *prompt* and wait for its transcript."""
    group = page.locator(f'#history-list .history-chat-group[data-chat-id="{_CHAT}"]')
    # The chat's rows sit folded under its header: unfold, then click the task.
    group.locator(".history-chat-header > .history-chat-collapse").click()
    group.locator(".sidebar-item").filter(has_text=prompt).click()
    expect(page.locator("#output > .rc")).to_contain_text(result, timeout=_TIMEOUT)


def _unfold(panel: Locator) -> Locator:
    """Click the header of the Trajectory or summary *panel*; returns the panels in its body."""
    panel.locator("> .collapse-header").click()
    return panel.locator("> .trajectory-sub > *, > .summary-sub > *")


def _preview(panel: Locator) -> str:
    """What the folded *panel* says it holds."""
    return panel.locator("> .collapse-header .collapse-preview").inner_text()


def _frames(sent: list[dict], kind: str) -> list[dict]:
    """The frames of type *kind* the page sent."""
    return [f for f in sent if f.get("type") == kind]


def _expect_stub(page: Page) -> None:
    """*page* shows the task as its text, the folded stub and its result, and no other panel."""
    output = page.locator("#output")
    stub = output.locator("> .trajectory.collapsed.lazy-unloaded")
    expect(stub).to_have_count(1, timeout=_TIMEOUT)
    expect(output.locator("> .rc")).to_contain_text(_RESULT, timeout=_TIMEOUT)
    assert _PROMPT in output.locator("> .task-panel").inner_text()
    assert _preview(stub) == f"{_OUTLINE_PANELS} events"
    # Nothing of the trajectory is on the page, now or a moment later.
    page.wait_for_timeout(500)
    assert stub.locator("> .trajectory-sub > *").count() == 0
    assert output.locator(".ev, .llm-panel").count() == 3, "the task, the stub, the result"


async def _drop_pages(harness: _ServerHarness) -> None:
    """Close the daemon's end of every page's socket: each page connects again."""
    for ws in list(harness.server._printer._remote_clients):
        await ws.close()


class TestLazyChatLoading:
    """Browser E2E: a transcript is fetched piece by piece, as its panels are opened."""

    def test_a_finished_task_opens_as_a_trajectory_stub(self, browser, harness) -> None:
        """A click on the task's CHATS row shows its text, the stub and the result."""
        task_id = _seed(harness)
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page)
            _expect_stub(page)
            resumes = _frames(sent, "resumeSession")
            assert [(f["chatId"], f["taskId"], f["lazy"]) for f in resumes] == [
                (_CHAT, task_id, True)
            ]
            assert _frames(sent, "getTaskEvents") == []
        assert errors == []

    def test_opening_the_trajectory_loads_its_outline(self, browser, harness) -> None:
        """The Trajectory panel's header is clicked: one fetch brings the outline."""
        task_id = _seed(harness)
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page)
            trajectory = page.locator("#output > .trajectory")
            panels = _unfold(trajectory)
            expect(panels).to_have_count(_OUTLINE_PANELS, timeout=_TIMEOUT)
            assert page.locator("#output > .trajectory:is(.collapsed, .lazy-unloaded)").count() == 0
            tab_id = page.evaluate("window._testApi.getActiveTabId()")
            fetches = _frames(sent, "getTaskEvents")
            assert [(f["taskId"], f["tabId"], "seg" in f) for f in fetches] == [
                (task_id, tab_id, False)
            ]
            # The summaries came folded and empty: what they folded is not on the page.
            summaries = trajectory.locator("> .trajectory-sub > .tc-summary")
            assert summaries.count() == 2
            assert trajectory.locator(".tc-summary.collapsed.lazy-unloaded").count() == 2
            assert trajectory.locator(".summary-sub > *").count() == 0
            text = trajectory.text_content() or ""
            assert "look into the folder" not in text and "measure a.txt" not in text
            # The panels no summary folded came with them.
            assert "skip the hidden ones" in text and "Wrapping up." in text
            assert panels.last.locator(".tc-h-name").inner_text() == "finish"
            # Folded again, the panel counts what it holds: what the stub said.
            trajectory.locator("> .collapse-header").click()
            assert _preview(trajectory) == f"{_OUTLINE_PANELS} events"
            # Opened again, it asks for nothing: it holds its events.
            trajectory.locator("> .collapse-header").click()
            page.wait_for_timeout(500)
            assert len(_frames(sent, "getTaskEvents")) == 1
            assert panels.count() == _OUTLINE_PANELS
        assert errors == []

    def test_opening_a_summary_loads_the_panels_it_folded(self, browser, harness) -> None:
        """A summary's header is clicked: one fetch brings the panels of its steps."""
        task_id = _seed(harness)
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page)
            trajectory = page.locator("#output > .trajectory")
            expect(_unfold(trajectory)).to_have_count(_OUTLINE_PANELS, timeout=_TIMEOUT)
            first, second = trajectory.locator("> .trajectory-sub > .tc-summary").all()
            expect(_unfold(first)).to_have_count(_FOLDED_PANELS[0], timeout=_TIMEOUT)
            text = first.locator("> .summary-sub").text_content() or ""
            for folded in ("You are the agent.", "Count them.", "Let me look.",
                           "look into the folder"):
                assert folded in text
            # Only that summary was asked for: the other is still folded and empty.
            assert trajectory.locator(".tc-summary.collapsed.lazy-unloaded").count() == 1
            assert second.locator("> .summary-sub > *").count() == 0
            expect(_unfold(second)).to_have_count(_FOLDED_PANELS[1], timeout=_TIMEOUT)
            assert "measure a.txt" in (second.locator("> .summary-sub").text_content() or "")
            assert trajectory.locator(".lazy-unloaded").count() == 0
            fetches = _frames(sent, "getTaskEvents")
            assert [(f["taskId"], f.get("seg")) for f in fetches] == [
                (task_id, None), (task_id, 0), (task_id, 1)
            ]
        assert errors == []

    def _expect_stub_restored(self, page: Page, sent: list[dict], tab_id: str, task_id: str
                              ) -> None:
        """A page that just loaded shows the chat's tab as the stub, asked for in pieces."""
        _expect_stub(page)
        assert [f.get("lazy") for f in _frames(sent, "ready")] == [True]
        resumes = _frames(sent, "resumeSession")
        assert [(f["tabId"], f["taskId"], f["lazy"]) for f in resumes] == [
            (tab_id, task_id, True)
        ]
        assert _frames(sent, "getTaskEvents") == []

    def test_a_reloaded_page_gets_the_stub_again(self, browser, harness) -> None:
        """Reloaded, or opened in another browser, the page shows the stub, not the outline.

        Such a page starts on a tab of its own, learns of the chat's
        tab from the daemon and asks for its transcript.
        """
        task_id = _seed(harness)
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page)
            trajectory = page.locator("#output > .trajectory")
            expect(_unfold(trajectory)).to_have_count(_OUTLINE_PANELS, timeout=_TIMEOUT)
            tab_id = page.evaluate("window._testApi.getActiveTabId()")
            sent.clear()
            page.reload()
            self._expect_stub_restored(page, sent, tab_id, task_id)
        assert errors == []
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            self._expect_stub_restored(page, sent, tab_id, task_id)
        assert errors == []

    def test_a_reconnected_page_gets_the_stub_again(self, browser, harness) -> None:
        """The page's connection drops: its next ``ready`` names the tab on screen.

        The daemon answers that ``ready`` with the tab's transcript,
        in pieces as the ``ready`` asked: the stub replaces the outline
        the page had loaded, and nothing more is asked for.
        """
        _seed(harness)
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page)
            trajectory = page.locator("#output > .trajectory")
            expect(_unfold(trajectory)).to_have_count(_OUTLINE_PANELS, timeout=_TIMEOUT)
            tab_id = page.evaluate("window._testApi.getActiveTabId()")
            sent.clear()
            asyncio.run_coroutine_threadsafe(_drop_pages(harness), harness.loop).result(60)
            _expect_stub(page)
            assert [(f.get("tabId"), f.get("lazy")) for f in _frames(sent, "ready")] == [
                (tab_id, True)
            ]
            assert _frames(sent, "resumeSession") == []
            assert _frames(sent, "getTaskEvents") == []
        assert errors == []

    def test_a_neighbouring_task_arrives_as_a_stub(self, browser, harness) -> None:
        """Scrolling past the top edge pulls the chat's previous task in, as a stub that opens."""
        first = _seed(harness)
        second = _seed(harness, "measure the folder", "Four kilobytes.")
        context, page, sent, errors = _open(browser, harness)
        with closing(context):
            _open_task(page, "measure the folder", "Four kilobytes.")
            output = page.locator("#output")
            # A stub's transcript is too short to scroll: a wheel turn up at
            # its top edge reads on into the task before.
            output.hover()
            page.mouse.wheel(0, -300)
            neighbour = output.locator("> .adjacent-task")
            expect(neighbour).to_have_count(1, timeout=_TIMEOUT)
            assert neighbour.get_attribute("data-task-id") == first
            assert _PROMPT in neighbour.locator("> .task-panel").inner_text()
            assert _RESULT in neighbour.locator("> .rc").inner_text()
            stub = neighbour.locator("> .trajectory.collapsed.lazy-unloaded")
            assert _preview(stub) == f"{_OUTLINE_PANELS} events"
            assert stub.locator("> .trajectory-sub > *").count() == 0
            asked = _frames(sent, "getAdjacentTask")
            assert (asked[0]["taskId"], asked[0]["direction"]) == (second, "prev")
            assert [f.get("lazy") for f in asked] == [True] * len(asked)
            assert _frames(sent, "getTaskEvents") == []
            # The neighbour's Trajectory panel opens as the tab's own does.
            panels = _unfold(neighbour.locator("> .trajectory"))
            expect(panels).to_have_count(_OUTLINE_PANELS, timeout=_TIMEOUT)
            tab_id = page.evaluate("window._testApi.getActiveTabId()")
            fetches = _frames(sent, "getTaskEvents")
            assert [(f["taskId"], f["tabId"], "seg" in f) for f in fetches] == [
                (first, tab_id, False)
            ]
            # The tab's own task was not touched.
            assert output.locator("> .trajectory.collapsed.lazy-unloaded").count() == 1
        assert errors == []
