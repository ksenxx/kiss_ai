# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Summary folding and user-response layout in the real chat webview.

Load the production remote server and shared webview assets in Chromium.
Events enter through the same message listener used by both hosts; no UI,
transport, or browser functions are replaced. Finished-task Trajectory
folding is independent: open that outer panel before checking its summaries.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from kiss.tests.agents.vscode.test_remote_panels_match_extension import (
    _load_page,
    _start_remote_server,
)

PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
    "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)
QUESTION = "Which branch should I use?"
ANSWER = "release <b>literally</b>\nnext line"


@pytest.fixture
def chat_page(tmp_path: Path) -> Iterator[Page]:
    """Serve an isolated production webapp and close it after each test."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_remote_server, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "remote server did not start"
        assert "error" not in state, state
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(ignore_https_errors=True)
            try:
                page = context.new_page()
                profiler = context.new_cdp_session(page)
                profiler.send("Profiler.enable")
                profiler.send(
                    "Profiler.startPreciseCoverage", {"callCount": True, "detailed": True}
                )
                _load_page(page, f"https://127.0.0.1:{state['port']}/")
                expect(page.locator("#kiss-server-loading")).to_be_hidden()
                yield page
                coverage = profiler.send("Profiler.takePreciseCoverage")
                (tmp_path / "webview-coverage.json").write_text(json.dumps(coverage))
            finally:
                context.close()
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
        assert not thread.is_alive(), "remote server did not stop"


def _send(page: Page, events: list[dict[str, Any]], tab: str | None = None) -> None:
    """Deliver inbound host events to the real chat's message listener."""
    page.evaluate(
        """({events, tab}) => {
          const tabId = tab || window._testApi.getActiveTabId();
          for (const event of events)
            window.dispatchEvent(new MessageEvent('message', {
              data: {...event, tabId},
            }));
        }""",
        {"events": events, "tab": tab},
    )


def _summary(description: str) -> dict[str, Any]:
    """Build a summary tool-call event."""
    return {"type": "tool_call", "name": "summary", "description": description}


def _events() -> list[dict[str, Any]]:
    """Interleave ordinary panels, media, a question, and user/agent replies."""
    return [
        {"type": "prompt", "text": "keep this prompt boundary"},
        {"type": "tool_call", "name": "Read", "path": "first.txt"},
        {"type": "tool_result", "content": "ordinary text"},
        {"type": "tool_call", "name": "screenshot"},
        {
            "type": "tool_result",
            "content": "image",
            "images": [{"path": "shot.png", "mime": "image/png", "b64": PNG}],
        },
        {"type": "tool_call", "name": "ask_user_question", "extras": {"question": QUESTION}},
        {"type": "tool_result", "tool_name": "ask_user_question", "content": ANSWER},
        {"type": "prompt", "steer": True, "text": "Keep this steering message"},
        {"type": "ask_answer", "question": "Why?", "text": "Agent answer", "success": True},
        {"type": "tool_call", "name": "Read", "path": "last.txt"},
        {"type": "tool_result", "content": "last ordinary text"},
        _summary("First digest"),
    ]


def _assert_preserved(page: Page, root: str = "#output") -> None:
    """Check ordering and actual visibility while the summary is folded."""
    summary = page.locator(f"{root} > .tc-summary").first
    expect(summary.locator(":scope > .tc-h")).to_have_attribute("aria-expanded", "false")
    expect(summary.locator(".summary-sub > .tc")).to_have_count(2)
    expect(summary.locator(".summary-sub img, .summary-sub .tc-question")).to_have_count(0)
    panels = page.locator(
        f"{root} > .tc:has(img), {root} > .tc-question, {root} > .user-msg, {root} > .ask-answer"
    )
    expect(panels).to_have_count(5)
    for panel in panels.all():
        expect(panel).to_be_visible()
        assert not panel.evaluate("el => el.classList.contains('collapsed')")
        assert panel.evaluate(
            "el => !!(el.previousElementSibling && "
            "el.compareDocumentPosition(el.parentElement.querySelector('.tc-summary')) & 2)"
        ), "persistent panels must follow the summary"
    expect(page.locator(f"{root} > .tc-question + .user-msg .task-panel-text")).to_have_text(ANSWER)
    expect(page.locator(f"{root} > .user-msg b")).to_have_count(0)


@pytest.mark.parametrize("replay", [False, True])
def test_summary_leaves_persistent_panels_after_digest(chat_page: Page, replay: bool) -> None:
    """Live and replayed summaries fold only ordinary panels, in order."""
    page = chat_page
    _send(page, [{"type": "clear"}, {"type": "status", "running": True}])
    if replay:
        _send(page, [{"type": "task_events", "task": "test", "task_id": 123, "events": _events()}])
    else:
        _send(page, _events())
    _assert_preserved(page)
    first_summary = page.locator("#output > .tc-summary").first
    # Opening and closing the digest never hides its preserved siblings.
    first_summary.locator(":scope > .tc-h").click()
    expect(first_summary.locator(".summary-sub")).to_be_visible()
    first_summary.locator(":scope > .tc-h").click()
    _assert_preserved(page)
    _send(page, [{"type": "tool_result", "content": "summary accepted"}])
    expect(first_summary.locator(":scope > .bash-panel")).to_have_text("summary accepted")
    _send(
        page,
        [
            {"type": "tool_call", "name": "Read", "path": "next.txt"},
            {"type": "tool_result", "content": "next"},
            _summary("Second digest"),
        ],
    )
    second = page.locator("#output > .tc-summary").nth(1)
    expect(second.locator(".summary-sub > .tc")).to_have_count(1)
    expect(second.locator(".summary-sub .user-msg, .summary-sub img")).to_have_count(0)
    assert second.evaluate("el => el.previousElementSibling.classList.contains('ask-answer')")
    _assert_preserved(page)


def test_user_response_is_a_right_aligned_user_message(chat_page: Page) -> None:
    """User answers use the same bubble and edge alignment as user messages."""
    page = chat_page
    _send(page, [{"type": "clear"}, {"type": "status", "running": True}, *_events()])
    response = page.locator("#output > .tc-question + .user-msg")
    expect(response).to_be_visible()
    expect(response.locator(".task-panel-h")).to_contain_text("Response")
    expect(response.locator(".task-panel-text")).to_have_text(ANSWER)
    geometry = response.evaluate("""el => {
      const r = el.getBoundingClientRect();
      const parent = el.parentElement;
      const p = parent.getBoundingClientRect();
      const s = getComputedStyle(parent);
      const content = parent.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
      return {
        width: r.width / content,
        gap: p.left + parent.clientWidth - parseFloat(s.paddingRight) - r.right,
      };
    }""")
    assert geometry["width"] == pytest.approx(0.8, abs=0.01), geometry
    assert geometry["gap"] == pytest.approx(0, abs=1), geometry
    # The user can still explicitly fold their response.
    response.locator(".task-panel-h").click()
    expect(response.locator(".task-panel-text")).to_be_hidden()
    _send(page, [_summary("Third digest")])
    expect(response.locator(".task-panel-text")).to_be_hidden()


@pytest.mark.parametrize(
    "media",
    ["![diagram](/media/kiss-icon.png)", '<video controls width="160" height="90"></video>'],
)
def test_thoughts_with_media_fold_into_summary(chat_page: Page, media: str) -> None:
    """A Thoughts panel joins the digest even when its Markdown shows media.

    Only tool-result media panels stay expanded after the summary; the
    agent's own Thoughts are folded like every other recounted step.
    """
    page = chat_page
    events: list[dict[str, Any]] = [
        {"type": "text_delta", "text": "ordinary thoughts"},
        {"type": "text_end"},
        {"type": "tool_call", "name": "screenshot"},
        {
            "type": "tool_result",
            "content": "image",
            "images": [{"path": "shot.png", "mime": "image/png", "b64": PNG}],
        },
        {"type": "text_delta", "text": media},
        {"type": "text_end"},
        {"type": "tool_call", "name": "Read"},
        {"type": "tool_result", "content": "after media"},
        _summary("Media digest"),
    ]
    _send(page, [{"type": "clear"}, {"type": "status", "running": True}, *events])
    summary = page.locator("#output > .tc-summary")
    expect(summary.locator(":scope > .tc-h")).to_have_attribute("aria-expanded", "false")
    expect(summary.locator(".summary-sub > .llm-panel")).to_have_count(2)
    expect(summary.locator(".summary-sub > .tc")).to_have_count(1)
    expect(page.locator("#output > .llm-panel")).to_have_count(0)
    expect(page.locator("#output > .tc-summary ~ .tc:has(img.tr-img)")).to_be_visible()
    thought_media = summary.locator(".summary-sub .llm-panel img, .summary-sub .llm-panel video")
    expect(thought_media).to_have_count(1)
    expect(thought_media).to_be_hidden()
    # Opening the digest shows the folded Thoughts and its media again.
    summary.locator(":scope > .tc-h").click()
    expect(thought_media).to_be_visible()
    summary.locator(":scope > .tc-h").click()
    # Finished-task replay folds the Thoughts the same way inside the
    # independent outer Trajectory fold.
    events.append({"type": "result", "summary": "done", "success": True})
    _send(
        page,
        [
            {"type": "status", "running": False},
            {"type": "task_events", "task": "media replay", "task_id": 456, "events": events},
        ],
    )
    page.locator("#output > .trajectory > .trajectory-h").click()
    expect(page.locator(".trajectory-sub > .llm-panel")).to_have_count(0)
    expect(page.locator(".trajectory-sub > .tc-summary .summary-sub > .llm-panel")).to_have_count(2)
    expect(page.locator(".trajectory-sub > .tc-summary ~ .tc:has(img.tr-img)")).to_be_visible()
    expect(page.locator(".tc-summary > .summary-sub")).to_be_hidden()


@pytest.mark.parametrize("content,is_error", [("", False), ("interrupted", True)])
def test_empty_and_failed_question_results(chat_page: Page, content: str, is_error: bool) -> None:
    """An empty reply is a user bubble; an error remains tool output."""
    page = chat_page
    _send(
        page,
        [
            {"type": "clear"},
            {"type": "status", "running": True},
            {"type": "tool_call", "name": "ask_user_question", "extras": {"question": QUESTION}},
            {"type": "tool_result", "content": content, "is_error": is_error},
            _summary("Question digest"),
        ],
    )
    expect(page.locator("#output > .tc-question")).to_be_visible()
    if is_error:
        expect(page.locator("#output .tc-question .tr.err")).to_contain_text(content)
        expect(page.locator("#output > .user-msg")).to_have_count(0)
    else:
        response = page.locator("#output > .tc-question + .user-msg")
        expect(response).to_be_visible()
        expect(response.locator(".task-panel-text")).to_have_text("")


def test_summary_at_welcome_boundary(chat_page: Page) -> None:
    """A fresh chat's welcome element is never adopted into a summary."""
    page = chat_page
    expect(page.locator("#output > #welcome")).to_be_visible()
    _send(page, [_summary("No earlier events"), _summary("Another digest")])
    expect(page.locator("#output > #welcome")).to_have_count(1)
    expect(page.locator("#output > .tc-summary")).to_have_count(2)
    expect(page.locator("#output > .tc-summary > .summary-sub > *")).to_have_count(0)


def test_background_summary_does_not_leak_panels(chat_page: Page) -> None:
    """A hidden chat keeps its panels and responses out of the visible chat."""
    page = chat_page
    # The fresh page's tab is a local placeholder the daemon's tab
    # registry never lists; the registry drops it the moment another
    # tab is registered (the "+" button registers the tab it opens).
    # A bare ``status`` event does not register a tab (only a run
    # does), so the hidden chat must be a tab opened through "+".
    placeholder = page.evaluate("window._testApi.getActiveTabId()")
    page.locator("#new-chat-btn").click()
    first = page.evaluate("window._testApi.getActiveTabId()")
    assert first != placeholder
    expect(page.locator(f'.chat-tab[data-tab-id="{placeholder}"]')).to_have_count(0)
    _send(page, [{"type": "clear"}, {"type": "status", "running": True}])
    page.locator("#new-chat-btn").click()
    second = page.evaluate("window._testApi.getActiveTabId()")
    assert first != second
    # The running first chat survives the "+": only an idle chat is
    # retired when the user leaves it.
    assert first in {t["id"] for t in page.evaluate("window._testApi.openTabs()")}
    _send(page, _events(), tab=first)
    expect(
        page.locator("#output .tc-summary, #output .tc-question, #output .tc-question-answer")
    ).to_have_count(0)
    # Chats have no tab row: the Chats-panel pick is switchToTab.
    page.evaluate("id => window._testApi.switchToTab(id)", first)
    _assert_preserved(page)
    page.evaluate("id => window._testApi.switchToTab(id)", second)
    expect(
        page.locator("#output .tc-summary, #output .tc-question, #output .tc-question-answer")
    ).to_have_count(0)
