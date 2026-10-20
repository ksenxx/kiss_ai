# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from other test modules)
"""End-to-end test of the phone swipe hint, in a REAL headless Chromium
against a REAL :class:`RemoteAccessServer`.

A phone (mobile user agent, 390px wide): the composer's chats (burger)
and task-info buttons are hidden, and a sticky "swipe-hint" toast tells
the user to swipe between the panels.  Ticking its "Do not show this
again" box removes the toast and records the choice in this browser's
localStorage, so a reload shows no hint and no ``dismissNotice`` frame
reaches the daemon.  A desktop browser at the same width keeps both
buttons (it cannot swipe) and gets no hint.
"""

from __future__ import annotations

import json

from kiss.tests.conftest import goto_retrying_network_change, mark_version_seen
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_mobile_swipe_screens import (
    _swipe_left,
    _touch_session,
    _wait_screen,
)
from kiss.tests.webapp.test_workspace_sections import (
    browser,  # noqa: F401  (module fixture used by param name)
)

TOAST = '.kiss-notification[data-notification-id="swipe-hint"]'
BOX = TOAST + " .kiss-notification-opt-out-box"
KEY = "kiss-swipe-hint-dismissed"
PHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Safari/604.1"
)


def _open_phone_page(browser, harness, user_agent: str | None):
    """Open the remote page 390px wide, under *user_agent* when given.

    Returns ``(context, page, sent_frames)``; *sent_frames* is a live
    list of the JSON frames the page sent over its WebSocket.
    """
    kwargs = {"ignore_https_errors": True, "viewport": {"width": 390, "height": 844}}
    if user_agent:
        kwargs["user_agent"] = user_agent
        kwargs["has_touch"] = True
        kwargs["is_mobile"] = True
    context = browser.new_context(**kwargs)
    mark_version_seen(context)
    page = context.new_page()
    sent_frames: list[dict] = []

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    page.on("websocket", _on_ws)
    goto_retrying_network_change(page, harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "!document.body.classList.contains('remote-desktop')", timeout=15000
    )
    return context, page, sent_frames


def test_phone_hides_drawer_buttons_and_hints_the_swipe(browser, harness):
    context, page, sent = _open_phone_page(browser, harness, PHONE_UA)
    try:
        assert page.evaluate("document.body.classList.contains('remote-mobile')")
        assert page.locator("#menu-btn").is_visible() is False
        assert page.locator("#meta-drawer-btn").is_visible() is False

        toast = page.locator(TOAST)
        toast.wait_for(state="visible", timeout=15000)
        assert "swipe left or right" in toast.inner_text().lower()
        label = page.locator(TOAST + " label.kiss-notification-opt-out")
        assert label.inner_text().strip() == "Do not show this again"
        # Sticky: still there after the plain toasts' 5s lifetime.
        page.wait_for_timeout(5500)
        assert toast.is_visible()

        page.click(BOX)
        toast.wait_for(state="detached", timeout=5000)
        assert page.evaluate(f"localStorage.getItem('{KEY}')") == "1"
        page.wait_for_timeout(300)
        assert [f for f in sent if f.get("type") == "dismissNotice"] == []

        # The record holds: a reload shows no hint.
        page.reload()
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        page.wait_for_function(
            "!document.body.classList.contains('remote-desktop')", timeout=15000
        )
        page.wait_for_timeout(1500)
        assert page.locator(TOAST).count() == 0

        # Closing the task-info drawer (a finger opened it, focus sits on
        # its close button) hands focus to the footer's New chat button,
        # not to the hidden toggle and not to the textbox.
        cdp = _touch_session(page)
        _swipe_left(cdp)
        _wait_screen(page, "info")
        assert page.evaluate("document.activeElement.id") == "meta-close"
        page.keyboard.press("Escape")
        _wait_screen(page, "chat")
        assert page.evaluate("document.activeElement.id") == "new-chat-btn"
    finally:
        context.close()


def test_desktop_browser_at_phone_width_keeps_the_buttons(browser, harness):
    context, page, _ = _open_phone_page(browser, harness, None)
    try:
        assert not page.evaluate("document.body.classList.contains('remote-mobile')")
        page.wait_for_selector("#menu-btn", state="visible", timeout=15000)
        assert page.locator("#meta-drawer-btn").is_visible()
        page.wait_for_timeout(1500)
        assert page.locator(TOAST).count() == 0
    finally:
        context.close()
