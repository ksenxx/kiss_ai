# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_explorer_scm_commands /
#   kiss.tests.webapp.test_workspace_sections are re-exported
#   for pytest by name)
"""The phone's machine drop-down survives the daemon's reply to its own tap.

A phone opens the native ``<select>`` picker on the ``focus`` a tap gives
``#chat-machine``, and that focus asks the daemon for a fresh machine list
(the drop-down's ``focus`` listener).  iOS dismisses a picker whose options
are replaced under it (or keeps showing the old ones), so the reply must
leave the ``<option>`` nodes alone when it changes nothing.  After a page
refresh nothing is focused yet, so the first tap always goes through this
path: that is the "no list of machines after I refresh the app" report.

Real ``RemoteAccessServer`` + headless Chromium with a phone's viewport,
touch screen and user agent; the tap is Playwright's touch tap.  Chromium
reports its open picker through the ``:open`` pseudo-class, so the test
sees the list open on the tap and still open after the reply; Chromium
patches an open picker when the options change, iOS does not, so the
iOS-relevant check is that the option nodes themselves are untouched.
"""

from __future__ import annotations

import json

from kiss.tests.conftest import goto_retrying_network_change, mark_version_seen
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _dismiss_toasts
from kiss.tests.webapp.test_workspace_sections import (
    browser,  # noqa: F401  (module fixture used by param name)
)

_IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 "
    "Safari/604.1"
)

# Tags every option node, so a later look can tell a repaint (new nodes)
# from an untouched list.
_MARK_JS = """() => {
  const sel = document.getElementById('chat-machine');
  Array.from(sel.options).forEach(o => { o.__kept = true; });
  return Array.from(sel.options).map(o => [o.value, o.textContent]);
}"""
_STATE_JS = """() => {
  const sel = document.getElementById('chat-machine');
  return {
    kept: Array.from(sel.options).every(o => o.__kept === true),
    options: Array.from(sel.options).map(o => [o.value, o.textContent]),
    focused: document.activeElement === sel,
    open: sel.matches(':open'),
    visible: sel.getClientRects().length > 0,
    phone: document.body.classList.contains('remote-mobile') &&
      !document.body.classList.contains('remote-desktop'),
  };
}"""


def _open_phone_page(browser, harness):
    """The remote page as an iPhone shows it; returns ``(context, page,
    sent, received)`` where the lists hold the page's WebSocket frames."""
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": 390, "height": 844},
        is_mobile=True, has_touch=True, user_agent=_IPHONE_UA,
    )
    mark_version_seen(context)
    page = context.new_page()
    sent: list[dict] = []
    received: list[dict] = []

    def _record(frames, payload) -> None:
        try:
            frames.append(json.loads(payload))
        except Exception:
            pass

    def _on_ws(ws) -> None:
        ws.on("framesent", lambda payload: _record(sent, payload))
        ws.on("framereceived", lambda payload: _record(received, payload))

    page.on("websocket", _on_ws)
    goto_retrying_network_change(page, harness.base_url + "/")
    return context, page, sent, received


def _count(frames, kind: str) -> int:
    return len([f for f in frames if f.get("type") == kind])


def _wait_for_count(page, frames, kind: str, above: int) -> None:
    for _ in range(80):
        if _count(frames, kind) > above:
            return
        page.wait_for_timeout(250)
    raise AssertionError(f"no new {kind} frame")


def _wait_for_boot(page, received) -> None:
    """The page is up and the boot's machine list has been painted."""
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    _wait_for_count(page, received, "machines", 0)
    page.wait_for_function(
        "document.getElementById('chat-machine').options.length > 0",
        timeout=15000,
    )


def test_a_tap_after_a_refresh_keeps_the_picker_s_options(browser, harness):
    context, page, sent, received = _open_phone_page(browser, harness)
    try:
        _wait_for_boot(page, received)
        page.reload()
        sent.clear()
        received.clear()
        _wait_for_boot(page, received)
        _dismiss_toasts(page)
        before = page.evaluate(_MARK_JS)
        assert not page.evaluate(_STATE_JS)["open"], "closed before the tap"
        asked = _count(sent, "getMachines")
        answered = _count(received, "machines")
        page.tap("#chat-machine")
        _wait_for_count(page, sent, "getMachines", asked)
        opened = page.evaluate(_STATE_JS)
        assert opened["phone"], "the phone layout is on"
        assert opened["focused"], "the tap focused the drop-down"
        assert opened["open"], "the tap opened the list"
        _wait_for_count(page, received, "machines", answered)
        page.wait_for_timeout(300)
        state = page.evaluate(_STATE_JS)
        assert state["open"], "the list is still open after the reply"
        assert state["visible"]
        assert state["options"] == before
        assert state["kept"], "the daemon's reply replaced the option nodes"
        # A phone has no Escape key: tapping away blurs the drop-down, which
        # closes its picker.  (Escape closes it on Linux Chromium but not on
        # macOS Chromium, where the picker stays open.)
        page.locator("#chat-machine").blur()
        page.wait_for_function(
            "!document.getElementById('chat-machine').matches(':open')",
            timeout=5000,
        )
    finally:
        context.close()
