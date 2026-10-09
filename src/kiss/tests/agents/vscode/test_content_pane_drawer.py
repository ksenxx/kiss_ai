# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_explorer_scm_commands are used by param name)
"""End-to-end browser tests for the desktop remote page's content pane
and task-info panel.

* With no file, browser or terminal tab open there is no content pane:
  the chat fills the window between the history panel and the docked
  task-info panel, and never runs under the panel.
* The machine name sits in bold green at the top centre of the chat.
* A file opened from the Explorer opens the content pane, which runs
  under the task-info panel, and slides the panel off the right edge
  (animated), leaving a drawer tab there; the tab brings the panel back
  ON TOP of the pane.
* A click in the content pane slides the panel off again.  Closing the
  last content tab docks the panel beside the chat again.

Every test drives a REAL headless Chromium (Playwright) against a REAL
:class:`RemoteAccessServer` over ``wss://`` whose work dir is a REAL
git repository — no mocks.
"""

from __future__ import annotations

import pytest

from kiss.tests.agents.vscode.test_workspace_sections import (
    _explorer_row,
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_RECT_JS = "sel => document.querySelector(sel).getBoundingClientRect().toJSON()"


def _rect(page, selector: str) -> dict:
    return dict(page.evaluate(_RECT_JS, selector))


def _body_has(page, cls: str) -> bool:
    return bool(page.evaluate("c => document.body.classList.contains(c)", cls))


_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none']) "


def _open_file(page, name: str = "feature.txt") -> None:
    """Open *name* from the Explorer: a text file lands in the editor
    (Monaco, or the plain fallback), a Markdown file in its preview
    frame; both mount asynchronously."""
    page.wait_for_selector(".explorer-row.is-file", timeout=15000)
    _explorer_row(page, name).click()
    page.wait_for_selector("body.content-pane-open", state="attached", timeout=15000)
    page.wait_for_selector("#content-tab-area", state="visible", timeout=15000)
    page.wait_for_selector(
        _VIEW + ".monaco-editor, " + _VIEW + ".content-code-fallback, "
        + _VIEW + ".content-html-frame",
        timeout=30000,
    )


def _bring_panel_back(page) -> None:
    """The file open slid the task-info panel off the right edge; the
    drawer tab brings it back on top of the content pane."""
    page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
    page.locator("#meta-drawer").click()
    page.wait_for_selector("body:not(.meta-hidden)", state="attached", timeout=5000)
    page.wait_for_function(
        "w => document.getElementById('meta-panel').getBoundingClientRect().right <= w + 1"
        " && document.getElementById('meta-panel').getBoundingClientRect().left < w - 100",
        arg=page.evaluate("innerWidth"),
        timeout=5000,
    )
    page.wait_for_function(
        "getComputedStyle(document.getElementById('meta-drawer')).visibility === 'hidden'",
        timeout=5000,
    )


def test_no_content_pane_until_a_file_opens_and_the_panel_overlays_it(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        assert not _body_has(page, "content-pane-open")
        assert page.locator("#content-tab-area").count() == 0 or page.locator(
            "#content-tab-area"
        ).is_hidden()
        assert page.locator("#pane-resizer").is_hidden()
        assert page.locator("#content-tab-bar").is_hidden()
        # The chat ends where the docked panel begins: no overlap.
        app = _rect(page, "#app")
        panel = _rect(page, "#meta-panel")
        output = _rect(page, "#output")
        assert app["right"] <= panel["left"] + 1
        assert output["right"] <= panel["left"] + 1
        assert output["width"] > 300

        _open_file(page)
        # The open slides the panel off the pane; the drawer brings it
        # back over it.
        _bring_panel_back(page)
        content = _rect(page, "#content-tab-area")
        panel = _rect(page, "#meta-panel")
        width = page.evaluate("innerWidth")
        # The pane runs to the window's right edge, under the panel...
        assert content["right"] == pytest.approx(width, abs=1)
        assert content["right"] > panel["left"] + 100
        # ...and the panel is the thing painted there.
        probe_x = panel["left"] + panel["width"] / 2
        probe_y = content["top"] + content["height"] / 2
        top = page.evaluate(
            "([x, y]) => { const el = document.elementFromPoint(x, y);"
            " return el && el.closest('#meta-panel') ? 'panel'"
            " : (el ? el.id || el.className : ''); }",
            [probe_x, probe_y],
        )
        assert top == "panel"
    finally:
        context.close()


def test_machine_name_sits_in_bold_green_at_the_top_centre_of_the_chat(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.wait_for_function(
            "document.getElementById('chat-machine').textContent.length > 0",
            timeout=15000,
        )
        name = page.locator("#chat-machine").inner_text()
        assert name == page.locator("#meta-machine").inner_text()
        style = page.evaluate(
            """() => {
              const el = document.getElementById('chat-machine');
              const cs = getComputedStyle(el);
              const green = getComputedStyle(document.body).getPropertyValue('--green').trim();
              const probe = document.createElement('span');
              probe.style.color = green;
              document.body.appendChild(probe);
              const greenRgb = getComputedStyle(probe).color;
              probe.remove();
              return {color: cs.color, green: greenRgb, weight: cs.fontWeight,
                      align: cs.textAlign};
            }"""
        )
        assert style["color"] == style["green"]
        assert style["weight"] in ("700", "bold")
        assert style["align"] == "center"
        # Top centre of the chat pane: above the transcript, centred on it.
        machine = _rect(page, "#chat-machine")
        output = _rect(page, "#output")
        assert machine["bottom"] <= output["top"] + 1
        mid_machine = machine["left"] + machine["width"] / 2
        mid_output = output["left"] + output["width"] / 2
        assert mid_machine == pytest.approx(mid_output, abs=2)
        # With a file open the strip stays over the chat, not the pane.
        _open_file(page)
        machine = _rect(page, "#chat-machine")
        output = _rect(page, "#output")
        assert machine["right"] <= output["right"] + 1
        assert machine["left"] + machine["width"] / 2 == pytest.approx(
            output["left"] + output["width"] / 2, abs=2,
        )
    finally:
        context.close()


_BORDER_JS = """sel => {
  const cs = getComputedStyle(document.querySelector(sel));
  return [cs.borderBottomWidth, cs.borderBottomStyle, cs.borderBottomColor];
}"""


def test_machine_name_is_parted_from_the_chat_by_the_same_hairline_as_the_tab_rows(
    browser, harness
):
    """The strip ends in the 1px border every other boundary of the page
    uses, and with a file open that hairline runs on from the content
    tab row's: both rows end at the same height."""
    context, page, _ = _open_page(browser, harness)
    try:
        page.wait_for_function(
            "document.getElementById('chat-machine').textContent.length > 0",
            timeout=15000,
        )
        border = page.evaluate(_BORDER_JS, "#chat-machine")
        assert border[0] == "1px" and border[1] == "solid"
        # The chat alone: the hairline is the only thing between the
        # name and the transcript.
        machine = _rect(page, "#chat-machine")
        output = _rect(page, "#output")
        assert machine["bottom"] == pytest.approx(output["top"], abs=1)

        _open_file(page)
        assert page.evaluate(_BORDER_JS, "#content-tab-bar") == border
        machine = _rect(page, "#chat-machine")
        tabs = _rect(page, "#content-tab-bar")
        assert machine["top"] == pytest.approx(tabs["top"], abs=1)
        assert machine["bottom"] == pytest.approx(tabs["bottom"], abs=1)
        # The name sits at the shared row's vertical centre, in one
        # line that still ellipsizes (a flex or grid strip would not).
        text = page.evaluate(
            """() => {
              const el = document.getElementById('chat-machine');
              const r = document.createRange();
              r.selectNodeContents(el);
              const t = r.getBoundingClientRect();
              const b = el.getBoundingClientRect();
              return {mid: (t.top + t.bottom) / 2 - (b.top + b.bottom) / 2,
                      display: getComputedStyle(el).display,
                      overflow: getComputedStyle(el).textOverflow};
            }"""
        )
        assert abs(text["mid"]) <= 2
        assert text["display"] == "block"
        assert text["overflow"] == "ellipsis"
    finally:
        context.close()


def test_click_in_the_pane_hides_the_panel_and_the_drawer_brings_it_back(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        _open_file(page)
        width = page.evaluate("innerWidth")
        height = page.evaluate("innerHeight")
        # The open itself hid the panel: the drawer shows and brings it
        # back.
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        assert page.locator("#meta-panel").get_attribute("inert") is not None
        page.wait_for_function(
            "getComputedStyle(document.getElementById('meta-drawer')).opacity === '1'",
            timeout=5000,
        )
        _bring_panel_back(page)
        assert page.locator("#meta-panel").get_attribute("inert") is None
        panel = _rect(page, "#meta-panel")
        content = _rect(page, "#content-tab-area")
        # The panel slides, it does not jump: a transform transition.
        transition = page.evaluate(
            "() => { const cs = getComputedStyle(document.getElementById('meta-panel'));"
            " return [cs.transitionProperty, cs.transitionDuration]; }"
        )
        assert "transform" in transition[0]
        assert transition[1] not in ("", "0s")
        # Drawer: invisible and untouchable while the panel is up.
        drawer = page.evaluate(
            "() => { const cs = getComputedStyle(document.getElementById('meta-drawer'));"
            " return [cs.display, cs.opacity, cs.pointerEvents, cs.visibility]; }"
        )
        assert drawer == ["flex", "0", "none", "hidden"]
        # Invisible, it is not in the tab order either.
        page.evaluate("document.getElementById('meta-drawer').focus()")
        assert page.evaluate("document.activeElement.id") != "meta-drawer"
        # The panel's own close button hides it too (a wide panel may
        # cover the whole pane): focus goes with it to the drawer...
        close = page.locator("#meta-close")
        assert close.is_visible()
        close.focus()
        close.click()
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        page.wait_for_function(
            "getComputedStyle(document.getElementById('meta-drawer')).opacity === '1'",
            timeout=5000,
        )
        assert page.evaluate("document.activeElement.id") == "meta-drawer"
        # ...and back to the close button when the drawer reopens it.
        page.keyboard.press("Enter")
        page.wait_for_selector("body:not(.meta-hidden)", state="attached", timeout=5000)
        assert page.evaluate("document.activeElement.id") == "meta-close"
        page.wait_for_function(
            "getComputedStyle(document.getElementById('meta-drawer')).visibility === 'hidden'",
            timeout=5000,
        )

        # A click in the editor, left of the panel.
        page.mouse.click(content["left"] + 40, content["top"] + content["height"] / 2)
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        assert page.locator("#meta-panel").get_attribute("inert") is not None
        page.wait_for_function(
            "w => document.getElementById('meta-panel').getBoundingClientRect().left >= w - 1",
            arg=width,
            timeout=5000,
        )
        # The drawer sits at the right edge, vertically centred, and
        # shows now.
        page.wait_for_function(
            "getComputedStyle(document.getElementById('meta-drawer')).opacity === '1'",
            timeout=5000,
        )
        d = _rect(page, "#meta-drawer")
        assert d["right"] == pytest.approx(width, abs=1)
        assert d["top"] + d["height"] / 2 == pytest.approx(height / 2, abs=2)
        assert d["width"] < 30 and d["height"] < 80
        assert page.locator("#meta-drawer").get_attribute("aria-expanded") == "false"
        # The editor now owns the whole width, nothing of the panel
        # over it.
        top = page.evaluate(
            "([x, y]) => { const el = document.elementFromPoint(x, y);"
            " return el && el.closest('#content-tab-area') ? 'pane' : ''; }",
            [panel["left"] + panel["width"] / 2, content["top"] + content["height"] / 2],
        )
        assert top == "pane"

        page.locator("#meta-drawer").click()
        page.wait_for_selector("body:not(.meta-hidden)", state="attached", timeout=5000)
        assert page.locator("#meta-panel").get_attribute("inert") is None
        page.wait_for_function(
            "w => document.getElementById('meta-panel').getBoundingClientRect().right <= w + 1"
            " && document.getElementById('meta-panel').getBoundingClientRect().left < w - 100",
            arg=width,
            timeout=5000,
        )
        page.wait_for_function(
            "getComputedStyle(document.getElementById('meta-drawer')).opacity === '0'",
            timeout=5000,
        )
        assert page.locator("#meta-drawer").get_attribute("aria-expanded") == "true"

        # Hide it again, then close the file: the panel docks beside the
        # chat, never hidden there.
        page.mouse.click(content["left"] + 40, content["top"] + content["height"] / 2)
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        page.click("#content-tab-list .chat-tab.active .chat-tab-close")
        page.wait_for_selector("body:not(.content-pane-open)", state="attached", timeout=15000)
        assert not _body_has(page, "meta-hidden")
        assert page.locator("#meta-panel").get_attribute("inert") is None
        page.wait_for_function(
            "w => document.getElementById('meta-panel').getBoundingClientRect().right <= w + 1"
            " && document.getElementById('meta-panel').getBoundingClientRect().left < w - 100",
            arg=width,
            timeout=5000,
        )
        app = _rect(page, "#app")
        panel = _rect(page, "#meta-panel")
        assert app["right"] <= panel["left"] + 1
        assert page.locator("#meta-drawer").is_hidden()
    finally:
        context.close()


def test_a_click_inside_the_preview_frame_hides_the_panel_too(browser, harness):
    """A Markdown file shows in a sandboxed preview frame whose clicks
    never reach the page; the focus the frame takes does."""
    context, page, _ = _open_page(browser, harness)
    try:
        _open_file(page, "README.md")
        _bring_panel_back(page)
        frame = _rect(page, _VIEW + ".content-html-frame")
        assert frame["width"] > 100 and frame["height"] > 100
        panel = _rect(page, "#meta-panel")
        # A click in the preview, left of the panel.
        x = min(frame["left"] + 40, panel["left"] - 40)
        page.mouse.click(x, frame["top"] + frame["height"] / 2)
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        page.locator("#meta-drawer").click()
        page.wait_for_selector("body:not(.meta-hidden)", state="attached", timeout=5000)
    finally:
        context.close()
