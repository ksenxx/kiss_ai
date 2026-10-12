# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_explorer_scm_commands are used by param name)
"""End-to-end browser tests for two desktop remote page details.

* The Inject promptlet sheet (``#tricks-panel``, opened by the
  composer's "Inject promptlet" button) rises INSIDE the chat pane: it
  and its scrim stop at the docked history panel on the left and at
  the docked task-info panel (or, with a file open, the content pane)
  on the right, instead of spanning the whole window.  The Working
  directory sheet shares the geometry.
* Every section header of the task-info panel (Task Info, Explorer,
  Source Control, Schedule, Apps, Spend ...) is a short row: the band
  from the header's top to its body's top is a fifth shorter than the
  18px icon-button row plus 8px gap it used to be (itself a fifth
  shorter than the 22px row and 10px gap before that), and a
  collapsed section drops the gap: it costs its header row plus the
  separator and nothing else.

Every test drives a REAL headless Chromium (Playwright) against a REAL
:class:`RemoteAccessServer` over ``wss://`` whose work dir is a REAL
git repository -- no mocks.
"""

from __future__ import annotations

import pytest

from kiss.tests.agents.vscode.test_content_pane_drawer import _open_file
from kiss.tests.agents.vscode.test_workspace_sections import (
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_RECT_JS = "sel => document.querySelector(sel).getBoundingClientRect().toJSON()"

# The header band before this change: an 18px row (the refresh / folder
# icon buttons set its height) plus the 8px gap below it.  The Task
# Info header has no button: its text line (11px at line-height 1.1 =
# 12.1px) plus the gap.  Both must now be at least a fifth shorter.
_OLD_BUTTON_HEADER_BAND_PX = 26.0
_OLD_TEXT_HEADER_BAND_PX = 20.1

# Every rendered section header: its row height plus the gap it keeps
# below itself (a Loading... status line may follow the gap).
_HEADER_BANDS_JS = """
() => [...document.querySelectorAll('#meta-panel .meta-section .meta-section-hdr')]
  .filter(hdr => hdr.getBoundingClientRect().height > 0)
  .map(hdr => ({
    id: hdr.parentElement.id,
    hasButton: !!hdr.querySelector('button:not(.meta-section-toggle)'),
    row: hdr.getBoundingClientRect().height,
    band: hdr.getBoundingClientRect().height
      + parseFloat(getComputedStyle(hdr).marginBottom),
  }))
"""


def _rect(page, selector: str) -> dict:
    return dict(page.evaluate(_RECT_JS, selector))


def _open_sheet(page, button: str, sheet: str) -> None:
    """Press *button* and wait for *sheet* to finish its slide in."""
    page.locator(button).click()
    page.wait_for_selector(f"{sheet}.open", state="attached", timeout=5000)
    page.wait_for_function(
        "sel => Math.abs(document.querySelector(sel).getBoundingClientRect()"
        ".bottom - document.getElementById('app').getBoundingClientRect().bottom) < 1",
        arg=sheet,
        timeout=5000,
    )


def _assert_inside_chat_pane(page, sheet: str, overlay: str, right_edge: float) -> None:
    """*sheet* and *overlay* lie within the chat pane: right of the
    history panel, left of *right_edge* (the docked task-info panel or
    the content pane), bottom on the chat's bottom edge."""
    chat = _rect(page, "#output")
    composer = _rect(page, "#input-area")
    sidebar = _rect(page, "#sidebar")
    box = _rect(page, sheet)
    scrim = _rect(page, overlay)
    assert box["width"] > 200, f"the sheet must still be a usable width: {box}"
    assert box["left"] >= sidebar["right"] - 1, f"{sheet} runs under the history panel: {box}"
    assert box["left"] >= chat["left"] - 1, f"{sheet} starts left of the chat: {box} vs {chat}"
    assert box["right"] <= right_edge + 1, (
        f"{sheet} runs past the chat pane's right edge {right_edge}: {box}"
    )
    assert box["bottom"] == pytest.approx(composer["bottom"], abs=1), (
        f"{sheet} must sit on the composer's bottom edge: {box} vs {composer}"
    )
    assert scrim["left"] == pytest.approx(chat["left"], abs=1), f"scrim: {scrim} vs {chat}"
    assert scrim["right"] == pytest.approx(right_edge, abs=1), f"scrim: {scrim}"
    assert scrim["top"] == pytest.approx(0, abs=1), f"scrim must cover the chat column: {scrim}"
    assert scrim["bottom"] == pytest.approx(composer["bottom"], abs=1), f"scrim: {scrim}"


def test_promptlet_sheet_stays_inside_the_chat_beside_the_docked_panels(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.wait_for_selector("#meta-panel", state="visible")
        panel = _rect(page, "#meta-panel")
        _open_sheet(page, "#tricks-btn", "#tricks-panel")
        _assert_inside_chat_pane(page, "#tricks-panel", "#tricks-overlay", panel["left"])
        # The scrim stops at the chat pane, so the task-info panel is
        # neither dimmed nor shielded: its refresh button still works.
        hit = page.evaluate(
            "() => { const b = document.getElementById('meta-spend-refresh');"
            " const r = b.getBoundingClientRect();"
            " return document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2)"
            " === b || b.contains(document.elementFromPoint(r.left + 2, r.top + 2)); }"
        )
        assert hit, "the Task Info panel must stay reachable while the sheet is open"
        # The sheet's own close button still closes it.
        page.locator("#tricks-panel-close").click()
        page.wait_for_selector("#tricks-panel:not(.open)", state="attached", timeout=5000)
    finally:
        context.close()


def test_promptlet_sheet_stays_in_the_chat_pane_beside_an_open_file(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        _open_file(page)
        handle = _rect(page, "#pane-resizer")
        _open_sheet(page, "#tricks-btn", "#tricks-panel")
        _assert_inside_chat_pane(page, "#tricks-panel", "#tricks-overlay", handle["left"])
        # The content pane is not under the scrim: a click there still
        # lands on the pane (it slides the task-info panel away, as
        # always), and the sheet stays open.
        content = _rect(page, "#content-tab-area")
        page.mouse.click(content["left"] + content["width"] / 2, content["top"] + 20)
        page.wait_for_selector("body.meta-hidden", state="attached", timeout=5000)
        assert page.locator("#tricks-panel.open").count() == 1
        page.locator("#tricks-overlay").click(position={"x": 10, "y": 10})
        page.wait_for_selector("#tricks-panel:not(.open)", state="attached", timeout=5000)
    finally:
        context.close()


def test_task_info_section_headers_are_a_fifth_shorter(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.wait_for_selector("#meta-panel", state="visible")
        bands = page.evaluate(_HEADER_BANDS_JS)
        ids = {b["id"] for b in bands}
        assert {"meta-section-info", "meta-explorer", "meta-schedule"} <= ids, bands
        for band in bands:
            old = _OLD_BUTTON_HEADER_BAND_PX if band["hasButton"] else _OLD_TEXT_HEADER_BAND_PX
            assert band["band"] <= old * 0.8 + 0.5, (
                f"{band['id']}: header band {band['band']:.1f}px is not a fifth "
                f"shorter than {old}px: {band}"
            )
            assert band["band"] >= old * 0.6, (
                f"{band['id']}: header band collapsed to {band['band']:.1f}px: {band}"
            )
        # The icon buttons set the row height: they shrank with it and
        # still hold their 14px glyph.
        button = page.evaluate(
            "() => { const b = document.getElementById('meta-schedule-refresh');"
            " const r = b.getBoundingClientRect(); const s = b.querySelector('svg')"
            ".getBoundingClientRect(); return {h: r.height, w: r.width, svg: s.height}; }"
        )
        assert button["h"] <= 18 * 0.8 + 0.5, button
        assert button["svg"] <= button["h"], button

        # Collapsed, the Schedule section keeps no gap under its header:
        # from its header's top to the next header's top there is only
        # the header row and the separator (9px line inside 2px margins).
        page.locator("#meta-schedule .meta-section-toggle").click()
        page.wait_for_selector("#meta-schedule.collapsed", state="attached", timeout=5000)
        collapsed = page.evaluate(
            "() => { const hdr = document.querySelector('#meta-schedule .meta-section-hdr');"
            " const next = document.querySelector('#meta-machines .meta-section-hdr');"
            " return {row: hdr.getBoundingClientRect().height,"
            " gap: parseFloat(getComputedStyle(hdr).marginBottom),"
            " pitch: next.getBoundingClientRect().top - hdr.getBoundingClientRect().top}; }"
        )
        assert collapsed["gap"] == 0, collapsed
        assert collapsed["pitch"] == pytest.approx(collapsed["row"] + 13, abs=1), collapsed
        assert collapsed["pitch"] <= (_OLD_BUTTON_HEADER_BAND_PX + 13) * 0.8, collapsed
        page.locator("#meta-schedule .meta-section-toggle").click()
        page.wait_for_selector("#meta-schedule:not(.collapsed)", state="attached", timeout=5000)
    finally:
        context.close()
