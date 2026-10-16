# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Every running-task spinner is the composer's wait spinner.

Two places show a "task is running" ring:

* the composer's ``#wait-spinner`` under the chat textbox;
* the ``.status-spinner`` at the left of every running row in the
  history panel and in the chat tab strip.

They are one CSS rule, so the tests compare their *computed* styles in
a real Chromium: size, ring thickness, one bright leading arc over a
faint track, and the very same keyframe animation.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.webapp.test_history_failed_red_cross import (
    _open_history_page,
    _post_history,
    _sample_sessions,
)


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


# The ring properties that must agree between the surfaces.  Colours are
# deliberately left out: each surface picks its own via `color`.
_RING_PROBE = """
(sel) => {
  const el = document.querySelector(sel);
  if (!el) return {error: 'no element ' + sel};
  const cs = getComputedStyle(el);
  const track = cs.borderBottomColor;
  return {
    width: cs.width,
    height: cs.height,
    borderRadius: cs.borderRadius,
    borderTopWidth: cs.borderTopWidth,
    borderRightWidth: cs.borderRightWidth,
    borderBottomWidth: cs.borderBottomWidth,
    borderLeftWidth: cs.borderLeftWidth,
    // A single leading arc: the top edge is bright, the other three
    // edges are the faint track.
    singleArc: cs.borderTopColor !== track &&
      cs.borderRightColor === track && cs.borderLeftColor === track,
    animationName: cs.animationName,
    animationDuration: cs.animationDuration,
    animationTimingFunction: cs.animationTimingFunction,
    animationIterationCount: cs.animationIterationCount,
    animationPlayState: cs.animationPlayState,
    opacity: cs.opacity,
    visible: el.offsetWidth > 0 && el.offsetHeight > 0,
  };
}
"""


def _ring(page, selector: str) -> dict:
    info = dict(page.evaluate(_RING_PROBE, selector))
    assert "error" not in info, info
    return info


def _wait_for_composer_spinner(page) -> None:
    """Wait until ``#wait-spinner`` is active (main.js shows it 250ms
    into a run) and has faded in (a 0.2s opacity transition)."""
    page.wait_for_function(
        "(el => el && getComputedStyle(el).opacity === '1')"
        "(document.querySelector('#wait-spinner.active'))",
        timeout=5000,
    )


def _assert_same_ring(a: dict, b: dict) -> None:
    for key in (
        "width", "height", "borderRadius",
        "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
        "animationName", "animationDuration", "animationTimingFunction",
        "animationIterationCount",
    ):
        assert a[key] == b[key], f"{key}: {a[key]!r} != {b[key]!r}\n{a}\n{b}"
    assert a["singleArc"] and b["singleArc"], (a, b)
    assert a["animationName"] not in ("none", ""), a
    assert a["animationIterationCount"] == "infinite", a


def test_history_row_spinner_is_the_composer_wait_spinner(_browser) -> None:
    """The running row's ring and the active ``#wait-spinner`` share
    every geometric and animation property and both spin."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        # The composer's spinner turns while the chat's task runs.
        page.evaluate(
            "() => window.__post({type: 'status', running: true,"
            " tabId: window._testApi.getActiveTabId(), startTs: 1700000000000,"
            " taskId: 'task-x'})"
        )
        _wait_for_composer_spinner(page)
        wait = _ring(page, "#wait-spinner.active")
        row = _ring(page, ".sidebar-item.running-item > .sidebar-item-running.status-spinner")
        _assert_same_ring(wait, row)
        assert wait["width"] == "12px" and wait["height"] == "12px", wait
        assert wait["borderTopWidth"] == "2px", wait
        assert wait["animationDuration"] == "0.8s", wait
        assert wait["animationTimingFunction"] == "linear", wait
        assert wait["animationPlayState"] == "running", wait
        assert row["animationPlayState"] == "running", row
        assert wait["opacity"] == "1" and wait["visible"], wait
        assert row["visible"], row
    finally:
        context.close()


# Freeze the running row's spinner at an instant of its animation and
# report where its box is.  The ring is rotated by the `rotate` property;
# a ring that turns in place keeps its bounding-box centre fixed, one
# that orbits (an offset applied through `transform`, which the browser
# composes BEFORE `rotate`) moves it around the offset's origin.
_FRAME_PROBE = """
(ms) => {
  const el = document.querySelector(
    '.sidebar-item.running-item > .sidebar-item-running.status-spinner');
  const anim = el.getAnimations()[0];
  anim.pause();
  anim.currentTime = ms;
  const r = el.getBoundingClientRect();
  // The first line of the task text: the box of its first character.
  const text = el.parentElement.querySelector('.sidebar-item-text');
  const range = document.createRange();
  range.setStart(text.firstChild, 0);
  range.setEnd(text.firstChild, 1);
  const line = range.getBoundingClientRect();
  const cs = getComputedStyle(el);
  return {
    cx: (r.left + r.right) / 2, cy: (r.top + r.bottom) / 2,
    rotate: cs.rotate, transform: cs.transform,
    firstLineCentre: (line.top + line.bottom) / 2,
  };
}
"""


def test_history_row_spinner_turns_in_place(_browser) -> None:
    """The running row's ring rotates about its own centre: frozen at
    four instants of one turn, the ring's box stays put (and stays
    centred on the first line of the task text) while its rotation
    angle changes."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        frames = [dict(page.evaluate(_FRAME_PROBE, ms)) for ms in (0, 200, 400, 600)]
    finally:
        context.close()
    angles = {f["rotate"] for f in frames}
    assert len(angles) == 4, f"the animation was not sampled at distinct angles: {frames}"
    for f in frames:
        assert f["transform"] == "none", f"an offset through `transform` makes it orbit: {f}"
    first = frames[0]
    for f in frames[1:]:
        assert abs(f["cx"] - first["cx"]) < 0.01 and abs(f["cy"] - first["cy"]) < 0.01, (
            f"the ring revolves instead of rotating: {frames}"
        )
    assert abs(first["cy"] - first["firstLineCentre"]) < 1.0, (
        f"the ring is not centred on the first text line: {first}"
    )


def test_tab_strip_spinner_is_the_composer_wait_spinner(_browser) -> None:
    """The chat tab strip's running icon is the same ring too."""
    context, page = _open_history_page(_browser)
    try:
        page.evaluate(
            "() => window.__post({type: 'status', running: true,"
            " tabId: window._testApi.getActiveTabId(), startTs: 1700000000000,"
            " taskId: 'task-y'})"
        )
        _wait_for_composer_spinner(page)
        _assert_same_ring(
            _ring(page, "#wait-spinner.active"),
            _ring(page, ".chat-tab-spinner.status-spinner"),
        )
    finally:
        context.close()

