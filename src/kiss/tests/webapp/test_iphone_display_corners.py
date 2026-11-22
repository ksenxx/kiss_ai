# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the composer's bottom corners follow an iPhone's display corners.

``media/displayCorners.js`` (inlined in ``<head>`` by ``_build_html``) acts
only on a listed iPhone that runs the page from its Home Screen icon
(``navigator.standalone``).  Held upright, the page asks for the whole
screen (``viewport-fit=cover``) and ``<html>`` carries ``display-corners``;
``remote-codex.css`` then keeps ONE gap between the composer box and the
left, right and bottom edge of the glass and rounds the box's bottom
corners by the display's radius less that gap, so the two curves share a
centre.  Held sideways, in a browser tab or on a screen that is not
listed, the page is left as it was.

Real ``RemoteAccessServer`` + headless Chromium made to look like the
phone: its screen size and pixel ratio, an iPhone user agent,
``navigator.standalone`` defined by an init script (only iOS has it) and
the home indicator's 34px bottom safe-area inset set through the DevTools
protocol.  What Chromium cannot show is WebKit's own answer to
``viewport-fit=cover``; the tests check that the page asks for it.
"""

from __future__ import annotations

import pytest

from kiss.tests.conftest import goto_retrying_network_change, mark_version_seen
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_phone_machine_dropdown import _IPHONE_UA
from kiss.tests.webapp.test_workspace_sections import (
    browser,  # noqa: F401  (module fixture used by param name)
)

_HOME_INDICATOR_INSET = 34
_STANDALONE_JS = (
    "Object.defineProperty(Navigator.prototype, 'standalone', {get() { return true; }});"
)
_FOLLOWS_JS = "() => document.documentElement.classList.contains('display-corners')"
_GEOMETRY_JS = """() => {
  const box = document.getElementById('input-container');
  const style = getComputedStyle(box);
  const rect = box.getBoundingClientRect();
  const footer = document.getElementById('input-footer').getBoundingClientRect();
  return {
    viewport: document.querySelector('meta[name=viewport]').content,
    follows: document.documentElement.classList.contains('display-corners'),
    left: rect.left,
    right: innerWidth - rect.right,
    bottom: innerHeight - rect.bottom,
    height: rect.height,
    top: [parseFloat(style.borderTopLeftRadius), parseFloat(style.borderTopRightRadius)],
    low: [parseFloat(style.borderBottomLeftRadius), parseFloat(style.borderBottomRightRadius)],
    footerAbove: innerHeight - footer.bottom,
  };
}"""


def _open_phone(browser, harness, width, height, ratio, standalone):  # noqa: F811
    """The page as an iPhone with that screen shows it, held upright;
    returns ``(context, page)``."""
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": width, "height": height},
        screen={"width": width, "height": height},
        device_scale_factor=ratio, is_mobile=True, has_touch=True, user_agent=_IPHONE_UA,
    )
    mark_version_seen(context)
    if standalone:
        context.add_init_script(_STANDALONE_JS)
    page = context.new_page()
    context.new_cdp_session(page).send(
        "Emulation.setSafeAreaInsetsOverride", {"insets": {"bottom": _HOME_INDICATOR_INSET}}
    )
    goto_retrying_network_change(page, harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    return context, page


@pytest.mark.parametrize(
    ("width", "height", "ratio", "radius"),
    [(402, 874, 3, 62), (414, 896, 2, 41.5)],
    ids=["iphone-17", "iphone-11"],
)
def test_composer_follows_display_corners_of_home_screen_app(
    browser, harness, width, height, ratio, radius  # noqa: F811
):
    """Upright, the box is concentric with the glass; sideways it is ordinary."""
    context, page = _open_phone(browser, harness, width, height, ratio, standalone=True)
    try:
        upright = page.evaluate(_GEOMETRY_JS)
        assert upright["follows"] is True, upright
        assert upright["viewport"].endswith(",viewport-fit=cover"), upright
        gap = upright["left"]
        assert gap > 0, upright
        assert upright["right"] == pytest.approx(gap, abs=0.01), upright
        assert upright["bottom"] == pytest.approx(gap, abs=0.01), upright
        assert upright["low"] == pytest.approx([radius - gap] * 2, abs=0.01), upright
        assert upright["footerAbove"] >= _HOME_INDICATOR_INSET, upright

        # Only the button row shows: the box stays tall enough for its
        # corners (a lower box gets all its radii scaled down), and the
        # buttons stay above the home indicator.
        page.click("#input-drawer-btn")
        page.wait_for_selector("#input-area.drawer-collapsed")
        collapsed = page.evaluate(_GEOMETRY_JS)
        assert collapsed["low"] == upright["low"], collapsed
        assert collapsed["height"] >= collapsed["low"][0] + collapsed["top"][0], collapsed
        assert collapsed["bottom"] == pytest.approx(gap, abs=0.01), collapsed
        assert collapsed["footerAbove"] >= _HOME_INDICATOR_INSET, collapsed
        page.click("#input-drawer-btn")
        page.wait_for_selector("#input-area:not(.drawer-collapsed)")

        page.set_viewport_size({"width": height, "height": width})
        page.wait_for_function(f"() => !({_FOLLOWS_JS})()")
        sideways = page.evaluate(_GEOMETRY_JS)
        assert sideways["viewport"] + ",viewport-fit=cover" == upright["viewport"], sideways
        assert sideways["low"] == sideways["top"] == upright["top"], sideways

        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_function(_FOLLOWS_JS)
        assert page.evaluate(_GEOMETRY_JS) == upright
    finally:
        context.close()


@pytest.mark.parametrize(
    ("width", "height", "standalone"),
    [(402, 874, False), (360, 800, True)],
    ids=["browser-tab", "unlisted-screen"],
)
def test_page_is_left_alone_elsewhere(browser, harness, width, height, standalone):  # noqa: F811
    """A browser tab and an unlisted screen keep the ordinary composer,
    which stays above the home indicator as a whole."""
    context, page = _open_phone(browser, harness, width, height, 3, standalone)
    try:
        seen = page.evaluate(_GEOMETRY_JS)
        assert seen["follows"] is False, seen
        assert "viewport-fit" not in seen["viewport"], seen
        assert seen["low"] == seen["top"], seen
        assert seen["bottom"] >= _HOME_INDICATOR_INSET, seen
    finally:
        context.close()
