# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — closing the active tab nukes the whole browser session.

When only the *active tab* is closed (user clicks the tab's ✕, or the
page calls ``window.close()``) while the context and other tabs are
still alive, ``_ensure_browser`` treats the tool as dead and tears down
the ENTIRE browser (closing every surviving tab, discarding the
context) before relaunching a blank one.  The correct recovery — as
``_check_for_new_tab`` already does for openings — is to adopt a
surviving tab from the still-live context.
"""

from pathlib import Path

from kiss.agents.sorcar.web_use_tool import WebUseTool

PAGE_A = """<!DOCTYPE html>
<html><head><title>Page A</title></head>
<body><h1>Alpha content</h1><a href="#">alpha link</a></body></html>"""

PAGE_B = """<!DOCTYPE html>
<html><head><title>KeepMe</title></head>
<body><h1>Survivor content</h1><a href="#">survivor link</a></body></html>"""


def test_closing_active_tab_adopts_surviving_tab(tmp_path: Path) -> None:
    a = tmp_path / "a.html"
    a.write_text(PAGE_A)
    b = tmp_path / "b.html"
    b.write_text(PAGE_B)

    tool = WebUseTool(user_data_dir=None, headless=True)
    try:
        tool.go_to_url(a.as_uri())
        ctx = tool._context
        assert ctx is not None

        # A second, real tab in the same context (e.g. opened earlier
        # via a target=_blank link).
        survivor = ctx.new_page()
        survivor.goto(b.as_uri())

        # The user closes the ACTIVE tab out from under the tool.
        tool._page.close()

        out = tool.get_page_content()

        # The surviving tab (and the whole session) must be preserved…
        assert tool._context is ctx, "browser session was torn down"
        assert not survivor.is_closed(), "surviving tab was closed"
        # …and the tool must now operate on the surviving tab.
        assert "KeepMe" in out, out
    finally:
        tool.close()
