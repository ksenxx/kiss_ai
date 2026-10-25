# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   the fixture parameter names below)
"""Pictures an agent names by file path show up in the real webapp.

A result's ``<img src="tmp/plot.png">`` used to render as a broken
image: the page's origin serves no files.  ``media/main.js`` now
fetches each such picture over the authenticated WebSocket with a
tokened ``openFile`` (``web_server._handle_open_file`` echoes the
token) and shows it as a ``data:`` URI; a previewed ``.html`` file
gets the same treatment, relative to its own directory.

Real daemon (``RemoteAccessServer``), real Chromium (Playwright):

* a result with a relative PNG, an absolute SVG and a missing file:
  the two pictures are decoded by the browser (``naturalWidth > 0``),
  the missing one shows its alt text and opens no tab;
* a clicked ``.html`` link renders in a preview frame whose picture,
  named relative to the file, is decoded inside the frame.
"""

from __future__ import annotations

import base64

import pytest

from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_result_ls_listing_file_links import (
    _deliver_result,
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)
_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
    '<rect width="40" height="20" fill="#36c"/></svg>\n'
)


def _write_pictures(harness) -> None:
    """A PNG, an SVG and a report naming a picture, in the work dir."""
    (harness.work_dir / "tmp").mkdir(exist_ok=True)
    (harness.work_dir / "tmp" / "plot.png").write_bytes(_PNG)
    (harness.work_dir / "tmp" / "d.svg").write_text(_SVG, encoding="utf-8")
    (harness.work_dir / "reports").mkdir(exist_ok=True)
    (harness.work_dir / "reports" / "fig.png").write_bytes(_PNG)
    (harness.work_dir / "reports" / "r.html").write_text(
        "<!DOCTYPE html><html><body><h1>Report</h1>"
        '<img id="fig" src="fig.png" alt="figure">'
        "<script>document.title = 'loaded'</script></body></html>\n",
        encoding="utf-8",
    )


@pytest.mark.usefixtures("harness")
class TestResultInlineImages:
    """Browser E2E: file-path pictures of results and previews load."""

    def test_result_pictures_load_and_missing_one_is_marked(
        self, browser, harness,
    ) -> None:
        _write_pictures(harness)
        context, page, sent = _open_page(browser, harness)
        try:
            svg_abs = str(harness.work_dir / "tmp" / "d.svg")
            _deliver_result(
                page,
                "<p>Plot: <img id=\"plot\" src=\"tmp/plot.png\" alt=\"plot\"> "
                f'<img id="svg" src="{svg_abs}" alt="svg"> '
                '<img id="gone" src="tmp/none.png" alt="gone"></p>',
            )
            page.wait_for_function(
                "() => { const i = document.getElementById('plot');"
                " return i && i.complete && i.naturalWidth > 0; }",
                timeout=30000,
            )
            page.wait_for_function(
                "() => { const i = document.getElementById('svg');"
                " return i && i.complete && i.naturalWidth > 0; }",
                timeout=30000,
            )
            page.wait_for_selector("#gone.inline-img-missing", timeout=30000)

            srcs = page.evaluate(
                "() => ['plot', 'svg', 'gone'].map(id =>"
                " document.getElementById(id).getAttribute('src'))"
            )
            assert srcs[0].startswith("data:image/png;base64,")
            assert srcs[1].startswith("data:image/svg+xml;base64,")
            assert srcs[2] is None
            assert page.evaluate(
                "() => document.getElementById('gone').title"
            ) == "Cannot show tmp/none.png"
            # The pictures were fetched as tokened openFile commands,
            # relative to the task's work dir, and opened no tab.
            opens = [f for f in sent if f.get("type") == "openFile"]
            assert sorted(f["path"] for f in opens) == sorted(
                ["tmp/plot.png", svg_abs, "tmp/none.png"],
            )
            assert all(f["token"].startswith("img-") for f in opens)
            assert page.evaluate(
                "() => window._testApi.openTabs()"
                ".filter(t => t.isContentTab).length"
            ) == 0
        finally:
            context.close()

    def test_html_preview_picture_loads_inside_the_frame(
        self, browser, harness,
    ) -> None:
        _write_pictures(harness)
        context, page, sent = _open_page(browser, harness)
        try:
            _deliver_result(
                page, "<p>See <code>reports/r.html</code></p>",
            )
            link = page.locator('#output .rc-body [data-path="reports/r.html"]')
            link.wait_for(state="visible", timeout=30000)
            link.click()
            page.wait_for_selector(".content-html-frame", timeout=30000)
            # The frame's document is parsed and serialized again with
            # the picture inlined; its script still runs.
            page.wait_for_function(
                "() => { const f = document.querySelector("
                "'.content-html-frame'); return f && f.srcdoc"
                ".includes('src=\"data:image/png;base64,'); }",
                timeout=30000,
            )
            frame = next(
                f for f in page.frames
                if f != page.main_frame and "Report" in (f.content() or "")
            )
            frame.wait_for_function(
                "() => { const i = document.getElementById('fig');"
                " return i && i.complete && i.naturalWidth > 0"
                " && document.title === 'loaded'; }",
                timeout=30000,
            )
            opens = [f for f in sent if f.get("type") == "openFile"]
            assert [f["path"] for f in opens if "token" in f] == [
                str(harness.work_dir / "reports" / "fig.png"),
            ]
        finally:
            context.close()
