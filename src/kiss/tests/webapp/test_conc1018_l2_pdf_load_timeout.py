# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_scm_worktrees_and_actions are intentionally
#   shadowed by test parameters of the same name)
"""The PDF viewer gives up on a pdf.js download that stalls.

A CDN connection that neither answers nor drops used to leave the
viewer tab on "Loading…" until the browser's own network timeout,
minutes later; pdfView.js now aborts the load after 15 s and shows
its failure note, the way terminalTab.js already did for xterm.js.
A stall is produced with a ``page.route`` handler that never
fulfils the request (the real pdf.js module and worker URLs on
``cdn.jsdelivr.net``); everything else is a REAL headless Chromium
against a REAL :class:`RemoteAccessServer` — no mocks.
"""

from __future__ import annotations

import time

from kiss.tests.server.test_scm_worktrees_and_actions import (
    harness,  # noqa: F401  (module fixture used by param name)
    worktree,  # noqa: F401
)
from kiss.tests.webapp.test_sidebar_menus import (
    _PDF_VIEWER,
    _PDFJS_BUILD_GLOB,
    _explorer_row,
    _open_page,
    _open_report_pdf,
    _panel_click,
    _pdf_note,
    _route_pdfjs,
    _wait_pdf_rendered,
    browser,  # noqa: F401
)

# The viewer's own limit is 15 s; Chromium's network stack would wait
# far longer (its idle-connection timeout is measured in minutes).
_TIMEOUT_NOTE = "Cannot display report.pdf: Error: pdf.js did not load within 15 s"
_MAX_WAIT_S = 25


def _stall_file(name: str):
    """A ``_route_pdfjs`` decision that never answers requests for
    *name* and lets every other pdf.js download through."""

    def decide(route, n):
        if not route.request.url.rsplit("/", 1)[1].startswith(name):
            route.continue_()

    return decide


def _check_stall_reported_then_recovers(browser, harness, stalled: str) -> None:
    """Open report.pdf while *stalled* never downloads: the note names
    the timeout within the limit and the stalled file was asked for
    once (a timeout is not a dropped connection, so no retry).  Once
    the CDN answers again the next viewer draws the page: a timed-out
    load is forgotten like any other failure."""
    context, page, frames = _open_page(browser, harness)
    requests = _route_pdfjs(context, _stall_file(stalled))
    try:
        t0 = time.monotonic()
        _open_report_pdf(page)
        note = page.locator(_PDF_VIEWER + " div.content-binary-note")
        note.wait_for(timeout=_MAX_WAIT_S * 1000)
        assert time.monotonic() - t0 < _MAX_WAIT_S
        assert _pdf_note(page) == _TIMEOUT_NOTE
        assert requests.count(stalled) == 1
        context.unroute(_PDFJS_BUILD_GLOB)
        _panel_click(_explorer_row(page, "report.pdf"))
        note.wait_for(state="detached", timeout=15000)
        _wait_pdf_rendered(page)
        assert page.locator(_PDF_VIEWER + " div.content-binary-note").count() == 0
    finally:
        context.close()


def test_pdf_viewer_gives_up_on_a_stalled_worker_download(browser, harness, worktree):
    """The worker script never arrives (the module did)."""
    _check_stall_reported_then_recovers(browser, harness, "pdf.worker.min.mjs")


def test_pdf_viewer_gives_up_on_a_stalled_module_import(browser, harness, worktree):
    """The module import never settles (the worker arrived); a dynamic
    import() cannot be aborted, so the viewer stops waiting for it."""
    _check_stall_reported_then_recovers(browser, harness, "pdf.min.mjs")


# Records the MIME type of every blob: URL the page allocates from now
# on (the worker script's is text/javascript; pdf.js makes others of
# its own while drawing).
_RECORD_OBJECT_URLS = """() => {
  window.__objectUrls = [];
  const create = URL.createObjectURL.bind(URL);
  URL.createObjectURL = blob => { window.__objectUrls.push(blob.type); return create(blob); };
}"""


def test_pdf_viewer_timed_out_import_that_later_succeeds_allocates_no_worker_url(
    browser, harness, worktree
):
    """The module import of a timed-out load is held, not dropped, and
    released after the viewer gave up: the worker body had arrived in
    time, and the late import must not turn it into a blob: URL that
    no viewer will ever use or revoke (it would stay allocated for the
    life of the page, once per failed open).  The next open allocates
    exactly the one URL its viewer uses."""
    context, page, _frames = _open_page(browser, harness)
    held: list = []

    def hold_module(route, n):
        if route.request.url.rsplit("/", 1)[1].startswith("pdf.min.mjs"):
            held.append(route)
        else:
            route.continue_()

    _route_pdfjs(context, hold_module)
    try:
        _open_report_pdf(page)
        note = page.locator(_PDF_VIEWER + " div.content-binary-note")
        note.wait_for(timeout=_MAX_WAIT_S * 1000)
        assert _pdf_note(page) == _TIMEOUT_NOTE
        assert len(held) == 1
        page.evaluate(_RECORD_OBJECT_URLS)
        with page.expect_response(lambda r: "pdf.min.mjs" in r.url, timeout=30000):
            held[0].continue_()
        # Let the late import settle and its continuation run.
        page.wait_for_timeout(3000)
        assert page.evaluate("window.__objectUrls") == []
        context.unroute(_PDFJS_BUILD_GLOB)
        _panel_click(_explorer_row(page, "report.pdf"))
        note.wait_for(state="detached", timeout=15000)
        _wait_pdf_rendered(page)
        urls = page.evaluate("window.__objectUrls")
        assert urls.count("text/javascript") == 1, urls
    finally:
        context.close()
