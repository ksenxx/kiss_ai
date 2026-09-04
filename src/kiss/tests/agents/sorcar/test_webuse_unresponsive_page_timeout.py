# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: web tools must not hang forever on an unresponsive page.

A live task froze inside ``get_page_content(text_only=True)`` right after
``go_to_url`` had failed with ``Page.goto: Timeout 30000ms exceeded``.
The page's main thread was stuck, and Playwright's ``Page.title()`` has
no timeout: it waits for the frame's execution context to answer, which
a stuck renderer never does. The tool call never returned and the task
sat idle for good.

The test serves a real page whose script spins forever right after
``DOMContentLoaded`` and drives a real headless Chromium through
``WebUseTool`` from a worker thread; every tool call must come back
(with an error string) well before the worker-thread join deadline.
"""

import http.server
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.web_use_tool import WebUseTool

_STUCK_PAGE = (
    b"<html><head><title>stuck page</title></head><body><p>hello</p>"
    b"<script>setTimeout(function () { while (true) {} }, 0);</script>"
    b"</body></html>"
)
_OK_PAGE = (
    b"<html><head><title>fine page</title></head><body><p>hello</p>"
    b'<a href="/stuck">go</a></body></html>'
)
# A top-level SVG document has an <svg> root, not <html>: the title must
# be read through the document root, not a hard-coded "html" selector.
_SVG_PAGE = b'<svg xmlns="http://www.w3.org/2000/svg"><title>svg title</title></svg>'

# The page is fully responsive except that reading document.title runs a
# never-returning getter.  A locator-based title read is only bounded
# during element resolution, so it hangs here; the whole-operation
# deadline of wait_for_function must turn this into a TimeoutError.
_HOSTILE_TITLE_PAGE = (
    b"<html><head><title>real</title></head><body><p>hostile</p><script>"
    b'Object.defineProperty(document, "title", {get() { while (true) {} }});'
    b"</script></body></html>"
)
# Healthy pages whose event handlers wedge the renderer in response to
# our own raw input: the pre-input liveness probe passes, then the
# timeout-less keyboard/mouse call blocks until the input watchdog kills
# Chromium and the pending call raises.
_KEYWEDGE_PAGE = (
    b"<html><head><title>keywedge</title></head><body><p>k</p><script>"
    b'addEventListener("keydown", function () { while (true) {} });'
    b"</script></body></html>"
)
_INPUTWEDGE_PAGE = (
    b"<html><head><title>inputwedge</title></head><body>"
    b'<input aria-label="box"><script>'
    b'addEventListener("keydown", function () { while (true) {} });'
    b"</script></body></html>"
)
_WHEELWEDGE_PAGE = (
    b"<html><head><title>wheelwedge</title></head><body><p>w</p><script>"
    b'addEventListener("mousemove", function () { while (true) {} });'
    b"</script></body></html>"
)

# Each bounded read gives up after 10 s (``_PAGE_READ_TIMEOUT_MS``); the
# join deadline leaves generous slack for browser start-up on slow CI.
_JOIN_TIMEOUT_S = 120


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.startswith("/stuck"):
            body, ctype = _STUCK_PAGE, "text/html"
        elif self.path.startswith("/svg"):
            body, ctype = _SVG_PAGE, "image/svg+xml"
        elif self.path.startswith("/hostile-title"):
            body, ctype = _HOSTILE_TITLE_PAGE, "text/html"
        elif self.path.startswith("/keywedge"):
            body, ctype = _KEYWEDGE_PAGE, "text/html"
        elif self.path.startswith("/inputwedge"):
            body, ctype = _INPUTWEDGE_PAGE, "text/html"
        elif self.path.startswith("/wheelwedge"):
            body, ctype = _WHEELWEDGE_PAGE, "text/html"
        else:
            body, ctype = _OK_PAGE, "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


def _serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _drive(
    tool: WebUseTool,
    base: str,
    out: dict[str, object],
    calls: Callable[[WebUseTool, str, dict[str, object]], None] | None = None,
) -> None:
    """Run the tool calls and close the tool on one thread.

    Playwright's sync API is bound to the thread that started it, so the
    orderly ``close()`` must happen here too; the test thread only falls
    back to a cross-thread close when this worker never finishes.
    """
    try:
        (calls or _drive_calls)(tool, base, out)
    finally:
        tool.close()


def _drive_calls(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    out["ok_tree"] = tool.go_to_url(f"{base}/ok")
    out["ok_text"] = tool.get_page_content(text_only=True)
    # The aria snapshot of an SVG document fails (it has no <body>; a
    # pre-existing limit), but its title must still be read via the root.
    tool.go_to_url(f"{base}/svg")
    out["svg_tabs"] = tool.go_to_url("tab:list")
    t0 = time.monotonic()
    out["stuck_nav"] = tool.go_to_url(f"{base}/stuck")
    out["stuck_nav_s"] = time.monotonic() - t0
    t0 = time.monotonic()
    out["stuck_text"] = tool.get_page_content(text_only=True)
    out["stuck_text_s"] = time.monotonic() - t0
    out["tabs"] = tool.go_to_url("tab:list")
    # Interaction tools: keyboard/mouse/count calls have no Playwright
    # timeout of their own, so each must be guarded by the liveness probe.
    # click(1) resolves the link remembered from the healthy page's tree.
    out["click"] = tool.click(1)
    out["press"] = tool.press_key("End")
    out["scroll"] = tool.scroll("down", 1)
    out["done"] = True


def test_tools_return_errors_instead_of_hanging_on_stuck_page(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(target=_drive, args=(tool, base, out), daemon=True)
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"

        # Sanity: a responsive page is read through the same bounded path.
        assert str(out["ok_tree"]).startswith("Page: fine page\n")
        assert '[1] link "go"' in str(out["ok_tree"])
        assert str(out["ok_text"]).startswith("Page: fine page\n")
        assert "hello" in str(out["ok_text"])
        assert f"[0] svg title - {base}/svg (active)" in str(out["svg_tabs"])

        # The stuck page: navigation reports an error (its read timed out)
        # and get_page_content returns an error string instead of hanging.
        assert str(out["stuck_nav"]).startswith("Error navigating to ")
        assert "Timeout" in str(out["stuck_nav"])
        assert str(out["stuck_text"]).startswith("Error getting page content: ")
        assert "Timeout" in str(out["stuck_text"])
        assert float(out["stuck_nav_s"]) < 60  # type: ignore[arg-type]
        assert float(out["stuck_text_s"]) < 60  # type: ignore[arg-type]

        # Listing tabs still works while the active tab is unresponsive.
        tabs = str(out["tabs"])
        assert tabs.startswith("Open tabs (")
        assert "(unresponsive)" in tabs
        assert f"{base}/stuck (active)" in tabs

        # Interaction tools fail fast with the liveness probe's timeout.
        assert str(out["click"]).startswith("Error clicking element 1: ")
        assert "Timeout" in str(out["click"])
        assert str(out["press"]).startswith("Error pressing key 'End': ")
        assert "Timeout" in str(out["press"])
        assert str(out["scroll"]).startswith("Error scrolling down: ")
        assert "Timeout" in str(out["scroll"])
    finally:
        if worker.is_alive():
            # The worker is wedged inside Playwright: reap Chromium from
            # here (the tool kills the recorded browser PID directly).
            tool.close()
        srv.shutdown()
        srv.server_close()


def _drive_hostile_title(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Hostile ``document.title`` getter: every read is a bounded error."""
    out["ok_tree"] = tool.go_to_url(f"{base}/ok")
    t0 = time.monotonic()
    out["hostile_nav"] = tool.go_to_url(f"{base}/hostile-title")
    out["hostile_nav_s"] = time.monotonic() - t0
    t0 = time.monotonic()
    out["hostile_text"] = tool.get_page_content(text_only=True)
    out["hostile_text_s"] = time.monotonic() - t0
    out["hostile_tabs"] = tool.go_to_url("tab:list")
    # Evaluating the hostile getter left the renderer spinning forever
    # (the while(true) never exits), so the tab is permanently wedged —
    # the agent's documented escape hatch is close_browser(), after
    # which the next call relaunches a fresh browser.
    out["closed"] = tool.close_browser()
    out["recovered"] = tool.go_to_url(f"{base}/ok")
    out["done"] = True


def test_hostile_title_getter_times_out_instead_of_hanging(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive, args=(tool, base, out), kwargs={"calls": _drive_hostile_title},
        daemon=True,
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["ok_tree"]).startswith("Page: fine page\n")
        assert str(out["hostile_nav"]).startswith("Error navigating to ")
        assert "Timeout" in str(out["hostile_nav"])
        assert str(out["hostile_text"]).startswith("Error getting page content: ")
        assert "Timeout" in str(out["hostile_text"])
        assert float(out["hostile_nav_s"]) < 60  # type: ignore[arg-type]
        assert float(out["hostile_text_s"]) < 60  # type: ignore[arg-type]
        tabs = str(out["hostile_tabs"])
        assert tabs.startswith("Open tabs (")
        assert f"(unresponsive) - {base}/hostile-title (active)" in tabs
        assert str(out["closed"]).startswith("Browser closed")
        assert str(out["recovered"]).startswith("Page: fine page\n")
    finally:
        if worker.is_alive():
            tool.close()
        srv.shutdown()
        srv.server_close()


def _drive_input_wedge(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Handlers that wedge the renderer on our input: watchdog unblocks the call.

    Each wedged raw input call is unblocked by the input watchdog killing
    Chromium; the next ``go_to_url`` must transparently relaunch a fresh
    browser, proving the tool recovers after every kill.
    """
    out["key_tree"] = tool.go_to_url(f"{base}/keywedge")
    t0 = time.monotonic()
    out["press"] = tool.press_key("End")
    out["press_s"] = time.monotonic() - t0
    out["input_tree"] = tool.go_to_url(f"{base}/inputwedge")
    t0 = time.monotonic()
    out["type"] = tool.type_text(1, "x")
    out["type_s"] = time.monotonic() - t0
    out["wheel_tree"] = tool.go_to_url(f"{base}/wheelwedge")
    t0 = time.monotonic()
    out["scroll"] = tool.scroll("down", 1)
    out["scroll_s"] = time.monotonic() - t0
    out["recovered"] = tool.go_to_url(f"{base}/ok")
    out["done"] = True


def test_input_wedging_handlers_are_unblocked_by_watchdog(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    # A tool that has not launched a browser has no recorded PID: the
    # watchdog guard degrades to a no-op context (nullcontext branch).
    with tool._input_hang_watchdog():
        pass
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive, args=(tool, base, out), kwargs={"calls": _drive_input_wedge},
        daemon=True,
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S + 60)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["key_tree"]).startswith("Page: keywedge\n")
        assert str(out["press"]).startswith("Error pressing key 'End': ")
        assert float(out["press_s"]) < 60  # type: ignore[arg-type]
        assert str(out["input_tree"]).startswith("Page: inputwedge\n")
        assert '[1] textbox "box"' in str(out["input_tree"])
        assert str(out["type"]).startswith("Error typing into element 1: ")
        assert float(out["type_s"]) < 60  # type: ignore[arg-type]
        assert str(out["wheel_tree"]).startswith("Page: wheelwedge\n")
        assert str(out["scroll"]).startswith("Error scrolling down: ")
        assert float(out["scroll_s"]) < 60  # type: ignore[arg-type]
        assert str(out["recovered"]).startswith("Page: fine page\n")
    finally:
        if worker.is_alive():
            tool.close()
        srv.shutdown()
        srv.server_close()
