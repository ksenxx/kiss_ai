# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Page reads: the serialization step is bounded too, and its handle is released.

``WebUseTool._read_page`` bounds a read with ``wait_for_function``'s
driver-side deadline, then fetches the answer with ``JSHandle.json_value``,
a second renderer round trip that had no deadline at all.  A page whose
``document.title`` getter returns an object with a never-returning
enumerable getter passes the first, bounded wait (the getter returns at
once) and wedged the serialization forever: ``get_page_content``,
``go_to_url`` and every other title read hung the task until the browser
was killed from outside.  The serialization now runs under the read's
own deadline (``_cdp_call``), so the tool returns its usual error string.

The ``JSHandle`` of every read also stayed registered in the driver
(``json_value`` does not dispose it, and Python's garbage collector
never reaches a channel object the connection still owns): one leaked
handle per title, tree, viewport, challenge or liveness read, for the
life of the page.  Every temporary handle is now disposed with a bounded
round trip.

Real headless Chromium, real loopback pages, no doubles.
"""

from __future__ import annotations

import gc
import http.server
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

_PLAIN_PAGE = b"<html><head><title>plain page</title></head><body><p>hello</p></body></html>"
# document.title answers at once with an object whose only enumerable
# property spins forever when read: the bounded wait passes, the
# serialization of its value never returns.
_HOSTILE_SERIALIZATION_PAGE = (
    b"<html><head><title>real</title></head><body><p>hostile</p><script>"
    b'Object.defineProperty(document, "title", {get() { return {get x() { while (true) {} }}; }});'
    b"</script></body></html>"
)
# Generous: each bounded read gives up after 10 s, plus browser start-up on slow CI.
_JOIN_TIMEOUT_S = 150


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        body = _HOSTILE_SERIALIZATION_PAGE if self.path.startswith("/hostile") else _PLAIN_PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


@pytest.fixture
def base() -> Iterator[str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _drive_hostile_serialization(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Every read of the hostile page is a bounded error; the browser recovers."""
    try:
        out["ok_tree"] = tool.go_to_url(f"{base}/plain")
        t0 = time.monotonic()
        out["hostile_nav"] = tool.go_to_url(f"{base}/hostile")
        out["hostile_nav_s"] = time.monotonic() - t0
        t0 = time.monotonic()
        out["hostile_text"] = tool.get_page_content(text_only=True)
        out["hostile_text_s"] = time.monotonic() - t0
        out["hostile_tabs"] = tool.go_to_url("tab:list")
        # The renderer is left spinning inside the getter for good: the
        # agent's documented escape hatch is close_browser().
        out["closed"] = tool.close_browser()
        out["recovered"] = tool.go_to_url(f"{base}/plain")
        out["done"] = True
    finally:
        tool.close()


def test_a_title_whose_serialization_hangs_is_a_bounded_error(tmp_path, base: str) -> None:
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive_hostile_serialization, args=(tool, base, out), daemon=True
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["ok_tree"]).startswith("Page: plain page\n")
        assert str(out["hostile_nav"]).startswith("Error navigating to ")
        assert "Timeout" in str(out["hostile_nav"])
        assert str(out["hostile_text"]).startswith("Error getting page content: ")
        assert "Timeout" in str(out["hostile_text"])
        assert float(out["hostile_nav_s"]) < 60  # type: ignore[arg-type]
        assert float(out["hostile_text_s"]) < 60  # type: ignore[arg-type]
        assert f"(unresponsive) - {base}/hostile (active)" in str(out["hostile_tabs"])
        assert str(out["closed"]).startswith("Browser closed")
        assert str(out["recovered"]).startswith("Page: plain page\n")
    finally:
        if worker.is_alive():
            tool.close()  # reap the wedged Chromium from here


def _live_js_handles(tool: WebUseTool) -> int:
    """How many JSHandle channel objects the tool's Playwright connection still owns."""
    gc.collect()
    objects = tool._playwright._impl_obj._connection._objects.values()
    return sum(1 for o in objects if getattr(o, "_type", None) == "JSHandle")


def _drive_reads(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Repeat every kind of temporary-handle read on one page; count the handles left."""
    try:
        out["tree"] = tool.go_to_url(f"{base}/plain")
        before = _live_js_handles(tool)
        for _ in range(20):
            assert tool._page_title(tool._page) == "plain page"
            assert tool._read_page(tool._page, "() => [innerWidth, innerHeight]")[0] > 0
            tool._require_responsive_renderer()
            assert str(tool.wait_for("hello", timeout=5)).startswith("Page: plain page\n")
            assert str(tool.scroll("down")).startswith("Page: plain page\n")  # liveness probe
        out["leaked"] = _live_js_handles(tool) - before
        out["done"] = True
    finally:
        tool.close()


def test_page_reads_leave_no_js_handles_behind(tmp_path, base: str) -> None:
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(target=_drive_reads, args=(tool, base, out), daemon=True)
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["tree"]).startswith("Page: plain page\n")
        assert out["leaked"] == 0, f"{out['leaked']} JSHandle channels left behind by the reads"
    finally:
        if worker.is_alive():
            tool.close()


if __name__ == "__main__":
    pytest.main([__file__])
