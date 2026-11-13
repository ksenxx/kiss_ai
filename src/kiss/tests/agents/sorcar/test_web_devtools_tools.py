# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for the DevTools-grade browser tools of WebUseTool.

Covers ``evaluate_script``, ``list_console_messages``,
``list_network_requests``, ``wait_for`` and ``attach_to_chrome`` against a
real Chromium and a local HTTP server — no mocks, patches, or test doubles.
The server page logs to the console, throws an uncaught error, issues
fetches, and mutates its DOM on a timer, so every tool observes real
browser behavior.

Branches not covered here, and why (unreachable without test doubles):

* The ``except Exception`` fallbacks of the event handlers (a driver
  event with a broken payload): they require sabotaging the Playwright
  driver itself.
* ``_try_ensure_browser``'s error return (Chromium missing/unlaunchable):
  same pre-existing pattern every other tool shares; installing a broken
  Chromium would destabilize the whole suite.
* ``attach_to_chrome``'s "no browser contexts found" branch: a real
  Chrome always has its default context.
* ``wait_for``'s non-timeout error branch: needs the page to die in the
  middle of ``wait_for_function`` at an uncontrolled moment.
* ``_close_browser_only``'s disconnect-exception branch: the user's
  browser exiting inside the disconnect call.
* ``evaluate_script``'s "no CDP session" branch: console capture would
  have had to fail at adoption while the page still answers later reads —
  not arrangeable deterministically.
"""

from __future__ import annotations

import http.server
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.sorcar_agent import BROWSER_TOOL_NAMES
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.tests.conftest import posix_only

_PAGE = b"""<html><head><title>devtools-test</title></head><body>
<button id="go" aria-label="Go">Go</button>
<p id="ephemeral">Ephemeral notice</p>
<script>
console.log("page boot");
console.error("boot error");
fetch("/api/data");
setTimeout(function () { throw new Error("uncaught boom"); }, 30);
setTimeout(function () {
  var e = document.getElementById("ephemeral");
  if (e) e.remove();
}, 300);
document.getElementById("go").addEventListener("click", function () {
  console.warn("button clicked");
  fetch("/api/more");
  setTimeout(function () {
    var p = document.createElement("p");
    p.textContent = "Loaded!";
    document.body.appendChild(p);
  }, 200);
});
</script></body></html>"""

_PLAIN_PAGE = b"<html><head><title>plain</title></head><body><p>nothing here</p></body></html>"

# The first "Loading" copy is hidden; only the second is visible.
_DUP_PAGE = (
    b"<html><head><title>dup</title></head><body>"
    b'<p style="display:none">Loading</p><p>Loading</p>'
    b"</body></html>"
)

# Two visible "Loading" copies, removed/hidden on demand from the test
# (timers would race go_to_url's settling wait).  hideFirst keeps the
# node in the DOM: a first-match-only implementation must mistake its
# hiding for the text disappearing while the second copy stays visible.
_DUPSTAY_PAGE = (
    b"<html><head><title>dupstay</title></head><body>"
    b'<p id="a">Loading</p><p id="b">Loading</p><script>'
    b'function hideFirst() { document.getElementById("a").style.display = "none"; }'
    b'function removeSecond() { document.getElementById("b").remove(); }'
    b"</script></body></html>"
)

# Text living in an open shadow root (wrapped AND directly under the
# root), a submit input whose label is its value attribute, and an input
# suppressed by content-visibility:hidden (must count as NOT present).
_SHADOW_PAGE = (
    b"<html><head><title>shadow</title></head><body>"
    b'<div id="host"></div><div id="host2"></div><div id="host3"></div><div id="host4"></div>'
    b'<div id="host5">UnslottedLight</div>'
    b'<div id="host6"><span>Slotted</span></div>'
    b'<div id="host7"><span>SlotSkipped</span></div>'
    b'<div id="host8">SlotInvisible</div>'
    b'<div id="host9"><span>Sp</span></div>'
    b'<input type="submit" value="SubmitReady">'
    b'<div style="content-visibility:hidden"><input value="ContentHidden"></div>'
    b"<script>"
    b'document.getElementById("host").attachShadow({mode: "open"})'
    b".innerHTML = '<p>ShadowReady</p>';"
    b'document.getElementById("host2").attachShadow({mode: "open"})'
    b".textContent = 'DirectReady';"
    b'document.getElementById("host3").attachShadow({mode: "open"})'
    b".innerHTML = '<b>Sha</b>dowSplit';"
    b'document.getElementById("host4").attachShadow({mode: "open"})'
    b".innerHTML = '<svg width=\"120\" height=\"40\">"
    b"<text x=\"5\" y=\"25\">ShadowVector</text></svg>';"
    b'document.getElementById("host5").attachShadow({mode: "open"})'
    b".textContent = 'ShadowOnly';"
    b'document.getElementById("host6").attachShadow({mode: "open"})'
    b".innerHTML = '<slot></slot> successfully';"
    b'document.getElementById("host7").attachShadow({mode: "open"})'
    b".innerHTML = '<div style=\"content-visibility:hidden\"><slot></slot></div>';"
    b'document.getElementById("host8").attachShadow({mode: "open"})'
    b".innerHTML = '<div style=\"visibility:hidden\"><slot></slot></div>';"
    b'document.getElementById("host9").attachShadow({mode: "open"})'
    b".innerHTML = '<p><slot></slot>liced</p>';"
    b"</script></body></html>"
)


# Text split across inline markup (a highlighted search hit, a partly
# bold word): one text node holds "Sa", the next "ved".  A hidden copy
# split the same way, a split inside <script>, and mixed visible/hidden
# content (a hidden fragment completing a word; a hidden duplicate next
# to a visible copy) must read as they render.
_SPLIT_PAGE = (
    b'<html><head><meta charset="utf-8"><title>split</title></head><body>'
    b'<p id="msg"><b>Sa</b>ved</p>'
    b'<p style="display:none"><b>Hid</b>den copy</p>'
    b'<div><p><span>Nes</span>ted</p></div>'
    b'<script>var s = "Scripted";</script>'
    b'<p><span hidden>Pa</span>rtly</p>'
    b'<p id="twin">Twin<span hidden>Twin</span></p>'
    b'<table><tr><td>Ta</td><td>ble</td></tr></table>'
    b'<select><option>Chosen</option><option hidden>Concealed</option></select>'
    b'<svg width="120" height="40"><text x="5" y="25">VectorText</text></svg>'
    b'<p>Li<br>ne</p>'
    b'<p>Spaced   out</p>'
    b'<p>Wo<span hidden>xx</span>rd</p>'
    b'<p><span style="display:contents">Contents</span></p>'
    b'<textarea id="area">OldDefault</textarea>'
    b"<p>Changes\nsaved</p>"
    b"<p>Message sent \xf0\x9f\x91\x8d</p>"
    b'<p style="white-space:pre-line">Preline   kept</p>'
    b"<p>Hard&nbsp;space</p>"
    b"<p>Soft<br hidden> break</p>"
    b'<select><option label="Completed">Internal</option></select>'
    b"<pre>Pre <b> formatted</b></pre>"
    b'<input type="checkbox" value="TickValue"><input type="radio" value="DotValue">'
    b'<input type="password" value="UnrenderedSecret"><input type="submit" value="Send it">'
    b'<input type="hidden" value="HiddenField">'
    b'<table><tr style="visibility:collapse"><td>CollapsedRow</td></tr>'
    b"<tr><td>ShownRow</td></tr></table>"
    b'<p style="visibility:collapse">CollapsedPara</p>'
    b"</body></html>"
)

# Text a stylesheet displays in another case than the page spells it.
_CASED_PAGE = (
    '<html><head><meta charset="utf-8"><title>cased</title><style>'
    ".up{text-transform:uppercase} .cap{text-transform:capitalize}"
    "</style></head><body>"
    '<p class="up" id="note">saved fine</p>'
    '<p>Changes <span class="up">stored</span> ok</p>'
    "<p class=\"cap\"><b>he</b>llo wide it's e.g.the world</p>"
    '<p style="text-transform:lowercase">SHOUT Text</p>'
    '<input class="up" value="field value">'
    '<div lang="tr"><p class="up">iptal</p></div>'
    '<p class="up" lang="de">schlie\u00dfen</p>'
    '<p class="up" lang="el">\u0391\u03c0\u03bf\u03b8\u03ae\u03ba\u03b5\u03c5\u03c3\u03b7</p>'
    '<p class="up" lang="not a tag">odd tongue</p>'
    # Letters whose other case is two letters, or another letter than the one mapped back.
    '<p class="up" lang="el">\u1fb3\u03b4\u03c9</p>'
    '<p style="text-transform:lowercase">STRA\u1e9eE</p>'
    # What Chromium does not map as String.toUpperCase does: a word's first letter goes to
    # title case, Georgian has no upper case, and a three-letter language tag is not Turkish.
    '<p class="cap">\u01c6uro \u1fb3\u03b4\u03c9</p>'
    '<p class="up">\u10d2\u10d0\u10db\u10d0</p>'
    '<p class="up" lang="tur">bitir</p>'
    "</body></html>"
).encode()


# Text a stylesheet draws before or after an element: the accessibility tree shows it,
# and so does a screenshot.
_DRAWN_PAGE = (
    '<html><head><meta charset="utf-8"><title>drawn</title><style>'
    '.st::after { content: " DONE"; }'
    ".state::before { content: attr(data-state); }"
    '.note:before { content: "Note"; display: block; }'  # as CSS 2 wrote it
    '.up::after { content: "new"; text-transform: uppercase; }'
    # An arrow behind a link whose address holds what reads like the end of the rule's
    # selector, on a button that draws a zero-width space, as GitHub's do.
    'a[href$="::after,"]::after { content: "\\2197"; } .zw::before { content: "\\200B"; }'
    # Selectors with an escape and with no element of their own; a string with escapes.
    '.sm\\:tw::after { content: "TwDrawn"; } #kids > ::after { content: "!"; }'
    '.esc::after { content: "Say \\"hi\\"\\a now"; }'
    '@media screen { .media::before { content: "MediaDrawn"; } }'
    '.gone::before { content: "GoneDrawn"; display: none; }'
    '.unseen::before { content: "UnseenDrawn"; visibility: hidden; }'
    '.off { display: none; } .off::before { content: "OffDrawn"; }'
    'body::before { content: "BannerDrawn"; display: block; }'
    'html::after { content: "RootDrawn"; display: block; }'
    # Of the elements that hold no boxes of their own, some draw: an image that failed,
    # a date field, a restyled checkbox.
    'img::after { content: attr(alt) "Drawn"; } svg::before { content: "SvgDrawn"; }'
    'input::before, textarea::before { content: attr(name) "Drawn"; }'
    '.alt::before { content: "Glyph" / "AltDrawn"; }'
    '.count { counter-increment: n; } .count::before { content: counters(n, "+") ". "; }'
    # An icon font's glyph is no text, alone or before some.
    '.icon::before { content: "\\f007"; } .badge::after { content: "\\f00c  Done"; }'
    "</style></head><body>"
    '<p><span class="st">Status</span></p>'
    '<p><span class="state" data-state="Synced"></span></p>'
    '<div class="note">body text</div>'
    '<p class="up">release</p>'
    '<p>Open <a class="zw" href="#::after,">Docs</a> page</p>'
    '<p class="sm:tw">t</p><p id="kids"><b>x</b><i>y</i></p><p class="esc"></p>'
    '<p class="media">m</p><p class="gone">g</p><p class="unseen">u</p><p class="off">o</p>'
    '<img alt="Loaded" src="data:image/gif;base64,'
    'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">'
    '<img alt="Failed" src="data:image/gif;base64,AAAA">'
    '<input name="Field"><input name="When" type="date"><textarea name="Area"></textarea>'
    '<svg width="8" height="8"></svg>'
    '<input name="Box" type="checkbox"><input name="Tick" type="checkbox" style="appearance:none">'
    '<p class="alt">a</p><ol><li class="count">first</li></ol>'
    '<p>My <i class="icon"></i>Profile</p><p>All<span class="badge"></span></p>'
    '<details><summary>More</summary><p class="st">Folded</p></details>'
    '<div class="st" style="display:contents">Flat</div>'
    '<div id="host"><span>lit</span></div><x-part id="part"></x-part><p class="card">c</p>'
    '<p id="far">f<span id="none"></span></p><p class="adopt">d</p>'
    "<script>"
    'const shadow = document.getElementById("host").attachShadow({mode: "open"});'
    "shadow.innerHTML ="
    " '<style>:host::before { content: \"HostDrawn\"; } p::after { content: \"InnerDrawn\"; }"
    " ::slotted(span)::after { content: \"SlotDrawn\"; }</style>'"
    " + '<p>in</p><b class=\"adopt\">sh</b><slot></slot>';"
    # One sheet for the document and for a shadow tree, as component libraries share theirs.
    "const shared = new CSSStyleSheet();"
    "shared.replaceSync('.adopt::after { content: \"AdoptedDrawn\"; }');"
    "document.adoptedStyleSheets = [shared]; shadow.adoptedStyleSheets = [shared];"
    'document.getElementById("part").attachShadow({mode: "open"}).innerHTML ='
    " '<b part=\"label\">parted</b>';"
    "</script></body></html>"
)

# What the page of another origin cannot read: the rules of this sheet.
_FAR_SHEET = b'#far::after { content: "FarDrawn"; }'


class _CountingServer(http.server.ThreadingHTTPServer):
    effect_count: int


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.startswith("/api/data"):
            self._send(b'{"ok": true}', "application/json")
        elif self.path.startswith("/api/more"):
            self._send(b'{"more": true}', "application/json")
        elif self.path.startswith("/slow"):
            time.sleep(2)
            self._send(b'{"slow": true}', "application/json")
        elif self.path.startswith("/trunc"):
            # Announces 500 bytes, sends 2, closes: the body read fails
            # AFTER a 200 status was seen (net::ERR_CONTENT_LENGTH_MISMATCH).
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b"ab")
            self.close_connection = True
        elif self.path.startswith("/plain"):
            self._send(_PLAIN_PAGE, "text/html")
        elif self.path.startswith("/dupstay"):
            self._send(_DUPSTAY_PAGE, "text/html")
        elif self.path.startswith("/dup"):
            self._send(_DUP_PAGE, "text/html")
        elif self.path.startswith("/shadow"):
            self._send(_SHADOW_PAGE, "text/html")
        elif self.path.startswith("/split"):
            self._send(_SPLIT_PAGE, "text/html")
        elif self.path.startswith("/cased"):
            self._send(_CASED_PAGE, "text/html")
        elif self.path.startswith("/drawn"):
            self._send(_DRAWN_PAGE.encode(), "text/html")
        elif self.path.startswith("/far.css"):
            self._send(_FAR_SHEET, "text/css")
        else:
            self._send(_PAGE, "text/html")

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        if self.path.startswith("/effect"):
            cast(_CountingServer, self.server).effect_count += 1
        self._send(b"{}", "application/json")

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


def _serve() -> tuple[_CountingServer, str]:
    srv = _CountingServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.effect_count = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _dead_port() -> int:
    """Return a TCP port nothing listens on (bind, take the port, close)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    sock.close()
    return port


def _kill_driver(web: WebUseTool) -> None:
    """Kill the Node driver behind *web*'s Playwright outright (a driver crash).

    ``asyncio.subprocess.Process.kill``: SIGKILL on POSIX, ``TerminateProcess``
    on Windows (which has no ``signal.SIGKILL``).
    """
    assert web._playwright is not None
    web._playwright._impl_obj._connection._transport._proc.kill()


def _eventually(predicate: Any, timeout: float = 10.0) -> Any:
    """Call *predicate* until it returns a truthy value; return that or the last value."""
    deadline = time.monotonic() + timeout
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(0.2)
        result = predicate()
    return result


@pytest.fixture
def server() -> Any:
    srv, base = _serve()
    yield base
    srv.shutdown()


@pytest.fixture
def counting_server() -> Any:
    """The base URL and the server itself, for assertions on /effect counts."""
    srv, base = _serve()
    yield base, srv
    srv.shutdown()


@pytest.fixture
def web(tmp_path: Path) -> Any:
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    yield tool
    tool.close()


def _chromium_persists_cookies() -> bool:
    """Whether a Chromium profile on this machine keeps its cookies across a relaunch.

    Chromium stores cookies encrypted, on Windows with DPAPI; a logon
    session without the user's credentials (an ssh login by key) gets
    ``ERROR_ACCESS_DENIED`` from ``CryptProtectData``, and Chromium then
    writes no cookie to the profile at all.
    """
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    plain = Blob(5, ctypes.create_string_buffer(b"probe", 5))
    sealed = Blob()
    sealed_ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(plain), None, None, None, None, 0, ctypes.byref(sealed)
    )
    if sealed_ok:
        ctypes.windll.kernel32.LocalFree(sealed.pbData)
    return bool(sealed_ok)


needs_cookie_persistence = pytest.mark.skipif(
    not _chromium_persists_cookies(),
    reason="Chromium persists no cookies here: DPAPI refuses this logon session",
)


def _find_id(tree: str, label: str) -> int:
    match = re.search(rf"\[(\d+)\].*{label}", tree)
    assert match, f"Element {label!r} not found in tree:\n{tree}"
    return int(match.group(1))


def _set_extra_style(web: WebUseTool, css: str) -> None:
    """Make *css* the rules of the page's extra style sheet, which the first call adds."""
    web.evaluate_script(
        "() => { let style = document.getElementById('extra');"
        " if (!style) { style = document.createElement('style'); style.id = 'extra';"
        " document.head.appendChild(style); }"
        f" style.textContent = {json.dumps(css)}; return 1; }}"
    )


def _pseudo_reads(web: WebUseTool) -> list[str]:
    """Return, and forget, the ids (else tag names) of the elements whose
    ``::before`` or ``::after`` style the page saw read."""
    names = web.evaluate_script(
        "() => { const names = [...window.pseudoReads]; window.pseudoReads.clear(); return names; }"
    )
    return list(json.loads(names))


def _console_containing(web: WebUseTool, needle: str) -> str:
    """The page's console listing when it mentions *needle*, else empty."""
    output = web.list_console_messages()
    return output if needle in output else ""


def _guidance_shown(web: WebUseTool) -> str:
    """get_page_content() when it carries the attach_to_chrome guidance, else empty."""
    content = web.get_page_content()
    return content if "attach_to_chrome" in content else ""


def _network_containing(web: WebUseTool, needle: str) -> str:
    """The page's network listing when it mentions *needle*, else empty."""
    output = web.list_network_requests()
    return output if needle in output else ""


class _UserChrome:
    """A stand-in for the user's real Chrome, remote-debugging enabled.

    Lives on its own thread because the Playwright sync API is bound to
    the thread (and event loop) that created it, and the WebUseTool
    under test already owns the test thread's loop.
    """

    def __init__(self, port: int):
        self.port = port
        self._calls: queue.Queue[tuple[str, str | None, queue.Queue[Any]]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="user-chrome", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        api = web_stealth.playwright_api()
        pw = api.sync_playwright().start()
        browser: Any = None
        while True:
            command, arg, reply = self._calls.get()
            if command == "launch":
                browser = pw.chromium.launch(
                    headless=True, args=[f"--remote-debugging-port={self.port}"]
                )
                reply.put(browser.is_connected())
            elif command == "new_page":
                page = browser.new_page()
                page.goto(arg)
                reply.put(True)
            elif command == "is_connected":
                reply.put(browser.is_connected())
            elif command == "pages":
                reply.put(
                    len([p for c in browser.contexts for p in c.pages if not p.is_closed()])
                )
            elif command == "close_pages":
                for context in browser.contexts:
                    for page in context.pages:
                        page.close()
                reply.put(True)
            elif command == "close_browser":
                browser.close()
                reply.put(True)
            else:  # "stop"
                if browser is not None and browser.is_connected():
                    browser.close()
                pw.stop()
                reply.put(True)
                return

    def _call(self, command: str, arg: str | None = None) -> Any:
        reply: queue.Queue[Any] = queue.Queue()
        self._calls.put((command, arg, reply))
        return reply.get(timeout=60)

    def launch(self) -> None:
        assert self._call("launch"), "user Chrome failed to launch"

    def new_page(self, url: str) -> None:
        self._call("new_page", url)

    def is_connected(self) -> bool:
        return bool(self._call("is_connected"))

    def page_count(self) -> int:
        return int(self._call("pages"))

    def close_pages(self) -> None:
        """The user closes every tab (the browser keeps running)."""
        self._call("close_pages")

    def close_browser(self) -> None:
        self._call("close_browser")

    def stop(self) -> None:
        self._call("stop")
        self._thread.join(timeout=30)


class TestEvaluateScript:
    def test_function_expression_and_json_result(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        assert web.evaluate_script("() => document.title") == "devtools-test"
        # A bare expression is wrapped as a function body.
        assert web.evaluate_script("document.title") == "devtools-test"
        # Non-string results come back as JSON.
        result = web.evaluate_script("() => ({a: 1, list: [1, 2]})")
        assert json.loads(result) == {"a": 1, "list": [1, 2]}
        assert web.evaluate_script("() => 40 + 2") == "42"

    def test_async_script(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        result = web.evaluate_script(
            "async () => { const r = await fetch('/api/data');"
            " return (await r.json()).ok; }"
        )
        assert result == "true"

    def test_throwing_script_and_empty_script(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        result = web.evaluate_script("() => { throw new Error('kaboom'); }")
        assert result.startswith("Error evaluating script: ")
        assert "kaboom" in result
        assert web.evaluate_script("   ") == "Error evaluating script: empty script."

    def test_result_truncation(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        result = web.evaluate_script("() => 'x'.repeat(30000)")
        assert len(result) > 20000
        assert result.endswith("... [truncated]")

    def test_bare_expressions(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        # Parenthesized and arrow-containing expressions are not functions.
        assert web.evaluate_script("(1 + 2)") == "3"
        assert json.loads(web.evaluate_script("[1,2].map(x => x*2)")) == [2, 4]

    def test_session_recreated_when_adoption_could_not_start_it(
        self, web: WebUseTool, server: str
    ) -> None:
        web.go_to_url(server)
        # The state a failed capture start leaves behind for the page that
        # stays current: the tools that need the session recreate it.
        web._page_records[web._page].session = None
        assert web.evaluate_script("() => 1 + 1") == "2"
        web._page_records[web._page].session = None
        web.list_console_messages(clear=True)
        assert web._page_records[web._page].console_on
        web.evaluate_script("() => console.log('after restart')")
        assert "after restart" in _eventually(lambda: _console_containing(web, "after restart"))

    def test_cyclic_result_is_an_error_not_a_raise(
        self, web: WebUseTool, server: str
    ) -> None:
        web.go_to_url(server)
        result = web.evaluate_script("() => { const a = {}; a.self = a; return a; }")
        assert result.startswith("Error evaluating script: ")

    def test_navigating_script_runs_its_side_effects_once(
        self, web: WebUseTool, counting_server: Any
    ) -> None:
        base, srv = counting_server
        web.go_to_url(base)
        result = web.evaluate_script(
            f"async () => {{ await fetch('{base}/effect', {{method: 'POST'}});"
            " location.href = '/done'; return new Promise(() => {}); }"
        )
        # The navigation cancelled the one-shot evaluation (an error) —
        # it must NOT have been re-armed in the new document.
        assert result.startswith("Error evaluating script: ")
        assert srv.effect_count == 1


class TestListConsoleMessages:
    def test_console_and_page_errors(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        output = _eventually(lambda: _console_containing(web, "uncaught boom"))
        assert "[log] page boot" in output
        assert "[error] boot error" in output
        assert "[pageerror]" in output and "uncaught boom" in output

    def test_clear_and_action_attribution(self, web: WebUseTool, server: str) -> None:
        tree = web.go_to_url(server)
        assert "page boot" in web.list_console_messages(clear=True)
        assert web.list_console_messages() == "No console messages recorded for the current page."
        # A click's console output is attributable to the click after clear.
        tree = web.click(_find_id(tree, "Go"))
        output = _eventually(lambda: _console_containing(web, "button clicked"))
        assert "[warning] button clicked" in output

    def test_no_messages_on_quiet_page(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/plain")
        assert web.list_console_messages() == "No console messages recorded for the current page."

    def test_queued_events_are_pumped_by_the_listing(
        self, web: WebUseTool, server: str
    ) -> None:
        web.go_to_url(server)
        web.list_console_messages(clear=True)
        assert (
            web.evaluate_script("() => { setTimeout(() => console.log('late-marker'), 300);"
                                " return 1; }")
            == "1"
        )
        time.sleep(0.7)
        # No other tool call in between: the listing's own round trip
        # must deliver the queued event first.
        assert "late-marker" in web.list_console_messages()


class TestListNetworkRequests:
    def test_requests_with_status(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        output = web.list_network_requests()
        assert f"GET 200 {server}/ (document)" in output
        assert f"GET 200 {server}/api/data (fetch)" in output

    def test_failed_request(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        dead = _dead_port()
        result = web.evaluate_script(
            f"async () => {{ try {{ await fetch('http://127.0.0.1:{dead}/x');"
            " return 'unexpected'; } catch (e) { return 'threw'; } }"
        )
        assert result == "threw"
        output = _eventually(lambda: _network_containing(web, "FAILED"))
        assert f"127.0.0.1:{dead}" in output
        assert "FAILED:" in output

    def test_pending_request(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        assert web.evaluate_script("() => { fetch('/slow'); return 1; }") == "1"
        output = _eventually(lambda: _network_containing(web, "/slow"))
        line = next(line for line in output.splitlines() if "/slow" in line)
        assert line.startswith("GET ... ")

    def test_body_failure_after_200_is_reported_failed(
        self, web: WebUseTool, server: str
    ) -> None:
        web.go_to_url(server)
        result = web.evaluate_script(
            "async () => { try { const r = await fetch('/trunc');"
            " await r.text(); return 'no-fail'; } catch (e) { return 'failed'; } }"
        )
        assert result == "failed"
        output = _eventually(lambda: _network_containing(web, "/trunc"))
        line = next(line for line in output.splitlines() if "/trunc" in line)
        assert "FAILED:" in line

    def test_clear_drops_pending_requests(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        assert web.evaluate_script("() => { fetch('/slow'); return 1; }") == "1"
        _eventually(lambda: _network_containing(web, "/slow"))
        assert web._page_records[web._page].pending, "pending entry missing before clear"
        web.list_network_requests(clear=True)
        assert web._page_records[web._page].pending == {}
        # The request finishes (2 s server-side) AFTER the clear: its
        # retired entry must not resurrect in the listing.
        time.sleep(2.5)
        assert "/slow" not in web.list_network_requests()

    def test_url_filter_no_match_and_clear(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        filtered = web.list_network_requests(url_filter="/api/")
        assert "/api/data" in filtered
        assert "(document)" not in filtered
        assert web.list_network_requests(url_filter="/nonexistent") == (
            "No network requests matching '/nonexistent'."
        )
        web.list_network_requests(clear=True)
        assert (
            web.list_network_requests() == "No network requests recorded for the current page."
        )


class TestWaitFor:
    def test_appear_after_click(self, web: WebUseTool, server: str) -> None:
        tree = web.go_to_url(server)
        web.click(_find_id(tree, "Go"))
        tree = web.wait_for("Loaded!")
        assert "Loaded!" in tree
        assert "devtools-test" in tree

    def test_disappear(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        tree = web.wait_for("Ephemeral notice", disappear=True)
        assert "Ephemeral notice" not in tree

    def test_timeout_and_floor(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(server)
        result = web.wait_for("NeverShowsUp", timeout=0)
        assert result == "Timed out after 0.5s waiting for text 'NeverShowsUp' to appear."

    def test_immediate_success_with_huge_timeout(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/plain")
        # The 120 s cap applies but the text is already there: no waiting.
        tree = web.wait_for("nothing here", timeout=9999)
        assert "nothing here" in tree

    def test_hidden_copy_counts_for_nothing(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/dup")
        # The second, visible "Loading" satisfies the wait — a tree, not a
        # timeout message (which would itself contain the text).
        tree = web.wait_for("Loading", timeout=2)
        assert tree.startswith("Page: dup")
        assert "- paragraph: Loading" in tree
        # ...and the remaining visible copy blocks disappearance.
        result = web.wait_for("Loading", timeout=0.5, disappear=True)
        assert result == "Timed out after 0.5s waiting for text 'Loading' to disappear."

    def test_disappear_waits_for_all_copies(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/dupstay")
        tree = web.wait_for("Loading", timeout=2)
        assert tree.startswith("Page: dupstay")
        web.evaluate_script("() => { setTimeout(hideFirst, 200); return 1; }")
        # The first copy is hidden but the second stays visible:
        # disappearance must NOT complete.
        result = web.wait_for("Loading", timeout=1, disappear=True)
        assert result.startswith("Timed out after 1s")
        web.evaluate_script("() => { setTimeout(removeSecond, 100); return 1; }")
        tree = web.wait_for("Loading", disappear=True)
        assert tree.startswith("Page: dupstay")
        assert "Loading" not in tree

    def test_shadow_dom_and_input_label(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/shadow")
        tree = web.wait_for("ShadowReady", timeout=2)
        assert tree.startswith("Page: shadow")
        # A text node directly under the shadow root (no wrapping element).
        tree = web.wait_for("DirectReady", timeout=2)
        assert tree.startswith("Page: shadow")
        tree = web.wait_for("SubmitReady", timeout=2)
        assert tree.startswith("Page: shadow")
        # content-visibility:hidden suppresses rendering: not present.
        result = web.wait_for("ContentHidden", timeout=0.5)
        assert result.startswith("Timed out after 0.5s")
        # Inline text split across the children of a shadow root, and SVG
        # text inside a shadow root, read as rendered.
        assert web.wait_for("ShadowSplit", timeout=2).startswith("Page: shadow")
        assert web.wait_for("ShadowVector", timeout=2).startswith("Page: shadow")
        # Light DOM a shadow tree without a <slot> leaves unrendered.
        assert web.wait_for("ShadowOnly", timeout=2).startswith("Page: shadow")
        assert web.wait_for("UnslottedLight", timeout=0.5).startswith("Timed out")
        # Slotted light DOM reads where the <slot> renders it, not where
        # it sits in the tree (which would be after the shadow text).
        assert web.wait_for("Slotted successfully", timeout=2).startswith("Page: shadow")
        assert web.wait_for("successfully Slotted", timeout=0.5).startswith("Timed out")
        # ...and its visibility and line box follow the slot's shadow
        # ancestors, not the host: a hidden wrapper around the slot hides
        # the slotted text, and an inline split across the slot joins.
        assert web.wait_for("SlotSkipped", timeout=0.5).startswith("Timed out")
        assert web.wait_for("SlotInvisible", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Spliced", timeout=2).startswith("Page: shadow")
        assert web.wait_for("Slotted", disappear=True, timeout=0.5).startswith("Timed out")
        assert web.wait_for("SlotSkipped", disappear=True, timeout=2).startswith("Page: shadow")
        assert web.wait_for("SlotInvisible", disappear=True, timeout=2).startswith("Page: shadow")

    def test_text_split_across_inline_markup(self, web: WebUseTool, server: str) -> None:
        web.go_to_url(f"{server}/split")
        tree = web.wait_for("Saved", timeout=2)
        assert tree.startswith("Page: split")
        tree = web.wait_for("Nested", timeout=2)
        assert tree.startswith("Page: split")
        # A hidden split copy counts for nothing, and script source is not
        # rendered text.
        assert web.wait_for("Hidden copy", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Scripted", timeout=0.5).startswith("Timed out")
        # "Partly" is only complete with its hidden fragment: not readable.
        assert web.wait_for("Partly", timeout=0.5).startswith("Timed out")
        assert web.wait_for("rtly", timeout=2).startswith("Page: split")
        # A hidden duplicate beside the visible copy does not hide it...
        assert web.wait_for("Twin", timeout=2).startswith("Page: split")
        # ...and once the visible copy goes, the hidden one keeps nothing.
        web.evaluate_script("() => { document.getElementById('msg').remove(); return 1; }")
        tree = web.wait_for("Saved", disappear=True)
        assert tree.startswith("Page: split")
        web.evaluate_script(
            "() => { document.getElementById('twin').firstChild.remove(); return 1; }"
        )
        assert web.wait_for("Twin", disappear=True).startswith("Page: split")
        # Two table cells never read as one word; a hidden <option> of a
        # collapsed select is not on screen, its selected label is; SVG
        # text is.
        assert web.wait_for("Table", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Concealed", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Chosen", timeout=2).startswith("Page: split")
        assert web.wait_for("VectorText", timeout=2).startswith("Page: split")
        # A <br> breaks the word; runs of white space read as one space;
        # hidden noise inside a word does not break it; display:contents
        # text has no box of its own but is on screen.
        assert web.wait_for("Line", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Spaced out", timeout=2).startswith("Page: split")
        assert web.wait_for("Word", timeout=2).startswith("Page: split")
        assert web.wait_for("Contents", timeout=2).startswith("Page: split")
        # A textarea shows its value, not its stale default text.
        assert web.wait_for("OldDefault", timeout=2).startswith("Page: split")
        web.evaluate_script("() => { document.getElementById('area').value = 'Typed'; return 1; }")
        assert web.wait_for("OldDefault", timeout=0.5).startswith("Timed out")
        assert web.wait_for("Typed", timeout=2).startswith("Page: split")
        # Only a value rendered as text counts: a submit button's label
        # does; a checkbox's or radio's submission value, a password's
        # cleartext and a hidden field's value do not.  Nor does text under
        # visibility:collapse (a collapsed table row, a collapsed block).
        assert web.wait_for("Send it", timeout=2).startswith("Page: split")
        assert web.wait_for("TickValue", timeout=0.5).startswith("Timed out")
        assert web.wait_for("DotValue", timeout=0.5).startswith("Timed out")
        assert web.wait_for("UnrenderedSecret", timeout=0.5).startswith("Timed out")
        assert web.wait_for("HiddenField", timeout=0.5).startswith("Timed out")
        assert web.wait_for("TickValue", disappear=True, timeout=2).startswith("Page: split")
        assert web.wait_for("CollapsedRow", timeout=0.5).startswith("Timed out")
        assert web.wait_for("ShownRow", timeout=2).startswith("Page: split")
        assert web.wait_for("CollapsedPara", timeout=0.5).startswith("Timed out")
        assert web.wait_for("CollapsedRow", disappear=True, timeout=2).startswith("Page: split")
        # White space is forgiving on both sides: a source line break,
        # &nbsp;, a white-space:pre-line run, a hidden <br> and preserved
        # spaces across inline markup all read as one space, and the
        # needle may spell them either way; an emoji is one letter.
        assert web.wait_for("Changes saved", timeout=2).startswith("Page: split")
        assert web.wait_for("Changes\nsaved", timeout=2).startswith("Page: split")
        assert web.wait_for("Message sent \U0001F44D", timeout=2).startswith("Page: split")
        assert web.wait_for("Preline kept", timeout=2).startswith("Page: split")
        assert web.wait_for("Hard space", timeout=2).startswith("Page: split")
        assert web.wait_for("Hard\u00a0space", timeout=2).startswith("Page: split")
        assert web.wait_for("Soft break", timeout=2).startswith("Page: split")
        assert web.wait_for("Pre formatted", timeout=2).startswith("Page: split")
        assert web.wait_for("Pre  formatted", timeout=2).startswith("Page: split")
        # A collapsed <select> shows its selected option's label attribute,
        # not the option's text.
        assert web.wait_for("Completed", timeout=2).startswith("Page: split")
        assert web.wait_for("Internal", timeout=0.5).startswith("Timed out")
        # A body without a box of its own (display:contents) still shows
        # its children.
        web.evaluate_script("() => { document.body.style.display = 'contents'; return 1; }")
        assert web.wait_for("Nested", timeout=2).startswith("Page: split")

    def test_text_matches_as_written_and_as_displayed(self, web: WebUseTool, server: str) -> None:
        """Text under a CSS ``text-transform`` is found by either spelling.

        The accessibility tree spells it as the page wrote it, a screenshot
        and the text-only content as it is displayed: both are what the
        agent has read, and a displayed copy keeps a disappearance waiting.
        """
        tree = web.go_to_url(f"{server}/cased")
        assert "saved fine" in tree
        assert "SAVED FINE" in web.get_page_content(text_only=True)
        for seen in (
            "saved fine",
            "SAVED FINE",
            "Changes stored ok",
            "Changes STORED ok",
            # A word split over markup is capitalized once; words as Chromium finds them.
            "Hello Wide It's E.G.The World",
            "shout text",
            "FIELD VALUE",
            # Each language's own mapping (Turkish from an ancestor, German and Greek
            # from the element), and the common one under a tag that names no language.
            "\u0130PTAL",
            "SCHLIESSEN",
            "\u0391\u03a0\u039f\u0398\u0397\u039a\u0395\u03a5\u03a3\u0397",
            "ODD TONGUE",
            "\u0391\u0399\u0394\u03a9",
            "stra\u00dfe",
            "\u01c5uro \u1fbc\u03b4\u03c9",
            "BITIR",
        ):
            assert web.wait_for(seen, timeout=2).startswith("Page: cased"), seen
        for unseen in (
            "Saved Fine", "IPTAL", "HeLlo",
            "\u01c4uro", "\u1c92\u1c90\u1c9b\u1c90", "B\u0130T\u0130R",
        ):
            assert web.wait_for(unseen, timeout=0.5).startswith("Timed out"), unseen
        assert web.wait_for("SAVED FINE", timeout=0.5, disappear=True).startswith("Timed out")
        web.evaluate_script("() => { document.getElementById('note').remove(); return 1; }")
        assert web.wait_for("SAVED FINE", disappear=True).startswith("Page: cased")
        # The language is the one the browser computed for the element (here the
        # page's Content-Language): no lang attribute names it.
        web.evaluate_script(
            "() => { const meta = document.createElement('meta');"
            " meta.httpEquiv = 'content-language'; meta.content = 'tr';"
            " document.head.appendChild(meta); return 1; }"
        )
        assert web.wait_for("F\u0130ELD VALUE", timeout=2).startswith("Page: cased")
        assert web.wait_for("FIELD VALUE", timeout=0.5).startswith("Timed out")

    def test_text_a_stylesheet_draws(self, web: WebUseTool, server: str) -> None:
        """The strings of a ``::before`` or ``::after`` are read where they are displayed.

        The accessibility tree shows them, so the agent waits for them; a wait
        for "DONE" timed out with the text on screen, and a wait for it to
        disappear returned at once.  Text is matched with and without them.
        """
        tree = web.go_to_url(f"{server}/drawn")
        assert "Status DONE" in tree
        seen = (
            "Status DONE",
            "Synced",
            "Note body text",
            # As written and as displayed, like any text under a text-transform.
            "releasenew",
            "releaseNEW",
            # With the arrow the sheet draws, and as the page's own text reads.
            "Open Docs\u2197 page",
            "Open Docs page",
            "MediaDrawnm",
            "Glypha",
            # A counter's number is not read, like the marker of a list item.
            ". first",
            "My Profile",
            "All Done",
            "Flat DONE",
            "BannerDrawn",
            "RootDrawn",
            "FailedDrawn",
            "WhenDrawn",
            "TickDrawn",
            "tTwDrawn",
            "x!y!",
            'Say "hi" now',
            # The rules of a shadow tree: for its host, its own elements and what it slots.
            "HostDrawn",
            "inInnerDrawn",
            "litSlotDrawn",
            "dAdoptedDrawn",
            "shAdoptedDrawn",
        )
        unseen = (
            "GoneDrawn", "UnseenDrawn", "OffDrawn", "AltDrawn", "Folded DONE",
            "LoadedDrawn", "FieldDrawn", "BoxDrawn", "AreaDrawn", "SvgDrawn", "+. first",
        )
        for text in seen:
            assert web.wait_for(text, timeout=2).startswith("Page: drawn"), text
        for text in unseen:
            assert web.wait_for(text, timeout=0.5).startswith("Timed out"), text
        # A drawn copy keeps a disappearance waiting.
        assert web.wait_for("DONE", timeout=0.5, disappear=True).startswith("Timed out")
        web.evaluate_script(
            "() => { for (const el of document.querySelectorAll('.st')) el.className = '';"
            " return 1; }"
        )
        assert web.wait_for("DONE", disappear=True).startswith("Page: drawn")

    def test_only_the_elements_a_rule_may_draw_by_are_read(
        self, web: WebUseTool, server: str
    ) -> None:
        """The ``::before`` and ``::after`` of every element are read only where they must be.

        Reading them costs more than the rest of a poll, so the rules of the
        style sheets say which elements to read: the page counts whose are.
        Every element is read where a rule's selector does not find its
        elements, which the text of such a rule shows.
        """
        web.go_to_url(f"{server}/drawn")
        web.evaluate_script(
            "() => { const read = window.getComputedStyle; window.pseudoReads = new Set();"
            " window.getComputedStyle = (el, pseudo) => {"
            "   if (pseudo) window.pseudoReads.add(el.id || el.tagName);"
            "   return read.call(window, el, pseudo); };"
            " return 1; }"
        )
        assert web.wait_for("Status DONE", timeout=2).startswith("Page: drawn")
        reads = _pseudo_reads(web)
        assert "SPAN" in reads and "far" not in reads
        # Rules that draw no text: without a `content` (those of a @position-try have no
        # such property), with one of no text.
        _set_extra_style(
            web,
            "@position-try --below { top: anchor(bottom); } #far::before { color: red; }"
            ' #far::after { content: ""; } @font-face { font-family: f; src: local("f"); }',
        )
        assert web.wait_for("Status DONE", timeout=2).startswith("Page: drawn")
        assert "far" not in _pseudo_reads(web)
        # A selector finds its elements where the rule draws by them, hovered ones too.
        _set_extra_style(web, 'p:hover > #none::after, #far::after { content: "FoundDrawn"; }')
        assert web.wait_for("fFoundDrawn", timeout=2).startswith("Page: drawn")
        assert "part" not in _pseudo_reads(web)
        # ... also the selectors of an imported sheet, once it has loaded.
        _set_extra_style(web, '@import url("/far.css");')
        assert web.wait_for("fFarDrawn", timeout=2).startswith("Page: drawn")
        _pseudo_reads(web)
        assert web.wait_for("fFarDrawn", timeout=2).startswith("Page: drawn")
        assert "part" not in _pseudo_reads(web)
        rules = "".join(f'.rule{i}::before {{ content: "r"; }}' for i in range(501))
        everywhere = (
            # A nested rule, a scoped one, one for a part of a shadow tree, a keyframe,
            # one with a namespace prefix, which a query cannot resolve.
            ('.card { &::after { content: "NestedDrawn"; } }', "cNestedDrawn"),
            (
                "@namespace h url(http://www.w3.org/1999/xhtml);"
                ' h|p#far::after { content: "SpaceDrawn"; }',
                "fSpaceDrawn",
            ),
            ('@scope (.card) { :scope::before { content: "ScopeDrawn"; } }', "ScopeDrawnc"),
            ('x-part::part(label)::before { content: "PartDrawn"; }', "PartDrawnparted"),
            (
                '#far::before { content: ""; animation: key 1000s; }'
                ' @keyframes key { from, to { content: "KeyDrawn"; } }',
                "KeyDrawnf",
            ),
            # More rules than are worth a query each.
            (rules + '#far::after { content: "ManyDrawn"; }', "fManyDrawn"),
        )
        for css, text in everywhere:
            _set_extra_style(web, css)
            assert web.wait_for(text, timeout=2).startswith("Page: drawn"), text
            assert "part" in _pseudo_reads(web), text
        # The rules of another origin's sheet, which the page cannot read.
        _set_extra_style(web, "")
        far = server.replace("127.0.0.1", "localhost")
        loaded = web.evaluate_script(
            "() => new Promise((done) => {"
            " const link = document.createElement('link'); link.rel = 'stylesheet';"
            f" link.href = '{far}/far.css'; link.onload = () => done('loaded');"
            " document.head.appendChild(link); })"
        )
        assert "loaded" in loaded
        for text in ("fFarDrawn", "Status DONE", "Open Docs page"):
            assert web.wait_for(text, timeout=2).startswith("Page: drawn"), text
        assert "part" in _pseudo_reads(web)
        assert web.wait_for("ManyDrawn", timeout=0.5).startswith("Timed out")


class TestAttachToChrome:
    def test_attach_drive_detach_and_reconnect_guidance(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)

            tree = web.attach_to_chrome(f"http://localhost:{chrome.port}")
            assert "devtools-test" in tree
            assert web.evaluate_script("() => document.title") == "devtools-test"

            # close_browser() detaches; the user's browser keeps running.
            assert web.close_browser().startswith("Browser closed.")
            assert chrome.is_connected()

            # Re-attach, then the user's browser exits: tools must say how
            # to reconnect instead of silently swapping in another browser.
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            chrome.close_browser()
            content = _eventually(lambda: _guidance_shown(web))
            assert "disconnected" in content

            # After close_browser() the built-in Chromium takes over again.
            web.close_browser()
            web.go_to_url(f"{server}/plain")
            assert web.evaluate_script("() => document.title") == "plain"
        finally:
            chrome.stop()

    def test_attach_to_dead_port_and_tab_less_browser(
        self, web: WebUseTool, server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A driver that cannot be spawned is an error string too (it used to raise).
        with monkeypatch.context() as env:
            env.setenv("PLAYWRIGHT_NODEJS_PATH", "/nonexistent/node")
            result = web.attach_to_chrome(f"http://localhost:{_dead_port()}")
        assert result.startswith("Error attaching to Chrome at ")
        assert "/nonexistent/node" in result or "cannot find the file specified" in result

        result = web.attach_to_chrome(f"http://localhost:{_dead_port()}")
        assert result.startswith("Error attaching to Chrome at ")

        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            # No page was ever opened: nothing to adopt.
            result = web.attach_to_chrome(f"http://localhost:{chrome.port}")
            assert result.startswith("Error attaching to Chrome at ")
            assert "no open tabs" in result
        finally:
            chrome.stop()

    def test_stale_live_flag_never_closes_the_users_tab(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            # The state a show_browser() session left behind before the
            # fix: with _live still set, teardown took the live branch and
            # ran page.close() on the USER's tab.
            web._live = True
            assert web.close_browser().startswith("Browser closed.")
            web._live = False
            assert chrome.is_connected()
            assert chrome.page_count() == 1
        finally:
            chrome.stop()

    def test_users_cookies_never_leave_the_attached_chrome(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            web.evaluate_script(
                "() => { document.cookie = 'session=secret'; localStorage.t = 'tok'; return 1; }"
            )
            # What show_browser() would carry into the daemon's browser:
            # the URL only.  The user's jar and localStorage stay put, and
            # a jar that was not read is not an empty one (nothing of what
            # an earlier switch carried is deleted for it).
            session = web._capture_session()
            assert session.url.startswith(server)
            assert session.cookies is None
            assert session.local_storage == []
            web.close_browser()
            # The built-in Chromium's jar is untouched by the attachment.
            web.go_to_url(server)
            assert web.evaluate_script("() => document.cookie") == ""
            session = web._capture_session()
            assert session.url.startswith(server)
        finally:
            chrome.stop()

    def test_detach_with_a_dead_driver_returns_and_spares_the_users_chrome(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            # The tool's Node driver crashes while attached: close_browser()
            # must not send it a close (it would never return) and must
            # not touch the user's browser, which has its own driver.
            _kill_driver(web)
            time.sleep(0.3)
            started = time.monotonic()
            assert web.close_browser().startswith("Browser closed.")
            assert time.monotonic() - started < 30
            assert web._attached_url is None and web._playwright is None
            assert chrome.is_connected()
            assert chrome.page_count() == 1
            # A fresh driver and the built-in Chromium take over.
            web.go_to_url(f"{server}/plain")
            assert web.evaluate_script("() => document.title") == "plain"

            # The death noticed by an earlier call (the failed evaluate):
            # the disconnect is skipped outright.
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            _kill_driver(web)
            time.sleep(0.3)
            assert web.evaluate_script("() => 1").startswith("Error")
            started = time.monotonic()
            assert web.close_browser().startswith("Browser closed.")
            assert time.monotonic() - started < 30
            assert web._playwright is None
            assert chrome.is_connected()
            assert chrome.page_count() == 1
        finally:
            chrome.stop()

    @posix_only("the driver is frozen with SIGSTOP")
    def test_detach_with_a_stalled_driver_is_bounded_and_spares_the_users_chrome(
        self, web: WebUseTool, server: str
    ) -> None:
        """The tool's Node driver stops answering while attached (frozen
        here): the disconnect sent to it never returned, and nothing else
        could end it (the user's browser is never signalled).  It is cut
        off on the Python side and the tool's own driver discarded; the
        user's Chrome and its tab survive."""
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            driver_pid = web._playwright._impl_obj._connection._transport._proc.pid
            os.kill(driver_pid, signal.SIGSTOP)
            resume = threading.Timer(60, os.kill, args=(driver_pid, signal.SIGCONT))
            resume.daemon = True
            resume.start()
            started = time.monotonic()
            try:
                assert web.close_browser().startswith("Browser closed.")
            finally:
                resume.cancel()
                try:
                    os.kill(driver_pid, signal.SIGCONT)
                except OSError:
                    pass
            elapsed = time.monotonic() - started
            assert elapsed < 30, f"close_browser() waited {elapsed:.0f}s for the frozen driver"
            assert web._attached_url is None and web._playwright is None
            assert chrome.is_connected()
            assert chrome.page_count() == 1
            # A fresh driver and the built-in Chromium take over.
            web.go_to_url(f"{server}/plain")
            assert web.evaluate_script("() => document.title") == "plain"
        finally:
            chrome.stop()

    @needs_cookie_persistence
    def test_hiding_the_browser_while_attached_returns_to_the_stealth_browser(
        self, web: WebUseTool, server: str
    ) -> None:
        """The user's Chrome is a visible browser: ``show_browser(False)`` leaves
        it (running) for the headless stealth Chromium on the same page, and
        the cookie an earlier switch carried into that profile survives the
        switch (the attached capture carries no jar, not an empty one: it is
        the profile on disk that keeps the cookie)."""
        web.go_to_url(f"{server}/plain")
        web._context.add_cookies(
            [{"name": "carried", "value": "earlier", "url": server, "expires": time.time() + 3600}]
        )
        web._headless = False
        web.show_browser(visible=False)
        assert [c["name"] for c in web._carried_cookies] == ["carried"]
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")

            tree = web.show_browser(visible=False)

            assert tree.startswith("Page: devtools-test"), tree
            assert web._attached_url is None and web._browser_pid is not None
            assert chrome.is_connected() and chrome.page_count() == 1
            assert web.evaluate_script("() => document.cookie") == "carried=earlier"
            assert web.show_browser(visible=False) == "Browser is already headless."
        finally:
            chrome.stop()

    def test_tab_the_user_closed_is_told_apart_from_a_disconnect(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            chrome.new_page(server)
            web.attach_to_chrome(f"http://localhost:{chrome.port}")
            chrome.close_pages()
            # The first call after the close may fail on the closed page
            # itself (its close is learnt during the call); the next one
            # diagnoses the state the user left.
            content = _eventually(lambda: _guidance_shown(web))
            assert "The attached tab was closed" in content, content
            assert "open one there" in content and "disconnected" not in content
            assert chrome.is_connected()
        finally:
            chrome.stop()

    def test_unreadable_tab_is_an_error_not_a_raise(
        self, web: WebUseTool, server: str
    ) -> None:
        chrome = _UserChrome(_dead_port())
        try:
            chrome.launch()
            # An SVG document has no <body>: the accessibility-tree read
            # times out; attach must report that, not raise, and must
            # roll the half-installed attachment back.
            chrome.new_page(
                "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'>"
                "<text>svg tab</text></svg>"
            )
            result = web.attach_to_chrome(f"http://localhost:{chrome.port}")
            assert result.startswith("Error attaching to Chrome at ")
            assert chrome.is_connected()
            web.go_to_url(f"{server}/plain")
            assert web.evaluate_script("() => document.title") == "plain"
        finally:
            chrome.stop()


class TestToolRegistration:
    def test_get_tools_matches_browser_tool_names(self) -> None:
        tool = WebUseTool(user_data_dir=None, headless=True)
        try:
            assert {f.__name__ for f in tool.get_tools()} == set(BROWSER_TOOL_NAMES)
        finally:
            tool.close()
