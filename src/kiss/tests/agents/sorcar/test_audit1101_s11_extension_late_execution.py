# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An extension command answered "deadline exceeded" never starts its side effects later.

The extension raced a command's promise against its deadline and freed
the queue at the deadline, but the command's own stages went on: an
``evaluate`` whose ``chrome.scripting.executeScript`` sat waiting on a
busy renderer was reported failed, and when the renderer recovered, the
injection ran; on a page whose CSP blocks ``eval`` it then even started
a fresh debugger attachment (the deadline's one-shot ``detach`` had run
before that debugger existed) and evaluated the script that way.  KISS
had long been told the command failed, so a click or a form submission
landed with nobody expecting it.

The deadline now travels through ``evaluate`` and its two paths: it is
checked before the injection, inside the injected function (a delayed
injection refuses expired work), before the CSP fallback, and again
after the debugger attached.  The shipped extension runs in a real
headless Chromium against a real bridge and a real loopback page with a
strict CSP; nothing is replaced.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from patchright.sync_api import Page, sync_playwright

from kiss.agents.sorcar.chrome_bridge import ChromeBridge, ChromeBridgeError, extension_dir

_PAGE = b'<!doctype html><title>late page</title><body data-late="0"><p>late page</p></body>'
# A page whose own script shadows the clock: a top-level ``let`` is a global
# lexical binding every classic script of the realm sees, injected ones too.
_SHADOW = b'<script nonce="test">let Date = "the page shadows Date";</script>'
_CSP = "script-src 'nonce-test'; object-src 'none'"
_INCREMENT = "document.body.dataset.late = String(Number(document.body.dataset.late) + 1)"
# The timed-out calls' timeout.  The extension's deadline runs a tenth
# short of it (``ChromeBridge._send``), so its "exceeded" reply has 0.6 s
# to arrive; 0.12 s (a 1.2 s timeout) and 0.3 s (3 s) were not enough on a
# loaded Mac.  Both stalls below last 8 s, well past the 5.4 s deadline.
_TIMEOUT = 6.0
_EXCEEDED = r"command exceeded its 5\.\d+s deadline"
# Keeps the renderer's main thread busy for 8 s, starting 150 ms from now.
_BUSY_RENDERER = """() => {
  setTimeout(() => { const end = Date.now() + 8000; while (Date.now() < end) {} }, 150);
}"""
# Keeps the renderer busy for 8 s the first time the page's CSP rejects a
# script: right after the extension's eval probe, before its debugger
# fallback reaches the page.
_STALL_ON_CSP_VIOLATION = """() => {
  document.body.dataset.stage = 'armed';
  document.addEventListener('securitypolicyviolation', () => {
    document.body.dataset.stage = 'busy';
    const end = Date.now() + 8000;
    while (Date.now() < end) {}
    document.body.dataset.stage = 'done';
  }, { once: true });
}"""


class _CspPageHandler(BaseHTTPRequestHandler):
    """Serves the one test page, with and without the strict CSP."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = _PAGE + _SHADOW if "shadow" in self.path else _PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        if self.path.startswith("/csp"):
            self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output quiet."""


@dataclass
class _Chrome:
    bridge: ChromeBridge
    page: Page  # driven over CDP by the test (outside the extension)
    base: str

    def tab_id(self) -> int:
        tabs = self.bridge.call("list_tabs", timeout=10)
        return int(next(tab["id"] for tab in tabs if tab["url"] == self.page.url))


@pytest.fixture(scope="module")
def chrome(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Chrome]:
    """A bridge, and Chromium running the installed extension connected to it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KISS_HOME", str(tmp_path_factory.mktemp("kiss_home")))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CspPageHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        bridge = ChromeBridge(write_config=True)
        bridge.start(takeover=True)
        ext = str(extension_dir())
        profile: Path = tmp_path_factory.mktemp("chrome_profile")
        try:
            with sync_playwright() as p:
                context = p.chromium.launch_persistent_context(
                    str(profile),
                    channel="chromium",
                    headless=True,
                    args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
                )
                try:
                    page = context.new_page()
                    page.goto(f"{base}/csp")
                    assert bridge.wait_for_extension(30), "the extension never connected"
                    yield _Chrome(bridge, page, base)
                finally:
                    context.close()
        finally:
            bridge.stop()
            httpd.shutdown()
            httpd.server_close()


def _late_count(chrome: _Chrome) -> str:
    # DOM attributes are shared between Patchright's isolated world and
    # the extension's MAIN world (window expandos are not).
    return str(chrome.page.evaluate("() => document.body.dataset.late"))


@pytest.mark.parametrize("path", ["/csp", "/plain"])
def test_a_command_timed_out_on_a_busy_renderer_never_runs_later(
    chrome: _Chrome, path: str
) -> None:
    """With and without the CSP fallback: the deadline reply is final, the DOM stays untouched."""
    chrome.page.goto(f"{chrome.base}{path}")
    tab_id = chrome.tab_id()
    assert _late_count(chrome) == "0"
    chrome.page.evaluate(_BUSY_RENDERER)
    time.sleep(0.4)  # the renderer is busy now
    began = time.monotonic()
    with pytest.raises(ChromeBridgeError, match=_EXCEEDED):
        chrome.bridge.call("evaluate", {"tabId": tab_id, "script": _INCREMENT}, timeout=_TIMEOUT)
    assert time.monotonic() - began < _TIMEOUT + 2
    assert chrome.bridge.extension_connected
    time.sleep(5)  # the renderer has recovered and every late stage has had its chance
    assert _late_count(chrome) == "0", "the timed-out command ran after its deadline reply"
    # The tab is still usable by the next command, debugger-free or not.
    increment = {"tabId": tab_id, "script": _INCREMENT}
    assert chrome.bridge.call("evaluate", increment, timeout=10) == "1"
    assert _late_count(chrome) == "1"


def test_a_command_timed_out_in_the_csp_fallback_never_runs_later(chrome: _Chrome) -> None:
    """The renderer turns busy after the probe: the queued debugger evaluation stays unrun."""
    chrome.page.goto(f"{chrome.base}/csp")
    tab_id = chrome.tab_id()
    assert _late_count(chrome) == "0"
    chrome.page.evaluate(_STALL_ON_CSP_VIOLATION)
    began = time.monotonic()
    with pytest.raises(ChromeBridgeError, match=_EXCEEDED):
        chrome.bridge.call("evaluate", {"tabId": tab_id, "script": _INCREMENT}, timeout=_TIMEOUT)
    assert time.monotonic() - began < _TIMEOUT + 2
    assert chrome.bridge.extension_connected
    time.sleep(5)  # the renderer has recovered and the queued evaluation has had its chance
    stage = chrome.page.evaluate("() => document.body.dataset.stage")
    assert stage == "done", f"the renderer never stalled on the CSP violation (stage {stage!r})"
    assert _late_count(chrome) == "0", "the timed-out command ran after its deadline reply"
    increment = {"tabId": tab_id, "script": _INCREMENT}
    assert chrome.bridge.call("evaluate", increment, timeout=10) == "1"
    assert _late_count(chrome) == "1"


def _evaluate(chrome: _Chrome, tab_id: int, script: str) -> object:
    return chrome.bridge.call("evaluate", {"tabId": tab_id, "script": script}, timeout=10)


def test_the_csp_fallback_keeps_the_script_semantics(chrome: _Chrome) -> None:
    """On the debugger path, the in-page gate changes no completion value, promise or error."""
    chrome.page.goto(f"{chrome.base}/csp")
    tab_id = chrome.tab_id()
    assert _evaluate(chrome, tab_id, "6 * 7") == 42
    assert _evaluate(chrome, tab_id, "var kissGate = 'declared';") is None
    assert _evaluate(chrome, tab_id, "kissGate") == "declared"
    settled = "new Promise((ok) => setTimeout(() => ok('later'), 20))"
    assert _evaluate(chrome, tab_id, settled) == "later"
    with pytest.raises(ChromeBridgeError, match=r"; debugger: Error: boom"):
        _evaluate(chrome, tab_id, "throw new Error('boom')")
    # Merge-gate review: the gate shares the script's top-level scope, so a
    # script declaring a name the gate uses must still run (a `let` hoists
    # over the whole script; a gate reading the bare name hit its temporal
    # dead zone).  Last in this document: the lexical bindings persist in
    # the page's realm until the next navigation.
    assert _evaluate(chrome, tab_id, "let Date = 1; Date") == 1
    assert _evaluate(chrome, tab_id, "let Error = 'shadowed'; Error") == "shadowed"
    assert _evaluate(chrome, tab_id, "'use strict'; let globalThis = 2; globalThis") == 2


@pytest.mark.parametrize("path", ["/shadow", "/csp-shadow"])
def test_a_page_shadowing_the_clock_still_runs_scripts(chrome: _Chrome, path: str) -> None:
    """A page's own top-level ``let Date`` breaks neither gate.

    The injected function runs in the page's script scope and the debugger
    script in the page's global scope: both saw the page's binding where
    they read ``Date.now()`` and failed with a ``TypeError`` before any
    script ran.  Both read the clock off the global object now, and the
    script itself still sees the page's binding, as it did unguarded.
    """
    chrome.page.goto(f"{chrome.base}{path}")
    tab_id = chrome.tab_id()
    assert _evaluate(chrome, tab_id, "6 * 7") == 42
    assert _evaluate(chrome, tab_id, "Date") == "the page shadows Date"


def test_the_csp_fallback_keeps_the_script_strict(chrome: _Chrome) -> None:
    """On the debugger path, a leading ``'use strict'`` still makes the script strict.

    The gate ahead of the script ended its directive prologue, so the
    directive became a plain string statement and a strict script ran
    sloppy: ``this`` of a plain call was the window and an assignment
    to an undeclared name created a global instead of throwing.  Only
    a directive gets this treatment; a script without one stays sloppy.
    """
    chrome.page.goto(f"{chrome.base}/csp")
    tab_id = chrome.tab_id()
    plain_this = "(function () { return this === undefined; })()"
    assert _evaluate(chrome, tab_id, f"'use strict'; {plain_this}") is True
    assert _evaluate(chrome, tab_id, f"'other directive'; 'use strict'; {plain_this}") is True
    # A directive ended by a line break (no semicolon), after a comment.
    with pytest.raises(ChromeBridgeError, match=r"; debugger: ReferenceError: kissUnbound"):
        _evaluate(chrome, tab_id, '// strict\n"use strict"\nkissUnbound = 73; kissUnbound')
    assert _evaluate(chrome, tab_id, plain_this) is False
    assert _evaluate(chrome, tab_id, "/* 'use strict'; */ kissSloppy = 73; kissSloppy") == 73
    assert _evaluate(chrome, tab_id, "'use strict'.length; kissMember = 73; kissMember") == 73
    # Round 3 review: a comment before the directive's semicolon does not
    # hide it, and a line break before a token that continues the
    # expression is no end of statement (the round-2 regex got both wrong).
    assert _evaluate(chrome, tab_id, f"'use strict' /* comment */; {plain_this}") is True
    assert _evaluate(chrome, tab_id, f"'other' /* comment */; 'use strict'; {plain_this}") is True
    assert _evaluate(chrome, tab_id, f"'use strict'\n.length; {plain_this}") is False
    assert _evaluate(chrome, tab_id, f"'use strict'\n+ 'not a directive'; {plain_this}") is False
    # A prologue of blank-line separated string statements made the
    # round-2 regex backtrack for minutes, stalling the service worker.
    assert _evaluate(chrome, tab_id, "'other'\n\n" * 60 + "42") == 42


if __name__ == "__main__":
    pytest.main([__file__])
