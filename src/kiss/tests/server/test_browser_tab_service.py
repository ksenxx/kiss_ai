# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the browser-tab service and default-browser resolution.

This module exercises the daemon-side pieces: address-bar normalisation,
resolving the machine's default browser from real ``.desktop`` files and
``$BROWSER``/``$KISS_BROWSER``, the ``browser*`` daemon commands driving
a real browser through :class:`AgentServer`, the error paths
(unreachable URL, a "browser" that exits at once), wheel/paste input,
what a finger's drag would scroll, hiding a viewer, a viewer
disconnecting, and shutdown with pages still open.

The macOS (``NSWorkspace``) and Windows (registry) probes cannot run on
a Linux CI host; they are exercised only through their shared helpers.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import gc
import http.server
import os
import shutil
import signal
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.core import default_browser as db
from kiss.server.browser_tab import (
    _ATTACH_TIMEOUT,
    _CLOSE_TIMEOUT,
    _INTERRUPT_PROBE_TIMEOUT,
    BrowserTabService,
    _key_params,
    _mouse_params,
    home_url,
    launch_failure_summary,
    normalize_url,
)
from kiss.server.server import AgentServer
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.two_client_harness import TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"


class ConnPrinter(MemoryPrinter):
    """A :class:`MemoryPrinter` that also keeps per-connection (``connId``) events."""

    def broadcast(self, event: dict[str, Any]) -> None:
        if event.get("connId"):
            self.emitted.append(event)
            return
        super().broadcast(event)


def _wait(pred: Callable[[], Any], what: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = pred()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _events(printer: MemoryPrinter, kind: str, **match: Any) -> list[dict[str, Any]]:
    return [
        e
        for e in list(printer.emitted)
        if e["type"] == kind and all(e.get(k) == v for k, v in match.items())
    ]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_url_and_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_BROWSER_HOME", raising=False)
    assert home_url() == "https://www.google.com"
    assert normalize_url("") == "https://www.google.com"
    monkeypatch.setenv("KISS_BROWSER_HOME", "http://intranet/")
    assert normalize_url("   ") == "http://intranet/"
    assert normalize_url("https://example.com/x?y=1") == "https://example.com/x?y=1"
    assert normalize_url("about:blank") == "about:blank"
    assert normalize_url("example.com") == "https://example.com"
    assert normalize_url("docs.python.org/3/") == "https://docs.python.org/3/"
    # A local server has no certificate, like a LAN address.
    assert normalize_url("localhost:8080/app") == "http://localhost:8080/app"
    assert normalize_url("localhost") == "http://localhost"
    assert normalize_url("192.168.1.10:3000") == "http://192.168.1.10:3000"
    assert normalize_url("localhost.example.com") == "https://localhost.example.com"
    assert normalize_url("kiss sorcar agent") == (
        "https://www.google.com/search?q=kiss%20sorcar%20agent"
    )
    assert normalize_url("what is cdp") == "https://www.google.com/search?q=what%20is%20cdp"


def test_key_and_mouse_translation() -> None:
    down = _key_params({"action": "down", "key": "a", "code": "KeyA", "keyCode": 65})
    assert down["type"] == "keyDown" and down["text"] == "a" and down["unmodifiedText"] == "a"
    assert down["windowsVirtualKeyCode"] == 65 and down["modifiers"] == 0
    enter = _key_params({"action": "down", "key": "Enter", "code": "Enter", "keyCode": 13})
    assert enter["type"] == "keyDown" and enter["text"] == "\r"
    arrow = _key_params({"action": "down", "key": "ArrowLeft", "code": "ArrowLeft", "keyCode": 37})
    assert arrow["type"] == "rawKeyDown" and "text" not in arrow
    # A shortcut sends no text (Ctrl+A must select all, not type "a").
    ctrl_a = _key_params(
        {"action": "down", "key": "a", "code": "KeyA", "keyCode": 65, "ctrl": True}
    )
    assert ctrl_a["type"] == "rawKeyDown" and ctrl_a["modifiers"] == 2
    up = _key_params({"action": "up", "key": "a", "keyCode": 65, "shift": True, "repeat": True})
    assert up["type"] == "keyUp" and up["modifiers"] == 8 and up["autoRepeat"] is True
    assert (
        _key_params({"action": "down", "key": "b", "keyCode": None})["windowsVirtualKeyCode"] == 0
    )

    press = _mouse_params(
        {
            "action": "mousePressed",
            "x": 10.5,
            "y": 20,
            "button": "left",
            "buttons": 1,
            "clickCount": 2,
            "alt": True,
            "meta": True,
        }
    )
    assert press == {
        "type": "mousePressed",
        "x": 10.5,
        "y": 20.0,
        "button": "left",
        "buttons": 1,
        "clickCount": 2,
        "modifiers": 5,
    }
    wheel = _mouse_params({"action": "mouseWheel", "x": 1, "y": 2, "deltaY": 120})
    assert wheel["deltaX"] == 0.0 and wheel["deltaY"] == 120.0 and wheel["button"] == "none"
    assert "deltaX" not in _mouse_params({})


# Playwright's message for a Chrome whose update left no framework to load
# (kiss-web-stderr.log, 2026-11-03), command line shortened.
_FRAMEWORK = (
    "/Applications/Google Chrome.app/Contents/MacOS/../Frameworks/"
    "Google Chrome Framework.framework/Versions/156.0.8078.80/Google Chrome Framework"
)
_LAUNCH_LINE = "<launching> /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless"
_CHROME_ABORT_MESSAGE = "\n".join(
    [
        "BrowserType.launch_persistent_context: Failed to launch the browser process.",
        "Browser logs:",
        "",
        _LAUNCH_LINE,
        "<launched> pid=83207",
        f"[pid=83207][err] dlopen {_FRAMEWORK}: dlopen({_FRAMEWORK}, 0x0105): tried: "
        f"'{_FRAMEWORK}' (no such file), '/System/Volumes/Preboot/Cryptexes/OS{_FRAMEWORK}' "
        "(no such file)",
        "[pid=83207] <process did exit: exitCode=null, signal=SIGABRT>",
        "[pid=83207] starting temporary directories cleanup",
        "Call log:",
        f"  - {_LAUNCH_LINE}",
        "  - <launched> pid=83207",
        f"  - [pid=83207][err] dlopen {_FRAMEWORK} (no such file)",
        "  - [pid=83207] <process did exit: exitCode=null, signal=SIGABRT>",
        "  - [pid=83207] <gracefully close end>",
        "",
    ]
)


def test_launch_failure_summary() -> None:
    """The headline, the browser's own complaint and its exit; no command line, no call log."""
    lines = launch_failure_summary(Exception(_CHROME_ABORT_MESSAGE)).split("\n")
    assert lines[0] == "Failed to launch the browser process."
    assert lines[1].startswith("dlopen /Applications/Google Chrome.app") and len(lines[1]) <= 300
    assert lines[2] == "<process did exit: exitCode=null, signal=SIGABRT>"
    assert len(lines) == 3
    # A message without browser output keeps its headline only; no message, no text.
    missing = RuntimeError("BrowserType.launch: Executable doesn't exist at /x\nCall log:\n  - a")
    assert launch_failure_summary(missing) == "Executable doesn't exist at /x"
    assert launch_failure_summary(TimeoutError()) == ""
    # Only Playwright's ``Class.method: `` prefix goes; a file name stays.
    denied = RuntimeError("chrome.exe: Access is denied.")
    assert launch_failure_summary(denied) == "chrome.exe: Access is denied."


# ---------------------------------------------------------------------------
# Default-browser resolution (Linux probes with real files and scripts)
# ---------------------------------------------------------------------------


def _fake_binary(directory: Path, name: str, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path = directory / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="Linux .desktop / xdg probes")
def test_default_browser_resolution_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    chromium = _fake_binary(bindir, "fake-chromium")
    firefox = _fake_binary(bindir, "firefox")
    apps = tmp_path / "share" / "applications"
    apps.mkdir(parents=True)
    (apps / "my-browser.desktop").write_text(
        "[Desktop Entry]\nName=My Browser\n"
        f"Exec=env GTK_USE_PORTAL=1 {chromium} --profile-directory=Default %U\n"
    )
    (apps / "ff.desktop").write_text(f"[Desktop Entry]\nExec={firefox} %u\n")
    # A fake xdg-settings names the .desktop file, as the real one does.
    desktop_choice = tmp_path / "choice.txt"
    # (shell builtins only: PATH holds nothing but this directory)
    _fake_binary(bindir, "xdg-settings", f'#!/bin/sh\nread line < {desktop_choice}; echo "$line"\n')
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "nowhere"))
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.delenv("KISS_BROWSER", raising=False)

    # Chromium-family default: streamed directly, marked as the default.
    desktop_choice.write_text("my-browser.desktop\n")
    assert db.default_browser_executable() == str(chromium)
    resolved = db.resolve_browser()
    assert resolved == db.ResolvedBrowser("Chromium", str(chromium), True)

    # Firefox default: not streamable; no Chromium-family browser on PATH
    # (the fake one is not under a well-known name) -> bundled Chromium.
    desktop_choice.write_text("ff.desktop\n")
    assert db.default_browser_executable() == str(firefox)
    resolved = db.resolve_browser()
    assert resolved.executable is None and resolved.is_default is False
    assert "firefox" in resolved.note and "cannot be streamed" in resolved.note
    # The directories above the binary do not name the browser: Firefox in the
    # home of a user called ``operator`` is not Opera, and is not streamed.
    operator_home = tmp_path / "operator"
    operator_home.mkdir()
    (apps / "ff-home.desktop").write_text(
        f"[Desktop Entry]\nExec={_fake_binary(operator_home, 'firefox')} %u\n"
    )
    desktop_choice.write_text("ff-home.desktop\n")
    resolved = db.resolve_browser()
    assert resolved.executable is None and resolved.is_default is False
    assert "cannot be streamed" in resolved.note
    desktop_choice.write_text("ff.desktop\n")
    # ... unless a well-known Chromium-family binary is installed.
    brave = _fake_binary(bindir, "brave-browser")
    resolved = db.resolve_browser()
    assert resolved == db.ResolvedBrowser("Brave", str(brave), False, resolved.note)
    brave.unlink()

    # A .desktop name with no file falls back to a binary of that name.
    desktop_choice.write_text("firefox.desktop\n")
    assert db.default_browser_executable() == str(firefox)
    # A broken Exec line (unresolvable binary) yields nothing.
    (apps / "broken.desktop").write_text("[Desktop Entry]\nExec=/no/such/browser %U\n")
    desktop_choice.write_text("broken.desktop\n")
    assert db.default_browser_executable() is None
    assert "No default browser" in db.resolve_browser().note

    # $BROWSER wins over xdg; $KISS_BROWSER wins over everything.
    monkeypatch.setenv("BROWSER", f"{chromium} --new-window %s")
    assert db.default_browser_executable() == str(chromium)
    monkeypatch.setenv("BROWSER", "")
    monkeypatch.setenv("KISS_BROWSER", str(firefox))
    assert db.resolve_browser() == db.ResolvedBrowser("firefox", str(firefox), False)
    monkeypatch.setenv("KISS_BROWSER", "/no/such/binary")
    assert db.resolve_browser().executable is None

    # xdg-settings failing -> xdg-mime is asked next.
    _fake_binary(bindir, "xdg-settings", "#!/bin/sh\nexit 1\n")
    _fake_binary(bindir, "xdg-mime", "#!/bin/sh\necho my-browser.desktop\n")
    monkeypatch.delenv("KISS_BROWSER", raising=False)
    assert db.default_browser_executable() == str(chromium)
    _fake_binary(bindir, "xdg-mime", "#!/bin/sh\nexit 1\n")
    assert db.default_browser_executable() is None


def test_family_name_and_mac_bundle_helper(tmp_path: Path) -> None:
    assert db.family_name("/usr/bin/google-chrome-stable") == "Google Chrome"
    assert db.family_name("/Applications/Brave Browser.app") == "Brave"
    assert (
        db.family_name(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
        == "Microsoft Edge"
    )
    assert db.family_name("/usr/bin/firefox") is None
    # Only the last component names the browser, whichever separator the path uses.
    for firefox in (
        "/home/operator/firefox/firefox",
        r"C:\Users\operator\AppData\Local\Mozilla Firefox\firefox.exe",
        "/home/brave/chrome/firefox",
    ):
        assert db.family_name(firefox) is None, firefox
    # A minimal .app bundle: Info.plist names the executable.
    app = tmp_path / "Fake.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
        b"<key>CFBundleExecutable</key><string>Fake</string></dict></plist>"
    )
    assert db._mac_app_executable(str(app)) is None  # binary missing
    # The binary is found as Popen would run it: on Windows that is never an
    # extensionless file (``shutil.which`` with ``PATHEXT``), so the stand-in is a ``.cmd``.
    binary = "Fake.cmd" if sys.platform == "win32" else "Fake"
    exe = _fake_binary(app / "Contents" / "MacOS", binary)
    found = db._mac_app_executable(str(app))
    assert found is not None and Path(found).resolve() == exe.resolve()
    # An unreadable bundle is skipped: garbage, XML cut short by an update, a non-dict root.
    xml = b'<?xml version="1.0"?><plist version="1.0">'
    for unreadable in (b"not a plist", xml + b"<dict>", xml + b"<array/></plist>"):
        (app / "Contents" / "Info.plist").write_bytes(unreadable)
        assert db._mac_app_executable(str(app)) is None
    assert db._mac_app_executable(str(tmp_path / "Missing.app")) is None


# ---------------------------------------------------------------------------
# The daemon commands against a real browser
# ---------------------------------------------------------------------------


# A sign-in page shaped like Duo's "Use your security key" prompt: on load
# it asks the browser for a security key and offers an "Other options"
# button at a fixed position.  The button retitles the page when clicked.
_SECURITY_KEY_PAGE = b"""<html><head><title>Login</title></head>
<body style='margin:0'>
<script>
if (window.PublicKeyCredential) {
  const challenge = new Uint8Array(32);
  navigator.credentials.get({publicKey: {challenge, timeout: 60000,
    allowCredentials: [{type: 'public-key', id: new Uint8Array(16)}]}})
    .catch(() => {});
}
</script>
<button style='position:absolute;left:100px;top:100px;width:200px;height:50px'
  onclick="document.title = 'Other options'">Other options</button>
</body></html>"""


def _strip(name: str, top: int, left: int, style: str = "overflow-x:auto") -> str:
    """A 300px box *top* px down the page: a 1200px line in it, scrolled *left* px sideways."""
    return (
        f"<div id='{name}' style='position:absolute;left:0;top:{top}px;width:300px;height:40px;"
        f"{style}'><div style='width:1200px;height:20px'></div></div>"
        f"<script>document.getElementById('{name}').scrollLeft = {left};</script>"
    )


# What the page below embeds: one box.  The test scrolls it sideways in
# the two frames (/strip) and leaves it alone where a plugin embeds it
# (/still).  A script of the embedded page cannot scroll it as the page
# loads: a frame of another site is rendered by a process of its own
# and is told its size after its page was parsed; until then its boxes
# may have no layout, and a box without one does not scroll.
_BOX_PAGE = "<!doctype html><body style='margin:0'>" + _strip("inner", 0, 0) + "</body>"
# Scrolls that box 80px sideways; where it stands then.
_SCROLL_BOX_JS = (
    "(() => { const box = document.getElementById('inner'); box.scrollLeft = 80;"
    " return box.scrollLeft; })()"
)


def _wide(name: str, root: str, body: str) -> str:
    """A document 3000px wide titled *name*; *root* and *body* style its two outer elements."""
    return (
        f"<!doctype html><html style='{root}'><head><title>{name}</title></head>"
        f"<body style='margin:0;{body}'><div style='width:3000px;height:20px'></div></body></html>"
    )


_DOCUMENTS = {
    "/strip": _BOX_PAGE,
    "/still": _BOX_PAGE,
    # Reads right to left, as its body says; the body always shows a scrollbar.
    "/rtl": _wide("rtl", "", "direction:rtl;overflow-y:scroll"),
    # Its body forbids sideways scrolling; so does the root of the next one.
    "/hidden": _wide("hidden", "", "overflow-x:hidden"),
    "/clip": _wide("clip", "overflow-x:clip", "direction:rtl"),
    # Its body is what scrolls, in a window that has nothing to scroll.
    "/own": _wide("own", "overflow-y:clip", "overflow:auto;width:300px;height:100px"),
    # An old page: without a doctype the body stands for the window.
    "/old": _wide("old", "", "direction:rtl").removeprefix("<!doctype html>"),
}

# Things a finger can land on, 50px apart down the left side of a
# document 3000px wide: boxes scrolled sideways or not, one that reads
# right to left at each of its ends, one that only a script scrolls, one
# whose lines run from right to left, one whose text runs upwards, one
# of a fractional width, one in a shadow root, one in a shadow root
# around a child of the page, one around a shadow root, a shadow root
# with nothing in it, the box above in a frame of this site (whose
# border and padding move its page 8px in) and in a frame of another
# site ({other}: this server under its other name), that box as a
# plugin embeds it, a text field, and frames the page draws at twice
# their size, at half of it, zoomed, and from inside a shadow root.
_FRAME = (
    "<iframe src='/strip' style='display:block;border:0;width:300px;height:{height}px;{style}'>"
    "</iframe>"
)
_SCROLLERS_PAGE = (
    "<!doctype html><html><head><title>Scrollers</title></head><body style='margin:0'>"
    + _strip("scrolled", 0, 100)
    + _strip("edge", 50, 0)
    + _strip("rtl-start", 100, 0, "overflow-x:auto;direction:rtl")
    + _strip("rtl-end", 150, -900, "overflow-x:auto;direction:rtl")
    + _strip("clipped", 200, 100, "overflow-x:hidden")
    + _strip("vertical", 650, 0, "overflow-x:auto;writing-mode:vertical-rl")
    + _strip("upwards", 700, 0, "overflow-x:auto;writing-mode:vertical-lr;direction:rtl")
    + _strip("fraction", 750, -2000, "overflow-x:auto;direction:rtl;width:300.3px")
    + _strip("around", 900, 60)
    + "<div id='empty' style='position:absolute;left:0;top:850px;width:300px;height:40px'></div>"
    "<script>document.getElementById('empty').attachShadow({{mode: 'open'}});</script>"
    + "<input id='field' style='position:absolute;left:0;top:800px;width:300px;height:30px'"
    " value='" + "word " * 200 + "'>"
    "<script>document.getElementById('field').scrollLeft = 300;</script>"
    + "<div id='host' style='position:absolute;left:0;top:250px;width:300px;height:40px'></div>"
    "<div id='slotted' style='position:absolute;left:0;top:300px;width:300px;height:40px'>"
    "<div style='width:1200px;height:20px'></div></div>"
    "<script>"
    "const box = \"<div id='s' style='width:300px;height:40px;overflow-x:auto'>\";"
    "const host = document.getElementById('host').attachShadow({{mode: 'open'}});"
    "host.innerHTML = box + \"<div style='width:1200px;height:20px'></div></div>\";"
    "host.getElementById('s').scrollLeft = 60;"
    "const slotted = document.getElementById('slotted').attachShadow({{mode: 'open'}});"
    "slotted.innerHTML = box + '<slot></slot></div>';"
    "slotted.getElementById('s').scrollLeft = 60;"
    "const within = document.querySelector('#around > div').attachShadow({{mode: 'open'}});"
    "within.innerHTML = \"<div style='width:1200px;height:20px'></div>\";"
    "</script>"
    "<div style='position:absolute;left:0;top:950px'>"
    + _FRAME.format(height=40, style="transform:scale(2);transform-origin:0 0")
    + "</div><div style='position:absolute;left:0;top:1050px'>"
    + _FRAME.format(height=80, style="transform:scale(.5);transform-origin:0 0")
    + "</div><div style='position:absolute;left:0;top:1100px'>"
    + _FRAME.format(height=40, style="zoom:2")
    + "</div><div id='framed' style='position:absolute;left:0;top:1200px'></div>"
    "<script>document.getElementById('framed').attachShadow({{mode: 'open'}}).innerHTML = \""
    + _FRAME.format(height=40, style="")
    + "\";</script>"
    "<iframe src='/strip' style='position:absolute;left:0;top:350px;width:300px;height:60px;"
    "border:5px solid;padding:3px'></iframe>"
    "<iframe src='http://{other}/strip' style='position:absolute;left:0;top:450px;width:300px;"
    "height:40px;border:0'></iframe>"
    "<object data='/still' type='text/html' style='position:absolute;left:0;top:500px;"
    "width:300px;height:40px'></object>"
    "<embed src='/still' type='text/html' style='position:absolute;left:0;top:550px;"
    "width:300px;height:40px'>"
    "<div style='width:3000px;height:20px'></div></body></html>"
)


class _Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = (
            b"<html><head><title>Tall</title></head><body style='margin:0'>"
            b"<input id='q' autofocus><div style='height:5000px'></div></body></html>"
        )
        if self.path == "/security-key":
            body = _SECURITY_KEY_PAGE
        elif self.path in _DOCUMENTS:
            body = _DOCUMENTS[self.path].encode()
        elif self.path == "/scrollers":
            other = self.headers["Host"].replace("127.0.0.1", "localhost")
            body = _SCROLLERS_PAGE.format(other=other).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


@pytest.fixture
def page_server() -> Any:
    # Threaded with daemon handler threads: Chromium opens speculative
    # connections it never writes to, and a single-threaded server would
    # sit in that handler's readline while shutdown() waited for it.
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def daemon(tmp_path: Path) -> Any:
    printer = ConnPrinter()
    server = AgentServer(printer)
    server.browser_tabs = BrowserTabService(printer, tmp_path / "profile")
    yield server, printer
    server.browser_tabs.shutdown()
    web_stealth.stop_virtual_display()


@pytest.fixture
def live_daemon() -> Any:
    """The tree's live harness: a real daemon on its local endpoint, for real clients."""
    harness = TwoClientHarness()
    harness.setUp()
    yield harness
    harness.server._agent_server.browser_tabs.shutdown()
    web_stealth.stop_virtual_display()
    harness.tearDown()


def _evaluate(service: BrowserTabService, tab_id: str, expression: str, frame: Any = None) -> Any:
    """Read something from the real page, or from *frame* of it (white-box check of an effect)."""
    import asyncio  # noqa: PLC0415

    target = frame or service._pages[tab_id].page
    assert service._loop is not None
    return asyncio.run_coroutine_threadsafe(target.evaluate(expression), service._loop).result(10)


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_browser_commands_end_to_end(daemon: Any, page_server: str) -> None:
    server, printer = daemon
    service = server.browser_tabs

    # An unreachable URL still opens the tab and reports the failure on it.
    server._handle_command({"type": "browserOpen", "url": "http://127.0.0.1:9/", "connId": "c1"})
    opened = _wait(lambda: _events(printer, "openBrowserTab"), "openBrowserTab", timeout=60)[0]
    tab_id = opened["tab_id"]
    # Announced to everyone unfocused, plus a focused copy for the requester.
    assert opened["tabId"] == "" and opened["focus"] is False and opened["browser"]
    assert opened["popup"] is False
    focused = _wait(lambda: _events(printer, "openBrowserTab", connId="c1"), "focus copy")[0]
    assert focused["tab_id"] == tab_id and focused["focus"] is True
    err = _wait(lambda: _events(printer, "browserError", tab_id=tab_id), "browserError")[0]
    assert "127.0.0.1:9" in err["text"] or "ERR_" in err["text"]

    # Navigate to the real page; state follows.
    server._handle_command(
        {"type": "browserNavigate", "tab_id": tab_id, "action": "go", "url": page_server}
    )
    state = _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Tall"), "state")[0]
    # The failed navigation left an error page in history: back is real.
    assert state["url"] == page_server and state["canGoForward"] is False

    # No viewer -> no frames.  A visible viewer -> frames sized to it.
    time.sleep(0.5)
    assert not _events(printer, "browserFrame")
    server._handle_command(
        {
            "type": "browserViewport",
            "tab_id": tab_id,
            "width": 500,
            "height": 400,
            "visible": True,
            "connId": "c1",
        }
    )
    frame = _wait(lambda: _events(printer, "browserFrame", tab_id=tab_id), "frame")[0]
    assert frame["connId"] == "c1" and (frame["width"], frame["height"]) == (500, 400)
    assert base64.b64decode(frame["data"])[:3] == b"\xff\xd8\xff"  # JPEG magic

    # Wheel scrolls the real page; pasted text lands in the focused input.
    server._handle_command(
        {
            "type": "browserInput",
            "tab_id": tab_id,
            "event": {"kind": "mouse", "action": "mouseWheel", "x": 100, "y": 100, "deltaY": 600},
        }
    )
    _wait(lambda: _evaluate(service, tab_id, "window.scrollY") > 0, "page scrolled")
    server._handle_command(
        {"type": "browserInput", "tab_id": tab_id, "event": {"kind": "text", "text": "pasted"}}
    )
    _wait(
        lambda: _evaluate(service, tab_id, "document.getElementById('q').value") == "pasted",
        "paste",
    )
    # Unknown kinds and malformed events are ignored, not fatal.
    server._handle_command({"type": "browserInput", "tab_id": tab_id, "event": {"kind": "nope"}})
    server._handle_command({"type": "browserInput", "tab_id": tab_id, "event": "not-a-dict"})
    server._handle_command(
        {"type": "browserInput", "tab_id": "browser__404", "event": {"kind": "text", "text": "x"}}
    )

    # Reload keeps streaming; hiding the only viewer stops frames.
    server._handle_command({"type": "browserNavigate", "tab_id": tab_id, "action": "reload"})
    _wait(lambda: len(_events(printer, "browserState", tab_id=tab_id)) >= 2, "reload state")
    server._handle_command(
        {"type": "browserViewport", "tab_id": tab_id, "visible": False, "connId": "c1"}
    )
    time.sleep(0.4)
    before = len(_events(printer, "browserFrame"))
    time.sleep(0.6)
    assert len(_events(printer, "browserFrame")) == before

    # A second viewer whose connection drops: frames stop again.
    server._handle_command(
        {
            "type": "browserViewport",
            "tab_id": tab_id,
            "width": 300,
            "height": 300,
            "visible": True,
            "connId": "c2",
        }
    )
    _wait(lambda: _events(printer, "browserFrame", connId="c2"), "frame for c2")
    service.viewer_gone("c2")
    time.sleep(0.4)
    before = len(_events(printer, "browserFrame"))
    time.sleep(0.6)
    assert len(_events(printer, "browserFrame")) == before
    assert service._pages[tab_id].viewers == {}

    # Commands for unknown tabs are harmless.
    server._handle_command({"type": "browserNavigate", "tab_id": "browser__404", "action": "back"})
    server._handle_command({"type": "browserClose", "tab_id": "browser__404"})
    server._handle_command(
        {
            "type": "browserViewport",
            "tab_id": "browser__404",
            "width": 1,
            "height": 1,
            "connId": "c1",
        }
    )

    # Shutdown with a live page: announced closed everywhere, browser gone,
    # and later opens are refused quietly.
    assert [e["tab_id"] for e in service.open_events()] == [tab_id]
    service.shutdown()
    assert _events(printer, "closeBrowserTab", tab_id=tab_id)
    assert service._context is None and service.open_events() == []
    errors = len(_events(printer, "browserError"))
    service.open("http://127.0.0.1:9/", "c1")
    service.viewer_gone("c1")
    time.sleep(0.2)
    assert len(_events(printer, "openBrowserTab")) == 2  # the pair from the first open only
    assert len(_events(printer, "browserError")) == errors  # quietly: no error for it either


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_finger_is_told_whether_a_drag_to_the_right_would_scroll_the_page(
    daemon: Any, page_server: str,
) -> None:
    """A ``touch`` input event asks what is under a finger; the page's viewers get the answer.

    ``scrollsLeft`` says that something under the point (the element,
    an ancestor, a frame around it or the document) scrolls sideways
    and is not at its left end, or that a plugin draws there, which
    cannot be asked: a drag to the right there is the page's, not a
    swipe of the surface showing it.
    """
    server, printer = daemon
    service = server.browser_tabs
    server._handle_command(
        {"type": "browserOpen", "url": page_server + "scrollers", "connId": "c1"}
    )
    tab_id = _wait(lambda: _events(printer, "openBrowserTab"), "the tab", timeout=60)[0]["tab_id"]

    def show(conn_id: str, visible: bool) -> None:
        server._handle_command({
            "type": "browserViewport", "tab_id": tab_id, "width": 500, "height": 1300,
            "visible": visible, "connId": conn_id,
        })

    def loaded(title: str) -> None:
        """Wait for the page titled *title* (the blank one a tab starts on is
        "complete" too) and its frames: the one of another site loads last."""
        _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title=title), title)
        _wait(
            lambda: _evaluate(service, tab_id, "document.readyState") == "complete",
            f"the frames of {title}",
        )

    def touch(y: int) -> str:
        """Put a finger down 100px in and *y* px down; the id its answer carries."""
        touch_id = f"touch-{y}-{len(_events(printer, 'browserTouch'))}"
        server._handle_command({
            "type": "browserInput", "tab_id": tab_id,
            "event": {"kind": "touch", "id": touch_id, "x": 100, "y": y},
        })
        return touch_id

    def scrolls_left(y: int) -> bool:
        """What both viewers are told about a finger put down *y* px down."""
        touch_id = touch(y)
        told = _wait(
            lambda: len(found := _events(printer, "browserTouch", id=touch_id)) == 2 and found,
            f"both viewers told about y={y}",
        )
        assert {(e["connId"], e["tab_id"]) for e in told} == {("c1", tab_id), ("c2", tab_id)}
        answer = told[0]["scrollsLeft"]
        assert isinstance(answer, bool) and told[1]["scrollsLeft"] is answer
        return answer

    show("c1", True)
    show("c2", True)
    loaded("Scrollers")
    frames = [f for f in service._pages[tab_id].page.frames if f.url.endswith("/strip")]
    _wait(
        lambda: [_evaluate(service, tab_id, _SCROLL_BOX_JS, frame) for frame in frames] == [80] * 6,
        "the box of each frame scrolled",
    )
    # What is under the finger: how far down the page it is (a point on
    # a box is on the 20px line inside it, 10px down it) and the answer.
    under_the_finger = {
        "a box scrolled sideways": (10, True),
        "a box at its left end": (60, False),
        "a right-to-left box at its start": (110, True),
        "a right-to-left box at its left end": (160, False),
        "a box only a script scrolls": (210, False),
        "a box in a shadow root": (260, True),
        "a child of the page in a shadow root's box": (310, True),
        "the border of a frame": (352, False),
        "a box in a frame of this site": (368, True),
        "the last lines of that box": (396, True),
        "that frame's page under the box": (400, False),
        "a box in a frame of another site": (460, True),
        "a page embedded as an object": (510, True),
        "a page embedded as a plugin": (560, True),
        "the document at its left end": (600, False),
        "a box whose lines run from right to left, at its start": (660, True),
        "a box whose text runs upwards, at its left end": (710, False),
        "a right-to-left box of a fractional width at its left end": (760, False),
        "a text field whose text is scrolled": (810, True),
        "an empty shadow root": (860, False),
        "a shadow root in a box scrolled sideways": (910, True),
        "a box in a frame drawn at twice its size": (1010, True),
        "under the box, in a frame drawn at half its size": (1075, False),
        "a box in a frame zoomed to twice its size": (1160, True),
        "a box in a frame in a shadow root": (1210, True),
        "nothing: a point off the page": (5000, False),
    }
    assert {what: scrolls_left(y) for what, (y, _) in under_the_finger.items()} == {
        what: expected for what, (_, expected) in under_the_finger.items()
    }
    # The document scrolled sideways: whatever the finger lands on, a
    # drag to the right scrolls the page back.
    server._handle_command({
        "type": "browserInput", "tab_id": tab_id,
        "event": {"kind": "mouse", "action": "mouseWheel", "x": 100, "y": 600, "deltaX": 100},
    })
    _wait(lambda: _evaluate(service, tab_id, "window.scrollX") == 100, "the document scrolled")
    assert scrolls_left(600) and scrolls_left(60)
    # Other documents, each after the script that scrolls it: one that
    # reads right to left starts at its right end, one that forbids
    # sideways scrolling moves under no finger, and in one the body is
    # what scrolls.
    left_end = "scrollTo(-100000, 0)"
    documents = {
        "rtl": {"0": True, left_end: False},
        "hidden": {"scrollTo(300, 0)": False},
        "clip": {"0": False},
        "own": {"document.body.scrollLeft = 300": True, "document.body.scrollLeft = 0": False},
        "old": {"0": True, left_end: False},
    }
    for name, scripts in documents.items():
        server._handle_command({
            "type": "browserNavigate", "tab_id": tab_id, "action": "go", "url": page_server + name,
        })
        loaded(name)
        for script, expected in scripts.items():
            _evaluate(service, tab_id, script)
            assert scrolls_left(10) is expected, (name, script)
    # Nobody showing the page: nobody to tell.
    show("c1", False)
    show("c2", False)
    told = len(_events(printer, "browserTouch"))
    touch(10)
    time.sleep(0.5)
    assert len(_events(printer, "browserTouch")) == told


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_closing_a_tab_whose_address_bar_navigation_is_in_flight_reports_no_error(
    daemon: Any, page_server: str,
) -> None:
    """The navigation a close aborts is no failure: no surface gets an error for the gone tab."""
    server, printer = daemon
    # Accepts and never answers: a navigation to it stays in flight.
    hung = socket.socket()
    hung.bind(("127.0.0.1", 0))
    hung.listen(8)
    hung.settimeout(30)
    held: list[socket.socket] = []
    try:
        server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
        loaded = _wait(lambda: _events(printer, "browserState", title="Tall"), "page", timeout=60)
        tab_id = loaded[0]["tab_id"]
        server._handle_command({
            "type": "browserNavigate", "tab_id": tab_id, "action": "go",
            "url": f"http://127.0.0.1:{hung.getsockname()[1]}/",
        })
        held.append(hung.accept()[0])  # the navigation reached the server
        server._handle_command({"type": "browserClose", "tab_id": tab_id})
        _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "close")
        time.sleep(1.0)  # the aborted navigation fails as the page closes
        assert _events(printer, "browserError") == []
    finally:
        for conn in held:
            conn.close()
        hung.close()


def _unclosed_event_loops() -> list[Any]:
    """Every event loop object alive in this process that is not closed."""
    return [
        obj for obj in gc.get_objects()
        if issubclass(type(obj), asyncio.AbstractEventLoop) and not obj.is_closed()
    ]


@pytest.mark.skipif(not thread_start_can_be_starved(), reason="requires enforced RLIMIT_NPROC")
def test_an_open_whose_loop_thread_cannot_start_is_refused_to_its_asker(daemon: Any) -> None:
    """No thread for the private loop: the asker is told (its ``openId`` echoed), no loop leaks."""
    server, printer = daemon
    service = server.browser_tabs
    before = _unclosed_event_loops()
    with nproc_limit_lowered_to_one():
        server._handle_command({
            "type": "browserOpen", "url": "http://127.0.0.1:9/", "connId": "c1", "openId": "o1",
        })
    refusals = _events(printer, "browserError", connId="c1")
    assert [(e["tab_id"], e["openId"]) for e in refusals] == [("", "o1")]
    assert "thread" in refusals[0]["text"]
    assert service._loop is None and service._thread is None
    # The loop no thread will ever run is closed there and then: left
    # open, its selector and self-pipe stay until a collection finds it.
    leaked = [loop for loop in _unclosed_event_loops() if not any(loop is b for b in before)]
    assert leaked == []


def test_commands_for_no_tab_start_no_browser_loop(daemon: Any) -> None:
    """Only an open starts the service's thread: nothing else has a tab to act on before it."""
    server, _printer = daemon
    service = server.browser_tabs
    for cmd in (
        {"type": "browserViewport", "tab_id": "browser__1", "visible": False, "connId": "c1"},
        {"type": "browserFrameAck", "tab_id": "browser__1", "connId": "c1"},
        {"type": "browserNavigate", "tab_id": "browser__1", "action": "reload"},
        {"type": "browserInput", "tab_id": "browser__1", "event": {"kind": "text", "text": "x"}},
        {"type": "browserClose", "tab_id": "browser__1"},
    ):
        server._handle_command(cmd)
    service.interrupt("browser__1")
    service.viewer_gone("c1")
    assert service._loop is None and service._thread is None


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_browser_that_cannot_launch_falls_back_to_bundled_chromium(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, page_server: str, tmp_path: Path
) -> None:
    """A default browser that dies on start (a half-updated Chrome) still shows the page."""
    server, printer = daemon
    # A "browser" that exits immediately: Playwright fails to launch it.
    fake = _fake_binary(tmp_path, "not-a-browser")
    monkeypatch.setenv("KISS_BROWSER", str(fake))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c9"})
    opened = _wait(
        lambda: _events(printer, "openBrowserTab", connId="c9"), "openBrowserTab", timeout=90
    )[0]
    assert opened["browser"] == "Chromium (bundled)" and opened["isDefaultBrowser"] is False
    # The badge's tooltip names the browser that failed and Playwright's
    # headline, not its command line or call log.
    assert opened["note"].startswith("not-a-browser could not be started: ")
    assert "Failed to launch the browser process" in opened["note"]
    assert "<launching>" not in opened["note"] and "Call log" not in opened["note"]
    assert not _events(printer, "browserError")
    _wait(lambda: _events(printer, "browserState", tab_id=opened["tab_id"], title="Tall"), "state")
    server.browser_tabs.shutdown()

    # Without the bundled Chromium either, the requester gets a short error
    # (the headline, not the call log) and no tab.
    server.browser_tabs = BrowserTabService(printer, tmp_path / "profile2")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "no-browsers"))
    server._handle_command({"type": "browserOpen", "url": "", "connId": "c10"})
    err = _wait(lambda: _events(printer, "browserError", connId="c10"), "launch error", timeout=60)[
        0
    ]
    assert err["tab_id"] == "" and err["text"]
    assert "Call log" not in err["text"] and len(err["text"]) < 1000
    assert not _events(printer, "openBrowserTab", connId="c10")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_headed_launch_falls_back_to_headless(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, page_server: str
) -> None:
    """A desktop session without a reachable window server still streams."""
    server, printer = daemon
    monkeypatch.delenv("KISS_HEADLESS", raising=False)
    monkeypatch.setenv("DISPLAY", ":99")  # nothing listens here
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for var in ("KISS_BROWSER", "BROWSER"):
        monkeypatch.delenv(var, raising=False)
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    opened = _wait(lambda: _events(printer, "openBrowserTab"), "openBrowserTab", timeout=90)[0]
    _wait(lambda: _events(printer, "browserState", tab_id=opened["tab_id"], title="Tall"), "state")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(shutil.which("Xvfb") is None, reason="Xvfb not installed")
def test_headless_environment_runs_a_headed_browser_on_xvfb(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, page_server: str
) -> None:
    """Without a display the tab still shows no headless or automation tell.

    Google's sign-in answers a ``HeadlessChrome`` client-hint brand or
    ``navigator.webdriver`` with "Couldn't sign you in", which would
    break every OAuth hand-off streamed through the tab.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    web_stealth.stop_virtual_display()
    assert service.open_for_user(page_server) is True
    tab_id = _events(printer, "openBrowserTab")[0]["tab_id"]
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Tall"), "state")
    brands = _evaluate(service, tab_id, "navigator.userAgentData.brands.map(b => b.brand).join()")
    assert "HeadlessChrome" not in brands and "Chrom" in brands
    assert "HeadlessChrome" not in _evaluate(service, tab_id, "navigator.userAgent")
    assert _evaluate(service, tab_id, "navigator.webdriver") is False
    assert web_stealth.virtual_display() is not None, "the browser runs on the shared Xvfb"


def _click_other_options(server: AgentServer, tab_id: str) -> None:
    """Click the (100..300, 100..150) button through the streamed tab's input path."""
    for action in ("mousePressed", "mouseReleased"):
        server._handle_command(
            {
                "type": "browserInput",
                "tab_id": tab_id,
                "event": {
                    "kind": "mouse",
                    "action": action,
                    "x": 200,
                    "y": 125,
                    "button": "left",
                    "buttons": 1 if action == "mousePressed" else 0,
                    "clickCount": 1,
                },
            }
        )


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(shutil.which("Xvfb") is None, reason="Xvfb not installed")
def test_security_key_prompt_does_not_wedge_the_streamed_tab(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, page_server: str
) -> None:
    """A page asking for a security key stays clickable in the Browser tab.

    Chrome answers ``navigator.credentials.get`` with a native modal
    "Use your security key" dialog.  On an Xvfb display nobody sees it,
    the screencast does not carry it, and it swallows every click on the
    page underneath, so Duo's "Other options" button could not be
    pressed.  An unattended browser therefore exposes no WebAuthn API:
    the page skips the request and the click lands.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    web_stealth.stop_virtual_display()
    assert service.open_for_user(page_server + "security-key") is True
    tab_id = _events(printer, "openBrowserTab")[0]["tab_id"]
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Login"), "state")
    assert web_stealth.virtual_display() is not None, "the browser runs headed on Xvfb"
    assert _evaluate(service, tab_id, "typeof window.PublicKeyCredential") == "undefined"
    _click_other_options(server, tab_id)
    _wait(
        lambda: _evaluate(service, tab_id, "document.title") == "Other options",
        "the click to reach the page",
    )


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_without_xvfb_the_tab_falls_back_to_masked_headless(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, page_server: str, tmp_path: Path
) -> None:
    """No Xvfb binary: real headless mode, with the HeadlessChrome UA token masked."""
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    web_stealth.stop_virtual_display()
    # A PATH with every current executable except Xvfb, so browser
    # discovery (KISS_BROWSER, installed browsers) keeps working.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for entry in os.environ["PATH"].split(os.pathsep):
        for exe in Path(entry).glob("*") if Path(entry).is_dir() else ():
            if exe.name != "Xvfb" and not (bin_dir / exe.name).is_symlink():
                (bin_dir / exe.name).symlink_to(exe)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert shutil.which("Xvfb") is None
    assert service.open_for_user(page_server) is True
    tab_id = _events(printer, "openBrowserTab")[0]["tab_id"]
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Tall"), "state")
    assert web_stealth.virtual_display() is None
    assert "HeadlessChrome" not in _evaluate(service, tab_id, "navigator.userAgent")
    assert _evaluate(service, tab_id, "navigator.webdriver") is False
    # Headless Chrome shows the same invisible security-key dialog.
    assert _evaluate(service, tab_id, "typeof window.PublicKeyCredential") == "undefined"


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_open_for_user_focuses_every_surface_and_reports_failures(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sign-in hand-off opens a focused tab everywhere and blocks until it loads."""
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")

    assert service.open_for_user(page_server) is True
    # Announced to every surface (tabId ""), then focused everywhere: the
    # user has to act on the page, so no surface stays on another tab.
    opened = _events(printer, "openBrowserTab")
    assert [event["focus"] for event in opened] == [False, True]
    assert {event["tabId"] for event in opened} == {""}
    tab_id = opened[0]["tab_id"]
    assert opened[1]["tab_id"] == tab_id
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Tall"), "state")
    assert _evaluate(service, tab_id, "location.href") == page_server

    # A page that cannot load is no hand-off: the blank tab is closed again.
    assert service.open_for_user("http://127.0.0.1:9/") is False
    _wait(lambda: len(_events(printer, "closeBrowserTab")) == 1 or None, "tab closed")
    assert set(service._pages) == {tab_id}

    # After shutdown nothing can be opened, and the failure is reported, not raised.
    service.shutdown()
    assert service.open_for_user(page_server) is False


def _task_on(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Start *coro* as a task on the service's loop (what ``wait_for`` cancels on a timeout)."""
    started: concurrent.futures.Future[asyncio.Task[Any]] = concurrent.futures.Future()
    loop.call_soon_threadsafe(lambda: started.set_result(loop.create_task(coro)))
    return started.result(5)


async def _cancelled(task: asyncio.Task[Any]) -> bool:
    await asyncio.wait({task})
    return task.cancelled()


def _attaching_new_page(local: dict[str, Any]) -> bool:
    """The first resumption of a new page's ``_register``: its CDP session just came back."""
    return local.get("is_new") is True and "cdp" not in local


def _coroutine_name(task: asyncio.Task[Any] | None) -> str:
    """The qualified name of the coroutine *task* runs (``""`` for no task)."""
    return getattr(task.get_coro() if task is not None else None, "__qualname__", "")


def _entered_by_page_event(local: dict[str, Any]) -> bool:
    """``_register`` entered by the context's own ``page`` registration (a task of its own)."""
    return _coroutine_name(asyncio.current_task()).endswith("._register")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.parametrize("held_at", [_attaching_new_page, _entered_by_page_event])
def test_an_open_cancelled_while_its_page_attaches_leaves_no_page_behind(
    live_daemon: TwoClientHarness, page_server: str, monkeypatch: pytest.MonkeyPatch,
    held_at: Callable[[dict[str, Any]], bool],
) -> None:
    """An open cancelled during the page's attach closes that page instead of losing it.

    ``asyncio.wait_for(_open, _OPEN_TIMEOUT)`` cancels the open's task;
    landing inside ``_register`` (a browser slow to answer CDP) it used
    to leave the page behind: attached by the context's own ``page``
    registration, it was announced as a blank tab nobody asked for,
    or, when the open's registration was the one cancelled, it stayed
    open in the browser with no record, so no surface showed it and
    nothing could close it.  The open's task is cancelled at exactly
    that point: the loop is held when the new page's ``_register``
    resumes with its CDP session, and the cancel is queued before it
    goes on.  Afterwards the browser holds the first tab alone.

    Held instead where the ``page`` event registers the new page, the
    cancel lands earlier, while the open still awaits ``new_page()``:
    the open then had no page to close, and the tab the page event
    announced stayed open on every surface.

    The surfaces are two real clients of a real daemon: what each was
    told to open and to close is read off its own connection.
    """
    service = live_daemon.server._agent_server.browser_tabs
    surfaces = [live_daemon.open_client(name) for name in ("a", "b")]
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    first = live_daemon.wait_event(surfaces[0], "openBrowserTab")["tab_id"]
    loop = service._loop
    context = service._context
    assert loop is not None and context is not None
    # The first tab, plus the blank page a persistent context starts with.
    pages_before = set(context.pages)

    attaching = Pause(BrowserTabService._register.__code__, held_at)
    with attaching:
        task = _task_on(loop, service._open(page_server, ""))
        assert attaching.paused.wait(30), "the new page never reached its attach"
        loop.call_soon_threadsafe(task.cancel)
        attaching.release()
    assert asyncio.run_coroutine_threadsafe(_cancelled(task), loop).result(30) is True

    _wait(lambda: set(context.pages) == pages_before or None, "the unwanted page to close")
    _wait(lambda: set(service._pages) == {first} or None, "the unwanted page to be forgotten")
    with service._lock:
        assert set(service._by_page) == {id(service._pages[first].page)}
    # The browser is intact: the next open lands in it.
    assert service.open_for_user(page_server) is True
    assert service._context is context and len(context.pages) == len(pages_before) + 1
    (second,) = set(service._pages) - {first}
    # A connection gets its frames in the daemon's order: once it is
    # told to focus the new tab, a blank tab the page event's
    # registration announced meanwhile was closed again there, and
    # the first tab was not.
    for surface in surfaces:
        live_daemon.wait_event(surface, "openBrowserTab", tab_id=second, focus=True)
        opened = {e["tab_id"] for e in surface.of_type("openBrowserTab")}
        closed = {e["tab_id"] for e in surface.of_type("closeBrowserTab")}
        assert opened - closed == {first, second}, (surface.name, opened, closed)


def _browser_pid(profile_dir: Path) -> int:
    """The browser process of *profile_dir*: the profile on its command line, no ``--type``.

    Chromium rewrites its own process title, after which ``cmdline`` holds the
    arguments joined by spaces instead of NULs, so both separators split it
    (the pytest profile path has no spaces).
    """
    own = f"--user-data-dir={profile_dir}".encode()
    pids = []
    for proc in Path("/proc").iterdir():
        try:
            raw = (proc / "cmdline").read_bytes() if proc.name.isdigit() else b""
            args = raw.replace(b"\0", b" ").split()
        except OSError:
            continue
        if own in args and not any(arg.startswith(b"--type=") for arg in args):
            pids.append(int(proc.name))
    assert len(pids) == 1, pids
    return pids[0]


def _entered_by_open(local: dict[str, Any]) -> bool:
    """``_register`` entered by the open's ``_new_tab`` (inside the open's task)."""
    return _coroutine_name(asyncio.current_task()).endswith("._open")


async def _tasks_running(name: str) -> int:
    return len([task for task in asyncio.all_tasks() if _coroutine_name(task).endswith(name)])


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(sys.platform != "linux", reason="SIGSTOP and /proc")
def test_a_cancelled_open_whose_browser_stops_answering_settles_within_the_bound(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup of a cancelled open does not wait for an attachment that never comes.

    The context's ``page`` event registers a new page first (its task
    is queued before the ``new_page`` reply wakes the open), so the
    open's own registration is the second one, waiting on
    ``attached``; the open's cancel hits that wait and leaves the
    page event's attach running.  A browser that stops answering
    (here: SIGSTOP, sent while the loop is held at the page event's
    ``_register`` entry) never completes that attach: the attach is
    bounded (``_ATTACH_TIMEOUT``), so the registration forgets the
    page and the cleanup takes the bounded close path instead of
    waiting on ``attached`` for good with the record and the page
    behind it.  Once the browser answers again the queued close takes
    the page down, and the next open lands in the same browser.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    first = _events(printer, "openBrowserTab")[0]["tab_id"]
    loop = service._loop
    context = service._context
    assert loop is not None and context is not None
    pages_before = set(context.pages)
    pid = _browser_pid(service._profile_dir)

    page_event = Pause(BrowserTabService._register.__code__, _entered_by_page_event)
    open_side = Pause(BrowserTabService._register.__code__, _entered_by_open)
    stopped = False
    try:
        with Pauses(page_event, open_side):
            task = _task_on(loop, service._open(page_server, ""))
            assert page_event.paused.wait(30), "the page event never registered the new page"
            os.kill(pid, signal.SIGSTOP)
            stopped = True
            page_event.release()
            assert open_side.paused.wait(30), "the open never reached its own registration"
            # Queued behind the held step: the cancel lands once the
            # open waits on ``attached``.
            loop.call_soon_threadsafe(task.cancel)
            open_side.release()
        assert asyncio.run_coroutine_threadsafe(_cancelled(task), loop).result(30) is True

        def settled() -> bool:
            with service._lock:
                forgotten = set(service._pages) == {first}
            cleanups = asyncio.run_coroutine_threadsafe(
                _tasks_running("._close_unwanted"), loop,
            ).result(10)
            return forgotten and cleanups == 0

        _wait(
            settled,
            "the cancelled open's page to be forgotten and its cleanup to end",
            timeout=_ATTACH_TIMEOUT + _CLOSE_TIMEOUT + 10,
        )
        # Never attached, never announced.
        assert {e["tab_id"] for e in _events(printer, "openBrowserTab")} == {first}
    finally:
        if stopped:
            os.kill(pid, signal.SIGCONT)
    # The close sent to the stopped browser takes the page down once it answers again.
    _wait(lambda: set(context.pages) == pages_before or None, "the unwanted page to close")
    with service._lock:
        assert set(service._by_page) == {id(service._pages[first].page)}
    assert not _events(printer, "closeBrowserTab", tab_id=first)
    assert service.open_for_user(page_server) is True
    assert service._context is context and len(context.pages) == len(pages_before) + 1


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(sys.platform != "linux", reason="SIGSTOP and /proc")
def test_an_open_whose_browser_stops_answering_fails_and_closes_its_page(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open whose attach reaches its bound fails, and its page is closed, not left behind.

    Nobody cancels this open: an agent thread's ``open_for_user`` just
    waits it out.  The browser stops answering (SIGSTOP, sent while the
    loop is held at the page event's ``_register`` entry) once the page
    exists, so the attach reaches ``_ATTACH_TIMEOUT``, both
    registrations forget the page and the open fails within that bound.
    The page used to be the failed open's only loose end: alive in the
    browser once it answered again, with no record, so no surface showed
    it and nothing could close it.  The failed open closes it like a
    cancelled one does, and the next open lands in the same browser.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    first = _events(printer, "openBrowserTab")[0]["tab_id"]
    context = service._context
    assert context is not None
    pages_before = set(context.pages)
    pid = _browser_pid(service._profile_dir)

    page_event = Pause(BrowserTabService._register.__code__, _entered_by_page_event)
    agent_thread = concurrent.futures.ThreadPoolExecutor(1)
    stopped = False
    try:
        with page_event:
            opening = agent_thread.submit(service.open_for_user, page_server)
            assert page_event.paused.wait(30), "the page event never registered the new page"
            os.kill(pid, signal.SIGSTOP)
            stopped = True
            page_event.release()
        # Settled by the attach bound, not by the open's own (longer) one.
        assert opening.result(_ATTACH_TIMEOUT + 10) is False
        # Never attached, never announced.
        with service._lock:
            assert set(service._pages) == {first}
            assert set(service._by_page) == {id(service._pages[first].page)}
        assert {e["tab_id"] for e in _events(printer, "openBrowserTab")} == {first}
    finally:
        if stopped:
            os.kill(pid, signal.SIGCONT)
        agent_thread.shutdown()
    # The close sent to the stopped browser takes the page down once it answers again.
    _wait(lambda: set(context.pages) == pages_before or None, "the failed open's page to close")
    assert not _events(printer, "closeBrowserTab", tab_id=first)
    assert service.open_for_user(page_server) is True
    assert service._context is context and len(context.pages) == len(pages_before) + 1


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(sys.platform != "linux", reason="SIGSTOP and /proc")
def test_an_interrupt_the_browser_does_not_answer_settles_and_closes_the_tab(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The input watchdog's recovery gives up within its bound on a browser that stops answering.

    ``interrupt`` first asks the page's CDP session to terminate the
    running script, then probes the page, and closes the tab when it
    still does not answer.  A browser that has stopped answering
    (SIGSTOP) answers neither request: the terminate request used to
    be unbounded, so the recovery never reached its probe nor the
    close fallback, and the wedged tab (and the agent's blocked input
    call behind the watchdog) stayed that way.  Both requests are
    bounded by ``_INTERRUPT_PROBE_TIMEOUT`` now, so the recovery falls
    through to the bounded close, which takes the page down once the
    browser answers again.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    tab_id = _events(printer, "openBrowserTab")[0]["tab_id"]
    loop = service._loop
    assert loop is not None
    pid = _browser_pid(service._profile_dir)
    os.kill(pid, signal.SIGSTOP)
    try:
        recovery = asyncio.run_coroutine_threadsafe(service._interrupt(tab_id), loop)
        # Settled by the probe and close bounds, not by the test's patience.
        recovery.result(_INTERRUPT_PROBE_TIMEOUT + 3 * _CLOSE_TIMEOUT + 10)
        assert service._pages[tab_id].closing is True
    finally:
        os.kill(pid, signal.SIGCONT)
    # The close sent to the stopped browser takes the tab down once it answers again.
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the wedged tab to close")


def test_shutdown_settles_an_outstanding_submission(daemon: Any) -> None:
    """Closing the private loop cancels outstanding futures and runs their finalizers."""
    server, _printer = daemon
    service = server.browser_tabs
    entered = threading.Event()
    finalized = threading.Event()

    async def pending_operation() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    future = service._run_threadsafe(pending_operation())
    assert entered.wait(10)
    service.shutdown()
    assert future.done(), "shutdown abandoned the submitted future"
    assert future.cancelled()
    assert finalized.is_set(), "the pending operation's finalizer never ran"


def _registering_popup(local: dict[str, Any]) -> bool:
    """``_register`` entered for a page the browser opened itself."""
    return local.get("popup") is True


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
@pytest.mark.skipif(sys.platform != "linux", reason="SIGSTOP and /proc")
def test_a_popup_whose_attach_fails_is_closed_not_left_behind(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A popup the service cannot attach is closed like a failed open's page.

    A ``window.open`` page is registered by the context's ``page``
    event alone: no open waits for it, so when its attach reaches
    ``_ATTACH_TIMEOUT`` (the browser stops answering, SIGSTOP, once
    the popup exists) nothing used to close it.  The page lived on
    with no record: no surface showed it, nothing could close it, and
    it kept the browser alive.  The registration now closes a popup
    it gave up on, and the close lands once the browser answers.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    first = _events(printer, "openBrowserTab")[0]["tab_id"]
    loop = service._loop
    context = service._context
    assert loop is not None and context is not None
    pages_before = set(context.pages)
    pid = _browser_pid(service._profile_dir)

    popup_event = Pause(BrowserTabService._register.__code__, _registering_popup)
    stopped = False
    try:
        with popup_event:
            opening = asyncio.run_coroutine_threadsafe(
                service._pages[first].page.evaluate("void window.open('about:blank')"), loop,
            )
            assert popup_event.paused.wait(30), "the page event never registered the popup"
            os.kill(pid, signal.SIGSTOP)
            stopped = True
            popup_event.release()
        assert len(context.pages) == len(pages_before) + 1
        _wait(lambda: len(service._pages) == 2 or None, "the popup to be reserved")
        # Never attached, never announced: the registration forgets the popup within the bound.
        _wait(
            lambda: set(service._pages) == {first} or None,
            "the popup to be forgotten",
            timeout=_ATTACH_TIMEOUT + 10,
        )
    finally:
        if stopped:
            os.kill(pid, signal.SIGCONT)
    # The browser answers the ``window.open`` only now, and the close sent
    # to it meanwhile takes the popup down.
    opening.result(30)
    _wait(lambda: set(context.pages) == pages_before or None, "the popup to close")
    assert {e["tab_id"] for e in _events(printer, "openBrowserTab")} == {first}
    assert not _events(printer, "closeBrowserTab")
