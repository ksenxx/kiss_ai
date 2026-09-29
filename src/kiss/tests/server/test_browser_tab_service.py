# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the browser-tab service and default-browser resolution.

The multi-surface behaviour lives in ``test_browser_tab_all_surfaces``;
this module exercises the daemon-side pieces that test does not reach:
address-bar normalisation, resolving the machine's default browser
from real ``.desktop`` files and ``$BROWSER``/``$KISS_BROWSER``, the
``browser*`` daemon commands driving a real browser through
:class:`VSCodeServer`, the error paths (unreachable URL, a "browser"
that exits at once), wheel/paste input, hiding a viewer, a viewer
disconnecting, and shutdown with pages still open.

The macOS (``NSWorkspace``) and Windows (registry) probes cannot run on
a Linux CI host; they are exercised only through their shared helpers.
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core import default_browser as db
from kiss.server.browser_tab import (
    BrowserTabService,
    _key_params,
    _mouse_params,
    decode_frame,
    home_url,
    normalize_url,
)
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter

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
    assert normalize_url("localhost:8080/app") == "https://localhost:8080/app"
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
    # A minimal .app bundle: Info.plist names the executable.
    app = tmp_path / "Fake.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
        b"<key>CFBundleExecutable</key><string>Fake</string></dict></plist>"
    )
    assert db._mac_app_executable(str(app)) is None  # binary missing
    exe = _fake_binary(app / "Contents" / "MacOS", "Fake")
    assert db._mac_app_executable(str(app)) == str(exe)
    (app / "Contents" / "Info.plist").write_bytes(b"not a plist")
    assert db._mac_app_executable(str(app)) is None
    assert db._mac_app_executable(str(tmp_path / "Missing.app")) is None


# ---------------------------------------------------------------------------
# The daemon commands against a real browser
# ---------------------------------------------------------------------------


class _Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = (
            b"<html><head><title>Tall</title></head><body style='margin:0'>"
            b"<input id='q' autofocus><div style='height:5000px'></div></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


@pytest.fixture
def page_server() -> Any:
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def daemon(tmp_path: Path) -> Any:
    printer = ConnPrinter()
    server = VSCodeServer(printer)
    server.browser_tabs = BrowserTabService(printer, tmp_path / "profile")
    yield server, printer
    server.browser_tabs.shutdown()


def _evaluate(service: BrowserTabService, tab_id: str, expression: str) -> Any:
    """Read something from the real page (white-box check of an input effect)."""
    import asyncio  # noqa: PLC0415

    rec = service._pages[tab_id]
    assert service._loop is not None
    return asyncio.run_coroutine_threadsafe(rec.page.evaluate(expression), service._loop).result(10)


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
    assert decode_frame(frame)[:3] == b"\xff\xd8\xff"  # JPEG magic

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
    service.open("http://127.0.0.1:9/", "c1")
    service.viewer_gone("c1")
    time.sleep(0.2)
    assert len(_events(printer, "openBrowserTab")) == 2  # the pair from the first open only


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_browser_that_cannot_launch_reports_to_requester(
    daemon: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, printer = daemon
    # A "browser" that exits immediately: Playwright fails to launch it.
    fake = Path(tempfile.mkdtemp()) / "not-a-browser"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("KISS_BROWSER", str(fake))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    server._handle_command({"type": "browserOpen", "url": "", "connId": "c9"})
    err = _wait(lambda: _events(printer, "browserError", connId="c9"), "launch error", timeout=60)[
        0
    ]
    assert err["tab_id"] == "" and err["text"]
    assert not _events(printer, "openBrowserTab")
    shutil.rmtree(fake.parent, ignore_errors=True)


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
