# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``kiss-app`` opens the webapp as a standalone app.

:mod:`kiss.server.standalone_app` waits for the daemon's loopback URL,
picks a Chromium-family browser and starts it in app mode under the
brand home's own profile directory; ``--install-launcher`` registers a
brand-named desktop launcher.  The command line is exercised through
the real ``kiss-app`` console script of this checkout's venv, with a
stub "browser" that records its arguments, and (where an X server can
be started) with a real browser against a local HTTP server.  The
branded cases run the same code from a re-branded copy of the package
(:mod:`kiss.tests.branded_package`).
"""

from __future__ import annotations

import http.server
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core.brand import BRAND
from kiss.server import standalone_app
from kiss.tests.branded_package import BRAND_ICON_PNG, SEAMLESS_LOOP, make_branded_package
from kiss.tests.conftest import find_free_port, posix_only

pytestmark = posix_only("runs the kiss-app console script and stub browsers under bash")

_SRC = Path(__file__).resolve().parents[2]
_KISS_APP = Path(sys.executable).parent / "kiss-app"
_LOOPBACK = "http://127.0.0.1:8787"
_MACOS = platform.system() == "Darwin"


def _app_flag(url: str, product_name: str = BRAND["product_name"]) -> str:
    """The ``--app=`` argument kiss-app passes for *url* on this platform.

    macOS: the daemon's ``media/appWindow.html`` page, which opens *url*
    as a popup window (the Dock-click fix, see the module); elsewhere
    *url* itself.
    """
    if not _MACOS:
        return f"--app={url}"
    query = urllib.parse.urlencode({"url": url, "title": product_name})
    return f"--app={urllib.parse.urljoin(url, '/media/appWindow.html')}?{query}"


# The flags that follow the common ones on this platform: macOS lets the
# appWindow.html page open its popup without a click.
_PLATFORM_FLAGS = ["--disable-popup-blocking"] if _MACOS else []


def _write_stub_browser(
    path: Path, log: Path, *, sleep: float = 0, exit_code: int = 0, stderr: str = ""
) -> Path:
    """A fake browser: logs its arguments one per line, prints *stderr*, sleeps, exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" >> "{log}"\n'
        f"[ -z '{stderr}' ] || echo '{stderr}' >&2\nsleep {sleep}\nexit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _stub_calls(log: Path) -> list[str]:
    for _ in range(50):
        if log.exists():
            return log.read_text(encoding="utf-8").splitlines()
        time.sleep(0.1)
    return []


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """A clean environment: temp HOME and KISS_HOME, no browser override, no XDG dirs."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("KISS_HOME", "KISS_APP_BROWSER", "XDG_DATA_HOME")
    }
    env["HOME"] = str(tmp_path / "home")
    env["KISS_HOME"] = str(tmp_path / "kiss-home")
    env.update(extra)
    return env


def _browser_env(tmp_path: Path, browser: Path, **extra: str) -> dict[str, str]:
    """``_env`` naming *browser*, with Playwright's browsers directory still known.

    Playwright looks for its browsers under HOME unless
    ``PLAYWRIGHT_BROWSERS_PATH`` names the directory, and the app asks
    it whether *browser* is the bundled Chromium (``sandboxed``).
    """
    registry = next((p.parent for p in browser.parents if p.name.startswith("chromium-")), None)
    if registry is not None:
        extra["PLAYWRIGHT_BROWSERS_PATH"] = str(registry)
    return _env(tmp_path, KISS_APP_BROWSER=str(browser), **extra)


def _run(
    args: list[str], env: dict[str, str], exe: Path = _KISS_APP
) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(exe), *args], capture_output=True, text=True, env=env, timeout=120)


def _site(port: int = 0) -> tuple[http.server.HTTPServer, str]:
    """Start a :class:`_Hits` site (a stand-in daemon) on *port*; return it and its URL."""
    _Hits.paths = []
    _Hits.app_window_open = False
    _Hits.close = False
    httpd = http.server.HTTPServer(("127.0.0.1", port), _Hits)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}/"


# --- the daemon's URL -------------------------------------------------------


def test_loopback_url_prefers_loopback_then_rewrites_local(tmp_path: Path) -> None:
    """``loopback`` wins; an old file with only ``local`` is rewritten to 127.0.0.1."""
    url_file = tmp_path / "remote-url.json"
    url_file.write_text(json.dumps({"local": "http://localhost:8787", "loopback": _LOOPBACK}))
    assert standalone_app.read_loopback_url(url_file) == _LOOPBACK
    url_file.write_text(
        json.dumps({"local": "http://localhost:9999", "tunnel": "https://t.example"})
    )
    assert standalone_app.read_loopback_url(url_file) == "http://127.0.0.1:9999"


@pytest.mark.parametrize(
    "content", ["", "not json", "[1, 2]", '{"tunnel": "https://t.example"}', '{"loopback": 5}']
)
def test_unusable_url_file_yields_no_url(tmp_path: Path, content: str) -> None:
    """Missing, malformed, non-object or URL-less files all mean "no daemon URL"."""
    url_file = tmp_path / "remote-url.json"
    assert standalone_app.read_loopback_url(url_file) is None
    url_file.write_text(content)
    assert standalone_app.read_loopback_url(url_file) is None


def test_wait_picks_up_a_url_file_written_late(tmp_path: Path) -> None:
    """The daemon's file appearing after a second is found; a never-written one times out.

    Only the file is waited for: the daemon answering at the URL is not
    (the window opens on the start page meanwhile).
    """
    url = f"http://127.0.0.1:{find_free_port()}/"
    url_file = tmp_path / "remote-url.json"

    def write_late() -> None:
        time.sleep(1.0)
        url_file.write_text(json.dumps({"loopback": url}))

    threading.Thread(target=write_late).start()
    started = time.monotonic()
    assert not standalone_app.daemon_answers(url)
    assert standalone_app.wait_for_url(url_file, None, 10) == url
    assert 1.0 <= time.monotonic() - started < 8
    started = time.monotonic()
    assert standalone_app.wait_for_url(tmp_path / "never.json", None, 0.6) is None
    assert 0.6 <= time.monotonic() - started < 4
    # --url skips the file.
    assert standalone_app.wait_for_url(tmp_path / "never.json", url, 0) == url


def test_daemon_answers_with_any_http_status(tmp_path: Path) -> None:
    """Any HTTP answer counts, a 404 from a daemon without the app-window route too."""
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    assert not standalone_app.daemon_answers(url)
    httpd = _site(port)[0]
    try:
        assert standalone_app.daemon_answers(url)
    finally:
        httpd.shutdown()

    class _NotFound(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert standalone_app.daemon_answers(f"http://127.0.0.1:{httpd.server_port}/")
    finally:
        httpd.shutdown()


def test_start_page_url_names_the_app_and_the_daemon_probe() -> None:
    """The start page is the packaged file, told what to probe and where to go next."""
    url = standalone_app.start_page_url("http://127.0.0.1:8787/x", "http://127.0.0.1:8787")
    page, _sep, query = url.partition("?")
    assert page == standalone_app.APP_START_PAGE.as_uri()
    assert standalone_app.APP_START_PAGE.is_file()
    assert urllib.parse.parse_qs(query) == {
        "next": ["http://127.0.0.1:8787/x"],
        "probe": ["http://127.0.0.1:8787/api/app-window"],
        "title": [BRAND["product_name"]],
    }
    branded = urllib.parse.parse_qs(
        standalone_app.start_page_url("http://h/", "http://h/", SEAMLESS_LOOP, popup=True)
        .partition("?")[2]
    )
    assert (branded["title"], branded["popup"]) == ([SEAMLESS_LOOP["product_name"]], ["1"])


def test_print_command_does_not_wait_for_the_daemon(tmp_path: Path) -> None:
    """``--print-command`` shows the command for the URL (given, or the file's) at once."""
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "browser.log")
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    url = f"http://127.0.0.1:{find_free_port()}"
    url_file = Path(env["KISS_HOME"]) / "remote-url.json"
    url_file.parent.mkdir(parents=True)
    url_file.write_text(json.dumps({"loopback": url}))
    started = time.monotonic()
    proc = _run(["--print-command"], env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[:2] == [str(stub), _app_flag(url)]
    proc = _run(["--print-command", "--url", _LOOPBACK], env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[:2] == [str(stub), _app_flag(_LOOPBACK)]
    assert time.monotonic() - started < 4
    assert not (tmp_path / "browser.log").exists()


# --- choosing the browser ---------------------------------------------------


def test_browser_override_wins_and_must_exist(tmp_path: Path) -> None:
    """``KISS_APP_BROWSER`` names the engine; an unresolvable one means no browser at all."""
    stub = _write_stub_browser(tmp_path / "bin" / "mybrowser", tmp_path / "log")
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    proc = _run(["--print-command", "--url", _LOOPBACK], env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [
        str(stub),
        _app_flag(_LOOPBACK),
        f"--user-data-dir={tmp_path / 'kiss-home' / 'webapp-profile'}",
        f"--profile-directory={standalone_app.brand_slug()}",
        f"--class={standalone_app.brand_slug()}",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--use-mock-keychain",
        *_PLATFORM_FLAGS,
    ]
    proc = _run(
        ["--print-command", "--url", _LOOPBACK],
        _env(tmp_path, KISS_APP_BROWSER="no-such-browser-xyz"),
    )
    assert proc.returncode == 1
    assert "No Chromium-family browser found" in proc.stderr
    assert _LOOPBACK in proc.stderr


def test_without_override_a_real_browser_is_found() -> None:
    """A system Chromium-family browser, else the Playwright Chromium, is used."""
    bundled = standalone_app.playwright_chromium()
    found = standalone_app.find_browser()
    assert found is not None and found.is_file() and os.access(found, os.X_OK)
    # An unknown platform has no system candidates: only the bundled browser remains.
    assert standalone_app.find_browser(system="Plan9") == bundled
    assert bundled is None or (bundled.is_file() and os.access(bundled, os.X_OK))


def test_the_bundled_chromium_runs_unsandboxed_on_linux(tmp_path: Path) -> None:
    """On Linux the Playwright Chromium gets ``--no-sandbox``; a system browser never does.

    The bundled browser has no setuid sandbox helper, so a distribution
    that restricts unprivileged user namespaces refuses to start it
    sandboxed ("No usable sandbox!"); other platforms keep the sandbox.
    """
    bundled = standalone_app.playwright_chromium()
    if bundled is None:
        pytest.skip("needs the Playwright Chromium")
    on_linux = platform.system() == "Linux"
    proc = _run(["--print-command", "--url", _LOOPBACK], _browser_env(tmp_path, bundled))
    assert proc.returncode == 0, proc.stderr
    assert ("--no-sandbox" in proc.stdout.split()) is on_linux, proc.stdout
    # A browser outside Playwright's browsers directory is a system browser.
    stub = _write_stub_browser(tmp_path / "bin" / "chromium", tmp_path / "log")
    proc = _run(["--print-command", "--url", _LOOPBACK], _env(tmp_path, KISS_APP_BROWSER=str(stub)))
    assert proc.returncode == 0, proc.stderr
    assert "--no-sandbox" not in proc.stdout.split(), proc.stdout
    assert standalone_app.sandboxed(bundled, system="Linux") is False
    assert standalone_app.sandboxed(stub, system="Linux") is True
    assert standalone_app.sandboxed(bundled, system="Darwin") is True


def test_system_candidates_per_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS and Windows look in their stock install locations before the bundled browser."""
    monkeypatch.delenv("KISS_APP_BROWSER", raising=False)
    bundled = standalone_app.playwright_chromium()
    # macOS: the first browser installed under /Applications, else the bundled one.
    mac = next((Path(c) for c in standalone_app._MACOS_BROWSERS if Path(c).is_file()), bundled)
    assert standalone_app.find_browser(system="Darwin") == mac
    # Windows candidates are joined under the Program Files roots; a file at
    # that (literal, backslash-named) path is found like any executable.
    chrome = _write_stub_browser(
        tmp_path / r"Google\Chrome\Application\chrome.exe", tmp_path / "log"
    )
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert standalone_app.find_browser(system="Windows") == chrome
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty"))
    assert standalone_app.find_browser(system="Windows") == bundled


# --- opening the window -----------------------------------------------------


def test_opens_the_daemon_url_detached(tmp_path: Path) -> None:
    """Without ``--url`` the daemon's loopback URL is awaited, then the browser starts detached."""
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log, sleep=8)
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    url_file = Path(env["KISS_HOME"]) / "remote-url.json"
    httpd, url = _site()
    url = url.rstrip("/")

    def write_late() -> None:
        time.sleep(1.0)
        url_file.parent.mkdir(parents=True, exist_ok=True)
        url_file.write_text(json.dumps({"local": "http://localhost:8787", "loopback": url}))

    threading.Thread(target=write_late).start()
    started = time.monotonic()
    try:
        proc = _run(["--wait", "15"], env)
    finally:
        httpd.shutdown()
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, proc.stderr
    # kiss-app returned once the browser had survived its startup grace
    # period, while the (8 s) browser was still running: detached.
    assert standalone_app.STARTUP_GRACE_SECS < elapsed < 6, elapsed
    assert _stub_calls(log) == [
        _app_flag(url),
        f"--user-data-dir={env['KISS_HOME']}/webapp-profile",
        f"--profile-directory={standalone_app.brand_slug()}",
        f"--class={standalone_app.brand_slug()}",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--use-mock-keychain",
        *_PLATFORM_FLAGS,
    ]


def test_without_a_daemon_url_fails_with_a_message(tmp_path: Path) -> None:
    """No URL file within ``--wait`` seconds: exit 1, nothing launched."""
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log)
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    proc = _run(["--wait", "0"], env)
    assert proc.returncode == 1
    assert f"{BRAND['product_name']} is not running" in proc.stderr
    assert "remote-url.json" in proc.stderr
    assert not log.exists()


def test_foreground_runs_the_browser_in_place(tmp_path: Path) -> None:
    """``--foreground`` execs the browser: its exit status is kiss-app's."""
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log, exit_code=7)
    httpd, url = _site()
    try:
        proc = _run(
            ["--foreground", "--url", url, "--wait", "0"],
            _env(tmp_path, KISS_APP_BROWSER=str(stub)),
        )
    finally:
        httpd.shutdown()
    assert proc.returncode == 7
    assert _stub_calls(log)[0] == _app_flag(url)


def test_a_daemon_not_answering_yet_gets_the_start_page(tmp_path: Path) -> None:
    """The window opens at once on the packaged start page, set to move on to the app.

    After a reboot the app is relaunched before the daemon is spawned:
    the URL file is there, nothing answers at its URL yet.  The browser
    is started anyway, on ``media/appStart.html`` (a file:// page)
    told to probe the daemon and then show the app's page (on macOS
    the popup-opening page).
    """
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log, exit_code=0)
    url = f"http://127.0.0.1:{find_free_port()}"
    started = time.monotonic()
    proc = _run(["--url", url, "--wait", "0"], _env(tmp_path, KISS_APP_BROWSER=str(stub)))
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - started < standalone_app.STARTUP_GRACE_SECS + 2
    flag = _stub_calls(log)[0]
    page, _sep, query = flag.partition("?")
    assert page == f"--app={standalone_app.APP_START_PAGE.as_uri()}"
    assert urllib.parse.parse_qs(query) == {
        "next": [_app_flag(url).removeprefix("--app=")],
        "probe": [f"{url}/api/app-window"],
        "title": [BRAND["product_name"]],
        **({"popup": ["1"]} if _MACOS else {}),
    }
    # --print-command names the app itself: it does not ask the daemon.
    proc = _run(
        ["--print-command", "--url", url], _env(tmp_path, KISS_APP_BROWSER=str(stub)),
    )
    assert proc.stdout.split()[1] == _app_flag(url)


def test_the_start_page_waits_for_the_daemon_and_then_shows_the_app(tmp_path: Path) -> None:
    """In a real Chromium the start page probes until the daemon answers, then navigates.

    The page is a file:// origin asking an http://127.0.0.1 port: the
    probe must be allowed from there (a no-cors fetch) and a refused
    connection must keep the page waiting rather than fail it.
    """
    from playwright.sync_api import sync_playwright

    port = find_free_port()
    daemon = f"http://127.0.0.1:{port}"
    url = standalone_app.start_page_url(f"{daemon}/landing", daemon)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url)
            assert page.title() == BRAND["product_name"]
            assert page.text_content("#name") == BRAND["product_name"]
            page.wait_for_timeout(1500)
            assert page.url == url, "the page moved on without a daemon"
            assert page.text_content("#dots")
            httpd, _ = _site(port)
            try:
                page.wait_for_url(f"{daemon}/landing", timeout=5000)
            finally:
                httpd.shutdown()
            assert "/landing" in _Hits.paths
        finally:
            browser.close()


def test_browser_dying_at_once_is_a_failure_with_its_stderr(tmp_path: Path) -> None:
    """A browser that exits non-zero within the grace period means no window: exit 1."""
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(
        tmp_path / "bin" / "chrome", log, exit_code=3, stderr="Missing X server or $DISPLAY"
    )
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    proc = _run(["--url", _LOOPBACK, "--wait", "0"], env)
    assert proc.returncode == 1
    app_log = Path(env["KISS_HOME"]) / "kiss-app.log"
    assert f"The browser exited at once ({stub}); its stderr is in {app_log}:" in proc.stderr
    assert "    Missing X server or $DISPLAY" in proc.stderr
    assert app_log.read_text(encoding="utf-8") == "Missing X server or $DISPLAY\n"


def test_browser_handing_off_to_a_running_instance_is_a_success(tmp_path: Path) -> None:
    """A browser that exits 0 at once (the URL went to an already-open instance): exit 0."""
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log, exit_code=0)
    started = time.monotonic()
    proc = _run(["--url", _LOOPBACK, "--wait", "0"], _env(tmp_path, KISS_APP_BROWSER=str(stub)))
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - started < standalone_app.STARTUP_GRACE_SECS + 1


# --- a real browser window --------------------------------------------------


_MOVED_BOUNDS = (123, 77, 640, 480)
"""Where ``_Hits``'s page moves the macOS popup window to: left, top, width, height."""


class _Hits(http.server.BaseHTTPRequestHandler):
    """A one-page site that records every request path.

    Like the daemon it serves the packaged ``media/appWindow.html`` and
    ``media/appWindow.js`` (the macOS popup window, its bounds keeper
    and its open/closing reports), and its page loads that script, so
    the window kiss-app opens on macOS is the real popup.  The page
    also moves and resizes a popup once to ``_MOVED_BOUNDS`` (a popup
    may move itself; an app window may not), so a relaunch shows
    whether the bounds were kept, and closes itself once ``close`` is
    set (it asks ``/close`` every 200 ms).  Like the daemon, the site
    answers kiss-app's question whether the popup is open
    (``/api/app-window``) with ``app_window_open``, which the test sets:
    the daemon learns it from the popup's WebSocket connection, which
    this page does not make.
    """

    paths: list[str] = []
    app_window_open = False
    close = False

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.paths.append(self.path)
        path, _, _query = self.path.partition("?")
        media = Path(path).name
        ctype = "text/html"
        if path.startswith("/media/") and media in ("appWindow.html", "appWindow.js"):
            body = (standalone_app.MEDIA_DIR / media).read_bytes()
            ctype = "text/javascript" if media.endswith(".js") else ctype
        elif path == standalone_app.APP_WINDOW_API:
            body = json.dumps({"open": self.app_window_open}).encode()
            ctype = "application/json"
        elif path == "/close":
            body = b"1" if _Hits.close else b"0"
            ctype = "text/plain"
        else:
            body = (
                b"<title>Standalone window test</title><h1>standalone ok</h1>"
                b'<script src="/media/appWindow.js"></script><script>'
                b"if (window.name === 'kiss-app-window' && !localStorage.getItem('moved')) {"
                b"  localStorage.setItem('moved', '1');"
                + f"  window.moveTo({_MOVED_BOUNDS[0]}, {_MOVED_BOUNDS[1]});"
                f"  window.resizeTo({_MOVED_BOUNDS[2]}, {_MOVED_BOUNDS[3]});".encode()
                + b"}"
                b"setInterval(function () {"
                b"  fetch('/close').then(r => r.text())"
                b"    .then(t => { if (t === '1') window.close(); });"
                b"}, 200);</script>"
            )
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        pass


def test_real_browser_opens_the_page_in_app_mode(tmp_path: Path) -> None:
    """The chosen browser really starts in ``--app`` mode and loads the URL.

    Needs an X server (``Xvfb``) and a Chromium-family browser; the
    window's WM_CLASS is checked when ``xwininfo`` is available.
    """
    xvfb = shutil.which("Xvfb")
    browser = standalone_app.find_browser()
    if xvfb is None or browser is None:
        pytest.skip("needs Xvfb and a Chromium-family browser")
    display = f":{find_free_port() % 1000 + 100}"
    x_server = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1280x900x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _Hits.paths = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Hits)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/"
    # The child's HOME is a temp dir: name the browser found with the real one.
    env = _browser_env(tmp_path, browser, DISPLAY=display)
    profile = Path(env["KISS_HOME"]) / "webapp-profile"
    try:
        time.sleep(1.0)
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        for _ in range(300):
            if "/" in _Hits.paths:
                break
            time.sleep(0.1)
        assert "/" in _Hits.paths, _Hits.paths
        # Chromium honoured --profile-directory: the profile is the brand
        # slug, which its Wayland app id ends in, not "Default".
        for _ in range(300):
            if (profile / standalone_app.brand_slug()).is_dir():
                break
            time.sleep(0.1)
        assert (profile / standalone_app.brand_slug()).is_dir(), list(profile.iterdir())
        assert not (profile / "Default").exists()
        xwininfo = shutil.which("xwininfo")
        if xwininfo:
            # The window carries the page title as its name and, as its
            # WM_CLASS, the URL-derived instance Chromium gives app
            # windows plus the brand slug as its class (the --class
            # flag), which the launcher's file name matches; a cold
            # profile can take a while to map it.
            wanted = (
                "Standalone window test",
                f'("127.0.0.1" "{standalone_app.brand_slug()}")',
            )
            tree = ""
            for _ in range(600):
                tree = subprocess.run(
                    [xwininfo, "-root", "-tree", "-display", display],
                    capture_output=True,
                    text=True,
                ).stdout
                if all(w in tree for w in wanted):
                    break
                time.sleep(0.1)
            assert all(w in tree for w in wanted), tree
    finally:
        # (no leading dashes: pkill would read them as its own options)
        subprocess.run(["pkill", "-f", f"user-data-dir={profile}"], check=False)
        httpd.shutdown()
        x_server.terminate()
        x_server.wait(timeout=10)


@pytest.mark.skipif(platform.system() != "Darwin", reason="checks the macOS brand bundle")
def test_macos_real_browser_runs_from_the_brand_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On macOS the real browser starts from the brand bundle's clone and loads the page.

    The clone is built under the child's temp HOME.  Its helpers (GPU,
    renderer, network) run under Chromium's sandbox, which only lets
    them read the bundle they were started from: a clone made of
    symlinks dies at once with "GPU process isn't usable", so the page
    loading proves the copies and the re-signed executable work.

    The window is the popup ``media/appWindow.html`` opens.  A reopen
    (``open -a``, a second kiss-app) brings it to the front, also
    while the popup is between connections (the daemon restarted): a
    second kiss-app waits for it to connect again rather than opening
    a second popup, and a start handed to the running browser while
    the popup is connected opens none either.  Once the user closes
    the popup the browser lives on without a window: a second kiss-app,
    told so by the site (as the daemon would, from the popup's
    WebSocket connection), hands it the command line for a new popup
    instead of a reopen (which would open an empty browser window).
    The daemon quits the browser that has no window left: at once
    (:func:`standalone_app.quit_closed_app`, armed by the popup's page
    going away) or, when the connection merely dropped, after a few
    sweeps (:func:`standalone_app.sweep_windowless_app`), so the next
    start is a fresh one, with the popup's bounds kept.
    """
    browser = standalone_app.find_browser()
    if browser is None or standalone_app.macos_browser_bundle(browser) is None:
        pytest.skip("needs a Chromium-family browser in an application bundle")
    httpd, url = _site()
    env = _browser_env(tmp_path, browser)
    # The daemon-side check looks for the brand bundle under HOME when it
    # is armed: the child's.
    monkeypatch.setenv("HOME", env["HOME"])
    profile = Path(env["KISS_HOME"]) / "webapp-profile"
    bundle = tmp_path / "home" / "Applications" / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / browser.name

    def popup_connects_late() -> None:
        time.sleep(3.0)
        _Hits.app_window_open = True

    try:
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert standalone_app.macos_bundle_browser(bundle, browser) == exe_copy
        assert _wait_for(lambda: "/" in _Hits.paths), (
            _Hits.paths, (Path(env["KISS_HOME"]) / "kiss-app.log").read_text(),
        )
        # The window and its helpers run from the clone, not the browser's own bundle.
        assert standalone_app.macos_bundle_in_use(bundle)
        # kiss-app made sure the site answers before it started the
        # browser.  The --app page was the popup opener, which asked the
        # site whether the popup was open already, then closed itself:
        # one window, the popup, showing the page.
        assert _Hits.paths[0] == standalone_app.APP_WINDOW_API, _Hits.paths
        assert _Hits.paths[1].startswith("/media/appWindow.html?url="), _Hits.paths
        assert _Hits.paths[2] == standalone_app.APP_WINDOW_API, _Hits.paths
        _Hits.app_window_open = True
        pid = _macos_engine_pid(exe_copy, profile)
        assert standalone_app.macos_bundle_pid(bundle) == pid
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS], _Hits.paths
        # A click on the running app's icon (what `open -a` sends: a
        # reopen event) brings that window to the front and opens no
        # other, which it would for an app window (Chromium opens a new
        # browser window when it finds no window of its own to focus).
        subprocess.run(["open", "-a", str(bundle)], check=True)
        time.sleep(3)
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        # A second kiss-app while the window is open does the same.
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert "already open" in proc.stdout
        time.sleep(3)
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        assert _Hits.paths.count("/") == 1, _Hits.paths
        # The popup between connections (the daemon restarted; the
        # reconnect is due within seconds): a second kiss-app waits for
        # it rather than handing the browser a start for a second popup.
        _Hits.app_window_open = False
        threading.Thread(target=popup_connects_late).start()
        started = time.monotonic()
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert "already open" in proc.stdout
        assert 3 <= time.monotonic() - started < standalone_app.APP_WINDOW_SETTLE_SECS
        assert _Hits.app_window_open
        time.sleep(3)
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        assert _Hits.paths.count("/") == 1, _Hits.paths
        # A start handed to the running browser while the popup is
        # connected (two kiss-app at once): the --app page learns so
        # from the site and closes without opening another popup.
        command = standalone_app.app_command(
            exe_copy,
            standalone_app.app_window_url(url),
            profile,
            standalone_app.brand_slug(),
            sandbox=standalone_app.sandboxed(browser),
            popups=True,
        )
        pages = len(_Hits.paths)
        assert subprocess.run(command, env=env, timeout=60).returncode == 0
        assert _wait_for(
            lambda: any(p.startswith("/media/appWindow.html?url=") for p in _Hits.paths[pages:])
        ), _Hits.paths[pages:]
        time.sleep(3)
        assert _Hits.paths.count("/") == 1, _Hits.paths
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        # The daemon's check leaves a browser alone whose popup is
        # connected, and one that still has a window on screen after
        # the popup's page went away (a reload).
        assert standalone_app.app_window_event("open")
        assert not standalone_app.quit_closed_app(bundle)
        assert not standalone_app.sweep_windowless_app(bundle)
        assert standalone_app.app_window_event("closing")
        time.sleep(standalone_app.APP_WINDOW_QUIT_GRACE_SECS + 1)
        assert standalone_app.macos_bundle_in_use(bundle)
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        # The user closes the window: the browser lives on without one.
        # kiss-app, told by the site that the popup is gone (and stays
        # so for the settling time: a minimized popup between
        # connections looks the same from outside), hands its command
        # line to the running browser, which opens a new popup where
        # the old one was, instead of sending a reopen.
        _Hits.close = True
        assert _macos_windows(pid, []) == []
        assert standalone_app.macos_bundle_in_use(bundle)
        _Hits.close = False
        _Hits.app_window_open = False
        started = time.monotonic()
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert "already open" not in proc.stdout
        assert time.monotonic() - started >= standalone_app.APP_WINDOW_SETTLE_SECS
        assert _wait_for(lambda: _Hits.paths.count("/") == 2), _Hits.paths
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS]
        # Closed again, with the popup's connection merely dropped (the
        # daemon's keepalive timed out): no quit at once (the window
        # may be minimized), but the sweeps that find the browser
        # without popup and window several times in a row quit it.
        _Hits.close = True
        assert _macos_windows(pid, []) == []
        assert standalone_app.app_window_event("open")
        assert standalone_app.app_window_event("dropped")
        time.sleep(standalone_app.APP_WINDOW_QUIT_GRACE_SECS + 1)
        assert standalone_app.macos_bundle_in_use(bundle)
        sweeps = [
            standalone_app.sweep_windowless_app(bundle)
            for _ in range(standalone_app.APP_WINDOW_SWEEP_STRIKES)
        ]
        assert sweeps == [False] * (standalone_app.APP_WINDOW_SWEEP_STRIKES - 1) + [True]
        assert _wait_for(lambda: not standalone_app.macos_bundle_in_use(bundle))
        assert standalone_app.macos_bundle_pid(bundle) is None
        assert not standalone_app.quit_closed_app(bundle)
        assert not standalone_app.sweep_windowless_app(bundle)
        # The popup's bounds (moved once by the page) survive a relaunch:
        # the browser forgets a popup's bounds, appWindow.js keeps them.
        _Hits.close = False
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert _wait_for(lambda: _Hits.paths.count("/") == 3), _Hits.paths
        pid = _macos_engine_pid(exe_copy, profile)
        assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS], _Hits.paths
        # Closed with its page going away: the daemon's check quits the
        # browser without a window after the grace period.
        _Hits.close = True
        assert _macos_windows(pid, []) == []
        assert standalone_app.app_window_event("open")
        assert standalone_app.app_window_event("closing")
        assert _wait_for(lambda: not standalone_app.macos_bundle_in_use(bundle))
        assert standalone_app.macos_bundle_pid(bundle) is None
    finally:
        subprocess.run(["pkill", "-f", f"user-data-dir={profile}"], check=False)
        httpd.shutdown()


@pytest.mark.skipif(not _MACOS, reason="the brand bundle is macOS-only")
def test_macos_real_browser_started_before_the_daemon_shows_the_start_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a reboot the app is relaunched before the daemon: a window at once, the app later.

    kiss-app starts the browser on the packaged start page while
    nothing answers at the daemon's URL (the file survived the reboot):
    a window is on screen within seconds, titled after the app.  Once
    the site answers, the page moves on to the popup opener, which
    opens the popup with the app and closes itself: one window, the
    popup, as on a start with the daemon up.  A second kiss-app
    meanwhile brings the waiting window to the front (the site cannot
    be asked, so the window on screen decides) instead of opening
    another.
    """
    browser = standalone_app.find_browser()
    if browser is None or standalone_app.macos_browser_bundle(browser) is None:
        pytest.skip("needs a Chromium-family browser in an application bundle")
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    env = _browser_env(tmp_path, browser)
    monkeypatch.setenv("HOME", env["HOME"])
    profile = Path(env["KISS_HOME"]) / "webapp-profile"
    bundle = tmp_path / "home" / "Applications" / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / browser.name
    _Hits.paths = []
    _Hits.app_window_open = False
    _Hits.close = False
    try:
        started = time.monotonic()
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert time.monotonic() - started < 15, "kiss-app waited for the daemon"
        pid = _macos_engine_pid(exe_copy, profile)
        assert _wait_for(lambda: bool(standalone_app.macos_windows_on_screen(pid))), (
            (Path(env["KISS_HOME"]) / "kiss-app.log").read_text()
        )
        assert _Hits.paths == [], "nothing served the site yet"
        # The page moves itself into a popup right after its first
        # paint (a reopen counts popups, not app windows): let the move
        # settle (two windows for a moment on a slow first start of the
        # clone) before a click on the icon, which brings the waiting
        # window to the front and opens no second window.
        time.sleep(3)
        assert _wait_for(lambda: standalone_app.macos_windows_on_screen(pid) == 1)
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert "already open" in proc.stdout
        time.sleep(3)
        assert standalone_app.macos_windows_on_screen(pid) == 1
        httpd, _ = _site(port)
        try:
            assert _wait_for(lambda: "/" in _Hits.paths), _Hits.paths
            assert _Hits.paths[0] == standalone_app.APP_WINDOW_API, _Hits.paths
            assert _Hits.paths[1].startswith("/media/appWindow.html?url="), _Hits.paths
            assert _Hits.paths[2] == standalone_app.APP_WINDOW_API, _Hits.paths
            _Hits.app_window_open = True
            assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS], _Hits.paths
            assert _Hits.paths.count("/") == 1, _Hits.paths
            # The user closes the window and the daemon goes down (a
            # restart).  The browser lives on without a window (no
            # daemon sweeps it here), so the next kiss-app hands it the
            # start page; once the daemon is back the popup shows the
            # app where it was: the opener restores the bounds the
            # page saved, the page itself moves only once per profile.
            _Hits.close = True
            assert _macos_windows(pid, []) == []
            _Hits.close = False
            _Hits.app_window_open = False
        finally:
            httpd.shutdown()
            httpd.server_close()  # frees the port for the site's return below
        proc = _run(["--url", url], env)
        assert proc.returncode == 0, proc.stderr
        assert "already open" not in proc.stdout
        assert _wait_for(lambda: bool(standalone_app.macos_windows_on_screen(pid)))
        time.sleep(3)
        assert _wait_for(lambda: standalone_app.macos_windows_on_screen(pid) == 1)
        httpd, _ = _site(port)  # starts a fresh path log
        try:
            assert _wait_for(lambda: "/" in _Hits.paths), _Hits.paths
            assert _Hits.paths[0] == standalone_app.APP_WINDOW_API, _Hits.paths
            assert _Hits.paths[1].startswith("/media/appWindow.html?url="), _Hits.paths
            assert _macos_windows(pid, [_MOVED_BOUNDS]) == [_MOVED_BOUNDS], _Hits.paths
            assert _Hits.paths.count("/") == 1, _Hits.paths
        finally:
            httpd.shutdown()
    finally:
        subprocess.run(["pkill", "-f", f"user-data-dir={profile}"], check=False)


def _wait_for(condition: Callable[[], bool], timeout: float = 30) -> bool:
    """Whether *condition* became true within *timeout* seconds (asked every 0.1 s)."""
    deadline = time.monotonic() + timeout
    while True:
        if condition():
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)


def _macos_engine_pid(exe: Path, profile: Path) -> int:
    """The pid of the browser process *exe* running with *profile* as its user data dir."""
    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    for line in listing.stdout.splitlines():
        pid, _, command = line.strip().partition(" ")
        if command.startswith(f"{exe} ") and f"--user-data-dir={profile} " in command:
            return int(pid)
    raise AssertionError(f"no {exe} with --user-data-dir={profile} in\n{listing.stdout}")


_WINDOWS_JXA = """
ObjC.import('CoreGraphics');
function run(argv) {
  var ref = $.CGWindowListCopyWindowInfo(
    $.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements,
    $.kCGNullWindowID);
  var list = ObjC.deepUnwrap(ObjC.castRefToObject(ref));
  var out = [];
  for (var i = 0; i < list.length; i++) {
    var w = list[i];
    if (w.kCGWindowOwnerPID === parseInt(argv[0], 10) && w.kCGWindowLayer === 0) {
      var b = w.kCGWindowBounds;
      out.push([b.X, b.Y, b.Width, b.Height]);
    }
  }
  return JSON.stringify(out);
}
"""


def _macos_windows(pid: int, expected: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """The on-screen windows of process *pid* as (left, top, width, height), front to back.

    Waits up to 30 s for them to be *expected*, then returns whatever is
    there.  The list comes from the window server (JXA over
    CoreGraphics), which needs no accessibility permission; a minimized
    window is off screen and not listed.
    """
    windows: list[tuple[int, ...]] = []
    for _ in range(300):
        out = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", _WINDOWS_JXA, str(pid)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        windows = [tuple(int(v) for v in w) for w in json.loads(out)]
        if windows == expected:
            break
        time.sleep(0.1)
    return windows


# --- desktop launchers ------------------------------------------------------


def _desktop_fields(entry: Path) -> dict[str, str]:
    lines = entry.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[Desktop Entry]"
    return dict(line.split("=", 1) for line in lines[1:])


def test_linux_launcher_carries_the_brand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``.desktop`` entry named after the product, with its icon, exec and window class."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    entry = standalone_app.install_launcher(
        BRAND, icon, exe, home=tmp_path, system="Linux", state_dir=state
    )
    assert (
        entry
        == tmp_path / ".local" / "share" / "applications" / f"{standalone_app.brand_slug()}.desktop"
    )
    assert entry is not None and os.access(entry, os.X_OK)
    fields = _desktop_fields(entry)
    assert fields["Name"] == BRAND["product_name"]
    assert fields["Comment"] == BRAND["tagline"]
    assert fields["Exec"] == f'env KISS_HOME="{state}" "{exe}"'
    assert fields["Icon"] == str(icon)
    # The Wayland app id of the window (X11 matches the file name).
    assert fields["StartupWMClass"] == "chrome-127.0.0.1__-kiss-sorcar"
    assert fields["Type"] == "Application" and fields["Terminal"] == "false"

    branded = standalone_app.install_launcher(
        SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Linux"
    )
    assert branded is not None and branded.name == "seamless-loop.desktop"
    fields = _desktop_fields(branded)
    assert fields["Name"] == "Seamless Loop"
    assert fields["Comment"] == SEAMLESS_LOOP["tagline"]
    assert fields["StartupWMClass"] == "chrome-127.0.0.1__-seamless-loop"

    # An absolute XDG_DATA_HOME relocates the entry; uninstall removes exactly it.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    moved = standalone_app.install_launcher(BRAND, icon, exe, home=tmp_path, system="Linux")
    assert moved == tmp_path / "xdg" / "applications" / entry.name
    assert moved is not None
    assert standalone_app.uninstall_launcher(BRAND, home=tmp_path, system="Linux") == moved
    assert not moved.exists() and entry.exists()
    assert standalone_app.uninstall_launcher(BRAND, home=tmp_path, system="Linux") is None
    # A relative XDG_DATA_HOME is ignored, as the XDG spec says.
    monkeypatch.setenv("XDG_DATA_HOME", "relative/data")
    assert standalone_app.install_launcher(BRAND, icon, exe, home=tmp_path, system="Linux") == entry
    assert standalone_app.uninstall_launcher(BRAND, home=tmp_path, system="Linux") == entry


def _key_file_string(raw: str) -> str:
    """Decode a desktop entry string value: ``\\s`` ``\\n`` ``\\t`` ``\\r`` ``\\\\`` sequences."""
    out, i = "", 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            out += {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}[raw[i + 1]]
            i += 2
        else:
            out += raw[i]
            i += 1
    return out


def _parse_desktop_exec(raw: str) -> list[str]:
    """Split a raw ``Exec`` line as the desktop entry spec prescribes.

    The key-file layer decodes its escape sequences first; then arguments
    are separated by spaces, a double-quoted argument may hold spaces
    with backslash escaping the next character, and after unquoting
    ``%%`` is a literal percent sign (other ``%x`` are field codes).
    """
    args: list[str] = []
    current, quoted, escaped = "", False, False
    for ch in _key_file_string(raw) + " ":
        if escaped:
            current, escaped = current + ch, False
        elif quoted and ch == "\\":
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif ch == " " and not quoted:
            args.append(current.replace("%%", "%"))
            current = ""
        else:
            current += ch
    return [a for a in args if a]


def test_launcher_paths_are_escaped_for_their_format(tmp_path: Path) -> None:
    """Desktop-entry field codes and quoting, and shell metacharacters, cannot break the launchers.

    The desktop entry's ``Exec`` is parsed the way the spec and GLib do
    (key-file escapes, then quoted arguments with backslash escapes, then
    ``%%`` -> ``%``), and, where GLib's own launcher is importable by the
    system ``python3`` (``python3-gi``), launched by it for real; the
    macOS launcher is run by ``sh``.  Both run a kiss-app stub in the
    odd path, which records the ``KISS_HOME`` and arguments it got.
    """
    odd = tmp_path / 'we%ird $dir "q" `b` \\s %f'
    exe, state = odd / "kiss-app", odd / "state"
    log = tmp_path / "log"
    _write_stub_browser(exe, log)
    exe.write_text(
        exe.read_text(encoding="utf-8").replace('"$@"', '"$KISS_HOME" "$@"'), encoding="utf-8"
    )
    entry = standalone_app.install_launcher(
        BRAND, tmp_path / "i.png", exe, home=tmp_path, system="Linux", state_dir=state
    )
    assert entry is not None
    raw_exec = _desktop_fields(entry)["Exec"]
    assert _parse_desktop_exec(raw_exec) == ["env", f"KISS_HOME={state}", str(exe)]
    assert "%%f" in raw_exec and '\\\\"' in raw_exec and "\\\\$" in raw_exec and "\\\\`" in raw_exec
    system_python = shutil.which("python3", path="/usr/bin:/bin")
    gio_launch = (
        "import sys, gi; gi.require_version('Gio', '2.0'); from gi.repository import Gio; "
        "sys.exit(0 if Gio.DesktopAppInfo.new_from_filename(sys.argv[1]).launch([], None) else 2)"
    )
    if (
        system_python
        and subprocess.run([system_python, "-c", "import gi"], capture_output=True).returncode == 0
    ):
        proc = subprocess.run(
            [system_python, "-c", gio_launch, str(entry)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert _stub_calls(log) == [str(state)]
        log.unlink()

    script = tmp_path / "launcher"
    script.write_text(standalone_app.macos_launcher_script(exe, state), encoding="utf-8")
    script.chmod(0o755)
    proc = subprocess.run(["sh", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert _stub_calls(log) == [str(state), "--foreground"]


def test_macos_launcher_is_an_app_bundle(tmp_path: Path) -> None:
    """``~/Applications/<Product Name>.app`` with Info.plist and a launcher that execs kiss-app.

    Run on any platform: the icon conversion needs macOS' ``sips`` and
    ``iconutil``, so the bundle has an icon only where they exist.
    """
    icon, exe = tmp_path / "icon.png", tmp_path / "kiss-app"
    icon.write_bytes(BRAND_ICON_PNG)
    state = tmp_path / "state dir"
    # A bare browser binary: the plain bundle, whatever browsers the host has.
    bare = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "log")
    bundle = standalone_app.install_launcher(
        SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Darwin", state_dir=state, browser=bare
    )
    assert bundle == tmp_path / "Applications" / "Seamless Loop.app"
    assert bundle is not None
    with (bundle / "Contents" / "Info.plist").open("rb") as f:
        info = plistlib.load(f)
    assert info["CFBundleName"] == "Seamless Loop"
    assert info["CFBundleDisplayName"] == "Seamless Loop"
    assert info["CFBundleIdentifier"] == "org.kiss-sorcar.seamless-loop"
    assert info["CFBundleExecutable"] == "launcher"
    launcher = bundle / "Contents" / "MacOS" / "launcher"
    assert os.access(launcher, os.X_OK)
    assert launcher.read_text(encoding="utf-8") == (
        f"#!/bin/sh\nKISS_HOME='{state}'\nexport KISS_HOME\nexec {exe} --foreground\n"
    )
    icns = bundle / "Contents" / "Resources" / "AppIcon.icns"
    has_tools = shutil.which("sips") and shutil.which("iconutil")
    assert icns.is_file() == bool(has_tools)
    assert ("CFBundleIconFile" in info) == bool(has_tools)
    assert not (bundle / "Contents" / "Resources" / "AppIcon.iconset").exists()

    # Re-installing replaces the plain bundle; uninstall removes it.
    (bundle / "stale").write_text("x")
    assert (
        standalone_app.install_launcher(
            SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Darwin", browser=bare
        )
        == bundle
    )
    assert not (bundle / "stale").exists()
    assert (
        standalone_app.uninstall_launcher(SEAMLESS_LOOP, home=tmp_path, system="Darwin") == bundle
    )
    assert not bundle.exists()
    assert standalone_app.uninstall_launcher(SEAMLESS_LOOP, home=tmp_path, system="Darwin") is None


def test_other_platforms_have_no_launcher(tmp_path: Path) -> None:
    """Windows (and anything else) gets no launcher file."""
    assert (
        standalone_app.install_launcher(
            BRAND, tmp_path / "i.png", tmp_path / "x", home=tmp_path, system="Windows"
        )
        is None
    )
    assert standalone_app.uninstall_launcher(BRAND, home=tmp_path, system="Windows") is None
    assert list(tmp_path.iterdir()) == []


def _expected_launcher(tmp_path: Path, product_name: str, slug: str) -> Path:
    """Where the CLI puts the launcher on this platform (see ``_env``: HOME, XDG_DATA_HOME)."""
    if platform.system() == "Darwin":
        return tmp_path / "home" / "Applications" / f"{product_name}.app"
    return tmp_path / "xdg" / "applications" / f"{slug}.desktop"


def test_launcher_command_line(tmp_path: Path) -> None:
    """``--install-launcher`` / ``--uninstall-launcher`` report what they did."""
    env = _env(tmp_path, XDG_DATA_HOME=str(tmp_path / "xdg"))
    proc = _run(["--install-launcher"], env)
    assert proc.returncode == 0, proc.stderr
    entry = _expected_launcher(tmp_path, BRAND["product_name"], standalone_app.brand_slug())
    assert proc.stdout.strip() == f"Installed {entry}"
    if entry.suffix == ".desktop":
        fields = _desktop_fields(entry)
        assert fields["Exec"] == f'env KISS_HOME="{env["KISS_HOME"]}" "{_KISS_APP.resolve()}"'
        assert fields["Icon"] == str(standalone_app.ICON_FILE)
        assert Path(fields["Icon"]).is_file()
    else:
        script = (entry / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")
        assert f"KISS_HOME={env['KISS_HOME']}\n" in script
    proc = _run(["--uninstall-launcher"], env)
    assert proc.stdout.strip() == f"Removed {entry}"
    assert not entry.exists()
    proc = _run(["--uninstall-launcher"], env)
    assert proc.stdout.strip() == "No launcher to remove."


def _stub_file_managers(tmp_path: Path) -> tuple[Path, Path]:
    """Stub ``xdg-open`` and ``open`` in a bin directory; return it and their shared log."""
    log = tmp_path / "opener.log"
    bin_dir = tmp_path / "bin"
    _write_stub_browser(bin_dir / "xdg-open", log)
    _write_stub_browser(bin_dir / "open", log)
    return bin_dir, log


def test_reveal_launcher_shows_its_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher's folder goes to ``xdg-open`` (Linux) or Finder selects the bundle (macOS)."""
    bin_dir, log = _stub_file_managers(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    home = tmp_path / "home"
    # Nothing installed yet: nothing to show, nothing opened.
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Linux") is None
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Darwin") is None
    assert standalone_app.launcher_path(BRAND, home=home, system="Windows") is None
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Windows") is None
    assert not log.exists()

    entry = standalone_app.install_launcher(
        BRAND, tmp_path / "i.png", tmp_path / "kiss-app", home=home, system="Linux"
    )
    assert entry is not None
    assert entry == standalone_app.launcher_path(BRAND, home=home, system="Linux")
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Linux") == entry
    assert _stub_calls(log) == [str(entry.parent)]

    log.unlink()
    bundle = home / "Applications" / f"{BRAND['product_name']}.app"
    bundle.mkdir(parents=True)
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Darwin") == bundle
    assert _stub_calls(log) == ["-R", str(bundle)]

    # An opener that fails at once (no desktop to open a window on) is
    # no reveal either.
    log.unlink()
    _write_stub_browser(tmp_path / "broken" / "xdg-open", log, exit_code=4)
    monkeypatch.setenv("PATH", str(tmp_path / "broken"))
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Linux") is None
    assert _stub_calls(log) == [str(entry.parent)]

    # Linux without xdg-open: nothing to open it with.
    log.unlink()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert standalone_app.reveal_launcher(BRAND, home=home, system="Linux") is None
    assert not log.exists()


def test_reveal_launcher_command_line(tmp_path: Path) -> None:
    """``--reveal-launcher`` fails before the launcher exists and shows it after."""
    bin_dir, log = _stub_file_managers(tmp_path)
    env = _env(
        tmp_path, XDG_DATA_HOME=str(tmp_path / "xdg"), PATH=f"{bin_dir}:{os.environ['PATH']}"
    )
    proc = _run(["--reveal-launcher"], env)
    assert proc.returncode == 1
    assert proc.stderr.strip() == "No launcher to show: none installed, or no file manager opened."
    assert not log.exists()

    assert _run(["--install-launcher"], env).returncode == 0
    entry = _expected_launcher(tmp_path, BRAND["product_name"], standalone_app.brand_slug())
    proc = _run(["--reveal-launcher"], env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"Showing {entry.name} in {entry.parent}"
    if entry.suffix == ".desktop":
        assert _stub_calls(log) == [str(entry.parent)]
    else:
        assert _stub_calls(log) == ["-R", str(entry)]


# --- a branded install ------------------------------------------------------


@pytest.fixture(scope="module")
def branded_src(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ``src`` directory of a Seamless Loop-branded copy of the package."""
    return make_branded_package(tmp_path_factory.mktemp("branded"))


def _run_branded(
    branded_src: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    env = dict(env, PYTHONPATH=str(branded_src))
    return subprocess.run(
        [sys.executable, "-m", "kiss.server.standalone_app", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_branded_app_uses_the_brand_home_and_window_class(
    branded_src: Path, tmp_path: Path
) -> None:
    """A Seamless Loop install keeps its profile under ``~/.s10s`` and uses its own class."""
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "log")
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub))
    del env["KISS_HOME"]  # the brand's own default home decides
    proc = _run_branded(branded_src, ["--print-command", "--url", _LOOPBACK], env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [
        str(stub),
        _app_flag(_LOOPBACK, "Seamless Loop"),
        f"--user-data-dir={tmp_path / 'home' / '.s10s' / 'webapp-profile'}",
        "--profile-directory=seamless-loop",
        "--class=seamless-loop",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--use-mock-keychain",
        *_PLATFORM_FLAGS,
    ]
    proc = _run_branded(branded_src, ["--wait", "0"], env)
    assert proc.returncode == 1
    assert "Seamless Loop is not running" in proc.stderr
    assert str(tmp_path / "home" / ".s10s" / "remote-url.json") in proc.stderr
    assert "Seamless Loop webapp" in _run_branded(branded_src, ["--help"], env).stdout


def test_branded_launcher_names_and_icon(branded_src: Path, tmp_path: Path) -> None:
    """The launcher of a branded install is Seamless Loop's, for its home, with its icon."""
    env = _env(tmp_path, XDG_DATA_HOME=str(tmp_path / "xdg"))
    del env["KISS_HOME"]  # the brand's own default home decides
    proc = _run_branded(branded_src, ["--install-launcher"], env)
    assert proc.returncode == 0, proc.stderr
    entry = _expected_launcher(tmp_path, "Seamless Loop", "seamless-loop")
    assert proc.stdout.strip() == f"Installed {entry}"
    brand_home = tmp_path / "home" / ".s10s"
    brand_icon = branded_src / "kiss" / "webapp" / "media" / "kiss-icon.png"
    if entry.suffix == ".desktop":
        fields = _desktop_fields(entry)
        assert fields["Name"] == "Seamless Loop"
        assert fields["Comment"] == SEAMLESS_LOOP["tagline"]
        # The brand's own profile name keeps its windows apart from stock's.
        assert fields["StartupWMClass"] == "chrome-127.0.0.1__-seamless-loop"
        assert fields["Exec"].startswith(f'env KISS_HOME="{brand_home}" ')
        assert Path(fields["Icon"]) == brand_icon
    else:
        script = (entry / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")
        assert f"KISS_HOME={brand_home}\n" in script
    assert brand_icon.read_bytes() == BRAND_ICON_PNG
    assert brand_icon.read_bytes() != standalone_app.ICON_FILE.read_bytes()
    # The stock launcher is untouched by the brand's.
    assert not _expected_launcher(
        tmp_path, BRAND["product_name"], standalone_app.brand_slug()
    ).exists()
    assert (
        _run_branded(branded_src, ["--uninstall-launcher"], env).stdout.strip()
        == f"Removed {entry}"
    )


# --- the command line, in process -------------------------------------------
#
# The same paths the subprocess tests take, run in this interpreter so
# coverage sees them.  Not reachable here without test doubles, hence
# untested: a broken Playwright package (``playwright_chromium`` returning
# None from its except branch), the ``.icns`` conversion (needs macOS'
# ``sips`` and ``iconutil``) and the ``--foreground`` exec, which would
# replace this interpreter (the subprocess test above covers it).


@pytest.mark.skipif(platform.system() != "Linux", reason="checks the Linux .desktop launcher")
def test_main_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` end to end: print, no daemon, launch, launcher install/remove, no browser."""
    log, udd_log = tmp_path / "browser.log", tmp_path / "udd.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log)
    _write_stub_browser(tmp_path / "bin" / "update-desktop-database", udd_log)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss-home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("KISS_APP_BROWSER", str(stub))

    assert standalone_app.main(["--print-command", "--url", _LOOPBACK]) == 0
    assert capsys.readouterr().out.split()[:2] == [str(stub), f"--app={_LOOPBACK}"]

    assert standalone_app.main(["--wait", "0"]) == 1
    assert f"{BRAND['product_name']} is not running" in capsys.readouterr().err

    url_file = tmp_path / "kiss-home" / "remote-url.json"
    url_file.parent.mkdir()
    httpd, url = _site()
    url_file.write_text(json.dumps({"loopback": url}))
    try:
        assert standalone_app.main([]) == 0
    finally:
        httpd.shutdown()
    assert _stub_calls(log)[:2] == [
        f"--app={url}",
        f"--user-data-dir={tmp_path / 'kiss-home' / 'webapp-profile'}",
    ]

    assert standalone_app.main(["--install-launcher"]) == 0
    entry = tmp_path / "xdg" / "applications" / f"{standalone_app.brand_slug()}.desktop"
    assert capsys.readouterr().out.strip() == f"Installed {entry}"
    assert _stub_calls(udd_log) == [str(entry.parent)]
    assert standalone_app.main(["--uninstall-launcher"]) == 0
    assert capsys.readouterr().out.strip() == f"Removed {entry}"
    assert standalone_app.main(["--uninstall-launcher"]) == 0
    assert capsys.readouterr().out.strip() == "No launcher to remove."

    monkeypatch.setenv("KISS_APP_BROWSER", "no-such-browser-xyz")
    assert standalone_app.find_browser() is None
    assert standalone_app.main(["--print-command", "--url", _LOOPBACK]) == 1
    assert "No Chromium-family browser found" in capsys.readouterr().err


# --- The window's icon: Wayland app id and the macOS clone bundle ------------


@pytest.mark.parametrize(
    ("browser", "url", "expected"),
    [
        ("/usr/bin/google-chrome", "http://127.0.0.1:8787", "chrome-127.0.0.1__-kiss-sorcar"),
        ("/usr/bin/chromium", "http://127.0.0.1/", "chrome-127.0.0.1__-kiss-sorcar"),
        ("/opt/pw/chrome-linux/chrome", "http://localhost:1/", "chrome-localhost__-kiss-sorcar"),
        ("/usr/bin/brave-browser", "http://127.0.0.1:8787/", "brave-127.0.0.1__-kiss-sorcar"),
        ("/usr/bin/microsoft-edge", "http://127.0.0.1:8787/", "msedge-127.0.0.1__-kiss-sorcar"),
        ("/usr/bin/chrome", "http://127.0.0.1:1/a/b?x=1", "chrome-127.0.0.1__a_b-kiss-sorcar"),
    ],
)
def test_wayland_app_id_follows_chromium(browser: str, url: str, expected: str) -> None:
    """``<executable>-<host>_<path>-<profile>`` with path separators as ``_``."""
    assert standalone_app.wayland_app_id(Path(browser), "kiss-sorcar", url) == expected
    # The port is not part of the id; the profile (brand slug) is.
    assert standalone_app.wayland_app_id(Path(browser), "kiss-sorcar") == (
        standalone_app.wayland_app_id(Path(browser), "kiss-sorcar", "http://127.0.0.1:65000/")
    )
    assert standalone_app.wayland_app_id(Path(browser), "seamless-loop", url) == expected.replace(
        "-kiss-sorcar", "-seamless-loop"
    )


def test_linux_launcher_names_the_window_of_the_browser_it_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``StartupWMClass`` is the Wayland id for the browser the launcher will run."""
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    brave = _write_stub_browser(tmp_path / "bin" / "brave-browser", tmp_path / "log")
    monkeypatch.setenv("KISS_APP_BROWSER", str(brave))
    entry = standalone_app.install_launcher(
        BRAND, tmp_path / "icon.png", tmp_path / "kiss-app", home=tmp_path, system="Linux"
    )
    assert entry is not None
    assert _desktop_fields(entry)["StartupWMClass"] == "brave-127.0.0.1__-kiss-sorcar"
    # The entry's file name is the X11 class kiss-app passes.
    assert entry.stem == standalone_app.brand_slug()
    # Without any browser the entry still assumes Chrome's prefix.
    monkeypatch.setenv("KISS_APP_BROWSER", "no-such-browser-xyz")
    entry = standalone_app.install_launcher(
        BRAND, tmp_path / "icon.png", tmp_path / "kiss-app", home=tmp_path, system="Linux"
    )
    assert entry is not None
    assert _desktop_fields(entry)["StartupWMClass"] == "chrome-127.0.0.1__-kiss-sorcar"


_BROWSER_PLIST: dict[str, Any] = {
    "CFBundleName": "Chromium",
    "CFBundleDisplayName": "Chromium",
    "CFBundleIdentifier": "org.chromium.Chromium",
    "CFBundleExecutable": "Chromium",
    "CFBundleIconFile": "app.icns",
    "CFBundleIconName": "AppIcon",
    "CFBundleShortVersionString": "140.0.7339.16",
    "CFBundleVersion": "7339.16",
    "CFBundlePackageType": "APPL",
    "LSMinimumSystemVersion": "11.0",
    "NSHighResolutionCapable": True,
    "NSSupportsAutomaticGraphicsSwitching": True,
    "NSMicrophoneUsageDescription": "Chromium needs the microphone for sites that ask.",
    "NSCameraUsageDescription": "Chromium needs the camera for sites that ask.",
    "CFBundleURLTypes": [{"CFBundleURLSchemes": ["http", "https"]}],
    "CFBundleDocumentTypes": [{"CFBundleTypeExtensions": ["html"]}],
    "UTExportedTypeDeclarations": [{"UTTypeIdentifier": "org.chromium.extension"}],
    "NSUserActivityTypes": ["NSUserActivityTypeBrowsingWeb"],
    "NSServices": [{"NSMenuItem": {"default": "Search"}}],
    "ASWebAuthenticationSessionWebBrowserSupportCapabilities": {"IsSupported": True},
    "NSAppleScriptEnabled": True,
    "OSAScriptingDefinition": "scripting.sdef",
    "LSHasLocalizedDisplayName": "1",
    "LSEnvironment": {"MallocNanoZone": "0"},
    "NSPrincipalClass": "BrowserCrApplication",
    "KSProductID": "org.chromium.Chromium",
    "KSUpdateURL": "https://example.invalid/update",
}


def _fake_browser_app(root: Path, name: str = "Chromium", version: str = "140.0.7339.16") -> Path:
    """A ``.app`` bundle shaped like Chromium's; returns its main executable.

    The framework directory holds a file (``Resources/chrome.pak``) and
    a relative symlink (``Versions/Current``), as Chromium's does, so a
    clone of it can be checked for both.
    """
    contents = root / f"{name}.app" / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    framework = contents / "Frameworks" / f"{name} Framework.framework"
    (framework / "Versions" / version / "Resources").mkdir(parents=True)
    (framework / "Versions" / version / "Resources" / "chrome.pak").write_bytes(b"pak")
    (framework / "Versions" / "Current").symlink_to(version)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump({**_BROWSER_PLIST, "CFBundleShortVersionString": version}, f)
    return _write_stub_browser(contents / "MacOS" / name, root / "browser.log")


def _bundle_plist(bundle: Path) -> dict[str, Any]:
    with (bundle / "Contents" / "Info.plist").open("rb") as f:
        return dict(plistlib.load(f))


def _signed_identifier(exe: Path) -> str | None:
    """The code-signing identifier of *exe* on macOS; None elsewhere (no ``codesign``)."""
    if platform.system() != "Darwin":
        return None
    out = subprocess.run(["codesign", "-dv", str(exe)], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if line.startswith("Identifier="):
            return line.removeprefix("Identifier=")
    return None


def test_macos_bundle_is_a_clone_of_the_browser(tmp_path: Path) -> None:
    """The brand bundle runs the browser's binary under the brand's identity.

    Its executable and Frameworks are real copies of the browser's
    (symlinks would put the framework outside the bundle path that
    Chromium's sandbox lets its helpers read), the copied binary is
    re-signed as the brand on macOS, and its Info.plist is the
    browser's minus the handler registrations and updater keys, renamed
    after the brand.
    """
    browser = _fake_browser_app(tmp_path / "apps")
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    icon.write_bytes(BRAND_ICON_PNG)
    bundle = standalone_app.install_launcher(
        SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Darwin", state_dir=state, browser=browser
    )
    assert bundle == tmp_path / "Applications" / "Seamless Loop.app"
    assert bundle is not None
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    frameworks = bundle / "Contents" / "Frameworks"
    assert exe_copy.is_file() and not exe_copy.is_symlink()
    assert os.access(exe_copy, os.X_OK)
    assert exe_copy.read_bytes() == browser.read_bytes()
    assert frameworks.is_dir() and not frameworks.is_symlink()
    framework = frameworks / "Chromium Framework.framework"
    pak = framework / "Versions" / "140.0.7339.16" / "Resources" / "chrome.pak"
    assert pak.read_bytes() == b"pak" and not pak.is_symlink()
    current = framework / "Versions" / "Current"
    assert current.is_symlink() and os.readlink(current) == "140.0.7339.16"
    if platform.system() == "Darwin":
        assert _signed_identifier(exe_copy) == "org.kiss-sorcar.seamless-loop"
    # The launcher script is still the bundle's executable.
    assert os.access(bundle / "Contents" / "MacOS" / "launcher", os.X_OK)

    info = _bundle_plist(bundle)
    assert info["CFBundleName"] == info["CFBundleDisplayName"] == "Seamless Loop"
    assert info["CFBundleIdentifier"] == "org.kiss-sorcar.seamless-loop"
    assert info["CFBundleExecutable"] == "launcher"
    assert info["LSUIElement"] is False
    # Carried over from the browser: what Chromium needs at run time.
    for key in (
        "CFBundleShortVersionString",
        "CFBundleVersion",
        "LSMinimumSystemVersion",
        "NSHighResolutionCapable",
        "NSSupportsAutomaticGraphicsSwitching",
        "NSMicrophoneUsageDescription",
        "NSCameraUsageDescription",
        "LSEnvironment",
        "NSPrincipalClass",
    ):
        assert info[key] == _BROWSER_PLIST[key], key
    # Dropped: handler registrations, the asset-catalog icon, Keystone.
    for key in standalone_app._MACOS_PLIST_DROPPED_KEYS + ("KSProductID", "KSUpdateURL"):
        assert key not in info, key
    has_tools = shutil.which("sips") and shutil.which("iconutil")
    assert ("CFBundleIconFile" in info) == bool(has_tools)
    if has_tools:
        assert info["CFBundleIconFile"] == "AppIcon"

    # The clone is current for that browser version, and only for it:
    # not for another version of it, another browser, or no bundle.
    assert standalone_app.macos_bundle_browser(bundle, browser) == exe_copy
    newer = _fake_browser_app(tmp_path / "newer", version="141.0.7400.1")
    assert standalone_app.macos_bundle_browser(bundle, newer) is None
    edge = _fake_browser_app(tmp_path / "edge", name="Microsoft Edge")
    assert standalone_app.macos_bundle_browser(bundle, edge) is None
    assert standalone_app.macos_bundle_browser(tmp_path / "none.app", browser) is None

    # Reinstalling for the same browser keeps the clone (a reinstall
    # must not pull the helpers out from under a running window) and
    # rewrites the launcher; a newer browser replaces it.
    marker = frameworks / "marker"
    marker.write_text("kept")
    launcher = bundle / "Contents" / "MacOS" / "launcher"
    launcher.write_text("stale")
    assert standalone_app.install_launcher(
        SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Darwin", state_dir=state, browser=browser
    )
    assert marker.read_text() == "kept"
    assert "--foreground" in launcher.read_text()
    assert standalone_app.install_launcher(
        SEAMLESS_LOOP, icon, exe, home=tmp_path, system="Darwin", state_dir=state, browser=newer
    )
    assert not marker.exists()
    assert exe_copy.read_bytes() == newer.read_bytes()
    assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "141.0.7400.1"
    assert standalone_app.macos_bundle_browser(bundle, newer) == exe_copy
    assert standalone_app.macos_bundle_browser(bundle, browser) is None


def test_macos_stale_clone_in_use_is_left_alone(tmp_path: Path) -> None:
    """A stale clone a window still runs from is kept until a later launch.

    Replacing it would delete the executables the running browser
    starts its helpers from; the launch falls back to the browser
    itself, which hands the URL to the running instance.
    """
    browser = _fake_browser_app(tmp_path / "apps")
    icon, exe = tmp_path / "icon.png", tmp_path / "kiss-app"
    icon.write_bytes(BRAND_ICON_PNG)
    bundle = standalone_app.install_launcher(
        BRAND, icon, exe, home=tmp_path, system="Darwin", browser=browser
    )
    assert bundle is not None
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    assert not standalone_app.macos_bundle_in_use(bundle)
    # A "window": the clone's own executable, kept running by its sleep.
    _write_stub_browser(exe_copy, tmp_path / "log", sleep=30)
    window = subprocess.Popen([str(exe_copy), "--app=http://127.0.0.1:1/"])
    try:
        # Popen returns once the child has exec'd, but the kernel publishes
        # the new command line a moment later; `ps` can still miss it.
        assert _wait_for(lambda: standalone_app.macos_bundle_in_use(bundle) is True)
        newer = _fake_browser_app(tmp_path / "newer", version="141.0.7400.1")
        launcher_before = (bundle / "Contents" / "MacOS" / "launcher").read_text()
        assert standalone_app.macos_launch_browser(newer, home=tmp_path) == newer
        assert (bundle / "Contents" / "MacOS" / "launcher").read_text() == launcher_before
        assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "140.0.7339.16"
        assert exe_copy.exists()
        # Nor is it replaced by the plain bundle: a browser whose plist
        # cannot be read, or a bare binary, waits as well.
        unreadable = _fake_browser_app(tmp_path / "unreadable")
        (tmp_path / "unreadable" / "Chromium.app" / "Contents" / "Info.plist").unlink()
        bare = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "log")
        for browser in (unreadable, bare):
            assert (
                standalone_app.write_macos_launcher(
                    tmp_path / "Applications", BRAND, icon, exe, tmp_path / "state", browser
                )
                is None
            )
            assert exe_copy.exists() and (bundle / "Contents" / "Frameworks").is_dir()
    finally:
        window.kill()
        window.wait()
    assert not standalone_app.macos_bundle_in_use(bundle)
    assert standalone_app.macos_launch_browser(newer, home=tmp_path) == exe_copy
    assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "141.0.7400.1"


def test_macos_app_page_opens_the_webapp_as_a_popup_of_its_origin() -> None:
    """The macOS ``--app`` page is ``media/appWindow.html`` on the webapp's origin.

    Its query names the webapp URL and the brand (the page's title);
    the command line lets that page open its popup without a click.
    """
    title = urllib.parse.quote_plus(BRAND["product_name"])
    assert standalone_app.app_window_url(_LOOPBACK) == (
        f"{_LOOPBACK}/media/appWindow.html?url=http%3A%2F%2F127.0.0.1%3A8787&title={title}"
    )
    assert standalone_app.app_window_url("https://t.trycloudflare.com/", SEAMLESS_LOOP) == (
        "https://t.trycloudflare.com/media/appWindow.html"
        "?url=https%3A%2F%2Ft.trycloudflare.com%2F&title=Seamless+Loop"
    )
    command = standalone_app.app_command(Path("chrome"), _LOOPBACK, Path("p"), "s", popups=True)
    assert command[-1] == "--disable-popup-blocking"
    assert "--disable-popup-blocking" not in standalone_app.app_command(
        Path("chrome"), _LOOPBACK, Path("p"), "s"
    )


@pytest.mark.skipif(not _MACOS, reason="`open -a` is macOS' Launch Services")
def test_macos_activate_running_needs_a_window_open_can_reach(tmp_path: Path) -> None:
    """Nothing running from the bundle, or a bundle ``open`` cannot launch: False.

    (The real-browser test above covers the window being brought to
    the front.)
    """
    bundle = tmp_path / "Applications" / f"{BRAND['product_name']}.app"
    assert not standalone_app.macos_activate_running(bundle)
    # A "window" running from what is not an application bundle (no
    # Info.plist, no launcher): ``open -a`` refuses it.
    exe = bundle / "Contents" / "MacOS" / "Chromium"
    _write_stub_browser(exe, tmp_path / "log", sleep=30)
    window = subprocess.Popen([str(exe), "--app=http://127.0.0.1:1/"])
    try:
        assert _wait_for(lambda: standalone_app.macos_bundle_in_use(bundle) is True)
        assert not standalone_app.macos_activate_running(bundle)
    finally:
        window.kill()
        window.wait()


def test_macos_clone_failure_leaves_the_launcher_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A browser bundle without Frameworks cannot be cloned: the plain bundle, no leftovers."""
    browser = _fake_browser_app(tmp_path / "apps")
    shutil.rmtree(tmp_path / "apps" / "Chromium.app" / "Contents" / "Frameworks")
    icon, exe = tmp_path / "icon.png", tmp_path / "kiss-app"
    icon.write_bytes(BRAND_ICON_PNG)
    bundle = standalone_app.install_launcher(
        BRAND, icon, exe, home=tmp_path, system="Darwin", browser=browser
    )
    assert bundle is not None
    assert [p.name for p in (bundle / "Contents" / "MacOS").iterdir()] == ["launcher"]
    assert not (bundle / "Contents" / "Frameworks").exists()
    assert "NSMicrophoneUsageDescription" not in _bundle_plist(bundle)
    err = capsys.readouterr().err
    assert "Could not clone" in err and "cp -R" in err
    assert standalone_app.macos_launch_browser(browser, home=tmp_path) == browser


def test_macos_bundle_without_a_browser_bundle_is_the_launcher_alone(tmp_path: Path) -> None:
    """A bare binary (or no browser) gives the plain wrapper bundle of before."""
    bare = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "log")
    icon, exe = tmp_path / "icon.png", tmp_path / "kiss-app"
    icon.write_bytes(BRAND_ICON_PNG)
    has_tools = bool(shutil.which("sips") and shutil.which("iconutil"))
    bundle = standalone_app.macos_bundle_path(tmp_path)
    for browser in (bare, None):
        # (write_macos_launcher directly: install_launcher would look
        # the host's browsers up in place of None)
        assert standalone_app.write_macos_launcher(
            tmp_path / "Applications", BRAND, icon, exe, tmp_path / "state", browser
        ) is None, "a plain bundle has no clone"
        assert not (bundle / "Contents" / "Frameworks").exists()
        assert [p.name for p in (bundle / "Contents" / "MacOS").iterdir()] == ["launcher"]
        info = _bundle_plist(bundle)
        assert info["CFBundleName"] == BRAND["product_name"]
        assert "NSMicrophoneUsageDescription" not in info
        # The plain bundle still names its icon (Finder, the applications
        # menu) wherever the icon could be converted.
        assert info.get("CFBundleIconFile") == ("AppIcon" if has_tools else None)
        assert (bundle / "Contents" / "Resources" / "AppIcon.icns").is_file() == has_tools
        assert standalone_app.macos_bundle_browser(bundle, bare) is None
    # A bundle whose Info.plist cannot be read is treated like a bare binary.
    broken = _fake_browser_app(tmp_path / "broken")
    (tmp_path / "broken" / "Chromium.app" / "Contents" / "Info.plist").write_bytes(b"not a plist")
    assert standalone_app.write_macos_launcher(
        tmp_path / "Applications", BRAND, icon, exe, tmp_path / "state", broken
    ) is None
    assert not (bundle / "Contents" / "Frameworks").exists()
    assert standalone_app.macos_browser_bundle(broken) is not None
    assert standalone_app.macos_browser_bundle(bare) is None


def test_macos_launch_goes_through_the_bundle_and_repairs_a_stale_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``kiss-app`` starts the browser by the bundle's copy, rebuilding it when stale."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss-home"))
    browser = _fake_browser_app(tmp_path / "apps")
    monkeypatch.setenv("KISS_APP_BROWSER", str(browser))
    bundle = tmp_path / "Applications" / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    # No bundle yet: launching builds it and uses the copy.
    assert standalone_app.macos_launch_browser(browser, home=tmp_path) == exe_copy
    assert exe_copy.is_file() and _bundle_plist(bundle)["CFBundleName"] == BRAND["product_name"]
    # The same browser version at another path (Playwright re-downloaded
    # the same revision) keeps the clone; a new revision rebuilds it.
    moved = _fake_browser_app(tmp_path / "apps-moved")
    shutil.rmtree(tmp_path / "apps")
    assert standalone_app.macos_launch_browser(moved, home=tmp_path) == exe_copy
    assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "140.0.7339.16"
    newer = _fake_browser_app(tmp_path / "apps2", version="141.0.7400.1")
    assert standalone_app.macos_launch_browser(newer, home=tmp_path) == exe_copy
    assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "141.0.7400.1"
    # A clone whose plist was cut short (an interrupted write) is stale
    # too, and rebuilt rather than tripping over.
    (bundle / "Contents" / "Info.plist").write_bytes(b'<?xml version="1.0"?><plist><dict>')
    assert standalone_app.macos_bundle_browser(bundle, newer) is None
    assert standalone_app.macos_launch_browser(newer, home=tmp_path) == exe_copy
    assert _bundle_plist(bundle)["CFBundleShortVersionString"] == "141.0.7400.1"
    # A bare binary is launched directly.
    bare = _write_stub_browser(tmp_path / "bin" / "chrome", tmp_path / "log")
    assert standalone_app.macos_launch_browser(bare, home=tmp_path) == bare
    # The launched command really runs the browser through the copy
    # (the stub behind it logs its arguments).
    log = tmp_path / "apps2" / "browser.log"
    command = standalone_app.app_command(exe_copy, _LOOPBACK, tmp_path / "profile", "kiss-sorcar")
    assert standalone_app.launch(command, tmp_path / "launch.log", foreground=False)
    assert _stub_calls(log)[0] == f"--app={_LOOPBACK}"


@posix_only("directory permissions")
def test_macos_launch_falls_back_to_the_browser_when_the_bundle_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unwritable ``~/Applications`` costs the icon, not the window."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss-home"))
    browser = _fake_browser_app(tmp_path / "apps")
    apps = tmp_path / "Applications"
    apps.mkdir()
    apps.chmod(0o555)
    try:
        assert standalone_app.macos_launch_browser(browser, home=tmp_path) == browser
    finally:
        apps.chmod(0o755)
    assert not any(apps.iterdir())
    err = capsys.readouterr().err
    assert "Could not write" in err and "Permission denied" in err
