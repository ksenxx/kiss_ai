# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the browser tool looks and behaves like a person's Chrome.

Bot-protection vendors blocked 95 hosts in the task history.  These tests
drive a REAL Chromium (no mocks) against local HTTP servers and check the
properties :mod:`kiss.agents.sorcar.web_stealth` provides:

1. Engine: Patchright drives the browser; the bundled Chromium (or Chrome)
   channel is chosen; on Linux the "no window" default is a headed
   Chromium on a process-wide Xvfb display, and the display is shared,
   restartable and torn down cleanly.
2. Fingerprint: no ``HeadlessChrome`` token, ``navigator.webdriver`` off,
   real window geometry, no emulated timezone / scale factor.
3. Human input: a click arrives after a curved pointer path and a held
   button; typing has an uneven cadence; scrolling wheels in uneven
   notches.  Elements without a box or with an unhittable point still
   get clicked (plain-click fallbacks).
4. Challenge pages: a Cloudflare-style ``cf-mitigated: challenge``
   interstitial that clears is waited out and the real page returned; one
   that stays is reported with a ``Note:``; a Google "unusual traffic"
   page is answered by re-running the query on Bing.

Unreachable without doubles (documented, not mocked): the ``chrome``
channel branch of ``chrome_channel`` (needs Google Chrome installed), the
``playwright`` fallback import (needs Patchright uninstalled), the
non-Linux branch of ``virtual_display``, an Xvfb that starts but never
reports its display (15 s timeout), and ``_glide_onto`` returning
``None`` for an element without a rendered box — ``scroll_into_view_if_needed``
already waits for visibility, so that guard only fires when the element
is removed between that call and ``bounding_box()``.
"""

from __future__ import annotations

import atexit
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import WebUseTool

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("Xvfb") is None,
    reason="virtual display tests need Linux with Xvfb",
)
# ``resource`` (rlimits) does not exist on Windows; the module is Linux-only
# anyway, so skip at import instead of failing collection there.
resource = pytest.importorskip("resource")

_EVENTS_PAGE = """<!doctype html><html><head><title>events</title></head>
<body style="margin:0">
<div style="height:1400px"></div>
<button id="target" style="width:180px;height:50px">Press me</button>
<input id="box" aria-label="Name">
<a id="wrapped" href="#" style="display:inline-block;width:200px">wrapped link</a>
<pre id="log"></pre>
<script>
window.ev = {moves: [], down: 0, up: 0, keys: [], wheels: [], clicked: 0};
document.addEventListener('mousemove', e => ev.moves.push([e.clientX, e.clientY]));
document.addEventListener('mousedown', e => ev.down = performance.now());
document.addEventListener('mouseup', e => ev.up = performance.now());
document.addEventListener('wheel', e => ev.wheels.push(e.deltaY));
document.getElementById('target').addEventListener('click', () => {
  ev.clicked++; document.getElementById('target').textContent = 'Pressed';
});
document.getElementById('box').addEventListener('keydown', e => ev.keys.push(performance.now()));
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Serve the probe pages.

    ``/events`` records input events in ``window.ev``; ``/challenge``
    answers the first *N* requests per path with a Cloudflare-style
    interstitial (``cf-mitigated: challenge`` header, "Just a moment..."
    title, a script reloading after 1 s) and content afterwards;
    ``/blocked`` never clears; ``/sorry/index`` is Google's "unusual
    traffic" page.
    """

    challenge_hits: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer the request according to its path."""
        path = urlparse(self.path).path
        headers = {}
        if path == "/events":
            body = _EVENTS_PAGE
        elif path.startswith("/challenge"):
            n = self.challenge_hits.get(path, 0) + 1
            self.challenge_hits[path] = n
            if n <= 2:
                headers["cf-mitigated"] = "challenge"
                body = (
                    "<title>Just a moment...</title><body>Performing security "
                    "verification<script>setTimeout(()=>location.reload(),1000)</script></body>"
                )
            else:
                body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/blocked":
            headers["cf-mitigated"] = "challenge"
            body = "<title>Just a moment...</title><body>Verifying you are human.</body>"
        elif path == "/turnstile":
            # Cloudflare's interactive managed challenge: the widget iframe
            # (served by the test's route on challenges.cloudflare.com)
            # tells the page when its box is ticked; the page then loads
            # the content, as the real challenge reloads it.
            headers["cf-mitigated"] = "challenge"
            body = (
                "<title>Just a moment...</title><body><h2>Performing security "
                "verification</h2><iframe src='https://challenges.cloudflare.com/"
                "cdn-cgi/challenge-platform/turnstile/widget' style='width:300px;"
                "height:65px;border:0'></iframe><script>addEventListener('message',"
                " e => { if (e.data === 'ticked') location.href = '/turnstile-done'; })"
                "</script></body>"
            )
        elif path == "/turnstile-done":
            body = "<title>Real content</title><body><h1>Welcome</h1></body>"
        elif path == "/sorry/index":
            body = (
                "<title>https://www.google.com/search</title><body>Our systems have "
                "detected unusual traffic from your computer network. IP address: 10.0.0.1"
                "</body>"
            )
        else:
            body = "<title>inert</title><body>a page</body>"
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr request log."""


_TURNSTILE_WIDGET = (
    "<body style='margin:0;padding:20px;font:14px sans-serif'>"
    "<input type=checkbox id=cb aria-label='Verify you are human'"
    " style='width:24px;height:24px;vertical-align:middle'"
    " onclick=\"parent.postMessage('ticked', '*')\"> Verify you are human</body>"
)


def _serve_turnstile_widget(route) -> None:
    """Answer the challenges.cloudflare.com iframe request with a checkbox widget."""
    route.fulfill(status=200, content_type="text/html", body=_TURNSTILE_WIDGET)


@pytest.fixture(scope="module")
def server():
    """Yield the base URL of a local server serving the probe pages."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    """Yield one default-configured WebUseTool shared by the module's tests."""
    base = tmp_path_factory.mktemp("stealth")
    web = WebUseTool(user_data_dir=str(base / "profile"), work_dir=str(base))
    try:
        yield web
    finally:
        web.close()


def _main(tool: WebUseTool, expression: str) -> object:
    """Evaluate in the page's main world.

    Patchright evaluates in an isolated world by default (so page scripts
    cannot see the automation); the recorder object ``window.ev`` and
    the masked ``navigator.userAgent`` live in the main world, where site
    scripts run.
    """
    return tool._page.evaluate(expression, isolated_context=False)


def _ev(tool: WebUseTool) -> dict:
    ev = _main(tool, "window.ev")
    assert isinstance(ev, dict)
    return ev


# ---------------------------------------------------------------------------
# 1. Engine and virtual display
# ---------------------------------------------------------------------------


def test_patchright_drives_the_browser_headed_on_a_virtual_display(tool, server):
    """Default tool: Patchright API, headed Chromium on the shared Xvfb display."""
    assert web_stealth.playwright_api().__name__ == "patchright.sync_api"
    assert web_stealth.playwright_package() == "patchright"
    assert web_stealth.chrome_channel() in ("chrome", "chromium")

    tool.go_to_url(f"{server}/inert")
    display = web_stealth.virtual_display()
    assert display is not None and display.startswith(":")
    assert web_stealth.virtual_display() == display, "one Xvfb per process"
    assert tool._headless is True, "still 'no window' from the caller's view"
    assert tool._chromium_headless is False, "but Chromium itself runs headed"
    assert tool._context_args() == {"no_viewport": True}
    # The browser really is on that display: its process environment says so.
    environ = open(f"/proc/{tool._browser_pid}/environ", "rb").read().split(b"\0")
    assert f"DISPLAY={display}".encode() in environ


def test_fingerprint_has_no_automation_tells(tool, server):
    """UA, webdriver flag, window geometry and emulation look like a desktop Chrome."""
    tool.go_to_url(f"{server}/inert")
    probe = tool._page.evaluate(
        "() => ({ua: navigator.userAgent, wd: navigator.webdriver,"
        " inner: [innerWidth, innerHeight], outer: [outerWidth, outerHeight],"
        " dpr: devicePixelRatio,"
        " tz: Intl.DateTimeFormat().resolvedOptions().timeZone})"
    )
    assert "HeadlessChrome" not in probe["ua"] and "Chrome/" in probe["ua"]
    assert probe["wd"] is False
    # A headed window has browser chrome: the outer height exceeds the
    # inner one, unlike headless where the two coincide.
    assert probe["outer"][1] > probe["inner"][1]
    assert probe["outer"] == list(tool.viewport)
    assert tuple(probe["inner"]) == tool._viewport_size()
    assert probe["dpr"] == 1  # the machine's real scale, no Retina emulation
    # No timezone override: whatever the machine reports is what a
    # person's browser here would report too.
    assert probe["tz"]
    if time.tzname[0] == "UTC":
        assert probe["tz"] in ("UTC", "Etc/UTC")


def test_virtual_display_restarts_after_stop_and_falls_back_without_xvfb(tool, server):
    """Stopping the display kills Xvfb; without the binary the tool goes headless."""
    assert web_stealth.virtual_display() is not None
    proc = web_stealth._XVFB_PROC
    assert proc is not None and proc.poll() is None
    xvfb_pid = int(subprocess.run(
        ["pgrep", "-P", str(proc.pid), "-x", "Xvfb"], capture_output=True, text=True
    ).stdout.split()[0])
    web_stealth.stop_virtual_display()
    assert proc.poll() is not None, "wrapper shell exited"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and os.path.exists(f"/proc/{xvfb_pid}"):
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{xvfb_pid}"), "Xvfb itself was killed"
    assert web_stealth._XVFB_PROC is None
    web_stealth.stop_virtual_display()  # idempotent when nothing runs
    # Restart cycles neither leak descriptors nor pile up atexit callbacks.
    fds_before = len(os.listdir("/proc/self/fd"))
    callbacks_before = atexit._ncallbacks()
    for _ in range(2):
        assert web_stealth.virtual_display() is not None
        web_stealth.stop_virtual_display()
    assert len(os.listdir("/proc/self/fd")) == fds_before
    assert atexit._ncallbacks() == callbacks_before
    # Descriptor exhaustion: the pipe still fits but Popen's own pipe does
    # not; the tool falls back to headless and closes both pipe ends.
    highest = max(int(fd) for fd in os.listdir("/proc/self/fd"))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (highest + 3, hard))
    try:
        assert web_stealth.virtual_display() is None
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert len(os.listdir("/proc/self/fd")) == fds_before

    # No Xvfb binary on PATH -> no display -> the launch falls back to
    # real headless mode (and masks the HeadlessChrome UA token).  Only
    # one sync Playwright driver may live per thread, so the shared tool
    # is closed first (it relaunches on its next call).
    tool.close()
    saved_path = os.environ["PATH"]
    os.environ["PATH"] = str(os.devnull)
    try:
        assert web_stealth.virtual_display() is None
        headless_tool = WebUseTool(user_data_dir=None)
        try:
            headless_tool.go_to_url(f"{server}/inert")
            assert headless_tool._chromium_headless is True
            assert headless_tool._context_args() == {
                "viewport": {"width": 1280, "height": 900}
            }
            assert headless_tool._viewport_size() == (1280, 900)
            # Site scripts (main world) and the request header see a
            # headed Chrome; only Patchright's isolated world keeps the
            # raw token, and no page can reach that world.
            main_ua = _main(headless_tool, "navigator.userAgent")
            assert isinstance(main_ua, str) and "HeadlessChrome" not in main_ua
            assert "HeadlessChrome" in headless_tool._page.evaluate("navigator.userAgent")
        finally:
            headless_tool.close()
    finally:
        os.environ["PATH"] = saved_path

    # A fresh display comes up on demand and the shared tool relaunches on it.
    fresh = web_stealth.virtual_display()
    assert fresh is not None
    tool.go_to_url(f"{server}/inert")
    assert tool._chromium_headless is False


def test_broken_xvfb_binary_means_no_virtual_display(tool, tmp_path):
    """An Xvfb that exits at once yields no display (and no leaked wrapper)."""
    tool.close()
    web_stealth.stop_virtual_display()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "Xvfb"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    saved_path = os.environ["PATH"]
    os.environ["PATH"] = f"{fake_bin}:{saved_path}"
    try:
        assert web_stealth.virtual_display() is None
        assert web_stealth._XVFB_PROC is None
    finally:
        os.environ["PATH"] = saved_path
    assert web_stealth.virtual_display() is not None


# ---------------------------------------------------------------------------
# 2. Human-like input
# ---------------------------------------------------------------------------


def test_click_travels_a_curved_path_and_holds_the_button(tool, server):
    """The pointer glides in, pauses, presses for a while, and the click lands."""
    tree = tool.go_to_url(f"{server}/events")
    button_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'button "Press me"' in line
    )
    tool._page.mouse.move(5, 5)  # far from the (scrolled-away) button
    tool._mouse_xy = (5.0, 5.0)
    _main(tool, "ev.moves = []")
    result = tool.click(button_id)
    assert 'button "Pressed"' in result
    ev = _ev(tool)
    assert ev["clicked"] == 1
    assert len(ev["moves"]) >= 6, "a glide, not a teleport"
    # Not a straight line: the intermediate points bow away from the chord.
    (x0, y0), (x1, y1) = ev["moves"][0], ev["moves"][-1]
    chord = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    max_dev = max(
        abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / chord for x, y in ev["moves"][1:-1]
    )
    assert max_dev > 2.0
    assert 40 <= ev["up"] - ev["down"] <= 400, "button held like a finger press"
    assert tool._mouse_xy is not None


def test_hover_glides_without_pressing(tool, server):
    """action="hover" moves onto the element and presses nothing."""
    tree = tool.go_to_url(f"{server}/events")
    button_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'button "Press me"' in line
    )
    tool.click(button_id, action="hover")
    ev = _ev(tool)
    assert ev["clicked"] == 0 and ev["down"] == 0
    assert len(ev["moves"]) >= 6


def test_typing_has_uneven_cadence_and_correct_text(tool, server):
    """Keystroke intervals vary and pause at spaces; the value is exact."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    text = "hello human world"
    tool.type_text(box_id, text)
    assert tool._page.evaluate("document.getElementById('box').value") == text
    keys = _ev(tool)["keys"]
    # Control+a and Backspace precede the text.
    typed = keys[-len(text):]
    gaps = [b - a for a, b in zip(typed, typed[1:])]
    assert statistics.pstdev(gaps) > 5, "not a metronome"
    # Playwright waits *after* each key, so the inter-word pause is the
    # gap following the space: 90-260 ms versus 35-115 ms per letter.
    after_space = [gaps[i] for i, ch in enumerate(text[:-1]) if ch == " "]
    after_letter = [gaps[i] for i, ch in enumerate(text[:-1]) if ch != " "]
    assert min(after_space) >= 80
    assert max(after_letter) <= 150


def test_type_text_with_enter_submits_after_a_pause(tool, server):
    """press_enter waits a moment, then submits."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    _main(
        tool,
        "document.getElementById('box').addEventListener('keydown',"
        " e => { if (e.key === 'Enter') ev.enter = performance.now(); })",
    )
    tool.type_text(box_id, "go", press_enter=True)
    ev = _ev(tool)
    # keys[-1] is the Enter keydown itself; keys[-2] is the last letter.
    assert abs(ev["enter"] - ev["keys"][-1]) < 5
    assert ev["enter"] - ev["keys"][-2] >= 150


def test_scroll_wheels_in_uneven_notches(tool, server):
    """Wheel deltas vary around the nominal step and the page really scrolls."""
    tool.go_to_url(f"{server}/events")
    tool.scroll("down", 4)
    ev = _ev(tool)
    assert len(ev["wheels"]) == 4
    assert all(240 <= d <= 360 for d in ev["wheels"])
    assert len(set(ev["wheels"])) > 1
    assert tool._page.evaluate("scrollY") > 0
    tool.scroll("up", 2)
    assert _ev(tool)["wheels"][-1] < 0


def test_click_falls_back_when_the_glided_point_is_covered(tool, server):
    """A point that receives no events within 5 s yields to the plain click.

    An overlay covers the button for 6 s: the positioned click times out
    (5 s), the plain click's own actionability wait then sees the overlay
    go away and lands.
    """
    tool.go_to_url(
        "data:text/html,<button id=b style='width:300px;height:60px'"
        " onclick=\"this.textContent='done'\">covered</button>"
        "<div id=o style='position:absolute;left:0;top:0;width:400px;height:100px'></div>"
        "<script>setTimeout(() => document.getElementById('o').remove(), 6000)</script>"
    )
    locator = tool._page.get_by_role("button", name="covered")
    start = time.monotonic()
    tool._human_click(locator)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "done"
    assert 5 <= time.monotonic() - start < 30
    # Same for a hover: cover the button again, the positioned hover
    # times out and the plain hover lands once the overlay is gone.
    tool._page.evaluate(
        "() => { const o = document.createElement('div'); o.id = 'o2';"
        " o.style.cssText = 'position:absolute;left:0;top:0;width:400px;height:100px';"
        " document.body.append(o); setTimeout(() => o.remove(), 6000);"
        " document.getElementById('b').textContent = 'covered';"
        " document.getElementById('b').onmouseover = () =>"
        " document.getElementById('b').textContent = 'hovered'; }"
    )
    start = time.monotonic()
    tool._human_click(locator, hover_only=True)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "hovered"
    assert 5 <= time.monotonic() - start < 30


def test_glide_failure_falls_back_to_plain_click(tool, server):
    """A Playwright error while gliding still ends in a click.

    The button is hidden for 12 s: ``scroll_into_view_if_needed`` gives up
    after the 10 s page-read timeout (a Playwright error), and the plain
    click's 30 s wait then catches the button becoming visible.
    """
    tool.go_to_url(
        "data:text/html,<button id=b hidden onclick=\"this.textContent='done'\">go</button>"
        "<script>setTimeout(() => document.getElementById('b').hidden = false, 12000)</script>"
    )
    locator = tool._page.get_by_role("button", name="go", include_hidden=True)
    start = time.monotonic()
    tool._human_click(locator)
    assert tool._page.evaluate("document.getElementById('b').textContent") == "done"
    assert 10 <= time.monotonic() - start < 40


def test_mouse_path_shape_and_typing_chunks():
    """Pure helpers: Bezier waypoints end on target; chunks rebuild the text."""
    path = web_stealth.mouse_path((0.0, 0.0), (400.0, 100.0))
    assert path[-1] == (400.0, 100.0)
    assert 6 <= len(path) <= 24
    assert web_stealth.mouse_path((10.0, 10.0), (10.4, 10.2)) == [(10.4, 10.2)]
    chunks = web_stealth.typing_chunks("a  b c")
    assert "".join(c for c, _ in chunks) == "a  b c"
    assert all(90 <= d <= 260 for c, d in chunks if c == " ")
    assert all(35 <= d <= 115 for c, d in chunks if c != " ")
    assert web_stealth.typing_chunks("") == []
    assert web_stealth.typing_duration_secs([("ab", 100), (" ", 200)]) == 0.4


# ---------------------------------------------------------------------------
# 3. Challenge pages
# ---------------------------------------------------------------------------


def test_cloudflare_style_challenge_is_waited_out(tool, server):
    """An interstitial that clears on its own yields the real page, no note."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/challenge/{time.time_ns()}")
    assert tree.startswith("Page: Real content")
    assert "Note:" not in tree
    assert time.monotonic() - start < 12


def test_persistent_block_is_reported_with_a_note(tool, server):
    """An interstitial that never clears is described plainly to the agent."""
    start = time.monotonic()
    tree = tool.go_to_url(f"{server}/blocked")
    elapsed = time.monotonic() - start
    first = tree.splitlines()[0]
    assert first.startswith("Note: 127.0.0.1:") and "(Cloudflare challenge)" in first
    assert "show_browser()" in first
    assert "Page: Just a moment..." in tree
    assert 12 <= elapsed < 40


def test_turnstile_checkbox_is_ticked_like_a_person(tool, server):
    """The interactive "Verify you are human" box is found in the widget frame and pressed.

    The widget origin is Cloudflare's, so the test answers that request
    itself with a checkbox page; everything else — frame discovery, the
    curved approach, the held press, the page's reaction — is real.
    """
    tool.go_to_url(f"{server}/inert")
    tool._context.route("https://challenges.cloudflare.com/**", _serve_turnstile_widget)
    try:
        start = time.monotonic()
        tree = tool.go_to_url(f"{server}/turnstile")
    finally:
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.startswith("Page: Real content"), tree[:300]
    assert "Note:" not in tree
    assert time.monotonic() - start < 2 * 12 + 10


def test_turnstile_is_not_ticked_when_the_widget_shows_no_box(tool, server):
    """Without a checkbox in the widget frame nothing is pressed and the block is reported."""
    tool.go_to_url(f"{server}/inert")

    def spinner(route) -> None:
        route.fulfill(status=200, content_type="text/html", body="<body>Verifying...</body>")

    tool._context.route("https://challenges.cloudflare.com/**", spinner)
    try:
        tree = tool.go_to_url(f"{server}/turnstile")
    finally:
        tool._context.unroute("https://challenges.cloudflare.com/**")
    assert tree.splitlines()[0].startswith("Note:") and "(Cloudflare challenge)" in tree
    assert tool._turnstile_checkbox() is None


def test_live_cloudflare_protected_site_loads(tool):
    """A real Cloudflare-protected publisher that challenged this network loads.

    journals.sagepub.com showed the Turnstile checkbox to every earlier
    configuration from this machine; with a fresh profile (no clearance
    cookie) the page must come back as content.  Skipped offline.
    """
    try:
        import socket

        socket.create_connection(("journals.sagepub.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("no Internet access")
    tool.close()  # fresh browser: no clearance cookie from earlier tests
    tree = tool.go_to_url("https://journals.sagepub.com/")
    assert tree.startswith("Page: Sage Journals"), tree[:200]


def test_challenge_vendor_recognises_each_vendor():
    """Titles/bodies seen in the task history map to their vendor labels."""
    cases = {
        ("Just a moment...", "Performing security verification"): "Cloudflare challenge",
        ("stackoverflow.com", "Verifying you are human. Ray ID: 9a"): "Cloudflare challenge",
        ("Attention Required! | Cloudflare", "Sorry, you have been blocked"): "Cloudflare block",
        ("medium.com", "Sorry, you have been blocked ... Cloudflare Ray ID"): "Cloudflare block",
        ("Access Denied", "Reference #18.x errors.edgesuite.net"): "Akamai block",
        ("", "You don't have permission to access ... Reference #18.2"): "Akamai block",
        ("Oh noes!", "Sad Anubis"): "Anubis proof-of-work check",
        ("Egyptian gods", "Anubis was the god of the dead"): None,
        ("Making sure you're not a bot!", "Loading..."): "Anubis proof-of-work check",
        ("dblp", "Oh noes! Sad Anubis: the check failed"): "Anubis proof-of-work check",
        ("", "Request unsuccessful. Incapsula incident ID: 12"): "Imperva Incapsula block",
        ("Robot or human?", "Press & Hold"): "PerimeterX press-and-hold check",
        ("Walmart", "Hold the button to confirm that you're human. Press & Hold"):
            "PerimeterX press-and-hold check",
        ("Are you a robot?", "ScienceDirect"): "CAPTCHA",
        ("Security check required", ""): "CAPTCHA",
        ("https://www.google.com/search",
         "Our systems have detected unusual traffic from your computer network. IP address: 1.2"):
            "Google 'unusual traffic' page",
        # Ordinary content that merely talks about these things.
        ("Newest 'python' Questions - Stack Overflow", "Ask Question"): None,
        ("CAPTCHA accessibility guide", "A captcha asks: are you a robot? Security check required"):
            None,
        ("ACL tutorial", "The message 'You don't have permission to access' means ..."): None,
        ("Blog: unusual traffic", "Google shows 'unusual traffic from your computer network'"):
            None,
        ("Docs", "Just a moment... while the demo loads; verify you are human later"): None,
    }
    for (title, body), expected in cases.items():
        assert web_stealth.challenge_vendor(title, body) == expected, (title, body)


def test_search_fallback_url_maps_google_queries_to_bing():
    """Google search / sorry URLs become the same query on Bing; others do not."""
    assert web_stealth.search_fallback_url(
        "https://www.google.com/search?q=playwright+stealth&hl=en"
    ) == "https://www.bing.com/search?q=playwright+stealth"
    sorry = (
        "https://www.google.com/sorry/index?continue=https://www.google.com/search"
        "%3Fq%3Dpatchright%2Bpython&q=EhAmAGAlD2ecTQ"
    )
    assert web_stealth.search_fallback_url(sorry) == (
        "https://www.bing.com/search?q=patchright+python"
    )
    assert web_stealth.search_fallback_url("https://www.google.com/sorry/index") is None
    assert web_stealth.search_fallback_url("https://www.google.com/maps") is None
    assert web_stealth.search_fallback_url("https://www.google.com/search") is None
    assert web_stealth.search_fallback_url("https://www.bing.com/search?q=x") is None
    assert web_stealth.search_fallback_url("https://google.co.uk/search?q=tea") == (
        "https://www.bing.com/search?q=tea"
    )


def test_google_unusual_traffic_page_is_reported_when_no_query_is_known(tool, server):
    """A Google-style block without a recoverable query gets the generic note."""
    tree = tool.go_to_url(f"{server}/sorry/index")
    first = tree.splitlines()[0]
    assert first.startswith("Note:") and "(Google 'unusual traffic' page)" in first


def test_google_unusual_traffic_page_falls_back_to_bing(tool):
    """A real Google search from this network is answered on Bing when blocked.

    Talks to google.com and bing.com; skipped when offline.
    """
    try:
        import socket

        socket.create_connection(("www.google.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("no Internet access")
    tree = tool.go_to_url("https://www.google.com/search?q=patchright+python")
    if "unusual traffic" not in tree.splitlines()[0]:
        pytest.skip("Google did not block this network right now")
    assert "opened on Bing instead" in tree.splitlines()[0]
    assert "URL: https://www.bing.com/search?q=patchright+python" in tree


def test_headless_fallback_still_waits_out_challenges(tool, server):
    """The challenge logic does not depend on the display mode."""
    tool.close()  # one sync driver per thread
    saved_path = os.environ["PATH"]
    web_stealth.stop_virtual_display()
    os.environ["PATH"] = str(os.devnull)
    try:
        web = WebUseTool(user_data_dir=None)
        try:
            tree = web.go_to_url(f"{server}/challenge/{time.time_ns()}")
            assert tree.startswith("Page: Real content")
            assert web._chromium_headless is True
        finally:
            web.close()
    finally:
        os.environ["PATH"] = saved_path


def test_tab_list_uses_the_selected_api_timeout_error(tool, server):
    """tab:list survives an unresponsive tab under the Patchright API."""
    tool.go_to_url(f"{server}/inert")
    listing = tool.go_to_url("tab:list")
    assert listing.startswith("Open tabs (") and "(active)" in listing


def test_typing_deadline_scales_with_text(tool, server):
    """A long text is typed completely: the watchdog does not fire early."""
    tree = tool.go_to_url(f"{server}/events")
    box_id = next(
        int(line.split("]")[0].split("[")[1])
        for line in tree.splitlines() if 'textbox "Name"' in line
    )
    text = "word " * 40
    tool.type_text(box_id, text.strip())
    assert tool._page.evaluate("document.getElementById('box').value") == text.strip()
    assert tool._is_alive()
