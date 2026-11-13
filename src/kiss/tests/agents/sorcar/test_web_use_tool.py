# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for web_use_tool.py module."""

import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import (
    _ACCOUNTS_GOOGLE_URL_RE,
    _SINGLETON_FILES,
    WebUseTool,
    _abort_route,
    _activate_app,
    _get_frontmost_app,
    _kill_driver_process,
    _profile_lock_state,
)
from kiss.tests.conftest import posix_only

# Without pidfds (macOS) asyncio's _ThreadedChildWatcher reaps the driver
# from its own thread the moment it dies, parked loop or not: the dead
# driver is never left as this process's unreaped child.
needs_pidfd_child_watcher = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"),
    reason="only asyncio's pidfd child watcher leaves a dead driver unreaped",
)


class _FailureCollector:
    """Callable that records (url, errorText) for requestfailed events."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str | None]] = []

    def __call__(self, request: Any) -> None:
        self.failures.append((request.url, request.failure))


class _AbortCallTracker:
    """Stand-in for a Playwright Route that records abort() calls."""

    def __init__(self) -> None:
        self.aborted = 0

    def abort(self) -> None:
        self.aborted += 1

FORM_PAGE = b"""<!DOCTYPE html>
<html><head><title>Test Form</title></head>
<body>
  <h1>Test Form Page</h1>
  <a href="/second">Go to second page</a>
  <form>
    <label for="username">Username</label>
    <input type="text" id="username" name="username" placeholder="Enter username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" placeholder="Enter password">
    <label for="color">Color</label>
    <select id="color" name="color">
      <option value="red">Red</option>
      <option value="green">Green</option>
      <option value="blue">Blue</option>
    </select>
    <label for="bio">Bio</label>
    <textarea id="bio" name="bio" placeholder="Bio"></textarea>
    <button type="submit">Submit</button>
  </form>
  <button id="action-btn" onclick="document.title='Clicked!'">Action</button>
  <div id="hover-target" onmouseover="this.textContent='Hovered!'"
       style="padding:20px;background:#eee;" role="button" tabindex="0">Hover me</div>
</body></html>"""

SECOND_PAGE = b"""<!DOCTYPE html>
<html><head><title>Second Page</title></head>
<body>
  <h1>Second Page</h1>
  <a href="/">Back to form</a>
  <p>Content on second page.</p>
</body></html>"""

LONG_PAGE = b"""<!DOCTYPE html>
<html><head><title>Long Page</title></head>
<body style="height: 5000px;">
  <h1>Top of page</h1>
  <div style="position: absolute; top: 3000px;">
    <p>Bottom content</p>
  </div>
</body></html>"""

ROLE_PAGE = b"""<!DOCTYPE html>
<html><head><title>Role Page</title></head>
<body>
  <div role="button" tabindex="0">Role Button</div>
  <div role="link" tabindex="0">Role Link</div>
  <div contenteditable="true" role="textbox" aria-label="Editable div">Editable div</div>
</body></html>"""

EMPTY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Empty</title></head>
<body></body></html>"""

NEW_TAB_PAGE = b"""<!DOCTYPE html>
<html><head><title>New Tab Page</title></head>
<body>
  <a href="/second" target="_blank" id="newtab-link">Open in new tab</a>
</body></html>"""

KEY_PAGE = b"""<!DOCTYPE html>
<html><head><title>Key Test</title></head>
<body>
  <input type="text" id="key-input" onkeydown="this.value=event.key">
  <div id="key-result"></div>
</body></html>"""

# Rows with a "Delete" button and an unlabelled text field each; a click
# or a keystroke is logged in the title.  "Decor" is hidden from the
# accessibility tree but visible on screen, as is all the last list item
# holds, and the "Ghost" buttons take no pointer events: Playwright hands
# out no element reference for those.
ROWS_PAGE = b"""<!DOCTYPE html>
<html><head><title>Rows</title></head>
<body>
  <div aria-hidden="true"><button>Decor</button> art</div>
  <ul>
    <li>one <button onclick="document.title += ' one'">Delete</button>
      <input oninput="document.title += ' 1'"></li>
    <li>two <button onclick="document.title += ' two'">Delete</button>
      <input oninput="document.title += ' 2'"></li>
    <li>three <button onclick="document.title += ' three'">Delete</button>
      <input oninput="document.title += ' 3'"></li>
  </ul>
  <button id="g1" style="pointer-events:none">Ghost</button>
  <button id="g2" style="pointer-events:none">Ghost</button>
  <ol><li><div aria-hidden="true">bullet</div></li></ol>
</body></html>"""

# Two cards that are links of one name, next to a link hidden from the
# accessibility tree but visible on screen.  The snapshot the ids come from
# leaves a card's name to the heading inside it.  Below them: text in a
# plain container, a container that is nothing but its label (on screen,
# then without a box), an iframe, a link that says more than a name holds,
# a link named by some of what it contains, and two links the pointer
# reaches only through what they hold (one takes no pointer events, the
# other has no box of its own).
CARDS_PAGE = b"""<!DOCTYPE html>
<html><head><title>Cards</title></head>
<body><main>
  <a id="one" href="#" onclick="document.title='first';return false"><h2>Open</h2></a>
  <a id="two" href="#" onclick="document.title='second';return false"><h2>Open</h2></a>
  <a aria-hidden="true" href="#">decorative</a>
  <div>plain words</div>
  <div aria-label="Price: it's 5" style="width:9px;height:9px"></div>
  <span title="unseen"></span>
  <iframe srcdoc="<button>Inner</button>"></iframe>
  <a href="#"><p>%s</p></a>
</main>
<a href="#" onclick="document.title='rich';return false"><h3>It's: 5</h3><p>stars <b>now</b></p>
  <figure><img alt="Photo" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" width="9"
    height="9"></figure>
  <div aria-hidden="true">unsaid <i>words</i></div><ul><li>item</li></ul></a>
<a href="#" aria-label="Swatch" style="pointer-events:none"
  onclick="document.title='swatch';return false"><span
  style="display:inline-block;width:9px;height:9px;pointer-events:auto"></span></a>
<div style="position:relative;height:9px"><a href="#" aria-label="Badge"
  onclick="document.title='badge';return false"><span
  style="position:absolute;width:9px;height:9px"></span></a></div>
</body></html>""" % (b"word " * 190)

# Its tree: what Playwright's plain ARIA snapshot of the page says, with an id
# on each link and the label of the container that is on screen as its text.
CARDS_TREE = f"""\
- main:
  - [1] link "Open":
    - /url: "#"
    - heading "Open" [level=2]
  - [2] link "Open":
    - /url: "#"
    - heading "Open" [level=2]
  - text: plain words
  - text: "Price: it's 5"
  - iframe
  - [3] link:
    - /url: "#"
    - paragraph: {('word ' * 190).strip()}
- [4] 'link "It''s: 5 stars now item"':
  - /url: "#"
  - 'heading "It''s: 5" [level=3]'
  - paragraph: stars now
  - figure:
    - img "Photo"
  - list:
    - listitem: item
- [5] link "Swatch":
  - /url: "#"
- [6] link "Badge":
  - /url: "#\""""

# A button next to an iframe of another site (``localhost``, so a renderer of
# its own) whose script never returns: the iframe is wedged, the page is not.
FRAMED_PAGE = b"""<!DOCTYPE html>
<html><head><title>Framed</title></head>
<body>
  <button onclick="document.title = 'Pressed'">Press</button>
  <iframe src="http://localhost:%s/spin"></iframe>
</body></html>"""
SPIN_PAGE = b"<!DOCTYPE html><html><body><script>while (true) {}</script></body></html>"

# A form whose submission the server answers only after SLOW_POST_SECS,
# longer than the 5 s bound of the positioned click in ``_human_click``.
SLOW_FORM_PAGE = b"""<!DOCTYPE html>
<html><head><title>Slow Form</title></head>
<body>
  <form method="post" action="/slowform">
    <button type="submit">Submit slowly</button>
  </form>
</body></html>"""
SLOW_POST_SECS = 7.0
SLOW_POSTS: list[str] = []

# ``document.title`` answers at once with an object whose own getter never
# returns: the title's bounded wait passes and its serialization wedges.
HOSTILE_TITLE_PAGE = (
    b"<!DOCTYPE html><html><head><title>Hostile</title></head><body><p>hostile</p><script>"
    b'Object.defineProperty(document, "title", {get() { return {get x() { while (true) {} }}; }});'
    b"</script></body></html>"
)


@pytest.fixture(scope="module")
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            pages = {
                "/": FORM_PAGE,
                "/second": SECOND_PAGE,
                "/long": LONG_PAGE,
                "/roles": ROLE_PAGE,
                "/empty": EMPTY_PAGE,
                "/newtab": NEW_TAB_PAGE,
                "/keytest": KEY_PAGE,
                "/rows": ROWS_PAGE,
                "/cards": CARDS_PAGE,
                "/framed": FRAMED_PAGE % self.headers.get("Host", "").rsplit(":", 1)[-1].encode(),
                "/spin": SPIN_PAGE,
                "/slowform": SLOW_FORM_PAGE,
                "/hostile-title": HOSTILE_TITLE_PAGE,
            }
            content = pages.get(self.path, FORM_PAGE)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:  # noqa: N802
            SLOW_POSTS.append(self.path)
            time.sleep(SLOW_POST_SECS)
            self.do_GET()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def web_tool():
    tool = WebUseTool(user_data_dir=None, headless=True)
    yield tool
    tool.close()


def _in_own_thread(body: Any, timeout: float = 100) -> None:
    """Run *body* in a fresh thread, as every real task has, and re-raise its failure.

    A sync Playwright is one per thread and the module's ``web_tool``
    already owns this one; a body that hangs fails instead of hanging
    the session (the thread is a daemon).
    """
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            body()
        except BaseException as exc:  # noqa: BLE001 - reported in the test thread
            errors.append(exc)

    thread = threading.Thread(target=_run, name="web-tool-under-test", daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the browser tool hung"
    if errors:
        raise errors[0]


def _kill_driver(tool: WebUseTool) -> None:
    """SIGKILL the Node driver behind *tool*'s Playwright (a driver crash)."""
    os.kill(tool._playwright._impl_obj._connection._transport._proc.pid, signal.SIGKILL)


def _assert_process_gone(pid: int, what: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"{what} (pid {pid}) survived")


def _tool_started_on_a_worker_thread(http_server: str) -> tuple[WebUseTool, int]:
    """Create a tool and load a page on a worker thread; return it with its driver's pid.

    The worker has exited when this returns, so the sync API's greenlet
    is unreachable: a later call from the test thread is the daemon's
    atexit hook closing a tool a pool worker created.
    """
    box: dict[str, Any] = {}

    def body() -> None:
        tool = WebUseTool(user_data_dir=None, headless=True)
        assert "Test Form" in tool.go_to_url(http_server + "/")
        box["tool"] = tool
        box["pid"] = tool._playwright._impl_obj._connection._transport._proc.pid

    _in_own_thread(body)
    return box["tool"], box["pid"]


class TestNavigation:

    def test_go_to_invalid_url(self, web_tool):
        result = web_tool.go_to_url("http://localhost:99999/nonexistent")
        assert "Error" in result

    def test_tab_list_shows_a_tab_whose_title_cannot_be_serialized_as_unresponsive(
        self, web_tool, http_server
    ):
        """One tab whose title read times out must not abort the whole listing.

        ``_read_page`` raises the built-in ``TimeoutError`` when the page's
        answer cannot be serialized; the per-tab recovery of ``tab:list``
        caught only Playwright's unrelated ``TimeoutError`` class, so the
        healthy tabs were lost with the listing.  The hostile page is
        served from another site (``localhost``) so the renderer it
        wedges is not the healthy tab's.
        """
        assert "Second Page" in web_tool.go_to_url(f"{http_server}/second")
        healthy = web_tool._page
        hostile_url = f"{http_server.replace('127.0.0.1', 'localhost')}/hostile-title"
        hostile = web_tool._context.new_page()
        try:
            hostile.goto(hostile_url, wait_until="domcontentloaded")
            listing = web_tool.go_to_url("tab:list")
        finally:
            hostile.close()
        assert listing.startswith("Open tabs ("), listing
        assert f"Second Page - {http_server}/second (active)" in listing
        assert f"(unresponsive) - {hostile_url}" in listing
        assert web_tool._page is healthy
        assert "Second Page" in web_tool.go_to_url(f"{http_server}/second")


class TestClick:
    def test_slow_form_submission_is_posted_once(self, web_tool, http_server):
        """A click whose navigation outlives the 5 s positioned-click bound is not repeated.

        Playwright's click timeout also covers the navigation the click
        started; ``_human_click`` took every timeout for "point not
        hittable" and clicked again at the centre, submitting a form the
        server answers slowly a second time.
        """
        del SLOW_POSTS[:]
        tree = web_tool.go_to_url(f"{http_server}/slowform")
        match = re.search(r'\[(\d+)\] button "Submit slowly"', tree)
        assert match, tree
        result = web_tool.click(int(match.group(1)))
        assert result.startswith("Page: Slow Form"), result
        assert SLOW_POSTS == ["/slowform"]

    def test_an_id_names_its_element_whatever_the_page_does_meanwhile(self, web_tool, http_server):
        """An id acts on the element it was shown next to, or on nothing.

        Ids were looked up again by role, name and position on every use.
        After the page changed under the agent, the id of a removed row's
        button clicked the next row's (the positions shift), the last
        row's id clicked the first row's (the "only match" and "first
        visible" fallbacks), and an id the tree never had was numbered
        against a snapshot the agent had not seen.  Pinning the ids to
        element references read before and after the tree left the ones
        that could not be matched to be found by position still: of two
        cards of one name that had traded places, the first id pressed
        the second.  The tree and the reference behind each id are now
        one snapshot of the page.
        """

        def rows(role: str) -> list[int]:
            tree = web_tool.go_to_url(f"{http_server}/rows")
            return [int(i) for i in re.findall(rf"\[(\d+)\] {role}", tree)]

        def title() -> str:
            return str(web_tool.evaluate_script("() => document.title"))

        def refused(result: str, element_id: int) -> bool:
            return f"element {element_id}: " in result and "get_page_content()" in result

        # A row inserted above moves every position; the ids stay on their elements.
        insert_row = (
            "() => document.querySelector('ul').insertAdjacentHTML('afterbegin',"
            " '<li>zero <button onclick=\"document.title += \\' zero\\'\">Delete</button>"
            "<input oninput=\"document.title += \\' 0\\'\"></li>')"
        )
        arrive_for_a_while = (
            "() => { window.busy = true; const until = performance.now() + 300;"
            f" const add = {insert_row};"
            " (function tick() { add();"
            " if (performance.now() < until) setTimeout(tick, 0); else window.busy = false; })(); }"
        )
        deletes = rows('button "Delete"')
        assert len(deletes) == 3
        web_tool.evaluate_script(insert_row)
        assert web_tool.click(deletes[1]).startswith("Page: Rows two")
        fields = rows("textbox")
        web_tool.evaluate_script(insert_row)
        assert web_tool.type_text(fields[2], "x").startswith("Page: Rows 3")

        # A removed element's id is refused: the next row's button is not clicked...
        deletes = rows('button "Delete"')
        web_tool.evaluate_script("() => document.querySelectorAll('li')[1].remove()")
        assert refused(web_tool.click(deletes[1]), deletes[1])
        # ...nor the first row's for the last one's...
        web_tool.evaluate_script("() => document.querySelectorAll('li')[1].remove()")
        assert refused(web_tool.click(deletes[2]), deletes[2])
        # ...and an element swapped for a look-alike is not the one the agent saw.
        web_tool.evaluate_script(
            "() => { const li = document.querySelector('li'); li.replaceWith(li.cloneNode(true)); }"
        )
        assert refused(web_tool.click(deletes[0]), deletes[0])
        assert refused(web_tool.click(len(web_tool._elements) + 1), len(web_tool._elements) + 1)
        assert title() == "Rows"
        # An id of a document the tab has left is refused the same way, on the
        # document it went back to as well (Playwright's own error named a frame).
        web_tool.go_to_url(f"{http_server}/keytest")
        deletes = rows('button "Delete"')
        web_tool.evaluate_script("() => history.back()")
        deadline = time.monotonic() + 10
        while title() != "Key Test" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert refused(web_tool.click(deletes[0]), deletes[0])
        assert title() == "Key Test"

        # Two cards of one name trade places: the id presses the card it was shown
        # next to.  A card's name, which the snapshot leaves to the heading inside
        # it, is shown; the link hidden from the accessibility tree is not, nor is
        # what the iframe holds.
        tree = web_tool.go_to_url(f"{http_server}/cards")
        assert tree.split("\n\n", 1)[1] == CARDS_TREE
        web_tool.evaluate_script(
            "() => { const two = document.getElementById('two'); two.parentNode.prepend(two); }"
        )
        assert web_tool.click(1).startswith("Page: first")
        assert web_tool.click(4).startswith("Page: rich")
        # A link the pointer cannot reach itself is pressed through the node
        # inside it that it can reach.
        assert web_tool.click(5).startswith("Page: swatch")
        assert web_tool.click(6).startswith("Page: badge")

        # An element the pointer cannot reach, nor anything inside it (a button
        # that takes no pointer events, an option of a closed list), has no
        # reference to pin an id to: it is shown without one.  What is hidden from
        # the accessibility tree is not shown at all.
        rows("button")
        tree = web_tool.get_page_content()
        assert tree.endswith('- button "Ghost"\n- button "Ghost"\n- list:\n  - listitem'), tree
        assert "Decor" not in tree
        tree = web_tool.go_to_url(http_server)
        assert '- option "Red" [selected]' in tree and "] option" not in tree, tree

        # Rows arriving while the page is being read: the tree and its ids are one
        # snapshot, so the id next to row two's button presses that button.
        rows("button")
        for pressed in range(1, 4):
            web_tool.evaluate_script(arrive_for_a_while)
            tree = web_tool.get_page_content()
            deadline = time.monotonic() + 30
            while web_tool.evaluate_script("() => String(window.busy)") == "true":
                assert time.monotonic() < deadline
                time.sleep(0.05)
            match = re.search(r'text: two\n\s+- \[(\d+)\] button "Delete"', tree)
            assert match, tree
            assert web_tool.click(int(match.group(1))).startswith("Page: Rows" + " two" * pressed)

    def test_ids_work_on_a_page_whose_iframe_does_not_answer(self, web_tool, http_server):
        """An iframe that never answers holds the tree up for seconds, and breaks nothing.

        The snapshot the tree and its ids come from reads the page's iframes too.
        One wedged in a renderer of its own keeps it waiting until its deadline
        for them (2 s), when the page's own part comes back.
        """
        try:
            started = time.monotonic()
            tree = web_tool.go_to_url(f"{http_server}/framed")
            match = re.search(r'\[(\d+)\] button "Press"', tree)
            assert match, tree
            assert web_tool.click(int(match.group(1))).startswith("Page: Pressed")
            assert time.monotonic() - started < 60
        finally:
            web_tool.go_to_url("about:blank")

    def test_click_does_not_adopt_a_tab_another_task_opened(self, web_tool, http_server):
        """A tab that appears during the click but was not opened by the page stays alone.

        Tools driving the daemon's Browser tab share one context with the
        tabs of every other task; ``_adopt_new_tab`` compared page counts
        and switched to the newest page of the context whatever opened
        it.  The other task's tab is opened at the real synchronisation
        point: while the clicked link's request is being routed.
        """
        tree = web_tool.go_to_url(f"{http_server}/")
        page = web_tool._page
        match = re.search(r'\[(\d+)\] link "Go to second page"', tree)
        assert match, tree
        opened: list[Any] = []

        def open_another_tab(route: Any) -> None:
            other = web_tool._context.new_page()
            other.goto(f"{http_server}/long", wait_until="domcontentloaded")
            opened.append(other)
            route.continue_()

        page.route("**/second", open_another_tab)
        try:
            result = web_tool.click(int(match.group(1)))
        finally:
            page.unroute("**/second")
            for other in opened:
                other.close()
        assert len(opened) == 1
        assert result.startswith("Page: Second Page"), result
        assert web_tool._page is page

    def test_click_adopts_the_tab_the_page_opened(self, web_tool, http_server):
        """A ``target=_blank`` link's tab, opened by the clicked page, is adopted."""
        tree = web_tool.go_to_url(f"{http_server}/newtab")
        page = web_tool._page
        match = re.search(r'\[(\d+)\] link "Open in new tab"', tree)
        assert match, tree
        result = web_tool.click(int(match.group(1)))
        popup = web_tool._page
        try:
            assert result.startswith("Page: Second Page"), result
            assert popup is not page and popup.opener() is page
        finally:
            popup.close()
            web_tool._adopt_page(page)


class TestCrashRecovery:
    """Verify the tool auto-recovers after Chromium/context dies unexpectedly.

    Simulates the "Google Chrome for Testing quit unexpectedly" scenario by
    closing the browser context out from under the tool.
    """

    def test_auto_relaunch_after_context_close(self, web_tool, http_server):
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()
        web_tool._context.close()
        assert not web_tool._is_alive()
        result = web_tool.go_to_url(http_server + "/")
        assert "Test" in result
        assert web_tool._is_alive()

    def test_on_page_crash_preserves_context(self, web_tool, http_server):
        """_on_page_crash must keep _context alive for cleanup.

        Before the fix, both page crash and browser close used the same
        _on_browser_lost handler which cleared _context, making the main
        browser process unreachable and leaking it.
        """
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()
        ctx = web_tool._context
        brw = web_tool._browser

        web_tool._on_page_crash()

        assert web_tool._page is None
        assert web_tool._context is ctx
        assert web_tool._browser is brw
        assert not web_tool._is_alive()

    def test_page_crash_closes_old_browser(self, web_tool, http_server):
        """After a page renderer crash, recovery must close the old browser.

        Reproduces the root cause of "Google Chrome for Testing quit
        unexpectedly": when only the renderer crashes, the old browser
        process must be terminated during recovery.  Before the fix,
        _on_browser_lost cleared the context reference, leaking the
        main browser process.
        """
        web_tool.go_to_url(http_server + "/")
        assert web_tool._is_alive()

        old_browser = web_tool._browser
        assert old_browser is not None

        web_tool._on_page_crash()
        assert not web_tool._is_alive()
        assert old_browser.is_connected()

        result = web_tool.go_to_url(http_server + "/")
        assert "Test Form" in result
        assert web_tool._is_alive()

        assert not old_browser.is_connected()

    @posix_only("SIGKILL of the driver process")
    @pytest.mark.timeout(120)
    def test_driver_crash_relaunches_instead_of_hanging(self, http_server):
        """A dead Playwright driver must not wedge the tool; the next call relaunches.

        Reproduces the hang seen when the Node driver crashed on an internal
        assertion: ``_is_alive`` trusted ``page.is_closed()``, local state
        the dead driver never updated, and the recovery's graceful
        ``context.close()`` waited forever for a close event nobody sends
        (the watchdog's kill of Chromium unwedges nothing without a
        driver to observe it).  The tool must detect the dead driver,
        discard it and come back with a fresh driver and browser.
        """

        def body() -> None:
            tool = WebUseTool(user_data_dir=None, headless=True)
            try:
                assert "Test Form" in tool.go_to_url(http_server + "/")
                old_playwright = tool._playwright
                old_browser_pid = tool._browser_pid
                assert old_browser_pid is not None
                _kill_driver(tool)
                # The call during which the driver dies may fail (like the
                # one during a renderer crash); the one after it must succeed.
                results = [tool.go_to_url(http_server + "/")]
                if "Test Form" not in results[0]:
                    results.append(tool.go_to_url(http_server + "/"))
                assert "Test Form" in results[-1], results
                assert tool._is_alive()
                assert tool._playwright is not None
                assert tool._playwright is not old_playwright
                assert tool._browser_pid not in (None, old_browser_pid)
                # The old Chromium (a child of the dead driver, reaped by
                # init, never a zombie of this process) must be gone too.
                _assert_process_gone(old_browser_pid, "old Chromium")
            finally:
                tool.close()

        _in_own_thread(body)

    @posix_only("SIGKILL of the driver process")
    @pytest.mark.timeout(120)
    def test_close_after_driver_died_while_idle_returns(self, http_server):
        """``close()`` after a driver death nobody has observed yet must return.

        The first close is what learns of the death (it fails); the second
        object's close would never return.  Chromium must be gone after.
        """

        def body() -> None:
            tool = WebUseTool(user_data_dir=None, headless=True)
            assert "Test Form" in tool.go_to_url(http_server + "/")
            browser_pid = tool._browser_pid
            assert browser_pid is not None
            _kill_driver(tool)
            time.sleep(0.3)
            started = time.monotonic()
            assert tool.close() == "Browser closed."
            assert time.monotonic() - started < 30
            assert tool._playwright is None and tool._page is None
            _assert_process_gone(browser_pid, "Chromium")
            # And the tool is usable again afterwards.
            try:
                assert "Test Form" in tool.go_to_url(http_server + "/")
            finally:
                tool.close()

        _in_own_thread(body)

    @needs_pidfd_child_watcher
    @pytest.mark.timeout(120)
    def test_off_thread_close_reaps_a_driver_that_died_unobserved(self, http_server):
        """A driver dead under a parked loop stays a zombie until ``close()`` reaps it.

        Off the thread that started Playwright, ``stop()`` cannot run and
        the loop whose child watcher would reap the driver is parked with
        that thread's greenlet.  The close must reap the zombie without
        signalling it (``waitid`` reports the exit first) and leave no
        driver or Chromium behind.
        """
        tool, driver_pid = _tool_started_on_a_worker_thread(http_server)
        browser_pid = tool._browser_pid
        assert browser_pid is not None
        os.kill(driver_pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while os.waitid(os.P_PID, driver_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            assert time.monotonic() < deadline, "the killed driver did not exit"
            time.sleep(0.05)
        assert tool.close() == "Browser closed."
        assert tool._playwright is None
        # A zombie still answers kill(pid, 0): gone means reaped.
        _assert_process_gone(driver_pid, "Playwright driver")
        _assert_process_gone(browser_pid, "Chromium")

    @needs_pidfd_child_watcher
    @pytest.mark.timeout(120)
    def test_a_reaped_driver_is_not_signalled_again(self, http_server):
        """Once the driver is reaped its pid is nobody's to signal.

        The loop's child watcher can reap the driver before the transport
        records the exit (``returncode`` still unset), and the pid may
        belong to another process by then.  The test reaps it the way the
        watcher would; the kill helper must find nothing to signal or
        reap, and the close must still complete.
        """
        tool, driver_pid = _tool_started_on_a_worker_thread(http_server)
        os.kill(driver_pid, signal.SIGKILL)
        assert os.waitpid(driver_pid, 0)[0] == driver_pid
        assert tool._playwright._impl_obj._connection._transport._proc.returncode is None
        assert _kill_driver_process(tool._playwright) is None
        assert tool.close() == "Browser closed."
        assert tool._playwright is None


class TestSingletonLockCleanup:
    """Stale Singleton{Lock,Cookie,Socket} from a previously crashed Chromium
    must be removed before launching a persistent context."""

    def test_cleans_stale_singleton_files(self, tmp_path):
        (tmp_path / _SINGLETON_FILES[0]).symlink_to("stale-host-99999")
        (tmp_path / _SINGLETON_FILES[1]).write_text("stale")
        tool = WebUseTool(user_data_dir=str(tmp_path), headless=True)
        try:
            tool._clean_singleton_locks(str(tmp_path))
            for name in _SINGLETON_FILES:
                assert not (tmp_path / name).exists()
                assert not (tmp_path / name).is_symlink()
        finally:
            tool.close()


class TestFocusHelpers:
    """Tests for _get_frontmost_app and _activate_app focus management."""

    def test_get_frontmost_app_returns_string_on_macos(self):
        """On macOS, _get_frontmost_app should return the current app name."""
        result = _get_frontmost_app()
        if sys.platform == "darwin":
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            assert result is None

    def test_activate_app_none_is_noop(self):
        """_activate_app(None) should silently do nothing."""
        _activate_app(None)

    def test_activate_app_with_valid_app(self):
        """_activate_app with a real app should not raise."""
        if sys.platform == "darwin":
            _activate_app("Finder")

    def test_activate_app_with_nonexistent_app(self):
        """_activate_app with a bogus name should not raise (best-effort)."""
        _activate_app("NonExistentApp12345")

    def test_ensure_browser_calls_focus_helpers(self, web_tool):
        """_ensure_browser should save and restore focus even in headless mode."""
        web_tool._ensure_browser()
        web_tool._context.close()
        assert not web_tool._is_alive()
        web_tool._ensure_browser()
        assert web_tool._is_alive()


class TestConcurrentProfileAccess:
    """Two WebUseTool instances sharing the same user_data_dir must not collide.

    Reproduces the "Something went wrong when opening your profile" error that
    occurs when multiple tasks (e.g. local extension + remote web server) try to
    use Chromium with the same profile directory simultaneously.
    """

    def test_second_instance_uses_different_profile(self, http_server, tmp_path):
        """When the profile is locked by one instance, a second must auto-select a variant.

        Each tool runs in its own thread (realistic: real tasks come from
        separate extension/web-server threads). Thread A locks the profile,
        signals readiness, then thread B launches and must detect the lock
        and use a numbered variant instead of crashing with the Chromium
        profile error.
        """
        profile_dir = str(tmp_path / "shared_profile")

        a_ready = threading.Event()
        b_done = threading.Event()
        results: dict[str, str] = {}
        errors: dict[str, Exception] = {}

        def _run_a() -> None:
            tool = WebUseTool(user_data_dir=profile_dir, headless=True)
            try:
                results["a"] = tool.go_to_url(http_server + "/")
                a_ready.set()
                b_done.wait(timeout=60)
            except Exception as exc:
                errors["a"] = exc
                a_ready.set()
            finally:
                tool.close()

        def _run_b() -> None:
            a_ready.wait(timeout=60)
            tool = WebUseTool(user_data_dir=profile_dir, headless=True)
            try:
                results["b"] = tool.go_to_url(http_server + "/")
            except Exception as exc:
                errors["b"] = exc
            finally:
                tool.close()
                b_done.set()

        t1 = threading.Thread(target=_run_a)
        t2 = threading.Thread(target=_run_b)
        t1.start()
        t2.start()
        t2.join(timeout=90)
        b_done.set()
        t1.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert "Test Form" in results["a"]
        assert "Test Form" in results["b"]

    def test_is_profile_in_use_no_lock(self, tmp_path):
        """Profile without SingletonLock is not in use."""
        assert not _profile_lock_state(str(tmp_path))[1]

    @posix_only("Chromium's SingletonLock symlink is the POSIX profile lock")
    def test_is_profile_in_use_stale_lock(self, tmp_path):
        """Profile with SingletonLock pointing to a dead PID is not in use."""
        (tmp_path / "SingletonLock").symlink_to("hostname-999999999")
        assert not _profile_lock_state(str(tmp_path))[1]

    @posix_only("Chromium's SingletonLock symlink is the POSIX profile lock")

    def test_is_profile_in_use_live_lock(self, tmp_path):
        """Profile with SingletonLock pointing to this process is in use."""
        import os

        (tmp_path / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")
        assert _profile_lock_state(str(tmp_path))[1]

    @posix_only("Chromium's SingletonLock symlink is the POSIX profile lock")

    def test_resolve_skips_locked_profiles(self, tmp_path):
        """_resolve_user_data_dir skips the configured dir when it's locked."""
        import os

        profile = str(tmp_path / "profile")
        Path(profile).mkdir()
        (Path(profile) / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")

        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            resolved = tool._resolve_user_data_dir()
            assert resolved == f"{profile}_1"
        finally:
            tool.close()

    @posix_only("Chromium's SingletonLock symlink is the POSIX profile lock")

    def test_resolve_skips_multiple_locked(self, tmp_path):
        """_resolve_user_data_dir skips numbered variants that are also locked."""
        import os

        profile = str(tmp_path / "profile")
        for suffix in ["", "_1", "_2"]:
            d = Path(f"{profile}{suffix}")
            d.mkdir(parents=True)
            (d / "SingletonLock").symlink_to(f"hostname-{os.getpid()}")

        tool = WebUseTool(user_data_dir=profile, headless=True)
        try:
            resolved = tool._resolve_user_data_dir()
            assert resolved == f"{profile}_3"
        finally:
            tool.close()

    def test_resolve_returns_none_for_no_dir(self):
        """_resolve_user_data_dir returns None when user_data_dir is None."""
        tool = WebUseTool(user_data_dir=None, headless=True)
        try:
            assert tool._resolve_user_data_dir() is None
        finally:
            tool.close()


class TestAccountsGoogleUrlRegex:
    """Pure unit tests for the accounts.google.com URL regex."""

    def test_matches_https_root(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match("https://accounts.google.com/")

    def test_matches_http_root(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match("http://accounts.google.com/")

    def test_matches_signin_path(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com/signin/v2/identifier"
        )

    def test_matches_oauth_path(self):
        assert _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com/o/oauth2/v2/auth?foo=bar"
        )

    def test_rejects_other_google_host(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("https://www.google.com/")

    def test_rejects_subdomain_prefix(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("https://myaccounts.google.com/")

    def test_rejects_homograph_suffix(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match(
            "https://accounts.google.com.evil.com/"
        )

    def test_rejects_ftp_scheme(self):
        assert not _ACCOUNTS_GOOGLE_URL_RE.match("ftp://accounts.google.com/")


class TestAbortRoute:
    """Unit tests for the _abort_route helper."""

    def test_calls_route_abort_once(self):
        tracker = _AbortCallTracker()
        _abort_route(tracker)
        assert tracker.aborted == 1


def _run_block_test(profile: str, http_url: str, out: dict) -> None:
    """Worker for test_request_to_accounts_google_is_aborted.

    Runs in its own thread so it gets a fresh asyncio thread-local state,
    which is required because the module-scoped ``web_tool`` fixture parks
    a Playwright greenlet on the main thread that leaves
    ``asyncio._running_loop`` set, causing a fresh
    ``sync_playwright().start()`` on the main thread to fail with
    "Playwright Sync API inside the asyncio loop".
    """
    tool = WebUseTool(user_data_dir=profile, headless=True)
    try:
        tool.go_to_url(http_url + "/")
        out["alive_before"] = tool._is_alive()

        collector = _FailureCollector()
        tool._context.on("requestfailed", collector)

        result = tool.go_to_url("https://accounts.google.com/signin")
        out["result_type"] = type(result).__name__
        tool._page.wait_for_timeout(500)
        out["failures"] = list(collector.failures)
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    finally:
        tool.close()


def _run_other_host_test(profile: str, http_url: str, out: dict) -> None:
    """Worker for test_other_hosts_are_not_blocked (runs in its own thread)."""
    tool = WebUseTool(user_data_dir=profile, headless=True)
    try:
        out["body"] = tool.go_to_url(http_url + "/")
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    finally:
        tool.close()


class TestAccountsGoogleRouteBlocking:
    """Verify the route registered in _launch_browser blocks accounts.google.com.

    These integration tests launch a real persistent-context browser
    (the only path where the route is currently registered) and confirm
    that any request to accounts.google.com is aborted before it leaves
    the browser.

    They run in a worker thread because the module-scoped ``web_tool``
    fixture leaves a Playwright greenlet parked on the main thread, which
    makes the next ``sync_playwright().start()`` on that thread raise.
    """

    def test_request_to_accounts_google_is_aborted(self, tmp_path, http_server):
        profile = str(tmp_path / "profile")
        out: dict = {}
        t = threading.Thread(
            target=_run_block_test, args=(profile, http_server, out)
        )
        t.start()
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
        assert "error" not in out, f"worker raised: {out.get('error')}"
        assert out["alive_before"] is True
        assert out["result_type"] is not None

        matches = [
            (url, err)
            for url, err in out["failures"]
            if "accounts.google.com" in url
        ]
        assert matches, (
            "No requestfailed event captured for accounts.google.com; "
            f"all failures: {out['failures']}"
        )
        url, err = matches[0]
        assert err is not None
        assert "ERR_FAILED" in err or "ABORTED" in err, (
            f"Expected an aborted-request error, got: {err!r}"
        )

    def test_other_hosts_are_not_blocked(self, tmp_path, http_server):
        """Sanity check: requests to other hosts must still succeed."""
        profile = str(tmp_path / "profile2")
        out: dict = {}
        t = threading.Thread(
            target=_run_other_host_test, args=(profile, http_server, out)
        )
        t.start()
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
        assert "error" not in out, f"worker raised: {out.get('error')}"
        assert "Test Form" in out["body"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
