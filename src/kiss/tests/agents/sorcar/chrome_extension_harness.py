# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real Chromium running the installed bridge extension, watched over the DevTools protocol.

The end-to-end tests of ``chrome_extension/background.js`` load the
extension the way a user does (the folder :func:`install_extension`
writes), let it connect to a real :class:`ChromeBridge` and drive real
loopback pages.  Nothing of the extension is replaced or instrumented:
which of its code ran is read from V8 itself (``Profiler`` precise
coverage with block granularity), in the service worker and in every
page it injects code into.

The browser is started by its own command line and keeps its own
defaults, so that a tab does what a user's tab does: a browser-automation
launcher (Playwright) switches the pop-up blocker and the back/forward
cache off, has Chromium replace a tab it discards instead of unloading it
in place, and holds a page that is loaded again after a crash with a
DevTools session of its own.

The DevTools connection (:class:`Cdp`) attaches to every page and to the
extension's service worker as they appear, starts the coverage before
their first script runs and can act as the browser's user does where the
headless browser has no window for it (switch the extension off and on,
close a tab, end the worker).
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

BACKGROUND_JS = (
    Path(__file__).parents[3] / "agents" / "sorcar" / "chrome_extension" / "background.js"
)
# Pages of the browser itself.  A toolbar click sent to one of them (a component
# extension's background page counts as a tab) ends the browser.
_BROWSER_PAGES = ("chrome://", "chrome-extension://")
# What the browser is started with besides its profile and the extension: no
# window, a DevTools port, no first-run pages, no calls home, no keyring
# prompt, colours as a page names them, and what a container needs (no
# sandbox, no /dev/shm).  None of it changes what a tab or an extension does.
_FLAGS = (
    "--headless=new",
    "--remote-debugging-port=0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-search-engine-choice-screen",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--metrics-recording-only",
    "--password-store=basic",
    "--use-mock-keychain",
    "--force-color-profile=srgb",
    "--no-sandbox",
    "--disable-dev-shm-usage",
)


def start_chromium(binary: str, profile: Path, extension: Path) -> subprocess.Popen[bytes]:
    """Start the Chromium *binary* on the profile dir *profile* with the unpacked *extension*.

    A new profile gets Developer mode switched on, as the profile of a
    user who loaded an unpacked extension has it: Chrome reloads such an
    extension in that mode only.
    """
    preferences = profile / "Default" / "Preferences"
    if not preferences.exists():
        preferences.parent.mkdir(parents=True)
        preferences.write_text(json.dumps({"extensions": {"ui": {"developer_mode": True}}}))
    return subprocess.Popen(
        [
            binary,
            *_FLAGS,
            f"--user-data-dir={profile}",
            f"--load-extension={extension}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_chromium(browser: subprocess.Popen[bytes]) -> None:
    """End a browser :func:`start_chromium` started."""
    browser.terminate()
    try:
        browser.wait(15)
    except subprocess.TimeoutExpired:
        browser.kill()
        browser.wait()


class Cdp:
    """A DevTools-protocol connection to the browser, for the commands of all its sessions."""

    def __init__(self, url: str, on_event: Callable[[dict[str, Any]], None]) -> None:
        """Connect to the browser endpoint *url*; *on_event* gets every event, on the reader."""
        self._ws = connect(url, max_size=None)
        self._ids = itertools.count(1)
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._on_event = on_event
        threading.Thread(target=self._read, daemon=True).start()

    def post(
        self, method: str, params: dict[str, Any] | None = None, session: str = ""
    ) -> Future[dict[str, Any]]:
        """Send a command and return the future of its reply.

        The one way to send from an event handler: it runs on the thread
        that reads the replies, so it must not wait for one.

        Args:
            method: The protocol method, e.g. ``"Profiler.enable"``.
            params: Its parameters.
            session: The target's session id; empty for the browser itself.
        """
        reply: Future[dict[str, Any]] = Future()
        with self._lock:
            frame: dict[str, Any] = {"id": next(self._ids), "method": method}
            if params:
                frame["params"] = params
            if session:
                frame["sessionId"] = session
            self._pending[frame["id"]] = reply
            self._ws.send(json.dumps(frame))
        return reply

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session: str = "",
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Send a command and return its result.

        Raises:
            RuntimeError: the browser answered with an error.
            TimeoutError: no answer within *timeout* seconds.
        """
        return self.post(method, params, session).result(timeout)

    def close(self) -> None:
        """Close the connection; the browser keeps running."""
        self._ws.close()

    def _read(self) -> None:
        try:
            with suppress(ConnectionClosed):  # the browser exited
                self._read_frames()
        finally:  # nobody waits for an answer that cannot come
            with self._lock:
                pending, self._pending = list(self._pending.values()), {}
            for reply in pending:
                reply.set_exception(RuntimeError("the DevTools connection closed"))

    def _read_frames(self) -> None:
        for raw in self._ws:
            message = json.loads(raw)
            if "id" not in message:
                self._on_event(message)
                continue
            with self._lock:
                reply = self._pending.pop(message["id"], None)
            if reply is None:
                continue
            if "error" in message:
                reply.set_exception(RuntimeError(json.dumps(message["error"])))
            else:
                reply.set_result(message.get("result", {}))


@dataclass
class _Text:
    """A run of characters as they reach a page, each with the source span it stands for."""

    text: str
    spans: list[tuple[int, int]]


class _Hole:
    """A value interpolated at run time (``${limit.at}``): any text, standing for no source."""


def _cook(
    source: str, start: int, end: int, templates: dict[str, list[_Text | _Hole]]
) -> list[_Text | _Hole]:
    """The body ``source[start:end]`` of a template literal, as V8 cooks it.

    An escape becomes its character (mapped to both source characters),
    ``${NAME}`` of a template in *templates* becomes that template's
    parts, and any other interpolation a hole.
    """
    parts: list[_Text | _Hole] = []
    run = _Text("", [])
    position = start
    while position < end:
        if source.startswith("${", position):
            close = source.index("}", position)
            if run.text:
                parts.append(run)
                run = _Text("", [])
            parts.extend(templates.get(source[position + 2 : close], [_Hole()]))
            position = close + 1
        elif source[position] == "\\":
            run.text += {"\\": "\\", "n": "\n"}[source[position + 1]]
            run.spans.append((position, position + 2))
            position += 2
        else:
            run.text += source[position]
            run.spans.append((position, position + 1))
            position += 1
    if run.text:
        parts.append(run)
    return parts


def _page_code(source: str) -> list[list[_Text | _Hole]]:
    """The code of *source* that runs in pages, never in the worker.

    The sources the worker keeps as template literals (``KISS_*_JS``,
    ``KISS_TARGET_JS`` only as part of the two that interpolate it), the
    deadline gate ``evalViaDebugger`` puts before a script, and
    ``runCode``, which Chrome injects by its source text.
    """
    templates: dict[str, list[_Text | _Hole]] = {}
    regions = []
    for match in re.finditer(r"const (KISS_\w+_JS) = `", source):
        parts = _cook(source, match.end(), source.index("`", match.end()), templates)
        templates[match.group(1)] = parts
        if match.group(1) != "KISS_TARGET_JS":
            regions.append(parts)
    gate = source.index("const gate = `") + len("const gate = `")
    regions.append(_cook(source, gate, source.index("`", gate), templates))
    start = source.index("async function runCode(")
    end = source.index("\n}\n", start) + 2
    regions.append([_Text(source[start:end], [(i, i + 1) for i in range(start, end)])])
    return regions


def _offset_counts(functions: list[dict[str, Any]], length: int) -> list[int]:
    """How often V8 ran each character of a script, from its coverage *functions*.

    V8 lists a function before the functions inside it and a block
    before the blocks inside it, so the innermost count is written last.
    A function without block counts tells nothing about its branches
    (its code was compiled before the coverage started): it counts as
    not run.  Only the first *length* characters are counted: the
    installed worker is the source and line ends after it.
    """
    counts = [0] * length
    for function in functions:
        for block in function["ranges"]:
            count = block["count"] if function["isBlockCoverage"] else 0
            start, end = block["startOffset"], min(block["endOffset"], length)
            counts[start:end] = [count] * (end - start)
    return counts


def _region_pattern(region: list[_Text | _Hole]) -> re.Pattern[str]:
    """A pattern finding *region* in a page script, one group per run of its text."""
    return re.compile(
        "".join(
            f"({re.escape(part.text)})" if isinstance(part, _Text) else ".*?" for part in region
        ),
        re.DOTALL,
    )


class ExtensionChromium:
    """Headless Chromium running the extension installed in ``$KISS_HOME``.

    The coverage of ``background.js`` adds up over every service worker
    and every page for as long as the browser runs: :meth:`collect` reads
    it, :meth:`uncovered` reports what never ran.
    """

    def __init__(self, binary: str, profile: Path, extension: Path) -> None:
        """Start the Chromium *binary* on the profile dir *profile* with the unpacked *extension*.

        Args:
            binary: The path of the Chromium executable.
            profile: An empty directory for the browser profile.
            extension: The extension directory (``extension_dir()``).
        """
        self._source = BACKGROUND_JS.read_text(encoding="utf-8")
        regions = _page_code(self._source)
        self._regions = [
            (_region_pattern(region), [p for p in region if isinstance(p, _Text)])
            for region in regions
        ]
        self._page_offsets = {
            offset
            for region in regions
            for part in region
            if isinstance(part, _Text)
            for start, end in part.spans
            for offset in range(start, end)
        }
        self._worker_counts = [0] * len(self._source)
        self._page_counts = [0] * len(self._source)
        self._functions: dict[str, tuple[int, int]] = {}  # the worker's, by name
        self._sessions: dict[str, dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()
        self._script_sources: dict[tuple[str, str], str] = {}
        self._worker_started = threading.Event()
        self._worker_stopped = threading.Event()  # at the line stop_worker_at named
        self.extension_id = ""  # known once chrome://extensions has been asked
        self._process = start_chromium(binary, profile, extension)
        try:
            self._cdp = Cdp(_devtools_url(profile), self._on_event)
            self._cdp.call(
                "Target.setAutoAttach",
                {
                    "autoAttach": True,
                    "waitForDebuggerOnStart": True,
                    "flatten": True,
                    "filter": [{"type": "service_worker"}, {"type": "page"}],
                },
            )
            self.open_manager()
            self.extension_id = self._unpacked_extension()
            # The worker Chrome started with the browser ran before this
            # connection could measure it: the one measured is the one
            # new_worker() starts, once the first has run its script (Chrome
            # does not always end a worker it is told to while it starts).
            deadline = time.monotonic() + 30
            while not self._workers():
                assert time.monotonic() < deadline, "the extension's service worker never started"
                time.sleep(0.02)
            self.worker_eval("1")
            self.new_worker()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the browser."""
        stop_chromium(self._process)

    def open_manager(self) -> None:
        """Open chrome://extensions, the page that may switch extensions off and on.

        :meth:`new_worker` works through it: a test that closed every tab
        of the browser opens it again.
        """
        manager = self._cdp.call("Target.createTarget", {"url": "chrome://extensions"})
        self._manager = self._session_of(manager["targetId"])

    def new_worker(self) -> None:
        """Switch the extension off and on, as its switch on chrome://extensions does.

        Chrome ends the service worker (and what it held: its socket, its
        debugger attachments, its alarms) and starts a new one, whose
        coverage is recorded from its first statement on.
        """
        self.collect()
        self._set_enabled(False)
        self._worker_started.clear()
        self._set_enabled(True)
        assert self._worker_started.wait(30), "the extension's service worker never restarted"
        # A worker that outlived the switch would dial the bridge beside the
        # new one, and the bridge keeps one connection: the newest.
        for _, outlived in self._workers()[:-1]:
            with suppress(RuntimeError, TimeoutError):  # gone by itself meanwhile
                self._cdp.call("Target.closeTarget", {"targetId": outlived["targetId"]}, timeout=5)

    @contextmanager
    def switched_off(self) -> Iterator[None]:
        """Keep the extension switched off, as by its switch on chrome://extensions, for a block.

        Chrome ends the service worker, and starts one when the block ends:
        the worker that the manifest it loaded names, read from disk then.
        """
        self.collect()
        self._set_enabled(False)
        try:
            yield
        finally:
            self._set_enabled(True)

    def set_developer_mode(self, on: bool) -> None:
        """Flip the Developer mode switch of chrome://extensions."""
        self._evaluate(
            self._manager,
            "chrome.developerPrivate.updateProfileConfiguration("
            f"{{inDeveloperMode: {json.dumps(on)}}})",
        )

    def extension_state(self) -> str:
        """What chrome://extensions shows of the extension: ``ENABLED``, ``DISABLED``, ..."""
        states = self._evaluate(
            self._manager,
            "chrome.developerPrivate.getExtensionsInfo({includeDisabled: true}).then((all) =>"
            f" all.filter((e) => e.id === {json.dumps(self.extension_id)}).map((e) => e.state))",
        )
        return str(states[0])

    def kill_worker(self) -> None:
        """End the service worker without a word to it, as Chrome ends an idle or a wedged one."""
        self.collect()
        self._cdp.call("Target.closeTarget", {"targetId": self._worker()[1]["targetId"]})

    def stop_worker_at(self, code: str) -> None:
        """Have the service worker stop before the line of ``background.js`` that holds *code*.

        For a line that ends the worker (the extension reloading itself):
        what the worker ran is gone with it, so :meth:`let_worker_go`
        records it while the worker stands still, and nothing of what
        the line does is replaced.
        """
        line = next(n for n, text in enumerate(self._source.split("\n")) if code in text)
        session = self._worker()[0]
        self._worker_stopped.clear()
        self._cdp.call("Debugger.enable", session=session)
        self._cdp.call(
            "Debugger.setBreakpointByUrl",
            {  # the worker's file is named after its build
                "lineNumber": line,
                "urlRegex": f"^chrome-extension://{self.extension_id}/background-.*",
            },
            session=session,
        )

    def let_worker_go(self) -> None:
        """Wait for the stop :meth:`stop_worker_at` set up, record the coverage, let it run on."""
        assert self._worker_stopped.wait(60), "the service worker never reached the line"
        session = self._worker()[0]
        self._collect_worker(session)
        self._cdp.call("Debugger.disable", session=session)  # it runs on, with no stop left in it

    def click_toolbar_icon(self, times: int = 1) -> None:
        """Click the extension's toolbar icon, the user's way to wake the bridge.

        Args:
            times: How many clicks, sent together: the worker has them
                all before it has finished handling the first.
        """
        action = {"id": self.extension_id, "targetId": self._web_tab()}
        for click in [self._cdp.post("Extensions.triggerAction", action) for _ in range(times)]:
            click.result(15)

    def worker_eval(self, expression: str) -> Any:
        """Evaluate *expression* in the service worker's global scope and return its value.

        For what a test must see of the worker's state and cannot through
        the bridge (its alarms); a promise is awaited.
        """
        session = self._worker()[0]
        # A worker that has just started has no globals, not even ``chrome``,
        # until Chrome has run background.js in it.
        deadline = time.monotonic() + 15
        while self._evaluate(session, "typeof connect") != "function":
            assert time.monotonic() < deadline, "the service worker never ran background.js"
            time.sleep(0.02)
        return self._evaluate(session, expression)

    def page_eval(self, url: str, expression: str) -> Any:
        """Evaluate *expression* in the page showing *url* and return its value.

        The test's own look at a page, past the extension: a tab whose
        command queue is busy still answers here.
        """
        return self._evaluate(self._session_of(self.pages()[url]), expression)

    def page_listeners(self, url: str, event: str) -> int:
        """How many listeners the window of the page showing *url* has for *event*."""
        count = f"(getEventListeners(window)[{json.dumps(event)}] || []).length"
        return int(self._evaluate(self._session_of(self.pages()[url]), count, console=True))

    def pages(self) -> dict[str, str]:
        """The open pages, as URL to DevTools target id (the extension manager's left out)."""
        targets = self._cdp.call("Target.getTargets", {"filter": [{"type": "page"}]})
        return {
            info["url"]: info["targetId"]
            for info in targets["targetInfos"]
            if not info["url"].startswith(_BROWSER_PAGES)
        }

    def close_page(self, target_id: str) -> None:
        """Close a tab the way its close button does."""
        self._cdp.call("Target.closeTarget", {"targetId": target_id})

    def crash_page(self, target_id: str) -> None:
        """End the process of a tab's page, as a crash does: the tab stays, with no page in it."""
        self.collect()
        self._cdp.post("Page.crash", session=self._session_of(target_id))  # never answered

    def discard_tab(self, tab_id: int) -> None:
        """Unload the page of the tab *tab_id* (the extension's id for it), as Memory Saver does."""
        self.collect()
        assert self.worker_eval(f"chrome.tabs.discard({tab_id}).then((tab) => tab.id)") == tab_id

    def close_pages(self) -> None:
        """Record the coverage of every open page, then close them all."""
        self.collect()
        for target_id in self.pages().values():
            self.close_page(target_id)

    def collect(self) -> None:
        """Add what ran since the last call to the coverage of ``background.js``.

        V8 keeps the counts per process and forgets them with it, so a
        test calls this before it closes a tab or navigates one away.
        """
        with self._sessions_lock:
            sessions = dict(self._sessions)
        pages: dict[str, list[str]] = {}  # processes (V8 isolates) and their sessions
        for session, info in sessions.items():
            try:
                if self._is_worker(info):
                    self._collect_worker(session)
                elif info["type"] == "page":
                    isolate = self._cdp.call("Runtime.getIsolateId", session=session, timeout=5)
                    pages.setdefault(isolate["id"], []).append(session)
            except (RuntimeError, TimeoutError):
                continue  # the target went away, or is a worker Chrome has stopped
        for isolate_id, members in pages.items():
            self._collect_pages(isolate_id, members)

    def uncovered(self, function: str = "") -> list[str]:
        """The code of ``background.js`` that never ran, as ``line: source`` entries.

        Worker code counts as run when a service worker ran it; the code
        the worker injects into pages when a page ran it.

        Args:
            function: Only the code of the worker function of this name
                (a command's, say); empty for the whole file.
        """
        self.collect()
        first, last = self._functions[function] if function else (0, len(self._source))
        counts = [
            self._page_counts[offset] if offset in self._page_offsets else worker
            for offset, worker in enumerate(self._worker_counts)
        ]
        missing = []
        offset = first
        while offset < last:
            if counts[offset]:
                offset += 1
                continue
            end = offset
            while end < last and not counts[end]:
                end += 1
            text = self._source[offset:end]
            if _CODE.search(_COMMENT.sub("", text)):
                line = self._source.count("\n", 0, offset) + 1
                missing.append(f"{line}: {' '.join(text.split())[:160]}")
            offset = end
        return missing

    # -- the DevTools side --------------------------------------------------

    def _on_event(self, message: dict[str, Any]) -> None:
        """Start the coverage of every new page and worker before its first script runs."""
        params = message.get("params", {})
        if message.get("method") == "Target.attachedToTarget":
            session, info = params["sessionId"], params["targetInfo"]
            with self._sessions_lock:
                self._sessions[session] = info
            self._measure(session, info["type"])
            if self._is_worker(info):
                self._worker_started.set()
        elif message.get("method") == "Inspector.targetReloadedAfterCrash":
            # A worker Chrome started again after it had ended, a page loaded again
            # after a crash: the same target in a new process, which waits like a
            # new one and is announced by no attachedToTarget.
            with self._sessions_lock:
                info = self._sessions.get(message["sessionId"])
            if info is not None:
                self._measure(message["sessionId"], info["type"])
        elif message.get("method") == "Debugger.paused":
            self._worker_stopped.set()  # pages skip every pause (_measure)
        elif message.get("method") == "Target.detachedFromTarget":
            with self._sessions_lock:
                self._sessions.pop(params["sessionId"], None)

    def _measure(self, session: str, kind: str) -> None:
        """Start the coverage of the process *session* is attached to, then let it run."""
        self._cdp.post("Profiler.enable", session=session)
        self._cdp.post(
            "Profiler.startPreciseCoverage", {"callCount": True, "detailed": True}, session
        )
        if kind == "page":  # the sources of the scripts injected into it
            self._cdp.post("Debugger.enable", session=session)
            self._cdp.post("Debugger.setSkipAllPauses", {"skip": True}, session)
        self._cdp.post("Runtime.runIfWaitingForDebugger", session=session)

    def _session_of(self, target_id: str) -> str:
        """The session attached to the target *target_id* (it attaches within moments)."""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with self._sessions_lock:
                for session, info in self._sessions.items():
                    if info["targetId"] == target_id:
                        return session
            time.sleep(0.02)
        raise AssertionError(f"no DevTools session for target {target_id}")

    def _unpacked_extension(self) -> str:
        """The id of the one extension loaded from a folder, as chrome://extensions knows it."""
        unpacked = (
            "globalThis.chrome && chrome.management"
            " ? new Promise((done) => chrome.management.getAll(done)).then((all) =>"
            " all.filter((e) => e.installType === 'development').map((e) => e.id))"
            " : []"  # the page has not loaded yet
        )
        deadline = time.monotonic() + 15
        while True:
            found = self._evaluate(self._manager, unpacked)
            if len(found) == 1:
                return str(found[0])
            assert time.monotonic() < deadline, f"unpacked extensions in the browser: {found}"
            time.sleep(0.02)

    def _is_worker(self, info: dict[str, Any]) -> bool:
        """Whether the target *info* is a service worker of the extension under test.

        Chrome runs extensions of its own beside it, one with a
        ``background.js`` too.
        """
        ours = f"chrome-extension://{self.extension_id}/"
        return bool(info["type"] == "service_worker" and info["url"].startswith(ours))

    def _workers(self) -> list[tuple[str, dict[str, Any]]]:
        """The sessions and target infos of the extension's service workers, the newest last."""
        with self._sessions_lock:
            return [
                (session, info) for session, info in self._sessions.items() if self._is_worker(info)
            ]

    def _worker(self) -> tuple[str, dict[str, Any]]:
        """The session and the target info of the extension's service worker."""
        return self._workers()[-1]

    def _web_tab(self) -> str:
        """The DevTools tab-target id of a tab showing a web page; one is opened when none is."""
        if not self.pages():
            self._cdp.call("Target.createTarget", {"url": "about:blank"})
        targets = self._cdp.call("Target.getTargets", {"filter": [{"type": "tab"}]})
        return next(
            str(info["targetId"])
            for info in targets["targetInfos"]
            if not info["url"].startswith(_BROWSER_PAGES)
        )

    def _evaluate(self, session: str, expression: str, console: bool = False) -> Any:
        """Evaluate *expression* in *session*; *console* adds what only DevTools' console has."""
        reply = self._cdp.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
                "includeCommandLineAPI": console,
            },
            session=session,
        )
        assert "exceptionDetails" not in reply, reply
        return reply["result"].get("value")

    def _set_enabled(self, enabled: bool) -> None:
        self._evaluate(
            self._manager,
            "new Promise((done) => chrome.management.setEnabled("
            f"{json.dumps(self.extension_id)}, {json.dumps(enabled)}, done))",
        )

    def _collect_worker(self, session: str) -> None:
        taken = self._cdp.call("Profiler.takePreciseCoverage", session=session, timeout=5)
        for script in taken["result"]:
            # The worker's own file, whatever build names it (what a test
            # evaluates in the worker has no URL).
            if script["url"].startswith(f"chrome-extension://{self.extension_id}/"):
                counts = _offset_counts(script["functions"], len(self._source))
                self._worker_counts = [a + b for a, b in zip(self._worker_counts, counts)]
                for function in script["functions"]:
                    whole = function["ranges"][0]
                    self._functions[function["functionName"]] = (
                        whole["startOffset"],
                        whole["endOffset"],
                    )

    def _collect_pages(self, isolate_id: str, sessions: list[str]) -> None:
        """Add the coverage of the injected scripts of one renderer process.

        A script without a URL is an injected one (or a page's ``eval``):
        its text tells which of the extension's page code it holds.
        """
        try:
            taken = self._cdp.call("Profiler.takePreciseCoverage", session=sessions[0], timeout=5)
        except (RuntimeError, TimeoutError):
            return
        for script in taken["result"]:
            if script["url"]:
                continue
            source = self._script_source(isolate_id, sessions, script["scriptId"])
            counts: list[int] = []
            for pattern, texts in self._regions:
                for match in pattern.finditer(source):
                    counts = counts or _offset_counts(script["functions"], len(source))
                    for group, text in enumerate(texts, start=1):
                        for index, (start, end) in enumerate(text.spans):
                            count = counts[match.start(group) + index]
                            for offset in range(start, end):
                                self._page_counts[offset] += count

    def _script_source(self, isolate_id: str, sessions: list[str], script_id: str) -> str:
        """The text of a page script; any session of its process may know it."""
        key = (isolate_id, script_id)
        if key not in self._script_sources:
            self._script_sources[key] = ""
            for session in sessions:
                try:
                    reply = self._cdp.call(
                        "Debugger.getScriptSource", {"scriptId": script_id}, session, timeout=5
                    )
                except (RuntimeError, TimeoutError):
                    continue
                self._script_sources[key] = reply["scriptSource"]
                break
        return self._script_sources[key]


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_CODE = re.compile(r"[^\s{}()\[\];,]")  # V8 also reports the closing brace after a return


def _devtools_url(profile: Path) -> str:
    """The browser's DevTools endpoint, from the file Chromium writes into its profile dir."""
    port_file = profile / "DevToolsActivePort"
    deadline = time.monotonic() + 30
    while not port_file.exists() or port_file.read_text().count("\n") < 1:
        assert time.monotonic() < deadline, "Chromium never published its DevTools port"
        time.sleep(0.05)
    port, path = port_file.read_text().split("\n")[:2]
    return f"ws://127.0.0.1:{port}{path}"
